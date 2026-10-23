"""Build a separate Korean-field correction corpus for the pinned Glaive source.

This module never edits a prepared source database. It adapts the same raw row twice:
once with the active English field map to recover the existing family keys, and once
with the Korean field map to create a distinct corrected episode. The result is a
native episode JSONL file plus checksummed import metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from megatoken.io import sha256_file

from .adapters import AdapterContext, SchemaError, _tools, adapt_row
from .schema import Episode, canonical_json, make_records, sequential_meridians, stable_id
from .sources import SourceSpec

KOREAN_SOURCE_ID = "heegyu-glaive-ko"
KOREAN_SOURCE_REVISION = "c8b54e70bfbdd2fd6be037557e6feaf1abf14f5d"
CORRECTED_SOURCE_ID = "heegyu-glaive-ko-korean-correction-v1"
NATIVE_IMPORT_ID = "heegyu-glaive-ko-korean-correction-native-v1"
DEFAULT_MAX_RECORD_BYTES = 16_384
EXPECTED_PINNED_ROWS = 15_170


class CorrectionRowRejected(ValueError):
    """A source row that cannot safely join and adapt both language views."""

    def __init__(self, stage: str, error_type: str):
        self.stage = stage
        self.error_type = error_type
        super().__init__(f"{stage}:{error_type}")


@dataclass(frozen=True, slots=True)
class CorrectionReceipt:
    source_id: str
    revision: str
    source_fingerprint: str
    correction_fingerprint: str
    native_source_id: str
    input_rows: int
    emitted_rows: int
    rejected_rows: int
    rows_with_system_context: int
    rows_with_normalized_tools: int
    rows_with_verbatim_tool_context: int
    rows_linked_to_english_family: int
    rejected_row_indices: tuple[int, ...]
    rejection_counts: dict[str, int]
    output_sha256: str
    output_size_bytes: int


def korean_correction_source(original: SourceSpec) -> SourceSpec:
    """Return the same pinned source under a new identity and Korean message field."""
    if original.source_id != KOREAN_SOURCE_ID:
        raise ValueError(f"Expected source {KOREAN_SOURCE_ID!r}")
    if original.revision != KOREAN_SOURCE_REVISION:
        raise ValueError("The Korean correction only supports the audited pinned revision")
    if original.format != "hf" or original.adapter != "simulated_tool_roles":
        raise ValueError("The pinned correction source format or adapter changed")
    return replace(
        original,
        source_id=CORRECTED_SOURCE_ID,
        max_rows=EXPECTED_PINNED_ROWS,
        terms=replace(
            original.terms,
            license_url="https://huggingface.co/datasets/heegyu/glaive-function-calling-v2-ko",
        ),
        field_map={**dict(original.field_map), "messages": "conversations_ko"},
    )


def _context(source: SourceSpec, row_index: int, *, max_record_bytes: int) -> AdapterContext:
    return AdapterContext(
        source_id=source.source_id,
        revision=source.revision,
        uri=source.uri,
        stream=source.stream,
        row_index=row_index,
        max_record_bytes=max_record_bytes,
        field_map=source.field_map,
        options=source.options,
    )


def _raw_context_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return canonical_json(value).encode("utf-8")


def _tool_schema_list(value: Any) -> str | None:
    """Normalize a structurally usable function description, else leave it as context."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
    else:
        parsed = value
    candidates = parsed if isinstance(parsed, list) else [parsed]
    if not candidates or any(not isinstance(item, Mapping) for item in candidates):
        return None

    # A tool description is not useful as a tool unless it identifies a function and
    # exposes an object-shaped argument schema. Keep unconventional descriptions as
    # observed context instead of manufacturing a tool from incomplete metadata.
    for candidate in candidates:
        function = candidate.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            parameters = function.get("parameters")
        elif isinstance(candidate.get("id"), str) and isinstance(
            candidate.get("inputSchema"), Mapping
        ):
            name = candidate["id"]
            schema = candidate["inputSchema"]
            parameters = schema.get("jsonSchema", schema)
        else:
            name = candidate.get("name")
            parameters = candidate.get("parameters")
        if not isinstance(name, str) or not name.strip():
            return None
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except json.JSONDecodeError:
                return None
        if not isinstance(parameters, Mapping):
            return None

    try:
        normalized = json.loads(_tools(list(candidates)))
    except (SchemaError, TypeError, ValueError):
        return None
    if not isinstance(normalized, list) or len(normalized) != len(candidates):
        return None

    from jsonschema.exceptions import SchemaError as JsonSchemaError
    from jsonschema.validators import validator_for

    try:
        for item in normalized:
            function = item.get("function") if isinstance(item, Mapping) else None
            function = function if isinstance(function, Mapping) else item
            name = function.get("name")
            parameters = function.get("parameters")
            if not isinstance(name, str) or not name.strip() or not isinstance(parameters, dict):
                return None
            validator_for(parameters).check_schema(parameters)
    except (JsonSchemaError, TypeError, ValueError):
        return None
    return canonical_json(normalized)


def _context_record(
    episode: Episode,
    *,
    payload: bytes,
    role: str,
    source_field: str,
    source_uri: str,
    max_record_bytes: int,
):
    logical_id = stable_id("korean-correction-context-v1", episode.episode_id, source_field)
    return make_records(
        payload,
        logical_id=logical_id,
        role=role,  # type: ignore[arg-type]
        turn_index=0,
        policy_mask=False,
        provenance="natural" if role == "system" else "unknown",
        max_bytes=max_record_bytes,
        source_uri=source_uri,
        source_field=source_field,
        transform="identity",
    )


def correct_korean_row(
    row: Mapping[str, Any],
    row_index: int,
    *,
    original_source: SourceSpec,
    corrected_source: SourceSpec | None = None,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> Episode:
    """Adapt one raw row and link its Korean family to the existing English family.

    The input mapping is never modified. Both adapters hash that original mapping, so
    the corrected lineage retains the upstream revision, raw-row hash, and row index.
    """
    corrected_source = corrected_source or korean_correction_source(original_source)
    if corrected_source.source_id != CORRECTED_SOURCE_ID:
        raise ValueError("Corrected source identity changed")
    if corrected_source.revision != original_source.revision:
        raise ValueError("English and Korean views must use the same pinned revision")
    if corrected_source.field_map.get("messages") != "conversations_ko":
        raise ValueError("Corrected view must map messages to conversations_ko")

    try:
        english = adapt_row(
            row,
            original_source.adapter,
            _context(original_source, row_index, max_record_bytes=max_record_bytes),
        )
    except (SchemaError, ValueError) as error:
        raise CorrectionRowRejected("english_lineage", type(error).__name__) from error

    try:
        korean = adapt_row(
            row,
            corrected_source.adapter,
            _context(corrected_source, row_index, max_record_bytes=max_record_bytes),
        )
    except (SchemaError, ValueError) as error:
        raise CorrectionRowRejected("korean_adaptation", type(error).__name__) from error

    english_lineage, korean_lineage = english.lineage, korean.lineage
    if (
        english_lineage.raw_sha256 != korean_lineage.raw_sha256
        or english_lineage.revision != korean_lineage.revision
        or english_lineage.source_row != korean_lineage.source_row
    ):
        raise ValueError("The two language views did not retain identical raw-row provenance")
    if english_lineage.source_row != row_index:
        raise ValueError("The adapted row index differs from the source reader cursor")

    merged_keys = set(english_lineage.family_keys) | set(korean_lineage.family_keys)
    primary_family = english_lineage.family_id
    merged_keys.discard(primary_family)
    lineage = replace(
        korean_lineage,
        family_id=primary_family,
        ancestor_ids=tuple(sorted(merged_keys)),
    )
    if not set(english_lineage.family_keys).issubset(lineage.family_keys):
        raise ValueError("Korean lineage does not contain every existing English family key")

    records = list(korean.records)
    # The source's system prompt is a separate column, not a message in either
    # conversation list. Preserve it as unscored system context if one is present.
    system_message = row.get("system_message")
    if system_message not in (None, "") and not any(record.role == "system" for record in records):
        records = (
            list(
                _context_record(
                    korean,
                    payload=_raw_context_bytes(system_message),
                    role="system",
                    source_field="system_message",
                    source_uri=corrected_source.uri,
                    max_record_bytes=max_record_bytes,
                )
            )
            + records
        )

    tool_schema = _tool_schema_list(row.get("function_description"))
    tools_json = korean.tools_json
    if tool_schema is not None:
        tools_json = tool_schema
    elif row.get("function_description") not in (None, ""):
        # Invalid or unfamiliar descriptions remain visible as unscored evidence.
        records = (
            list(
                _context_record(
                    korean,
                    payload=_raw_context_bytes(row["function_description"]),
                    role="document",
                    source_field="function_description",
                    source_uri=corrected_source.uri,
                    max_record_bytes=max_record_bytes,
                )
            )
            + records
        )

    # Chat adapters already create sequential meridians. Rebuild those after adding
    # the prelude so every new physical record gets an ordered event of its own.
    return sequential_meridians(
        replace(
            korean,
            records=tuple(records),
            lineage=lineage,
            tools_json=tools_json,
            tracks=(),
            meridians=(),
        )
    )


def _native_manifest(
    *,
    corrected_source: SourceSpec,
    output_name: str,
    row_count: int,
    digest: str,
    size: int,
) -> dict[str, Any]:
    terms = asdict(corrected_source.terms)
    return {
        "schema_version": 1,
        "sources": [
            {
                "source_id": NATIVE_IMPORT_ID,
                "uri": output_name,
                "format": "jsonl",
                "adapter": "native",
                "stream": corrected_source.stream,
                "revision": f"{corrected_source.revision}+correction-v1",
                "subset": None,
                "split": "train",
                "max_rows": row_count,
                "files": [{"path": output_name, "sha256": digest, "size_bytes": size}],
                "terms": terms,
                "options": {
                    "preserve_native_stream": True,
                    "quality": {"min_scored_positions": 1},
                },
                "field_map": {},
                "notes": (
                    "Separate Korean-field correction addendum; family keys include the "
                    "existing English view. Not yet merged into the main prepared-data pin."
                ),
            }
        ],
    }


def build_correction_corpus(
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    *,
    original_source: SourceSpec,
    output_dir: str | Path,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
    expected_input_rows: int | None = None,
) -> CorrectionReceipt:
    """Write an immutable correction addendum, receipt, and native import manifest.

    ``output_dir`` must not exist. The active preparation directory is never opened.
    Row-level data errors are recorded only by stage, exception type, and source index;
    exception messages and source payloads are not placed in the receipt. Rejected
    source row indices are retained so a bounded follow-up can reproduce the skip.
    """
    corrected_source = korean_correction_source(original_source)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    episodes_path = root / "episodes.jsonl"
    temporary = root / "episodes.jsonl.part"
    receipt_path = root / "receipt.json"
    manifest_path = root / "source-manifest.json"
    counts: Counter[str] = Counter()
    rejected_indices: list[int] = []
    input_rows = emitted = system_rows = normalized_tools = verbatim_tools = linked = 0
    previous_row_index = -1
    output_digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            for row_index, row in rows:
                if row_index <= previous_row_index:
                    raise ValueError("Correction source rows must have strictly increasing indices")
                previous_row_index = row_index
                input_rows += 1
                try:
                    episode = correct_korean_row(
                        row,
                        row_index,
                        original_source=original_source,
                        corrected_source=corrected_source,
                        max_record_bytes=max_record_bytes,
                    )
                except CorrectionRowRejected as error:
                    counts[f"{error.stage}:{error.error_type}"] += 1
                    rejected_indices.append(row_index)
                    continue

                encoded = (canonical_json(episode.to_dict()) + "\n").encode("utf-8")
                handle.write(encoded)
                output_digest.update(encoded)
                emitted += 1
                linked += 1
                system_rows += int(any(record.role == "system" for record in episode.records))
                normalized_tools += int(episode.tools_json != "[]")
                verbatim_tools += int(
                    any(
                        record.source_span is not None
                        and record.source_span.field == "function_description"
                        for record in episode.records
                    )
                )
            handle.flush()
            os.fsync(handle.fileno())

        if emitted == 0:
            raise ValueError("No Korean correction episodes passed both source adapters")
        if expected_input_rows is not None and input_rows != expected_input_rows:
            raise ValueError(
                f"Pinned source yielded {input_rows} rows; expected {expected_input_rows}"
            )
        os.replace(temporary, episodes_path)
        digest = output_digest.hexdigest()
        size = episodes_path.stat().st_size
        manifest = _native_manifest(
            corrected_source=corrected_source,
            output_name=episodes_path.name,
            row_count=emitted,
            digest=digest,
            size=size,
        )
        native_source = SourceSpec.from_dict(manifest["sources"][0], base=root)
        receipt = CorrectionReceipt(
            source_id=corrected_source.source_id,
            revision=corrected_source.revision,
            source_fingerprint=original_source.fingerprint,
            correction_fingerprint=corrected_source.fingerprint,
            native_source_id=native_source.source_id,
            input_rows=input_rows,
            emitted_rows=emitted,
            rejected_rows=input_rows - emitted,
            rows_with_system_context=system_rows,
            rows_with_normalized_tools=normalized_tools,
            rows_with_verbatim_tool_context=verbatim_tools,
            rows_linked_to_english_family=linked,
            rejected_row_indices=tuple(rejected_indices),
            rejection_counts=dict(sorted(counts.items())),
            output_sha256=digest,
            output_size_bytes=size,
        )
        _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(receipt_path, asdict(receipt))
        if sha256_file(episodes_path) != digest:
            raise ValueError("Correction output checksum changed after writing")
        return receipt
    except BaseException:
        # Preserve the partial artifact for diagnosis; never remove or modify an
        # existing output. The absence of receipt/manifest marks an incomplete build.
        raise


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    payload = (canonical_json(value) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
