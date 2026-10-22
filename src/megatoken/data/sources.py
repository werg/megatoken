"""Pinned, streaming source readers. Importing this module performs no network I/O."""

from __future__ import annotations

import gzip
import itertools
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from megatoken.io import content_hash, sha256_file, write_json

PARQUET_BATCH_BYTES = 64 << 20


@dataclass(frozen=True, slots=True)
class TermsReview:
    approved: bool = False
    license: str = "unreviewed"
    license_url: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("Terms approval must be an explicit JSON boolean")

    def require_approval(self) -> None:
        if not self.approved or not self.reviewed_by or not self.reviewed_at:
            raise ValueError(
                "Source terms are unapproved; record approved/reviewed_by/reviewed_at in the source manifest"
            )


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    uri: str
    format: str
    adapter: str
    stream: str
    revision: str
    terms: TermsReview = TermsReview()
    files: tuple[SourceFile, ...] = ()
    subset: str | None = None
    split: str = "train"
    max_rows: int | None = None
    field_map: Mapping[str, str | list[str]] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.source_id, self.uri, self.adapter, self.stream, self.revision)):
            raise ValueError(
                "Sources require identity, URI, adapter, stream, and a pinned revision"
            )
        if self.format not in {"jsonl", "parquet", "hf"}:
            raise ValueError("Source format must be jsonl, parquet, or hf")
        if self.format == "hf" and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(
                "Hugging Face sources require an immutable 40-character commit, not main/a tag"
            )
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive, or null for an explicitly unbounded run")

    @property
    def fingerprint(self) -> str:
        """What is read, not who approved it or where the files sit on this machine."""
        value = asdict(self)
        value["terms"] = {"license": self.terms.license, "license_url": self.terms.license_url}
        value["files"] = [{**item, "path": Path(item["path"]).name} for item in value["files"]]
        if self.format != "hf":
            value["uri"] = Path(self.uri).name
        if value["options"].get("lineage_join"):
            join = value["options"]["lineage_join"]
            value["options"] = {
                **value["options"],
                "lineage_join": {**join, "path": Path(join["path"]).name},
            }
        return content_hash(value)

    @property
    def repository_id(self) -> str:
        return (
            self.uri.removeprefix("hf://datasets/")
            .removeprefix("hf://")
            .removeprefix("https://huggingface.co/datasets/")
        )

    def validate_local_files(self) -> None:
        self.terms.require_approval()
        if self.format == "hf":
            return
        if not self.files:
            raise ValueError("Local sources require an explicit checksummed files list")
        for source_file in self.files:
            path = Path(source_file.path)
            if not source_file.sha256:
                raise ValueError(f"Missing source checksum: {path}")
            if source_file.size_bytes is not None and path.stat().st_size != source_file.size_bytes:
                raise ValueError(f"Source size changed: {path}")
            if sha256_file(path) != source_file.sha256:
                raise ValueError(f"Source checksum changed: {path}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, base: Path | None = None) -> SourceSpec:
        obj = dict(value)
        # Human-readable acquisition notes do not silently enter reader options.
        for key in ("notes", "readiness", "publisher_card", "metadata_checked_at"):
            obj.pop(key, None)
        obj["terms"] = TermsReview(**obj.get("terms", {}))
        files = []
        for entry in obj.get("files", ()):
            item = SourceFile(**entry)
            if base is not None and obj.get("format") != "hf" and not Path(item.path).is_absolute():
                item = replace(item, path=str((base / item.path).resolve()))
            files.append(item)
        obj["files"] = tuple(files)
        options = dict(obj.get("options", {}))
        if base is not None and options.get("lineage_join"):
            join = dict(options["lineage_join"])
            location = Path(join["path"])
            if not location.is_absolute():
                join["path"] = str((base / location).resolve())
            options["lineage_join"] = join
        obj["options"] = options
        return cls(**obj)


def load_source_manifest(path: str | Path) -> tuple[SourceSpec, ...]:
    location = Path(path)
    obj = json.loads(location.read_text())
    if obj.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema")
    sources = tuple(
        SourceSpec.from_dict(value, base=location.resolve().parent) for value in obj["sources"]
    )
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("Source IDs must be unique within one manifest")
    return sources


def iter_source_rows(
    source: SourceSpec, *, start_row: int = 0
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream deterministic logical rows; Parquet skips completed row groups.

    A Hugging Face streaming resume counts logical rows sequentially. This avoids
    format-dependent Arrow ``skip`` behavior but reads preceding rows again; it
    never stores a whole dataset in Python memory. Local JSONL/GZip resumes scan
    old lines without parsing them.
    """
    source.terms.require_approval()
    if start_row < 0:
        raise ValueError("start_row cannot be negative")
    if source.max_rows is not None and start_row >= source.max_rows:
        return
    if source.format == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face streaming requires the 'data' extra (datasets)"
            ) from exc
        kwargs: dict[str, Any] = {
            "path": source.repository_id,
            "name": source.subset,
            "split": source.split,
            "revision": source.revision,
            "streaming": True,
        }
        if source.files:
            kwargs["data_files"] = {source.split: [item.path for item in source.files]}
        if source.options.get("read_batch_rows"):
            # Parquet rows that are megabytes long overflow Arrow's 2 GiB list offsets when read
            # in the file's own (large) batches; a smaller batch reads the same rows.
            kwargs["batch_size"] = int(source.options["read_batch_rows"])
        dataset = load_dataset(**kwargs)
        # datasets 4.8.5 Arrow skip can repeat fully skipped Parquet row groups.
        # Count actual emitted rows so persisted logical cursors stay exact.
        for row_index, row in enumerate(
            itertools.islice(dataset, start_row, source.max_rows), start=start_row
        ):
            yield row_index, dict(row)
        return
    row_index = 0
    for source_file in source.files:
        path = Path(source_file.path)
        if source.format == "jsonl":
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if source.max_rows is not None and row_index >= source.max_rows:
                        return
                    current = row_index
                    row_index += 1
                    if current < start_row:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"Expected a JSON object at source row {current} in {path}"
                        )
                    yield current, obj
        else:
            try:
                import pyarrow.parquet as parquet
            except ImportError as exc:
                raise RuntimeError("Parquet streaming requires the 'data' extra (pyarrow)") from exc
            reader = parquet.ParquetFile(path)
            for group in range(reader.num_row_groups):
                meta = reader.metadata.row_group(group)
                count = meta.num_rows
                if row_index + count <= start_row:
                    row_index += count
                    continue
                # Arrow cannot hand back a nested column over 2 GiB in one piece, and long
                # trajectories reach that within a few rows: keep each batch near 64 MiB.
                row_bytes = meta.total_byte_size / max(count, 1)
                batch_size = max(1, min(128, int(PARQUET_BATCH_BYTES / max(row_bytes, 1))))
                for batch in reader.iter_batches(batch_size=batch_size, row_groups=[group]):
                    for row in batch.to_pylist():
                        if source.max_rows is not None and row_index >= source.max_rows:
                            return
                        current = row_index
                        row_index += 1
                        if current >= start_row:
                            yield current, row


def pin_local_source(
    path: str | Path,
    *,
    source_id: str,
    adapter: str,
    stream: str,
    terms: TermsReview,
    max_rows: int | None = None,
) -> SourceSpec:
    file = Path(path).resolve()
    sha = sha256_file(file)
    format = "parquet" if file.suffix == ".parquet" else "jsonl"
    return SourceSpec(
        source_id=source_id,
        uri=str(file),
        format=format,
        adapter=adapter,
        stream=stream,
        revision=f"sha256:{sha}",
        files=(SourceFile(str(file), sha, file.stat().st_size),),
        terms=terms,
        max_rows=max_rows,
    )


def pin_hf_source(
    repository_id: str,
    *,
    source_id: str,
    adapter: str,
    stream: str,
    revision: str = "main",
    subset: str | None = None,
    split: str = "train",
    terms: TermsReview | None = None,
    max_rows: int | None = 1000,
) -> SourceSpec:
    """Resolve metadata only; no corpus download and no terms acceptance."""
    terms = terms or TermsReview()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Revision resolution requires huggingface_hub") from exc
    info = HfApi().dataset_info(repository_id, revision=revision)
    return SourceSpec(
        source_id=source_id,
        uri=f"hf://datasets/{repository_id}",
        format="hf",
        adapter=adapter,
        stream=stream,
        revision=info.sha,
        subset=subset,
        split=split,
        terms=terms,
        max_rows=max_rows,
    )


def write_source_manifest(path: str | Path, sources: tuple[SourceSpec, ...]) -> None:
    write_json(path, {"schema_version": 1, "sources": [asdict(source) for source in sources]})
