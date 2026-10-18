"""Resumable import -> external-memory dedup -> frozen split -> atomic shards."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from megatoken.io import content_hash, write_json

from .adapters import AdapterContext, SchemaError, adapt_row
from .dedup import DedupConfig, DedupStore
from .lineage import LineageJoin
from .quality import QualityPolicy, validate_quality, validate_row_filters
from .schema import Codec, encode_record
from .sources import SourceSpec, iter_source_rows, load_source_manifest
from .storage import PreparedDataset, export_prepared


def prepare_sources(
    sources: Iterable[SourceSpec] | str | Path,
    output: str | Path,
    codec: Codec,
    *,
    dedup: DedupConfig | None = None,
    max_record_bytes: int = 16384,
    max_record_tokens: int = 4096,
    shard_bytes: int = 128 << 20,
    benchmark_quarantine: dict[str, Any] | None = None,
    reject_invalid: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    _interrupt_after: int | None = None,
) -> PreparedDataset:
    """Prepare source rows, checking source identity before resuming any cursor.

    Malformed rows fail the default run. ``reject_invalid=True`` writes their
    explicit source/row/error receipts and continues; an operator can inspect
    the rejection distribution before authorizing scale. `_interrupt_after` is
    a deterministic crash-injection hook for the resume invariant test.
    """
    dedup = dedup or DedupConfig()
    if min(max_record_bytes, max_record_tokens, shard_bytes) < 1:
        raise ValueError("Physical record and shard limits must be positive")
    source_list = tuple(
        load_source_manifest(sources) if isinstance(sources, (str, Path)) else sources
    )
    if not source_list:
        raise ValueError("At least one explicit source is required")
    if len({source.source_id for source in source_list}) != len(source_list):
        raise ValueError("Duplicate source IDs")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    for source in source_list:
        source.validate_local_files()
    quarantine = benchmark_quarantine or {"family_keys": [], "content_sha256": []}
    identity = content_hash(
        {
            "sources": [asdict(source) for source in source_list],
            "codec": codec.fingerprint,
            "dedup": asdict(dedup),
            "max_record_bytes": max_record_bytes,
            "max_record_tokens": max_record_tokens,
            "benchmark_quarantine": quarantine,
            "reject_invalid": reject_invalid,
            "adapter_version": 2,
            "schema_version": 1,
        }
    )
    if (root / "manifest.json").exists():
        ready = PreparedDataset(root, split=None)
        if ready.manifest["pipeline_fingerprint"] != identity:
            ready.close()
            raise ValueError(
                "An existing prepared dataset has a different pipeline/source identity"
            )
        return ready
    with DedupStore(root / "preparation.sqlite", dedup) as store, ExitStack() as resources:
        previous = store.get_meta("pipeline_fingerprint")
        if previous is not None and previous != identity:
            raise ValueError(
                "Preparation inputs/configuration changed; refuse to resume an incompatible cursor"
            )
        store.set_meta("pipeline_fingerprint", identity)
        store.connection.commit()
        if not store.frozen:
            store.quarantine_benchmarks(
                family_keys=quarantine.get("family_keys", ()),
                content_sha256=quarantine.get("content_sha256", ()),
            )
            processed = 0
            for source in source_list:
                start, complete = store.source_cursor(source.fingerprint)
                if complete:
                    continue
                join_config = source.options.get("lineage_join")
                lineage_join = None
                if join_config:
                    lineage_join = LineageJoin(
                        join_config["path"],
                        join_config["sha256"],
                        root / f"lineage-{source.fingerprint[:16]}.sqlite",
                    )
                    resources.callback(lineage_join.close)
                last = start - 1
                for row_index, row in iter_source_rows(source, start_row=start):
                    context = AdapterContext(
                        source_id=source.source_id,
                        revision=source.revision,
                        uri=source.uri,
                        stream=source.stream,
                        row_index=row_index,
                        max_record_bytes=max_record_bytes,
                        field_map=source.field_map,
                        options=source.options,
                        codec=codec,
                        max_record_tokens=max_record_tokens,
                    )
                    with store.transaction():
                        try:
                            validate_row_filters(row, source.options.get("row_filters", []))
                            if lineage_join is not None:
                                row = lineage_join.apply(row, join_config.get("row_key", "uuid"))
                            episode = adapt_row(row, source.adapter, context)
                            # A rejected low-quality variant still contributes its
                            # known ancestor links before any train/eval split.
                            store.register_episode_lineage(episode)
                            quality = dict(source.options.get("quality", {}))
                            if "allowed_outcomes" in quality:
                                quality["allowed_outcomes"] = tuple(quality["allowed_outcomes"])
                            validate_quality(episode, codec, QualityPolicy(**quality))
                            if any(
                                encode_record(record, codec).positions > max_record_tokens
                                for record in episode.records
                            ):
                                raise ValueError(
                                    "A committed native physical record exceeds max_record_tokens; preserve its boundary or configure a larger supported capacity"
                                )
                            status = store.add(episode, codec)
                            detail = episode.episode_id
                        except (SchemaError, ValueError, TypeError, KeyError) as error:
                            if not reject_invalid:
                                raise ValueError(
                                    f"Source {source.source_id}, row {row_index}: {error}"
                                ) from error
                            status, detail = "rejected", str(error)
                        store.mark_source(
                            source.fingerprint, row_index, status=status, detail=detail
                        )
                    last = row_index
                    processed += 1
                    if progress and processed % 100 == 0:
                        progress({"source": source.source_id, "row": row_index, "status": status})
                    if _interrupt_after is not None and processed >= _interrupt_after:
                        if lineage_join is not None:
                            lineage_join.close()
                        raise InterruptedError(
                            "Injected preparation interruption after durable row commit"
                        )
                if lineage_join is not None:
                    lineage_join.close()
                with store.transaction():
                    # Empty sources still become durably complete without a fake row.
                    store.connection.execute(
                        "INSERT INTO source_progress VALUES(?,?,1) ON CONFLICT(source_key) DO UPDATE SET next_row=excluded.next_row,complete=1",
                        (source.fingerprint, last + 1),
                    )
            store.freeze()
        receipts = [asdict(source) | {"fingerprint": source.fingerprint} for source in source_list]
        export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=receipts,
            pipeline_fingerprint=identity,
            shard_bytes=shard_bytes,
        )
        # Quarantine/reject details are metadata, never concatenated to transcripts.
        report = {
            "schema_version": 1,
            "rows": [
                dict(row)
                for row in store.connection.execute(
                    "SELECT source_key,row_index,status,detail FROM source_rows WHERE status='rejected' LIMIT 10000"
                )
            ],
            "sample_limit": 10000,
            "total_rejected": store.census()["rejected_rows"],
        }
        write_json(root / "rejections.json", report)
    return PreparedDataset(root, split=None)
