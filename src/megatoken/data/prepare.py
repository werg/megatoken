"""Resumable import -> external-memory dedup -> frozen split -> atomic shards."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from megatoken.io import content_hash, write_json

from .adapters import (
    AdapterContext,
    ChatFramingError,
    SchemaError,
    adapt_row,
    source_row_lineage,
    source_row_supervision,
)
from .chunking import ChunkingConfig
from .dedup import DedupConfig, DedupStore
from .lineage import LineageJoin
from .quality import validate_row_filters
from .review import ParserReviewPolicy, ParserReviewQueue, source_quality, validate_prepared_episode
from .schema import SCHEMA_VERSION, Codec, canonical_json
from .sources import SourceSpec, iter_source_rows, load_source_manifest
from .storage import PreparedDataset, export_prepared


def prepare_sources(
    sources: Iterable[SourceSpec] | str | Path,
    output: str | Path,
    codec: Codec,
    *,
    dedup: DedupConfig | None = None,
    max_record_bytes: int = 16384,
    max_record_tokens: int = 1024,
    chunking: ChunkingConfig | None = None,
    review: ParserReviewPolicy | None = None,
    shard_bytes: int = 128 << 20,
    benchmark_quarantine: dict[str, Any] | None = None,
    reject_invalid: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    _interrupt_after: int | None = None,
) -> PreparedDataset:
    """Prepare source rows, checking source identity before resuming any cursor.

    Parser faults/ambiguities are captured in the review queue and excluded from
    accepted shards. Ordinary schema/quality failures fail the default run.
    ``reject_invalid=True`` writes their
    explicit source/row/error receipts and continues; an operator can inspect
    the rejection distribution before authorizing scale. `_interrupt_after` is
    a deterministic crash-injection hook for the resume invariant test.
    """
    dedup = dedup or DedupConfig()
    review = review or ParserReviewPolicy()
    if min(max_record_bytes, max_record_tokens, shard_bytes) < 1:
        raise ValueError("Physical record and shard limits must be positive")
    chunking = (chunking or ChunkingConfig()).with_hard_limit(
        max_record_tokens, max_bytes=max_record_bytes
    )
    max_record_tokens = chunking.hard_max_tokens
    max_record_bytes = chunking.max_bytes
    chunking_identity = chunking.identity(codec)
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
            "chunking": chunking_identity,
            "parser_review": asdict(review),
            "benchmark_quarantine": quarantine,
            "reject_invalid": reject_invalid,
            "adapter_version": 4,
            "schema_version": SCHEMA_VERSION,
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
        review_queue = ParserReviewQueue(store.connection)
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
                    raw_row = row
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
                        chunking=chunking,
                    )
                    with store.transaction():
                        try:
                            validate_row_filters(row, source.options.get("row_filters", []))
                            if lineage_join is not None:
                                row = lineage_join.apply(row, join_config.get("row_key", "uuid"))
                            try:
                                episode = adapt_row(row, source.adapter, context)
                            except ChatFramingError as error:
                                if not review.enabled:
                                    raise
                                # Even an unparseable frame must register its
                                # source ancestry before family splits freeze.
                                lineage = source_row_lineage(row, source.adapter, context)
                                family = store.register_family(lineage.family_keys)
                                supervision = source_row_supervision(row, source.adapter, context)
                                if supervision.benchmark_tags:
                                    store.connection.execute(
                                        "INSERT OR IGNORE INTO benchmarks VALUES(?, 'family')",
                                        (family,),
                                    )
                                detail = review_queue.capture(
                                    source=source,
                                    row_index=row_index,
                                    raw_row=raw_row,
                                    adapted_row=row,
                                    episode=None,
                                    parser_identity=chunking_identity,
                                    chunking=chunking,
                                    policy=review,
                                    reasons=error.diagnostics,
                                    error=str(error),
                                )
                                status, episode = "parser_review", None
                            # A rejected low-quality variant still contributes its
                            # known ancestor links before any train/eval split.
                            if episode is not None:
                                store.register_episode_lineage(episode)
                                validate_prepared_episode(
                                    episode,
                                    codec,
                                    source_quality(source),
                                    max_record_tokens=max_record_tokens,
                                    max_record_bytes=max_record_bytes,
                                )
                                reasons = review.reasons(episode, adapter=source.adapter)
                                if reasons:
                                    detail = review_queue.capture(
                                        source=source,
                                        row_index=row_index,
                                        raw_row=raw_row,
                                        adapted_row=row,
                                        episode=episode,
                                        parser_identity=chunking_identity,
                                        chunking=chunking,
                                        policy=review,
                                        reasons=reasons,
                                    )
                                    status = "parser_review"
                                else:
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
        store.set_meta("parser_review_summary", canonical_json(review_queue.summary()))
        store.connection.commit()
        receipts = [asdict(source) | {"fingerprint": source.fingerprint} for source in source_list]
        export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=receipts,
            pipeline_fingerprint=identity,
            shard_bytes=shard_bytes,
            chunking=chunking,
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
        write_json(root / "parser-review.json", review_queue.summary() | {"policy": asdict(review)})
    return PreparedDataset(root, split=None)
