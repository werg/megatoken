"""Resumable import -> external-memory dedup -> frozen split -> atomic shards."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from megatoken.io import content_hash
from megatoken.teacher import PINNED_TEACHER

from .adapters import (
    AdapterContext,
    ChatFramingError,
    SchemaError,
    adapt_row,
    source_row_lineage,
    source_row_supervision,
    validate_options,
)
from .chunking import ChunkingConfig
from .compression import CompressionPolicy, DocumentCompressor
from .dedup import DedupConfig, DedupStore
from .lineage import LineageJoin
from .media_assets import TensorAssetStore
from .quality import RowFilterReject, validate_row_filters
from .relations import EXTRACTOR as NAVIGATION_EXTRACTOR
from .review import ParserReviewPolicy, ParserReviewQueue, source_quality, validate_prepared_episode
from .schema import SCHEMA_VERSION, Codec, canonical_json, encode_episode
from .sources import SourceSpec, iter_source_rows, load_source_manifest
from .storage import PreparedDataset, export_prepared

# Rows are committed in batches; a row's writes are all-or-nothing within its batch.
COMMIT_EVERY_ROWS = 512


def prepare_sources(
    sources: Iterable[SourceSpec] | str | Path,
    output: str | Path,
    codec: Codec,
    *,
    dedup: DedupConfig | None = None,
    max_record_bytes: int | None = None,
    max_record_tokens: int | None = None,
    chunking: ChunkingConfig | None = None,
    review: ParserReviewPolicy | None = None,
    shard_bytes: int = 128 << 20,
    benchmark_quarantine: dict[str, Any] | None = None,
    reject_invalid: bool = False,
    compression: CompressionPolicy | None = None,
    teacher_packs: str | Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    ingest_only: bool = False,
    decisions: str | Path | None = None,
) -> PreparedDataset | None:
    """Prepare source rows, checking source identity before resuming any cursor.

    ``max_record_tokens`` / ``max_record_bytes`` set the hard physical-record
    limits (raising or lowering the chunking policy's own); by default the
    policy's limits stand.

    Parser faults/ambiguities are captured in the review queue and excluded from
    accepted shards. A row that a source's declared ``row_filters`` exclude is
    recorded as ``filtered``. Any other invalid row (a schema or quality
    failure) fails the default run; ``reject_invalid=True`` writes explicit
    source/row/error receipts for those and continues, so an operator can inspect
    the rejection distribution before authorizing scale. Errors that are not
    about the row's data (a bug in an adapter) always stop the run.

    ``compression`` reads long observed fields (reference documents, large tool
    results) as teacher features (data/compression.py), with EmbeddingGemma2
    loaded from the ``teacher_packs`` manifest; the features are stored as
    immutable assets under ``output``, which training uses as its media root.

    One run is normally enough. To prepare one manifest's shards on separate machines
    (``data/sharding.py``), run each with ``ingest_only=True``: it reads and deduplicates
    its rows within the shard, leaves ``digest.sqlite`` beside ``preparation.sqlite`` and
    returns ``None``. ``reconcile`` then joins the digests, and running the same call again
    with that shard's ``decisions`` file applies the global verdicts (splits, cross-shard
    duplicates, family isolation) in place of the shard-local freeze and writes the root.
    """
    if (compression is None) != (teacher_packs is None):
        raise ValueError("Compressed reading needs both a policy and the teacher pack manifest")
    if ingest_only and decisions is not None:
        raise ValueError("A shard is either ingested or finished with decisions, not both")
    dedup = dedup or DedupConfig()
    review = review or ParserReviewPolicy()
    if (
        min(
            1 if max_record_bytes is None else max_record_bytes,
            1 if max_record_tokens is None else max_record_tokens,
            shard_bytes,
        )
        < 1
    ):
        raise ValueError("Physical record and shard limits must be positive")
    chunking = (chunking or ChunkingConfig()).with_record_limits(
        max_tokens=max_record_tokens, max_bytes=max_record_bytes
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
        validate_options(source.adapter, source.options)
    quarantine = benchmark_quarantine or {"family_keys": [], "content_sha256": []}
    identity = content_hash(
        {
            "sources": [source.fingerprint for source in source_list],
            "codec": codec.fingerprint,
            "dedup": asdict(dedup),
            "max_record_bytes": max_record_bytes,
            "max_record_tokens": max_record_tokens,
            "chunking": chunking_identity,
            "parser_review": asdict(review),
            "benchmark_quarantine": quarantine,
            "reject_invalid": reject_invalid,
            "compression": None
            if compression is None
            else {
                **asdict(compression),
                "roles": sorted(compression.roles),
                "teacher": PINNED_TEACHER,
            },
            "schema_version": SCHEMA_VERSION,
            "navigation_extractor": NAVIGATION_EXTRACTOR,
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
            compressor = None
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
                if compression is not None and compressor is None:
                    from megatoken.omni.embeddinggemma2 import EmbeddingGemma2

                    teacher = EmbeddingGemma2.from_manifest(teacher_packs)
                    if teacher.extractor != PINNED_TEACHER:
                        raise ValueError("The pack manifest's teacher is not the pinned teacher")
                    compressor = DocumentCompressor(teacher, TensorAssetStore(root), compression)
                rows_in_batch = 0
                try:
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
                            chunking=chunking,
                            compressor=compressor,
                            assets=TensorAssetStore(root),
                        )
                        with store.row_savepoint():
                            try:
                                status, detail = _ingest_row(
                                    store,
                                    review_queue,
                                    source,
                                    row_index,
                                    row,
                                    context,
                                    lineage_join,
                                    join_config,
                                    review=review,
                                    chunking=chunking,
                                    chunking_identity=chunking_identity,
                                )
                            except RowFilterReject as error:
                                status, detail = "filtered", str(error)
                            except (SchemaError, ValueError) as error:
                                if not reject_invalid:
                                    raise ValueError(
                                        f"Source {source.source_id}, row {row_index}: {error}"
                                    ) from error
                                status, detail = "rejected", str(error)
                            except Exception as error:
                                error.add_note(
                                    f"Source {source.source_id}, row {row_index}: this is a "
                                    "pipeline error, not an invalid row, so it is never rejected"
                                )
                                raise
                            store.mark_source(
                                source.fingerprint, row_index, status=status, detail=detail
                            )
                        last = row_index
                        processed += 1
                        rows_in_batch += 1
                        if rows_in_batch >= COMMIT_EVERY_ROWS:
                            store.connection.commit()
                            rows_in_batch = 0
                        if progress and processed % 100 == 0:
                            progress(
                                {"source": source.source_id, "row": row_index, "status": status}
                            )
                finally:
                    # Whatever stopped the loop, every completed row is durable.
                    store.connection.commit()
                with store.transaction():
                    # Empty sources still become durably complete without a fake row.
                    store.connection.execute(
                        "INSERT INTO source_progress VALUES(?,?,1) ON CONFLICT(source_key) DO UPDATE SET next_row=excluded.next_row,complete=1",
                        (source.fingerprint, last + 1),
                    )
            if ingest_only:
                store.write_digest(root / "digest.sqlite")
                return None
            if decisions is None:
                store.freeze()
            else:
                store.apply_decisions(decisions)
        store.set_meta("parser_review_summary", canonical_json(review_queue.summary()))
        store.connection.commit()
        receipts = [asdict(source) | {"fingerprint": source.fingerprint} for source in source_list]
        # Quarantine/reject details are metadata, never concatenated to transcripts.
        # They are written with the data files, before the manifest commits the export.
        rejections = {
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
        export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=receipts,
            pipeline_fingerprint=identity,
            shard_bytes=shard_bytes,
            chunking=chunking,
            sidecars={
                "rejections.json": rejections,
                "parser-review.json": review_queue.summary() | {"policy": asdict(review)},
            },
        )
    return PreparedDataset(root, split=None)


def _ingest_row(
    store: DedupStore,
    review_queue: ParserReviewQueue,
    source: SourceSpec,
    row_index: int,
    row: dict[str, Any],
    context: AdapterContext,
    lineage_join: LineageJoin | None,
    join_config: dict[str, Any] | None,
    *,
    review: ParserReviewPolicy,
    chunking: ChunkingConfig,
    chunking_identity: dict[str, Any],
) -> tuple[str, str]:
    """One source row into the store; returns its status and detail."""
    raw_row = row
    validate_row_filters(row, source.options.get("row_filters", []))
    if lineage_join is not None:
        row = lineage_join.apply(row, join_config.get("row_key", "uuid"))
    try:
        episode = adapt_row(row, source.adapter, context)
    except ChatFramingError as error:
        if not review.enabled:
            raise
        # Even an unparseable frame must register its source ancestry before
        # family splits freeze.
        lineage = source_row_lineage(row, source.adapter, context)
        family = store.register_family(lineage.family_keys)
        supervision = source_row_supervision(row, source.adapter, context)
        if supervision.benchmark_tags:
            store.connection.execute(
                "INSERT OR IGNORE INTO benchmarks VALUES(?, 'family')", (family,)
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
        return "parser_review", detail
    from .relations import with_local_relations

    episode = with_local_relations(episode)
    # A rejected low-quality variant still contributes its known ancestor links
    # before any train/eval split.
    store.register_episode_lineage(episode)
    encoded = encode_episode(episode, context.codec)
    validate_prepared_episode(
        episode,
        encoded,
        source_quality(source),
        max_record_tokens=context.max_record_tokens,
        max_record_bytes=context.max_record_bytes,
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
        return "parser_review", detail
    return store.add(episode, context.codec, encoded), episode.episode_id
