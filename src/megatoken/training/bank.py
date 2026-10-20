"""Reference native KB training owner: eligible ANN IDs, live sparse leaves, KV admission.

Discovery and source-byte delivery remain discrete. Native continuation through the
returned state trains the selected query/key/value/projector/adapter path. The
SQLite writer is the sole sparse-row authority; its transactions never replace
the model's MasterAdamW optimizer or duplicate neural optimizer moments.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from megatoken.data.retrieval import RetrievalExample
from megatoken.memory.context import admit_kb
from megatoken.memory.bilateral import selected_joint_groups
from megatoken.memory.contracts import Collection, Visibility
from megatoken.memory.growth import CatalogChunk, GrowthAllocator
from megatoken.memory.index import ExactIndex
from megatoken.memory.optimizer import RowID, SparseRowAdam
from megatoken.memory.reader import selected_evidence
from megatoken.memory.retrieval import RetrievalSession
from megatoken.memory.recipe import RetrievalRecipe
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model.core import ROLE_IDS
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.training.memory import (
    LocalMemoryWindow, LocalRetrievalTarget, annotated_ranking_loss,
)
from megatoken.training.source_features import NativeSourceFeatures


def semantic_region(key: np.ndarray) -> str:
    """Stable coarse catalog partition independent of free semantic VALUES."""
    vector = np.asarray(key, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or vector.size == 0:
        raise ValueError("Demand partition requires a finite semantic address")
    index = int(np.argmax(np.abs(vector)))
    return f"{index}:{int(vector[index] >= 0)}"


@dataclass(frozen=True)
class ReadResult:
    state: Any
    loss: Tensor
    source_hits: int
    memory_hits: int
    memory_groups: int
    source_value_examples: int = 0
    source_committed: int = 0
    source_bytes: int = 0


class BankTrainingOwner:
    """Per-rank CPU reference; shared SQLite has ONE authoritative writer.

    Calls to read() are made immediately BEFORE each annotated native
    supervised meridian. No future source or teacher target is passed to ANN.
    """

    def __init__(self, bank: SQLiteKB, recipe: RetrievalRecipe, *,
                 device: torch.device, sparse_lr: float = 1e-3,
                 source_features: NativeSourceFeatures | None = None):
        if bank.space.fingerprint != recipe.space.fingerprint or (
            bank.value_dim != recipe.model.semantic_value_dim
        ):
            raise ValueError("Bank/training semantic layout mismatch")
        if recipe.phase not in {"progressive_warmup", "free_kb_pretraining"}:
            raise ValueError("Bank updates are allowed only in free-KB pretraining phases")
        if recipe.loss_weights.get("source_value", 0) and source_features is None:
            raise ValueError("Source-value training requires actual detached native E(source) bytes")
        self.bank = bank
        self.source_features = source_features
        self.recipe = recipe
        self.device = device
        self.index = ExactIndex(bank)
        self.sparse = SparseRowAdam(bank, lr=sparse_lr)
        self.selected = LocalMemoryWindow(self.sparse, "semantic-v1", device)
        self.growth = GrowthAllocator(bank, recipe.growth)
        self.query_demand = 0
        self.pending_demand: list[tuple[str, str, str, int]] = []
        self.reads = 0

    def _rank(self, query: Tensor, ids: tuple[int, ...],
              *, collection: Collection, positive: Tensor,
              complete: bool) -> Tensor:
        # A missed annotated positive is omitted, never trained as no-answer.
        if not ids:
            return query.sum() * 0
        keys = torch.stack([
            self.selected.field(collection, "record", identity, "key")
            for identity in ids
        ]).to(query.device)
        scores = selected_evidence(query, keys[None].expand(query.shape[0], -1, -1))
        numerator, valid_count = annotated_ranking_loss(
            scores, positive[None], torch.tensor([complete], device=query.device),
        )
        return numerator if valid_count else query.sum() * 0

    def read(self, state: Any, query: Tensor, annotation: RetrievalExample,
             *, global_count: int, multi: MultiTrackGenerator | None = None,
             coverage: RetrievalSession | None = None) -> ReadResult:
        if query.shape != (1, self.recipe.space.dimension) or global_count < 1:
            raise ValueError("Training KB read requires one query and global event count")
        if annotation.query.space_fingerprint != self.recipe.space.fingerprint:
            raise ValueError("Query/KV semantic-space revision mismatch")
        scope = annotation.query_teacher_inputs.ranges[0].scope
        # Deadline is a delivery limit, not permission to see future evidence.
        # Selected inputs must be available at the actual native boundary.
        committed = int(state.meridians_seen[0])
        if annotation.query.meridian > committed:
            raise ValueError("KB query cannot see a future native meridian")
        if multi is not None and committed != annotation.query.meridian + 1:
            raise ValueError(
                "Training retrieval requires a committed P query immediately "
                "after its recorded native boundary"
            )
        visibility = Visibility(scope, min(annotation.query.meridian, committed))
        q = query.detach().float().cpu().numpy()[0]
        source_hits = self.index.search(
            q, collection=Collection.SOURCE, scope=scope,
            visibility=visibility, limit=self.recipe.max_selected_source,
        )
        memory_hits = self.index.search(
            q, collection=Collection.MEMORY, scope=scope,
            visibility=visibility, limit=self.recipe.max_selected_memory,
        )
        positives = set(annotation.positive_content_ids)
        # Ranking labels apply only to truly eligible, selected source IDs.
        s_ids = source_hits.ids
        s_positive = torch.tensor([
            hit.record.content_id in positives for hit in source_hits.hits
        ], device=self.device, dtype=torch.bool)
        s_loss = self._rank(
            query, s_ids, collection=Collection.SOURCE, positive=s_positive,
            complete=annotation.annotations_complete and not positives,
        )
        source_value = query.sum() * 0
        source_value_examples = 0
        if s_ids and self.source_features is not None:
            observed_features = torch.stack([
                self.source_features.get(hit.record) for hit in source_hits.hits
            ])[None].to(query.device)
            target = LocalRetrievalTarget(
                teacher_query=query.detach(),
                useful_source=s_positive[None],
                annotated_complete=torch.tensor(
                    [annotation.annotations_complete and not positives],
                    device=query.device, dtype=torch.bool,
                ),
                source_features=observed_features.detach(),
            )
            source_value, source_value_examples = self.selected.source_loss(
                query, s_ids, target,
                eligible=torch.ones(
                    (1, len(s_ids)), dtype=torch.bool, device=query.device,
                ),
            )
        # Memory candidates reference STABLE ORIGINAL source assignments.
        m_hits = memory_hits.hits
        m_ids = tuple(hit.record.id for hit in m_hits)
        m_positive = torch.tensor([
            hit.record.source_assignment in positives for hit in m_hits
        ], device=self.device, dtype=torch.bool)
        m_loss = self._rank(
            query, m_ids, collection=Collection.MEMORY, positive=m_positive,
            complete=annotation.annotations_complete and not positives,
        )
        groups_count = 0
        if m_ids:
            groups, _ = selected_joint_groups(
                bank=self.bank, selected=self.selected, query=query,
                records=[hit.record for hit in m_hits],
                visibility=visibility, annotation=annotation,
                anchor=int(state.meridians_seen[0]),
                max_groups=self.recipe.max_kb_positions,
            )
            for group in groups:
                state = admit_kb(state, group)
            groups_count = len(groups)
        # Demand is counted from the canonical SOURCE catalog, before source
        # coverage or pair/ranking bonuses. Empty catalogs cannot fabricate KB.
        if source_hits.hits:
            region = semantic_region(source_hits.hits[0].record.key)
            # Defer writes to the SINGLE primary-rank update barrier; queries
            # on workers may not mutate the canonical SQLite demand catalog.
            self.pending_demand.append((
                f"{annotation.episode_id}/{annotation.event_id}",
                region, annotation.split_family, annotation.query.meridian,
            ))
            self.query_demand += 1
        committed_sources = 0
        committed_bytes = 0
        if multi is not None:
            if coverage is None or self.source_features is None:
                raise RuntimeError(
                    "Native RAG training requires a per-episode MaxSim ledger "
                    "and immutable source byte reader"
                )
            epochs = self.bank.revisions()
            scoring_epoch = epochs.scoring_identity(
                f"native-p-r:{state.model_version}", self.recipe.space,
            )
            boundary = replace(annotation.query, snapshot_id=scoring_epoch)
            coverage.incremental(
                list(source_hits.hits), boundary=boundary,
                threshold=float(self.recipe.extra_rules.get("source_threshold", 0.35)),
            )
            reserved = coverage.reserve(
                max_bytes=self.recipe.max_source_bytes,
                max_events=min(4, self.recipe.max_selected_source),
                now=annotation.query.meridian,
                request_id=f"{annotation.episode_id}/{annotation.event_id}",
            )
            for obligation in reserved:
                row = self.bank.get(Collection.SOURCE, obligation.record_id)
                if not row.active or not visibility.permits(row.footprint):
                    coverage.release(
                        obligation.interval, f"{annotation.episode_id}/{annotation.event_id}",
                    )
                    raise ValueError("Source became ineligible before native RAG admission")
                data = self.source_features.read_interval(row, obligation.interval)
                state = multi.tracks.observe_group(
                    state,
                    (ObservedEvent(
                        "retrieved_source", "text", data,
                        role_id=ROLE_IDS["document"],
                    ),),
                    encoder_grad=False,
                )
                coverage.commit(
                    obligation.interval,
                    request_id=f"{annotation.episode_id}/{annotation.event_id}",
                    receipt=(
                        f"native-rag:{annotation.episode_id}/{annotation.event_id}:"
                        f"{obligation.interval.content_id}:{obligation.interval.version}:"
                        f"{obligation.interval.start}:{obligation.interval.end}"
                    ),
                )
                committed_sources += 1
                committed_bytes += len(data)
        self.reads += 1
        scale = self.recipe.loss_weights.get("query_ranking", 0.)
        loss = (scale * (s_loss + m_loss) + (
            self.recipe.loss_weights.get("source_value", 0.) * source_value
        )) / global_count
        return ReadResult(
            state, loss, len(s_ids), len(m_ids), groups_count,
            source_value_examples, committed_sources, committed_bytes,
        )

    def catalog(self, scope: str) -> list[CatalogChunk]:
        """Source records are the allocation catalog, never model outputs."""
        catalog = []
        for row in self.bank.scan(Collection.SOURCE, scope):
            if not row.active:
                continue
            catalog.append(CatalogChunk(
                content_id=row.content_id, key=row.key, scope=row.scope,
                region=semantic_region(row.key), footprint=row.footprint,
                family=row.footprint.ranges[0].family or row.content_id,
                group=row.group,
            ))
        return catalog

    def grow_at_barrier(self, *, scopes: tuple[str, ...],
                        step: int, budget: int) -> int:
        """Only after one completed, globally synchronized optimizer step."""
        if budget < 0:
            raise ValueError("Growth must have a nonnegative, bounded budget")
        from megatoken.memory.contracts import Visibility
        n = 0
        for scope in sorted(set(scopes)):
            # Historical prepared training provenance stays eligible here;
            # serving never uses this permissive barrier visibility.
            rows = self.catalog(scope)
            if not rows:
                continue
            cursor = max(r.available_at for c in rows for r in c.footprint.ranges)
            n += len(self.growth.grow(
                rows, visibility=Visibility(scope, cursor), step=step,
                budget=max(0, budget - n),
            ))
            if n >= budget:
                break
        return n


def synchronize_sparse_rows(owner: BankTrainingOwner, distributed: Any, *,
                            optimizer_step: int, max_norm: float) -> int:
    """Reference all-rank union/SUM of touched sparse rows. Never average.

    Gradients already use true global evidence denominators. Keep a sorted
    canonical ID/field identity to avoid collisions of record/pair IDs.
    Only rank zero writes SQLite; all other ranks discard stale leaves and
    wait for the authoritative transaction to commit.
    """
    local: dict[tuple[str, str, int, str, str], list[float]] = {}
    for row, leaf in owner.sparse.leaves.items():
        if leaf.grad is None:
            continue
        key = (row.collection.value, row.kind, row.id, row.field, row.layout)
        values = leaf.grad.detach().float().cpu().reshape(-1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite local selected KB gradient")
        local[key] = values.tolist()
    gathered = distributed.all_gather_object(local)
    changed = 0
    error = None
    if distributed.is_primary:
        try:
            combined: dict[tuple[str, str, int, str, str], np.ndarray] = {}
            for rank_values in gathered:
                for key, values in rank_values.items():
                    v = np.asarray(values, dtype=np.float32)
                    if key in combined:
                        if combined[key].shape != v.shape:
                            raise RuntimeError("Sparse gradient row geometry differs by rank")
                        combined[key] += v
                    else:
                        combined[key] = v.copy()
            # One transaction writes every touched sparse field and its moments.
            for key, grad in sorted(combined.items()):
                identity = RowID(Collection(key[0]), key[1], key[2], key[3], key[4])
                leaf = owner.sparse.select(identity)
                if leaf.numel() != grad.size or not np.isfinite(grad).all():
                    raise ValueError("Aggregated sparse gradient shape/nonfiniteness")
                leaf.grad = torch.from_numpy(grad.copy()).reshape_as(leaf)
            changed = owner.sparse.step(step=optimizer_step, max_norm=max_norm)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        owner.sparse.leaves.clear()
        owner.sparse._selected_revisions.clear()
    # All ranks receive publication failure instead of hanging in a barrier.
    failures = distributed.all_gather_object(error)
    if any(x is not None for x in failures):
        raise RuntimeError(f"Sparse KB publication failed on primary: {failures}")
    distributed.barrier()
    return changed


def synchronize_query_demand(owner: BankTrainingOwner, distributed: Any) -> int:
    """Deduplicate global query demand at the same single-writer step barrier."""
    local = tuple(owner.pending_demand)
    gathered = distributed.all_gather_object(local)
    written = 0
    error = None
    if distributed.is_primary:
        try:
            with owner.bank.transaction():
                for batch in gathered:
                    for event_id, region, family, cursor in batch:
                        written += int(owner.growth.demand.observe(
                            event_id=event_id, region=region,
                            family=family, step=cursor,
                        ))
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = distributed.all_gather_object(error)
    if any(errors):
        raise RuntimeError(f"Query demand publication failed: {errors}")
    owner.pending_demand.clear()
    distributed.barrier()
    return written
