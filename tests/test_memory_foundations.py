"""Focused CPU acceptance gates for the independent source/KB retrieval contract."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from megatoken.generation.embedding import EmbeddingRealizer, EmbeddingSpec
from megatoken.memory.contracts import (
    Collection, Footprint, QueryBoundary, SemanticSpace, SourceRange, Visibility,
)
from megatoken.memory.context import admit_kb
from megatoken.memory.growth import CatalogChunk, GrowthAllocator, GrowthPolicy
from megatoken.memory.index import CoarseIVF, ExactIndex
from megatoken.memory.relations import pair_evidence
from megatoken.memory.retrieval import Interval, RetrievalSession, subtract
from megatoken.memory.semantic import AdmittedGroup, Constituent
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model.config import ModelConfig
from megatoken.model.core import FacetedCore
from megatoken.model.multitrack import MultiTrackCore


def space():
    return SemanticSpace("kb-space", "v1", "pplx-pilot", "pinned-weight-sha",
                         8, "doc-format-v1", "query-format-v1", "ordered-chunks")


def support(source="document", *, start=0, end=8, available_at=0, scope="team"):
    return Footprint((SourceRange(
        source, "version1", start, end, "sha256-content", scope,
        available_at=available_at, family="project-a",
    ),))


def test_visibility_requires_all_context_not_just_returned_chunk():
    visible = Visibility("team", committed_cursor=5)
    full = Footprint((
        *support().ranges,
        *support("future", available_at=10).ranges,
    ))
    assert visible.permits(support())
    assert not visible.permits(full)
    assert not Visibility("elsewhere", 100).permits(support())
    assert not Visibility("team", 10, excluded_families=frozenset({"project-a"})
                          ).permits(support())


def test_sqlite_rollback_reopen_generations_moments_and_snapshot(tmp_path):
    path = tmp_path / "bank.sqlite"
    with SQLiteKB(path, space(), value_dim=16) as kb:
        a = kb.insert_source(
            scope="team", footprint=support(), content_id="original-a",
            asset_ref="source/chunk/000", key=np.ones(8),
        )
        m = kb.insert_memory(
            scope="team", footprint=support(), assignment="original-a",
            key=np.ones(8), value=np.zeros(16), birth_step=0,
        )
        assert kb.get(Collection.SOURCE, a).value is None
        assert kb.get(Collection.MEMORY, m).value.shape == (16,)
        first = kb.revisions()
        with pytest.raises(RuntimeError, match="inject rollback"):
            with kb.transaction():
                kb.insert_memory(
                    scope="team", footprint=support(), assignment="would-roll-back",
                    key=np.ones(8), value=np.zeros(16), birth_step=0,
                )
                raise RuntimeError("inject rollback")
        assert len(kb.scan(Collection.MEMORY, "team")) == 1
        changed = kb.update_record(
            Collection.MEMORY, m, value=np.ones(16),
            expected_payload_revision=1,
        )
        assert changed.payload_revision == 2
        assert kb.revisions().key == first.key
        assert kb.revisions().payload > first.payload
        kb.write_moments(
            collection=Collection.MEMORY, kind="record", entity_id=m,
            field="value", layout_generation="semantic-v1",
            first=np.ones(16), second=np.ones(16), updates=1, step=9,
            row_revision=2,
        )
        assert kb.moments(collection=Collection.MEMORY, kind="record",
                          entity_id=m, field="value",
                          layout_generation="semantic-v1")[2:] == (1, 9, 2)
        kb.backup(tmp_path / "snapshot.sqlite")
    with SQLiteKB(tmp_path / "snapshot.sqlite", space(), value_dim=16,
                  readonly=True) as snapshot:
        assert snapshot.get(Collection.MEMORY, m).payload_revision == 2
        with pytest.raises(PermissionError):
            snapshot.update_record(Collection.MEMORY, m, value=np.zeros(16))



def test_source_version_tombstone_cascades_and_is_idempotent(tmp_path):
    with SQLiteKB(tmp_path / "revocation.sqlite", space(), value_dim=16) as kb:
        old = kb.insert_memory(
            scope="team", footprint=support("retired"), assignment="retired",
            key=np.ones(8), value=np.zeros(16), birth_step=0,
        )
        still_valid = kb.insert_memory(
            scope="team", footprint=support("unrelated"), assignment="unrelated",
            key=np.arange(1, 9), value=np.zeros(16), birth_step=0,
        )
        # Deliberately omit retired context from the pair support to emulate a
        # prior incomplete relation schema. Endpoint retirement still revokes.
        kb.insert_relation(
            scope="team", left=old, right=still_valid, prior=0.5,
            value=np.zeros(16), dependence_logits=(-2., -2.),
            footprint=support("unrelated"),
        )
        kb.update_record(
            Collection.MEMORY, old, posting_cell=0,
            centroids_revision="pinned-centroids",
        )
        assert kb.db.execute(
            "SELECT count(*) FROM postings WHERE record_id=?", (old,)
        ).fetchone()[0] == 1
        previous = kb.revisions().index
        retired = kb.tombstone_source_version("retired", "version1")
        assert retired == {"source": 0, "memory": 1, "relation": 1}
        assert kb.revisions().index > previous
        assert not kb.relations("team")
        assert [r.id for r in kb.scan(Collection.MEMORY, "team")] == [still_valid]
        assert kb.db.execute(
            "SELECT count(*) FROM postings WHERE record_id=?", (old,)
        ).fetchone()[0] == 0
        epoch = kb.revisions()
        assert kb.tombstone_source_version("retired", "version1") == {
            "source": 0, "memory": 0, "relation": 0,
        }
        assert kb.revisions() == epoch



def test_retired_version_cannot_be_reallocated_via_old_pair_window(tmp_path):
    with SQLiteKB(tmp_path / "retired.sqlite", space(), value_dim=16) as kb:
        obsolete = support("obsolete")
        valid = support("still-valid")
        left = kb.insert_memory(
            scope="team", footprint=obsolete, assignment="old-a",
            key=np.ones(8), value=np.zeros(16), birth_step=0,
        )
        right = kb.insert_memory(
            scope="team", footprint=valid, assignment="old-b",
            key=np.arange(1, 9), value=np.zeros(16), birth_step=0,
        )
        kb.tombstone_source_version("obsolete", "version1")
        assert kb.footprint_revoked(obsolete)
        with pytest.raises(ValueError, match="Retired"):
            kb.insert_memory(
                scope="team", footprint=obsolete, assignment="new-a",
                key=np.ones(8), value=np.zeros(16), birth_step=1,
            )
        with pytest.raises(ValueError, match="retired"):
            kb.insert_relation(
                scope="team", left=left, right=right, prior=0.5,
                value=np.zeros(16), dependence_logits=(0., 0.),
                footprint=valid,
            )
        kb.put_growth_state("readiness", {"coherence": True, "embeddings": True})
        growth = GrowthAllocator(
            kb, GrowthPolicy(value_dim=16, seed=5, min_pair_windows=2),
        )
        assert growth.admit_pairs({
            "window1": ("old-a", "old-b"),
            "window2": ("old-a", "old-b"),
        }, scope="team", visibility=Visibility("team", 10)) == ()


def test_exact_and_ivf_filter_before_truncation(tmp_path):
    with SQLiteKB(tmp_path / "kb.sqlite", space(), value_dim=16) as kb:
        allowed = kb.insert_source(
            scope="team", footprint=support("allowed"), content_id="yes",
            asset_ref="allowed", key=[1] + [0] * 7,
        )
        kb.insert_source(
            scope="team", footprint=support("forbidden", available_at=100),
            content_id="no", asset_ref="forbidden", key=[1] + [0] * 7,
        )
        v = Visibility("team", committed_cursor=10)
        query = np.array([1] + [0] * 7, dtype=np.float32)
        exact = ExactIndex(kb).search(
            query, collection=Collection.SOURCE, scope="team",
            visibility=v, limit=1,
        )
        assert exact.ids == (allowed,)
        ivf = CoarseIVF(kb, centroids=np.stack([query, -query]), probes=1).search(
            query, collection=Collection.SOURCE, scope="team",
            visibility=v, limit=2,
        )
        assert ivf.ids == (allowed,) and ivf.shortfall == 1



def test_maxsim_deduplicates_byte_coverage_across_chunk_aliases(tmp_path):
    from megatoken.memory.base import Hit

    with SQLiteKB(tmp_path / "aliases.sqlite", space(), value_dim=16) as kb:
        a = kb.insert_source(
            scope="team", footprint=support("physical-document"),
            content_id="original/chunk-a", asset_ref="document-v1",
            key=np.ones(8),
        )
        b = kb.insert_source(
            scope="team", footprint=support("physical-document"),
            content_id="alias/chunk-b", asset_ref="document-v1",
            key=np.ones(8),
        )
        first = kb.get(Collection.SOURCE, a)
        second = kb.get(Collection.SOURCE, b)
        receipt = QueryBoundary(
            "retrieval-1", 0, "source", "main", "request",
            space().fingerprint, "scoring-epoch-1", 4,
        )
        coverage = RetrievalSession()
        obligations = coverage.incremental(
            [Hit(first, 0.8), Hit(second, 0.81)],
            boundary=receipt, threshold=0.5,
        )
        assert len(obligations) == 1
        assert obligations[0].interval.content_id == "physical-document"
        picked = coverage.reserve(
            max_bytes=128, max_events=4, now=1, request_id="read",
        )
        assert len(picked) == 1
        coverage.commit(
            picked[0].interval, request_id="read", receipt="native-meridian-1",
        )
        assert not coverage.incremental(
            [Hit(second, 0.95)], boundary=receipt, threshold=0.5,
        )


def test_incremental_coverage_deferred_retries_and_score_epoch(tmp_path):
    from megatoken.memory.base import Hit
    with SQLiteKB(tmp_path / "kb.sqlite", space(), value_dim=16) as kb:
        identity = kb.insert_source(
            scope="team", footprint=support(), key=np.ones(8),
            content_id="doc-hash", asset_ref="source",
        )
        row = kb.get(Collection.SOURCE, identity)
        s = RetrievalSession()
        boundary = QueryBoundary("event1", 1, "source", "main", "hash",
                                 space().fingerprint, "score-epoch-1", 4)
        qualified = s.incremental([Hit(row, 0.9)], boundary=boundary, threshold=0.7)
        assert len(qualified) == 1
        assert s.incremental([Hit(row, 0.95)], boundary=boundary, threshold=0.7) == []
        reserved = s.reserve(max_bytes=32, max_events=1, now=1, request_id="fetch")
        assert reserved == tuple(qualified)
        assert s.reserve(max_bytes=32, max_events=1, now=1, request_id="other") == ()
        s.release(reserved[0].interval, "fetch")
        assert s.reserve(max_bytes=32, max_events=1, now=2, request_id="fetch2")
        s.commit(reserved[0].interval, request_id="fetch2", receipt="native-C-commit")
        s.new_epoch("different-key-epoch")
        assert not s.maxima and len(s.injected) == 1
        assert not s.incremental(
            [Hit(row, 0.98)], boundary=QueryBoundary(
                "event2", 2, "source", "main", "hash",
                space().fingerprint, "different-key-epoch", 5,
            ), threshold=0.7,
        )
    remaining = subtract(Interval("a", "v1", 0, 30),
                         [Interval("a", "v1", 5, 10), Interval("a", "v1", 12, 25)])
    assert [(x.start, x.end) for x in remaining] == [(0, 5), (10, 12), (25, 30)]


def test_demand_bootstrap_preserves_birth_and_existing_values(tmp_path):
    with SQLiteKB(tmp_path / "kb.sqlite", space(), value_dim=16) as kb:
        growth = GrowthAllocator(kb, GrowthPolicy(value_dim=16, seed=21, max_points=3))
        available = Visibility("team", 1)
        catalog = [
            CatalogChunk(f"source-{i}", np.eye(8, dtype=np.float32)[i],
                         "team", "dense" if i < 2 else "tail",
                         support(f"source-{i}"), f"family-{i}")
            for i in range(3)
        ]
        with pytest.raises(RuntimeError, match="coherence"):
            growth.grow(catalog, visibility=available, step=1, budget=1)
        kb.put_growth_state("readiness", {"coherence": True, "embeddings": True})
        assert growth.demand.observe(event_id="query1", region="dense",
                                     family="family-a", step=1)
        assert not growth.demand.observe(event_id="query1", region="dense",
                                         family="family-a", step=1)
        ids = growth.grow(catalog, visibility=available, step=2, budget=2)
        assert len(ids) == 2
        old = kb.get(Collection.MEMORY, ids[0]).value.copy()
        growth.grow(catalog, visibility=available, step=3, budget=1)
        assert np.array_equal(kb.get(Collection.MEMORY, ids[0]).value, old)
        assert len(kb.scan(Collection.MEMORY, "team")) == 3



def test_dense_growth_repeats_high_demand_region_before_exploration(tmp_path):
    with SQLiteKB(tmp_path / "density.sqlite", space(), value_dim=16) as kb:
        policy = GrowthPolicy(
            value_dim=16, seed=7, dense_fraction=0.75, max_points=20,
        )
        growth = GrowthAllocator(kb, policy)
        kb.put_growth_state("readiness", {"coherence": True, "embeddings": True})
        for i in range(10):
            assert growth.demand.observe(
                event_id=f"hot-query-{i}", region="high-demand",
                family=f"family-{i}", step=0,
            )
        catalog = [
            CatalogChunk(
                f"hot-{i}", np.eye(8, dtype=np.float32)[i % 8],
                "team", "high-demand", support(f"hot-{i}"), f"hot-family-{i}",
            ) for i in range(5)
        ] + [
            CatalogChunk(
                f"cold-{i}", np.eye(8, dtype=np.float32)[i % 8],
                "team", "low-demand", support(f"cold-{i}"), f"cold-family-{i}",
            ) for i in range(5)
        ]
        ids = growth.grow(
            catalog, visibility=Visibility("team", committed_cursor=10),
            step=1, budget=4,
        )
        assignments = [kb.get(Collection.MEMORY, i).source_assignment for i in ids]
        assert assignments[:3] == ["hot-0", "hot-1", "hot-2"]
        assert assignments[3] == "cold-0"


def test_correlations_absent_checked_partner_attenuates_only_correct_endpoint():
    base = torch.tensor([0.5, 0.7], requires_grad=True)
    logits = torch.tensor([[1.0, 2.0]], requires_grad=True)
    inputs = dict(
        left=torch.tensor([0]), right=torch.tensor([1]),
        dependence_logits=logits, prior=torch.ones(1),
        pair_strength=torch.ones(1),
        left_partner=torch.zeros(1), right_partner=torch.ones(1),
        left_checked=torch.ones(1), right_checked=torch.ones(1),
        distinct_support=torch.ones(1),
        beta=0.5,
    )
    out = pair_evidence(base, **inputs)
    assert out.point[0] < base[0]
    assert torch.allclose(out.point[1], base[1])
    assert out.pair[0] > 0
    (out.point.sum() + out.pair.sum()).backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    inputs.update(left_checked=torch.zeros(1), distinct_support=torch.zeros(1))
    unknown = pair_evidence(base.detach(), **inputs)
    assert torch.allclose(unknown.point, base.detach())
    assert torch.equal(unknown.pair, torch.zeros_like(unknown.pair))


def test_causal_embedding_realizer_sees_all_facets_and_role_asymmetry():
    r = EmbeddingRealizer(4, 16, 8)
    raw = torch.randn(2, 4, 16, requires_grad=True)
    query = r(raw, EmbeddingSpec("teacher-v1", "query", 8))
    document = r(raw, EmbeddingSpec("teacher-v1", "document", 8))
    assert query.shape == (2, 8)
    assert torch.allclose(query.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert not torch.allclose(query, document)
    (query[:, 0] + document[:, 0]).sum().backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0


def test_admission_keeps_native_frontier_and_allows_full_kv_gradients():
    model = FacetedCore(ModelConfig.tiny(
        semantic_dim=8, semantic_value_dim=16, max_kb_positions=2
    ))
    tracks = MultiTrackCore(model)
    original = tracks.initial_state(1)
    semantic = torch.randn(1, 16, requires_grad=True)
    coefficient = torch.tensor([0.8], requires_grad=True)
    constituent = Constituent(
        "point:1", semantic, coefficient,
        torch.tensor([-6.0], requires_grad=True), 0, "payload-v1",
    )
    group = AdmittedGroup("group:1", (constituent,), 0, "admission-v1", "scope:1")
    admitted = admit_kb(original, group)
    assert torch.equal(admitted.raw, original.raw)
    assert torch.equal(admitted.meridians_seen, original.meridians_seen)
    assert torch.equal(admitted.events_seen, original.events_seen)
    assert admitted.history.slots == original.history.slots == 0
    materialized = tracks.materialize_memory(admitted)
    assert materialized is not None
    assert materialized.features.keys.shape == (1, 2, 4, 8)
    assert materialized.features.mask.all()
    projected = tracks.predict_group(admitted, tracks=("main",), effort=1)
    projected.square().mean().backward()
    assert semantic.grad is not None and semantic.grad.abs().sum() > 0
    assert coefficient.grad is not None and coefficient.grad.abs().sum() > 0
    assert model.memory_projector.to_kv.weight.grad is not None
    assert model.memory_adapters.controller.weight.grad is not None
