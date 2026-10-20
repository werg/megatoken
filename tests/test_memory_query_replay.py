"""Exact retrieval-query replay and failed-boundary atomicity."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.memory.contracts import Footprint, SemanticSpace, SourceRange, Visibility
from megatoken.memory.index import ExactIndex
from megatoken.memory.replay import QueryActionReceipt, replay_query_action
from megatoken.memory.retrieval import RetrievalService, RetrievalSession
from megatoken.memory.runtime import NativeRetrieval, RetrievalPolicy
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime


def _generator():
    torch.manual_seed(7)
    core = FacetedCore(ModelConfig.tiny(
        semantic_dim=8, semantic_value_dim=16, max_kb_positions=2,
    ))
    return MultiTrackGenerator(FacetedGenerator(core, ByteCodec.byte_fallback()))


def test_selected_query_tape_replay_reproduces_kv_vector_and_policy_density():
    generator = _generator()
    original = generator.tracks.initial_state(1)
    query, served, raw = generator.retrieve_embedding(
        original, main_request=b"Locate design decision",
        purpose="source", space_fingerprint="pinned-space",
        effort=1, generator=torch.Generator().manual_seed(31),
    )
    tape = QueryActionReceipt.capture(
        event_id="lookup-1", meridian=0,
        model_version=generator.generator.core.model_version,
        request=b"Locate design decision", purpose="source",
        space_fingerprint="pinned-space", selected=raw,
        vector=query, effort=1, snapshot_id="epoch-1",
    )
    parsed = QueryActionReceipt.from_json(tape.to_json())
    reconstructed, query_again, policy_log_prob = replay_query_action(
        generator, original, parsed,
    )
    assert torch.isfinite(policy_log_prob)
    torch.testing.assert_close(query_again, query)
    torch.testing.assert_close(reconstructed.history.keys, served.history.keys)
    torch.testing.assert_close(reconstructed.history.values, served.history.values)
    assert reconstructed.meridians_seen.item() == 1
    with pytest.raises(ValueError, match="model version"):
        replay_query_action(generator, original, replace(tape, model_version=17))


def test_incremental_source_coverage_snapshot_roundtrips_and_rejects_corruption(tmp_path):
    from megatoken.memory.base import Hit
    from megatoken.memory.contracts import Collection, QueryBoundary
    from megatoken.memory.retrieval import Interval

    space = SemanticSpace(
        "shared", "v1", "teacher", "pinned", 8, "doc", "query", "context",
    )
    support = Footprint((SourceRange(
        "source", "v1", 0, 12, "sha", "team",
    ),))
    with SQLiteKB(tmp_path / "bank.sqlite", space, value_dim=16) as bank:
        row_id = bank.insert_source(
            scope="team", footprint=support, content_id="source-0",
            asset_ref="source-0.bytes", key=np.ones(8),
        )
        row = bank.get(Collection.SOURCE, row_id)
        coverage = RetrievalSession()
        boundary = QueryBoundary(
            "event-1", 0, "source", "main", "request",
            space.fingerprint, "score-1", 5,
        )
        coverage.incremental([Hit(row, 0.9)], boundary=boundary, threshold=0.5)
        selected = coverage.reserve(
            max_bytes=12, max_events=1, now=1, request_id="fetch",
        )
        assert len(selected) == 1
        revived = RetrievalSession.from_dict(coverage.to_dict())
        assert revived.to_dict() == coverage.to_dict()
        item = selected[0].interval
        revived.commit(item, request_id="fetch", receipt="native-commit")
        assert revived.injected == [item]
        assert not coverage.injected
        snapshot = revived.to_dict()
        snapshot["injected"].append(snapshot["injected"][0])
        with pytest.raises(ValueError, match="Overlapping"):
            RetrievalSession.from_dict(snapshot)


def test_failed_search_does_not_publish_query_or_kb_session_state(tmp_path):
    generator = _generator()
    runtime = MultiTrackRuntime(generator)
    session = runtime.open()
    before = session.state
    space = SemanticSpace(
        "space", "v1", "teacher", "pinned", 8, "doc", "query", "context",
    )
    with SQLiteKB(tmp_path / "bank.sqlite", space, value_dim=16) as bank:
        class FailingIndex(ExactIndex):
            def search(self, *args, **kwargs):
                raise RuntimeError("injected ANN failure")

        retrieval = NativeRetrieval(
            generator, runtime, RetrievalService(FailingIndex(bank)),
            space, RetrievalPolicy(),
        )
        with pytest.raises(RuntimeError, match="injected ANN failure"):
            retrieval.boundary(
                session, request=b"search", purpose="source",
                visibility=Visibility("team", 0), request_id="request-1",
                effort=1,
            )
        assert session.state is before
        assert session.revision == 0
        assert not session.retrieval_receipts
        assert not retrieval.ledger(session).maxima
