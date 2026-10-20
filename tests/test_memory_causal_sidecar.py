"""Causal source-footprint and teacher-target sidecar release gates."""
from dataclasses import replace
import base64
from hashlib import sha256

import pytest

from megatoken.data.retrieval import RetrievalExample, read_sidecar, write_sidecar
from megatoken.memory.contracts import Footprint, QueryBoundary, SourceRange


def _support(name, *, available_at=0, scope="team"):
    return Footprint((
        SourceRange(
            name, "v1", 0, 10, "sha-source-bytes", scope,
            available_at=available_at, family="fixture",
        ),
    ))


def _example():
    boundary = QueryBoundary(
        "event-1", meridian=3, purpose="source",
        originating_track="main", request_hash="request-digest",
        space_fingerprint="space-digest",
        snapshot_id="scoring-snapshot", deadline=4,
    )
    return RetrievalExample(
        episode_id="episode-1", event_id="event-1", query=boundary,
        query_teacher_inputs=_support("request", available_at=3),
        document_key_context=_support("document", available_at=3),
        optimization_supervision=_support("future-label", available_at=100),
        payload_support=_support("document", available_at=3),
        positive_content_ids=("document/chunk",), support_weights=(1.,),
        annotations_complete=True, source_snapshot="pinned-corpus",
        split_family="family-a", teacher_target=(1., 0.),
    )


def test_query_teacher_cannot_use_future_context():
    with pytest.raises(ValueError, match="Query teacher sees future"):
        replace(_example(), query_teacher_inputs=_support("future", available_at=4))


def test_candidate_key_cannot_use_unavailable_contextual_neighbors():
    with pytest.raises(ValueError, match="not causally eligible"):
        replace(_example(), document_key_context=_support("future", available_at=4))


def test_optimizer_labels_may_use_future_outcomes_but_not_as_query_inputs():
    example = _example()
    assert example.optimization_supervision.ranges[0].available_at == 100
    assert example.query_teacher_inputs.ranges[0].available_at <= example.query.meridian


@pytest.mark.parametrize("target", [(float("nan"), 0.), (float("inf"), 1.), (0., 0.)])
def test_invalid_teacher_target_rejected(target):
    with pytest.raises(ValueError, match="finite and nonzero"):
        replace(_example(), teacher_target=target)


def test_retrieval_sidecar_round_trip_preserves_source_support_and_digest(tmp_path):
    path = tmp_path / "retrieval.jsonl"
    example = _example()
    write_sidecar(path, [example])
    assert read_sidecar(path) == [example]


def test_explicit_query_request_bytes_are_hashed_and_replayable():
    message = b"Find the original fact"
    old = _example()
    annotated = replace(
        old,
        query=replace(old.query, request_hash=sha256(message).hexdigest()),
        main_request_b64=base64.b64encode(message).decode("ascii"),
    )
    assert annotated.request_content() == message
    with pytest.raises(ValueError, match="causal request hash"):
        replace(annotated, main_request_b64=base64.b64encode(b"hidden answer").decode("ascii"))
    with pytest.raises(ValueError, match="base64"):
        replace(annotated, main_request_b64="not*base64")
    with pytest.raises(ValueError, match="requires pinned main-request bytes"):
        old.request_content()
