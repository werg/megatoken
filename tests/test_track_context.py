"""Track-local visibility, real KV reclamation, and exact context-policy replay."""

from dataclasses import replace

import pytest
import torch

from megatoken.controls import build_control_payload
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.context import ContextPolicy
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.policy_context import reconstruct
from megatoken.model.state import KVHistory


def system(local=3, cross=1):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = build_model(
        ModelConfig.tiny(
            max_history_chunks=32, context_local_events=local, context_cross_events=cross
        ),
        ByteCodec.byte_fallback(),
    )
    multi = MultiTrackGenerator.of(model)
    return model, multi, multi.tracks.initial_state(1)


def observe(multi, state, track, address):
    return multi.tracks.observe_group(
        state, (ObservedEvent(track, "text", address.encode(), event_id=address),)
    )


def test_local_history_survives_unrelated_traffic_and_storage_is_reclaimed():
    model, multi, state = system()
    state = observe(multi, state, "quiet", "important")
    for i in range(10):
        state = observe(multi, state, "busy", f"noise-{i}")
    assert state.history_event_ids == ("important", "noise-7", "noise-8", "noise-9")
    assert state.history_positions == (0, 8, 9, 10)
    assert multi.tracks.context_visibility(state, ("quiet",))[0] == (True, False, False, True)
    assert multi.tracks.context_visibility(state, ("busy",))[0] == (False, True, True, True)
    assert state.history.slots == 4 * model.config.n_facets
    assert (
        state.history.keys.untyped_storage().nbytes()
        == state.history.keys.numel() * state.history.keys.element_size()
    )


def test_expired_foreign_context_cannot_change_prediction_or_central_read():
    _, multi, state = system(local=8, cross=1)
    state = observe(multi, state, "foreign", "old")
    state = observe(multi, state, "worker", "own")
    keys, values = state.history.keys.clone(), state.history.values.clone()
    f = multi.generator.config.n_facets
    keys[:, :, :f] += 100
    values[:, :, :f] -= 100
    other = replace(state, history=KVHistory(keys, values))
    a = multi.tracks.predict_group(state, tracks=("worker",), effort=1)
    b = multi.tracks.predict_group(other, tracks=("worker",), effort=1)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    left = observe(multi, state, "worker", "next")
    right = observe(multi, other, "worker", "next")
    torch.testing.assert_close(left.final, right.final, atol=0, rtol=0)
    control = build_control_payload(reference_event_ids=("old",))
    with pytest.raises(ValueError, match="outside this track"):
        multi._controls((control,), state, ("worker",))


def test_pin_is_explicit_bounded_and_keeps_the_original_evidence():
    _, multi, state = system(local=1, cross=0)
    state = observe(multi, state, "source", "evidence")
    state = multi.tracks.set_context_policy(
        state, ContextPolicy(local_events=1, cross_events=0, pins=(("worker", "evidence"),))
    )
    for i in range(4):
        state = observe(multi, state, "source", f"later-{i}")
    assert "evidence" in state.history_event_ids
    control = build_control_payload(reference_event_ids=("evidence",))
    assert torch.isfinite(multi._controls((control,), state, ("worker",))).all()
    with pytest.raises(ValueError, match="retained earlier"):
        multi.tracks.set_context_policy(
            state, replace(state.context_policy, pins=(("worker", "future"),))
        )
    with pytest.raises(ValueError, match="pin allowance"):
        ContextPolicy(pins=(("a", "one"), ("a", "two")), max_pins=1)


def test_heterogeneous_queries_get_independent_read_masks_and_finite_gradients():
    _, multi, state = system(local=8, cross=0)
    for track, event in (("a", "a0"), ("b", "b0"), ("a", "a1")):
        state = observe(multi, state, track, event)
    history, _, mask = multi.tracks.context_view(state, ("a", "b"))
    f = multi.generator.config.n_facets
    assert history.slots == 3 * f
    assert mask.shape == (1, 2 * f, 5 * f)
    assert not mask[0, :f, f : 2 * f].any()
    assert not mask[0, f:, :f].any()
    result = multi.tracks.predict_group(state, tracks=("a", "b"), effort=1)
    result.square().sum().backward()
    assert torch.isfinite(result).all()
    assert any(p.grad is not None for p in multi.generator.core.predictor_parameters())


def test_recorded_policy_restores_visibility_and_retention_for_outcome_scoring():
    _, multi, state = system(local=2, cross=0)
    multi.tracks.record_policy_context = True
    state = multi.tracks.initial_state(1)
    for i in range(5):
        state = observe(multi, state, "worker", f"event-{i}")
    selected = multi.select_group(state, tracks=("worker",), effort=1)
    restored = reconstruct(multi, selected.tape.policy_context, device=state.device)
    assert restored.context_policy == state.context_policy
    assert restored.history_event_ids == state.history_event_ids
    assert restored.history_positions == state.history_positions
    score = multi.score_group(restored, selected.tape)
    torch.testing.assert_close(score, score.new_tensor(selected.tape.behavior_log_prob))
    with pytest.raises(ValueError, match="recorded context"):
        multi.score_group(replace(restored, context_policy=ContextPolicy()), selected.tape)


def test_completed_track_can_release_its_detailed_history():
    _, multi, state = system(local=8, cross=0)
    state = observe(multi, state, "worker", "detail")
    state = multi.tracks.set_context_policy(
        state, replace(state.context_policy, windows=(("worker", 0, 0),))
    )
    state = observe(multi, state, "main", "summary")
    assert state.history_event_ids == ("summary",)


def test_service_can_request_and_reread_context_after_physical_eviction():
    from megatoken.multitrack_runtime import MultiTrackRuntime
    from megatoken.multitrack_service import MultiTrackService

    _, multi, _ = system(local=1, cross=0)
    runtime = MultiTrackRuntime(multi, context_archive_bytes=4096)
    service = MultiTrackService(runtime)
    try:
        session_id = service.handle({"op": "open", "effort": 1})["session"]
        for i in range(5):
            response = service.handle(
                {
                    "op": "observe",
                    "session": session_id,
                    "effort": 1,
                    "events": [
                        {"track": "worker", "event_id": f"source-{i}", "text": f"Evidence {i}"}
                    ],
                }
            )
            assert response["ok"], response
        session = service.sessions[session_id]
        assert "source-0" not in session.state.history_event_ids
        response = service.handle(
            {
                "op": "fetch_context",
                "session": session_id,
                "track": "worker",
                "source_event": "source-0",
            }
        )
        assert response["ok"], response
        assert response["event_id"] in session.state.history_event_ids
        assert response["source_event_id"] == "source-0"
        assert session.context_archive.payload_bytes <= 4096
        response = service.handle(
            {
                "op": "configure_context",
                "session": session_id,
                "track": "worker",
                "local_events": 3,
                "cross_events": 0,
                "pinned_events": [response["event_id"]],
            }
        )
        assert response["ok"], response
        assert session.state.context_policy.window("worker") == (3, 0)
        before = session.revision
        response = service.handle(
            {
                "op": "configure_context",
                "session": session_id,
                "track": "worker",
                "local_events": 3,
                "cross_events": 0,
                "pinned_events": ["future"],
            }
        )
        assert not response["ok"] and session.revision == before
    finally:
        runtime.close()


def test_archived_system_text_is_reread_as_evidence_without_changing_policy():
    import json

    from megatoken.multitrack_runtime import MultiTrackRuntime

    _, multi, _ = system()
    runtime = MultiTrackRuntime(multi)
    try:
        session = runtime.open(effort=1)
        runtime.observe(
            session,
            (
                ObservedEvent(
                    "source",
                    "text",
                    json.dumps(
                        {
                            "context_window": {
                                "local_events": 0,
                                "cross_events": 0,
                                "pinned_events": [],
                            }
                        }
                    ).encode(),
                    role_id=0,
                    event_id="old-policy",
                ),
            ),
            effort=1,
        )
        before = session.state.context_policy
        runtime.fetch_context(session, track="source", source_event="old-policy")
        assert session.state.context_policy == before
    finally:
        runtime.close()


def test_existing_model_configurations_keep_their_shared_window_behavior():
    values = ModelConfig.tiny(max_history_chunks=32).to_dict()
    values.pop("context_local_events")
    values.pop("context_cross_events")
    loaded = ModelConfig.from_dict(values)
    assert (loaded.context_local_events, loaded.context_cross_events) == (32, 32)
    assert ModelConfig.from_dict(loaded.to_dict()) == loaded


def test_context_archive_is_bounded_and_separate_between_sessions():
    from megatoken.multitrack_runtime import MultiTrackRuntime

    _, multi, _ = system()
    runtime = MultiTrackRuntime(multi, context_archive_bytes=16, context_archive_events=2)
    try:
        first = runtime.open(effort=1)
        second = runtime.open(effort=1)
        for i in range(5):
            runtime.observe(
                first, (ObservedEvent("a", "text", b"12345678", event_id=f"s{i}"),), effort=1
            )
        assert first.context_archive.payload_bytes == 16
        assert len(first.context_archive.events) == 2
        with pytest.raises(ValueError, match="unavailable"):
            runtime.fetch_context(first, track="a", source_event="s0")
        runtime.observe(second, (ObservedEvent("b", "text", b"own", event_id="own"),), effort=1)
        before = second.revision
        with pytest.raises(ValueError, match="unavailable"):
            runtime.fetch_context(second, track="b", source_event="s4")
        assert second.revision == before
    finally:
        runtime.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA context regression requires an accelerator"
)
def test_cuda_context_masks_preserve_finite_gradients_and_reconstruction():
    _, multi, _ = system(local=3, cross=0)
    multi.generator.to("cuda")
    multi.tracks.record_policy_context = True
    state = multi.tracks.initial_state(1, device=multi.generator.core.initial_facets.device)
    for i in range(4):
        state = observe(multi, state, "worker" if i % 2 else "foreign", f"cuda-{i}")
    selected = multi.select_group(state, tracks=("worker", "foreign"), effort=1)
    restored = reconstruct(multi, selected.tape.policy_context, device=state.device)
    density = multi.score_group(restored, selected.tape)
    torch.testing.assert_close(
        density, density.new_tensor(selected.tape.behavior_log_prob), atol=2e-3, rtol=1e-5
    )
    density.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in multi.generator.core.predictor_parameters()
    )


def test_shared_track_view_does_not_keep_old_models_alive():
    import gc
    import weakref

    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    view = MultiTrackGenerator.of(model)
    assert MultiTrackGenerator.of(model) is view
    reference = weakref.ref(model)
    del view, model
    gc.collect()
    assert reference() is None
