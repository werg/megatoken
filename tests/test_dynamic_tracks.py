"""Dynamic identity, same-track peers, categorical policy and asynchronous commits."""

from dataclasses import replace
from threading import Event

import pytest
import torch
from test_multitrack_media import AUDIO, _media_target, _system, _text_target
from test_multitrack_runtime import _runtime

from megatoken.controls import effective_event_payloads
from megatoken.data.schema import MediaSpan, Meridian, TrackEvent
from megatoken.generation.structure import ActionSpec
from megatoken.lifecycle import TrackRegistry
from megatoken.model.multitrack import GroupActionTape


def test_unbounded_lifetime_with_bounded_active_work_and_stable_replies():
    registry = TrackRegistry(max_active=1)
    addresses = []
    for i in range(24):
        description = registry.create(f"Temporary task {i}")
        addresses.append(description.address)
        with pytest.raises(ValueError, match="Active work capacity"):
            registry.create("Too much active work")
        registry.change(description.address, "complete")
    assert len(set(addresses)) == 24
    registry.change(addresses[0], "rename", label="Original work")
    assert registry.tracks[addresses[0]].status == "completed"
    assert registry.tracks[addresses[-1]].label == ""
    registry.change(addresses[0], "resume")
    assert registry.selectable(addresses[0], set())


def test_routing_ignores_addresses_but_reads_description_and_receives_gradients():
    core, _, multi = _system()
    context = torch.randn(1, 2, core.config.facet_dim)
    purpose = torch.randn(1, 2, core.config.control_dim, requires_grad=True)
    first = core.expert_pools.router_features(
        context, track_ids=torch.tensor([[0, 1]]), track_semantics=purpose
    )
    remapped = core.expert_pools.router_features(
        context, track_ids=torch.tensor([[900, 301]]), track_semantics=purpose
    )
    torch.testing.assert_close(first, remapped, rtol=0, atol=0)
    changed = core.expert_pools.router_features(context, track_semantics=purpose + 1)
    assert not torch.allclose(first, changed)
    first.square().sum().backward()
    assert purpose.grad.abs().sum() > 0
    assert core.expert_pools.router_semantic_projection.weight.grad.abs().sum() > 0
    state = multi.tracks.initial_state(1)
    for i in range(12):
        state = multi.tracks.advance_group(
            state,
            torch.randn_like(state.raw[:, :1]),
            tracks=(f"work-{i}",),
            role_ids=torch.tensor([[2]]),
        )
    assert len(state.track_registry) == 12
    # Reassign every equality handle without changing the relationships.
    count = len(state.track_registry)
    remapped = replace(
        state,
        track_registry=tuple(reversed(state.track_registry)),
        history_tracks=count - 1 - state.history_tracks,
    )
    torch.testing.assert_close(
        multi.tracks.predict_group(state, tracks=("work-11",), effort=1),
        multi.tracks.predict_group(remapped, tracks=("work-11",), effort=1),
    )


def test_same_track_speech_and_caption_replay_and_event_permutation():
    _, _, multi = _system()
    state = multi.tracks.initial_state(1)
    arguments = dict(
        tracks=("reply", "reply"),
        event_ids=("speech", "caption"),
        specs=(AUDIO, ActionSpec(max_tokens=4)),
        controls=(b"Say hello", b"Caption the speech"),
        effort=1,
    )
    generated = multi.generate_group(
        state, **arguments, generator=torch.Generator().manual_seed(42)
    )
    restored = GroupActionTape.from_json(generated.tape.to_json())
    replayed = multi.replay_group(state, restored, specs=arguments["specs"], effort=1)
    torch.testing.assert_close(generated.state.final, replayed.state.final)
    assert generated.state.history_tracks[0, -2:].tolist() == [0, 0]
    reversed_args = {
        **arguments,
        "event_ids": ("caption", "speech"),
        "specs": tuple(reversed(arguments["specs"])),
        "controls": tuple(reversed(arguments["controls"])),
    }
    reversed_group = multi.generate_group(
        state, **reversed_args, generator=torch.Generator().manual_seed(42)
    )
    torch.testing.assert_close(
        generated.tape.raw(state.device),
        reversed_group.tape.raw(state.device)[:, [1, 0]],
        atol=1e-5,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="unique"):
        multi.generate_group(state, tracks=("reply", "reply"), effort=1)


def test_modality_choice_has_probability_and_supervision():
    core, model, multi = _system()
    state = multi.tracks.initial_state(1)
    options = ((ActionSpec(max_tokens=4), AUDIO),)
    result = multi.generate_group(
        state,
        tracks=("reply",),
        modality_options=options,
        effort=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert result.tape.modality_masks == ((True, True, False, False),)
    assert result.tape.modality_log_probs[0] < 0
    assert result.tape.behavior_log_prob == pytest.approx(
        sum(result.tape.event_log_probs) + sum(result.tape.modality_log_probs), rel=1e-5
    )
    spec = next(
        p
        for p in options[0]
        if ("text" if isinstance(p, ActionSpec) else p.modality) == result.tape.modalities[0]
    )
    multi.replay_group(state, result.tape, specs=(spec,), effort=1)
    trained = multi.supervised_meridian(
        state,
        tracks=("reply", "reply"),
        event_ids=("a", "b"),
        plans=(_text_target(model, b"hi"), _media_target(model, "audio", AUDIO)),
        control_payloads=(b"Respond clearly",) * 2,
        effort=1,
    )
    loss, count = trained.losses["modality"]
    assert count == 2
    loss.backward()
    assert core.modality_head.bias.grad.abs().sum() > 0
    assert core.modality_head._bank.weight.grad.abs().sum() > 0
    assert effective_event_payloads(
        b"", ("opaque-a",), (("Respond", 0),)
    ) == effective_event_payloads(b"", ("opaque-b",), (("Respond", 0),))


def test_selection_survives_renderer_failure_and_input_can_arrive(monkeypatch):
    _, runtime = _runtime()
    session = runtime.open()
    entered, release = Event(), Event()

    def failing(selected):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(runtime.generator, "render_selected", failing)
    revision = runtime.start_generation(
        session, tracks=("reply",), specs=(ActionSpec(max_tokens=4),), effort=1
    )
    try:
        assert entered.wait(5)
        assert session.revision == revision
        runtime.observe_text(session, "user", b"Stop speaking", effort=1)
        assert session.revision > revision
        selected_raw = session.rendering[revision].selection.raw.clone()
        release.set()
        with pytest.raises(RuntimeError, match="decoder unavailable"):
            session.rendering[revision].future.result(timeout=5)
        work = runtime.poll_generation(session, revision)
        assert work.status == "failed"
        torch.testing.assert_close(work.selection.raw, selected_raw)
        assert work.selection.tape.event_ids
        assert work.selection.tape.render_status == "selected"
    finally:
        release.set()
        runtime.close()


def test_stopped_delivery_cannot_erase_or_deliver_selected_content():
    _, runtime = _runtime()
    session = runtime.open()
    revision = runtime.start_generation(
        session, tracks=("reply",), specs=(ActionSpec(max_tokens=4),), effort=1
    )
    try:
        work = session.rendering[revision]
        work.future.result(timeout=5)
        runtime.stop_delivery(session, revision)
        assert runtime.poll_generation(session, revision).status == "cancelled"
        with pytest.raises(ValueError, match="uncancelled"):
            runtime.delivered(session, revision, work.selection.tape.event_ids)
    finally:
        runtime.close()


def test_exact_media_clock_and_availability_are_different():
    span = MediaSpan("recording", 0, 100, start_tick=0, end_tick=4800, ticks_per_second=48000)
    event = TrackEvent("a", "mic", "audio", 0, "user", media_span=span, available_at_ns=500_000_000)
    with pytest.raises(ValueError, match="availability"):
        Meridian(0, (event,), selected_at_ns=100_000_000)
    assert Meridian(0, (event,), selected_at_ns=500_000_000).events[0].media_span.end_tick == 4800


def test_runtime_track_retirement_late_results_and_ready_dependencies():
    from megatoken.multitrack_runtime import ConcurrentToolRunner, ReadyWork

    _, runtime = _runtime()
    session = runtime.open()
    runner = ConcurrentToolRunner(1)
    try:
        first = runtime.track_action(session, "create", purpose="Find available facts", budget=4)
        original = first.address
        session.tool_routes["late-call"] = (original, "original-action")
        runner.submit("late-call", lambda: b"Verified fact")
        runtime.track_action(session, "complete", address=original)
        runtime.track_action(session, "rename", address=original, label="Archived research")
        next_track = runtime.track_action(session, "create", purpose="Explain the facts", budget=4)
        assert next_track.address != original
        runner.pending["late-call"].result(timeout=5)
        runtime.collect_tools(session, runner, effort=1)
        assert session.events[-1][1] == (original,)
        assert "result:late-call" in session.completed_events
        runtime.enqueue(
            session,
            ReadyWork(
                "explanation",
                next_track.address,
                ActionSpec(max_tokens=4),
                prerequisites=("result:late-call",),
            ),
        )
        runtime.enqueue(
            session,
            ReadyWork(
                "blocked",
                next_track.address,
                ActionSpec(max_tokens=4),
                prerequisites=("future-result",),
            ),
        )
        revision = runtime.schedule_ready(session, effort=1)
        assert session.rendering[revision].selection.tape.event_ids == ("explanation",)
        assert [work.event_id for work in session.queue] == ["blocked"]
        session.rendering[revision].future.result(timeout=5)
        runtime.poll_generation(session, revision)
        runtime.delivered(session, revision, ("explanation",))
        runtime.release_generation(session, revision)
        assert revision not in session.rendering
        runtime.track_action(
            session, "assign", address=next_track.address, purpose="Check the facts"
        )
        assert runtime.schedule_ready(session, effort=1) is None
        assert ("stale", ("blocked",)) in session.scheduling_log
    finally:
        runner.close()
        runtime.close()


def test_typed_lifecycle_uses_committed_policy_messages_and_executes_once():
    import json

    from megatoken.tool_schema import compile_tool_spec

    _, runtime = _runtime()
    session = runtime.open()
    # Fixed arguments isolate protocol correctness from untrained-model quality.
    spec = compile_tool_spec(
        json.dumps(
            [
                {
                    "name": "track_create",
                    "parameters": {
                        "type": "object",
                        "const": {
                            "purpose": "Find facts",
                            "completion": "Facts checked",
                            "budget": 4,
                        },
                    },
                }
            ]
        ),
        128,
    )
    try:
        provisional = runtime.speculate(session, tracks=("assistant",), specs=(spec,), effort=1)
        with pytest.raises(ValueError, match="committed"):
            runtime.execute_lifecycle(session, provisional.result)
        group = runtime.accept(session, provisional)
        outcomes = runtime.execute_lifecycle(session, group)
        assert len(outcomes) == 1
        assert outcomes[0]["track"]["purpose"] == "Find facts"
        assert runtime.execute_lifecycle(session, group) == []
        assert len(session.tracks.tracks) == 1
        assert not session.tool_ready
    finally:
        runtime.close()


def test_failed_acceptance_preserves_work_budget(monkeypatch):
    _, runtime = _runtime()
    session = runtime.open()
    track = runtime.track_action(session, "create", purpose="Give a brief response", budget=1)
    try:
        branch = runtime.speculate(
            session, tracks=(track.address,), specs=(ActionSpec(max_tokens=4),), effort=1
        )

        def unavailable(*args, **kwargs):
            raise RuntimeError("lookup unavailable")

        monkeypatch.setattr(runtime.memory, "generated", unavailable)
        with pytest.raises(RuntimeError, match="lookup unavailable"):
            runtime.accept(session, branch)
        assert session.tracks.tracks[track.address].budget == 1
        assert session.pending is branch
    finally:
        runtime.close()


def test_model_control_messages_can_queue_then_select_work():
    import json

    from megatoken.tool_schema import compile_tool_spec

    _, runtime = _runtime()
    session = runtime.open()

    def action(name, arguments):
        spec = compile_tool_spec(
            json.dumps([{"name": name, "parameters": {"type": "object", "const": arguments}}]), 128
        )
        group = runtime.generate(session, tracks=("assistant",), specs=(spec,), effort=1)
        assert group.tape.valid == (True,)
        return group, runtime.execute_lifecycle(session, group)

    try:
        _, created = action(
            "track_create", {"purpose": "Find facts", "completion": "Facts checked", "budget": 4}
        )
        address = created[0]["track"]["address"]
        group, requested = action(
            "track_output", {"address": address, "request": "Reply", "modality": "auto"}
        )
        event_id = requested[0]["event_id"]
        assert runtime.execute_lifecycle(session, group) == []
        revision = runtime.schedule_ready(session, effort=1)
        work = session.rendering[revision]
        assert work.selection.tape.event_ids == (event_id,)
        assert work.selection.tape.tracks == (address,)
        work.future.result(timeout=5)
        assert runtime.poll_generation(session, revision).result.tape.render_status == "rendered"
    finally:
        runtime.close()


def test_prepared_timing_roundtrips_and_dependencies_require_prior_groups():
    from megatoken.data.schema import Dependency, Episode, Lineage, PhysicalRecord, TrackSpec

    source = TrackEvent(
        "source",
        "conversation",
        "text",
        0,
        "user",
        speaker="A",
        available_at_ns=500,
        media_span=MediaSpan("audio", 0, 1, 0, 48, 48000),
    )
    answer = TrackEvent("answer", "conversation", "text", 1, "assistant", speaker="B")
    episode = Episode(
        "timed",
        (
            PhysicalRecord("source", "user", b"hello"),
            PhysicalRecord("answer", "assistant", b"hi", turn_index=1, policy_mask=True),
        ),
        Lineage("fixture", "v1", "family"),
        dependencies=(Dependency("source", "answer", "evidence"),),
        tracks=(TrackSpec("conversation", "user"),),
        meridians=(Meridian(0, (source,), 500), Meridian(1, (answer,), 600)),
    )
    assert Episode.from_dict(episode.to_dict()) == episode
    with pytest.raises(ValueError, match="earlier meridian"):
        replace(episode, meridians=(Meridian(0, (source, replace(answer, meridian_id=0)), 600),))
