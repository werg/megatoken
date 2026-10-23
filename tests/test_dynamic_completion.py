"""Whole-policy replay, executed resource contracts and publication visibility."""

import json
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest
import torch
from test_multitrack_media import AUDIO, _system

from megatoken.data.media_timing import VideoTimeline, sample_intervals
from megatoken.data.schema import Episode, Lineage, PhysicalRecord, Supervision, canonical_json
from megatoken.environments import SnapshotEnvironment, SnapshotToolAdapter
from megatoken.generation.structure import ActionSpec
from megatoken.model.multitrack import GroupActionTape, ObservedEvent
from megatoken.model.policy_context import reconstruct
from megatoken.retrieval.collection import SourceCollection
from megatoken.teacher import PINNED_TEACHER
from megatoken.training.history import NativeHistory
from megatoken.training.losses import LossLedger
from megatoken.training.outcome import OutcomeCredit, policy_outcome


def test_exact_causal_policy_context_and_verified_policy_gradient():
    core, _, multi = _system()
    multi.tracks.record_policy_context = True
    state = multi.tracks.initial_state(1)
    state = multi.tracks.describe_track(state, "reply", "Answer clearly")
    state = multi.tracks.observe_group(state, (ObservedEvent("user", "text", b"hello"),))
    first = multi.generate_group(
        state, tracks=("reply",), effort=1, specs=(ActionSpec(max_tokens=4),)
    )
    second = multi.generate_group(
        first.state,
        tracks=("reply", "reply"),
        event_ids=("voice", "caption"),
        specs=(AUDIO, ActionSpec(max_tokens=4)),
        controls=(b"Speak", b"Caption"),
        effort=1,
    )
    tape = GroupActionTape.from_json(second.tape.to_json())
    assert len(tape.policy_context) == tape.meridian_id == 2
    restored = reconstruct(multi, tape.policy_context, device=state.device)
    torch.testing.assert_close(restored.final, first.state.final)
    restored.final.retain_grad()
    density = multi.score_group(restored, tape)
    assert float(density.detach()) == pytest.approx(tape.behavior_log_prob, abs=1e-4)
    density.backward(retain_graph=True)
    assert restored.final.grad is not None and torch.isfinite(restored.final.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in core.central.parameters()
    )
    core.zero_grad(set_to_none=True)
    terms, metrics = policy_outcome(
        multi, tape, OutcomeCredit(1, 0.5, 2, "local_assertions"), device=state.device
    )
    terms["policy_outcome"][0].backward()
    assert metrics["outcome_decisions"] == 1
    assert sum(p.grad.abs().sum() for p in core.predictor_parameters() if p.grad is not None) > 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in core.modality_head.parameters()
    )
    # Decoder failure does not erase the sampled decision's policy density.
    failed = replace(tape, payloads=(b"", b""), emitted_ids=((), ()), valid=(False, False))
    failed_terms, _ = policy_outcome(
        multi,
        failed,
        OutcomeCredit(-1, 0, 2, "external_tests"),
        device=state.device,
    )
    assert torch.isfinite(failed_terms["policy_outcome"][0])
    ledger = LossLedger(
        {"policy_outcome": 1},
        {"policy_outcome": 1},
        {"policy_outcome": 1.0},
    )
    ledger.backward_loss(failed_terms).backward()
    ledger.finish()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in core.parameters()
        if parameter.grad is not None
    )
    with pytest.raises(ValueError, match="complete committed"):
        policy_outcome(
            multi,
            replace(tape, policy_context=None),
            OutcomeCredit(1, 0.5, 2, "external_tests"),
            device=state.device,
        )
    with pytest.raises(ValueError, match="independent"):
        OutcomeCredit(1, 0, 2, "simulation_judge")


def test_latent_outcome_update_preserves_the_exact_rendered_action_map():
    from megatoken.training.optimizer import MasterAdamW
    from megatoken.training.outcome import latent_policy_parameter_ids

    _, model, multi = _system()
    multi.tracks.record_policy_context = True
    state = multi.tracks.initial_state(1)
    selected = multi.select_group(
        state,
        tracks=("reply",),
        effort=1,
        specs=(ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=2),),
    )
    before = multi.render_selected(selected)
    optimizer = MasterAdamW(model, lr=0.001)
    policy = latent_policy_parameter_ids(model)
    assert policy.isdisjoint(id(p) for p in model.core.shared_parameters())
    assert policy.isdisjoint(id(p) for p in model.decoder.parameters())
    assert policy.isdisjoint(id(p) for p in model.media_realizers.parameters())
    optimizer.freeze(
        [name for name, p in optimizer.model_parameters.items() if id(p) not in policy]
    )
    frozen = {
        name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad
    }
    learned = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    try:
        terms, _ = policy_outcome(
            multi, before.tape, OutcomeCredit(1, 0.5, 1, "local_assertions"), device=state.device
        )
        terms["policy_outcome"][0].backward()
        optimizer.step()
        assert any(
            not torch.equal(p, learned[name])
            for name, p in model.named_parameters()
            if name in learned
        )
        assert all(
            torch.equal(p, frozen[name]) for name, p in model.named_parameters() if name in frozen
        )
        after = multi.render_selected(selected)
        assert after.tape.payloads == before.tape.payloads
        assert after.tape.emitted_ids == before.tape.emitted_ids
    finally:
        optimizer.close()


def test_native_same_track_peers_have_explicit_bindings_and_one_parent():
    _, model, multi = _system()
    state = multi.tracks.initial_state(1)
    result = multi.generate_group(
        state,
        tracks=("reply", "reply"),
        event_ids=("a", "b"),
        effort=1,
        specs=(ActionSpec(max_tokens=4, min_tokens=1),) * 2,
    )
    records = tuple(
        PhysicalRecord(
            f"record-{i}",
            "assistant",
            payload,
            emitted_ids=ids,
            codec_fingerprint=model.codec.fingerprint,
        )
        for i, (payload, ids) in enumerate(
            zip(result.tape.payloads, result.tape.emitted_ids, strict=True)
        )
    )
    entry = {
        "record_cursor": 0,
        "record_ids": [r.record_id for r in records],
        "tape": json.loads(result.tape.to_json()),
    }
    episode = Episode(
        "peers",
        records,
        Lineage("test", "v1", "peers"),
        supervision=Supervision(annotations_json=canonical_json({"native_action_tapes": [entry]})),
    )
    native = NativeHistory.from_episode(episode, model)
    assert native.group_for_records((0, 1)) == result.tape
    assert native.paired_tape(0).group is native.paired_tape(1).group
    assert native.events_for_record(1, include_record=False) == ()
    entry["record_ids"].reverse()
    with pytest.raises(ValueError, match="contiguous"):
        NativeHistory.from_episode(
            replace(
                episode,
                supervision=Supervision(
                    annotations_json=canonical_json({"native_action_tapes": [entry]})
                ),
            ),
            model,
        )


def test_snapshot_adapters_conflicting_writes_and_idempotent_delivery():
    env = SnapshotEnvironment()
    initial = env.reset(
        {
            "prompt": "edit",
            "initial_files": {"a": "0", "b": "0"},
            "expected_files": {"a": "1", "b": "2"},
        }
    )
    bound = SnapshotToolAdapter(env, initial.state_id)

    def execute(identity, name, args):
        return bound.execute_selected(
            action_id=identity, payload=canonical_json({"name": name, "arguments": args}).encode()
        )

    original = execute("read", "read", {"path": "a"})
    execute("write-a", "write", {"path": "a", "content": "1"})
    execute("write-b", "write", {"path": "b", "content": "2"})
    assert execute("read", "read", {"path": "a"}) == original
    assert json.loads(execute("old-read", "read", {"path": "a"}))["observation"]["content"] == "0"
    with pytest.raises(ValueError, match="Conflicting"):
        execute("conflict", "write", {"path": "a", "content": "bad"})
    assert env.verify().reward == 1


def test_live_publication_revisions_and_cross_connection_index_refresh(tmp_path):
    keys = np.eye(3, 768, dtype=np.float32)
    with SourceCollection.create(tmp_path, name="live", teacher=PINNED_TEACHER) as writer:
        writer.add_text(b"first", keys[0], origin="one")
        writer.reindex(cells=1)
        with SourceCollection.open(tmp_path) as reader:
            assert len(reader.search(keys[0], 3)) == 1
            revision = reader.revision
            row = writer.add_text(b"arrived", keys[1], origin="two")
            receipt = writer.publication(row)
            assert receipt["revision"] == revision + 1
            assert len(reader.search(keys[1], 3, max_revision=revision)) == 1
            assert (
                len(reader.search(keys[1], 3, available_at_ns=receipt["available_at_ns"] - 1)) == 1
            )
            assert reader.search(keys[1], 1)[0].row == row
            writer.reindex(cells=2)
            assert reader.search(keys[1], 1, probes=2)[0].row == row
            assert writer.add_text(b"arrived", keys[1], origin="retry") is None
            assert writer.revision == receipt["revision"]


def test_variable_frame_clock_resampling_preserves_available_source_indices():
    timeline = VideoTimeline(Fraction(1, 1000), (0, 30, 100), (30, 100, 200))
    assert timeline.uniform_indices(0, 200, Fraction(20)) == (0, 1, 2, 2)
    assert timeline.uniform_indices(0, 100, Fraction(20)) == (0, 1)
    with pytest.raises(ValueError, match="bounds"):
        timeline.uniform_indices(0, 200, Fraction(1000), max_frames=10)
    assert sample_intervals(45, 2, max_seconds=10) == ((0, 20), (20, 40), (40, 45))


def test_failed_render_without_physical_record_trains_through_the_real_ledger(tmp_path):
    from training_fixture import TEXT_ONLY_ASSETS, tiny_train_config

    from megatoken.data import WindowConfig
    from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
    from megatoken.training.engine import batch_counters, prepare_batches
    from megatoken.training.losses import LossLedger
    from megatoken.training.meridians import forward_meridian_batch
    from megatoken.training.stream import EpisodeWindow, _WindowPlan
    from megatoken.training.transducer import TransducerTrainer

    _, model, multi = _system()
    multi.tracks.record_policy_context = True
    state = multi.tracks.observe_group(
        multi.tracks.initial_state(1), (ObservedEvent("user", "text", b"hello"),)
    )
    selected = multi.select_group(state, tracks=("assistant",), effort=1)
    tape = selected.tape
    prompt = PhysicalRecord("prompt", "user", b"hello")
    credit = OutcomeCredit(0, 0.5, 1, "local_assertions")
    episode = Episode(
        "failed-render",
        (prompt,),
        Lineage("test", "v1", "failed-render"),
        tracks=(TrackSpec("user", "user"),),
        meridians=(Meridian(0, (TrackEvent("prompt", "user", "text", 0, "user"),)),),
        supervision=Supervision(
            annotations_json=canonical_json(
                {
                    "native_action_tapes": [
                        {
                            "record_cursor": 1,
                            "record_ids": [None],
                            "tape": json.loads(tape.to_json()),
                            "outcome_credit": asdict(credit),
                        }
                    ]
                }
            )
        ),
    )
    plan = _WindowPlan(episode, model.codec, WindowConfig())
    assert len(plan) == 1 and plan[0].start == plan[0].end == 1
    config = tiny_train_config(tmp_path, effort=1)
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, plan[0], TEXT_ONLY_ASSETS)], config)
    assert batch.counts["policy_outcome"] == 1
    counters = batch_counters([batch])
    assert counters.P == counters.G == 0 and counters.Q == 1
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    metrics = forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        transducer=TransducerTrainer(model.transducer),
    )
    ledger.finish()
    assert metrics["outcome_decisions"] == 1
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.predictor_parameters()
    )


def test_executed_lifecycle_source_announces_purpose_after_creation_and_keeps_late_replies():
    from megatoken.controls import resolve_event_control
    from megatoken.data.dynamic_execution import execution_episode

    episode = execution_episode(0, delegated=True)
    assert Episode.from_dict(episode.to_dict()) == episode and episode.track_revisions
    workers = {r.track_id for r in episode.track_revisions}
    assert all(t.description == "" for t in episode.tracks if t.track_id in workers)
    events = {e.record_id: e for g in episode.meridians for e in g.events}
    positions = {r.record_id: i for i, r in enumerate(episode.records)}
    announced = {r.track_id: positions[r.record_id] for r in episode.track_revisions}
    reads = {a.action_id: a for a in episode.actions if a.tool_name == "read"}
    cancelled = next(a for a in episode.actions if a.tool_name == "track_cancel")
    cancel_position = positions[cancelled.record_ids[0]]
    assert all(positions[a.record_ids[0]] < cancel_position for a in reads.values())
    replies = [r for r in episode.records if r.result_of in reads]
    assert all(positions[r.record_id] > cancel_position for r in replies)
    for action in reads.values():
        event = events[action.record_ids[0]]
        assert announced[event.track_id] < positions[event.record_id]
        purpose = next(r.purpose for r in episode.track_revisions if r.track_id == event.track_id)
        assert purpose.startswith("Read")
        assert b"available_input_tracks" not in resolve_event_control(episode, event)
    with pytest.raises(ValueError, match="public announcement"):
        replace(
            episode,
            track_revisions=(
                replace(episode.track_revisions[0], purpose="Hidden future task"),
                *episode.track_revisions[1:],
            ),
        )


def test_native_history_cache_does_not_retain_a_finished_model():
    import gc
    import weakref

    _, model, multi = _system()
    episode = Episode(
        "cache", (PhysicalRecord("prompt", "user", b"hello"),), Lineage("test", "v1", "cache")
    )
    NativeHistory.of(episode, model)
    owner = weakref.ref(model)
    del multi, model
    gc.collect()
    assert owner() is None


def test_partial_speech_delivery_survives_stop_without_erasing_selection():
    from test_multitrack_runtime import _runtime

    _, runtime = _runtime()
    try:
        session = runtime.open(effort=1)
        revision = runtime.start_generation(
            session, tracks=("voice",), event_ids=("speech",), specs=(AUDIO,), effort=1
        )
        session.rendering[revision].future.result(timeout=10)
        runtime.poll_generation(session, revision)
        tape = session.rendering[revision].selection.tape.to_json()
        runtime.delivered(session, revision, ("speech",), extents={"speech": 128})
        work = runtime.stop_delivery(session, revision)
        assert work.status == "cancelled" and work.delivery_progress == {"speech": 128}
        assert work.delivery_log[0]["unit"] == "samples"
        assert work.selection.tape.to_json() == tape
        stopped = session.revision
        runtime.stop_delivery(session, revision)
        assert session.revision == stopped
        runtime.release_generation(session, revision)
        assert not session.reserved_output_bytes
    finally:
        runtime.close()


def test_lifecycle_control_meaning_survives_opaque_address_remapping():
    from megatoken.controls import build_control_payload, semantic_lifecycle_controls

    def control(address, purpose):
        payload = build_control_payload(
            seed=b"Choose useful work",
            tools_json=canonical_json(
                [
                    {
                        "name": "track_resume",
                        "parameters": {
                            "type": "object",
                            "properties": {"address": {"type": "string", "enum": [address]}},
                        },
                    }
                ]
            ),
        )
        return semantic_lifecycle_controls(payload, {address: (purpose, 2)})

    assert control("arbitrary-123", "Read source") == control("other-999", "Read source")
    assert control("arbitrary-123", "Read source") != control("arbitrary-123", "Write answer")
