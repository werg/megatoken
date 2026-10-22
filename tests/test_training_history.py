"""Native event chronology at teacher prefixes and live recurrent windows."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_generation_policy import invalid_materialization_spec

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.collection import TeacherRequest
from megatoken.data.schema import Episode, Lineage, PhysicalRecord, Supervision, canonical_json
from megatoken.data.student import ModelTeacher
from megatoken.generation import ActionSpec, ByteCodec, FacetedGenerator, FieldSpec, SchemaSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.tensors import byte_tensor
from megatoken.training.history import NativeHistory, replay_raw_tape


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(918)
    return FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback())


def observe(model, state, record):
    values, lengths = byte_tensor([record.payload], state.device)
    return model.core.observe(
        values,
        state,
        lengths=lengths,
        role_ids=torch.tensor([ROLE_TO_ID[record.role]]),
        encoder_grad=False,
    )


def annotation_entry(cursor, tape):
    return {"record_cursor": cursor, "tape": json.loads(tape.to_json())}


@pytest.fixture
def collected(model):
    prompt = PhysicalRecord("prompt", "user", b"choose an action")
    tool = PhysicalRecord("tool", "tool", b'{"ok":true}', turn_index=2)
    invalid_spec = invalid_materialization_spec(model)
    empty_spec = ActionSpec(min_tokens=0, max_tokens=0)
    action_spec = ActionSpec(
        allow_text=False,
        schemas=(
            SchemaSpec(
                "finish",
                (
                    FieldSpec("name", kind="json", fixed_json='"finish"'),
                    FieldSpec("arguments", kind="json", fixed_json="{}"),
                ),
            ),
        ),
        max_tokens=128,
    )
    with torch.no_grad():
        initial = model.core.initial_state(1)
        factual = observe(model, initial, prompt)
        invalid = model.sample(
            factual, spec=invalid_spec, generator=torch.Generator().manual_seed(3)
        )
        empty = model.sample(
            invalid.state, spec=empty_spec, generator=torch.Generator().manual_seed(4)
        )
        action = model.sample(
            empty.state, spec=action_spec, generator=torch.Generator().manual_seed(5)
        )
        after_tool = observe(model, action.state, tool)
        trailing = model.sample(
            after_tool, spec=empty_spec, generator=torch.Generator().manual_seed(6)
        )
    assert not invalid.tape.valid and "including fixed syntax" in invalid.tape.error
    assert invalid.payload in {
        invalid_spec.schemas[0].serialize({"value": value}) for value in (b"\t", b"\n")
    }
    assert empty.tape.valid and empty.payload == b""
    assert action.tape.valid and action.payload
    generated = PhysicalRecord(
        "action",
        "assistant",
        action.payload,
        turn_index=1,
        emitted_ids=action.emitted_ids,
        codec_fingerprint=model.codec.fingerprint,
        provenance="student",
    )
    entries = [
        annotation_entry(1, invalid.tape),
        annotation_entry(1, empty.tape),
        annotation_entry(1, action.tape),
        annotation_entry(3, trailing.tape),
    ]
    episode = Episode(
        "native-episode",
        (prompt, generated, tool),
        Lineage("fixture", "v1", "native-family"),
        supervision=Supervision(annotations_json=canonical_json({"native_action_tapes": entries})),
    )
    return SimpleNamespace(
        episode=episode,
        invalid=invalid,
        empty=empty,
        action=action,
        after_tool=after_tool,
        trailing=trailing,
        entries=entries,
    )


def assert_same_context(actual, expected):
    assert torch.equal(actual.facets, expected.facets)
    for name in ("records_seen", "last_record_ids", "role_ids", "episode_ids"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    for name in ("final",):
        left, right = getattr(actual, name), getattr(expected, name)
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
    assert actual.history.mask is expected.history.mask is None
    torch.testing.assert_close(actual.history.keys, expected.history.keys, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.history.values, expected.history.values, atol=1e-6, rtol=1e-6)


def test_native_history_is_built_once_per_episode_and_model(model, collected, monkeypatch):
    builds = []
    original = NativeHistory.from_episode.__func__

    def counting(cls, episode, owner):
        builds.append(owner)
        return original(cls, episode, owner)

    monkeypatch.setattr(NativeHistory, "from_episode", classmethod(counting))
    first = NativeHistory.of(collected.episode, model)
    assert NativeHistory.of(collected.episode, model) is first and len(builds) == 1
    # Another model validates the same tapes independently.
    other = type(model)(model.core, model.codec)
    assert NativeHistory.of(collected.episode, other) is not first and len(builds) == 2


def test_native_invalid_empty_action_and_tool_replay_matches_live_context(model, collected):
    history = NativeHistory.from_episode(collected.episode, model)
    assert history.has_events
    assert history.events_for_record(0, include_record=True) == ()
    assert history.events_for_record(1, include_record=True) == (
        collected.invalid.tape,
        collected.empty.tape,
        collected.action.tape,
    )
    assert history.has_generated_record(1)
    assert history.paired_tape(1) == collected.action.tape
    assert history.paired_tape(0) is None
    assert history.paired_tape(3) is None
    assert not history.has_generated_record(3)
    state = model.core.initial_state(1)
    for cursor, record in enumerate(collected.episode.records):
        for tape in history.events_for_record(cursor, include_record=True):
            state = replay_raw_tape(model, state, tape)
            raw = torch.from_numpy(tape.facet_array().copy()).reshape(1, *tape.facet_shape)
            assert torch.equal(state.facets, raw)
        if not history.has_generated_record(cursor):
            state = observe(model, state, record)
    assert_same_context(state, collected.after_tool)
    for tape in history.events_for_record(history.record_count, include_record=True):
        state = replay_raw_tape(model, state, tape)
    assert_same_context(state, collected.trailing.state)
    assert state.records_seen.item() == 6  # Every selected event was already C-processed.


def test_target_native_tape_is_excluded_but_earlier_zero_record_events_recur(model, collected):
    history = NativeHistory.from_episode(collected.episode, model)
    before_target = history.events_for_record(1, include_record=False)
    assert before_target == (collected.invalid.tape, collected.empty.tape)
    state = observe(model, model.core.initial_state(1), collected.episode.records[0])
    for tape in before_target:
        state = replay_raw_tape(model, state, tape)
    assert_same_context(state, collected.empty.state)
    assert not torch.equal(state.facets, collected.action.state.facets)
    # A different target tensor in metadata cannot affect its preceding prefix.
    changed = replace(collected.action.tape, facets=(0.0,) * model.latent_dimensions)
    entries = [*collected.entries]
    entries[2] = annotation_entry(1, changed)
    episode = replace(
        collected.episode,
        supervision=Supervision(annotations_json=canonical_json({"native_action_tapes": entries})),
    )
    changed_history = NativeHistory.from_episode(episode, model)
    assert changed_history.events_for_record(1, include_record=False) == before_target
    # Trailing zero-record events are still available at an end-of-episode boundary.
    assert history.events_for_record(3, include_record=False) == (collected.trailing.tape,)


def test_fixed_raw_replay_skips_encoder_decoder_prediction_and_keeps_history_gradients(
    model, collected, monkeypatch
):
    history = NativeHistory.from_episode(collected.episode, model)
    state = observe(model, model.core.initial_state(1), collected.episode.records[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("raw history used an encoder, decoder, or unused prediction stack")

    rng = torch.random.get_rng_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(model.core, "encode_chunk", forbidden)
        patch.setattr(model.core.remix, "forward", forbidden)
        patch.setattr(model.decoder, "forward", forbidden)
        patch.setattr(model.facet_distribution, "forward", forbidden)
        for tape in history.events_for_record(1, include_record=True):
            state = replay_raw_tape(model, state, tape)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert_same_context(state, collected.action.state)
    state.history.values.retain_grad()
    history_values = state.history.values
    after_tool = observe(model, state, collected.episode.records[2])
    plan = model.prepare_target(target_bytes=b"next", representation_mode="recurrent")
    result = model.supervised_step(after_tool, plan=plan)
    result.losses["lexical"][0].backward()
    assert history_values.grad is not None and history_values.grad.float().norm() > 0
    assert torch.isfinite(history_values.grad).all()
    assert all(parameter.grad is None for parameter in model.core.encoder_parameters())


def malformed_episode(collected, entries, *, records=None):
    # Some failures are normally caught even earlier by data schema validation;
    # the history parser also has to reject them at its own input boundary.
    return SimpleNamespace(
        records=collected.episode.records if records is None else records,
        supervision=SimpleNamespace(
            annotations_json=canonical_json({"native_action_tapes": entries})
        ),
    )


@pytest.mark.parametrize("cursor", [-1, 4, "1", 1.5, True])
def test_native_cursor_range_and_type_are_strict(model, collected, cursor):
    entry = {**collected.entries[0], "record_cursor": cursor}
    with pytest.raises(ValueError, match="record_cursor"):
        NativeHistory.from_episode(malformed_episode(collected, [entry]), model)


def test_legacy_native_cursor_cannot_be_inferred(model, collected):
    entry = {"tape": collected.entries[0]["tape"]}
    with pytest.raises(ValueError, match="re-collect"):
        NativeHistory.from_episode(malformed_episode(collected, [entry]), model)


@pytest.mark.parametrize("kind", ["reordered", "duplicate", "after_consumed", "missing_record"])
def test_native_order_cannot_reuse_a_consumed_physical_record(model, collected, kind):
    invalid, empty, action, trailing = collected.entries
    entries = {
        "reordered": [trailing, invalid],
        "duplicate": [action, action],
        "after_consumed": [action, empty],
        "missing_record": [{**action, "record_cursor": 3}],
    }[kind]
    with pytest.raises(ValueError, match="record"):
        NativeHistory.from_episode(malformed_episode(collected, entries), model)


@pytest.mark.parametrize("kind", ["codec", "geometry", "bytes", "fp32", "effort", "valid"])
def test_native_tape_identity_and_raw_tensor_are_checked(model, collected, kind):
    value = dict(collected.entries[0]["tape"])
    if kind == "codec":
        value["codec_fingerprint"] = "another-codec"
    elif kind == "geometry":
        value["facet_shape"] = [1, model.latent_dimensions]
    elif kind == "bytes":
        value["emitted_ids"] = [1]
    elif kind == "fp32":
        # V4 normally packs FP32 bytes. Exercise the list input's precision
        # guard explicitly: the Python float 0.1 is not exactly representable.
        value["facets"] = [0.1, *collected.invalid.tape.facet_array().tolist()[1:]]
    elif kind == "effort":
        value["effort"] = 5
    else:
        value["valid"] = 1
    with pytest.raises(ValueError, match="Native history entry"):
        NativeHistory.from_episode(
            malformed_episode(collected, [{"record_cursor": 1, "tape": value}]), model
        )


@pytest.mark.parametrize("kind", ["role", "payload", "ids", "codec", "parts"])
def test_native_nonempty_output_requires_its_unchanged_whole_assistant_record(
    model, collected, kind
):
    record = collected.episode.records[1]
    if kind == "role":
        record = replace(record, role="user")
    elif kind == "payload":
        record = replace(record, payload=b"changed")
    elif kind == "ids":
        record = replace(record, emitted_ids=None)
    elif kind == "codec":
        record = replace(record, codec_fingerprint="other")
    else:
        record = replace(record, part_count=2)
    records = (collected.episode.records[0], record, collected.episode.records[2])
    with pytest.raises(ValueError, match="native"):
        NativeHistory.from_episode(
            malformed_episode(collected, collected.entries, records=records), model
        )


def test_absent_native_metadata_is_an_empty_index_and_invalid_lookup_fails(model, collected):
    episode = replace(collected.episode, supervision=Supervision())
    history = NativeHistory.from_episode(episode, model)
    assert not history.has_events
    assert not history.has_generated_record(1)
    assert history.events_for_record(1, include_record=False) == ()
    with pytest.raises(ValueError, match="cursor"):
        history.events_for_record(4, include_record=True)


def test_valid_empty_student_draw_does_not_require_an_absent_record_on_next_request(model):
    class Engine:
        manifest = {"model_sha256": "a" * 64}
        config = model.config
        codec = model.codec

        def __init__(self):
            self.observations, self.draws = 0, 0

        def new_session(self):
            self.session = SimpleNamespace(state=model.core.initial_state(1))
            return self.session

        def observe_physical(self, session, payload, *, role, token_ids):
            self.observations += 1
            assert self.codec.decode(token_ids) == payload
            session.state = observe(model, session.state, PhysicalRecord("input", role, payload))

        def generate(self, session, **kwargs):
            self.draws += 1
            result = model.sample(
                session.state,
                spec=ActionSpec(max_tokens=0),
                generator=torch.Generator().manual_seed(self.draws),
            )
            session.state = result.state
            return result

    engine = Engine()
    student = ModelTeacher(_engine=engine)
    records = (PhysicalRecord("prompt", "user", b"continue"),)
    for index in range(2):
        response = student.respond(
            TeacherRequest(
                request_id=f"empty-{index}",
                messages_json="[]",
                tools_json="[]",
                max_output_tokens=8,
                codec_fingerprint=model.codec.fingerprint,
                session_id="empty-session",
                records=records,
            )
        )
        assert response.content == b"" and not response.calls
        assert json.loads(response.native_tape_json)["valid"] is True
    assert engine.observations == 1 and engine.draws == 2
    assert engine.session.state.records_seen.item() == 3


def integration_episodes(model, collected):
    """A scored native target and a later teacher target with a native prefix."""
    entries = [*collected.entries]
    future = replace(collected.action.tape, facets=(0.0,) * model.latent_dimensions)
    entries[2] = annotation_entry(1, future)
    target = replace(
        collected.episode,
        episode_id="scored-native-target",
        records=(
            collected.episode.records[0],
            replace(collected.episode.records[1], policy_mask=True),
            collected.episode.records[2],
        ),
        supervision=Supervision(annotations_json=canonical_json({"native_action_tapes": entries})),
        control_seed=b"public causal instruction",
    )
    prefix = replace(
        collected.episode,
        episode_id="native-prefix",
        records=(
            *collected.episode.records,
            PhysicalRecord(
                "correction",
                "assistant",
                b"corrected answer",
                turn_index=3,
                policy_mask=True,
                provenance="teacher",
            ),
        ),
        control_seed=b"public causal instruction",
    )
    return (
        (target, 1, collected.empty.state),
        (prefix, 3, collected.trailing.state),
    )


def capture_core_calls(model, monkeypatch):
    trace = SimpleNamespace(encoded=[], predictions=[], native_controls=[])
    original_encode = model.core.encode_chunk
    original_predict = model.core.predict_facets
    original_advance = model.core.advance_raw

    def encode(values, lengths=None):
        lengths_ = [values.shape[1]] * values.shape[0] if lengths is None else lengths.tolist()
        trace.encoded.extend(
            bytes(row[:length].tolist()) for row, length in zip(values, lengths_, strict=True)
        )
        return original_encode(values, lengths)

    def predict(state, *args, **kwargs):
        control = kwargs.get("control", args[0] if args else None)
        trace.predictions.append((state.detach(), None if control is None else control.detach()))
        return original_predict(state, *args, **kwargs)

    def advance(state, raw, *args, **kwargs):
        trace.native_controls.append(kwargs.get("control", args[0] if args else None))
        return original_advance(state, raw, *args, **kwargs)

    monkeypatch.setattr(model.core, "encode_chunk", encode)
    monkeypatch.setattr(model.core, "predict_facets", predict)
    monkeypatch.setattr(model.core, "advance_raw", advance)
    return trace


def integration_config():
    from megatoken.training.config import TrainConfig

    return TrainConfig(
        phase="D",
        precision="fp32",
        device="cpu",
        generated_capacity=128,
        representation_mode="recurrent",
        use_effort_mixture=False,
        microbatch_episodes=2,
        # The evaluator reads none of the retrieval artifacts; the config only
        # has to satisfy the every-invocation validation.
        budget=64,
    )


def test_evaluation_replays_native_prefix_and_excludes_native_target_facets(
    model, collected, monkeypatch
):
    from megatoken.training.evaluation import evaluate_records

    class Dataset:
        def __init__(self, episode):
            self.episode = episode

        def __len__(self):
            return 1

        def iter_episodes(self):
            yield self.episode

    trace = capture_core_calls(model, monkeypatch)
    for episode, cursor, expected in integration_episodes(model, collected):
        trace.encoded.clear()
        trace.predictions.clear()
        trace.native_controls.clear()
        metrics = evaluate_records(
            model,
            Dataset(episode),
            model.codec,
            integration_config(),
            torch.device("cpu"),
            episodes=1,
        )
        assert model.training  # The evaluator restores the caller's mode.
        assert metrics["generated_chunks"] == 1 and metrics["prior_scored_positions"] > 0
        assert len(trace.predictions) == 2  # Sampling and teacher CE share the same causal input.
        for actual, control in trace.predictions:
            assert_same_context(actual, expected)
            assert control is not None
        assert all(value is None for value in trace.native_controls)
        if cursor == 3:
            assert collected.action.payload not in trace.encoded
        else:
            assert trace.encoded.count(collected.action.payload) == 1


def test_fixed_native_replay_retains_central_gradients_and_ignores_prospective_controls(
    model, monkeypatch
):
    prompt = PhysicalRecord("prompt", "user", b"continue with controls")
    state = observe(model, model.core.initial_state(1), prompt)
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=3)
    first = model.sample(
        state,
        spec=spec,
        control_payload=b"condition one\x00",
        generator=torch.Generator().manual_seed(61),
    )
    second = model.sample(
        first.state,
        spec=spec,
        control_payload=b"condition two\xff",
        generator=torch.Generator().manual_seed(62),
    )
    records = tuple(
        PhysicalRecord(
            f"native-{index}",
            "assistant",
            value.payload,
            turn_index=index,
            emitted_ids=value.emitted_ids,
            codec_fingerprint=model.codec.fingerprint,
        )
        for index, value in enumerate((first, second), 1)
    )
    episode = Episode(
        "native-controls",
        (prompt, *records),
        Lineage("fixture", "v1", "controls"),
        supervision=Supervision(
            annotations_json=canonical_json(
                {
                    "native_action_tapes": [
                        annotation_entry(1, first.tape),
                        annotation_entry(2, second.tape),
                    ]
                }
            )
        ),
    )
    native = NativeHistory.from_episode(episode, model)
    replayed = observe(model, model.core.initial_state(1), prompt)

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed-action C replay encoded a prospective control")

    monkeypatch.setattr(model.core, "encode_control", forbidden)
    for cursor in (1, 2):
        (tape,) = native.events_for_record(cursor, include_record=True)
        replayed = replay_raw_tape(model, replayed, tape)
    assert_same_context(replayed, second.state)
    replayed.history.values.square().sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.float().norm() > 0
        for parameter in model.core.central_parameters()
    )
    assert all(parameter.grad is None for parameter in model.core.control_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.core.encoder_parameters())
