"""Block-causal multi-track reference path and the exact joint action tape."""

import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.latents import DiagonalGaussian
from megatoken.generation.policy import FacetedGenerator
from megatoken.generation.structure import ActionSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import GroupActionTape, MultiTrackCore, MultiTrackGenerator


def _tiny(*, history=8):
    torch.manual_seed(27)
    config = ModelConfig.tiny(max_history_chunks=history)
    core = FacetedCore(config)
    return core, MultiTrackCore(core)


def _selected(core, tracks, scale=1.0):
    return torch.randn(1, len(tracks), core.config.n_facets, core.config.facet_dim) * scale


def _roles(tracks):
    return torch.tensor([[3] * len(tracks)], dtype=torch.long)


def test_singleton_matches_original_core_and_predictor():
    core, multi = _tiny()
    old_initial = core.initial_state(1)
    initial = multi.initial_state(1)
    old_prior = core.predict_facets(old_initial, effort=1)
    new_prior = multi.predict_group(initial, tracks=("assistant",), effort=1)
    torch.testing.assert_close(new_prior[:, 0], old_prior, atol=1e-5, rtol=1e-5)

    raw = _selected(core, ("user",))[:, 0]
    old = core.advance_raw(old_initial, raw, role_ids=torch.tensor([3]))
    new = multi.advance_group(
        initial, raw[:, None], tracks=("user",), role_ids=torch.tensor([[3]])
    )
    torch.testing.assert_close(new.raw[:, 0], old.facets)
    torch.testing.assert_close(new.early[:, 0], old.early, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new.final[:, 0], old.final, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new.history.keys, old.history.keys, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new.history.values, old.history.values, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        multi.predict_group(new, tracks=("assistant",), effort=1)[:, 0],
        core.predict_facets(old, effort=1),
        atol=1e-5, rtol=1e-5,
    )


def test_joint_observation_reads_peers_and_prediction_does_not_commit():
    core, multi = _tiny()
    state = multi.initial_state(1)
    raw = _selected(core, ("camera", "audio"))
    another = raw.clone()
    another[:, 1] = another[:, 1] + 2.0
    a = multi.advance_group(state, raw, tracks=("camera", "audio"), role_ids=_roles((1, 2)))
    b = multi.advance_group(state, another, tracks=("camera", "audio"), role_ids=_roles((1, 2)))
    assert a.meridian_sizes == (2,)
    assert a.history.slots == 2 * core.config.n_facets
    assert not torch.allclose(a.final[:, 0], b.final[:, 0])
    before = a.history.keys.clone()
    predicted = multi.predict_group(a, tracks=("transcript", "reply"), effort=1)
    assert predicted.shape == raw.shape
    torch.testing.assert_close(before, a.history.keys)
    assert not torch.allclose(predicted[:, 0], predicted[:, 1])


def test_eviction_drops_whole_meridians():
    core, multi = _tiny(history=3)
    state = multi.initial_state(1)
    pair = ("worker", "critic")
    first = multi.advance_group(state, _selected(core, pair), tracks=pair, role_ids=_roles(pair))
    assert first.meridian_sizes == (2,)
    one = multi.advance_group(
        first, _selected(core, ("tool",)), tracks=("tool",), role_ids=_roles((1,))
    )
    assert one.meridian_sizes == (2, 1)
    second = multi.advance_group(
        one, _selected(core, pair), tracks=pair, role_ids=_roles(pair)
    )
    assert second.meridian_sizes == (1, 2)
    assert second.history.slots == 3 * core.config.n_facets
    third = multi.advance_group(
        second, _selected(core, pair), tracks=pair, role_ids=_roles(pair)
    )
    assert third.meridian_sizes == (2,)
    assert third.history.slots == 2 * core.config.n_facets


def test_invalid_identity_and_group_geometry():
    core, multi = _tiny()
    state = multi.initial_state(1)
    with pytest.raises(ValueError, match="unique"):
        multi.advance_group(
            state, _selected(core, ("a", "b")), tracks=("same", "same"),
            role_ids=_roles((1, 2)),
        )
    with pytest.raises(ValueError, match="roles|role"):
        multi.advance_group(
            state, _selected(core, ("a", "b")), tracks=("a", "b"),
            role_ids=torch.tensor([3]),
        )
    with pytest.raises(ValueError, match="control"):
        multi.predict_group(
            state, tracks=("a", "b"),
            control=torch.randn(1, 3, core.config.control_dim),
        )


def test_observe_text_group():
    core, multi = _tiny()
    state = multi.observe_text_group(
        multi.initial_state(1), (b"Hello", b"World"), tracks=("speaker", "screen")
    )
    assert state.history.slots == 2 * core.config.n_facets
    assert state.frontier_tracks == ("speaker", "screen")
    assert state.meridian_sizes == (2,)


def test_grouped_generation_tape_and_exact_replay():
    core, multi = _tiny()
    model = FacetedGenerator(core, ByteCodec.byte_fallback())
    serving = MultiTrackGenerator(model)
    state = multi.observe_text_group(
        multi.initial_state(1), (b"Say hi",), tracks=("user",)
    )
    specs = (ActionSpec(max_tokens=8), ActionSpec(max_tokens=8))
    sample = serving.generate_text_group(
        state, tracks=("answer", "caption"), specs=specs, effort=1,
        generator=torch.Generator().manual_seed(5),
    )
    tape = GroupActionTape.from_json(sample.tape.to_json())
    assert tape.facet_shape == (2, core.config.n_facets, core.config.facet_dim)
    assert tape.effort == 1
    assert len(tape.event_log_probs) == 2
    assert sum(tape.event_log_probs) == pytest.approx(tape.behavior_log_prob, rel=1e-5)
    assert sample.state.meridian_sizes == (1, 2)
    replay = serving.replay_text_group(state, tape, specs=specs, effort=1)
    torch.testing.assert_close(replay.state.final, sample.state.final, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(replay.log_prob, sample.log_prob, atol=1e-4, rtol=1e-5)
    with pytest.raises(ValueError, match="specifications"):
        serving.replay_text_group(state, tape, specs=(ActionSpec(max_tokens=9),) * 2, effort=1)
    with pytest.raises(ValueError, match="effort"):
        serving.replay_text_group(state, tape, specs=specs, effort=3)
    tampered = GroupActionTape.to_json(tape)
    import json
    corrupted = json.loads(tampered)
    corrupted["event_log_probs"][0] += 4
    with pytest.raises(ValueError, match="does not sum"):
        GroupActionTape.from_json(json.dumps(corrupted))


def test_group_history_backpropagates_across_meridians():
    core, multi = _tiny()
    raw = _selected(core, ("first", "second")).requires_grad_(True)
    state = multi.advance_group(
        multi.initial_state(1), raw, tracks=("first", "second"), role_ids=_roles((1, 2))
    )
    latter = multi.predict_group(state, tracks=("reply",), effort=1)
    latter.square().sum().backward()
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())
    assert float(raw.grad.abs().sum()) > 0


def test_meridian_eviction_is_after_context_read(monkeypatch):
    core, multi = _tiny(history=3)
    state = multi.initial_state(1)
    for tracks in (("a", "b"), ("c",)):
        state = multi.advance_group(
            state, _selected(core, tracks), tracks=tracks, role_ids=_roles(tracks)
        )
    seen = []
    original = core.central.forward

    def spy(raw, history, *args, **kwargs):
        seen.append(history.slots)
        return original(raw, history, *args, **kwargs)

    monkeypatch.setattr(core.central, "forward", spy)
    final = multi.advance_group(
        state, _selected(core, ("a", "b")), tracks=("a", "b"), role_ids=_roles((1, 2))
    )
    assert seen == [3 * core.config.n_facets]
    assert final.meridian_sizes == (1, 2)
    assert final.history_tracks.tolist() == [[2, 0, 1]]
    assert int(final.meridians_seen[0]) == 3
    assert int(final.events_seen[0]) == 5


def test_track_registration_order_does_not_change_joint_causal_geometry():
    core, multi = _tiny()
    initial = multi.initial_state(1)
    raw = _selected(core, ("left", "right"))
    roles = _roles((1, 2))
    ab = multi.advance_group(initial, raw, tracks=("left", "right"), role_ids=roles)
    ba = multi.advance_group(
        initial, raw[:, [1, 0]], tracks=("right", "left"), role_ids=roles[:, [1, 0]]
    )
    torch.testing.assert_close(ab.final[:, 0], ba.final[:, 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ab.final[:, 1], ba.final[:, 0], atol=1e-5, rtol=1e-5)
    next_raw = _selected(core, ("left",))
    next_roles = _roles((1,))
    ab2 = multi.advance_group(ab, next_raw, tracks=("left",), role_ids=next_roles)
    ba2 = multi.advance_group(ba, next_raw, tracks=("left",), role_ids=next_roles)
    torch.testing.assert_close(ab2.final, ba2.final, atol=1e-5, rtol=1e-5)


def test_every_member_of_a_meridian_has_the_same_rope_time():
    core, multi = _tiny()
    first = multi.advance_group(
        multi.initial_state(1), _selected(core, ("mic", "camera")),
        tracks=("mic", "camera"), role_ids=_roles((1, 2)),
    )
    expected = core.central.memory(
        first.final.flatten(1, 2),
        temporal_positions=torch.tensor([0], dtype=torch.long),
        slot_codes=core.central.slot_codes().repeat(2, 1),
    )
    torch.testing.assert_close(first.history.keys, expected.keys, atol=0, rtol=0)
    second = multi.advance_group(
        first, _selected(core, ("mic",)), tracks=("mic",), role_ids=_roles((1,))
    )
    expected_next = core.central.memory(
        second.final.flatten(1, 2),
        temporal_positions=torch.tensor([1], dtype=torch.long),
        slot_codes=core.central.slot_codes(),
    )
    torch.testing.assert_close(
        second.history.keys[:, :, -core.config.n_facets:],
        expected_next.keys, atol=0, rtol=0,
    )
    torch.testing.assert_close(
        second.history.keys[:, :, :2 * core.config.n_facets],
        first.history.keys, atol=0, rtol=0,
    )


def test_oversized_meridian_fails_before_central_compute(monkeypatch):
    core, multi = _tiny(history=2)
    initial = multi.initial_state(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Central work must not run for an invalid group")

    monkeypatch.setattr(core.central, "forward", forbidden)
    names = ("a", "b", "c")
    with pytest.raises(ValueError, match="historical event capacity"):
        multi.advance_group(
            initial, _selected(core, names), tracks=names, role_ids=_roles(names)
        )


def test_group_gaussian_noise_tracks_identity_not_physical_order():
    mean = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(1, 3, 4, 8)
    scales = torch.zeros_like(mean) - 2
    names = ("z", "a", "q")
    original = MultiTrackGenerator._sample_policy(
        DiagonalGaussian(mean, scales), names, torch.Generator().manual_seed(102)
    )
    order = (1, 2, 0)
    permuted = MultiTrackGenerator._sample_policy(
        DiagonalGaussian(mean[:, order], scales[:, order]),
        tuple(names[i] for i in order),
        torch.Generator().manual_seed(102),
    )
    torch.testing.assert_close(permuted, original[:, order], atol=0, rtol=0)
