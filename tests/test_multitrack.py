"""Block-causal multi-track reference path and the exact joint action tape."""

import pytest
import torch

from megatoken.generation.codec import ByteCodec
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
    assert sample.state.meridian_sizes == (1, 2)
    replay = serving.replay_text_group(state, tape, specs=specs, effort=1)
    torch.testing.assert_close(replay.state.final, sample.state.final, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(replay.log_prob, sample.log_prob, atol=1e-4, rtol=1e-5)
    with pytest.raises(ValueError, match="specifications"):
        serving.replay_text_group(state, tape, specs=(ActionSpec(max_tokens=9),) * 2, effort=1)


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
