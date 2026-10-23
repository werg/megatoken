"""Three-stage central KV reasoning, temporal RoPE, and immutable cache ownership."""

from dataclasses import replace

import pytest
import torch

from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.attention import AttentionFeatures, AttentionRead, temporal_rope
from megatoken.model.core import P_HISTORY_READ_STAGES


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(182)


def test_temporal_rope_is_relative_and_preserves_same_record_attention():
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn_like(query)
    current = torch.tensor([20_000_000, 45_000_000], dtype=torch.long)
    preceding = current - torch.tensor([17, 381], dtype=torch.long)
    same_time = (temporal_rope(query, current) * temporal_rope(key, current)).sum(-1)
    torch.testing.assert_close(same_time, (query * key).sum(-1), atol=3e-5, rtol=3e-5)
    relative = (temporal_rope(query, current) * temporal_rope(key, preceding)).sum(-1)
    shift = 1_337_000
    translated = (
        temporal_rope(query, current + shift) * temporal_rope(key, preceding + shift)
    ).sum(-1)
    torch.testing.assert_close(relative, translated, atol=3e-5, rtol=3e-5)
    # FP32 position rounding would alias these consecutive late timestamps.
    first = temporal_rope(query, torch.tensor([16_777_216, 16_777_216]))
    second = temporal_rope(query, torch.tensor([16_777_217, 16_777_217]))
    assert (first - second).abs().max() > 0.01


def test_facet_slot_codes_address_unordered_slots_without_temporal_offsets():
    core = FacetedCore(ModelConfig.tiny()).eval()
    with torch.no_grad():
        core.central.slot_embedding.zero_()
        core.central.slot_embedding[0, 0] = 4.0
        core.central.slot_embedding[1, 0] = -4.0
    features = torch.ones(1, core.config.n_facets, core.config.facet_dim)
    keys = core.central.memory(
        features,
        temporal_positions=torch.zeros(1, dtype=torch.long),
        slot_codes=core.central.slot_codes(),
    ).keys
    assert not torch.allclose(keys[:, :, 0], keys[:, :, 1])
    # All facets share the same *temporal* phase, not one step per slot.
    raw_keys = core.central.memory(features, slot_codes=core.central.slot_codes()).keys
    torch.testing.assert_close(keys, raw_keys, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_three_temporary_writes_are_read_once_and_never_enter_cache(checkpointing, monkeypatch):
    core = FacetedCore(ModelConfig.tiny(checkpoint_stages=checkpointing))
    reads = [[] for _ in core.central.layers]
    for index, layer in enumerate(core.central.layers):
        original = layer.attention.read

        def recorded(query, features, *, _index=index, _original=original, **kwargs):
            reads[_index].append(features.keys.shape[2])
            return _original(query, features, **kwargs)

        monkeypatch.setattr(layer.attention, "read", recorded)

    raw = torch.randn(1, core.config.n_facets, core.config.facet_dim, requires_grad=True)
    initial = core.initial_state(1)
    state = core.advance_raw(initial, raw)
    assert reads == [[4], [4, 4], [4, 4], [4, 4]]
    assert state.history.slots == core.config.n_facets
    final = core.central.memory(
        state.final,
        temporal_positions=initial.records_seen,
        slot_codes=core.central.slot_codes(),
    )
    torch.testing.assert_close(state.history.keys, final.keys, rtol=0, atol=0)
    torch.testing.assert_close(state.history.values, final.values, rtol=0, atol=0)

    core.predict_facets(state, effort=1).square().mean().backward()
    assert raw.grad is not None and raw.grad.norm() > 0
    for writer in core.central.temporary_memories:
        assert writer.key_value.router.weight.grad is not None
        assert writer.key_value.router.weight.grad.norm() > 0
        assert writer.key_value._bank.weight.grad is not None
        assert writer.key_value._bank.weight.grad.norm() > 0
    for layer in core.central.layers[1:]:
        assert layer.temporary_gate.grad is not None
        assert layer.temporary_gate.grad.abs() > 0


def test_predictor_reads_history_at_only_four_fixed_stages(monkeypatch):
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.initial_state(1)
    for _ in range(3):
        state = core.advance_raw(state, torch.randn(1, core.config.n_facets, core.config.facet_dim))
    recorded = [[] for _ in core.stages]
    for index, stage in enumerate(core.stages):
        original = stage.attention.read

        def spy(query, features, *, _index=index, _original=original, **kwargs):
            recorded[_index].append(features.keys.shape[2])
            return _original(query, features, **kwargs)

        monkeypatch.setattr(stage.attention, "read", spy)

    history_before = (state.history.keys.clone(), state.history.values.clone())
    core.predict_facets(state, effort=3)
    assert P_HISTORY_READ_STAGES == (3, 6, 9, 12)
    assert len(recorded) == core.config.prediction_layers
    for stage, reads in enumerate(recorded, 1):
        expected = core.config.n_facets + (
            state.history.slots if stage in P_HISTORY_READ_STAGES else 0
        )
        assert reads == [expected]
    torch.testing.assert_close(state.history.keys, history_before[0], atol=0, rtol=0)
    torch.testing.assert_close(state.history.values, history_before[1], atol=0, rtol=0)


def test_old_rotated_keys_and_values_survive_until_full_chunk_eviction():
    config = ModelConfig.tiny(max_history_chunks=3)
    core = FacetedCore(config).eval()
    raw = torch.randn(1, config.n_facets, config.facet_dim)
    first = core.advance_raw(core.initial_state(1), raw)
    rotated = first.history.keys.clone()
    original_values = first.history.values.clone()
    second = core.advance_raw(first, torch.randn_like(raw))
    third = core.advance_raw(second, torch.randn_like(raw))
    assert third.history.slots == 3 * config.n_facets
    torch.testing.assert_close(
        third.history.keys[:, :, : config.n_facets],
        rotated,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        third.history.values[:, :, : config.n_facets],
        original_values,
        rtol=0,
        atol=0,
    )
    fourth = core.advance_raw(third, torch.randn_like(raw))
    assert fourth.history.slots == 3 * config.n_facets
    torch.testing.assert_close(
        fourth.history.keys[:, :, : config.n_facets],
        second.history.keys[:, :, config.n_facets :],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fourth.history.values[:, :, : config.n_facets],
        second.history.values[:, :, config.n_facets :],
        rtol=0,
        atol=0,
    )


def test_old_architecture_version_cannot_reuse_rotary_history_cache():
    config = ModelConfig.tiny()
    with pytest.raises(ValueError, match="requires version 11"):
        ModelConfig.tiny(architecture_version=5)
    with pytest.raises(ValueError, match="architecture_version=11"):
        ModelConfig.from_dict({**config.to_dict(), "architecture_version": 5})
    core = FacetedCore(config)
    state = core.advance_raw(core.initial_state(1), torch.randn(1, 4, 32))
    with pytest.raises(ValueError, match="geometry"):
        core.predict_facets(replace(state, geometry=(5, *state.geometry[1:])))


def test_same_track_attention_bias_respects_identity_not_track_number():
    torch.manual_seed(55)
    attention = AttentionRead(8, query_heads=2, kv_heads=1, head_dim=4, track_relations=True).eval()
    query = torch.randn(1, 2, 8)
    keys = torch.randn(1, 1, 4, 4)
    values = torch.randn_like(keys)
    tracks = torch.tensor([[4, 8, 4, 8]])
    queries = torch.tensor([[4, 8]])
    with torch.no_grad():
        attention.same_track_bias.fill_(3.0)
    features = AttentionFeatures(keys, values, None, tracks)
    with pytest.raises(ValueError, match="both query and key"):
        attention.read(query, features)
    result = attention.read(query, features, query_track_ids=queries)
    perm = torch.tensor([1, 3, 0, 2])
    scrambled = AttentionFeatures(keys[:, :, perm], values[:, :, perm], None, tracks[:, perm])
    torch.testing.assert_close(
        result,
        attention.read(query, scrambled, query_track_ids=queries),
        atol=1e-6,
        rtol=1e-6,
    )
    renamed = AttentionFeatures(keys, values, None, tracks.where(tracks == 4, 17) + 50)
    torch.testing.assert_close(
        result,
        attention.read(query, renamed, query_track_ids=queries.where(queries == 4, 17) + 50),
        atol=1e-6,
        rtol=1e-6,
    )
    with torch.no_grad():
        attention.same_track_bias.zero_()
    assert not torch.allclose(result, attention.read(query, features, query_track_ids=queries))

