"""Contextual v4 objectives exercised through the real window/optimizer engine."""

import copy
from dataclasses import replace

import pytest
import torch

from megatoken.data import WindowConfig, iter_windows
from megatoken.data.schema import Episode, Lineage, LogicalAction, PhysicalRecord, digest_bytes
from megatoken.generation import ByteCodec
from megatoken.loading import build_model, load_model_config
from megatoken.model import ModelConfig
from megatoken.tensors import byte_tensor
from megatoken.training.config import TrainConfig
from megatoken.training.engine import batch_counters, forward_batch, prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.stream import EpisodeWindow


def window_fixture(*, mask_first=False):
    codec = ByteCodec.byte_fallback()
    episode = Episode(
        "tower",
        (
            PhysicalRecord("prompt", "user", b"go"),
            PhysicalRecord(
                "first",
                "assistant",
                b"one",
                turn_index=1,
                policy_mask=True,
                free_spans=() if mask_first else None,
            ),
            PhysicalRecord("second", "assistant", b"two", turn_index=2, policy_mask=True),
        ),
        Lineage("fixture", "v4", "tower-family"),
        control_seed=b"public control",
    )
    windows = list(iter_windows(episode, codec, WindowConfig(max_records=4, max_decisions=4)))
    assert len(windows) == 1 and windows[0].start == 0
    return codec, [EpisodeWindow(episode, windows[0])]


def config_for(mode, *, reconstruction=0.0):
    config = TrainConfig(
        phase="B0" if mode == "bootstrap" else "P1",
        representation_mode=mode,
        precision="fp32",
        device="cpu",
        generated_capacity=8,
        microbatch_episodes=1,
        use_stage_mixture=False,
        use_effort_mixture=False,
        effort=1,
    )
    weights = dict.fromkeys(config.loss_weights, 0.0)
    weights.update(lexical=1.0, reconstruction=reconstruction)
    return replace(config, loss_weights=weights)


@pytest.mark.parametrize("mode", ["bootstrap", "recurrent"])
def test_real_trainer_keeps_selected_tower_and_only_appends_live_context(mode, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(882)
    codec, windows = window_fixture(mask_first=True)
    model = build_model(ModelConfig.tiny(checkpoint_stages=True), codec)
    config = config_for(mode)
    (batch,) = prepare_batches(model, windows, config)
    assert batch.targets[1].denominators["lexical"] == 0
    assert batch.targets[2].denominators["lexical"] == 3
    captured, encodings = [], {}
    original_step, original_encode = model.supervised_step, model.core.encode_chunk

    def encode(values, lengths=None):
        result = original_encode(values, lengths)
        payload = bytes(values[0, : int(lengths[0])].tolist())
        encodings[payload] = result
        return result

    def supervised(state, *args, **kwargs):
        result = original_step(state, *args, **kwargs)
        result.state.facets.retain_grad()
        captured.append((state, result.state))
        return result

    monkeypatch.setattr(model.core, "encode_chunk", encode)
    monkeypatch.setattr(model, "supervised_step", supervised)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    first_source, first_selected = captured[0]
    second_source, second_selected = captured[1]
    assert second_source is first_selected
    assert [
        state.records_seen.item() for state in (first_source, first_selected, second_selected)
    ] == [1, 2, 3]
    assert second_selected.history.slots == 3 * model.config.n_facets
    assert first_selected.final is not None
    # The first chunk has no own lexical objective. Its gradient must come from
    # the later target through raw recurrence and/or the selected historical KV.
    assert first_selected.facets.grad is not None
    assert first_selected.facets.grad.norm() > 0
    if mode == "bootstrap":
        assert first_selected.facets is encodings[b"one"]
    else:
        assert not torch.equal(first_selected.facets, encodings[b"one"])
    assert batch_counters([batch]).P == 6  # Two actual CE passes on the second target.


def test_direct_encoder_auxiliary_cannot_change_main_gradients_or_clipped_update():
    torch.set_num_threads(1)
    torch.manual_seed(771)
    codec, windows = window_fixture()
    initial = build_model(ModelConfig.tiny(checkpoint_stages=True), codec)
    models = [initial, copy.deepcopy(initial)]
    reports, optimizers, gradients = [], [], []
    for model, weight in zip(models, (1.0, 1000.0), strict=True):
        config = config_for("recurrent", reconstruction=weight)
        (batch,) = prepare_batches(model, windows, config)
        assert batch.counts["encoder_alignment"] == 0  # No invented native labels.
        assert batch.counts["facet_prediction"] == 0  # No E-to-forecast pull.
        optimizer = MasterAdamW(model, lr=0.01)
        optimizers.append(optimizer)
        optimizer.isolate_gradient_parameters("encoder", model.core.encoder_parameters())
        encoder_ids = {id(parameter) for parameter in model.core.encoder_parameters()}
        torch.manual_seed(639)
        ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
        forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
        ledger.finish()
        gradients.append(
            {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in model.named_parameters()
                if id(parameter) not in encoder_ids
            }
        )
        reports.append(optimizer.step(max_norm=0.01))
    try:
        assert reports[0]["gradient_norm_model"] == reports[1]["gradient_norm_model"]
        assert reports[1]["gradient_norm_encoder"] > 100 * reports[0]["gradient_norm_encoder"]
        for name, gradient in gradients[0].items():
            if gradient is None:
                assert gradients[1][name] is None
            else:
                torch.testing.assert_close(gradient, gradients[1][name], atol=0, rtol=0)
            # Compare served parameters, precise masters and moments after an
            # actual independently clipped optimizer transaction.
            torch.testing.assert_close(
                optimizers[0].masters[name], optimizers[1].masters[name], atol=0, rtol=0
            )
            left = optimizers[0].optimizer.state.get(optimizers[0].masters[name], {})
            right = optimizers[1].optimizer.state.get(optimizers[1].masters[name], {})
            assert left.keys() == right.keys()
            for field in left:
                torch.testing.assert_close(left[field], right[field], atol=0, rtol=0)
    finally:
        for optimizer in optimizers:
            optimizer.close()


def test_canonical_presets_and_legacy_architecture_rejection():
    assert load_model_config("32x512").name == "32x512"
    assert load_model_config("32x256").name == "32x256"
    for alias in ("recurrent32", "compact32"):
        with pytest.raises(ValueError, match="command-line aliases"):
            load_model_config(alias)
    with pytest.raises(ValueError, match="architecture_version=8"):
        ModelConfig.from_dict({**ModelConfig.tiny().to_dict(), "architecture_version": 3})


def test_offline_tool_result_uses_recorded_action_not_counterfactual_sample(monkeypatch):
    """An offline tool result must never condition on a different sampled call."""
    torch.set_num_threads(1)
    torch.manual_seed(901)
    call = b'{"name":"read","arguments":{"path":"A"}}'
    episode = Episode(
        "tool-causality",
        (
            PhysicalRecord("prompt", "user", b"read A", turn_index=0),
            PhysicalRecord(
                "call",
                "assistant",
                call,
                turn_index=1,
                policy_mask=True,
                action_id="a1",
                execution_id="e1",
            ),
            PhysicalRecord(
                "observation",
                "tool",
                b"A contents",
                turn_index=2,
                result_of="a1",
                execution_id="e1",
                provenance="executed",
            ),
            PhysicalRecord("next", "assistant", b"done", turn_index=3, policy_mask=True),
        ),
        Lineage("fixture", "v4", "tool-causality"),
        actions=(
            LogicalAction(
                "a1",
                "call1",
                "read",
                '{"path":"A"}',
                ("call",),
                digest_bytes(call),
                execution_id="e1",
            ),
        ),
    )
    codec = ByteCodec.byte_fallback()
    windows = list(iter_windows(episode, codec, WindowConfig(max_records=8, max_decisions=4)))
    assert len(windows) == 1
    model = build_model(ModelConfig.tiny(), codec)
    config = replace(config_for("recurrent"), generated_capacity=64)
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, windows[0])], config)
    seen = []
    original = model.supervised_step

    def capture(state, *args, **kwargs):
        result = original(state, *args, **kwargs)
        seen.append((kwargs.get("teacher_forced_action"), result.state.facets.detach().clone()))
        return result

    monkeypatch.setattr(model, "supervised_step", capture)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    assert [forced for forced, _ in seen] == [True, False]
    ids, lengths = byte_tensor([call], torch.device("cpu"))
    with torch.no_grad():
        encoded_recorded_call = model.core.encode_chunk(ids, lengths)
    torch.testing.assert_close(seen[0][1], encoded_recorded_call, atol=0, rtol=0)
