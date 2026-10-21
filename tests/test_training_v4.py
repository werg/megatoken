"""Contextual v4 objectives exercised through the real window/optimizer engine."""

import copy
from dataclasses import replace

import pytest
import torch
from retrieval_fixture import MODEL_OVERRIDES, make_episode, retrieval_aux, retrieval_train_config

from megatoken.data import WindowConfig, iter_windows
from megatoken.generation import ByteCodec
from megatoken.loading import build_model, load_model_config
from megatoken.model import ModelConfig
from megatoken.training.engine import prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.stream import EpisodeWindow


def window_fixture():
    codec = ByteCodec.byte_fallback()
    episode = make_episode(0)
    windows = list(iter_windows(episode, codec, WindowConfig(max_records=4, max_decisions=4)))
    assert len(windows) == 1 and windows[0].start == 0
    return codec, [EpisodeWindow(episode, windows[0])]


def config_for(base, *, reconstruction=0.0):
    config = base
    weights = dict.fromkeys(config.loss_weights, 0.0)
    weights.update(lexical=1.0, reconstruction=reconstruction)
    return replace(config, loss_weights=weights)


def test_direct_encoder_auxiliary_cannot_change_main_gradients_or_clipped_update(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(771)
    codec, windows = window_fixture()
    base = retrieval_train_config(
        tmp_path,
        phase="P1",
        representation_mode="recurrent",
        microbatch_episodes=1,
        max_accumulation_windows=1,
    )
    initial = build_model(
        ModelConfig.tiny(checkpoint_stages=True, **MODEL_OVERRIDES),
        codec,
    )
    models = [initial, copy.deepcopy(initial)]
    reports, optimizers, gradients = [], [], []
    for model, weight in zip(models, (1.0, 1000.0), strict=True):
        config = config_for(base, reconstruction=weight)
        aux = retrieval_aux(config, supervised=False)
        (batch,) = prepare_batches(model, windows, config)
        assert batch.counts["encoder_alignment"] == 0  # No invented native labels.
        assert batch.counts["facet_prediction"] == 0  # No E-to-forecast pull.
        optimizer = MasterAdamW(model, lr=0.01)
        optimizers.append(optimizer)
        optimizer.isolate_gradient_parameters("encoder", model.core.encoder_parameters())
        encoder_ids = {id(parameter) for parameter in model.core.encoder_parameters()}
        torch.manual_seed(639)
        ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
        forward_meridian_batch(model, batch, ledger, config, torch.device("cpu"), 0, aux)
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
    with pytest.raises(ValueError, match="architecture_version=9"):
        ModelConfig.from_dict({**ModelConfig.tiny().to_dict(), "architecture_version": 3})
