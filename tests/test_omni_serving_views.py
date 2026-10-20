"""Serving/training must use the same masked frozen feature fusion through E."""
import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent


def test_served_audio_fuses_whistle_and_mimi_before_one_facets_readout():
    torch.manual_seed(89)
    whistle = "pinned-whistle-semantic-v1"
    mimi = "pinned-pocket-mimi-acoustic-v1"
    model = build_model(
        ModelConfig.tiny(media_feature_views={whistle: 6, mimi: 4}),
        ByteCodec.byte_fallback(),
    )
    model.eval()
    core = MultiTrackGenerator(model).tracks
    state = core.initial_state(1, device=torch.device("cpu"))
    primary = torch.randn(1, 3, 6)
    primary_mask = torch.tensor([[True, True, False]])
    acoustic = torch.randn(1, 2, 4)
    acoustic_mask = torch.ones(1, 2, dtype=torch.bool)
    event = ObservedEvent(
        track="audio", modality="audio", payload=primary,
        extractor=whistle, valid_mask=primary_mask,
        source_views=((mimi, acoustic, acoustic_mask),),
    )
    with torch.no_grad():
        fused = core.observe_group(state, (event,), encoder_grad=False)
        changed = primary.clone()
        changed[:, -1] = 9999.0
        repeated = core.observe_group(
            state,
            (ObservedEvent("audio", "audio", changed, extractor=whistle,
                           valid_mask=primary_mask,
                           source_views=((mimi, acoustic, acoustic_mask),)),),
            encoder_grad=False,
        )
    assert fused.raw.shape == (1, 1, model.config.n_facets, model.config.facet_dim)
    assert torch.allclose(fused.raw, repeated.raw, atol=1e-5)


def test_serving_refuses_unidentified_source_views():
    with pytest.raises(ValueError, match="primary extractor"):
        ObservedEvent(
            "audio", "audio", torch.randn(1, 4, 6),
            source_views=(("external", torch.randn(1, 3, 4),
                           torch.ones(1, 3, dtype=torch.bool)),),
        )
    with pytest.raises(ValueError, match="nonempty boolean"):
        ObservedEvent(
            "audio", "audio", torch.randn(1, 4, 6), extractor="pinned-v1",
            valid_mask=torch.zeros(1, 4, dtype=torch.bool),
        )
