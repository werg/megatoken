"""Real media assets through the existing grouped loss ledger and checkpoint path."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from megatoken.data import Episode, Lineage, PhysicalRecord, WindowConfig, iter_windows
from megatoken.data.media_assets import (
    MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore,
)
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec, encode_record
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.training.config import TrainConfig
from megatoken.training.engine import batch_counters, forward_batch, prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.stream import EpisodeWindow


def _case(tmp_path, *, external=False):
    assets = TensorAssetStore(tmp_path)
    wave = np.sin(np.arange(512, dtype=np.float32) / 21).astype("<f4")
    image = np.linspace(0, 1, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    wave_asset = assets.put(wave, modality="audio", sample_rate=16000)
    image_asset = assets.put(image, modality="image")
    feature_id = "pinned-audio-test-v1"
    features = assets.put(
        np.ones((12, 16), dtype=np.float32), modality="audio",
        kind="features", extractor=feature_id, sample_rate=16000,
    ) if external else wave_asset
    observed = MediaRecord("audio", features)
    generated = MediaRecord("image", image_asset, image_asset)
    tracks = (
        TrackSpec("mic", "user"), TrackSpec("prompt", "user"),
        TrackSpec("caption", "assistant"), TrackSpec("picture", "assistant"),
    )
    records = (
        PhysicalRecord("sound", "user", observed.to_bytes(),
                       media_type=MEDIA_RECORD_TYPE),
        PhysicalRecord("prompt", "user", b"Describe the sound and draw a scene"),
        PhysicalRecord("caption", "assistant", b"bright", turn_index=1,
                       policy_mask=True),
        PhysicalRecord("picture", "assistant", generated.to_bytes(),
                       turn_index=1, policy_mask=True,
                       media_type=MEDIA_RECORD_TYPE),
    )
    meridians = (
        Meridian(0, (
            TrackEvent("sound", "mic", "audio", 0, "user",
                       asset_sha256=features.sha256),
            TrackEvent("prompt", "prompt", "text", 0, "user"),
        )),
        Meridian(1, (
            TrackEvent("caption", "caption", "text", 1, "assistant"),
            TrackEvent("picture", "picture", "image", 1, "assistant",
                       asset_sha256=image_asset.sha256),
        )),
    )
    episode = Episode(
        "omni", records, Lineage("local", "revision", "family"),
        tracks=tracks, meridians=meridians,
    )
    model = build_model(
        ModelConfig.tiny(**({
            "audio_feature_dim": 16, "audio_feature_id": feature_id,
        } if external else {})),
        ByteCodec.byte_fallback(),
    )
    config = TrainConfig(
        phase="B0", dataset=str(tmp_path), output=str(tmp_path / "run"),
        model="tiny", device="cpu", precision="fp32",
        budget_meter="G", budget=4, eval_every=0, effort=1,
        use_stage_mixture=False, use_effort_mixture=False,
        max_history_tokens=128, generated_capacity=16,
    )
    return assets, episode, model, config


def test_media_reference_has_zero_text_tokens_and_corruption_fails(tmp_path):
    assets, episode, _, _ = _case(tmp_path)
    codec = ByteCodec.byte_fallback()
    for record in (episode.records[0], episode.records[-1]):
        encoded = encode_record(record, codec)
        assert encoded.tokens == encoded.P == encoded.positions == 0
    ref = MediaRecord.from_bytes(episode.records[0].payload).source
    expected = assets.load(ref).copy()
    assert expected.shape == (512,)
    path = tmp_path / ref.relative_path
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA256"):
        assets.load(ref)


@pytest.mark.parametrize("external", (False, True))
@pytest.mark.parametrize("mode", ("bootstrap", "recurrent"))
def test_mixed_media_meridian_gradients_and_loss_counts(tmp_path, external, mode):
    torch.set_num_threads(1)
    torch.manual_seed(901)
    _, episode, model, config = _case(tmp_path, external=external)
    config = replace(
        config,
        representation_mode=mode,
        phase="B0" if mode == "bootstrap" else "P1",
    )
    windows = list(iter_windows(
        episode, ByteCodec.byte_fallback(),
        WindowConfig(max_records=8, max_decisions=4),
    ))
    batch, = prepare_batches(
        model, [EpisodeWindow(episode, windows[0])], config,
    )
    assert batch.counts["visual_prediction"] == 1
    assert batch.counts["visual_reconstruction"] == 1
    assert batch.counts["audio_prediction"] == 0
    assert batch.counts["lexical"] == len(b"bright")
    assert batch.counts["facet_prediction"] == (
        2 * model.config.n_facets * model.config.facet_dim
        if mode == "bootstrap" else 0
    )
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    model.core.set_model_version(0)
    forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    assert model.media_realizers.visual_filter[0].weight.grad is not None
    assert model.core.frontend.facet_queries.grad is not None
    if external:
        # Recurrent observations are intentionally detached from main gradients.
        if mode == "bootstrap":
            assert model.core.media_stems.external_audio.weight.grad is not None
        else:
            assert model.core.media_stems.external_audio.weight.grad is None
    counters = batch_counters([batch])
    assert counters.G == len(episode.records)
    assert counters.media_events == 2
    assert counters.image_events == 1
    # Offline feature caches do not assert an original sample count without
    # a verified linked raw source; only the media-event count is universal.
    assert counters.audio_samples == (0 if external else 512)
    assert counters.P == 2 * len(b"bright")


def test_feature_revision_does_not_fall_back_to_unpinned_source(tmp_path):
    _, episode, model, config = _case(tmp_path, external=True)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    window = next(iter_windows(episode, ByteCodec.byte_fallback()))
    with pytest.raises(ValueError, match="revision/width|fingerprint"):
        prepare_batches(model, [EpisodeWindow(episode, window)], config)


def test_joint_text_audio_and_visual_targets_have_distinct_objectives(tmp_path):
    from dataclasses import replace

    torch.set_num_threads(1)
    assets, ep, model, config = _case(tmp_path)
    raw = assets.put(
        np.sin(np.arange(512, dtype=np.float32) / 21),
        modality="audio", sample_rate=16000,
    )
    audio = MediaRecord("audio", raw, raw)
    voice_record = PhysicalRecord(
        "voice", "assistant", audio.to_bytes(), turn_index=1,
        media_type=MEDIA_RECORD_TYPE, policy_mask=True,
    )
    second = Meridian(1, (
        *ep.meridians[1].events,
        TrackEvent("voice", "voice", "audio", 1, "assistant",
                   asset_sha256=raw.sha256),
    ))
    ep = replace(ep, records=(*ep.records, voice_record),
                 tracks=(*ep.tracks, TrackSpec("voice", "assistant")),
                 meridians=(ep.meridians[0], second))
    window = next(iter_windows(ep, ByteCodec.byte_fallback()))
    batch, = prepare_batches(
        model, [EpisodeWindow(ep, window)], config,
    )
    assert batch.counts["audio_prediction"] == 1
    assert batch.counts["audio_reconstruction"] == 1
    assert batch.counts["visual_prediction"] == 1
    assert batch.counts["lexical"] == len(b"bright")
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    assert model.media_realizers.audio_filter[0].weight.grad is not None
    assert model.media_realizers.visual_filter[0].weight.grad is not None
    assert batch_counters([batch]).audio_samples == 1024


def test_real_event_budget_checkpoint_and_media_data(tmp_path):
    from megatoken.data import DedupConfig, DedupStore
    from megatoken.data.storage import export_prepared
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    _, ep, _, config = _case(tmp_path)
    source = DedupStore(
        tmp_path / "index.sqlite", DedupConfig(
            train_fraction=1.0, validation_fraction=0.0,
        ),
    )
    try:
        with source.transaction():
            source.add(ep, ByteCodec.byte_fallback())
        source.freeze()
        export_prepared(
            source, tmp_path, codec=ByteCodec.byte_fallback(),
            source_receipts=[], pipeline_fingerprint="omni-event-pilot-v1",
        )
    finally:
        source.close()
    # B0's text-token budget would not account for image/audio events. G is an
    # explicit event budget with its own advance/accumulation and checkpoint.
    config = replace(
        config, max_steps=1, accumulate_events=4, accumulate_scored=100_000,
        accumulate_ingested=100_000, max_accumulation_windows=1,
        microbatch_episodes=1, checkpoint_every=0, use_stage_mixture=False,
    )
    result = train(config)
    payload, _ = load_checkpoint(result["checkpoint"])
    assert result["global_step"] == 1
    assert result["phase_consumed"] == result["counters"]["G"]
    assert result["counters"]["G"] == 4
    assert result["counters"]["media_events"] == 2
    assert payload["phase_consumed"] == 4


def test_video_training_is_one_32_facet_megatoken(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(922)
    assets, ep, model, config = _case(tmp_path)
    clip = np.stack((np.zeros((3, 8, 8), dtype=np.float32),
                     np.ones((3, 8, 8), dtype=np.float32)))
    video = assets.put(clip, modality="video", fps_num=8)
    replacement = MediaRecord("video", video, video)
    from megatoken.data.schema import Meridian, TrackEvent

    records = (*ep.records[:-1], replace(
        ep.records[-1], payload=replacement.to_bytes(),
    ))
    changed = replace(ep, records=records, meridians=(
        ep.meridians[0],
        Meridian(1, (ep.meridians[1].events[0],
                     TrackEvent("picture", "picture", "video", 1, "assistant",
                                asset_sha256=video.sha256))),
    ))
    window = next(iter_windows(changed, ByteCodec.byte_fallback()))
    batch, = prepare_batches(model, [EpisodeWindow(changed, window)], config)
    assert batch.counts["visual_prediction"] == 1
    assert batch_counters([batch]).video_frames == 2
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    assert model.media_realizers.visual_filter[0].weight.grad is not None


def test_generated_pretrained_vision_features_train_e_in_recurrent_mode(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(937)
    assets, ep, _, config = _case(tmp_path)
    reference = MediaRecord.from_bytes(ep.records[-1].payload)
    identifier = "vision-features-pinned-test-v1"
    cached = assets.put(
        np.arange(12 * 16, dtype=np.float32).reshape(12, 16) / 192,
        modality="image", kind="features", extractor=identifier,
    )
    target = MediaRecord("image", cached, reference.target)
    records = (*ep.records[:-1], replace(ep.records[-1],
                                        payload=target.to_bytes()))
    meridian = Meridian(1, (
        ep.meridians[1].events[0],
        TrackEvent("picture", "picture", "image", 1, "assistant",
                   asset_sha256=cached.sha256),
    ))
    ep = replace(ep, records=records, meridians=(ep.meridians[0], meridian))
    model = build_model(ModelConfig.tiny(
        vision_feature_dim=16, vision_feature_id=identifier,
    ), ByteCodec.byte_fallback())
    config = replace(config, phase="P1", representation_mode="recurrent")
    window = next(iter_windows(ep, ByteCodec.byte_fallback()))
    batch, = prepare_batches(model, [EpisodeWindow(ep, window)], config)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    forward_batch(model, batch, ledger, config, torch.device("cpu"), 0)
    ledger.finish()
    assert model.core.media_stems.external_vision.weight.grad is not None
    assert model.media_realizers.visual_filter[0].weight.grad is not None
