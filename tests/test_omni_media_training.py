"""Prepared media (teacher features -> realizer latents) through the grouped loss ledger."""

from dataclasses import replace

import pytest
import torch
from media_support import FakeRealizer, install_fake_realizers, prepared_record
from retrieval_fixture import MODEL_OVERRIDES, retrieval_aux, retrieval_train_config

from megatoken.data import Episode, Lineage, PhysicalRecord, WindowConfig, iter_windows
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec, encode_record
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.teacher import PINNED_TEACHER
from megatoken.training.engine import batch_counters, prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow

CODEC = ByteCodec.byte_fallback()
OTHER_TEACHER = PINNED_TEACHER.replace("914f7f89", "0" * 8)


def _generated(name, modality, record, turn=1):
    return (
        PhysicalRecord(
            name,
            "assistant",
            record.to_bytes(),
            turn_index=turn,
            policy_mask=True,
            media_type=MEDIA_RECORD_TYPE,
        ),
        TrackSpec(name, "assistant"),
        TrackEvent(name, name, modality, turn, "assistant", asset_sha256=record.source.sha256),
    )


def _case(tmp_path):
    """Observed audio + prompt in meridian 0; a caption and a generated image in meridian 1."""
    store = TensorAssetStore(tmp_path)
    sound = prepared_record(store, "audio", seed=1, target=False, tokens=10)
    picture = prepared_record(store, "image", seed=2, tokens=14)
    record, spec, event = _generated("picture", "image", picture)
    episode = Episode(
        "omni",
        (
            PhysicalRecord("sound", "user", sound.to_bytes(), media_type=MEDIA_RECORD_TYPE),
            PhysicalRecord("prompt", "user", b"Describe the sound and draw a scene"),
            PhysicalRecord("caption", "assistant", b"bright", turn_index=1, policy_mask=True),
            record,
        ),
        Lineage("local", "revision", "family"),
        tracks=(
            TrackSpec("mic", "user"),
            TrackSpec("prompt", "user"),
            TrackSpec("caption", "assistant"),
            spec,
        ),
        meridians=(
            Meridian(
                0,
                (
                    TrackEvent(
                        "sound", "mic", "audio", 0, "user", asset_sha256=sound.source.sha256
                    ),
                    TrackEvent("prompt", "prompt", "text", 0, "user"),
                ),
            ),
            Meridian(1, (TrackEvent("caption", "caption", "text", 1, "assistant"), event)),
        ),
    )
    # The track name of a generated media record is its own name ("picture").
    episode = replace(
        episode,
        tracks=(*episode.tracks[:3], TrackSpec("picture", "assistant")),
    )
    model = install_fake_realizers(build_model(ModelConfig.tiny(**MODEL_OVERRIDES), CODEC))
    config = retrieval_train_config(
        tmp_path / "retrieval",
        phase="B0",
        media_root=str(tmp_path),
        budget_meter="G",
        budget=4,
        generated_capacity=16,
        max_history_tokens=128,
        max_record_bytes=4096,
    )
    return store, episode, model, config


def _forward(model, episode, config, *, window_config=None):
    window = next(iter_windows(episode, CODEC, window_config or WindowConfig()))
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, window)], config)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    model.core.set_model_version(0)
    forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        retrieval_aux(config, supervised=False),
    )
    ledger.finish()
    return batch, ledger


def _condition_grad(model, name):
    return model.media_realizers.packs[name].condition.project.weight.grad


def test_media_reference_has_zero_text_tokens_and_corruption_fails(tmp_path):
    store, episode, _, _ = _case(tmp_path)
    for record in (episode.records[0], episode.records[-1]):
        encoded = encode_record(record, CODEC)
        assert encoded.tokens == encoded.P == encoded.positions == 0
    reference = MediaRecord.from_bytes(episode.records[-1].payload)
    expected = store.load(reference.source)
    assert expected.shape == reference.source.shape == (14, 512)
    assert store.load(reference.target).shape == (4, 2, 2)
    path = tmp_path / reference.source.relative_path
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA256"):
        store.load(reference.source)
    # Corrupting the target latents fails closed the same way.
    target = tmp_path / reference.target.relative_path
    target.write_bytes(target.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="SHA256"):
        store.load(reference.target)


@pytest.mark.parametrize("mode", ("bootstrap", "recurrent"))
def test_mixed_media_meridian_gradients_and_loss_counts(tmp_path, mode):
    torch.set_num_threads(1)
    torch.manual_seed(901)
    _, episode, model, config = _case(tmp_path)
    config = replace(config, representation_mode=mode, phase="B0" if mode == "bootstrap" else "P1")
    window = next(iter_windows(episode, CODEC, WindowConfig(max_records=8, max_decisions=4)))
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, window)], config)
    assert batch.counts["visual_prediction"] == 1
    assert batch.counts["visual_reconstruction"] == 1
    assert batch.counts["audio_prediction"] == 0
    assert batch.counts["lexical"] == len(b"bright")
    assert batch.counts["facet_prediction"] == (
        2 * model.config.n_facets * model.config.facet_dim if mode == "bootstrap" else 0
    )
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    model.core.set_model_version(0)
    forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        retrieval_aux(config, supervised=False),
    )
    ledger.finish()
    # The image realizer's own objective trains its facet condition; the audio
    # realizer is untouched because the audio event is only observed.
    assert _condition_grad(model, "sana") is not None
    assert _condition_grad(model, "sana").abs().sum() > 0
    assert _condition_grad(model, "pocket") is None
    assert model.core.frontend.facet_queries.grad is not None
    if mode == "bootstrap":
        assert model.core.media_encoder.projection.weight.grad is not None
    counters = batch_counters([batch])
    assert counters.G == len(episode.records)
    assert counters.media_events == 2
    assert counters.image_events == 1
    # An observed audio event realizes nothing, so it declares no output samples.
    assert counters.audio_samples == 0
    assert counters.video_frames == 0
    assert counters.P == 2 * len(b"bright")


def test_wrong_teacher_identity_is_rejected(tmp_path):
    _, episode, _, config = _case(tmp_path)
    model = install_fake_realizers(
        build_model(ModelConfig.tiny(**MODEL_OVERRIDES, media_extractor=OTHER_TEACHER), CODEC)
    )
    window = next(iter_windows(episode, CODEC))
    with pytest.raises(ValueError, match="different teacher"):
        prepare_batches(model, [EpisodeWindow(episode, window)], config)
    # A non-pinned identity cannot even be written as an asset or a model.
    with pytest.raises(ValueError, match="pinned EmbeddingGemma2"):
        prepared_record(TensorAssetStore(tmp_path), "image", extractor="other-teacher-v1")
    with pytest.raises(ValueError, match="pinned EmbeddingGemma2"):
        ModelConfig.tiny(**MODEL_OVERRIDES, media_extractor="other-teacher-v1")


def test_target_from_another_realizer_pack_is_rejected(tmp_path):
    torch.set_num_threads(1)
    _, episode, model, config = _case(tmp_path)
    # A realizer with another checkpoint digest cannot consume these latents.
    other = FakeRealizer("image", model.config.n_facets, model.config.facet_dim)
    other.checkpoint_identity = "f" * 64
    model.media_realizers.packs["sana"] = other
    with pytest.raises(ValueError, match="different realizer pack"):
        _forward(model, episode, config)


def test_joint_text_audio_and_visual_targets_have_distinct_objectives(tmp_path):
    torch.set_num_threads(1)
    store, episode, model, config = _case(tmp_path)
    voice = prepared_record(store, "audio", seed=3, tokens=9)
    record, spec, event = _generated("voice", "audio", voice)
    episode = replace(
        episode,
        records=(*episode.records, record),
        tracks=(*episode.tracks, spec),
        meridians=(
            episode.meridians[0],
            Meridian(1, (*episode.meridians[1].events, event)),
        ),
    )
    batch, ledger = _forward(model, episode, config)
    assert batch.counts["audio_prediction"] == 1
    assert batch.counts["audio_reconstruction"] == 1
    assert batch.counts["visual_prediction"] == 1
    assert batch.counts["lexical"] == len(b"bright")
    assert _condition_grad(model, "pocket").abs().sum() > 0
    assert _condition_grad(model, "sana").abs().sum() > 0
    # Different realizers, different latents: the two gradients are not one signal.
    assert not torch.equal(_condition_grad(model, "pocket"), _condition_grad(model, "sana"))
    counters = batch_counters([batch])
    assert counters.audio_samples == 48
    assert counters.image_events == 1
    assert counters.media_events == 3


def test_codec_padding_is_not_supervised(tmp_path):
    torch.set_num_threads(1)
    store, episode, model, config = _case(tmp_path)
    voice = prepared_record(store, "audio", seed=3, tokens=9, valid_frames=2)
    record, spec, event = _generated("voice", "audio", voice)
    episode = replace(
        episode,
        records=(*episode.records[:2], record),
        tracks=(*episode.tracks[:2], spec),
        meridians=(
            episode.meridians[0],
            Meridian(1, (event,)),
        ),
    )
    window = next(iter_windows(episode, CODEC))
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, window)], config)
    (step,) = [s for s in batch.steps if s.generated]
    (plan,) = [p for p in step.plans if hasattr(p, "resolve")]
    target = plan.resolve(store, torch.device("cpu"))
    assert target.target_mask.tolist() == [[True, True, False, False, False, False]]
    assert target.realizer == voice.target.extractor


@pytest.mark.parametrize("mode", ("bootstrap", "recurrent"))
def test_video_training_is_one_32_facet_megatoken(tmp_path, mode):
    torch.set_num_threads(1)
    torch.manual_seed(922)
    store, episode, model, config = _case(tmp_path)
    # A video clip is many teacher tokens and many latent frames, but exactly one event.
    clip = prepared_record(store, "video", seed=4, tokens=40)
    record, spec, event = _generated("picture", "video", clip)
    episode = replace(
        episode,
        records=(*episode.records[:-1], record),
        meridians=(
            episode.meridians[0],
            Meridian(1, (episode.meridians[1].events[0], event)),
        ),
    )
    config = replace(config, representation_mode=mode, phase="B0" if mode == "bootstrap" else "P1")
    batch, _ = _forward(model, episode, config)
    assert batch.counts["visual_prediction"] == 1
    counters = batch_counters([batch])
    assert counters.video_frames == 4
    assert counters.image_events == 0
    assert counters.media_events == 2
    assert _condition_grad(model, "sana_video").abs().sum() > 0
    # 40 teacher tokens are reduced to the one canonical facet block.
    raw = model.core.encode_media(
        torch.from_numpy(store.load(clip.source))[None], extractor=PINNED_TEACHER
    )
    assert raw.shape == (1, model.config.n_facets, model.config.facet_dim)


def test_real_event_budget_checkpoint_and_media_data(tmp_path, monkeypatch):
    """The engine path; realizers come from a locked pack manifest, replaced here by fakes."""
    from retrieval_fixture import write_retrieval_artifacts

    from megatoken.data import DedupConfig, DedupStore
    from megatoken.data.storage import export_prepared
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    _, ep, _, config = _case(tmp_path)
    # Mandatory retrieval supervision needs a document-role source event in
    # meridian 0 and a query at the generated caption.
    ep = replace(
        ep,
        records=(PhysicalRecord("doc", "document", b"the sound is a bright tone"), *ep.records),
        tracks=(TrackSpec("doc", "document"), *ep.tracks),
        meridians=(
            Meridian(0, (TrackEvent("doc", "doc", "text", 0, "document"), *ep.meridians[0].events)),
            ep.meridians[1],
        ),
    )
    data = tmp_path / "omni-data"
    source = DedupStore(
        tmp_path / "index.sqlite", DedupConfig(train_fraction=1.0, validation_fraction=0.0)
    )
    try:
        with source.transaction():
            source.add(ep, CODEC)
        source.freeze()
        export_prepared(
            source,
            data,
            codec=CODEC,
            source_receipts=[],
            pipeline_fingerprint="omni-event-pilot-v1",
        )
    finally:
        source.close()
    artifacts = write_retrieval_artifacts(
        tmp_path / "omni-retrieval", [ep], doc="doc", request="prompt", answer="caption"
    )
    packs = tmp_path / "packs.json"
    packs.write_text("{}")
    monkeypatch.setattr(
        "megatoken.omni.loader.install_realizers",
        lambda model, path, *, device: install_fake_realizers(model),
    )
    config = replace(
        config,
        dataset=str(data),
        output=str(tmp_path / "run"),
        packs=str(packs),
        **artifacts,
        max_steps=1,
        accumulate_events=5,
        accumulate_scored=100_000,
        accumulate_ingested=100_000,
        max_accumulation_windows=1,
        budget=5,
        microbatch_episodes=1,
        checkpoint_every=0,
        use_stage_mixture=False,
    )
    result = train(config)
    payload, _ = load_checkpoint(result["checkpoint"])
    assert result["global_step"] == 1
    assert result["phase_consumed"] == result["counters"]["G"]
    assert result["counters"]["G"] == 5
    assert result["counters"]["media_events"] == 2
    assert result["counters"]["image_events"] == 1
    assert payload["phase_consumed"] == 5
    # Frozen realizer donors never enter the checkpoint; trainable conditions do.
    names = set(payload["model"])
    assert any(name.startswith("media_realizers.packs.sana.condition.") for name in names)
    assert not any(".donor." in name for name in names)
