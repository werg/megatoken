"""Causal concordant compiler, true codec target geometry, and replay contracts."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from retrieval_fixture import MODEL_OVERRIDES, retrieval_aux, retrieval_train_config
from torch import nn

from megatoken.controls import resolve_event_control
from megatoken.data.concordance import (
    AlignedBundle,
    BundleArtifact,
    TaskView,
    compile_task,
    generated,
    group,
    observed,
    standard_views,
)
from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.data.packing import WindowConfig, iter_windows
from megatoken.data.schema import Lineage
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.native_latents import (
    NativeLatentFlow,
    PackSpec,
    masked_flow_mse,
)
from megatoken.model.relations import RelationProjection, SelectedRelation
from megatoken.training.concordance_recipe import load_concordance_recipe
from megatoken.training.engine import prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow


def _bundle(tmp_path):
    store = TensorAssetStore(tmp_path)
    wave = store.put(np.zeros(512, np.float32), modality="audio", sample_rate=16000)
    return AlignedBundle(
        "recording-001",
        Lineage("recording", "r1", "speaker-ancestry", frozen_split="train"),
        (
            BundleArtifact("speech", "source_speech", "audio", media=MediaRecord("audio", wave)),
            BundleArtifact(
                "transcript",
                "verbatim_transcript",
                "text",
                payload=b"secret expected transcript",
                provenance="teacher",
            ),
        ),
        "explicit-verified-terms",
        "audio-16k-v1",
        "human-reviewed",
    )


def test_standard_asr_compiles_with_causal_distinct_request(tmp_path):
    bundle = _bundle(tmp_path)
    (view,) = standard_views(bundle)
    assert view.family == "transcription"
    episode = compile_task(bundle, view)
    assert len(episode.meridians) == 2
    first, second = episode.meridians
    assert not episode.records[0].policy_mask
    assert episode.records[1].policy_mask
    p0 = resolve_event_control(episode, first.events[0])
    p1 = resolve_event_control(episode, second.events[0])
    assert p0 != p1
    assert b"secret expected transcript" not in p1
    assert b"Transcribe exactly" in p1
    assert b"source_speech" in p1
    assert all(
        "secret expected transcript" not in x for x in (p0.decode("utf-8"), p1.decode("utf-8"))
    )
    assert episode.supervision.annotations_json != "{}"
    assert episode.lineage.family_id == "speaker-ancestry"
    assert episode.lineage.frozen_split == "train"


def test_same_step_output_peer_cannot_be_named_as_a_current_reference(tmp_path):
    bundle = _bundle(tmp_path)
    view = TaskView(
        "leaky",
        "transcription",
        (
            group(observed("speech", "microphone", "Listen to source speech")),
            group(
                generated(
                    "transcript", "text", "Transcribe", reference_artifact_ids=("transcript",)
                )
            ),
        ),
    )
    with pytest.raises(ValueError, match="Future|same-meridian"):
        compile_task(bundle, view)


def test_edit_requires_precise_base_and_increasing_request_revision(tmp_path):
    bundle = _bundle(tmp_path)
    changed = replace(
        bundle.artifacts[1],
        artifact_id="revision2",
        label="second",
        parent_id="transcript",
        version="r2",
    )
    bundle = replace(bundle, artifacts=(*bundle.artifacts, changed))
    view = TaskView(
        "edit-1",
        "speech_edit",
        (
            group(observed("speech", "mic", "Original utterance")),
            group(generated("revision2", "transcript", "Correct transcript", revision=1)),
            group(generated("transcript", "transcript", "Initial version", revision=0)),
        ),
    )
    with pytest.raises(ValueError, match="exact base"):
        compile_task(bundle, view)


def test_media_asset_native_masks_and_clock_revisions(tmp_path):
    store = TensorAssetStore(tmp_path)
    source = store.put(np.zeros((3, 8, 8), np.float32), modality="image")
    target = store.put(
        np.ones((4, 4), np.float32),
        modality="image",
        kind="codec_latents",
        codec_id="sana-dcae-r1",
        profile="sana-image-600m",
        preprocessing_id="crop-v3",
        geometry=(3, 8, 8),
        clock_id="source-clock",
        start_ms=0,
        end_ms=40,
    )
    mask = store.put_mask(np.ones((4, 4), bool), modality="image")
    record = MediaRecord("image", source, target, target_mask=mask)
    assert MediaRecord.from_bytes(record.to_bytes()) == record
    assert store.load_mask(mask).shape == (4, 4)
    with pytest.raises(ValueError, match="Codec latents need"):
        replace(target, codec_id="")
    with pytest.raises(ValueError, match="time interval"):
        replace(target, end_ms=0)


class _FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def forward(self, noisy, timestep, conditioning):
        del timestep
        return noisy * self.scale + conditioning[..., :4].square().mean() * torch.ones_like(noisy)


class _IdentityDecoder(nn.Module):
    def forward(self, value):
        return value


def test_native_flow_consumes_every_facet_and_masks_codec_padding():
    torch.set_num_threads(1)
    pack = PackSpec(
        "sana-image-600m",
        "sana_image",
        "not-loaded",
        "a" * 64,
        "revision",
        "sana-dcae-r1",
        16,
        steps=2,
    )
    native = NativeLatentFlow(
        facet_count=4,
        facet_width=32,
        pack=pack,
        denoiser=_FakeDenoiser(),
        decoder=_IdentityDecoder(),
    )
    raw = torch.randn((1, 4, 32), requires_grad=True)
    target = torch.randn(1, 2, 4, 4)
    mask = torch.ones(1, 2, 4, 4, dtype=torch.bool)
    mask[..., -1] = False
    loss = native.flow_loss(raw, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None and raw.grad.norm() > 0
    assert native.conditioner.to_donor.weight.grad is not None
    assert native.denoiser.scale.grad is None  # frozen pretrained donor
    spec = MediaOutputSpec(
        "image",
        height=8,
        width=8,
        codec_id=pack.codec_id,
        pack_id=pack.pack_id,
        latent_shape=(2, 4, 4),
    )
    a = native.sample(raw.detach(), spec, seed=7)
    b = native.sample(raw.detach(), spec, seed=7)
    torch.testing.assert_close(a, b)
    assert a.shape == (1, 2, 4, 4)


def test_masked_flow_mse_uses_only_valid_coordinates():
    result = masked_flow_mse(
        torch.tensor([[1.0, 1000.0]]), torch.tensor([[0.0, 0.0]]), torch.tensor([[True, False]])
    )
    assert result.item() == 1
    with pytest.raises(ValueError, match="valid"):
        masked_flow_mse(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_relation_projection_differentiates_selected_not_teacher():
    projection = RelationProjection(32)
    raw = torch.randn(1, 2, 4, 32, requires_grad=True)
    teachers = torch.randn(1, 2, 4, 32, requires_grad=True)
    loss, count = projection(
        raw, teachers, ("text", "audio"), (SelectedRelation("same_spoken_content", 0, 1),)
    )
    assert count == 1
    loss.backward()
    assert raw.grad is not None and raw.grad.norm() > 0
    assert teachers.grad is None


def test_concordance_recipe_loads_all_phases_and_requires_existing_streams():
    for phase in ("A", "B", "C", "D"):
        recipe = load_concordance_recipe(
            "configs/train/concordance-recipe.v1.json",
            phase=phase,
        )
        assert abs(sum(recipe.mixture.values()) - 1) < 1e-8
        assert recipe.max_media_events_per_window >= 3
    phase_b = load_concordance_recipe("configs/train/concordance-recipe.v1.json")
    assert phase_b.mixture["speech_translation"] == 0.15


def test_native_codec_target_enters_existing_grouped_loss_ledger(tmp_path):
    torch.set_num_threads(1)
    store = TensorAssetStore(tmp_path)
    image = store.put(np.zeros((3, 8, 8), np.float32), modality="image")
    native = store.put(
        np.zeros((4, 4), np.float32),
        modality="image",
        kind="codec_latents",
        codec_id="sana-dcae-r1",
        preprocessing_id="transform-v1",
        profile="sana-image-600m",
        geometry=(3, 8, 8),
        clock_id="input",
        start_ms=0,
        end_ms=40,
    )
    bundle = AlignedBundle(
        "picture",
        Lineage("photo", "r1", "camera-family", frozen_split="train"),
        (
            BundleArtifact("request", "creation_request", "text", payload=b"orange sphere"),
            BundleArtifact("picture", "image", "image", media=MediaRecord("image", image, native)),
            BundleArtifact("caption", "display_caption", "text", payload=b"orange sphere"),
        ),
        "licensed",
        "crop-v1",
        "verified",
    )
    compiled = compile_task(
        bundle, next(view for view in standard_views(bundle) if view.family == "image_creation")
    )
    model = build_model(ModelConfig.tiny(**MODEL_OVERRIDES), ByteCodec.byte_fallback())
    pack = PackSpec(
        "sana-image-600m",
        "sana_image",
        "not-loaded",
        "a" * 64,
        "revision",
        "sana-dcae-r1",
        16,
        steps=2,
    )
    model.media_realizers.install_native(
        "image",
        NativeLatentFlow(
            facet_count=4,
            facet_width=32,
            pack=pack,
            denoiser=_FakeDenoiser(),
            decoder=_IdentityDecoder(),
        ),
    )
    config = retrieval_train_config(
        tmp_path / "retrieval",
        phase="B0",
        budget_meter="G",
        budget=10,
        media_root=str(tmp_path),
        max_history_tokens=128,
        generated_capacity=32,
        effort=1,
        max_record_bytes=1024,
    )
    window = next(
        iter_windows(
            compiled, ByteCodec.byte_fallback(), WindowConfig(max_records=6, max_decisions=3)
        )
    )
    (batch,) = prepare_batches(
        model,
        [EpisodeWindow(compiled, window)],
        config,
    )
    assert batch.counts["visual_prediction"] == 1
    assert batch.counts["relation_semantic"] == 1
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
    assert model.media_realizers.native["image"].conditioner.to_donor.weight.grad is not None
