"""Causal concordant compiler over prepared media records: no target leakage, one loss path."""

from dataclasses import replace

import pytest
import torch
from media_support import install_fake_realizers, prepared_record
from retrieval_fixture import MODEL_OVERRIDES, retrieval_aux, retrieval_train_config

from megatoken.controls import resolve_event_control
from megatoken.data.concordance import (
    AlignedBundle,
    BundleArtifact,
    TaskView,
    compile_task,
    edit_task_view,
    generated,
    group,
    observed,
    standard_views,
)
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.packing import WindowConfig, iter_windows
from megatoken.data.schema import Lineage
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.relations import RelationProjection, SelectedRelation
from megatoken.training.engine import prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow


def _bundle(tmp_path):
    store = TensorAssetStore(tmp_path)
    speech = prepared_record(store, "audio", seed=11, target=False)
    return AlignedBundle(
        "recording-001",
        Lineage("recording", "r1", "speaker-ancestry", frozen_split="train"),
        (
            BundleArtifact("speech", "source_speech", "audio", media=speech),
            BundleArtifact(
                "transcript",
                "verbatim_transcript",
                "text",
                payload=b"secret expected transcript",
                provenance="teacher",
            ),
        ),
        "explicit-verified-terms",
        "teacher-features-v1",
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


def _media(store, modality, seed):
    """A fully prepared event: observed teacher features plus a realizer target."""
    return prepared_record(store, modality, seed=seed, tokens=10)


def test_observed_media_never_exposes_its_realization_target(tmp_path):
    store = TensorAssetStore(tmp_path)
    full = _media(store, "image", 3)
    artifact = BundleArtifact("picture", "image", "image", media=full)
    shown, media_type = artifact.record_bytes(generated=False)
    assert media_type == MEDIA_RECORD_TYPE
    reader = MediaRecord.from_bytes(shown)
    assert reader == full.observed() and reader.target is None
    assert reader.output_geometry is None
    assert full.target.sha256.encode() not in shown
    assert b'output_geometry":null' in shown
    # A generated event carries the target and its geometry for E/R supervision.
    realized, _ = artifact.record_bytes(generated=True)
    assert MediaRecord.from_bytes(realized) == full
    # An observation-only record cannot be selected as a generated output.
    bare = BundleArtifact(
        "bare", "image", "image", media=prepared_record(store, "image", seed=4, target=False)
    )
    assert bare.record_bytes(generated=False)[0]
    with pytest.raises(ValueError, match="lacks native target"):
        bare.record_bytes(generated=True)
    with pytest.raises(ValueError, match="matching MediaRecord"):
        BundleArtifact("wrong", "audio", "audio", media=full)


def test_compiled_media_records_keep_target_only_on_generated_meridians(tmp_path):
    store = TensorAssetStore(tmp_path)
    bundle = AlignedBundle(
        "av-001",
        Lineage("clip", "r1", "scene-family", frozen_split="train"),
        (
            BundleArtifact("request", "creation_request", "text", payload=b"a red ball"),
            BundleArtifact("video", "video", "video", media=_media(store, "video", 5)),
            BundleArtifact("sound", "soundtrack", "audio", media=_media(store, "audio", 6)),
            BundleArtifact("caption", "display_caption", "text", payload=b"a red ball"),
        ),
        "licensed",
        "teacher-features-v1",
        "verified",
    )
    (view,) = (v for v in standard_views(bundle) if v.family == "av_creation")
    episode = compile_task(bundle, view)
    media = [
        (r, MediaRecord.from_bytes(r.payload))
        for r in episode.records
        if r.media_type == MEDIA_RECORD_TYPE
    ]
    assert {m.modality for _, m in media} == {"video", "audio"}
    assert all(r.policy_mask and m.target is not None for r, m in media)
    # Meridian events bind each record to the digest of its observed teacher features.
    bound = {e.asset_sha256 for g in episode.meridians for e in g.events if e.asset_sha256}
    assert bound == {m.source.sha256 for _, m in media}
    annotations = episode.supervision.annotations_json
    assert "synchronized_with" in annotations and "describes" in annotations


def test_image_creation_target_enters_existing_grouped_loss_ledger(tmp_path):
    torch.set_num_threads(1)
    store = TensorAssetStore(tmp_path)
    bundle = AlignedBundle(
        "picture",
        Lineage("photo", "r1", "camera-family", frozen_split="train"),
        (
            BundleArtifact("request", "creation_request", "text", payload=b"orange sphere"),
            BundleArtifact("picture", "image", "image", media=_media(store, "image", 7)),
            BundleArtifact("caption", "display_caption", "text", payload=b"orange sphere"),
        ),
        "licensed",
        "teacher-features-v1",
        "verified",
    )
    compiled = compile_task(
        bundle, next(view for view in standard_views(bundle) if view.family == "image_creation")
    )
    model = install_fake_realizers(
        build_model(ModelConfig.tiny(**MODEL_OVERRIDES), ByteCodec.byte_fallback())
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
        max_record_bytes=4096,
    )
    window = next(
        iter_windows(
            compiled, ByteCodec.byte_fallback(), WindowConfig(max_records=6, max_decisions=3)
        )
    )
    (batch,) = prepare_batches(model, [EpisodeWindow(compiled, window)], config)
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
    assert model.media_realizers.packs["sana"].condition.project.weight.grad is not None


def test_edit_view_requires_the_prior_base_and_compiles_a_verified_chain(tmp_path):
    store = TensorAssetStore(tmp_path)
    original = BundleArtifact("base", "original_image", "image", media=_media(store, "image", 8))
    edited = BundleArtifact(
        "edit",
        "edited_image",
        "image",
        media=_media(store, "image", 9),
        parent_id="base",
        version="v1",
    )
    bundle = AlignedBundle(
        "edit-001",
        Lineage("photo", "r1", "edit-family", frozen_split="train"),
        (
            original,
            edited,
            BundleArtifact("ask", "image_edit_instruction", "text", payload=b"make it blue"),
        ),
        "licensed",
        "teacher-features-v1",
        "verified",
    )
    (view,) = standard_views(bundle)
    assert view.family == "image_edit"
    episode = compile_task(bundle, view)
    assert [d.kind for d in episode.dependencies] == ["edit"]
    media = [
        MediaRecord.from_bytes(r.payload)
        for r in episode.records
        if r.media_type == MEDIA_RECORD_TYPE
    ]
    # The base is only observed; the edited version is the supervised output.
    assert [(m.source, m.target is not None) for m in media] == [
        (original.media.source, False),
        (edited.media.source, True),
    ]
    explicit = edit_task_view(
        family="image_edit", view_id="e", original="base", instruction="ask", edited="edit"
    )
    twin = compile_task(bundle, explicit)
    assert [r.payload for r in twin.records] == [r.payload for r in episode.records]
    # The edited target cannot be compiled without its exact observed base.
    only_edit = TaskView(
        "no-base",
        "image_edit",
        (
            group(observed("ask", "instruction", "Read the edit request")),
            group(generated("edit", "edited", "Edit it", revision=1)),
        ),
    )
    with pytest.raises(ValueError, match="exact base"):
        compile_task(bundle, only_edit)
    # An edit whose parent has another modality is not a version chain.
    with pytest.raises(ValueError, match="matching causal parent"):
        replace(
            bundle,
            artifacts=(*bundle.artifacts, replace(edited, artifact_id="bad", parent_id="ask")),
        )
