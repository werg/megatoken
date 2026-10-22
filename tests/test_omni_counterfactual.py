"""Counterfactual evidence siblings change exactly one evidence track and never leak targets."""

import pytest
from media_support import prepared_record

from megatoken.data.concordance import (
    AlignedBundle,
    BundleArtifact,
    TaskView,
    compile_task,
    standard_views,
)
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import Lineage

INSTRUCTION = b"Which stock remains after shipment?"


def _bundle(store, *, index, variant="base"):
    """A grounded-answer task needing sound, picture and text evidence together.

    ``audio`` swaps only the acoustic evidence; ``visual`` swaps only the picture.
    Siblings share one source family so they can never cross a frozen split.
    """
    seeds = {"audio": 100 + index, "video": 200 + index}
    if variant == "audio":
        seeds["audio"] += 1000
    elif variant == "visual":
        seeds["video"] += 1000
    elif variant != "base":
        raise ValueError("Unknown counterfactual variant")
    answer = f"red:{seeds['video']};blue:{seeds['audio']}".encode()
    return AlignedBundle(
        f"cf-{index}-{variant}",
        Lineage("procedural", "r1", f"parent-{index}", frozen_split="train"),
        (
            BundleArtifact(
                "sound",
                "audio_evidence",
                "audio",
                media=prepared_record(store, "audio", seed=seeds["audio"]),
            ),
            BundleArtifact(
                "picture",
                "video_evidence",
                "video",
                media=prepared_record(store, "video", seed=seeds["video"]),
            ),
            BundleArtifact("note", "text_evidence", "text", payload=INSTRUCTION),
            BundleArtifact("answer", "grounded_answer", "text", payload=answer),
        ),
        "synthetic",
        "teacher-features-v1",
        "procedural",
    )


def _view(bundle):
    (view,) = (v for v in standard_views(bundle) if v.family == "cross_track_evidence")
    return view


def _episodes(tmp_path, index):
    store = TensorAssetStore(tmp_path)
    bundles = {v: _bundle(store, index=index, variant=v) for v in ("base", "audio", "visual")}
    return bundles, {v: compile_task(b, _view(b)) for v, b in bundles.items()}


def _source(episode, modality):
    (record,) = (
        r
        for r in episode.records
        if r.media_type == MEDIA_RECORD_TYPE
        and MediaRecord.from_bytes(r.payload).modality == modality
    )
    return MediaRecord.from_bytes(record.payload)


@pytest.mark.parametrize("index", [0, 1])
def test_counterfactual_pairs_change_only_the_targeted_evidence(tmp_path, index):
    _, eps = _episodes(tmp_path, index)
    base, swapped_audio, swapped_visual = eps["base"], eps["audio"], eps["visual"]

    # The written evidence and its request are untouched by either swap.
    text = [
        [r.payload for r in ep.records if r.media_type != MEDIA_RECORD_TYPE][0]
        for ep in eps.values()
    ]
    assert text == [INSTRUCTION] * 3

    # Swapping sound leaves the camera unchanged; swapping the picture leaves sound unchanged.
    assert _source(base, "video") == _source(swapped_audio, "video")
    assert _source(base, "audio") != _source(swapped_audio, "audio")
    assert _source(base, "audio") == _source(swapped_visual, "audio")
    assert _source(base, "video") != _source(swapped_visual, "video")

    answers = {ep.records[-1].payload for ep in eps.values()}
    assert len(answers) == 3
    assert len({ep.lineage.family_id for ep in eps.values()}) == 1
    assert len({ep.episode_id for ep in eps.values()}) == 3
    # The task needs all three tracks, observed together before the answer is made.
    for ep in eps.values():
        first = ep.meridians[0]
        assert [e.modality for e in first.events] == ["audio", "video", "text"]
        assert ep.meridians[1].events[0].modality == "text"
        assert ep.stream == "cross_track_evidence"


@pytest.mark.parametrize("index", [0, 1])
def test_counterfactual_evidence_is_observed_media(tmp_path, index):
    bundles, eps = _episodes(tmp_path, index)
    for variant, ep in eps.items():
        for artifact in ("sound", "picture"):
            media = bundles[variant].by_id(artifact).media
            assert media.target is not None  # prepared for generation elsewhere
            # Evidence is observed: it keeps its own latents for encode-and-
            # reconstruct, and training reads only its features.
            assert _source(ep, media.modality) == media
            record = next(
                r
                for r in ep.records
                if r.media_type == MEDIA_RECORD_TYPE and media.source.sha256.encode() in r.payload
            )
            assert not record.policy_mask
        # Meridian events bind observed media to their pinned feature digest.
        digests = {e.asset_sha256 for e in ep.meridians[0].events if e.asset_sha256}
        assert digests == {
            bundles[variant].by_id(a).media.source.sha256 for a in ("sound", "picture")
        }


def test_sibling_swaps_use_distinct_teacher_features_of_one_modality(tmp_path):
    bundles, _ = _episodes(tmp_path, 0)
    sound = {v: b.by_id("sound").media.source for v, b in bundles.items()}
    assert sound["base"].sha256 == sound["visual"].sha256 != sound["audio"].sha256
    # Same teacher, shape and clock-free contract; only the content differs.
    assert {a.extractor for a in sound.values()} == {sound["base"].extractor}
    assert {a.shape for a in sound.values()} == {sound["base"].shape}


def test_unknown_counterfactual_variant_and_incomplete_evidence_rejected(tmp_path):
    store = TensorAssetStore(tmp_path)
    with pytest.raises(ValueError, match="counterfactual"):
        _bundle(store, index=0, variant="speech")
    complete = _bundle(store, index=0)
    # No view is invented when one evidence track is missing.
    partial = AlignedBundle(
        "cf-partial",
        complete.lineage,
        tuple(a for a in complete.artifacts if a.artifact_id != "sound"),
        "synthetic",
        "teacher-features-v1",
        "procedural",
    )
    assert standard_views(partial) == ()
    with pytest.raises(ValueError, match="known family"):
        TaskView("x", "speech", _view(complete).groups)
