"""A creative content plan is a genuinely selected earlier event, not a prompt."""

from media_support import prepared_record

from megatoken.data.concordance import (
    AlignedBundle,
    BundleArtifact,
    compile_task,
    standard_views,
)
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import Lineage


def test_creative_plan_precedes_atomic_peer_generation(tmp_path):
    assets = TensorAssetStore(tmp_path)
    artifacts = (
        BundleArtifact(
            "request", "creation_request", "text", payload=b"Create a dog chasing a ball"
        ),
        BundleArtifact(
            "plan", "content_plan", "text", payload=b"Dog crosses the yard toward a red ball"
        ),
        BundleArtifact(
            "caption",
            "display_caption",
            "text",
            payload=b"Dog follows a red ball",
            provenance="teacher",
        ),
        BundleArtifact("video", "video", "video", media=prepared_record(assets, "video", seed=1)),
        BundleArtifact(
            "audio", "soundtrack", "audio", media=prepared_record(assets, "audio", seed=2)
        ),
    )
    bundle = AlignedBundle(
        "bundle",
        Lineage("origin", "r1", "family", frozen_split="train"),
        artifacts,
        "CC-BY-4.0",
        "plan-v1",
        "publisher-annotated",
    )
    (view,) = [v for v in standard_views(bundle) if v.family == "av_creation"]
    episode = compile_task(bundle, view)
    assert [len(g.events) for g in episode.meridians] == [1, 1, 3]
    assert [r.policy_mask for r in episode.records] == [False, True, True, True, True]
    # The plan is generated (supervised) before, and conditions, the three peers.
    assert episode.meridians[1].events[0].track_id == "content_plan"
    generated = [
        MediaRecord.from_bytes(r.payload)
        for r in episode.records
        if r.media_type == MEDIA_RECORD_TYPE
    ]
    assert [m.modality for m in generated] == ["video", "audio"]
    assert all(m.target is not None and m.output_geometry is not None for m in generated)
    assert episode.stream == "av_creation"
    assert episode.lineage.frozen_split == "train"
