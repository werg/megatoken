"""A creative content plan is a genuinely selected earlier event, not a prompt."""

import pytest

from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.omni.bundles import AlignedBundle, Artifact, TaskView, compile_views
from tests.media_support import prepared_record


def test_creative_plan_precedes_atomic_five_track_generation(tmp_path):
    assets = TensorAssetStore(tmp_path)

    def media(id_, modality, seed):
        record = prepared_record(assets, modality, seed=seed)
        return Artifact(id_, modality, record.to_bytes(), media_type=MEDIA_RECORD_TYPE)

    artifacts = (
        Artifact("request", "text", b"Create a dog chasing a ball"),
        Artifact(
            "plan",
            "text",
            b"Dog crosses the yard toward a red ball",
            annotation_type="content_plan",
        ),
        Artifact("caption", "text", b"Dog follows a red ball", annotation_type="display_caption"),
        media("video", "video", 1),
        media("audio", "audio", 2),
        media("opening", "image", 3),
        media("ending", "image", 4),
    )
    requests = {a.artifact_id: "Generate or interpret " + a.artifact_id for a in artifacts}
    task = TaskView(
        family="av_creation",
        inputs=("request",),
        content_plan="plan",
        outputs=("caption", "video", "audio", "opening", "ending"),
        requests=requests,
        event_revisions={"plan": 2, "video": 3},
    )
    compiled = compile_views(AlignedBundle("bundle", "family", "train", artifacts), (task,))[0]
    episode = compiled.episode
    assert [len(g.events) for g in episode.meridians] == [1, 1, 5]
    assert [r.policy_mask for r in episode.records] == [False, True, True, True, True, True, True]
    assert episode.meridians[1].events[0].request_revision == 2
    assert episode.meridians[2].events[1].request_revision == 3
    assert episode.control_seed == b""
    # Generated media keep their realizer latent target and output geometry.
    generated = [
        MediaRecord.from_bytes(r.payload)
        for r in episode.records
        if r.media_type == MEDIA_RECORD_TYPE
    ]
    assert [m.modality for m in generated] == ["video", "audio", "image", "image"]
    assert all(m.target is not None and m.output_geometry is not None for m in generated)
    assert compiled.source_family == "family"
    assert compiled.split == "train"


def test_content_plan_cannot_be_declared_as_external_input():
    with pytest.raises(ValueError, match="separately generated"):
        TaskView(
            family="image_creation",
            inputs=("plan",),
            outputs=("image",),
            content_plan="plan",
            requests={"plan": "Generate plan", "image": "Render image"},
        )


def test_observed_tool_artifact_is_not_relabelled_as_user():
    bundle = AlignedBundle(
        "bundle",
        "origin",
        "validation",
        (
            Artifact("actual_result", "text", b'{"weather":"rain"}', observed_role="tool"),
            Artifact("answer", "text", b"Rain expected"),
        ),
    )
    task = TaskView(
        "cross_track",
        ("actual_result",),
        ("answer",),
        {"actual_result": "Read tool result", "answer": "Answer using observed facts"},
    )
    episode = compile_views(bundle, (task,))[0].episode
    assert episode.records[0].role == "tool"
    assert episode.meridians[0].events[0].role == "tool"
