"""Sequential edit chains reuse bounded modality/semantic tracks across versions."""
import numpy as np

from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.omni.bundles import (
    AlignedBundle, Artifact, Edit, compile_edit_trajectory,
)


def test_recorded_four_step_edits_do_not_allocate_new_tracks_per_version(tmp_path):
    store = TensorAssetStore(tmp_path)
    artifacts = []
    for index in range(5):
        image = store.put(
            np.ones((3, 8, 8), dtype=np.float32) * index,
            modality="image", artifact_version=f"v{index}",
            source_family="scene-family",
        )
        artifacts.append(Artifact(
            f"image-v{index}", "image",
            MediaRecord("image", image, image).to_bytes(),
            version=f"v{index}", track_id="picture",
            media_type=MEDIA_RECORD_TYPE,
        ))
    for index in range(1, 5):
        artifacts.append(Artifact(
            f"caption-v{index}", "text", f"new caption {index}".encode(),
            version=f"v{index}", track_id="caption",
        ))
    edits = tuple(
        Edit(f"v{index}", f"v{index+1}", "change red to blue")
        for index in range(4)
    )
    bundle = AlignedBundle(
        "scene-bundle", "scene-family", "train", tuple(artifacts), edits=edits,
    )
    trajectory = compile_edit_trajectory(
        bundle,
        base_artifact_id="image-v0",
        steps=tuple(
            (edits[index], f"image-v{index+1}", (f"caption-v{index+1}",))
            for index in range(4)
        ),
    )
    episode = trajectory.episode
    assert len(episode.meridians) == 1 + 2 * 4
    assert {track.track_id for track in episode.tracks} == {
        "picture", "edit_request", "caption",
    }
    assert len(episode.tracks) == 3
    for group in episode.meridians[2::2]:
        assert {event.track_id for event in group.events} == {"picture", "caption"}
