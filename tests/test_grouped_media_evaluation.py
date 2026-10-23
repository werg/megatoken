"""Generated evaluation uses prepared evidence and public media bounds."""

from types import SimpleNamespace

import torch
from test_multitrack_media import AUDIO, _system

from megatoken.data.media_assets import MEDIA_RECORD_TYPE, TensorAssetStore
from megatoken.data.schema import Episode, Lineage, Meridian, PhysicalRecord, TrackEvent, TrackSpec
from megatoken.omni.quality import DecodedQualityArchive
from megatoken.training.evaluation import evaluate_served_records
from tests.media_support import prepared_record


def test_served_evaluation_generates_same_track_media_from_checked_observations(tmp_path):
    torch.set_num_threads(1)
    _, model, _ = _system()
    store = TensorAssetStore(tmp_path)
    image = prepared_record(store, "image", target=False)
    speech = prepared_record(store, "audio")
    records = (
        PhysicalRecord("scene", "user", image.to_bytes(), media_type=MEDIA_RECORD_TYPE),
        PhysicalRecord(
            "speech", "assistant", speech.to_bytes(), policy_mask=True, media_type=MEDIA_RECORD_TYPE
        ),
        PhysicalRecord("caption", "assistant", b"A caption", policy_mask=True, media_type="text/plain"),
    )
    episode = Episode(
        "evaluation",
        records,
        Lineage("test", "v1", "heldout-family"),
        tracks=(TrackSpec("input", "user", "image"), TrackSpec("reply", "assistant")),
        meridians=(
            Meridian(0, (TrackEvent("scene", "input", "image", 0, "user"),)),
            Meridian(
                1,
                (
                    TrackEvent("speech", "reply", "audio", 1, "assistant"),
                    TrackEvent("caption", "reply", "text", 1, "assistant"),
                ),
            ),
        ),
    )
    dataset = SimpleNamespace(assets=store, iter_episodes=lambda: iter((episode,)))
    report = evaluate_served_records(
        model,
        dataset,
        SimpleNamespace(effort=1, generated_capacity=4),
        torch.device("cpu"),
        episodes=1,
        output_specs={"audio": AUDIO},
        decoded_evaluator=DecodedQualityArchive(
            tmp_path / "decoded",
            evaluator_metadata={"reference": "test fixture; no speech recognizer"},
        ),
    )
    assert report["skipped_media_episodes"] == 0
    assert report["decoded_media_events"] == 1
    assert report["generated_text_chunks"] == 1
    assert report["generated_chunks"] == 2
    assert report["media_quality_established"] is False
    import numpy as np

    archived = report["decoded_quality"][0]["sampled_assets"]["speech"]
    actual = np.load(tmp_path / "decoded" / archived["path"], allow_pickle=False)
    assert actual.ndim == 1 and actual.size > 0 and np.isfinite(actual).all()
    assert report["observation_mode"] == "recorded_observation_replay"
    assert report["tool_actions_executed"] is False
