"""Counterfactual omnimodal pilot must need each independent evidence track."""

import pytest

from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from scripts.prepare_omni_pilot import make_episode


@pytest.mark.parametrize("index", [0, 1])
def test_counterfactual_pairs_change_only_the_targeted_evidence(tmp_path, index):
    store = TensorAssetStore(tmp_path)
    baseline = make_episode(store, seed=91, index=index)
    audio_swap = make_episode(store, seed=91, index=index, variant="audio")
    visual_swap = make_episode(store, seed=91, index=index, variant="visual")

    # Keep original instructions and removal amounts identical across all three.
    for record_idx in (0, 1):
        assert baseline.records[record_idx].payload == audio_swap.records[record_idx].payload
        assert baseline.records[record_idx].payload == visual_swap.records[record_idx].payload

    def source(episode, record_idx):
        return MediaRecord.from_bytes(episode.records[record_idx].payload).source

    # Swapping sound leaves camera unchanged; swapping picture leaves sound unchanged.
    assert source(baseline, 2).sha256 == source(audio_swap, 2).sha256
    assert source(baseline, 3).sha256 != source(audio_swap, 3).sha256
    assert source(baseline, 3).sha256 == source(visual_swap, 3).sha256
    assert source(baseline, 2).sha256 != source(visual_swap, 2).sha256

    answers = {episode.records[4].payload for episode in (baseline, audio_swap, visual_swap)}
    assert len(answers) == 3
    assert {episode.lineage.family_id for episode in (baseline, audio_swap, visual_swap)} == {
        baseline.lineage.family_id
    }
    assert len({episode.episode_id for episode in (baseline, audio_swap, visual_swap)}) == 3
    assert all(
        episode.meridians[0].events[2].modality == ("video" if index % 2 else "image")
        for episode in (baseline, audio_swap, visual_swap)
    )


def test_unknown_counterfactual_variant_rejected(tmp_path):
    with pytest.raises(ValueError, match="counterfactual"):
        make_episode(TensorAssetStore(tmp_path), seed=91, index=0, variant="speech")
