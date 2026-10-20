"""Aligned bundle compiler writes canonical native Episode stream without any data prep."""

import base64
import json

import pytest

from megatoken.data.schema import Episode
from scripts.compile_aligned_bundles import compile_manifest


def _bundle(source_family="family", split="train"):
    def artifact(name, text):
        return {
            "artifact_id": name,
            "modality": "text",
            "payload_base64": base64.b64encode(text.encode()).decode(),
        }

    return {
        "bundle_id": "recording",
        "source_family": source_family,
        "split": split,
        "artifacts": [
            artifact("recording", "Source audio transcript"),
            artifact("plan", "A chosen scene and exact speech content"),
            artifact("caption", "Scene description"),
            artifact("spoken", "Spoken utterance"),
        ],
        "views": [
            {
                "family": "verbalization",
                "inputs": ["recording"],
                "content_plan": "plan",
                "outputs": ["caption", "spoken"],
                "requests": {
                    "recording": "Read literal source words",
                    "plan": "Commit coherent content selection",
                    "caption": "Describe committed content",
                    "spoken": "Speak committed content",
                },
            }
        ],
    }


def test_compiled_output_is_direct_native_input(tmp_path):
    source = tmp_path / "bundles.jsonl"
    source.write_text(json.dumps(_bundle()) + "\n")
    output = tmp_path / "episodes.jsonl"
    receipt = compile_manifest(str(source), str(output))
    assert receipt["bundles"] == 1
    assert receipt["episodes"] == 1
    decoded = json.loads(output.read_text())
    episode = Episode.from_dict(decoded)
    assert [len(m.events) for m in episode.meridians] == [1, 1, 2]
    assert episode.lineage.frozen_split == "train"
    assert [x.policy_mask for x in episode.records] == [False, True, True, True]
    with pytest.raises(ValueError, match="overwrite"):
        compile_manifest(str(source), str(output))


def test_compiler_rejects_same_family_cross_split(tmp_path):
    source = tmp_path / "bundles.jsonl"
    first = _bundle("speaker-A", "train")
    second = _bundle("speaker-A", "test")
    second["bundle_id"] = "recording-2"
    source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="frozen split"):
        compile_manifest(str(source), str(tmp_path / "out.jsonl"))
