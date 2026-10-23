import copy
import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from megatoken.contracts import MAIN_TRACK
from megatoken.data.legacy_linear import convert_linear_episode, convert_prepared
from megatoken.data.schema import (
    Episode,
    Lineage,
    PhysicalRecord,
    canonical_json,
    sequential_meridians,
)
from megatoken.generation.codec import ByteCodec
from megatoken.io import content_hash, sha256_file


def legacy():
    episode = sequential_meridians(
        Episode(
            "old",
            (
                PhysicalRecord("q", "user", "한국어?".encode()),
                PhysicalRecord("a", "assistant", "네".encode(), policy_mask=True),
            ),
            Lineage("original", "pinned", "family", ancestor_ids=("english-family",)),
        )
    )
    value = episode.to_dict()
    value["schema_version"] = 3
    value["tracks"] = [{"track_id": role, "role": role} for role in ("user", "assistant")]
    for record, group in zip(value["records"], value["meridians"], strict=True):
        group["events"][0]["track_id"] = record["role"]
    return episode, value


def test_conversion_preserves_bytes_masks_and_family_without_mutating_input():
    original, value = legacy()
    before = copy.deepcopy(value)
    converted = convert_linear_episode(value)
    assert converted.records == original.records
    assert converted.lineage == original.lineage
    assert converted.content_sha256 == original.content_sha256
    assert {event.track_id for group in converted.meridians for event in group.events} == {
        MAIN_TRACK
    }
    assert value == before


@pytest.mark.parametrize("kind", ["parallel", "tape", "instruction", "revision", "custom_track"])
def test_unsupported_causal_meaning_is_never_silently_removed(kind):
    _, value = legacy()
    if kind == "parallel":
        value["meridians"][0]["events"] = (
            *value["meridians"][0]["events"],
            value["meridians"][1]["events"][0],
        )
    elif kind == "tape":
        value["supervision"]["annotations_json"] = '{"native_action_tapes":[{"version":"old"}]}'
    elif kind == "instruction":
        value["meridians"][0]["events"][0]["instruction"] = "Keep this meaning"
    elif kind == "revision":
        value["track_revisions"] = [{"revision": 1}]
    else:
        value["tracks"][0]["description"] = "Independent worker"
    with pytest.raises(ValueError):
        convert_linear_episode(value)


def prepared_legacy(root):
    from megatoken.data import DedupConfig, DedupStore
    from megatoken.data.storage import export_prepared

    episode, raw = legacy()
    episode = replace(episode, lineage=replace(episode.lineage, frozen_split="validation"))
    codec = ByteCodec.byte_fallback()
    with DedupStore(root.parent / "spool.sqlite", DedupConfig()) as store:
        store.add(episode, codec)
        store.freeze()
        manifest = export_prepared(
            store, root, codec=codec, source_receipts=[], pipeline_fingerprint="old"
        )
    raw["lineage"]["frozen_split"] = "validation"
    line = (canonical_json(raw) + "\n").encode()
    shard = root / manifest["shards"][0]["path"]
    shard.write_bytes(line)
    with sqlite3.connect(root / manifest["index"]["path"]) as db:
        db.execute(
            "UPDATE episodes SET offset=0,length=?,sha256=?",
            (len(line), hashlib.sha256(line).hexdigest()),
        )
    manifest["schema_version"] = 3
    manifest["shards"][0]["sha256"] = sha256_file(shard)
    manifest["index"]["sha256"] = sha256_file(root / manifest["index"]["path"])
    manifest["fingerprint"] = content_hash(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return codec, episode


def test_prepared_conversion_preserves_heldout_split_and_leaves_original_intact(tmp_path):
    from megatoken.data.storage import PreparedDataset

    source = tmp_path / "original"
    codec, episode = prepared_legacy(source)
    original_hashes = {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "converted"
    receipt = convert_prepared(source, output, codec)
    assert receipt["splits"] == {"validation": 1}
    with PreparedDataset(output, split="validation") as dataset:
        (actual,) = dataset.iter_episodes()
        assert actual.records == episode.records
        assert set(episode.lineage.family_keys) <= set(actual.lineage.family_keys)
        assert actual.lineage.frozen_split == "validation"
    with PreparedDataset(output, split="train") as dataset:
        assert len(dataset) == 0
    assert original_hashes == {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()}


def test_corrupt_original_prepared_index_is_rejected_before_publication(tmp_path):
    source = tmp_path / "original"
    codec, _ = prepared_legacy(source)
    with (source / "index.sqlite").open("ab") as handle:
        handle.write(b"corrupt")
    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="checksum"):
        convert_prepared(source, output, codec)
    assert not (output / "manifest.json").exists()
