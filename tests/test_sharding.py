"""Shards of a source manifest read every row exactly once, balanced, and keep source identity."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from megatoken.data.sharding import plan_shards
from megatoken.data.sources import SourceFile, SourceSpec, TermsReview, iter_source_rows

TERMS = {"approved": True, "reviewed_by": "tester", "reviewed_at": "2026-01-01"}


def hf_source(source_id: str, **fields) -> dict:
    return {
        "source_id": source_id,
        "uri": f"hf://datasets/example/{source_id}",
        "format": "hf",
        "adapter": "chat",
        "stream": "chat",
        "revision": "a" * 40,
        "terms": TERMS,
        "split": "train",
        **fields,
    }


def manifest(*sources: dict) -> dict:
    return {"schema_version": 1, "sources": list(sources)}


def test_full_plan_gives_every_file_to_exactly_one_shard():
    chat = hf_source("chat")
    code = hf_source("code")
    listings = {
        "chat": [(f"hf://x/chat-{i}.parquet", 100_000 + 1000 * i) for i in range(30)],
        "code": [(f"hf://x/code-{i}.parquet", 300_000) for i in range(12)],
    }
    shards = plan_shards(manifest(chat, code), 6, listings, full=True)
    taken = Counter(
        item["path"] for shard in shards for source in shard["sources"] for item in source["files"]
    )
    assert taken == Counter(name for files in listings.values() for name, _ in files)
    assert all(source["max_rows"] is None for shard in shards for source in shard["sources"])
    # One spec per source in each shard, and shards close to the average load.
    for shard in shards:
        ids = [source["source_id"] for source in shard["sources"]]
        assert len(ids) == len(set(ids))
    loads = [
        sum(
            size
            for source in shard["sources"]
            for item in source["files"]
            for name, size in listings[source["source_id"]]
            if name == item["path"]
        )
        for shard in shards
    ]
    assert max(loads) <= 1.3 * sum(loads) / len(loads)


def test_a_source_too_big_for_one_shard_is_striped_across_shards():
    giant = hf_source("giant")
    small = hf_source("small")
    listings = {
        "giant": [("hf://x/giant.jsonl", 90_000_000)],
        "small": [(f"hf://x/small-{i}.parquet", 100_000) for i in range(8)],
    }
    shards = plan_shards(manifest(giant, small), 4, listings, full=True)
    stripes = sorted(
        tuple(source["stripe"])
        for shard in shards
        for source in shard["sources"]
        if source["source_id"] == "giant"
    )
    count = stripes[0][1]
    assert stripes == [(index, count) for index in range(count)]
    holders = [
        i for i, shard in enumerate(shards) for s in shard["sources"] if s["source_id"] == "giant"
    ]
    assert len(holders) == len(set(holders))  # no shard holds two stripes of one source


def test_capped_plan_matches_scaled_caps_and_keeps_sources_whole():
    capped = manifest(
        hf_source("a", max_rows=100, stream="chat"),
        hf_source("b", max_rows=50, stream="executed_trajectories"),
    )
    shards = plan_shards(capped, 2, scale=3)
    got = {s["source_id"]: s["max_rows"] for shard in shards for s in shard["sources"]}
    assert got == {"a": 300, "b": 150}
    assert all(len(shard["sources"]) == 1 for shard in shards)
    with pytest.raises(ValueError, match="max_rows cap"):
        plan_shards(manifest(hf_source("u")), 2)
    with pytest.raises(ValueError, match="no file listing"):
        plan_shards(manifest(hf_source("u")), 2, {}, full=True)


def test_stripes_partition_rows_and_keep_global_row_numbers(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"n": n}) + "\n" for n in range(23)))

    def spec(stripe):
        return SourceSpec(
            source_id="s",
            uri=str(path),
            format="jsonl",
            adapter="chat",
            stream="chat",
            revision="sha256:" + "0" * 64,
            terms=TermsReview(approved=True, reviewed_by="t", reviewed_at="2026-01-01"),
            files=(SourceFile(str(path)),),
            stripe=stripe,
        )

    parts = [
        [(number, row["n"]) for number, row in iter_source_rows(spec((i, 3)))] for i in range(3)
    ]
    assert all(number == value for part in parts for number, value in part)
    assert sorted(number for part in parts for number, _ in part) == list(range(23))
    resumed = [number for number, _ in iter_source_rows(spec((1, 3)), start_row=10)]
    assert resumed == [n for n in range(10, 23) if n % 3 == 1]
    with pytest.raises(ValueError, match="stripe"):
        spec((3, 3))


def test_a_source_without_a_stripe_keeps_its_identity():
    from pathlib import Path

    from megatoken.data.sources import load_source_manifest

    sources = {
        source.source_id: source
        for source in load_source_manifest(
            Path(__file__).parents[1] / "configs/data/sources.agentic.json"
        )
    }
    # Fingerprint recorded before sources could be striped; roots prepared then must still resume.
    assert sources["smol-smoltalk"].fingerprint == (
        "50652b91f339b1eea515cacd4d3e09b63e1122cd47ae69832192aaf3343b9b38"
    )
    striped = SourceSpec(**{**hf_source("x"), "terms": TermsReview(approved=True)}, stripe=(0, 2))
    plain = SourceSpec(**{**hf_source("x"), "terms": TermsReview(approved=True)})
    assert plain.fingerprint != striped.fingerprint
