"""The agentic coding manifests: pinned, licence-recorded, uncapped, and checked for overlap."""

import json
from pathlib import Path

import pytest

from megatoken.data.adapters import PROFILES, validate_options
from megatoken.data.recipes import stage_recipe
from megatoken.data.sources import load_source_manifest

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs/data"
CODING = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic-coding.json")}
GATED = {
    s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic-coding-gated.json")
}
EVERY = {**CODING, **GATED}
OTHERS = [
    load_source_manifest(CONFIGS / f"sources.{name}.json")
    for name in (
        "agentic",
        "agentic-gated",
        "agentic-more",
        "agentic-more-gated",
        "agentic-chat",
        "agentic-chat-gated",
        "later-stages",
    )
]
STREAMS = {
    "executed_trajectories",
    "verified_trajectories",
    "verified_successful_multiturn",
    "mixed_quality_episodes",
    "environment_trajectories",
    "synthetic_code_dialogue",
}


def read_key(source):
    """What a source reads: two sources with one key read the same rows."""
    return (
        source.uri,
        source.subset or "default",
        source.split,
        tuple(sorted(item.path for item in source.files)),
        json.dumps(source.options.get("row_filters", []), sort_keys=True),
        source.stream,
    )


def test_every_source_is_approved_pinned_uncapped_and_names_its_licence():
    for source in EVERY.values():
        assert source.terms.approved and source.terms.reviewed_by == "Gabriel Pickard", source
        assert source.terms.reviewed_at == "2026-10-22", source
        assert source.terms.license and source.terms.license_url, source
        assert source.max_rows is None, f"{source.source_id} keeps a smoke-run row cap"
        assert source.revision and source.revision != "main", source
        assert source.stream in STREAMS, source
        assert source.adapter in PROFILES, source
        validate_options(source.adapter, source.options)


def test_no_coding_id_is_used_by_another_manifest():
    taken = {s.source_id for manifest in OTHERS for s in manifest}
    assert sorted(taken & set(EVERY)) == []


def test_no_other_manifest_reads_the_same_rows_in_the_same_stream():
    mine = {read_key(s): s.source_id for s in EVERY.values()}
    assert len(mine) == len(EVERY), "two coding sources read the same rows"
    for manifest in OTHERS:
        for source in manifest:
            assert read_key(source) not in mine, (source.source_id, mine[read_key(source)])


def test_verified_streams_read_only_rows_with_a_recorded_passing_outcome():
    for source in EVERY.values():
        if source.stream in {"verified_trajectories", "verified_successful_multiturn"}:
            labelled = source.options.get("row_filters") or "outcome" in source.field_map
            curated_release = source.source_id in {"coderforge-reward1", "s-verified-coderforge"}
            assert labelled or curated_release, source.source_id
            assert "assume_outcome" not in source.options, source.source_id


def test_unverified_runs_are_admitted_only_by_name():
    for source in EVERY.values():
        if source.stream in {"executed_trajectories", "mixed_quality_episodes"}:
            quality = source.options.get("quality", {})
            labelled = source.field_map.get("outcome") or source.options.get("row_filters")
            stated = source.options.get("assume_outcome") or source.options.get(
                "admit_unknown_outcome"
            )
            assert labelled or stated or quality.get("min_scored_positions") == 0, source.source_id


def test_the_streams_the_sources_feed_are_in_the_stage_plan():
    planned = set(stage_recipe("P1").mixture) | set(stage_recipe("M2").mixture)
    assert {"executed_trajectories", "verified_trajectories"} <= planned


def test_the_gated_manifest_names_what_to_click_on_the_hub():
    assert set(GATED) == {"openswe-trajectories"}
    source = GATED["openswe-trajectories"]
    assert source.uri == "hf://datasets/GAIR/OpenSWE"
    assert "agree" in source.terms.notes.lower() or "gated" in source.terms.notes.lower()


def test_evaluation_sets_are_not_training_sources():
    for source in EVERY.values():
        text = (source.source_id + source.uri).lower()
        assert "swe-bench-pro" not in text and "terminal-bench" not in text, source.source_id
        assert "swebench-verified" not in text and "humaneval" not in text, source.source_id


def test_the_run_file_prepares_these_sources_and_every_hub_source_is_listed():
    job = json.loads((ROOT / "configs/cloud/prepare-full-agentic-coding.json").read_text())
    assert job["sources"] == "configs/data/sources.agentic-coding.json"
    assert job["review_config"] == "configs/data/parser-review.agentic.json"
    assert not job["resources"].get("gpu")
    listing = json.loads((ROOT / job["listing"]).read_text())
    assert set(listing) == set(CODING)
    assert all(listing[source_id] for source_id in CODING)


def test_a_capped_smoke_plan_partitions_the_uncapped_manifest():
    from megatoken.data.sharding import plan_shards

    manifest = json.loads((CONFIGS / "sources.agentic-coding.json").read_text())
    shards = plan_shards(manifest, 16, cap=300)
    capped = {s["source_id"]: s["max_rows"] for shard in shards for s in shard["sources"]}
    assert capped == {s["source_id"]: 300 for s in manifest["sources"]}
    with pytest.raises(ValueError, match="cap"):
        plan_shards(manifest, 16, cap=0)


def test_the_smoke_job_reads_the_coding_manifest_with_a_row_cap():
    from megatoken import cloud

    spec = json.loads((ROOT / "configs/cloud/prepare-agentic-coding-mix.json").read_text())
    job = cloud.validate(spec)
    command = " ".join(job["command"])
    assert "configs/data/sources.agentic-coding.json" in command and "--cap" in command
    assert "--full" not in command and job["resources"]["gpu"] == 0


def test_every_listed_file_is_at_the_revision_its_source_pins():
    listing = json.loads((CONFIGS / "files.agentic-coding.json").read_text())
    for source_id, source in CODING.items():
        for name, size in listing[source_id]:
            assert f"@{source.revision}/" in name and size >= 0, (source_id, name)
