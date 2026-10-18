import json
from dataclasses import replace

import pytest

from megatoken.data import (
    DedupConfig,
    DedupStore,
    Episode,
    Lineage,
    PreparedDataset,
    SourceSpec,
    Supervision,
    TermsReview,
    make_records,
    pin_local_source,
    prepare_sources,
    stage_recipe,
    write_demo_source,
)
from megatoken.data.lineage import LineageJoin
from megatoken.generation.codec import ByteCodec
from megatoken.io import sha256_file


def document(name, text, family, **lineage):
    return Episode(
        name,
        make_records(text.encode(), logical_id=name, role="document", policy_mask=True),
        Lineage("test", "fixed", family, **lineage),
        stream="natural_documents",
    )


def test_external_dedup_unions_before_splitting_and_quarantines_benchmark_family(tmp_path):
    codec = ByteCodec.byte_fallback()
    content = " ".join(f"word{i}" for i in range(220))
    first = document("first", content, "parent:a")
    duplicate = document("duplicate", content, "parent:b")
    near = document("near", content.replace("word100", "changed100"), "parent:c")
    child = document(
        "child", "entirely different descendant", "parent:d", ancestor_ids=("parent:c",)
    )
    with DedupStore(tmp_path / "dedup.sqlite") as store:
        store.quarantine_benchmarks(family_keys=("parent:b",))
        with store.transaction():
            store.add(first, codec)
            store.add(duplicate, codec)
            store.add(near, codec)
            store.add(child, codec)
        store.freeze()
        assert {store.split_of(f"parent:{x}") for x in "abcd"} == {"quarantine"}
        assert not list(store.accepted_rows())
        assert store.census()["duplicates"] >= 2


def test_inherited_evaluation_split_cannot_be_reassigned(tmp_path):
    codec = ByteCodec.byte_fallback()
    episode = document(
        "child", "a generated evaluation descendant", "held-out-parent", frozen_split="test"
    )
    with DedupStore(
        tmp_path / "store", DedupConfig(train_fraction=1, validation_fraction=0)
    ) as store:
        with store.transaction():
            store.add(episode, codec)
        store.freeze()
        assert store.split_of("held-out-parent") == "test"
        with pytest.raises(ValueError, match="Unknown ancestor"):
            store.split_of("new-parent-after-generation")


def test_chunk_independent_exact_dedup_and_near_benchmark_detection(tmp_path):
    codec = ByteCodec.byte_fallback()
    text = " ".join(f"word{index}" for index in range(220))
    benchmark = replace(
        document("benchmark", text, "held-out"),
        supervision=Supervision(benchmark_tags=("held-out-suite",)),
    )
    rechunked = Episode(
        "rechunked",
        make_records(
            text.encode(), logical_id="copy", role="document", policy_mask=True, max_bytes=17
        ),
        Lineage("test", "fixed", "another-family"),
    )
    near = document("near", text.replace("word100", "changed100"), "near-family")
    assert benchmark.content_sha256 == rechunked.content_sha256
    with DedupStore(tmp_path / "families") as store:
        with store.transaction():
            store.add(benchmark, codec)
            assert store.add(rechunked, codec) == "duplicate"
            store.add(near, codec)
        store.freeze()
        assert store.split_of("near-family") == "quarantine"
        assert not list(store.accepted_rows())


def test_time_holdout_compares_instants_across_offsets(tmp_path):
    before = document(
        "before", "one earlier document", "before-family", created_at="2026-01-01T01:00:00+02:00"
    )
    after = document(
        "after",
        "a different later document",
        "after-family",
        created_at="2025-12-31T23:30:00-01:00",
    )
    config = DedupConfig(train_fraction=1, validation_fraction=0, evaluation_after="2026-01-01")
    with DedupStore(tmp_path / "families", config) as store:
        with store.transaction():
            store.add(before, ByteCodec.byte_fallback())
            store.add(after, ByteCodec.byte_fallback())
        store.freeze()
        assert store.split_of("before-family") == "train"
        assert store.split_of("after-family") == "test"


def test_quality_rejection_still_quarantines_known_benchmark_ancestry(tmp_path):
    benchmark = document("short", "held-out", "parent")
    descendant = document(
        "child",
        "A sufficiently long otherwise acceptable training descendant.",
        "child-family",
        ancestor_ids=("parent",),
    )
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "\n".join(json.dumps(episode.to_dict()) for episode in (benchmark, descendant)) + "\n"
    )
    source = pin_local_source(
        source_path,
        source_id="fixture",
        adapter="native",
        stream="natural_documents",
        terms=TermsReview(True, "fixture", reviewed_by="test", reviewed_at="2026-10-17"),
    )
    source = replace(source, options={"quality": {"min_tokens": 20}})
    with prepare_sources(
        [source],
        tmp_path / "prepared",
        ByteCodec.byte_fallback(),
        reject_invalid=True,
        benchmark_quarantine={"content_sha256": [benchmark.content_sha256]},
    ) as dataset:
        assert len(dataset) == 0
        assert dataset.manifest["census"]["rejected_rows"] == 1
        assert dataset.manifest["census"]["quarantined"] == 1


def test_resume_after_atomic_row_commit_matches_uninterrupted_dataset(tmp_path):
    codec = ByteCodec.byte_fallback()
    source = write_demo_source(tmp_path / "raw.jsonl", count=7)
    config = DedupConfig(train_fraction=1, validation_fraction=0)
    with pytest.raises(InterruptedError):
        prepare_sources([source], tmp_path / "resumed", codec, dedup=config, _interrupt_after=3)
    with prepare_sources(
        [source], tmp_path / "resumed", codec, dedup=config, shard_bytes=1000
    ) as resumed:
        resumed_ids = [episode.episode_id for episode in resumed.iter_episodes()]
        resumed_census = resumed.manifest["census"]
        assert len(resumed_ids) == 7
        assert set(
            episode.episode_id for episode in resumed.iter_episodes(rank=0, world_size=2)
        ).isdisjoint(episode.episode_id for episode in resumed.iter_episodes(rank=1, world_size=2))
    with prepare_sources(
        [source], tmp_path / "straight", codec, dedup=config, shard_bytes=1000
    ) as straight:
        assert resumed_ids == [episode.episode_id for episode in straight.iter_episodes()]
        assert resumed_census == straight.manifest["census"]
        assert straight.manifest["census"]["P"] == 0
        assert straight.manifest["census"]["free_position_candidates"] > 0


def test_changed_sources_and_tampered_shards_fail_closed(tmp_path):
    codec = ByteCodec.byte_fallback()
    source = write_demo_source(tmp_path / "source.jsonl", count=2)
    with prepare_sources([source], tmp_path / "prepared", codec) as dataset:
        shard = dataset.root / dataset.manifest["shards"][0]["path"]
    with shard.open("ab") as handle:
        handle.write(b"corrupt")
    with PreparedDataset(tmp_path / "prepared", split=None) as dataset:
        with pytest.raises(ValueError, match="checksum"):
            dataset[0]
    with (tmp_path / "source.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="Source size changed"):
        prepare_sources([source], tmp_path / "prepared", codec)


def test_lineage_join_is_exact_checked_and_cannot_modify_content(tmp_path):
    path = tmp_path / "parents.jsonl"
    path.write_text(
        json.dumps(
            {
                "row_id": "variant",
                "family_id": "original-document",
                "ancestor_ids": ["older-snapshot"],
            }
        )
        + "\n"
    )
    join = LineageJoin(path, sha256_file(path), tmp_path / "parents.sqlite")
    result = join.apply({"uuid": "variant", "text": "kept byte exact"}, "uuid")
    assert result["text"] == "kept byte exact" and result["family_id"] == "original-document"
    with pytest.raises(ValueError, match="No original-parent"):
        join.apply({"uuid": "unknown"}, "uuid")
    join.close()
    with pytest.raises(ValueError, match="checksum"):
        LineageJoin(path, "wrong", tmp_path / "other.sqlite")


def test_source_revision_and_terms_are_explicit():
    with pytest.raises(ValueError, match="immutable"):
        SourceSpec("s", "hf://datasets/owner/name", "hf", "chat", "chat", "main")
    with pytest.raises(ValueError, match="unapproved"):
        TermsReview().require_approval()


def test_current_stage_budget_units_and_proposed_mixtures():
    assert {
        phase: (stage_recipe(phase).meter, stage_recipe(phase).limit)
        for phase in ("B0", "P1", "M1", "M2", "LC", "S", "D", "R")
    } == {
        "B0": ("I", 250_000_000),
        "P1": ("D", 400_000_000_000),
        "M1": ("D", 60_000_000_000),
        "M2": ("D", 40_000_000_000),
        "LC": ("D", 20_000_000_000),
        "S": ("P", 2_000_000_000),
        "D": ("P", 1_000_000_000),
        "R": ("Z", 32_000_000_000),
    }
    assert stage_recipe("M1").exposures["issue_pr_sequences"] == 24_480_000_000
    assert stage_recipe("M1").exposures["build_logs"] == 720_000_000
    assert stage_recipe("LC").constraints["writer_reader_scored_windows_min_fraction"] == 0.3
