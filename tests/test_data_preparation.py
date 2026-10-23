import json
from dataclasses import replace

import pytest
from prepare_support import interrupt_after_rows

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


def test_resume_after_atomic_row_commit_matches_uninterrupted_dataset(tmp_path, monkeypatch):
    codec = ByteCodec.byte_fallback()
    source = write_demo_source(tmp_path / "raw.jsonl", count=7)
    config = DedupConfig(train_fraction=1, validation_fraction=0)
    with monkeypatch.context() as crash:
        interrupt_after_rows(crash, 3)
        with pytest.raises(InterruptedError):
            prepare_sources([source], tmp_path / "resumed", codec, dedup=config)
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
        for phase in ("B0", "MB", "P1", "M1", "M2", "LC", "S", "D")
    } == {
        "B0": ("I", 250_000_000),
        "MB": ("G", 4_000_000),
        "P1": ("D", 400_000_000_000),
        "M1": ("D", 60_000_000_000),
        "M2": ("D", 40_000_000_000),
        "LC": ("D", 20_000_000_000),
        "S": ("P", 2_000_000_000),
        "D": ("Q", 10_000_000),
    }
    m1 = stage_recipe("M1")
    # The new cross-track evidence tasks take 0.1% from issue/PR sequences.
    assert round((m1.mixture["issue_pr_sequences"] + m1.mixture["cross_track_evidence"])
                 * m1.limit) == 16_213_080_000
    assert round(m1.mixture["build_logs"] * m1.limit) == 636_120_000
    # Media is part of the plan: a bridge stage, then explicit shares and replay.
    assert stage_recipe("MB").constraints["trainable"] == "media_bridge"
    p1 = stage_recipe("P1").mixture
    media = ("audio_identity", "image_identity", "video_identity", "transcription")
    assert all(p1[name] > 0 for name in media) and p1["long_document_reading"] == pytest.approx(
        0.0285
    )
    assert stage_recipe("M2").mixture["image_identity"] > 0


AGENTIC = ("executed_trajectories", "verified_trajectories", "tool_use", "environment_trajectories")
RAW_WEB_TEXT = ("natural_documents", "synthetic_documents")


def test_stage_plan_leans_on_agentic_and_chat_data_over_raw_text():
    p1 = stage_recipe("P1").mixture
    # Executed, verified and tool-use agent episodes and chat SFT dominate the text share;
    # raw web text is cut to well under half of its earlier 0.369 (two points of it
    # went to reading from a knowledge base and from a source collection).
    assert (sum(p1.get(name, 0) for name in AGENTIC)
            + p1["cross_track_evidence"]) == pytest.approx(0.26695)
    assert p1["chat"] == pytest.approx(0.076)
    assert sum(p1[name] for name in RAW_WEB_TEXT) == pytest.approx(0.133)
    assert p1["base_reading"] == p1["collection_reading"] == pytest.approx(0.0095)
    # Broad replay retains its proportions while dynamic execution gains supply.
    assert p1["long_document_reading"] == pytest.approx(0.0285)
    assert sum(
        v
        for k, v in p1.items()
        if k
        in {
            "audio_identity",
            "image_identity",
            "video_identity",
            "transcription",
            "verbalization",
            "speech_translation",
            "image_description",
            "audio_description",
            "video_description",
            "image_creation",
            "av_creation",
            "image_edit",
            "edit_judging",
            "edit_plan",
            "speech_edit",
            "av_edit",
            "av_understanding",
            "endpoints",
            "endpoint_conditioned_video",
            "speech_counterfactual",
        }
    ) == pytest.approx(0.1425)
    for phase in ("M1", "M2", "LC"):
        later = stage_recipe(phase).mixture
        assert later["executed_trajectories"] >= 0.076 and later["verified_trajectories"] >= 0.076
    assert stage_recipe("M1").mixture["executed_trajectories"] == pytest.approx(0.1995)
    m2 = stage_recipe("M2").mixture
    assert (m2["verified_trajectories"] + m2["speech_counterfactual"]
            + m2["cross_track_evidence"]) == pytest.approx(0.4655)
    s = stage_recipe("S").mixture
    assert s["tool_use"] + s["typed_exact_instructions"] + s["chat"] >= 0.225
    assert s["natural_document_replay"] <= 0.02


def test_shared_boilerplate_never_quarantines_distinct_documents_or_hides_a_near_duplicate(
    tmp_path,
):
    codec = ByteCodec.byte_fallback()
    header = " ".join(f"licence{i}" for i in range(80))

    def body(index, changed=None):
        words = [f"doc{index}word{i}" for i in range(200)]
        if changed is not None:
            words[changed] = "changed"
        return header + " " + " ".join(words)

    config = DedupConfig(max_candidates=4)
    with DedupStore(tmp_path / "families", config) as store:
        with store.transaction():
            statuses = [
                store.add(document(f"d{index}", body(index), f"family-{index}"), codec)
                for index in range(30)
            ]
            # After the header's shingles are shared by far more than four documents,
            # a near duplicate of an early document must still be found.
            near = store.add(document("near", body(3, changed=100), "near-family"), codec)
        assert statuses == ["accepted_candidate"] * 30
        assert near == "duplicate"
        assert store.census()["quarantined"] == 0
        assert (
            store.connection.execute(
                "SELECT duplicate_of FROM episodes WHERE episode_id='near'"
            ).fetchone()[0]
            == "d3"
        )


@pytest.mark.parametrize("train,validation", [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)])
def test_split_fractions_that_sum_to_one_are_accepted_despite_float_rounding(train, validation):
    DedupConfig(train_fraction=train, validation_fraction=validation)
    with pytest.raises(ValueError, match="proportions"):
        DedupConfig(train_fraction=train, validation_fraction=validation + 0.01)


def test_episode_content_hash_is_computed_once_and_not_serialized():
    episode = document("cached", "some text", "family")
    first = episode.content_sha256
    assert first == episode.content_sha256
    assert "_content_sha256" not in episode.to_dict()
    assert Episode.from_dict(episode.to_dict()).content_sha256 == first
    assert replace(episode, stream="other").content_sha256 == first


def filter_source(tmp_path, rows, *, adapter="document", options=None, **kwargs):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    source = pin_local_source(
        path,
        source_id="fixture",
        adapter=adapter,
        stream="natural_documents",
        terms=TermsReview(True, "test", reviewed_by="test", reviewed_at="2026-10-22"),
    )
    return replace(source, options=options or {}, **kwargs)


def test_declared_row_filters_select_rows_without_aborting_the_default_run(tmp_path):
    rows = [
        {
            "text": f"A document number {index} with enough words.",
            "family_id": f"f{index}",
            "metadata": {"score": index},
        }
        for index in range(6)
    ]
    source = filter_source(
        tmp_path,
        rows,
        options={"row_filters": [{"path": "metadata.score", "op": "gte", "value": 3}]},
    )
    with prepare_sources([source], tmp_path / "out", ByteCodec.byte_fallback()) as dataset:
        census = dataset.manifest["census"]
        assert census["accepted_episodes"] == 3
        assert census["filtered_rows"] == 3 and census["rejected_rows"] == 0
        assert json.loads((tmp_path / "out" / "rejections.json").read_text())["rows"] == []


def test_an_adapter_bug_is_never_swallowed_as_an_invalid_row(tmp_path, monkeypatch):
    source = filter_source(tmp_path, [{"text": "fine", "family_id": "f"}])

    def broken(row, adapter, context):
        return {}["missing"]

    monkeypatch.setattr("megatoken.data.prepare.adapt_row", broken)
    with pytest.raises(KeyError) as raised:
        prepare_sources([source], tmp_path / "out", ByteCodec.byte_fallback(), reject_invalid=True)
    assert any("row 0" in note for note in raised.value.__notes__)


def test_misspelled_adapter_options_fail_before_any_row_is_read(tmp_path):
    source = filter_source(
        tmp_path,
        [{"text": "fine", "family_id": "f"}],
        options={"allow_pending_final_actions": True},
    )
    with pytest.raises(ValueError, match="allow_pending_final_actions"):
        prepare_sources([source], tmp_path / "out", ByteCodec.byte_fallback())
    assert not (tmp_path / "out" / "preparation.sqlite").exists()


def test_record_limits_can_raise_the_chunking_policy_and_manifest_commits_last(
    tmp_path, monkeypatch
):
    from megatoken.data import storage
    from megatoken.data.chunking import ChunkingConfig

    written = []
    real = storage.write_json
    monkeypatch.setattr(
        storage, "write_json", lambda path, value: (written.append(path.name), real(path, value))
    )
    source = write_demo_source(tmp_path / "raw.jsonl", count=3)
    with prepare_sources(
        [source],
        tmp_path / "out",
        ByteCodec.byte_fallback(),
        chunking=ChunkingConfig(),
        max_record_tokens=4096,
        max_record_bytes=65536,
    ) as dataset:
        config = dataset.manifest["chunking"]["config"]
        assert (config["hard_max_tokens"], config["max_bytes"]) == (4096, 65536)
        assert config["soft_max_tokens"] == 768
    assert written[-1] == "manifest.json"
    assert set(written[:-1]) == {"rejections.json", "parser-review.json"}


def test_prepared_roots_are_checksummed_once_per_process_and_unknown_splits_are_errors(
    tmp_path, monkeypatch
):
    from megatoken.data import storage

    source = write_demo_source(tmp_path / "raw.jsonl", count=3)
    prepare_sources([source], tmp_path / "out", ByteCodec.byte_fallback()).close()
    hashed = []
    real = storage.sha256_file
    monkeypatch.setattr(storage, "sha256_file", lambda path: (hashed.append(path), real(path))[1])
    storage._VERIFIED_FILES.clear()
    for _ in range(3):
        with PreparedDataset(tmp_path / "out", split="train") as dataset:
            list(dataset.iter_episodes())
    first_pass = len(hashed)
    assert first_pass == len(set(hashed)) > 0
    with pytest.raises(ValueError, match="Unknown split"):
        PreparedDataset(tmp_path / "out", split="trian")
    (tmp_path / "out" / "episodes-000000.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        with PreparedDataset(tmp_path / "out", split="train") as dataset:
            list(dataset.iter_episodes())


def test_source_identity_ignores_review_stamps_and_where_the_files_sit(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"text": "a document", "family_id": "f"}\n')
    moved = tmp_path / "elsewhere" / "rows.jsonl"
    moved.parent.mkdir()
    moved.write_bytes(rows.read_bytes())

    def pinned(path, reviewed_at):
        return pin_local_source(
            path,
            source_id="fixture",
            adapter="document",
            stream="natural_documents",
            terms=TermsReview(True, "test", reviewed_by="someone", reviewed_at=reviewed_at),
        )

    first, later, relocated = (
        pinned(rows, "2026-10-01"),
        pinned(rows, "2026-10-22"),
        pinned(moved, "2026-10-22"),
    )
    assert first.fingerprint == later.fingerprint == relocated.fingerprint
    moved.write_text('{"text": "a different document", "family_id": "f"}\n')
    assert pinned(moved, "2026-10-22").fingerprint != first.fingerprint


def test_rows_before_a_pipeline_error_stay_committed_and_the_failing_row_does_not(
    tmp_path, monkeypatch
):
    from megatoken.data import prepare

    rows = [
        {"text": f"Document number {index} is here.", "family_id": f"f{index}"}
        for index in range(5)
    ]
    source = filter_source(tmp_path, rows)
    real = prepare.adapt_row

    def flaky(row, adapter, context):
        if context.row_index == 3:
            raise RuntimeError("adapter bug")
        return real(row, adapter, context)

    monkeypatch.setattr(prepare, "adapt_row", flaky)
    with pytest.raises(RuntimeError, match="adapter bug"):
        prepare_sources([source], tmp_path / "out", ByteCodec.byte_fallback())
    monkeypatch.setattr(prepare, "adapt_row", real)
    with DedupStore(tmp_path / "out" / "preparation.sqlite") as store:
        assert store.source_cursor(source.fingerprint) == (3, False)
        assert store.connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 3
    with prepare_sources([source], tmp_path / "out", ByteCodec.byte_fallback()) as dataset:
        assert dataset.manifest["census"]["accepted_episodes"] == 5
