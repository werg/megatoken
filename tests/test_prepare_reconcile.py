"""Shards prepared apart and reconciled agree with one run over the same rows."""

import json
from dataclasses import replace

import pytest

from megatoken.data import (
    DedupConfig,
    Episode,
    Lineage,
    PreparedCorpus,
    PreparedDataset,
    Supervision,
    TermsReview,
    make_records,
    pin_local_source,
    prepare_sources,
    reconcile,
)
from megatoken.generation.codec import ByteCodec

CODEC = ByteCodec.byte_fallback()
CONFIG = DedupConfig(train_fraction=0.5, validation_fraction=0.25, max_descendants_per_parent=2)
TERMS = TermsReview(True, "fixture", reviewed_by="test", reviewed_at="2026-10-17")


def document(name, text, family, **fields):
    supervision = fields.pop("supervision", Supervision())
    return Episode(
        name,
        make_records(text.encode(), logical_id=name, role="document", policy_mask=True),
        Lineage("test", "fixed", family, **fields),
        stream="natural_documents",
        supervision=supervision,
    )


def body(index, changed=None):
    words = [f"{index:x}{i:x}" for i in range(180)]
    if changed is not None:
        words[changed] = "changed"
    return " ".join(words)


def source(path, episodes, source_id):
    path.write_text("\n".join(json.dumps(episode.to_dict()) for episode in episodes) + "\n")
    return replace(
        pin_local_source(
            path, source_id=source_id, adapter="native", stream="natural_documents", terms=TERMS
        )
    )


def accepted(root):
    with PreparedDataset(root, split=None) as dataset:
        return {
            episode.episode_id: dataset.connection.execute(
                "SELECT split,family_id FROM episodes WHERE episode_id=?", (episode.episode_id,)
            ).fetchone()["split"]
            for episode in dataset.iter_episodes()
        }


def shard_episodes():
    """Three shards whose rows meet across shards: a duplicate, a near duplicate, a family."""
    shared = body(0)
    return [
        [
            document("a-original", shared, "family:a"),
            document("a-unique", body(1), "family:b"),
            document("a-sibling", body(2), "family:shared"),
        ],
        [
            document("b-exact-copy", shared, "family:z"),  # same text as a-original
            document("b-near-copy", body(1, changed=90), "family:y"),  # near a-unique
            document("b-sibling", body(3), "family:shared"),  # same family as a-sibling
            document("b-other", body(4), "family:c"),
        ],
        [
            document("c-sibling", body(5), "family:shared"),
            document("c-other", body(6), "family:d"),
            document("c-linked", body(7), "family:e", ancestor_ids=("family:c",)),
        ],
    ]


def prepare_shards(tmp_path, groups):
    sources = [source(tmp_path / f"raw-{i}.jsonl", g, f"shard{i}") for i, g in enumerate(groups)]
    for index, one in enumerate(sources):
        assert (
            prepare_sources(
                [one], tmp_path / f"shard-{index}", CODEC, dedup=CONFIG, ingest_only=True
            )
            is None
        )
        assert not (tmp_path / f"shard-{index}" / "manifest.json").exists()
    digests = [tmp_path / f"shard-{index}" / "digest.sqlite" for index in range(len(sources))]
    summary = reconcile(digests, tmp_path / "verdicts")
    roots = []
    for index, one in enumerate(sources):
        prepare_sources(
            [one],
            tmp_path / f"shard-{index}",
            CODEC,
            dedup=CONFIG,
            decisions=tmp_path / "verdicts" / f"decisions-{index:04d}.sqlite",
        ).close()
        roots.append(tmp_path / f"shard-{index}")
    return sources, roots, summary


def test_reconciled_shards_hold_no_cross_shard_duplicate_and_one_split_per_family(tmp_path):
    groups = shard_episodes()
    sources, roots, summary = prepare_shards(tmp_path, groups)
    kept = {}
    for root in roots:
        kept.update(accepted(root))
    assert "b-exact-copy" not in kept and "b-near-copy" not in kept
    assert {"a-original", "a-unique"} <= set(kept)
    assert summary["cross_shard_duplicates"] == 2
    # Documents of one family stay in one split although three shards read them.
    siblings = {
        name: kept[name] for name in ("a-sibling", "b-sibling", "c-sibling") if name in kept
    }
    assert len(siblings) == CONFIG.max_descendants_per_parent  # the cap counts across shards
    assert len(set(siblings.values())) == 1
    # A document linked to another family by ancestry follows it.
    assert kept.get("c-linked") == kept.get("b-other") or "c-linked" not in kept
    # Shards are still ordinary roots that a corpus reads together.
    with PreparedCorpus(roots, split=None) as corpus:
        assert len(corpus) == len(kept)


def test_reconciliation_equals_one_run_over_the_same_rows_in_shard_order(tmp_path):
    groups = shard_episodes()
    _, roots, _ = prepare_shards(tmp_path, groups)
    sharded = {}
    for root in roots:
        sharded.update(accepted(root))
    whole = source(
        tmp_path / "whole.jsonl",
        [episode for group in groups for episode in group],
        "whole",
    )
    with prepare_sources([whole], tmp_path / "whole", CODEC, dedup=CONFIG) as dataset:
        single = {
            episode.episode_id: dataset.connection.execute(
                "SELECT split FROM episodes WHERE episode_id=?", (episode.episode_id,)
            ).fetchone()["split"]
            for episode in dataset.iter_episodes()
        }
    assert sharded == single


def test_benchmark_family_quarantine_crosses_shards(tmp_path):
    benchmark = document(
        "a-benchmark", body(10), "family:held", supervision=Supervision(benchmark_tags=("bench",))
    )
    groups = [
        [benchmark, document("a-fine", body(11), "family:f")],
        [
            document("b-descendant", body(12), "family:held"),
            document("b-fine", body(13), "family:g"),
        ],
    ]
    _, roots, _ = prepare_shards(tmp_path, groups)
    kept = {}
    for root in roots:
        kept.update(accepted(root))
    assert "a-benchmark" not in kept and "b-descendant" not in kept
    assert {"a-fine", "b-fine"} <= set(kept)


def test_variant_cap_counts_across_shards(tmp_path):
    groups = [
        [
            document(f"{shard}-{i}", body(8 + 2 * shard + i), "parent:one", task_id="task-1")
            for i in range(2)
        ]
        for shard in range(3)
    ]
    _, roots, _ = prepare_shards(tmp_path, groups)
    kept = {}
    for root in roots:
        kept.update(accepted(root))
    assert len(kept) == CONFIG.max_descendants_per_parent


def test_finishing_needs_this_shards_verdicts_and_matching_dedup_settings(tmp_path):
    groups = shard_episodes()[:2]
    sources = [source(tmp_path / f"raw-{i}.jsonl", g, f"shard{i}") for i, g in enumerate(groups)]
    for index, one in enumerate(sources):
        prepare_sources([one], tmp_path / f"s{index}", CODEC, dedup=CONFIG, ingest_only=True)
    with pytest.raises(ValueError, match="not both"):
        prepare_sources(
            [sources[0]], tmp_path / "x", CODEC, ingest_only=True, decisions=tmp_path / "d.sqlite"
        )
    other = DedupConfig(near_threshold=0.9)
    prepare_sources([sources[1]], tmp_path / "odd", CODEC, dedup=other, ingest_only=True)
    with pytest.raises(ValueError, match="different dedup"):
        reconcile(
            [tmp_path / "s0" / "digest.sqlite", tmp_path / "odd" / "digest.sqlite"], tmp_path / "v"
        )
    reconcile([tmp_path / "s0" / "digest.sqlite"], tmp_path / "alone")
    # Shard 1's rows were not in that reconciliation, so shard 0's verdicts cannot finish it.
    with pytest.raises(ValueError, match="no verdict"):
        prepare_sources(
            [sources[1]],
            tmp_path / "s1",
            CODEC,
            dedup=CONFIG,
            decisions=tmp_path / "alone" / "decisions-0000.sqlite",
        )
