"""Source collections: the disk index agrees with exact search; items of every medium."""

from __future__ import annotations

import numpy as np
import pytest
from media_support import prepared_record

from megatoken.data.media_assets import TensorAssetStore
from megatoken.retrieval.collection import SourceCollection, text_item_id
from megatoken.retrieval.index import rank_exact, unit
from megatoken.retrieval.session import SourceSession
from megatoken.teacher import EMBEDDING_WIDTH, PINNED_TEACHER


def clustered_keys(count: int, *, clusters: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, EMBEDDING_WIDTH))
    rows = centers[rng.integers(0, clusters, count)] + 0.6 * rng.standard_normal(
        (count, EMBEDDING_WIDTH)
    )
    return unit(rows)


def text_collection(tmp_path, keys: np.ndarray, **options) -> SourceCollection:
    collection = SourceCollection.create(tmp_path / "docs", name="docs", teacher=PINNED_TEACHER)
    for number, key in enumerate(keys):
        collection.add_text(f"passage {number}".encode(), key, origin=f"doc-{number}.txt")
    collection.reindex(**options)
    return collection


def test_source_search_records_snapshot_and_excludes_later_publication(tmp_path):
    keys = clustered_keys(2)
    collection = text_collection(tmp_path, keys[:1], cells=1)
    try:
        source = SourceSession({"docs": collection}, threshold=-1)
        first = source.candidates(keys[0])
        original = source.search_snapshots["docs"]
        assert original["revision"] == 1
        later = collection.add_text(b"later evidence", keys[1], origin="later.txt")
        assert collection.publication(later)["revision"] > original["revision"]
        (chosen,) = source.choose(first, 0)
        assert chosen.snapshot_revision == chosen.publication_revision == 1
        assert chosen.available_at_ns <= chosen.searched_at_ns
        source.revision_limits = {"docs": 1}
        frozen = source.candidates(keys[1])
        assert all(candidate.hit.row != later for candidate in frozen)
        source.available_at_ns = 0
        assert source.candidates(keys[1]) == []
        assert source.search_snapshots["docs"]["available_at_ns"] == 0
    finally:
        collection.close()


def test_the_index_returns_exactly_what_exact_search_returns(tmp_path):
    keys = clustered_keys(900)
    collection = text_collection(tmp_path, keys, cells=16)
    assert collection.info()["index"]["cells"] == 16
    queries = unit(keys[::90] + 0.2 * np.random.default_rng(1).standard_normal((10, 768)))
    for query in queries:
        exact = collection.search_exact(query, 10)
        # Probing every cell makes int8 ranking plus exact rescoring the exact top-k.
        indexed = collection.search(query, 10, probes=16)
        assert [hit.row for hit in indexed] == [hit.row for hit in exact]
        assert [hit.score for hit in indexed] == pytest.approx([hit.score for hit in exact])
        # With the calibrated probe count it is the exact ranking of the probed cells.
        probed = collection.search(query, 10)
        assert [hit.score for hit in probed] == sorted((hit.score for hit in probed), reverse=True)
        assert {hit.row for hit in probed} <= {hit.row for hit in collection.search(query, 900)}


def test_int8_pruning_never_drops_an_exact_winner():
    rng = np.random.default_rng(3)
    keys = unit(rng.standard_normal((500, 768)) + 4.0)  # all close together: scores nearly tie
    scales = np.abs(keys).max(axis=1) / 127
    codes = np.rint(keys / scales[:, None]).astype(np.int8)
    ids = np.arange(500)
    for _ in range(20):
        query = unit(rng.standard_normal(768) + 4.0).astype(np.float64)
        top = rank_exact(ids, scales, codes, query, 7, lambda chosen: keys[chosen])
        exact = sorted(range(500), key=lambda i: (-float(keys[i].astype(np.float64) @ query), i))
        assert [i for i, _ in top] == exact[:7]


def test_search_is_deterministic_and_skips_observed_items_before_ranking(tmp_path):
    keys = clustered_keys(200, seed=2)
    collection = text_collection(tmp_path, keys)
    query = keys[5]
    first = collection.search(query, 5)
    assert first == collection.search(query, 5)
    assert first[0].row == 6  # rows start at 1; item 5 is its own best match
    skipped = collection.search(query, 5, exclude={first[0].row})
    assert [hit.row for hit in skipped][:4] == [hit.row for hit in first][1:5]
    assert len(skipped) == 5 and first[0].row not in {hit.row for hit in skipped}


def test_items_of_every_medium_share_one_collection(tmp_path):
    store = TensorAssetStore(tmp_path / "prepared")
    collection = SourceCollection.create(tmp_path / "mixed", name="mixed", teacher=PINNED_TEACHER)
    clips = {
        modality: prepared_record(store, modality, seed=seed, target=True)
        for seed, modality in enumerate(("audio", "image", "video"), start=1)
    }
    piece = prepared_record(store, "text", seed=9, target=False)
    for modality, record in clips.items():
        collection.add_record(record, store, origin=f"clip.{modality}")
    collection.add_record(piece, store, origin="long.txt#bytes=0-9000")
    passage = b"The vault opens with key 31."
    key = np.random.default_rng(4).standard_normal(768)
    collection.add_text(passage, key, origin="notes.txt")
    # An item twice is held once.
    assert collection.add_text(passage, key, origin="again.txt") is None
    assert collection.add_record(clips["image"], store, origin="again.png") is None
    collection.reindex()
    assert collection.count == 5
    for record in (*clips.values(), piece):
        embedding = store.load(record.embedding)
        (hit,) = collection.search(embedding, 1, probes=collection.info()["index"]["cells"])
        item = collection.item(hit.row)
        assert item.item_id == record.source.source_family
        expected = {"text": "compressed_text"}.get(record.modality, record.modality)
        assert item.modality == expected and item.text is None
        # The collection stands alone: its own copy of the features, no target.
        copy = collection.features(item)
        assert np.array_equal(copy, store.load(record.source))
        assert item.record.target is None and item.record.embedding == record.embedding
    (hit,) = collection.search(key, 1)
    text = collection.item(hit.row)
    assert (text.modality, text.text, text.origin) == ("text", passage, "notes.txt")
    assert text.item_id == text_item_id(passage)
    assert collection.info()["modalities"] == {
        "audio": 1,
        "compressed_text": 1,
        "image": 1,
        "text": 1,
        "video": 1,
    }


def test_items_added_later_are_found_in_their_cell(tmp_path):
    keys = clustered_keys(300, seed=5)
    collection = text_collection(tmp_path, keys[:250], cells=9)
    for number in range(250, 300):
        collection.add_text(f"late {number}".encode(), keys[number], origin="late")
    assert collection.info()["indexed_items"] == 300
    for number in range(250, 300, 7):
        assert collection.search(keys[number], 1, probes=9)[0].item_id == text_item_id(
            f"late {number}".encode()
        )


def test_a_mounted_collection_is_read_only_and_checks_its_teacher(tmp_path):
    keys = clustered_keys(50, seed=6)
    text_collection(tmp_path, keys).close()
    mounted = SourceCollection.open(tmp_path / "docs", teacher=PINNED_TEACHER)
    other = SourceCollection.open(tmp_path / "docs")  # a second reader, as another rank
    with pytest.raises(PermissionError):
        mounted.add_text(b"x", keys[0], origin="x")
    assert mounted.search(keys[3], 3) == other.search(keys[3], 3)
    with pytest.raises(ValueError, match="teacher"):
        SourceCollection.open(tmp_path / "docs", teacher="another")
    with pytest.raises(FileExistsError):
        SourceCollection.create(tmp_path / "docs", name="docs", teacher=PINNED_TEACHER)


def test_a_session_observes_each_item_once_and_only_above_its_threshold(tmp_path):
    keys = clustered_keys(80, seed=7)
    collection = text_collection(tmp_path, keys)
    collection.set_threshold(0.9)
    session = SourceSession({"docs": collection}, max_sources=2)
    query = keys[10]
    first = session.choose(session.candidates(query), lookup=0)
    assert [r.item.text for r in first] == [b"passage 10"]  # only one item reaches 0.9
    session.commit(first)
    again = session.choose(session.candidates(query), lookup=1)
    assert b"passage 10" not in [r.item.text for r in again]
    session.threshold = -1.0
    best = session.choose(session.candidates(query), lookup=2)
    assert len(best) == 2 and len({r.item.row for r in best}) == 2
    assert all(r.item.row not in session.observed["docs"] for r in best)
    assert session.log == first


def test_collections_rank_together_with_ties_to_the_earlier_mount(tmp_path):
    keys = clustered_keys(40, seed=8)
    one = text_collection(tmp_path / "a", keys)
    two = SourceCollection.create(tmp_path / "b", name="b", teacher=PINNED_TEACHER)
    for number, key in enumerate(keys):
        two.add_text(f"other {number}".encode(), key, origin="b")
    two.reindex()
    session = SourceSession({"a": one, "b": two}, max_sources=2, threshold=0.0)
    chosen = session.choose(session.candidates(keys[3]), lookup=0)
    assert [(r.collection, r.item.text) for r in chosen] == [
        ("a", b"passage 3"),
        ("b", b"other 3"),
    ]


def test_a_collection_that_has_doubled_asks_to_be_reindexed(tmp_path):
    keys = clustered_keys(12, seed=3)
    collection = SourceCollection.create(tmp_path / "grow", name="grow", teacher=PINNED_TEACHER)
    try:
        assert not collection.needs_reindex  # empty: nothing to index
        for number, key in enumerate(keys[:4]):
            collection.add_text(f"passage {number}".encode(), key, origin=f"doc-{number}.txt")
        assert collection.needs_reindex  # never indexed
        collection.reindex()
        assert not collection.needs_reindex
        for number, key in enumerate(keys[4:7], start=4):
            collection.add_text(f"passage {number}".encode(), key, origin=f"doc-{number}.txt")
        assert not collection.needs_reindex  # 7 <= 2 * 4
        collection.add_text(b"passage 9", keys[9], origin="doc-9.txt")
        collection.add_text(b"passage 10", keys[10], origin="doc-10.txt")
        assert collection.needs_reindex  # 9 > 2 * 4
    finally:
        collection.close()
