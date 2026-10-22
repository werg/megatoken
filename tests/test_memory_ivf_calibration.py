"""IVF probe calibration, its publication, and exact rescoring of probed cells."""

import numpy as np
import pytest

from megatoken.memory.contracts import (
    Collection,
    Footprint,
    SemanticSpace,
    SourceRange,
    Visibility,
)
from megatoken.memory.disk_ivf import SQLiteDiskIVF
from megatoken.memory.index import ExactIndex
from megatoken.memory.index_cli import calibrate_nprobe
from megatoken.memory.sqlite import SQLiteKB

SPACE = SemanticSpace("live", "v1", "teacher", "revision", 8, "doc", "query", "context")


def unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


def test_calibration_picks_the_smallest_probe_count_meeting_the_target():
    rng = np.random.default_rng(0)
    keys = unit(rng.standard_normal((2000, 8)))
    centroids = unit(rng.standard_normal((16, 8)))
    queries = unit(rng.standard_normal((64, 8)))
    nprobe, curve = calibrate_nprobe(keys, centroids, queries, k=10, target_recall=0.9)
    assert curve[16] == 1.0  # probing every cell is exact
    assert curve[nprobe] >= 0.9
    assert all(curve[n] < 0.9 for n in curve if n < nprobe)
    assert calibrate_nprobe(keys, centroids, queries, k=10, target_recall=1.0)[0] <= 16
    with pytest.raises(ValueError):
        calibrate_nprobe(keys, centroids, queries, target_recall=0)


def _bank(path, keys, *, hidden=()):
    bank = SQLiteKB(path, SPACE, value_dim=4)
    for i, key in enumerate(keys):
        later = 5 if i in hidden else 0
        bank.insert_source(
            scope="team",
            footprint=Footprint(
                (SourceRange(f"s{i}", "v1", 0, 1, f"d{i}", "team", available_at=later),)
            ),
            content_id=f"c{i}",
            asset_ref="a",
            key=key,
        )
    return bank


def test_full_probe_search_equals_exact_search_and_filters_visibility(tmp_path):
    rng = np.random.default_rng(1)
    keys = unit(rng.standard_normal((300, 8)))
    centroids = unit(rng.standard_normal((6, 8)))
    with _bank(tmp_path / "kb.sqlite", keys, hidden={0, 1, 2}) as bank:
        index = SQLiteDiskIVF(bank, centroids, nprobe=6, batch_size=7)
        index.rebuild(scope="team", collection=Collection.SOURCE)
        visible = Visibility("team", 1)
        for query in unit(rng.standard_normal((5, 8))):
            search = dict(collection=Collection.SOURCE, scope="team", visibility=visible, limit=9)
            approximate = index.search(query, **search, excluded={5})
            exact = ExactIndex(bank).search(query, **search, excluded={5})
            assert approximate.ids == exact.ids
            assert {0, 1, 2, 5}.isdisjoint(approximate.ids)
            assert approximate.candidates_evaluated == 299  # every key but the excluded one
            assert approximate.shortfall == 0
        assert index.search(keys[3], **search, excluded=None).hits[0].record.id == 4


def test_int8_codes_never_change_the_exact_ranking_of_near_ties(tmp_path):
    # Keys closer together than the int8 code error: only exact FP64 rescoring
    # can order them, and it must match exact search including ID tie-breaks.
    rng = np.random.default_rng(3)
    base = unit(rng.standard_normal((4, 8)))
    keys = unit(np.repeat(base, 40, axis=0) + 1e-3 * rng.standard_normal((160, 8)))
    keys[7] = keys[6]  # an exact duplicate: ties break by record ID
    centroids = unit(rng.standard_normal((3, 8)))
    with _bank(tmp_path / "kb.sqlite", keys) as bank:
        index = SQLiteDiskIVF(bank, centroids, nprobe=3)
        index.rebuild(scope="team", collection=Collection.SOURCE)
        search = dict(
            collection=Collection.SOURCE, scope="team", visibility=Visibility("team", 0), limit=25
        )
        for query in [*base, keys[6], *unit(rng.standard_normal((4, 8)))]:
            approximate = index.search(query, **search)
            exact = ExactIndex(bank).search(query, **search)
            assert approximate.ids == exact.ids
            assert [h.similarity for h in approximate.hits] == pytest.approx(
                [h.similarity for h in exact.hits], abs=1e-12
            )


def test_readers_take_the_published_probe_count(tmp_path):
    rng = np.random.default_rng(2)
    keys = unit(rng.standard_normal((50, 8)))
    centroids = unit(rng.standard_normal((4, 8)))
    with _bank(tmp_path / "kb.sqlite", keys) as bank:
        with pytest.raises(RuntimeError, match="no IVF publication"):
            SQLiteDiskIVF.published(bank, centroids)
        index = SQLiteDiskIVF(bank, centroids, nprobe=3)
        bank.put_growth_state(
            "ivf_publication",
            {"centroid_vectors_sha256": index.identity.centroid_sha256, "nprobe": 3},
        )
        assert SQLiteDiskIVF.published(bank, centroids).nprobe == 3
        with pytest.raises(RuntimeError, match="no IVF publication"):
            SQLiteDiskIVF.published(bank, unit(rng.standard_normal((4, 8))))
    with pytest.raises(ValueError):
        SQLiteDiskIVF(_bank(tmp_path / "b.sqlite", keys), centroids, nprobe=5)
