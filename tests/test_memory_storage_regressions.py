from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace

import numpy as np
import pytest

from megatoken.memory.contracts import Collection, Visibility
from megatoken.memory.growth import GrowthAllocator, GrowthPolicy
from megatoken.memory.index import CoarseIVF
from megatoken.memory.sqlite import SQLiteKB
from tests.test_memory_causal_sidecar import _example, _support
from tests.test_memory_foundations import space, support


def _memory(kb, name, key=None, footprint=None):
    return kb.insert_memory(
        scope="team", footprint=footprint or support(name), assignment=name,
        key=np.ones(8) if key is None else key, value=np.zeros(16), birth_step=0,
    )


def _growth(kb, **kw):
    kb.put_growth_state("readiness", {"coherence": True, "embeddings": True})
    return GrowthAllocator(kb, GrowthPolicy(value_dim=16, seed=1, **kw))


def test_retired_relation_pair_is_not_reproposed(tmp_path):
    with SQLiteKB(tmp_path / "a.sqlite", space(), value_dim=16) as kb:
        a, b, c = (_memory(kb, n) for n in "abc")
        rel = kb.insert_relation(
            scope="team", left=a, right=b, prior=.5, value=np.zeros(16),
            dependence_logits=(0., 0.), footprint=support("pairsrc"),
        )
        kb.tombstone_source_version("pairsrc", "version1")
        assert not kb.relations("team")
        growth = _growth(kb)
        windows = {"w1": ("a", "b", "c"), "w2": ("a", "b", "c")}
        ids = growth.admit_pairs(windows, scope="team", visibility=Visibility("team", 9))
        assert rel not in ids and len(ids) == 2  # (a,c),(b,c); (a,b) stays retired


def test_pair_budget_is_respected_without_wasted_degrees(tmp_path):
    with SQLiteKB(tmp_path / "b.sqlite", space(), value_dim=16) as kb:
        for n in "abcd":
            _memory(kb, n)
        growth = _growth(kb, max_pairs=1)
        ids = growth.admit_pairs(
            {"w1": ("a", "b"), "w2": ("a", "b"), "w3": ("c", "d"), "w4": ("c", "d")},
            scope="team", visibility=Visibility("team", 9))
        assert len(ids) == 1


def test_retrieval_supervision_is_causal_with_serving_meridian():
    # meridian=3, deadline=4: serving at cursor<=3 can never return available_at=4.
    with pytest.raises(ValueError, match="not causally eligible"):
        replace(_example(), document_key_context=_support("late", available_at=4))
    with pytest.raises(ValueError, match="not causally eligible"):
        replace(_example(), payload_support=_support("late", available_at=4))


def test_insert_revocation_check_is_inside_transaction(tmp_path):
    with SQLiteKB(tmp_path / "c.sqlite", space(), value_dim=16) as kb:
        fp = support("racy")
        seen = []
        original = kb.footprint_revoked

        def probe(footprint):
            seen.append(kb._depth)
            return original(footprint)
        kb.footprint_revoked = probe
        _memory(kb, "x", footprint=fp)
        assert seen and all(depth > 0 for depth in seen)


def test_readonly_bank_is_not_immutable_and_sees_live_commits(tmp_path):
    path = tmp_path / "d.sqlite"
    with SQLiteKB(path, space(), value_dim=16) as kb:
        _memory(kb, "a")
        with SQLiteKB(path, space(), value_dim=16, readonly=True) as ro:
            assert len(ro.scan(Collection.MEMORY, "team")) == 1
            _memory(kb, "b")
            assert len(ro.scan(Collection.MEMORY, "team")) == 2
            with pytest.raises(PermissionError):
                _memory(ro, "c")
            with pytest.raises(sqlite3.OperationalError):
                ro.db.execute("DELETE FROM records")


def test_read_snapshot_is_single_transaction(tmp_path):
    path = tmp_path / "e.sqlite"
    with SQLiteKB(path, space(), value_dim=16) as kb:
        kb.db.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
        _memory(kb, "a")
        with SQLiteKB(path, space(), value_dim=16, readonly=True) as ro:
            with ro.read_snapshot():
                assert len(ro.scan(Collection.MEMORY, "team")) == 1
                writer = threading.Thread(target=_memory, args=(kb, "b"))
                writer.start()
                writer.join()
                assert len(ro.scan(Collection.MEMORY, "team")) == 1
            assert len(ro.scan(Collection.MEMORY, "team")) == 2


def test_coarse_ivf_rejects_negative_limit(tmp_path):
    with SQLiteKB(tmp_path / "f.sqlite", space(), value_dim=16) as kb:
        index = CoarseIVF(kb, np.eye(8)[:2], probes=1)
        with pytest.raises(ValueError, match="Negative"):
            index.search(np.ones(8), collection=Collection.MEMORY, scope="team",
                         visibility=Visibility("team", 1), limit=-1)


def test_retired_rows_cannot_be_updated(tmp_path):
    with SQLiteKB(tmp_path / "g.sqlite", space(), value_dim=16) as kb:
        a = _memory(kb, "a", footprint=support("gone"))
        b = _memory(kb, "b")
        c = _memory(kb, "c")
        rel = kb.insert_relation(
            scope="team", left=b, right=c, prior=.5, value=np.zeros(16),
            dependence_logits=(0., 0.), footprint=support("rel"),
        )
        kb.update_record(Collection.MEMORY, b, strength=.3)  # active still fine
        kb.tombstone_source_version("gone", "version1")
        kb.tombstone_source_version("rel", "version1")
        with pytest.raises(ValueError, match="retired"):
            kb.update_record(Collection.MEMORY, a, strength=.5,
                             posting_cell=0, centroids_revision="r")
        assert kb.db.execute("SELECT count(*) FROM postings WHERE record_id=?",
                             (a,)).fetchone()[0] == 0
        with pytest.raises(ValueError, match="retired"):
            kb.update_relation(rel, strength=.5)
