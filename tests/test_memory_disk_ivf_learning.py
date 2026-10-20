"""Moved trainable keys and disk postings are one atomic sparse Adam transaction."""
import numpy as np
import torch

from megatoken.memory.contracts import (
    Collection, Footprint, SemanticSpace, SourceRange, Visibility,
)
from megatoken.memory.disk_ivf import SQLiteDiskIVF
from megatoken.memory.optimizer import RowID, SparseRowAdam
from megatoken.memory.sqlite import SQLiteKB


def test_sparse_key_moves_ivf_cell_without_stale_posting(tmp_path):
    space = SemanticSpace(
        "live", "v1", "teacher", "revision", 8, "doc", "query", "context",
    )
    support = Footprint((
        SourceRange("source", "v1", 0, 20, "source-digest", "team"),
    ))
    forward = np.array([1.] + [0.] * 7, dtype=np.float32)
    backwards = -forward
    centroids = np.stack([forward, backwards])
    with SQLiteKB(tmp_path / "bank.sqlite", space, value_dim=16) as bank:
        bank.insert_source(
            scope="team", footprint=support,
            content_id="source/chunk", asset_ref="source.bin", key=forward,
        )
        record_id = bank.insert_memory(
            scope="team", footprint=support, assignment="source/chunk",
            key=forward, value=np.zeros(16), birth_step=0,
        )
        index = SQLiteDiskIVF(bank, centroids, nprobe=1)
        index.rebuild(scope="team", collection=Collection.SOURCE)
        index.rebuild(scope="team", collection=Collection.MEMORY)
        visible = Visibility("team", committed_cursor=1)
        assert index.search(
            backwards, collection=Collection.MEMORY, scope="team",
            visibility=visible, limit=1,
        ).hits == ()
        sparse = SparseRowAdam(bank, lr=2., index=index)
        identifier = RowID(Collection.MEMORY, "record", record_id, "key", "semantic-v1")
        selected = sparse.select(identifier)
        selected[0].backward()
        assert sparse.step(step=1) == 1
        moved = bank.get(Collection.MEMORY, record_id)
        assert moved.key[0] < 0
        selected = index.search(
            backwards, collection=Collection.MEMORY, scope="team",
            visibility=visible, limit=1,
        )
        assert selected.ids == (record_id,)
        posting = bank.db.execute(
            "SELECT cell,key_revision FROM postings WHERE record_id=?",
            (record_id,),
        ).fetchone()
        assert posting["cell"] == 1
        assert posting["key_revision"] == moved.key_revision
        moments = bank.moments(
            collection=Collection.MEMORY, kind="record", entity_id=record_id,
            field="key", layout_generation="semantic-v1",
        )
        assert moments[2] == 1
