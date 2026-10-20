"""Atomic selected-row ownership and sparse optimizer step chronology."""
import numpy as np
import pytest

from megatoken.memory.contracts import (
    Collection,
    Footprint,
    SemanticSpace,
    SourceRange,
)
from megatoken.memory.optimizer import RowID, SparseRowAdam
from megatoken.memory.sqlite import SQLiteKB


def _space():
    return SemanticSpace(
        "optimizer-test", "v1", "local-teacher", "sha-pinned", 8,
        "document-v1", "query-v1", "window-v1",
    )


def _footprint():
    return Footprint((
        SourceRange("source", "v1", 0, 12, "digest", "scope"),
    ))


def _bank(path):
    bank = SQLiteKB(path, _space(), value_dim=6)
    row = bank.insert_memory(
        scope="scope", footprint=_footprint(),
        assignment="source/0", key=np.ones(8), value=np.zeros(6), birth_step=0,
    )
    return bank, row


def test_multi_field_sparse_step_commits_coherent_row_and_moments(tmp_path):
    bank, row = _bank(tmp_path / "bank.sqlite")
    try:
        sparse = SparseRowAdam(bank)
        keyid = RowID(Collection.MEMORY, "record", row, "key", "v1")
        valueid = RowID(Collection.MEMORY, "record", row, "value", "v1")
        key = sparse.select(keyid)
        value = sparse.select(valueid)
        (key[0] + value[1]).backward()
        assert sparse.step(step=1) == 2
        assert bank.moments(
            collection=Collection.MEMORY, kind="record",
            entity_id=row, field="key", layout_generation="v1",
        )[2] == 1
        assert bank.moments(
            collection=Collection.MEMORY, kind="record",
            entity_id=row, field="value", layout_generation="v1",
        )[2] == 1
    finally:
        bank.close()


def test_selected_sparse_row_rejects_concurrent_mutation_without_partial_commit(tmp_path):
    bank, row = _bank(tmp_path / "bank.sqlite")
    try:
        sparse = SparseRowAdam(bank)
        keyid = RowID(Collection.MEMORY, "record", row, "key", "v1")
        key = sparse.select(keyid)
        key.sum().backward()
        bank.update_record(Collection.MEMORY, row, strength=0.5)
        prior = bank.get(Collection.MEMORY, row)
        with pytest.raises(RuntimeError, match="Stale selected KB row"):
            sparse.step(step=1)
        after = bank.get(Collection.MEMORY, row)
        np.testing.assert_array_equal(after.key, prior.key)
        assert bank.moments(
            collection=Collection.MEMORY, kind="record",
            entity_id=row, field="key", layout_generation="v1",
        ) is None
        # A rejected barrier must not poison later steps with the stale leaf.
        sparse.select(keyid).sum().backward()
        assert sparse.step(step=1) == 1
    finally:
        bank.close()


def test_sparse_step_rejects_row_retired_after_selection(tmp_path):
    bank, row = _bank(tmp_path / "bank.sqlite")
    try:
        sparse = SparseRowAdam(bank)
        keyid = RowID(Collection.MEMORY, "record", row, "key", "v1")
        sparse.select(keyid).sum().backward()
        bank.tombstone_source_version("source", "v1")
        assert not bank.get(Collection.MEMORY, row).active
        with pytest.raises(RuntimeError, match="Stale selected KB row"):
            sparse.step(step=1)
        assert bank.moments(
            collection=Collection.MEMORY, kind="record",
            entity_id=row, field="key", layout_generation="v1",
        ) is None
    finally:
        bank.close()


def test_same_step_replay_does_not_advance_sparse_moments(tmp_path):
    bank, row = _bank(tmp_path / "bank.sqlite")
    try:
        sparse = SparseRowAdam(bank)
        identity = RowID(Collection.MEMORY, "record", row, "value", "v1")
        sparse.select(identity).sum().backward()
        sparse.step(step=1)
        sparse.select(identity).sum().backward()
        with pytest.raises(RuntimeError, match="not strictly increasing"):
            sparse.step(step=1)
    finally:
        bank.close()
