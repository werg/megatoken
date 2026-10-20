"""DiLoCo free-KB row-sync: coordinator-owned rows, sparse pseudo-gradient merges."""
import json
import sqlite3

import numpy as np
import pytest
import torch

from megatoken.memory.contracts import Collection, Footprint, SemanticSpace, SourceRange
from megatoken.memory.growth import GrowthPolicy
from megatoken.memory.optimizer import RowID, SparseRowAdam
from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe
from megatoken.model.config import ModelConfig
from megatoken.training.diloco_kb import (
    KBCoordinator,
    KBSync,
    initialize_kb,
    install_snapshot,
    load_kb_update,
    read_field,
    row_key,
    save_kb_update,
)
from megatoken.training.diloco_protocol import merge_pseudogradients


class Conflict(ValueError):
    pass


def _recipe():
    space = SemanticSpace(
        "shared", "r1", "local", "pinned-teacher", 8,
        "document", "query", "ordered-source",
    )
    return RetrievalRecipe(
        schema="megatoken-retrieval-recipe-v1",
        space=space,
        model=ModelConfig.tiny(semantic_dim=8, semantic_value_dim=4, max_kb_positions=2),
        growth=GrowthPolicy(value_dim=4, seed=9, max_points=10, max_pairs=5),
        loss_weights={"embedding": 1., "query_ranking": 0.5, "source_value": 0.},
        phase="free_kb_pretraining",
        max_selected_source=2, max_selected_memory=2,
        max_kb_positions=2, max_readout_groups=2,
    )


def _footprint(source="source-a"):
    return Footprint((SourceRange(source, "v1", 0, 20, "sha", "team", family="f"),))


def _setup(tmp_path, *, quorum=2, workers=2, staleness=1, growth_budget=0):
    recipe = _recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe.to_dict()))
    bank_path = tmp_path / "initial.sqlite"
    with open_bank_for_recipe(recipe, bank_path) as bank:
        rows = [
            bank.insert_memory(
                scope="team", footprint=_footprint(source), assignment=f"{source}/0",
                key=np.eye(8)[i], value=np.full(4, 0.5, dtype=np.float32), birth_step=0,
            )
            for i, source in enumerate(("source-a", "source-b"))
        ]
        bank.put_growth_state("readiness", {"coherence": True, "embeddings": True})
        database_id = bank.database_id
    root = tmp_path / "coordinator"
    root.mkdir()
    settings = initialize_kb(
        root, recipe_path=str(recipe_path), bank_path=str(bank_path),
        centroids_path=None, centroids_sha256=None, nprobe=8,
        scopes=("team",), growth_budget=growth_budget,
    )
    assert settings["database_id"] == database_id
    coordinator = KBCoordinator(
        root, settings, outer_lr=0.7, outer_momentum=0.9, max_staleness=staleness,
        min_quorum=quorum, workers=workers, grace_seconds=0,
    )
    return recipe, coordinator, rows


def _submit(coordinator, tmp_path, worker, rows, *, sequence=1, base=0, tokens=10,
            steps=2, demand=()):
    path = tmp_path / f"update-{worker}-{sequence}.safetensors"
    save_kb_update(path, rows, list(demand), wire_dtype="fp32")
    return coordinator.submit(worker, sequence, base, steps, tokens, path, Conflict)


def test_quorum_merge_applies_token_weighted_outer_nesterov_per_row(tmp_path):
    _, coordinator, (a, b) = _setup(tmp_path)
    try:
        value_a = RowID(Collection.MEMORY, "record", a, "value", "v1")
        value_b = RowID(Collection.MEMORY, "record", b, "value", "v1")
        delta0 = np.full(4, 0.2, dtype=np.float32)
        delta1 = np.full(4, -0.1, dtype=np.float32)
        only1 = np.full(4, 0.3, dtype=np.float32)
        assert _submit(coordinator, tmp_path, 0, {row_key(value_a): delta0},
                       tokens=10)["status"] == "queued"
        assert coordinator.version() == 0  # Below quorum: nothing merges.
        result = _submit(coordinator, tmp_path, 1,
                         {row_key(value_a): delta1, row_key(value_b): only1}, tokens=30)
        assert result["version"] == 1
        merged_a = merge_pseudogradients(
            [torch.from_numpy(delta0), torch.from_numpy(delta1)], [10, 30], [2, 2], [0, 0],
        )
        # First outer step: velocity = g, theta' = theta - lr * (g + mu * g).
        expected_a = 0.5 - 0.7 * (merged_a.numpy() * 1.9)
        expected_b = 0.5 - 0.7 * (only1 * 1.9)  # Only its single contributor votes.
        np.testing.assert_allclose(read_field(coordinator.bank, value_a), expected_a, rtol=1e-6)
        np.testing.assert_allclose(read_field(coordinator.bank, value_b), expected_b, rtol=1e-6)
        # Replays of committed sequences are idempotent; stale bases are refused.
        assert _submit(coordinator, tmp_path, 0, {row_key(value_a): delta0})[
            "status"] == "already_merged"
        coordinator.max_staleness = 0
        with pytest.raises(Conflict, match="Stale KB baseline"):
            _submit(coordinator, tmp_path, 0, {row_key(value_a): delta0}, sequence=2, base=0)
    finally:
        coordinator.close()


def test_published_snapshot_is_versioned_and_stripped(tmp_path):
    _, coordinator, (a, _) = _setup(tmp_path, quorum=1, workers=1)
    try:
        key = row_key(RowID(Collection.MEMORY, "record", a, "value", "v1"))
        _submit(coordinator, tmp_path, 0, {key: np.full(4, 0.1, dtype=np.float32)})
        version, path = coordinator.published()
        assert version == 1
        with sqlite3.connect(path) as db:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            assert not {"diloco_kb_outer", "diloco_kb_meta"} & tables
            assert db.execute("SELECT COUNT(*) FROM optimizer_rows").fetchone()[0] == 0
        assert len(list(path.parent.glob("public-*.sqlite"))) == 1
    finally:
        coordinator.close()


def test_retired_rows_are_skipped_and_demand_is_recorded(tmp_path):
    _, coordinator, (a, b) = _setup(tmp_path, quorum=1, workers=1)
    try:
        coordinator.bank.tombstone_source_version("source-a", "v1")
        retired = RowID(Collection.MEMORY, "record", a, "value", "v1")
        live = RowID(Collection.MEMORY, "record", b, "strength", "v1")
        before = coordinator.bank.get(Collection.MEMORY, a).value.copy()
        _submit(
            coordinator, tmp_path, 0,
            {row_key(retired): np.ones(4, dtype=np.float32),
             row_key(live): np.array([10.], dtype=np.float32)},
            demand=[("episode/event", "0:1", "family", 3)],
        )
        assert coordinator.version() == 1
        np.testing.assert_array_equal(coordinator.bank.get(Collection.MEMORY, a).value, before)
        # Overshooting outer steps cannot make a mass negative.
        assert coordinator.bank.get(Collection.MEMORY, b).strength == 0
        assert coordinator.bank.db.execute(
            "SELECT COUNT(*) FROM demand_events WHERE event_id='episode/event'"
        ).fetchone()[0] == 1
    finally:
        coordinator.close()


def test_restart_resumes_committed_version_and_sequence_ledger(tmp_path):
    _, coordinator, (a, _) = _setup(tmp_path, quorum=1, workers=1)
    key = row_key(RowID(Collection.MEMORY, "record", a, "value", "v1"))
    _submit(coordinator, tmp_path, 0, {key: np.full(4, 0.1, dtype=np.float32)})
    settings = coordinator.settings
    coordinator.close()
    restarted = KBCoordinator(
        tmp_path / "coordinator", settings, outer_lr=0.7, outer_momentum=0.9,
        max_staleness=1, min_quorum=1, workers=1, grace_seconds=0,
    )
    try:
        assert restarted.version() == 1
        assert _submit(restarted, tmp_path, 0, {key: np.zeros(4, dtype=np.float32) + 1})[
            "status"] == "already_merged"
    finally:
        restarted.close()


def test_learner_deltas_install_and_keep_local_moments(tmp_path):
    recipe, coordinator, (a, _) = _setup(tmp_path, quorum=1, workers=1)
    try:
        _, public = coordinator.published()
        local = tmp_path / "learner.sqlite"
        local.write_bytes(public.read_bytes())
        sync = KBSync(version=0, wire_dtype="fp32", local_steps=1)
        bank = open_bank_for_recipe(recipe, local)
        sparse = SparseRowAdam(bank, lr=0.1)
        identity = RowID(Collection.MEMORY, "record", a, "value", "v1")
        sparse.select(identity).sum().backward()
        sync.capture(sparse.leaves)
        sparse.step(step=1)
        rows = sync.deltas(bank)
        np.testing.assert_allclose(
            rows[row_key(identity)],
            0.5 - read_field(bank, identity),
        )
        bank.close()
        update = tmp_path / "learner-update.safetensors"
        save_kb_update(update, rows, [], wire_dtype="bf16")
        assert set(load_kb_update(update).rows) == {row_key(identity)}
        coordinator.submit(0, 1, 0, 1, 5, update, Conflict)
        version, published = coordinator.published()
        downloaded = tmp_path / "downloaded.sqlite"
        downloaded.write_bytes(published.read_bytes())
        install_snapshot(downloaded, local)
        sync.installed(version)
        with open_bank_for_recipe(recipe, local) as installed:
            np.testing.assert_allclose(
                read_field(installed, identity), read_field(coordinator.bank, identity),
            )
            # Learner-local sparse Adam moments survive the canonical install.
            assert installed.moments(
                collection=Collection.MEMORY, kind="record", entity_id=a,
                field="value", layout_generation="v1",
            )[2] == 1
        assert sync.version == 1 and not sync.baselines
    finally:
        coordinator.close()


def _synced_rows(path):
    from megatoken.training.diloco_kb import SYNCED_TABLES

    with sqlite3.connect(path) as db:
        return {
            table: sorted(map(tuple, db.execute(f"SELECT * FROM {table}").fetchall()))
            for table in SYNCED_TABLES
        }


def test_incremental_changes_match_full_snapshot_and_revert_local_training(tmp_path):
    from megatoken.training.diloco_kb import KBChangesExpired, apply_changes

    recipe, coordinator, (a, b) = _setup(tmp_path, quorum=1, workers=1, growth_budget=2)
    try:
        version0, public = coordinator.published()
        local = tmp_path / "learner.sqlite"
        local.write_bytes(public.read_bytes())
        # Learner trains row b locally; the coordinator never receives it.
        bank = open_bank_for_recipe(recipe, local)
        sync = KBSync(version=version0, wire_dtype="fp32", local_steps=1)
        sparse = SparseRowAdam(bank, lr=0.5)
        trained = RowID(Collection.MEMORY, "record", b, "value", "v1")
        sparse.select(trained).sum().backward()
        sync.capture(sparse.leaves)
        sparse.step(step=1)
        assert not np.allclose(read_field(bank, trained), 0.5)
        # Canonical changes: a row update, demand+growth, then a retirement.
        key = row_key(RowID(Collection.MEMORY, "record", a, "value", "v1"))
        _submit(coordinator, tmp_path, 0, {key: np.full(4, 0.1, dtype=np.float32)},
                demand=[("episode/0", "0:1", "f", 1)])
        coordinator.bank.tombstone_source_version("source-a", "v1")
        _submit(coordinator, tmp_path, 0, {key: np.full(4, 0.1, dtype=np.float32)},
                sequence=2, base=1)
        assert coordinator.version() == 2
        export = tmp_path / "changes.sqlite"
        records, relations = sync.trained_rows()
        assert records == (b,) and relations == ()
        assert coordinator.export_changes(
            version0, export, records=records, relations=relations,
        ) == 2
        assert coordinator.export_changes(2, tmp_path / "none.sqlite") is None
        assert apply_changes(bank, export) == 2
        bank.close()
        _, snapshot = coordinator.published()
        assert _synced_rows(local) == _synced_rows(snapshot)
        # Retained history bounds incremental pulls; older bases need the snapshot.
        coordinator.retained_versions = 1
        _submit(coordinator, tmp_path, 0,
                {row_key(RowID(Collection.MEMORY, "record", b, "value", "v1")):
                 np.full(4, 0.1, dtype=np.float32)}, sequence=3, base=2)
        with pytest.raises(KBChangesExpired):
            coordinator.export_changes(1, tmp_path / "expired.sqlite")
        assert coordinator.export_changes(2, tmp_path / "recent.sqlite") == 3
    finally:
        coordinator.close()


def test_snapshot_has_no_coordinator_triggers(tmp_path):
    recipe, coordinator, (a, _) = _setup(tmp_path, quorum=1, workers=1)
    try:
        _, public = coordinator.published()
        with sqlite3.connect(public) as db:
            assert db.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
            ).fetchone()[0] == 0
        local = tmp_path / "learner.sqlite"
        local.write_bytes(public.read_bytes())
        with open_bank_for_recipe(recipe, local) as bank:
            # Learner writes must not reference coordinator-only tables.
            bank.update_record(Collection.MEMORY, a, strength=0.3)
    finally:
        coordinator.close()
