"""Live free-KB phases through train(): synchronous sparse updates and DiLoCo row sync."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import socket
import time
import urllib.error
import urllib.request
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from megatoken.memory.contracts import Collection
from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import DecoupledConfig
from megatoken.training.diloco_server import serve_coordinator
from megatoken.training.engine import train

from retrieval_fixture import SCOPE, live_kb_train_config


def _rows(path, recipe):
    """{(collection, id): (key, value, strength)} of every active row, plus moment count."""
    out = {}
    with open_bank_for_recipe(recipe, path, readonly=True) as bank:
        for collection in (Collection.SOURCE, Collection.MEMORY):
            for row in bank.scan(collection, SCOPE):
                out[(collection.value, row.id)] = (
                    np.asarray(row.key, dtype=np.float32).copy(),
                    None if row.value is None else np.asarray(row.value, dtype=np.float32).copy(),
                    row.strength,
                )
        moments = bank.db.execute("SELECT COUNT(*) FROM optimizer_rows").fetchone()[0]
        demand = bank.db.execute("SELECT COUNT(*) FROM demand_events").fetchone()[0]
    return out, moments, demand


def _changed(before, after):
    changed = 0
    for identity, (key, value, strength) in after.items():
        old = before.get(identity)
        if old is None:
            continue
        same = np.array_equal(old[0], key) and (
            (old[1] is None and value is None)
            or (old[1] is not None and value is not None and np.array_equal(old[1], value))
        ) and old[2] == strength
        changed += not same
    return changed


def _metrics(run):
    return [json.loads(line) for line in (Path(run) / "metrics.jsonl").read_text().splitlines()
            if '"event": "train"' in line]


@pytest.fixture
def recipe_of():
    return lambda cfg: RetrievalRecipe.load(cfg.retrieval_memory_recipe)


@pytest.mark.parametrize("phase", ["M1", "M2"])
def test_synchronous_live_kb_train_updates_rows_and_resumes(tmp_path, phase, recipe_of):
    torch.set_num_threads(1)
    config = live_kb_train_config(tmp_path, phase=phase, checkpoint_every=1)
    recipe = recipe_of(config)
    initial, moments_initial, demand_initial = _rows(config.retrieval_bank, recipe)
    assert moments_initial == 0 and demand_initial == 0
    first = train(config)
    assert first["global_step"] == 2
    steps = _metrics(config.output)
    assert [m["global_step"] for m in steps] == [1, 2]
    assert sum(m["memory_sparse_updated"] for m in steps) > 0
    assert sum(m["memory_demand_published"] for m in steps) > 0
    assert sum(m["memory_grown"] for m in steps) > 0
    checkpoint = Path(first["checkpoint"])
    assert (checkpoint / "kb.sqlite").is_file()
    assert (checkpoint / "kb.sqlite.manifest.json").is_file()
    payload, _ = load_checkpoint(first["checkpoint"])
    assert payload["global_step"] == 2
    manifest = json.loads((checkpoint / "kb.sqlite.manifest.json").read_text())
    assert manifest["optimizer_step"] == 2
    # Step 1 grew memory rows; step 2 trained them: values and moments moved.
    step1 = next(p for p in checkpoint.parent.iterdir() if "000000001" in p.name)
    mid, moments_mid, _ = _rows(step1 / "kb.sqlite", recipe)
    live, moments_live, demand_live = _rows(config.retrieval_bank, recipe)
    snap, _, _ = _rows(checkpoint / "kb.sqlite", recipe)
    assert demand_live > 0 and moments_live > 0
    assert len(mid) > len(initial) and set(mid) <= set(live)
    assert _changed(mid, live) > 0
    assert all(np.array_equal(live[k][0], snap[k][0]) for k in live)
    # Exact resume continues from the published snapshot and keeps training rows.
    resumed = train(dataclasses.replace(config, max_steps=1), resume=first["checkpoint"])
    assert resumed["global_step"] == 3
    after, _, _ = _rows(config.retrieval_bank, recipe)
    assert _changed(live, after) > 0
