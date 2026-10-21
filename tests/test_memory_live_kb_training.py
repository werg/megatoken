"""Live free-KB phases through train(): synchronous sparse updates and DiLoCo row sync."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import socket
import sqlite3
import time
import urllib.error
import urllib.request
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from retrieval_fixture import SCOPE, live_kb_train_config

from megatoken.memory.contracts import Collection
from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import DecoupledConfig
from megatoken.training.diloco_server import serve_coordinator
from megatoken.training.engine import train


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


def _copy_live(path, destination):
    """Consistent copy of a bank another process holds open (WAL-aware)."""
    source = sqlite3.connect(str(path), timeout=10)
    target = sqlite3.connect(str(destination))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return destination


def _changed(before, after):
    changed = 0
    for identity, (key, value, strength) in after.items():
        old = before.get(identity)
        if old is None:
            continue
        same = (
            np.array_equal(old[0], key)
            and (
                (old[1] is None and value is None)
                or (old[1] is not None and value is not None and np.array_equal(old[1], value))
            )
            and old[2] == strength
        )
        changed += not same
    return changed


def _metrics(run):
    return [
        json.loads(line)
        for line in (Path(run) / "metrics.jsonl").read_text().splitlines()
        if '"event": "train"' in line
    ]


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
    run_bank = Path(config.output) / "kb.sqlite"
    live, moments_live, demand_live = _rows(run_bank, recipe)
    # The configured bank is an immutable input; the run trains its own copy.
    untouched, moments_input, demand_input = _rows(config.retrieval_bank, recipe)
    assert untouched.keys() == initial.keys() and moments_input == demand_input == 0
    snap, _, _ = _rows(checkpoint / "kb.sqlite", recipe)
    assert demand_live > 0 and moments_live > 0
    assert len(mid) > len(initial) and set(mid) <= set(live)
    assert _changed(mid, live) > 0
    assert all(np.array_equal(live[k][0], snap[k][0]) for k in live)
    # Exact resume continues from the published snapshot and keeps training rows.
    resumed = train(dataclasses.replace(config, max_steps=1), resume=first["checkpoint"])
    assert resumed["global_step"] == 3
    after, _, _ = _rows(run_bank, recipe)
    assert _changed(live, after) > 0


def _get(url, token, path):
    request = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_diloco_http_free_kb_row_sync_train_and_resume(tmp_path, monkeypatch, recipe_of):
    torch.set_num_threads(1)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    token = "test-diloco-token-" + "b" * 32
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    settings = DecoupledConfig(
        coordinator_url=url,
        run_id="kb-http-v1",
        workers=1,
        worker_id=0,
        local_steps=1,
        poll_steps=1,
        heartbeat_steps=1,
        io_workers=2,
        timeout_seconds=30,
        max_offline_steps=20,
    )
    config = live_kb_train_config(
        tmp_path,
        phase="M1",
        max_steps=6,
        checkpoint_every=1,
        inner_optimizer="muon",
        decoupled=settings,
    )
    recipe = recipe_of(config)
    initial, _, _ = _rows(config.retrieval_bank, recipe)
    # Initialize exactly as `megatoken diloco init` does, from a config file.
    from megatoken.cli import _diloco_initialize

    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(dataclasses.asdict(config)))
    root = tmp_path / "coordinator"
    _diloco_initialize(
        Namespace(
            config=str(config_path),
            root=str(root),
            resume=None,
            quorum=1,
            fragment_mib=1,
            outer_lr=0.7,
            outer_momentum=0.9,
            grace_seconds=0.0,
            max_staleness=2,
        )
    )
    server = multiprocessing.get_context("spawn").Process(
        target=serve_coordinator,
        args=(str(root), "127.0.0.1", port, token),
    )
    server.start()
    try:
        for _ in range(200):
            try:
                status = _get(url, token, "/v1/status")
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise AssertionError("Coordinator failed to start")
        assert status["kb_version"] == 0
        first = train(config)
        assert first["global_step"] == 6
        state, _ = load_checkpoint(first["checkpoint"])
        assert state["diloco"]["kb"]["version"] >= 1
        status = _get(url, token, "/v1/status")
        assert status["kb_version"] >= 1
        # The worker never grew locally: all ids come from the coordinator bank.
        canonical = _copy_live(root / "kb" / "canonical.sqlite", tmp_path / "canonical-copy.sqlite")
        rows, _, demand = _rows(canonical, recipe)
        assert demand > 0
        # Only the coordinator grows: memory rows exist beyond the imported sources.
        assert len(rows) > len(initial) and any(k[0] == "memory" for k in rows)
        assert set(initial) <= set(rows)
        # Learner deltas were merged into canonical rows: outer momentum is nonzero.
        with open_bank_for_recipe(recipe, canonical, readonly=True) as bank:
            momenta = bank.db.execute("SELECT row, momentum FROM diloco_kb_outer").fetchall()
        assert momenta and any(np.frombuffer(m, dtype="<f4").any() for _, m in momenta)
        assert all(r.startswith("memory|record|") for r, _ in momenta)
        assert status["kb_version"] >= 2
        learner, learner_moments, _ = _rows(
            Path(config.output) / "worker-00" / "diloco" / "kb.sqlite", recipe
        )
        assert set(learner) <= set(rows)
        assert any(k[0] == "memory" for k in learner)
        assert learner_moments > 0  # canonical installs keep the learner's own moments
        # After bootstrap, learners pull incremental change sets: the full
        # snapshot is built lazily and only for version 0 here.
        assert [p.name for p in (root / "kb").glob("public-*.sqlite")] == [
            "public-000000000.sqlite"
        ]
        assert (Path(first["checkpoint"]) / "kb.sqlite").is_file()
        resumed = train(dataclasses.replace(config, max_steps=1), resume=first["checkpoint"])
        assert resumed["global_step"] == 7
        state2, _ = load_checkpoint(resumed["checkpoint"])
        assert state2["diloco"]["local_steps"] == 7
        assert state2["diloco"]["kb"]["version"] >= state["diloco"]["kb"]["version"]
        assert state2["diloco"]["kb"]["sequence"] >= state["diloco"]["kb"]["sequence"]
    finally:
        server.terminate()
        server.join(timeout=10)
        if server.is_alive():
            server.kill()
            server.join()
