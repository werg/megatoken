"""Pack acquisition is restartable: nothing half-downloaded is ever locked or blocks a rerun."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REVISION = "a" * 40
INVENTORY = {"embeddinggemma2": {"repository": "org/model", "revision": REVISION}}


@pytest.fixture
def acquire(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "acquire_packs.py"
    spec = importlib.util.spec_from_file_location("acquire_packs_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = types.SimpleNamespace(fail_after_first_file=False, downloads=0)

    class HfApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repository, revision):
            return types.SimpleNamespace(sha=revision)

    def snapshot_download(*, repo_id, revision, local_dir, token, **patterns):
        state.downloads += 1
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}")
        (target / ".cache").mkdir()
        (target / ".cache" / "bookkeeping").write_text("machine specific")
        if state.fail_after_first_file:
            raise ConnectionError("download interrupted")
        (target / "model.safetensors").write_bytes(b"weights")

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi, hub.snapshot_download = HfApi, snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return module, state


def test_acquire_locks_every_file_with_its_checksum(acquire, tmp_path):
    module, _ = acquire
    result = module.acquire(tmp_path / "packs", INVENTORY)
    assert result["packs"] == ["embeddinggemma2"]
    lock = json.loads((tmp_path / "packs" / "lock.json").read_text())
    entry = lock["packs"]["embeddinggemma2"]
    assert sorted(entry["files"]) == [
        "embeddinggemma2/config.json",
        "embeddinggemma2/model.safetensors",
    ]
    assert not list((tmp_path / "packs").glob(".*.partial"))


def test_an_interrupted_download_leaves_no_snapshot_and_a_rerun_succeeds(acquire, tmp_path):
    module, state = acquire
    root = tmp_path / "packs"
    state.fail_after_first_file = True
    with pytest.raises(ConnectionError):
        module.acquire(root, INVENTORY)
    assert not (root / "embeddinggemma2").exists() and not (root / "lock.json").exists()
    state.fail_after_first_file = False
    module.acquire(root, INVENTORY)  # the leftover temporary directory is replaced
    assert (root / "embeddinggemma2" / "model.safetensors").read_bytes() == b"weights"
    assert not list(root.glob(".*.partial"))


def test_a_snapshot_the_lock_does_not_list_is_replaced_but_a_locked_one_is_immutable(
    acquire, tmp_path
):
    module, state = acquire
    root = tmp_path / "packs"
    orphan = root / "embeddinggemma2"
    orphan.mkdir(parents=True)
    (orphan / "stale.bin").write_bytes(b"from a run that died before locking")
    module.acquire(root, INVENTORY)
    assert not (orphan / "stale.bin").exists() and state.downloads == 1
    with pytest.raises(FileExistsError, match="immutable"):
        module.acquire(root, INVENTORY)
    assert state.downloads == 1
    # The lock is written atomically: no temporary file survives.
    assert [p.name for p in root.iterdir() if p.name.startswith(".lock")] == []
