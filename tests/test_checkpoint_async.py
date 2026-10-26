"""Background checkpoint writes and durable object-store round trips."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import types
from pathlib import Path

import pytest
import torch

from megatoken.io import sha256_file
from megatoken.training.checkpoint import (
    CheckpointWriter,
    commit_checkpoint,
    load_checkpoint,
    resolve_checkpoint,
    snapshot_to_cpu,
    stage_checkpoint,
)
from megatoken.training.diloco_storage import CheckpointStore


def test_snapshot_is_private_and_keeps_shared_storage_shared():
    weight = torch.arange(6, dtype=torch.float32)
    state = {"model": {"w": weight}, "optimizer": {"master": weight, "moments": [weight * 2]}}
    snapshot = snapshot_to_cpu(state)
    weight.add_(100)
    assert snapshot["model"]["w"].tolist() == list(range(6))
    assert snapshot["model"]["w"] is snapshot["optimizer"]["master"]
    assert snapshot["optimizer"]["moments"][0].tolist() == [2.0 * i for i in range(6)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned device snapshot needs CUDA")
def test_device_snapshot_is_pinned_private_and_keeps_sharing():
    weight = torch.arange(6, dtype=torch.float32, device="cuda")
    snapshot = snapshot_to_cpu({"a": weight, "b": weight, "c": weight[2:]})
    weight.add_(100)
    assert snapshot["a"].device.type == "cpu" and snapshot["a"].is_pinned()
    assert snapshot["a"] is snapshot["b"]
    assert snapshot["a"].tolist() == list(range(6))
    assert snapshot["c"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_writer_overlaps_the_caller_and_publishes_atomically(tmp_path):
    root = tmp_path / "checkpoints"
    release = threading.Event()
    writer = CheckpointWriter()
    try:
        staged = stage_checkpoint(root, "P1-000000001-periodic")
        (staged / "transducer.pt").write_bytes(b"bank")

        def write():
            release.wait(timeout=30)
            commit_checkpoint(
                root,
                "P1-000000001-periodic",
                staged,
                {"model": {"w": torch.ones(3)}},
                {"global_step": 1},
                extra={
                    "transducer_checkpoint": {
                        "file": "transducer.pt",
                        "sha256": sha256_file(staged / "transducer.pt"),
                    }
                },
            )

        writer.submit(write)
        # Training continues; nothing is visible until the write commits.
        assert not (root / "latest.json").exists()
        release.set()
        writer.wait()
        payload, metadata = load_checkpoint(root)
        assert payload["model"]["w"].tolist() == [1.0, 1.0, 1.0]
        assert metadata["global_step"] == 1
        state_path = resolve_checkpoint(root) / "state.pt"
        checksum = sha256_file(state_path)
        # Optimizer restores may update a CPU tensor loaded from the private
        # mapping. That must never mutate the committed checkpoint itself.
        payload["model"]["w"].add_(7)
        restored, _ = load_checkpoint(root)
        assert restored["model"]["w"].tolist() == [1.0, 1.0, 1.0]
        assert sha256_file(state_path) == checksum
        assert (resolve_checkpoint(root) / "transducer.pt").read_bytes() == b"bank"
        assert not list(root.glob(".checkpoint-*"))
    finally:
        writer.close()


def test_writer_surfaces_a_failed_write_at_the_next_wait():
    writer = CheckpointWriter()

    def fail():
        raise OSError("disk full")

    writer.submit(fail)
    with pytest.raises(OSError, match="disk full"):
        writer.wait()
    writer.close()


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, filename, bucket, key):
        self.objects[f"{bucket}/{key}"] = Path(filename).read_bytes()

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[f"{Bucket}/{Key}"] = Body

    def get_object(self, Bucket, Key):
        return {"Body": types.SimpleNamespace(read=lambda: self.objects[f"{Bucket}/{Key}"])}

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[f"{bucket}/{key}"])

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(
            key.removeprefix(f"{Bucket}/")
            for key in self.objects
            if key.startswith(f"{Bucket}/{Prefix}")
        )
        start = int(ContinuationToken or 0)
        page = keys[start : start + 2]  # tiny pages: pruning must follow the continuation
        more = start + 2 < len(keys)
        return {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": more,
            **({"NextContinuationToken": str(start + 2)} if more else {}),
        }

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            del self.objects[f"{Bucket}/{item['Key']}"]


def test_object_store_round_trips_the_transducer_bank_with_the_weights(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "botocore.exceptions",
        types.SimpleNamespace(ClientError=type("ClientError", (Exception,), {})),
    )
    root = tmp_path / "checkpoints"
    name = "M2-000000004-periodic"
    staged = stage_checkpoint(root, name)
    bank = staged / "transducer.pt"
    bank.write_bytes(b"canonical rows")
    (staged / "transducer.manifest.json").write_text("{}")
    source = commit_checkpoint(
        root,
        name,
        staged,
        {"model": {"w": torch.zeros(2)}},
        {"global_step": 4},
        extra={"transducer_checkpoint": {"file": "transducer.pt", "sha256": sha256_file(bank)}},
    )
    store = CheckpointStore.__new__(CheckpointStore)
    store.bucket, store.prefix, store.s3 = "bucket", "run/worker-00", _FakeS3()
    pointer = store.upload(source)
    assert set(pointer["files"]) == {
        "state.pt",
        "metadata.json",
        "transducer.pt",
        "transducer.manifest.json",
    }
    # A fresh sandbox has lost its scratch volume entirely.
    shutil.rmtree(root)
    restored = store.restore(root)
    assert restored == root / name
    assert (restored / "transducer.pt").read_bytes() == b"canonical rows"
    payload, metadata = load_checkpoint(root)
    assert metadata["transducer_checkpoint"]["file"] == "transducer.pt"
    assert json.loads((root / "latest.json").read_text())["checkpoint"] == name
    # Tampered remote objects are rejected before publication.
    shutil.rmtree(root)
    store.s3.objects[f"bucket/run/worker-00/{name}/transducer.pt"] = b"tampered"
    with pytest.raises(ValueError, match="transducer.pt checksum mismatch"):
        store.restore(root)
    assert not (root / name).exists()


def _commit(root, name, step, *, pinned=False):
    staged = stage_checkpoint(root, name)
    bank = staged / "transducer.pt"
    bank.write_bytes(b"rows")
    return commit_checkpoint(
        root,
        name,
        staged,
        {"model": {"w": torch.zeros(1)}},
        {"global_step": step},
        extra={"transducer_checkpoint": {"file": "transducer.pt", "sha256": sha256_file(bank)}},
        pinned=pinned,
        keep=99,
    )


def test_object_store_keeps_only_what_local_retention_keeps(tmp_path):
    root = tmp_path / "checkpoints"
    store = CheckpointStore.__new__(CheckpointStore)
    store.bucket, store.prefix, store.s3 = "bucket", "run/worker-00", _FakeS3()
    names = ["P-000000001-periodic", "P-000000002-periodic", "P-000000003-periodic"]
    store.upload(_commit(root, "P-000000000-phase-start", 0, pinned=True), keep=2)
    for step, name in enumerate(names, start=1):
        store.upload(_commit(root, name, step), keep=2)
    remote = {key.split("/")[3] for key in store.s3.objects}
    # The two newest periodic checkpoints and the pinned one; the oldest periodic is gone.
    assert remote == {"P-000000000-phase-start", names[1], names[2], "latest.json"}
    # A checkpoint whose upload has not committed metadata yet is never pruned.
    store.s3.objects["bucket/run/worker-00/P-000000009-periodic/state.pt"] = b"partial"
    store.upload(_commit(root, "P-000000004-periodic", 4), keep=2)
    assert "bucket/run/worker-00/P-000000009-periodic/state.pt" in store.s3.objects
    assert "bucket/run/worker-00/P-000000002-periodic/state.pt" not in store.s3.objects


def test_stale_staging_directories_from_crashed_writes_are_removed(tmp_path):
    root = tmp_path / "checkpoints"
    crashed = root / ".checkpoint-crashed"
    crashed.mkdir(parents=True)
    (crashed / "state.pt").write_bytes(b"half a checkpoint")
    staged = stage_checkpoint(root, "P1-000000001-periodic")
    assert not crashed.exists() and staged.exists()


def test_the_transducer_bank_is_fsynced_before_it_is_published(tmp_path, monkeypatch):
    from megatoken.training import checkpoint

    synced = []
    real = checkpoint.os.fsync

    def capture_fsynced_file(fd):
        metadata = checkpoint.os.fstat(fd)
        synced.append((metadata.st_dev, metadata.st_ino))
        real(fd)

    monkeypatch.setattr(checkpoint.os, "fsync", capture_fsynced_file)
    staged = stage_checkpoint(tmp_path / "checkpoints", "P1-000000001-periodic")
    checkpoint.transducer_attachment(lambda: {"bank": torch.zeros(2)})(staged)
    transducer = (staged / "transducer.pt").stat()
    assert (transducer.st_dev, transducer.st_ino) in synced
