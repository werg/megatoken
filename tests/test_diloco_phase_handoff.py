"""A DiLoCo phase transition: a new coordinator seeded from the previous export."""

from __future__ import annotations

import dataclasses
import json
from argparse import Namespace

import pytest
import torch
from training_fixture import tiny_train_config

from megatoken.cli import _diloco_initialize
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import DecoupledConfig
from megatoken.training.diloco_export import export_coordinator
from megatoken.training.diloco_server import Coordinator
from megatoken.training.transducer_sync import TransducerHost
from tests.media_support import install_fake_realizers


def _initialize(tmp_path, config, root, resume=None):
    path = tmp_path / f"{root}.json"
    path.write_text(json.dumps(dataclasses.asdict(config)))
    _diloco_initialize(
        Namespace(
            config=str(path),
            root=str(tmp_path / root),
            resume=resume,
            quorum=1,
            fragment_mib=1,
            outer_lr=0.7,
            outer_momentum=0.9,
            grace_seconds=0.0,
            max_staleness=2,
        )
    )
    return tmp_path / root


def _host(root):
    manifest = json.loads((root / "manifest.json").read_text())
    return TransducerHost(root, manifest["transducer"], workers=2, min_quorum=1, grace_seconds=0.0)


def _export(tmp_path, config, root, name):
    result = export_coordinator(root, tmp_path / name, config)
    return result["checkpoint"], load_checkpoint(result["checkpoint"], map_location="cpu")[0]


def test_media_phase_coordinator_starts_from_the_text_phase_export(tmp_path, monkeypatch):
    def settings(run_id):
        return DecoupledConfig(coordinator_url="http://127.0.0.1:9", run_id=run_id, workers=2)

    text = tiny_train_config(tmp_path, output=str(tmp_path / "runs"), decoupled=settings("text"))
    text_root = _initialize(tmp_path, text, "text-coordinator")
    # Give the text phase's canonical bank some learned rows, as a running coordinator would.
    coordinator = Coordinator(text_root)
    host = coordinator.transducer
    g = host.bank.geometry
    learned = host.bank.add_edges(
        departure=torch.randn(5, g.location_dim),
        key=torch.randn(5, g.query_dim),
        value=torch.randn(5, g.value_dim),
        destination=torch.randn(5, g.location_dim),
        bias=torch.zeros(5),
        write_mass=torch.ones(5),
        step=0,
    )
    coordinator.close()
    checkpoint, text_export = _export(tmp_path, text, text_root, "text-export")
    assert text_export["export_only"] and text_export["packs_fingerprint"] is None
    assert all(value.dtype == torch.float32 for value in text_export["master_weights"].values())

    # The media phase installs realizers (faked: no pretrained weights in tests).
    monkeypatch.setattr(
        "megatoken.omni.loader.install_realizers",
        lambda model, manifest, *, device: install_fake_realizers(model),
    )
    lock = tmp_path / "lock.json"
    lock.write_text('{"version": 1}')
    media = dataclasses.replace(text, packs=str(lock), decoupled=settings("media"))
    media_root = _initialize(tmp_path, media, "media-coordinator", resume=checkpoint)
    media_checkpoint, media_export = _export(tmp_path, media, media_root, "media-export")
    # The new phase's coordinator starts from the previous phase's bank.
    assert _host(media_root).bank.edge_ids().tolist() == learned.tolist()

    old, new = text_export["master_weights"], media_export["master_weights"]
    added = set(new) - set(old)
    assert set(old) <= set(new)
    assert added and all(name.startswith("media_realizers.packs.") for name in added)
    assert all(".condition." in name for name in added)  # frozen donors never become masters
    for name, value in old.items():
        torch.testing.assert_close(new[name], value, rtol=0, atol=0)
    assert media_export["packs_fingerprint"] is not None

    # A phase may add realizers but never swap them or drop them.
    swapped = tmp_path / "other-lock.json"
    swapped.write_text('{"version": 1, "other": true}')
    with pytest.raises(ValueError, match="pack manifest differs"):
        _initialize(
            tmp_path,
            dataclasses.replace(media, packs=str(swapped), decoupled=settings("swap")),
            "swap-coordinator",
            resume=media_checkpoint,
        )
    with pytest.raises(ValueError, match="pack manifest differs"):
        _initialize(
            tmp_path,
            dataclasses.replace(text, decoupled=settings("drop")),
            "drop-coordinator",
            resume=media_checkpoint,
        )
