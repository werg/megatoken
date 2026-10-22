"""Export a consistent *stopped-coordinator* model for standard evaluation.

A global DiLoCo synchronizer owns outer momentum, not the independent inner
optimizer states. Exports are evaluation/inference snapshots and the seed of
the next phase's coordinator (with their global FP32 masters), NOT restartable
local-training checkpoints. Stop the coordinator before exporting: stopping saves its bank.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from megatoken.training.checkpoint import save_checkpoint, transducer_attachment
from megatoken.training.diloco_protocol import Fragment, load_tensors, training_identity
from megatoken.training.diloco_server import _metadata


def export_coordinator(root: str | Path, output: str | Path, config: object) -> dict:
    root = Path(root)
    from megatoken.io import content_hash, strict_json_loads

    manifest = strict_json_loads((root / "manifest.json").read_bytes())
    fingerprint = manifest.pop("fingerprint")
    if content_hash(manifest) != fingerprint:
        raise ValueError("Coordinator manifest checksum mismatch")
    if manifest["train_identity"] != training_identity(config):
        raise ValueError("Export config differs from the coordinator's training objective")
    snapshot = torch.load(root / "bootstrap.pt", weights_only=True, map_location="cpu")["model"]
    # Preserve buffers from the initialization, since this outer optimizer only
    # synchronizes trainable parameter masters (no moving batch-norm statistics).
    params = {
        part["name"]: snapshot[part["name"]].float().clone()
        for fragment in manifest["fragments"]
        for part in fragment["slices"]
    }
    versions = []
    for spec in manifest["fragments"]:
        fragment = Fragment.from_dict(spec)
        path = root / "fragments" / f"{fragment.index:05d}.safetensors"
        versions.append(int(_metadata(path)["version"]))
        data = load_tensors(path)["theta"]
        if data.dtype != torch.float32 or data.numel() != fragment.length:
            raise ValueError("Corrupt coordinator fragment during export")
        cursor = 0
        for part in fragment.slices:
            target = params[part.name].reshape(-1)
            target[part.offset : part.offset + part.length].copy_(
                data[cursor : cursor + part.length]
            )
            cursor += part.length
    for name, value in params.items():
        snapshot[name] = value.to(snapshot[name].dtype)
    # The global FP32 masters seed the next phase's coordinator (diloco init --resume).
    from megatoken.io import sha256_file
    from megatoken.training.transducer_sync import TransducerHost

    host = TransducerHost(
        root,
        manifest["transducer"],
        workers=manifest["workers"],
        min_quorum=manifest["min_quorum"],
        grace_seconds=manifest["grace_seconds"],
    )

    name = f"outer-{max(versions):09d}-{int(time.time())}"
    directory = save_checkpoint(
        output,
        name,
        {
            "model": snapshot,
            "model_config": manifest["model_config"],
            "train_config": asdict(config),
            "codec_fingerprint": manifest["codec_fingerprint"],
            "packs_fingerprint": sha256_file(config.packs) if config.packs else None,
            "export_only": True,
            "master_weights": params,
            "outer_fragment_versions": versions,
            "global_step": max(versions),
        },
        {
            "global_step": max(versions),
            "phase": config.phase,
            "phase_consumed": -1,
            "export_only": True,
            "run_id": manifest["run_id"],
            "outer_fragment_versions": versions,
        },
        pinned=True,
        attachments=transducer_attachment(host.checkpoint_state),
    )
    return {"checkpoint": str(directory), "versions": versions, "export_only": True}
