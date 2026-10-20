"""Offline, byte-pinned native donor loading for the existing model owner."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch

from .donors import PocketFacetsStudent, SanaFacetsStudent


def read_locked_manifest(filename: str):
    """No implicit Hub downloads, mutable tags, unverified file substitution."""
    path = Path(filename).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Pack manifest must declare version 1")
    packs = value.get("packs")
    if not isinstance(packs, dict) or not packs or set(packs) - {
        "pocket", "sana", "sana_video", "whistle", "minicpm"
    }:
        raise ValueError("Supported residents: Pocket, Sana, Sana-Video")
    for name, item in packs.items():
        if set(item) - {"snapshot", "config", "checkpoint", "weights", "files", "revision", "repository"}:
            raise ValueError("Unexpected pretrained pack manifest fields")
        if not isinstance(item.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", item["revision"]) or not isinstance(item.get("files"), dict) or not item["files"]:
            raise ValueError("Each pack requires a revision and file checksums")
        if name == "pocket":
            if not item.get("config"):
                raise ValueError("Pocket requires its YAML config and exact Mimi weights")
        elif name in {"sana", "sana_video", "minicpm"} and not item.get("snapshot"):
            raise ValueError("Pack requires a locally pinned snapshot")
        elif name == "whistle" and not item.get("weights"):
            raise ValueError("Whistle requires a locally pinned .cact file")
        verified = set()
        for relative, expected in item["files"].items():
            if (not isinstance(relative,str) or Path(relative).is_absolute()
                    or ".." in Path(relative).parts or len(expected) != 64
                    or any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError("Invalid relative checksum manifest entry")
            filename = (path.parent / relative).resolve()
            if not filename.is_relative_to(path.parent):
                raise ValueError("Pack artifact escapes manifest directory")
            if not filename.is_file():
                raise FileNotFoundError(str(filename))
            hasher = hashlib.sha256()
            with filename.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected:
                raise ValueError(f"Pretrained artifact checksum mismatch: {relative}")
            verified.add(filename)
        for required in ("config", "checkpoint", "weights"):
            if item.get(required):
                target = (path.parent / item[required]).resolve()
                if not target.is_file() or target not in verified:
                    raise ValueError(f"Unhashed pack {required} file")
        if item.get("snapshot"):
            snapshot = (path.parent / item["snapshot"]).resolve()
            if not snapshot.is_relative_to(path.parent) or not snapshot.is_dir():
                raise ValueError("Snapshot must be inside the manifest directory")
            if name in {"sana", "sana_video"} and (snapshot / "model_index.json") not in verified:
                raise ValueError("Diffusers snapshot requires a hashed model_index.json")
            for asset in snapshot.rglob("*"):
                if asset.is_symlink() or (asset.is_file() and asset.resolve() not in verified):
                    raise ValueError(f"Unhashed or symlinked snapshot asset: {asset}")
        if name == "pocket":
            import yaml
            config_file = (path.parent / item["config"]).resolve()
            config = yaml.safe_load(config_file.read_text("utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Pocket config must be a YAML mapping")
            def walk(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key in {"weights_path", "weights_path_without_voice_cloning", "tokenizer_path"} and isinstance(value, str):
                            if value.startswith(("hf://", "https://", "http://")):
                                raise ValueError(f"Remote Pocket resource in {key}")
                            resource = Path(value)
                            if not resource.is_absolute():
                                resource = config_file.parent / resource
                            if resource.resolve() not in verified:
                                raise ValueError(f"Unhashed Pocket resource {key}")
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)
            walk(config)
    return path.parent, packs


def install_pretrained_packs(model, manifest: str, *, device, dtype=torch.bfloat16):
    """All packs become registered modules before optimizer and checkpoint restore."""
    base, packs = read_locked_manifest(manifest)
    resident = model.media_realizers
    for name, item in packs.items():
        if name not in {"pocket", "sana", "sana_video"}:
            continue
        if name == "pocket":
            student = PocketFacetsStudent.from_checkpoint(
                config=str(base / item["config"]),
                checkpoint=str(base / item["checkpoint"]) if item.get("checkpoint") else None,
                facet_width=model.config.facet_dim,
            )
        else:
            student = SanaFacetsStudent.from_checkpoint(
                snapshot=str(base / item["snapshot"]),
                modality="image" if name == "sana" else "video",
                facet_width=model.config.facet_dim, torch_dtype=dtype,
            )
        student = student.to(device)
        student.eval()
        resident.register_pretrained(name, student)
    return model


def write_manifest_template():
    return {
        "version": 1,
        "packs": {
            "pocket": {
                "revision": "REPLACE_WITH_PINNED_COMMIT",
                "config": "pocket/pocket.yaml",
                "files": {
                    "pocket/pocket.yaml": "REPLACE_WITH_SHA256",
                    "pocket/flow_lm.safetensors": "REPLACE_WITH_SHA256",
                    "pocket/mimi.safetensors": "REPLACE_WITH_SHA256",
                },
            },
            "sana": {
                "revision": "REPLACE_WITH_PINNED_COMMIT",
                "snapshot": "sana",
                "files": {"sana/model_index.json": "REPLACE_WITH_SHA256"},
            },
            "sana_video": {
                "revision": "REPLACE_WITH_PINNED_COMMIT",
                "snapshot": "sana_video",
                "files": {"sana_video/model_index.json": "REPLACE_WITH_SHA256"},
            },
        },
    }
