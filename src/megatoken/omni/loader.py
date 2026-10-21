"""Locked, byte-pinned pack manifests: realizer installation and teacher verification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .realizers import PocketRealizer, SanaRealizer


def snapshot_files(directory: Path):
    """Every entry of a snapshot that is part of the model.

    Hub download bookkeeping under ``.cache/`` differs per machine and download,
    so it is neither locked nor allowed to bind an identity.
    """
    for entry in sorted(directory.rglob("*")):
        if ".cache" not in entry.relative_to(directory).parts:
            yield entry


def read_locked_manifest(filename: str):
    """No implicit Hub downloads, mutable tags, unverified file substitution."""
    path = Path(filename).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Pack manifest must declare version 1")
    packs = value.get("packs")
    if (
        not isinstance(packs, dict)
        or not packs
        or set(packs) - {"pocket", "sana", "sana_video", "embeddinggemma2"}
    ):
        raise ValueError("Supported packs: Pocket, Sana, Sana-Video, EmbeddingGemma2")
    for name, item in packs.items():
        if set(item) - {
            "snapshot",
            "config",
            "checkpoint",
            "files",
            "revision",
            "repository",
        }:
            raise ValueError("Unexpected pretrained pack manifest fields")
        if (
            not isinstance(item.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", item["revision"])
            or not isinstance(item.get("files"), dict)
            or not item["files"]
        ):
            raise ValueError("Each pack requires a revision and file checksums")
        if name == "pocket":
            if not item.get("config"):
                raise ValueError("Pocket requires its YAML config and exact Mimi weights")
        elif not item.get("snapshot"):
            raise ValueError("Pack requires a locally pinned snapshot")
        verified = set()
        for relative, expected in item["files"].items():
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)
            ):
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
        for required in ("config", "checkpoint"):
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
            for asset in snapshot_files(snapshot):
                if asset.is_symlink() or (asset.is_file() and asset.resolve() not in verified):
                    raise ValueError(f"Unhashed or symlinked snapshot asset: {asset}")
        if name == "pocket":
            import yaml

            config_file = (path.parent / item["config"]).resolve()
            config = yaml.safe_load(config_file.read_text("utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Pocket config must be a YAML mapping")

            def walk(node, base=config_file.parent, verified=verified):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key in {
                            "weights_path",
                            "weights_path_without_voice_cloning",
                            "tokenizer_path",
                        } and isinstance(value, str):
                            if value.startswith(("hf://", "https://", "http://")):
                                raise ValueError(f"Remote Pocket resource in {key}")
                            resource = Path(value)
                            if not resource.is_absolute():
                                resource = base / resource
                            if resource.resolve() not in verified:
                                raise ValueError(f"Unhashed Pocket resource {key}")
                        walk(value, base, verified)
                elif isinstance(node, list):
                    for value in node:
                        walk(value, base, verified)

            walk(config)
    return path.parent, packs


def build_realizer(
    base: Path, name: str, item: dict, *, facet_count: int, facet_width: int, device
):
    """One realizer from an already verified pack entry, its identity bound to its files."""
    if name == "pocket":
        realizer = PocketRealizer.from_checkpoint(
            config=str(base / item["config"]),
            checkpoint=str(base / item["checkpoint"]) if item.get("checkpoint") else None,
            facet_count=facet_count,
            facet_width=facet_width,
        )
    elif name in ("sana", "sana_video"):
        realizer = SanaRealizer.from_checkpoint(
            snapshot=str(base / item["snapshot"]),
            modality="image" if name == "sana" else "video",
            facet_count=facet_count,
            facet_width=facet_width,
        )
    else:
        raise ValueError(f"{name} is not a realizer pack")
    realizer.checkpoint_identity = hashlib.sha256(
        json.dumps(
            {"revision": item["revision"], "files": item["files"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return realizer.to_device(device)


def install_realizers(model, manifest: str, *, device) -> None:
    """Install every locked realizer before optimizer enumeration and checkpoint restore.

    Only their facet conditions become parameters; the frozen donors stay
    outside checkpoints. The EmbeddingGemma2 teacher may share the manifest but
    is never installed into the model. The manifest is verified once.
    """
    base, packs = read_locked_manifest(manifest)
    for name in sorted(set(packs) - {"embeddinggemma2"}):
        model.media_realizers.register(
            build_realizer(
                base,
                name,
                packs[name],
                facet_count=model.config.n_facets,
                facet_width=model.config.facet_dim,
                device=device,
            )
        )
