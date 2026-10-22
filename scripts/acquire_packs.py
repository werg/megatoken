#!/usr/bin/env python3
"""Download the pinned pretrained packs and write their lock: the only way packs are made.

    python scripts/acquire_packs.py --root packs [--inventory configs/packs/inventory.json]
        [--only sana sana_video]

Each pack is fetched at its exact 40-hex revision, restricted to the files the
model actually loads, into ``<root>/<pack>``; ``<root>/lock.json`` then records
a SHA256 for every file (``megatoken.omni.loader.read_locked_manifest``).

- ``pocket``: the English model and tokenizer only, plus a fully local
  ``pocket.yaml`` derived from the installed ``pocket_tts`` English config.
  ``kyutai/pocket-tts`` is gated: accept its terms on Hugging Face and supply a
  token (``HF_TOKEN``). Its ungated sibling zeroes the Mimi encoder and must
  not be used: realization targets need the encoder.
- ``sana`` / ``sana_video``: Diffusers snapshots without the Gemma-2 text
  encoder and tokenizer (facets replace the prompt) and without fp16 copies.
- ``embeddinggemma2``: the complete snapshot (the media teacher).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from megatoken.io import fsync_directory, sha256_file, write_json
from megatoken.omni.loader import read_locked_manifest, snapshot_files

PATTERNS = {
    "pocket": {
        "allow_patterns": [
            "languages/english/model.safetensors",
            "languages/english/tokenizer.json",
        ]
    },
    "sana": {"ignore_patterns": ["text_encoder/*", "tokenizer/*", "*.fp16.*"]},
    "sana_video": {"ignore_patterns": ["text_encoder/*", "tokenizer/*"]},
    "embeddinggemma2": {},
}


def pocket_config(snapshot: Path) -> Path:
    """The installed package's English config with every resource made local."""
    from importlib.resources import files

    import yaml

    config = yaml.safe_load(files("pocket_tts").joinpath("config/english.yaml").read_text())
    config["weights_path"] = "languages/english/model.safetensors"
    config.pop("weights_path_without_voice_cloning", None)
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = "languages/english/tokenizer.json"
    path = snapshot / "pocket.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def acquire(root: Path, inventory: dict, *, token: str | None = None) -> dict:
    """Fetch each pack into a temporary directory, rename it into place, then lock it.

    An interrupted run leaves either a temporary directory (removed on the next run) or a
    finished snapshot the lock does not list yet (replaced on the next run); a rerun never
    needs hand cleanup. Every pack is locked as soon as it is in place.
    """
    from huggingface_hub import HfApi, snapshot_download

    if not inventory or set(inventory) - PATTERNS.keys():
        raise ValueError(f"Packs are {sorted(PATTERNS)}")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "lock.json"
    packs = json.loads(lock.read_text())["packs"] if lock.exists() else {}
    for name, item in sorted(inventory.items()):
        repository, revision = item["repository"], item["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pack revisions are immutable 40-hex commits")
        if HfApi(token=token).model_info(repository, revision=revision).sha != revision:
            raise ValueError(f"The Hub resolved another commit for {repository}")
        snapshot = root / name
        if snapshot.exists() and name in packs:
            raise FileExistsError(f"Packs are immutable; remove {snapshot} first")
        partial = root / f".{name}.partial"
        for leftover in (partial, snapshot):  # an unlocked snapshot is an interrupted run's
            shutil.rmtree(leftover, ignore_errors=True)
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=str(partial),
            token=token,
            **PATTERNS[name],
        )
        shutil.rmtree(partial / ".cache", ignore_errors=True)
        entry = {"repository": repository, "revision": revision, "snapshot": name}
        if name == "pocket":
            entry["config"] = f"{name}/{pocket_config(partial).relative_to(partial).as_posix()}"
        entry["files"] = {
            f"{name}/{path.relative_to(partial).as_posix()}": sha256_file(path)
            for path in snapshot_files(partial)
            if path.is_file()
        }
        partial.rename(snapshot)
        fsync_directory(root)
        packs[name] = entry
        write_json(lock, {"version": 1, "packs": packs})
    read_locked_manifest(str(lock))
    return {"lock": str(lock), "packs": sorted(packs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=Path("configs/packs/inventory.json"))
    parser.add_argument("--only", nargs="+", help="Acquire just these packs")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if args.only:
        inventory = {name: inventory[name] for name in args.only}
    print(json.dumps(acquire(args.root, inventory), indent=2))


if __name__ == "__main__":
    main()
