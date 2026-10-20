#!/usr/bin/env python3
"""Acquire pinned pretrained *weights* only; no dataset processing or training.

Inventory example:
{"sana":{"repository":"Efficient-Large-Model/Sana_600M_1024px_diffusers",
         "revision":"40hex..."},
 "sana_video":{"repository":"Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
               "revision":"40hex..."},
 "minicpm":{"repository":"openbmb/MiniCPM-V-4.7-1B","revision":"40hex..."}}.

The exact 40-hex commit and immutable local destination are compulsory.
Pocket multi-repository tokenizer/model resources must be copied into its
local snapshot and referenced by a fully local Pocket YAML before locking.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def acquire(root: str, inventory: dict, *, token: str | None = None):
    from huggingface_hub import HfApi, snapshot_download

    root_path = Path(root).resolve()
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Supply at least one named pinned pretrained component")
    allowed = {"whistle", "pocket", "minicpm", "sana", "sana_video"}
    if set(inventory) - allowed:
        raise ValueError("Unknown pretrained pack")
    root_path.mkdir(parents=True, exist_ok=True)
    acquired = {}
    for name, item in sorted(inventory.items()):
        revision = item.get("revision")
        repository = item.get("repository")
        if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
            raise ValueError("Model revisions must be immutable 40-character commits")
        if not isinstance(repository, str) or repository.count("/") != 1:
            raise ValueError("Hugging Face model repository must be owner/name")
        output = root_path / name
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite an existing pack: {output}")
        actual = HfApi(token=token).model_info(repository, revision=revision).sha
        if actual != revision:
            raise ValueError(f"Hub resolved unexpected commit for {repository}")
        snapshot_download(
            repo_id=repository,
            repo_type="model",
            revision=revision,
            local_dir=str(output),
            token=token,
        )
        cache = output / ".cache"
        if cache.exists():
            shutil.rmtree(cache)
        # Avoid a hidden download cache from being mistaken for an executable
        # checkpoint. The lock helper will hash *every* remaining file.
        acquired[name] = {"repository": repository, "revision": revision, "snapshot": name}
    return acquired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument(
        "--output", required=True, help="Path for acquisition receipt JSON (not pack lock)"
    )
    args = parser.parse_args()
    requested = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    results = acquire(args.root, requested)
    Path(args.output).write_text(
        json.dumps(results, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
