#!/usr/bin/env python3
"""Create a complete local SHA256 manifest from existing pinned pack snapshots.

The inventory is JSON: {"sana": {"revision": "<40hex>", "snapshot": "sana"},
"pocket": {"revision": "<40hex>", "config": "pocket/local.yaml",
"snapshot": "pocket"}}. All referenced files must already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from megatoken.omni.loader import read_locked_manifest

_ALLOWED = {"pocket", "sana", "sana_video", "embeddinggemma2"}
_PATHS = {"snapshot", "config", "checkpoint"}


def create_lock(root: str, inventory: dict) -> dict:
    base = Path(root).resolve()
    if not isinstance(inventory, dict) or not inventory or set(inventory) - _ALLOWED:
        raise ValueError("Inventory has unknown or missing packs")
    locked = {}
    for name, item in sorted(inventory.items()):
        if not isinstance(item, dict) or not item.get("revision"):
            raise ValueError("Each pack requires a pinned revision")
        files: dict[str, str] = {}
        for key in _PATHS:
            if key not in item:
                continue
            relative = Path(item[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Pack paths must be local and relative")
            location = base / relative
            if not location.exists() or location.is_symlink():
                raise ValueError("Pack has missing or symlinked resources")
            children = location.rglob("*") if location.is_dir() else (location,)
            for entry in children:
                if entry.is_symlink():
                    raise ValueError("Symlink inside snapshot; make an immutable local copy")
                if not entry.is_file():
                    continue
                sha = hashlib.sha256()
                with entry.open("rb") as source:
                    for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        sha.update(data)
                files[entry.relative_to(base).as_posix()] = sha.hexdigest()
        if not files:
            raise ValueError("Pack has no files")
        locked[name] = dict(item, files=dict(sorted(files.items())))
    return {"version": 1, "packs": locked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.output).resolve()
    if out.parent != root:
        raise ValueError("Write the lock inside its pack root")
    inventory = json.loads(Path(args.inventory).read_text("utf-8"))
    lock = create_lock(str(root), inventory)
    out.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    read_locked_manifest(str(out))
    print(out)


if __name__ == "__main__":
    main()
