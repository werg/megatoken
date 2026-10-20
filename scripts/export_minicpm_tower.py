#!/usr/bin/env python3
"""Export trained MiniCPM-V visual tower/merger from a SHA256-locked parent.

No LLM or language-model weights are instantiated. Use the exported directory
as a distinct `minicpm_export` pack after generating a fresh full checksum lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from megatoken.omni.loader import read_locked_manifest
from megatoken.omni.perception import export_minicpm_submodules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Pinned parent weights lock")
    parser.add_argument("--output", required=True, help="New local export directory")
    args = parser.parse_args()
    base, packs = read_locked_manifest(args.manifest)
    item = packs.get("minicpm")
    if item is None or not item.get("snapshot"):
        raise ValueError("MiniCPM parent must have a pinned source snapshot")
    parent = (base / item["snapshot"]).resolve()
    output = Path(args.output).resolve()
    if not parent.is_dir() or output == parent or output.is_relative_to(parent):
        raise ValueError("Output cannot overwrite or nest in the verified parent")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an earlier visual export")
    result = export_minicpm_submodules(str(parent), str(output))
    fingerprints = {}
    for name in ("vision_tower.safetensors", "merger.safetensors", "config.json"):
        artifact = output / name
        fingerprints[name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "source_revision": item["revision"],
        "source_snapshot": item["snapshot"],
        "submodule_tensors": result,
        "sha256": fingerprints,
    }
    (output / "export_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
