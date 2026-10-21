"""Create a complete SHA256 lock for a *previously pinned* local HF snapshot.

Get the source only by immutable HF commit first, for example:
  hf download google/embeddinggemma-2 --revision <40-hex> --local-dir <snapshot>

This command does not authenticate a model revision against the Hub; it pins
the bytes already present and records the caller-supplied download revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use the exact immutable 40-hex HF revision")
    root = Path(args.snapshot).resolve()
    if not (root / "config.json").is_file():
        parser.error("Missing local EmbeddingGemma2 config.json")
    manifest = {}
    for file in sorted(root.rglob("*")):
        if file.is_symlink():
            parser.error("Do not pin snapshot symlinks")
        if not file.is_file() or file.name == "teacher-lock.json":
            continue
        relative = file.relative_to(root).as_posix()
        with file.open("rb") as stream:
            manifest[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    if not manifest:
        parser.error("Local HF snapshot is empty")
    lock = {"model": "google/embeddinggemma-2", "revision": args.revision, "sha256": manifest}
    (root / "teacher-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
