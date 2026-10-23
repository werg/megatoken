#!/usr/bin/env python3
"""Freeze local preparation/training/serving code with per-file content checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Explicit pinned replacement: relative/path=local/path; recorded in the snapshot",
    )
    args = parser.parse_args()
    repo, root = args.repo.resolve(), args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("src", "scripts", "configs", "docs", "tests"):
        shutil.copytree(
            repo / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
    for name in ("pyproject.toml", "README.md", "AGENTS.md"):
        if (repo / name).is_file():
            shutil.copy2(repo / name, root / name)
    overlays = {}
    for specification in args.overlay:
        relative, separator, original = specification.partition("=")
        destination = (root / relative).resolve()
        if not separator or not destination.is_relative_to(root) or not destination.is_file():
            raise ValueError(
                "An explicit overlay must replace an existing file inside the snapshot"
            )
        shutil.copyfile(original, destination)
        overlays[relative] = {
            "path": str(Path(original).resolve()),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    (root / "snapshot.json").write_text(
        json.dumps({"files": files, "explicit_overlays": overlays}, indent=2) + "\n"
    )
    archive = root.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
    print(
        json.dumps(
            {
                "root": str(root),
                "archive": str(archive),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": len(files),
                "overlays": overlays,
            }
        )
    )


if __name__ == "__main__":
    main()
