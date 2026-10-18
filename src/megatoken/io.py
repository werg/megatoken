"""Durable local writes and content identities shared by every pipeline stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO


def strict_json_loads(value: str | bytes) -> Any:
    """Read standard JSON without silently accepting ambiguous object keys."""

    def invalid_constant(name):
        raise ValueError(f"nonstandard JSON constant {name}")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key {key}")
            value[key] = item
        return value

    return json.loads(value, parse_constant=invalid_constant, object_pairs_hook=unique_object)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def atomic_binary_writer(path: str | Path) -> Iterator[BinaryIO]:
    """Readers observe either the previous complete file or the new complete file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    with atomic_binary_writer(path) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode())
        handle.write(b"\n")


def read_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix in {".yaml", ".yml"}:
            import yaml

            value = yaml.safe_load(handle)
        else:
            value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a configuration mapping in {source}")
    return dict(value)
