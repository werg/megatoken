#!/usr/bin/env python3
"""Stage immutable training input artifacts from S3 to a sandbox's local disk.

The S3 prefix must contain codec/codec.json and prepared/manifest.json,
plus their referenced tokenizer/data files. Fails before training if empty.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from megatoken.io import fsync_directory


def stage(uri: str, destination: str | Path) -> int:
    import boto3

    parts = urlsplit(uri)
    if parts.scheme != "s3" or not parts.netloc:
        raise ValueError("S3 artifact URI must be s3://bucket/prefix")
    prefix = parts.path.strip("/")
    prefix = f"{prefix}/" if prefix else ""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=parts.netloc, Prefix=prefix
    ):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            if key.endswith("/"):
                continue
            relative = Path(key.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("S3 data contains an unsafe relative path")
            output = root / relative
            if relative.parts[0] not in {"prepared", "codec"}:
                raise ValueError("Data prefix may only contain prepared/ and codec/ assets")
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".staging-", dir=output.parent)
            os.close(fd)
            try:
                client.download_file(parts.netloc, key, name)
                if Path(name).stat().st_size != entry["Size"]:
                    raise ValueError(f"Downloaded artifact size differs for {relative}")
                os.replace(name, output)
            finally:
                Path(name).unlink(missing_ok=True)
            count += 1
    if not count:
        raise ValueError("No artifacts found in the supplied S3 prefix")
    for required in ("codec/codec.json", "prepared/manifest.json"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"Missing required staged asset {required}")
    fsync_directory(root)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--destination", default="artifacts")
    args = parser.parse_args()
    print(f"Staged {stage(args.s3, args.destination)} verified-size artifact objects")


if __name__ == "__main__":
    main()
