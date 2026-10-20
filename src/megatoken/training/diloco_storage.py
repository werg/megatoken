"""Immutable worker checkpoint uploads to S3-compatible durable object storage.

The latest pointer is committed only after all checkpoint objects upload.
Workers recover the same per-rank cursor, optimizer, RNG and pending DiLoCo
outbox, even when CoreWeave reclaims the original sandbox scratch disk.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from megatoken.io import fsync_directory, sha256_file, write_json


class CheckpointStore:
    def __init__(self, uri: str, run_id: str, worker_id: int) -> None:
        parsed = urlsplit(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("Expected checkpoint_s3=s3://bucket/prefix")
        if any(part in {".", ".."} for part in parsed.path.split("/")):
            raise ValueError("Unsafe checkpoint S3 prefix")
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Install megatoken[coreweave] for durable checkpoint storage"
            ) from exc
        self.bucket = parsed.netloc
        self.prefix = "/".join(
            x for x in (parsed.path.strip("/"), run_id, f"worker-{worker_id:02d}") if x
        )
        # Uses short-lived workload credentials, AWS_* env or attached role.
        self.s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def upload(self, checkpoint: str | Path) -> dict:
        path = Path(checkpoint)
        meta = path / "metadata.json"
        state = path / "state.pt"
        metadata = json.loads(meta.read_text())
        if metadata["checkpoint_name"] != path.name:
            raise ValueError("Checkpoint directory and metadata differ")
        state_sha = sha256_file(state)
        if state_sha != metadata["state_sha256"]:
            raise ValueError("Refusing to upload corrupted checkpoint")
        meta_sha = sha256_file(meta)
        for file in (state, meta):
            self.s3.upload_file(str(file), self.bucket, self._key(f"{path.name}/{file.name}"))
        pointer = {
            "checkpoint": path.name,
            "state_sha256": state_sha,
            "metadata_sha256": meta_sha,
            "global_step": metadata["global_step"],
        }
        # S3 put is atomic: missing/broken snapshots are never advertised.
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key("latest.json"),
            Body=json.dumps(pointer, sort_keys=True).encode(),
            ContentType="application/json",
        )
        return pointer

    def restore(self, checkpoints: str | Path) -> Path | None:
        try:
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise ImportError("Install megatoken[coreweave]") from exc
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self._key("latest.json"))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        pointer = json.loads(obj["Body"].read())
        name = pointer["checkpoint"]
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("Invalid remote checkpoint pointer")
        root = Path(checkpoints)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / name
        if destination.exists():
            if sha256_file(destination / "state.pt") != pointer["state_sha256"]:
                raise ValueError("Local checkpoint conflicts with remote checkpoint name")
            if sha256_file(destination / "metadata.json") != pointer["metadata_sha256"]:
                raise ValueError("Local checkpoint metadata differs from remote")
        else:
            staging = Path(tempfile.mkdtemp(prefix=".download-", dir=root))
            try:
                for filename in ("state.pt", "metadata.json"):
                    self.s3.download_file(
                        self.bucket, self._key(f"{name}/{filename}"), str(staging / filename)
                    )
                if sha256_file(staging / "state.pt") != pointer["state_sha256"]:
                    raise ValueError("Remote checkpoint state checksum mismatch")
                if sha256_file(staging / "metadata.json") != pointer["metadata_sha256"]:
                    raise ValueError("Remote checkpoint metadata checksum mismatch")
                fsync_directory(staging)
                os.rename(staging, destination)
                fsync_directory(root)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        write_json(
            root / "latest.json",
            {"checkpoint": name, "metadata_sha256": pointer["metadata_sha256"]},
        )
        return destination
