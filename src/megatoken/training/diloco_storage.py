"""Immutable worker checkpoint uploads to S3-compatible durable object storage.

The latest pointer is committed only after all checkpoint objects upload; older
checkpoints are then pruned by the local retention rule (newest ``keep`` periodic
ones, every pinned one, and the one the pointer names).
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
            raise ImportError("Install megatoken[storage] for durable checkpoint storage") from exc
        self.bucket = parsed.netloc
        self.prefix = "/".join(
            x for x in (parsed.path.strip("/"), run_id, f"worker-{worker_id:02d}") if x
        )
        # Uses short-lived workload credentials, AWS_* env or attached role.
        self.s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"))

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def upload(self, checkpoint: str | Path, *, keep: int = 3) -> dict:
        """Upload every checkpoint file (weights, transducer bank, manifests), then the
        pointer, then prune what retention no longer keeps."""
        path = Path(checkpoint)
        meta = path / "metadata.json"
        metadata = json.loads(meta.read_text())
        if metadata["checkpoint_name"] != path.name:
            raise ValueError("Checkpoint directory and metadata differ")
        if sha256_file(path / "state.pt") != metadata["state_sha256"]:
            raise ValueError("Refusing to upload corrupted checkpoint")
        bank = metadata["transducer_checkpoint"]
        if sha256_file(path / bank["file"]) != bank["sha256"]:
            raise ValueError("Refusing to upload a corrupted transducer bank")
        files = {
            file.name: sha256_file(file)
            for file in sorted(path.iterdir())
            if file.is_file() and file.name != "metadata.json"
        }
        files["metadata.json"] = sha256_file(meta)
        for name in files:
            self.s3.upload_file(str(path / name), self.bucket, self._key(f"{path.name}/{name}"))
        pointer = {
            "checkpoint": path.name,
            "files": files,
            "state_sha256": files["state.pt"],
            "metadata_sha256": files["metadata.json"],
            "global_step": metadata["global_step"],
        }
        # S3 put is atomic: missing/broken snapshots are never advertised.
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key("latest.json"),
            Body=json.dumps(pointer, sort_keys=True).encode(),
            ContentType="application/json",
        )
        self.prune(path.name, keep)
        return pointer

    def prune(self, current: str, keep: int) -> None:
        """Delete remote periodic checkpoints beyond the newest ``keep`` (pinned ones stay)."""
        if keep < 1:
            raise ValueError("At least one periodic checkpoint must be retained")
        objects: dict[str, list[str]] = {}
        token: dict[str, str] = {}
        while True:
            page = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{self.prefix}/", **token)
            for item in page.get("Contents", []):
                name, _, rest = item["Key"].removeprefix(f"{self.prefix}/").partition("/")
                if rest:
                    objects.setdefault(name, []).append(item["Key"])
            if not page.get("IsTruncated"):
                break
            token = {"ContinuationToken": page["NextContinuationToken"]}
        periodic = []
        for name, keys in objects.items():
            if self._key(f"{name}/metadata.json") not in keys:
                continue  # an upload still in flight has no metadata yet: keep it
            body = self.s3.get_object(Bucket=self.bucket, Key=self._key(f"{name}/metadata.json"))
            metadata = json.loads(body["Body"].read())
            if not metadata.get("pinned", True):
                periodic.append((metadata.get("global_step", 0), name))
        for _, name in sorted(periodic, reverse=True)[keep:]:
            if name != current:
                keys = objects[name]
                for start in range(0, len(keys), 1000):
                    self.s3.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": [{"Key": key} for key in keys[start : start + 1000]]},
                    )

    def restore(self, checkpoints: str | Path) -> Path | None:
        try:
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise ImportError("Install megatoken[storage]") from exc
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
        files = pointer["files"]
        if not isinstance(files, dict) or any(
            not isinstance(f, str) or not f or Path(f).name != f or f.startswith(".") for f in files
        ):
            raise ValueError("Invalid remote checkpoint file list")
        if not {"state.pt", "metadata.json"} <= set(files):
            raise ValueError("Remote checkpoint lacks weights or metadata")
        root = Path(checkpoints)
        root.mkdir(parents=True, exist_ok=True)
        for stale in root.glob(".download-*"):
            shutil.rmtree(stale, ignore_errors=True)  # an earlier restore died mid-download
        destination = root / name
        if destination.exists():
            for filename, digest in files.items():
                if sha256_file(destination / filename) != digest:
                    raise ValueError(f"Local checkpoint {filename} conflicts with remote")
        else:
            staging = Path(tempfile.mkdtemp(prefix=".download-", dir=root))
            try:
                for filename, digest in files.items():
                    self.s3.download_file(
                        self.bucket, self._key(f"{name}/{filename}"), str(staging / filename)
                    )
                    if sha256_file(staging / filename) != digest:
                        raise ValueError(f"Remote checkpoint {filename} checksum mismatch")
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
