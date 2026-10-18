"""Durable, authenticated, HTTP(S)-reachable Decoupled DiLoCo synchronizer.

No GPU-to-GPU connectivity is required. All update files are validated
safetensors; committed fragment versions contain the deduplication sequence
ledger, so replay after a syncer crash cannot apply an update twice. Deploy
behind a TLS reverse proxy; this server itself intentionally speaks HTTP.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch
from safetensors import safe_open

from megatoken.io import atomic_binary_writer, content_hash, strict_json_loads, write_json
from megatoken.training.diloco_protocol import (
    Fragment,
    load_tensors,
    merge_pseudogradients,
    outer_nesterov,
    pack_fragment,
    parameter_identity,
    plan_fragments,
    save_tensors,
)


class Conflict(ValueError):
    """A stale or duplicated submission, or a live worker holding the slot."""


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.metadata() or {}


def initialize_coordinator(
    root: str | Path,
    *,
    run_id: str,
    model_state: dict[str, torch.Tensor],
    masters: dict[str, torch.Tensor],
    model_config: dict,
    codec_fingerprint: str,
    train_identity: str,
    workers: int = 8,
    min_quorum: int = 4,
    fragment_bytes: int = 64 << 20,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    grace_seconds: float = 8.0,
    max_staleness: int = 2,
    token_budget: int = 0,
) -> dict:
    """Initialize a new run; never silently overwrite an existing trajectory."""
    root = Path(root)
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", run_id)
        or not 1 <= min_quorum <= workers <= 256
        or not 0 <= grace_seconds <= 3600
        or not 0 <= max_staleness <= 32
        or token_budget < 0
        or not (0 < outer_lr <= 4 and 0 <= outer_momentum < 1)
    ):
        raise ValueError("Invalid DiLoCo coordinator configuration")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Refusing to overwrite a DiLoCo coordinator run")
    root.mkdir(parents=True, exist_ok=True)
    fragments = plan_fragments(masters, target_bytes=fragment_bytes)
    manifest = {
        "protocol": 1,
        "run_id": run_id,
        "workers": workers,
        "min_quorum": min_quorum,
        "fragment_bytes": fragment_bytes,
        "outer_lr": outer_lr,
        "outer_momentum": outer_momentum,
        "grace_seconds": grace_seconds,
        "max_staleness": max_staleness,
        "token_budget": token_budget,
        "model_config": model_config,
        "codec_fingerprint": codec_fingerprint,
        "train_identity": train_identity,
        "parameter_identity": parameter_identity(masters),
        "fragments": [item.to_dict() for item in fragments],
    }
    manifest["fingerprint"] = content_hash(manifest)
    try:
        with atomic_binary_writer(root / "bootstrap.pt") as handle:
            torch.save({"model": {k: v.detach().cpu() for k, v in model_state.items()}}, handle)
        for fragment in fragments:
            values = pack_fragment(masters, fragment)
            save_tensors(
                root / "fragments" / f"{fragment.index:05d}.safetensors",
                {"theta": values, "momentum": torch.zeros_like(values)},
                metadata={"version": "0", "last_seq": "{}"},
            )
            save_tensors(
                root / "public" / f"{fragment.index:05d}.safetensors",
                {"theta": values},
                metadata={"version": "0"},
            )
        write_json(root / "progress.json", {"workers": {}, "leases": {}, "merges": 0})
        # Manifest is the last write: a run is discoverable only when complete.
        write_json(root / "manifest.json", manifest)
    except BaseException:
        # Partial initialization is not a valid run (manifest is committed last).
        (root / "manifest.json").unlink(missing_ok=True)
        raise
    return manifest


class Coordinator:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = strict_json_loads((self.root / "manifest.json").read_bytes())
        identity = self.manifest.pop("fingerprint")
        if content_hash(self.manifest) != identity:
            raise ValueError("Coordinator manifest fingerprint mismatch")
        self.manifest["fingerprint"] = identity
        self.fragments = tuple(Fragment.from_dict(x) for x in self.manifest["fragments"])
        self.progress = strict_json_loads((self.root / "progress.json").read_bytes())
        self.lock = threading.RLock()
        self.quorum_since: dict[int, float] = {}
        self._sync_public()
        # A queued quorum surviving a restart gets an immediate opportunity to merge.
        self.tick()

    def _path(self, index: int) -> Path:
        if index < 0 or index >= len(self.fragments):
            raise ValueError("Unknown fragment index")
        return self.root / "fragments" / f"{index:05d}.safetensors"

    def _published(self, index: int) -> Path:
        return self.root / "public" / f"{index:05d}.safetensors"

    def _pending_dir(self, index: int) -> Path:
        return self.root / "pending" / f"{index:05d}"

    def _version(self, index: int) -> int:
        return int(_metadata(self._path(index))["version"])

    def _sync_public(self) -> None:
        with self.lock:
            for fragment in self.fragments:
                index = fragment.index
                version = self._version(index)
                public = self._published(index)
                if not public.exists() or int(_metadata(public)["version"]) != version:
                    theta = load_tensors(self._path(index))["theta"]
                    save_tensors(
                        public, {"theta": theta},
                        metadata={"version": str(version)},
                    )

    def _write_progress(self) -> None:
        write_json(self.root / "progress.json", self.progress)

    def _lease_valid(self, worker: int, session: str) -> None:
        lease = self.progress["leases"].get(str(worker))
        if lease is None or lease["session"] != session or lease["expires"] < time.time():
            raise Conflict("Worker lease expired or belongs to a different session")

    def lease(self, worker: int, session: str) -> dict:
        with self.lock:
            if not 0 <= worker < self.manifest["workers"] or not re.fullmatch(
                r"[a-f0-9-]{36}", session
            ):
                raise ValueError("Invalid worker ID or session UUID")
            previous = self.progress["leases"].get(str(worker))
            if (
                previous
                and previous["expires"] > time.time()
                and previous["session"] != session
            ):
                raise Conflict("Worker ID already leased by a live session")
            self.progress["leases"][str(worker)] = {
                "session": session,
                "expires": time.time() + 180,
            }
            self._write_progress()
            return self.status()

    def heartbeat(self, worker: int, session: str, steps: int, tokens: int) -> dict:
        with self.lock:
            self._lease_valid(worker, session)
            if type(steps) is not int or type(tokens) is not int or steps < 0 or tokens < 0:
                raise ValueError("Progress counters must be nonnegative integers")
            previous = self.progress["workers"].get(str(worker), {"steps": 0, "tokens": 0})
            # Restored learner checkpoints can replay already reported windows.
            # A high-water mark prevents this replay counting training tokens twice.
            self.progress["workers"][str(worker)] = {
                "steps": max(steps, previous["steps"]),
                "tokens": max(tokens, previous["tokens"]),
            }
            self.progress["leases"][str(worker)]["expires"] = time.time() + 180
            self._write_progress()
            return self.status()

    def status(self) -> dict:
        with self.lock:
            global_tokens = sum(x["tokens"] for x in self.progress["workers"].values())
            return {
                "run_id": self.manifest["run_id"],
                "global_tokens": global_tokens,
                "token_budget": self.manifest["token_budget"],
                "finished": bool(
                    self.manifest["token_budget"] and global_tokens >= self.manifest["token_budget"]
                ),
                "merges": self.progress["merges"],
                "versions": [self._version(f.index) for f in self.fragments],
                "live_workers": sum(
                    x["expires"] >= time.time() for x in self.progress["leases"].values()
                ),
                "worker_progress": self.progress["workers"],
                "pending": {
                    str(f.index): len(list(self._pending_dir(f.index).glob("*.safetensors")))
                    for f in self.fragments
                },
            }

    def submit(
        self,
        index: int,
        worker: int,
        session: str,
        sequence: int,
        base_version: int,
        steps: int,
        tokens: int,
        incoming: Path,
    ) -> dict:
        with self.lock:
            self._lease_valid(worker, session)
            self._path(index)  # Reject an unknown fragment before touching queued files.
            if (
                type(sequence) is not int or sequence < 1
                or type(base_version) is not int or base_version < 0
                or type(steps) is not int or steps <= 0
                or type(tokens) is not int or tokens <= 0
            ):
                raise ValueError("Invalid update metadata")
            fragment = self.fragments[index]
            version = self._version(index)
            state_meta = _metadata(self._path(index))
            last_seq = json.loads(state_meta.get("last_seq", "{}"))
            if sequence <= last_seq.get(str(worker), 0):
                return {"status": "already_merged", "version": version}
            if base_version > version or version - base_version > self.manifest["max_staleness"]:
                raise Conflict("Stale fragment baseline; fetch current global weights")
            tensors = load_tensors(incoming)
            if tensors.keys() != {"delta"}:
                raise ValueError("Expected exactly one safetensors delta")
            delta = tensors["delta"]
            if (
                delta.dtype not in (torch.float32, torch.bfloat16)
                or delta.shape != (fragment.length,)
                or not torch.isfinite(delta).all()
            ):
                raise ValueError("Delta dtype, shape, or finiteness invalid")
            target = self._pending_dir(index) / f"{worker:03d}.safetensors"
            if target.exists():
                old = _metadata(target)
                if int(old["sequence"]) == sequence:
                    return {"status": "already_queued", "version": version}
                raise Conflict("One fragment update per worker may be in flight")
            save_tensors(
                target,
                {"delta": delta.float()},
                metadata={
                    "worker": str(worker),
                    "sequence": str(sequence),
                    "base_version": str(base_version),
                    "steps": str(steps),
                    "tokens": str(tokens),
                },
            )
            self._try_merge(index)
            return {"status": "queued", "version": self._version(index)}

    def _try_merge(self, index: int) -> None:
        fragment = self.fragments[index]
        path = self._path(index)
        meta = _metadata(path)
        version = int(meta["version"])
        last_seq = json.loads(meta.get("last_seq", "{}"))
        proposals = []
        for file in sorted(self._pending_dir(index).glob("*.safetensors")):
            item = _metadata(file)
            worker = int(item["worker"])
            sequence = int(item["sequence"])
            age = version - int(item["base_version"])
            if sequence <= last_seq.get(str(worker), 0) or (
                age > self.manifest["max_staleness"] or age < 0
            ):
                file.unlink()
                continue
            proposals.append((file, item, age))
        if len(proposals) < self.manifest["min_quorum"]:
            self.quorum_since.pop(index, None)
            return
        since = self.quorum_since.setdefault(index, time.monotonic())
        if (
            len(proposals) < self.manifest["workers"]
            and time.monotonic() - since < self.manifest["grace_seconds"]
        ):
            return
        self.quorum_since.pop(index, None)
        data = load_tensors(path)
        delta = merge_pseudogradients(
            [load_tensors(file)["delta"] for file, _, _ in proposals],
            [int(item["tokens"]) for _, item, _ in proposals],
            [int(item["steps"]) for _, item, _ in proposals],
            [age for _, _, age in proposals],
        )
        theta, momentum = outer_nesterov(
            data["theta"],
            data["momentum"],
            delta,
            lr=self.manifest["outer_lr"],
            momentum=self.manifest["outer_momentum"],
        )
        for _, item, _ in proposals:
            last_seq[item["worker"]] = int(item["sequence"])
        # Atomic fragment publication includes last_seq. This is the commit
        # record; replayed queued proposals will be discarded on restart.
        save_tensors(
            path, {"theta": theta, "momentum": momentum},
            metadata={"version": str(version + 1), "last_seq": json.dumps(last_seq)},
        )
        save_tensors(
            self._published(index), {"theta": theta.to(torch.bfloat16)},
            metadata={"version": str(version + 1)},
        )
        for file, _, _ in proposals:
            file.unlink(missing_ok=True)
        self.progress["merges"] += 1
        self._write_progress()

    def tick(self) -> None:
        with self.lock:
            for fragment in self.fragments:
                self._try_merge(fragment.index)


def serve_coordinator(root: str | Path, host: str, port: int, token: str) -> None:
    """Bind only to a private host or place an authenticated TLS proxy in front."""
    if not token or len(token) < 24:
        raise ValueError("Set MEGATOKEN_DILOCO_TOKEN to a random secret (>=24 chars)")
    coordinator = Coordinator(root)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            # Do not leak authorization headers, session identifiers or query values.
            pass

        def _auth(self) -> bool:
            if hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return True
            self._json(401, {"error": "Unauthorized"})
            return False

        def _json(self, status: int, value: dict) -> None:
            body = json.dumps(value, sort_keys=True, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream_file(self, file, version: int | None = None) -> None:
            with file:
                size = os.fstat(file.fileno()).st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "no-store")
                if version is not None:
                    self.send_header("X-Fragment-Version", str(version))
                self.end_headers()
                for chunk in iter(lambda: file.read(4 << 20), b""):
                    self.wfile.write(chunk)

        def _file(self, path: Path) -> None:
            self._stream_file(path.open("rb"))

        def _read_json(self) -> dict:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Invalid JSON payload size")
            result = strict_json_loads(self.rfile.read(size))
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result

        def do_GET(self) -> None:
            if not self._auth():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/v1/manifest":
                    self._json(200, coordinator.manifest)
                elif parsed.path == "/v1/status":
                    self._json(200, coordinator.status())
                elif parsed.path == "/v1/bootstrap":
                    self._file(coordinator.root / "bootstrap.pt")
                elif re.fullmatch(r"/v1/fragments/[0-9]+", parsed.path):
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    with coordinator.lock:
                        version = coordinator._version(index)
                        query = parse_qs(parsed.query)
                        unchanged = query.get("after") == [str(version)]
                        # Open under the state lock, but never hold that lock
                        # while a slow HTTP client streams many MiB.
                        file = None if unchanged else coordinator._published(index).open("rb")
                    if unchanged:
                        self.send_response(204)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                    else:
                        self._stream_file(file, version)
                else:
                    self._json(404, {"error": "Unknown endpoint"})
            except (ValueError, OSError) as exc:
                self._json(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            if not self._auth():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/v1/lease":
                    value = self._read_json()
                    result = coordinator.lease(int(value["worker"]), str(value["session"]))
                elif parsed.path == "/v1/progress":
                    value = self._read_json()
                    result = coordinator.heartbeat(
                        int(value["worker"]),
                        str(value["session"]),
                        value["steps"],
                        value["tokens"],
                    )
                elif re.fullmatch(r"/v1/updates/[0-9]+", parsed.path):
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    coordinator._path(index)
                    maximum = coordinator.fragments[index].length * 4 + 8192
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= maximum:
                        raise ValueError("Oversized or empty fragment update")
                    tempdir = coordinator.root / "incoming"
                    tempdir.mkdir(exist_ok=True)
                    fd, filename = tempfile.mkstemp(dir=tempdir, suffix=".safetensors")
                    try:
                        with os.fdopen(fd, "wb") as handle:
                            remaining = length
                            while remaining:
                                block = self.rfile.read(min(4 << 20, remaining))
                                if not block:
                                    raise ValueError("Truncated fragment upload")
                                handle.write(block)
                                remaining -= len(block)
                            handle.flush()
                            os.fsync(handle.fileno())
                        result = coordinator.submit(
                            index,
                            int(self.headers["X-Worker"]),
                            str(self.headers["X-Session"]),
                            int(self.headers["X-Sequence"]),
                            int(self.headers["X-Base-Version"]),
                            int(self.headers["X-Steps"]),
                            int(self.headers["X-Tokens"]),
                            Path(filename),
                        )
                    finally:
                        Path(filename).unlink(missing_ok=True)
                else:
                    self._json(404, {"error": "Unknown endpoint"})
                    return
                self._json(200, result)
            except Conflict as exc:
                self._json(409, {"error": str(exc)})
            except (KeyError, ValueError, OSError) as exc:
                self._json(400, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.request_queue_size = 32
    stop = threading.Event()

    def reconcile() -> None:
        while not stop.wait(1):
            coordinator.tick()

    thread = threading.Thread(target=reconcile, daemon=True)
    thread.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop.set()
        server.server_close()
        thread.join(timeout=2)
