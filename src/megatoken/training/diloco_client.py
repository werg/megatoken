"""Nonblocking DiLoCo worker transport, restart state and optimizer reconciliation.

Network work runs on an I/O pool. Live model mutation happens exclusively
between training optimizer transactions on the training thread; threads never
touch CUDA tensors. Durable outbox files make failed uploads retryable.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from typing import Any

import torch

from megatoken.io import atomic_binary_writer, content_hash
from megatoken.training.diloco_protocol import (
    Fragment,
    install_fragment,
    load_tensors,
    pack_fragment,
    parameter_identity,
    save_tensors,
)


class RemoteError(RuntimeError):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"DiLoCo coordinator HTTP {status}: {reason}")
        self.status = status


class Transport:
    def __init__(self, url: str, token: str, timeout: float, allow_http: bool = False):
        self.url = url.rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("Coordinator must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and not (
            allow_http or parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("Use HTTPS for the coordinator outside localhost")
        if not token or len(token) < 24:
            raise ValueError("MEGATOKEN_DILOCO_TOKEN is missing or too short")
        self.token = token
        self.timeout = timeout
        # A coordinator behind a gateway that admits only share-token holders
        # (megatoken.cloud sandboxes) also needs that gateway credential.
        gateway = os.environ.get("MEGATOKEN_DILOCO_GATEWAY_TOKEN", "")
        self.gateway = {"X-Sandbox-Share-Token": gateway} if gateway else {}

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        file: Path | None = None,
        headers: dict[str, str] | None = None,
        output: Path | None = None,
    ) -> tuple[int, Any, Message]:
        if json_body is not None and file is not None:
            raise ValueError("Cannot send JSON and a binary body simultaneously")
        body = (
            json.dumps(json_body, sort_keys=True, allow_nan=False).encode()
            if json_body is not None
            else file.read_bytes()
            if file
            else None
        )
        request = urllib.request.Request(
            self.url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                **self.gateway,
                **({"Content-Type": "application/json"} if json_body is not None else {}),
                **(headers or {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                # Case-insensitive: HTTP/2 gateways lowercase header names.
                response_headers = response.headers
                if output is not None and response.status == 200:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("wb") as handle:
                        for block in iter(lambda: response.read(4 << 20), b""):
                            handle.write(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    return response.status, output, response_headers
                raw = response.read()
                return (
                    response.status,
                    json.loads(raw)
                    if raw and response.headers.get_content_type() == "application/json"
                    else raw,
                    response_headers,
                )
        except urllib.error.HTTPError as error:
            raw = error.read(4096).decode(errors="replace")
            raise RemoteError(error.code, raw) from error

    def json(self, method: str, path: str, value: dict | None = None) -> dict:
        _, result, _ = self.request(method, path, json_body=value)
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON coordinator response from {path}")
        return result

    def fetch_file(
        self,
        path: str,
        directory: Path,
        *,
        json_body: dict | None = None,
    ) -> tuple[Path | None, Message]:
        fd, name = tempfile.mkstemp(dir=directory, suffix=".download")
        os.close(fd)
        target = Path(name)
        try:
            status, _, headers = self.request(
                "GET" if json_body is None else "POST",
                path,
                json_body=json_body,
                output=target,
            )
            if status == 204:
                target.unlink()
                return None, headers
            if status != 200:
                raise ValueError(f"Unexpected HTTP status {status}")
            return target, headers
        except BaseException:
            target.unlink(missing_ok=True)
            raise


def fetch_kb_snapshot(settings: Any, destination: Path) -> int:
    """Download the coordinator's canonical free-KB snapshot before training."""
    transport = Transport(
        settings.coordinator_url,
        os.environ.get("MEGATOKEN_DILOCO_TOKEN", ""),
        settings.timeout_seconds,
        allow_http=settings.allow_private_http,
    )
    manifest = transport.json("GET", "/v1/manifest")
    if manifest.get("kb") is None:
        raise ValueError("DiLoCo coordinator was initialized without a free KB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    path, headers = transport.fetch_file("/v1/kb", destination.parent)
    assert path is not None
    os.replace(path, destination)
    return int(headers["X-KB-Version"])


class DecoupledWorker:
    """One learner per sandbox, without process groups or blocking collectives."""

    def __init__(
        self,
        settings: Any,
        *,
        model_config: dict,
        codec_fingerprint: str,
        masters: dict[str, torch.Tensor],
        output: Path,
        token_budget: int,
        train_identity: str,
        dataset_fingerprint: str,
    ) -> None:
        self.settings = settings
        self.worker = settings.resolved_worker_id
        self.transport = Transport(
            settings.coordinator_url,
            os.environ.get("MEGATOKEN_DILOCO_TOKEN", ""),
            settings.timeout_seconds,
            allow_http=settings.allow_private_http,
        )
        self.manifest = self.transport.json("GET", "/v1/manifest")
        if (
            self.manifest["run_id"] != settings.run_id
            or self.manifest["workers"] != settings.workers
            or content_hash(self.manifest["model_config"]) != content_hash(model_config)
            or self.manifest["codec_fingerprint"] != codec_fingerprint
            or self.manifest["parameter_identity"] != parameter_identity(masters)
            or self.manifest["token_budget"] != token_budget
            or self.manifest["train_identity"] != train_identity
            or self.manifest["dataset_fingerprint"] != dataset_fingerprint
        ):
            raise ValueError("DiLoCo worker architecture/codec/topology/budget differs from syncer")
        self.fragments = tuple(Fragment.from_dict(x) for x in self.manifest["fragments"])
        self.output = output
        self.outbox = output / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.io = ThreadPoolExecutor(max_workers=settings.io_workers, thread_name_prefix="diloco")
        # Heartbeats and KB sync are small and latency-sensitive: give them their
        # own lane so bulk fragment transfers can never starve them.
        self.control_io = ThreadPoolExecutor(max_workers=2, thread_name_prefix="diloco-control")
        # Coordinator versions from the latest status; drives targeted fetches.
        self.remote_versions: list[int] | None = None
        self.remote_kb_version: int | None = None
        self.session = str(uuid.uuid4())
        self.versions = [-1] * len(self.fragments)
        self.baselines: list[torch.Tensor | None] = [None] * len(self.fragments)
        self.steps_since = [0] * len(self.fragments)
        self.tokens_since = [0] * len(self.fragments)
        self.sequences = [0] * len(self.fragments)
        self.pending: dict[int, dict] = {}
        self.awaiting_global: set[int] = set()
        self.downloads: dict[int, Future] = {}
        self.heartbeat: Future | None = None
        self.local_steps = 0
        self.local_tokens = 0
        self.global_tokens = 0
        self.merges = 0
        self.finished = False
        self.last_heartbeat_step = -settings.heartbeat_steps
        self.last_poll_step = -settings.poll_steps
        self.last_progress_error = ""
        self.offline_steps = 0
        self._poll_cursor = 0
        self.kb = None
        self._kb_force_fetch = False
        self.transducer = None
        self.remote_transducer_version: int | None = None

    def attach_transducer(self, sync: Any) -> None:
        """Synchronize v10 transducer rounds through the coordinator (transducer_sync)."""
        if self.manifest.get("transducer") is None:
            raise ValueError("DiLoCo coordinator was initialized without a transducer bank")
        self.transducer = sync

    def attach_kb(self, kb: Any) -> None:
        """Synchronize free-KB rows through the coordinator (see diloco_kb)."""
        if self.manifest.get("kb") is None:
            raise ValueError("DiLoCo coordinator was initialized without a free KB")
        self.kb = kb

    def _lease(self) -> None:
        status = self.transport.json(
            "POST", "/v1/lease", {"worker": self.worker, "session": self.session}
        )
        self._update_status(status)

    def _update_status(self, value: dict) -> None:
        self.global_tokens = int(value["global_tokens"])
        self.merges = int(value["merges"])
        self.finished = bool(value["finished"])
        versions = value.get("versions")
        if isinstance(versions, list) and len(versions) == len(self.fragments):
            self.remote_versions = [int(v) for v in versions]
        if value.get("kb_version") is not None:
            self.remote_kb_version = int(value["kb_version"])
        if value.get("transducer_version") is not None:
            self.remote_transducer_version = int(value["transducer_version"])

    def bootstrap(self, model: torch.nn.Module, optimizer: Any) -> None:
        """Get identical initialization, then reconcile with latest global fragments."""
        self._lease()
        path, _ = self.transport.fetch_file("/v1/bootstrap", self.outbox)
        assert path is not None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model"], strict=True)
        finally:
            path.unlink(missing_ok=True)
        optimizer.rebase(list(optimizer.masters))
        self._fetch_all(optimizer)

    def resume(self, value: dict, optimizer: Any) -> None:
        if value["protocol"] != 1 or value["fingerprint"] != self.manifest["fingerprint"]:
            raise ValueError("DiLoCo checkpoint belongs to a different coordinator run")
        if value["worker"] != self.worker:
            raise ValueError("DiLoCo worker checkpoint belongs to another dataset shard")
        self.session = value["session"]
        self.versions = list(value["versions"])
        self.baselines = [part.float().cpu() for part in value["baselines"]]
        self.steps_since = list(value["steps_since"])
        self.tokens_since = list(value["tokens_since"])
        self.sequences = list(value["sequences"])
        self.local_steps = int(value["local_steps"])
        self.local_tokens = int(value["local_tokens"])
        self.pending = {int(k): dict(v, future=None) for k, v in value["pending"].items()}
        self.awaiting_global = set(value.get("awaiting_global", []))
        if self.kb is not None:
            if value.get("kb") is None:
                raise ValueError("DiLoCo checkpoint lacks free-KB synchronization state")
            self.kb.load_state_dict(value["kb"], self.outbox)
        self._lease()
        # The committed checkpoint embeds each pending upload for recovery when
        # a sandbox's ephemeral scratch volume has been discarded.
        for index, entry in list(self.pending.items()):
            filename = self.outbox / f"{index:05d}-{entry['sequence']:09d}.safetensors"
            if not filename.exists():
                payload = value.get("pending_payloads", {}).get(str(index))
                if payload is None:
                    raise FileNotFoundError(f"Checkpoint lacks pending upload {index}")
                entry["delta"] = payload
            entry["filename"] = str(filename)
            self._send_async(index)
        if self.kb is not None and self.kb.pending is not None:
            self.kb.pending["future"] = self.control_io.submit(
                self._upload_kb, dict(self.kb.pending)
            )
        if self.transducer is not None:
            if value.get("transducer") is None:
                raise ValueError("DiLoCo checkpoint lacks transducer synchronization state")
            self.transducer.load_state_dict(value["transducer"])
        self._fetch_all(optimizer)

    def _fetch(self, index: int, version: int) -> tuple[int, torch.Tensor] | None:
        path, headers = self.transport.fetch_file(
            f"/v1/fragments/{index}?after={version}", self.outbox
        )
        if path is None:
            return None
        try:
            vector = load_tensors(path)["theta"]
            next_version = int(headers["X-Fragment-Version"])
            if (
                vector.shape != (self.fragments[index].length,)
                or vector.dtype not in (torch.bfloat16, torch.float32)
                or not bool(vector.isfinite().all())
            ):
                raise ValueError("Invalid global fragment received")
            return next_version, vector.float()
        finally:
            path.unlink(missing_ok=True)

    def _install(self, index: int, update: tuple[int, torch.Tensor] | None, optimizer: Any) -> bool:
        """Install a newer global fragment *without discarding local progress*.

        ``baselines[index]`` is the anchor that local progress is measured from:
        the last installed global, or the upload point once a pseudo-gradient
        was queued. Steps trained since the anchor are carried onto the new
        global (overlapped/streaming DiLoCo), so learners never idle or waste
        compute while a round is in flight, and the next pseudo-gradient
        contains exactly that residual progress.
        """
        if update is None:
            return False
        version, tensor = update
        if version <= self.versions[index]:
            return False
        anchor = self.baselines[index]
        if anchor is None:
            values = tensor
        else:
            values = tensor + (pack_fragment(optimizer.masters, self.fragments[index]) - anchor)
        install_fragment(optimizer, self.fragments[index], values)
        self.versions[index] = version
        self.baselines[index] = tensor
        self.awaiting_global.discard(index)
        return True

    def _fetch_all(self, optimizer: Any) -> None:
        for index in range(len(self.fragments)):
            self._install(index, self._fetch(index, self.versions[index]), optimizer)

    def _upload(self, index: int, entry: dict) -> dict:
        _, response, _ = self.transport.request(
            "POST",
            f"/v1/updates/{index}",
            file=self._write_outbox(entry),
            headers={
                "X-Worker": str(self.worker),
                "X-Session": self.session,
                "X-Sequence": str(entry["sequence"]),
                "X-Base-Version": str(entry["version"]),
                "X-Steps": str(entry["steps"]),
                "X-Tokens": str(entry["tokens"]),
                "Content-Type": "application/octet-stream",
            },
        )
        return response

    def _upload_kb(self, entry: dict) -> dict:
        _, response, _ = self.transport.request(
            "POST",
            "/v1/kb/updates",
            file=Path(entry["filename"]),
            headers={
                "X-Worker": str(self.worker),
                "X-Session": self.session,
                "X-Sequence": str(entry["sequence"]),
                "X-Base-Version": str(entry["version"]),
                "X-Steps": str(entry["steps"]),
                "X-Tokens": str(entry["tokens"]),
                "Content-Type": "application/octet-stream",
            },
        )
        return response

    def _fetch_kb(
        self,
        version: int,
        trained: tuple[tuple[int, ...], tuple[int, ...]] = ((), ()),
    ) -> tuple[int, Path, str] | None:
        """Rows changed since ``version`` plus trained rows; snapshot as fallback."""
        if version >= 0:
            try:
                path, headers = self.transport.fetch_file(
                    "/v1/kb/changes",
                    self.outbox,
                    json_body={
                        "after": version,
                        "records": list(trained[0]),
                        "relations": list(trained[1]),
                    },
                )
                if path is None:
                    return None
                return int(headers["X-KB-Version"]), path, "changes"
            except RemoteError as exc:
                if exc.status != 410:
                    raise
        path, headers = self.transport.fetch_file(f"/v1/kb?after={version}", self.outbox)
        if path is None:
            return None
        return int(headers["X-KB-Version"]), path, "snapshot"

    def fetch_transducer(self, version: int) -> tuple[int, Path, str] | None:
        """Changes since ``version``; the whole published bank when history expired."""
        if version >= 0:
            try:
                path, headers = self.transport.fetch_file(
                    f"/v1/transducer/changes?after={version}", self.outbox
                )
                if path is None:
                    return None
                return int(headers["X-Transducer-Version"]), path, "changes"
            except RemoteError as exc:
                if exc.status != 410:
                    raise
        path, headers = self.transport.fetch_file(f"/v1/transducer?after={version}", self.outbox)
        if path is None:
            return None
        return int(headers["X-Transducer-Version"]), path, "snapshot"

    def _upload_transducer(self, entry: dict) -> dict:
        _, response, _ = self.transport.request(
            "POST",
            "/v1/transducer/rounds",
            file=Path(entry["filename"]),
            headers={
                "X-Worker": str(self.worker),
                "X-Session": self.session,
                "X-Sequence": str(entry["sequence"]),
                "X-Base-Version": str(entry["version"]),
                "X-Steps": str(entry["steps"]),
                "X-Tokens": "1",
                "Content-Type": "application/octet-stream",
            },
        )
        return response

    def report_transducer_validation(self, step: int, loss: float) -> None:
        """Worker 0's bank-only validation loss, for the coordinator's switch rule."""
        self.control_io.submit(
            self.transport.json,
            "POST",
            "/v1/transducer/validation",
            {"worker": self.worker, "session": self.session, "step": step, "loss": loss},
        )

    def _transducer_step(self, steps: int, trainer: Any) -> tuple[int, Any]:
        """Upload a round every 32 local steps; fetch newer canonical versions.

        Returns (errors, ready (version, path, kind) or None) for the caller to
        install between optimizer steps.
        """
        from megatoken.training.transducer_diloco import LOCAL_STEPS
        from megatoken.training.transducer_sync import submit_round

        sync = self.transducer
        errors = 0
        ready = None
        sync.steps_since += 1
        entry = sync.pending
        if entry is not None and entry["future"] is not None and entry["future"].done():
            try:
                entry["future"].result()
                Path(entry["filename"]).unlink(missing_ok=True)
                sync.pending = None
            except RemoteError as exc:
                if exc.status == 409:
                    # Stale or duplicate: drop it and re-base on the canonical bank.
                    Path(entry["filename"]).unlink(missing_ok=True)
                    sync.pending = None
                    sync.force_snapshot = True
                else:
                    errors += 1
                    entry["future"] = None
                    self.last_progress_error = str(exc)
            except (OSError, TimeoutError, ValueError) as exc:
                errors += 1
                entry["future"] = None
                self.last_progress_error = str(exc)
        if sync.pending is not None and sync.pending["future"] is None:
            sync.pending["future"] = self.control_io.submit(
                self._upload_transducer, dict(sync.pending)
            )
        if sync.download is not None and sync.download.done():
            try:
                result = sync.download.result()
                if result is not None and (result[0] > sync.version or result[2] == "snapshot"):
                    ready = result
                    sync.force_snapshot = False
                elif result is not None:
                    result[1].unlink(missing_ok=True)
            except (OSError, TimeoutError, ValueError, RemoteError) as exc:
                errors += 1
                self.last_progress_error = str(exc)
            sync.download = None
        if sync.steps_since >= LOCAL_STEPS and sync.pending is None:
            submission = submit_round(trainer, sync, worker=self.worker, step=steps)
            filename = self.outbox / f"transducer-{submission.sequence:09d}.pt"
            with atomic_binary_writer(filename) as handle:
                torch.save(submission.state_dict(), handle)
            sync.pending = {
                "filename": str(filename),
                "sequence": submission.sequence,
                "version": submission.base_version,
                "steps": LOCAL_STEPS,
                "future": None,
            }
            sync.pending["future"] = self.control_io.submit(
                self._upload_transducer, dict(sync.pending)
            )
        newer = (
            self.remote_transducer_version > sync.version
            if self.remote_transducer_version is not None
            else steps % self.settings.poll_steps == 0
        )
        if ready is None and sync.download is None and (sync.force_snapshot or newer):
            sync.download = self.control_io.submit(
                self.fetch_transducer, -1 if sync.force_snapshot else sync.version
            )
        return errors, ready

    def _kb_step(self, steps: int, tokens: int, bank: Any) -> tuple[int, Any]:
        """Advance KB uploads/polls. Returns (errors, ready (version, path, kind) or None)."""
        from megatoken.training.diloco_kb import save_kb_update

        kb = self.kb
        errors = 0
        ready = None
        kb.steps_since += 1
        kb.tokens_since += tokens
        entry = kb.pending
        if entry is not None and entry["future"] is not None and entry["future"].done():
            try:
                entry["future"].result()
                Path(entry["filename"]).unlink(missing_ok=True)
                kb.pending = None
                kb.awaiting_global = True
            except RemoteError as exc:
                if exc.status == 409 and "Stale" in str(exc):
                    # The baseline aged out: drop it and reinstall the canonical bank.
                    Path(entry["filename"]).unlink(missing_ok=True)
                    kb.pending = None
                    self._kb_force_fetch = True
                else:
                    errors += 1
                    entry["future"] = None
                    entry["retry_after"] = steps + min(64, 2 ** min(6, entry.get("attempts", 0)))
                    entry["attempts"] = entry.get("attempts", 0) + 1
                    self.last_progress_error = str(exc)
            except (OSError, TimeoutError, ValueError) as exc:
                errors += 1
                entry["future"] = None
                entry["retry_after"] = steps + min(64, 2 ** min(6, entry.get("attempts", 0)))
                entry["attempts"] = entry.get("attempts", 0) + 1
                self.last_progress_error = str(exc)
        if (
            kb.pending is not None
            and kb.pending["future"] is None
            and (steps >= kb.pending["retry_after"])
        ):
            kb.pending["future"] = self.control_io.submit(self._upload_kb, dict(kb.pending))
        if kb.download is not None and kb.download.done():
            try:
                result = kb.download.result()
                if result is not None and result[0] > kb.version:
                    ready = result
                elif result is not None:
                    result[1].unlink(missing_ok=True)
                self._kb_force_fetch = False
            except (OSError, TimeoutError, ValueError, RemoteError) as exc:
                errors += 1
                self.last_progress_error = str(exc)
            kb.download = None
        if (
            kb.steps_since >= self.settings.local_steps
            and kb.pending is None
            and not kb.awaiting_global
        ):
            rows = kb.deltas(bank)
            if rows or kb.demand:
                sequence = kb.sequence + 1
                filename = self.outbox / f"kb-{sequence:09d}.safetensors"
                save_kb_update(filename, rows, kb.demand, wire_dtype=self.settings.wire_dtype)
                kb.sequence = sequence
                kb.demand = []
                kb.pending = {
                    "filename": str(filename),
                    "sequence": sequence,
                    "version": kb.version,
                    "steps": kb.steps_since,
                    "tokens": max(1, kb.tokens_since),
                    "future": None,
                    "retry_after": 0,
                }
                kb.pending["future"] = self.control_io.submit(self._upload_kb, dict(kb.pending))
            kb.uploaded(bank)
        # Fetch as soon as the coordinator reports a newer canonical version;
        # without status information fall back to periodic polling.
        newer = (
            self.remote_kb_version > kb.version
            if self.remote_kb_version is not None
            else steps % self.settings.poll_steps == 0
        )
        if ready is None and kb.download is None and (self._kb_force_fetch or newer):
            kb.download = self.control_io.submit(
                self._fetch_kb,
                -1 if self._kb_force_fetch else kb.version,
                kb.trained_rows(),
            )
        return errors, ready

    def _send_async(self, index: int) -> None:
        entry = self.pending[index]
        entry["future"] = self.io.submit(self._upload, index, dict(entry))

    def _write_outbox(self, entry: dict) -> Path:
        filename = Path(entry["filename"])
        if not filename.exists():
            staging = filename.with_suffix(".partial")
            save_tensors(staging, {"delta": entry["delta"]})
            os.replace(staging, filename)
        return filename

    def _queue(self, index: int, optimizer: Any) -> None:
        if index in self.pending or index in self.awaiting_global or self.versions[index] < 0:
            return
        if self.steps_since[index] <= 0 or self.tokens_since[index] <= 0:
            return
        baseline = self.baselines[index]
        assert baseline is not None
        current = pack_fragment(optimizer.masters, self.fragments[index])
        delta = baseline - current
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite local DiLoCo pseudogradient")
        sequence = self.sequences[index] + 1
        self.sequences[index] = sequence
        self.pending[index] = {
            "filename": str(self.outbox / f"{index:05d}-{sequence:09d}.safetensors"),
            "sequence": sequence,
            "version": self.versions[index],
            "steps": self.steps_since[index],
            "tokens": self.tokens_since[index],
            # Serialized by the I/O lane; the training thread never touches disk.
            "delta": delta.to(
                torch.bfloat16 if self.settings.wire_dtype == "bf16" else torch.float32
            ),
            "future": None,
            "retry_after": 0,
        }
        # Progress from here on is measured against the upload point, so the
        # steps trained while this round is in flight survive the global install.
        self.baselines[index] = current
        self.steps_since[index] = 0
        self.tokens_since[index] = 0
        self._send_async(index)

    def after_step(
        self,
        optimizer: Any,
        *,
        steps: int,
        tokens: int,
        bank: Any = None,
        transducer: Any = None,
    ) -> dict:
        """Called *after* each complete optimizer step; never blocks on network.

        With an attached KB, ``bank`` is the learner's local SQLite bank and the
        result's ``kb_snapshot`` is a downloaded (version, path, kind) for the
        caller to apply between optimizer steps; kind is "changes" or "snapshot".
        """
        kb_errors, kb_ready = (
            self._kb_step(steps, tokens, bank) if self.kb is not None else (0, None)
        )
        transducer_errors, transducer_ready = (
            self._transducer_step(steps, transducer) if self.transducer is not None else (0, None)
        )
        kb_errors += transducer_errors
        self.local_steps = steps
        self.local_tokens += tokens
        for i in range(len(self.fragments)):
            self.steps_since[i] += 1
            self.tokens_since[i] += tokens
        errors = 0
        for index, entry in list(self.pending.items()):
            future = entry["future"]
            if future is None or not future.done():
                continue
            try:
                future.result()
                Path(entry["filename"]).unlink(missing_ok=True)
                del self.pending[index]
                self.awaiting_global.add(index)
                self.offline_steps = 0
            except RemoteError as exc:
                # Stale proposals are invalidated; the next pull overwrites
                # this fragment and reestablishes a legitimate baseline.
                if exc.status == 409 and "Stale" in str(exc):
                    Path(entry["filename"]).unlink(missing_ok=True)
                    del self.pending[index]
                    self.downloads[index] = self.io.submit(self._fetch, index, -1)
                else:
                    errors += 1
                    entry["future"] = None
                    entry["retry_after"] = steps + min(64, 2 ** min(6, entry.get("attempts", 0)))
                    entry["attempts"] = entry.get("attempts", 0) + 1
            except (OSError, TimeoutError, ValueError) as exc:
                errors += 1
                entry["future"] = None
                entry["retry_after"] = steps + min(64, 2 ** min(6, entry.get("attempts", 0)))
                entry["attempts"] = entry.get("attempts", 0) + 1
                self.last_progress_error = str(exc)
        for index, entry in list(self.pending.items()):
            if entry["future"] is None and steps >= entry["retry_after"]:
                self._send_async(index)

        updated = False
        for index, future in list(self.downloads.items()):
            if not future.done():
                continue
            try:
                updated = self._install(index, future.result(), optimizer) or updated
                self.offline_steps = 0
            except (OSError, TimeoutError, ValueError, RemoteError) as exc:
                errors += 1
                self.last_progress_error = str(exc)
            del self.downloads[index]
        if self.remote_versions is not None:
            # Status-driven: fetch exactly the fragments the coordinator advanced.
            for index, remote in enumerate(self.remote_versions):
                if remote > self.versions[index] and index not in self.downloads:
                    self.downloads[index] = self.io.submit(self._fetch, index, self.versions[index])
        elif steps - self.last_poll_step >= self.settings.poll_steps:
            self.last_poll_step = steps
            for _ in range(min(self.settings.io_workers, len(self.fragments))):
                index = self._poll_cursor % len(self.fragments)
                self._poll_cursor += 1
                if index not in self.downloads:
                    self.downloads[index] = self.io.submit(self._fetch, index, self.versions[index])

        if self.heartbeat is not None and self.heartbeat.done():
            try:
                self._update_status(self.heartbeat.result())
                self.offline_steps = 0
            except (OSError, TimeoutError, ValueError, RemoteError) as exc:
                errors += 1
                self.last_progress_error = str(exc)
            self.heartbeat = None
        if (
            self.heartbeat is None
            and steps - self.last_heartbeat_step >= self.settings.heartbeat_steps
        ):
            self.last_heartbeat_step = steps
            self.heartbeat = self.control_io.submit(
                self.transport.json,
                "POST",
                "/v1/progress",
                {
                    "worker": self.worker,
                    "session": self.session,
                    "steps": self.local_steps,
                    "tokens": self.local_tokens,
                },
            )

        # H steps per fragment with stable staggered offsets; no global barrier.
        for index in range(len(self.fragments)):
            offset = index * self.settings.local_steps // len(self.fragments)
            first_due = self.settings.local_steps + offset
            if (
                self.local_steps >= first_due
                and self.steps_since[index] >= self.settings.local_steps
                and index not in self.pending
                and index not in self.awaiting_global
            ):
                self._queue(index, optimizer)
        errors += kb_errors
        if errors:
            self.offline_steps += 1
        if self.offline_steps > self.settings.max_offline_steps:
            raise RuntimeError(
                f"DiLoCo coordinator unreachable for too many steps: {self.last_progress_error}"
            )
        return {
            "updated": int(updated),
            "pending": len(self.pending),
            "merges": self.merges,
            **(
                {
                    "kb_version": self.kb.version,
                    "kb_pending": int(self.kb.pending is not None),
                    "kb_snapshot": kb_ready,
                }
                if self.kb is not None
                else {}
            ),
            **(
                {
                    "transducer_version": self.transducer.version,
                    "transducer_pending": int(self.transducer.pending is not None),
                    "transducer_ready": transducer_ready,
                }
                if self.transducer is not None
                else {}
            ),
        }

    def state_dict(self) -> dict:
        """Checkpoint baseline+cursor, not live Futures or socket objects."""
        return {
            "protocol": 1,
            "fingerprint": self.manifest["fingerprint"],
            "worker": self.worker,
            "session": self.session,
            "versions": list(self.versions),
            "baselines": [tensor.detach().clone() for tensor in self.baselines],
            "steps_since": list(self.steps_since),
            "tokens_since": list(self.tokens_since),
            "sequences": list(self.sequences),
            "local_steps": self.local_steps,
            "local_tokens": self.local_tokens,
            "pending": {
                str(k): {
                    key: value for key, value in entry.items() if key not in ("future", "delta")
                }
                for k, entry in self.pending.items()
            },
            "awaiting_global": sorted(self.awaiting_global),
            "pending_payloads": {
                str(k): (
                    entry["delta"] if "delta" in entry else load_tensors(entry["filename"])["delta"]
                )
                for k, entry in self.pending.items()
            },
            "kb": None if self.kb is None else self.kb.state_dict(),
            "transducer": None if self.transducer is None else self.transducer.state_dict(),
        }

    def metrics(self) -> dict:
        return {
            "worker": self.worker,
            "global_tokens": self.global_tokens,
            "local_tokens": self.local_tokens,
            "versions": self.versions,
            "pending_uploads": len(self.pending),
            "inflight_downloads": len(self.downloads),
            "merges": self.merges,
            "offline_steps": self.offline_steps,
        }

    def close(self) -> None:
        self.io.shutdown(wait=True, cancel_futures=False)
        self.control_io.shutdown(wait=True, cancel_futures=False)
        if (
            self.transducer is not None
            and self.transducer.download is not None
            and self.transducer.download.done()
        ):
            try:
                result = self.transducer.download.result()
            except Exception:
                result = None
            if result is not None:
                result[1].unlink(missing_ok=True)
        if self.kb is not None and self.kb.download is not None and self.kb.download.done():
            try:
                result = self.kb.download.result()
            except Exception:
                result = None
            if result is not None:
                result[1].unlink(missing_ok=True)
