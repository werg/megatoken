"""Durable, fail-closed native request journal (storage only).

This module records exact requests before dispatch and exact response bytes before
returning them. It deliberately does not restore native model state or authorize
replay/commit in another runtime.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Callable
from pathlib import Path


class NativeRequestJournalError(RuntimeError):
    pass


class NativeRequestBusy(NativeRequestJournalError):
    pass


class NativeRequestConflict(NativeRequestJournalError):
    pass


class NativeRequestOutcomeUnknown(NativeRequestJournalError):
    pass


_SCHEMA_ID = "semantic-native-request-identity/1"
_SCHEMA_INTENT = "semantic-native-request-intent/1"
_SCHEMA_OUTCOME = "semantic-native-request-outcome/1"
_SCHEMA_SUPERSEDED = "semantic-native-request-superseded/1"
_SCOPE_KEYS = ("session_id", "stream_id", "session_generation")
_IDENTITY_KEYS = {
    "schema",
    "session_id",
    "stream_id",
    "session_generation",
    "ordinal",
    "operation",
    "model_identity",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise NativeRequestJournalError("record is not canonical JSON data") from exc


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_identity(identity: object) -> dict:
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise NativeRequestJournalError("invalid native request identity fields")
    item = dict(identity)
    if item["schema"] != _SCHEMA_ID:
        raise NativeRequestJournalError("unsupported native request identity schema")
    for key in ("session_id", "stream_id", "operation"):
        if not isinstance(item[key], str) or not _ID.fullmatch(item[key]):
            raise NativeRequestJournalError(f"invalid {key}")
    for key in ("session_generation", "ordinal"):
        if type(item[key]) is not int or item[key] < 1 or item[key] > 2**53 - 1:
            raise NativeRequestJournalError(f"invalid {key}")
    model = item["model_identity"]
    if not isinstance(model, dict) or set(model) not in (
        {"checkpoint", "tokenizer"},
        {"checkpoint", "tokenizer", "training"},
    ):
        raise NativeRequestJournalError("model identity requires checkpoint and tokenizer hashes")
    if any(not isinstance(value, str) or not _HEX64.fullmatch(value) for value in model.values()):
        raise NativeRequestJournalError("model identities must be lowercase SHA-256 hashes")
    if len(_canonical(item)) > 4096:
        raise NativeRequestJournalError("native request identity exceeds size limit")
    return item


class DurableNativeRequestJournal:
    """POSIX local journal with immutable ordinal slots and non-stealable flock.

    The callback is invoked exactly once only after the intent is durably linked.
    An intent without its outcome is permanently unknown and is never dispatched
    again by this API. A completed slot returns its exact stored response bytes.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        max_requests: int = 4096,
        max_request_bytes: int = 1 << 20,
        max_result_bytes: int = 1 << 20,
        max_log_bytes: int = 64 << 20,
    ):
        if os.name != "posix" or not hasattr(fcntl, "flock"):
            raise NativeRequestJournalError("POSIX flock is required")
        if any(
            type(x) is not int or x < 1
            for x in (max_requests, max_request_bytes, max_result_bytes, max_log_bytes)
        ):
            raise ValueError("journal limits must be positive integers")
        self.root = Path(directory)
        self.max_requests = max_requests
        self.max_request_bytes = max_request_bytes
        self.max_result_bytes = max_result_bytes
        self.max_log_bytes = max_log_bytes
        self._ensure_private_dir(self.root)

    def _after_directory_fsync(self, path: Path, reason: str) -> None:
        """Private fault-injection seam; production implementation has no side effects."""

    def _sync_directory(self, path: Path, reason: str) -> None:
        fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            st = os.fstat(fd)
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() and path in (self.root,):
                raise NativeRequestJournalError("unsafe journal directory during sync")
            os.fsync(fd)
        finally:
            os.close(fd)
        self._after_directory_fsync(path, reason)

    def _ensure_private_dir(self, path: Path) -> None:
        absolute = path.absolute()
        cursor = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            parent = cursor
            cursor = cursor / part
            try:
                component = cursor.lstat()
            except FileNotFoundError:
                try:
                    cursor.mkdir(mode=0o700)
                    self._sync_directory(parent, "mkdir-parent")
                    self._sync_directory(cursor, "mkdir-self")
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise NativeRequestJournalError(
                        "cannot create and sync journal directory"
                    ) from exc
                try:
                    component = cursor.lstat()
                except OSError as exc:
                    raise NativeRequestJournalError("journal directory disappeared") from exc
            if stat.S_ISLNK(component.st_mode) or not stat.S_ISDIR(component.st_mode):
                raise NativeRequestJournalError(
                    "journal path may not traverse symlinks or non-directories"
                )
        st = path.lstat()
        if st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) & 0o077:
            raise NativeRequestJournalError(
                "journal directory must be owner-only and owned by this user"
            )
        # Re-ack both the directory entry and the directory itself before relying on
        # a scope for callback dispatch, including when another process created it.
        self._sync_directory(path.parent, "dir-parent")
        self._sync_directory(path, "dir-self")

    def _scope(self, identity: dict) -> tuple[Path, str]:
        scope = {key: identity[key] for key in _SCOPE_KEYS}
        scope_id = _digest(_canonical(scope))
        path = self.root / scope_id
        self._ensure_private_dir(path)
        return path, scope_id

    def _paths(self, directory: Path, ordinal: int) -> tuple[Path, Path]:
        return directory / f"{ordinal:020d}.intent.json", directory / f"{ordinal:020d}.outcome.json"

    def _used_bytes(self, directory: Path) -> int:
        total = 0
        for item in directory.iterdir():
            if not re.fullmatch(r"[0-9]{20}\.(?:intent|outcome|superseded)\.json", item.name):
                continue
            try:
                st = item.lstat()
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(st.st_mode)
                or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) & 0o077
                or st.st_nlink != 1
            ):
                raise NativeRequestJournalError("unsafe journal file")
            total += st.st_size
            if total > self.max_log_bytes:
                raise NativeRequestJournalError("journal scope exceeds byte limit")
        return total

    def _clean_orphan_temps(self, directory: Path) -> None:
        removed = False
        for item in directory.iterdir():
            if not item.name.startswith(".tmp-"):
                continue
            if not re.fullmatch(r"\.tmp-[0-9a-f]{32}", item.name):
                raise NativeRequestJournalError("unrecognized temporary journal entry")
            st = item.lstat()
            if (
                not stat.S_ISREG(st.st_mode)
                or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) & 0o077
                or st.st_nlink not in (1, 2)
            ):
                raise NativeRequestJournalError("unsafe temporary journal entry")
            if st.st_nlink == 2:
                matches = []
                for candidate in directory.iterdir():
                    if candidate.name.startswith(".tmp-"):
                        continue
                    if not re.fullmatch(
                        r"[0-9]{20}\.(?:intent|outcome|superseded)\.json", candidate.name
                    ):
                        continue
                    try:
                        target_st = candidate.lstat()
                    except FileNotFoundError:
                        continue
                    if (target_st.st_dev, target_st.st_ino) == (st.st_dev, st.st_ino):
                        matches.append(candidate)
                if len(matches) != 1:
                    raise NativeRequestJournalError(
                        "linked temporary has no unique official journal destination"
                    )
                target_st = matches[0].lstat()
                if (
                    not stat.S_ISREG(target_st.st_mode)
                    or target_st.st_uid != os.geteuid()
                    or stat.S_IMODE(target_st.st_mode) & 0o077
                    or target_st.st_nlink != 2
                ):
                    raise NativeRequestJournalError("unsafe linked journal destination")
            item.unlink()
            removed = True
        if removed:
            self._sync_directory(directory, "temp-cleanup")

    def _lock(self, directory: Path):
        lock_path = directory / "scope.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags, 0o600)
            st = os.fstat(fd)
            if (
                not stat.S_ISREG(st.st_mode)
                or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) & 0o077
                or st.st_nlink != 1
            ):
                os.close(fd)
                raise NativeRequestJournalError("unsafe scope lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError as exc:
            try:
                os.close(fd)
            except (OSError, UnboundLocalError):
                pass
            raise NativeRequestBusy("native request scope is held by another process") from exc
        except OSError as exc:
            try:
                os.close(fd)
            except (OSError, UnboundLocalError):
                pass
            raise NativeRequestJournalError("cannot securely lock native request scope") from exc

    def _locked(self, identity: dict):
        class Guard:
            def __init__(self, outer, ident):
                self.outer, self.ident, self.fd, self.directory, self.scope_id = (
                    outer,
                    ident,
                    None,
                    None,
                    None,
                )

            def __enter__(self):
                self.directory, self.scope_id = self.outer._scope(self.ident)
                self.fd = self.outer._lock(self.directory)
                try:
                    self.outer._clean_orphan_temps(self.directory)
                except BaseException:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                    os.close(self.fd)
                    self.fd = None
                    raise
                return self.directory, self.scope_id

            def __exit__(self, *_):
                if self.fd is not None:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                    os.close(self.fd)

        return Guard(self, identity)

    def _read_record(self, path: Path, limit: int) -> dict | None:
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            self._sync_directory(path.parent, "record-absent")
            return None
        except OSError as exc:
            raise NativeRequestJournalError("cannot securely read journal record") from exc
        try:
            st = os.fstat(fd)
            if (
                not stat.S_ISREG(st.st_mode)
                or st.st_uid != os.geteuid()
                or stat.S_IMODE(st.st_mode) & 0o077
                or st.st_nlink != 1
                or st.st_size > limit
            ):
                raise NativeRequestJournalError("unsafe or oversized journal record")
            chunks = []
            remaining = limit + 1
            while remaining:
                block = os.read(fd, min(65536, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            raw = b"".join(chunks)
            if len(raw) > limit:
                raise NativeRequestJournalError("oversized journal record")
            os.fsync(fd)
            self._sync_directory(path.parent, "record-read")
        finally:
            os.close(fd)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NativeRequestJournalError("corrupt journal record") from exc
        if not isinstance(value, dict) or _canonical(value) != raw:
            raise NativeRequestJournalError("noncanonical journal record")
        return value

    def _publish(self, path: Path, record: dict, limit: int) -> None:
        raw = _canonical(record)
        if len(raw) > limit:
            raise NativeRequestJournalError("journal record exceeds size limit")
        temp = path.parent / (".tmp-" + secrets.token_hex(16))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temp, flags, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise NativeRequestJournalError("short journal write")
                view = view[written:]
            os.fsync(fd)
            self._after_temp_fsync(path.name, temp)
        finally:
            os.close(fd)
        try:
            os.link(temp, path, follow_symlinks=False)
        except FileExistsError as exc:
            try:
                os.unlink(temp)
            except FileNotFoundError:
                pass
            raise NativeRequestConflict("immutable journal slot already exists") from exc
        except OSError:
            try:
                os.unlink(temp)
            except FileNotFoundError:
                pass
            raise
        self._after_link_before_temp_unlink(path.name, path, temp)
        os.unlink(temp)
        self._sync_directory(path.parent, "record-publish")
        self._after_published(path.name, path)

    def _after_temp_fsync(self, target_name: str, temp_path: Path) -> None:
        """Private fault-injection seam; production implementation has no side effects."""

    def _after_published(self, target_name: str, path: Path) -> None:
        """Private fault-injection seam used to model a lost acknowledgement."""

    def _after_link_before_temp_unlink(self, target_name: str, path: Path, temp: Path) -> None:
        """Private fault seam for a crash after link but before temp cleanup."""

    def _request_id(self, scope_id: str, ordinal: int) -> str:
        return _digest(
            _canonical(
                {"schema": "semantic-native-request-slot/1", "scope": scope_id, "ordinal": ordinal}
            )
        )

    def _check_slot(
        self,
        directory: Path,
        scope_id: str,
        identity: dict,
        request: bytes,
        previous_outcome: str | None,
    ):
        ordinal = identity["ordinal"]
        intent_path, outcome_path = self._paths(directory, ordinal)
        intent = self._read_record(intent_path, self.max_request_bytes * 2 + 8192)
        if intent is None:
            if (
                os.path.lexists(outcome_path)
                or self._read_record(outcome_path, self.max_result_bytes * 2 + 8192) is not None
                or os.path.lexists(directory / f"{ordinal:020d}.superseded.json")
            ):
                raise NativeRequestJournalError("orphan outcome exists without its intent")
            if any(
                int(match.group(1)) > ordinal
                for item in directory.iterdir()
                if (
                    match := re.fullmatch(
                        r"([0-9]{20})\.(?:intent|outcome|superseded)\.json", item.name
                    )
                )
            ):
                raise NativeRequestJournalError(
                    "missing request slot precedes retained higher ordinal"
                )
            return None, None
        expected_intent = {
            "schema": _SCHEMA_INTENT,
            "request_id": self._request_id(scope_id, ordinal),
            "identity": identity,
            "request_sha256": _digest(request),
            "request_b64": base64.b64encode(request).decode("ascii"),
            "previous_outcome_sha256": previous_outcome,
        }
        if intent != expected_intent:
            raise NativeRequestConflict("request slot identity or exact bytes differ")
        outcome = self._read_record(outcome_path, self.max_result_bytes * 2 + 8192)
        if outcome is None:
            if os.path.lexists(directory / f"{ordinal:020d}.superseded.json"):
                raise NativeRequestJournalError("superseded marker has no original outcome")
            return intent, None
        try:
            result = base64.b64decode(outcome["result_b64"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise NativeRequestJournalError("invalid stored outcome bytes") from exc
        expected_outcome = {
            "schema": _SCHEMA_OUTCOME,
            "request_id": self._request_id(scope_id, ordinal),
            "request_sha256": _digest(request),
            "result_sha256": _digest(result),
            "result_b64": base64.b64encode(result).decode("ascii"),
        }
        if outcome != expected_outcome or len(result) > self.max_result_bytes:
            raise NativeRequestJournalError("stored outcome integrity failure")
        self._superseded(directory, identity, request, outcome)
        return intent, result

    def _superseded(self, directory, identity, request_bytes, outcome):
        marker = self._read_record(directory / f"{identity['ordinal']:020d}.superseded.json", 8192)
        if marker is None:
            return None
        if (
            type(marker) is not dict
            or set(marker)
            != {
                "schema",
                "request_id",
                "request_sha256",
                "outcome_sha256",
                "mutation_sha256",
                "operation",
            }
            or marker.get("schema") != _SCHEMA_SUPERSEDED
            or marker.get("request_id") != outcome["request_id"]
            or marker.get("request_sha256") != _digest(request_bytes)
            or marker.get("outcome_sha256") != _digest(_canonical(outcome))
            or type(marker.get("mutation_sha256")) is not str
            or not _HEX64.fullmatch(marker["mutation_sha256"])
            or type(marker.get("operation")) is not str
            or not _ID.fullmatch(marker["operation"])
        ):
            raise NativeRequestJournalError("superseded native request marker differs")
        return marker

    def _scan_prior(self, directory: Path, scope_id: str, identity: dict) -> tuple[str | None, int]:
        previous = None
        highest = 0
        for ordinal in range(1, identity["ordinal"]):
            intent_path, outcome_path = self._paths(directory, ordinal)
            if not os.path.lexists(intent_path):
                for item in directory.iterdir():
                    match = re.fullmatch(
                        r"([0-9]{20})\.(?:intent|outcome|superseded)\.json", item.name
                    )
                    if match and int(match.group(1)) > ordinal:
                        raise NativeRequestJournalError(
                            "missing request intent precedes retained later records"
                        )
                break
            intent = self._read_record(intent_path, self.max_request_bytes * 2 + 8192)
            if (
                not isinstance(intent, dict)
                or set(intent)
                != {
                    "schema",
                    "request_id",
                    "identity",
                    "request_sha256",
                    "request_b64",
                    "previous_outcome_sha256",
                }
                or intent.get("schema") != _SCHEMA_INTENT
                or intent.get("request_id") != self._request_id(scope_id, ordinal)
            ):
                raise NativeRequestJournalError("corrupt prior request intent")
            prior_identity = _validate_identity(intent["identity"])
            if (
                prior_identity["ordinal"] != ordinal
                or any(prior_identity[key] != identity[key] for key in _SCOPE_KEYS)
                or intent.get("previous_outcome_sha256") != previous
            ):
                raise NativeRequestJournalError("broken request history link")
            if prior_identity["model_identity"] != identity["model_identity"]:
                raise NativeRequestConflict("model identity changed within session generation")
            try:
                prior_request = base64.b64decode(intent["request_b64"], validate=True)
            except (ValueError, TypeError) as exc:
                raise NativeRequestJournalError("invalid prior request bytes") from exc
            if (
                len(prior_request) > self.max_request_bytes
                or intent.get("request_b64") != base64.b64encode(prior_request).decode("ascii")
                or intent.get("request_sha256") != _digest(prior_request)
            ):
                raise NativeRequestJournalError("prior request integrity failure")
            outcome = self._read_record(outcome_path, self.max_result_bytes * 2 + 8192)
            if outcome is None:
                raise NativeRequestOutcomeUnknown("earlier request outcome is unknown")
            try:
                result = base64.b64decode(outcome["result_b64"], validate=True)
            except (KeyError, ValueError, TypeError) as exc:
                raise NativeRequestJournalError("invalid prior outcome") from exc
            if (
                not isinstance(outcome, dict)
                or set(outcome)
                != {"schema", "request_id", "request_sha256", "result_sha256", "result_b64"}
                or outcome.get("result_b64") != base64.b64encode(result).decode("ascii")
                or outcome.get("request_id") != self._request_id(scope_id, ordinal)
                or outcome.get("request_sha256") != intent.get("request_sha256")
                or outcome.get("result_sha256") != _digest(result)
                or outcome.get("schema") != _SCHEMA_OUTCOME
                or len(result) > self.max_result_bytes
            ):
                raise NativeRequestJournalError("prior outcome integrity failure")
            self._superseded(directory, prior_identity, prior_request, outcome)
            previous = _digest(_canonical(outcome))
            highest = ordinal
        return previous, highest

    def execute(
        self, identity: dict, request_bytes: bytes, callback: Callable[[bytes], bytes]
    ) -> bytes:
        identity = _validate_identity(identity)
        if not isinstance(request_bytes, bytes) or len(request_bytes) > self.max_request_bytes:
            raise NativeRequestJournalError("request must be bounded exact bytes")
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._locked(identity) as (directory, scope_id):
            previous, highest = self._scan_prior(directory, scope_id, identity)
            ordinal = identity["ordinal"]
            if ordinal > self.max_requests:
                raise NativeRequestJournalError("request count limit exceeded")
            if ordinal > highest + 1:
                raise NativeRequestConflict("request ordinal is not the next slot")
            intent, result = self._check_slot(
                directory, scope_id, identity, request_bytes, previous
            )
            if intent is not None:
                if result is None:
                    raise NativeRequestOutcomeUnknown("admitted request has no durable outcome")
                return result
            if ordinal != highest + 1:
                raise NativeRequestConflict("request ordinal is not the next slot")
            if (
                self._used_bytes(directory)
                + len(
                    _canonical(
                        {
                            "schema": _SCHEMA_INTENT,
                            "request_id": self._request_id(scope_id, ordinal),
                            "identity": identity,
                            "request_sha256": _digest(request_bytes),
                            "request_b64": base64.b64encode(request_bytes).decode("ascii"),
                            "previous_outcome_sha256": previous,
                        }
                    )
                )
                > self.max_log_bytes
            ):
                raise NativeRequestJournalError("journal scope byte limit exceeded")
            intent_path, _ = self._paths(directory, ordinal)
            record = {
                "schema": _SCHEMA_INTENT,
                "request_id": self._request_id(scope_id, ordinal),
                "identity": identity,
                "request_sha256": _digest(request_bytes),
                "request_b64": base64.b64encode(request_bytes).decode("ascii"),
                "previous_outcome_sha256": previous,
            }
            self._publish(intent_path, record, self.max_request_bytes * 2 + 8192)
            # The callback can run only after intent publication and directory fsync.
            result_bytes = callback(request_bytes)
            if not isinstance(result_bytes, bytes) or len(result_bytes) > self.max_result_bytes:
                raise NativeRequestJournalError("callback must return bounded exact bytes")
            _, outcome_path = self._paths(directory, ordinal)
            outcome_record = {
                "schema": _SCHEMA_OUTCOME,
                "request_id": self._request_id(scope_id, ordinal),
                "request_sha256": _digest(request_bytes),
                "result_sha256": _digest(result_bytes),
                "result_b64": base64.b64encode(result_bytes).decode("ascii"),
            }
            if self._used_bytes(directory) + len(_canonical(outcome_record)) > self.max_log_bytes:
                raise NativeRequestJournalError("journal scope byte limit exceeded")
            self._publish(outcome_path, outcome_record, self.max_result_bytes * 2 + 8192)
            return result_bytes

    def inspect(
        self, identity: dict, request_bytes: bytes, *, require_latest: bool = False
    ) -> dict:
        identity = _validate_identity(identity)
        if not isinstance(request_bytes, bytes) or len(request_bytes) > self.max_request_bytes:
            raise NativeRequestJournalError("request must be bounded exact bytes")
        with self._locked(identity) as (directory, scope_id):
            previous, highest = self._scan_prior(directory, scope_id, identity)
            ordinal = identity["ordinal"]
            if require_latest and any(
                int(match.group(1)) > ordinal
                for item in directory.iterdir()
                if (
                    match := re.fullmatch(
                        r"([0-9]{20})\.(?:intent|outcome|superseded)\.json", item.name
                    )
                )
            ):
                raise NativeRequestConflict(
                    "a later native request exists; earlier poststate cannot recover"
                )
            intent, result = self._check_slot(
                directory, scope_id, identity, request_bytes, previous
            )
            if intent is None:
                if ordinal <= highest:
                    raise NativeRequestJournalError("missing request slot in retained history")
                return {
                    "status": "never-admitted",
                    "request_id": self._request_id(scope_id, ordinal),
                }
            if result is None:
                return {"status": "unknown", "request_id": self._request_id(scope_id, ordinal)}
            _, outcome_path = self._paths(directory, ordinal)
            outcome = self._read_record(outcome_path, self.max_result_bytes * 2 + 8192)
            if (
                require_latest
                and self._superseded(directory, identity, request_bytes, outcome) is not None
            ):
                raise NativeRequestConflict(
                    "native poststate was superseded before a later mutation"
                )
            return {
                "status": "completed",
                "request_id": self._request_id(scope_id, ordinal),
                "result_bytes": result,
                "outcome_sha256": _digest(_canonical(outcome)),
            }

    def has_history(self, identity: dict) -> bool:
        """Check exact scope occupancy before a fresh process can mutate state."""
        identity = _validate_identity(identity)
        with self._locked(identity) as (directory, _scope_id):
            self._used_bytes(directory)
            return any(
                re.fullmatch(r"[0-9]{20}\.(?:intent|outcome|superseded)\.json", item.name)
                for item in directory.iterdir()
            )

    def supersede(self, identity: dict, request_bytes: bytes, mutation: dict) -> None:
        """Durably revoke state-recovery eligibility before an unjournaled mutation.

        The completed response remains readable as historical storage evidence.
        A later ordinal can establish a new independently acknowledged poststate.
        """
        identity = _validate_identity(identity)
        if not isinstance(request_bytes, bytes) or len(request_bytes) > self.max_request_bytes:
            raise NativeRequestJournalError("request must be bounded exact bytes")
        if (
            type(mutation) is not dict
            or type(mutation.get("op")) is not str
            or not _ID.fullmatch(mutation["op"])
        ):
            raise NativeRequestJournalError("supersession requires an exact mutation request")
        mutation_digest = _digest(_canonical(mutation))
        with self._locked(identity) as (directory, scope_id):
            previous, _ = self._scan_prior(directory, scope_id, identity)
            intent, result = self._check_slot(
                directory, scope_id, identity, request_bytes, previous
            )
            if intent is None or result is None:
                raise NativeRequestOutcomeUnknown(
                    "only a completed native outcome may be superseded"
                )
            _, outcome_path = self._paths(directory, identity["ordinal"])
            outcome = self._read_record(outcome_path, self.max_result_bytes * 2 + 8192)
            if self._superseded(directory, identity, request_bytes, outcome) is not None:
                return
            marker = {
                "schema": _SCHEMA_SUPERSEDED,
                "request_id": outcome["request_id"],
                "request_sha256": _digest(request_bytes),
                "outcome_sha256": _digest(_canonical(outcome)),
                "mutation_sha256": mutation_digest,
                "operation": mutation["op"],
            }
            if self._used_bytes(directory) + len(_canonical(marker)) > self.max_log_bytes:
                raise NativeRequestJournalError("journal scope byte limit exceeded")
            path = directory / f"{identity['ordinal']:020d}.superseded.json"
            self._publish(path, marker, 8192)
            if self._superseded(directory, identity, request_bytes, outcome) != marker:
                raise NativeRequestJournalError("superseded marker durable readback differs")
