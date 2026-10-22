"""Persistent native-parser isolation with a watchdog over startup and pipe I/O.

Each preparation process reuses one worker, serializing concurrent requests and
caching loaded grammars. Timeouts, crashes, and invalid evidence discard that
worker and return bounded lexical evidence for the affected source document.

The executable/module are fixed; corpus bytes never select commands, execute as
Python, or trigger grammar downloads. Registered provider factories are trusted
application code. This is crash isolation, not a security sandbox for providers.
"""

from __future__ import annotations

import atexit
import collections
import os
import pickle
import queue
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, BinaryIO

from .types import Boundary, Diagnostic, ParseLimits, ParseResult, Region

if TYPE_CHECKING:
    from .syntax import Scan

_HEADER = struct.Struct("!Q")
_MAX_MESSAGE = 64 << 20
_PROTOCOL_VERSION = 1
_OWNER_ENV = "MEGATOKEN_PARSER_OWNER_PID"
_LOG_LIMIT = 1 << 20  # The worker's stderr log is emptied before a request when it passes this.
_LOG_TAIL = 400  # Bytes of that log quoted in a failure diagnostic.
_CACHED_PAYLOAD = 4096  # Largest payload whose scan is remembered, and how many are.
_CACHED_SCANS = 1024


class _ProviderError(RuntimeError):
    """A trusted provider cannot be reconstructed in the isolated interpreter."""


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    pieces = bytearray()
    while len(pieces) < size:
        part = stream.read(size - len(pieces))
        if not part:
            raise EOFError("Native parser pipe closed")
        pieces.extend(part)
    return bytes(pieces)


def _receive(stream: BinaryIO):
    size = _HEADER.unpack(_read_exact(stream, _HEADER.size))[0]
    if size > _MAX_MESSAGE:
        raise ValueError("Native parser message exceeds the transport budget")
    # Only our locally spawned Python peer writes this private pipe. Corpus bytes
    # are nested bytes, never themselves passed to pickle.loads. Provider imports
    # have the same trust requirement as registering the factory in-process.
    return pickle.loads(_read_exact(stream, size))


def _send(stream: BinaryIO, payload: bytes) -> None:
    if len(payload) > _MAX_MESSAGE:
        raise ValueError("Native parser message exceeds the transport budget")
    for piece in (_HEADER.pack(len(payload)), payload):
        pending = memoryview(piece)
        while pending:
            written = stream.write(pending)
            if written is None or written <= 0:
                raise BrokenPipeError("Native parser pipe stopped accepting bytes")
            pending = pending[written:]
    stream.flush()


@dataclass
class _Request:
    payload: bytes
    ready: threading.Event = field(default_factory=threading.Event)
    value: object = None
    error: Exception | None = None


class _Worker:
    def __init__(self, providers: bytes):
        self.owner_pid = os.getpid()
        self.providers = providers
        self.requests: queue.Queue[_Request | None] = queue.Queue()
        self.ready = threading.Event()
        self.closed = threading.Event()
        # Parsing is deterministic: a payload scanned before (agent traces repeat file views
        # and command output constantly) is answered from here, for as long as this worker lives.
        self.scans: collections.OrderedDict[tuple, Scan] = collections.OrderedDict()
        self.close_lock = threading.Lock()
        self.startup_error: Exception | None = None
        self.process: subprocess.Popen | None = None
        self.process_closed = False
        # Native crashes explain themselves on stderr (assertion text, signal
        # names, grammar load errors). An anonymous append-only file keeps them
        # for the failure diagnostic without a pipe that could fill and stall.
        self.log = tempfile.TemporaryFile(mode="a+b")
        environment = os.environ.copy()
        # Module invocation works from notebooks/stdin as well as entry points.
        # Reproduce trusted application imports, including local grammar plugins;
        # no corpus filename or language hint ever enters this import path.
        environment["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(path) for path in sys.path if isinstance(path, str)
        )
        environment[_OWNER_ENV] = str(self.owner_pid)
        self.thread = threading.Thread(
            target=self._communicate,
            args=(environment,),
            name="megatoken-parser-pipe",
            daemon=True,
        )
        try:
            self.thread.start()
        except Exception:
            self.close()
            raise

    def _communicate(self, environment: dict[str, str]) -> None:
        try:
            # Linux's parent-death signal follows the *creating thread*. Spawn
            # on this persistent thread, not on a short-lived caller/pool thread.
            # Process creation is also inside the startup watchdog this way.
            self.process = subprocess.Popen(
                [sys.executable, "-m", "megatoken.data.parsing.workers", "--worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
                close_fds=True,
                start_new_session=os.name == "posix",
                env=environment,
            )
            if self.closed.is_set():
                # Process creation may finish after the caller timed out. It
                # still owns a cleanup obligation and must never leak that child.
                with self.close_lock:
                    self._close_process()
                return
            assert self.process.stdout is not None and self.process.stdin is not None
            _send(self.process.stdin, self.providers)
            hello = _receive(self.process.stdout)
            if isinstance(hello, tuple) and len(hello) == 2 and hello[0] == "provider_error":
                raise _ProviderError(hello[1])
            if hello != ("ready", _PROTOCOL_VERSION):
                raise RuntimeError("Invalid native parser startup handshake")
        except Exception as error:
            self.startup_error = error
        finally:
            self.ready.set()
        if self.startup_error is not None:
            return
        while not self.closed.is_set():
            request = self.requests.get()
            if request is None:
                return
            try:
                _send(self.process.stdin, request.payload)
                request.value = _receive(self.process.stdout)
            except Exception as error:
                request.error = error
            finally:
                request.ready.set()
            if request.error is not None:
                return

    def call(self, payload: bytes, timeout: float):
        # Both waits consume one deadline. Writes run on the daemon thread, so a
        # full OS pipe cannot defeat the watchdog by blocking the calling thread.
        deadline = time.monotonic() + timeout
        if self.closed.is_set():
            raise RuntimeError("Native parser worker is closed")
        if not self.ready.wait(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Native parser startup exceeded its request watchdog")
        if self.startup_error is not None:
            raise self.startup_error
        if os.fstat(self.log.fileno()).st_size > _LOG_LIMIT:
            self.log.truncate(0)
        request = _Request(payload)
        self.requests.put(request)
        if not request.ready.wait(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("Native parser exceeded its request watchdog")
        if request.error is not None:
            raise request.error
        return request.value

    def stderr_tail(self) -> str:
        """The last lines the worker wrote to stderr, for failure diagnostics."""
        try:
            size = os.fstat(self.log.fileno()).st_size
            raw = os.pread(self.log.fileno(), min(size, _LOG_TAIL), max(0, size - _LOG_TAIL))
        except (OSError, ValueError):
            return ""
        return " ".join(raw.decode("utf-8", "replace").split())

    def _stop(self, *, force: bool = False) -> None:
        assert self.process is not None
        try:
            if os.name == "posix":
                # A fresh session also lets cleanup stop provider descendants
                # that might otherwise retain a cancelled worker's pipe handles.
                os.killpg(self.process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                self.process.kill()
            else:
                self.process.terminate()
        except ProcessLookupError:
            pass

    def detach_after_fork(self) -> None:
        """Drop inherited raw pipes without touching parent locks or its child."""
        if self.process is None:
            return
        self.process.returncode = 0  # This Popen is not our child to wait for.
        for stream in (self.process.stdin, self.process.stdout, self.log):
            if stream is not None:
                stream.close()

    def _close_process(self) -> None:
        if self.process is None or self.process_closed:
            return
        self.process_closed = True
        try:
            self._stop()
            try:
                self.process.wait(timeout=0.15)
            except subprocess.TimeoutExpired:
                pass
            # Also stop descendants if the direct child already exited on TERM.
            self._stop(force=True)
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                # Bound shutdown even for an uninterruptible OS process. Popen
                # can reap it later; no subsequent request reuses this child.
                pass
        finally:
            # Raw FileIO has no buffered-I/O lock that a blocked communication
            # thread could retain while this thread tries to close its stream.
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()

    def close(self) -> None:
        if self.owner_pid != os.getpid():
            self.detach_after_fork()
            return
        if not self.close_lock.acquire(blocking=False):
            return
        try:
            self.closed.set()
            self.requests.put(None)
            self._close_process()
        finally:
            self.close_lock.release()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.25)
        if not self.thread.is_alive():
            self.log.close()


_worker: _Worker | None = None
_mutex = threading.Lock()


def _after_fork() -> None:
    global _worker, _mutex
    inherited, _worker = _worker, None
    # Another parent thread may have held the mutex at fork. Child preparation
    # starts its own worker and must not retain the parent's pipe descriptors.
    _mutex = threading.Lock()
    if inherited is not None:
        inherited.detach_after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def close_parser_worker() -> None:
    """Release this process's native worker; later parsing starts a fresh one."""
    global _worker
    with _mutex:
        previous, _worker = _worker, None
        if previous is not None:
            previous.close()


def _shutdown() -> None:
    # Interpreter exit must not wait on a mutex held by a daemon caller whose
    # configured watchdog could be minutes. Normal API cleanup remains serialized.
    if _worker is not None:
        _worker.close()


atexit.register(_shutdown)


def _provider_snapshot() -> bytes:
    from .registry import parser_provider_specs

    specs = parser_provider_specs()
    for spec in specs:
        factory = spec.parser_factory
        if getattr(factory, "__module__", None) in {"__main__", "__mp_main__"}:
            # pickle.dumps accepts __main__ functions, but the fresh interpreter
            # cannot reconstruct a notebook or stdin definition.
            raise _ProviderError("Provider factories must live in an importable module")
    snapshot = pickle.dumps(specs, protocol=pickle.HIGHEST_PROTOCOL)
    if len(snapshot) > _MAX_MESSAGE:
        raise _ProviderError("Provider configuration exceeds the transport budget")
    return snapshot


def _validate_scan(scan: object, byte_length: int, limits: ParseLimits) -> Scan:
    from .syntax import Injection, Scan

    if not isinstance(scan, Scan) or not isinstance(scan.result, ParseResult):
        raise ValueError("Native parser returned an invalid evidence object")
    if type(scan.nodes) is not int or not 0 <= scan.nodes <= limits.max_nodes:
        raise ValueError("Native parser returned an invalid node count")
    result = scan.result
    for entries, expected in (
        (result.boundaries, Boundary),
        (result.regions, Region),
        (result.diagnostics, Diagnostic),
        (scan.injections, Injection),
    ):
        if not isinstance(entries, list) or any(not isinstance(item, expected) for item in entries):
            raise ValueError("Native parser returned invalid evidence entries")
    if len(result.regions) > limits.max_regions or len(scan.injections) > limits.max_injections:
        raise ValueError("Native parser exceeded the evidence budget")
    for names in (result.parsers, result.languages):
        if not isinstance(names, set) or any(not isinstance(name, str) for name in names):
            raise ValueError("Native parser returned invalid parser/language names")
    for item in chain(result.boundaries, result.regions):
        if type(item.depth) is not int or item.depth < 0:
            raise ValueError("Native parser returned an invalid evidence depth")
        if not isinstance(item.kind, str) or not isinstance(item.source, str):
            raise ValueError("Native parser returned invalid evidence metadata")
    for item in result.boundaries:
        if type(item.offset) is not int or type(item.mandatory) is not bool:
            raise ValueError("Native parser returned an invalid boundary")
    for item in chain(result.regions, result.diagnostics, scan.injections):
        if type(item.start) is not int or type(item.end) is not int:
            raise ValueError("Native parser returned an invalid source coordinate")
    for item in result.regions:
        if (
            type(item.protect) is not bool
            or not isinstance(item.path, tuple)
            or any(not isinstance(part, str) for part in item.path)
        ):
            raise ValueError("Native parser returned an invalid region")
    for item in result.diagnostics:
        if any(not isinstance(value, str) for value in (item.code, item.parser, item.detail)):
            raise ValueError("Native parser returned an invalid diagnostic")
    for item in chain(result.regions, scan.injections):
        if item.language is not None and not isinstance(item.language, str):
            raise ValueError("Native parser returned an invalid language hint")
    result.validate(byte_length)
    for item in scan.injections:
        if not 0 <= item.start < item.end <= byte_length or not isinstance(item.ranges, tuple):
            raise ValueError("Native parser returned an invalid injection span")
        if not isinstance(item.kind, str) or not isinstance(item.source, str):
            raise ValueError("Native parser returned invalid injection metadata")
        previous = item.start
        for span in item.ranges:
            if not isinstance(span, tuple) or len(span) != 2:
                raise ValueError("Native parser returned an invalid injection range")
            start, end = span
            if (
                type(start) is not int
                or type(end) is not int
                or not previous <= start < end <= item.end
            ):
                raise ValueError("Native parser injection ranges must be bounded and disjoint")
            previous = end
    return scan


def _failure_scan(
    payload: bytes, language: str, limits: ParseLimits, code: str, detail: str
) -> Scan:
    from .syntax import scan_generic

    result = scan_generic(payload, language, limits)
    result.result.diagnostics.append(Diagnostic(code, 0, len(payload), "native-worker", detail))
    return result


def scan_syntax_isolated(
    payload: bytes, language: str, limits: ParseLimits, backend: str = "tree_sitter"
) -> Scan:
    """Return checked byte evidence, recovering when a native parser fails.

    Providers install once per interpreter; registry changes restart it instead
    of mixing identities. Factories must be pickleable and importable without a
    ``__main__`` script. Concurrent threads share one serialized worker.
    """
    global _worker
    from .registry import grammar_available

    if not isinstance(payload, bytes):
        raise TypeError("Isolated syntax parsing requires bytes")
    if backend != "fallback" and not grammar_available(language):
        # Without a grammar nothing native runs: the lexical fallback and its
        # diagnostic come back immediately, with no round trip to the worker.
        from .syntax import scan_syntax

        return scan_syntax(payload, language, limits, backend)
    try:
        providers = _provider_snapshot()
    except (pickle.PickleError, AttributeError, TypeError, _ProviderError) as error:
        return _failure_scan(
            payload,
            language,
            limits,
            "parser_provider_not_serializable",
            f"Use an importable factory for isolation: {type(error).__name__}",
        )
    request = pickle.dumps((payload, language, limits, backend), protocol=pickle.HIGHEST_PROTOCOL)
    if len(request) > _MAX_MESSAGE:
        return _failure_scan(
            payload, language, limits, "native_parser_message_budget", "Request exceeds pipe budget"
        )
    with _mutex:
        try:
            if (
                _worker is None
                or (_worker.process is not None and _worker.process.poll() is not None)
                or _worker.providers != providers
            ):
                previous, _worker = _worker, None
                if previous is not None:
                    previous.close()
                _worker = _Worker(providers)
            key = payload, language, limits, backend
            if key in _worker.scans:
                _worker.scans.move_to_end(key)
                return _worker.scans[key]
            result = _validate_scan(
                _worker.call(request, limits.worker_timeout_ms / 1000), len(payload), limits
            )
            if _CACHED_SCANS and len(payload) <= _CACHED_PAYLOAD:
                _worker.scans[key] = result
                if len(_worker.scans) > _CACHED_SCANS:
                    _worker.scans.popitem(last=False)
            return result
        except (OSError, EOFError, ValueError, RuntimeError, pickle.PickleError) as error:
            previous, _worker = _worker, None
            exit_code = (
                previous.process.poll()
                if previous is not None and previous.process is not None
                else None
            )
            stderr = previous.stderr_tail() if previous is not None else ""
            if previous is not None:
                previous.close()
            code = (
                "native_parser_timeout"
                if isinstance(error, TimeoutError)
                else "parser_provider_not_serializable"
                if isinstance(error, _ProviderError)
                else "native_parser_crash"
            )
            return _failure_scan(
                payload,
                language,
                limits,
                code,
                f"{type(error).__name__}; exit_code={exit_code}; worker will restart"
                + (f"; stderr: {stderr}" if stderr else ""),
            )


def _parent_death_signal() -> None:
    """Also stop a hung native parser if its preparation process is killed."""
    if not sys.platform.startswith("linux") or _OWNER_ENV not in os.environ:
        return
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    operation = library.prctl
    operation.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    operation.restype = ctypes.c_int
    if operation(1, signal.SIGKILL, 0, 0, 0) != 0:  # Linux PR_SET_PDEATHSIG.
        raise OSError(ctypes.get_errno(), "Cannot configure native parser parent-death signal")
    # Close the race where the owner died just before PR_SET_PDEATHSIG.
    if os.getppid() != int(os.environ[_OWNER_ENV]):
        os._exit(0)


def _serve() -> None:
    _parent_death_signal()
    # Reserve private protocol descriptors, then detach ordinary stdin/stdout.
    # Provider logging, including native fd-1 writes, cannot corrupt the protocol
    # or accidentally consume the next source document.
    reader = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    writer = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    with open(os.devnull, "rb") as empty:
        os.dup2(empty.fileno(), sys.stdin.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    from .registry import install_parser_provider_specs
    from .syntax import scan_syntax

    with reader, writer:
        try:
            providers = _receive(reader)
            install_parser_provider_specs(providers)
        except Exception as error:
            _send(writer, pickle.dumps(("provider_error", type(error).__name__)))
            return
        _send(writer, pickle.dumps(("ready", _PROTOCOL_VERSION)))
        while True:
            try:
                payload, language, limits, backend = _receive(reader)
            except EOFError:
                return
            try:
                result = scan_syntax(payload, language, limits, backend)
            except Exception as error:
                result = _failure_scan(
                    payload, language, limits, "native_parser_exception", type(error).__name__
                )
            _send(writer, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Internal native-parser worker; use megatoken chunks for diagnostics")
    _serve()
