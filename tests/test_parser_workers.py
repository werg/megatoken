"""Native isolation failure tests exercise real pipes and fresh interpreters."""

from __future__ import annotations

import importlib
import io
import json
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from megatoken.data.parsing import registry, workers
from megatoken.data.parsing.syntax import Injection, Scan, scan_syntax
from megatoken.data.parsing.types import Boundary, Diagnostic, ParseLimits, ParseResult, Region

LIMITS = ParseLimits()
SOURCE = b'def greet(name):\n    """Hello. Welcome back."""\n    return name + "!"\n'


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    workers.close_parser_worker()
    monkeypatch.setattr(registry, "_providers", {})
    yield
    workers.close_parser_worker()


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """A trusted importable provider, never derived from corpus content."""
    name = "_megatoken_test_native_provider"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    path = tmp_path / f"{name}.py"
    path.write_text(
        """import os
import signal
import sys
import time
from megatoken.data.parsing.registry import get_parser

class ControlledParser:
    def parse(self, payload):
        if payload == b"__crash__":
            os.kill(os.getpid(), signal.SIGKILL)
        if payload == b"__hang__":
            time.sleep(30)
        if payload == b"__ignore_term__":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(30)
        if payload == b"__bad_tree__":
            return object()
        return get_parser("python").parse(payload)

def factory(language):
    return ControlledParser()

def noisy_factory(language):
    print("Provider logging must never enter the protocol", flush=True)
    os.write(1, b"Native provider logging\\n")
    assert sys.stdin.read(1) == ""
    return get_parser("python")

def offline_factory(language):
    def forbid_network(event, args):
        if event.startswith("socket."):
            raise RuntimeError("Network use is forbidden by this test")
    sys.addaudithook(forbid_network)
    return get_parser("python")
""",
        encoding="utf-8",
    )
    importlib.invalidate_caches()
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def register(factory, *, name="test-native", language="test_native"):
    registry.register_parser_provider(name, "v1", [language], factory)
    return language


def diagnostic_codes(scan):
    return {item.code for item in scan.result.diagnostics}


def assert_bounded(scan, payload):
    scan.result.validate(len(payload))
    assert all(0 <= item.start <= item.end <= len(payload) for item in scan.result.diagnostics)


@pytest.mark.parametrize(
    ("language", "payload"),
    [
        ("python", SOURCE),
        ("javascript", b"function f() { return `Hello ${name}.`; }\n"),
        ("rust", b'fn main() { println!("Hello."); }'),
        ("html", b"<p>Hello &#x1f680;.</p><script>let x = 1;</script>"),
        ("python", b'def incomplete(x:\n  return "broken\xff\n'),
    ],
)
def test_real_native_round_trip_matches_direct_evidence(language, payload):
    direct = scan_syntax(payload, language, LIMITS)
    isolated = workers.scan_syntax_isolated(payload, language, LIMITS)
    assert isolated == direct
    assert f"tree_sitter:{language}" in isolated.result.parsers
    assert_bounded(isolated, payload)


def test_successful_requests_reuse_one_worker_and_cleanup_is_idempotent():
    workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    original = workers._worker
    assert original is not None
    for _ in range(4):
        result = workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
        assert "tree_sitter:python" in result.result.parsers
        assert workers._worker is original
    workers.close_parser_worker()
    workers.close_parser_worker()
    assert original.process.poll() is not None
    assert not original.thread.is_alive()
    assert original.process.stdin.closed and original.process.stdout.closed
    workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    assert workers._worker is not original


def test_custom_provider_installs_once_and_new_registration_restarts_worker(provider):
    language = register(provider.factory)
    for _ in range(3):
        result = workers.scan_syntax_isolated(SOURCE, language, LIMITS)
        assert f"tree_sitter:{language}" in result.result.parsers
        assert not result.result.diagnostics
    original = workers._worker
    register(provider.factory, name="another-provider", language="another_native")
    result = workers.scan_syntax_isolated(SOURCE, language, LIMITS)
    assert f"tree_sitter:{language}" in result.result.parsers
    assert workers._worker is not original
    assert original.process.poll() is not None


def test_provider_prints_and_native_stdout_cannot_corrupt_protocol(provider):
    language = register(provider.noisy_factory)
    for _ in range(2):
        result = workers.scan_syntax_isolated(SOURCE, language, LIMITS)
        assert f"tree_sitter:{language}" in result.result.parsers
        assert not result.result.diagnostics


def test_corpus_is_data_and_bundled_grammar_loading_is_offline(provider, tmp_path):
    language = register(provider.offline_factory)
    marker = tmp_path / "source-was-executed"
    source = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "import socket\nsocket.create_connection(('127.0.0.1', 9))\n"
    ).encode()
    result = workers.scan_syntax_isolated(source, language, LIMITS)
    assert f"tree_sitter:{language}" in result.result.parsers
    assert not result.result.diagnostics
    assert not marker.exists()


@pytest.mark.parametrize("main_module", [False, True])
def test_unserializable_and_notebook_provider_factories_are_diagnosed(main_module):
    def local_factory(language):
        return registry.get_parser("python")

    if main_module:
        local_factory.__module__ = "__main__"
    language = register(local_factory)
    result = workers.scan_syntax_isolated(SOURCE, language, LIMITS)
    assert "parser_provider_not_serializable" in diagnostic_codes(result)
    assert "generic_lexical:v2" in result.result.parsers
    assert workers._worker is None
    assert_bounded(result, SOURCE)


def test_provider_missing_from_fresh_interpreter_is_diagnosed_and_recovers(provider):
    language = register(provider.factory)
    Path(provider.__file__).unlink()
    result = workers.scan_syntax_isolated(SOURCE, language, LIMITS)
    assert "parser_provider_not_serializable" in diagnostic_codes(result)
    assert workers._worker is None
    registry._providers.clear()
    recovered = workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    assert "tree_sitter:python" in recovered.result.parsers


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal/crash semantics")
@pytest.mark.parametrize(
    ("source", "code", "timeout_ms"),
    [
        (b"__crash__", "native_parser_crash", 5_000),
        (b"__hang__", "native_parser_timeout", 60),
        (b"__ignore_term__", "native_parser_timeout", 60),
    ],
)
def test_crash_or_timeout_has_bounded_fallback_and_next_request_succeeds(
    source, code, timeout_ms, provider
):
    language = register(provider.factory)
    workers.scan_syntax_isolated(SOURCE, language, LIMITS)
    original = workers._worker
    started = time.monotonic()
    result = workers.scan_syntax_isolated(
        source, language, replace(LIMITS, worker_timeout_ms=timeout_ms)
    )
    assert time.monotonic() - started < 2.0
    assert code in diagnostic_codes(result)
    assert "generic_lexical:v2" in result.result.parsers
    assert workers._worker is None
    assert original.process.poll() is not None
    assert not original.thread.is_alive()
    assert_bounded(result, source)
    recovered = workers.scan_syntax_isolated(SOURCE, language, LIMITS)
    assert f"tree_sitter:{language}" in recovered.result.parsers
    assert workers._worker is not original


def test_tree_walk_exception_falls_back_inside_child_without_poisoning_it(provider):
    language = register(provider.factory)
    result = workers.scan_syntax_isolated(b"__bad_tree__", language, LIMITS)
    assert "native_parser_exception" in diagnostic_codes(result)
    assert "generic_lexical:v2" in result.result.parsers
    original = workers._worker
    recovered = workers.scan_syntax_isolated(SOURCE, language, LIMITS)
    assert f"tree_sitter:{language}" in recovered.result.parsers
    assert workers._worker is original


def substitute_child(monkeypatch, program, *, once=False):
    real_popen = subprocess.Popen
    calls = 0

    def spawn(command, **kwargs):
        nonlocal calls
        calls += 1
        assert command == [sys.executable, "-m", "megatoken.data.parsing.workers", "--worker"]
        assert not kwargs.get("shell", False)
        if once and calls > 1:
            return real_popen(command, **kwargs)
        return real_popen([sys.executable, "-c", program], **kwargs)

    monkeypatch.setattr(workers.subprocess, "Popen", spawn)


@pytest.mark.parametrize("ready", [False, True])
def test_watchdog_covers_blocked_startup_and_request_pipe_writes(monkeypatch, ready):
    # This trusted synthetic peer never reads stdin. The 2 MiB write necessarily
    # exceeds the OS pipe capacity, before or after its optional ready handshake.
    program = "import pickle, struct, sys, time\n"
    if ready:
        program += (
            f"hello = pickle.dumps(('ready', {workers._PROTOCOL_VERSION}))\n"
            "sys.stdout.buffer.write(struct.pack('!Q', len(hello)) + hello)\n"
            "sys.stdout.buffer.flush()\n"
        )
    program += "time.sleep(30)\n"
    substitute_child(monkeypatch, program)
    large = b"x" * (2 << 20)
    worker = workers._Worker(pickle.dumps(()) if ready else large)
    try:
        if ready:
            assert worker.ready.wait(3)
            assert worker.startup_error is None
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            worker.call(large, timeout=0.06)
        assert time.monotonic() - started < 1.0
    finally:
        worker.close()
    assert worker.process.poll() is not None
    assert not worker.thread.is_alive()


def test_corrupt_response_is_recoverable_and_next_request_uses_real_worker(monkeypatch):
    program = (
        "import pickle, struct, sys, time\n"
        f"hello = pickle.dumps(('ready', {workers._PROTOCOL_VERSION}))\n"
        "for message in (hello, b'not a pickle'):\n"
        "    sys.stdout.buffer.write(struct.pack('!Q', len(message)) + message)\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(30)\n"
    )
    substitute_child(monkeypatch, program, once=True)
    result = workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    assert "native_parser_crash" in diagnostic_codes(result)
    assert workers._worker is None
    assert_bounded(result, SOURCE)
    recovered = workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    assert "tree_sitter:python" in recovered.result.parsers


@pytest.mark.parametrize(
    "bad_scan",
    [
        None,
        Scan(nodes=-1),
        Scan(nodes=LIMITS.max_nodes + 1),
        Scan(ParseResult(boundaries=[Boundary(1.5, "line")])),
        Scan(ParseResult(boundaries=[Boundary(9999, "line")])),
        Scan(ParseResult(regions=[Region(-1, 2, "string")])),
        Scan(ParseResult(regions=[Region(0, 2, "string", depth="bad")])),
        Scan(ParseResult(diagnostics=[Diagnostic("error", 0, 9999, "test")])),
        Scan(ParseResult(parsers={7})),
        Scan(injections=[Injection(0, 9999)]),
        Scan(injections=[Injection(1, 5, ranges=((0, 2),))]),
        Scan(injections=[Injection(1, 5, ranges=((1, 3), (2, 4)))]),
        Scan(injections=[Injection(1, 5, ranges=((1, 6),))]),
        Scan(injections=[Injection(1, 5, ranges=((1, 1),))]),
    ],
)
def test_invalid_evidence_never_reaches_source_mapping(monkeypatch, bad_scan):
    workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    original = workers._worker
    monkeypatch.setattr(original, "call", lambda payload, timeout: bad_scan)
    result = workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    assert "native_parser_crash" in diagnostic_codes(result)
    assert workers._worker is None
    assert original.process.poll() is not None
    assert_bounded(result, SOURCE)


def test_request_and_response_transport_budgets(monkeypatch):
    with pytest.raises(ValueError, match="transport budget"):
        workers._receive(io.BytesIO(workers._HEADER.pack(workers._MAX_MESSAGE + 1)))
    monkeypatch.setattr(workers, "_MAX_MESSAGE", 512)
    payload = b"a" * 4096
    result = workers.scan_syntax_isolated(payload, "python", LIMITS)
    assert "native_parser_message_budget" in diagnostic_codes(result)
    assert workers._worker is None
    assert_bounded(result, payload)


def test_protocol_handles_short_reads_and_writes():
    class ShortIO(io.BytesIO):
        def write(self, payload):
            return super().write(payload[:3])

        def read(self, size=-1):
            return super().read(min(size, 2))

    stream = ShortIO()
    value = {"source": bytes(range(256)), "label": "完整"}
    workers._send(stream, pickle.dumps(value))
    stream.seek(0)
    assert workers._receive(stream) == value
    with pytest.raises(EOFError):
        workers._receive(stream)


def test_concurrent_callers_share_one_worker_without_crossed_responses():
    sources = [SOURCE + f"answer = {index}\n".encode() for index in range(24)]
    expected = [scan_syntax(payload, "python", LIMITS) for payload in sources]

    def parse(payload):
        result = workers.scan_syntax_isolated(payload, "python", LIMITS)
        return result, workers._worker.process.pid

    with ThreadPoolExecutor(max_workers=8) as pool:
        completed = list(pool.map(parse, sources))
    assert [result for result, _ in completed] == expected
    assert len({pid for _, pid in completed}) == 1


def test_worker_survives_the_short_lived_thread_that_first_requested_it():
    completed = []
    caller = threading.Thread(
        target=lambda: completed.append(workers.scan_syntax_isolated(SOURCE, "python", LIMITS))
    )
    caller.start()
    caller.join(timeout=5)
    assert not caller.is_alive() and len(completed) == 1
    original = workers._worker
    assert original is not None
    for _ in range(3):
        result = workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
        assert "tree_sitter:python" in result.result.parsers
        assert workers._worker is original


def test_process_creation_finishing_after_timeout_still_cleans_up(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_popen = subprocess.Popen

    def delayed_spawn(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(workers.subprocess, "Popen", delayed_spawn)
    worker = workers._Worker(pickle.dumps(()))
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError):
            worker.call(b"source", timeout=0.04)
        worker.close()
    finally:
        release.set()
        worker.thread.join(timeout=3)
        worker.close()
    assert not worker.thread.is_alive()
    assert worker.process is not None and worker.process.poll() is not None
    assert worker.process.stdin.closed and worker.process.stdout.closed


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires POSIX fork")
@pytest.mark.filterwarnings("ignore:This process.*is multi-threaded:DeprecationWarning")
def test_fork_detaches_inherited_handles_and_resets_a_locked_parent_mutex():
    workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
    parent_worker = workers._worker
    received, sent = os.pipe()
    # Fork with the lock held to reproduce a preparation process forking while a
    # different parent thread is using the shared worker.
    workers._mutex.acquire()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(received)
        try:
            assert workers._worker is None
            assert parent_worker.process.stdin.closed and parent_worker.process.stdout.closed
            result = workers.scan_syntax_isolated(SOURCE, "python", LIMITS)
            answer = {
                "ok": "tree_sitter:python" in result.result.parsers,
                "pid": workers._worker.process.pid,
            }
            workers.close_parser_worker()
            os.write(sent, json.dumps(answer).encode())
        finally:
            os.close(sent)
            os._exit(0)
    os.close(sent)
    workers._mutex.release()
    try:
        # A select deadline turns a regression into a test failure, not a hung CI
        # process with a child waiting forever on an inherited mutex.
        import select

        readable, _, _ = select.select([received], [], [], 8)
        assert readable, "Forked preparation child did not finish"
        answer = json.loads(os.read(received, 4096))
        assert answer["ok"] and answer["pid"] != parent_worker.process.pid
        assert workers._worker is parent_worker
        assert parent_worker.process.poll() is None
        assert (
            "tree_sitter:python"
            in workers.scan_syntax_isolated(SOURCE, "python", LIMITS).result.parsers
        )
    finally:
        os.close(received)
        os.kill(child_pid, signal.SIGKILL)
        os.waitpid(child_pid, 0)


def process_stopped(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    if sys.platform.startswith("linux"):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            return state == "Z"
        except FileNotFoundError:
            return True
    return False


def await_stopped(pid):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if process_stopped(pid):
            return
        time.sleep(0.02)
    pytest.fail(f"Native parser process {pid} survived owner shutdown")


@pytest.mark.skipif(os.name != "posix", reason="Process-lifetime observation uses POSIX")
def test_module_worker_starts_from_stdin_and_is_reaped_at_interpreter_exit():
    program = (
        "from megatoken.data.parsing import workers\n"
        "from megatoken.data.parsing.types import ParseLimits\n"
        "result = workers.scan_syntax_isolated(b'x = 1', 'python', ParseLimits())\n"
        "assert 'tree_sitter:python' in result.result.parsers\n"
        "print(workers._worker.process.pid, flush=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-"], input=program, text=True, capture_output=True, timeout=10, check=True
    )
    await_stopped(int(completed.stdout.strip()))


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal and process-lifetime semantics")
def test_interpreter_exit_does_not_wait_for_a_daemon_callers_long_watchdog():
    program = (
        "import os, signal, threading, time\n"
        "from megatoken.data.parsing import workers\n"
        "from megatoken.data.parsing.types import ParseLimits\n"
        "limits = ParseLimits(worker_timeout_ms=60000)\n"
        "workers.scan_syntax_isolated(b'x = 1', 'python', limits)\n"
        "pid = workers._worker.process.pid\n"
        "os.kill(pid, signal.SIGSTOP)\n"
        "thread = threading.Thread(target=workers.scan_syntax_isolated, "
        "args=(b'x = 2', 'python', limits), daemon=True)\n"
        "thread.start()\n"
        "while not workers._mutex.locked(): time.sleep(0.001)\n"
        "print(pid, flush=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-"], input=program, text=True, capture_output=True, timeout=5, check=True
    )
    await_stopped(int(completed.stdout.strip()))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux parent-death signal")
def test_hung_native_worker_dies_if_its_owner_is_killed():
    program = (
        "import os, signal, time\n"
        "from megatoken.data.parsing import workers\n"
        "from megatoken.data.parsing.types import ParseLimits\n"
        "workers.scan_syntax_isolated(b'x = 1', 'python', ParseLimits())\n"
        "pid = workers._worker.process.pid\n"
        "print(pid, flush=True)\n"
        "os.kill(pid, signal.SIGSTOP)\n"
        "time.sleep(30)\n"
    )
    owner = subprocess.Popen(
        [sys.executable, "-c", program], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        import select

        readable, _, _ = select.select([owner.stdout], [], [], 8)
        assert readable, "Owner did not start its native worker"
        worker_pid = int(owner.stdout.readline().strip())
        owner.kill()
        owner.wait(timeout=3)
        await_stopped(worker_pid)
    finally:
        owner.kill()
        owner.wait(timeout=3)
        owner.stdout.close()
        owner.stderr.close()
