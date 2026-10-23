#!/usr/bin/env python3
"""Serve soak: does one long chat session keep answering, or does the service stop after 64?

THE QUESTION THIS ANSWERS
    Reviewers reported that `megatoken serve` refuses further generation once a session holds 64
    retained results, unless the client sends the separate "release" request.  This probe
    reproduces or refutes that exactly, over the real JSON-lines protocol (the program `megatoken
    serve`, started as a subprocess), and reports the first failure with the exact requests and
    responses around it.  It also checks that a long, well-behaved session (one that releases)
    stays healthy: no failure, steady latency, steady server memory, bounded history.

WHAT IT DOES
    Three scenarios, each in its own single session of the service:
    with_release     N rounds of: observe one user message, generate one reply, then send
                     `release_generation` for that reply.  Every 10th round it also creates a track
                     and runs two lifecycle actions on it (progress, then complete).
    without_release  The same N rounds but the client never sends a release.  Expected (if the
                     reviewers are right): the 65th generate fails with
                     "Retained output capacity exhausted".
    recovery         Continues the failed session: releases the oldest retained replies, then
                     generates again, showing a release is what unblocks it.

HOW TO RUN
    Smoke (CPU, builds its own random tiny model bundle, well under a minute on a quiet machine):
        /home/werg/megatoken-runs/env/dev/bin/python scripts/probes/probe_serve_soak.py \\
            --smoke --output /tmp/soak-smoke.json
    Realistic (150 rounds, against the bundle written by the rehearsal probe or by `megatoken export`):
        /home/werg/megatoken-runs/env/dev/bin/python scripts/probes/probe_serve_soak.py \\
            --bundle /path/to/served/bundle --requests 150 --output /tmp/soak.json

WHAT THE REPORTED NUMBERS MEAN
    requests                      N, the rounds attempted per scenario (default 150, smoke 70).
    *.completed_rounds            rounds that fully succeeded before the first failure (or N).
                                  Healthy for with_release: equal to N.
    *.first_failure               null if nothing failed; otherwise the round number, the request
                                  and the service's error text.  Healthy for with_release: null.
                                  Without release the limit is expected at round 65.
    *.latency_ms_first10 / last10 mean time of one generate round trip at the start and at the end
                                  of the run.  Healthy: last10 within about 2x of first10 (a steady
                                  climb would mean growing per-session work).
    *.server_rss_mb_start / end   resident memory of the service process.  Healthy: the end value
                                  is not much above the start (a few tens of MB at most).
    *.retained_events_max         most events the session ever kept in its working history; the
                                  service evicts old ones itself, so this must stay small (tiny
                                  model: 8).
    finding.limit_confirmed       true when the no-release scenario failed at the 65th generate with
                                  the retained-output message.
    recovery.recovered            true when releasing made generation work again.

IF IT FAILS / WHAT TO CHANGE
    If with_release fails, that is a real serving fault: the first_failure block has the exact
    request, and `megatoken serve --model BUNDLE` can replay it by hand.  If without_release fails
    at round 65 (expected), clients of the synchronous `generate` request must send
    {"op":"release_generation","session":S,"selection_revision":R} after each reply, where R is the
    `revision` value in the generate response; or the service should release the synchronous
    results itself.  See the "finding" text in the output for the source lines involved.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
RETAINED_LIMIT_TEXT = "Retained output capacity exhausted"


def finite_tree(value: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(path or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            bad.extend(finite_tree(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            bad.extend(finite_tree(item, f"{path}[{index}]"))
    return bad


def trim(value: Any, limit: int = 100) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in {"tape", "selection"}:
                out[key] = f"<{key}: {len(json.dumps(item))} chars>"
            else:
                out[key] = trim(item, limit)
        return out
    if isinstance(value, list):
        return [trim(item, limit) for item in value[:4]] + (["..."] if len(value) > 4 else [])
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


class Service:
    """`megatoken serve` as a subprocess, one JSON line in, one JSON line out."""

    def __init__(self, bundle: Path, device: str, workdir: Path, label: str):
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(name, "1")
        self.argv = [PYTHON, "-m", "megatoken", "serve", "--model", str(bundle)]
        self.argv += ["--device", device]
        self.stderr_path = workdir / f"serve-{label}.stderr"
        self.stderr = self.stderr_path.open("w")
        self.process = subprocess.Popen(  # noqa: S603 - fixed argv
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            env=env,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.counter = 0

    def _pump(self) -> None:
        for line in self.process.stdout:  # type: ignore[union-attr]
            self.lines.put(line)
        self.lines.put(None)

    def call(self, request: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
        self.counter += 1
        request = {**request, "id": self.counter}
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no answer to {request['op']} within {timeout:.0f}s") from None
        if line is None:
            tail = "\n".join(self.stderr_path.read_text().splitlines()[-15:])
            raise RuntimeError(f"service exited (code {self.process.poll()}); stderr:\n{tail}")
        return json.loads(line)

    def rss(self) -> float | None:
        return rss_mb(self.process.pid)

    def stop(self) -> int:
        with contextlib.suppress(Exception):
            assert self.process.stdin is not None
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait()
        self.stderr.close()
        return code


def build_random_bundle(directory: Path) -> Path:
    """A tiny random-weight bundle (the protocol does not care that it is untrained)."""
    sys.path[:0] = [str(REPO / "tests"), str(REPO / "src")]
    from test_served import checkpoint_model

    from megatoken.served import export_served

    # Lifecycle context announcements are atomic records. The byte-codec bundle
    # must fit their JSON as well as the short generated replies; 64 bytes cannot.
    _, checkpoint = checkpoint_model(directory / "checkpoint", max_capacity=128)
    path = directory / "bundle"
    export_served(checkpoint, path)
    return path


def reply_request(session: str, round_number: int, seed: int) -> dict[str, Any]:
    return {
        "op": "generate",
        "session": session,
        "outputs": [{"track": "assistant", "modality": "text", "max_tokens": 6, "kind": "utf8"}],
        "seed": seed + round_number,
        "effort": 1,
    }


def run_scenario(
    service: Service,
    rounds: int,
    seed: int,
    *,
    release: bool,
    track_every: int,
    stop_at_failure: bool = True,
) -> dict[str, Any]:
    """One session, ``rounds`` chat rounds; returns a summary and the exchanges near any failure."""
    opened = service.call({"op": "open", "effort": 1, "metadata": {"task": "chat"}})
    if not opened.get("ok"):
        raise RuntimeError(f"open failed: {opened}")
    session = opened["session"]
    rss_start = service.rss()
    latencies: list[float] = []
    exchanges: list[dict[str, Any]] = []  # a rolling record of the last few exchanges
    first_failure: dict[str, Any] | None = None
    revisions: list[int] = []
    completed = 0
    retained_max = 0
    tracks_made = lifecycle_done = 0
    address = ""
    status_samples: list[dict[str, Any]] = []

    def record(request: dict[str, Any], response: dict[str, Any]) -> None:
        exchanges.append({"request": trim(request), "response": trim(response)})
        del exchanges[:-6]

    for number in range(1, rounds + 1):
        observed_request = {
            "op": "observe_text",
            "session": session,
            "track": "user",
            "role": "user",
            "text": f"message {number}: what is {number} plus {number}?",
            "effort": 1,
        }
        observed = service.call(observed_request)
        record(observed_request, observed)
        failed: tuple[str, dict, dict] | None = None
        if not observed.get("ok"):
            failed = ("observe_text", observed_request, observed)
        else:
            retained_max = max(retained_max, observed.get("retained_events", 0))
            request = reply_request(session, number, seed)
            started = time.time()
            response = service.call(request)
            elapsed = (time.time() - started) * 1000
            record(request, response)
            if not response.get("ok"):
                failed = ("generate", request, response)
            else:
                latencies.append(elapsed)
                revisions.append(response["revision"])
                payload = base64.b64decode(response["outputs"][0]["payload_b64"])
                if not math.isfinite(response["log_prob"]) or len(payload) > 6 * 8:
                    failed = ("generate (bad answer)", request, response)
        if failed is None and release:
            release_request = {
                "op": "release_generation",
                "session": session,
                "selection_revision": revisions[-1],
            }
            released = service.call(release_request)
            record(release_request, released)
            if not released.get("ok"):
                failed = ("release_generation", release_request, released)
            else:
                revisions.pop()
        if failed is None and track_every and number % track_every == 0:
            for operation, extra in (
                ("create", {"values": {"purpose": f"side task {number}", "completion": "done"}}),
                ("progress", {"values": {"progress": "working"}}),
                ("complete", {}),
            ):
                request = {"op": "track_action", "session": session, "operation": operation}
                request.update(extra)
                if operation != "create":
                    request["track"] = address
                answer = service.call(request)
                record(request, answer)
                if not answer.get("ok"):
                    failed = (f"track_action {operation}", request, answer)
                    break
                if operation == "create":
                    address = answer["track"]["address"]
                    tracks_made += 1
                else:
                    lifecycle_done += 1
        if number % max(1, rounds // 5) == 0 and failed is None:
            status = service.call({"op": "status", "session": session})
            if status.get("ok"):
                retained_max = max(retained_max, status["retained_events"])
                status_samples.append(
                    {
                        "round": number,
                        "retained_meridians": status["retained_meridians"],
                        "retained_events": status["retained_events"],
                        "meridians_seen": status["meridians"],
                    }
                )
        if failed is not None:
            what, request, response = failed
            first_failure = {
                "round": number,
                "generates_completed_before": len(latencies),
                "failed_request": what,
                "request": trim(request),
                "error": response.get("error"),
                "exchanges_leading_to_failure": list(exchanges),
            }
            if stop_at_failure:
                break
        else:
            completed = number
    summary = {
        "completed_rounds": completed,
        "first_failure": first_failure,
        "generate_round_trips_ok": len(latencies),
        "latency_ms_first10": round(sum(latencies[:10]) / max(1, len(latencies[:10])), 2),
        "latency_ms_last10": round(sum(latencies[-10:]) / max(1, len(latencies[-10:])), 2),
        "server_rss_mb_start": rss_start,
        "server_rss_mb_end": service.rss(),
        "retained_events_max": retained_max,
        "tracks_created": tracks_made,
        "lifecycle_actions": lifecycle_done,
        "status_samples": status_samples,
        "unreleased_revisions": len(revisions),
    }
    return {
        "session": session,
        "summary": summary,
        "unreleased": revisions,
        "last_exchanges": list(exchanges),
    }


def recover(service: Service, session: str, revisions: list[int], seed: int) -> dict[str, Any]:
    """After the limit: show a release unblocks generation again."""
    blocked = service.call(reply_request(session, 9000, seed))
    steps = [
        {"step": "generate while full", "ok": blocked.get("ok"), "error": blocked.get("error")}
    ]
    released = 0
    for revision in revisions[:3]:
        answer = service.call(
            {"op": "release_generation", "session": session, "selection_revision": revision}
        )
        steps.append(
            {
                "step": f"release_generation {revision}",
                "ok": answer.get("ok"),
                "response": trim(answer),
            }
        )
        released += bool(answer.get("ok"))
    # the session must still accept an observation, then a generate
    service.call(
        {
            "op": "observe_text",
            "session": session,
            "track": "user",
            "text": "one more question",
            "effort": 1,
        }
    )
    again = service.call(reply_request(session, 9001, seed))
    steps.append(
        {"step": "generate after release", "ok": again.get("ok"), "error": again.get("error")}
    )
    return {
        "recovered": bool(again.get("ok")) and not blocked.get("ok"),
        "released": released,
        "steps": steps,
    }


def where(name: str, text: str) -> str:
    """``file:line`` of the first line containing ``text`` in src/megatoken/``name``."""
    path = REPO / "src" / "megatoken" / name
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if text in line:
                return f"src/megatoken/{name}:{number}"
    except OSError:
        pass
    return f"src/megatoken/{name}"


def soak(args, workdir: Path) -> dict[str, Any]:
    bundle = args.bundle
    if bundle is None:
        bundle = build_random_bundle(workdir)
    result: dict[str, Any] = {
        "probe": "serve_soak",
        "smoke": args.smoke,
        "seed": args.seed,
        "requests": args.requests,
        "bundle": str(bundle),
        "problems": [],
    }
    started = time.time()
    # scenario 1: a client that releases every reply
    service = Service(bundle, args.device, workdir, "release")
    try:
        result["with_release"] = run_scenario(
            service, args.requests, args.seed, release=True, track_every=10
        )
    finally:
        result["server_exit_with_release"] = service.stop()
    # scenario 2: a client that never releases
    service = Service(bundle, args.device, workdir, "norelease")
    try:
        result["without_release"] = run_scenario(
            service, args.requests, args.seed, release=False, track_every=10
        )
        failure = result["without_release"]["summary"]["first_failure"]
        if failure is not None and RETAINED_LIMIT_TEXT in str(failure["error"]):
            result["recovery"] = recover(
                service,
                result["without_release"]["session"],
                result["without_release"]["unreleased"],
                args.seed,
            )
    finally:
        result["server_exit_without_release"] = service.stop()
    failure = result["without_release"]["summary"]["first_failure"]
    confirmed = (
        failure is not None
        and failure["failed_request"] == "generate"
        and RETAINED_LIMIT_TEXT in str(failure["error"])
    )
    generates_ok = result["without_release"]["summary"]["generate_round_trips_ok"]
    result["finding"] = {
        "limit_confirmed": confirmed,
        "generates_that_succeeded_without_release": generates_ok,
        "failed_at_generate": generates_ok + 1 if failure is not None else None,
        "error": None if failure is None else failure["error"],
        "text": (
            "Confirmed: without release calls, the synchronous generate request succeeds "
            f"{generates_ok} times in a session and the next one fails with "
            f"'{failure['error'] if failure else ''}'. "
            f"The limit is `max_retained_outputs = 64` ({where('multitrack_runtime.py', 'self.max_retained_outputs = ')}), "
            f"checked in speculate() ({where('multitrack_runtime.py', '>= self.max_retained_outputs')}) "
            "over session.selected_groups; accept() adds every committed reply to "
            f"session.selected_groups ({where('multitrack_runtime.py', 'session.selected_groups[session.revision] = result')}) "
            "and only release_generation() removes it "
            f"({where('multitrack_runtime.py', 'session.selected_groups.pop(revision')}). "
            "The service's generate response "
            f"({where('multitrack_service.py', 'speculative,')}) returns the tape "
            "and `selection_revision`; release that selection after consuming its output."
            if confirmed
            else (
                "Not reproduced in this run: no failure with the retained-output message "
                f"(rounds attempted {args.requests}; the limit needs more than 64 generates)."
            )
        ),
    }
    release_failure = result["with_release"]["summary"]["first_failure"]
    if release_failure is not None:
        result["problems"].append(
            f"with_release failed at round {release_failure['round']}: {release_failure['error']}"
        )
    summary = result["with_release"]["summary"]
    if summary["latency_ms_first10"] and summary["latency_ms_last10"] > 3 * max(
        summary["latency_ms_first10"], 5.0
    ):
        result["problems"].append(
            "generate latency grew more than 3x over the with_release run "
            f"({summary['latency_ms_first10']} -> {summary['latency_ms_last10']} ms)"
        )
    for name in ("server_exit_with_release", "server_exit_without_release"):
        if result[name] != 0:
            result["problems"].append(f"{name}: service exit code {result[name]}")
    result["ok"] = not result["problems"]
    result["seconds"] = round(time.time() - started, 1)
    bad = finite_tree(result)
    if bad:
        raise ValueError(f"non-finite numbers in the result: {bad[:5]}")
    return result


def print_summary(result: dict[str, Any]) -> None:
    for name in ("with_release", "without_release"):
        s = result[name]["summary"]
        failure = s["first_failure"]
        print(
            f"{name:16} rounds ok {s['completed_rounds']}/{result['requests']}, "
            f"generate latency {s['latency_ms_first10']} -> {s['latency_ms_last10']} ms, "
            f"server memory {s['server_rss_mb_start']} -> {s['server_rss_mb_end']} MB, "
            f"first failure: "
            + ("none" if failure is None else f"round {failure['round']}: {failure['error']}")
        )
    if "recovery" in result:
        print(
            f"recovery: {result['recovery']['recovered']} (released {result['recovery']['released']})"
        )
    print(result["finding"]["text"])
    print("soak", "OK" if result["ok"] else f"PROBLEMS: {result['problems']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--smoke", action="store_true", help="70 rounds against a random tiny bundle"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="write the full result JSON here")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--bundle", type=Path, help="served bundle (default: build a random tiny one)"
    )
    parser.add_argument("--requests", type=int, help="generate rounds per scenario (default 150)")
    args = parser.parse_args(argv)
    if args.requests is None:
        args.requests = 70 if args.smoke else 150
    if args.requests < 1:
        parser.error("--requests must be positive")
    lock = None
    if args.device == "cuda":
        import fcntl

        lock = open("/tmp/megatoken-gpu.lock", "w")  # noqa: SIM115 - held for the whole run
        fcntl.flock(lock, fcntl.LOCK_EX)
    workdir = Path(tempfile.mkdtemp(prefix="megatoken-soak-"))
    try:
        result = soak(args, workdir)
    finally:
        if lock is not None:
            lock.close()
        shutil.rmtree(workdir, ignore_errors=True)
    print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
