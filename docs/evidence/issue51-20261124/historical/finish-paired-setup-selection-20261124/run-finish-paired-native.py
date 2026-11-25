#!/usr/bin/env python3
"""Run only the paired P-to-C native checks for issue51 source fixtures.

This helper is intentionally external to both git worktrees. It is prepared for
execution after the final source artifacts have been exported and synced.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PRODUCER = ROOT / "producer-dev"
CONSUMER = ROOT / "consumer"
LOG = ROOT / "finish-paired-native.log"
STATUS = ROOT / "finish-paired-native-status.json"

# Keep these in exact order with their owning test modules. The TAP pattern is
# built from these titles with regex escaping, so unrelated tests in each file
# are excluded.
SELECTIONS = [
    (
        "test/semantic-promise-adoption-getter-source-v8.test.js",
        "native independently checks adoption constructor jobs, retained scopes and selected completions",
    ),
    (
        "test/semantic-promise-mutation-profile-source-v8.test.js",
        "actual native independently replays every eligible mutation-boundary control",
    ),
    (
        "test/semantic-promise-object-profile-source-v8.test.js",
        "native independently replays every eligible Promise object control",
    ),
    (
        "test/semantic-promise-object-review-source-v8.test.js",
        "native independently checks every Promise review source and retained continuation",
    ),
    (
        "test/semantic-promise-object-review-v8.test.js",
        "native independently replays retained invocations and captured adoption selection",
    ),
    (
        "test/semantic-promise-object-source-v8.test.js",
        "independent native checker replays every new source and retained shared state",
    ),
    (
        "test/semantic-promise-object-v8.test.js",
        "independent native transitions reproduce all Promise object cases and reject altered records",
    ),
    (
        "test/semantic-protocol-promise-adoption-source-v8.test.js",
        "actual native adoption consumer independently replays every source context and restored cut",
    ),
    (
        "test/semantic-shared-generator-source-v8.test.js",
        "actual native shared replay checks every generated context and continuation",
    ),
]
EXPECTED_TEST_COUNT = 9


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def clean(repo: Path) -> bool:
    return not git(repo, "status", "--porcelain", "--untracked-files=all")


def one_line(command: list[str], cwd: Path | None = None, env=None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def parse_tap(log_text: str) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    counts = {"pass": 0, "fail": 0, "skip": 0, "cancel": 0}
    summary: dict[str, int] = {}
    observed: dict[str, int] = {title: 0 for _, title in SELECTIONS}
    outcome_for = {"✔": "pass", "✖": "fail", "﹣": "skip"}
    for raw in log_text.splitlines():
        line = raw.lstrip()
        marker = line[:1]
        if marker in outcome_for:
            title = line[1:].split(" (", 1)[0].strip()
            if title in observed:
                observed[title] += 1
                counts[outcome_for[marker]] += 1
        match = re.search(r"^ℹ\s+(tests|pass|fail|skipped|cancelled)\s+(\d+)\s*$", line)
        if match:
            key, value = match.groups()
            summary[key] = int(value)
    return counts, observed, summary


def main() -> int:
    files = [filename for filename, _ in SELECTIONS]
    titles = [title for _, title in SELECTIONS]
    pattern = "^(?:" + "|".join(re.escape(title) for title in titles) + ")$"
    started = time.monotonic()
    producer_before = git(PRODUCER, "rev-parse", "HEAD")
    consumer_before = git(CONSUMER, "rev-parse", "HEAD")
    producer_clean_before = clean(PRODUCER)
    consumer_clean_before = clean(CONSUMER)
    if not producer_clean_before or not consumer_clean_before:
        raise RuntimeError("refusing paired-native check: producer and consumer must both start clean")

    env = dict(os.environ)
    node_bin = ROOT / "node24" / "node_modules" / ".bin"
    env["PATH"] = str(node_bin) + os.pathsep + env.get("PATH", "")
    env["MEGATOKEN_ROOT"] = str(CONSUMER)
    env["MEGATOKEN_PYTHON"] = str(ROOT / ".venv" / "bin" / "python")
    env["PYTHONPATH"] = str(CONSUMER / "src")

    node = shutil.which("node", path=env["PATH"])
    if node is None:
        raise RuntimeError(f"Node executable not found with pinned PATH prefix {node_bin}")
    node_version = one_line([node, "--version"])
    if not re.match(r"^v24\.", node_version):
        raise RuntimeError(f"paired native run requires Node 24; got {node_version}")
    python_version = one_line([env["MEGATOKEN_PYTHON"], "--version"])

    command = [
        node,
        "--experimental-vm-modules",
        "--test",
        "--test-concurrency=2",
        "--test-name-pattern",
        pattern,
        *files,
    ]
    exit_code = 1
    error_text = None
    try:
        with LOG.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=PRODUCER,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_code = result.returncode
    except Exception as exc:  # retain a status artifact for setup/runtime errors
        error_text = f"{type(exc).__name__}: {exc}"
        LOG.write_text(error_text + "\n", encoding="utf-8")

    log_text = LOG.read_text(encoding="utf-8", errors="replace")
    counts, observed, summary = parse_tap(log_text)
    producer_after = git(PRODUCER, "rev-parse", "HEAD")
    consumer_after = git(CONSUMER, "rev-parse", "HEAD")
    producer_clean_after = clean(PRODUCER)
    consumer_clean_after = clean(CONSUMER)

    status = {
        "schema": "issue51-paired-native-run/1",
        "producerRevision": producer_before,
        "consumerRevision": consumer_before,
        "producerRevisionAfter": producer_after,
        "consumerRevisionAfter": consumer_after,
        "producerCleanBefore": producer_clean_before,
        "consumerCleanBefore": consumer_clean_before,
        "producerCleanAfter": producer_clean_after,
        "consumerCleanAfter": consumer_clean_after,
        "nodeVersion": node_version,
        "pythonVersion": python_version,
        "files": files,
        "testTitles": titles,
        "expectedTestCount": EXPECTED_TEST_COUNT,
        "testCounts": counts,
        "observedSelectedTests": observed,
        "tapSummary": summary,
        "command": command,
        "exitCode": exit_code,
        "durationSeconds": round(time.monotonic() - started, 3),
        "logPath": str(LOG),
        "error": error_text,
    }
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    requirements = [
        producer_clean_before,
        consumer_clean_before,
        producer_clean_after,
        consumer_clean_after,
        producer_before == producer_after,
        consumer_before == consumer_after,
        exit_code == 0,
        counts == {"pass": 9, "fail": 0, "skip": 0, "cancel": 0},
        all(observed[title] == 1 for title in titles),
        sum(observed.values()) == EXPECTED_TEST_COUNT,
        error_text is None,
    ]
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if all(requirements) else 1


if __name__ == "__main__":
    raise SystemExit(main())
