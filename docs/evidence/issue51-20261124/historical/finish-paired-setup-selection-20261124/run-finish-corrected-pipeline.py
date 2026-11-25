"""Run the final issue-51 checks at explicitly approved producer/consumer pins.

The producer generation and canonical standard gate must already have passed
on the expected producer pin. This coordinator checks that receipt and does not
rerun or replace it. It launches the existing gate drivers once, preserves
their individual receipts, performs no push, and stops before final evidence
recording or documentation/merge work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback


ROOT = Path(__file__).resolve().parent
PRODUCER = ROOT / "producer-dev"
CONSUMER = ROOT / "consumer"
PYTHON = str(ROOT / ".venv/bin/python")
STATUS_PATH = ROOT / "finish-corrected-pipeline-status.json"
ARCHIVE_MANIFEST = ROOT / "finish-pre-correction-5553de1a-6bb45749/manifest.json"
ARCHIVE_PRODUCER = "5553de1a10761626c5604a57453291039d73683d"
ARCHIVE_CONSUMER = "6bb4574931c4fdf19588a86aef96577b22ff6541"
TOKENIZER_SHA256 = "40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60"
TOKENIZER_IDENTITY = "7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2"
HEX40 = re.compile(r"^[0-9a-f]{40}$")

ENV = dict(os.environ)
ENV["PATH"] = str(ROOT / "node24/node_modules/.bin") + os.pathsep + ENV.get("PATH", "")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def clean_pin(repo: Path, expected: str) -> None:
    assert git(repo, "rev-parse", "HEAD") == expected, (repo, expected)
    assert not git(repo, "status", "--porcelain", "--untracked-files=all"), repo


def json_file(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_ancestor(repo: Path, older: str, newer: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, (repo, older, newer, result.stderr)


def canonical_generated_paths(repo: Path, older: str, newer: str) -> list[str]:
    changed = git(repo, "diff", "--name-only", older, newer).splitlines()
    allowed_exact = {
        "contracts/semantic-execution-v8.json",
        "contracts/semantic-native-policy-v8.json",
        "test/fixtures/semantic-protocol-region-v8.json.gz.b64",
        "test/fixtures/semantic-protocol-source-v8.json",
        "test/fixtures/semantic-regions-v8.json",
    }
    assert all(path in allowed_exact for path in changed), changed
    return changed


def last_json(path: Path) -> dict:
    rows = [line for line in path.read_text().splitlines() if line.lstrip().startswith("{")]
    assert rows, path
    return json.loads(rows[-1])


def validate_canonical_gate(expected_producer: str) -> tuple[dict, dict, str]:
    generation = json_file(ROOT / "finish-canonical-status.json")
    assert generation.get("current") is None
    assert len(generation.get("steps", [])) == 10, generation
    assert all(step.get("exitCode") == 0 for step in generation["steps"]), generation
    assert generation["steps"][7]["command"] == [
        "node", "tools/generate-protocol-source-fixtures.mjs"
    ]
    assert generation["steps"][9]["command"] == ["npm", "run", "contracts:check"]

    recheck = json_file(ROOT / "finish-canonical-recheck-status.json")
    assert recheck.get("producerRevision") == expected_producer, recheck
    assert recheck.get("generationSourceRevision") == generation.get("implementationRevision"), recheck
    assert recheck.get("generatedDataUnchanged") is True, recheck
    assert not recheck.get("current"), recheck
    artifact_revision = recheck.get("generatedArtifactRevision")
    assert artifact_revision == generation.get("producerRevision"), (generation, recheck)
    source_revision = generation.get("implementationRevision")
    assert HEX40.fullmatch(source_revision or ""), generation
    assert_ancestor(PRODUCER, source_revision, artifact_revision)
    canonical_generated_paths(PRODUCER, source_revision, artifact_revision)
    mode = recheck.get("mode")
    if mode == "verified-generation-artifact":
        assert artifact_revision == expected_producer, recheck
        assert recheck.get("steps") == [], recheck
        evidence = recheck.get("evidenceLogs")
        assert isinstance(evidence, list) and len(evidence) == 2, recheck
        by_index = {row.get("stepIndex"): row for row in evidence}
        assert set(by_index) == {7, 9}, evidence
        assert by_index[7].get("purpose") == "protocol-source-fixture-generation"
        assert by_index[7].get("valid") is True and by_index[7].get("mode") == "generated"
        assert by_index[9].get("purpose") == "standard-contract-gate"
        assert by_index[9].get("valid") is True and by_index[9].get("mode") == "checked"
        for index, record in by_index.items():
            path = Path(record["log"])
            assert path.is_file() and sha256(path) == record["sha256"], (index, record)
            observed = last_json(path)
            assert observed.get("valid") is True and observed.get("mode") == record["mode"], observed
    elif mode == "fresh-recheck":
        assert_ancestor(PRODUCER, artifact_revision, expected_producer)
        assert len(recheck.get("steps", [])) == 2, recheck
        expected_commands = [
            ["node", "tools/generate-protocol-source-fixtures.mjs", "--check"],
            ["npm", "run", "contracts:check"],
        ]
        assert [step.get("command") for step in recheck["steps"]] == expected_commands, recheck
        assert all(step.get("exitCode") == 0 and step.get("producerTreeUnchanged") is True
                   for step in recheck["steps"]), recheck
        data_changes = git(
            PRODUCER, "diff", "--name-only", artifact_revision, expected_producer,
            "--", "contracts", "test/fixtures"
        ).splitlines()
        assert not data_changes, data_changes
        for step in recheck["steps"]:
            log = ROOT / step["log"]
            assert log.is_file(), log
            observed = last_json(log)
            assert observed.get("valid") is True and observed.get("mode") == "checked", observed
    else:
        raise AssertionError(("unsupported canonical verification mode", mode, recheck))
    assert not git(PRODUCER, "status", "--porcelain", "--untracked-files=all")
    return generation, recheck, sha256(ROOT / "finish-canonical-recheck-status.json")


def preflight(args: argparse.Namespace) -> tuple[dict, dict, str, dict, str]:
    assert HEX40.fullmatch(args.expected_producer), args.expected_producer
    assert HEX40.fullmatch(args.expected_consumer), args.expected_consumer
    assert not STATUS_PATH.exists(), STATUS_PATH
    clean_pin(PRODUCER, args.expected_producer)
    clean_pin(CONSUMER, args.expected_consumer)

    archive = json_file(ARCHIVE_MANIFEST)
    assert archive.get("producerRevision") == ARCHIVE_PRODUCER, archive
    assert archive.get("consumerRevision") == ARCHIVE_CONSUMER, archive
    assert archive.get("schema") == "issue51-historical-correction-run/1", archive
    archive_hash = sha256(ARCHIVE_MANIFEST)

    for path in (
        ROOT / "cohort-finish.json",
        ROOT / "artifacts-finish.jsonl",
        ROOT / "prepared-finish",
        ROOT / "finish-preparation-report.json",
    ):
        assert not path.exists(), path
    for path in (
        ROOT / "run-finish-paired-native.py",
        ROOT / "finish-paired-native-status.json",
        ROOT / "finish-paired-native.log",
    ):
        assert path.is_file() if path.name == "run-finish-paired-native.py" else not path.exists(), path

    generation, recheck, recheck_hash = validate_canonical_gate(args.expected_producer)
    return generation, recheck, recheck_hash, archive, archive_hash


def save(status: dict, *, exclusive: bool = False) -> None:
    if exclusive:
        with STATUS_PATH.open("x") as stream:
            stream.write(json.dumps(status, indent=2) + "\n")
        return
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(STATUS_PATH)


def phase(status: dict, name: str) -> None:
    status["phase"] = name
    save(status)
    print(name, flush=True)


def run_step(status: dict, name: str, command: list[str], *, cwd: Path = ROOT) -> None:
    log = ROOT / f"finish-corrected-{name}.log"
    assert not log.exists(), log
    started = time.monotonic()
    with log.open("w") as output:
        result = subprocess.run(command, cwd=cwd, env=ENV, stdout=output, stderr=subprocess.STDOUT)
    status["steps"].append(
        {
            "name": name,
            "command": command,
            "cwd": str(cwd),
            "log": log.name,
            "exitCode": result.returncode,
            "seconds": round(time.monotonic() - started, 3),
            "logSha256": sha256(log),
        }
    )
    save(status)
    assert result.returncode == 0, (name, result.returncode, log)


def capture_json_stdout(log: Path, destination: Path) -> dict:
    raw = log.read_bytes()
    value = json.loads(raw)
    assert isinstance(value, dict), (log, type(value))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(destination)
    return value


def launch(status: dict, jobs: dict, name: str, driver: str) -> None:
    log = ROOT / f"finish-corrected-{name}-driver.log"
    assert not log.exists(), log
    with log.open("w") as output:
        process = subprocess.Popen(
            [PYTHON, "-u", str(ROOT / driver)],
            cwd=ROOT,
            env=ENV,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    jobs[name] = process
    status["jobs"][name] = {"pid": process.pid, "driver": driver, "log": log.name}
    save(status)


def wait_job(status: dict, jobs: dict, name: str) -> dict:
    process = jobs[name]
    exit_code = process.wait()
    receipt_path = {
        "producer": ROOT / "finish-producer-status.json",
        "export": ROOT / "finish-cohort-export-status.json",
        "paired-native": ROOT / "finish-paired-native-status.json",
        "native": ROOT / "finish-native-status.json",
        "cohort": ROOT / "finish-cohort-status.json",
        "preparation": ROOT / "finish-preparation-status.json",
    }[name]
    receipt = json_file(receipt_path)
    status["jobs"][name].update(exitCode=exit_code, receipt=receipt_path.name)
    save(status)
    assert exit_code == 0, (name, exit_code, status["jobs"][name])
    assert receipt.get("producerRevision") == status["producerRevision"], (name, receipt)
    if name in ("paired-native", "native", "cohort", "preparation"):
        assert receipt.get("consumerRevision") == status["consumerRevision"], (name, receipt)
    if name == "producer":
        assert len(receipt.get("steps", [])) == 2, receipt
        assert all(step.get("exitCode") == 0 for step in receipt["steps"]), receipt
        assert receipt.get("current") is None, receipt
    elif name == "export":
        assert receipt.get("producerTreeUnchanged") is True, receipt
        assert receipt.get("exitCode") == 0, receipt
    elif name == "paired-native":
        assert receipt.get("exitCode") == 0, receipt
        assert receipt.get("expectedTestCount") == 9, receipt
        assert receipt.get("testCounts") == {"pass": 9, "fail": 0, "skip": 0, "cancel": 0}, receipt
        assert receipt.get("producerRevisionAfter") == status["producerRevision"], receipt
        assert receipt.get("consumerRevisionAfter") == status["consumerRevision"], receipt
        assert receipt.get("producerCleanBefore") is True and receipt.get("consumerCleanBefore") is True, receipt
        assert receipt.get("producerCleanAfter") is True and receipt.get("consumerCleanAfter") is True, receipt
        assert receipt.get("nodeVersion", "").startswith("v24."), receipt
        assert receipt.get("error") is None, receipt
        assert len(receipt.get("testTitles", [])) == 9, receipt
        assert len(receipt.get("files", [])) == 9, receipt
        assert "--test-concurrency=2" in receipt.get("command", []), receipt
        observed = receipt.get("observedSelectedTests", {})
        assert len(observed) == 9 and all(count == 1 for count in observed.values()), receipt
    elif name in ("native", "cohort", "preparation"):
        assert receipt.get("exitCode") == 0, receipt
    return receipt


def commit_consumer_data(status: dict, message: str) -> None:
    assert not git(CONSUMER, "ls-files", "--others", "--exclude-standard").splitlines()
    assert not git(CONSUMER, "diff", "--cached", "--name-only").splitlines()
    changed = git(CONSUMER, "diff", "--name-only").splitlines()
    allowed_exact = {
        "src/megatoken/data/contracts/semantic-execution-v8.json",
        "tests/fixtures/semantic-protocol-source-v8.json",
        "tests/test_protocol_v8.py",
        "tests/test_issue51_object_source_v8.py",
    }
    assert all(
        path in allowed_exact
        or (path.startswith("tests/fixtures/semantic-") and path.endswith(".json.gz.b64"))
        for path in changed
    ), changed
    assert not git(CONSUMER, "diff", "--name-only", "--diff-filter=D").splitlines()
    if changed:
        subprocess.run(["git", "-C", str(CONSUMER), "add", "--", *changed], check=True)
        subprocess.run(["git", "-C", str(CONSUMER), "commit", "-m", message], check=True)
    assert not git(CONSUMER, "status", "--porcelain", "--untracked-files=all")
    status["consumerRevision"] = git(CONSUMER, "rev-parse", "HEAD")
    save(status)


def validate_export(status: dict) -> int:
    report_path = ROOT / "cohort-finish.json"
    report = json_file(report_path)
    assert report.get("producerRevision") == status["producerRevision"], report
    assert report.get("producerTreeClean") is True, report
    cases = report.get("cases")
    assert isinstance(cases, list) and cases
    identities = [row.get("id") for row in cases]
    assert all(isinstance(identity, str) and identity for identity in identities)
    assert len(identities) == len(set(identities)), "duplicate exported case IDs"
    supported_cases = [row for row in cases if row.get("artifact") is not None]
    artifact_path = ROOT / "artifacts-finish.jsonl"
    digest = hashlib.sha256()
    with artifact_path.open("rb") as actual_stream:
        for case in supported_cases:
            expected_line = (json.dumps(case["artifact"], separators=(",", ":")) + "\n").encode()
            actual_line = actual_stream.readline()
            assert actual_line == expected_line, (case["id"], "JSONL record differs from exported artifact")
            digest.update(actual_line)
        assert actual_stream.read(1) == b"", "JSONL has extra artifact records"
    sync_status = json_file(ROOT / "finish-cohort-sync.json")
    assert sync_status.get("producerRevision") == status["producerRevision"], sync_status
    assert sync_status.get("producerTreeClean") is True, sync_status
    assert sync_status.get("supported") == len(supported_cases), sync_status
    assert sync_status.get("declared") == len(cases), sync_status
    status["export"] = {
        "report": report_path.name,
        "reportSha256": sha256(report_path),
        "supported": len(supported_cases),
        "declared": len(cases),
        "artifactsJsonl": artifact_path.name,
        "artifactsJsonlSha256": digest.hexdigest(),
        "supportedIds": [row["id"] for row in supported_cases],
    }
    save(status)
    return len(supported_cases)


def validate_preparation(status: dict, supported: int) -> dict:
    path = ROOT / "prepared-finish/semantic-v8-report.json"
    report = json_file(path)
    assert report.get("schema") == "semantic-preparation/8", report
    assert report.get("producerRevision") == status["producerRevision"], report
    assert report.get("consumerRevision") == status["consumerRevision"], report
    counts = report.get("counts", {})
    assert counts.get("attempted") == supported == counts.get("admitted"), (supported, counts)
    assert counts.get("refused", 0) == 0, counts
    assert report.get("refusals", []) == [], report
    assert report.get("optimizerSteps") == 0, report
    assert report.get("learningClaim") is None, report
    assert report.get("tokenizerSha256") == TOKENIZER_SHA256, report
    assert report.get("tokenizerIdentity") == TOKENIZER_IDENTITY, report
    status["preparation"] = {
        "report": "prepared-finish/semantic-v8-report.json",
        "reportSha256": sha256(path),
        "counts": counts,
        "learningClaim": report["learningClaim"],
        "optimizerSteps": report["optimizerSteps"],
        "tokenizerSha256": report["tokenizerSha256"],
        "tokenizerIdentity": report["tokenizerIdentity"],
    }
    save(status)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-producer", required=True)
    parser.add_argument("--expected-consumer", required=True)
    args = parser.parse_args()

    _, recheck, recheck_hash, archive, archive_hash = preflight(args)
    status = {
        "schema": "issue51-corrected-final-pipeline/1",
        "phase": "preflight-passed",
        "expectedProducerRevision": args.expected_producer,
        "producerRevision": args.expected_producer,
        "initialConsumerRevision": args.expected_consumer,
        "consumerRevision": args.expected_consumer,
        "canonicalGenerationStatusSha256": sha256(ROOT / "finish-canonical-status.json"),
        "canonicalRecheckStatusSha256": recheck_hash,
        "canonicalRecheckMode": recheck["mode"],
        "canonicalGateRerun": False,
        "historicalArchiveManifest": ARCHIVE_MANIFEST.relative_to(ROOT).as_posix(),
        "historicalArchiveManifestSha256": archive_hash,
        "historicalPins": {
            "producerRevision": archive["producerRevision"],
            "consumerRevision": archive["consumerRevision"],
        },
        "steps": [],
        "jobs": {},
    }
    jobs = {}
    save(status, exclusive=True)

    try:
        phase(status, "producer-full-tests-and-export")
        launch(status, jobs, "producer", "run-finish-producer-checks.py")
        launch(status, jobs, "export", "run-finish-cohort-export.py")

        phase(status, "canonical-consumer-sync")
        run_step(status, "canonical-sync", [PYTHON, str(ROOT / "sync-object-canonical.py")])
        canonical_sync = capture_json_stdout(
            ROOT / "finish-corrected-canonical-sync.log", ROOT / "finish-canonical-sync.json"
        )
        assert canonical_sync.get("producerRevision") == args.expected_producer, canonical_sync
        status["canonicalSync"] = {
            "receipt": "finish-canonical-sync.json",
            "receiptSha256": sha256(ROOT / "finish-canonical-sync.json"),
        }
        save(status)
        run_step(status, "support-fixtures", [PYTHON, str(ROOT / "run-finish-support-fixtures.py")])
        run_step(
            status,
            "format-canonical-test",
            [str(ROOT / ".venv/bin/ruff"), "format", str(CONSUMER / "tests/test_protocol_v8.py")],
        )
        clean_pin(PRODUCER, args.expected_producer)
        commit_consumer_data(status, "Synchronize corrected canonical protocol fixtures")

        phase(status, "waiting-for-issue-export")
        export_receipt = wait_job(status, jobs, "export")
        assert export_receipt["producerRevision"] == args.expected_producer

        phase(status, "full-cohort-sync")
        run_step(status, "cohort-sync", [PYTHON, str(ROOT / "sync-finish-cohort.py")])
        cohort_sync = capture_json_stdout(
            ROOT / "finish-corrected-cohort-sync.log", ROOT / "finish-cohort-sync.json"
        )
        assert cohort_sync.get("producerRevision") == args.expected_producer, cohort_sync
        assert cohort_sync.get("producerTreeClean") is True, cohort_sync
        status["cohortSync"] = {
            "receipt": "finish-cohort-sync.json",
            "receiptSha256": sha256(ROOT / "finish-cohort-sync.json"),
            "declared": cohort_sync.get("declared"),
            "preservedOriginals": cohort_sync.get("preservedOriginals"),
            "supported": cohort_sync.get("supported"),
            "refused": cohort_sync.get("refused"),
        }
        save(status)
        run_step(
            status,
            "format-cohort-test",
            [str(ROOT / ".venv/bin/ruff"), "format", str(CONSUMER / "tests/test_issue51_object_source_v8.py")],
        )
        commit_consumer_data(status, "Synchronize complete retained object source cohort")
        clean_pin(PRODUCER, args.expected_producer)
        status["consumerRevision"] = git(CONSUMER, "rev-parse", "HEAD")
        save(status)
        supported = validate_export(status)

        # This short exact-pair gate protects source-native Promise and
        # adoption interactions that are not in the broad consumer modules.
        phase(status, "paired-native-focused-gate")
        launch(status, jobs, "paired-native", "run-finish-paired-native.py")
        paired = wait_job(status, jobs, "paired-native")
        status["pairedNativeCounts"] = paired.get("testCounts")
        save(status)

        phase(status, "final-consumer-verification-and-preparation")
        launch(status, jobs, "native", "run-finish-native-checks.py")
        launch(status, jobs, "cohort", "run-finish-cohort-checks.py")
        launch(status, jobs, "preparation", "run-finish-preparation.py")
        run_step(status, "python-style", [PYTHON, str(ROOT / "check-finish-python-style.py")])

        phase(status, "awaiting-all-final-drivers")
        for name in ("producer", "native", "cohort", "preparation"):
            wait_job(status, jobs, name)
        clean_pin(PRODUCER, args.expected_producer)
        clean_pin(CONSUMER, status["consumerRevision"])
        preparation = validate_preparation(status, supported)
        style = json_file(ROOT / "finish-python-style-status.json")
        assert style.get("consumerRevision") == status["consumerRevision"], style
        assert style.get("steps") and all(step.get("exitCode") == 0 for step in style["steps"]), style
        status["pythonStyle"] = {
            "receipt": "finish-python-style-status.json",
            "receiptSha256": sha256(ROOT / "finish-python-style-status.json"),
            "files": style.get("files", []),
            "steps": style["steps"],
        }
        assert sha256(ARCHIVE_MANIFEST) == archive_hash, "historical archive changed during run"
        clean_pin(PRODUCER, args.expected_producer)
        clean_pin(CONSUMER, status["consumerRevision"])
        status["phase"] = "ready-for-evidence-review"
        status["preparationReportSha256"] = sha256(ROOT / "prepared-finish/semantic-v8-report.json")
        status["allFinalDriversPassed"] = True
        status["pushesPerformed"] = False
        status["recorderPerformed"] = False
        status["mergePerformed"] = False
        save(status)
        print(
            json.dumps(
                {
                    "phase": status["phase"],
                    "producerRevision": status["producerRevision"],
                    "consumerRevision": status["consumerRevision"],
                    "supportedArtifacts": supported,
                    "preparationCounts": preparation["counts"],
                    "status": STATUS_PATH.name,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        status["phase"] = "stopped-for-review"
        status["error"] = str(error)
        status["jobsStillRunning"] = [name for name, process in jobs.items() if process.poll() is None]
        status["runningPids"] = {
            name: process.pid for name, process in jobs.items() if process.poll() is None
        }
        try:
            save(status)
        finally:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
