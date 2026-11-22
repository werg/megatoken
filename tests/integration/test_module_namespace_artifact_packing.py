"""Actual source-qualified namespace artifact through native V8 preparation and packing.

This isolated technical fixture uses its own one-member registry. It is not
merged into the immutable production corpus population and makes no learning
claim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from megatoken.data import PreparedDataset, WindowConfig, iter_windows
from megatoken.data.protocol_v8 import ProtocolV8Error, decode_protocol_record
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec


def _run_checked(command, *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=cwd, check=False, timeout=timeout, capture_output=True, text=True
    )
    if result.returncode:
        raise AssertionError(
            f"Command failed ({result.returncode}): {command!r}\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
    return result


def _revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def test_actual_namespace_artifact_is_source_rechecked_and_packed(tmp_path):
    producer = Path(os.environ["TRACING_JS_SELF_ROOT"]).resolve()
    consumer = Path(os.environ["MEGATOKEN_ROOT"]).resolve()
    tokenizer = Path(os.environ["MEGATOKEN_TOKENIZER_JSON"]).resolve()
    artifact_path = tmp_path / "namespace-source-artifact.json"
    prepared_path = tmp_path / "prepared-namespace-v8"

    emitter = producer / "tools/emit-semantic-v8-namespace-artifact.mjs"
    _run_checked(
        ["node", str(emitter), "--output", str(artifact_path)],
        cwd=producer,
        timeout=180,
    )
    assert artifact_path.is_file()
    artifact = json.loads(artifact_path.read_text())
    program = next(row for row in artifact["public"]["timeline"] if row["kind"] == "program")
    plan = program["fields"]["regionProgram"]["moduleNamespacePlan"]
    assert plan["schema"] == "semantic-module-namespace-plan/1"
    identity = artifact["sidecar"]["splitIdentity"]
    registry = identity["registry"]
    assert registry["digest"] == artifact["sidecar"]["familyRegistryDigest"]
    assert len(registry["sources"]) == 1
    assert registry["sources"][0]["id"] == identity["sourceId"]

    script = consumer / "scripts/prepare_semantic_v8.py"
    _run_checked(
        [
            sys.executable,
            str(script),
            "--input",
            str(artifact_path),
            "--output",
            str(prepared_path),
            "--producer-root",
            str(producer),
            "--producer-revision",
            _revision(producer),
            "--consumer-revision",
            _revision(consumer),
            "--tokenizer-json",
            str(tokenizer),
            "--max-positions",
            "8192",
            "--max-history",
            "1024",
        ],
        cwd=consumer,
        timeout=300,
    )

    report = json.loads((prepared_path / "semantic-v8-report.json").read_text())
    assert report["counts"]["attempted"] == 1
    assert report["counts"]["admitted"] == 1
    assert report["counts"].get("refused", 0) == 0
    assert report["optimizerSteps"] == 0
    assert report["learningClaim"] is None

    with PreparedDataset(prepared_path, split=None) as prepared:
        assert len(prepared) == 1
        episode = prepared[0]
    assert episode.records
    assert any(record.policy_mask and record.role == "assistant" for record in episode.records)
    semantic = json.loads(episode.supervision.annotations_json)["semantic_execution"]
    assert semantic["admission"] == "admitted"
    assert semantic["modelExecutionOutcome"] == "unmeasured"
    assert semantic["sourceAdmission"]["sourceValidated"] is True
    assert semantic["sourceAdmission"]["projectionValidated"] is True
    assert semantic["nativeAdmission"]["checks"]["sourceBinding"] == "passed"
    assert semantic["nativeAdmission"]["checks"]["tokenizer"] == "passed"
    assert semantic["nativeAdmission"]["tokenizer"]["sha256"] == LING_TOKENIZER_SHA256
    assert semantic["nativeAdmission"]["counts"]["protocolRecords"] > 0
    assert semantic["nativeTrainingDigest"]
    assert semantic["tokenizerEvidence"]["sha256"] == LING_TOKENIZER_SHA256

    codec = ByteCodec.from_tokenizer_json(tokenizer, expected_sha256=LING_TOKENIZER_SHA256)
    required_kinds = {
        "namespace.create",
        "cell.initialize",
        "cell.write",
        "namespace.keys",
        "namespace.read",
        "namespace.descriptor",
    }
    packed_targets = {}
    for window in iter_windows(
        episode, codec, WindowConfig(max_window_tokens=8192, max_decisions=4)
    ):
        for encoded in window.records:
            record = encoded.record
            if record.role != "assistant" or not record.policy_mask:
                continue
            try:
                protocol_record = decode_protocol_record(record.payload.decode("utf-8"))
            except ProtocolV8Error:
                continue
            if protocol_record["kind"] not in required_kinds:
                continue
            assert encoded.token_ids
            assert len(encoded.token_ids) == len(encoded.free_mask)
            assert encoded.P > 0
            packed_targets[protocol_record["kind"]] = encoded
    assert set(packed_targets) == required_kinds
