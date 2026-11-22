"""One original TLA graph through exact source, native replay and physical packing.

The one-member registry is isolated from all immutable production populations.
No optimizer runs and no model-behavior claim follows from this technical gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from megatoken.controls import canonical_semantic_observation
from megatoken.data import PreparedDataset, WindowConfig, iter_windows
from megatoken.data.protocol_v8 import decode_protocol_record
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec


def run(command, cwd, timeout):
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False
    )
    assert result.returncode == 0, result.stdout[-4000:] + "\n" + result.stderr[-4000:]
    return result


def revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def test_original_module_scheduler_artifact_is_rechecked_and_physically_packed(tmp_path):
    required = ("TRACING_JS_SELF_ROOT", "MEGATOKEN_ROOT", "MEGATOKEN_TOKENIZER_JSON")
    available = {key: os.environ.get(key) for key in required}
    if not any(available.values()):
        pytest.skip("Exact paired producer, consumer and tokenizer assets are required")
    assert all(available.values()), f"Partially configured module artifact gate: {available!r}"
    producer = Path(available["TRACING_JS_SELF_ROOT"]).resolve()
    consumer = Path(available["MEGATOKEN_ROOT"]).resolve()
    tokenizer = Path(available["MEGATOKEN_TOKENIZER_JSON"]).resolve()
    artifact_path = tmp_path / "module-scheduler-source-artifact.json"
    prepared_path = tmp_path / "prepared-module-scheduler"
    run(
        [
            "node",
            str(producer / "tools/emit-semantic-v8-module-scheduler-artifact.mjs"),
            "--output",
            str(artifact_path),
        ],
        producer,
        600,
    )
    artifact = json.loads(artifact_path.read_text())
    generated = [row for row in artifact["public"]["timeline"] if row["mode"] == "generated"]
    assert len(generated) == len(artifact["contexts"]) == 708
    source = artifact["sidecar"]["sourceArtifact"]["sidecar"]
    assert source["compiler"]["moduleSchedulerOwnership"]["ownedEventCount"] == 10
    assert source["moduleSchedulerPlan"]["schema"] == "semantic-bound-module-scheduler-plan/1"
    assert len(artifact["sidecar"]["splitIdentity"]["registry"]["sources"]) == 1
    run(
        [
            sys.executable,
            str(consumer / "scripts/prepare_semantic_v8.py"),
            "--input",
            str(artifact_path),
            "--output",
            str(prepared_path),
            "--producer-root",
            str(producer),
            "--producer-revision",
            revision(producer),
            "--consumer-revision",
            revision(consumer),
            "--tokenizer-json",
            str(tokenizer),
            "--max-positions",
            "8192",
            "--max-history",
            "1024",
            "--source-recheck-timeout-seconds",
            "600",
        ],
        consumer,
        1800,
    )
    report_bytes = (prepared_path / "semantic-v8-report.json").read_bytes()
    report = json.loads(report_bytes)
    assert report["counts"]["attempted"] == report["counts"]["admitted"] == 1
    assert report["counts"].get("refused", 0) == 0
    assert report["optimizerSteps"] == 0 and report["learningClaim"] is None
    with PreparedDataset(prepared_path, split=None) as prepared:
        assert len(prepared) == 1
        episode = prepared[0]
    semantic = json.loads(episode.supervision.annotations_json)["semantic_execution"]
    assert semantic["admission"] == "admitted" and semantic["modelExecutionOutcome"] == "unmeasured"
    assert all(
        semantic["sourceAdmission"][key] is True
        for key in ("sourceValidated", "projectionValidated", "replayValidated")
    )
    assert all(
        semantic["nativeAdmission"]["checks"][key] == "passed"
        for key in ("replay", "contexts", "continuation", "masks", "sourceBinding", "tokenizer")
    )
    assert semantic["nativeAdmission"]["tokenizer"]["sha256"] == LING_TOKENIZER_SHA256
    assert semantic["nativeTrainingDigest"]
    codec = ByteCodec.from_tokenizer_json(tokenizer, expected_sha256=LING_TOKENIZER_SHA256)
    expected_targets = {
        f"semantic:v8:target:{row['sequence']}": row["fields"]["line"].encode() for row in generated
    }
    expected_contexts = {
        f"semantic:v8:context:{row['sequence']}": canonical_semantic_observation(context)
        for row, context in zip(generated, artifact["contexts"], strict=True)
    }
    groups = {parts[0].logical_id: parts for parts in episode.iter_logical_records()}
    context_parts = 0
    for identity, payload in expected_contexts.items():
        parts = groups[identity]
        assert b"".join(row.payload for row in parts) == payload
        assert all(row.role == "system" and not row.policy_mask for row in parts)
        context_parts += len(parts)
    for identity, payload in expected_targets.items():
        parts = groups[identity]
        assert b"".join(row.payload for row in parts) == payload
        assert all(row.role == "assistant" and row.policy_mask for row in parts)
        sequence = identity.rsplit(":", 1)[1]
        context = groups["semantic:v8:context:" + sequence]
        assert any(
            edge.writer_id == context[-1].record_id
            and edge.reader_id == parts[0].record_id
            and edge.kind == "v8-public-context"
            for edge in episode.dependencies
        )
    windows = tuple(
        iter_windows(episode, codec, WindowConfig(max_window_tokens=8192, max_decisions=4))
    )
    encoded = [row for window in windows for row in window.records]
    assert len(encoded) == len(episode.records)
    assert all(row.tokens <= 8192 for row in encoded)
    assert all(row.P == 0 for row in encoded if not row.record.policy_mask)
    selected = [row for row in encoded if row.record.policy_mask]
    assert all(row.P > 0 and len(row.free_mask) == len(row.token_ids) for row in selected)
    assert {row.record.logical_id for row in selected} == set(expected_targets)
    counts = Counter()
    for payload in expected_targets.values():
        if payload.startswith(b"protocol "):
            counts[decode_protocol_record(payload.decode())["kind"]] += 1
    for kind in (
        "module.scheduler.initialize",
        "async.call",
        "async.await",
        "async.resume",
        "async.complete",
        "generator.create",
        "generator.resume",
    ):
        assert counts[kind] > 0
    assert counts["module.scheduler.initialize"] == 1
    context_tokens = [len(codec.encode(payload)) for payload in expected_contexts.values()]
    manifest_bytes = (prepared_path / "manifest.json").read_bytes()
    evidence = {
        "schema": "semantic-v8-module-scheduler-artifact-pack/1",
        "producerRevision": semantic["sourceAdmission"]["producerRevision"],
        "consumerRevision": semantic["nativeAdmission"]["consumerRevision"],
        "artifactSha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "sourceFamily": source["sourceFamily"],
        "sourceRegistry": artifact["sidecar"]["splitIdentity"],
        "datasetBinding": artifact["sidecar"]["datasetBinding"],
        "sourceAdmission": semantic["sourceAdmission"],
        "nativeAdmission": semantic["nativeAdmission"],
        "nativeTrainingDigest": semantic["nativeTrainingDigest"],
        "tokenizerEvidence": semantic["tokenizerEvidence"],
        "logicalGeneratedTargets": len(expected_targets),
        "physicalSelectedTargets": len(selected),
        "logicalContexts": len(expected_contexts),
        "physicalContextParts": context_parts,
        "maxWholeContextTokens": max(context_tokens),
        "contextsOver8192Tokens": sum(n > 8192 for n in context_tokens),
        "allContextBytesReassembled": True,
        "allTargetBytesReassembled": True,
        "allContextDependenciesPreserved": True,
        "allPhysicalRecordsWithin8192Tokens": True,
        "criticalTargetCounts": dict(sorted(counts.items())),
        "packedWindows": len(windows),
        "totalScoredPositions": sum(row.P for row in selected),
        "selectedTokenAndMaskDigest": hashlib.sha256(
            json.dumps(
                [
                    {
                        "recordId": row.record.record_id,
                        "tokenIds": list(row.token_ids),
                        "freeMask": list(row.free_mask),
                    }
                    for row in selected
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "preparedReportRaw": report_bytes.decode(),
        "preparedReportSha256": hashlib.sha256(report_bytes).hexdigest(),
        "preparedManifestRaw": manifest_bytes.decode(),
        "preparedManifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "productionPopulationChanged": False,
        "optimizerUpdates": 0,
        "learnedSuccess": None,
    }
    print(
        "MODULE_SCHEDULER_ARTIFACT_PACK_EVIDENCE_JSON "
        + json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    )
