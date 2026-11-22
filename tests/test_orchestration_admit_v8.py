"""CLI receipt boundaries with explicit source/tokenizer unit-test doubles.

Native public replay and physical row construction run normally. The temporary
producer below only emits supplied receipt bytes; it is not source certification.
"""

import base64
import copy
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec
from scripts import admit_semantic_v8, qualify_semantic_v8_lifecycle


def document():
    path = Path(__file__).parent / "fixtures/semantic-orchestration-teacher-v8.json.gz.b64"
    return json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))


def producer_stub(root):
    tools = root / "tools"
    tools.mkdir(parents=True)
    (tools / "recheck-semantic-native-source.mjs").write_text(
        "import fs from 'node:fs'; const input=process.argv[process.argv.indexOf('--input')+1];"
        "process.stdout.write(fs.readFileSync(input+'.unit-receipt.json','utf8'));"
    )
    env = admit_semantic_v8._git_environment()
    subprocess.run(["git", "init", "--quiet"], cwd=root, env=env, check=True)
    subprocess.run(["git", "add", "."], cwd=root, env=env, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Explicit source receipt test double",
        ],
        cwd=root,
        env=env,
        check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, env=env, text=True
    ).strip()


def source_receipt(value, path, revision):
    result = {
        "schema": "semantic-native-source-recheck/1",
        "producerRevision": revision,
        "producerTreeClean": True,
        "artifactSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sourceArtifactDigest": stable_algorithm_hash(value["sidecar"]["sourceArtifact"]),
        "publicDigest": stable_algorithm_hash(value["public"]),
        "contextDigest": stable_algorithm_hash(value["contexts"]),
        "sourceValidated": True,
        "projectionValidated": True,
        "replayValidated": True,
    }
    if "orchestrationTranscript" in value["sidecar"]:
        result.update(
            orchestrationValidated=True,
            orchestrationTranscriptDigest=stable_algorithm_hash(
                value["sidecar"]["orchestrationTranscript"]
            ),
        )
    return result


def test_source_cli_requires_exact_conditional_transcript_receipt(tmp_path):
    root = tmp_path / "producer"
    revision = producer_stub(root)
    value = document()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(value))
    receipt_path = Path(str(path) + ".unit-receipt.json")
    expected = source_receipt(value, path, revision)
    receipt_path.write_text(json.dumps(expected) + "\n")
    assert admit_semantic_v8.recheck_source_binding(value, path, root, revision) == expected
    for field, altered in (
        ("orchestrationValidated", None),
        ("orchestrationValidated", False),
        ("orchestrationTranscriptDigest", None),
        ("orchestrationTranscriptDigest", "0" * 64),
    ):
        changed = copy.deepcopy(expected)
        if altered is None:
            del changed[field]
        else:
            changed[field] = altered
        receipt_path.write_text(json.dumps(changed) + "\n")
        with pytest.raises(ValueError, match="does not bind this artifact"):
            admit_semantic_v8.recheck_source_binding(value, path, root, revision)
    del value["sidecar"]["orchestrationTranscript"]
    path.write_text(json.dumps(value))
    naked = source_receipt(value, path, revision)
    receipt_path.write_text(json.dumps(naked) + "\n")
    assert admit_semantic_v8.recheck_source_binding(value, path, root, revision) == naked
    for extra in (
        {"orchestrationValidated": True},
        {"orchestrationTranscriptDigest": expected["orchestrationTranscriptDigest"]},
    ):
        receipt_path.write_text(json.dumps({**naked, **extra}) + "\n")
        with pytest.raises(ValueError, match="without its transcript"):
            admit_semantic_v8.recheck_source_binding(value, path, root, revision)


def test_admission_cli_binds_all_four_materialized_targets(tmp_path, monkeypatch, capsys):
    value = document()
    path, output, tokenizer = (
        tmp_path / name for name in ("input.json", "receipt.json", "codec.json")
    )
    path.write_text(json.dumps(value))
    tokenizer.write_text("{}")
    revision = "a" * 40
    monkeypatch.setattr(admit_semantic_v8, "checked_revision", lambda *_: revision)
    monkeypatch.setattr(
        ByteCodec, "from_tokenizer_json", lambda *_a, **_k: ByteCodec.byte_fallback()
    )
    monkeypatch.setattr(
        admit_semantic_v8,
        "recheck_source_binding",
        lambda *_: source_receipt(value, path, "b" * 40),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admit_semantic_v8.py",
            "--input",
            str(path),
            "--output",
            str(output),
            "--consumer-revision",
            revision,
            "--tokenizer-json",
            str(tokenizer),
            "--producer-root",
            str(tmp_path),
            "--producer-revision",
            "b" * 40,
        ],
    )
    assert admit_semantic_v8.main() == 0
    capsys.readouterr()
    receipt = json.loads(output.read_text())
    assert receipt["admitted"] is True
    assert receipt["checks"]["orchestration"] == "passed"
    assert receipt["counts"]["orchestrationRecords"] == 4
    assert receipt["orchestrationTranscriptDigest"] == stable_algorithm_hash(
        value["sidecar"]["orchestrationTranscript"]
    )
    prepare = admit_semantic_v8.prepare_native_v8_training

    def incomplete(*args, **kwargs):
        episode, counts = prepare(*args, **kwargs)
        return episode, {**counts, "orchestrationRecords": 3}

    monkeypatch.setattr(admit_semantic_v8, "prepare_native_v8_training", incomplete)
    assert admit_semantic_v8.main() == 0
    capsys.readouterr()
    refused = json.loads(output.read_text())
    assert refused["checks"]["orchestration"] == "untested"
    assert refused["admitted"] is False


def test_lifecycle_collector_reads_saved_rows_and_refuses_detached_counts(tmp_path):
    # This helper injects source/tokenizer audit receipts explicitly; the
    # physical episode and its on-disk prepared corpus are constructed normally.
    from tests.test_semantic_orchestration_corpus_v8 import prepare_episode

    value, _, _, _, _, episode, report = prepare_episode(tmp_path)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(value))
    result = qualify_semantic_v8_lifecycle.qualify_orchestration_preparation(
        source, tmp_path / "prepared-v8", report
    )
    assert result["transcriptDigest"] == stable_algorithm_hash(
        value["sidecar"]["orchestrationTranscript"]
    )
    assert result["logicalLifecycleTargets"] == 4
    assert result["physicalLifecycleTargetParts"] == sum(
        row.logical_id.startswith("semantic:v8:orchestration:target:") for row in episode.records
    )
    assert result["trackOwners"] == ["track:0", "track:1"]
    assert result["optimizerExposurePerTarget"] is None
    assert result["actualModelFragmentBudgetEquivalence"] is None
    changed = copy.deepcopy(report)
    changed["counts"]["orchestrationRecords"] = 3
    with pytest.raises(ValueError, match="omitted lifecycle targets"):
        qualify_semantic_v8_lifecycle.qualify_orchestration_preparation(
            source, tmp_path / "prepared-v8", changed
        )
    value["sidecar"].pop("orchestrationTranscript")
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Detached orchestration count"):
        qualify_semantic_v8_lifecycle.qualify_orchestration_preparation(
            source, tmp_path / "prepared-v8", report
        )
