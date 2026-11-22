"""Narrow admission shape checks for source-session V8 artifacts.

The fixture is a producer-generated local structure control. CLI tests use an
explicit source-receipt double, so they do not claim production source or
pinned-tokenizer admission.
"""

import base64
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

from megatoken.data.native_v8 import validate_v8_artifact
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec
from scripts import admit_semantic_v8


def session_fixture():
    payload = (
        Path(__file__).parent / "fixtures/semantic-source-session-v8.json.gz.b64"
    ).read_bytes()
    fixture = json.loads(gzip.decompress(base64.b64decode(payload)))
    assert fixture["fixtureAuthority"] == (
        "producer-generated local source-session structure control; its source and tokenizer "
        "receipts are not production admission evidence"
    )
    return fixture["artifact"]


def refresh_nested_digest(document):
    sidecar = document["sidecar"]
    sidecar["sourceArtifactDigest"] = stable_algorithm_hash(sidecar["sourceArtifact"])


def malformed_component_timeline(document):
    component = document["sidecar"]["sourceArtifact"]["sidecar"]["collection"]["components"][0]
    component["public"]["timeline"] = [None]
    component["sidecar"]["publicDigest"] = stable_algorithm_hash(component["public"])


def generic_source_receipt(document, path, revision):
    return {
        "schema": "semantic-native-source-recheck/1",
        "producerRevision": revision,
        "producerTreeClean": True,
        "artifactSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sourceArtifactDigest": stable_algorithm_hash(document["sidecar"]["sourceArtifact"]),
        "publicDigest": stable_algorithm_hash(document["public"]),
        "contextDigest": stable_algorithm_hash(document["contexts"]),
        "sourceValidated": True,
        "projectionValidated": True,
        "replayValidated": True,
    }


def test_source_session_guard_accepts_the_current_nested_plan_and_native_replay():
    document = session_fixture()
    plan = admit_semantic_v8.validate_source_session_shape(document)
    assert plan["schema"] == "semantic-source-session-plan/1"
    assert len(plan["calls"]) == 2
    receipt = validate_v8_artifact(document)
    assert all(
        receipt["checks"][name] == "passed" for name in ("replay", "contexts", "continuation")
    )


def test_source_session_guard_accepts_a_bound_persisted_native_receipt():
    document = copy.deepcopy(session_fixture())
    sidecar = document["sidecar"]
    sidecar["admission"] = "admitted"
    sidecar["nativeAdmission"] = {
        "schema": "semantic-native-admission/8",
        "admitted": True,
        "publicDigest": sidecar["publicDigest"],
        "contextsDigest": sidecar["contextDigest"],
        "sourceArtifactDigest": sidecar["sourceArtifactDigest"],
    }
    plan = admit_semantic_v8.validate_source_session_shape(document)
    assert len(plan["calls"]) == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["sidecar"]["sourceArtifact"]["sidecar"].update(
                schema="semantic-session-source-evidence/99"
            ),
            "nested source authority",
        ),
        (
            lambda d: d["sidecar"]["sourceArtifact"]["sidecar"]["collection"]["plan"].update(
                calls=[]
            ),
            "plan or component count",
        ),
        (
            lambda d: d["sidecar"]["sourceArtifact"]["sidecar"]["collection"]["plan"].update(
                calls=None
            ),
            "plan or component count",
        ),
        (
            lambda d: d["sidecar"]["sourceArtifact"]["sidecar"].update(
                selectionOrigin="controlled-or-model-output-unverified"
            ),
            "nested source authority",
        ),
        (
            lambda d: d["sidecar"]["sourceArtifact"]["sidecar"]["collection"]["components"][0][
                "sidecar"
            ].update(sourceFamily="foreign-family"),
            "component evidence",
        ),
        (
            lambda d: d["sidecar"]["sourceArtifact"]["sidecar"]["collection"]["plan"]["calls"][
                0
            ].update(inputDigest="0" * 64),
            "call digest",
        ),
        (
            lambda d: d["sidecar"]["sourceArtifact"]["sidecar"]["collection"]["components"][
                0
            ].update(public=None),
            "component evidence",
        ),
        (
            malformed_component_timeline,
            "component timeline",
        ),
    ],
)
def test_source_session_guard_rejects_changed_nested_authority(mutate, message):
    document = copy.deepcopy(session_fixture())
    mutate(document)
    refresh_nested_digest(document)
    with pytest.raises(ValueError, match=message):
        admit_semantic_v8.validate_source_session_shape(document)


def test_admission_cli_rechecks_valid_session_shape_before_emitting_receipt(
    tmp_path, monkeypatch, capsys
):
    document = session_fixture()
    input_path, output_path, tokenizer_path = (
        tmp_path / name for name in ("session.json", "receipt.json", "tokenizer.json")
    )
    input_path.write_text(json.dumps(document))
    tokenizer_path.write_text("{}")
    consumer_revision, producer_revision = "a" * 40, "b" * 40
    monkeypatch.setattr(admit_semantic_v8, "checked_revision", lambda *_: consumer_revision)
    monkeypatch.setattr(
        ByteCodec, "from_tokenizer_json", lambda *_args, **_kwargs: ByteCodec.byte_fallback()
    )
    source_checks = []

    def source_check(value, path, _root, revision):
        source_checks.append((value, path, revision))
        return generic_source_receipt(value, path, producer_revision)

    monkeypatch.setattr(admit_semantic_v8, "recheck_source_binding", source_check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admit_semantic_v8.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--consumer-revision",
            consumer_revision,
            "--tokenizer-json",
            str(tokenizer_path),
            "--producer-root",
            str(tmp_path),
            "--producer-revision",
            producer_revision,
        ],
    )
    assert admit_semantic_v8.main() == 0
    capsys.readouterr()
    assert len(source_checks) == 1
    receipt = json.loads(output_path.read_text())
    assert receipt["admitted"] is True
    assert receipt["sourceVerification"]["sourceArtifactDigest"] == stable_algorithm_hash(
        document["sidecar"]["sourceArtifact"]
    )


def test_admission_cli_refuses_malformed_session_before_source_receipt_check(tmp_path, monkeypatch):
    document = session_fixture()
    document["sidecar"]["sourceArtifact"]["sidecar"]["collection"]["plan"]["calls"] = []
    refresh_nested_digest(document)
    input_path, output_path = tmp_path / "session.json", tmp_path / "receipt.json"
    input_path.write_text(json.dumps(document))
    revision = "a" * 40
    monkeypatch.setattr(admit_semantic_v8, "checked_revision", lambda *_: revision)
    monkeypatch.setattr(
        admit_semantic_v8,
        "recheck_source_binding",
        lambda *_: pytest.fail("malformed session must be rejected before producer source recheck"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admit_semantic_v8.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--consumer-revision",
            revision,
            "--producer-root",
            str(tmp_path),
            "--producer-revision",
            "b" * 40,
        ],
    )
    with pytest.raises(ValueError, match="plan or component count"):
        admit_semantic_v8.main()
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("admission", "native_receipt"),
    [
        ("locally_verified", {"schema": "semantic-native-admission/8", "admitted": True}),
        (
            "admitted",
            {
                "schema": "semantic-native-admission/8",
                "admitted": True,
                "publicDigest": "0" * 64,
                "contextsDigest": "0" * 64,
                "sourceArtifactDigest": "0" * 64,
            },
        ),
    ],
)
def test_source_session_guard_refuses_unbound_or_inconsistent_native_receipt(
    admission, native_receipt
):
    document = copy.deepcopy(session_fixture())
    document["sidecar"]["admission"] = admission
    document["sidecar"]["nativeAdmission"] = native_receipt
    with pytest.raises(ValueError, match="outer evidence"):
        admit_semantic_v8.validate_source_session_shape(document)
