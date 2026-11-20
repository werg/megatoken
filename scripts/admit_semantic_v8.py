#!/usr/bin/env python3
"""Write a conservative native engineering receipt for semantic execution/8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.native_v8 import prepare_native_v8_training, validate_v8_artifact
from megatoken.data.semantic_contract_v8 import CONTRACT, CONTRACT_DIGEST, INTEROPERABILITY_DIGEST

_REPOSITORY_GIT_ENV = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
}


def _git_environment():
    """Keep unrelated caller settings while targeting another checkout."""
    return {key: value for key, value in os.environ.items() if key not in _REPOSITORY_GIT_ENV}


def checked_revision(root, expected):
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, env=_git_environment()
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        env=_git_environment(),
    ).strip()
    if actual != expected or dirty:
        raise ValueError("V8 native receipt requires the exact clean consumer revision")
    if not Path(__import__("megatoken").__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Imported Python package is not this consumer checkout")
    return actual


def recheck_source_binding(document, input_path, producer_root, producer_revision):
    """Ask the exact clean JS producer checkout to replay source correspondence."""
    if producer_root is None or producer_revision is None:
        return None
    root = Path(producer_root).resolve()
    if not root.is_dir():
        return None  # Authority is not installed; this cannot admit the artifact.

    def check_producer_checkout():
        try:
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                timeout=10,
                env=_git_environment(),
            ).strip()
            top = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                text=True,
                timeout=10,
                env=_git_environment(),
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                text=True,
                timeout=10,
                env=_git_environment(),
            ).strip()
            subprocess.check_output(
                ["git", "ls-files", "--error-unmatch", "tools/recheck-semantic-native-source.mjs"],
                cwd=root,
                text=True,
                timeout=10,
                env=_git_environment(),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("V8 source authority is not a pinned producer checkout") from error
        if actual != producer_revision or Path(top).resolve() != root or dirty:
            raise ValueError("V8 source authority requires the exact clean producer revision")

    check_producer_checkout()
    command = [
        "node",
        str(root / "tools" / "recheck-semantic-native-source.mjs"),
        "--producer-root",
        str(root),
        "--producer-revision",
        producer_revision,
        "--input",
        str(input_path.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=_git_environment(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    check_producer_checkout()
    if completed.returncode != 0:
        raise ValueError("Pinned producer rejected semantic source correspondence")
    if len(completed.stdout.encode("utf-8")) > 65536 or completed.stderr:
        raise ValueError("Pinned producer source receipt was not bounded and clean")
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Pinned producer returned a malformed source receipt") from error
    sidecar = document.get("sidecar", {})
    expected = {
        "schema": "semantic-native-source-recheck/1",
        "producerRevision": producer_revision,
        "producerTreeClean": True,
        "artifactSha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "sourceArtifactDigest": stable_algorithm_hash(sidecar.get("sourceArtifact")),
        "publicDigest": stable_algorithm_hash(document.get("public")),
        "contextDigest": stable_algorithm_hash(document.get("contexts")),
        "sourceValidated": True,
        "projectionValidated": True,
        "replayValidated": True,
    }
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Pinned producer source receipt does not bind this artifact")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--producer-root", type=Path)
    parser.add_argument("--producer-revision")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and receipt paths must differ")
    root = Path(__file__).resolve().parents[1]
    revision = checked_revision(root, args.consumer_revision)
    if (args.producer_root is None) != (args.producer_revision is None):
        parser.error("--producer-root and --producer-revision must be provided together")
    document = json.loads(args.input.read_text())
    result = validate_v8_artifact(document)
    sidecar = document.get("sidecar", {})
    source_artifact = sidecar.get("sourceArtifact") if isinstance(sidecar, dict) else None
    public = document["public"]
    contexts = document["contexts"]
    source_binding = "untested"
    source_verification = None
    if isinstance(sidecar, dict) and sidecar.get("schema") in (
        "semantic-region-evidence/1",
        "semantic-protocol-region-evidence/1",
    ):
        if (
            sidecar.get("representation") != "semantic-trace/8"
            or sidecar.get("referenceStatus") != "agreed"
            or sidecar.get("optimizerSteps") != 0
            or sidecar.get("learningClaim") is not None
            or sidecar.get("sourceArtifactDigest")
            != stable_algorithm_hash(sidecar.get("sourceArtifact"))
            or sidecar.get("publicDigest") != stable_algorithm_hash(public)
            or sidecar.get("contextDigest") != stable_algorithm_hash(contexts)
        ):
            raise ValueError("V8 source evidence no longer binds this public projection")
        if sidecar.get("schema") == "semantic-protocol-region-evidence/1":
            if (
                sidecar.get("admission") != "locally_verified"
                or sidecar.get("profile") != "compositional-regions/1"
            ):
                raise ValueError("V8 protocol source evidence has an unsupported region profile")
            profile = sidecar.get("protocolProfile")
            extensions = sidecar.get("protocolExtensions", [])
            source_sidecar = (
                source_artifact.get("sidecar") if isinstance(source_artifact, dict) else None
            )
            source_projection = (
                source_sidecar.get("compiler") if isinstance(source_sidecar, dict) else None
            )
            if not isinstance(source_projection, dict):
                raise ValueError("V8 protocol source projection inventory is missing")
            source_extensions = source_projection.get("extensions", [])
            if profile == "symbols-ordinary-properties/1":
                if extensions not in (None, []) or source_extensions not in (None, []):
                    raise ValueError("Base protocol source evidence cannot claim extensions")
            elif profile in (
                "protocol-source-extensions/1",
                "promise-joins-source/1",
                "promise-executors-source/1",
            ):
                supported = {
                    "owned-binary-memory/1",
                    "builtin-iterator/1",
                    "class-private-source/1",
                    "synchronous-generator-source/1",
                    "promise-join/1",
                    "promise-executor/1",
                    "number-conversion-source/1",
                    "ordinary-data-proxy-source/1",
                    "ordinary-array-map-source/1",
                }
                if (
                    not isinstance(extensions, list)
                    or not extensions
                    or not all(type(value) is str and value in supported for value in extensions)
                    or len(extensions) != len(set(extensions))
                    or not isinstance(source_extensions, list)
                    or source_extensions != extensions
                ):
                    raise ValueError(
                        "V8 protocol source evidence has unsupported or mismatched extensions"
                    )
                if profile == "promise-joins-source/1" and extensions != ["promise-join/1"]:
                    raise ValueError("V8 Promise join source evidence has an invalid inventory")
                if profile == "promise-executors-source/1" and extensions not in (
                    ["promise-executor/1"],
                    ["promise-executor/1", "promise-join/1"],
                ):
                    raise ValueError("V8 Promise executor source evidence has an invalid inventory")
            else:
                raise ValueError("V8 protocol source evidence has an unsupported protocol profile")
        source_verification = recheck_source_binding(
            document, args.input, args.producer_root, args.producer_revision
        )
        if source_verification is not None:
            source_binding = "passed"
    tokenizer = {"status": "untested", "identity": None, "sha256": None}
    training = None
    if args.tokenizer_json is not None:
        raw = args.tokenizer_json.read_bytes()
        tokenizer = {
            "status": "untested",
            "identity": None,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        # ByteCodec performs the repository's vocabulary and pinned tokenizer
        # verification when the tokenizer dependency is installed.
        try:
            from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec

            codec = ByteCodec.from_tokenizer_json(
                args.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
            )
            tokenizer = {
                "status": "passed",
                "identity": codec.fingerprint,
                "sha256": LING_TOKENIZER_SHA256,
            }
            _, training = prepare_native_v8_training(document, codec=codec)
        except ImportError:
            tokenizer["status"] = "resource_limited"
    checks = dict(result["checks"])
    checks["tokenizer"] = tokenizer["status"]
    checks["sourceBinding"] = source_binding
    checks["masks"] = (
        "passed"
        if training is not None
        and training["maskValidation"]
        and training["generatedRecords"] == result["counts"]["generatedRecords"]
        and training["contexts"] == len(contexts)
        else "untested"
    )
    receipt = {
        "schema": "semantic-native-admission/8",
        "traceFormat": "semantic-trace/8",
        "contextFormat": "semantic-context/8",
        "representation": "mixed",
        "componentGate": "native-consumer-only",
        "publicDigest": stable_algorithm_hash(public),
        "contextsDigest": stable_algorithm_hash(contexts),
        "sourceArtifactDigest": stable_algorithm_hash(source_artifact)
        if source_artifact is not None
        else None,
        "sourceVerification": source_verification,
        "sourceProfile": sidecar.get("profile") if isinstance(sidecar, dict) else None,
        "protocolProfile": sidecar.get("protocolProfile") if isinstance(sidecar, dict) else None,
        "protocolExtensions": sidecar.get("protocolExtensions", [])
        if isinstance(sidecar, dict)
        else [],
        "datasetBindingDigest": stable_algorithm_hash(sidecar["datasetBinding"])
        if isinstance(sidecar, dict) and sidecar.get("datasetBinding") is not None
        else None,
        "consumerRevision": revision,
        "contractDigest": CONTRACT_DIGEST,
        "fixtureDigest": CONTRACT["fixtureDigest"],
        "interoperabilityDigest": INTEROPERABILITY_DIGEST,
        "tokenizer": tokenizer,
        "checks": checks,
        "counts": {
            "records": len(public["timeline"]),
            "contexts": result["counts"]["contexts"],
            "generatedRecords": result["counts"]["generatedRecords"],
            "algorithmRecords": result["counts"].get("algorithmRecords", 0),
            "regionRecords": result["counts"].get("regionRecords", 0),
            "detailedRecords": result["counts"].get("detailedRecords", 0),
            "protocolRecords": result["counts"].get("protocolRecords", 0),
            "generatedTokens": training["generatedTokens"] if training else None,
            "observedTokens": training["observedTokens"] if training else None,
        },
        "framing": {
            "kind": training["framing"] if training else "untested",
            "eosTokenId": training["eosTokenId"] if training else None,
            "terminalTokens": training["terminalTokens"] if training else None,
            "physicalRecords": training["physicalRecords"] if training else None,
        },
        "nativeTrainingDigest": training["nativeTrainingDigest"] if training else None,
        "optimizerSteps": 0,
        "learningClaim": None,
        "admitted": bool(
            training is not None
            and all(
                checks.get(name) == "passed"
                for name in (
                    "replay",
                    "contexts",
                    "continuation",
                    "masks",
                    "sourceBinding",
                    "tokenizer",
                )
            )
        ),
    }
    # Engineering fixtures lack private source-bound records and exact physical
    # token rows. Never turn structural success into model-serving admission.
    if args.output.resolve() == args.input.resolve():
        parser.error("Input and receipt paths must differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(receipt, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
