#!/usr/bin/env python3
"""Write a conservative native engineering receipt for semantic execution/8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from megatoken.data.native_v8 import prepare_native_v8_training, validate_v8_artifact
from megatoken.data.semantic_contract_v8 import CONTRACT, CONTRACT_DIGEST, INTEROPERABILITY_DIGEST
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.source_identity_v8 import (
    DIRECT_SOURCE_SCHEMA,
    direct_source_identity_v8,
)

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

_SOURCE_SESSION_SCHEMA = "semantic-session-region-evidence/1"
_SOURCE_SESSION_ARTIFACT_SCHEMA = "semantic-session-source-evidence/1"
_SOURCE_SESSION_COLLECTION_SCHEMA = "semantic-source-session-collection/1"
_SOURCE_SESSION_PLAN_SCHEMA = "semantic-source-session-plan/1"
_SOURCE_SESSION_COMPONENT_SCHEMAS = {
    "semantic-region-evidence/1",
    "semantic-protocol-region-evidence/1",
    DIRECT_SOURCE_SCHEMA,
}
_SOURCE_SELECTION_ORIGINS = {"owned-interpreter-teacher", "certified_teacher"}


def validate_source_session_shape(document):
    """Validate the narrow source-session envelope before pinned source replay.

    The producer recheck remains authoritative for each original reference and
    the composed public projection. This guard verifies the exact nested wire
    shape and digest relationships consumed by that recheck.
    """
    sidecar = document.get("sidecar") if isinstance(document, dict) else None
    if not isinstance(sidecar, dict) or sidecar.get("schema") != _SOURCE_SESSION_SCHEMA:
        raise ValueError("V8 source session has an unsupported outer evidence schema")
    source_artifact = sidecar.get("sourceArtifact")
    source_sidecar = source_artifact.get("sidecar") if isinstance(source_artifact, dict) else None
    target = source_artifact.get("public") if isinstance(source_artifact, dict) else None
    admission = sidecar.get("admission")
    native_admission = sidecar.get("nativeAdmission")
    if (
        admission not in {"locally_verified", "admitted"}
        or sidecar.get("referenceStatus") != "agreed"
        or sidecar.get("representation") != "semantic-trace/8"
        or sidecar.get("profile") != "compositional-regions/1"
        or sidecar.get("selectionOrigin") != "owned-interpreter-teacher"
        or sidecar.get("optimizerSteps") != 0
        or sidecar.get("learningClaim") is not None
        or (admission == "locally_verified" and native_admission is not None)
        or (
            admission == "admitted"
            and (
                not isinstance(native_admission, dict)
                or native_admission.get("schema") != "semantic-native-admission/8"
                or native_admission.get("admitted") is not True
                or native_admission.get("publicDigest") != sidecar.get("publicDigest")
                or native_admission.get("contextsDigest") != sidecar.get("contextDigest")
                or native_admission.get("sourceArtifactDigest")
                != sidecar.get("sourceArtifactDigest")
            )
        )
        or sidecar.get("sourceArtifactDigest") != stable_algorithm_hash(source_artifact)
        or sidecar.get("publicDigest") != stable_algorithm_hash(document.get("public"))
        or sidecar.get("contextDigest") != stable_algorithm_hash(document.get("contexts"))
    ):
        raise ValueError("V8 source-session receipt does not bind its outer evidence")
    if (
        not isinstance(source_sidecar, dict)
        or not isinstance(target, dict)
        or source_sidecar.get("schema") != _SOURCE_SESSION_ARTIFACT_SCHEMA
        or source_sidecar.get("admission") != "locally_verified"
        or source_sidecar.get("referenceStatus") != "agreed"
        or source_sidecar.get("selectionOrigin") != "owned-interpreter-teacher"
        or source_sidecar.get("optimizerSteps") != 0
        or source_sidecar.get("learningClaim") is not None
        or target.get("schema") != "semantic-session-source-target/1"
    ):
        raise ValueError("V8 source-session nested source authority is malformed")

    family = sidecar.get("sourceFamily")
    if (
        not isinstance(family, str)
        or not family
        or sidecar.get("family") != family
        or source_sidecar.get("sourceFamily") != family
        or source_sidecar.get("sourceLineage") != family
    ):
        raise ValueError("V8 source-session family binding is malformed")
    collection = source_sidecar.get("collection")
    plan = collection.get("plan") if isinstance(collection, dict) else None
    components = collection.get("components") if isinstance(collection, dict) else None
    calls = plan.get("calls") if isinstance(plan, dict) else None
    if (
        not isinstance(collection, dict)
        or collection.get("schema") != _SOURCE_SESSION_COLLECTION_SCHEMA
        or not isinstance(plan, dict)
        or plan.get("schema") != _SOURCE_SESSION_PLAN_SCHEMA
        or not isinstance(components, list)
        or not 2 <= len(components) <= 16
        or not isinstance(calls, list)
        or len(components) != len(calls)
        or any(not isinstance(call, dict) for call in calls)
    ):
        raise ValueError("V8 source-session plan or component count is invalid")

    reference = source_sidecar.get("referenceInput")
    source = reference.get("source") if isinstance(reference, dict) else None
    entry = reference.get("entry") if isinstance(reference, dict) else None
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(entry, str)
        or not entry
        or not isinstance(reference.get("invocations"), list)
        or len(reference["invocations"]) != len(components)
        or plan.get("family") != family
        or plan.get("sourceDigest") != stable_algorithm_hash(source)
        or plan.get("entry") != entry
        or not isinstance(plan.get("sessionId"), str)
        or not plan["sessionId"]
        or type(plan.get("seed")) is not int
        or plan["seed"] < 0
        or (
            plan.get("producerRevision") is not None
            and (
                not isinstance(plan.get("producerRevision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", plan["producerRevision"]) is None
            )
        )
        or target.get("sourceDigest") != plan["sourceDigest"]
        or target.get("entry") != entry
        or target.get("componentPublicDigests") != [call.get("publicDigest") for call in calls]
    ):
        raise ValueError("V8 source-session source or reference binding is invalid")

    provenance = source_sidecar.get("sourceProvenance")
    if provenance != {"family": family, "sourceDigest": plan["sourceDigest"], "seed": plan["seed"]}:
        raise ValueError("V8 source-session provenance differs from its collection plan")

    program_fields = None
    component_public_digests = []
    for index, (component, call) in enumerate(zip(components, calls, strict=True)):
        component_sidecar = component.get("sidecar") if isinstance(component, dict) else None
        component_public = component.get("public") if isinstance(component, dict) else None
        if (
            not isinstance(component, dict)
            or not isinstance(component_public, dict)
            or component_public.get("schema") != "semantic-execution/8"
            or not isinstance(component.get("contexts"), list)
            or not isinstance(component_sidecar, dict)
            or component_sidecar.get("schema") not in _SOURCE_SESSION_COMPONENT_SCHEMAS
            or component_sidecar.get("admission") != "locally_verified"
            or component_sidecar.get("referenceStatus") != "agreed"
            or component_sidecar.get("optimizerSteps") != 0
            or component_sidecar.get("learningClaim") is not None
            or component_sidecar.get("nativeAdmission") is not None
            or any(
                key in component_sidecar
                for key in ("datasetBinding", "splitIdentity", "orchestrationTranscript")
            )
            or component_sidecar.get("family") != family
            or component_sidecar.get("sourceFamily") != family
            or (
                "selectionOrigin" in component_sidecar
                and component_sidecar.get("selectionOrigin") not in _SOURCE_SELECTION_ORIGINS
            )
        ):
            raise ValueError("V8 source-session component evidence is malformed")

        if component_sidecar["schema"] == DIRECT_SOURCE_SCHEMA:
            direct_source_identity_v8(component_sidecar)
            component_source_sidecar = component_sidecar
        else:
            component_source_artifact = component_sidecar.get("sourceArtifact")
            component_source_sidecar = (
                component_source_artifact.get("sidecar")
                if isinstance(component_source_artifact, dict)
                else None
            )
            if (
                not isinstance(component_source_sidecar, dict)
                or component_sidecar.get("sourceArtifactDigest")
                != stable_algorithm_hash(component_source_artifact)
                or component_sidecar.get("publicDigest")
                != stable_algorithm_hash(component.get("public"))
                or component_sidecar.get("contextDigest")
                != stable_algorithm_hash(component.get("contexts"))
            ):
                raise ValueError("V8 source-session component digest binding is invalid")
        if not isinstance(component_source_sidecar, dict):
            raise ValueError("V8 source-session component source authority is malformed")
        origin = component_source_sidecar.get("selectionOrigin")
        component_reference = component_source_sidecar.get("referenceInput")
        if (
            (
                component_sidecar["schema"] == "semantic-region-evidence/1"
                and origin not in _SOURCE_SELECTION_ORIGINS
            )
            or any(
                "selectionOrigin" in evidence
                and evidence["selectionOrigin"] not in _SOURCE_SELECTION_ORIGINS
                for evidence in (component_sidecar, component_source_sidecar)
            )
            or not isinstance(component_reference, dict)
            or component_reference.get("source") != source
            or component_reference.get("entry") != entry
            or component_source_sidecar.get("sourceFamily") != family
            or reference["invocations"][index] != component_reference
        ):
            raise ValueError("V8 source-session component source is not the declared family source")

        timeline = component_public.get("timeline")
        if not isinstance(timeline, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("fields"), dict) for row in timeline
        ):
            raise ValueError("V8 source-session component timeline is missing")
        programs = [row for row in timeline if row.get("kind") == "program"]
        roots = [
            row
            for row in timeline
            if row.get("kind") == "invocation"
            and row.get("fields", {}).get("parentInvocationId") is None
        ]
        if len(programs) != 1 or len(roots) != 1:
            raise ValueError("V8 source-session components need one observed program and root call")
        program, root = programs[0], roots[0]
        current_program = program["fields"]
        root_fields = root["fields"]
        if (
            program.get("mode") != "observed"
            or root.get("mode") != "observed"
            or current_program.get("session") != plan["sessionId"]
            or current_program.get("programId") != "program1"
            or current_program.get("source") != source
            or current_program.get("entry") != entry
            or current_program.get("digest") != plan["sourceDigest"]
            or root_fields.get("session") != plan["sessionId"]
            or root_fields.get("programId") != "program1"
            or root_fields.get("invocationId") != f"ingress{index + 1}"
            or root_fields.get("track") != "worker1"
            or root_fields.get("realmId") != f"realm{index + 1}"
            or root_fields.get("realmPolicy") != "fresh"
            or root_fields.get("triggerId") != f"outside{index + 1}"
            or root_fields.get("activation") != "call"
            or root_fields.get("inputGraph") is None
        ):
            raise ValueError("V8 source-session component root differs from its fixed source plan")
        if program_fields is None:
            program_fields = current_program
        elif current_program != program_fields:
            raise ValueError("V8 source-session immutable program declarations differ")
        component_public_digest = stable_algorithm_hash(component_public)
        component_context_digest = stable_algorithm_hash(component["contexts"])
        component_public_digests.append(component_public_digest)
        expected_call = {
            "component": index,
            "invocationId": root_fields["invocationId"],
            "realmId": root_fields["realmId"],
            "triggerId": root_fields["triggerId"],
            "inputDigest": stable_algorithm_hash(root_fields["inputGraph"]),
            "publicDigest": component_public_digest,
            "contextsDigest": component_context_digest,
        }
        if call != expected_call:
            raise ValueError("V8 source-session call digest differs from its original component")
    if component_public_digests != target["componentPublicDigests"]:
        raise ValueError("V8 source-session target differs from its component public artifacts")
    return plan


def _git_environment():
    """Keep unrelated caller settings while targeting another checkout."""
    return {key: value for key, value in os.environ.items() if key not in _REPOSITORY_GIT_ENV}


def checked_revision(root, expected=None):
    """Capture current clean code, optionally constrained to an earlier run cut."""
    root = Path(root).resolve()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, env=_git_environment()
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        env=_git_environment(),
    ).strip()
    if (
        re.fullmatch(r"[a-f0-9]{40}", actual) is None
        or dirty
        or (expected is not None and actual != expected)
    ):
        raise ValueError("V8 native receipt requires the exact clean consumer revision")
    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, env=_git_environment()
    ).strip()
    if Path(top).resolve() != root:
        raise ValueError("V8 native receipt requires the exact consumer repository root")
    # Include ignored importable files: package __file__ alone cannot reveal a
    # shadow module. Normal source-associated bytecode caches are not new code.
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "-z", "--", "src", "scripts"],
        cwd=root,
        text=True,
        env=_git_environment(),
    ).split("\0")
    if any(
        re.search(r"\.(?:py|pyc|pyo|so|pyd)$", name)
        and not re.search(r"(?:^|/)__pycache__/[^/]+\.pyc$", name)
        for name in untracked
        if name
    ):
        raise ValueError("V8 native consumer contains untracked importable code")
    if not Path(__import__("megatoken").__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Imported Python package is not this consumer checkout")
    return actual


def recheck_source_binding(
    document, input_path, producer_root, producer_revision, *, timeout_seconds=30
):
    """Ask the exact clean JS producer checkout to replay source correspondence."""
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 900:
        raise ValueError("Source recheck timeout must be an integer from 1 through 900 seconds")
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
            timeout=timeout_seconds,
            env=_git_environment(),
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            "Pinned producer source replay exceeded its declared time bound"
        ) from error
    except FileNotFoundError as error:
        raise ValueError("Pinned producer source replay executable is unavailable") from error
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
        "sourceArtifactDigest": stable_algorithm_hash(sidecar["sourceArtifact"])
        if sidecar.get("sourceArtifact") is not None
        else None,
        "publicDigest": stable_algorithm_hash(document.get("public")),
        "contextDigest": stable_algorithm_hash(document.get("contexts")),
        "sourceValidated": True,
        "projectionValidated": True,
        "replayValidated": True,
    }
    if sidecar.get("schema") == DIRECT_SOURCE_SCHEMA:
        expected.update(
            sourceIdentityDigest=direct_source_identity_v8(sidecar), identityValidated=True
        )
    if "orchestrationTranscript" in sidecar:
        expected.update(
            orchestrationTranscriptDigest=stable_algorithm_hash(sidecar["orchestrationTranscript"]),
            orchestrationValidated=True,
        )
    elif isinstance(receipt, dict) and any(
        key in receipt for key in ("orchestrationTranscriptDigest", "orchestrationValidated")
    ):
        raise ValueError("Source receipt cannot claim orchestration without its transcript")
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Pinned producer source receipt does not bind this artifact")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--consumer-revision",
        help="Optional exact revision for reproducing an observed run; defaults to the current clean checkout",
    )
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--producer-root", type=Path)
    parser.add_argument("--producer-revision")
    parser.add_argument("--source-recheck-timeout-seconds", type=int, default=30)
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
    has_orchestration = isinstance(sidecar, dict) and "orchestrationTranscript" in sidecar
    source_artifact = sidecar.get("sourceArtifact") if isinstance(sidecar, dict) else None
    public = document["public"]
    contexts = document["contexts"]
    source_binding = "untested"
    source_verification = None
    source_identity = None
    if isinstance(sidecar, dict) and sidecar.get("schema") in (
        "semantic-region-evidence/1",
        "semantic-protocol-region-evidence/1",
        _SOURCE_SESSION_SCHEMA,
        DIRECT_SOURCE_SCHEMA,
    ):
        direct = sidecar["schema"] == DIRECT_SOURCE_SCHEMA
        if sidecar["schema"] == _SOURCE_SESSION_SCHEMA:
            validate_source_session_shape(document)
        if direct:
            source_identity = direct_source_identity_v8(sidecar)
        if (
            sidecar.get("representation") != ("mixed" if direct else "semantic-trace/8")
            or sidecar.get("referenceStatus") != "agreed"
            or sidecar.get("optimizerSteps") != 0
            or sidecar.get("learningClaim") is not None
            or not direct
            and sidecar.get("sourceArtifactDigest")
            != stable_algorithm_hash(sidecar.get("sourceArtifact"))
            or sidecar.get("publicDigest") != stable_algorithm_hash(public)
            or sidecar.get("contextDigest") != stable_algorithm_hash(contexts)
        ):
            raise ValueError("V8 source evidence no longer binds this public projection")
        if direct and (
            sidecar.get("admission") != "locally_verified"
            or sidecar.get("profile") != "compositional-regions/1"
            or sidecar.get("protocolExtensions") != ["promise-join/1"]
        ):
            raise ValueError("V8 direct host source evidence has an unsupported protocol inventory")
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
                "composed-protocol-source/1",
                "promise-joins-source/1",
                "promise-executors-source/1",
                "promise-adoption-source/1",
                "bound-module-scheduler-source/1",
            ):
                supported = {
                    "owned-binary-memory/1",
                    "builtin-iterator/1",
                    "class-private-source/1",
                    "synchronous-generator-source/1",
                    "promise-join/1",
                    "promise-executor/1",
                    "promise-adoption/1",
                    "number-conversion-source/1",
                    "primitive-conversion-source/1",
                    "ordinary-completions-source/1",
                    "custom-iterator-source/1",
                    "ordinary-data-proxy-source/1",
                    "ordinary-array-map-source/1",
                    "ordinary-for-in/1",
                    "async-scoped-call-source/1",
                    "bound-module-scheduler-source/1",
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
                # These specialized profiles use the compiler's sorted source
                # components. Additional shared adapters use the composed profile.
                if profile == "promise-executors-source/1" and (
                    "promise-executor/1" not in extensions
                    or extensions != sorted(extensions)
                    or not set(extensions)
                    <= {
                        "async-scoped-call-source/1",
                        "promise-adoption/1",
                        "promise-executor/1",
                        "promise-join/1",
                    }
                ):
                    raise ValueError("V8 Promise executor source evidence has an invalid inventory")
                if profile == "promise-adoption-source/1" and extensions not in (
                    ["promise-adoption/1"],
                    ["promise-adoption/1", "promise-join/1"],
                ):
                    raise ValueError("V8 Promise adoption source evidence has an invalid inventory")
                if profile == "bound-module-scheduler-source/1":
                    plan = source_sidecar.get("moduleSchedulerPlan")
                    published = [
                        row["fields"].get("regionProgram", {}).get("moduleSchedulerPlan")
                        for row in public["timeline"]
                        if row["kind"] == "program"
                    ]
                    module_request = (
                        source_sidecar.get("sourceProvenance", {})
                        .get("upstream", {})
                        .get("moduleEvidence", {})
                        .get("request", {})
                    )
                    ownership = source_projection.get("moduleSchedulerOwnership", {})
                    if (
                        "bound-module-scheduler-source/1" not in extensions
                        or not isinstance(plan, dict)
                        or published != [plan]
                        or module_request.get("experimentalSchedulerPlan") is not True
                        or ownership.get("schema") != "bound-module-source-ownership/1"
                        or ownership.get("planDigest") != plan.get("digest")
                        or type(ownership.get("ownedEventCount")) is not int
                        or ownership["ownedEventCount"] <= 0
                        or len(ownership.get("cuts", [])) != ownership["ownedEventCount"]
                    ):
                        raise ValueError(
                            "V8 module scheduler evidence lacks its exact source plan and ownership"
                        )
                elif "bound-module-scheduler-source/1" in extensions:
                    raise ValueError("V8 module scheduler requires its declared source profile")
            else:
                raise ValueError("V8 protocol source evidence has an unsupported protocol profile")
        source_verification = recheck_source_binding(
            document,
            args.input,
            args.producer_root,
            args.producer_revision,
            timeout_seconds=args.source_recheck_timeout_seconds,
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
    if has_orchestration:
        checks["orchestration"] = (
            "passed"
            if training is not None
            and training.get("orchestrationRecords") == 4
            and result["counts"].get("orchestrationRecords") == 4
            and training.get("orchestrationTranscriptDigest")
            == result.get("orchestrationTranscriptDigest")
            == stable_algorithm_hash(sidecar["orchestrationTranscript"])
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
        **({"sourceIdentityDigest": source_identity} if source_identity is not None else {}),
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
                    *(("orchestration",) if has_orchestration else ()),
                )
            )
        ),
    }
    if has_orchestration:
        receipt["orchestrationTranscriptDigest"] = stable_algorithm_hash(
            sidecar["orchestrationTranscript"]
        )
        receipt["counts"]["orchestrationRecords"] = (
            training.get("orchestrationRecords") if training else None
        )
    # Engineering fixtures lack private source-bound records and exact physical
    # token rows. Never turn structural success into model-serving admission.
    checked_revision(root, revision)
    if args.output.resolve() == args.input.resolve():
        parser.error("Input and receipt paths must differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(receipt, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
