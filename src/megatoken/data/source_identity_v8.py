"""Bind direct V8 source evidence without manufacturing a nested artifact."""

from __future__ import annotations

import re

from .semantic_contract_v8 import CONTRACT_DIGEST, FIXTURE_DIGEST
from .semantic_json import stable_algorithm_hash

DIRECT_SOURCE_SCHEMA = "semantic-protocol-host-region-evidence/1"


def direct_source_identity_v8(sidecar):
    """Validate identity shape and pins; only fresh source replay proves truth."""
    identity = sidecar.get("identity")
    if (
        sidecar.get("schema") != DIRECT_SOURCE_SCHEMA
        or "sourceArtifact" in sidecar
        or sidecar.get("sourceArtifactDigest") is not None
        or not isinstance(identity, dict)
        or identity.get("schema") != "semantic-source-candidate-identity/8"
        or sidecar.get("identityDigest") != stable_algorithm_hash(identity)
    ):
        raise ValueError("V8 direct source identity is missing or its digest changed")
    execution, certification = identity.get("execution", {}), identity.get("certification", {})
    expected = {
        "execution": "semantic-execution/8",
        "trace": "semantic-trace/8",
        "context": "semantic-context/8",
        "harness": "compositional-regions/1",
        "representation": "mixed",
    }
    if (
        not isinstance(execution, dict)
        or any(execution.get(key) != value for key, value in expected.items())
        or not isinstance(certification, dict)
        or certification.get("schemaDigest") != CONTRACT_DIGEST
        or certification.get("fixtureDigest") != FIXTURE_DIGEST
        or certification.get("evidenceKind") != "deterministic-engineering"
        or certification.get("optimizerSteps") != 0
        or certification.get("learnedSuccess") is not None
    ):
        raise ValueError("V8 direct source identity has incompatible execution or evidence pins")
    return sidecar["identityDigest"]


def source_evidence_binding_v8(sidecar):
    if sidecar.get("schema") == DIRECT_SOURCE_SCHEMA:
        return {
            "sourceArtifactDigest": None,
            "sourceIdentityDigest": direct_source_identity_v8(sidecar),
        }
    if not isinstance(sidecar.get("sourceArtifact"), dict):
        raise ValueError("V8 source artifact is missing")
    return {"sourceArtifactDigest": stable_algorithm_hash(sidecar["sourceArtifact"])}


def validate_source_binding_v8(binding):
    """A prepared binding names exactly one source authority representation."""
    direct = "sourceIdentityDigest" in binding
    value = binding.get("sourceIdentityDigest" if direct else "sourceArtifactDigest")
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        or direct
        and ("sourceArtifactDigest" not in binding or binding["sourceArtifactDigest"] is not None)
    ):
        raise ValueError("V8 source binding requires an exact artifact or direct identity digest")
    return direct
