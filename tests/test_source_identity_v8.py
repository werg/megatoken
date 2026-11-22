"""Identity checks bind metadata; these controls never claim live source proof."""

import copy

import pytest

from megatoken.data.semantic_contract_v8 import CONTRACT_DIGEST, FIXTURE_DIGEST
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.source_identity_v8 import (
    DIRECT_SOURCE_SCHEMA,
    direct_source_identity_v8,
    source_evidence_binding_v8,
    validate_source_binding_v8,
)


def direct_source_sidecar():
    identity = {
        "schema": "semantic-source-candidate-identity/8",
        "execution": {
            "execution": "semantic-execution/8",
            "trace": "semantic-trace/8",
            "context": "semantic-context/8",
            "harness": "compositional-regions/1",
            "representation": "mixed",
        },
        "certification": {
            "schemaDigest": CONTRACT_DIGEST,
            "fixtureDigest": FIXTURE_DIGEST,
            "evidenceKind": "deterministic-engineering",
            "optimizerSteps": 0,
            "learnedSuccess": None,
        },
    }
    return {
        "schema": DIRECT_SOURCE_SCHEMA,
        "identity": identity,
        "identityDigest": stable_algorithm_hash(identity),
    }


def test_direct_identity_binds_a_single_source_representation():
    side = direct_source_sidecar()
    digest = direct_source_identity_v8(side)
    binding = source_evidence_binding_v8(side)
    assert binding == {"sourceArtifactDigest": None, "sourceIdentityDigest": digest}
    assert validate_source_binding_v8(binding)
    for changed in (
        {"sourceArtifactDigest": None},
        {**binding, "sourceIdentityDigest": None},
        {**binding, "sourceArtifactDigest": "f" * 64},
        {"sourceIdentityDigest": digest},
    ):
        with pytest.raises(ValueError, match="exact artifact or direct identity"):
            validate_source_binding_v8(changed)


@pytest.mark.parametrize("mutation", ["digest", "schema", "trace", "contract", "learned", "nested"])
def test_direct_identity_rejects_rehashed_contract_and_authority_changes(mutation):
    side = copy.deepcopy(direct_source_sidecar())
    if mutation == "digest":
        side["identityDigest"] = "0" * 64
    elif mutation == "schema":
        side["identity"]["schema"] = "semantic-source-candidate-identity/6"
    elif mutation == "trace":
        side["identity"]["execution"]["trace"] = "semantic-trace/6"
    elif mutation == "contract":
        side["identity"]["certification"]["schemaDigest"] = "0" * 64
    elif mutation == "learned":
        side["identity"]["certification"]["learnedSuccess"] = True
    else:
        side["sourceArtifact"] = {}
    if mutation != "digest":
        side["identityDigest"] = stable_algorithm_hash(side["identity"])
    with pytest.raises(ValueError, match="direct source identity"):
        direct_source_identity_v8(side)
