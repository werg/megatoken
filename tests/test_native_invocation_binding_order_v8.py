"""Invocation binding maps replay independently of JSON object key order."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from megatoken.data.native_v8 import (
    NativeV8Error,
    _initial_state,
    _invocation_bindings,
    _native_checkpoint,
    _restore_native_checkpoint,
    validate_v8_artifact,
)
from megatoken.data.prefix_v8 import validate_v8_prediction_prefix
from megatoken.data.semantic_json import stable_algorithm_json

FIXTURE = Path(__file__).parent / "fixtures/semantic-binding-order-v8.json"


def cases():
    raw = FIXTURE.read_bytes()
    provenance = json.loads(
        FIXTURE.with_name("semantic-binding-order-v8.provenance.json").read_text()
    )
    assert provenance["producerRevision"] == "29e802f379b03375ed893dc133baf25dd15a4ba9"
    assert provenance["producerTree"] == "208503c1f3cfe9a13050c5c8b1884ab24c80a5ce"
    assert hashlib.sha256(raw).hexdigest() == provenance["fixtureSha256"]
    document = json.loads(raw)
    assert document["schema"] == "semantic-binding-order-interoperability/1"
    assert document["qualification"]["admission"] == "not-claimed"
    assert document["qualification"]["native"] == "untested"
    return document["cases"]


def invocation_rows(artifact):
    timeline = artifact["public"]["timeline"]
    program = next(row["fields"] for row in timeline if row["kind"] == "program")
    invocation = next(row["fields"] for row in timeline if row["kind"] == "invocation")
    return program, invocation


def aliases_and_locals(artifact):
    program, invocation = invocation_rows(artifact)
    state = _initial_state(program, invocation, program["functions"])
    return state["captureAliases"], state["locals"]


def test_source_rechecked_binding_cases_replay_and_canonical_reordering():
    rows = {case["id"]: case for case in cases()}
    assert set(rows) == {
        "declared-parameter-order",
        "additional-binding-order",
        "additional-utf16-order",
    }

    expected_input_orders = {
        "declared-parameter-order": ["a", "z"],
        "additional-binding-order": ["z", "a"],
        "additional-utf16-order": ["\uff21", "\U00010000"],
    }
    expected_canonical_orders = {
        "declared-parameter-order": ["a", "z"],
        "additional-binding-order": ["a", "z"],
        "additional-utf16-order": ["\U00010000", "\uff21"],
    }

    for case_id, case in rows.items():
        artifact = case["artifact"]
        original_program, original_invocation = invocation_rows(artifact)
        assert list(original_invocation["bindings"]) == expected_input_orders[case_id]

        original_receipt = validate_v8_artifact(artifact)
        assert all(
            original_receipt["checks"][name] == "passed"
            for name in ("replay", "contexts", "continuation")
        )
        assert original_receipt["admitted"] is False

        aliases, locals_ = aliases_and_locals(artifact)
        assert aliases == case["initialState"]["aliases"]
        assert locals_ == case["initialState"]["locals"]

        # This uses the consumer's ECMAScript-compatible canonical JSON order,
        # including the UTF-16 ordering difference for U+10000 and U+FF21.
        canonical = json.loads(stable_algorithm_json(artifact))
        _, canonical_invocation = invocation_rows(canonical)
        assert list(canonical_invocation["bindings"]) == expected_canonical_orders[case_id]
        canonical_receipt = validate_v8_artifact(canonical)
        assert canonical_receipt["checks"] == original_receipt["checks"]
        assert canonical_receipt["admitted"] is False
        assert aliases_and_locals(canonical) == (aliases, locals_)


def test_source_rechecked_binding_cases_preserve_every_prefix_and_checkpoint():
    for case in cases():
        artifact = case["artifact"]
        for index, context in enumerate(artifact["contexts"]):
            public = {
                **artifact["public"],
                "timeline": artifact["public"]["timeline"][: context["visibleThrough"]],
            }
            checked = validate_v8_prediction_prefix(public, artifact["contexts"][:index], context)
            assert checked["context"] == context
            checkpoint = _native_checkpoint(public["timeline"], context)
            assert (
                _restore_native_checkpoint(
                    checkpoint, expected_prefix=public["timeline"], expected_state=context["state"]
                )
                == context["state"]
            )


def test_invalid_binding_names_and_indices_fail_before_binding_values_are_copied():
    roots = [["num", 4]]
    bad_bindings = (
        {"": 0},
        {1: 0},
        {"valid": 0, "bad": True},
        {"valid": 0, "bad": -1},
        {"valid": 0, "bad": 1},
        {"valid": 0, "bad": "0"},
        {"valid": 0, "bad": None},
    )
    for bindings in bad_bindings:
        invocation = {"inputGraph": {"roots": roots}, "bindings": bindings}
        with pytest.raises(NativeV8Error, match="observed argument"):
            with patch(
                "megatoken.data.native_v8.copy.deepcopy",
                side_effect=AssertionError("binding value copied before full validation"),
            ):
                _invocation_bindings({"params": []}, invocation)
