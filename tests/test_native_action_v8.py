"""Native dispatch receipts bind the committed request without proving model origin."""

import copy
import hashlib

import pytest

from megatoken.data.native_v8 import NativeV8Error, validate_v8_artifact
from megatoken.data.prefix_v8 import validate_v8_committed_prefix
from tests.test_native_recovery_v8 import (
    apply,
    artifact,
    holders,
    refused_without_mutation,
    through,
)


def test_native_action_receipt_is_bound_preserved_and_keeps_authentication_unclaimed():
    document = artifact(11)
    receipt = validate_v8_artifact(document)
    assert not receipt["admitted"]
    assert receipt["checks"]["sourceBinding"] == "untested"
    assert document["recovery"]["nativeActionAuthentication"] == "not-claimed"
    assert document["recovery"]["adapterCalls"] == 1
    dispatch = next(row for row in document["public"]["timeline"] if row["kind"] == "dispatch")
    native = dispatch["fields"]["nativeAction"]
    context = next(
        context
        for context in document["contexts"]
        if context["visibleThrough"] >= dispatch["sequence"]
    )
    intent = dict(context["obligations"]["intents"])["root/effect"]
    assert intent["nativeAction"] == native == document["recovery"]["nativeAction"]
    assert native["recordDigest"] == hashlib.sha256(intent["witness"].encode()).hexdigest()
    assert native["operation"] == intent["operation"]
    assert native["payload"] == intent["payload"]
    assert native["modelTrack"] == "controlled-model-track"
    assert native["authority"] == {
        "issuer": "selected-test",
        "scopes": ["dispatch"],
        "evidence": None,
    }
    assert native["signature"] == {"algorithm": "fixture-only", "value": "not-authenticated"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "semantic-native-action/3"),
        ("actionKey", ""),
        ("actionKey", False),
        ("invocationId", "foreign"),
        ("effectId", "foreign"),
        ("operation", "demo.readAsync"),
        ("payload", '[ "key" ]'),
        ("ownerTrack", "foreign"),
        ("recordDigest", "0" * 64),
        ("recordDigest", None),
    ],
)
def test_native_action_cannot_substitute_committed_request_binding(field, value):
    state, row = holders(11, before_kind="dispatch")
    row["fields"]["nativeAction"][field] = value
    refused_without_mutation(state, row, "request binding")


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "actionKey",
        "invocationId",
        "effectId",
        "operation",
        "payload",
        "ownerTrack",
        "recordDigest",
    ],
)
def test_native_action_requires_every_binding_field(field):
    state, row = holders(11, before_kind="dispatch")
    del row["fields"]["nativeAction"][field]
    refused_without_mutation(state, row, "plain request binding")


@pytest.mark.parametrize("native", [None, False, [], "receipt"])
def test_an_explicit_invalid_native_action_cannot_be_treated_as_absent(native):
    state, row = holders(11, before_kind="dispatch")
    row["fields"]["nativeAction"] = native
    refused_without_mutation(state, row, "plain request binding")
    document = artifact(11)
    dispatch = next(row for row in document["public"]["timeline"] if row["kind"] == "dispatch")
    through(document, dispatch["sequence"])
    dispatch["fields"]["nativeAction"] = native
    before = copy.deepcopy(document)
    with pytest.raises(ValueError, match="plain request binding"):
        validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -0.0, {1: "non-string-key"}, {1, 2}, (1, 2)]
)
def test_native_action_extensions_require_finite_plain_json_data(value):
    state, row = holders(11, before_kind="dispatch")
    row["fields"]["nativeAction"]["extension"] = value
    refused_without_mutation(state, row, "plain request binding")


@pytest.mark.parametrize("shape", ["cycle", "too-deep"])
def test_native_action_extensions_cannot_hide_cycles_or_unbounded_nesting(shape):
    state, row = holders(11, before_kind="dispatch")
    value = {}
    if shape == "cycle":
        value["self"] = value
    else:
        for _ in range(65):
            value = {"nested": value}
    row["fields"]["nativeAction"]["extension"] = value
    refused_without_mutation(state, row, "plain request binding")


def test_native_action_requires_an_independently_reconstructed_request_obligation():
    state, row = holders(11, before_kind="dispatch")
    state = (*state[:3], None)
    refused_without_mutation(state, row, "committed request obligation")


def test_context_cannot_replace_the_observed_native_action_metadata():
    document = artifact(11)
    context = next(
        context
        for context in document["contexts"]
        if context["obligations"]["intents"]
        and "nativeAction" in context["obligations"]["intents"][0][1]
    )
    context["obligations"]["intents"][0][1]["nativeAction"]["signature"]["value"] = "changed"
    with pytest.raises(NativeV8Error, match="Obligations differ"):
        validate_v8_artifact(document)


def test_unbound_extra_receipt_data_is_retained_without_becoming_authentication():
    state, row = holders(11, before_kind="dispatch")
    native = row["fields"]["nativeAction"]
    native["signature"] = {"algorithm": "external-format", "value": "opaque-to-native-replay"}
    native["authority"] = {"batch": [1, True, None, {"extension": "value"}]}
    apply(state, row)
    assert state[3]["intents"][0][1]["nativeAction"] == native
    assert state[3]["intents"][0][1]["status"] == "dispatched"
