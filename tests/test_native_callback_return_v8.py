"""Declared callback graphs retain exact heap authority and source continuations."""

import copy
import json

import pytest

from megatoken.data.detailed_v8 import encode_record, parse_record
from megatoken.data.host_operations import canonical_json
from megatoken.data.native_v8 import NativeV8Error, _apply_detail, validate_v8_artifact
from megatoken.data.observations_v8 import validate_observation_v8
from megatoken.data.prefix_v8 import validate_v8_prediction_prefix
from tests.test_prediction_prefix_v8 import examples, prefix


@pytest.mark.parametrize("case", range(4))
def test_original_source_callback_return_contexts_and_boundaries(case):
    document = examples("semantic-callback-return-v8.json.gz.b64")[case]
    receipt = validate_v8_artifact(document)
    assert not receipt["admitted"]
    assert all(
        receipt["checks"][name] == "passed" for name in ("replay", "contexts", "continuation")
    )
    # Full artifact replay already checks every record. Independently recover
    # the final child and parent prediction boundaries without a future target.
    children = {
        row["fields"]["invocationId"]
        for row in document["public"]["timeline"]
        if row["kind"] == "invocation" and row["fields"]["activation"] == "callback"
    }
    assert children
    last_child = max(
        index
        for index, context in enumerate(document["contexts"])
        if context["invocationId"] in children
    )
    for index in (last_child, len(document["contexts"]) - 1):
        public, contexts, context = prefix(document, index)
        assert validate_v8_prediction_prefix(public, contexts, context)["context"] == context


def graph_request():
    document = examples("semantic-callback-return-v8.json.gz.b64")[1]
    row = next(
        row
        for row in document["public"]["timeline"]
        if row["kind"] == "semantic" and parse_record(row["fields"]["line"])[0] == "request"
    )
    context = next(
        context
        for context in document["contexts"]
        if context["visibleThrough"] == row["sequence"] - 1
    )
    fields = parse_record(row["fields"]["line"])[1]
    return copy.deepcopy((context["state"], fields, row, context["obligations"]))


@pytest.mark.parametrize(
    "mutation",
    [
        "stale-property",
        "missing-object",
        "unknown-callback",
        "extra-object",
        "noncanonical",
        "missing-promise",
        "undeclared-callback",
        "json-codec",
        "accessor",
        "custom-prototype",
    ],
)
def test_invalid_graph_request_cannot_publish_effect_or_intent(mutation):
    state, fields, row, obligations = graph_request()
    supplied = json.loads(fields["payload"])
    if mutation == "stale-property":
        supplied["objects"][0]["properties"][0]["value"] = ["num", 12]
    elif mutation == "missing-object":
        supplied["objects"] = []
    elif mutation == "unknown-callback":
        supplied["objects"][0]["properties"][0]["value"] = ["callback", "unknown"]
    elif mutation == "extra-object":
        supplied["objects"].append({"id": "o999", "kind": "object", "properties": []})
    elif mutation == "missing-promise":
        fields.pop("promise")
    elif mutation == "undeclared-callback":
        fields.update(
            operation="natlang.writeCell",
            arguments=[fields["arguments"][0], fields["arguments"][2]],
        )
        fields.pop("promise")
        supplied["roots"] = [supplied["roots"][0], supplied["roots"][2]]
    elif mutation == "json-codec":
        fields["operation"] = "demo.store"
    elif mutation == "accessor":
        dict(state["heap"])["o2"]["descriptors"] = [["visit", {"kind": "accessor"}]]
    elif mutation == "custom-prototype":
        dict(state["heap"])["o2"]["prototype"] = ["ref", "o3"]
    fields["payload"] = (
        json.dumps(supplied) if mutation == "noncanonical" else canonical_json(supplied)
    )
    if mutation == "noncanonical":
        # The wire encoder refuses this before replay. Independently exercise
        # the decoder's graph materialization guard with the same bad fields.
        with pytest.raises(ValueError, match="canonical"):
            encode_record("request", fields, role="generated")
    else:
        row["fields"]["line"] = encode_record("request", fields, role="generated")
    before = copy.deepcopy((state, obligations))
    with pytest.raises(NativeV8Error):
        _apply_detail(state, "request", fields, row, obligations)
    assert (state, obligations) == before


def test_guest_callback_throw_provenance_is_only_valid_on_rejection():
    document = examples("semantic-callback-return-v8.json.gz.b64")[3]
    row = next(row for row in document["public"]["timeline"] if row["kind"] == "result")
    assert row["fields"]["status"] == "rejected"
    row["fields"]["errorOrigin"] = "guest-callback-throw"
    assert validate_observation_v8(row)
    row["fields"]["status"] = "fulfilled"
    with pytest.raises(ValueError, match="Fulfilled result"):
        validate_observation_v8(row)
