"""Retained invocation state is derived from prior records, never ID spelling."""

import copy
from unittest.mock import patch

import pytest

from megatoken.data import native_v8
from megatoken.data.native_v8 import NativeV8Error, _replay_shared_state, validate_v8_artifact
from megatoken.data.prefix_v8 import validate_v8_committed_prefix, validate_v8_prediction_prefix
from tests.test_prediction_prefix_v8 import examples, prefix


def artifacts():
    return examples("semantic-retained-invocations-v8.json.gz.b64")


@pytest.mark.parametrize("case", range(3))
def test_retained_calls_and_nonnumeric_callbacks_replay_all_prefixes(case):
    document = artifacts()[case]
    receipt = validate_v8_artifact(document)
    assert receipt["admitted"] is False
    assert receipt["checks"]["sourceBinding"] == "untested"
    assert all(receipt["checks"][key] == "passed" for key in ("replay", "contexts", "continuation"))
    for index in range(len(document["contexts"])):
        public, contexts, context = prefix(document, index)
        assert validate_v8_prediction_prefix(public, contexts, context)["context"] == context


def test_closure_calls_update_shared_capture_and_ordinary_calls_keep_retained_heap():
    document = artifacts()[0]
    final = _replay_shared_state(
        document["public"]["timeline"],
        document["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="ordinary",
    )
    state = final["state"]
    assert dict(state["locals"])["count"] == ["num", 10]
    assert dict(state["heap"])["fn"]["kind"] == "closure"
    assert state["captureAliases"] == [["initial", "a3b0"]]
    assert state["completion"] == {"kind": "return", "value": ["num", 9]}
    for identity, cell in (("next", "a1b0"), ("again123", "a2b0")):
        boundary = next(c for c in document["contexts"] if c["invocationId"] == identity)
        assert dict(boundary["state"]["captureAliases"])["value"] == cell
        assert boundary["input"]["retained"] == ["fn"]


def test_fresh_sibling_realms_do_not_share_cells_but_use_global_invocation_order():
    document = artifacts()[2]
    second = next(c for c in document["contexts"] if c["invocationId"] == "second")
    assert second["state"]["locals"] == [["a1b0", ["num", 8]]]
    assert second["state"]["captureAliases"] == [["initial", "a1b0"]]
    assert second["input"] == {"graph": {"roots": [["num", 8]], "objects": []}}


def retained_prefix():
    document = artifacts()[0]
    row = next(
        row
        for row in document["public"]["timeline"]
        if row["kind"] == "invocation" and row["fields"]["invocationId"] == "next"
    )
    document["public"]["timeline"] = document["public"]["timeline"][: row["sequence"]]
    document["contexts"] = [
        context for context in document["contexts"] if context["visibleThrough"] < row["sequence"]
    ]
    return document, row


@pytest.mark.parametrize("mutation", ["fresh-reuse", "wrong-code", "missing-owner"])
def test_retained_admission_refuses_before_importing_inputs(mutation):
    document, row = retained_prefix()
    if mutation == "fresh-reuse":
        row["fields"]["realmPolicy"] = "fresh"
    elif mutation == "wrong-code":
        row["fields"]["entry"] = "run"
    else:
        row["fields"]["parentInvocationId"] = "missing"
    before = copy.deepcopy(document)
    initialize = native_v8._initial_state

    def guarded_initialize(program, invocation, *args, **kwargs):
        if invocation["invocationId"] == "next":
            raise AssertionError("Invalid retained call reached graph import")
        return initialize(program, invocation, *args, **kwargs)

    with patch.object(native_v8, "_initial_state", side_effect=guarded_initialize):
        with pytest.raises(NativeV8Error):
            validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


@pytest.mark.parametrize("mutation", ["overwritten-closure", "unknown-receiver", "unknown-input"])
def test_retained_input_graph_cannot_replace_or_invent_existing_identities(mutation):
    document, row = retained_prefix()
    if mutation == "overwritten-closure":
        row["fields"]["inputGraph"]["objects"] = [{"id": "fn", "kind": "object", "properties": []}]
    elif mutation == "unknown-receiver":
        row["fields"]["receiver"] = ["ref", "unknown"]
    else:
        row["fields"]["inputGraph"]["roots"] = [["ref", "unknown"]]
    before = copy.deepcopy(document)
    with pytest.raises(NativeV8Error):
        validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before
