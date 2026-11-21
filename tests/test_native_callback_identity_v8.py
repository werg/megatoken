"""Retained captures cannot authorize an unrelated public function entry."""

import copy
from unittest.mock import patch

import pytest

from megatoken.data import native_v8
from megatoken.data.native_v8 import NativeV8Error, _closure_entry_function, validate_v8_artifact
from megatoken.data.prefix_v8 import validate_v8_committed_prefix
from tests.test_prediction_prefix_v8 import examples


def callback_prefix(kind):
    document = examples("semantic-callback-regions-v8.json.gz.b64")[0]
    row = next(
        item
        for item in document["public"]["timeline"]
        if item["kind"] == kind
        and (kind == "handler" or item["fields"]["activation"] == "callback")
    )
    document["public"]["timeline"] = document["public"]["timeline"][: row["sequence"]]
    document["contexts"] = [
        context for context in document["contexts"] if context["visibleThrough"] < row["sequence"]
    ]
    return document, row


@pytest.mark.parametrize("kind", ["invocation", "handler"])
@pytest.mark.parametrize("entry", ["run", "function:13", "missing"])
def test_forged_entry_is_refused_before_graph_import_or_handler_publication(kind, entry):
    document, row = callback_prefix(kind)
    validate_v8_committed_prefix(document["public"], document["contexts"])
    row["fields"]["entry"] = entry
    before = copy.deepcopy(document)
    initialize = native_v8._initial_state
    observe = native_v8._apply_observation

    def guarded_initialize(program, invocation, *args, **kwargs):
        if invocation.get("activation") == "callback":
            raise AssertionError("Forged closure reached input graph import")
        return initialize(program, invocation, *args, **kwargs)

    def guarded_observe(state, observed, *args, **kwargs):
        if kind == "handler" and observed["kind"] == "handler":
            raise AssertionError("Forged closure reached handler publication")
        return observe(state, observed, *args, **kwargs)

    with (
        patch.object(native_v8, "_initial_state", side_effect=guarded_initialize),
        patch.object(native_v8, "_apply_observation", side_effect=guarded_observe),
        pytest.raises(NativeV8Error, match="immutable function site"),
    ):
        validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


def test_changing_activation_does_not_bypass_closure_binding():
    document, row = callback_prefix("invocation")
    row["fields"].update(activation="call", entry="run")
    with pytest.raises(NativeV8Error, match="immutable function site"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


def test_matching_closure_backed_call_does_not_discard_its_retained_cells():
    document, row = callback_prefix("invocation")
    row["fields"]["activation"] = "call"
    before = copy.deepcopy(document)
    with pytest.raises(NativeV8Error, match="retained-call replay profile"):
        validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


@pytest.mark.parametrize("entry", ["function:77", "run"])
def test_host_owned_handler_resolves_realm_identity_before_explicit_profile_refusal(entry):
    document, row = callback_prefix("handler")
    row["fields"].update(ownerInvocationId=None, entry=entry)
    message = (
        "native lifecycle replay profile" if entry == "function:77" else "immutable function site"
    )
    with pytest.raises(NativeV8Error, match=message):
        validate_v8_committed_prefix(document["public"], document["contexts"])


@pytest.mark.parametrize("field,value", [("realmId", "other-realm"), ("programId", "unknown")])
def test_handler_requires_the_retained_realm_and_published_program(field, value):
    document, row = callback_prefix("handler")
    row["fields"][field] = value
    with pytest.raises(NativeV8Error, match="retained realm source"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


@pytest.mark.parametrize("entry,code", [("guest", "function:77"), ("function:77", "guest")])
def test_unique_declared_name_and_source_site_are_equivalent(entry, code):
    functions = [{"name": "run", "start": 13}, {"name": "guest", "start": 77}]
    closure = {"kind": "closure", "code": code}
    before = copy.deepcopy((functions, closure))
    assert _closure_entry_function(functions, entry, closure) == functions[1]
    assert (functions, closure) == before


@pytest.mark.parametrize("entry,code", [("guest", "function:77"), ("function:77", "guest")])
def test_one_unique_spelling_does_not_resolve_an_ambiguous_other_spelling(entry, code):
    functions = [{"name": "guest", "start": 77}, {"name": "guest", "start": 99}]
    with pytest.raises(NativeV8Error, match="one immutable function site"):
        _closure_entry_function(functions, entry, {"kind": "closure", "code": code})


@pytest.mark.parametrize(
    "functions,closure",
    [
        ([{"name": "guest", "start": 77}], {"kind": "closure", "code": "missing"}),
        ([{"name": "guest", "start": "77"}], {"kind": "closure", "code": "guest"}),
        ([{"name": "guest", "start": 77}], {"kind": "object"}),
        ([{"name": "guest", "start": 77}], None),
    ],
)
def test_closure_identity_must_be_a_declared_source_function(functions, closure):
    with pytest.raises(NativeV8Error):
        _closure_entry_function(functions, "guest", closure)


def test_unmodified_callback_fixture_still_replays_both_retained_invocations():
    document = examples("semantic-callback-regions-v8.json.gz.b64")[0]
    assert (
        sum(
            row["kind"] == "invocation" and row["fields"]["activation"] == "callback"
            for row in document["public"]["timeline"]
        )
        == 2
    )
    receipt = validate_v8_artifact(document)
    assert receipt["checks"]["continuation"] == "passed"
    assert not receipt["admitted"]
