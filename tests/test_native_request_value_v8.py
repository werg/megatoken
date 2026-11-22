"""An external JSON payload must retain the selected argument's value type."""

import copy

import pytest

from megatoken.data.detailed_v8 import encode_record, parse_record
from megatoken.data.host_operations import canonical_json, materialize_request
from megatoken.data.native_v8 import NativeV8Error, _apply_detail


def request(argument, payload):
    fields = {
        "effect": "effect",
        "operation": "demo.store",
        "arguments": [["str", "key"], ["bool" if type(argument) is bool else "num", argument]],
        "invocation": "root",
        "site": {"start": 0, "end": 5},
        "payload": canonical_json(["key", payload]),
    }
    return parse_record(encode_record("request", fields))


def before_request():
    return {"stack": ["root"], "frameFunctions": [["root", "run"]], "effects": [], "heap": []}


@pytest.mark.parametrize("argument,payload", [(True, 1), (False, 0), (1, True), (0, False)])
def test_request_payload_cannot_exchange_booleans_and_numbers(argument, payload):
    kind, fields = request(argument, payload)
    with pytest.raises(ValueError, match="payload differs from evaluated arguments"):
        materialize_request(fields)
    state = before_request()
    before = copy.deepcopy(state)
    with pytest.raises(NativeV8Error, match="payload differs from its tagged arguments"):
        _apply_detail(state, kind, fields, {"sequence": 1})
    assert state == before


@pytest.mark.parametrize("argument,payload", [(True, True), (False, False), (1, 1.0), (1.0, 1)])
def test_request_preserves_boolean_values_and_equivalent_ecmascript_numbers(argument, payload):
    kind, fields = request(argument, payload)
    expected, _ = materialize_request(fields)
    state = before_request()
    _apply_detail(state, kind, fields, {"sequence": 1})
    assert state["effects"][0][1]["arguments"] == expected
    assert state["effects"][0][1]["payload"] == fields["payload"]
