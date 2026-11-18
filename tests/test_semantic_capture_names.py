"""Captured cells use source binding names, while heap identities stay narrow."""

import json

import pytest

from megatoken.data.witness_v5 import WitnessV5Error, parse_record


def allocation(cell):
    return "allocate " + json.dumps(
        {
            "name": "o1",
            "kind": "closure",
            "properties": [],
            "code": "function:1",
            "captures": [{"binding": "keyFn", "cell": cell}],
        }
    )


@pytest.mark.parametrize("cell", ["keyFn", "_value", "$value", "value_1", "éclair"])
def test_capture_accepts_the_same_source_binding_name_as_bind(cell):
    parse_record("bind " + json.dumps({"name": cell, "value": ["num", 1]}))
    kind, fields = parse_record(allocation(cell))
    assert kind == "allocate"
    assert fields["captures"] == [{"binding": "keyFn", "cell": cell}]


@pytest.mark.parametrize("cell", ["", "1value", "value-name", "value name", None, 7])
def test_capture_rejects_malformed_source_binding_names(cell):
    with pytest.raises(WitnessV5Error, match="Malformed shared cell capture relation"):
        parse_record(allocation(cell))


def test_capture_does_not_relax_heap_object_identities():
    with pytest.raises(WitnessV5Error, match="Invalid witness identity"):
        parse_record(
            "read "
            + json.dumps({"object": "keyFn", "key": "value", "present": True, "value": ["num", 1]})
        )
