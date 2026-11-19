"""Captured cells use source binding names, while heap identities stay narrow."""

import json

import pytest

from megatoken.data.witness_v6 import WitnessV6Error, encode_record, parse_record


def allocation(cell):
    name = cell if isinstance(cell, str) else json.dumps(str(cell))
    return 'new @o1 closure {} code "function:1" captures {keyFn: ' + name + "}"


@pytest.mark.parametrize("cell", ["keyFn", "_value", "$value", "value_1", "éclair"])
def test_capture_accepts_the_same_source_binding_name_as_bind(cell):
    parse_record(encode_record("bind", {"name": cell, "value": ["num", 1]}))
    kind, fields = parse_record(allocation(cell))
    assert kind == "allocate"
    assert fields["captures"] == [{"binding": "keyFn", "cell": cell}]


@pytest.mark.parametrize("cell", ["", "1value", "value-name", "value name", None, 7])
def test_capture_rejects_malformed_source_binding_names(cell):
    with pytest.raises(WitnessV6Error, match="Invalid semantic-trace/6"):
        parse_record(allocation(cell))


def test_capture_does_not_relax_heap_object_identities():
    with pytest.raises(WitnessV6Error, match="Invalid semantic-trace/6"):
        parse_record('read @keyFn["value"] => 1')
