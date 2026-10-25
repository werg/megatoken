"""Independent native materialization checks use the shared declaration, not JS code."""

import copy

import pytest

from megatoken.data.host_operations import HOST_OPERATIONS, canonical_json, materialize_request


def request(operation, arguments, payload, **fields):
    return {
        "operation": operation,
        "arguments": arguments,
        "payload": canonical_json(payload),
        **fields,
    }


def test_catalog_contains_all_required_families_and_typed_lifetimes():
    for operation in (
        "demo.readAsync",
        "dom.release",
        "http.request",
        "http.responseReadChunk",
        "fs.openWrite",
        "fs.writeChunk",
        "fs.closeWrite",
        "net.connect",
        "natlang.call",
        "natlang.readCell",
        "natlang.writeCell",
    ):
        assert operation in HOST_OPERATIONS
        descriptor = HOST_OPERATIONS[operation]
        assert type(descriptor["arity"]) is int
        assert descriptor["codec"] in {"canonical-json/1", "owned-graph/1"}
        assert descriptor["cancellation"]
        assert descriptor["authority"]
    assert HOST_OPERATIONS["natlang.call"]["callbackGraph"] is True


def graph_fields(nodes):
    roots = [["str", "module"], ["str", "run"], ["ref", "o1"]]
    payload = {"schema": "owned-graph/1", "roots": roots, "objects": nodes}
    return request("natlang.call", roots, payload, promise="p1")


def test_owned_graph_preserves_cycles_sparse_arrays_special_scalars_and_snapshot():
    nodes = [
        {
            "id": "o1",
            "kind": "object",
            "properties": [
                {"key": "self", "value": ["ref", "o1"]},
                {"key": "array", "value": ["ref", "a1"]},
                {"key": "alias", "value": ["ref", "a1"]},
                {"key": "negativeZero", "value": ["num", "-0"]},
            ],
        },
        {
            "id": "a1",
            "kind": "array",
            "properties": [
                {"key": "2", "value": ["undefined"]},
                {"key": "length", "value": ["num", 4]},
            ],
        },
    ]
    fields = graph_fields(nodes)
    heap = {node["id"]: copy.deepcopy(node) for node in nodes}
    # Array length was allocated before its numeric element. The wire codec
    # must apply JavaScript's own-key order without changing the live values.
    heap["a1"]["properties"].reverse()
    actual, signature = materialize_request(fields, heap=heap, require_promise=True)
    assert actual["objects"] == nodes
    assert signature == HOST_OPERATIONS["natlang.call"]["signature"]
    stale = copy.deepcopy(heap)
    stale["o1"]["properties"][3]["value"] = ["num", 0]
    with pytest.raises(ValueError, match="snapshot"):
        materialize_request(fields, heap=stale, require_promise=True)


def test_owned_callback_inside_declared_graph_requires_committed_closure():
    node = {
        "id": "o1",
        "kind": "object",
        "properties": [{"key": "callback", "value": ["ref", "c1"]}],
    }
    fields = graph_fields(
        [{**node, "properties": [{"key": "callback", "value": ["callback", "c1"]}]}]
    )
    heap = {"o1": node, "c1": {"id": "c1", "kind": "closure", "properties": []}}
    result, _ = materialize_request(fields, heap=heap, require_promise=True)
    assert result["objects"][0]["properties"][0]["value"] == ["callback", "c1"]
    with pytest.raises(ValueError, match="closure"):
        materialize_request(fields, heap={"o1": node}, require_promise=True)


@pytest.mark.parametrize(
    "number,text",
    [
        (1, "1"),
        (-0.0, "0"),
        (1e-7, "1e-7"),
        (1e-6, "0.000001"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
    ],
)
def test_ecmascript_canonical_number_bytes(number, text):
    assert canonical_json(number) == text


def test_promise_contract_callback_slots_and_malformed_payload_refuse():
    fields = request("demo.readAsync", [["str", "x"]], ["x"])
    with pytest.raises(ValueError, match="Promise"):
        materialize_request(fields, require_promise=True)
    fields["promise"] = "p1"
    assert materialize_request(fields, require_promise=True)[0] == ["x"]
    fields["payload"] = '[ "x" ]'
    with pytest.raises(ValueError, match="canonical"):
        materialize_request(fields, require_promise=True)
