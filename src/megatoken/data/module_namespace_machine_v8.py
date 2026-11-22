"""Experimental shared-heap namespace operations; deliberately not admitted."""

from __future__ import annotations

import copy
import re
from typing import Any

from .module_namespace_state_v8 import (
    _canonical_value,
    _exact,
    _fail,
    verify_experimental_module_namespace_plan,
)

_ID = re.compile(r"^o[1-9][0-9]*$")
_LOCAL = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def _table(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, list):
        _fail(label + " table")
    result = {}
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or row[0] in result
        ):
            _fail(label + " identities")
        result[row[0]] = copy.deepcopy(row[1])
    return result


def _pairs(value: dict[str, Any]) -> list[list[Any]]:
    return [[key, copy.deepcopy(item)] for key, item in value.items()]


def _fresh(heap: dict[str, Any], identity: Any) -> None:
    if not isinstance(identity, str) or not _ID.fullmatch(identity) or identity in heap:
        _fail("fresh public heap identity required")


def _namespace(plan: dict[str, Any], module_file: Any) -> dict[str, Any]:
    row = next((item for item in plan["namespaces"] if item["file"] == module_file), None)
    if row is None:
        _fail("namespace has no source-owned linker row")
    cells = {item["id"]: item for item in plan["cells"]}
    return {
        "file": row["file"],
        "namespaceName": row["namespaceName"],
        "exports": [
            {
                "name": item["name"],
                "cell": item["cell"],
                "loweredName": cells[item["cell"]]["loweredName"],
            }
            for item in row["exports"]
        ],
    }


def _verify_namespace_node(node: Any, identity: str, plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(node, dict):
        _fail("namespace heap object is absent")
    expected = _namespace(plan, node.get("namespaceFile"))
    if (
        node.get("id") != identity
        or node.get("kind") != "module-namespace"
        or node.get("prototype") is not None
        or node.get("extensible") is not False
        or node.get("namespacePlanDigest") != plan["digest"]
        or node.get("namespaceFile") != expected["file"]
        or node.get("namespaceName") != expected["namespaceName"]
        or node.get("namespaceExports") != expected["exports"]
        or node.get("namespaceExportNames") != [item["name"] for item in expected["exports"]]
        or node.get("properties") != [[["symbol", "well-known:toStringTag"], ["str", "Module"]]]
    ):
        _fail("namespace heap object differs from its source-owned plan")
    return expected


def apply_experimental_module_namespace_shared_transition(
    shared_state: Any, plan: Any, transition: Any
) -> dict[str, Any]:
    """Apply bounded namespace effects to the same heap and locals used by V8."""
    verify_experimental_module_namespace_plan(plan)
    if not isinstance(shared_state, dict):
        _fail("shared machine state")
    heap = _table(shared_state.get("heap"), "heap")
    local_values = _table(shared_state.get("locals"), "locals")
    if not isinstance(transition, dict) or not isinstance(transition.get("kind"), str):
        _fail("namespace operation")
    next_heap = copy.deepcopy(heap)
    next_locals = copy.deepcopy(local_values)
    kind = transition["kind"]

    if kind == "namespace.create":
        if not _exact(transition, {"kind", "namespaceId", "moduleFile", "bind"}):
            _fail("namespace allocation fields")
        expected = _namespace(plan, transition["moduleFile"])
        if transition["bind"] != expected["namespaceName"]:
            _fail("namespace source binding differs from linker")
        namespace_id = transition["namespaceId"]
        _fresh(next_heap, namespace_id)
        exports = expected["exports"]
        next_heap[namespace_id] = {
            "id": namespace_id,
            "kind": "module-namespace",
            "prototype": None,
            "extensible": False,
            "namespacePlanDigest": plan["digest"],
            "namespaceFile": expected["file"],
            "namespaceName": expected["namespaceName"],
            "namespaceExports": exports,
            "namespaceExportNames": [item["name"] for item in exports],
            "properties": [[["symbol", "well-known:toStringTag"], ["str", "Module"]]],
        }
        next_locals[expected["namespaceName"]] = ["ref", namespace_id]
    elif kind == "namespace.keys":
        if not _exact(transition, {"kind", "namespaceId", "names", "resultId", "bind"}):
            _fail("namespace keys fields")
        namespace_id = transition["namespaceId"]
        namespace = _verify_namespace_node(next_heap.get(namespace_id), namespace_id, plan)
        names = [item["name"] for item in namespace["exports"]]
        if transition["names"] != names or not isinstance(transition["bind"], str):
            _fail("namespace keys differ from source order")
        result_id = transition["resultId"]
        _fresh(next_heap, result_id)
        next_heap[result_id] = {
            "id": result_id,
            "kind": "array",
            "length": len(names),
            "prototype": "ordinary",
            "properties": [
                [["str", str(index)], ["str", name]] for index, name in enumerate(names)
            ],
        }
        next_locals[transition["bind"]] = ["ref", result_id]
    elif kind == "namespace.read":
        if not _exact(transition, {"kind", "namespaceId", "exportName", "result", "bind"}):
            _fail("namespace read fields")
        namespace_id = transition["namespaceId"]
        namespace = _verify_namespace_node(next_heap.get(namespace_id), namespace_id, plan)
        binding = next(
            (item for item in namespace["exports"] if item["name"] == transition["exportName"]),
            None,
        )
        if binding is None or not isinstance(transition["bind"], str):
            _fail("namespace read export or target")
        current = next_locals.get(binding["loweredName"])
        if (
            current is None
            or not _canonical_value(current)
            or not _canonical_value(transition["result"])
            or current != transition["result"]
        ):
            _fail("namespace read differs from current live source cell")
        next_locals[transition["bind"]] = copy.deepcopy(current)
    elif kind == "namespace.descriptor":
        if not _exact(
            transition,
            {"kind", "namespaceId", "exportName", "result", "resultId", "bind"},
        ):
            _fail("namespace descriptor fields")
        namespace_id = transition["namespaceId"]
        namespace = _verify_namespace_node(next_heap.get(namespace_id), namespace_id, plan)
        binding = next(
            (item for item in namespace["exports"] if item["name"] == transition["exportName"]),
            None,
        )
        if binding is None or not isinstance(transition["bind"], str):
            _fail("namespace descriptor export or target")
        current = next_locals.get(binding["loweredName"])
        expected_result = {
            "value": copy.deepcopy(current),
            "writable": True,
            "enumerable": True,
            "configurable": False,
        }
        if not _canonical_value(current) or transition["result"] != expected_result:
            _fail("namespace descriptor differs from current live source cell")
        result_id = transition["resultId"]
        _fresh(next_heap, result_id)
        next_heap[result_id] = {
            "id": result_id,
            "kind": "object",
            "prototype": "ordinary",
            "properties": [
                [["str", "value"], copy.deepcopy(current)],
                [["str", "writable"], ["bool", True]],
                [["str", "enumerable"], ["bool", True]],
                [["str", "configurable"], ["bool", False]],
            ],
        }
        next_locals[transition["bind"]] = ["ref", result_id]
    else:
        _fail("unsupported shared namespace operation")

    result = copy.deepcopy(shared_state)
    result["heap"] = _pairs(next_heap)
    result["locals"] = _pairs(next_locals)
    return result


def validate_experimental_module_namespace_shared_state(shared_state: Any, plan: Any) -> bool:
    """Validate retained exotic heap nodes after JSON checkpoint restoration."""
    verify_experimental_module_namespace_plan(plan)
    if not isinstance(shared_state, dict):
        _fail("shared machine state")
    heap = _table(shared_state.get("heap"), "heap")
    local_values = _table(shared_state.get("locals"), "locals")
    for cell in plan["cells"]:
        if cell["loweredName"] in local_values and not _canonical_value(
            local_values[cell["loweredName"]]
        ):
            _fail("source cell is outside the scalar experimental subset")
    seen_namespaces = set()
    seen_bindings = set()
    for identity, node in heap.items():
        if isinstance(node, dict) and node.get("kind") == "module-namespace":
            if not isinstance(identity, str) or not _ID.fullmatch(identity):
                _fail("namespace identity")
            expected = _verify_namespace_node(node, identity, plan)
            if expected["file"] in seen_namespaces or expected["namespaceName"] in seen_bindings:
                _fail("duplicate restored namespace binding")
            seen_namespaces.add(expected["file"])
            seen_bindings.add(expected["namespaceName"])
            if local_values.get(expected["namespaceName"]) != ["ref", identity]:
                _fail("restored namespace local differs from its heap identity")
    return True
