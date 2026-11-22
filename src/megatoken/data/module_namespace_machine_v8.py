"""Experimental module namespace effects over the shared V8 heap and locals."""

from __future__ import annotations

import copy
import re
from typing import Any

from .module_namespace_state_v8 import _canonical_value, _exact, _fail, verify_experimental_module_namespace_plan

_ID = re.compile(r"^o[1-9][0-9]*$")


def _table(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, list):
        _fail(label + " table")
    result = {}
    for row in value:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str) or row[0] in result:
            _fail(label + " identities")
        result[row[0]] = copy.deepcopy(row[1])
    return result


def _pairs(value: dict[str, Any]) -> list[list[Any]]:
    return [[key, copy.deepcopy(item)] for key, item in value.items()]


def _resolved(shared_state: dict[str, Any], name: Any) -> str:
    if not isinstance(name, str) or not name or len(name) > 256:
        _fail("binding identity")
    aliases = _table(shared_state.get("captureAliases", []), "capture alias")
    result = aliases.get(name, name)
    if not isinstance(result, str) or not result or len(result) > 256:
        _fail("captured binding identity")
    return result


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
                "mutable": cells[item["cell"]]["mutable"],
            }
            for item in row["exports"]
        ],
    }


def _verify_namespace(node: Any, identity: str, plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(node, dict):
        _fail("namespace heap object is absent")
    expected = _namespace(plan, node.get("namespaceFile"))
    if (
        set(node)
        != {
            "id",
            "kind",
            "prototype",
            "extensible",
            "namespacePlanDigest",
            "namespaceFile",
            "namespaceName",
            "namespaceExports",
            "namespaceExportNames",
            "properties",
        }
        or node["id"] != identity
        or node["kind"] != "module-namespace"
        or node["prototype"] is not None
        or node["extensible"] is not False
        or node["namespacePlanDigest"] != plan["digest"]
        or node["namespaceFile"] != expected["file"]
        or node["namespaceName"] != expected["namespaceName"]
        or node["namespaceExports"] != expected["exports"]
        or node["namespaceExportNames"] != [item["name"] for item in expected["exports"]]
        or node["properties"] != [[["symbol", "well-known:toStringTag"], ["str", "Module"]]]
    ):
        _fail("namespace heap object differs from its source-owned plan")
    return expected


def _fresh(
    heap: dict[str, Any], identity: Any, shared_state: dict[str, Any]
) -> None:
    protocol = shared_state.get("protocol") or {}
    reserved = {row.get("id") for row in protocol.get("symbols", []) if isinstance(row, dict)}
    reserved.update(
        row.get("id")
        for row in (protocol.get("construction") or {}).get("privateNames", [])
        if isinstance(row, dict)
    )
    if (
        not isinstance(identity, str)
        or not _ID.fullmatch(identity)
        or identity in heap
        or identity in reserved
    ):
        _fail("fresh public heap identity required")


def _ordinary_result(kind: str, properties: list[list[Any]], length: int | None = None) -> dict[str, Any]:
    node = {
        "kind": kind,
        "code": None,
        "captures": [],
        "prototype": "ordinary",
        "properties": [],
    }
    if kind == "array":
        node["properties"].append(["length", ["num", length]])
    node["properties"].extend(copy.deepcopy(properties))
    return node


def apply_experimental_module_namespace_shared_transition(
    shared_state: Any, plan: Any, transition: Any
) -> dict[str, Any]:
    """Apply a strict namespace/source-cell record atomically to shared state."""
    verify_experimental_module_namespace_plan(plan)
    if not isinstance(shared_state, dict):
        _fail("shared machine state")
    heap = _table(shared_state.get("heap"), "heap")
    locals_map = _table(shared_state.get("locals"), "locals")
    if not isinstance(transition, dict) or not isinstance(transition.get("kind"), str):
        _fail("namespace operation")
    next_heap = copy.deepcopy(heap)
    next_locals = copy.deepcopy(locals_map)
    kind = transition["kind"]

    if kind in {"cell.initialize", "cell.write"}:
        if not _exact(transition, {"kind", "cell", "loweredName", "value"}) or not _canonical_value(transition["value"]):
            _fail("source cell lifecycle fields")
        cell = next((item for item in plan["cells"] if item["id"] == transition["cell"]), None)
        if cell is None or cell["loweredName"] != transition["loweredName"]:
            _fail("source cell identity differs from linker")
        namespace_cells = {
            export["cell"]
            for namespace_row in plan["namespaces"]
            for export in namespace_row["exports"]
        }
        if cell["id"] not in namespace_cells:
            _fail("source cell is not namespace-backed")
        if kind == "cell.write" and not cell["mutable"]:
            _fail("write to immutable source cell")
        storage = _resolved(shared_state, cell["loweredName"])
        if kind == "cell.initialize":
            if storage in next_locals:
                _fail("duplicate source cell initialization")
        elif storage not in next_locals:
            _fail("source cell write before initialization")
        next_locals[storage] = copy.deepcopy(transition["value"])

    elif kind == "namespace.create":
        if not _exact(transition, {"kind", "namespaceId", "moduleFile"}):
            _fail("namespace allocation fields")
        expected = _namespace(plan, transition["moduleFile"])
        identity = transition["namespaceId"]
        _fresh(next_heap, identity, shared_state)
        next_heap[identity] = {
            "id": identity,
            "kind": "module-namespace",
            "prototype": None,
            "extensible": False,
            "namespacePlanDigest": plan["digest"],
            "namespaceFile": expected["file"],
            "namespaceName": expected["namespaceName"],
            "namespaceExports": expected["exports"],
            "namespaceExportNames": [item["name"] for item in expected["exports"]],
            "properties": [[["symbol", "well-known:toStringTag"], ["str", "Module"]]],
        }
        owner_name = _resolved(shared_state, expected["namespaceName"])
        if owner_name in next_locals:
            _fail("namespace owner binding already exists")
        next_locals[owner_name] = ["ref", identity]

    elif kind in {"namespace.keys", "namespace.read", "namespace.descriptor"}:
        identity = transition.get("namespaceId")
        if not isinstance(identity, str) or not _ID.fullmatch(identity):
            _fail("namespace identity")
        expected = _verify_namespace(next_heap.get(identity), identity, plan)
        if kind == "namespace.keys":
            if not _exact(transition, {"kind", "namespaceId", "names", "resultId"}):
                _fail("namespace keys fields")
            names = [item["name"] for item in expected["exports"]]
            if transition["names"] != names:
                _fail("namespace keys differ from source order")
            for item in expected["exports"]:
                storage = _resolved(shared_state, item["loweredName"])
                if storage not in next_locals:
                    _fail("TDZ Object.keys namespace inspection")
                if not _canonical_value(next_locals[storage]):
                    _fail("Object.keys namespace export is outside scalar subset")
            result_id = transition["resultId"]
            _fresh(next_heap, result_id, shared_state)
            next_heap[result_id] = _ordinary_result(
                "array",
                [[str(index), ["str", name]] for index, name in enumerate(names)],
                len(names),
            )
        else:
            fields = (
                {"kind", "namespaceId", "exportName", "value"}
                if kind == "namespace.read"
                else {"kind", "namespaceId", "exportName", "value", "resultId"}
            )
            if not _exact(transition, fields):
                _fail("namespace value fields")
            export = next((item for item in expected["exports"] if item["name"] == transition["exportName"]), None)
            if export is None:
                _fail("namespace export absent")
            storage = _resolved(shared_state, export["loweredName"])
            if storage not in next_locals:
                _fail("TDZ namespace read")
            current = next_locals[storage]
            if not _canonical_value(current):
                _fail("namespace cell is outside scalar experimental subset")
            if kind == "namespace.read":
                if transition["value"] != current:
                    _fail("namespace read differs from current source cell")
            else:
                descriptor = {
                    "value": copy.deepcopy(current),
                    "writable": True,
                    "enumerable": True,
                    "configurable": False,
                }
                if transition["value"] != descriptor:
                    _fail("namespace descriptor differs from current source cell")
                result_id = transition["resultId"]
                _fresh(next_heap, result_id, shared_state)
                next_heap[result_id] = _ordinary_result(
                    "object",
                    [
                        ["value", copy.deepcopy(current)],
                        ["writable", ["bool", True]],
                        ["enumerable", ["bool", True]],
                        ["configurable", ["bool", False]],
                    ],
                )
    else:
        _fail("unsupported shared namespace transition")

    result = copy.deepcopy(shared_state)
    result["heap"] = _pairs(next_heap)
    result["locals"] = _pairs(next_locals)
    return result


def validate_experimental_module_namespace_shared_state(shared_state: Any, plan: Any) -> bool:
    verify_experimental_module_namespace_plan(plan)
    if not isinstance(shared_state, dict):
        _fail("shared machine state")
    heap = _table(shared_state.get("heap"), "heap")
    locals_map = _table(shared_state.get("locals"), "locals")
    seen_files, seen_names = set(), set()
    for identity, node in heap.items():
        if isinstance(node, dict) and node.get("kind") == "module-namespace":
            if not isinstance(identity, str) or not _ID.fullmatch(identity):
                _fail("namespace identity")
            expected = _verify_namespace(node, identity, plan)
            if expected["file"] in seen_files or expected["namespaceName"] in seen_names:
                _fail("duplicate namespace identity")
            seen_files.add(expected["file"])
            seen_names.add(expected["namespaceName"])
            storage = _resolved(shared_state, expected["namespaceName"])
            if locals_map.get(storage) != ["ref", identity]:
                _fail("restored namespace owner binding differs from heap identity")
    namespace_cells = {
        export["cell"]
        for namespace_row in plan["namespaces"]
        for export in namespace_row["exports"]
    }
    for cell in plan["cells"]:
        if cell["id"] not in namespace_cells:
            continue
        storage = _resolved(shared_state, cell["loweredName"])
        if storage in locals_map and not _canonical_value(locals_map[storage]):
            _fail("source cell is outside scalar experimental subset")
    return True


def assert_experimental_module_namespace_state_unchanged(
    before: Any, after: Any, plan: Any
) -> bool:
    """Protect shared cell/owner locals from every non-namespace transition."""
    verify_experimental_module_namespace_plan(plan)
    if not isinstance(before, dict) or not isinstance(after, dict):
        _fail("shared namespace transition snapshots")
    before_locals = _table(before.get("locals"), "locals")
    after_locals = _table(after.get("locals"), "locals")
    before_aliases = _table(before.get("captureAliases", []), "capture alias")
    after_aliases = _table(after.get("captureAliases", []), "capture alias")
    protected = {
        cell["loweredName"]
        for cell in plan["cells"]
        if any(
            export["cell"] == cell["id"]
            for namespace in plan["namespaces"]
            for export in namespace["exports"]
        )
    }
    protected.update(namespace["namespaceName"] for namespace in plan["namespaces"])
    missing = object()
    for name in protected:
        left_storage = before_aliases.get(name, name)
        right_storage = after_aliases.get(name, name)
        if left_storage != right_storage:
            _fail("non-namespace transition changed a protected capture alias")
        left = before_locals.get(left_storage, missing)
        right = after_locals.get(right_storage, missing)
        if left is missing and right is missing:
            continue
        if left is missing or right is missing or left != right:
            _fail("non-namespace transition changed a live namespace owner or source cell")
    before_heap = _table(before.get("heap"), "heap")
    after_heap = _table(after.get("heap"), "heap")
    left_nodes = {
        identity: node
        for identity, node in before_heap.items()
        if isinstance(node, dict) and node.get("kind") == "module-namespace"
    }
    right_nodes = {
        identity: node
        for identity, node in after_heap.items()
        if isinstance(node, dict) and node.get("kind") == "module-namespace"
    }
    if left_nodes != right_nodes:
        _fail("non-namespace transition changed an exotic namespace heap node")
    validate_experimental_module_namespace_shared_state(after, plan)
    return True
