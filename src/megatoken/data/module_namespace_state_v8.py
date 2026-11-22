"""Experimental, unadmitted source-bound module namespace state contract.

This module models namespace identity, live export cells, own keys, reads and
own descriptors. It is intentionally not wired into source projection or
producer/native admission.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

from megatoken.data.semantic_json import stable_algorithm_hash


_PLAN_SCHEMA = "semantic-module-namespace-plan/1"
_STATE_SCHEMA = "semantic-module-namespace-state/1"
_HASH = re.compile(r"^[a-f0-9]{64}$")
_LOCAL = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_BIGINT = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_ARRAY_INDEX = re.compile(r"^(?:0|[1-9][0-9]*)$")
_SPECIAL_NUMBERS = {"-0", "NaN", "+Infinity", "-Infinity"}
_CELL_KINDS = {"let", "const", "function", "class", "var"}
_MISSING = object()


def _fail(message: str) -> None:
    raise ValueError(f"Experimental module namespace/1: {message}")


def _exact(value: Any, fields: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == fields


def _js_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be", "surrogatepass")


def _js_length(value: str) -> int:
    return len(value.encode("utf-16-le", "surrogatepass")) // 2


def _canonical_path(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.startswith("/")
        and "\\" not in value
        and all(part and part not in {".", ".."} for part in value.split("/"))
    )


def _array_index_name(value: str) -> bool:
    if not _ARRAY_INDEX.fullmatch(value):
        return False
    return int(value) <= 0xFFFFFFFE


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and not _array_index_name(value)


def _mutable_kind(kind: str) -> bool:
    return kind != "const"


def _canonical_value(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) == 1:
        return isinstance(value[0], str) and value[0] in {"undefined", "null"}
    if len(value) != 2:
        return False
    tag, payload = value
    if tag == "bool":
        return type(payload) is bool
    if tag == "str":
        return isinstance(payload, str)
    if tag == "bigint":
        return isinstance(payload, str) and _BIGINT.fullmatch(payload) is not None
    if tag == "num":
        if isinstance(payload, str):
            return payload in _SPECIAL_NUMBERS
        if isinstance(payload, bool) or not isinstance(payload, (int, float)):
            return False
        try:
            if type(payload) is int:
                converted = float(payload)
                return math.isfinite(converted) and int(converted) == payload
            return math.isfinite(payload) and not (
                payload == 0 and math.copysign(1.0, payload) < 0
            )
        except OverflowError:
            return False
    return False


def _js_number_equal(left: int | float, right: int | float) -> bool:
    left_number = float(left)
    right_number = float(right)
    if left_number != right_number:
        return False
    if left_number == 0:
        return math.copysign(1.0, left_number) == math.copysign(1.0, right_number)
    return True


def _deep_strict_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _js_number_equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _deep_strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _deep_strict_equal(left[key], right[key]) for key in left
        )
    return left == right


def _cell_id(file: str, local: str) -> str:
    return f"{file}#{local}"


def build_experimental_module_namespace_plan(
    *, modules: Any, entry_module: Any, namespaces: Any, cells: Any
) -> dict[str, Any]:
    """Build a plan from canonical linker records; this grants no admission claim."""
    if (
        not isinstance(modules, Mapping)
        or not _canonical_path(entry_module)
        or entry_module not in modules
        or not isinstance(namespaces, list)
        or not isinstance(cells, list)
        or any(not isinstance(file, str) for file in modules)
    ):
        _fail("module graph, entry or linker records are malformed")
    files = sorted(modules, key=_js_sort_key)
    if (
        not 1 <= len(files) <= 64
        or any(
            not _canonical_path(file)
            or not isinstance(modules[file], str)
            or _js_length(modules[file]) > 65536
            for file in files
        )
    ):
        _fail("module graph exceeds bounds or contains an invalid source")

    cell_map: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if (
            not _exact(cell, {"file", "local", "kind", "mutable"})
            or type(cell["mutable"]) is not bool
            or not _canonical_path(cell["file"])
            or cell["file"] not in modules
            or not isinstance(cell["local"], str)
            or not _LOCAL.fullmatch(cell["local"])
            or not isinstance(cell["kind"], str)
            or cell["kind"] not in _CELL_KINDS
            or cell["mutable"] != _mutable_kind(cell["kind"])
        ):
            _fail("canonical source cell record is malformed")
        identity = _cell_id(cell["file"], cell["local"])
        if identity in cell_map:
            _fail("duplicate canonical source cell")
        cell_map[identity] = {
            "id": identity,
            "file": cell["file"],
            "local": cell["local"],
            "kind": cell["kind"],
            "mutable": _mutable_kind(cell["kind"]),
        }

    namespace_map: dict[str, dict[str, Any]] = {}
    for record in namespaces:
        if (
            not _exact(record, {"file", "exports"})
            or not _canonical_path(record["file"])
            or record["file"] not in modules
            or not isinstance(record["exports"], list)
            or record["file"] in namespace_map
        ):
            _fail("namespace link record is malformed or duplicated")
        exports: list[dict[str, str]] = []
        previous: str | None = None
        for entry in record["exports"]:
            if (
                not _exact(entry, {"name", "file", "local"})
                or not _name(entry["name"])
                or not _canonical_path(entry["file"])
                or not isinstance(entry["local"], str)
                or not _LOCAL.fullmatch(entry["local"])
            ):
                _fail("namespace export binding record is malformed")
            identity = _cell_id(entry["file"], entry["local"])
            if identity not in cell_map:
                _fail("namespace export does not resolve to a declared source cell")
            if previous is not None and _js_sort_key(previous) >= _js_sort_key(entry["name"]):
                _fail("namespace exports must be unique and lexically sorted")
            previous = entry["name"]
            exports.append({"name": entry["name"], "cell": identity})
        namespace_map[record["file"]] = {"file": record["file"], "exports": exports}

    body = {
        "schema": _PLAN_SCHEMA,
        "entryModule": entry_module,
        "modules": [
            {"file": file, "sourceDigest": stable_algorithm_hash(modules[file])}
            for file in files
        ],
        "cells": sorted(cell_map.values(), key=lambda row: _js_sort_key(row["id"])),
        "namespaces": sorted(
            namespace_map.values(), key=lambda row: _js_sort_key(row["file"])
        ),
    }
    return {**body, "digest": stable_algorithm_hash(body)}


def verify_experimental_module_namespace_plan(plan: Any) -> bool:
    fields = {"schema", "entryModule", "modules", "cells", "namespaces", "digest"}
    if (
        not _exact(plan, fields)
        or plan["schema"] != _PLAN_SCHEMA
        or not _canonical_path(plan["entryModule"])
        or not isinstance(plan["digest"], str)
        or not _HASH.fullmatch(plan["digest"])
        or not isinstance(plan["modules"], list)
        or not 1 <= len(plan["modules"]) <= 64
        or not isinstance(plan["cells"], list)
        or len(plan["cells"]) > 4096
        or not isinstance(plan["namespaces"], list)
        or len(plan["namespaces"]) > 64
    ):
        _fail("namespace plan integrity differs")

    prior: str | None = None
    files: set[str] = set()
    for row in plan["modules"]:
        if (
            not _exact(row, {"file", "sourceDigest"})
            or not _canonical_path(row["file"])
            or not isinstance(row["sourceDigest"], str)
            or not _HASH.fullmatch(row["sourceDigest"])
            or (prior is not None and _js_sort_key(prior) >= _js_sort_key(row["file"]))
        ):
            _fail("namespace plan module table is malformed")
        prior = row["file"]
        files.add(row["file"])
    if plan["entryModule"] not in files:
        _fail("namespace plan entry is absent")

    prior = None
    cell_ids: set[str] = set()
    for row in plan["cells"]:
        if (
            not _exact(row, {"id", "file", "local", "kind", "mutable"})
            or not _canonical_path(row["file"])
            or row["file"] not in files
            or not isinstance(row["local"], str)
            or not _LOCAL.fullmatch(row["local"])
            or not isinstance(row["kind"], str)
            or row["kind"] not in _CELL_KINDS
            or type(row["mutable"]) is not bool
            or row["mutable"] != _mutable_kind(row["kind"])
            or row["id"] != _cell_id(row["file"], row["local"])
            or (prior is not None and _js_sort_key(prior) >= _js_sort_key(row["id"]))
        ):
            _fail("namespace plan cell table is malformed")
        prior = row["id"]
        cell_ids.add(row["id"])

    prior = None
    namespace_files: set[str] = set()
    for row in plan["namespaces"]:
        if (
            not _exact(row, {"file", "exports"})
            or not _canonical_path(row["file"])
            or row["file"] not in files
            or row["file"] in namespace_files
            or (prior is not None and _js_sort_key(prior) >= _js_sort_key(row["file"]))
            or not isinstance(row["exports"], list)
        ):
            _fail("namespace plan namespace table is malformed")
        prior = row["file"]
        namespace_files.add(row["file"])
        previous = None
        for entry in row["exports"]:
            if (
                not _exact(entry, {"name", "cell"})
                or not _name(entry["name"])
                or not isinstance(entry["cell"], str)
                or entry["cell"] not in cell_ids
                or (previous is not None and _js_sort_key(previous) >= _js_sort_key(entry["name"]))
            ):
                _fail("namespace plan export table is malformed")
            previous = entry["name"]

    body = {key: value for key, value in plan.items() if key != "digest"}
    if plan["digest"] != stable_algorithm_hash(body):
        _fail("namespace plan integrity differs")
    return True


def create_experimental_module_namespace_state(
    plan: Any, initial_cells: Any = _MISSING
) -> dict[str, Any]:
    verify_experimental_module_namespace_plan(plan)
    if initial_cells is _MISSING:
        initial_cells = {}
    if not isinstance(initial_cells, Mapping):
        _fail("initial cells are malformed")
    declared = {cell["id"] for cell in plan["cells"]}
    if any(identity not in declared for identity in initial_cells):
        _fail("initial state contains an undeclared source cell")
    cells: dict[str, dict[str, Any]] = {}
    for cell in plan["cells"]:
        identity = cell["id"]
        if identity not in initial_cells:
            cells[identity] = {"initialized": False, "value": None, "version": 0}
            continue
        initial = initial_cells[identity]
        if (
            not _exact(initial, {"initialized", "value"})
            or type(initial["initialized"]) is not bool
            or (initial["initialized"] and not _canonical_value(initial["value"]))
        ):
            _fail("initial cell value is malformed")
        cells[identity] = {
            "initialized": initial["initialized"],
            "value": copy.deepcopy(initial["value"]) if initial["initialized"] else None,
            "version": 0,
        }
    return {
        "schema": _STATE_SCHEMA,
        "planDigest": plan["digest"],
        "cells": cells,
        "namespaces": {},
    }


def _verify_state(plan: dict[str, Any], state: Any) -> None:
    if (
        not _exact(state, {"schema", "planDigest", "cells", "namespaces"})
        or state["schema"] != _STATE_SCHEMA
        or state["planDigest"] != plan["digest"]
        or not isinstance(state["cells"], dict)
        or not isinstance(state["namespaces"], dict)
        or any(not isinstance(identity, str) for identity in state["cells"])
        or not _deep_strict_equal(
            sorted(state["cells"], key=_js_sort_key),
            sorted((row["id"] for row in plan["cells"]), key=_js_sort_key),
        )
    ):
        _fail("state is not bound to the canonical source plan")
    for cell in plan["cells"]:
        value = state["cells"][cell["id"]]
        if (
            not _exact(value, {"initialized", "value", "version"})
            or type(value["initialized"]) is not bool
            or type(value["version"]) is not int
            or value["version"] < 0
            or value["version"] > 9007199254740991
            or (value["initialized"] and not _canonical_value(value["value"]))
            or (not value["initialized"] and value["value"] is not None)
        ):
            _fail("source cell state is malformed")
    for identity, row in state["namespaces"].items():
        row_file = row.get("file") if isinstance(row, dict) else None
        planned = next((item for item in plan["namespaces"] if item["file"] == row_file), None)
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"obj[1-9][0-9]*", identity)
            or not _exact(row, {"file", "exports"})
            or not _deep_strict_equal(planned, row)
        ):
            _fail("namespace identity state is malformed")


def apply_experimental_module_namespace_transition(
    plan: Any, state: Any, transition: Any
) -> dict[str, Any]:
    verify_experimental_module_namespace_plan(plan)
    _verify_state(plan, state)
    next_state = copy.deepcopy(state)
    answer = {"state": next_state, "result": None}

    def require_cell(identity: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        declared = next((row for row in plan["cells"] if row["id"] == identity), None)
        if declared is None or identity not in next_state["cells"]:
            _fail("transition references an undeclared source cell")
        return declared, next_state["cells"][identity]

    def require_namespace(identity: Any) -> dict[str, Any]:
        if not isinstance(identity, str) or identity not in next_state["namespaces"]:
            _fail("transition references an uncreated namespace")
        row = next_state["namespaces"][identity]
        bound = next(
            (item for item in plan["namespaces"] if item["file"] == row["file"]), None
        )
        if bound is None or not _deep_strict_equal(row["exports"], bound["exports"]):
            _fail("namespace identity or source export table changed")
        return bound

    kind = transition.get("kind") if isinstance(transition, dict) else None
    if kind == "namespace.create":
        if (
            not _exact(transition, {"kind", "namespaceId", "moduleFile"})
            or not isinstance(transition["namespaceId"], str)
            or not re.fullmatch(r"obj[1-9][0-9]*", transition["namespaceId"])
            or transition["namespaceId"] in next_state["namespaces"]
            or any(
                row["file"] == transition["moduleFile"]
                for row in next_state["namespaces"].values()
            )
        ):
            _fail("namespace allocation identity is malformed or duplicated")
        planned = next(
            (row for row in plan["namespaces"] if row["file"] == transition["moduleFile"]),
            None,
        )
        if planned is None:
            _fail("namespace allocation has no source-owned link")
        next_state["namespaces"][transition["namespaceId"]] = copy.deepcopy(planned)
        return answer

    if kind in {"cell.initialize", "cell.write"}:
        if not _exact(transition, {"kind", "cell", "value"}) or not _canonical_value(
            transition["value"]
        ):
            _fail("cell transition is malformed")
        declared, current = require_cell(transition["cell"])
        if kind == "cell.initialize":
            if current["initialized"]:
                _fail("source cell was initialized more than once")
        elif not current["initialized"] or not declared["mutable"]:
            _fail("write is not allowed for the initialized source cell kind")
        current["initialized"] = True
        current["value"] = copy.deepcopy(transition["value"])
        current["version"] += 1
        return answer

    if kind in {"namespace.keys", "namespace.read", "namespace.descriptor"}:
        fields = (
            {"kind", "namespaceId", "names"}
            if kind == "namespace.keys"
            else {"kind", "namespaceId", "exportName", "value"}
        )
        if not _exact(transition, fields):
            _fail("namespace observation record is malformed")
        planned = require_namespace(transition["namespaceId"])
        if kind == "namespace.keys":
            names = [row["name"] for row in planned["exports"]]
            if not _deep_strict_equal(transition["names"], names):
                _fail("own-key result differs from source export names")
            answer["result"] = copy.deepcopy(names)
            return answer
        binding = next(
            (row for row in planned["exports"] if row["name"] == transition["exportName"]),
            None,
        )
        if binding is None:
            _fail("namespace read names a missing export")
        _, current = require_cell(binding["cell"])
        if not current["initialized"]:
            _fail("namespace read occurs before its source binding is initialized")
        if kind == "namespace.read":
            if not _deep_strict_equal(transition["value"], current["value"]):
                _fail("namespace read differs from the current live cell")
            answer["result"] = copy.deepcopy(current["value"])
            return answer
        descriptor = {
            "value": copy.deepcopy(current["value"]),
            "writable": True,
            "enumerable": True,
            "configurable": False,
        }
        if not _deep_strict_equal(transition["value"], descriptor):
            _fail("namespace descriptor differs from the live namespace data descriptor")
        answer["result"] = descriptor
        return answer

    _fail("unsupported namespace transition kind")
