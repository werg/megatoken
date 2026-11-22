"""Controls for experimental, non-admitted module namespace state parity."""

from copy import deepcopy

import pytest

from megatoken.data.module_namespace_state_v8 import (
    apply_experimental_module_namespace_transition,
    build_experimental_module_namespace_plan,
    create_experimental_module_namespace_state,
    verify_experimental_module_namespace_plan,
)
from megatoken.data.semantic_json import stable_algorithm_hash


MODULES = {
    "main.js": 'import * as ns from "./cells.js"; export function run(){ return Object.keys(ns); }',
    "cells.js": "export let value=1; export {value as alias}; export const fixed=2;",
}
CELLS = [
    {"file": "cells.js", "local": "value", "kind": "let", "mutable": True},
    {"file": "cells.js", "local": "fixed", "kind": "const", "mutable": False},
]
NAMESPACES = [
    {
        "file": "cells.js",
        "exports": [
            {"name": "alias", "file": "cells.js", "local": "value"},
            {"name": "fixed", "file": "cells.js", "local": "fixed"},
            {"name": "value", "file": "cells.js", "local": "value"},
        ],
    }
]


def _plan(**changes):
    fields = {
        "modules": MODULES,
        "entry_module": "main.js",
        "namespaces": NAMESPACES,
        "cells": CELLS,
    }
    fields.update(changes)
    return build_experimental_module_namespace_plan(**fields)


def test_plan_hashes_sources_and_live_aliases_read_current_cell():
    plan = _plan()
    assert verify_experimental_module_namespace_plan(plan)
    body = {key: value for key, value in plan.items() if key != "digest"}
    assert plan["digest"] == stable_algorithm_hash(body)

    state = create_experimental_module_namespace_state(
        plan,
        {
            "cells.js#value": {"initialized": True, "value": ["num", 1]},
            "cells.js#fixed": {"initialized": True, "value": ["num", 2]},
        },
    )
    created = apply_experimental_module_namespace_transition(
        plan,
        state,
        {"kind": "namespace.create", "namespaceId": "obj1", "moduleFile": "cells.js"},
    )
    listed = apply_experimental_module_namespace_transition(
        plan,
        created["state"],
        {"kind": "namespace.keys", "namespaceId": "obj1", "names": ["alias", "fixed", "value"]},
    )
    assert listed["result"] == ["alias", "fixed", "value"]
    written = apply_experimental_module_namespace_transition(
        plan,
        listed["state"],
        {"kind": "cell.write", "cell": "cells.js#value", "value": ["num", 9]},
    )
    read = apply_experimental_module_namespace_transition(
        plan,
        written["state"],
        {"kind": "namespace.read", "namespaceId": "obj1", "exportName": "alias", "value": ["num", 9]},
    )
    assert read["result"] == ["num", 9]
    descriptor = {
        "value": ["num", 9],
        "writable": True,
        "enumerable": True,
        "configurable": False,
    }
    observed = apply_experimental_module_namespace_transition(
        plan,
        read["state"],
        {
            "kind": "namespace.descriptor",
            "namespaceId": "obj1",
            "exportName": "value",
            "value": descriptor,
        },
    )
    assert observed["result"] == descriptor


def test_utf16_export_sort_and_numeric_index_names_match_javascript_order():
    names = ["𐀀", ""]
    plan = build_experimental_module_namespace_plan(
        modules={"main.js": "", "cells.js": ""},
        entry_module="main.js",
        cells=[{"file": "cells.js", "local": "value", "kind": "var", "mutable": True}],
        namespaces=[
            {
                "file": "cells.js",
                "exports": [
                    {"name": name, "file": "cells.js", "local": "value"} for name in names
                ],
            }
        ],
    )
    assert [row["name"] for row in plan["namespaces"][0]["exports"]] == names
    with pytest.raises(ValueError, match="malformed"):
        build_experimental_module_namespace_plan(
            modules={"main.js": "", "cells.js": ""},
            entry_module="main.js",
            cells=[{"file": "cells.js", "local": "value", "kind": "let", "mutable": True}],
            namespaces=[
                {
                    "file": "cells.js",
                    "exports": [
                        {"name": "4294967294", "file": "cells.js", "local": "value"}
                    ],
                }
            ],
        )


def test_tdz_const_writes_and_failed_transitions_leave_input_unchanged():
    plan = _plan()
    initial = create_experimental_module_namespace_state(plan)
    created = apply_experimental_module_namespace_transition(
        plan,
        initial,
        {"kind": "namespace.create", "namespaceId": "obj1", "moduleFile": "cells.js"},
    )["state"]
    snapshot = deepcopy(created)
    with pytest.raises(ValueError, match="before its source binding is initialized"):
        apply_experimental_module_namespace_transition(
            plan,
            created,
            {
                "kind": "namespace.read",
                "namespaceId": "obj1",
                "exportName": "value",
                "value": ["num", 1],
            },
        )
    assert created == snapshot
    initialized = apply_experimental_module_namespace_transition(
        plan,
        created,
        {"kind": "cell.initialize", "cell": "cells.js#fixed", "value": ["num", 2]},
    )["state"]
    with pytest.raises(ValueError, match="write is not allowed"):
        apply_experimental_module_namespace_transition(
            plan,
            initialized,
            {"kind": "cell.write", "cell": "cells.js#fixed", "value": ["num", 3]},
        )
    assert initialized["cells"]["cells.js#fixed"]["value"] == ["num", 2]


@pytest.mark.parametrize(
    "value",
    [
        ["bigint", "-0"],
        ["bigint", "+1"],
        ["num", "Infinity"],
        ["num", float("nan")],
        ["num", True],
        ["bool", 1],
        ["undefined", None],
        ["null", None],
        ["str", 1],
        ["object", "not scalar"],
    ],
)
def test_malformed_or_noncanonical_scalar_tags_are_refused(value):
    with pytest.raises(ValueError, match="initial cell value is malformed"):
        create_experimental_module_namespace_state(
            _plan(),
            {"cells.js#value": {"initialized": True, "value": value}},
        )


def test_namespace_observation_and_plan_tampering_are_refused():
    plan = _plan()
    changed = deepcopy(plan)
    changed["modules"][0]["sourceDigest"] = stable_algorithm_hash("changed")
    with pytest.raises(ValueError, match="integrity differs"):
        verify_experimental_module_namespace_plan(changed)

    state = create_experimental_module_namespace_state(plan)
    with pytest.raises(ValueError, match="malformed"):
        apply_experimental_module_namespace_transition(
            plan,
            state,
            {
                "kind": "namespace.create",
                "namespaceId": "obj1",
                "moduleFile": "cells.js",
                "extra": True,
            },
        )
    assert state["namespaces"] == {}


def test_bad_cell_and_namespace_state_shapes_refuse_without_mutation():
    plan = _plan()
    state = create_experimental_module_namespace_state(plan)
    forged = deepcopy(state)
    forged["cells"]["cells.js#value"]["version"] = -1
    with pytest.raises(ValueError, match="source cell state is malformed"):
        apply_experimental_module_namespace_transition(
            plan,
            forged,
            {"kind": "namespace.create", "namespaceId": "obj1", "moduleFile": "cells.js"},
        )
    malformed = deepcopy(state)
    malformed["namespaces"]["obj1"] = {"file": "cells.js", "exports": []}
    with pytest.raises(ValueError, match="namespace identity state is malformed"):
        apply_experimental_module_namespace_transition(
            plan,
            malformed,
            {"kind": "namespace.keys", "namespaceId": "obj1", "names": []},
        )


@pytest.mark.parametrize(
    "value",
    [
        ["undefined"],
        ["null"],
        ["bool", False],
        ["str", "live"],
        ["bigint", "0"],
        ["bigint", "-9007199254740993"],
        ["num", 0.0],
        ["num", 2**60],
        ["num", "-0"],
        ["num", "NaN"],
        ["num", "+Infinity"],
        ["num", "-Infinity"],
        ["num", 1.25],
    ],
)
def test_canonical_scalar_tags_are_accepted(value):
    state = create_experimental_module_namespace_state(
        _plan(),
        {"cells.js#value": {"initialized": True, "value": value}},
    )
    assert state["cells"]["cells.js#value"]["value"] == value


def test_explicit_null_initial_cells_is_not_the_omitted_default():
    with pytest.raises(ValueError, match="initial cells are malformed"):
        create_experimental_module_namespace_state(_plan(), None)



@pytest.mark.parametrize("value", [["num", -0.0], ["num", 9007199254740993]])
def test_noncanonical_negative_zero_and_inexact_js_integers_are_refused(value):
    with pytest.raises(ValueError, match="initial cell value is malformed"):
        create_experimental_module_namespace_state(
            _plan(),
            {"cells.js#value": {"initialized": True, "value": value}},
        )
