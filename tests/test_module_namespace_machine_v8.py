"""Controls for experimental namespace effects on shared heap and locals."""

from copy import deepcopy
import json

import pytest

from megatoken.data.module_namespace_state_v8 import build_experimental_module_namespace_plan
from megatoken.data.protocol_v8 import ProtocolV8Error, _internal_machine, _node
from megatoken.data.module_namespace_machine_v8 import (
    apply_experimental_module_namespace_shared_transition,
    validate_experimental_module_namespace_shared_state,
)


def _plan():
    return build_experimental_module_namespace_plan(
        modules={"main.js": "import * as ns from './cells.js';", "cells.js": "export let value=1; export {value as alias};"},
        entry_module="main.js",
        cells=[
            {
                "file": "cells.js",
                "local": "value",
                "kind": "let",
                "mutable": True,
                "loweredName": "__traced_eval_m1_value",
            }
        ],
        namespaces=[
            {
                "file": "cells.js",
                "namespaceName": "__traced_eval_namespace_0",
                "exports": [
                    {"name": "alias", "file": "cells.js", "local": "value"},
                    {"name": "value", "file": "cells.js", "local": "value"},
                ],
            }
        ],
    )


def _state():
    return {
        "heap": [],
        "locals": [["__traced_eval_m1_value", ["num", 1]]],
        "stack": [],
        "protocol": None,
    }


def _apply(state, kind, **fields):
    return apply_experimental_module_namespace_shared_transition(
        state, _plan(), {"kind": kind, **fields}
    )


def test_namespace_read_keys_and_descriptor_bind_real_shared_heap_results():
    plan = _plan()
    state = apply_experimental_module_namespace_shared_transition(
        _state(),
        plan,
        {
            "kind": "namespace.create",
            "namespaceId": "o15",
            "moduleFile": "cells.js",
            "bind": "__traced_eval_namespace_0",
        },
    )
    restored = json.loads(json.dumps(state))
    assert validate_experimental_module_namespace_shared_state(restored, plan)
    assert dict(restored["locals"])["__traced_eval_namespace_0"] == ["ref", "o15"]

    # A normal source local update is visible via both namespace export aliases.
    locals_map = dict(restored["locals"])
    locals_map["__traced_eval_m1_value"] = ["num", 9]
    restored["locals"] = [[key, value] for key, value in locals_map.items()]
    read = apply_experimental_module_namespace_shared_transition(
        restored,
        plan,
        {
            "kind": "namespace.read",
            "namespaceId": "o15",
            "exportName": "alias",
            "result": ["num", 9],
            "bind": "observed",
        },
    )
    assert dict(read["locals"])["observed"] == ["num", 9]

    keyed = apply_experimental_module_namespace_shared_transition(
        read,
        plan,
        {
            "kind": "namespace.keys",
            "namespaceId": "o15",
            "names": ["alias", "value"],
            "resultId": "o16",
            "bind": "keys",
        },
    )
    assert dict(keyed["locals"])["keys"] == ["ref", "o16"]
    assert dict(keyed["heap"])["o16"]["kind"] == "array"
    assert dict(keyed["heap"])["o16"]["properties"] == [
        [["str", "0"], ["str", "alias"]],
        [["str", "1"], ["str", "value"]],
    ]

    descriptor = {
        "value": ["num", 9],
        "writable": True,
        "enumerable": True,
        "configurable": False,
    }
    described = apply_experimental_module_namespace_shared_transition(
        keyed,
        plan,
        {
            "kind": "namespace.descriptor",
            "namespaceId": "o15",
            "exportName": "value",
            "result": descriptor,
            "resultId": "o17",
            "bind": "descriptor",
        },
    )
    assert dict(described["locals"])["descriptor"] == ["ref", "o17"]
    assert dict(described["heap"])["o17"]["kind"] == "object"
    assert validate_experimental_module_namespace_shared_state(
        json.loads(json.dumps(described)), plan
    )


def test_namespace_source_binding_and_heap_identity_are_exact():
    state = _state()
    with pytest.raises(ValueError, match="source binding"):
        apply_experimental_module_namespace_shared_transition(
            state,
            _plan(),
            {
                "kind": "namespace.create",
                "namespaceId": "o15",
                "moduleFile": "cells.js",
                "bind": "forgedAlias",
            },
        )
    with pytest.raises(ValueError, match="fresh public heap identity"):
        apply_experimental_module_namespace_shared_transition(
            state,
            _plan(),
            {
                "kind": "namespace.create",
                "namespaceId": "obj15",
                "moduleFile": "cells.js",
                "bind": "__traced_eval_namespace_0",
            },
        )


def test_live_read_cannot_supply_a_forged_value_or_reuse_heap_identity():
    state = _apply(
        _state(),
        "namespace.create",
        namespaceId="o15",
        moduleFile="cells.js",
        bind="__traced_eval_namespace_0",
    )
    with pytest.raises(ValueError, match="current live source cell"):
        _apply(
            state,
            "namespace.read",
            namespaceId="o15",
            exportName="value",
            result=["num", 2],
            bind="value",
        )
    with pytest.raises(ValueError, match="fresh public heap identity"):
        _apply(
            state,
            "namespace.keys",
            namespaceId="o15",
            names=["alias", "value"],
            resultId="o15",
            bind="keys",
        )


def test_checkpoint_restore_rejects_changed_namespace_exports_and_binding():
    state = _apply(
        _state(),
        "namespace.create",
        namespaceId="o15",
        moduleFile="cells.js",
        bind="__traced_eval_namespace_0",
    )
    changed = deepcopy(state)
    node = dict(dict(changed["heap"])["o15"])
    node["namespaceExportNames"] = ["value", "alias"]
    changed["heap"] = [["o15", node]]
    with pytest.raises(ValueError, match="source-owned plan"):
        validate_experimental_module_namespace_shared_state(changed, _plan())

    changed = deepcopy(state)
    changed["locals"] = [["__traced_eval_namespace_0", ["ref", "o16"]]]
    with pytest.raises(ValueError, match="heap identity"):
        validate_experimental_module_namespace_shared_state(changed, _plan())


def test_ordinary_property_protocol_rejects_namespace_exotics():
    state = _apply(
        _state(),
        "namespace.create",
        namespaceId="o15",
        moduleFile="cells.js",
        bind="__traced_eval_namespace_0",
    )
    machine = _internal_machine(
        {
            **state,
            "captureAliases": [],
            "frameFunctions": [],
        }
    )
    with pytest.raises(ProtocolV8Error, match="ordinary object required"):
        _node(machine, "o15")
