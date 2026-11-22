"""Controls for shared heap namespace operations and live source locals."""

import json
from copy import deepcopy

import pytest

from megatoken.data.module_namespace_machine_v8 import (
    apply_experimental_module_namespace_shared_transition as apply,
)
from megatoken.data.module_namespace_machine_v8 import (
    assert_experimental_module_namespace_state_unchanged,
)
from megatoken.data.module_namespace_machine_v8 import (
    validate_experimental_module_namespace_shared_state as validate_state,
)
from megatoken.data.module_namespace_state_v8 import build_experimental_module_namespace_plan
from megatoken.data.protocol_v8 import apply_protocol_to_shared_state, encode_protocol_record


def plan():
    return build_experimental_module_namespace_plan(
        modules={
            "main.js": "import * as ns from './cells.js';",
            "cells.js": "export let value=1; export {value as alias};",
        },
        entry_module="main.js",
        cells=[
            {
                "file": "cells.js",
                "local": "value",
                "kind": "let",
                "mutable": True,
                "loweredName": "__traced_eval_m1_value",
            },
            {
                "file": "cells.js",
                "local": "fixed",
                "kind": "const",
                "mutable": False,
                "loweredName": "__traced_eval_m1_fixed",
            },
        ],
        namespaces=[
            {
                "file": "cells.js",
                "namespaceName": "__traced_eval_namespace_0",
                "exports": [
                    {"name": "alias", "file": "cells.js", "local": "value"},
                    {"name": "fixed", "file": "cells.js", "local": "fixed"},
                    {"name": "value", "file": "cells.js", "local": "value"},
                ],
            }
        ],
    )


def initial():
    return {
        "heap": [],
        "locals": [
            ["__traced_eval_m1_value", ["num", 1]],
            ["__traced_eval_m1_fixed", ["num", 2]],
        ],
        "captureAliases": [],
        "frameFunctions": [],
        "stack": [],
        "protocol": None,
    }


def test_namespace_read_follows_single_live_cell_and_materializes_heap_results():
    source_plan = plan()
    state = apply(
        initial(),
        source_plan,
        {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
    )
    locals_map = dict(state["locals"])
    locals_map["__traced_eval_namespace_0"] = [
        "ref",
        "o15",
    ]  # ordinary source binding initialization
    state["locals"] = [[key, value] for key, value in locals_map.items()]
    assert validate_state(json.loads(json.dumps(state)), source_plan)

    written = apply(
        state,
        source_plan,
        {
            "kind": "cell.write",
            "cell": "cells.js#value",
            "loweredName": "__traced_eval_m1_value",
            "value": ["num", 9],
        },
    )
    read = apply(
        written,
        source_plan,
        {
            "kind": "namespace.read",
            "namespaceId": "o15",
            "exportName": "alias",
            "value": ["num", 9],
        },
    )
    keys = apply(
        read,
        source_plan,
        {
            "kind": "namespace.keys",
            "namespaceId": "o15",
            "names": ["alias", "fixed", "value"],
            "resultId": "o16",
        },
    )
    assert dict(keys["heap"])["o16"]["kind"] == "array"
    assert dict(keys["heap"])["o16"]["properties"] == [
        ["length", ["num", 3]],
        ["0", ["str", "alias"]],
        ["1", ["str", "fixed"]],
        ["2", ["str", "value"]],
    ]

    descriptor = {
        "value": ["num", 9],
        "writable": True,
        "enumerable": True,
        "configurable": False,
    }
    result = apply(
        keys,
        source_plan,
        {
            "kind": "namespace.descriptor",
            "namespaceId": "o15",
            "exportName": "value",
            "value": descriptor,
            "resultId": "o17",
        },
    )
    assert dict(result["heap"])["o17"]["properties"] == [
        ["value", ["num", 9]],
        ["writable", ["bool", True]],
        ["enumerable", ["bool", True]],
        ["configurable", ["bool", False]],
    ]
    assert validate_state(json.loads(json.dumps(result)), source_plan)


def test_tdz_and_immutable_cells_use_source_plan_and_local_presence():
    source_plan = plan()
    state = apply(
        {"heap": [], "locals": [], "captureAliases": [], "stack": [], "protocol": None},
        source_plan,
        {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
    )
    with pytest.raises(ValueError, match="TDZ"):
        apply(
            state,
            source_plan,
            {
                "kind": "namespace.read",
                "namespaceId": "o15",
                "exportName": "value",
                "value": ["num", 1],
            },
        )
    initialized = apply(
        state,
        source_plan,
        {
            "kind": "cell.initialize",
            "cell": "cells.js#value",
            "loweredName": "__traced_eval_m1_value",
            "value": ["undefined"],
        },
    )
    read = apply(
        initialized,
        source_plan,
        {
            "kind": "namespace.read",
            "namespaceId": "o15",
            "exportName": "alias",
            "value": ["undefined"],
        },
    )
    with pytest.raises(ValueError, match="immutable"):
        apply(
            read,
            source_plan,
            {
                "kind": "cell.write",
                "cell": "cells.js#fixed",
                "loweredName": "__traced_eval_m1_fixed",
                "value": ["num", 3],
            },
        )


def test_capture_aliases_share_the_same_export_cell():
    source_plan = plan()
    state = initial()
    state["locals"] = [["captured_cell", ["num", 1]], ["fixed_cell", ["num", 2]]]
    state["captureAliases"] = [
        ["__traced_eval_m1_value", "captured_cell"],
        ["__traced_eval_m1_fixed", "fixed_cell"],
        ["__traced_eval_namespace_0", "namespace_binding"],
    ]
    created = apply(
        state,
        source_plan,
        {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
    )
    locals_map = dict(created["locals"])
    locals_map["namespace_binding"] = ["ref", "o15"]
    created["locals"] = [[key, value] for key, value in locals_map.items()]
    updated = apply(
        created,
        source_plan,
        {
            "kind": "cell.write",
            "cell": "cells.js#value",
            "loweredName": "__traced_eval_m1_value",
            "value": ["num", 8],
        },
    )
    read = apply(
        updated,
        source_plan,
        {
            "kind": "namespace.read",
            "namespaceId": "o15",
            "exportName": "value",
            "value": ["num", 8],
        },
    )
    assert dict(read["locals"])["captured_cell"] == ["num", 8]
    assert validate_state(json.loads(json.dumps(read)), source_plan)


def test_namespace_heap_is_source_owned_and_generic_property_path_refuses_it():
    from megatoken.data.protocol_v8 import ProtocolV8Error, _internal_machine, _node

    source_plan = plan()
    state = apply(
        initial(),
        source_plan,
        {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
    )
    changed = deepcopy(state)
    node = dict(dict(changed["heap"])["o15"])
    node["namespaceExportNames"] = ["value", "alias", "fixed"]
    changed["heap"] = [["o15", node]]
    with pytest.raises(ValueError, match="source-owned plan"):
        validate_state(changed, source_plan)
    machine = _internal_machine({**state, "captureAliases": [], "frameFunctions": []})
    with pytest.raises(ProtocolV8Error, match="ordinary object required"):
        _node(machine, "o15")


def test_namespace_ids_and_result_ids_are_fresh_public_heap_ids():
    source_plan = plan()
    state = apply(
        initial(),
        source_plan,
        {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
    )
    with pytest.raises(ValueError, match="fresh public heap identity"):
        apply(
            state,
            source_plan,
            {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
        )
    with pytest.raises(ValueError, match="fresh public heap identity"):
        apply(
            state,
            source_plan,
            {
                "kind": "namespace.keys",
                "namespaceId": "o15",
                "names": ["alias", "fixed", "value"],
                "resultId": "obj16",
            },
        )


def test_protocol_lines_apply_namespace_effects_to_shared_machine_and_checkpoint():
    source_plan = plan()
    state = initial()
    state["stack"] = ["invocation-1"]
    create = {
        "kind": "namespace.create",
        "namespaceId": "o15",
        "moduleFile": "cells.js",
    }
    state = apply_protocol_to_shared_state(
        state, encode_protocol_record(create), namespace_plan=source_plan
    )
    assert state["protocol"]["sequence"] == 1
    state = json.loads(json.dumps(state))
    write = {
        "kind": "cell.write",
        "cell": "cells.js#value",
        "loweredName": "__traced_eval_m1_value",
        "value": ["num", 9],
    }
    state = apply_protocol_to_shared_state(
        state, encode_protocol_record(write), namespace_plan=source_plan
    )
    state = apply_protocol_to_shared_state(
        state,
        encode_protocol_record(
            {
                "kind": "namespace.read",
                "namespaceId": "o15",
                "exportName": "alias",
                "value": ["num", 9],
            }
        ),
        namespace_plan=source_plan,
    )
    state = apply_protocol_to_shared_state(
        state,
        encode_protocol_record(
            {
                "kind": "namespace.keys",
                "namespaceId": "o15",
                "names": ["alias", "fixed", "value"],
                "resultId": "o16",
            }
        ),
        namespace_plan=source_plan,
    )
    descriptor = {
        "kind": "namespace.descriptor",
        "namespaceId": "o15",
        "exportName": "value",
        "value": {
            "value": ["num", 9],
            "writable": True,
            "enumerable": True,
            "configurable": False,
        },
        "resultId": "o17",
    }
    state = apply_protocol_to_shared_state(
        state, encode_protocol_record(descriptor), namespace_plan=source_plan
    )
    assert state["protocol"]["sequence"] == 5
    assert dict(state["heap"])["o16"]["kind"] == "array"
    assert dict(state["heap"])["o17"]["kind"] == "object"
    with pytest.raises(ValueError, match="source-derived namespace plan"):
        apply_protocol_to_shared_state(
            state,
            encode_protocol_record(create),
        )


def test_protocol_namespace_descriptor_validates_before_shared_heap_allocation():
    source_plan = plan()
    state = initial()
    state["stack"] = ["invocation-1"]
    state = apply_protocol_to_shared_state(
        state,
        encode_protocol_record(
            {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"}
        ),
        namespace_plan=source_plan,
    )
    bad = {
        "kind": "namespace.descriptor",
        "namespaceId": "o15",
        "exportName": "value",
        "value": {
            "value": ["num", 1],
            "writable": False,
            "enumerable": True,
            "configurable": False,
        },
        "resultId": "o16",
    }
    with pytest.raises(ValueError):
        encode_protocol_record(bad)


def test_object_keys_reads_every_export_and_refuses_tdz():
    source_plan = plan()
    state = {
        "heap": [],
        "locals": [],
        "captureAliases": [],
        "frameFunctions": [],
        "stack": [],
        "protocol": None,
    }
    state["stack"] = ["invocation-1"]
    state = apply_protocol_to_shared_state(
        state,
        encode_protocol_record(
            {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"}
        ),
        namespace_plan=source_plan,
    )
    with pytest.raises(ValueError, match="TDZ Object.keys"):
        apply_protocol_to_shared_state(
            state,
            encode_protocol_record(
                {
                    "kind": "namespace.keys",
                    "namespaceId": "o15",
                    "names": ["alias", "fixed", "value"],
                    "resultId": "o16",
                }
            ),
            namespace_plan=source_plan,
        )


def test_generic_bind_cannot_bypass_namespace_owner_or_cell_lifecycle():
    source_plan = plan()
    state = initial()
    state["stack"] = ["invocation-1"]
    state = apply_protocol_to_shared_state(
        state,
        encode_protocol_record(
            {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"}
        ),
        namespace_plan=source_plan,
    )
    for name, value in [
        ("__traced_eval_namespace_0", ["null"]),
        ("__traced_eval_m1_value", ["num", 5]),
    ]:
        with pytest.raises(ValueError, match="lifecycle records"):
            apply_protocol_to_shared_state(
                state,
                encode_protocol_record({"kind": "bind", "name": name, "value": value}),
                namespace_plan=source_plan,
            )


def test_namespace_operation_cannot_bypass_pending_number_continuation_guard():
    from megatoken.data.number_v8 import SCHEMA as NUMBER_SCHEMA
    from megatoken.data.protocol_v8 import create_protocol_state

    source_plan = plan()
    state = initial()
    state["stack"] = ["invocation-1"]
    state["frameFunctions"] = [["invocation-1", "code-1"]]
    state["heap"] = [
        [
            "o20",
            {
                "id": "o20",
                "kind": "object",
                "prototype": None,
                "extensible": True,
                "properties": [],
            },
        ]
    ]
    protocol = create_protocol_state()
    protocol["numberConversions"] = {
        "schema": NUMBER_SCHEMA,
        "operations": [
            {
                "id": "conversion-1",
                "parent": "invocation-1",
                "input": ["ref", "o20"],
                "index": -1,
                "method": None,
                "builtin": None,
                "child": None,
                "phase": "select",
                "primitive": None,
                "result": None,
            }
        ],
    }
    state["protocol"] = protocol
    with pytest.raises(ValueError, match="pending continuation requires number.select"):
        apply_protocol_to_shared_state(
            state,
            encode_protocol_record(
                {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"}
            ),
            namespace_plan=source_plan,
        )


def test_detailed_bind_cannot_overwrite_namespace_cell_or_owner():
    from megatoken.data.protocol_v8 import ProtocolV8Error, assert_detailed_record_compatible

    source_plan = plan()
    state = apply(
        initial(),
        source_plan,
        {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
    )
    with pytest.raises(ProtocolV8Error, match="bypasses lifecycle"):
        assert_detailed_record_compatible(
            state,
            "bind",
            {"name": "__traced_eval_m1_value", "value": ["num", 5]},
            namespace_plan=source_plan,
        )


def test_compact_region_cannot_write_namespace_backed_cell():
    from megatoken.data.native_v8 import NativeV8Error, _replay_region
    from megatoken.data.semantic_json import stable_algorithm_hash

    source_plan = plan()
    state = initial()
    region = {
        "id": "r1",
        "digest": "region-digest",
        "completion": "normal",
        "operations": [
            {
                "kind": "bind",
                "binding": "__traced_eval_m1_value",
                "term": {"op": "literal", "value": ["num", 5]},
            }
        ],
    }
    program = {"definitionDigest": "definition-digest", "regions": [region]}
    context = {"state": state, "visibleThrough": 1}
    entry = {
        "definition": "definition-digest",
        "region": "r1",
        "regionDigest": "region-digest",
        "visibleThrough": 1,
        "stateDigest": stable_algorithm_hash(state),
    }
    target = 'region {"region":"r1","bindings":[],"decisions":[],"completion":{"kind":"normal"}}'
    with pytest.raises(NativeV8Error, match="Compact regions are unsupported"):
        _replay_region(program, context, target, entry, namespace_plan=source_plan)


def test_generic_protocol_outputs_cannot_mutate_cells_or_rebind_aliases():
    source_plan = plan()
    before = apply(
        initial(),
        source_plan,
        {"kind": "namespace.create", "namespaceId": "o15", "moduleFile": "cells.js"},
    )
    mutated = deepcopy(before)
    local_values = dict(mutated["locals"])
    local_values["__traced_eval_m1_value"] = ["num", 10]
    mutated["locals"] = [[key, value] for key, value in local_values.items()]
    with pytest.raises(ValueError, match="changed a live namespace owner or source cell"):
        assert_experimental_module_namespace_state_unchanged(before, mutated, source_plan)

    remapped = deepcopy(before)
    remapped["captureAliases"] = [["__traced_eval_m1_value", "other_cell"]]
    with pytest.raises(ValueError, match="changed a protected capture alias"):
        assert_experimental_module_namespace_state_unchanged(before, remapped, source_plan)
