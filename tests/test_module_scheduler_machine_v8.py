"""Bound module plans and initialization never supply independent lifecycle facts."""

import copy
import json

import pytest

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.module_scheduler_machine_v8 import (
    COMPLETION_REGISTRATION,
    FUNCTIONS,
    INITIALIZE,
    PLAN_SCHEMA,
    advance,
    assert_next,
    validate_shared_state,
    verify_bound_module_scheduler_plan,
)
from megatoken.data.module_scheduler_state_v8 import build_module_scheduler_plan_v8
from megatoken.data.native_v8 import _initial_state
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _internal_machine,
    apply_protocol_to_shared_state,
)


def fixture():
    source = (
        "// Unicode before callback identities: \U0001f310\n"
        "async function entry(){}\n"
        "  __traced_eval_modules[0]={index:0,entry:true,hasTLA:true,dependencies:[],parents:[]};\n"
        "  async function __traced_eval_async(record){}\n"
        "  function __traced_eval_sync(record){}\n"
        "  function __traced_eval_failed(record,error){}\n  " + COMPLETION_REGISTRATION
    )
    functions = dict(FUNCTIONS)
    for key, prefix in (
        ("fulfilled", "function(){__traced_eval_fulfilled"),
        ("rejected", "function(error){__traced_eval_failed"),
    ):
        position = source.index(prefix)
        functions[key] = "function:" + str(len(source[:position].encode("utf-16-le")) // 2)
    module_plan = build_module_scheduler_plan_v8(
        {
            "profile": "evaluated-module/1",
            "evaluationStrategy": "async-dependency-scheduler",
            "entryModule": "main.js",
            "modules": [{"file": "main.js", "source": "if(false)await 0;"}],
            "asyncDependencies": [
                {"file": "main.js", "index": 0, "hasTLA": True, "dependencies": []}
            ],
        }
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "sourceDigest": stable_algorithm_hash(source),
        "entry": "entry",
        "modulePlan": module_plan,
        "functions": functions,
    }
    plan["digest"] = stable_algorithm_hash(plan)
    state = _initial_state(
        {"entry": "entry"},
        {"invocationId": "root", "entry": "entry", "inputGraph": {"roots": [], "objects": []}},
    )
    record = {"kind": INITIALIZE, "object": "ledger", "planDigest": plan["digest"]}
    return source, plan, state, record


def select(state, record, plan):
    return apply_protocol_to_shared_state(
        state, "protocol " + json.dumps(record, separators=(",", ":")), module_scheduler_plan=plan
    )


def rehash(plan):
    plan["modulePlan"]["digest"] = stable_algorithm_hash(
        {key: value for key, value in plan["modulePlan"].items() if key != "digest"}
    )
    plan["digest"] = stable_algorithm_hash(
        {key: value for key, value in plan.items() if key != "digest"}
    )


def test_initialization_requires_active_source_plan_and_retains_exact_owner():
    source, plan, state, record = fixture()
    assert verify_bound_module_scheduler_plan(plan, source=source, entry="entry") == plan
    before = copy.deepcopy(state)
    selected = select(state, record, plan)
    assert state == before
    assert validate_shared_state(selected, plan)
    ledger = dict(selected["heap"])["ledger"]
    assert ledger["owner"] == "root" and ledger["plan"] == plan and ledger["sequence"] == 0
    assert ledger["state"]["modules"]["main.js"]["status"] == "waiting"
    with pytest.raises(ProtocolV8Error, match="source-bound"):
        select(state, record, None)


@pytest.mark.parametrize("mode", ["source", "entry", "TLA", "callback", "extra-field"])
def test_rehashed_plan_cannot_change_source_inventory_or_reaction_position(mode):
    source, plan, _, _ = fixture()
    if mode == "source":
        plan["sourceDigest"] = "0" * 64
    elif mode == "entry":
        plan["entry"] = "foreign"
    elif mode == "TLA":
        plan["modulePlan"]["modules"][0]["hasTLA"] = False
    elif mode == "callback":
        plan["functions"]["fulfilled"] = "function:0"
    else:
        plan["untrusted"] = True
    rehash(plan)
    with pytest.raises(ValueError):
        verify_bound_module_scheduler_plan(plan, source=source, entry="entry")


def test_initialization_is_first_once_and_cannot_reuse_an_existing_heap_identity():
    _, plan, state, record = fixture()
    with pytest.raises(ProtocolV8Error, match="initialize"):
        select(state, {"kind": "bind", "name": "x", "value": ["num", 1]}, plan)
    state = select(state, record, plan)
    before = copy.deepcopy(state)
    with pytest.raises(ProtocolV8Error, match="duplicate"):
        select(state, record, plan)
    assert state == before
    source, plan, state, record = fixture()
    state["heap"].append(["ledger", {"kind": "object", "properties": []}])
    with pytest.raises(ProtocolV8Error, match="fresh entry"):
        select(state, record, plan)


@pytest.mark.parametrize(
    "mode", ["missing-plan", "foreign-owner", "duplicate-ledger", "escape", "end"]
)
def test_retained_owner_and_lifecycle_cannot_be_bypassed(mode):
    _, plan, state, record = fixture()
    state = select(state, record, plan)
    if mode == "missing-plan":
        with pytest.raises(ValueError, match="active source-bound"):
            validate_shared_state(state, None)
    elif mode == "foreign-owner":
        dict(state["heap"])["ledger"]["owner"] = "foreign"
        with pytest.raises(ValueError, match="owner"):
            validate_shared_state(state, plan)
    elif mode == "duplicate-ledger":
        state["heap"].append(["duplicate", copy.deepcopy(dict(state["heap"])["ledger"])])
        with pytest.raises(ValueError, match="one active"):
            validate_shared_state(state, plan)
    else:
        record = (
            {"kind": "bind", "name": "escaped", "value": ["ref", "ledger"]}
            if mode == "escape"
            else {"kind": "end"}
        )
        with pytest.raises(ValueError, match="cannot escape|unfinished"):
            assert_next(_internal_machine(state), record, plan)


@pytest.mark.parametrize("record", [{"kind": "return"}, {"kind": "throw", "value": ["num", 7]}])
def test_unowned_completion_does_not_complete_an_unstarted_module(record):
    # A dependency rejection can bind a waiting parent that never enters its
    # driver. A later root completion has no current frame. None is absence of
    # ownership in both places; it must not identify a synchronous module call.
    _, plan, state, initialize_record = fixture()
    machine = _internal_machine(select(state, initialize_record, plan))
    machine["stack"] = []
    machine["heap"].update(
        {
            "module-record": {
                "kind": "object",
                "properties": {
                    "index": ["num", 0],
                    "entry": ["bool", True],
                    "hasTLA": ["bool", True],
                    "dependencies": ["ref", "dependencies"],
                    "iterator": ["ref", "iterator"],
                },
            },
            "dependencies": {"kind": "array", "properties": {"length": ["num", 0]}},
            "iterator": {"kind": "generator", "callee": ["ref", "module-body"]},
            "module-body": {"kind": "function", "code": "__traced_eval_module_0"},
        }
    )
    machine["heap"]["ledger"]["records"].append(
        {"file": "main.js", "object": "module-record", "frame": None, "execution": None}
    )
    before = copy.deepcopy(machine)
    advance(before, machine, record, plan)
    assert machine == before
