"""Plan-bound module lifecycle derived independently from shared public records."""

from __future__ import annotations

import copy
import re

from .algorithm_v7 import stable_algorithm_hash
from .module_scheduler_state_v8 import (
    _verify_state,
    apply_module_scheduler_action_v8,
    create_module_scheduler_state_v8,
    verify_module_scheduler_plan_v8,
)
from .promise_scheduler_v8 import rows

PROFILE = "bound-module-scheduler/1"
PLAN_SCHEMA = "semantic-bound-module-scheduler-plan/1"
INITIALIZE = "module.scheduler.initialize"
FIELDS = {INITIALIZE: {"kind", "object", "planDigest"}}
FUNCTIONS = {
    "asyncDriver": "__traced_eval_async",
    "syncDriver": "__traced_eval_sync",
    "failure": "__traced_eval_failed",
    "startAsync": "__traced_eval_start_async",
}
LEDGER_FIELDS = {
    "kind",
    "properties",
    "plan",
    "owner",
    "state",
    "records",
    "settlements",
    "sequence",
}
COMPLETION_REGISTRATION = """function __traced_eval_start_async(record){
    if(record.started)throw new TypeError('Module body already started');
    record.started=true;
    record.execution=__traced_eval_async(record);
    record.execution.then(function(){__traced_eval_fulfilled(record);},
      function(error){__traced_eval_failed(record,error);});
  }"""


def _need(condition, message):
    if not condition:
        raise ValueError("Bound module scheduler: " + message)


def _exact(value, fields, message):
    _need(type(value) is dict and set(value) == set(fields), message)


def _owner(machine):
    return next(iter(rows(machine.get("frameFunctions", {}))), None)


def _ledgers(machine):
    return [
        (identity, node)
        for identity, node in machine["heap"].items()
        if node.get("kind") == "module-scheduler"
    ]


def verify_bound_module_scheduler_plan(plan, *, source=None, entry=None):
    _exact(
        plan, {"schema", "sourceDigest", "entry", "modulePlan", "functions", "digest"}, "plan shape"
    )
    _need(
        plan["schema"] == PLAN_SCHEMA
        and stable_algorithm_hash({k: v for k, v in plan.items() if k != "digest"})
        == plan["digest"],
        "plan digest",
    )
    _need(
        isinstance(plan["entry"], str)
        and plan["entry"]
        and isinstance(plan["sourceDigest"], str)
        and re.fullmatch(r"[0-9a-f]{64}", plan["sourceDigest"]),
        "plan source identity",
    )
    verify_module_scheduler_plan_v8(plan["modulePlan"])
    functions = plan["functions"]
    _exact(functions, set(FUNCTIONS) | {"fulfilled", "rejected"}, "source function shape")
    _need(
        all(functions[key] == value for key, value in FUNCTIONS.items()), "driver source identities"
    )
    _need(
        all(
            isinstance(functions[key], str) and re.fullmatch(r"function:\d+", functions[key])
            for key in ("fulfilled", "rejected")
        )
        and functions["fulfilled"] != functions["rejected"],
        "completion callback identities",
    )
    if source is not None:
        _need(
            stable_algorithm_hash(source) == plan["sourceDigest"] and entry == plan["entry"],
            "plan differs from active source program",
        )
        modules = plan["modulePlan"]["modules"]
        positions = {row["file"]: row["index"] for row in modules}
        records = [
            line
            for line in source.split("\n")
            if re.match(r"  __traced_eval_modules\[\d+\]=\{index:", line)
        ]
        _need(len(records) == len(modules), "source module record inventory differs")
        for index, module in enumerate(modules):
            entry_flag = str(module["file"] == plan["modulePlan"]["entryModule"]).lower()
            tla_flag = str(module["hasTLA"]).lower()
            dependencies = ",".join(str(positions[file]) for file in module["dependencies"])
            expected = (
                f"  __traced_eval_modules[{index}]={{index:{index},entry:{entry_flag},"
                f"hasTLA:{tla_flag},dependencies:[{dependencies}],"
            )
            _need(records[index].startswith(expected), "source module dependencies or TLA differ")
        # Public function identities use JavaScript UTF-16 offsets. The native
        # check preserves those offsets even when original modules use Unicode.
        start = source.find(COMPLETION_REGISTRATION)
        _need(start >= 0, "exact completion registration source required")
        for key, callback in (
            ("fulfilled", "function(){__traced_eval_fulfilled(record);}"),
            ("rejected", "function(error){__traced_eval_failed(record,error);}"),
        ):
            position = start + COMPLETION_REGISTRATION.index(callback)
            offset = len(source[:position].encode("utf-16-le")) // 2
            _need(
                functions[key] == "function:" + str(offset),
                "callback differs from exact completion registration source",
            )
        for key, name in FUNCTIONS.items():
            pattern = (
                (r"async\s+" if key == "asyncDriver" else r"(?<!async )")
                + r"function\s+"
                + re.escape(name)
                + r"\s*\("
            )
            _need(len(re.findall(pattern, source)) == 1, "exact source helper required")
    return plan


def validate_record(record):
    if type(record) is not dict or record.get("kind") != INITIALIZE:
        return False
    _exact(record, FIELDS[INITIALIZE], "initialization shape")
    _need(
        isinstance(record["object"], str)
        and 0 < len(record["object"]) <= 256
        and isinstance(record["planDigest"], str)
        and re.fullmatch(r"[0-9a-f]{64}", record["planDigest"]),
        "initialization identity or plan",
    )
    return True


def initialize(machine, record, plan):
    validate_record(record)
    _need(
        plan is not None and plan["digest"] == record["planDigest"],
        "initialization requires its active source-bound plan",
    )
    verify_bound_module_scheduler_plan(plan)
    _need(
        machine["stack"] == [_owner(machine)]
        and machine["protocol"]["sequence"] == 0
        and not _ledgers(machine)
        and record["object"] not in machine["heap"],
        "module scheduler must initialize at its fresh entry boundary",
    )
    machine["heap"][record["object"]] = {
        "kind": "module-scheduler",
        "properties": {},
        "plan": copy.deepcopy(plan),
        "owner": _owner(machine),
        "state": create_module_scheduler_state_v8(plan["modulePlan"]),
        "records": [],
        "settlements": [],
        "sequence": 0,
    }


def _record_owner(machine, plan, identity):
    node = machine["heap"].get(identity, {})
    props = node.get("properties", {})
    index = props.get("index")
    _need(
        node.get("kind") == "object"
        and type(index) is list
        and len(index) == 2
        and index[0] == "num"
        and type(index[1]) is int,
        "module argument is not its owned record",
    )
    modules = plan["modulePlan"]["modules"]
    _need(0 <= index[1] < len(modules), "module index")
    module = modules[index[1]]
    _need(
        props.get("hasTLA") == ["bool", module["hasTLA"]]
        and props.get("entry") == ["bool", module["file"] == plan["modulePlan"]["entryModule"]],
        "module record differs from active source plan",
    )
    ref = props.get("dependencies")
    deps = machine["heap"].get(ref[1], {}) if type(ref) is list and ref[0] == "ref" else {}
    positions = {row["file"]: row["index"] for row in modules}
    _need(
        deps.get("kind") == "array"
        and deps["properties"].get("length") == ["num", len(module["dependencies"])]
        and all(
            deps["properties"].get(str(i)) == ["num", positions[file]]
            for i, file in enumerate(module["dependencies"])
        ),
        "module dependency array differs from source plan",
    )
    ref = props.get("iterator")
    iterator = machine["heap"].get(ref[1], {}) if type(ref) is list and ref[0] == "ref" else {}
    ref = iterator.get("callee")
    callee = machine["heap"].get(ref[1], {}) if type(ref) is list and ref[0] == "ref" else {}
    _need(
        iterator.get("kind") == "generator"
        and callee.get("code") == "__traced_eval_module_" + str(module["index"]),
        "module lacks its exact source environment generator",
    )
    return module


def _binding(ledger, module, identity):
    existing = next((row for row in ledger["records"] if row["file"] == module["file"]), None)
    _need(
        existing is None or existing["object"] == identity, "module changed source record identity"
    )
    _need(
        not any(
            row["file"] != module["file"] and row["object"] == identity for row in ledger["records"]
        ),
        "reused module source record identity",
    )
    if existing is not None:
        return existing
    result = {"file": module["file"], "object": identity, "frame": None, "execution": None}
    ledger["records"].append(result)
    return result


def _promise(machine, identity):
    node = machine["heap"].get(identity, {})
    _need(
        node.get("kind") == "promise" and node.get("external") is None,
        "module Promise is not internal",
    )
    return node


def _promise_fact(machine, identity):
    node = _promise(machine, identity)
    return {
        "id": identity,
        "status": node["state"],
        **({} if node["state"] == "pending" else {"value": copy.deepcopy(node["result"])}),
    }


def _settlement_facts(ledger, settlement):
    module = next(
        row for row in ledger["plan"]["modulePlan"]["modules"] if row["file"] == settlement["file"]
    )
    success = settlement["status"] == "fulfilled"
    kind = (
        ("module-execution-settle" if success else "module-evaluation-failure")
        if module["hasTLA"]
        else ("module-synchronous-complete" if success else "module-synchronous-failure")
    )
    p = settlement["promise"]
    frame = settlement["frame"]
    job = settlement["job"]
    return {
        "promises": {}
        if p is None
        else {
            p: {
                "id": p,
                "status": settlement["status"],
                "value": copy.deepcopy(settlement["value"]),
            }
        },
        "jobs": {
            job: {
                "id": job,
                "promiseId": p,
                "frameId": frame,
                "status": settlement["status"],
                "moduleFile": settlement["file"],
                "kind": kind,
                "value": copy.deepcopy(settlement["value"]),
            }
        },
        "frames": {}
        if module["hasTLA"]
        else {
            frame: {
                "id": frame,
                "moduleFile": settlement["file"],
                "executionPromiseId": None,
                "status": "completed",
                "completion": {
                    "type": "return" if success else "throw",
                    "value": copy.deepcopy(settlement["value"]),
                },
            }
        },
    }


def validate_state(machine, plan=None, *, require_active_plan=False):
    ledgers = _ledgers(machine)
    if not ledgers:
        return True
    _need(
        len(ledgers) == 1 and (not require_active_plan or plan is not None),
        "module ledger requires one active source-bound program",
    )
    ledger = ledgers[0][1]
    _exact(ledger, LEDGER_FIELDS, "ledger shape")
    if plan is None:
        plan = ledger["plan"]
    verify_bound_module_scheduler_plan(plan)
    _need(
        ledger["plan"] == plan
        and ledger["owner"] == _owner(machine)
        and ledger["properties"] == {}
        and type(ledger["sequence"]) is int
        and 0 <= ledger["sequence"] <= 32768,
        "retained ledger owner or shape",
    )
    _verify_state(plan["modulePlan"], ledger["state"])
    _need(
        type(ledger["records"]) is list
        and type(ledger["settlements"]) is list
        and len(ledger["records"]) <= len(plan["modulePlan"]["modules"])
        and len(ledger["settlements"]) <= len(plan["modulePlan"]["modules"]),
        "ledger bound",
    )
    files = set()
    objects = set()
    settled = set()
    frames = rows(machine["frameFunctions"])
    continuations = rows(machine["continuations"])
    for record in ledger["records"]:
        _exact(record, {"file", "object", "frame", "execution"}, "module record binding")
        module = _record_owner(machine, plan, record["object"])
        state = ledger["state"]["modules"].get(record["file"])
        _need(
            module["file"] == record["file"]
            and record["file"] not in files
            and record["object"] not in objects,
            "module record binding identity",
        )
        files.add(record["file"])
        objects.add(record["object"])
        _need(
            record["frame"] == state["frameId"]
            and record["execution"] == state["executionPromiseId"],
            "retained driver identity",
        )
        if record["frame"] is not None:
            _need(
                frames.get(record["frame"])
                == plan["functions"]["asyncDriver" if module["hasTLA"] else "syncDriver"],
                "retained driver outside source plan",
            )
            if module["hasTLA"]:
                promise = _promise(machine, record["execution"])
                call = promise.get("asyncCall", {})
                _need(
                    call.get("owner") == ledger["owner"]
                    and call.get("invocation") == record["frame"]
                    and call.get("arguments", [])[:1] == [["ref", record["object"]]],
                    "retained execution Promise has another call owner",
                )
                if state["status"] == "suspended":
                    latest = call.get("continuations", [])[-1:]
                    _need(
                        call.get("phase") == "suspended"
                        and latest
                        and continuations.get(latest[0], {}).get("promise")
                        == state["awaited"]["promiseId"],
                        "retained module await differs",
                    )
                if state["status"] in ("fulfilled", "rejected"):
                    _need(
                        call.get("phase") == "completed" and promise["state"] == state["status"],
                        "retained module settlement differs",
                    )
    _need(
        all(
            row["startOrder"] is None or file in files
            for file, row in ledger["state"]["modules"].items()
        ),
        "started module lacks record owner",
    )
    jobs = rows(machine["jobs"])
    for settlement in ledger["settlements"]:
        _exact(
            settlement,
            {"file", "promise", "job", "frame", "status", "value"},
            "module settlement binding",
        )
        module = next(
            (row for row in plan["modulePlan"]["modules"] if row["file"] == settlement["file"]),
            None,
        )
        state = ledger["state"]["modules"].get(settlement["file"], {})
        _need(
            module is not None
            and settlement["file"] not in settled
            and state.get("status") == settlement["status"]
            and state.get("completionJobId") == settlement["job"]
            and state.get("frameId") == settlement["frame"]
            and state.get("executionPromiseId") == settlement["promise"],
            "retained settlement owner",
        )
        settled.add(settlement["file"])
        if module["hasTLA"]:
            promise = _promise(machine, settlement["promise"])
            job = jobs.get(settlement["job"], {})
            owner = next(row for row in ledger["records"] if row["file"] == settlement["file"])
            callback_kind = "fulfilled" if settlement["status"] == "fulfilled" else "rejected"
            fn = machine["heap"].get(
                job.get("closure" if callback_kind == "fulfilled" else "rejectionClosure"), {}
            )
            capture = next(
                (row for row in fn.get("captures", []) if row["binding"] == "record"), None
            )
            _need(
                promise["state"] == settlement["status"]
                and promise["result"] == settlement["value"]
                and job.get("promise") == settlement["promise"]
                and job.get("owner") == ledger["owner"]
                and job.get("status") in ("running", "completed")
                and job.get("originalStatus") == settlement["status"]
                and job.get("originalResult") == settlement["value"]
                and fn.get("code") == plan["functions"][callback_kind]
                and capture is not None
                and machine["locals"].get(capture["cell"]) == ["ref", owner["object"]],
                "retained reaction authority",
            )
        else:
            _need(
                settlement["job"] == "module-sync:" + settlement["frame"],
                "synchronous completion identity",
            )
    _need(
        all(
            state["frameId"] is None
            or state["status"] not in ("fulfilled", "rejected")
            or file in settled
            for file, state in ledger["state"]["modules"].items()
        ),
        "settled module has no retained completion authority",
    )
    return True


def assert_next(machine, record, plan):
    ledgers = _ledgers(machine)
    if plan is None:
        _need(not ledgers, "module ledger lacks an active source plan")
        return True
    if record["kind"] == INITIALIZE:
        _need(not ledgers, "duplicate module ledger")
        return True
    _need(len(ledgers) == 1, "source-bound module execution must initialize its ledger first")
    _need(
        record.get("fields", record).get("object") != ledgers[0][0],
        "logical module ledger cannot be targeted by guest records",
    )
    if record["kind"] == "end":
        _need(
            all(
                row["status"] in ("fulfilled", "rejected")
                for row in ledgers[0][1]["state"]["modules"].values()
            ),
            "unfinished module lifecycle obligations",
        )

    def inspect(value):
        if type(value) is list:
            _need(
                value != ["ref", ledgers[0][0]],
                "logical module ledger cannot escape into guest values",
            )
            for child in value:
                inspect(child)
        elif type(value) is dict:
            for child in value.values():
                inspect(child)

    inspect(record)
    return True


def advance(before, after, record, plan):
    if plan is None:
        return
    assert_next(before, record, plan)
    ledgers = _ledgers(after)
    _need(len(ledgers) == 1, "module execution lost ledger")
    ledger = ledgers[0][1]
    kind = record["kind"]
    f = record.get("fields", record)

    def step(action, facts):
        ledger["state"] = apply_module_scheduler_action_v8(
            plan["modulePlan"], ledger["state"], action, facts
        )
        ledger["sequence"] += 1

    if kind in ("call", "async.call"):
        value = f.get("callee")
        callee = before["heap"].get(value[1]) if type(value) is list and value[0] == "ref" else None
        if callee and callee.get("code") in (
            plan["functions"]["asyncDriver"],
            plan["functions"]["syncDriver"],
            plan["functions"]["failure"],
        ):
            _need(
                f.get("arguments") and f["arguments"][0][0] == "ref", "driver lacks record argument"
            )
            module = _record_owner(after, plan, f["arguments"][0][1])
            entry = _binding(ledger, module, f["arguments"][0][1])
            if callee["code"] == plan["functions"]["failure"]:
                if ledger["state"]["modules"][module["file"]]["status"] == "waiting":
                    dependency = next(
                        (
                            file
                            for file in module["dependencies"]
                            if ledger["state"]["modules"][file]["status"] == "rejected"
                        ),
                        None,
                    )
                    failure = (
                        ledger["state"]["modules"][dependency]["failure"] if dependency else None
                    )
                    settled = next(
                        (
                            row
                            for row in ledger["settlements"]
                            if failure and row["file"] == failure["moduleFile"]
                        ),
                        None,
                    )
                    _need(
                        settled is not None and f["arguments"][1:2] == [settled["value"]],
                        "parent failure lacks dependency outcome",
                    )
                    step(
                        {
                            "type": "module.propagateFailure",
                            "file": module["file"],
                            "dependencyFile": dependency,
                            "promiseId": settled["promise"],
                            "jobId": settled["job"],
                        },
                        _settlement_facts(ledger, settled),
                    )
            else:
                _need(
                    module["hasTLA"] == (kind == "async.call")
                    and callee["code"]
                    == plan["functions"]["asyncDriver" if module["hasTLA"] else "syncDriver"],
                    "wrong module driver",
                )
                entry["frame"] = f["child"]
                entry["execution"] = f["promise"] if module["hasTLA"] else None
                facts = {
                    "frames": {
                        entry["frame"]: {
                            "id": entry["frame"],
                            "moduleFile": module["file"],
                            "executionPromiseId": entry["execution"],
                        }
                    },
                    "promises": {}
                    if entry["execution"] is None
                    else {entry["execution"]: _promise_fact(after, entry["execution"])},
                    "jobs": {},
                }
                step(
                    {
                        "type": "module.start",
                        "file": module["file"],
                        "frameId": entry["frame"],
                        "executionPromiseId": entry["execution"],
                    },
                    facts,
                )
    elif kind in ("async.await", "async.resume"):
        entry = next((row for row in ledger["records"] if row["frame"] == f["invocation"]), None)
        if entry is not None:
            awaited = (
                f["awaited"]
                if kind == "async.await"
                else rows(before["continuations"])[f["continuation"]]["promise"]
            )
            promise = _promise_fact(after, awaited)
            action = {
                "type": "module.await" if kind == "async.await" else "module.resume",
                "file": entry["file"],
                "frameId": entry["frame"],
                "promiseId": awaited,
            }
            facts = {
                "frames": {
                    entry["frame"]: {
                        "id": entry["frame"],
                        "moduleFile": entry["file"],
                        "executionPromiseId": entry["execution"],
                        "awaitedPromiseId": awaited,
                    }
                },
                "promises": {awaited: promise},
                "jobs": {},
            }
            if kind == "async.resume":
                action["jobId"] = f["continuation"]
                facts["jobs"][f["continuation"]] = {
                    "id": f["continuation"],
                    "promiseId": awaited,
                    "frameId": entry["frame"],
                    "status": promise["status"],
                    "moduleFile": entry["file"],
                    "kind": "promise-reaction",
                    "value": copy.deepcopy(promise["value"]),
                }
            step(action, facts)
    elif kind == "job" and f["action"] == "dispatch":
        job = rows(before["jobs"]).get(f["job"])
        entry = next(
            (row for row in ledger["records"] if job and row["execution"] == job.get("promise")),
            None,
        )
        if entry is not None:
            promise = _promise(before, entry["execution"])
            fn = before["heap"].get(
                job.get("closure")
                if promise["state"] == "fulfilled"
                else job.get("rejectionClosure"),
                {},
            )
            capture = next(
                (row for row in fn.get("captures", []) if row["binding"] == "record"), None
            )
            _need(
                fn.get("code")
                == plan["functions"]["fulfilled" if promise["state"] == "fulfilled" else "rejected"]
                and capture is not None
                and before["locals"].get(capture["cell"]) == ["ref", entry["object"]],
                "settlement reaction lacks exact source record closure",
            )
            settled = {
                "file": entry["file"],
                "promise": entry["execution"],
                "job": f["job"],
                "frame": entry["frame"],
                "status": promise["state"],
                "value": copy.deepcopy(promise["result"]),
            }
            step(
                {
                    "type": "module.complete"
                    if promise["state"] == "fulfilled"
                    else "module.reject",
                    "file": entry["file"],
                    "frameId": entry["frame"],
                    "promiseId": entry["execution"],
                    "jobId": f["job"],
                },
                _settlement_facts(ledger, settled),
            )
            ledger["settlements"].append(settled)
    elif kind in ("return", "throw"):
        frame = f.get("invocation", before["stack"][-1] if before["stack"] else None)
        entry = next(
            (
                row
                for row in ledger["records"]
                if frame is not None and row["frame"] == frame and row["execution"] is None
            ),
            None,
        )
        if entry is not None:
            settled = {
                "file": entry["file"],
                "promise": None,
                "job": "module-sync:" + entry["frame"],
                "frame": entry["frame"],
                "status": "fulfilled" if kind == "return" else "rejected",
                "value": copy.deepcopy(f["value"]),
            }
            step(
                {
                    "type": "module.complete" if kind == "return" else "module.reject",
                    "file": entry["file"],
                    "frameId": entry["frame"],
                    "promiseId": None,
                    "jobId": settled["job"],
                },
                _settlement_facts(ledger, settled),
            )
            ledger["settlements"].append(settled)
    validate_state(after, plan, require_active_plan=True)


def validate_shared_state(state, plan):
    from .protocol_v8 import _internal_machine

    return validate_state(_internal_machine(state), plan, require_active_plan=True)


def advance_shared(before, after, record, plan):
    from .protocol_v8 import _external_machine, _internal_machine

    previous = _internal_machine(before)
    selected = _internal_machine(after)
    advance(previous, selected, record, plan)
    return _external_machine(selected)
