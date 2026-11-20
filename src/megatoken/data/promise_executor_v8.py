"""Bounded replay of owned synchronous Promise executor calls."""

from __future__ import annotations

import copy

PROFILE = "promise-executor/1"
STATE_SCHEMA = "semantic-promise-executor-state/1"
BOUND = 4096
FIELDS = {
    "promise.executor.enter": {"kind", "promise", "executor", "child", "resolve", "reject"},
    "promise.resolver.call": {"kind", "resolver", "value"},
    "promise.executor.finish": {"kind", "promise", "invocation", "completion"},
}


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid Promise executor protocol: " + message)


def _identity(value):
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("bounded identity required")


def _known(machine, value, self_id=None):
    from .protocol_v8 import _known as known

    return known(machine, value, self_id)


def _promise(machine, identity):
    node = machine["heap"].get(identity)
    if (
        not isinstance(node, dict)
        or node.get("kind") != "promise"
        or node.get("external") is not None
    ):
        _fail("owned internal Promise required")
    return node


def _state(machine, create=False):
    protocol = machine["protocol"]
    state = protocol.get("promiseExecutors")
    if state is None and create:
        state = {"schema": STATE_SCHEMA, "capabilities": [], "executors": []}
        protocol["promiseExecutors"] = state
    return state


def _resolver_node(promise, mode):
    return {
        "kind": "closure",
        "code": f"Promise.executor.{mode}:{promise}",
        "captures": [],
        "functionKind": "ordinary",
        "lexicalReceiver": None,
        "prototype": "ordinary",
        "properties": {},
        "promiseResolver": {"promise": promise, "mode": mode},
    }


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind) if isinstance(kind, str) else None
    if fields is None:
        return False
    from . import protocol_v8 as p

    p._exact(record, fields, "Promise executor record")
    if kind == "promise.executor.enter":
        for key in ("promise", "child", "resolve", "reject"):
            _identity(record[key])
        p.validate_protocol_value(record["executor"])
        if record["executor"][0] != "ref" or record["resolve"] == record["reject"]:
            _fail("owned executor and distinct resolver identities required")
    elif kind == "promise.resolver.call":
        p.validate_protocol_value(record["resolver"])
        p.validate_protocol_value(record["value"])
        if record["resolver"][0] != "ref":
            _fail("owned resolver required")
    else:
        _identity(record["promise"])
        _identity(record["invocation"])
        p._exact(record["completion"], {"kind", "value"}, "executor completion")
        if record["completion"]["kind"] not in ("return", "throw"):
            _fail("completion kind")
        p.validate_protocol_value(record["completion"]["value"])
    return True


def _fresh(machine, identity):
    from .protocol_v8 import _assert_fresh_heap_identity

    _assert_fresh_heap_identity(machine, identity)
    jobs = machine.get("jobs", [])
    jobs = dict(jobs) if isinstance(jobs, list) else jobs
    continuations = machine.get("continuations", [])
    continuations = (
        set(continuations) if isinstance(continuations, dict) else {r[0] for r in continuations}
    )
    if identity in jobs or identity in continuations:
        _fail("resolver identity collision")


def apply_record(machine, record):
    if not validate_record(record):
        return False
    from . import protocol_v8 as p
    from .promise_join_v8 import _settle

    if record["kind"] == "promise.executor.enter":
        target = _promise(machine, record["promise"])
        data = _state(machine)
        if (
            target["state"] != "pending"
            or target.get("adopted") is not None
            or target.get("join")
            or data
            and any(row["promise"] == record["promise"] for row in data["capabilities"])
        ):
            _fail("executor requires a fresh pending Promise")
        if data and len(data["capabilities"]) >= BOUND:
            _fail("capability count bound")
        _fresh(machine, record["resolve"])
        _fresh(machine, record["reject"])
        _known(machine, record["executor"])
        executor = machine["heap"].get(record["executor"][1])
        if not executor or executor.get("kind") != "closure" or executor.get("promiseResolver"):
            _fail("executor must be an owned source closure")
        parent = machine["stack"][-1] if machine["stack"] else None
        if not parent:
            _fail("executor requires an active parent")
        machine["heap"][record["resolve"]] = _resolver_node(record["promise"], "resolve")
        machine["heap"][record["reject"]] = _resolver_node(record["promise"], "reject")
        machine["_internalProtocolTransition"] = True
        try:
            p._apply(
                machine,
                {
                    "kind": "call",
                    "callee": record["executor"],
                    "receiver": ["undefined"],
                    "arguments": [["ref", record["resolve"]], ["ref", record["reject"]]],
                    "child": record["child"],
                },
            )
        finally:
            machine.pop("_internalProtocolTransition", None)
        data = _state(machine, create=True)
        data["capabilities"].append(
            {
                "promise": record["promise"],
                "resolve": record["resolve"],
                "reject": record["reject"],
                "called": False,
                "resolution": None,
            }
        )
        data["executors"].append(
            {
                "promise": record["promise"],
                "parent": parent,
                "child": record["child"],
                "phase": "running",
                "completion": None,
            }
        )
        return True

    if record["kind"] == "promise.resolver.call":
        resolver = _known(machine, record["resolver"])
        _known(machine, record["value"])
        data = _state(machine)
        capability = next(
            (
                row
                for row in (data or {}).get("capabilities", [])
                if row["resolve"] == resolver[1] or row["reject"] == resolver[1]
            ),
            None,
        )
        if capability is None:
            _fail("undeclared resolving function")
        mode = "resolve" if capability["resolve"] == resolver[1] else "reject"
        if p._node(machine, resolver[1]) != _resolver_node(capability["promise"], mode):
            _fail("resolving function metadata changed")
        if capability["called"]:
            return True
        target = _promise(machine, capability["promise"])
        value = record["value"]
        if target["state"] == "pending" and mode == "resolve" and value[0] == "ref":
            _fail("object and Promise resolution require their separate assimilation protocol")
        capability["called"] = True
        capability["resolution"] = {"mode": mode, "value": copy.deepcopy(value)}
        if target["state"] == "pending":
            _settle(
                machine,
                capability["promise"],
                "fulfilled" if mode == "resolve" else "rejected",
                value,
            )
        return True

    data = _state(machine)
    entry = next(
        (row for row in (data or {}).get("executors", []) if row["promise"] == record["promise"]),
        None,
    )
    capability = next(
        (
            row
            for row in (data or {}).get("capabilities", [])
            if row["promise"] == record["promise"]
        ),
        None,
    )
    if (
        entry is None
        or entry["phase"] != "running"
        or capability is None
        or entry["child"] != record["invocation"]
        or not machine["stack"]
        or machine["stack"][-1] != record["invocation"]
        or len(machine["stack"]) < 2
        or machine["stack"][-2] != entry["parent"]
    ):
        _fail("executor completion lacks its active call")
    completion = record["completion"]
    _known(machine, completion["value"])
    machine["_internalProtocolTransition"] = True
    try:
        p._apply(
            machine,
            {
                "kind": completion["kind"],
                "invocation": record["invocation"],
                "value": completion["value"],
            },
        )
    finally:
        machine.pop("_internalProtocolTransition", None)
    target = _promise(machine, record["promise"])
    if completion["kind"] == "throw" and not capability["called"] and target["state"] == "pending":
        _settle(machine, record["promise"], "rejected", completion["value"])
    entry["phase"] = "complete"
    entry["completion"] = copy.deepcopy(completion)
    machine["lastReturn"] = {"kind": "return", "value": ["ref", record["promise"]]}
    return True


def assert_next(machine, record):
    fields = record.get("fields", record)
    if record.get("kind") == "job" and fields.get("action") == "complete":
        jobs = machine.get("jobs", [])
        jobs = dict(jobs) if isinstance(jobs, list) else jobs
        job = jobs.get(fields.get("job")) or {}
        returned = job.get("returned") or {}
        value = returned.get("value") or []
        if (
            job.get("target")
            and returned.get("kind") == "return"
            and value[:1] == ["ref"]
            and machine["heap"].get(value[1], {}).get("kind") == "promise"
        ):
            _fail("reaction Promise return requires its assimilation job protocol")
    data = machine.get("protocol", {}).get("promiseExecutors")
    if not data:
        return True
    active = next(
        (
            row
            for row in data["executors"]
            if row["phase"] == "running" and row["child"] == (machine.get("stack") or [None])[-1]
        ),
        None,
    )
    if active and record.get("kind") in ("return", "throw"):
        _fail("executor completion requires promise.executor.finish")
    if record.get("kind") == "end" and any(row["phase"] == "running" for row in data["executors"]):
        _fail("cannot end during an active Promise executor")
    if (
        record.get("kind") == "promise"
        and fields.get("action") in ("resolve", "reject", "adopt")
        and any(row["promise"] == fields.get("promise") for row in data["capabilities"])
    ):
        _fail("managed Promise settlement requires its resolver record")
    if isinstance(record.get("kind"), str) and record["kind"].startswith("property."):
        identities = [fields.get("object")]
        receiver = fields.get("receiver")
        if isinstance(receiver, list) and len(receiver) == 2 and receiver[0] == "ref":
            identities.append(receiver[1])
        if any(machine["heap"].get(identity, {}).get("promiseResolver") for identity in identities):
            _fail("resolving functions have no ordinary properties")
    return True


def assert_callable(machine, callee):
    if machine["heap"].get(callee[1], {}).get("promiseResolver"):
        _fail("resolving functions require their explicit settlement record")


def validate_state(machine):
    data = machine.get("protocol", {}).get("promiseExecutors")
    if data is None:
        if any(
            node.get("promiseResolver")
            for node in machine.get("heap", {}).values()
            if isinstance(node, dict)
        ):
            _fail("orphan resolver identity")
        return True
    from . import protocol_v8 as p

    p._exact(data, {"schema", "capabilities", "executors"}, "Promise executor state")
    if data["schema"] != STATE_SCHEMA or not all(
        type(data[key]) is list and len(data[key]) <= BOUND for key in ("capabilities", "executors")
    ):
        _fail("executor state schema or count bound")
    promises, resolver_ids, children = set(), set(), set()
    entries = {
        entry.get("promise"): entry for entry in data["executors"] if isinstance(entry, dict)
    }
    for capability in data["capabilities"]:
        p._exact(
            capability,
            {"promise", "resolve", "reject", "called", "resolution"},
            "Promise capability",
        )
        for key in ("promise", "resolve", "reject"):
            _identity(capability[key])
        if capability["promise"] in promises or capability["resolve"] == capability["reject"]:
            _fail("duplicate Promise capability")
        promises.add(capability["promise"])
        target = _promise(machine, capability["promise"])
        for mode in ("resolve", "reject"):
            identity = capability[mode]
            node = p._node(machine, identity) if identity in machine["heap"] else None
            if identity in resolver_ids or node != _resolver_node(capability["promise"], mode):
                _fail("resolver identity or metadata differs")
            resolver_ids.add(identity)
        if type(capability["called"]) is not bool or capability["called"] != (
            capability["resolution"] is not None
        ):
            _fail("resolving function first-call cell")
        resolution = capability["resolution"]
        agrees = False
        if resolution is not None:
            p._exact(resolution, {"mode", "value"}, "resolver first call")
            if resolution["mode"] not in ("resolve", "reject"):
                _fail("resolution mode")
            value = _known(machine, resolution["value"])
            agrees = (
                (resolution["mode"] == "reject" or value[0] != "ref")
                and target.get("adopted") is None
                and target["state"]
                == ("fulfilled" if resolution["mode"] == "resolve" else "rejected")
                and target.get("result") == value
            )
        entry = entries.get(capability["promise"])
        thrown = bool(
            entry
            and entry.get("phase") == "complete"
            and isinstance(entry.get("completion"), dict)
            and entry["completion"].get("kind") == "throw"
            and target["state"] == "rejected"
            and target.get("adopted") is None
            and target.get("result") == entry["completion"].get("value")
        )
        if capability["called"]:
            if not agrees and not thrown:
                _fail("resolver first call contradicts Promise settlement")
        elif not (target["state"] == "pending" and target.get("adopted") is None or thrown):
            _fail("executor throw and Promise settlement disagree")
    completed = set()
    for entry in data["executors"]:
        p._exact(
            entry, {"promise", "parent", "child", "phase", "completion"}, "executor continuation"
        )
        for key in ("promise", "parent", "child"):
            _identity(entry[key])
        if (
            entry["promise"] not in promises
            or entry["promise"] in completed
            or entry["child"] in children
        ):
            _fail("executor capability/call linkage")
        completed.add(entry["promise"])
        children.add(entry["child"])
        if (
            entry["parent"] not in machine["frameFunctions"]
            or entry["child"] not in machine["frameFunctions"]
        ):
            _fail("executor source frames missing")
        if entry["phase"] not in ("running", "complete"):
            _fail("executor phase")
        if entry["phase"] == "running":
            call = next(
                (
                    row
                    for row in machine["protocol"]["callFrames"]
                    if row["child"] == entry["child"]
                ),
                None,
            )
            cap = next(row for row in data["capabilities"] if row["promise"] == entry["promise"])
            if (
                entry["completion"] is not None
                or entry["parent"] not in machine["stack"]
                or entry["child"] not in machine["stack"]
                or not call
                or call["parent"] != entry["parent"]
                or call["arguments"] != [["ref", cap["resolve"]], ["ref", cap["reject"]]]
            ):
                _fail("unfinished executor call frame or resolving pair")
        else:
            p._exact(entry["completion"], {"kind", "value"}, "executor completion")
            _known(machine, entry["completion"]["value"])
            if (
                entry["completion"]["kind"] not in ("return", "throw")
                or entry["child"] in machine["stack"]
            ):
                _fail("completed executor state")
    if completed != promises:
        _fail("capability lacks its executor")
    for identity, node in machine["heap"].items():
        if isinstance(node, dict) and node.get("promiseResolver") and identity not in resolver_ids:
            _fail("orphan resolver closure")
    return True
