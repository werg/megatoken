"""Independent nested async call scopes on the existing Promise/FIFO machine."""

from __future__ import annotations

import copy

from .promise_scheduler_v8 import (
    assert_plain_resolution,
    promise,
    refresh_ready,
    rows,
    settle_promise,
    store_rows,
)

PROFILE = "async-scoped-call/1"
STATE_SCHEMA = "semantic-async-call-state/1"
BOUND = 4096
STATE_FIELDS = {
    "schema",
    "owner",
    "invocation",
    "callee",
    "receiver",
    "arguments",
    "parent",
    "phase",
    "attached",
    "continuations",
    "resumeScope",
    "completion",
    "resolution",
}
FIELDS = {
    "async.function": {"kind", "object"},
    "async.call": {"kind", "promise", "child", "callee", "receiver", "arguments"},
    "async.await": {"kind", "promise", "invocation", "awaited", "continuation"},
    "async.resume": {"kind", "promise", "invocation", "continuation", "status", "value"},
    "async.complete": {"kind", "promise", "invocation", "completion"},
    "async.resolve.enqueue": {"kind", "promise", "invocation", "target", "job", "settlementJob"},
    "async.resolve.subscribe": {"kind", "promise", "invocation", "target", "job", "settlementJob"},
    "async.resolve.settle": {"kind", "promise", "invocation", "target", "job", "status", "value"},
}
RESOLUTION_SCHEMA = "semantic-async-promise-resolution/1"
RESERVED_RESOLUTION_JOB_FIELDS = {
    "status",
    "owner",
    "promise",
    "reaction",
    "closure",
    "rejectionClosure",
    "target",
    "sequence",
    "asyncResolution",
}


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid async scoped call: " + message)


def _require(condition, message):
    if not condition:
        _fail(message)


def _id(value):
    _require(isinstance(value, str) and 0 < len(value) <= 256, "bounded identity required")


def _owner(machine):
    return next(iter(rows(machine.get("frameFunctions", {}))), None)


def _calls(machine):
    return [(identity, node) for identity, node in machine["heap"].items() if "asyncCall" in node]


def _completion(value, validate):
    from .protocol_v8 import _exact

    _exact(value, {"kind", "value"}, "async completion")
    _require(value["kind"] in ("return", "throw"), "completion kind")
    validate(value["value"])


def validate_record(record):
    from .protocol_v8 import _exact, validate_protocol_value

    fields = FIELDS.get(record.get("kind")) if type(record) is dict else None
    if fields is None:
        return False
    _exact(record, fields, "async call record")
    for field in ("object", "promise", "child", "invocation", "awaited", "continuation"):
        if field in record:
            _id(record[field])
    if record["kind"] == "async.call":
        validate_protocol_value(record["callee"])
        _require(record["callee"][0] == "ref", "owned async callable required")
        validate_protocol_value(record["receiver"])
        _require(
            type(record["arguments"]) is list and len(record["arguments"]) <= BOUND,
            "argument bound",
        )
        for argument in record["arguments"]:
            validate_protocol_value(argument)
    elif record["kind"] == "async.resume":
        _require(record["status"] in ("fulfilled", "rejected"), "resume settlement")
        validate_protocol_value(record["value"])
    elif record["kind"] == "async.resolve.settle":
        _require(record["status"] in ("fulfilled", "rejected"), "resolution settlement")
        validate_protocol_value(record["value"])
    elif record["kind"] == "async.complete":
        _completion(record["completion"], validate_protocol_value)
    return True


def _execution(machine, identity):
    node = promise(machine, identity)
    _require(node.get("external") is None, "owned internal execution Promise required")
    return node


def _scope(machine):
    return {
        "receiver": copy.deepcopy(machine.get("receiver", ["undefined"])),
        "aliases": [list(pair) for pair in machine.get("captureAliases", {}).items()],
    }


def _restore_scope(machine, scope):
    machine["receiver"] = copy.deepcopy(scope["receiver"])
    machine["captureAliases"] = dict(scope["aliases"])


def _detach(machine, call):
    stack = machine["stack"]
    _require(stack and stack[-1] == call["invocation"], "async body is not active")
    if call["attached"]:
        frames = machine["protocol"]["callFrames"]
        frame = frames[-1] if frames else None
        _require(
            frame is not None
            and frame["child"] == call["invocation"]
            and frame["parent"] == call["parent"]
            and len(stack) > 1
            and stack[-2] == call["parent"],
            "async first suspension lost its caller scope",
        )
        frames.pop()
        stack.pop()
        _restore_scope(
            machine, {"receiver": frame["priorReceiver"], "aliases": frame["priorAliases"]}
        )
    else:
        _require(
            len(stack) == 1
            and machine.get("activeJob") is None
            and call["resumeScope"] is not None,
            "detached async turn has a foreign caller or job",
        )
        stack.pop()
        _restore_scope(machine, call["resumeScope"])
        call["resumeScope"] = None


def apply_record(machine, record):
    from .protocol_v8 import _apply, _known

    if not validate_record(record):
        return False
    kind = record["kind"]
    if kind == "async.function":
        fn = machine["heap"].get(record["object"])
        _require(
            fn is not None
            and fn.get("kind") == "closure"
            and not any(
                fn.get(field)
                for field in ("asyncFunction", "generatorFunction", "promiseResolver", "callable")
            ),
            "async definition requires an undeclared source closure",
        )
        fn["asyncFunction"] = True
        return True
    if kind == "async.call":
        _known(machine, record["callee"])
        _known(machine, record["receiver"])
        for argument in record["arguments"]:
            _known(machine, argument)
        fn = machine["heap"].get(record["callee"][1])
        target = _execution(machine, record["promise"])
        _require(
            fn is not None and fn.get("kind") == "closure" and fn.get("asyncFunction") is True,
            "declared async source callable required",
        )
        capabilities = (machine["protocol"].get("promiseExecutors") or {}).get("capabilities", [])
        _require(
            target["state"] == "pending"
            and target.get("adopted") is None
            and not target.get("join")
            and "asyncCall" not in target
            and not any(row["promise"] == record["promise"] for row in capabilities),
            "async call requires a fresh unclaimed execution Promise",
        )
        _require(
            len(_calls(machine)) < BOUND and bool(machine["stack"]),
            "async call bound or missing parent",
        )
        parent = machine["stack"][-1]
        machine["_internalProtocolTransition"] = True
        try:
            _apply(
                machine,
                {
                    "kind": "call",
                    "child": record["child"],
                    "callee": record["callee"],
                    "receiver": record["receiver"],
                    "arguments": record["arguments"],
                },
            )
        finally:
            machine.pop("_internalProtocolTransition", None)
        target["asyncCall"] = {
            "schema": STATE_SCHEMA,
            "owner": _owner(machine),
            "invocation": record["child"],
            "callee": copy.deepcopy(record["callee"]),
            "receiver": copy.deepcopy(machine["receiver"]),
            "arguments": copy.deepcopy(record["arguments"]),
            "parent": parent,
            "phase": "running",
            "attached": True,
            "continuations": [],
            "resumeScope": None,
            "completion": None,
            "resolution": None,
        }
        return True
    target = _execution(machine, record["promise"])
    call = target.get("asyncCall")
    _require(
        type(call) is dict
        and call.get("schema") == STATE_SCHEMA
        and call.get("owner") == _owner(machine)
        and call.get("invocation") == record["invocation"],
        "execution Promise does not own this async invocation",
    )
    continuations = rows(machine.get("continuations", []))
    if kind == "async.await":
        _require(
            call["phase"] == "running"
            and target["state"] == "pending"
            and machine["stack"]
            and machine["stack"][-1] == call["invocation"],
            "await requires the running unsettled async invocation",
        )
        promise(machine, record["awaited"])
        identity = record["continuation"]
        _require(
            identity not in continuations and identity not in rows(machine.get("jobs", [])),
            "continuation identity is not fresh",
        )
        _require(len(call["continuations"]) < BOUND, "async await bound")
        saved, attached = _scope(machine), call["attached"]
        _detach(machine, call)
        machine["scheduler"]["registration"] += 1
        continuations[identity] = {
            "state": "waiting",
            "promise": record["awaited"],
            "invocation": call["invocation"],
            "sequence": machine["scheduler"]["registration"],
            "activeJob": None,
            "captureAliases": saved["aliases"],
            "receiver": saved["receiver"],
        }
        store_rows(machine, "continuations", continuations)
        call["attached"], call["phase"] = False, "suspended"
        call["continuations"].append(identity)
        if attached:
            machine["lastReturn"] = {"kind": "return", "value": ["ref", record["promise"]]}
        refresh_ready(machine)
        return True
    if kind == "async.resume":
        identity = record["continuation"]
        continuation = continuations.get(identity)
        _require(
            call["phase"] == "suspended"
            and not call["attached"]
            and call["continuations"][-1:] == [identity]
            and continuation is not None
            and continuation["invocation"] == call["invocation"]
            and continuation["state"] == "ready"
            and continuation["activeJob"] is None,
            "resume requires this async call's latest ready continuation",
        )
        _require(
            not machine["stack"]
            and machine.get("activeJob") is None
            and not machine["protocol"]["callFrames"]
            and not machine["protocol"]["pendingCalls"]
            and machine["ready"][:1] == [identity],
            "async resume violates FIFO or active turn ownership",
        )
        awaited = promise(machine, continuation["promise"])
        _require(
            awaited["state"] == record["status"] and awaited["result"] == record["value"],
            "async resume contradicts settled dependency",
        )
        call["resumeScope"] = _scope(machine)
        machine["ready"].pop(0)
        continuation["state"] = "resumed"
        store_rows(machine, "continuations", continuations)
        machine["stack"].append(call["invocation"])
        _restore_scope(
            machine,
            {"receiver": continuation["receiver"], "aliases": continuation["captureAliases"]},
        )
        call["phase"] = "running"
        return True
    if kind == "async.resolve.enqueue":
        resolution = call.get("resolution")
        target = machine["heap"].get(record["target"])
        _require(
            call["phase"] == "completed"
            and resolution is not None
            and resolution.get("schema") == RESOLUTION_SCHEMA
            and resolution.get("phase") == "enqueue-pending"
            and resolution.get("target") == record["target"]
            and call["completion"]["kind"] == "return"
            and call["completion"]["value"] == ["ref", record["target"]]
            and target is not None
            and target.get("kind") == "promise"
            and target.get("external") is None
            and record["target"] != record["promise"],
            "Promise-resolution enqueue differs from the exact completed async return",
        )
        _require(
            not _resolution_cycle(machine, record["promise"], record["target"]),
            "Promise-resolution dependency cycle",
        )
        jobs = rows(machine.get("jobs", []))
        continuations = rows(machine.get("continuations", []))
        _require(
            record["job"] != record["settlementJob"]
            and record["job"] not in jobs
            and record["settlementJob"] not in jobs
            and record["job"] not in continuations
            and record["settlementJob"] not in continuations,
            "Promise-resolution job identities must be fresh",
        )
        scheduler = machine.setdefault("scheduler", {"registration": 0, "ready": 0})
        resolution["assimilationJob"] = record["job"]
        resolution["settlementJob"] = record["settlementJob"]
        resolution["phase"] = "queued"
        scheduler["ready"] += 1
        jobs[record["job"]] = {
            "status": "queued",
            "owner": call["owner"],
            "promise": None,
            "reaction": "async-resolution-enqueue",
            "closure": None,
            "rejectionClosure": None,
            "target": record["promise"],
            "sequence": None,
            "readySequence": scheduler["ready"],
            "asyncResolution": {
                "promise": record["promise"],
                "target": record["target"],
                "phase": "enqueue",
            },
        }
        jobs[record["settlementJob"]] = {
            "status": "reserved",
            "owner": call["owner"],
            "promise": None,
            "reaction": "async-resolution-reserved",
            "closure": None,
            "rejectionClosure": None,
            "target": record["promise"],
            "sequence": None,
            "asyncResolution": {
                "promise": record["promise"],
                "target": record["target"],
                "phase": "reserved",
            },
        }
        machine.setdefault("ready", []).append(record["job"])
        store_rows(machine, "jobs", jobs)
        return True
    if kind == "async.resolve.subscribe":
        resolution = call.get("resolution")
        jobs = rows(machine.get("jobs", []))
        job = jobs.get(record["job"])
        reserved = jobs.get(record["settlementJob"])
        target = machine["heap"].get(record["target"])
        _require(
            isinstance(reserved, dict) and set(reserved) == RESERVED_RESOLUTION_JOB_FIELDS,
            "reserved async settlement job fields differ",
        )
        _require(
            call["phase"] == "completed"
            and resolution is not None
            and resolution.get("schema") == RESOLUTION_SCHEMA
            and resolution.get("phase") == "queued"
            and resolution.get("target") == record["target"]
            and resolution.get("assimilationJob") == record["job"]
            and resolution.get("settlementJob") == record["settlementJob"]
            and record["job"] != record["settlementJob"]
            and call["completion"]["kind"] == "return"
            and call["completion"]["value"] == ["ref", record["target"]]
            and job is not None
            and job["owner"] == call["owner"]
            and job["status"] == "queued"
            and job["reaction"] == "async-resolution-enqueue"
            and job.get("asyncResolution")
            == {"promise": record["promise"], "target": record["target"], "phase": "enqueue"}
            and reserved is not None
            and reserved.get("owner") == call["owner"]
            and reserved.get("status") == "reserved"
            and reserved.get("promise") is None
            and reserved.get("reaction") == "async-resolution-reserved"
            and reserved.get("closure") is None
            and reserved.get("rejectionClosure") is None
            and reserved.get("target") == record["promise"]
            and reserved.get("sequence") is None
            and reserved.get("asyncResolution")
            == {"promise": record["promise"], "target": record["target"], "phase": "reserved"}
            and machine["ready"][:1] == [record["job"]]
            and not machine["stack"]
            and machine.get("activeJob") is None
            and target is not None
            and target.get("kind") == "promise"
            and target.get("external") is None,
            "Promise-resolution subscription requires its exact FIFO enqueue job",
        )
        machine["ready"].pop(0)
        job["status"] = "completed"
        resolution["phase"] = "subscribed"
        scheduler = machine.setdefault("scheduler", {"registration": 0, "ready": 0})
        scheduler["registration"] += 1
        jobs[record["settlementJob"]] = {
            "status": "waiting",
            "owner": call["owner"],
            "promise": record["target"],
            "reaction": "async-resolution-settle",
            "closure": None,
            "rejectionClosure": None,
            "target": record["promise"],
            "sequence": scheduler["registration"],
            "asyncResolution": {
                "promise": record["promise"],
                "target": record["target"],
                "phase": "settle",
            },
        }
        store_rows(machine, "jobs", jobs)
        refresh_ready(machine)
        return True
    if kind == "async.resolve.settle":
        resolution = call.get("resolution")
        jobs = rows(machine.get("jobs", []))
        job = jobs.get(record["job"])
        target = machine["heap"].get(record["target"])
        outer = _execution(machine, record["promise"])
        _require(
            call["phase"] == "completed"
            and resolution is not None
            and resolution.get("schema") == RESOLUTION_SCHEMA
            and resolution.get("phase") == "subscribed"
            and resolution.get("target") == record["target"]
            and resolution.get("settlementJob") == record["job"]
            and resolution.get("assimilationJob") != record["job"]
            and job is not None
            and job["owner"] == call["owner"]
            and job["status"] == "queued"
            and job["reaction"] == "async-resolution-settle"
            and job["promise"] == record["target"]
            and job["target"] == record["promise"]
            and job.get("asyncResolution")
            == {"promise": record["promise"], "target": record["target"], "phase": "settle"}
            and machine["ready"][:1] == [record["job"]]
            and not machine["stack"]
            and machine.get("activeJob") is None
            and target is not None
            and target.get("kind") == "promise"
            and target.get("external") is None
            and target["state"] in ("fulfilled", "rejected")
            and target["state"] == record["status"]
            and target["result"] == record["value"]
            and outer["state"] == "pending"
            and outer.get("adopted") is None,
            "Promise-resolution settlement requires its exact subscribed FIFO observer",
        )
        if record["status"] == "fulfilled":
            assert_plain_resolution(machine, record["value"])
        machine["ready"].pop(0)
        job["status"] = "completed"
        resolution["phase"] = "settled"
        store_rows(machine, "jobs", jobs)
        settle_promise(machine, record["promise"], record["status"], record["value"])
        return True
    _require(
        call["phase"] == "running"
        and target["state"] == "pending"
        and machine["stack"]
        and machine["stack"][-1] == call["invocation"],
        "completion requires the running unsettled async invocation",
    )
    _known(machine, record["completion"]["value"])
    returned = record["completion"]["value"]
    adoption = (
        record["completion"]["kind"] == "return"
        and returned[0] == "ref"
        and machine["heap"].get(returned[1], {}).get("kind") == "promise"
    )
    if record["completion"]["kind"] == "return" and not adoption:
        assert_plain_resolution(machine, returned)
    if adoption:
        target = machine["heap"][returned[1]]
        _require(
            target.get("external") is None
            and returned[1] != record["promise"]
            and not _resolution_cycle(machine, record["promise"], returned[1]),
            "async Promise return requires a distinct owned target without a resolution cycle",
        )
    attached = call["attached"]
    _detach(machine, call)
    call["phase"], call["completion"] = "completed", copy.deepcopy(record["completion"])
    if adoption:
        call["resolution"] = {
            "schema": RESOLUTION_SCHEMA,
            "target": returned[1],
            "assimilationJob": None,
            "settlementJob": None,
            "phase": "enqueue-pending",
        }
    else:
        call["resolution"] = None
        settle_promise(
            machine,
            record["promise"],
            "fulfilled" if record["completion"]["kind"] == "return" else "rejected",
            record["completion"]["value"],
        )
    if attached:
        machine["lastReturn"] = {"kind": "return", "value": ["ref", record["promise"]]}
    return True


def assert_next(machine, record):
    calls = _calls(machine)
    kind = record["kind"]
    active = machine.get("stack", [])[-1:]
    if any(
        node["asyncCall"]["phase"] == "running" and active == [node["asyncCall"]["invocation"]]
        for _, node in calls
    ):
        _require(
            kind not in ("return", "throw", "await"),
            "async body requires its explicit continuation protocol",
        )
    fields = record.get("fields", record)
    if kind == "job":
        jobs = rows(machine.get("jobs", []))
        if jobs.get(fields.get("job"), {}).get("asyncResolution"):
            _fail("async Promise-resolution jobs require their explicit FIFO protocol")
    if kind == "async.complete" and fields.get("completion", {}).get("kind") == "return":
        value = fields["completion"].get("value", [])
        if value[:1] == ["ref"] and machine["heap"].get(value[1], {}).get("kind") == "promise":
            call = machine["heap"].get(fields.get("promise"), {}).get("asyncCall", {})
            _require(
                call.get("phase") == "running" and active == [call.get("invocation")],
                "async Promise return requires its exact active body",
            )
    if (
        any(
            (node["asyncCall"].get("resolution") or {}).get("phase") == "enqueue-pending"
            for _, node in calls
        )
        and kind != "async.resolve.enqueue"
    ):
        _fail("completed Promise return requires its explicit assimilation enqueue")
    if kind == "resume":
        _require(
            not any(
                record["continuation"] in node["asyncCall"]["continuations"] for _, node in calls
            ),
            "async continuation requires its explicit resume protocol",
        )
    if kind == "end":
        _require(
            not any(
                node["asyncCall"]["owner"] == _owner(machine)
                and node["asyncCall"]["phase"] != "completed"
                for _, node in calls
            ),
            "unfinished async call",
        )
    identity = (
        record.get("promise")
        if kind in ("promise", "promise.executor.enter")
        else record.get("result")
        if kind == "promise.join"
        else None
    )
    if identity is not None and (
        kind != "promise" or record.get("action") in ("resolve", "reject", "adopt")
    ):
        _require(
            "asyncCall" not in machine["heap"].get(identity, {}),
            "async execution Promise has a separate owner",
        )
    return True


def validate_state(machine):
    from .protocol_v8 import _exact, _known

    invocations, seen_continuations = set(), set()
    calls = _calls(machine)
    _require(len(calls) <= BOUND, "restored async call bound")
    for fn in machine["heap"].values():
        if "asyncFunction" in fn:
            _require(
                fn["asyncFunction"] is True
                and fn.get("kind") == "closure"
                and not any(
                    fn.get(field) for field in ("generatorFunction", "promiseResolver", "callable")
                ),
                "restored async callable capability",
            )
    continuations = rows(machine.get("continuations", []))
    frames = rows(machine.get("frameFunctions", {}))
    for identity, node in calls:
        call = node["asyncCall"]
        _exact(call, STATE_FIELDS, "retained async call")
        for field in ("owner", "invocation", "parent"):
            _id(call[field])
        _require(
            call["schema"] == STATE_SCHEMA
            and node.get("kind") == "promise"
            and node.get("external") is None
            and node.get("adopted") is None
            and not node.get("join")
            and type(call["attached"]) is bool
            and call["phase"] in ("running", "suspended", "completed"),
            "retained async call shape",
        )
        _known(machine, call["callee"])
        _known(machine, call["receiver"])
        fn = machine["heap"].get(call["callee"][1]) if call["callee"][0] == "ref" else None
        invocation = (call["owner"], call["invocation"])
        _require(
            fn is not None and fn.get("asyncFunction") is True and invocation not in invocations,
            "retained async callable or invocation identity",
        )
        invocations.add(invocation)
        _require(
            type(call["arguments"]) is list and len(call["arguments"]) <= BOUND,
            "retained async argument bound",
        )
        for value in call["arguments"]:
            _known(machine, value)
        _require(
            type(call["continuations"]) is list
            and len(call["continuations"]) <= BOUND
            and (not call["attached"] or not call["continuations"]),
            "retained async attachment",
        )
        for continuation_id in call["continuations"]:
            _id(continuation_id)
        _require(
            len(set(call["continuations"])) == len(call["continuations"]),
            "duplicate retained async continuation",
        )
        if call["phase"] == "completed":
            _completion(call["completion"], lambda value: _known(machine, value))
            _validate_resolution_state(machine, identity, node, call)
        else:
            _require(
                node["state"] == "pending"
                and call["completion"] is None
                and call["resolution"] is None,
                "unfinished async execution Promise is settled",
            )
        if call["owner"] != _owner(machine):
            continue
        _require(
            frames.get(call["invocation"]) == fn["code"] and call["parent"] in frames,
            "retained async source frame identity",
        )
        for index, continuation_id in enumerate(call["continuations"]):
            _id(continuation_id)
            continuation = continuations.get(continuation_id)
            _require(
                continuation_id not in seen_continuations
                and continuation is not None
                and continuation["invocation"] == call["invocation"]
                and continuation["activeJob"] is None,
                "retained async continuation ownership",
            )
            seen_continuations.add(continuation_id)
            promise(machine, continuation["promise"])
            _require(
                continuation["receiver"] == call["receiver"]
                and continuation["captureAliases"]
                == [[row["binding"], row["cell"]] for row in fn["captures"]],
                "retained async continuation receiver or captured cells",
            )
            suspended = index == len(call["continuations"]) - 1 and call["phase"] == "suspended"
            _require(
                continuation["state"] in ("waiting", "ready")
                if suspended
                else continuation["state"] == "resumed",
                "retained async continuation phase",
            )
        frame = next(
            (
                row
                for row in machine["protocol"]["callFrames"]
                if row["child"] == call["invocation"]
            ),
            None,
        )
        if call["phase"] == "running":
            _require(
                node["state"] == "pending"
                and call["completion"] is None
                and call["invocation"] in machine["stack"],
                "retained running async body",
            )
            _require(
                (
                    frame is not None
                    and frame["parent"] == call["parent"]
                    and call["resumeScope"] is None
                )
                if call["attached"]
                else (
                    frame is None
                    and machine["stack"][:1] == [call["invocation"]]
                    and call["resumeScope"] is not None
                    and machine.get("activeJob") is None
                ),
                "retained async call scope",
            )
            if machine["stack"][-1:] == [call["invocation"]]:
                _require(
                    machine.get("receiver") == call["receiver"]
                    and list(machine.get("captureAliases", {}).items())
                    == [(row["binding"], row["cell"]) for row in fn["captures"]],
                    "retained async receiver or live capture aliases",
                )
        else:
            _require(
                call["invocation"] not in machine["stack"]
                and frame is None
                and call["resumeScope"] is None,
                "inactive async frame retains a live scope",
            )
            if call["phase"] == "suspended":
                _require(
                    not call["attached"]
                    and bool(call["continuations"])
                    and node["state"] == "pending"
                    and call["completion"] is None,
                    "retained suspended async body",
                )
            else:
                _completion(call["completion"], lambda value: _known(machine, value))
                _validate_resolution_state(machine, identity, node, call)
        if call["resumeScope"] is not None:
            saved = call["resumeScope"]
            _exact(saved, {"receiver", "aliases"}, "async idle scope")
            _known(machine, saved["receiver"])
            _require(type(saved["aliases"]) is list, "async idle aliases")
            aliases = set()
            for pair in saved["aliases"]:
                _require(
                    type(pair) is list
                    and len(pair) == 2
                    and all(isinstance(value, str) for value in pair)
                    and pair[0] not in aliases
                    and pair[1] in machine["locals"],
                    "retained async idle alias",
                )
                aliases.add(pair[0])
    jobs = rows(machine.get("jobs", []))
    for job_id, job in jobs.items():
        link = job.get("asyncResolution")
        if link is None:
            continue
        resolution = (
            machine["heap"].get(link.get("promise"), {}).get("asyncCall", {}).get("resolution")
        )
        _require(
            resolution is not None
            and job_id in (resolution.get("assimilationJob"), resolution.get("settlementJob")),
            "orphan Promise-resolution job",
        )
    return True


def _resolution_cycle(machine, promise_id, target_id):
    heap = machine["heap"]
    seen = set()
    cursor = target_id
    while cursor and cursor not in seen:
        if cursor == promise_id:
            return True
        seen.add(cursor)
        resolution = heap.get(cursor, {}).get("asyncCall", {}).get("resolution")
        cursor = (
            resolution.get("target")
            if resolution and resolution.get("phase") != "settled"
            else None
        )
    return False


def _validate_resolution_state(machine, identity, node, call):
    from .promise_scheduler_v8 import assert_plain_resolution
    from .protocol_v8 import _exact

    completion = call["completion"]
    value = completion["value"]
    target_id = (
        value[1]
        if completion["kind"] == "return"
        and value[:1] == ["ref"]
        and machine["heap"].get(value[1], {}).get("kind") == "promise"
        else None
    )
    resolution = call.get("resolution")
    if target_id is None:
        _require(
            resolution is None
            and node["state"] == ("fulfilled" if completion["kind"] == "return" else "rejected")
            and node["result"] == value,
            "retained async completion differs from execution Promise",
        )
        return
    _exact(
        resolution,
        {"schema", "target", "assimilationJob", "settlementJob", "phase"},
        "retained async Promise resolution",
    )
    _require(
        resolution["schema"] == RESOLUTION_SCHEMA
        and resolution["target"] == target_id
        and target_id != identity
        and resolution["phase"] in ("enqueue-pending", "queued", "subscribed", "settled")
        and machine["heap"].get(target_id, {}).get("external") is None
        and not _resolution_cycle(machine, identity, target_id),
        "retained async Promise-resolution identity or phase differs",
    )
    jobs = rows(machine.get("jobs", []))
    ready = machine.get("ready", [])
    if resolution["phase"] == "enqueue-pending":
        _require(
            resolution["assimilationJob"] is None
            and resolution["settlementJob"] is None
            and node["state"] == "pending"
            and node["result"] is None,
            "pending async Promise-resolution enqueue changed its Promise",
        )
        return
    assimilation_id = resolution["assimilationJob"]
    settlement_id = resolution["settlementJob"]
    assimilation = jobs.get(assimilation_id)
    _require(
        isinstance(assimilation_id, str)
        and isinstance(settlement_id, str)
        and assimilation_id != settlement_id
        and assimilation is not None
        and assimilation.get("owner") == call["owner"]
        and assimilation.get("reaction") == "async-resolution-enqueue"
        and assimilation.get("target") == identity
        and assimilation.get("asyncResolution")
        == {"promise": identity, "target": target_id, "phase": "enqueue"},
        "retained async assimilation job differs",
    )
    if resolution["phase"] == "queued":
        settlement = jobs.get(settlement_id)
        _require(
            isinstance(settlement, dict) and set(settlement) == RESERVED_RESOLUTION_JOB_FIELDS,
            "queued async settlement reservation fields differ",
        )
        _require(
            assimilation.get("status") == "queued"
            and assimilation_id in ready
            and settlement is not None
            and settlement.get("owner") == call["owner"]
            and settlement.get("status") == "reserved"
            and settlement.get("promise") is None
            and settlement.get("reaction") == "async-resolution-reserved"
            and settlement.get("closure") is None
            and settlement.get("rejectionClosure") is None
            and settlement.get("target") == identity
            and settlement.get("sequence") is None
            and settlement.get("asyncResolution")
            == {"promise": identity, "target": target_id, "phase": "reserved"}
            and settlement_id not in ready
            and node["state"] == "pending"
            and node["result"] is None,
            "queued async assimilation job or unresolved Promise differs",
        )
        return
    _require(
        assimilation.get("status") == "completed" and assimilation_id not in ready,
        "retained async assimilation job is not completed",
    )
    settlement = jobs.get(settlement_id)
    target = machine["heap"].get(target_id)
    _require(
        settlement is not None
        and settlement.get("owner") == call["owner"]
        and settlement.get("reaction") == "async-resolution-settle"
        and settlement.get("promise") == target_id
        and settlement.get("target") == identity
        and settlement.get("asyncResolution")
        == {"promise": identity, "target": target_id, "phase": "settle"},
        "retained async settlement observer differs",
    )
    if resolution["phase"] == "subscribed":
        _require(
            node["state"] == "pending"
            and node["result"] is None
            and (
                settlement.get("status") == "waiting" and settlement_id not in ready
                if target["state"] == "pending"
                else settlement.get("status") == "queued" and settlement_id in ready
            ),
            "retained async settlement observer disagrees with its target",
        )
        return
    _require(
        target["state"] != "pending"
        and settlement.get("status") == "completed"
        and settlement_id not in ready
        and node["state"] == target["state"]
        and node["result"] == target["result"],
        "retained async Promise settlement differs from its exact target",
    )
    if target["state"] == "fulfilled":
        assert_plain_resolution(machine, target["result"])
