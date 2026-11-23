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
}
FIELDS = {
    "async.function": {"kind", "object"},
    "async.call": {"kind", "promise", "child", "callee", "receiver", "arguments"},
    "async.await": {"kind", "promise", "invocation", "awaited", "continuation"},
    "async.resume": {"kind", "promise", "invocation", "continuation", "status", "value"},
    "async.complete": {"kind", "promise", "invocation", "completion"},
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
        "pending": copy.deepcopy(machine.get("pending")),
    }


def _restore_scope(machine, scope):
    machine["receiver"] = copy.deepcopy(scope["receiver"])
    machine["captureAliases"] = dict(scope["aliases"])
    machine["pending"] = copy.deepcopy(scope["pending"])


def _own_pending(machine, invocation):
    pending = machine.get("pending")
    handler = pending.get("handler") if isinstance(pending, dict) else None
    if not isinstance(handler, dict) or handler.get("schema") != "semantic-handler-scopes/1":
        return copy.deepcopy(pending)
    frames = [frame for frame in handler["frames"] if frame["invocation"] == invocation]
    return (
        {
            "kind": "cleanup",
            "value": ["undefined"],
            "handler": {"schema": handler["schema"], "frames": copy.deepcopy(frames)},
        }
        if frames
        else None
    )


def _detach(machine, call):
    stack = machine["stack"]
    _require(
        stack and stack[-1] == call["invocation"] and call["resumeScope"] is not None,
        "async body is not active or lost its caller scope",
    )
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
        prior = _scope(machine)
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
            "resumeScope": prior,
            "completion": None,
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
        pending = _own_pending(machine, call["invocation"])
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
            "pending": pending,
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
            {
                "receiver": continuation["receiver"],
                "aliases": continuation["captureAliases"],
                "pending": continuation["pending"],
            },
        )
        call["phase"] = "running"
        return True
    _require(
        call["phase"] == "running"
        and target["state"] == "pending"
        and machine["stack"]
        and machine["stack"][-1] == call["invocation"],
        "completion requires the running unsettled async invocation",
    )
    _known(machine, record["completion"]["value"])
    adoption_target = None
    if record["completion"]["kind"] == "return":
        value = record["completion"]["value"]
        if value[0] == "ref" and machine["heap"].get(value[1], {}).get("kind") == "promise":
            adoption_target = value[1]
        else:
            assert_plain_resolution(machine, value)
    attached = call["attached"]
    _detach(machine, call)
    call["phase"], call["completion"] = "completed", copy.deepcopy(record["completion"])
    if adoption_target is not None:
        from .promise_adoption_v8 import stage_intent

        stage_intent(
            machine,
            record["promise"],
            adoption_target,
            {"kind": "async", "invocation": record["invocation"]},
        )
    else:
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
    from .protocol_v8 import _exact, _known, validate_pending_scope

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
    for _identity, node in calls:
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
            result = call["completion"]["value"]
            adoption = node.get("adoption") or {}
            target_is_promise = (
                call["completion"]["kind"] == "return"
                and result[0] == "ref"
                and machine["heap"].get(result[1], {}).get("kind") == "promise"
            )
            if target_is_promise:
                _require(
                    adoption.get("target") == result[1]
                    and adoption.get("cause") == {"kind": "async", "invocation": call["invocation"]}
                    and adoption.get("phase")
                    in ("intent", "queued", "reading", "observing", "completed", "rejected", "self"),
                    "retained async Promise result lacks adoption authority",
                )
                if adoption.get("phase") == "completed":
                    target = machine["heap"][result[1]]
                    _require(
                        node["state"] == target["state"] and node["result"] == target["result"],
                        "retained async adoption differs from target settlement",
                    )
                elif adoption.get("phase") == "self":
                    _require(
                        result[1] == _identity
                        and node["state"] == "rejected"
                        and node["result"] == ["ref", adoption.get("error")],
                        "retained async self-resolution differs from its TypeError",
                    )
                elif adoption.get("phase") == "rejected":
                    job_rows = machine.get("jobs", [])
                    returned = (dict(job_rows) if isinstance(job_rows, list) else job_rows).get(
                        adoption.get("job"), {}
                    ).get("returned", {})
                    _require(
                        returned.get("kind") == "throw"
                        and node["state"] == "rejected"
                        and node["result"] == returned.get("value"),
                        "retained async adoption rejection differs from its getter",
                    )
                else:
                    _require(node["state"] == "pending", "pending async adoption settled early")
            else:
                _require(
                    node["state"]
                    == ("fulfilled" if call["completion"]["kind"] == "return" else "rejected")
                    and node["result"] == result,
                    "retained async completion differs from execution Promise",
                )
        else:
            _require(
                node["state"] == "pending" and call["completion"] is None,
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
            validate_pending_scope(machine, continuation.get("pending", {}), [call["invocation"]])
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
                    and call["resumeScope"] is not None
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
                value = call["completion"]["value"]
                adoption = node.get("adoption") or {}
                adopts_promise = (
                    call["completion"]["kind"] == "return"
                    and value[0] == "ref"
                    and machine["heap"].get(value[1], {}).get("kind") == "promise"
                )
                if adopts_promise:
                    _require(
                        adoption.get("target") == value[1]
                        and adoption.get("cause")
                        == {"kind": "async", "invocation": call["invocation"]}
                        and adoption.get("phase")
                        in ("intent", "queued", "reading", "observing", "completed", "rejected", "self"),
                        "retained async completion lacks adoption authority",
                    )
                    if adoption.get("phase") == "completed":
                        target = machine["heap"][value[1]]
                        _require(
                            node["state"] == target["state"] and node["result"] == target["result"],
                            "retained async adoption differs from target settlement",
                        )
                    elif adoption.get("phase") == "self":
                        _require(
                            value[1] == _identity
                            and node["state"] == "rejected"
                            and node["result"] == ["ref", adoption.get("error")],
                            "retained async self-resolution differs from its TypeError",
                        )
                    elif adoption.get("phase") == "rejected":
                        job_rows = machine.get("jobs", [])
                        returned = (dict(job_rows) if isinstance(job_rows, list) else job_rows).get(
                            adoption.get("job"), {}
                        ).get("returned", {})
                        _require(
                            returned.get("kind") == "throw"
                            and node["state"] == "rejected"
                            and node["result"] == returned.get("value"),
                            "retained async adoption rejection differs from its getter",
                        )
                    else:
                        _require(
                            node["state"] == "pending", "async adoption settled before forwarding"
                        )
                else:
                    _require(
                        node["state"]
                        == ("fulfilled" if call["completion"]["kind"] == "return" else "rejected")
                        and node["result"] == value,
                        "retained async completion differs from execution Promise",
                    )
        if call["resumeScope"] is not None:
            saved = call["resumeScope"]
            _exact(saved, {"receiver", "aliases", "pending"}, "async idle scope")
            _known(machine, saved["receiver"])
            callers = (
                machine["stack"][: machine["stack"].index(call["invocation"])]
                if call["attached"]
                else []
            )
            validate_pending_scope(machine, saved["pending"], callers)
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
    return True
