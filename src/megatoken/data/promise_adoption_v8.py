"""FIFO adoption jobs for owned intrinsic Promises."""

from __future__ import annotations

import copy

PROFILE = "promise-adoption/1"
STATE_SCHEMA = "semantic-promise-adoption-state/1"
FIELDS = {
    "promise.adopt.enqueue": {"kind", "promise", "target", "job"},
    "promise.adopt.enter": {"kind", "promise", "target", "job", "forwarding"},
    "promise.adopt.forward": {"kind", "promise", "target", "job", "status", "value"},
    "promise.adopt.self": {"kind", "promise", "error"},
    "promise.reaction.complete": {"kind", "job", "owner", "promise", "target"},
}
BOUND = 4096


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid owned Promise adoption: " + message)


def _rows(value):
    return dict(value) if isinstance(value, list) else value


def _id(value):
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("bounded identity required")


def _promise(machine, identity):
    node = machine["heap"].get(identity)
    if not node or node.get("kind") != "promise" or node.get("external") is not None:
        _fail("owned internal Promise required")
    return node


def _assert_intrinsic_then(machine, identity):
    node = _promise(machine, identity)
    properties = node.get("properties", {})
    descriptors = node.get("descriptors", {})
    if (
        properties
        or descriptors
        or node.get("prototype", "ordinary") not in (None, "ordinary")
        or node.get("extensible") is False
        or node.get("callable") is not None
        or node.get("proxy") is not None
    ):
        _fail("Promise then lookup is overridden or has an unmodeled prototype")


def _adoption(machine, identity):
    return _promise(machine, identity).get("adoption")


def stage_intent(machine, adopter, target, cause):
    """Bind adoption to the exact completion event before the next record."""
    from .protocol_v8 import _known

    dest = _promise(machine, adopter)
    _promise(machine, target)
    _assert_intrinsic_then(machine, target)
    if (
        dest["state"] != "pending"
        or dest["result"] is not None
        or dest.get("adopted") is not None
        or dest.get("join")
        or _adoption(machine, adopter) is not None
    ):
        _fail("adoption requires a pending adopter")
    if sum(1 for node in machine["heap"].values() if node.get("kind") == "promise" and node.get("adoption") is not None) >= BOUND:
        _fail("Promise adoption bound")
    owner = next(iter(machine.get("frameFunctions", {})), None)
    _id(owner)
    row = {
        "schema": STATE_SCHEMA,
        "owner": owner,
        "target": target,
        "phase": "intent",
        "cause": copy.deepcopy(cause),
        "job": None,
        "forwarding": None,
        "error": None,
    }
    dest["adoption"] = row
    _known(machine, ["ref", target])


def stage_source_intent(machine, adopter, target):
    """Only for an otherwise unclaimed fresh internal Promise result."""
    node = _promise(machine, adopter)
    if node.get("asyncCall") or node.get("join") or (node.get("adoption") or {}).get("phase"):
        _fail("source adoption lacks a fresh internal Promise")
    stage_intent(machine, adopter, target, {"kind": "source"})


def _fresh_job(machine, identity):
    from .protocol_v8 import _assert_fresh_heap_identity

    _assert_fresh_heap_identity(machine, identity)
    jobs = _rows(machine.get("jobs", []))
    continuations = _rows(machine.get("continuations", []))
    if identity in jobs or identity in continuations or identity in machine["heap"]:
        _fail("job identity is not fresh")


def _turn(machine, identity, job, owner):
    root = next(iter(machine.get("frameFunctions", {})), None)
    if (
        machine.get("stack") or machine.get("activeJob") is not None
        or any(row.get("status") == "running" for row in _rows(machine.get("jobs", [])).values())
        or machine.get("ready", [])[:1] != [identity]
        or not job or job.get("status") != "queued" or job.get("owner") != owner
        or owner != root
    ):
        _fail("adoption step violates FIFO or active turn ownership")


def _check_cause(machine, adopter, row, target):
    cause = row.get("cause")
    if type(cause) is not dict or not isinstance(cause.get("kind"), str):
        _fail("adoption lacks a source-bound resolution cause")
    kind = cause["kind"]
    if kind == "resolver":
        if set(cause) != {"kind", "resolver"}:
            _fail("resolver cause fields")
        data = machine["protocol"].get("promiseExecutors") or {}
        cap = next((c for c in data.get("capabilities", []) if c.get("promise") == adopter), None)
        if not cap or cap.get("resolve") != cause["resolver"] or not cap.get("called"):
            _fail("resolver cause is not the exact resolving capability")
        resolution = cap.get("resolution") or {}
        if resolution.get("mode") != "resolve" or resolution.get("value") != ["ref", target]:
            _fail("resolver cause does not match the retained target")
    elif kind == "async":
        if set(cause) != {"kind", "invocation"}:
            _fail("async cause fields")
        node = _promise(machine, adopter)
        call = node.get("asyncCall") or {}
        completion = call.get("completion") or {}
        if call.get("invocation") != cause["invocation"] or call.get("phase") != "completed":
            _fail("async cause does not own this completed execution Promise")
        if completion.get("kind") != "return" or completion.get("value") != ["ref", target]:
            _fail("async cause does not match its returned Promise")
    elif kind == "reaction":
        if set(cause) != {"kind", "job"}:
            _fail("reaction cause fields")
        job = _rows(machine.get("jobs", [])).get(cause["job"])
        if not job or job.get("target") != adopter or job.get("status") != "completed":
            _fail("reaction cause does not own the derived Promise")
        if job.get("returned", {}).get("value") != ["ref", target]:
            _fail("reaction cause does not match returned Promise")
    elif kind == "source":
        if set(cause) != {"kind"}:
            _fail("source cause fields")
        node = _promise(machine, adopter)
        capabilities = (machine["protocol"].get("promiseExecutors") or {}).get("capabilities", [])
        jobs = _rows(machine.get("jobs", []))
        if (
            node.get("asyncCall") or node.get("join")
            or any(c.get("promise") == adopter for c in capabilities)
            or any(job.get("target") == adopter for job in jobs.values())
        ):
            _fail("source cause bypasses a protected Promise owner")
    else:
        _fail("unknown adoption cause")


def validate_record(record):
    from .protocol_v8 import _exact, _identity, validate_protocol_value

    if type(record) is not dict or record.get("kind") not in FIELDS:
        return False
    _exact(record, FIELDS[record["kind"]], "Promise adoption record")
    for key in ("promise", "target", "job", "forwarding", "owner"):
        if key in record:
            _identity(record[key])
    if record["kind"] == "promise.adopt.forward":
        if record["status"] not in ("fulfilled", "rejected"):
            _fail("forwarding settlement status")
        validate_protocol_value(record["value"])
    elif record["kind"] == "promise.adopt.self":
        _identity(record["error"])
    return True


def _queue_job(machine, identity, *, status, owner, promise=None, adopter=None, target=None):
    jobs = _rows(machine.get("jobs", []))
    continuations = _rows(machine.get("continuations", []))
    if identity in jobs or identity in continuations:
        _fail("duplicate adoption job")
    scheduler = machine.setdefault("scheduler", {"registration": 0, "ready": 0})
    job = {
        "status": status,
        "owner": owner,
        "closure": None,
        "reaction": "adopt" if status == "queued" else "adoption",
        "adoptionPromise": adopter,
        "adoptionTarget": target,
    }
    if status == "queued":
        scheduler["ready"] += 1
        job["readySequence"] = scheduler["ready"]
        machine.setdefault("ready", []).append(identity)
    else:
        job.update(promise=promise, rejectionClosure=None, target=None)
        scheduler["registration"] += 1
        job["sequence"] = scheduler["registration"]
        job["sequence"] = scheduler["registration"]
    jobs[identity] = job
    from .promise_scheduler_v8 import store_rows

    store_rows(machine, "jobs", jobs)


def apply_record(machine, record):
    from .promise_scheduler_v8 import refresh_ready, settle_promise
    kind = record["kind"]
    if kind == "promise.adopt.enqueue":
        adopter, target, identity = record["promise"], record["target"], record["job"]
        if adopter == target:
            _fail("direct self-resolution uses the self rejection record")
        row = _adoption(machine, adopter)
        _promise(machine, target)
        _assert_intrinsic_then(machine, target)
        if row is None or row["phase"] != "intent" or row["target"] != target:
            # Source-created root results are allowed only when no protected cause exists.
            stage_source_intent(machine, adopter, target)
            row = _adoption(machine, adopter)
        if row["phase"] != "intent" or row["target"] != target:
            _fail("enqueue does not consume exact resolution intent")
        _check_cause(machine, adopter, row, target)
        if (
            (not machine["stack"] and row.get("cause", {}).get("kind") != "reaction")
            or row["owner"] != next(iter(machine.get("frameFunctions", {})), None)
        ):
            _fail("adoption owner changed before enqueue")
        _fresh_job(machine, identity)
        if sum(1 for node in machine["heap"].values() if node.get("kind") == "promise" and node.get("adoption") is not None) >= BOUND:
            _fail("adoption bound")
        row.update(phase="queued", job=identity)
        _queue_job(machine, identity, status="queued", owner=row["owner"], adopter=adopter, target=target)
        return True
    if kind == "promise.adopt.enter":
        adopter, target, identity, forwarding = (
            record["promise"], record["target"], record["job"], record["forwarding"]
        )
        row = _adoption(machine, adopter)
        jobs = _rows(machine.get("jobs", []))
        job = jobs.get(identity)
        _turn(machine, identity, job, row.get("owner") if row else None)
        if (
            row["phase"] != "queued" or row["job"] != identity or row["target"] != target
            or not job or job.get("reaction") != "adopt"
            or job.get("adoptionPromise") != adopter or job.get("adoptionTarget") != target
            or job.get("owner") != row["owner"] or machine.get("ready", [])[:1] != [identity]
            or job.get("status") != "queued"
        ):
            _fail("assimilation job is not the FIFO head")
        _fresh_job(machine, forwarding)
        machine["ready"].pop(0)
        job["status"] = "completed"
        row.update(phase="observing", forwarding=forwarding)
        _queue_job(machine, forwarding, status="waiting", owner=row["owner"], promise=target, adopter=adopter, target=target)
        refresh_ready(machine)
        return True
    if kind == "promise.adopt.forward":
        adopter, target, identity = record["promise"], record["target"], record["job"]
        row = _adoption(machine, adopter)
        jobs = _rows(machine.get("jobs", []))
        job = jobs.get(identity)
        source = _promise(machine, target)
        _turn(machine, identity, job, row.get("owner") if row else None)
        if (
            row["phase"] != "observing" or row["forwarding"] != identity or row["target"] != target
            or not job or job.get("reaction") != "adoption"
            or job.get("adoptionPromise") != adopter or job.get("adoptionTarget") != target
            or job.get("owner") != row["owner"] or machine.get("ready", [])[:1] != [identity]
            or job.get("status") != "queued"
            or source["state"] != record["status"] or source["result"] != record["value"]
        ):
            _fail("forwarding job does not match FIFO target settlement")
        machine["ready"].pop(0)
        job["status"] = "completed"
        if record["status"] == "fulfilled":
            from .promise_scheduler_v8 import assert_plain_resolution

            assert_plain_resolution(machine, record["value"])
        settle_promise(machine, adopter, record["status"], record["value"], refresh=False)
        row["phase"] = "completed"
        refresh_ready(machine)
        return True
    if kind == "promise.adopt.self":
        adopter, error = record["promise"], record["error"]
        row = _adoption(machine, adopter)
        if row is None:
            stage_source_intent(machine, adopter, adopter)
            row = _adoption(machine, adopter)
        if row["phase"] != "intent" or row["target"] != adopter:
            _fail("self rejection lacks exact resolution intent")
        _check_cause(machine, adopter, row, adopter)
        _fresh_job(machine, error)
        machine["heap"][error] = {
            "kind": "object",
            "code": None,
            "captures": [],
            "prototype": "ordinary",
            "properties": {
                "name": ["str", "TypeError"],
                "message": ["str", "Chaining cycle detected for promise"],
            },
        }
        error_value = ["ref", error]
        settle_promise(machine, adopter, "rejected", error_value)
        row.update(phase="self", error=error)
        return True
    if kind == "promise.reaction.complete":
        identity, owner, adopter, target = (
            record["job"], record["owner"], record["promise"], record["target"]
        )
        jobs = _rows(machine.get("jobs", []))
        job = jobs.get(identity)
        _promise(machine, adopter)
        _assert_intrinsic_then(machine, target)
        returned = (job or {}).get("returned") or {}
        if (
            not job or job.get("status") != "running" or job.get("owner") != owner
            or owner != next(iter(machine.get("frameFunctions", {})), None)
            or not job.get("promise") or not job.get("target")
            or job.get("target") != adopter
            or returned.get("kind") != "return"
            or returned.get("value") != ["ref", target]
            or machine.get("stack") or machine.get("activeJob") != identity
            or job.get("reaction") not in ("then", "catch")
            or any(c.get("activeJob") == identity and c["state"] != "resumed"
                   for c in _rows(machine.get("continuations", [])).values())
        ):
            _fail("reaction completion does not bind its returned Promise")
        row = _adoption(machine, adopter)
        if _promise(machine, adopter)["state"] != "pending" or row is not None:
            _fail("reaction derived Promise is not fresh and pending")
        if job.get("reaction") == "finally":
            _fail("finally Promise return requires its cleanup protocol")
        job["status"] = "completed"
        if machine.get("activeJob") == identity:
            machine["captureAliases"] = dict(job["priorAliases"])
            if machine.get("protocol") is not None and "priorReceiver" in job:
                machine["receiver"] = copy.deepcopy(job["priorReceiver"])
            machine["activeJob"] = None
        stage_intent(machine, adopter, target, {"kind": "reaction", "job": identity})
        return True
    return False


def assert_next(machine, record):
    kind = record.get("kind")
    fields = record.get("fields", record)
    pending = [
        (identity, node.get("adoption"))
        for identity, node in machine.get("heap", {}).items()
        if node.get("kind") == "promise"
        and node.get("adoption") is not None
        and node["adoption"].get("phase") == "intent"
    ]
    if pending and record.get("kind") not in ("promise.adopt.enqueue", "promise.adopt.self"):
        _fail("resolution intent must be consumed before another record")
    if kind == "promise" and fields.get("action") in ("resolve", "reject", "adopt"):
        if machine["heap"].get(fields.get("promise"), {}).get("adoption") is not None:
            _fail("adoption-owned Promise requires its scheduler record")
    selected_identity = (
        fields.get("result") if kind == "promise.join"
        else fields.get("promise") if kind in ("promise.executor.enter", "async.call")
        else None
    )
    if selected_identity and machine["heap"].get(selected_identity, {}).get("adoption") is not None:
        _fail("adoption-owned Promise cannot be claimed by another owner")
    jobs = _rows(machine.get("jobs", []))
    selected_job = jobs.get(fields.get("job"))
    if kind == "job" and selected_job and selected_job.get("reaction") in ("adopt", "adoption"):
        _fail("adoption jobs require their explicit FIFO record")
    head = machine.get("ready", [])[:1]
    head_job = jobs.get(head[0]) if head else None
    if not machine.get("stack") and machine.get("activeJob") is None and head_job:
        if head_job.get("reaction") in ("adopt", "adoption"):
            expected = "promise.adopt.enter" if head_job["reaction"] == "adopt" else "promise.adopt.forward"
            if kind != expected or fields.get("job") != head[0]:
                _fail("ready adoption job must run before any later record")


def validate_state(machine):
    from .protocol_v8 import _assert_fresh_heap_identity, _exact, _identity

    rows = _rows(machine.get("jobs", []))
    ready = machine.get("ready", [])
    owned_jobs = set()
    for identity, node in machine["heap"].items():
        if node.get("kind") != "promise":
            continue
        row = node.get("adoption")
        if row is None:
            continue
        _exact(row, {"schema", "owner", "target", "phase", "cause", "job", "forwarding", "error"}, "Promise adoption metadata")
        if row["schema"] != STATE_SCHEMA or row["phase"] not in ("intent", "queued", "observing", "completed", "self"):
            _fail("adoption metadata schema or phase")
        for key in ("owner", "target", "job", "forwarding"):
            if row[key] is not None:
                _identity(row[key])
        if row["owner"] != next(iter(machine.get("frameFunctions", {})), None):
            _fail("retained adoption owner differs from its root invocation")
        _promise(machine, row["target"])
        _assert_intrinsic_then(machine, row["target"])
        if row["phase"] == "intent" and (row["target"] is None or row["cause"] is None):
            _fail("incomplete staged adoption intent")
        if row["phase"] in ("intent", "queued", "observing", "completed", "self"):
            _check_cause(machine, identity, row, row["target"])
        if row["phase"] == "intent":
            if (
                row["job"] is not None or row["forwarding"] is not None or row["error"] is not None
                or node["state"] != "pending" or node["result"] is not None
            ):
                _fail("staged adoption intent contains a future identity or early result")
        elif row["phase"] == "queued":
            _assert_fresh_heap_identity(machine, row["job"])
            job = rows.get(row["job"])
            if (
                row["job"] is None or row["forwarding"] is not None or row["error"] is not None
                or node["state"] != "pending" or node["result"] is not None
                or not job or job.get("status") != "queued" or job.get("owner") != row["owner"]
                or job.get("reaction") != "adopt" or job.get("closure") is not None
                or job.get("adoptionPromise") != identity or job.get("adoptionTarget") != row["target"]
                or type(job.get("readySequence")) is not int or job["readySequence"] <= 0
                or job["readySequence"] > machine["scheduler"]["ready"] or row["job"] in owned_jobs
                or row["job"] not in ready
            ):
                _fail("queued adoption job lost its exact FIFO binding")
            owned_jobs.add(row["job"])
        elif row["phase"] in ("observing", "completed"):
            _assert_fresh_heap_identity(machine, row["job"])
            _assert_fresh_heap_identity(machine, row["forwarding"])
            job = rows.get(row["job"])
            forwarding = rows.get(row["forwarding"])
            if (
                row["job"] is None or row["forwarding"] is None or row["error"] is not None
                or not job or job.get("status") != "completed" or job.get("owner") != row["owner"]
                or job.get("reaction") != "adopt" or job.get("closure") is not None
                or job.get("adoptionPromise") != identity or job.get("adoptionTarget") != row["target"]
                or type(job.get("readySequence")) is not int or job["readySequence"] <= 0
                or job["readySequence"] > machine["scheduler"]["ready"]
                or row["job"] in owned_jobs or row["forwarding"] in owned_jobs
            ):
                _fail("retained assimilation job linkage")
            owned_jobs.update((row["job"], row["forwarding"]))
            expected_status = (
                "completed" if row["phase"] == "completed"
                else "waiting" if _promise(machine, row["target"])["state"] == "pending"
                else "queued"
            )
            if (
                not forwarding or forwarding.get("status") != expected_status
                or forwarding.get("owner") != row["owner"] or forwarding.get("reaction") != "adoption"
                or forwarding.get("promise") != row["target"] or forwarding.get("closure") is not None
                or forwarding.get("rejectionClosure") is not None or forwarding.get("target") is not None
                or forwarding.get("adoptionPromise") != identity
                or forwarding.get("adoptionTarget") != row["target"]
                or type(forwarding.get("sequence")) is not int or forwarding["sequence"] <= 0
                or forwarding["sequence"] > machine["scheduler"]["registration"]
                or ((expected_status == "queued") != (row["forwarding"] in ready))
                or expected_status != "waiting"
                and (
                    type(forwarding.get("readySequence")) is not int
                    or forwarding["readySequence"] <= 0
                    or forwarding["readySequence"] > machine["scheduler"]["ready"]
                )
            ):
                _fail("retained forwarding registration linkage")
            if row["phase"] == "completed":
                target = _promise(machine, row["target"])
                if node["state"] != target["state"] or node["result"] != target["result"]:
                    _fail("completed adoption differs from target settlement")
        elif row["phase"] == "self":
            error = row["error"]
            _identity(error)
            error_node = machine["heap"].get(error)
            if (
                row["target"] != identity or row["job"] is not None or row["forwarding"] is not None
                or node.get("state") != "rejected" or node.get("result") != ["ref", error]
                or not error_node or error_node.get("kind") != "object"
                or error in rows or error in _rows(machine.get("continuations", []))
            ):
                _fail("self-resolution rejection lost its declared TypeError")
    for identity, job in rows.items():
        if job.get("reaction") in ("adopt", "adoption"):
            if identity not in owned_jobs or job.get("status") not in ("waiting", "queued", "completed"):
                _fail("orphan or invalid adoption job")
            if (job.get("status") == "queued") != (identity in ready):
                _fail("adoption job ready status differs from queue")
