"""Bounded independent replay for source-qualified Promise.all/race joins."""

from __future__ import annotations

import copy

from .promise_scheduler_v8 import refresh_ready as _refresh
from .promise_scheduler_v8 import settle_promise as _settle

PROFILE = "promise-join/1"
STATE_SCHEMA = "semantic-promise-join-state/1"
FIELDS = {
    "promise.join": {"kind", "mode", "result", "inputs", "jobs", "output", "error"},
    "promise.join.step": {"kind", "result", "job", "entry", "output", "error", "status", "value"},
}
BOUND = 4096


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid Promise join protocol: " + message)


def _identity(value, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("identity")


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind)
    if fields is None:
        return False
    if set(record) != fields:
        _fail("record fields")
    _identity(record["result"])
    if kind == "promise.join":
        if record["mode"] not in ("all", "race"):
            _fail("unsupported join mode")
        for key in ("inputs", "jobs"):
            if type(record[key]) is not list or len(record[key]) > BOUND:
                _fail("bounded " + key)
            for identity in record[key]:
                _identity(identity)
        if len(record["inputs"]) != len(record["jobs"]):
            _fail("input/job count")
        if record["mode"] == "all" and len(record["inputs"]) >= BOUND:
            _fail("all result must leave one property for array length")
        if len(set(record["jobs"])) != len(record["jobs"]):
            _fail("duplicate join job")
        _identity(record["output"], nullable=True)
        _identity(record["error"], nullable=True)
        if record["mode"] == "all" and not record["inputs"]:
            if record["output"] is None or record["error"] is not None:
                _fail("empty all allocation")
        elif record["output"] is not None or record["error"] is not None:
            _fail("future allocation identity")
    else:
        _identity(record["job"])
        for key in ("entry", "output", "error"):
            _identity(record[key], nullable=True)
        if record["entry"] is not None or record["error"] is not None:
            _fail("unused entry/error allocation")
        if record["status"] not in ("pending", "fulfilled", "rejected"):
            _fail("Promise status")
        from .protocol_v8 import validate_protocol_value

        if record["status"] == "pending":
            if record["value"] is not None:
                _fail("pending value")
        else:
            if record["value"] is None:
                _fail("settled value")
            validate_protocol_value(record["value"])
    return True


def _dict_rows(value):
    return dict(value) if isinstance(value, list) else value


def _list_rows(value):
    return [[key, row] for key, row in value.items()]


def _promise(machine, identity):
    node = machine["heap"].get(identity)
    if not node or node.get("kind") != "promise":
        _fail("Promise identity")
    if node.get("state") not in ("pending", "fulfilled", "rejected"):
        _fail("Promise state")
    if (node["state"] == "pending") != (node.get("result") is None):
        _fail("Promise settlement shape")
    return node


def _known(machine, value, self_id=None):
    from .protocol_v8 import _known as known

    return known(machine, value, self_id)


def _fresh_output(machine, identity):
    from .protocol_v8 import _assert_fresh_heap_identity

    _assert_fresh_heap_identity(machine, identity)
    if identity in _dict_rows(machine.get("jobs", [])):
        _fail("output identity collides with a job")


def apply_record(machine, record):
    if not validate_record(record):
        return False
    jobs = _dict_rows(machine.get("jobs", []))
    ready = machine.setdefault("ready", [])
    scheduler = machine.setdefault("scheduler", {"registration": 0, "ready": 0})
    result = _promise(machine, record["result"])
    if record["kind"] == "promise.join":
        if result["state"] != "pending" or result.get("join") is not None:
            _fail("result must be a fresh pending Promise")
        if record["result"] in record["inputs"]:
            _fail("result cannot be an input")
        continuations = machine.get("continuations", [])
        continuation_ids = (
            set(continuations)
            if isinstance(continuations, dict)
            else {row[0] for row in continuations}
        )
        occupied = set(machine["heap"]) | set(jobs) | continuation_ids
        for identity in record["jobs"]:
            if identity in occupied:
                _fail("join job identity collision")
        for identity in record["inputs"]:
            _promise(machine, identity)
        if record["mode"] == "all" and not record["inputs"]:
            output = record["output"]
            _fresh_output(machine, output)
            machine["heap"][output] = {
                "kind": "array",
                "properties": {"length": ["num", 0]},
                "prototype": "ordinary",
                "captures": [],
                "code": None,
            }
            _settle(machine, record["result"], "fulfilled", ["ref", output])
            result["join"] = {
                "schema": STATE_SCHEMA,
                "mode": "all",
                "inputs": [],
                "jobs": [],
                "observed": [],
                "values": [],
                "output": output,
                "error": None,
                "settled": True,
            }
        else:
            result["join"] = {
                "schema": STATE_SCHEMA,
                "mode": record["mode"],
                "inputs": copy.deepcopy(record["inputs"]),
                "jobs": copy.deepcopy(record["jobs"]),
                "observed": [False] * len(record["inputs"]),
                "values": [None] * len(record["inputs"]),
                "output": None,
                "error": None,
                "settled": False,
            }
            owner = machine.get("stack", [None])[0]
            for index, (identity, input_id) in enumerate(
                zip(record["jobs"], record["inputs"], strict=True)
            ):
                scheduler["registration"] += 1
                jobs[identity] = {
                    "status": "waiting",
                    "owner": owner,
                    "promise": input_id,
                    "reaction": "join",
                    "closure": None,
                    "rejectionClosure": None,
                    "target": None,
                    "sequence": scheduler["registration"],
                    "joinResult": record["result"],
                    "joinIndex": index,
                }
            machine["jobs"] = _list_rows(jobs)
            _refresh(machine)
        return True

    join = result.get("join")
    if (
        not join
        or join.get("schema") != STATE_SCHEMA
        or join.get("settled") != (result["state"] != "pending")
    ):
        _fail("unknown or stale join result")
    if record["job"] not in join["jobs"]:
        _fail("unregistered join job")
    index = join["jobs"].index(record["job"])
    job = jobs.get(record["job"])
    if (
        not job
        or job.get("reaction") != "join"
        or job.get("joinResult") != record["result"]
        or job.get("joinIndex") != index
        or job.get("promise") != join["inputs"][index]
    ):
        _fail("join scheduler linkage")
    if (
        machine.get("stack")
        or machine.get("activeJob")
        or ready[:1] != [record["job"]]
        or job["status"] != "queued"
    ):
        _fail("join step violates FIFO or stack turn")
    source = _promise(machine, join["inputs"][index])
    if source["state"] == "pending":
        _fail("input not settled")
    ready.pop(0)
    job["status"] = "completed"
    join["observed"][index] = True
    join["values"][index] = copy.deepcopy(source["result"])
    expected_output = None
    if not join["settled"]:
        if join["mode"] == "race":
            _settle(machine, record["result"], source["state"], source["result"])
            join["settled"] = True
        elif source["state"] == "rejected":
            _settle(machine, record["result"], "rejected", source["result"])
            join["settled"] = True
        elif all(join["observed"]):
            expected_output = record["output"]
            if expected_output is None:
                _fail("final Promise.all output identity")
            _fresh_output(machine, expected_output)
            props = {str(i): copy.deepcopy(value) for i, value in enumerate(join["values"])}
            props["length"] = ["num", len(join["values"])]
            machine["heap"][expected_output] = {
                "kind": "array",
                "properties": props,
                "prototype": "ordinary",
                "captures": [],
                "code": None,
            }
            join["output"] = expected_output
            _settle(machine, record["result"], "fulfilled", ["ref", expected_output])
            join["settled"] = True
    if record["output"] != expected_output:
        _fail("false output allocation identity")
    if result["state"] != record["status"] or result.get("result") != record["value"]:
        _fail("false join settlement claim")
    machine["jobs"] = _list_rows(jobs)
    _refresh(machine)
    return True


def validate_state(machine):
    """Check restored join metadata against heap and scheduler identities."""
    heap = machine.get("heap", {})
    jobs = _dict_rows(machine.get("jobs", []))
    ready = machine.get("ready", [])
    for result_id, result in heap.items():
        if not isinstance(result, dict) or result.get("kind") != "promise":
            continue
        join = result.get("join")
        if join is None:
            continue
        if (
            join.get("schema") != STATE_SCHEMA
            or join.get("mode") not in ("all", "race")
            or not all(
                type(join.get(key)) is list for key in ("inputs", "jobs", "observed", "values")
            )
            or len({len(join[key]) for key in ("inputs", "jobs", "observed", "values")}) != 1
            or type(join.get("settled")) is not bool
            or join["settled"] != (result["state"] != "pending")
            or join.get("mode") == "all"
            and len(join["inputs"]) >= BOUND
            or join.get("mode") == "race"
            and any(join["observed"])
            and not join["settled"]
        ):
            _fail("restored join metadata")
        for index, (input_id, job_id) in enumerate(zip(join["inputs"], join["jobs"], strict=True)):
            _identity(input_id)
            _identity(job_id)
            source = _promise(machine, input_id)
            if type(join["observed"][index]) is not bool:
                _fail("restored observation flag")
            if join["observed"][index]:
                if join["values"][index] != source.get("result"):
                    _fail("restored observed value changed")
                _known(machine, join["values"][index], result_id)
            elif join["values"][index] is not None:
                _fail("unobserved input value")
            job = jobs.get(job_id)
            if (
                not job
                or job.get("reaction") != "join"
                or job.get("joinResult") != result_id
                or job.get("joinIndex") != index
                or job.get("promise") != input_id
            ):
                _fail("restored scheduler linkage")
            if (job.get("status") == "completed") != join["observed"][index]:
                _fail("restored job completion differs from observed input")
    for job_id, job in jobs.items():
        if job.get("reaction") != "join":
            continue
        result = heap.get(job.get("joinResult"))
        if not result or not result.get("join") or job_id not in result["join"]["jobs"]:
            _fail("orphan join job")
        if job.get("status") not in ("waiting", "queued", "completed"):
            _fail("join job lifecycle")
        if job["status"] == "queued" and job_id not in ready:
            _fail("queued job absent from ready queue")
    return True
