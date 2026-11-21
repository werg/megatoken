"""Independent replay for bounded Promise.all, allSettled and race joins."""

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
        if record["mode"] not in ("all", "allSettled", "race"):
            _fail("unsupported join mode")
        for key in ("inputs", "jobs"):
            if type(record[key]) is not list or len(record[key]) > BOUND:
                _fail("bounded " + key)
            for identity in record[key]:
                _identity(identity)
        if len(record["inputs"]) != len(record["jobs"]):
            _fail("input/job count")
        if record["mode"] != "race" and len(record["inputs"]) >= BOUND:
            _fail("all result must leave one property for array length")
        if len(set(record["jobs"])) != len(record["jobs"]):
            _fail("duplicate join job")
        _identity(record["output"], nullable=True)
        _identity(record["error"], nullable=True)
        if record["mode"] != "race" and not record["inputs"]:
            if record["output"] is None or record["error"] is not None:
                _fail("empty all allocation")
        elif record["output"] is not None or record["error"] is not None:
            _fail("future allocation identity")
    else:
        _identity(record["job"])
        for key in ("entry", "output", "error"):
            _identity(record[key], nullable=True)
        if record["error"] is not None:
            _fail("unused error allocation")
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


def _owner(machine):
    """The controller root survives after its JavaScript stack has returned."""
    frames = _dict_rows(machine.get("frameFunctions", {}))
    return next(iter(frames), None) or next(iter(machine.get("stack", [])), None)


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


def assert_next(machine, record):
    fields = record.get("fields", record)
    if (
        record.get("kind") == "promise"
        and fields.get("action") in ("resolve", "reject", "adopt")
        and machine["heap"].get(fields.get("promise"), {}).get("join") is not None
    ):
        _fail("join result settlement requires its explicit join step")
    return True


def apply_record(machine, record):
    if not validate_record(record):
        return False
    jobs = _dict_rows(machine.get("jobs", []))
    ready = machine.setdefault("ready", [])
    scheduler = machine.setdefault("scheduler", {"registration": 0, "ready": 0})
    result = _promise(machine, record["result"])
    if record["kind"] == "promise.join":
        owner = _owner(machine)
        _identity(owner)
        if (
            result["state"] != "pending"
            or result.get("join") is not None
            or result.get("adopted") is not None
            or result.get("external") is not None
        ):
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
        if record["mode"] != "race" and not record["inputs"]:
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
                "owner": owner,
                "mode": record["mode"],
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
                "owner": owner,
                "mode": record["mode"],
                "inputs": copy.deepcopy(record["inputs"]),
                "jobs": copy.deepcopy(record["jobs"]),
                "observed": [False] * len(record["inputs"]),
                "values": [None] * len(record["inputs"]),
                "output": None,
                "error": None,
                "settled": False,
            }
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
        or join.get("mode") not in ("all", "allSettled", "race")
        or join.get("owner") != _owner(machine)
        or join.get("settled") != (result["state"] != "pending")
    ):
        _fail("unknown or stale join result")
    if record["job"] not in join["jobs"]:
        _fail("unregistered join job")
    index = join["jobs"].index(record["job"])
    job = jobs.get(record["job"])
    if (
        not job
        or job.get("owner") != join["owner"]
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
    if join["mode"] == "allSettled":
        entry = record["entry"]
        if entry is None:
            _fail("allSettled requires a fresh settlement entry")
        _fresh_output(machine, entry)
        machine["heap"][entry] = {
            "kind": "object",
            "properties": {
                "status": ["str", source["state"]],
                "value" if source["state"] == "fulfilled" else "reason": copy.deepcopy(
                    source["result"]
                ),
            },
            "prototype": "ordinary",
            "captures": [],
            "code": None,
        }
        observed_value = ["ref", entry]
    else:
        if record["entry"] is not None:
            _fail("unused settlement entry")
        observed_value = copy.deepcopy(source["result"])
    ready.pop(0)
    job["status"] = "completed"
    join["observed"][index] = True
    join["values"][index] = observed_value
    expected_output = None
    if not join["settled"]:
        if join["mode"] == "race":
            _settle(machine, record["result"], source["state"], source["result"])
            join["settled"] = True
        elif join["mode"] == "all" and source["state"] == "rejected":
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
    owner = _owner(machine)
    for result_id, result in heap.items():
        if not isinstance(result, dict) or result.get("kind") != "promise":
            continue
        join = result.get("join")
        if join is None:
            continue
        _identity(join.get("owner"))
        if (
            join.get("schema") != STATE_SCHEMA
            or result.get("adopted") is not None
            or result.get("external") is not None
            or join.get("mode") not in ("all", "allSettled", "race")
            or not all(
                type(join.get(key)) is list for key in ("inputs", "jobs", "observed", "values")
            )
            or len({len(join[key]) for key in ("inputs", "jobs", "observed", "values")}) != 1
            or type(join.get("settled")) is not bool
            or join["settled"] != (result["state"] != "pending")
            or len(join["inputs"]) > BOUND
            or join.get("mode") != "race"
            and len(join["inputs"]) >= BOUND
            or join.get("mode") != "race"
            and not join["inputs"]
            and (
                result["state"] != "fulfilled"
                or result.get("result") != ["ref", join.get("output")]
            )
            or join.get("mode") == "race"
            and any(join["observed"])
            and not join["settled"]
        ):
            _fail("restored join metadata")
        entries, seen_jobs = set(), set()
        for index, (input_id, job_id) in enumerate(zip(join["inputs"], join["jobs"], strict=True)):
            _identity(input_id)
            _identity(job_id)
            if job_id in seen_jobs:
                _fail("restored duplicate join job")
            seen_jobs.add(job_id)
            source = _promise(machine, input_id)
            if type(join["observed"][index]) is not bool:
                _fail("restored observation flag")
            if join["observed"][index]:
                value = join["values"][index]
                _known(machine, value, result_id)
                if source["state"] == "pending":
                    _fail("observed input remains pending")
                if join["mode"] == "allSettled":
                    entry_id = value[1] if value[0] == "ref" else None
                    entry = heap.get(entry_id, {})
                    if entry.get("kind") != "object" or entry_id in entries:
                        _fail("restored settlement entry identity")
                    entries.add(entry_id)
                    # Once exposed by fulfillment, these are ordinary mutable
                    # guest objects. Only unpublished entries retain this shape.
                    expected = {
                        "status": ["str", source["state"]],
                        "value" if source["state"] == "fulfilled" else "reason": source["result"],
                    }
                    if not join["settled"] and entry.get("properties") != expected:
                        _fail("restored pending entry differs from input")
                elif value != source.get("result"):
                    _fail("restored observed value changed")
            elif join["values"][index] is not None:
                _fail("unobserved input value")
            job = jobs.get(job_id)
            if owner is not None and join["owner"] != owner:
                if job is not None:
                    _fail("join job belongs to another invocation")
                continue
            if (
                not job
                or job.get("owner") != join["owner"]
                or job.get("reaction") != "join"
                or job.get("joinResult") != result_id
                or job.get("joinIndex") != index
                or job.get("promise") != input_id
            ):
                _fail("restored scheduler linkage")
            if (job.get("status") == "completed") != join["observed"][index]:
                _fail("restored job completion differs from observed input")
        if join["mode"] == "allSettled":
            if join["settled"] != all(join["observed"]) or result["state"] == "rejected":
                _fail("restored allSettled completion")
            if join["settled"] and (
                result.get("result") != ["ref", join.get("output")]
                or heap.get(join.get("output"), {}).get("kind") != "array"
            ):
                _fail("restored allSettled output")
        if join.get("error") is not None:
            _fail("unsupported join error storage")
        if join.get("output") is not None and result["state"] != "fulfilled":
            _fail("premature join output")
    for job_id, job in jobs.items():
        if job.get("reaction") != "join":
            continue
        result = heap.get(job.get("joinResult"))
        if not result or not result.get("join") or job_id not in result["join"]["jobs"]:
            _fail("orphan join job")
        if job.get("owner") != result["join"]["owner"] or (
            owner is not None and job.get("owner") != owner
        ):
            _fail("join job belongs to another invocation")
        if job.get("status") not in ("waiting", "queued", "completed"):
            _fail("join job lifecycle")
        if job["status"] == "queued" and job_id not in ready:
            _fail("queued job absent from ready queue")
    return True
