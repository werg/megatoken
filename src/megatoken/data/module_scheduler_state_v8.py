"""Source-bound async module DAG state; public/data admission is a separate gate."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping

SCHEMA = "semantic-module-scheduler-plan/1"
STATE_SCHEMA = "semantic-module-scheduler-state/1"
_HEX = re.compile(r"^[0-9a-f]{64}$")


class ModuleSchedulerError(ValueError):
    pass


def _fail(message):
    raise ModuleSchedulerError("Module scheduler state: " + message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _exact(value, keys):
    return isinstance(value, Mapping) and set(value) == set(keys)


def _id(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def build_module_scheduler_plan_v8(metadata):
    """Build only from fresh evaluated-module linker metadata containing source text."""
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("profile") != "evaluated-module/1"
        or metadata.get("evaluationStrategy") != "async-dependency-scheduler"
        or not isinstance(metadata.get("modules"), list)
        or not isinstance(metadata.get("asyncDependencies"), list)
        or not isinstance(metadata.get("entryModule"), str)
    ):
        _fail("unsupported linker metadata")
    module_rows = metadata["modules"]
    deps_rows = metadata["asyncDependencies"]
    module_by_file = {row.get("file"): row for row in module_rows if isinstance(row, Mapping)}
    if len(module_by_file) != len(module_rows) or len(module_rows) != len(deps_rows):
        _fail("module table mismatch")
    rows = []
    for index, row in enumerate(deps_rows):
        if (
            not isinstance(row, Mapping)
            or type(row.get("index")) is not int
            or row.get("index") != index
            or row.get("file") not in module_by_file
            or type(row.get("hasTLA")) is not bool
            or not isinstance(row.get("dependencies"), list)
        ):
            _fail("dependency row")
        source = module_by_file[row["file"]].get("source")
        if not isinstance(source, str):
            _fail("missing original source")
        dependencies = []
        for dependency_file in row["dependencies"]:
            if (
                not isinstance(dependency_file, str)
                or dependency_file not in module_by_file
                or dependency_file == row["file"]
            ):
                _fail("dependency file")
            dependencies.append(dependency_file)
        if len(set(dependencies)) != len(dependencies):
            _fail("duplicate dependency")
        rows.append(
            {
                "file": row["file"],
                "index": index,
                "sourceDigest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "dependencies": dependencies,
                "hasTLA": row["hasTLA"],
            }
        )
    by_file = {row["file"]: row for row in rows}
    entry = metadata["entryModule"]
    if entry not in by_file:
        _fail("entry module")
    visiting, visited = set(), set()

    def visit(file):
        if file in visiting:
            _fail("cyclic module graphs are outside this DAG profile")
        if file in visited:
            return
        visiting.add(file)
        for dependency in by_file[file]["dependencies"]:
            visit(dependency)
        visiting.remove(file)
        visited.add(file)

    for row in rows:
        visit(row["file"])
    body = {"schema": SCHEMA, "entryModule": entry, "modules": rows}
    return {**body, "digest": _digest(body)}


def verify_module_scheduler_plan_v8(plan):
    if (
        not _exact(plan, {"schema", "entryModule", "modules", "digest"})
        or plan.get("schema") != SCHEMA
        or not _id(plan.get("entryModule"))
        or not isinstance(plan.get("modules"), list)
        or not _id(plan.get("digest"))
    ):
        _fail("plan envelope")
    body = {
        "schema": plan["schema"],
        "entryModule": plan["entryModule"],
        "modules": plan["modules"],
    }
    if _digest(body) != plan["digest"]:
        _fail("plan digest")
    files = set()
    for index, row in enumerate(plan["modules"]):
        if (
            not _exact(row, {"file", "index", "sourceDigest", "dependencies", "hasTLA"})
            or not _id(row.get("file"))
            or type(row.get("index")) is not int
            or row.get("index") != index
            or not isinstance(row.get("sourceDigest"), str)
            or not _HEX.fullmatch(row["sourceDigest"])
            or not isinstance(row.get("dependencies"), list)
            or type(row.get("hasTLA")) is not bool
            or row["file"] in files
            or len(set(row["dependencies"])) != len(row["dependencies"])
        ):
            _fail("module row")
        files.add(row["file"])
    if plan["entryModule"] not in files:
        _fail("entry absent")
    by_file = {row["file"]: row for row in plan["modules"]}
    visiting, visited = set(), set()

    def visit(file):
        row = by_file.get(file)
        if row is None:
            _fail("unknown dependency")
        if file in visiting:
            _fail("cycle")
        if file in visited:
            return
        visiting.add(file)
        for dependency in row["dependencies"]:
            visit(dependency)
        visiting.remove(file)
        visited.add(file)

    for row in plan["modules"]:
        visit(row["file"])
    return plan


def create_module_scheduler_state_v8(plan):
    verify_module_scheduler_plan_v8(plan)
    return {
        "schema": STATE_SCHEMA,
        "planDigest": plan["digest"],
        "nextStartOrder": 0,
        "modules": {
            row["file"]: {
                "status": "waiting",
                "pendingDependencies": len(row["dependencies"]),
                "startOrder": None,
                "frameId": None,
                "executionPromiseId": None,
                "awaited": None,
                "completionJobId": None,
                "failure": None,
                "completedDependencies": [],
            }
            for row in plan["modules"]
        },
    }


def _verify_state(plan, state):
    if (
        not _exact(state, {"schema", "planDigest", "nextStartOrder", "modules"})
        or state.get("schema") != STATE_SCHEMA
        or state.get("planDigest") != plan["digest"]
        or type(state.get("nextStartOrder")) is not int
        or state["nextStartOrder"] < 0
        or not _exact(state.get("modules"), {row["file"] for row in plan["modules"]})
    ):
        _fail("state shape")
    orders, frames, executions = [], set(), set()
    for module in plan["modules"]:
        row = state["modules"][module["file"]]
        if (
            not _exact(
                row,
                {
                    "status",
                    "pendingDependencies",
                    "startOrder",
                    "frameId",
                    "executionPromiseId",
                    "awaited",
                    "completionJobId",
                    "failure",
                    "completedDependencies",
                },
            )
            or row.get("status") not in {"waiting", "running", "suspended", "fulfilled", "rejected"}
            or not isinstance(row.get("completedDependencies"), list)
            or len(set(row["completedDependencies"])) != len(row["completedDependencies"])
            or any(file not in module["dependencies"] for file in row["completedDependencies"])
            or type(row.get("pendingDependencies")) is not int
            or row.get("pendingDependencies")
            != len(module["dependencies"]) - len(row["completedDependencies"])
        ):
            _fail("module state")
        started = row["startOrder"] is not None
        if started:
            if (
                type(row["startOrder"]) is not int
                or row["startOrder"] < 0
                or row["startOrder"] >= state["nextStartOrder"]
                or not _id(row["frameId"])
                or (
                    not _id(row["executionPromiseId"])
                    if module["hasTLA"]
                    else row["executionPromiseId"] is not None
                )
                or row["frameId"] in frames
                or (
                    row["executionPromiseId"] is not None
                    and row["executionPromiseId"] in executions
                )
            ):
                _fail("module identity state")
            frames.add(row["frameId"])
            if row["executionPromiseId"] is not None:
                executions.add(row["executionPromiseId"])
            orders.append(row["startOrder"])
        elif row["frameId"] is not None or row["executionPromiseId"] is not None:
            _fail("unstarted identity")
        if row["status"] == "waiting" and (
            started
            or row["awaited"] is not None
            or row["completionJobId"] is not None
            or row["failure"] is not None
        ):
            _fail("waiting module state")
        if row["status"] == "running" and (
            not started
            or row["pendingDependencies"] != 0
            or row["awaited"] is not None
            or row["completionJobId"] is not None
            or row["failure"] is not None
        ):
            _fail("running module state")
        if row["status"] == "suspended" and (
            not started
            or row["pendingDependencies"] != 0
            or not _exact(row["awaited"], {"promiseId"})
            or not _id(row["awaited"]["promiseId"])
            or row["completionJobId"] is not None
            or row["failure"] is not None
        ):
            _fail("suspended module state")
        if row["status"] == "fulfilled" and (
            not started
            or row["pendingDependencies"] != 0
            or row["awaited"] is not None
            or not _id(row["completionJobId"])
            or row["failure"] is not None
        ):
            _fail("fulfilled module state")
        if row["status"] == "rejected" and (
            row["awaited"] is not None
            or not _exact(row["failure"], {"moduleFile", "promiseId", "jobId"})
            or row["failure"]["moduleFile"] not in {item["file"] for item in plan["modules"]}
            or (
                not _id(row["failure"]["promiseId"])
                if next(
                    item for item in plan["modules"] if item["file"] == row["failure"]["moduleFile"]
                )["hasTLA"]
                else row["failure"]["promiseId"] is not None
            )
            or not _id(row["failure"]["jobId"])
            or not _id(row["completionJobId"])
            or row["completionJobId"] != row["failure"]["jobId"]
            or (
                started
                and (
                    row["failure"]["moduleFile"] != module["file"]
                    or row["pendingDependencies"] != 0
                )
            )
        ):
            _fail("rejected module state")
    for module in plan["modules"]:
        row = state["modules"][module["file"]]
        for file in module["dependencies"]:
            dependency = state["modules"][file]
            completed = file in row["completedDependencies"]
            if dependency["status"] == "fulfilled" and not completed:
                _fail("missing completed dependency edge")
            if (
                completed
                and dependency["status"] != "fulfilled"
                and not (
                    row["status"] == "rejected"
                    and dependency["status"] == "rejected"
                    and row["failure"] == dependency["failure"]
                )
            ):
                _fail("completed dependency has no settled authority")
        if (
            row["status"] == "rejected"
            and row["startOrder"] is None
            and not any(
                state["modules"][file]["status"] == "rejected"
                and row["failure"] == state["modules"][file]["failure"]
                for file in row["completedDependencies"]
            )
        ):
            _fail("propagated failure has no rejected dependency")
    if len(set(orders)) != len(orders) or len(orders) != state["nextStartOrder"]:
        _fail("noncanonical start order")


def _settled_job(facts, action, status):
    promise = facts.get("promises", {}).get(action["promiseId"])
    job = facts.get("jobs", {}).get(action["jobId"])
    if (
        not isinstance(promise, Mapping)
        or promise.get("id") != action["promiseId"]
        or promise.get("status") != status
        or not isinstance(job, Mapping)
        or job.get("id") != action["jobId"]
        or job.get("promiseId") != action["promiseId"]
        or job.get("frameId") != action["frameId"]
        or job.get("status") != status
        or (
            action["type"] == "module.resume"
            and (job.get("kind") != "promise-reaction" or job.get("moduleFile") != action["file"])
        )
    ):
        _fail("resume is not bound to the existing promise job")
    if (
        "value" not in promise
        or "value" not in job
        or _canonical(promise["value"]) != _canonical(job["value"])
    ):
        _fail("promise/job outcome mismatch")
    return promise, job


def apply_module_scheduler_action_v8(plan, state, action, facts=None):
    """Return a new state after binding one action to current runtime facts."""
    verify_module_scheduler_plan_v8(plan)
    facts = facts or {}
    if (
        not isinstance(state, Mapping)
        or state.get("schema") != STATE_SCHEMA
        or state.get("planDigest") != plan["digest"]
        or not isinstance(action, Mapping)
    ):
        _fail("state/action envelope")
    _verify_state(plan, state)
    next_state = copy.deepcopy(state)
    row = next_state["modules"].get(action.get("file"))
    source = next((item for item in plan["modules"] if item["file"] == action.get("file")), None)
    if row is None or source is None:
        _fail("unknown module")
    kind = action.get("type")
    if kind == "module.start":
        if (
            not _exact(action, {"type", "file", "frameId", "executionPromiseId"})
            or row["status"] != "waiting"
            or row["pendingDependencies"] != 0
            or not _id(action["frameId"])
            or (
                not _id(action["executionPromiseId"])
                if source["hasTLA"]
                else action["executionPromiseId"] is not None
            )
        ):
            _fail("module start")
        if any(
            next_state["modules"][file]["status"] != "fulfilled" for file in source["dependencies"]
        ):
            _fail("dependencies have not completed")
        frame = facts.get("frames", {}).get(action["frameId"])
        promise = facts.get("promises", {}).get(action["executionPromiseId"])
        if (
            not isinstance(frame, Mapping)
            or frame.get("id") != action["frameId"]
            or frame.get("moduleFile") != action["file"]
            or frame.get("executionPromiseId") != action["executionPromiseId"]
            or any(
                item["frameId"] == action["frameId"]
                or (
                    action["executionPromiseId"] is not None
                    and item["executionPromiseId"] == action["executionPromiseId"]
                )
                for item in next_state["modules"].values()
            )
            or (
                source["hasTLA"]
                and (
                    not isinstance(promise, Mapping)
                    or promise.get("id") != action["executionPromiseId"]
                    or promise.get("status") != "pending"
                )
            )
        ):
            _fail("module execution frame/promise identity")
        row.update(
            status="running",
            frameId=action["frameId"],
            executionPromiseId=action["executionPromiseId"],
            startOrder=next_state["nextStartOrder"],
            completedDependencies=list(source["dependencies"]),
        )
        next_state["nextStartOrder"] += 1
    elif kind == "module.await":
        if (
            not _exact(action, {"type", "file", "frameId", "promiseId"})
            or row["status"] != "running"
            or row["frameId"] != action["frameId"]
            or not source["hasTLA"]
            or row["awaited"] is not None
            or not _id(action["promiseId"])
        ):
            _fail("module await")
        frame = facts.get("frames", {}).get(action["frameId"])
        if (
            not isinstance(frame, Mapping)
            or frame.get("id") != action["frameId"]
            or frame.get("moduleFile") != action["file"]
            or frame.get("executionPromiseId") != row["executionPromiseId"]
            or frame.get("awaitedPromiseId") != action["promiseId"]
        ):
            _fail("module await frame identity")
        promise = facts.get("promises", {}).get(action["promiseId"])
        if (
            not isinstance(promise, Mapping)
            or promise.get("id") != action["promiseId"]
            or promise.get("status") not in {"pending", "fulfilled", "rejected"}
        ):
            _fail("expected existing await promise")
        row["awaited"] = {"promiseId": action["promiseId"]}
        row["status"] = "suspended"
    elif kind == "module.resume":
        if (
            not _exact(action, {"type", "file", "frameId", "promiseId", "jobId"})
            or row["status"] != "suspended"
            or row["frameId"] != action["frameId"]
            or not isinstance(row["awaited"], Mapping)
            or row["awaited"].get("promiseId") != action["promiseId"]
            or not _id(action["jobId"])
        ):
            _fail("module resume")
        frame = facts.get("frames", {}).get(action["frameId"])
        if (
            not isinstance(frame, Mapping)
            or frame.get("id") != action["frameId"]
            or frame.get("moduleFile") != action["file"]
            or frame.get("executionPromiseId") != row["executionPromiseId"]
            or frame.get("awaitedPromiseId") != action["promiseId"]
        ):
            _fail("resume frame identity")
        promise = facts.get("promises", {}).get(action["promiseId"])
        if not isinstance(promise, Mapping) or promise.get("status") not in {
            "fulfilled",
            "rejected",
        }:
            _fail("awaited promise is unsettled")
        _settled_job(facts, action, promise["status"])
        # Rejection resumes the module body with a throw. A catch/finally may
        # handle it; only the execution completion establishes module failure.
        row["status"] = "running"
        row["awaited"] = None
    elif kind in {"module.complete", "module.reject"}:
        if (
            not _exact(action, {"type", "file", "frameId", "promiseId", "jobId"})
            or row["status"] not in {"running", "suspended"}
            or row["frameId"] != action["frameId"]
            or row["executionPromiseId"] != action["promiseId"]
            or not _id(action["jobId"])
        ):
            _fail("module settlement")
        if row["status"] == "suspended":
            _fail("cannot settle with unresolved await")
        rejecting = kind == "module.reject"
        status = "rejected" if rejecting else "fulfilled"
        if source["hasTLA"]:
            promise, job = _settled_job(facts, action, status)
        else:
            job = facts.get("jobs", {}).get(action["jobId"])
            frame = facts.get("frames", {}).get(action["frameId"])
            if (
                not isinstance(job, Mapping)
                or job.get("id") != action["jobId"]
                or job.get("promiseId") is not None
                or job.get("frameId") != action["frameId"]
                or job.get("status") != status
                or "value" not in job
                or not isinstance(frame, Mapping)
                or frame.get("id") != action["frameId"]
                or frame.get("moduleFile") != action["file"]
                or frame.get("executionPromiseId") is not None
                or frame.get("status") != "completed"
                or not isinstance(frame.get("completion"), Mapping)
                or frame["completion"].get("type") != ("throw" if rejecting else "return")
                or _canonical(frame["completion"].get("value")) != _canonical(job["value"])
            ):
                _fail("synchronous completion authority")
            promise = {"id": None, "value": job["value"]}
        row["status"] = "rejected" if rejecting else "fulfilled"
        row["awaited"] = None
        row["completionJobId"] = job["id"]
        expected_kind = (
            ("module-evaluation-failure" if rejecting else "module-execution-settle")
            if source["hasTLA"]
            else ("module-synchronous-failure" if rejecting else "module-synchronous-complete")
        )
        if (
            job.get("kind") != expected_kind
            or job.get("moduleFile") != action["file"]
            or "value" not in promise
            or "value" not in job
            or _canonical(promise["value"]) != _canonical(job["value"])
        ):
            _fail("module execution settlement authority")
        row["failure"] = (
            {"moduleFile": action["file"], "promiseId": promise["id"], "jobId": job["id"]}
            if rejecting
            else None
        )
    elif kind == "module.propagateFailure":
        if (
            not _exact(action, {"type", "file", "dependencyFile", "promiseId", "jobId"})
            or row["status"] != "waiting"
            or action["dependencyFile"] not in source["dependencies"]
            or action["dependencyFile"] in row["completedDependencies"]
            or next_state["modules"][action["dependencyFile"]]["status"] != "rejected"
            or next_state["modules"][action["dependencyFile"]]["failure"]["promiseId"]
            != action["promiseId"]
            or next_state["modules"][action["dependencyFile"]]["failure"]["jobId"]
            != action["jobId"]
            or (action["promiseId"] is not None and not _id(action["promiseId"]))
            or not _id(action["jobId"])
        ):
            _fail("dependency failure propagation")
        promise = facts.get("promises", {}).get(action["promiseId"])
        job = facts.get("jobs", {}).get(action["jobId"])
        root_file = next_state["modules"][action["dependencyFile"]]["failure"]["moduleFile"]
        asynchronous = next(item for item in plan["modules"] if item["file"] == root_file)["hasTLA"]
        if (
            asynchronous
            and (
                not isinstance(promise, Mapping)
                or promise.get("id") != action["promiseId"]
                or promise.get("status") != "rejected"
            )
            or not isinstance(job, Mapping)
            or job.get("id") != action["jobId"]
            or job.get("promiseId") != action["promiseId"]
            or job.get("status") != "rejected"
            or job.get("kind")
            != ("module-evaluation-failure" if asynchronous else "module-synchronous-failure")
            or job.get("moduleFile") != root_file
            or "value" not in job
            or asynchronous
            and ("value" not in promise or _canonical(promise["value"]) != _canonical(job["value"]))
        ):
            _fail("dependency propagation job identity")
        row["status"] = "rejected"
        row["failure"] = {
            "moduleFile": next_state["modules"][action["dependencyFile"]]["failure"]["moduleFile"],
            "promiseId": action["promiseId"],
            "jobId": job["id"],
        }
        row["completionJobId"] = job["id"]
        row["completedDependencies"].append(action["dependencyFile"])
        row["pendingDependencies"] -= 1
    else:
        _fail("unknown action type")
    if kind == "module.complete":
        for parent in plan["modules"]:
            parent_state = next_state["modules"][parent["file"]]
            if (
                action["file"] in parent["dependencies"]
                and action["file"] not in parent_state["completedDependencies"]
            ):
                parent_state["completedDependencies"].append(action["file"])
                parent_state["pendingDependencies"] -= 1
    _verify_state(plan, next_state)
    return next_state
