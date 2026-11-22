import copy

import pytest

from megatoken.data.module_scheduler_state_v8 import (
    apply_module_scheduler_action_v8,
    build_module_scheduler_plan_v8,
    create_module_scheduler_state_v8,
)

META = {
    "profile": "evaluated-module/1",
    "evaluationStrategy": "async-dependency-scheduler",
    "entryModule": "main.js",
    "modules": [
        {"file": "shared.js", "source": "export const x=1;"},
        {"file": "left.js", "source": "import {x} from './shared.js'; await 0;"},
        {"file": "right.js", "source": "import {x} from './shared.js'; await 0;"},
        {"file": "main.js", "source": "import './left.js'; import './right.js';"},
    ],
    "asyncDependencies": [
        {"file": "shared.js", "index": 0, "hasTLA": False, "dependencies": []},
        {"file": "left.js", "index": 1, "hasTLA": True, "dependencies": [0]},
        {"file": "right.js", "index": 2, "hasTLA": True, "dependencies": [0]},
        {"file": "main.js", "index": 3, "hasTLA": False, "dependencies": [1, 2]},
    ],
}


def _facts(file, frame, execution):
    return {
        "frames": {frame: {"id": frame, "moduleFile": file, "executionPromiseId": execution}},
        "promises": {execution: {"id": execution, "status": "pending"}},
        "jobs": {},
    }


def _await_facts(file, frame, execution, promise, status="pending", job=None):
    return {
        "frames": {frame: {"id": frame, "moduleFile": file,
                           "executionPromiseId": execution, "awaitedPromiseId": promise}},
        "promises": {promise: {"id": promise, "status": status,
                               **({"value": None} if status != "pending" else {})}},
        "jobs": ({job: {"id": job, "promiseId": promise, "frameId": frame,
                        "status": status, "kind": "promise-reaction",
                        "moduleFile": file, "value": None}} if job else {}),
    }


def _complete(plan, state, file, n):
    frame, promise, job = f"f{n}", f"p{n}", f"j{n}"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.start", "file": file,
                      "frameId": frame, "executionPromiseId": promise},
        _facts(file, frame, promise),
    )
    return apply_module_scheduler_action_v8(
        plan, state, {"type": "module.complete", "file": file,
                      "frameId": frame, "promiseId": promise, "jobId": job},
        {"frames": {}, "promises": {promise: {"id": promise, "status": "fulfilled", "value": None}},
         "jobs": {job: {"id": job, "promiseId": promise, "frameId": frame,
                        "status": "fulfilled", "kind": "module-execution-settle",
                        "moduleFile": file, "value": None}}},
    )


def test_dag_start_order_is_derived_and_shared_dependency_completes_once():
    plan = build_module_scheduler_plan_v8(META)
    state = create_module_scheduler_state_v8(plan)
    state = _complete(plan, state, "shared.js", 0)
    assert state["nextStartOrder"] == 1
    assert state["modules"]["shared.js"]["status"] == "fulfilled"
    assert state["modules"]["left.js"]["status"] == "waiting"
    assert state["modules"]["right.js"]["status"] == "waiting"


def test_resume_requires_same_existing_pending_promise_and_reaction_job():
    plan = build_module_scheduler_plan_v8(META)
    state = _complete(plan, create_module_scheduler_state_v8(plan), "shared.js", 0)
    frame, execution, awaited = "f1", "exec1", "await1"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.start", "file": "left.js",
                      "frameId": frame, "executionPromiseId": execution},
        _facts("left.js", frame, execution),
    )
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.await", "file": "left.js",
                      "frameId": frame, "promiseId": awaited},
        _await_facts("left.js", frame, execution, awaited),
    )
    with pytest.raises(ValueError, match="existing promise job"):
        apply_module_scheduler_action_v8(
            plan, state, {"type": "module.resume", "file": "left.js",
                          "frameId": frame, "promiseId": awaited, "jobId": "forged"},
            _await_facts("left.js", frame, execution, awaited, "fulfilled", None),
        )
    assert state["modules"]["left.js"]["status"] == "suspended"


def test_rejected_dependency_propagates_without_settling_independent_sibling():
    plan = build_module_scheduler_plan_v8(META)
    state = _complete(plan, create_module_scheduler_state_v8(plan), "shared.js", 0)

    lf, le, la = "left-frame", "left-exec", "left-await"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.start", "file": "left.js",
                      "frameId": lf, "executionPromiseId": le},
        _facts("left.js", lf, le),
    )
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.await", "file": "left.js",
                      "frameId": lf, "promiseId": la},
        _await_facts("left.js", lf, le, la),
    )
    rf, re, ra = "right-frame", "right-exec", "right-await"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.start", "file": "right.js",
                      "frameId": rf, "executionPromiseId": re},
        _facts("right.js", rf, re),
    )
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.await", "file": "right.js",
                      "frameId": rf, "promiseId": ra},
        _await_facts("right.js", rf, re, ra),
    )

    reject_job = "left-reaction"
    mismatch = _await_facts("left.js", lf, le, la, "rejected", reject_job)
    mismatch["jobs"][reject_job]["value"] = "wrong-reason"
    with pytest.raises(ValueError, match="outcome mismatch"):
        apply_module_scheduler_action_v8(
            plan, state, {"type": "module.resume", "file": "left.js",
                          "frameId": lf, "promiseId": la, "jobId": reject_job},
            mismatch,
        )
    assert state["modules"]["left.js"]["status"] == "suspended"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.resume", "file": "left.js",
                      "frameId": lf, "promiseId": la, "jobId": reject_job},
        _await_facts("left.js", lf, le, la, "rejected", reject_job),
    )
    exec_job = "left-exec-reject"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.reject", "file": "left.js",
                      "frameId": lf, "promiseId": le, "jobId": exec_job},
        {"promises": {le: {"id": le, "status": "rejected", "value": None}},
         "jobs": {exec_job: {"id": exec_job, "promiseId": le, "frameId": lf,
                             "status": "rejected", "kind": "module-evaluation-failure",
                             "moduleFile": "left.js", "value": None}}},
    )
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.propagateFailure", "file": "main.js",
                      "dependencyFile": "left.js", "promiseId": le,
                      "jobId": exec_job},
        {"promises": {le: {"id": le, "status": "rejected", "value": None}},
         "jobs": {exec_job: {"id": exec_job, "promiseId": le,
                             "frameId": lf, "status": "rejected", "kind": "module-evaluation-failure",
                             "moduleFile": "left.js", "value": None}}},
    )
    assert state["modules"]["main.js"]["status"] == "rejected"
    assert state["modules"]["right.js"]["status"] == "suspended"

    right_job = "right-reaction"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.resume", "file": "right.js",
                      "frameId": rf, "promiseId": ra, "jobId": right_job},
        _await_facts("right.js", rf, re, ra, "fulfilled", right_job),
    )
    right_done = "right-complete"
    state = apply_module_scheduler_action_v8(
        plan, state, {"type": "module.complete", "file": "right.js",
                      "frameId": rf, "promiseId": re, "jobId": right_done},
        {"promises": {re: {"id": re, "status": "fulfilled", "value": None}},
         "jobs": {right_done: {"id": right_done, "promiseId": re,
                               "frameId": rf, "status": "fulfilled",
                               "kind": "module-execution-settle",
                               "moduleFile": "right.js", "value": None}}},
    )
    assert state["modules"]["right.js"]["status"] == "fulfilled"
    assert state["modules"]["main.js"]["status"] == "rejected"

def test_plan_rejects_cycle_and_tampered_digest():
    cyclic = {**META, "asyncDependencies": [dict(row) for row in META["asyncDependencies"]]}
    cyclic["asyncDependencies"][0]["dependencies"] = [1]
    with pytest.raises(ValueError, match="cyclic"):
        build_module_scheduler_plan_v8(cyclic)
    plan = build_module_scheduler_plan_v8(META)
    plan["modules"][0]["hasTLA"] = True
    from megatoken.data.module_scheduler_state_v8 import verify_module_scheduler_plan_v8
    with pytest.raises(ValueError, match="digest"):
        verify_module_scheduler_plan_v8(plan)

def test_early_and_duplicate_completion_are_refused_without_mutation():
    plan = build_module_scheduler_plan_v8(META)
    initial = create_module_scheduler_state_v8(plan)
    frame, execution, job = "shared-frame", "shared-exec", "shared-job"
    pending = _facts("shared.js", frame, execution)
    with pytest.raises(ValueError, match="module settlement"):
        apply_module_scheduler_action_v8(
            plan, initial, {"type": "module.complete", "file": "shared.js",
                            "frameId": frame, "promiseId": execution, "jobId": job},
            pending,
        )
    assert initial["modules"]["shared.js"]["status"] == "waiting"
    malformed = copy.deepcopy(initial)
    malformed["nextStartOrder"] = 1
    malformed["modules"]["left.js"].update(
        status="running", startOrder=0, frameId="forged-frame",
        executionPromiseId="forged-exec"
    )
    with pytest.raises(ValueError, match="module state"):
        apply_module_scheduler_action_v8(
            plan, malformed, {"type": "module.start", "file": "shared.js",
                              "frameId": frame, "executionPromiseId": execution},
            pending,
        )
    assert initial["modules"]["left.js"]["status"] == "waiting"
    completed = _complete(plan, initial, "shared.js", 0)
    terminal = copy.deepcopy(completed)
    terminal["modules"]["shared.js"]["awaited"] = {"promiseId": "forged-await"}
    with pytest.raises(ValueError, match="module state"):
        apply_module_scheduler_action_v8(
            plan, terminal, {"type": "module.start", "file": "left.js",
                             "frameId": frame, "executionPromiseId": execution},
            pending,
        )
    with pytest.raises(ValueError, match="module start"):
        apply_module_scheduler_action_v8(
            plan, completed, {"type": "module.start", "file": "shared.js",
                              "frameId": frame, "executionPromiseId": execution},
            pending,
        )
    assert completed["modules"]["shared.js"]["status"] == "fulfilled"
    with pytest.raises(ValueError, match="module settlement"):
        apply_module_scheduler_action_v8(
            plan, completed, {"type": "module.complete", "file": "shared.js",
                              "frameId": "f0", "promiseId": "p0", "jobId": "j0"},
            {"promises": {"p0": {"id": "p0", "status": "fulfilled"}},
             "jobs": {"j0": {"id": "j0", "promiseId": "p0",
                             "frameId": "f0", "status": "fulfilled"}}},
        )
