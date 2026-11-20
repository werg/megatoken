"""Independent owned-Promise settlement and FIFO scheduling for shared v8 replay."""

from __future__ import annotations

import copy


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid owned Promise scheduler: " + message)


def rows(value):
    return dict(value) if isinstance(value, list) else value


def store_rows(machine, field, values):
    machine[field] = [[identity, value] for identity, value in values.items()]


def promise(machine, identity):
    node = machine["heap"].get(identity)
    if not node or node.get("kind") != "promise":
        _fail("unknown Promise identity")
    if node.get("state") not in ("pending", "fulfilled", "rejected"):
        _fail("invalid Promise state")
    if (node["state"] == "pending") != (node.get("result") is None):
        _fail("invalid Promise settlement")
    return node


def create_promise(machine, identity):
    from .protocol_v8 import _assert_fresh_heap_identity

    _assert_fresh_heap_identity(machine, identity)
    machine["heap"][identity] = {
        "kind": "promise",
        "properties": {},
        "state": "pending",
        "result": None,
        "adopted": None,
        "external": None,
        "reactions": [],
    }


def settle_promise(machine, identity, status, value, *, external=False, refresh=True):
    from .protocol_v8 import _known

    node = promise(machine, identity)
    if node["state"] != "pending" or node.get("adopted") or status not in ("fulfilled", "rejected"):
        _fail("settlement requires a pending Promise without an adoption lock")
    if node.get("external") and not external:
        _fail("external Promise requires an observed settlement")
    _known(machine, value)
    if status == "fulfilled" and value[0] == "ref":
        target = machine["heap"].get(value[1])
        if target and target.get("kind") == "promise":
            # Resolving to a Promise schedules assimilation work before its
            # reactions become runnable. Eagerly propagating a settlement and
            # sorting by registration order cannot represent that FIFO queue.
            _fail("Promise adoption requires explicit assimilation jobs")
    node["state"], node["result"] = status, copy.deepcopy(value)
    if refresh:
        refresh_ready(machine)


def refresh_ready(machine):
    jobs = rows(machine.get("jobs", []))
    continuations = rows(machine.get("continuations", []))
    ready = machine.setdefault("ready", [])
    scheduler = machine.setdefault("scheduler", {"registration": 0, "ready": 0})
    while True:
        if any(
            node.get("kind") == "promise" and node.get("adopted")
            for node in machine["heap"].values()
        ):
            _fail("Promise adoption requires explicit assimilation jobs")
        eligible = []
        for identity, continuation in continuations.items():
            if (
                continuation["state"] == "waiting"
                and promise(machine, continuation["promise"])["state"] != "pending"
            ):
                eligible.append((continuation["sequence"], identity, continuation, "state"))
        for identity, job in jobs.items():
            if (
                job["status"] == "waiting"
                and job.get("promise")
                and promise(machine, job["promise"])["state"] != "pending"
            ):
                eligible.append((job["sequence"], identity, job, "status"))
        for _, identity, item, field in sorted(eligible, key=lambda item: item[0]):
            item[field] = "ready" if field == "state" else "queued"
            scheduler["ready"] += 1
            item["readySequence"] = scheduler["ready"]
            ready.append(identity)
        # A finally callback may itself return a Promise. Its settlement can
        # make further jobs ready, so process one consequence before rescanning.
        completed_cleanup = False
        for job in jobs.values():
            if job["status"] != "waiting-cleanup":
                continue
            cleanup = promise(machine, job["cleanup"])
            if cleanup["state"] == "pending":
                continue
            job["status"] = "completed"
            if job.get("target"):
                rejected = cleanup["state"] == "rejected"
                settle_promise(
                    machine,
                    job["target"],
                    "rejected" if rejected else job["originalStatus"],
                    cleanup["result"] if rejected else job["originalResult"],
                    refresh=False,
                )
            completed_cleanup = True
            break
        if not completed_cleanup:
            break
    store_rows(machine, "jobs", jobs)
    if "continuations" in machine or continuations:
        store_rows(machine, "continuations", continuations)


def apply_detail(machine, kind, fields, *, root, functions=()):
    jobs = rows(machine.get("jobs", []))
    continuations = rows(machine.get("continuations", []))
    scheduler = machine.setdefault("scheduler", {"registration": 0, "ready": 0})
    ready = machine.setdefault("ready", [])

    def fresh_job(identity):
        if identity in jobs or identity in continuations:
            _fail("duplicate job or continuation identity")

    def closure(identity):
        node = machine["heap"].get(identity)
        if not node or node.get("kind") != "closure":
            _fail("callback requires an owned closure")
        if node.get("generatorFunction") or node.get("promiseResolver"):
            _fail("generator and resolving callbacks require their explicit call protocol")
        if any(capture["cell"] not in machine["locals"] for capture in node.get("captures", [])):
            _fail("callback capture is not initialized")
        return node

    if kind == "promise":
        identity, action = fields["promise"], fields["action"]
        if action == "create":
            create_promise(machine, identity)
        elif action in ("resolve", "reject", "adopt"):
            if action == "adopt":
                promise(machine, fields["target"])
            settle_promise(
                machine,
                identity,
                "rejected" if action == "reject" else "fulfilled",
                ["ref", fields["target"]] if action == "adopt" else fields["value"],
            )
        elif action == "react":
            promise(machine, identity)
            fresh_job(fields["job"])
            for name in ("closure", "rejectionClosure"):
                if fields.get(name):
                    closure(fields[name])
            if fields.get("target"):
                create_promise(machine, fields["target"])
            scheduler["registration"] += 1
            jobs[fields["job"]] = {
                "status": "waiting",
                "owner": root,
                "promise": identity,
                "reaction": fields["reaction"],
                "closure": fields.get("closure"),
                "rejectionClosure": fields.get("rejectionClosure"),
                "target": fields.get("target"),
                "sequence": scheduler["registration"],
            }
            store_rows(machine, "jobs", jobs)
            refresh_ready(machine)
        else:
            _fail("unsupported Promise action")
    elif kind == "await":
        promise(machine, fields["promise"])
        fresh_job(fields["continuation"])
        if not machine["stack"] or machine["stack"][-1] != fields["invocation"]:
            _fail("suspension invocation mismatch")
        if fields.get("returnPromise"):
            promise(machine, fields["returnPromise"])
            if len(machine["stack"]) != 1 or machine.get("callReturn"):
                _fail("async call return requires first root suspension")
            machine["callReturn"] = {"kind": "return", "value": ["ref", fields["returnPromise"]]}
        scheduler["registration"] += 1
        continuations[fields["continuation"]] = {
            "state": "waiting",
            "promise": fields["promise"],
            "invocation": machine["stack"].pop(),
            "sequence": scheduler["registration"],
            "activeJob": machine.get("activeJob"),
            "captureAliases": [[key, value] for key, value in machine["captureAliases"].items()],
        }
        if machine.get("protocol") is not None:
            continuations[fields["continuation"]]["receiver"] = copy.deepcopy(machine["receiver"])
        if machine.get("activeJob"):
            suspended = jobs[machine["activeJob"]]
            machine["captureAliases"] = dict(suspended["priorAliases"])
            if machine.get("protocol") is not None and "priorReceiver" in suspended:
                machine["receiver"] = copy.deepcopy(suspended["priorReceiver"])
            machine["activeJob"] = None
        store_rows(machine, "continuations", continuations)
        refresh_ready(machine)
    elif kind == "resume":
        continuation = continuations.get(fields["continuation"])
        if (
            machine["stack"]
            or ready[:1] != [fields["continuation"]]
            or not continuation
            or continuation["state"] != "ready"
            or continuation["invocation"] != fields["invocation"]
        ):
            _fail("resume violates continuation ownership or FIFO order")
        target = promise(machine, continuation["promise"])
        if target["state"] != fields["status"] or target["result"] != fields["result"]:
            _fail("resume contradicts its Promise settlement")
        ready.pop(0)
        continuation["state"] = "resumed"
        machine["stack"].append(continuation["invocation"])
        machine["activeJob"] = continuation["activeJob"]
        machine["captureAliases"] = dict(continuation["captureAliases"])
        if machine.get("protocol") is not None and "receiver" in continuation:
            machine["receiver"] = copy.deepcopy(continuation["receiver"])
    elif kind == "job":
        identity, action = fields["job"], fields["action"]
        if action == "enqueue":
            fresh_job(identity)
            if fields["owner"] != root:
                _fail("job owner differs from invocation domain")
            if fields.get("closure"):
                closure(fields["closure"])
            scheduler["ready"] += 1
            jobs[identity] = {
                "status": "queued",
                "owner": root,
                "closure": fields.get("closure"),
                "readySequence": scheduler["ready"],
            }
            ready.append(identity)
            store_rows(machine, "jobs", jobs)
        else:
            job = jobs.get(identity)
            if not job or job.get("owner") != fields["owner"]:
                _fail("job owner mismatch")
            if job.get("reaction") == "join":
                _fail("join jobs require their explicit protocol step")
            if action == "dispatch":
                if machine["stack"] or ready[:1] != [identity] or job["status"] != "queued":
                    _fail("dispatch violates stack turn or FIFO order")
                ready.pop(0)
                job["status"] = "running"
                callback, argument = job.get("closure"), ["undefined"]
                if job.get("promise"):
                    target = promise(machine, job["promise"])
                    if target["state"] == "pending":
                        _fail("reaction Promise is still pending")
                    job["originalStatus"] = target["state"]
                    job["originalResult"] = copy.deepcopy(target["result"])
                    if job["reaction"] == "then" and target["state"] == "rejected":
                        callback = job.get("rejectionClosure")
                    elif job["reaction"] == "catch" and target["state"] == "fulfilled":
                        callback = None
                    if job["reaction"] != "finally":
                        argument = target["result"]
                if callback:
                    node = closure(callback)
                    function = next(
                        (
                            fn
                            for fn in functions
                            if fn.get("name") == node["code"]
                            or "function:" + str(fn.get("start")) == node["code"]
                        ),
                        None,
                    )
                    if function is None:
                        _fail("callback has no immutable source function")
                    job["priorAliases"] = list(map(list, machine["captureAliases"].items()))
                    if machine.get("protocol") is not None:
                        job["priorReceiver"] = copy.deepcopy(machine["receiver"])
                        machine["receiver"] = copy.deepcopy(
                            node.get("lexicalReceiver")
                            if node.get("functionKind") == "arrow"
                            else ["undefined"]
                        )
                    machine["captureAliases"] = {
                        capture["binding"]: capture["cell"] for capture in node.get("captures", [])
                    }
                    for index, name in enumerate(function.get("params", [])):
                        if not name:
                            continue
                        base, ordinal = "j" + identity + "p" + str(index), 0
                        cell = base
                        while cell in machine["locals"]:
                            ordinal += 1
                            cell = base + "x" + str(ordinal)
                        machine["captureAliases"][name] = cell
                        machine["locals"][cell] = copy.deepcopy(
                            argument if index == 0 else ["undefined"]
                        )
                    machine["activeJob"] = identity
                    machine["stack"].append("job:" + identity)
                    machine["frameFunctions"]["job:" + identity] = node["code"]
            elif action == "complete":
                if job["status"] != "running":
                    _fail("completion requires a running job")
                if machine["stack"] or machine.get("activeJob") not in (None, identity):
                    _fail("callback frame has not completed")
                if any(
                    continuation.get("activeJob") == identity and continuation["state"] != "resumed"
                    for continuation in continuations.values()
                ):
                    _fail("callback has an unresumed continuation")
                returned = job.get("returned")
                if (
                    job.get("target")
                    and job.get("reaction") == "finally"
                    and returned
                    and returned["kind"] == "return"
                ):
                    _fail("normal finally completion requires explicit cleanup jobs")
                job["status"] = "completed"
                if machine.get("activeJob") == identity:
                    machine["captureAliases"] = dict(job["priorAliases"])
                    if machine.get("protocol") is not None and "priorReceiver" in job:
                        machine["receiver"] = copy.deepcopy(job["priorReceiver"])
                    machine["activeJob"] = None
                returned = job.get("returned")
                if job.get("promise") and job.get("target"):
                    store_rows(machine, "jobs", jobs)
                    if returned and returned["kind"] == "throw":
                        settle_promise(machine, job["target"], "rejected", returned["value"])
                    elif (
                        job["reaction"] == "finally"
                        and returned
                        and returned["value"][0] == "ref"
                        and machine["heap"].get(returned["value"][1], {}).get("kind") == "promise"
                    ):
                        job["status"], job["cleanup"] = "waiting-cleanup", returned["value"][1]
                        refresh_ready(machine)
                    elif job["reaction"] == "finally" or not returned:
                        settle_promise(
                            machine, job["target"], job["originalStatus"], job["originalResult"]
                        )
                    else:
                        settle_promise(machine, job["target"], "fulfilled", returned["value"])
                elif not job.get("promise") and returned and returned["kind"] == "throw":
                    machine.setdefault("jobErrors", []).append(
                        {"job": identity, "value": copy.deepcopy(returned["value"])}
                    )
            else:
                _fail("unsupported job action")
    else:
        return False
    return True
