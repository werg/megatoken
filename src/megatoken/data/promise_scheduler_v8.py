"""Independent owned-Promise settlement and FIFO scheduling for shared v8 replay."""

from __future__ import annotations

import copy


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid owned Promise scheduler: " + message)


def rows(value):
    return dict(value) if isinstance(value, list) else value


def selected_callback(job):
    if job and job.get("promise"):
        if job.get("reaction") == "then" and job.get("originalStatus") == "rejected":
            return job.get("rejectionClosure")
        if job.get("reaction") == "catch" and job.get("originalStatus") == "fulfilled":
            return None
    return job.get("closure") if job else None


def selected_child(machine, identity, job):
    if job and job.get("reaction") == "adopt":
        for node in machine.get("heap", {}).values():
            row = node.get("adoption") if isinstance(node, dict) else None
            if row and row.get("job") == identity and row.get("phase") == "reading":
                lookup = row.get("lookup") or {}
                if lookup.get("child"):
                    return lookup["child"]
    return "job:" + identity


def is_scheduled_callback_frame(machine, frame):
    """Identify the retained ordinary callback frame whose parameters we validate below."""
    active = machine.get("activeJob")
    job = rows(machine.get("jobs", [])).get(active) if active is not None else None
    if not job or job.get("reaction") in ("adopt", "adoption"):
        return False
    callback = selected_callback(job)
    return bool(
        job.get("status") == "running"
        and callback
        and frame.get("child") == selected_child(machine, active, job)
        and frame.get("callee") == ["ref", callback]
    )


def _callback_aliases(machine, identity, job, callback, frame):
    """Rebuild callback captures plus source-bound formal cells from the dispatch receipt."""
    from .protocol_v8 import _known

    node = machine.get("heap", {}).get(callback)
    definition = job.get("callbackDefinition")
    bindings = job.get("parameterBindings")
    if not isinstance(node, dict) or node.get("kind") != "closure":
        _fail("scheduled callback is not its retained closure")
    if (
        not isinstance(definition, dict)
        or set(definition) != {"code", "name", "start", "params"}
        or definition.get("code") != node.get("code")
        or definition.get("name") is not None
        and (not isinstance(definition.get("name"), str) or not definition["name"])
        or not isinstance(definition.get("code"), str)
        or not definition["code"]
        or type(definition.get("start")) not in (int, type(None))
        or (definition.get("start") is not None and definition["start"] < 0)
        or not (
            definition["code"] == definition["name"]
            or definition["start"] is not None
            and definition["code"] == "function:" + str(definition["start"])
        )
        or not isinstance(definition.get("params"), list)
        or len(definition["params"]) > 256
        or any(
            value is not None and (not isinstance(value, str) or not value)
            for value in definition["params"]
        )
        or not isinstance(bindings, list)
    ):
        _fail("scheduled callback source parameter definition")
    expected_indices = [
        index for index, name in enumerate(definition["params"]) if name is not None
    ]
    if len(bindings) != len(expected_indices):
        _fail("scheduled callback parameter binding count")
    aliases = {
        row["binding"]: row["cell"] for row in node.get("captures", [])
    }
    for row, index in zip(bindings, expected_indices, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"binding", "index", "cell", "initialValue"}
            or row.get("binding") != definition["params"][index]
            or type(row.get("index")) is not int
            or row.get("index") != index
            or not isinstance(row.get("cell"), str)
            or not row["cell"]
            or len(row["cell"]) > 256
        ):
            _fail("scheduled callback parameter binding receipt")
        base = "j" + identity + "p" + str(index)
        suffix = row["cell"][len(base) :]
        if row["cell"] != base and (
            not suffix.startswith("x")
            or not suffix[1:].isdigit()
            or suffix[1:].startswith("0")
            or int(suffix[1:]) < 1
        ):
            _fail("scheduled callback parameter cell identity")
        expected = frame["arguments"][index] if index < len(frame["arguments"]) else ["undefined"]
        if row["initialValue"] != expected or row["cell"] not in machine.get("locals", {}):
            _fail("scheduled callback parameter does not retain its call argument")
        _known(machine, row["initialValue"])
        _known(machine, machine["locals"][row["cell"]])
        aliases[row["binding"]] = row["cell"]
    return aliases


def _saved_completion(machine, job):
    if "priorPending" not in job or "priorLastReturn" not in job:
        _fail("active scheduler job lost its saved caller completion")
    prior_pending = job["priorPending"]
    if prior_pending is not None:
        if (
            not isinstance(prior_pending, dict)
            or not {"kind", "value"}.issubset(prior_pending)
            or set(prior_pending) - {"kind", "value", "handler"}
            or prior_pending["kind"] not in ("return", "throw", "cleanup")
        ):
            _fail("saved scheduler caller completion shape")
        from .protocol_v8 import _known

        _known(machine, prior_pending["value"])
    prior_last_return = job["priorLastReturn"]
    if prior_last_return is not None:
        if (
            not isinstance(prior_last_return, dict)
            or set(prior_last_return) != {"kind", "value"}
            or prior_last_return["kind"] not in ("return", "throw")
        ):
            _fail("saved scheduler caller return shape")
        from .protocol_v8 import _known

        _known(machine, prior_last_return["value"])


def _validate_callback_frame(machine, identity, job, callback, *, continuation=None):
    from .protocol_v8 import _known

    root = next(iter(machine.get("frameFunctions", {})), None)
    child = selected_child(machine, identity, job)
    frame = next(
        (row for row in machine["protocol"].get("callFrames", []) if row.get("child") == child),
        None,
    )
    if not job or job.get("status") != "running" or job.get("owner") != root or callback is None:
        _fail("active scheduler job lacks its selected callback")
    _saved_completion(machine, job)
    if job.get("returned") is not None:
        returned = job["returned"]
        if (
            set(returned) != {"kind", "value"}
            or returned["kind"] not in ("return", "throw")
            or frame is not None
            or child in machine.get("stack", [])
        ):
            _fail("completed scheduler callback retains an active call frame")
        _known(machine, returned["value"])
        return
    expected_arguments = (
        [job.get("originalResult")]
        if job.get("promise") and job.get("reaction") != "finally"
        else []
    )
    if (
        not frame
        or frame.get("parent") != root
        or frame.get("callee") != ["ref", callback]
        or frame.get("arguments") != expected_arguments
        or frame.get("priorReceiver") != job.get("priorReceiver")
        or frame.get("priorAliases") != job.get("priorAliases")
    ):
        _fail("active scheduler callback differs from its retained call frame")
    aliases = _callback_aliases(machine, identity, job, callback, frame)
    if continuation is not None:
        if (
            continuation.get("state") not in ("waiting", "ready")
            or continuation.get("invocation") != child
            or child in machine.get("stack", [])
            or root not in machine.get("stack", [])
            or continuation.get("captureAliases") != [[key, value] for key, value in aliases.items()]
            or continuation.get("receiver")
            != machine.get("locals", {}).get(child + ":this")
            or machine.get("captureAliases", {})
            != dict(job.get("priorAliases", []))
            or machine.get("receiver") != job.get("priorReceiver")
        ):
            _fail("suspended scheduler callback lost its retained scope")
    else:
        if (
            child not in machine.get("stack", [])
            or machine["stack"].index(child) != machine["stack"].index(root) + 1
        ):
            _fail("active scheduler callback differs from its retained call frame")
        if machine["stack"][-1] == child:
            if machine.get("captureAliases", {}) != aliases:
                _fail("active scheduler callback parameter aliases")
            if machine.get("receiver") != machine.get("locals", {}).get(child + ":this"):
                _fail("active scheduler callback receiver")


def validate_active_callback_state(machine):
    protocol = machine.get("protocol")
    active = machine.get("activeJob")
    if protocol is None:
        return True
    jobs = rows(machine.get("jobs", []))
    if active is not None:
        job = jobs.get(active)
        if not job or job.get("reaction") not in ("adopt", "adoption"):
            _validate_callback_frame(machine, active, job, selected_callback(job))
    for continuation in rows(machine.get("continuations", [])).values():
        identity = continuation.get("activeJob")
        if (
            identity is None
            or continuation.get("state") not in ("waiting", "ready")
            or identity == active
        ):
            continue
        job = jobs.get(identity)
        if not job or job.get("reaction") in ("adopt", "adoption"):
            continue
        _validate_callback_frame(
            machine,
            identity,
            job,
            selected_callback(job),
            continuation=continuation,
        )
    return True


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
    row = {
        "kind": "promise",
        "properties": {},
        "state": "pending",
        "result": None,
        "adopted": None,
        "external": None,
        "reactions": [],
    }
    if (machine.get("protocol") or {}).get("promiseObjects") is not None:
        from .promise_object_v8 import intrinsic_id

        row["prototype"] = ["ref", intrinsic_id("Promise.prototype")]
    machine["heap"][identity] = row


def assert_plain_resolution(machine, value):
    """Prove that a reference cannot invoke a then getter or assimilation job."""
    from .generator_v8 import INTRINSICS as GENERATOR_INTRINSICS
    from .protocol_v8 import _descriptor, _lookup_descriptor, _node

    if value[0] != "ref":
        return
    target = machine["heap"].get(value[1])
    if target and target.get("kind") == "promise":
        _fail("Promise adoption requires explicit assimilation jobs")
    current, seen = value[1], set()
    while current is not None:
        if current in seen:
            _fail("cyclic then lookup")
        seen.add(current)
        node = _node(machine, current)
        if _descriptor(node, "then") is not None:
            break
        prototype = node.get("prototype", "ordinary")
        if prototype is None or prototype == "ordinary":
            break
        if isinstance(prototype, str) and prototype in GENERATOR_INTRINSICS:
            break  # The shared property resolver checks this declared intrinsic.
        if not (isinstance(prototype, list) and len(prototype) == 2 and prototype[0] == "ref"):
            _fail("unmodeled prototype in then lookup")
        current = prototype[1]
    _, descriptor = _lookup_descriptor(machine, value[1], "then")
    if descriptor is None:
        return
    if descriptor["kind"] != "data":
        _fail("then getter requires an explicit resolution protocol")
    method = descriptor["value"]
    if method[0] == "ref":
        selected = machine["heap"].get(method[1])
        if (
            not selected
            or selected.get("kind") in ("closure", "proxy")
            or (
                selected.get("kind") == "intrinsic"
                and selected.get("callableKind") in ("function", "constructor")
            )
        ):
            _fail("callable then requires explicit assimilation jobs")
    elif method[0] not in ("undefined", "null", "bool", "num", "str", "bigint", "symbol"):
        _fail("unmodeled then value requires an explicit resolution protocol")


def settle_promise(machine, identity, status, value, *, external=False, refresh=True):
    from .protocol_v8 import _known

    node = promise(machine, identity)
    if node["state"] != "pending" or node.get("adopted") or status not in ("fulfilled", "rejected"):
        _fail("settlement requires a pending Promise without an adoption lock")
    if node.get("external") and not external:
        _fail("external Promise requires an observed settlement")
    _known(machine, value)
    if status == "fulfilled":
        assert_plain_resolution(machine, value)
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
            _fail("legacy adoption lock has no assimilation authority")
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
        if (
            node.get("generatorFunction")
            or node.get("promiseResolver")
            or node.get("asyncFunction")
        ):
            _fail("generator, async and resolving callbacks require their explicit call protocol")
        capture = node.get("callable", {}).get("lexicalConstruction")
        if capture is not None:
            if machine.get("protocol") is None:
                _fail("constructor-captured callbacks require an authenticated protocol frame")
            from .construction_v8 import _construction_capture

            _construction_capture(machine, capture)
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
            "pending": copy.deepcopy(machine.get("pending")),
        }
        machine["pending"] = None
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
        owner = continuation.get("activeJob") if continuation else None
        root = next(iter(machine.get("frameFunctions", {})), None)
        suspended_job_root = bool(owner) and machine["stack"] == [root]
        if (
            machine["stack"] and not suspended_job_root
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
        machine["pending"] = copy.deepcopy(continuation["pending"])
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
                    parameter_definitions = function.get("params", [])
                    if (
                        not isinstance(parameter_definitions, list)
                        or len(parameter_definitions) > 256
                        or any(
                            name is not None and (not isinstance(name, str) or not name)
                            for name in parameter_definitions
                        )
                        or function.get("name") is not None
                        and (not isinstance(function.get("name"), str) or not function["name"])
                        or function.get("start") is not None
                        and (type(function["start"]) is not int or function["start"] < 0)
                    ):
                        _fail("callback has malformed immutable parameter definitions")
                    job["callbackDefinition"] = {
                        "code": node["code"],
                        "name": function["name"],
                        "start": function.get("start"),
                        "params": copy.deepcopy(parameter_definitions),
                    }
                    child = "job:" + identity
                    if machine.get("protocol") is not None:
                        from .construction_v8 import _call
                        job["priorPending"] = copy.deepcopy(machine.get("pending"))
                        job["priorLastReturn"] = copy.deepcopy(machine.get("lastReturn"))
                        machine["pending"] = None
                        machine["lastReturn"] = None
                        if not root:
                            _fail("scheduled callback has no invocation owner")
                        machine["stack"].append(root)
                        arguments = [argument] if job.get("promise") and job.get("reaction") != "finally" else []
                        from .construction_v8 import _lexical_receiver

                        _call(
                            machine,
                            child,
                            ["ref", callback],
                            _lexical_receiver(
                                machine, node, node.get("lexicalReceiver", ["undefined"])
                            ),
                            arguments,
                        )
                        frame = machine["protocol"]["callFrames"][-1]
                        job["priorAliases"] = copy.deepcopy(frame["priorAliases"])
                        job["priorReceiver"] = copy.deepcopy(frame["priorReceiver"])
                    else:
                        job["priorAliases"] = list(map(list, machine["captureAliases"].items()))
                        machine["captureAliases"] = {
                            capture["binding"]: capture["cell"] for capture in node.get("captures", [])
                        }
                        machine["stack"].append(child)
                        machine["frameFunctions"][child] = node["code"]
                    parameter_bindings = []
                    for index, name in enumerate(parameter_definitions):
                        if name is None:
                            continue
                        base, ordinal = "j" + identity + "p" + str(index), 0
                        cell = base
                        while cell in machine["locals"]:
                            ordinal += 1
                            cell = base + "x" + str(ordinal)
                        machine["captureAliases"][name] = cell
                        initial_value = copy.deepcopy(argument if index == 0 else ["undefined"])
                        machine["locals"][cell] = initial_value
                        parameter_bindings.append(
                            {
                                "binding": name,
                                "index": index,
                                "cell": cell,
                                "initialValue": initial_value,
                            }
                        )
                    job["parameterBindings"] = parameter_bindings
                    machine["activeJob"] = identity
            elif action == "complete":
                if job["status"] != "running":
                    _fail("completion requires a running job")
                if any(
                    continuation.get("activeJob") == identity and continuation["state"] != "resumed"
                    for continuation in continuations.values()
                ):
                    _fail("callback has an unresumed continuation")
                if machine["stack"] or machine.get("activeJob") not in (None, identity):
                    _fail("callback frame has not completed")
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
                    if "priorPending" in job:
                        machine["pending"] = copy.deepcopy(job["priorPending"])
                    if "priorLastReturn" in job:
                        machine["lastReturn"] = copy.deepcopy(job["priorLastReturn"])
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
