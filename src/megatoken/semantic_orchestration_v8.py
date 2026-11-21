"""Bounded V8 coordinator authority on actual shared native child tracks.

The offered public cut is source/input ingress, not an admission receipt. Every
model action remains private until exact acknowledgement. This first profile
owns declared root children; dynamic callback and descendant admission is not
silently substituted for a typed child binding.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict

from megatoken.controls import canonical_semantic_observation
from megatoken.data.prefix_v8 import validate_v8_committed_prefix
from megatoken.data.semantic_json import stable_algorithm_hash, stable_algorithm_json
from megatoken.execution_context import execution_binding
from megatoken.io import strict_json_loads

VOLATILE_EVIDENCE = {"status", "generationSeconds", "preparationSeconds", "totalSeconds"}


def deterministic_selection(selection):
    return {key: value for key, value in selection.items() if key not in VOLATILE_EVIDENCE}


def _fail(message):
    from megatoken.semantic_model_v8 import NativeSelectionV8Error

    raise NativeSelectionV8Error(message)


def _description(value):
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _same(left, right):
    return stable_algorithm_json(left) == stable_algorithm_json(right)


def validate_ingress(session, public, *, max_active, child_budget):
    checked = validate_v8_committed_prefix(public, [])
    if (
        checked["session"] != session
        or type(max_active) is not int
        or not 1 <= max_active <= 16
        or type(child_budget) is not int
        or not 1 <= child_budget <= 100000
    ):
        _fail("V8 coordinator requires its bounded session and child work budget")
    programs, children = {}, []
    for row in public["timeline"]:
        if row["mode"] != "observed" or row["kind"] not in {"program", "invocation"}:
            _fail("V8 coordinator ingress accepts observed source and root inputs only")
        fields = row["fields"]
        if row["kind"] == "program":
            programs[fields["programId"]] = row
        else:
            if (
                fields["activation"] != "call"
                or fields["parentInvocationId"] is not None
                or fields["closureId"] is not None
                or fields["taskRevision"] != 1
            ):
                _fail("V8 coordinator profile requires declared root call invocations")
            program = programs[fields["programId"]]
            execution_binding(program, row)
            children.append({"program": program, "invocation": row})
    if not 1 <= len(children) <= 16:
        _fail("V8 coordinator requires one to sixteen declared root invocations")
    return copy.deepcopy(children)


def _work(adapter, session, track=None):
    work = adapter.sessions.get(session)
    if (
        work is None
        or work.orchestration is None
        or (track is not None and track != work.orchestration["track"])
    ):
        _fail("Unknown V8 coordinator session or track")
    return work


def _live(work):
    current = work.runtime.tracks.tracks[work.orchestration["track"]]
    if current.status in {"completed", "cancelled"}:
        _fail("Terminal V8 coordinator cannot admit or execute further work")


def _child(work, invocation):
    return next(
        (
            item
            for item in work.orchestration["children"]
            if item["execution"]["invocation_id"] == invocation
        ),
        None,
    )


def _adopt(adapter, work, branch):
    work.runtime = branch
    adapter.runtime.memory.sessions[branch.memory.session_id] = branch.memory


def open_orchestration(adapter, *, session, public, maxActive=1, childBudget=128, origin="model"):
    from megatoken.semantic_checkpoint_v8 import capture_random_state, restore_random_state

    with adapter.lock:
        if adapter.initialization is None:
            _fail("Initialize and attest the native V8 model before opening a coordinator")
        if origin not in {"model", "controlled"} or (
            origin == "controlled" and not adapter.allow_controlled_lifecycle
        ):
            _fail("Controlled V8 coordinator input requires explicit engineering authority")
        children = validate_ingress(session, public, max_active=maxActive, child_budget=childBudget)
        if session in adapter.sessions or session in adapter.closed_sessions:
            _fail("V8 coordinator opening requires a fresh session identity")
        random_before = capture_random_state()
        opening_count = len(adapter.openings)
        allocated_before = set(adapter.runtime.memory.sessions)
        work = None
        try:
            work = adapter._opened(session)
            native = work.runtime
            native.tracks.max_active = maxActive
            native.tracks.max_waiting = 17
            grants = sorted(
                {
                    grant
                    for child in children
                    for grant in child["program"]["fields"]["capabilities"]
                }
            )
            coordinator = adapter.runtime.track_action(
                native,
                "create",
                purpose="Execute the declared V8 child tasks and publish their exact results",
                completion="All declared children completed and their exact results were published",
                capabilities=tuple(dict.fromkeys(("text", *grants))),
                budget=min(
                    100000, (childBudget + 4 * adapter.max_fragments + 4) * len(children) + 1
                ),
            )
            for child in children:
                adapter.runtime.register_execution_context(native, **child)
                child["execution"] = _description(
                    execution_binding(**child, completion_target=coordinator.address)
                )
                child["track"] = None
                child["completion"] = None
            work.orchestration = {
                "schema": "semantic-orchestration/8",
                "initialPublic": copy.deepcopy(public),
                "maxActive": maxActive,
                "childBudget": childBudget,
                "origin": origin,
                "track": coordinator.address,
                "children": children,
                "decisions": [],
                "operations": [],
                "published": [],
                "cancelled": None,
            }
            maximum = adapter.runtime.generator.generator.config.max_history_chunks
            adapter.runtime.configure_context(
                native,
                track=coordinator.address,
                local_events=maximum,
                cross_events=0,
                pinned_events=(),
            )
            measured = adapter._observe(
                native,
                canonical_semantic_observation(
                    {
                        "schema": "semantic-orchestration-input/8",
                        "task": "Create each declared child; bootstrap is a separate host acknowledgement. Join its exact issued dependency, wait for its delivered completion, publish its exact kind/value as progress, then complete this coordinator.",
                        "sourceAdmissionClaim": False,
                        "public": public,
                        "children": children,
                    }
                ),
                "user",
                "semantic:v8:orchestration:ingress",
                track=coordinator.address,
            )
            visible = adapter.runtime.context_status(native, track=coordinator.address)[
                "visible_events"
            ]
            if not set(measured["events"]) <= set(visible):
                _fail("V8 coordinator source ingress exceeds visible native context")
            work.public = copy.deepcopy(public)
            # The complete ingress cut was just observed by the shared session;
            # each child separately observes its exact source/input at bootstrap.
            work.ingested_through = len(public["timeline"])
            return status_orchestration(adapter, session=session, track=coordinator.address)
        except Exception:
            if work is not None:
                adapter.sessions.pop(session, None)
                adapter.runtime.memory.close_session(work.runtime.memory)
            for identity in set(adapter.runtime.memory.sessions) - allocated_before:
                adapter.runtime.memory.close_session(adapter.runtime.memory.sessions[identity])
            del adapter.openings[opening_count:]
            restore_random_state(random_before)
            raise


def status_orchestration(adapter, *, session, track):
    with adapter.lock:
        work = _work(adapter, session, track)
        state = work.orchestration
        return {
            "schema": "semantic-orchestration-state/8",
            "session": session,
            "track": track,
            "origin": state["origin"],
            "sourceAdmissionClaim": False,
            "status": work.runtime.tracks.tracks[track].status,
            "children": copy.deepcopy(state["children"]),
            "decisions": copy.deepcopy(state["decisions"]),
            "tracks": [_description(item) for item in work.runtime.tracks.tracks.values()],
            "receipts": copy.deepcopy(work.runtime.execution_receipts),
            "published": list(state["published"]),
            "publicDigest": stable_algorithm_hash(work.public),
            "pendingSelection": work.pending["attempt"]["selectionId"] if work.pending else None,
        }


def _context(adapter, session, track):
    status = status_orchestration(adapter, session=session, track=track)
    return {
        "schema": "semantic-orchestration-context/8",
        "session": session,
        "track": track,
        "children": status["children"],
        "tracks": status["tracks"],
        "receipts": status["receipts"],
        "published": status["published"],
        "publicDigest": status["publicDigest"],
        "instruction": "Emit one JSON object {name,arguments}. Allowed actions are track_create, track_join, track_progress, track_complete and track_cancel. Copy a declared execution binding exactly into track_create, with purpose, completion, capabilities and budget. Join only this coordinator's issued child dependency. Progress must contain the exact JSON {kind,value} of a delivered child completion. Complete only after all declared children have delivered and published results. Other lifecycle actions are outside this bounded profile.",
    }


def select_orchestration(adapter, *, session, track, controlledRecord=None):
    from megatoken.generation.structure import ActionSpec
    from megatoken.semantic_checkpoint_v8 import capture_random_state
    from megatoken.semantic_model_v8 import ADAPTER, NativeSelectionV8Error, _fork_session

    with adapter.lock:
        work = _work(adapter, session, track)
        _live(work)
        if work.pending is not None:
            _fail("A V8 candidate requires commit or rejection before another selection")
        if len(adapter.attempts) >= adapter.max_attempts:
            _fail("V8 attempt accounting capacity exhausted")
        if not work.runtime.tracks.selectable(track, work.runtime.completed_events):
            _fail("V8 coordinator waits for actual child completion")
        controlled = work.orchestration["origin"] == "controlled"
        if controlled:
            if not adapter.allow_controlled_lifecycle or type(controlledRecord) is not str:
                _fail("Controlled V8 coordinator needs explicit original fixture bytes")
        elif controlledRecord is not None:
            _fail("Actual V8 coordinator cannot receive controlled decisions")
        context = _context(adapter, session, track)
        random_before = capture_random_state()
        branch = _fork_session(work.runtime)
        attempt = {
            "adapter": ADAPTER,
            "family": "orchestration",
            "selectionId": f"v8:{len(adapter.attempts)}",
            "session": session,
            "invocationId": None,
            "checkpointIdentity": adapter.model_identity,
            "tokenizerIdentity": adapter.codec.fingerprint,
            "trainingIdentity": adapter.training_identity,
            "codecFingerprint": adapter.codec.fingerprint,
            "origin": "controlled" if controlled else "actual-model-selected",
            "modelSelected": not controlled,
            "status": "submitted",
            "contextDigest": stable_algorithm_hash(context),
            "publicDigest": stable_algorithm_hash(work.public),
            "fragments": [],
            "payloadHex": None,
            "generatedTokens": 0,
            "observedTokens": 0,
            "contextTokens": 0,
            "observedBytes": 0,
            "contextBytes": 0,
            "observedPhysicalRecords": 0,
            "contextPhysicalRecords": 0,
            "instructionTokens": 0,
            "instructionBytes": 0,
            "generationSeconds": 0.0,
            "preparationSeconds": 0.0,
        }
        adapter.attempts.append(attempt)
        started = time.perf_counter()
        try:
            measured = adapter._observe(
                branch,
                canonical_semantic_observation(context),
                "system",
                f"semantic:v8:orchestration:context:{attempt['selectionId']}",
                track=track,
            )
            attempt.update(
                contextTokens=measured["tokens"],
                contextBytes=measured["bytes"],
                contextPhysicalRecords=measured["physicalRecords"],
            )
            required = set(measured["events"])
            before = adapter.runtime.context_status(branch, track=track)
            if not required <= set(before["visible_events"]):
                _fail("V8 coordinator context exceeds visible native history")
            attempt["contextBefore"] = before
            attempt["preparationSeconds"] = time.perf_counter() - started
            generation_started = time.perf_counter()
            for index in range(1 if controlled else adapter.max_fragments):
                if controlled:
                    if len(controlledRecord.encode()) > 65536:
                        _fail("V8 coordinator record exceeds its byte budget")
                    group = adapter.runtime.commit_fixture_action(
                        branch,
                        track=track,
                        payload=controlledRecord.encode(),
                        fixture_digest=stable_algorithm_hash(work.orchestration["initialPublic"]),
                        decision=len(work.orchestration["decisions"]),
                        family="lifecycle",
                    )
                else:
                    if not required <= set(
                        adapter.runtime.context_status(branch, track=track)["visible_events"]
                    ):
                        _fail("V8 coordinator context or prior fragments are no longer visible")
                    control = canonical_semantic_observation(
                        {"instruction": context["instruction"], "continuation": index > 0}
                    )
                    attempt["instructionTokens"] += len(adapter.codec.encode(control))
                    attempt["instructionBytes"] += len(control)
                    group = adapter.runtime.generate(
                        branch,
                        tracks=(track,),
                        event_ids=(f"{attempt['selectionId']}:fragment:{index}",),
                        specs=(
                            ActionSpec(
                                text_kind="utf8",
                                min_tokens=1,
                                max_tokens=adapter.max_tokens,
                                require_terminal=None,
                            ),
                        ),
                        controls=(control,),
                        effort=adapter.effort,
                        continuation_mode="encoded_text",
                    )
                tape = group.tape
                payload = tape.payloads[0] if len(tape.payloads) == 1 else None
                ids = tape.emitted_ids[0] if len(tape.emitted_ids) == 1 else ()
                fragment = {
                    "selectedEventId": tape.event_ids[0] if len(tape.event_ids) == 1 else None,
                    "meridian": tape.meridian_id,
                    "modelTrack": track,
                    "payloadHex": payload.hex() if isinstance(payload, bytes) else None,
                    "emittedTokenIds": list(ids),
                    "tokenIds": list(ids),
                    "terminal": tape.terminal[0] if len(tape.terminal) == 1 else None,
                }
                attempt["fragments"].append(fragment)
                attempt["generatedTokens"] += len(ids)
                if (
                    len(tape.payloads) != 1
                    or tape.valid != (True,)
                    or tape.modalities != ("text",)
                    or tape.tracks != (track,)
                    or not isinstance(payload, bytes)
                    or not ids
                    or any(type(value) is not int or value < 0 for value in ids)
                    or adapter.codec.decode(ids) != payload
                ):
                    _fail("Native coordinator emitted an invalid text fragment")
                required.update(tape.event_ids)
                original = b"".join(
                    bytes.fromhex(part["payloadHex"]) for part in attempt["fragments"]
                )
                attempt["payloadHex"] = original.hex()
                if len(original) > 65536:
                    _fail("V8 coordinator record exceeds its byte budget")
                if controlled:
                    branch.controlled_fixture_actions.pop(tape.event_ids[0])
                else:
                    adapter.runtime.release_generation(branch, branch.revision)
                if tape.terminal[0]:
                    break
            else:
                _fail("V8 coordinator exhausted its fragment budget without termination")
            record = original.decode("utf-8", errors="strict")
            attempt["generationSeconds"] = time.perf_counter() - generation_started
            attempt["contextAfter"] = adapter.runtime.context_status(branch, track=track)
            attempt["status"] = "awaiting-commit"
            attempt["totalSeconds"] = time.perf_counter() - started
            work.pending = {
                "family": "orchestration",
                "branch": branch,
                "attempt": attempt,
                "record": record,
                "randomBefore": random_before,
            }
            return {
                "schema": "semantic-orchestration-output/8",
                "record": record,
                "selection": copy.deepcopy(attempt),
            }
        except Exception as error:
            attempt.update(
                status="rejected", failure=str(error), totalSeconds=time.perf_counter() - started
            )
            raise NativeSelectionV8Error(str(error), evidence=copy.deepcopy(attempt)) from error


def _strict_record(record):
    try:
        action = strict_json_loads(record)
    except (TypeError, ValueError) as error:
        _fail("Coordinator did not select a complete JSON lifecycle action: " + str(error))
    if (
        type(action) is not dict
        or set(action) != {"name", "arguments"}
        or type(action["arguments"]) is not dict
    ):
        _fail("Coordinator requires exactly one typed lifecycle action")
    return action["name"], action["arguments"]


def validate_action(work, record):
    """Validate the complete action before mutating even a private continuation."""
    name, arguments = _strict_record(record)
    state, registry = work.orchestration, work.runtime.tracks
    coordinator = registry.tracks[state["track"]]
    if name == "track_create":
        if set(arguments) != {"purpose", "completion", "budget", "capabilities", "execution"}:
            _fail("V8 child creation requires its exact declared fields")
        binding = arguments["execution"]
        child = _child(work, binding.get("invocation_id")) if type(binding) is dict else None
        if child is None or not _same(binding, child["execution"]) or child["track"] is not None:
            _fail("Child creation changed or reused its immutable coordinator binding")
        if (
            type(arguments["purpose"]) is not str
            or not arguments["purpose"].strip()
            or type(arguments["completion"]) is not str
            or not arguments["completion"].strip()
            or type(arguments["budget"]) is not int
            or not 1 <= arguments["budget"] <= min(state["childBudget"], coordinator.budget)
            or type(arguments["capabilities"]) is not list
            or not _same(
                arguments["capabilities"],
                list(dict.fromkeys(("text", *binding["capability_grants"]))),
            )
            or not set(arguments["capabilities"]) <= set(coordinator.capabilities)
            or binding["completion_target"] != coordinator.address
        ):
            _fail("Child creation exceeds its coordinator scope, grants or budget")
    elif name == "track_join":
        if (
            set(arguments) != {"address", "completion_dependency"}
            or arguments["address"] != coordinator.address
        ):
            _fail("A coordinator can join only its own issued child dependency")
        reference = arguments["completion_dependency"]
        if (
            type(reference) is not dict
            or registry.dependencies.get(reference.get("id")) != reference
        ):
            _fail("Coordinator join contains a forged completion dependency")
        child = registry.tracks[reference["track"]]
        if (
            child.parent != coordinator.address
            or reference["id"] in work.runtime.execution_delivered
        ):
            _fail("Coordinator join must await an unfinished owned child")
    elif name == "track_progress":
        if (
            set(arguments) != {"address", "progress"}
            or arguments["address"] != coordinator.address
            or type(arguments["progress"]) is not str
        ):
            _fail("Coordinator progress must publish its exact delivered child result")
        try:
            value = strict_json_loads(arguments["progress"])
        except (ValueError, TypeError):
            _fail("Coordinator progress requires a JSON kind/value result")
        matches = [
            child
            for child in state["children"]
            if child["completion"] is not None
            and child["execution"]["invocation_id"] not in state["published"]
            and _same(value, child["completion"]["completion"])
        ]
        if not matches:
            _fail("Coordinator progress differs from an unpublished exact child completion")
    elif name in {"track_complete", "track_cancel"}:
        if set(arguments) != {"address"} or arguments["address"] != coordinator.address:
            _fail("Coordinator terminal action must name its own track")
        if name == "track_complete" and set(state["published"]) != {
            child["execution"]["invocation_id"] for child in state["children"]
        }:
            _fail("Coordinator completion requires every declared child result to be published")
    else:
        _fail("Lifecycle action is outside the bounded V8 coordinator profile")
    return name, arguments


def _cancel_tracks(adapter, branch):
    for track in tuple(branch.tracks.tracks.values()):
        if branch.tracks.tracks[track.address].status not in {"completed", "cancelled"}:
            adapter.runtime.track_action(branch, "cancel", address=track.address)


def commit_orchestration(adapter, *, session, track, record, selection):
    from megatoken.semantic_checkpoint_v8 import capture_random_state
    from megatoken.semantic_model_v8 import _fork_session

    with adapter.lock:
        work = _work(adapter, session, track)
        if type(selection) is not dict:
            _fail("Coordinator acknowledgement requires original selection evidence")
        prior = next(
            (
                item
                for item in work.orchestration["decisions"]
                if item["selectionId"] == selection.get("selectionId")
            ),
            None,
        )
        if prior is not None:
            index = int(prior["selectionId"].removeprefix("v8:"))
            expected = {**adapter.attempts[index], "status": "awaiting-commit"}
            if record != prior["record"] or not _same(selection, expected):
                _fail("Repeated coordinator acknowledgement changed its original selection")
            return copy.deepcopy(prior["outcome"])
        _live(work)
        pending = work.pending
        if (
            pending is None
            or pending.get("family") != "orchestration"
            or record != pending["record"]
            or not _same(selection, pending["attempt"])
        ):
            _fail("Coordinator acknowledgement differs from its exact pending candidate")
        # Generation has already charged the private coordinator track. Child
        # work cannot borrow the budget consumed by selecting its own creation.
        candidate_work = copy.copy(work)
        candidate_work.runtime = pending["branch"]
        name, arguments = validate_action(candidate_work, record)
        commit_random = capture_random_state()
        branch = _fork_session(pending["branch"])
        state = copy.deepcopy(work.orchestration)
        if name == "track_create":
            description = adapter.runtime.track_action(
                branch, "create", parent=track, ingress_key=selection["selectionId"], **arguments
            )
            child = next(
                child
                for child in state["children"]
                if child["execution"]["invocation_id"] == description.execution.invocation_id
            )
            child["track"] = description.address
            outcome = {
                "track": _description(description),
                "completion_dependency": branch.tracks.completion_dependency(description.address),
            }
        elif name == "track_join":
            description = branch.tracks.wait_for_execution(
                track, arguments["completion_dependency"], delivered=branch.completed_events
            )
            outcome = {"track": _description(description)}
        elif name == "track_progress":
            value = strict_json_loads(arguments["progress"])
            child = next(
                child
                for child in state["children"]
                if child["completion"] is not None
                and child["execution"]["invocation_id"] not in state["published"]
                and _same(value, child["completion"]["completion"])
            )
            state["published"].append(child["execution"]["invocation_id"])
            outcome = {
                "track": _description(adapter.runtime.track_action(branch, "progress", **arguments))
            }
        elif name == "track_complete":
            outcome = {
                "track": _description(
                    adapter.runtime.track_action(branch, "complete", address=track)
                )
            }
        else:
            _cancel_tracks(adapter, branch)
            state["cancelled"] = "selected coordinator cancellation"
            outcome = {"track": _description(branch.tracks.tracks[track])}
        outcome = {
            "schema": "semantic-orchestration-commit/8",
            "selectionId": selection["selectionId"],
            "committed": True,
            **outcome,
        }
        adapter._observe(
            branch,
            canonical_semantic_observation(outcome),
            "environment",
            f"semantic:v8:orchestration:receipt:{selection['selectionId']}",
            track=track,
        )
        state["decisions"].append(
            {
                "selectionId": selection["selectionId"],
                "selectionDigest": stable_algorithm_hash(deterministic_selection(selection)),
                "record": record,
                "outcome": copy.deepcopy(outcome),
            }
        )
        state["operations"].append(
            {
                "kind": "coordinator",
                "attempt": int(selection["selectionId"].removeprefix("v8:")),
                "randomBefore": pending["randomBefore"],
                "commitRandom": commit_random,
            }
        )
        _adopt(adapter, work, branch)
        work.orchestration = state
        pending["attempt"]["status"] = "committed"
        work.pending = None
        return outcome


def bootstrap_orchestration_child(adapter, *, session, track, childTrack):
    from megatoken.semantic_checkpoint_v8 import capture_random_state
    from megatoken.semantic_model_v8 import _fork_session

    with adapter.lock:
        work = _work(adapter, session, track)
        _live(work)
        if work.pending is not None:
            _fail("A pending V8 candidate must be resolved before child bootstrap")
        child = next(
            (child for child in work.orchestration["children"] if child["track"] == childTrack),
            None,
        )
        if child is None:
            _fail("Child bootstrap has no acknowledged coordinator reservation")
        description = work.runtime.tracks.tracks[childTrack]
        if description.status in {"completed", "cancelled"}:
            _fail("Terminal child cannot be bootstrapped")
        if description.bootstrap_ready:
            return {
                "schema": "semantic-orchestration-bootstrap/8",
                "track": _description(description),
                "bootstrapped": True,
            }
        random_before = capture_random_state()
        branch = _fork_session(work.runtime)
        description = adapter.runtime.bootstrap_execution(branch, childTrack)
        maximum = adapter.runtime.generator.generator.config.max_history_chunks
        adapter.runtime.configure_context(
            branch, track=childTrack, local_events=maximum, cross_events=0, pinned_events=()
        )
        outcome = {
            "schema": "semantic-orchestration-bootstrap/8",
            "track": _description(description),
            "bootstrapped": True,
        }
        adapter._observe(
            branch,
            canonical_semantic_observation(outcome),
            "environment",
            f"semantic:v8:orchestration:bootstrap:{childTrack}",
            track=track,
        )
        _adopt(adapter, work, branch)
        work.orchestration["operations"].append(
            {"kind": "bootstrap", "track": childTrack, "randomBefore": random_before}
        )
        return outcome


def semantic_track(work, invocation, public):
    if work.orchestration is None:
        from megatoken.contracts import MAIN_TRACK

        return MAIN_TRACK
    _live(work)
    initial = work.orchestration["initialPublic"]["timeline"]
    if not _same(public["timeline"][: len(initial)], initial):
        _fail("Child selection changed its coordinator's immutable source/input ingress")
    declared = {child["execution"]["invocation_id"] for child in work.orchestration["children"]}
    if any(
        row["kind"] == "invocation" and row["fields"]["invocationId"] not in declared
        for row in public["timeline"]
    ):
        _fail("Dynamic invocation admission is outside this bounded coordinator profile")
    child = _child(work, invocation)
    if child is None or child["track"] is None:
        _fail("Child selection needs an acknowledged coordinator reservation")
    if not work.runtime.tracks.selectable(child["track"], work.runtime.completed_events):
        _fail("Child selection requires observed bootstrap and actual dependency readiness")
    return child["track"]


def derive_completion(public, contexts, invocation):
    from megatoken.data.native_v8 import _replay_shared_state

    boundary = _replay_shared_state(
        public["timeline"], contexts, {}, roundtrip=False, boundary_invocation=invocation
    )
    completion = boundary["state"]["completion"]
    if (
        type(completion) is not dict
        or completion.get("kind") not in {"return", "throw"}
        or set(completion) != {"kind", "value"}
    ):
        _fail("V8 child completion lacks an independently replayed exact return or throw")
    return copy.deepcopy(completion), copy.deepcopy(boundary["state"]["heap"])


def recorded_completion(state, *, public, contexts, invocationId, completion):
    """Read an exact historical outcome without recreating delivery authority."""
    if state is None:
        return None
    child = next(
        (
            child
            for child in state["children"]
            if child["execution"]["invocation_id"] == invocationId
        ),
        None,
    )
    recorded = child["completion"] if child is not None else None
    if recorded is None:
        return None
    if (
        stable_algorithm_hash(public) != recorded["publicDigest"]
        or stable_algorithm_hash(contexts) != recorded["contextsDigest"]
        or (completion is not None and not _same(completion, recorded["completion"]))
    ):
        _fail("V8 completion retry changed its original public cut, contexts or result")
    return copy.deepcopy(recorded)


def deliver_completion(adapter, work, *, session, public, contexts, invocationId, completion):
    from megatoken.semantic_checkpoint_v8 import capture_random_state
    from megatoken.semantic_model_v8 import _fork_session

    _live(work)
    child = _child(work, invocationId)
    if child is None or child["track"] is None:
        _fail("V8 child completion has no acknowledged owning coordinator")
    exact, heap = derive_completion(public, contexts, invocationId)
    if completion is not None and not _same(completion, exact):
        _fail("Caller completion differs from independently replayed V8 semantics")
    result = {
        "schema": "semantic-child-completion/8",
        "session": session,
        "invocationId": invocationId,
        "track": child["track"],
        "completion": exact,
        "heap": heap,
        "publicDigest": stable_algorithm_hash(public),
        "contextsDigest": stable_algorithm_hash(contexts),
    }
    if child["completion"] is not None:
        if not _same(result, child["completion"]):
            _fail("V8 completion retry changed its immutable acknowledged result")
        return copy.deepcopy(result)
    description = work.runtime.tracks.tracks[child["track"]]
    if not description.bootstrap_ready or description.status in {"completed", "cancelled"}:
        _fail("Closed or unbootstrapped child cannot publish completion")
    random_before = capture_random_state()
    branch = _fork_session(work.runtime)
    adapter.runtime.complete_execution(branch, child["track"], **exact)
    adapter._observe(
        branch,
        canonical_semantic_observation(result),
        "environment",
        f"semantic:v8:orchestration:completion:{invocationId}",
        track=work.orchestration["track"],
    )
    _adopt(adapter, work, branch)
    child["completion"] = result
    work.orchestration["operations"].append(
        {
            "kind": "completion",
            "invocationId": invocationId,
            "acknowledgedThrough": len(public["timeline"]),
            "randomBefore": random_before,
        }
    )
    return copy.deepcopy(result)


def cancel_orchestration(adapter, *, session, track, reason):
    from megatoken.semantic_checkpoint_v8 import capture_random_state
    from megatoken.semantic_model_v8 import _fork_session

    with adapter.lock:
        work = _work(adapter, session, track)
        if type(reason) is not str or not reason:
            _fail("V8 coordinator cancellation requires a nonempty observed reason")
        if work.orchestration["cancelled"] is not None:
            if work.orchestration["cancelled"] != reason:
                _fail("V8 coordinator cancellation retry changed its original reason")
            return status_orchestration(adapter, session=session, track=track)
        if work.runtime.tracks.tracks[track].status == "completed":
            _fail("Completed V8 coordinator cannot be cancelled or resumed")
        random_before = capture_random_state()
        branch = _fork_session(work.runtime)
        _cancel_tracks(adapter, branch)
        receipt = {"schema": "semantic-orchestration-cancel/8", "track": track, "reason": reason}
        branch.execution_receipts.append(receipt)
        _adopt(adapter, work, branch)
        work.orchestration["cancelled"] = reason
        work.orchestration["operations"].append(
            {"kind": "cancel", "reason": reason, "randomBefore": random_before}
        )
        if work.pending is not None:
            work.pending["attempt"].update(status="cancelled", failure=reason)
            work.pending = None
        return status_orchestration(adapter, session=session, track=track)


def replay_semantic_selection(adapter, *, session, saved, attempts, index, entry):
    """Regenerate and acknowledge one original semantic action during recovery."""
    from megatoken.semantic_checkpoint_v8 import restore_random_state

    context_index = entry["contextIndex"]
    context = saved["contexts"][context_index]
    rows = saved["public"]["timeline"]
    through, acknowledged = context["visibleThrough"], entry["acknowledgedThrough"]
    if not through < acknowledged <= len(rows):
        _fail("Invalid V8 acknowledged-history cursor")
    adapter.attempts = copy.deepcopy(attempts[:index])
    restore_random_state(entry["randomBefore"])
    generated = adapter.select(
        session=session,
        public={**saved["public"], "timeline": rows[:through]},
        contexts=saved["contexts"][:context_index],
        context=context,
    )
    if not _same(
        deterministic_selection(generated["selection"]), deterministic_selection(attempts[index])
    ):
        _fail("Restored native V8 child selection differs from its original evidence")
    adapter.commit(
        session=session,
        public={**saved["public"], "timeline": rows[:acknowledged]},
        contexts=saved["contexts"][: context_index + 1],
        invocationId=context["invocationId"],
        record=generated["record"],
        selection=generated["selection"],
    )


def restore_orchestration(adapter, *, session, saved, attempts):
    """Replay the single ordered coordinator/child history on a staged runtime."""
    from megatoken.semantic_checkpoint_v8 import restore_random_state

    state = saved["orchestration"]
    keys = {
        "schema",
        "initialPublic",
        "maxActive",
        "childBudget",
        "origin",
        "track",
        "children",
        "decisions",
        "operations",
        "published",
        "cancelled",
    }
    if (
        type(state) is not dict
        or set(state) != keys
        or state["schema"] != "semantic-orchestration/8"
        or type(state["operations"]) is not list
        or len(state["operations"]) > adapter.max_attempts * 2 + 34
        or type(state["children"]) is not list
        or not 1 <= len(state["children"]) <= 16
    ):
        _fail("Invalid bounded V8 orchestration checkpoint")
    adapter.open_orchestration(
        session=session,
        public=state["initialPublic"],
        maxActive=state["maxActive"],
        childBudget=state["childBudget"],
        origin=state["origin"],
    )
    work = adapter.sessions[session]
    if work.orchestration["track"] != state["track"]:
        _fail("Restored coordinator address differs from its original allocation")
    expected_attempts = [
        index
        for index, attempt in enumerate(attempts)
        if attempt["session"] == session and attempt["status"] == "committed"
    ]
    replayed_attempts, semantic_indices = [], []
    for operation in state["operations"]:
        if type(operation) is not dict:
            _fail("Malformed V8 orchestration operation")
        kind = operation.get("kind")
        if kind == "coordinator":
            if (
                set(operation) != {"kind", "attempt", "randomBefore", "commitRandom"}
                or type(operation["attempt"]) is not int
                or not 0 <= operation["attempt"] < len(attempts)
            ):
                _fail("Malformed coordinator acknowledgement in ordered recovery history")
            index = operation["attempt"]
            original = attempts[index]
            if (
                original.get("family") != "orchestration"
                or original["session"] != session
                or original["status"] != "committed"
            ):
                _fail("Coordinator operation has no original committed selection")
            replayed_attempts.append(index)
            adapter.attempts = copy.deepcopy(attempts[:index])
            restore_random_state(operation["randomBefore"])
            controlled = (
                bytes.fromhex(original["payloadHex"]).decode("utf-8", errors="strict")
                if state["origin"] == "controlled"
                else None
            )
            generated = adapter.select_orchestration(
                session=session, track=state["track"], controlledRecord=controlled
            )
            if not _same(
                deterministic_selection(generated["selection"]), deterministic_selection(original)
            ):
                _fail("Restored native coordinator selection differs from its original evidence")
            restore_random_state(operation["commitRandom"])
            adapter.commit_orchestration(
                session=session, track=state["track"], **generated_without_schema(generated)
            )
        elif kind == "semantic":
            if (
                set(operation) != {"kind", "replay"}
                or type(operation["replay"]) is not int
                or operation["replay"] != len(semantic_indices)
                or not 0 <= operation["replay"] < len(saved["replay"])
            ):
                _fail("Reordered or duplicated semantic acknowledgement in coordinator history")
            semantic_indices.append(operation["replay"])
            entry = saved["replay"][operation["replay"]]
            index = entry["attempt"]
            if attempts[index].get("family") is not None:
                _fail("Child semantic operation was substituted with coordinator authority")
            replayed_attempts.append(index)
            replay_semantic_selection(
                adapter, session=session, saved=saved, attempts=attempts, index=index, entry=entry
            )
        elif kind == "bootstrap":
            if set(operation) != {"kind", "track", "randomBefore"}:
                _fail("Malformed V8 child bootstrap operation")
            restore_random_state(operation["randomBefore"])
            adapter.bootstrap_orchestration_child(
                session=session, track=state["track"], childTrack=operation["track"]
            )
        elif kind == "completion":
            if (
                set(operation) != {"kind", "invocationId", "acknowledgedThrough", "randomBefore"}
                or type(operation["acknowledgedThrough"]) is not int
                or not 0 < operation["acknowledgedThrough"] <= len(saved["public"]["timeline"])
            ):
                _fail("Malformed V8 child completion operation")
            restore_random_state(operation["randomBefore"])
            adapter.complete(
                session=session,
                public={
                    **saved["public"],
                    "timeline": saved["public"]["timeline"][: operation["acknowledgedThrough"]],
                },
                contexts=work.contexts,
                invocationId=operation["invocationId"],
            )
        elif kind == "cancel":
            if set(operation) != {"kind", "reason", "randomBefore"}:
                _fail("Malformed V8 coordinator cancellation operation")
            restore_random_state(operation["randomBefore"])
            adapter.cancel_orchestration(
                session=session, track=state["track"], reason=operation["reason"]
            )
        else:
            _fail("Unknown V8 orchestration operation")
    if replayed_attempts != expected_attempts or semantic_indices != list(
        range(len(saved["replay"]))
    ):
        _fail("Coordinator checkpoint omits, duplicates or reorders acknowledged selections")
    if not _same(work.orchestration, state):
        _fail("Restored coordinator bindings, receipts or ordered history differ")
    return len(replayed_attempts)


def generated_without_schema(generated):
    return {"record": generated["record"], "selection": generated["selection"]}
