"""Frozen semantic cohorts and free-running, independent-outcome evaluation.

The runner receives only source, invocation and its declared environment. Expected
outcomes never enter inference. Controlled selections and initialized fixed weights
remain engineering evidence and cannot produce a learned-success statistic.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter

from megatoken.data.host_operations import CONTRACT_DIGEST, canonical_json
from megatoken.data.schema import encode_record
from megatoken.data.semantic_harness_v4 import adapt_selected_harness_v4

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("compact",)
HARD_GATES = (
    "futureObservationLeaks",
    "unauthorizedEffects",
    "untrackedNativeFallback",
    "falseAdmissions",
    "replayNativeEffects",
)
OUTCOMES = ("completion", "state", "effects", "callbacks", "cleanup")
CONTROLLED_FIELDS = frozenset(
    {
        "controlledSelections",
        "controlledSelectionsByRepresentation",
        "controlledOutcome",
        "controlledCoordinatorActions",
    }
)


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _interactive_usage(evidence, codec):
    """Retain failed selections as cost, without duplicating accepted request failures."""
    timeline = evidence.get("episode", {}).get("public", {}).get("timeline", [])
    accepted = [event["line"].encode() for event in timeline if event.get("kind") == "semantic"]
    generated = sum(len(codec.encode(raw)) for raw in accepted)
    observed = sum(
        len(codec.encode(canonical_json(event).encode()))
        for event in timeline
        if event.get("kind") != "semantic"
    )
    rejected = []
    retained_ids = set()
    for receipt in evidence.get("rejections", []):
        raw = receipt.get("record")
        if not isinstance(raw, str):
            continue
        raw = raw.encode()
        if receipt.get("phase") == "native-commit-or-delivery" and raw in accepted:
            continue
        identity = receipt.get("selectedEventId") or (receipt.get("selection") or {}).get(
            "selectedEventId"
        )
        if identity is not None and identity in retained_ids:
            continue
        if identity is not None:
            retained_ids.add(identity)
        rejected.append(raw)
    failure = (evidence.get("failure") or {}).get("evidence") or {}
    encoded = failure.get("payloadHex")
    if isinstance(encoded, str):
        raw = bytes.fromhex(encoded)
    elif isinstance(encoded, list) and all(
        value is None or isinstance(value, str) for value in encoded
    ):
        raw = b"".join(bytes.fromhex(value) for value in encoded if value is not None)
    else:
        raw = None
    if raw is not None:
        identity = failure.get("selectedEventId") or (failure.get("selection") or {}).get(
            "selectedEventId"
        )
        already_retained = identity is not None and identity in retained_ids
        native_failure = any(
            receipt.get("phase") == "native-commit-or-delivery"
            and receipt.get("record", "").encode() == raw
            for receipt in evidence.get("rejections", [])
        )
        if (
            not already_retained
            and not (native_failure and raw in accepted)
            and raw not in rejected
        ):
            rejected.append(raw)
    rejected_tokens = sum(len(codec.encode(raw)) for raw in rejected)
    return generated, observed, rejected_tokens


def _original_emission_tokens(evidence, codec, *, canonical_tokens):
    """Meter original decoder streams by immutable selected-event identity."""
    fragments, seen = [], {}
    selections = list(evidence.get("selections", [])) + list(evidence.get("rejections", []))
    if evidence.get("selection") is not None:
        selections.append({"record": evidence.get("record"), "selection": evidence["selection"]})
    orchestration = evidence.get("orchestration") or {}
    selections.extend(orchestration.get("childSelections", []))
    selections.extend(orchestration.get("decisions", []))
    for item in selections:
        selection = item.get("selection")
        if selection is None:
            continue
        if (
            selection.get("modelSelected") is not True
            or selection.get("codecFingerprint") != codec.fingerprint
        ):
            raise ValueError(
                "Inference token evidence does not come from the attested actual model"
            )
        selected_fragments = selection.get("fragments")
        if not isinstance(selected_fragments, list) or not selected_fragments:
            raise ValueError("Actual model cost requires its original emitted fragment token IDs")
        if (
            item.get("record") is not None
            and b"".join(bytes.fromhex(fragment["payloadHex"]) for fragment in selected_fragments)
            != item["record"].encode()
        ):
            raise ValueError(
                "Original inference fragments do not reconstruct their reported record"
            )
        fragments.extend(selected_fragments)
    failure = (evidence.get("failure") or {}).get("evidence") or {}
    fragments.extend(failure.get("fragments", []))
    if failure.get("selection"):
        fragments.extend(failure["selection"].get("fragments", []))
    for decision in orchestration.get("decisions", []):
        fragments.extend(decision.get("evidence", {}).get("fragments", []))
    for fragment in fragments:
        identity, ids = fragment["selectedEventId"], fragment["emittedTokenIds"]
        if (
            not isinstance(identity, str)
            or not isinstance(ids, list)
            or any(type(value) is not int for value in ids)
        ):
            raise ValueError("Native decoder token evidence is malformed")
        raw = bytes.fromhex(fragment["payloadHex"])
        if codec.decode(ids) != raw:
            raise ValueError(
                "Native decoder token IDs do not reconstruct the original selected bytes"
            )
        signature = (tuple(ids), raw)
        if identity in seen and seen[identity] != signature:
            raise ValueError("Selected event identity was reused for different inference bytes")
        seen[identity] = signature
    if canonical_tokens and not seen:
        raise ValueError("Actual generated output omitted its original inference token evidence")
    return sum(len(ids) for ids, _ in seen.values())


def _orchestration_usage(evidence, codec):
    orchestration = evidence["orchestration"]
    if orchestration.get("schema") != "semantic-orchestration-evidence/4":
        raise ValueError("Missing native coordinator execution evidence")
    state = orchestration["finalState"]
    if state.get("schema") != "semantic-orchestration-state/4":
        raise ValueError("Coordinator report lacks its actual final native state")
    rows = state["observations"]
    seen, generated, observed = set(), 0, 0
    for row in rows:
        if row["eventId"] in seen or type(row["terminal"]) is not bool:
            raise ValueError("Native orchestration observation identity was repeated or changed")
        seen.add(row["eventId"])
        count = len(codec.encode(bytes.fromhex(row["payloadHex"])))
        if row["role"] == "assistant":
            generated += count
        else:
            observed += count
    return generated, observed, len(rows)


def _validate_orchestration(evidence, codec, origin):
    """Independently check native lifecycle ownership, bytes and causal receipts."""
    from megatoken.io import strict_json_loads
    from megatoken.semantic_model_adapter import _validate_prefix

    execution = evidence["orchestration"]
    state, coordinator = execution["finalState"], execution["track"]
    if (
        execution.get("validation") != {"semanticReplay": True, "lifecycle": True, "complete": True}
        or state["session"] != execution["session"]
        or state["track"] != coordinator
        or state["representation"] != evidence["representation"]
        or state["origin"] != ("controlled" if origin == "controlled" else "model")
        or state["status"] != "completed"
        or any(track["status"] != "completed" for track in state["tracks"])
    ):
        raise ValueError("Coordinator execution is incomplete or has inconsistent authority")
    proposed = {}
    for prefix in execution["proposals"]:
        program, invocation, rows = _validate_prefix(prefix)
        if len(rows) != 2 or invocation["invocationId"] in proposed:
            raise ValueError("Coordinator proposals contain future or duplicate child work")
        proposed[invocation["invocationId"]] = (program, invocation)
    messages, fragments = [], []
    for index, row in enumerate(state["observations"]):
        if fragments and (row["track"], row["role"]) != (
            fragments[0][1]["track"],
            fragments[0][1]["role"],
        ):
            raise ValueError("Native orchestration publication interleaves a partial message")
        fragments.append((index, row))
        if row["terminal"]:
            raw = b"".join(bytes.fromhex(part["payloadHex"]) for _, part in fragments)
            messages.append(
                (
                    fragments[0][0],
                    index,
                    row["track"],
                    row["role"],
                    raw,
                    tuple(part["eventId"] for _, part in fragments),
                )
            )
            fragments = []
    if fragments:
        raise ValueError("Coordinator evidence ends inside a physical message")
    created, completions, progressed = {}, {}, []
    for start, _end, track, role, raw, _identities in messages:
        if role != "assistant":
            try:
                item = strict_json_loads(raw.decode())
            except (ValueError, UnicodeError):
                continue
            if isinstance(item, dict) and item.get("schema") == "semantic-completion/4":
                dependency = item["dependency"]
                if (
                    track != coordinator
                    or item["target"] != coordinator
                    or item["parent"] != coordinator
                ):
                    # Nested callback completion belongs to its own execution parent.
                    continue
                completions[dependency["invocation"]] = (start, item)
    for decision in execution["decisions"]:
        selection, record, outcomes = (
            decision["selection"],
            decision["record"],
            decision["outcomes"],
        )
        if (
            selection["modelSelected"] != (origin != "controlled")
            or selection["modelTrack"] != coordinator
            or selection["codecFingerprint"] != codec.fingerprint
        ):
            raise ValueError("Coordinator selection changed its actual model/control authority")
        matching = [
            message
            for message in messages
            if message[2:4] == (coordinator, "assistant")
            and message[4] == record.encode()
            and any(
                identity == selection["selectedEventId"]
                or identity.startswith(selection["selectedEventId"] + ":part:")
                for identity in message[5]
            )
        ]
        if len(matching) != 1 or len(outcomes) != 1 or "error" in outcomes[0]:
            raise ValueError("Coordinator action lacks one exact committed native execution")
        start, end = matching[0][:2]
        action = strict_json_loads(record)
        name, args = action["name"], action["arguments"]
        receipt = outcomes[0]
        if receipt.get("action_id") != f"{selection['meridian']}:{selection['selectedEventId']}":
            raise ValueError("Lifecycle receipt is not correlated to its selected event")
        if name == "track_create":
            binding, child = args["execution"], receipt["track"]
            identity = binding["invocation_id"]
            if identity not in proposed or identity in created or child["parent"] != coordinator:
                raise ValueError("Coordinator created an unproposed, repeated or unowned child")
            program, invocation = proposed[identity]
            if (
                binding["source_digest"] != program["digest"]
                or binding["program_id"] != program["programId"]
                or not set(binding["capability_grants"]) <= set(program["capabilities"])
            ):
                raise ValueError("Child allocation changed its observed source or capability grant")
            if (
                receipt["completion_dependency"]["track"] != child["address"]
                or receipt["completion_dependency"]["invocation"] != identity
            ):
                raise ValueError("Child creation returned a foreign completion dependency")
            child_publications = []
            for message in messages:
                if message[2] != child["address"] or message[3] != "user":
                    continue
                item = strict_json_loads(message[4].decode())
                if item.get("kind") in {"program", "invocation"}:
                    child_publications.append((message[0], item))
            if (
                len(child_publications) < 2
                or child_publications[0][0] <= end
                or [item["kind"] for _, item in child_publications[:2]] != ["program", "invocation"]
            ):
                raise ValueError("Child selection source/input were not published after creation")
            acknowledgements = []
            for message in messages:
                if message[2] != coordinator or message[3] != "tool":
                    continue
                item = strict_json_loads(message[4].decode())
                if item.get("action_id") == receipt["action_id"]:
                    acknowledgements.append(message)
            if (
                len(acknowledgements) != 1
                or not end
                < acknowledgements[0][0]
                <= acknowledgements[0][1]
                < child_publications[0][0]
            ):
                raise ValueError(
                    "Native child code was not published after its exact creation acknowledgement"
                )
            if (
                child_publications[0][1]["digest"] != program["digest"]
                or child_publications[1][1]["inputGraph"] != invocation["inputGraph"]
            ):
                raise ValueError("Native child bootstrap changed source or arguments")
            created[identity] = {
                "track": child["address"],
                "dependency": receipt["completion_dependency"],
                "joined": False,
            }
        elif name == "track_join":
            dependency = args["completion_dependency"]
            identity = dependency["invocation"]
            if (
                args["address"] != coordinator
                or identity not in created
                or dependency != created[identity]["dependency"]
            ):
                raise ValueError(
                    "Coordinator join did not use its actual child's runtime dependency"
                )
            created[identity]["joined"] = True
        elif name == "track_progress":
            if args["address"] != coordinator:
                raise ValueError("Coordinator progress was redirected to another owner")
            progress = strict_json_loads(args["progress"])
            available = [
                identity
                for identity in proposed
                if identity not in progressed
                and identity in completions
                and completions[identity][0] < start
            ]
            if not available:
                raise ValueError("Coordinator predicted child progress before observed completion")
            identity = available[0]
            _, completion = completions[identity]
            if (
                identity not in created
                or not created[identity]["joined"]
                or completion["dependency"] != created[identity]["dependency"]
                or progress != {key: completion[key] for key in ("kind", "value")}
            ):
                raise ValueError("Coordinator progress differs from the delivered child result")
            progressed.append(identity)
        elif name == "track_complete":
            if args["address"] != coordinator or set(progressed) != set(proposed):
                raise ValueError(
                    "Coordinator completed before every joined child result was published"
                )
        else:
            raise ValueError(
                "Coordinator record is outside the declared child/join/progress policy"
            )
    if set(created) != set(proposed) or set(progressed) != set(proposed):
        raise ValueError("Coordinator omitted required child lifecycle work")
    tracks = {track["address"]: track for track in state["tracks"]}
    for row in execution["childSelections"]:
        selection = row["selection"]
        native = tracks.get(selection["modelTrack"])
        if (
            native is None
            or native["execution"]["invocation_id"] != row["invocationId"]
            or selection["modelSelected"] != (origin != "controlled")
            or selection["codecFingerprint"] != codec.fingerprint
        ):
            raise ValueError("Child selection did not use its actual native allocated owner")
        owner = native["parent"]
        seen = set()
        while owner is not None and owner != coordinator:
            if owner in seen or owner not in tracks:
                raise ValueError("Child native ownership has a cycle or missing ancestor")
            seen.add(owner)
            owner = tracks[owner]["parent"]
        if owner != coordinator:
            raise ValueError("Guest work escaped the native coordinator task tree")


def freeze_cohort(cases, *, name, thresholds):
    """Freeze task families, reference outcomes, thresholds and input bytes before runs."""
    if name not in {"common-support", "effectful-tracks-events"}:
        raise ValueError("Report the frozen common-support and effectful cohorts separately")
    if set(thresholds) != {"wholeTaskSuccess", "traceValid"} or any(
        type(x) not in (int, float) or not 0 <= x <= 1 for x in thresholds.values()
    ):
        raise ValueError("Declare whole-task and trace acceptance thresholds before evaluation")
    seen, families, normalized = set(), {}, []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "family",
            "split",
            "request",
            "expected",
            "referenceReceipt",
        }:
            raise ValueError(
                "Cohort cases require separate observed request and independent expected result"
            )
        if case["id"] in seen or case["split"] not in {"validation", "test"} or not case["family"]:
            raise ValueError("Evaluation case IDs are unique and whole families are held out")
        if case["family"] in families and families[case["family"]] != case["split"]:
            raise ValueError("A held-out task family cannot cross splits")
        if set(case["expected"]) != set(OUTCOMES):
            raise ValueError(
                "Independent reference must cover completion, state, effects, callbacks and cleanup"
            )
        if (
            case["referenceReceipt"].get("status") != "agreed"
            or case["referenceReceipt"].get("requestDigest") != _digest(case["request"])
            or case["referenceReceipt"].get("outcomeDigest") != _digest(case["expected"])
        ):
            raise ValueError("Reference receipts must bind exact input and outcome bytes")
        seen.add(case["id"])
        families[case["family"]] = case["split"]
        normalized.append(json.loads(canonical_json(case)))
    if not normalized:
        raise ValueError("Frozen evaluation cohort cannot be empty")
    cohort = {
        "schema": "semantic-evaluation-cohort/4",
        "name": name,
        "cases": normalized,
        "thresholds": thresholds,
        "seeds": list(SEEDS),
        "catalogDigest": CONTRACT_DIGEST,
    }
    return {**cohort, "digest": _digest(cohort)}


def verify_cohort(cohort):
    if (
        cohort.get("schema") != "semantic-evaluation-cohort/4"
        or cohort.get("catalogDigest") != CONTRACT_DIGEST
        or cohort.get("seeds") != list(SEEDS)
    ):
        raise ValueError("Incompatible semantic evaluation cohort")
    if cohort.get("digest") != _digest(
        {key: value for key, value in cohort.items() if key != "digest"}
    ):
        raise ValueError("Frozen cohort or acceptance thresholds changed")
    rebuilt = freeze_cohort(
        cohort["cases"],
        name=cohort["name"],
        thresholds=cohort["thresholds"],
    )
    if rebuilt != cohort:
        raise ValueError("Cohort is not its canonical frozen declaration")


def evaluate_cohort(
    cohort, runner, *, codec, seed, origin, representation="compact", checkpoint_identity=None
):
    """Execute a supplied production runner with no access to private expected labels."""
    verify_cohort(cohort)
    if (
        seed not in SEEDS
        or origin not in {"controlled", "fixed-weights", "trained"}
        or representation not in REPRESENTATIONS
    ):
        raise ValueError("Unregistered semantic evaluation seed, origin or representation")
    if origin != "controlled" and (
        not isinstance(checkpoint_identity, str) or len(checkpoint_identity) != 64
    ):
        raise ValueError("Model evaluations require their immutable served model digest")
    if origin == "trained" and any(case["split"] != "test" for case in cohort["cases"]):
        raise ValueError("Learned comparison reports require the frozen test-only cohort")
    rows = []
    for case in cohort["cases"]:
        started = time.perf_counter()
        evidence = None
        generated = observed = 0
        canonical_generated, model_generated = 0, None
        physical = None
        costs, hard = {}, {}
        try:
            # Deep copy prevents a runner from mutating the frozen declaration.
            request = json.loads(canonical_json(case["request"]))
            if origin != "controlled":
                for field in CONTROLLED_FIELDS:
                    request.pop(field, None)
            evidence = runner(
                {
                    "schema": "semantic-evaluation-request/4",
                    "caseId": case["id"],
                    "request": request,
                    "seed": seed,
                    "representation": representation,
                    "checkpointIdentity": checkpoint_identity,
                }
            )
            if (
                evidence.get("schema") != "semantic-evaluation-result/4"
                or evidence.get("origin") != origin
                or evidence.get("representation") != representation
            ):
                raise ValueError("Runner must report its real controlled/model authority")
            if origin != "controlled" and evidence.get("checkpointIdentity") != checkpoint_identity:
                raise ValueError("Runner used a different served model checkpoint")
            if origin != "controlled" and evidence.get("inferenceIdentity") != {
                "schema": "semantic-inference-identity/4",
                "seed": seed,
                "checkpointIdentity": checkpoint_identity,
                "tokenizerIdentity": codec.fingerprint,
            }:
                raise ValueError(
                    "Model runner lacks the actual seeded inference initialization receipt"
                )
            hard = evidence.get("hardGates")
            if (
                not isinstance(hard, dict)
                or set(hard) != set(HARD_GATES)
                or any(type(x) is not int or x < 0 for x in hard.values())
            ):
                raise ValueError(
                    "Runner must supply measured counts for every hard safety/correctness gate"
                )
            costs = evidence.get("costs", {})
            if set(costs) != {
                "computeSeconds",
                "peakMemoryBytes",
                "dispatches",
                "queueDepth",
                "leakedResources",
            } or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in costs.values()
            ):
                raise ValueError(
                    "Runner must report measured compute, memory, dispatch and resource costs"
                )
            rejected_tokens = 0
            if request.get("schema") == "selected-orchestration-evaluation-case/4":
                generated, observed, physical = _orchestration_usage(evidence, codec)
            else:
                generated, observed, rejected_tokens = _interactive_usage(evidence, codec)
                generated += rejected_tokens
            canonical_generated = generated
            if origin != "controlled":
                model_generated = _original_emission_tokens(
                    evidence, codec, canonical_tokens=canonical_generated
                )
            actual = evidence.get("outcome")
            if not isinstance(actual, dict) or set(actual) != set(OUTCOMES):
                raise ValueError(
                    "Runner result omitted execution state/effects/callback/cleanup evidence"
                )
            if "firstSemanticInconsistency" not in evidence:
                raise ValueError(
                    "Runner must report selected-controller semantic consistency evidence"
                )
            inconsistency = evidence["firstSemanticInconsistency"]
            if inconsistency is not None and not isinstance(inconsistency, dict):
                raise ValueError("First semantic violation must be an exact failure receipt")
            trace_valid = inconsistency is None and not evidence.get("failure")
            episode, _ = adapt_selected_harness_v4(
                evidence["episode"],
                codec=codec,
                allow_local=True,
                max_positions=1024,
                max_history=1024,
            )
            if request.get("schema") == "selected-orchestration-evaluation-case/4":
                _validate_orchestration(evidence, codec, origin)
            else:
                generated, observed = rejected_tokens, 0
                for record in episode.records:
                    count = encode_record(record, codec).tokens
                    generated += count if record.policy_mask else 0
                    observed += count if not record.policy_mask else 0
                physical = len(episode.records)
            canonical_generated = generated
            if model_generated is not None:
                generated = model_generated
            correct = {
                name: canonical_json(actual[name]) == canonical_json(case["expected"][name])
                for name in OUTCOMES
            }
            success = (
                all(correct.values())
                and trace_valid
                and not any(hard.values())
                and costs["leakedResources"] == 0
            )
            rows.append(
                {
                    "caseId": case["id"],
                    "family": case["family"],
                    "wholeTaskSuccess": success,
                    "traceValid": trace_valid,
                    "firstSemanticInconsistency": inconsistency,
                    "matches": correct,
                    "hardGates": hard,
                    "generatedTokens": generated,
                    "canonicalGeneratedTokens": canonical_generated,
                    "generatedTokenBasis": "original-decoder-token-ids"
                    if model_generated is not None
                    else "canonical-native-records",
                    "observedTokens": observed,
                    "physicalRecords": physical,
                    "latencySeconds": time.perf_counter() - started,
                    "costs": costs,
                    "costScope": evidence.get("costScope"),
                    "nativeRuntimeMetrics": evidence.get("nativeRuntimeMetrics"),
                    "failure": None
                    if success
                    else "runner-failure"
                    if evidence.get("failure")
                    else "outcome-or-hard-gate",
                    "failureEvidence": evidence.get("failure"),
                    "evidenceDigest": _digest(evidence),
                }
            )
        except (ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as error:
            if model_generated is not None:
                generated = model_generated
            hard = {
                key: value
                for key, value in (hard.items() if isinstance(hard, dict) else ())
                if key in HARD_GATES and type(value) is int and value >= 0
            }
            costs = {
                key: value
                for key, value in (costs.items() if isinstance(costs, dict) else ())
                if type(value) in (int, float) and math.isfinite(value) and value >= 0
            }
            rows.append(
                {
                    "caseId": case["id"],
                    "family": case["family"],
                    "wholeTaskSuccess": False,
                    "traceValid": False,
                    "generatedTokens": generated,
                    "canonicalGeneratedTokens": canonical_generated,
                    "generatedTokenBasis": "original-decoder-token-ids"
                    if model_generated is not None
                    else "canonical-native-records",
                    "observedTokens": observed,
                    "physicalRecords": physical,
                    "hardGates": hard if isinstance(hard, dict) else {},
                    "costs": costs if isinstance(costs, dict) else {},
                    "costScope": evidence.get("costScope") if isinstance(evidence, dict) else None,
                    "nativeRuntimeMetrics": evidence.get("nativeRuntimeMetrics")
                    if isinstance(evidence, dict)
                    else None,
                    "failureEvidence": evidence.get("failure")
                    if isinstance(evidence, dict)
                    else None,
                    "latencySeconds": time.perf_counter() - started,
                    "failure": type(error).__name__ + ": " + str(error),
                    "evidenceDigest": _digest(evidence) if evidence is not None else None,
                }
            )
    correct = sum(row["wholeTaskSuccess"] for row in rows)
    trace = sum(row["traceValid"] for row in rows)
    hard = {key: sum(row.get("hardGates", {}).get(key, 0) for row in rows) for key in HARD_GATES}
    summary = {
        "tasks": len(rows),
        "wholeTaskSuccess": correct / len(rows),
        "traceValid": trace / len(rows),
        "correctTasks": correct,
        "generatedTokens": sum(row.get("generatedTokens", 0) for row in rows),
        "canonicalGeneratedTokens": sum(row.get("canonicalGeneratedTokens", 0) for row in rows),
        "observedTokens": sum(row.get("observedTokens", 0) for row in rows),
        "computeSeconds": sum(row.get("costs", {}).get("computeSeconds", 0) for row in rows),
        "hardGates": hard,
        "failureCounts": dict(Counter(row["failure"] for row in rows if row["failure"])),
    }
    summary["generatedTokensPerCorrectTask"] = (
        summary["generatedTokens"] / correct if correct else None
    )
    summary["computeSecondsPerCorrectTask"] = (
        summary["computeSeconds"] / correct if correct else None
    )
    return {
        "schema": "semantic-evaluation-report/4",
        "cohortDigest": cohort["digest"],
        "cohort": cohort["name"],
        "evaluationSplits": sorted({case["split"] for case in cohort["cases"]}),
        "seed": seed,
        "origin": origin,
        "representation": representation,
        "checkpointIdentity": checkpoint_identity,
        "tokenizerIdentity": codec.fingerprint,
        "catalogDigest": CONTRACT_DIGEST,
        "thresholds": cohort["thresholds"],
        "summary": summary,
        "cases": rows,
        "learnedSuccess": summary["wholeTaskSuccess"] if origin == "trained" else None,
        "acceptance": origin == "trained"
        and all(summary[name] >= threshold for name, threshold in cohort["thresholds"].items())
        and not any(hard.values()),
        "optimizerStepsDuringEvaluation": 0,
    }


def compare_experiments(reports, budgets):
    """Compare all seeds under declared equal-example/token or measured-compute budgets."""
    if len(reports) != len(budgets) or not reports:
        raise ValueError("Every evaluation needs its actual training budget receipt")
    cohorts = {report["cohortDigest"] for report in reports}
    origins = {report["origin"] for report in reports}
    if len(cohorts) != 1 or len(origins) != 1:
        raise ValueError("Never pool different frozen cohorts or engineering and learned evidence")
    if origins == {"trained"} and any(
        report.get("evaluationSplits") != ["test"] for report in reports
    ):
        raise ValueError("Learned budget comparisons must use the frozen test-only cohort")
    groups = {}
    for report, budget in zip(reports, budgets, strict=True):
        if (
            budget.get("schema") != "semantic-experiment-budget/4"
            or budget.get("seed") != report["seed"]
            or budget.get("representation") != report["representation"]
            or budget.get("checkpointIdentity") != report["checkpointIdentity"]
            or budget.get("cohortDigest") != report["cohortDigest"]
        ):
            raise ValueError("Training budget receipt does not belong to its evaluation")
        regime = budget.get("regime")
        if regime not in {"fixed-examples", "fixed-tokens", "measured-compute"}:
            raise ValueError("Unregistered equal-budget experiment regime")
        for field in ("examples", "generatedTokens", "observedTokens", "computeSeconds"):
            if (
                type(budget.get(field)) not in (int, float)
                or not math.isfinite(budget[field])
                or budget[field] < 0
            ):
                raise ValueError("Comparison needs measured example/token/compute counts")
        groups.setdefault(regime, []).append((report, budget))
    result = {}
    for regime, group in groups.items():
        representations = REPRESENTATIONS
        expected = {(seed, representation) for seed in SEEDS for representation in representations}
        pairs = [(report["seed"], report["representation"]) for report, _ in group]
        if len(pairs) != len(set(pairs)) or set(pairs) != expected:
            raise ValueError(
                "Budget reports require the one canonical representation at seeds 17, 29 and 43"
            )
        meter = {"fixed-examples": "examples", "fixed-tokens": "generatedTokens"}.get(regime)
        if meter == "examples" and len({budget[meter] for _, budget in group}) != 1:
            raise ValueError("An equal-example comparison cannot hide differing actual exposure")
        if meter == "generatedTokens":
            if (
                len({budget.get("limit") for _, budget in group}) != 1
                or len({budget.get("maxTokenOvershoot") for _, budget in group}) != 1
                or any(
                    not isinstance(budget.get("limit"), (int, float))
                    or not isinstance(budget.get("maxTokenOvershoot"), int)
                    or not budget["limit"]
                    <= budget[meter]
                    <= budget["limit"] + budget["maxTokenOvershoot"]
                    for _, budget in group
                )
            ):
                raise ValueError(
                    "Fixed-token runs must retain the same limit and declared atomic-window allowance"
                )
        result[regime] = {
            representation: {
                "meanSuccess": sum(
                    report["summary"]["wholeTaskSuccess"]
                    for report, _ in group
                    if report["representation"] == representation
                )
                / len(SEEDS),
                "runs": [
                    {"report": report, "budget": budget}
                    for report, budget in group
                    if report["representation"] == representation
                ],
            }
            for representation in representations
        }
    return {
        "schema": "semantic-experiment-comparison/4",
        "cohortDigest": next(iter(cohorts)),
        "origin": next(iter(origins)),
        "regimes": result,
        "learnedEvidence": origins == {"trained"},
    }
