"""Recovery authority is reconstructed from observed cancellation and delivery."""

import copy

import pytest

from megatoken.data.native_v8 import NativeV8Error, _apply_observation, validate_v8_artifact
from megatoken.data.prefix_v8 import validate_v8_committed_prefix, validate_v8_prediction_prefix
from megatoken.data.semantic_json import stable_algorithm_hash
from tests.test_native_observation_boundaries_v8 import pending_delivery
from tests.test_prediction_prefix_v8 import examples
from tests.test_protocol_host_v8 import callback_example, through


def observation(example, kind, **fields):
    first = example["public"]["timeline"][1]["fields"]
    row = {
        "sequence": len(example["public"]["timeline"]) + 1,
        "mode": "observed",
        "kind": kind,
        "fields": {
            "session": first["session"],
            "track": "host" if kind in {"realm-close", "handler-close"} else first["track"],
            **fields,
        },
    }
    example["public"]["timeline"].append(row)
    return row


def empty_realm():
    example = copy.deepcopy(through(callback_example(), 2))
    return example, example["public"]["timeline"][1]["fields"]


def cancel(example, owner="invocation1", uncertain=()):
    return observation(
        example, "cancel", invocationId=owner, reason="test stop", uncertain=list(uncertain)
    )


def close(example, realm):
    return observation(example, "realm-close", realmId=realm, reason="test stop")


def test_cancel_then_explicit_realm_close_survives_public_replay_and_cannot_reopen():
    example, invocation = empty_realm()
    cancel(example, invocation["invocationId"])
    close(example, invocation["realmId"])
    validate_v8_committed_prefix(example["public"], example["contexts"])
    reopened = copy.deepcopy(example["public"]["timeline"][1])
    reopened["sequence"] = len(example["public"]["timeline"]) + 1
    reopened["fields"].update(invocationId="another", triggerId="another", realmPolicy="retained")
    example["public"]["timeline"].append(reopened)
    with pytest.raises(NativeV8Error, match="Closed realm cannot be resurrected"):
        validate_v8_artifact(example)


@pytest.mark.parametrize(
    "fault", ("uncancelled", "unknown", "duplicate", "wrong-track", "empty-reason")
)
def test_realm_closure_requires_exact_preceding_lifecycle_authority(fault):
    example, invocation = empty_realm()
    if fault != "uncancelled":
        cancel(example, invocation["invocationId"])
    if fault == "duplicate":
        close(example, invocation["realmId"])
    row = close(example, "unknown" if fault == "unknown" else invocation["realmId"])
    if fault == "wrong-track":
        row["fields"]["track"] = invocation["track"]
    if fault == "empty-reason":
        row["fields"]["reason"] = ""
    with pytest.raises(NativeV8Error, match="Realm closure"):
        validate_v8_artifact(example)


def test_cancelled_invocation_cannot_select_again_even_with_rehashed_context():
    original = callback_example()
    example, invocation = empty_realm()
    cancel(example, invocation["invocationId"])
    context = copy.deepcopy(original["contexts"][0])
    context["visibleThrough"] = len(example["public"]["timeline"])
    context["prefixDigest"] = stable_algorithm_hash(example["public"]["timeline"])
    selected = copy.deepcopy(original["public"]["timeline"][2])
    selected["sequence"] = context["visibleThrough"] + 1
    example["public"]["timeline"].append(selected)
    example["contexts"].append(context)
    with pytest.raises(NativeV8Error, match="follows invocation cancellation"):
        validate_v8_artifact(example)


def test_cancel_receipt_keeps_dispatched_effect_uncertainty_and_requires_handler_close():
    original = callback_example()
    handler = next(row for row in original["public"]["timeline"] if row["kind"] == "handler")
    example = copy.deepcopy(through(original, handler["sequence"]))
    dispatched = next(row for row in example["public"]["timeline"] if row["kind"] == "dispatch")
    owner, effect = (dispatched["fields"][key] for key in ("invocationId", "effectId"))
    forged = copy.deepcopy(example)
    cancel(forged, owner)
    with pytest.raises(NativeV8Error, match="uncertain effects differ"):
        validate_v8_artifact(forged)
    cancel(example, owner, [effect])
    unfinished = copy.deepcopy(example)
    close(unfinished, handler["fields"]["realmId"])
    with pytest.raises(NativeV8Error, match="precedes handler closure"):
        validate_v8_artifact(unfinished)
    observation(
        example, "handler-close", subscriptionId=handler["fields"]["subscriptionId"], discarded=[]
    )
    close(example, handler["fields"]["realmId"])
    validate_v8_committed_prefix(example["public"], example["contexts"])


def test_callback_allocator_advances_only_from_delivered_canonical_event_ids():
    state, dispatched, settled, obligations = pending_delivery()
    obligations["callbacks"] = [
        [
            "root/e1/1",
            {
                "state": "active",
                "lifetime": "subscription",
                "subscriptionId": "callback1",
                "closureId": "closure",
                "sequence": 0,
                "queued": [],
            },
        ]
    ]
    obligations["handlers"] = [["callback1", {"state": "active", "seen": [], "queue": []}]]
    for event_id, sequence in (
        ("callbackevent1", 1),
        ("callbackevent3", 3),
        ("callbackevent2", 3),
        ("teacherevent9", 3),
        ("callbackevent01", 3),
        ("callbackevent9007199254740992", 3),
    ):
        _apply_observation(
            state,
            {
                "kind": "event",
                "fields": {
                    "subscriptionId": "callback1",
                    "eventId": event_id,
                },
            },
            dispatched,
            settled,
            obligations,
        )
        assert obligations["callbacks"][0][1]["sequence"] == sequence
        assert obligations["callbacks"][0][1]["queued"] == []
    before = copy.deepcopy((state, dispatched, settled, obligations))
    with pytest.raises(NativeV8Error, match="Duplicate source event"):
        _apply_observation(
            state,
            {
                "kind": "event",
                "fields": {
                    "subscriptionId": "callback1",
                    "eventId": "callbackevent3",
                },
            },
            dispatched,
            settled,
            obligations,
        )
    assert (state, dispatched, settled, obligations) == before


def test_actual_native_callback_recovery_fixture_matches_independent_public_replay():
    example = examples("semantic-host-recovery-v8.json.gz.b64")[0]
    assert example["qualification"] == "controlled-machine-regression-not-source-certified"
    receipt = validate_v8_artifact(example)
    assert receipt["counts"]["contexts"] == 10
    assert receipt["checks"]["continuation"] == "passed"
    assert receipt["checks"]["sourceBinding"] == "untested"
    validate_v8_committed_prefix(example["public"], example["contexts"])
    midpoint = example["midpoint"]
    public = {**example["public"], "timeline": example["public"]["timeline"][: midpoint["cut"]]}
    contexts = [
        context for context in example["contexts"] if context["visibleThrough"] < midpoint["cut"]
    ]
    context = midpoint["context"]
    checked = validate_v8_prediction_prefix(public, contexts, context)
    assert checked["context"] == context
    assert all(
        callback["sequence"] == 2 and callback["queued"] == []
        for _, callback in context["obligations"]["callbacks"]
    )
    forged = copy.deepcopy(context)
    forged["obligations"]["callbacks"][0][1]["sequence"] += 1
    with pytest.raises(NativeV8Error, match="context differs from independent replay"):
        validate_v8_prediction_prefix(public, contexts, forged)
