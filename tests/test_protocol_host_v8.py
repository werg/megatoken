"""Independent host ownership checks across retained V8 invocation controllers."""

import copy

import pytest

from megatoken.data.detailed_v8 import encode_record, parse_record
from megatoken.data.native_v8 import (
    NativeV8Error,
    _apply_detail,
    _apply_observation,
    _initial_state,
    _replay_shared_state,
    validate_v8_artifact,
)
from tests.test_native_observation_boundaries_v8 import pending_delivery, result_row
from tests.test_prediction_prefix_v8 import examples


def callback_example():
    return examples("semantic-callback-regions-v8.json.gz.b64")[0]


def through(example, sequence):
    return {
        **example,
        "public": {**example["public"], "timeline": example["public"]["timeline"][:sequence]},
        "contexts": [row for row in example["contexts"] if row["visibleThrough"] < sequence],
    }


def test_pending_callback_capability_is_reconstructed_before_dispatch():
    example = callback_example()
    request = next(
        row
        for row in example["public"]["timeline"]
        if row["kind"] == "semantic" and parse_record(row["fields"]["line"])[0] == "request"
    )
    cut = through(example, request["sequence"])
    boundary = _replay_shared_state(
        cut["public"]["timeline"],
        cut["contexts"],
        {},
        roundtrip=True,
        boundary_invocation=request["fields"]["invocationId"],
    )
    # The fixture's later observation independently names the same capability.
    observed = next(row for row in example["contexts"] if row["obligations"]["callbacks"])
    expected = copy.deepcopy(observed["obligations"])
    expected["handlers"] = []
    expected["intents"][0][1].update(status="committed", attempts=0)
    expected["callbacks"][0][1]["state"] = "pending"
    assert boundary["obligations"] == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [("realmId", "foreign-realm"), ("lifetime", "subscription"), ("queueLimit", 1)],
)
def test_handler_cannot_change_a_pending_callback_capability(field, value):
    example = callback_example()
    handler = next(row for row in example["public"]["timeline"] if row["kind"] == "handler")
    cut = copy.deepcopy(through(example, handler["sequence"]))
    cut["public"]["timeline"][-1]["fields"][field] = value
    with pytest.raises(
        NativeV8Error, match="pending callback capability|retained realm source or owner"
    ):
        validate_v8_artifact(cut)


def test_separate_host_registration_may_use_the_same_pending_guest_closure():
    example = callback_example()
    handler = next(row for row in example["public"]["timeline"] if row["kind"] == "handler")
    cut = copy.deepcopy(through(example, handler["sequence"]))
    cut["public"]["timeline"][-1]["fields"]["subscriptionId"] = "outside"
    boundary = _replay_shared_state(
        cut["public"]["timeline"],
        cut["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="invocation1",
    )
    assert list(dict(boundary["obligations"]["handlers"])) == ["outside"]
    callback = boundary["obligations"]["callbacks"][0][1]
    assert callback["subscriptionId"] == "callback1"
    assert callback["state"] == "pending"


def test_observed_duplicate_cannot_replace_a_retained_controller():
    example = callback_example()
    initial = example["public"]["timeline"][1]
    cut = copy.deepcopy(through(example, initial["sequence"]))
    duplicate = copy.deepcopy(initial)
    duplicate["sequence"] += 1
    cut["public"]["timeline"].append(duplicate)
    with pytest.raises(NativeV8Error, match="Duplicate invocation controller"):
        validate_v8_artifact(cut)


def test_generated_owner_cannot_interrupt_another_running_callback():
    example = callback_example()
    callback = next(row for row in example["contexts"] if row["invocationId"] == "event1")
    cut = copy.deepcopy(through(example, callback["visibleThrough"] + 1))
    cut["public"]["timeline"][-1]["fields"]["invocationId"] = "invocation1"
    with pytest.raises(NativeV8Error, match="interrupts another invocation"):
        _replay_shared_state(cut["public"]["timeline"], cut["contexts"], {}, roundtrip=False)


def test_effect_ids_are_owned_by_invocation_and_receipts_cannot_cross_owners():
    dispatched, settled = set(), set()
    obligations = {"intents": [], "handlers": [], "callbacks": [], "closureQueue": []}
    for owner in ("root", "callback"):
        state = _initial_state(
            {}, {"invocationId": owner, "entry": "run", "inputGraph": {"roots": [], "objects": []}}
        )
        request = {
            "invocation": owner,
            "effect": "e1",
            "operation": "demo.lookup",
            "site": {"start": 0, "end": 8},
            "arguments": [["str", "key"]],
            "payload": '["key"]',
        }
        _apply_detail(state, "request", request, {}, obligations, root=owner)
        dispatch = {
            "sequence": 1,
            "kind": "dispatch",
            "fields": {"invocationId": owner, "effectId": "e1", "key": owner + "/e1", "attempt": 1},
        }
        forged = copy.deepcopy(dispatch)
        forged["fields"].update(invocationId="foreign", key="foreign/e1")
        before = copy.deepcopy((state, dispatched, settled, obligations))
        with pytest.raises(NativeV8Error, match="another invocation"):
            _apply_observation(state, forged, dispatched, settled, obligations)
        assert (state, dispatched, settled, obligations) == before
        _apply_observation(state, dispatch, dispatched, settled, obligations)
        fields = {"effect": "e1", "status": "fulfilled", "result": ["num", 7]}
        receipt = {
            "sequence": 2,
            "kind": "result",
            "fields": {
                "session": "session",
                "track": "worker",
                "invocationId": owner,
                "effectId": "e1",
                "status": "fulfilled",
                "result": ["num", 7],
                "line": encode_record("observe", fields, role="observed"),
            },
        }
        forged = copy.deepcopy(receipt)
        forged["fields"]["invocationId"] = "foreign"
        before = copy.deepcopy((state, dispatched, settled, obligations))
        with pytest.raises(NativeV8Error, match="another invocation"):
            _apply_observation(state, forged, dispatched, settled, obligations)
        assert (state, dispatched, settled, obligations) == before
        _apply_observation(state, receipt, dispatched, settled, obligations)
    assert dispatched == settled == {"root/e1", "callback/e1"}
    assert all(row[1]["status"] == "settled" for row in obligations["intents"])


def test_reaction_request_keeps_source_frame_but_host_intent_belongs_to_controller():
    state = _initial_state(
        {}, {"invocationId": "callback", "entry": "run", "inputGraph": {"roots": [], "objects": []}}
    )
    state["stack"] = ["job:reaction"]
    state["frameFunctions"].append(["job:reaction", "handler"])
    obligations = {"intents": [], "handlers": [], "callbacks": [], "closureQueue": []}
    request = {
        "invocation": "job:reaction",
        "effect": "e1",
        "operation": "demo.store",
        "site": {"start": 0, "end": 8},
        "arguments": [["str", "key"], ["num", 7]],
        "payload": '["key",7]',
    }
    before = copy.deepcopy((state, obligations))
    with pytest.raises(NativeV8Error, match="another invocation frame"):
        _apply_detail(
            state,
            "request",
            {**request, "invocation": "callback"},
            {},
            obligations,
            root="callback",
        )
    assert (state, obligations) == before
    _apply_detail(state, "request", request, {}, obligations, root="callback")
    assert state["effects"][0][1]["invocation"] == "job:reaction"
    assert obligations["intents"][0][0] == "callback/e1"
    assert obligations["intents"][0][1]["invocationId"] == "callback"
    dispatched, settled = set(), set()
    dispatch = {
        "kind": "dispatch",
        "sequence": 1,
        "fields": {
            "invocationId": "callback",
            "effectId": "e1",
            "key": "callback/e1",
            "attempt": 1,
        },
    }
    _apply_observation(state, dispatch, dispatched, settled, obligations)
    receipt = result_row(["num", 7])
    receipt["fields"]["invocationId"] = "job:reaction"
    before = copy.deepcopy((state, dispatched, settled, obligations))
    with pytest.raises(NativeV8Error, match="another invocation"):
        _apply_observation(state, receipt, dispatched, settled, obligations)
    assert (state, dispatched, settled, obligations) == before
    receipt["fields"]["invocationId"] = "callback"
    _apply_observation(state, receipt, dispatched, settled, obligations)
    assert dispatched == settled == {"callback/e1"}
    assert obligations["intents"][0][1]["status"] == "settled"


@pytest.mark.parametrize(
    ("lifetime", "status", "expected"),
    [
        ("call", "fulfilled", "revoked"),
        ("call", "rejected", "revoked"),
        ("subscription", "fulfilled", "active"),
        ("subscription", "rejected", "revoked"),
    ],
)
def test_observed_result_retires_call_callbacks_and_failed_subscriptions(
    lifetime, status, expected
):
    state, dispatched, settled, obligations = pending_delivery()
    obligations["callbacks"] = [
        [
            "root/e1/1",
            {
                "state": "active",
                "lifetime": lifetime,
                "subscriptionId": "callback1",
                "closureId": "closure",
            },
        ]
    ]
    obligations["handlers"] = [["callback1", {"state": "active", "seen": [], "queue": []}]]
    _apply_observation(
        state, result_row(["num", 7], status=status), dispatched, settled, obligations
    )
    assert obligations["callbacks"][0][1]["state"] == expected
    assert obligations["handlers"][0][1]["state"] == (
        "closing" if expected == "revoked" else "active"
    )
    # Previously queued native arrivals remain explicitly observed while a
    # handler drains. The close receipt prevents any subsequent delivery.
    event = {"kind": "event", "fields": {"subscriptionId": "callback1", "eventId": "event1"}}
    _apply_observation(state, event, dispatched, settled, obligations)
    assert obligations["handlers"][0][1]["seen"] == ["event1"]
    close = {"kind": "handler-close", "fields": {"subscriptionId": "callback1", "discarded": []}}
    _apply_observation(state, close, dispatched, settled, obligations)
    before = copy.deepcopy((state, dispatched, settled, obligations))
    with pytest.raises(NativeV8Error, match="active source handler"):
        _apply_observation(state, event, dispatched, settled, obligations)
    assert (state, dispatched, settled, obligations) == before
