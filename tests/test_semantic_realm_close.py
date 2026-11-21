"""A journaled realm closure is terminal in both existing envelope consumers."""

import copy

import pytest
from test_detailed_v7 import _interop_exports
from test_semantic_harness_v6 import fixture

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.detailed_harness_v7 import adapt_detailed_harness_v7
from megatoken.data.semantic_harness_v6 import adapt_selected_harness_v6


def _case(version):
    if version == 6:
        return fixture(), adapt_selected_harness_v6
    data = next(row["export"] for row in _interop_exports() if row["case"] == "branch-1")
    return copy.deepcopy(data), adapt_detailed_harness_v7


def _append(data, kind, track, **fields):
    public = data["public"]
    rows = public["timeline"]
    rows.append(
        {
            "schema": public["schema"],
            "session": rows[0]["session"],
            "sequence": len(rows) + 1,
            "ownerSequence": sum(row["track"] == track for row in rows) + 1,
            "track": track,
            "mode": "observed",
            "kind": kind,
            **fields,
        }
    )
    if "publicDigest" in data["sidecar"]:
        data["sidecar"]["publicDigest"] = stable_algorithm_hash(public)


def _close(data):
    calls = [row for row in data["public"]["timeline"] if row["kind"] == "invocation"]
    assert len({row["realmId"] for row in calls}) == 1
    for call in calls:
        _append(
            data,
            "cancel",
            call["track"],
            invocationId=call["invocationId"],
            reason="finished realm",
            uncertain=[],
        )
    _append(data, "realm-close", "host", realmId=calls[0]["realmId"], reason="finished realm")
    return calls[0]


@pytest.mark.parametrize("version", [6, 7])
def test_journaled_realm_close_replays_but_never_reopens_or_accepts_a_second_closure(version):
    data, adapt = _case(version)
    call = _close(data)
    _, receipt = adapt(data, allow_local=True)
    assert receipt.invocations == 1
    repeated = copy.deepcopy(data)
    _append(repeated, "realm-close", "host", realmId=call["realmId"], reason="again")
    with pytest.raises(ValueError, match="Realm close"):
        adapt(repeated, allow_local=True)

    fields = {
        key: value
        for key, value in call.items()
        if key not in {"schema", "session", "sequence", "ownerSequence", "track", "mode", "kind"}
    }
    fields.update(
        invocationId="resurrected",
        triggerId="newtrigger",
        ingressKey="newtrigger",
        realmPolicy="retained",
    )
    _append(data, "invocation", call["track"], **fields)
    with pytest.raises(ValueError, match="retained JS realm"):
        adapt(data, allow_local=True)


@pytest.mark.parametrize("version", [6, 7])
def test_realm_close_requires_prior_cancellation_and_rejects_forged_authority(version):
    original, adapt = _case(version)
    call = next(row for row in original["public"]["timeline"] if row["kind"] == "invocation")
    _append(original, "realm-close", "host", realmId=call["realmId"], reason="too early")
    with pytest.raises(ValueError, match="cancellation"):
        adapt(original, allow_local=True)
    for fields in ({"track": call["track"]}, {"reason": ""}, {"realmId": "unknown"}):
        data, _ = _case(version)
        _close(data)
        data["public"]["timeline"][-1].update(fields)
        if "publicDigest" in data["sidecar"]:
            data["sidecar"]["publicDigest"] = stable_algorithm_hash(data["public"])
        with pytest.raises(ValueError):
            adapt(data, allow_local=True)


@pytest.mark.parametrize("version", [6, 7])
def test_closed_realms_require_handler_retirement_and_refuse_new_subscriptions(version):
    data, adapt = _case(version)
    call = next(row for row in data["public"]["timeline"] if row["kind"] == "invocation")
    handler = {
        "subscriptionId": "subscription1",
        "programId": call["programId"],
        "realmId": call["realmId"],
        "trackId": call["track"],
        "realmPolicy": "retained",
        "entry": call["entry"],
        "queueLimit": 8,
        "closureId": None,
        "lifetime": "subscription",
        "ownerInvocationId": call["invocationId"],
    }
    _append(data, "handler", "host", **handler)
    unretired = copy.deepcopy(data)
    _close(unretired)
    with pytest.raises(ValueError, match="handler retirement"):
        adapt(unretired, allow_local=True)
    _append(data, "handler-close", "host", subscriptionId="subscription1", discarded=[])
    _close(data)
    adapt(data, allow_local=True)
    _append(data, "handler", "host", **{**handler, "subscriptionId": "subscription2"})
    with pytest.raises(ValueError, match="Handler registration"):
        adapt(data, allow_local=True)
