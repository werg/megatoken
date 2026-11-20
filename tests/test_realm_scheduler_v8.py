"""Global Promise readiness is independent of controller traversal order."""

import copy
import json

import pytest

from megatoken.data.native_v8 import (
    NativeV8Error,
    _assert_realm_scheduler_turn,
    _initial_state,
    _refresh_realm_readiness,
)


def peers():
    controllers = {}
    for owner in ("root", "callback"):
        state = _initial_state(
            {},
            {"invocationId": owner, "entry": "run", "inputGraph": {"roots": [], "objects": []}},
        )
        state["stack"] = []
        state["scheduler"] = {"registration": 5, "ready": 7}
        state["heap"] = [
            ["shared", {"kind": "promise", "state": "fulfilled", "result": ["num", 7]}]
        ]
        controllers[owner] = state
    root, callback = controllers.values()
    root["jobs"] = [
        ["earlier", {"owner": "root", "status": "queued", "readySequence": 5}],
        [
            "root-job",
            {
                "owner": "root",
                "status": "queued",
                "promise": "shared",
                "sequence": 3,
                "readySequence": 6,
            },
        ],
    ]
    root["continuations"] = [
        ["root-await", {"state": "waiting", "promise": "shared", "sequence": 5}]
    ]
    root["ready"] = ["earlier", "root-job"]
    callback["jobs"] = [
        [
            "callback-first",
            {"owner": "callback", "status": "waiting", "promise": "shared", "sequence": 1},
        ],
        [
            "callback-last",
            {
                "owner": "callback",
                "status": "queued",
                "promise": "shared",
                "sequence": 4,
                "readySequence": 7,
            },
        ],
    ]
    callback["continuations"] = [
        ["callback-await", {"state": "waiting", "promise": "shared", "sequence": 2}]
    ]
    callback["ready"] = ["callback-last"]
    return controllers


def test_shared_pending_promise_publishes_waiters_in_global_registration_order():
    controllers = peers()
    _refresh_realm_readiness(controllers, 5)
    expected = {
        "earlier": 5,
        "callback-first": 6,
        "callback-await": 7,
        "root-job": 8,
        "callback-last": 9,
        "root-await": 10,
    }
    assert {
        identity: item["readySequence"]
        for state in controllers.values()
        for identity, item in state["jobs"] + state["continuations"]
    } == expected
    assert controllers["root"]["ready"] == ["earlier", "root-job", "root-await"]
    assert controllers["callback"]["ready"] == ["callback-first", "callback-await", "callback-last"]
    assert all(
        state["scheduler"] == {"registration": 5, "ready": 10} for state in controllers.values()
    )
    restored = json.loads(json.dumps(controllers))
    _refresh_realm_readiness(restored, 10)
    assert restored == controllers
    reversed_controllers = dict(reversed(list(peers().items())))
    _refresh_realm_readiness(reversed_controllers, 5)
    assert reversed_controllers == controllers


def test_global_turn_preserves_preexisting_microtask_then_oldest_ready_waiter():
    controllers = peers()
    _refresh_realm_readiness(controllers, 5)
    _assert_realm_scheduler_turn(controllers, "root", "earlier")
    with pytest.raises(NativeV8Error, match="realm-wide FIFO"):
        _assert_realm_scheduler_turn(controllers, "callback", "callback-first")
    controllers["root"]["ready"].pop(0)
    _assert_realm_scheduler_turn(controllers, "callback", "callback-first")
    with pytest.raises(NativeV8Error, match="realm-wide FIFO"):
        _assert_realm_scheduler_turn(controllers, "root", "root-job")
    controllers["root"]["activeJob"] = "earlier"
    with pytest.raises(NativeV8Error, match="unfinished callback"):
        _assert_realm_scheduler_turn(controllers, "callback", "callback-first")


@pytest.mark.parametrize("mutation", ["registration", "owner", "scheduler"])
def test_invalid_shared_readiness_is_rejected_before_any_queue_changes(mutation):
    controllers = peers()
    if mutation == "registration":
        controllers["callback"]["jobs"][0][1]["sequence"] = 3
    elif mutation == "owner":
        controllers["callback"]["jobs"][0][1]["owner"] = "root"
    else:
        controllers["callback"]["scheduler"]["ready"] += 1
    before = copy.deepcopy(controllers)
    with pytest.raises(NativeV8Error, match="registration|another invocation|scheduler counters"):
        _refresh_realm_readiness(controllers, 5)
    assert controllers == before


def test_explicit_enqueue_after_cutoff_keeps_its_published_order():
    controllers = peers()
    controllers["root"]["jobs"][0][1]["readySequence"] = 8
    for state in controllers.values():
        state["scheduler"]["ready"] = 8
    _refresh_realm_readiness(controllers, 5)
    assert controllers["root"]["jobs"][0][1]["readySequence"] == 8
    assert controllers["callback"]["jobs"][0][1]["readySequence"] == 9
