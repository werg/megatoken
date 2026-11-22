"""V8 request/result replay must use its own typed records, never witness_v6."""

import builtins
from unittest.mock import patch

from megatoken.data.detailed_v8 import encode_record
from megatoken.data.native_v8 import _apply_detail, _apply_observation


def test_request_dispatch_and_result_replay_with_witness_v6_unavailable():
    state = {
        "heap": [],
        "locals": [],
        "captureAliases": [],
        "effects": [],
        "stack": ["root"],
        "frameFunctions": [["root", "run"]],
        "scheduler": {"ready": 0},
    }
    obligations = {"intents": [], "handlers": [], "callbacks": [], "closureQueue": []}
    request = {
        "effect": "e1",
        "operation": "demo.lookup",
        "arguments": [["str", "key"]],
        "invocation": "root",
        "site": {"start": 4, "end": 22},
        "payload": '["key"]',
    }
    request_line = encode_record("request", request, role="generated")
    request_row = {
        "sequence": 1,
        "fields": {"session": "session1", "track": "worker1", "line": request_line},
    }
    original_import = builtins.__import__

    def import_without_v6(name, globals=None, locals=None, fromlist=(), level=0):
        package = (globals or {}).get("__package__", "")
        if (
            name in {"megatoken.data.witness_v6", ".witness_v6"}
            or "witness_v6" in fromlist
            and (name in {"", "."} or package == "megatoken.data")
        ):
            raise AssertionError("V8 effects attempted to import the V6 witness renderer")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=import_without_v6):
        # Confirm this hook catches the exact relative-import form used by the
        # old implementation (`from . import witness_v6`).
        try:
            __import__(
                "",
                {"__package__": "megatoken.data"},
                {},
                ("witness_v6",),
                1,
            )
        except AssertionError as error:
            assert "V6 witness renderer" in str(error)
        else:
            raise AssertionError("V6 relative import was not blocked")
        _apply_detail(
            state,
            "request",
            request,
            request_row,
            obligations,
            {},
            {"root": 1},
        )
        dispatch = {
            "sequence": 2,
            "kind": "dispatch",
            "fields": {
                "invocationId": "root",
                "effectId": "e1",
                "key": "root/e1",
                "attempt": 1,
            },
        }
        _apply_observation(state, dispatch, set(), set(), obligations, {})
        result = {"effect": "e1", "result": ["str", "done"], "status": "fulfilled"}
        result_line = encode_record("observe", result, role="observed")
        receipt = {
            "sequence": 3,
            "kind": "result",
            "fields": {
                "session": "session1",
                "track": "worker1",
                "invocationId": "root",
                "effectId": "e1",
                "status": "fulfilled",
                "result": ["str", "done"],
                "line": result_line,
                "errorOrigin": None,
            },
        }
        _apply_observation(state, receipt, {"root/e1"}, set(), obligations, {3: 2})

    intent = dict(obligations["intents"])["root/e1"]
    assert intent["witness"] == request_line
    assert intent["result"] == {
        "schema": "semantic-lifecycle/8",
        "session": "session1",
        "sequence": 3,
        "ownerSequence": 2,
        "track": "worker1",
        "mode": "observed",
        "kind": "result",
        "invocationId": "root",
        "effectId": "e1",
        "status": "fulfilled",
        "result": ["str", "done"],
        "line": result_line,
        "errorOrigin": None,
    }
