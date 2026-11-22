from __future__ import annotations

import os
import sys
from pathlib import Path

from megatoken.semantic_request_journal_v8 import (
    DurableNativeRequestJournal,
    NativeRequestBusy,
)

ROOT = Path(os.environ["JOURNAL_ROOT"])
MODE = os.environ["JOURNAL_MODE"]
IDENTITY = {
    "schema": "semantic-native-request-identity/1",
    "session_id": "session-a",
    "stream_id": "stream-a",
    "session_generation": 1,
    "ordinal": 1,
    "operation": "generate",
    "model_identity": {"checkpoint": "a" * 64, "tokenizer": "b" * 64},
}
REQUEST = b"request"


class CrashingJournal(DurableNativeRequestJournal):
    def _after_temp_fsync(self, target_name, temp_path):
        if MODE == "before-admission" and target_name.endswith(".intent.json"):
            os._exit(70)

    def _after_published(self, target_name, path):
        if MODE == "after-outcome" and target_name.endswith(".outcome.json"):
            os._exit(72)

    def _after_link_before_temp_unlink(self, target_name, path, temp):
        if MODE == "after-link-intent" and target_name.endswith(".intent.json"):
            os._exit(74)
        if MODE == "after-link-outcome" and target_name.endswith(".outcome.json"):
            os._exit(75)
        if MODE == "after-link-superseded" and target_name.endswith(".superseded.json"):
            os._exit(76)


journal = CrashingJournal(ROOT)
if MODE == "after-admission":
    journal.execute(IDENTITY, REQUEST, lambda _: os._exit(71))
elif MODE == "after-outcome":
    journal.execute(IDENTITY, REQUEST, lambda _: b"response")
elif MODE == "after-link-intent":
    journal.execute(IDENTITY, REQUEST, lambda _: os._exit(71))
elif MODE == "after-link-outcome":
    journal.execute(IDENTITY, REQUEST, lambda _: b"response")
elif MODE == "after-link-superseded":
    journal.execute(IDENTITY, REQUEST, lambda _: b"response")
    journal.supersede(IDENTITY, REQUEST, {"op": "select", "session": "session-a"})
elif MODE == "before-admission":
    journal.execute(IDENTITY, REQUEST, lambda _: b"must-not-run")
elif MODE == "hold":

    def wait_for_release(_):
        print("ADMITTED", flush=True)
        sys.stdin.readline()
        return b"response"

    result = journal.execute(IDENTITY, REQUEST, wait_for_release)
    print("RESULT:" + result.decode("ascii"), flush=True)
elif MODE == "contend":
    try:
        journal.execute(IDENTITY, REQUEST, lambda _: b"wrong")
    except NativeRequestBusy:
        raise SystemExit(73) from None
    raise SystemExit(74)
else:
    raise SystemExit("unknown test mode")
