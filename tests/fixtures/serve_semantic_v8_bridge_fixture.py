from __future__ import annotations

import copy
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from megatoken.data.semantic_json import stable_algorithm_hash

ROOT = Path(__file__).resolve().parents[2]
script = ROOT / "scripts" / "serve_semantic_v8.py"
namespace = runpy.run_path(str(script))["main"].__globals__
model_identity = {"checkpoint": "a" * 64, "tokenizer": "b" * 64}


class Runtime:
    def __init__(self):
        self.memory = SimpleNamespace(sessions={})

    def close(self):
        return None


class Adapter:
    def __init__(self, runtime, **kwargs):
        self.runtime = runtime
        self.model_identity = model_identity["checkpoint"]
        self.codec = SimpleNamespace(fingerprint=model_identity["tokenizer"])
        self.training_identity = None
        self.initialization = {
            "checkpointIdentity": self.model_identity,
            "tokenizerIdentity": self.codec.fingerprint,
            "trainingIdentity": None,
        }
        self.sessions = {}
        self.attempts = []
        self.state = None

    def checkpoint(self):
        if self.state is None:
            raise ValueError("fresh adapter has no checkpoint")
        return copy.deepcopy(self.state)

    def restore(self, *, checkpoint):
        if self.state is not None:
            raise ValueError("restore requires empty adapter")
        self.state = copy.deepcopy(checkpoint)
        self.attempts = copy.deepcopy(checkpoint["attempts"])

    def cancel(self, **kwargs):
        return {"closed": True}

    def metrics(self):
        return {}


bundle = SimpleNamespace(model=None, chunking=None, manifest={"model_sha256": model_identity["checkpoint"]})
namespace.update(
    ServedBundle=lambda *args, **kwargs: bundle,
    MultiTrackGenerator=SimpleNamespace(of=lambda model: None),
    MultiTrackRuntime=lambda *args, **kwargs: Runtime(),
    NativeSemanticInferenceV8=Adapter,
)


def dispatch(adapter, request):
    with open(os.environ["CALLS_PATH"], "a", encoding="utf-8") as output:
        output.write(request["id"] + "\n")
    if os.environ.get("FAIL_AFTER_DISPATCH") == "1":
        raise RuntimeError("injected failure after commit dispatch")
    public = request["public"]
    session = request["session"]
    selection_id = request["selection"]["selectionId"]
    adapter.attempts = [{"selectionId": selection_id, "status": "committed", "session": session}]
    adapter.state = {
        "schema": "semantic-model-checkpoint/8",
        "initialization": adapter.initialization,
        "attempts": adapter.attempts,
        "sessions": {
            session: {"public": copy.deepcopy(public), "unacknowledgedSelection": None}
        },
    }
    adapter.state["digest"] = stable_algorithm_hash(adapter.state)
    return {
        "schema": "semantic-native-commit/8",
        "selectionId": selection_id,
        "publicDigest": stable_algorithm_hash(public),
        "committed": True,
    }


namespace["dispatch"] = dispatch
sys.argv = [
    str(script),
    "--served",
    "unused",
    "--request-journal",
    os.environ["JOURNAL_PATH"],
    "--execution-lock-fd",
    os.environ["LOCK_FD"],
    "--execution-lock-session",
    os.environ["LOCK_SESSION"],
    "--execution-lock-stream",
    os.environ["LOCK_STREAM"],
    "--execution-lock-path",
    os.environ["LOCK_PATH"],
]
raise SystemExit(namespace["main"]())
