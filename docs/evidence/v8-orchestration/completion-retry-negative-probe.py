"""Model-free negative controls for the historical V8 completion repair."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import runpy

EVIDENCE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
with contextlib.redirect_stdout(io.StringIO()):
    setup = runpy.run_path(str(EVIDENCE / "completion-retry-probe.py"))["setup"]


def negative(mutation):
    adapter, state, request = setup()
    adapter.complete(**request)
    state.runtime.tracks.change(state.orchestration["track"], "complete")
    request = copy.deepcopy(request)
    mutation(request)
    before = copy.deepcopy((state.orchestration, state.runtime.tracks.checkpoint(), state.public))
    try:
        adapter.complete(**request)
        result = {"refused": False}
    except ValueError as error:
        result = {"refused": True, "error": str(error)}
    result["authorityUnchanged"] = before == (
        state.orchestration, state.runtime.tracks.checkpoint(), state.public
    )
    return result


def changed_source(request):
    request["public"]["timeline"][0]["fields"]["source"] += " "


def changed_context(request):
    request["contexts"][0]["prefixDigest"] = "0" * 64


adapter, state, request = setup()
original = adapter.complete(**request)
adapter.sessions.pop(request["session"])
adapter.closed_sessions[request["session"]] = {"orchestration": copy.deepcopy(state.orchestration)}
before_closed = copy.deepcopy(adapter.closed_sessions)
closed = adapter.complete(**request)

print(json.dumps({
    "scope": "model-free immutable receipt read; assembled saved state; no model checkpoint claim",
    "hashes": {
        name: hashlib.sha256((ROOT / "src/megatoken" / name).read_bytes()).hexdigest()
        for name in ("semantic_model_v8.py", "semantic_orchestration_v8.py")
    },
    "controls": {
        "changedOriginalPublicCut": negative(changed_source),
        "changedOriginalContexts": negative(changed_context),
        "changedCompletion": negative(
            lambda request: request.update(completion={"kind": "return", "value": ["num", 99]})
        ),
        "unknownInvocation": negative(lambda request: request.update(invocationId="invocation99")),
        "closedSessionExactRetry": {
            "sameChildReceipt": closed["childCompletion"] == original["childCompletion"],
            "sessionRetained": closed["sessionRetained"],
            "noSessionReopened": not adapter.sessions,
            "savedAuthorityUnchanged": adapter.closed_sessions == before_closed,
        },
    },
}, indent=2, sort_keys=True))
