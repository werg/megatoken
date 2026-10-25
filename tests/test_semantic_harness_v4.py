"""Cross-language controlled trajectory, dynamic tracks and no-future-observation checks."""

import copy
import hashlib
import json

import pytest

from megatoken.data import WindowConfig, iter_windows
from megatoken.data.schema import Episode
from megatoken.data.semantic_harness_v4 import (
    SelectedHarnessError, adapt_selected_harness_v4,
)
from megatoken.generation.codec import ByteCodec


def line(kind, fields):
    return kind + " " + json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def fixture():
    source = "function run(key){return host.readAsync(key);}"
    call = "host.readAsync(key)"
    start = source.index(call)
    sha = hashlib.sha256(source.encode()).hexdigest()
    events = []
    owners = {}

    def add(kind, mode, track, **fields):
        owners[track] = owners.get(track, 0) + 1
        events.append({
            "schema": "semantic-execution/4", "session": "session1",
            "sequence": len(events) + 1, "ownerSequence": owners[track],
            "track": track, "mode": mode, "kind": kind, **fields,
        })

    add("program", "observed", "host", programId="p", source=source,
        digest=sha, entry="run", capabilities=["demo.readAsync"],
        sites=[{"start": start, "end": start + len(call),
                "role": "CallExpression", "operation": "demo.readAsync"}])
    add("invocation", "observed", "worker:0", programId="p", invocationId="call1",
        realmId="realm1", triggerId="outside1", completionPolicy="immediate-root",
        entry="run", sourceDigest=sha, inputGraph={
            "roots": [["str", "counter"]], "objects": [],
        })
    add("semantic", "generated", "worker:0", invocationId="call1",
        traceFormat="semantic-trace/3", line=line("request", {
            "effect": "fx1", "operation": "demo.readAsync",
            "invocation": "i1",
            "site": {"start": start, "end": start + len(call)},
            "arguments": [["str", "counter"]], "payload": '["counter"]',
        }))
    add("dispatch", "observed", "worker:0", invocationId="call1", effectId="fx1",
        key="call1/fx1", attempt=1)
    add("result", "observed", "worker:0", invocationId="call1", effectId="fx1",
        line=line("observe", {
            "effect": "fx1", "status": "fulfilled", "result": ["num", 7],
        }), status="fulfilled", result=["num", 7])
    add("semantic", "generated", "worker:0", invocationId="call1",
        traceFormat="semantic-trace/3", line=line("return", {"value": ["num", 7]}))
    add("semantic", "generated", "worker:0", invocationId="call1",
        traceFormat="semantic-trace/3", line=line("end", {"kind": "return"}))
    return {
        "public": {"schema": "semantic-execution/4",
                   "traceFormat": "semantic-trace/3",
                   "profile": "selected-harness/1", "timeline": events},
        "sidecar": {"schema": "semantic-supervision/2",
                    "admission": "locally_verified",
                    "referenceStatus": "unverified"},
    }


def test_selected_harness_import_uses_actual_dynamic_track_and_one_bound_action():
    episode, receipt = adapt_selected_harness_v4(
        fixture(), allow_local=True, codec=ByteCodec.byte_fallback()
    )
    assert receipt.invocations == 1
    assert receipt.actions == 1
    assert receipt.observations == 4
    assert receipt.tracks == 2
    assert episode.actions[0].call_id == "session1:call1:fx1"
    assert episode.actions[0].tool_name == "demo.readAsync"
    assert json.loads(episode.actions[0].arguments_json) == {"args": ["counter"]}
    assert [r.policy_mask for r in episode.records] == [
        False, False, True, False, False, True, True,
    ]
    assert episode.records[4].result_of == episode.actions[0].action_id
    assert Episode.from_dict(episode.to_dict()) == episode
    assert list(iter_windows(episode, ByteCodec.byte_fallback(),
                             WindowConfig(max_records=16, max_decisions=16)))
    assert episode.supervision.outcome == "unknown"


@pytest.mark.parametrize("mutation", [
    lambda d: d["public"].update(traceFormat="semantic-trace/4"),
    lambda d: d["public"]["timeline"][1].update(sourceDigest="bad"),
    lambda d: d["public"]["timeline"][2].update(mode="observed"),
    lambda d: d["public"]["timeline"][2].update(track="other"),
    lambda d: d["public"]["timeline"][2].update(sequence=4),
    lambda d: d["public"]["timeline"][2].update(ownerSequence=8),
    lambda d: d["public"]["timeline"][2].update(line=line("observe", {
        "effect": "fx1", "status": "fulfilled", "result": ["num", 7],
    })),
    lambda d: d["public"]["timeline"][2].update(line=line("request", {
        "effect": "fx1", "operation": "demo.readAsync", "invocation": "i1",
        "site": {"start": 0, "end": 3},
        "arguments": [["str", "counter"]], "payload": '["counter"]',
    })),
    lambda d: d["public"]["timeline"][4].update(effectId="other"),
    lambda d: d["public"]["timeline"][4].update(result=["num", 999]),
    lambda d: d["public"]["timeline"][3].update(attempt=2),
])
def test_invalid_or_forged_public_provenance_is_rejected(mutation):
    doc = copy.deepcopy(fixture())
    mutation(doc)
    with pytest.raises(SelectedHarnessError):
        adapt_selected_harness_v4(doc, allow_local=True)


def test_future_result_cannot_be_placed_ahead_of_selected_request():
    doc = fixture()
    rows = doc["public"]["timeline"]
    # Chronology must be physically committed; renumbering doesn't hide the
    # causal violation. A delivery before a request has no matching action.
    rows[2], rows[4] = rows[4], rows[2]
    for i, event in enumerate(rows):
        event["sequence"] = i + 1
        if event["track"] == "worker:0":
            event["ownerSequence"] = i
    with pytest.raises(SelectedHarnessError):
        adapt_selected_harness_v4(doc, allow_local=True)


def test_program_once_and_more_than_eight_lifetime_tracks():
    doc = fixture()
    rows = doc["public"]["timeline"]
    # Repeated independent invocations reuse exactly one source publication.
    original = rows[1].copy()
    rows[:] = [rows[0]]
    owners = {"host": 1}
    for index in range(10):
        inv = f"call{index}"
        track = f"worker:{index}"
        node = {**original, "sequence": len(rows) + 1,
                "track": track, "ownerSequence": 1, "invocationId": inv,
                "triggerId": f"outside{index}"}
        rows.append(node)
        for n, (kind, data) in enumerate((
            ("return", {"value": ["num", index]}),
            ("end", {"kind": "return"}),
        ), start=2):
            rows.append({
                "schema": "semantic-execution/4", "session": "session1",
                "sequence": len(rows) + 1, "ownerSequence": n,
                "track": track, "mode": "generated", "kind": "semantic",
                "invocationId": inv, "traceFormat": "semantic-trace/3",
                "line": line(kind, data),
            })
    ep, receipt = adapt_selected_harness_v4(doc, allow_local=True)
    assert receipt.invocations == 10
    assert receipt.tracks == 11
    assert receipt.actions == 0
    assert sum(record.policy_mask for record in ep.records) == 20


def test_no_unadmitted_trajectory_can_be_labeled_success():
    with pytest.raises(SelectedHarnessError):
        adapt_selected_harness_v4(fixture())
    doc = fixture()
    doc["sidecar"]["admission"] = "admitted"
    doc["sidecar"]["referenceStatus"] = "agreed"
    with pytest.raises(SelectedHarnessError):
        adapt_selected_harness_v4(doc, admission_verifier=lambda _: False)
