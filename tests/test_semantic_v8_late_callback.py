"""Original listener input and actual native continuation, with supplied decoding.

The paired producer fixture executes the original JavaScript independently,
registers once through the production harness, checkpoints, journals a later
arrival, and restores that arrival before delivering the callback. These are
engineering controls, not learned execution or source-admission qualification.
"""

from __future__ import annotations

import base64
import copy
import gzip
import json
from pathlib import Path

from tests.test_semantic_v8_orchestration import (
    acknowledge,
    bootstrap,
    chosen,
    commit_child,
    real_runtime,
    recover,
    request,
    select_child,
)


def original_listener():
    path = Path(__file__).parent / "fixtures/late-callback-original-input-v8.json.gz.b64"
    return json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))


def completed_root(adapter, decoder, document):
    public = document["initialPublic"]
    session = public["timeline"][0]["fields"]["session"]
    opened = request(
        adapter,
        "open-orchestration",
        session=session,
        public=public,
        origin="controlled",
        childBudget=64,
    )
    selected = chosen(
        adapter,
        opened,
        "track_create",
        {
            "purpose": "Run the original listener once",
            "completion": "Deliver its exact return",
            "budget": 64,
            "capabilities": ["text", "demo.listen"],
            "execution": copy.deepcopy(opened["children"][0]["execution"]),
        },
    )
    created = acknowledge(adapter, opened, selected)
    bootstrap(adapter, opened, created["track"]["address"])
    acknowledge(
        adapter,
        opened,
        chosen(
            adapter,
            opened,
            "track_join",
            {
                "address": opened["track"],
                "completion_dependency": created["completion_dependency"],
            },
        ),
    )
    decoder["records"] = [record.encode() for record in document["records"][:4]]
    for index in range(4):
        selected = select_child(adapter, document, index)
        commit_child(adapter, document, index, selected)
    result = adapter.complete(
        session=session,
        public=document["rootPublic"],
        contexts=document["contexts"][:4],
        invocationId=document["rootInvocation"],
    )
    assert result["childCompletion"]["completion"] == {"kind": "return", "value": ["num", 1]}
    acknowledge(
        adapter,
        opened,
        chosen(
            adapter,
            opened,
            "track_progress",
            {"address": opened["track"], "progress": json.dumps(result["childCompletion"]["completion"])},
        ),
    )
    packet = adapter.checkpoint()
    assert adapter.status_orchestration(session=session, track=opened["track"])["status"] == "active"
    return opened, packet


def test_actual_journalled_callback_can_continue_after_root_checkpoint():
    document = original_listener()
    assert document["reference"] == {"registrations": 1, "root": 1, "callback": 5}
    assert document["registrations"] == 1
    assert document["rootPublic"]["timeline"] == document["admissionPublic"]["timeline"][:-2]
    assert document["ingressRecords"][-1]["arrival"]["event"]["eventId"] == "after-checkpoint"
    with real_runtime() as (open_adapter, decoder):
        adapter = open_adapter()
        opened, packet = completed_root(adapter, decoder, document)
        decoder["records"] = [record.encode() for record in document["records"][:4]]
        adapter, receipt = recover(open_adapter, adapter, opened["session"], packet)
        assert receipt["hostDispatches"] == 0
        decoder["records"] = [record.encode() for record in document["records"][4:]]
        # Red baseline: this real post-checkpoint callback reaches the existing
        # explicit dynamic-admission refusal before any callback generation.
        selected = select_child(adapter, document, 4)
        assert selected["record"] == document["records"][4]
