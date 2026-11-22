"""Paired source fixtures exercise retained controllers through the public V8 API."""

import copy
from functools import lru_cache

import pytest

from megatoken.data.detailed_v8 import parse_record
from megatoken.data.native_v8 import (
    NativeV8Error,
    _native_checkpoint,
    _replay_shared_state,
    _restore_native_checkpoint,
    validate_v8_artifact,
)
from megatoken.data.prefix_v8 import validate_v8_prediction_prefix
from megatoken.data.protocol_v8 import decode_protocol_record
from megatoken.data.semantic_json import stable_algorithm_hash
from tests.test_prediction_prefix_v8 import examples, prefix

IDS = (
    "join-host-join",
    "join-retained-callback",
    "callback-owned-join",
    "callback-owned-join-awaited",
)


@lru_cache
def artifacts():
    return {row["id"]: row for row in examples("semantic-host-protocol-v8.json.gz.b64")}


def target(example, index):
    context = example["contexts"][index]
    return example["public"]["timeline"][context["visibleThrough"]]


def record(row):
    if row["kind"] == "protocol":
        parsed = decode_protocol_record(row["fields"]["line"])
        return parsed["kind"], parsed
    return parse_record(row["fields"]["line"])


def boundary_indices(example):
    indices = {0, len(example["contexts"]) - 1}
    previous_owner = None
    for index, context in enumerate(example["contexts"]):
        row = target(example, index)
        kind, _ = record(row)
        previous = example["public"]["timeline"][context["visibleThrough"] - 1]
        if (
            row["kind"] == "protocol"
            or kind in {"request", "await", "resume", "job", "end"}
            or previous["mode"] == "observed"
            or context["invocationId"] != previous_owner
        ):
            indices.add(index)
        previous_owner = context["invocationId"]
    return sorted(indices)


@pytest.mark.parametrize(("identity", "outcome"), zip(IDS, (12, 7, 0, 6), strict=True))
def test_source_host_composition_replays_every_context_and_restored_continuation(identity, outcome):
    example = artifacts()[identity]
    receipt = validate_v8_artifact(example)
    assert all(receipt["checks"][key] == "passed" for key in ("replay", "contexts", "continuation"))
    assert receipt["counts"]["contexts"] == len(example["contexts"])
    assert receipt["counts"]["protocolRecords"] > 0
    assert receipt["admitted"] is False  # Compact public fixtures contain no source authority.
    final = _replay_shared_state(
        example["public"]["timeline"],
        example["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="invocation1",
    )
    assert final["state"]["completion"] == {"kind": "return", "value": ["num", outcome]}
    assert all(intent["status"] == "settled" for _, intent in final["obligations"]["intents"])


@pytest.mark.parametrize("identity", IDS)
def test_host_protocol_prediction_and_restore_boundaries_need_no_future_target(identity):
    example = artifacts()[identity]
    for index in boundary_indices(example):
        public, contexts, context = prefix(example, index)
        checked = validate_v8_prediction_prefix(public, contexts, context)
        assert checked["context"] == context
        checkpoint = _native_checkpoint(public["timeline"], checked["context"])
        restored = _restore_native_checkpoint(
            checkpoint,
            expected_prefix=public["timeline"],
            expected_state=context["state"],
            expected_cursor=context["regionCursor"],
            expected_obligations=context["obligations"],
        )
        assert restored == context["state"]


@pytest.mark.parametrize("field", ("join-owner", "job-owner", "ready-order", "result-receipt"))
def test_rehashed_owner_readiness_and_effect_contexts_are_not_native_authority(field):
    example = artifacts()["callback-owned-join-awaited"]
    for candidate, original in enumerate(example["contexts"]):
        owned = [
            node
            for _, node in original["state"]["heap"]
            if (node.get("join") or {}).get("owner") == "event1"
        ]
        jobs = [item for _, item in original["state"]["jobs"] if item["owner"] == "event1"]
        intents = [item for _, item in original["obligations"]["intents"] if item.get("result")]
        if (
            original["invocationId"] == "event1"
            and owned
            and jobs
            and intents
            and original["state"]["ready"]
        ):
            index = candidate
            break
    else:
        raise AssertionError(
            "fixture omits a retained callback boundary with completed host receipt"
        )
    public, contexts, context = prefix(example, index)
    if field == "join-owner":
        next(
            node
            for _, node in context["state"]["heap"]
            if (node.get("join") or {}).get("owner") == "event1"
        )["join"]["owner"] = "invocation1"
    elif field == "job-owner":
        context["state"]["jobs"][0][1]["owner"] = "invocation1"
    elif field == "ready-order":
        context["state"]["scheduler"]["ready"] += 1
    else:
        next(item for _, item in context["obligations"]["intents"] if item.get("result"))["result"][
            "sequence"
        ] += 1
    context["prefixDigest"] = stable_algorithm_hash(public["timeline"])
    with pytest.raises(NativeV8Error, match="context differs from independent replay"):
        validate_v8_prediction_prefix(public, contexts, context)
    checkpoint = _native_checkpoint(public["timeline"], context)
    with pytest.raises(NativeV8Error, match="differs? from verified replay"):
        _restore_native_checkpoint(
            checkpoint,
            expected_prefix=public["timeline"],
            expected_state=original["state"],
            expected_cursor=original["regionCursor"],
            expected_obligations=original["obligations"],
        )


@pytest.mark.parametrize("action", ("promise.join.step", "job"))
def test_callback_jobs_cannot_be_selected_by_a_root_with_its_own_truthful_context(action):
    example = artifacts()["callback-owned-join-awaited"]
    index = next(
        index
        for index, context in enumerate(example["contexts"])
        if context["invocationId"] == "event1"
        and record(target(example, index))[0] == action
        and (action != "job" or record(target(example, index))[1]["action"] == "dispatch")
    )
    public, contexts, context = prefix(example, index)
    # The forged selection gets the true root state. Rejection must therefore
    # come from invocation/scheduler ownership, not a fabricated root context.
    boundary = _replay_shared_state(
        public["timeline"], contexts, {}, roundtrip=False, boundary_invocation="invocation1"
    )
    context.update(invocationId="invocation1", **boundary)
    row = copy.deepcopy(target(example, index))
    row["fields"]["invocationId"] = "invocation1"
    public["timeline"].append(row)
    with pytest.raises(NativeV8Error, match="realm-wide FIFO|another invocation|unknown or stale"):
        validate_v8_artifact({"public": public, "contexts": contexts + [context]})


def test_observed_callback_result_cannot_settle_the_root_controller():
    example = artifacts()["callback-owned-join-awaited"]
    result = next(
        row
        for row in example["public"]["timeline"]
        if row["kind"] == "result" and row["fields"]["invocationId"] == "event1"
    )
    cut = result["sequence"]
    forged = {
        "public": {
            **example["public"],
            "timeline": copy.deepcopy(example["public"]["timeline"][:cut]),
        },
        "contexts": [row for row in example["contexts"] if row["visibleThrough"] < cut],
    }
    forged["public"]["timeline"][-1]["fields"]["invocationId"] = "invocation1"
    with pytest.raises(NativeV8Error, match="effect|invocation|delivery"):
        validate_v8_artifact(forged)
