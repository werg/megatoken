"""Observed host handlers own their exact event routes and retained state."""

import copy
from unittest.mock import patch

import pytest

from megatoken.data import native_v8
from megatoken.data.native_v8 import NativeV8Error, _replay_shared_state, validate_v8_artifact
from megatoken.data.prefix_v8 import validate_v8_committed_prefix, validate_v8_prediction_prefix
from tests.test_prediction_prefix_v8 import examples, prefix


def artifact(case=3):
    return examples("semantic-retained-invocations-v8.json.gz.b64")[case]


def through(document, row):
    document["public"]["timeline"] = document["public"]["timeline"][: row["sequence"]]
    document["contexts"] = [
        context for context in document["contexts"] if context["visibleThrough"] < row["sequence"]
    ]
    return document


@pytest.mark.parametrize("case", (3, 4, 5, 7))
def test_host_handler_registration_delivery_retention_and_close_replay_every_boundary(case):
    document = artifact(case)
    receipt = validate_v8_artifact(document)
    assert not receipt["admitted"]
    assert receipt["checks"]["sourceBinding"] == "untested"
    assert all(receipt["checks"][key] == "passed" for key in ("replay", "contexts", "continuation"))
    for index in range(len(document["contexts"])):
        public, contexts, context = prefix(document, index)
        assert validate_v8_prediction_prefix(public, contexts, context)["context"] == context
    final = _replay_shared_state(
        document["public"]["timeline"],
        document["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="event2",
    )
    assert final["obligations"]["intents"] == []
    assert final["obligations"]["callbacks"] == []
    assert final["obligations"]["closureQueue"] == []
    handler = dict(final["obligations"]["handlers"])["outside"]
    assert handler["ownerInvocationId"] == ("owner" if case == 7 else None)
    assert handler["state"] == "closed"
    assert handler["seen"] == ["first", "second"]
    assert handler["queue"] == []
    if case == 3:
        assert dict(final["state"]["locals"])["count"] == ["num", 11]
    elif case == 4:
        assert final["state"]["locals"] == [["a1b0", ["num", 8]]]
    elif case == 7:
        assert final["state"]["locals"] == [["a2b0", ["num", 8]]]
    else:
        assert final["state"]["locals"] == [["a0b0", ["num", 2]], ["a1b0", ["num", 8]]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("programId", "other"),
        ("realmId", "other"),
        ("track", "other"),
        ("triggerId", "other"),
        ("invocationId", "event2"),
        ("invocationId", "event01"),
    ],
)
def test_event_route_substitution_refuses_before_marking_delivery(field, value):
    document = artifact()
    row = next(row for row in document["public"]["timeline"] if row["kind"] == "event")
    through(document, row)
    row["fields"][field] = value
    before = copy.deepcopy(document)
    observe = native_v8._apply_observation

    def guarded_observe(state, observed, *args, **kwargs):
        if observed["kind"] == "event":
            raise AssertionError("Forged event reached handler mutation")
        return observe(state, observed, *args, **kwargs)

    with patch.object(native_v8, "_apply_observation", side_effect=guarded_observe):
        with pytest.raises(NativeV8Error):
            validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation", "call"),
        ("track", "other"),
        ("triggerId", "other"),
        ("closureId", None),
        ("parentInvocationId", "root"),
        ("taskRevision", 2),
        ("receiver", ["num", 2]),
        ("bindings", {"count": 0}),
        ("deadline", 123),
        ("ingressKey", "other"),
    ],
)
def test_callback_activation_must_preserve_its_preceding_event(field, value):
    document = artifact()
    row = next(
        row
        for row in document["public"]["timeline"]
        if row["kind"] == "invocation" and row["fields"]["invocationId"] == "event1"
    )
    through(document, row)
    row["fields"][field] = value
    before = copy.deepcopy(document)
    initialize = native_v8._initial_state

    def guarded_initialize(program, invocation, *args, **kwargs):
        if invocation["invocationId"] == "event1":
            raise AssertionError("Forged activation reached input import")
        return initialize(program, invocation, *args, **kwargs)

    with patch.object(native_v8, "_initial_state", side_effect=guarded_initialize):
        with pytest.raises(NativeV8Error, match="differs from its observed event delivery"):
            validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


def test_parentless_callback_requires_one_immediately_preceding_event():
    document = artifact()
    row = next(row for row in document["public"]["timeline"] if row["kind"] == "event")
    event_index = row["sequence"] - 1
    invocation = document["public"]["timeline"][event_index + 1]
    through(document, invocation)
    del document["public"]["timeline"][event_index]
    invocation["sequence"] -= 1
    with pytest.raises(NativeV8Error, match="no independently replayed owner"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


def test_pending_event_cannot_be_skipped_to_select_another_invocation():
    document = artifact()
    event = next(row for row in document["public"]["timeline"] if row["kind"] == "event")
    through(document, event)
    with pytest.raises(NativeV8Error, match="precedes the delivered event"):
        _replay_shared_state(
            document["public"]["timeline"],
            document["contexts"],
            {},
            roundtrip=True,
            boundary_invocation="root",
        )


def test_event_cannot_interrupt_an_existing_realm_turn():
    document = artifact()
    handler = next(row for row in document["public"]["timeline"] if row["kind"] == "handler")
    event = next(row for row in document["public"]["timeline"] if row["kind"] == "event")
    # Retain allocation of the closure, while the creating invocation is still
    # running. Host registration is legal; event execution must wait.
    through(document, document["public"]["timeline"][3])
    for row in (handler, event):
        row["sequence"] = len(document["public"]["timeline"]) + 1
        document["public"]["timeline"].append(row)
    with pytest.raises(NativeV8Error, match="interrupt a running JavaScript stack"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


def test_event_identity_cannot_be_delivered_twice():
    document = artifact()
    second = [row for row in document["public"]["timeline"] if row["kind"] == "event"][1]
    through(document, second)
    second["fields"].update(eventId="first", triggerId="event:outside:first")
    with pytest.raises(NativeV8Error, match="Duplicate source event"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


def test_closed_host_handler_cannot_receive_more_events():
    document = artifact()
    event = copy.deepcopy(
        next(row for row in document["public"]["timeline"] if row["kind"] == "event")
    )
    event["sequence"] = len(document["public"]["timeline"]) + 1
    event["fields"].update(eventId="third", invocationId="event3", triggerId="event:outside:third")
    document["public"]["timeline"].append(event)
    with pytest.raises(NativeV8Error, match="no active source handler"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


@pytest.mark.parametrize(
    "field,value",
    [("realmPolicy", "fresh"), ("ownerInvocationId", "missing"), ("queueLimit", 0)],
)
def test_host_registration_requires_usable_realm_owner_and_bounded_queue(field, value):
    document = artifact()
    row = next(row for row in document["public"]["timeline"] if row["kind"] == "handler")
    through(document, row)
    row["fields"][field] = value
    before = copy.deepcopy(document)
    with pytest.raises(ValueError):
        validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


def test_handler_close_publishes_unique_discarded_arrival_identities():
    document = artifact()
    row = document["public"]["timeline"][-1]
    assert row["kind"] == "handler-close"
    row["fields"]["discarded"] = ["queued-first", "queued-second"]
    final = _replay_shared_state(
        document["public"]["timeline"],
        document["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="event2",
    )
    assert dict(final["obligations"]["handlers"])["outside"]["seen"] == [
        "first",
        "second",
        "queued-first",
        "queued-second",
    ]
    for discarded in (["first"], ["queued-first", "queued-first"]):
        row["fields"]["discarded"] = discarded
        with pytest.raises(NativeV8Error):
            validate_v8_artifact(document)


def test_actual_queued_arrival_retirement_replays_before_the_next_call():
    document = artifact(6)
    assert not any(row["kind"] == "event" for row in document["public"]["timeline"])
    close = next(row for row in document["public"]["timeline"] if row["kind"] == "handler-close")
    assert close["fields"]["discarded"] == ["queued-first", "queued-second"]
    receipt = validate_v8_artifact(document)
    assert not receipt["admitted"]
    for index in range(len(document["contexts"])):
        public, contexts, context = prefix(document, index)
        assert validate_v8_prediction_prefix(public, contexts, context)["context"] == context
    after = next(context for context in document["contexts"] if context["invocationId"] == "after")
    handler = dict(after["obligations"]["handlers"])["outside"]
    assert handler["state"] == "closed"
    assert handler["seen"] == ["queued-first", "queued-second"]
    assert handler["queue"] == []
