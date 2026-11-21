"""Independent reconstruction of external recovery authority and dispatch attempts."""

import copy

import pytest

from megatoken.data.detailed_v8 import encode_record
from megatoken.data.native_v8 import (
    NativeV8Error,
    _apply_observation,
    _replay_shared_state,
    validate_v8_artifact,
)
from megatoken.data.prefix_v8 import validate_v8_committed_prefix, validate_v8_prediction_prefix
from tests.test_prediction_prefix_v8 import examples, prefix


def artifact(case=0):
    return examples("semantic-recovery-v8.json.gz.b64")[case]


def through(document, sequence):
    document["public"]["timeline"] = document["public"]["timeline"][:sequence]
    document["contexts"] = [
        context for context in document["contexts"] if context["visibleThrough"] < sequence
    ]
    return document


def recovery_row(document):
    return next(
        row
        for row in document["public"]["timeline"]
        if row["kind"] in {"reconcile", "callback-reattach"}
    )


def holders(case=0):
    document = artifact(case)
    row = copy.deepcopy(recovery_row(document))
    through(document, row["sequence"] - 1)
    boundary = _replay_shared_state(
        document["public"]["timeline"],
        document["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="root",
    )
    obligations = boundary["obligations"]
    dispatched = {key for key, intent in obligations["intents"] if intent["status"] == "dispatched"}
    settled = {key for key, intent in obligations["intents"] if intent["status"] == "settled"}
    return (boundary["state"], dispatched, settled, obligations), row


def apply(state, row):
    _apply_observation(state[0], row, state[1], state[2], state[3])


def refused_without_mutation(state, row, message):
    before = copy.deepcopy(state)
    with pytest.raises(NativeV8Error, match=message):
        apply(state, row)
    assert state == before


def dispatch_row(attempt):
    return {
        "kind": "dispatch",
        "fields": {
            "invocationId": "root",
            "effectId": "effect",
            "key": "root/effect",
            "attempt": attempt,
        },
    }


@pytest.mark.parametrize("case", range(6))
def test_recovery_replays_every_public_prediction_and_json_boundary(case):
    document = artifact(case)
    receipt = validate_v8_artifact(document)
    assert not receipt["admitted"]
    assert all(
        receipt["checks"][key] == "untested" for key in ("sourceBinding", "masks", "tokenizer")
    )
    assert all(receipt["checks"][key] == "passed" for key in ("replay", "contexts", "continuation"))
    for index in range(len(document["contexts"])):
        public, contexts, context = prefix(document, index)
        assert validate_v8_prediction_prefix(public, contexts, context)["context"] == context
    final = _replay_shared_state(
        document["public"]["timeline"],
        document["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="event2" if case == 4 else "root",
    )
    intent = final["obligations"]["intents"][0][1]
    assert intent["status"] == ("lost" if case == 2 else "settled")
    assert intent["attempts"] == (2 if case == 3 else 1)
    if case == 2:
        assert document["recovery"]["outcome"] == "unresolved-lost-effect"
        assert final["state"]["completion"] is None
        assert dict(final["state"]["effects"])["effect"]["status"] == "requested"
    elif case == 4:
        assert document["recovery"]["registrations"] == 1
        assert document["recovery"]["reattachments"] == 1
        handler = dict(final["obligations"]["handlers"])["callback1"]
        assert handler["state"] == "closed"
        assert handler["seen"] == ["first", "second"]
        assert final["obligations"]["callbacks"][0][1]["state"] == "revoked"
        assert sum(row["kind"] == "handler" for row in document["public"]["timeline"]) == 1
    else:
        assert document["recovery"]["adapterCalls"] == 1
        assert final["state"]["completion"]["kind"] == ("throw" if case == 1 else "return")


def test_durable_owned_return_recovery_preserves_receiver_aliases_cycles_and_captured_cells():
    document = artifact(5)
    assert document["recovery"]["durableArrivals"] == 1
    assert document["recovery"]["independentReaderRestores"] == 3
    assert document["recovery"]["adapterCalls"] == 1
    child_contexts = [
        context for context in document["contexts"] if context["invocationId"] == "child"
    ]
    assert len(child_contexts) == 2  # Separate saved return and end boundaries.
    for context in child_contexts:
        value = context["state"]["receiver"]
        assert context["input"]["graph"]["roots"] == [value, value]
        assert dict(context["state"]["heap"])[value[1]]["properties"] == [
            ["value", ["num", 4]],
            ["self", value],
        ]
        assert dict(context["state"]["locals"])["saved"] == ["ref", "retained"]
        assert context["state"]["captureAliases"] == [
            ["saved", "saved"],
            ["value", "a1b0"],
            ["alias", "a1b1"],
        ]
        assert dict(context["state"]["locals"])["a1b0"] == value
        assert dict(context["state"]["locals"])["a1b1"] == value
        assert dict(context["obligations"]["intents"])["root/effect"]["status"] == "dispatched"
    final = _replay_shared_state(
        document["public"]["timeline"],
        document["contexts"],
        {},
        roundtrip=True,
        boundary_invocation="root",
    )
    assert final["state"]["completion"]["value"] == document["recovery"]["callbackResult"]
    assert (
        dict(final["state"]["heap"])["external"]["result"] == document["recovery"]["callbackResult"]
    )


def test_completing_the_recovered_child_does_not_supply_its_parents_missing_result():
    state, _ = holders(5)
    assert state[3]["intents"][0][1]["status"] == "dispatched"
    assert state[3]["intents"][0][1]["result"] is None
    assert state[0]["completion"] is None
    assert dict(state[0]["heap"])["external"]["state"] == "pending"
    assert state[1] == {"root/effect"}
    assert state[2] == set()


def test_reconciliation_cannot_import_an_unknown_returned_identity_after_child_recovery():
    state, row = holders(5)
    apply(state, row)
    result = copy.deepcopy(
        next(row for row in artifact(5)["public"]["timeline"] if row["kind"] == "result")
    )
    result["fields"]["result"] = ["ref", "missing"]
    result["fields"]["line"] = encode_record(
        "observe",
        {"effect": "effect", "status": "fulfilled", "result": ["ref", "missing"]},
        role="observed",
    )
    refused_without_mutation(state, result, "unknown identity")


@pytest.mark.parametrize(
    "field,value",
    [
        ("invocationId", "foreign"),
        ("effectId", "foreign"),
        ("track", "foreign"),
        ("evidenceId", ""),
        ("evidenceId", None),
        ("disposition", "retry"),
    ],
)
def test_reconciliation_requires_exact_owner_effect_track_and_external_evidence(field, value):
    document = artifact()
    row = recovery_row(document)
    through(document, row["sequence"])
    row["fields"][field] = value
    before = copy.deepcopy(document)
    with pytest.raises(ValueError):
        validate_v8_committed_prefix(document["public"], document["contexts"])
    assert document == before


@pytest.mark.parametrize("status", ["committed", "settled", "lost", "cancelled"])
def test_reconciliation_cannot_overwrite_an_ineligible_effect(status):
    state, row = holders()
    state[3]["intents"][0][1]["status"] = status
    refused_without_mutation(state, row, "outstanding dispatched effect")


@pytest.mark.parametrize("evidence", ["", None, False, 0])
def test_reconciliation_refusal_preserves_the_entire_outstanding_effect(evidence):
    state, row = holders()
    row["fields"]["evidenceId"] = evidence
    refused_without_mutation(state, row, "independent evidence")


@pytest.mark.parametrize("field", ["invocationId", "effectId", "track", "status"])
def test_reconciled_result_must_match_the_immediately_preceding_receipt(field):
    document = artifact()
    row = recovery_row(document)
    result = document["public"]["timeline"][row["sequence"]]
    assert result["kind"] == "result"
    through(document, result["sequence"])
    result["fields"][field] = "rejected" if field == "status" else "foreign"
    if field in {"effectId", "status"}:
        fields = result["fields"]
        fields["line"] = encode_record(
            "observe",
            {"effect": fields["effectId"], "status": fields["status"], "result": fields["result"]},
            role="observed",
        )
    with pytest.raises(NativeV8Error, match="exact observed result"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


def test_reconciliation_cannot_be_skipped_by_another_observation_or_prediction():
    document = artifact()
    row = recovery_row(document)
    through(document, row["sequence"])
    with pytest.raises(NativeV8Error, match="precedes the reconciled effect"):
        _replay_shared_state(
            document["public"]["timeline"],
            document["contexts"],
            {},
            roundtrip=True,
            boundary_invocation="root",
        )
    repeated = copy.deepcopy(row)
    repeated["sequence"] += 1
    document["public"]["timeline"].append(repeated)
    with pytest.raises(NativeV8Error, match="exact observed result"):
        validate_v8_committed_prefix(document["public"], document["contexts"])


@pytest.mark.parametrize("attempt", [0, 1, 3, True, 2.0, 9007199254740992])
def test_retry_must_increment_the_preserved_dispatch_attempt_exactly(attempt):
    state, row = holders(3)
    apply(state, row)
    assert state[1] == set()
    assert state[3]["intents"][0][1]["attempts"] == 1
    refused_without_mutation(state, dispatch_row(attempt), "dispatch identity|requested intent")
    apply(state, dispatch_row(2))
    assert state[1] == {"root/effect"}
    assert state[3]["intents"][0][1]["attempts"] == 2


def test_an_opaque_evidence_identifier_may_describe_more_than_one_verified_attempt():
    state, row = holders(3)
    for attempt in (2, 3):
        # Authenticating independent evidence belongs to the embedding. A
        # reused batch receipt is not made invalid by inventing uniqueness.
        apply(state, row)
        apply(state, dispatch_row(attempt))
    assert state[3]["intents"][0][1]["attempts"] == 3


@pytest.mark.parametrize("case", [0, 1, 2])
def test_only_not_dispatched_evidence_makes_retry_eligible(case):
    state, row = holders(case)
    refused_without_mutation(state, dispatch_row(2), "Duplicate effect dispatch")
    apply(state, row)
    refused_without_mutation(state, dispatch_row(2), "Duplicate effect dispatch")


def test_lost_effect_cannot_later_accept_a_result_or_another_reconciliation():
    state, row = holders(2)
    apply(state, row)
    fulfilled = artifact()
    result = next(row for row in fulfilled["public"]["timeline"] if row["kind"] == "result")
    refused_without_mutation(state, result, "dispatched obligation")
    row["fields"]["disposition"] = "not-dispatched"
    refused_without_mutation(state, row, "outstanding dispatched effect")


@pytest.mark.parametrize(
    "field,value",
    [
        ("subscriptionId", "foreign"),
        ("realmId", "foreign"),
        ("track", "foreign"),
        ("evidenceId", ""),
        ("evidenceId", None),
    ],
)
def test_reattachment_requires_the_exact_retained_handler_and_evidence(field, value):
    state, row = holders(4)
    row["fields"][field] = value
    refused_without_mutation(state, row, "exact live handler")


@pytest.mark.parametrize("status", ["closing", "closed", "pending"])
def test_reattachment_cannot_revive_a_handler_being_retired(status):
    state, row = holders(4)
    state[3]["handlers"][0][1]["state"] = status
    refused_without_mutation(state, row, "exact live handler")


@pytest.mark.parametrize("status", ["active", "lost"])
def test_reattachment_respects_public_active_state_but_never_revives_a_revoked_capability(status):
    state, row = holders(4)
    state[3]["handlers"][0][1]["state"] = status
    state[3]["callbacks"][0][1]["state"] = "revoked"
    refused_without_mutation(state, row, "exact live handler")
    state[3]["callbacks"][0][1]["state"] = status
    before = copy.deepcopy(state)
    apply(state, row)
    assert state[0:3] == before[0:3]
    assert state[3]["intents"] == before[3]["intents"]
    assert state[3]["handlers"][0][1]["state"] == "active"
    assert state[3]["callbacks"][0][1]["state"] == "active"


def test_host_handler_reattachment_does_not_invent_guest_registration():
    state, row = holders(4)
    state[3]["callbacks"] = []
    state[3]["handlers"][0][1]["ownerInvocationId"] = None
    apply(state, row)
    assert state[3]["callbacks"] == []


def test_effect_receipt_before_invocation_is_rejected_instead_of_silently_ignored():
    document = artifact()
    row = recovery_row(document)
    document["public"]["timeline"] = [document["public"]["timeline"][0], row]
    document["contexts"] = []
    row["sequence"] = 2
    with pytest.raises(NativeV8Error, match="no independently replayed invocation"):
        validate_v8_committed_prefix(document["public"], document["contexts"])
