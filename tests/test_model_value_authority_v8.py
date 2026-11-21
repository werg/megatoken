"""Supplied-decoder authority controls; these do not measure model quality."""

import copy

import pytest

from megatoken.data.semantic_json import stable_algorithm_json
from megatoken.semantic_checkpoint_v8 import capture_random_state, continuation_fingerprint
from megatoken.semantic_model_v8 import NativeSelectionV8Error
from tests.test_native_action_v8 import receipt_document_with_flag
from tests.test_prediction_prefix_v8 import examples
from tests.test_semantic_v8_serving_restore import adapter_for, commit, rehash, select, target


@pytest.fixture(scope="module")
def committed_checkpoint():
    example = examples("semantic-regions-v8.json.gz.b64")[0]
    adapter, _ = adapter_for([target(example, 0)])
    commit(adapter, example, 0, select(adapter, example, 0))
    return example, adapter.checkpoint()


def replaced(packet, path, value):
    changed = copy.deepcopy(packet)
    owner = changed
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value
    return rehash(changed)


def refused_checkpoint(example, packet):
    restored, runtime = adapter_for([target(example, 0)])
    before_random = capture_random_state()
    before = stable_algorithm_json(packet)
    with pytest.raises(NativeSelectionV8Error):
        restored.restore(checkpoint=packet)
    assert not restored.sessions and not restored.attempts and not runtime.memory.sessions
    assert capture_random_state() == before_random
    assert stable_algorithm_json(packet) == before


@pytest.mark.parametrize(
    "path,value",
    [
        (("attempts", 0, "modelSelected"), 1),
        (("attempts", 0, "fragments", 0, "terminal"), 1),
        (("initialization", "nativeAdmissionClaim"), 0),
        (("initialization", "generationSettings", "effort"), True),
        (("settings", "effort"), True),
    ],
)
def test_rehashed_model_checkpoint_cannot_exchange_booleans_and_numbers(
    committed_checkpoint, path, value
):
    example, packet = committed_checkpoint
    refused_checkpoint(example, replaced(packet, path, value))


def test_rehashed_model_checkpoint_requires_an_integer_context_index(committed_checkpoint):
    example, packet = committed_checkpoint
    session = next(iter(packet["sessions"]))
    refused_checkpoint(
        example, replaced(packet, ("sessions", session, "replay", 0, "contextIndex"), False)
    )


@pytest.mark.parametrize(
    "path",
    [
        ("settings", "effort"),
        ("initialization", "generationSettings", "effort"),
        ("attempts", 0, "generatedTokens"),
        ("attempts", 0, "fragments", 0, "meridian"),
        ("openings", 0, "metadataTokens"),
        ("openings", 0, "metadataBytes"),
    ],
)
def test_model_checkpoint_preserves_ecmascript_number_equivalence(committed_checkpoint, path):
    example, packet = committed_checkpoint
    value = packet
    for key in path:
        value = value[key]
    assert type(value) is int
    changed = replaced(packet, path, float(value))
    restored, _ = adapter_for([target(example, 0)])
    receipt = restored.restore(checkpoint=changed)
    assert receipt["restored"] and receipt["reexecutedSelections"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("metadataTokens", True),
        ("metadataTokens", -1),
        ("metadataTokens", 1.5),
        ("metadataTokens", 2**53),
        ("metadataBytes", False),
        ("metadataBytes", "1"),
        ("metadataBytes", None),
        ("openSeconds", -1),
        ("openSeconds", False),
    ],
)
def test_opening_costs_refuse_invalid_json_numbers_before_native_state(
    committed_checkpoint, field, value
):
    example, packet = committed_checkpoint
    refused_checkpoint(example, replaced(packet, ("openings", 0, field), value))


def model_after_acknowledged_dispatch(value):
    example = receipt_document_with_flag(value)
    adapter, runtime = adapter_for([target(example, 0), target(example, 1)])
    output = select(adapter, example, 0)
    # The acknowledgement includes the real host receipts following the request.
    # They remain bound even before the next selection ingests their model bytes.
    public = copy.deepcopy(example["public"])
    public["timeline"] = public["timeline"][: example["contexts"][1]["visibleThrough"]]
    adapter.commit(
        session=output["selection"]["session"],
        public=public,
        contexts=example["contexts"][:1],
        invocationId=example["contexts"][0]["invocationId"],
        record=output["record"],
        selection=output["selection"],
    )
    return adapter, runtime, output["selection"]["session"]


@pytest.mark.parametrize("observed,changed", [(True, 1), (False, 0), (1, True), (0, False)])
def test_selection_cannot_rewrite_acknowledged_dispatch_metadata(observed, changed):
    adapter, runtime, session = model_after_acknowledged_dispatch(observed)
    work = adapter.sessions[session]
    before = stable_algorithm_json(work.public)
    before_runtime = continuation_fingerprint(work.runtime)
    before_random = capture_random_state()
    with pytest.raises(NativeSelectionV8Error, match="rewrites acknowledged native history"):
        select(adapter, receipt_document_with_flag(changed), 1)
    assert stable_algorithm_json(work.public) == before
    assert continuation_fingerprint(work.runtime) == before_runtime
    assert capture_random_state() == before_random
    assert work.pending is None and len(adapter.attempts) == 1
    assert len(runtime.fragments) == 1


@pytest.mark.parametrize("observed,changed", [(1, 1.0), (1.0, 1)])
def test_selection_keeps_equivalent_numeric_receipt_representations(observed, changed):
    adapter, _, session = model_after_acknowledged_dispatch(observed)
    example = receipt_document_with_flag(changed)
    selected = select(adapter, example, 1)
    assert selected["record"].encode() == target(example, 1)
    assert commit(adapter, example, 1, selected)["committed"]
    assert adapter.sessions[session].pending is None
