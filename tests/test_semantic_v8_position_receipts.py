"""Independent prepared positions versus actual CE/gather selections, including forgery controls."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from megatoken.data.schema import ByteSpan, PhysicalRecord
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation import ActionSpec, ByteCodec, FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.training.positions_v8 import (
    TextReadoutReceipt,
    advance_position_exposure,
    empty_position_exposure,
    independent_text_target,
    position_exposure_report,
    restore_position_exposure,
    validate_position_exposure,
)


def record(**changes):
    return PhysicalRecord(
        record_id="physical-1",
        logical_id="logical-1",
        role="assistant",
        payload=b"aaaa",
        policy_mask=True,
        free_spans=(ByteSpan(1, 3),),
        **changes,
    )


def oracle_for(value=None, codec=None):
    return independent_text_target(
        record() if value is None else value,
        codec=codec or ByteCodec.byte_fallback(),
        generated_capacity=8,
        max_capacity=16,
    )


def begin(oracle, branch="prior"):
    receipt = TextReadoutReceipt(oracle, branch)
    receipt.begin((b"aaaa",), (tuple(oracle["tokenIds"]),), (oracle["capacity"],))
    receipt.categorical("length", [0], [3], list(range(1, 9)))
    receipt.categorical("mode", [0], [0], [0])
    receipt.categorical("terminal", [0], [1], [0, 1])
    return receipt


def completed(oracle, branch="prior"):
    receipt = begin(oracle, branch)
    indices = [i for i, bit in enumerate(oracle["freeMask"]) if bit]
    if indices:
        receipt.lexical(
            "utf8",
            [(0, i) for i in indices],
            [oracle["tokenIds"][i] for i in indices],
            torch.ones(256, dtype=torch.bool),
        )
    return receipt.finish()


def target_receipt():
    oracle = oracle_for()
    return {
        "logicalTargetId": "logical-1",
        "physicalRecordId": "physical-1",
        "status": "verified",
        "reasons": [],
        "oracle": oracle,
        "readouts": {branch: completed(oracle, branch) for branch in ("prior", "reconstruction")},
        "modality": {"label": 0, "support": [[0, 1]]},
    }


def objective():
    return {
        "effectiveLossWeights": {
            "lexical": 1.0,
            "reconstruction": 0.1,
            "structure": 0.1,
            "terminal": 0.1,
            "modality": 0.25,
            "router_z": 0.001,
        }
    }


def exposure(receipt=None):
    return {
        "schema": "semantic-v8-optimizer-position-exposure/1",
        "registryDigest": "a" * 64,
        "objectiveDigest": stable_algorithm_hash(objective()),
        "sourceId": "source-1",
        "windowIndex": 3,
        "records": [target_receipt() if receipt is None else receipt],
        "unmeasuredObjectives": {"router_z": 1},
    }


def state_after_one():
    state = empty_position_exposure(registry_digest="a" * 64, objective=objective())
    return advance_position_exposure(
        state,
        [exposure()],
        registry_digest="a" * 64,
        objective=objective(),
        optimizer_step=1,
    )


def reseal(value):
    value["digest"] = stable_algorithm_hash(
        {key: child for key, child in value.items() if key != "digest"}
    )
    return value


def test_independent_oracle_intersects_exact_byte_span_and_emitted_mask():
    codec = ByteCodec(
        tuple(bytes([b]) for b in range(256)) + (b"ab",), merge_ranks={(b"a", b"b"): 0}
    )
    assert codec.encode(b"abab") == [256, 256]
    original = PhysicalRecord(
        "physical",
        "assistant",
        b"abab",
        logical_id="logical",
        policy_mask=True,
        emitted_ids=(97, 98, 256),
        emitted_free_mask=(True, False, True),
        codec_fingerprint=codec.fingerprint,
        free_spans=(ByteSpan(1, 4),),
    )
    oracle = oracle_for(original, codec)
    assert oracle["tokenIds"] == [97, 98, 256]  # preserve actual emitted partition
    assert oracle["tokenByteEdges"] == [0, 1, 2, 4]
    assert oracle["freeMask"] == [False, False, True]
    assert (
        oracle_for(replace(original, free_spans=(ByteSpan(3, 4),)), codec)["freeMask"]
        == [False] * 3
    )
    cached = replace(original, emitted_ids=None, emitted_free_mask=None, cached_ids=(97, 98, 256))
    assert oracle_for(cached, codec)["tokenIds"] == [97, 98, 256]


def test_complete_positions_distinguish_repeated_token_labels_and_zero_mask():
    receipt = completed(oracle_for())
    assert receipt["positions"] == [1, 2]
    assert receipt["labels"] == [97, 97]
    zero = completed(oracle_for(replace(record(), free_spans=())))
    assert zero["positions"] == [] and zero["alphabetSha256"] is None
    assert set(zero["decisions"]) == {"length", "mode", "terminal"}


@pytest.mark.parametrize(
    "coordinates,labels",
    [
        ([(0, 0), (0, 1)], [97, 97]),  # shifted mask, identical count and labels
        ([(0, 1), (0, 1)], [97, 97]),  # duplicated position replaces distinct target
        ([(0, 1), (0, 4)], [97, 97]),  # invented coordinate
        ([(0, 1), (0, 2)], [98, 97]),  # wrong label at a correct coordinate
        ([(False, 1), (0, 2)], [97, 97]),  # bool is not canonical integer metadata
    ],
)
def test_actual_selection_rejects_shift_duplicate_invention_or_wrong_label(coordinates, labels):
    receipt = begin(oracle_for())
    with pytest.raises(ValueError, match="V8 position audit"):
        receipt.lexical("utf8", coordinates, labels, torch.ones(256, dtype=torch.bool))


def test_actual_selection_rejects_omitted_free_position_and_changed_alphabet():
    receipt = begin(oracle_for())
    receipt.lexical("utf8", [(0, 1)], [97], torch.ones(256, dtype=torch.bool))
    with pytest.raises(ValueError, match="omitted"):
        receipt.finish()
    receipt = begin(oracle_for())
    alphabet = torch.ones(256, dtype=torch.bool)
    alphabet[0] = False
    with pytest.raises(ValueError, match="alphabet mask changed"):
        receipt.lexical("utf8", [(0, 1), (0, 2)], [97, 97], alphabet)


@pytest.mark.parametrize(
    "support,label",
    [
        (list(range(2, 10)), 2),  # correct resolved label, shifted public support
        ([1, 2, 3, 4, 5, 7, 8], 3),  # omitted class
        ([1, 2, 3, 4, 4, 5, 6, 7, 8], 3),  # duplicate mask entry
        (list(range(1, 9)), 4),  # label shift within valid support
    ],
)
def test_actual_categorical_selection_rejects_changed_support_or_label(support, label):
    receipt = TextReadoutReceipt(oracle_for(), "prior")
    receipt.begin((b"aaaa",), ((97, 97, 97, 97),), (4,))
    with pytest.raises(ValueError, match="V8 position audit"):
        receipt.categorical("length", [0], [label], support)


def test_readout_refuses_missing_pass_decision_or_noncanonical_row():
    receipt = begin(oracle_for())
    with pytest.raises(ValueError, match="duplicated"):
        receipt.categorical("mode", [0], [0], [0])
    receipt = TextReadoutReceipt(oracle_for(), "reconstruction")
    receipt.begin((b"aaaa",), ((97, 97, 97, 97),), (4,))
    with pytest.raises(ValueError, match="row membership"):
        receipt.categorical("mode", [False], [0], [0])
    with pytest.raises(ValueError, match="omitted"):
        receipt.finish()


@pytest.mark.parametrize(
    "value",
    [
        replace(record(), policy_mask=False),
        replace(record(), payload=b"", free_spans=()),
        replace(record(), logical_id=""),
        replace(record(), part_index=False),
    ],
)
def test_independent_oracle_rejects_observation_empty_or_noncanonical_input(value):
    with pytest.raises(ValueError, match="V8 position audit"):
        oracle_for(value)


@pytest.mark.parametrize(
    "mutation", ["shift", "duplicate", "mask", "support_bool", "objective_bool", "missing_pass"]
)
def test_resealed_checkpoint_cannot_forge_position_receipts(mutation):
    state = state_after_one()
    target = state["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"]["physical-1"][
        "receipt"
    ]
    prior = target["readouts"]["prior"]
    if mutation == "shift":
        prior["positions"] = [0, 1]
    elif mutation == "duplicate":
        prior["positions"] = [1, 1]
    elif mutation == "mask":
        prior["mask"] = [True, True, False, False]
    elif mutation == "support_bool":
        prior["decisions"]["mode"]["support"] = [[False, True]]
    elif mutation == "objective_bool":
        state["objective"]["effectiveLossWeights"]["lexical"] = True
    else:
        del target["readouts"]["reconstruction"]
    with pytest.raises(ValueError, match="V8 position audit"):
        validate_position_exposure(reseal(state), registry_digest="a" * 64, objective=objective())


def test_position_state_resumes_exactly_and_counts_repeated_optimizer_occurrences():
    first = state_after_one()
    first_copy = deepcopy(first)
    report = position_exposure_report(first)
    assert report["positiveWeightPositionCounts"] == {
        "lexical": 2,
        "reconstruction": 2,
        "structure": 4,
        "terminal": 2,
        "modality": 1,
    }
    assert report["unmeasuredObjectiveCounts"] == {"router_z": 1}
    assert report["routerInternalPositionCoverage"] == "unmeasured"
    restored = restore_position_exposure(
        {"train_config": {"semantic_v8_policy": {}}, "semantic_v8_position_exposure": first},
        registry_digest="a" * 64,
        objective=objective(),
        global_step=1,
    )
    second = advance_position_exposure(
        restored,
        [exposure()],
        registry_digest="a" * 64,
        objective=objective(),
        optimizer_step=2,
    )
    assert first == first_copy  # an uncommitted candidate cannot mutate saved history
    assert second["previousDigest"] == first["digest"]
    assert position_exposure_report(second)["scoredOccurrences"] == 2
    with pytest.raises(ValueError, match="missing"):
        restore_position_exposure(
            {"train_config": {"semantic_v8_policy": {}}},
            registry_digest="a" * 64,
            objective=objective(),
            global_step=1,
        )


def test_actual_decoder_receipts_leave_losses_and_gradients_unchanged():
    torch.set_num_threads(1)
    torch.manual_seed(913)
    model = FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback()).eval()
    plan = model.prepare_target(
        target_bytes=b"aaaa",
        free_mask=(False, True, True, False),
        mode_specs=[ActionSpec(text_kind="utf8", min_tokens=1, max_tokens=8)],
    )
    raw = torch.randn(1, model.config.n_facets, model.config.facet_dim, requires_grad=True)
    plain = model.decoder.teacher_raw(model.core, raw, plan)
    plain_sum = sum(value for value, _ in plain.values())
    plain_gradient = torch.autograd.grad(plain_sum, raw)[0]
    sink = TextReadoutReceipt(oracle_for(), "prior")
    audited = model.decoder.teacher_raw(model.core, raw, plan, position_sink=sink)
    audited_sum = sum(value for value, _ in audited.values())
    audited_gradient = torch.autograd.grad(audited_sum, raw)[0]
    assert sink.finish()["positions"] == [1, 2]
    for name in plain:
        assert plain[name][1] == audited[name][1]
        torch.testing.assert_close(plain[name][0], audited[name][0], rtol=0, atol=0)
    torch.testing.assert_close(plain_gradient, audited_gradient, rtol=0, atol=0)
    shifted = replace(plan, free_masks=((True, True, False, False),))
    with pytest.raises(ValueError, match="position, mask or label"):
        model.decoder.teacher_raw(
            model.core, raw, shifted, position_sink=TextReadoutReceipt(oracle_for(), "prior")
        )
