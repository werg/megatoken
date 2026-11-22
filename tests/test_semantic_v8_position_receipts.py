"""Independent prepared positions versus actual CE/gather selections, including forgery controls."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatoken.contracts import LOSS_NAMES
from megatoken.data.schema import ByteSpan, PhysicalRecord
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation import ActionSpec, ByteCodec, FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.training.config import TrainConfig
from megatoken.training.positions_v8 import (
    PositionWindowAudit,
    TextReadoutReceipt,
    advance_position_exposure,
    compare_position_inventory,
    empty_position_exposure,
    independent_position_inventory,
    independent_text_target,
    position_exposure_report,
    qualify_position_history,
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
        "trainingConfigDigest": "c" * 64,
        "modelConfigDigest": "d" * 64,
        "codecFingerprint": ByteCodec.byte_fallback().fingerprint,
        "installedModalities": [0],
        "routingBalanceCoefficient": 0.005,
        "effectiveLossWeights": {
            **dict.fromkeys(LOSS_NAMES, 0.0),
            "lexical": 1.0,
            "reconstruction": 0.1,
            "structure": 0.1,
            "terminal": 0.1,
            "modality": 0.25,
            "router_z": 0.001,
        },
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
        router_internal_positions={"p": 4, "shape_bootstrap": 2},
        router_internal_assignments={"p": 4, "shape_bootstrap": 2},
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
    empty = empty_position_exposure(registry_digest="a" * 64, objective=objective())
    assert position_exposure_report(empty)["routerInternalPositionAccounting"] == "unmeasured"
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
        router_internal_positions={"p": 3, "shape_bootstrap": 1},
        router_internal_assignments={"p": 3, "shape_bootstrap": 1},
    )
    assert first == first_copy  # an uncommitted candidate cannot mutate saved history
    assert second["previousDigest"] == first["digest"]
    assert position_exposure_report(second)["scoredOccurrences"] == 2
    assert position_exposure_report(second)["routerInternalPositionCounts"] == {
        "p": 7,
        "shape_bootstrap": 3,
    }
    assert position_exposure_report(second)["routerInternalPositionCount"] == 10
    assert position_exposure_report(second)["routerInternalAssignmentCounts"] == {
        "p": 7,
        "shape_bootstrap": 3,
    }
    assert position_exposure_report(second)["routerInternalAssignmentCount"] == 10
    assert position_exposure_report(second)["routerInternalPositionAuditUpdates"] == 2
    assert (
        position_exposure_report(second)["routerInternalPositionAccounting"]
        == "complete-per-update"
    )
    assert "expertIds" not in json.dumps(second)
    with pytest.raises(ValueError, match="selector set changed"):
        advance_position_exposure(
            restored,
            [exposure()],
            registry_digest="a" * 64,
            objective=objective(),
            optimizer_step=2,
            router_internal_positions={"p": 3},
            router_internal_assignments={"p": 3},
        )
    with pytest.raises(ValueError, match="missing"):
        restore_position_exposure(
            {"train_config": {"semantic_v8_policy": {}}},
            registry_digest="a" * 64,
            objective=objective(),
            global_step=1,
        )


def test_explicit_stage_change_archives_old_objective_without_reusing_its_credit():
    original = state_after_one()
    changed = {**objective(), "trainingConfigDigest": "e" * 64}
    payload = {
        "train_config": {"semantic_v8_policy": {}},
        "semantic_v8_position_exposure": original,
    }
    with pytest.raises(ValueError, match="another objective"):
        restore_position_exposure(
            payload, registry_digest="a" * 64, objective=changed, global_step=1
        )
    fresh = restore_position_exposure(
        payload,
        registry_digest="a" * 64,
        objective=changed,
        global_step=1,
        phase_transition=True,
    )
    assert fresh["startingGlobalStep"] == fresh["lastGlobalStep"] == 1
    assert fresh["optimizerUpdates"] == 0 and fresh["sources"] == {}
    assert fresh["previousStage"] == original
    sample = exposure()
    sample["objectiveDigest"] = stable_algorithm_hash(changed)
    current = advance_position_exposure(
        fresh,
        [sample],
        registry_digest="a" * 64,
        objective=changed,
        optimizer_step=2,
    )
    report = position_exposure_report(current)
    assert report["optimizerUpdates"] == report["scoredOccurrences"] == 1
    assert report["previousStageDigest"] == original["digest"]
    assert current["previousStage"] == original
    assert current["routerInternalPositionCounts"] == {}
    assert current["previousStage"]["routerInternalPositionCounts"] == {
        "p": 4,
        "shape_bootstrap": 2,
    }
    broken = deepcopy(current)
    broken["previousStage"]["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"][
        "physical-1"
    ]["receipt"]["readouts"]["prior"]["positions"] = [0, 1]
    reseal(broken["previousStage"])
    with pytest.raises(ValueError, match="token positions"):
        validate_position_exposure(reseal(broken), registry_digest="a" * 64, objective=changed)
    broken = deepcopy(current)
    broken["previousStage"]["objective"] = None
    reseal(broken["previousStage"])
    with pytest.raises(ValueError, match="objective metadata"):
        validate_position_exposure(reseal(broken), registry_digest="a" * 64, objective=changed)


@pytest.mark.parametrize("field", ["semantic_v8_position_digest", "global_step"])
def test_resume_and_stage_transition_require_metadata_to_bind_position_history(field):
    state = state_after_one()
    payload = {
        "train_config": {"semantic_v8_policy": {}},
        "semantic_v8_position_exposure": state,
    }
    metadata = {"semantic_v8_position_digest": state["digest"], "global_step": 1}
    metadata[field] = "f" * 64 if field == "semantic_v8_position_digest" else 2
    for phase_transition in (False, True):
        with pytest.raises(ValueError, match="checkpoint metadata"):
            restore_position_exposure(
                payload,
                registry_digest="a" * 64,
                objective=objective(),
                global_step=1,
                phase_transition=phase_transition,
                checkpoint_metadata=metadata,
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


def prepared_episode(records, *, source="source-1", tools="[]"):
    return SimpleNamespace(
        stream="semantic_execution",
        lineage=SimpleNamespace(frozen_split="train"),
        tools_json=tools,
        records=records,
        meridians=[
            SimpleNamespace(
                events=[
                    SimpleNamespace(
                        record_id=row.record_id,
                        modality="text",
                        settings_json="{}",
                    )
                    for row in records
                ]
            )
        ],
        supervision=SimpleNamespace(
            annotations_json=json.dumps(
                {
                    "semantic_execution": {
                        "schema": "semantic-execution/8",
                        "sourceId": source,
                        "datasetBinding": {
                            "sourceId": source,
                            "registryDigest": "a" * 64,
                            "split": "train",
                        },
                    }
                }
            )
        ),
    )


def prepared_config():
    return TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-test-tokenizer",
        max_steps=1,
        generated_capacity=8,
        semantic_v8_policy={
            "schema": "semantic-training-policy/8",
            "producerRevision": "a" * 40,
            "consumerRevision": "b" * 40,
            "registryDigest": "a" * 64,
        },
    )


def prepared_inventory(episodes):
    return independent_position_inventory(
        episodes,
        codec=ByteCodec.byte_fallback(),
        config=prepared_config(),
        model_config=ModelConfig.tiny(),
        modalities=[0],
        prepared_manifest_fingerprint="b" * 64,
        expected_source_ids=["source-1"],
    )


def state_and_presence(inventory):
    sample = exposure()
    sample["objectiveDigest"] = stable_algorithm_hash(inventory["objective"])
    state = advance_position_exposure(
        empty_position_exposure(registry_digest="a" * 64, objective=inventory["objective"]),
        [sample],
        registry_digest="a" * 64,
        objective=inventory["objective"],
        optimizer_step=1,
    )
    presence = {
        name: deepcopy(state[name])
        for name in (
            "registryDigest",
            "startingGlobalStep",
            "lastGlobalStep",
            "optimizerUpdates",
            "sources",
        )
    }
    return state, presence


def test_independent_inventory_keeps_logical_and_physical_parts_and_ignores_observations():
    parts = [
        replace(record(), record_id=f"part-{index}", part_index=index, part_count=2)
        for index in range(2)
    ]
    observation = replace(record(), record_id="observed", logical_id="observed", policy_mask=False)
    inventory = prepared_inventory([prepared_episode([observation, *parts])])
    assert inventory["logicalTargetCount"] == 1
    assert inventory["physicalTargetRecordCount"] == 2
    assert inventory["positiveWeightPositionCounts"]["lexical"] == 4
    rows = inventory["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"]
    assert set(rows) == {"part-0", "part-1"}
    assert rows["part-0"]["oracle"]["terminal"] is False
    assert rows["part-1"]["oracle"]["terminal"] is True


@pytest.mark.parametrize(
    "mutation", [None, "omitted", "invented", "joint_shift", "joint_duplicate"]
)
def test_final_comparison_is_independent_of_planner_and_optimizer_agreement(mutation):
    inventory = prepared_inventory([prepared_episode([record()])])
    state, presence = state_and_presence(inventory)
    rows = state["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"]
    if mutation == "omitted":
        rows.clear()
        presence["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"].clear()
    elif mutation == "invented":
        invented = deepcopy(rows["physical-1"])
        invented["receipt"]["physicalRecordId"] = "invented"
        invented["receipt"]["oracle"]["physicalRecordId"] = "invented"
        reseal(invented["receipt"]["oracle"])
        rows["invented"] = invented
    elif mutation == "joint_shift":
        receipt = rows["physical-1"]["receipt"]
        receipt["oracle"]["freeMask"] = [True, True, False, False]
        reseal(receipt["oracle"])
        for readout in receipt["readouts"].values():
            readout["positions"] = [0, 1]
            readout["mask"] = [True, True, False, False]
        # Internal agreement and every denominator remain correct. Only the
        # independently re-read prepared free spans expose the shifted mask.
    elif mutation == "joint_duplicate":
        rows["physical-1"]["scoredOccurrences"] = 2
        presence["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"]["physical-1"][
            "scoredOccurrences"
        ] = 2
    reseal(state)
    comparison = compare_position_inventory(
        inventory, state, target_exposure=presence, global_step=1
    )
    assert comparison["exactOnePassPreparedTextPositions"] is (mutation is None)
    if mutation == "joint_shift":
        assert comparison["positiveWeightPositionCountsMatch"] is True
        assert comparison["preparedOracleMismatches"] == [["source-1", "logical-1", "physical-1"]]
    if mutation == "joint_duplicate":
        assert comparison["independentOccurrenceMismatches"]


def test_unsupported_structured_branch_never_receives_verified_position_credit():
    inventory = prepared_inventory([prepared_episode([record()], tools='[{"name":"tool"}]')])
    assert inventory["verifiedPhysicalTargets"] == 0
    assert inventory["unmeasuredPhysicalTargets"] == 1
    expected = inventory["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"][
        "physical-1"
    ]
    assert expected["oracle"] is None
    assert "structured" in expected["reasons"][0]


def test_observation_only_window_receives_no_position_credit_or_invented_targets():
    observed = replace(record(), policy_mask=False)
    episode = prepared_episode([observed])
    item = SimpleNamespace(episode=episode, window=SimpleNamespace(index=0, start=0, end=1))
    config = SimpleNamespace(semantic_v8_policy={"registryDigest": "a" * 64}, generated_capacity=8)
    audit = PositionWindowAudit(
        SimpleNamespace(windows=[item]),
        codec=ByteCodec.byte_fallback(),
        config=config,
        model_config=ModelConfig.tiny(),
        modalities=[0],
        objective=objective(),
    )
    assert audit.finish()["records"] == []
    with pytest.raises(ValueError, match="invented, observed or repeated"):
        audit.step([SimpleNamespace(record=observed)])


@pytest.mark.parametrize(
    "mutation", [None, "parent", "metadata", "drop", "changed_mask", "hidden_events"]
)
def test_bounded_position_history_preserves_receipts_and_never_claims_full_pass(mutation):
    from dataclasses import asdict

    inventory = prepared_inventory([prepared_episode([record()])])
    state, presence = state_and_presence(inventory)
    sample = exposure()
    sample["objectiveDigest"] = stable_algorithm_hash(inventory["objective"])
    after = advance_position_exposure(
        state,
        [sample],
        registry_digest="a" * 64,
        objective=inventory["objective"],
        optimizer_step=2,
    )
    if mutation == "parent":
        after["previousDigest"] = "f" * 64
    elif mutation == "drop":
        after["sources"] = {}
    elif mutation == "hidden_events":
        after["unmeasuredObjectives"] = {}
    elif mutation == "changed_mask":
        receipt = after["sources"]["source-1"]["logicalTargets"]["logical-1"]["records"][
            "physical-1"
        ]["receipt"]
        receipt["oracle"]["freeMask"] = [True, True, False, False]
        reseal(receipt["oracle"])
        for row in receipt["readouts"].values():
            row["positions"] = [0, 1]
            row["mask"] = [True, True, False, False]
    reseal(after)
    payload = {
        "train_config": asdict(prepared_config()),
        "model_config": ModelConfig.tiny().to_dict(),
        "codec_fingerprint": ByteCodec.byte_fallback().fingerprint,
        "global_step": 1,
        "semantic_v8_position_exposure": state,
        "semantic_v8_target_exposure": presence,
    }
    resumed = {
        **payload,
        "global_step": 2,
        "semantic_v8_position_exposure": after,
        "semantic_v8_target_exposure": {
            **presence,
            "lastGlobalStep": 2,
            "optimizerUpdates": 2,
            "sources": deepcopy(after["sources"]),
        },
    }
    arguments = {
        "inventory": inventory,
        "first_metadata": {"semantic_v8_position_digest": state["digest"]},
        "resumed_metadata": {
            "semantic_v8_position_digest": after["digest"] if mutation != "metadata" else "f" * 64
        },
    }
    if mutation is None:
        report = qualify_position_history(payload, resumed, **arguments)
        assert report["historyContinued"] is True
        assert report["fullPopulationQualified"] is False
        assert report["fullPopulationComparison"]["independentOccurrenceMismatches"]
        assert report["routerInternalPositionCoverage"] == "unmeasured"
    else:
        with pytest.raises(ValueError, match="V8 position audit"):
            qualify_position_history(payload, resumed, **arguments)


def test_full_controlled_corpus_executes_exact_independent_positions(tmp_path, monkeypatch):
    """Actual bounded optimizer pass; fixture source/tokenizer admission is injected."""
    import subprocess
    import time
    from dataclasses import asdict
    from pathlib import Path

    from test_semantic_v8_training_policy import prepared_v8_fixture

    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.data import PreparedCorpus, iter_windows
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.engine import train
    from megatoken.training.windows import window_config

    torch.set_num_threads(1)
    _, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", lambda _path, *, vocab_size: codec)
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(prepared)],
        output=str(tmp_path / "fit"),
        device="cpu",
        precision="fp32",
        max_steps=1,
        checkpoint_every=1,
        keep_checkpoints=1,
        microbatch_episodes=1,
        accumulate_scored=1,
        accumulate_ingested=1,
        max_accumulation_windows=1,
        allow_unverified_data=True,
        semantic_v8_policy=policy,
    )
    with PreparedCorpus([prepared], split="train") as corpus:
        episodes = [corpus.fetch(index)[0] for index in range(len(corpus))]
    windows = [
        window
        for episode in episodes
        for window in iter_windows(episode, codec, window_config(config))
    ]
    updates = sum(window.Q > 0 for window in windows)
    assert 1 <= updates <= 8, "Keep the actual controlled position gate finite"
    config = replace(config, max_steps=updates)
    started = time.perf_counter()
    result = train(config)
    elapsed = time.perf_counter() - started
    payload, metadata = load_checkpoint(result["checkpoint"])
    manifest = json.loads((prepared / "manifest.json").read_text())
    inventory = independent_position_inventory(
        episodes,
        codec=codec,
        config=config,
        model_config=ModelConfig.from_dict(payload["model_config"]),
        modalities=[0],
        prepared_manifest_fingerprint=manifest["fingerprint"],
        expected_source_ids=["source-1"],
    )
    state = payload["semantic_v8_position_exposure"]
    comparison = compare_position_inventory(
        inventory,
        state,
        target_exposure=payload["semantic_v8_target_exposure"],
        global_step=updates,
    )
    assert comparison["exactOnePassPreparedTextPositions"] is True
    assert inventory["physicalTargetRecordCount"] == sum(
        record.policy_mask for episode in episodes for record in episode.records
    )
    assert inventory["physicalTargetRecordCount"] == 4
    assert metadata["semantic_v8_position_digest"] == state["digest"]
    assert state["optimizerUpdates"] == result["global_step"] == updates
    stage_config = replace(config, phase="SE", max_steps=1, output=str(tmp_path / "fit-next-stage"))
    stage_result = train(stage_config, resume=result["checkpoint"], phase_transition=True)
    stage_payload, stage_metadata = load_checkpoint(stage_result["checkpoint"])
    stage_positions = stage_payload["semantic_v8_position_exposure"]
    assert stage_positions["startingGlobalStep"] == updates
    assert stage_positions["lastGlobalStep"] == stage_result["global_step"] == updates + 1
    assert stage_positions["optimizerUpdates"] == 1
    assert stage_positions["previousStage"] == state
    assert stage_positions["objective"] != state["objective"]
    assert stage_positions["sources"] != state["sources"]
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "semantic-v8-controlled-position-evidence/1",
        "sourceCommit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runtime": {
            "torch": str(torch.__version__),
            "device": "cpu",
            "precision": "fp32",
            "seed": config.seed,
        },
        "boundaries": {
            "sourceAdmission": "injected fixture receipt",
            "tokenizerAdmission": "injected byte-fallback fixture",
            "sourcePopulation": "one controlled prepared fixture, not the 64-source support population",
        },
        "elapsedSeconds": elapsed,
        "windowCount": len(windows),
        "optimizerUpdates": updates,
        "totalOptimizerUpdatesExecuted": updates + 1,
        "phaseTransition": {
            "from": config.phase,
            "to": stage_config.phase,
            "checkpointMetadata": stage_metadata,
            "positionExposure": stage_positions,
            "summary": position_exposure_report(stage_positions),
            "previousObjectiveArchived": True,
            "newStageExposureStartsEmpty": True,
        },
        "config": asdict(config),
        "preparedManifestFingerprint": manifest["fingerprint"],
        "checkpointMetadata": metadata,
        "independentPreparedPositionInventory": inventory,
        "optimizerPositionExposure": state,
        "optimizerPositionSummary": position_exposure_report(state),
        "targetExposure": payload["semantic_v8_target_exposure"],
        "comparison": comparison,
        "routerInternalPositionCoverage": "unmeasured",
        "learnedSuccess": None,
    }
    (tmp_path / "position-proof.json").write_text(json.dumps(reseal(report), indent=2) + "\n")
