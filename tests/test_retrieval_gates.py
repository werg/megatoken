"""Held-out bootstrap gates run the training forward and gate on measured values."""

from __future__ import annotations

import json
import shutil

from retrieval_fixture import (
    live_kb_train_config,
    make_episode,
    prepare_dataset,
    retrieval_train_config,
    write_retrieval_artifacts,
)

from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe
from megatoken.training.engine import train
from megatoken.training.gates import BOOTSTRAP_GATES, GateThresholds, bootstrap_gates

LENIENT = GateThresholds(
    max_lexical_ce=1e9,
    max_terminal_ce=1e9,
    min_document_centered_cosine=-1.0,
    min_query_centered_cosine=-1.0,
    min_recall_ratio=0.0,
    recall_k=2,
)


def _setup(tmp_path):
    checkpoint = train(retrieval_train_config(tmp_path / "p1", max_steps=1, checkpoint_every=1))[
        "checkpoint"
    ]
    bank = tmp_path / "catalog.sqlite"
    shutil.copyfile(live_kb_train_config(tmp_path / "live").retrieval_bank, bank)
    items = [make_episode(i) for i in range(3)]
    heldout = prepare_dataset(tmp_path / "heldout", items, heldout=True)
    sidecars = write_retrieval_artifacts(tmp_path / "heldout" / "retrieval", items)
    recipe = sidecars["retrieval_memory_recipe"]
    with open_bank_for_recipe(RetrievalRecipe.load(recipe), bank) as opened:
        opened.put_growth_state("readiness", {"coherence": False, "embeddings": False})
    return dict(
        checkpoint=checkpoint,
        dataset=heldout,
        query_sidecar=sidecars["retrieval_sidecar"],
        document_sidecar=sidecars["retrieval_document_sidecar"],
        recipe_path=recipe,
        bank_path=bank,
        device="cpu",
    )


def _readiness(arguments):
    recipe = RetrievalRecipe.load(arguments["recipe_path"])
    with open_bank_for_recipe(recipe, arguments["bank_path"], readonly=True) as bank:
        return bank.growth_state("readiness", {})


def test_measured_gates_write_evidence_receipt_and_readiness(tmp_path):
    arguments = _setup(tmp_path)
    result = bootstrap_gates(
        **arguments, output=tmp_path / "pass", thresholds=LENIENT, mark_ready=True
    )
    assert result["ready"] and set(BOOTSTRAP_GATES) <= set(result["passed"])
    evidence = result["evidence"]
    assert evidence["coherence"]["lexical_ce_per_token"] > 0
    assert evidence["embedding_alignment"]["documents"] == 3
    assert evidence["query_alignment"]["queries"] == 3
    assert 0 <= evidence["query_alignment"]["student_recall"] <= 1
    receipt = json.loads((tmp_path / "pass" / "progressive_warmup.gate.json").read_text())
    assert receipt["phase"] == "progressive_warmup"
    assert set(receipt["successful"]) == set(result["passed"])
    for gate in receipt["successful"]:
        assert json.loads(open(receipt["evidence"][gate]).read())["passed"]
    readiness = _readiness(arguments)
    assert readiness["coherence"] and readiness["embeddings"]


def test_failed_gate_writes_evidence_but_no_receipt_or_readiness(tmp_path):
    arguments = _setup(tmp_path)
    strict = GateThresholds(**{**LENIENT.__dict__, "max_lexical_ce": 0.0})
    result = bootstrap_gates(
        **arguments, output=tmp_path / "fail", thresholds=strict, mark_ready=True
    )
    assert not result["ready"] and "coherence" not in result["passed"]
    assert (tmp_path / "fail" / "coherence.json").exists()
    assert not (tmp_path / "fail" / "progressive_warmup.gate.json").exists()
    assert not _readiness(arguments)["coherence"]
