"""Breaking migration: old trainers and memory-disabled checkpoints never run."""

from __future__ import annotations

import argparse
import inspect
from hashlib import sha256

import pytest

from megatoken.cli import parser
from megatoken.data.document_embeddings import (
    DocumentEmbeddingExample, read_document_sidecar, write_document_sidecar,
)
from megatoken.memory.contracts import Footprint, SourceRange
from megatoken.model.config import ModelConfig
from megatoken.model.core import FacetedCore
from megatoken.model.multitrack import MultiTrackCore, MultiTrackGenerator
from megatoken.training.config import TrainConfig


def config(**overrides):
    settings = {
        "model": "tiny",
        "phase": "B0",
        "budget": 64,
        "retrieval_sidecar": "query.jsonl",
        "retrieval_document_sidecar": "document.jsonl",
        "retrieval_memory_recipe": "pinned-retrieval-recipe.json",
    }
    settings.update(overrides)
    return TrainConfig(**settings)


def test_required_training_assets_cannot_be_switched_off():
    with pytest.raises(ValueError, match="query and document"):
        TrainConfig(budget=64)
    with pytest.raises(ValueError, match="query and document"):
        config(retrieval_document_sidecar="")
    with pytest.raises(ValueError, match="Pin actual"):
        config(retrieval_sidecar="PIN_PREPARED_QUERY")
    with pytest.raises(ValueError, match="local router"):
        config(router_balance_mode="local")
    with pytest.raises(ValueError, match="grouped megatoken"):
        config(budget_meter="auto")
    decoupled = {"coordinator_url": "https://example.invalid", "run_id": "run"}
    # DiLoCo is the target methodology: allowed without a live bank, gated with one.
    assert config(decoupled=decoupled).decoupled.run_id == "run"
    with pytest.raises(ValueError, match="KB row-sync"):
        config(decoupled=decoupled, retrieval_bank="kb.sqlite",
               retrieval_gate_receipt="gate.json")
    assert config().router_balance_mode == "global"
    assert config().budget_meter == "G"


def test_memory_geometry_is_required_even_in_coherence_bootstrap():
    geometry = ModelConfig.tiny()
    assert geometry.semantic_dim > 0
    assert geometry.semantic_value_dim > 0
    assert geometry.max_kb_positions >= 1
    with pytest.raises(ValueError, match="required on every model"):
        ModelConfig.tiny(semantic_dim=0, semantic_value_dim=0, max_kb_positions=0)
    for field in ("semantic_dim", "memory_lora_rank", "max_kb_positions"):
        old = geometry.to_dict()
        del old[field]
        with pytest.raises(ValueError, match="Old non-retrieval"):
            ModelConfig.from_dict(old)
    model = FacetedCore(geometry)
    state = MultiTrackCore(model).initial_state(1)
    assert state.kb is not None and state.kb.groups == ()
    assert model.memory_projector is not None
    assert model.memory_adapters is not None
    assert model.memory_seed is not None


def test_query_events_cannot_opt_out_of_native_commit():
    assert "commit" not in inspect.signature(
        MultiTrackGenerator.retrieve_embedding
    ).parameters


def test_public_legacy_training_commands_are_absent():
    command_action = next(
        action for action in parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    for previous in ("rl", "smoke"):
        assert previous not in command_action.choices
    diloco = command_action.choices["diloco"]
    sub = next(a for a in diloco._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == {"init", "serve", "export"}
    train = command_action.choices["train"]
    config_action = next(
        action for action in train._actions
        if action.dest == "config"
    )
    assert config_action.required


def test_document_teacher_roundtrip_and_original_byte_causality(tmp_path):
    content = b"only these original source bytes"
    span = SourceRange(
        "project/document", "v1", 0, len(content), sha256(content).hexdigest(),
        "workspace", available_at=0, family="source-family",
    )
    label = DocumentEmbeddingExample(
        episode_id="episode-a",
        event_id="source-block-a",
        payload_sha256=sha256(content).hexdigest(),
        teacher_inputs=Footprint((span,)),
        source_span=span,
        teacher_target=tuple([0.5] * 8),
        preprocessing_revision="pinned-document-teacher-v1",
    )
    path = tmp_path / "teacher-document.jsonl"
    write_document_sidecar(path, [label])
    assert read_document_sidecar(path) == [label]
    label.validate_event(payload=content, clock=1)
    with pytest.raises(ValueError, match="differs"):
        label.validate_event(payload=b"spoofed", clock=1)
    with pytest.raises(ValueError, match="unavailable"):
        label.validate_event(payload=content, clock=-1)
    data = path.read_bytes()
    path.write_bytes(data + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        read_document_sidecar(path)
