"""Model+KB publication must expose either a complete pair or nothing."""

from pathlib import Path

import pytest
import torch

from megatoken.memory.contracts import Footprint, SemanticSpace, SourceRange
from megatoken.memory.growth import GrowthPolicy
from megatoken.memory.recipe import (
    RetrievalRecipe,
    checkpoint_bank,
    validate_checkpoint,
)
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model.config import ModelConfig
from megatoken.training.checkpoint import (
    load_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)


def _recipe():
    config = ModelConfig.tiny(
        semantic_dim=8,
        semantic_value_dim=16,
        max_kb_positions=2,
    )
    space = SemanticSpace(
        "local",
        "r1",
        "teacher",
        "pinned",
        8,
        "document",
        "query",
        "ordered",
    )
    return RetrievalRecipe(
        schema="megatoken-retrieval-recipe-v1",
        model=config,
        space=space,
        growth=GrowthPolicy(value_dim=16, seed=1),
        loss_weights={"native": 1.0},
        phase="free_kb_pretraining",
        max_kb_positions=2,
        max_readout_groups=2,
    )


def _bank(tmp_path, recipe):
    bank = SQLiteKB(
        tmp_path / "working-bank.sqlite",
        recipe.space,
        value_dim=recipe.model.semantic_value_dim,
    )
    bank.insert_memory(
        scope="team",
        footprint=Footprint((SourceRange("source", "v1", 0, 8, "source-bytes", "team"),)),
        key=[1.0] + [0.0] * 7,
        value=[0.01] * 16,
        assignment="original-source",
        birth_step=0,
    )
    return bank


def test_checkpoint_atomically_bundles_versioned_model_and_bank(tmp_path):
    recipe = _recipe()
    with _bank(tmp_path, recipe) as bank:
        root = tmp_path / "checkpoints"
        name = "step-001"

        def attachment(stage: Path):
            return {
                "memory_checkpoint": checkpoint_bank(
                    recipe,
                    bank,
                    checkpoint_file=stage / "kb.sqlite",
                    model_checkpoint_id=name,
                    optimizer_step=1,
                    growth_cursor="cursor-1",
                    query_model_snapshot="native-query-1",
                )
            }

        destination = save_checkpoint(
            root,
            name,
            {"model": {"x": torch.ones(2)}},
            {"global_step": 1},
            attachments=attachment,
        )
        payload, metadata = load_checkpoint(destination)
        assert payload["model"]["x"].tolist() == [1.0, 1.0]
        assert metadata["memory_checkpoint"]["optimizer_step"] == 1
        validate_checkpoint(
            recipe,
            destination / "kb.sqlite",
            model_checkpoint_id=name,
        )
        assert resolve_checkpoint(root) == destination
        with (destination / "kb.sqlite").open("ab") as stream:
            stream.write(b"not-part-of-pinned-snapshot")
        with pytest.raises(ValueError, match="memory artifact"):
            load_checkpoint(destination)


def test_failed_memory_snapshot_never_publishes_a_neural_checkpoint(tmp_path):
    root = tmp_path / "checkpoints"

    def fail(_stage):
        raise RuntimeError("injected memory backup failure")

    with pytest.raises(RuntimeError, match="injected memory backup failure"):
        save_checkpoint(
            root,
            "step-001",
            {"model": {}},
            {"global_step": 1},
            attachments=fail,
        )
    assert not (root / "step-001").exists()
    assert not (root / "latest.json").exists()
