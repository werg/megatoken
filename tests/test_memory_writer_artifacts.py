"""Prepared writer artifacts are pinned, checked and gated before publication."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from megatoken.memory.recipe import PHASE_REQUIREMENTS, GateReceipt, RetrievalRecipe
from megatoken.training.writer_cli import fit_pinned_dataset
from megatoken.training.writer_data import WriterDataset, WriterShard


def _recipe():
    config = Path(__file__).resolve().parents[1] / "configs/retrieval-memory-pilot.json"
    data = json.loads(config.read_text())
    data["phase"] = "local_source_writer"
    data["space"]["teacher_revision"] = "pinned-revision-test"
    data["growth"]["partition_id"] = "pinned-split-test"
    return RetrievalRecipe.from_dict(data)


def _receipt():
    phase = "local_source_writer"
    names = PHASE_REQUIREMENTS[phase]
    return GateReceipt(
        phase, {key: f"receipt-{key}" for key in names},
        "model-snapshot", "bank-snapshot", "corpus-manifest", tuple(names),
    )


def test_writer_dataset_rejects_missing_provenance():
    with pytest.raises(ValueError, match="source identity"):
        WriterShard("shard.safetensors", "a" * 64, "", "assignments")


def test_writer_distillation_cli_publishes_only_pinned_frozen_target_outputs(tmp_path):
    pytest.importorskip("safetensors.torch")
    from safetensors.torch import save_file

    recipe = _recipe()
    receipt = _receipt()
    f = recipe.model.n_facets
    d = recipe.model.facet_dim
    semantic = recipe.space.dimension
    value = recipe.model.semantic_value_dim
    shard_path = tmp_path / "source.safetensors"
    save_file({
        "encoded": torch.randn(1, 4, f, d),
        "valid_blocks": torch.ones(1, 4, dtype=torch.bool),
        "slots": torch.tensor([[0]], dtype=torch.long),
        "key": torch.randn(1, 1, semantic),
        "value": torch.randn(1, 1, value),
        "strength": torch.ones(1, 1) * 0.5,
        "lifetime_logit": torch.ones(1, 1) * -7,
        "distance": torch.zeros(1, 1),
        "valid": torch.ones(1, 1, dtype=torch.bool),
    }, str(shard_path))
    ref = WriterShard(
        path=shard_path.name,
        sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
        source_footprint_digest="complete-source-footprint",
        assignment_digest="original-source-to-record-assignments",
    )
    manifest = WriterDataset(
        schema="megatoken-writer-dataset-v1", kind="source",
        recipe_fingerprint=recipe.fingerprint,
        semantic_space_fingerprint=recipe.space.fingerprint,
        model_checkpoint_id=receipt.model_snapshot,
        free_bank_sha256=receipt.kb_snapshot,
        source_context_policy=recipe.space.context_recipe,
        mode="suffix",
        layout={
            "facets": f, "facet_dim": d, "semantic_dim": semantic,
            "value_dim": value, "max_blocks": 4,
        },
        shards=(ref,),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        **{key: value for key, value in manifest.__dict__.items() if key != "shards"},
        "shards": [ref.__dict__],
    }))
    result = fit_pinned_dataset(
        recipe, receipt, manifest, dataset_path=manifest_path,
        output=tmp_path / "writer-output", epochs=1, lr=0.001,
        hidden=16, heads=4, seed=17, device=torch.device("cpu"),
    )
    assert result["metrics"][0]["valid_targets"] == 1
    assert not result["behavioral_continuation_verified"]
    assert (tmp_path / "writer-output" / "writer.safetensors").exists()
    with pytest.raises(FileExistsError):
        fit_pinned_dataset(
            recipe, receipt, manifest, dataset_path=manifest_path,
            output=tmp_path / "writer-output", epochs=1, lr=0.001,
            hidden=16, heads=4, seed=17, device=torch.device("cpu"),
        )
