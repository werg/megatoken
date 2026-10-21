"""Pinned offline source/pair writer-fitting entry point.

Usage:
  megatoken-memory-writer --recipe recipe.json --gate-receipt receipt.json
      --dataset source-writer.json --output writer-artifacts --epochs 1

This is a distillation launcher; native frozen-reader continuation fitting is
available through training.hindsight's behavior callback, not silently claimed
by a target-only run.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch

from megatoken.memory.recipe import GateReceipt, RetrievalRecipe, validate_gate
from megatoken.memory.writers import RelationWriter, SourceWriter
from megatoken.training.hindsight import (
    WriterFitSettings,
    fit_relation_writer_epoch,
    fit_source_writer_epoch,
)
from megatoken.training.writer_data import WriterDataset, pair_batches, source_batches


def fit_pinned_dataset(
    recipe: RetrievalRecipe,
    receipt: GateReceipt,
    dataset: WriterDataset,
    *,
    dataset_path: Path,
    output: Path,
    epochs: int,
    lr: float,
    hidden: int,
    heads: int,
    seed: int,
    device: torch.device,
) -> dict:
    validate_gate(recipe, receipt)
    if receipt.phase != recipe.phase:
        raise RuntimeError("Writer checkpoint requires an exact phase gate receipt")
    if recipe.phase not in {
        "local_source_writer",
        "cross_source_writer",
        "corpus_reproduction",
    }:
        raise RuntimeError("Source writer cannot train before free-memory pretraining")
    if dataset.kind == "pair" and recipe.phase != "corpus_reproduction":
        raise RuntimeError("Pair reproduction requires the post-pretraining corpus phase")
    if recipe.phase in {"local_source_writer", "cross_source_writer"} and (
        dataset.mode != "suffix" or dataset.kind != "source"
    ):
        raise RuntimeError("Local/cross-source writers require the physical suffix view")
    if recipe.phase == "corpus_reproduction" and dataset.mode != "full":
        raise RuntimeError("Corpus reproduction must use complete permitted source footprints")
    if (
        dataset.recipe_fingerprint != recipe.fingerprint
        or dataset.semantic_space_fingerprint != recipe.space.fingerprint
        or dataset.model_checkpoint_id != receipt.model_snapshot
        or dataset.free_bank_sha256 != receipt.kb_snapshot
        or dataset.source_context_policy != recipe.space.context_recipe
    ):
        raise RuntimeError("Writer inputs are incompatible with frozen target coordinates")
    if epochs < 1 or lr <= 0:
        raise ValueError("Writer epochs and learning rate must be positive")
    if output.exists():
        raise FileExistsError("Writer publication must use an empty new directory")
    layout = dataset.layout
    if (
        layout["semantic_dim"] != recipe.model.semantic_dim
        or layout["value_dim"] != recipe.model.semantic_value_dim
    ):
        raise ValueError("Prepared writer dimensions differ from canonical KB geometry")
    torch.manual_seed(seed)
    if dataset.kind == "source":
        if (
            layout["facets"] != recipe.model.n_facets
            or layout["facet_dim"] != recipe.model.facet_dim
        ):
            raise ValueError("Source writer must consume the actual native E facet layout")
        writer = SourceWriter(
            facets=layout["facets"],
            facet_dim=layout["facet_dim"],
            semantic_dim=layout["semantic_dim"],
            value_dim=layout["value_dim"],
            max_blocks=layout["max_blocks"],
            hidden=hidden,
            heads=heads,
        ).to(device)
    else:
        writer = RelationWriter(
            key_dim=layout["semantic_dim"],
            value_dim=layout["value_dim"],
            statistics_dim=layout["statistics_dim"],
            hidden=hidden,
        ).to(device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=lr, weight_decay=0.0)
    settings = WriterFitSettings(learning_rate=lr, distillation_weight=1.0)
    metrics = []
    for epoch in range(epochs):
        if dataset.kind == "source":
            reports = fit_source_writer_epoch(
                writer,
                source_batches(dataset_path, dataset, device=device),
                settings=settings,
                optimizer=optimizer,
            )
        else:
            reports = fit_relation_writer_epoch(
                writer,
                pair_batches(dataset_path, dataset, device=device),
                settings=settings,
                optimizer=optimizer,
            )
        if reports["updates"] == 0:
            raise RuntimeError("No valid source/pair assignments were fitted")
        metrics.append({"epoch": epoch + 1, **reports})
    from safetensors.torch import save_file

    # Publish atomically into an otherwise absent output directory; neither
    # weights nor receipts can be mistaken for the optimized free KB.
    parent = output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".writer-stage-", dir=parent))
    try:
        weights = staging / "writer.safetensors"
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in writer.state_dict().items()
            },
            str(weights),
        )
        metadata = {
            "schema": "megatoken-writer-checkpoint-v1",
            "kind": dataset.kind,
            "mode": dataset.mode,
            "recipe": recipe.fingerprint,
            "semantic_space": recipe.space.fingerprint,
            "native_model_snapshot": receipt.model_snapshot,
            "free_kb_snapshot": receipt.kb_snapshot,
            "source_context_policy": dataset.source_context_policy,
            "dataset_manifest_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
            "shards": [asdict(ref) for ref in dataset.shards],
            "layout": layout,
            "epochs": epochs,
            "seed": seed,
            "hidden": hidden,
            "heads": heads,
            "learning_rate": lr,
            "training_objective": "frozen-target-distillation",
            "behavioral_continuation_verified": False,
            "metrics": metrics,
            "weights_sha256": sha256(weights.read_bytes()).hexdigest(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
    return metadata


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit frozen-coordinate KB source/pair writers")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    recipe = RetrievalRecipe.load(args.recipe)
    received = json.loads(args.gate_receipt.read_text())
    receipt = GateReceipt(**{**received, "successful": tuple(received["successful"])})
    dataset = WriterDataset.load(args.dataset)
    report = fit_pinned_dataset(
        recipe,
        receipt,
        dataset,
        dataset_path=args.dataset,
        output=args.output,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        heads=args.heads,
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
