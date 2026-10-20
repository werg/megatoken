"""Strict retrieval-memory curriculum, phase gate, and coherent KB artifact manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import argparse
import json
from pathlib import Path

from megatoken.model.config import ARCHITECTURE_VERSION, ModelConfig
from megatoken.io import sha256_file

from .contracts import SemanticSpace
from .growth import GrowthPolicy
from .sqlite import SQLiteKB


PHASE_ORDER = (
    "coherence_embedding",
    "progressive_warmup",
    "free_kb_pretraining",
    "local_source_writer",
    "cross_source_writer",
    "corpus_reproduction",
    "acceptance",
    "experience",
)

PHASE_REQUIREMENTS = {
    "coherence_embedding": (),
    "progressive_warmup": ("coherence", "embedding_alignment", "query_alignment"),
    "free_kb_pretraining": ("coherence", "embedding_alignment", "query_alignment",
                            "empty_bank", "eligible_gradient", "index_recall"),
    "local_source_writer": ("native_usefulness", "final_free_bank_checkpoint"),
    "cross_source_writer": ("local_writer_quality", "permitted_suffix"),
    "corpus_reproduction": ("transfer_quality", "frozen_target_assignment"),
    "acceptance": ("point_reconstruction", "pair_reconstruction",
                   "fresh_index_recall", "behavioral_uptake"),
    "experience": ("frozen_reader_policy", "verified_behavior_replay",
                   "regenerated_write_acceptance"),
}


@dataclass(frozen=True)
class RetrievalRecipe:
    schema: str
    space: SemanticSpace
    model: ModelConfig
    growth: GrowthPolicy
    loss_weights: dict[str, float]
    phase: str = "coherence_embedding"
    router_balance_mode: str = "global"
    max_selected_source: int = 32
    max_selected_memory: int = 32
    max_source_bytes: int = 16384
    max_kb_positions: int = 4
    max_readout_groups: int = 4
    extra_rules: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != "megatoken-retrieval-recipe-v1":
            raise ValueError("Unknown retrieval-memory training recipe schema")
        if self.phase not in PHASE_ORDER or self.router_balance_mode != "global":
            raise ValueError("Retrieval pretraining must use declared phases/global router balance")
        if self.model.semantic_dim != self.space.dimension:
            raise ValueError("Embedding readout dimension differs from the teacher/KB space")
        if self.model.semantic_value_dim != self.growth.value_dim or (
            self.model.max_kb_positions != self.max_kb_positions
        ):
            raise ValueError("Memory value and native K/V geometry must match the recipe")
        if min(self.max_selected_source, self.max_selected_memory, self.max_source_bytes,
               self.max_kb_positions, self.max_readout_groups) < 0:
            raise ValueError("Retrieval budgets must be bounded and nonnegative")
        if self.max_kb_positions < self.max_readout_groups:
            raise ValueError("A KB snapshot cannot contain more groups than materialized slots")
        if not self.loss_weights or any(value < 0 for value in self.loss_weights.values()):
            raise ValueError("Pretraining losses must have explicit nonnegative weights")

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def to_dict(self):
        return {
            "schema": self.schema, "space": asdict(self.space),
            "model": self.model.to_dict(), "growth": asdict(self.growth),
            "loss_weights": dict(self.loss_weights), "phase": self.phase,
            "router_balance_mode": self.router_balance_mode,
            "max_selected_source": self.max_selected_source,
            "max_selected_memory": self.max_selected_memory,
            "max_source_bytes": self.max_source_bytes,
            "max_kb_positions": self.max_kb_positions,
            "max_readout_groups": self.max_readout_groups,
            "extra_rules": dict(self.extra_rules),
        }

    @classmethod
    def from_dict(cls, data: dict):
        data = dict(data)
        data["space"] = SemanticSpace(**data["space"])
        model = dict(data["model"])
        model.setdefault("architecture_version", ARCHITECTURE_VERSION)
        data["model"] = ModelConfig.from_dict(model)
        data["growth"] = GrowthPolicy(**data["growth"])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class GateReceipt:
    phase: str
    evidence: dict[str, str]
    model_snapshot: str
    kb_snapshot: str
    corpus_manifest: str
    successful: tuple[str, ...]

    def __post_init__(self):
        if self.phase not in PHASE_ORDER or not self.model_snapshot or not self.kb_snapshot:
            raise ValueError("Phase receipt must pin model and KB identities")


def validate_gate(recipe: RetrievalRecipe, receipt: GateReceipt) -> None:
    if PHASE_ORDER.index(receipt.phase) > PHASE_ORDER.index(recipe.phase):
        raise ValueError("A future-phase receipt cannot advance an earlier recipe")
    passed = set(receipt.successful)
    needed = set(PHASE_REQUIREMENTS[recipe.phase])
    missing = needed - passed
    if missing:
        raise RuntimeError(f"Phase {recipe.phase} has unverified gates: {sorted(missing)}")
    if any(not receipt.evidence.get(g) for g in needed):
        raise RuntimeError("Every completed gate requires a traceable evidence artifact")


def open_bank_for_recipe(recipe: RetrievalRecipe, path: str | Path, *,
                         readonly: bool = False) -> SQLiteKB:
    return SQLiteKB(path, recipe.space, value_dim=recipe.model.semantic_value_dim,
                    readonly=readonly)


def record_readiness(recipe: RetrievalRecipe, bank: SQLiteKB, *,
                     coherence: bool, embedding_alignment: bool,
                     query_alignment: bool) -> None:
    if recipe.phase != "coherence_embedding":
        raise RuntimeError("Readiness is set only at the curriculum bootstrap handoff")
    bank.put_growth_state("readiness", {
        "coherence": coherence,
        "embeddings": embedding_alignment and query_alignment,
        "space": recipe.space.fingerprint,
        "model": recipe.model.to_dict(),
        "recipe": recipe.fingerprint,
    })


def checkpoint_bank(
    recipe: RetrievalRecipe, bank: SQLiteKB, *,
    checkpoint_file: str | Path,
    model_checkpoint_id: str,
    optimizer_step: int,
    growth_cursor: str,
    query_model_snapshot: str,
) -> dict:
    """Publish a self-contained SQLite backup next to the matching neural step."""
    if optimizer_step < 0 or not model_checkpoint_id or not query_model_snapshot:
        raise ValueError("Checkpoint requires coordinated model, query and KB identities")
    location = bank.backup(checkpoint_file)
    info = {
        "schema": "megatoken-memory-checkpoint-v1",
        "recipe": recipe.fingerprint,
        "space": recipe.space.fingerprint,
        "kb_database_id": bank.database_id,
        "kb_sha256": sha256_file(location),
        "kb_file": location.name,
        "model_checkpoint_id": model_checkpoint_id,
        "query_model_snapshot": query_model_snapshot,
        "optimizer_step": optimizer_step,
        "growth_cursor": growth_cursor,
        "revisions": asdict(bank.revisions()),
    }
    location.with_suffix(location.suffix + ".manifest.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n"
    )
    return info


def validate_checkpoint(recipe: RetrievalRecipe, path: str | Path, *,
                        model_checkpoint_id: str) -> dict:
    path = Path(path)
    info = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    if info.get("schema") != "megatoken-memory-checkpoint-v1" or (
        info["recipe"] != recipe.fingerprint or
        info["space"] != recipe.space.fingerprint or
        info["model_checkpoint_id"] != model_checkpoint_id or
        sha256_file(path) != info["kb_sha256"]
    ):
        raise RuntimeError("Model/KB snapshot content or identity is incompatible")
    with open_bank_for_recipe(recipe, path, readonly=True) as bank:
        if asdict(bank.revisions()) != info["revisions"]:
            raise RuntimeError("KB snapshot generations do not match the publication manifest")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a pinned retrieval-memory recipe")
    parser.add_argument("recipe", help="Path to strict JSON recipe")
    parser.add_argument("--gate-receipt", help="JSON gate evidence for the configured phase")
    parser.add_argument("--bank", help="Existing canonical bank/snapshot for compatibility")
    args = parser.parse_args(argv)
    recipe = RetrievalRecipe.load(args.recipe)
    if args.gate_receipt:
        payload = json.loads(Path(args.gate_receipt).read_text())
        receipt = GateReceipt(**{
            **payload, "successful": tuple(payload["successful"])
        })
        validate_gate(recipe, receipt)
    if args.bank:
        with open_bank_for_recipe(recipe, args.bank, readonly=True) as bank:
            print("KB identity:", bank.database_id)
            print("Revisions:", asdict(bank.revisions()))
    print(json.dumps({
        "phase": recipe.phase,
        "fingerprint": recipe.fingerprint,
        "semantic_space": recipe.space.fingerprint,
        "required_gates": PHASE_REQUIREMENTS[recipe.phase],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
