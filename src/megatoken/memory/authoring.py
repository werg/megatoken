"""Frozen learned-coordinate new-source authoring and full shadow-bank reproduction.

Serving sees actual source E facets and immutable assignment/source metadata; it
never reads optimized memory target values, future answers or a latent free bank.
Source and pair writers are separately registered/frozen checkpoint artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from megatoken.io import sha256_file
from megatoken.memory.contracts import Footprint, SemanticSpace, SourceRange
from megatoken.memory.reproduction import (
    RebuildResult, ReproducedPair, ReproducedPoint,
    ReproducedSource, regenerate_bank,
)
from megatoken.memory.writers import RelationWriter, SourceWriter


@dataclass(frozen=True)
class SourceAssignment:
    """Immutable original assignment correspondence, not a writer input tensor."""
    identity: str
    content_id: str
    scope: str
    footprint: Footprint
    evidence_span: SourceRange
    asset_ref: str
    original_source: str
    original_distance: float
    batch_index: int
    output_slot: int


@dataclass(frozen=True)
class PairAssignment:
    left_assignment: str
    right_assignment: str
    scope: str
    footprint: Footprint
    prior: float
    statistics: Tensor   # source-window statistics, NEVER optimized pair target


class FrozenWriterArtifact:
    def __init__(self, root: Path, *, expected_recipe: str,
                 expected_space: str, expected_model_snapshot: str,
                 expected_free_bank: str, kind: str):
        from safetensors.torch import load_file

        info = json.loads((root / "manifest.json").read_text())
        if info.get("schema") != "megatoken-writer-checkpoint-v1" or (
            info["kind"] != kind or info["mode"] != "full"
            or info["recipe"] != expected_recipe
            or info["semantic_space"] != expected_space
            or info["native_model_snapshot"] != expected_model_snapshot
            or info["free_kb_snapshot"] != expected_free_bank
        ):
            raise ValueError("Writer artifact is not from the accepted frozen final space")
        file = root / "writer.safetensors"
        if sha256_file(file) != info["weights_sha256"]:
            raise RuntimeError("Frozen writer weights differ from their manifest")
        shape = info["layout"]
        if kind == "source":
            model = SourceWriter(
                facets=shape["facets"], facet_dim=shape["facet_dim"],
                semantic_dim=shape["semantic_dim"], value_dim=shape["value_dim"],
                max_blocks=shape["max_blocks"],
                hidden=info["hidden"], heads=info["heads"],
            )
        elif kind == "pair":
            model = RelationWriter(
                key_dim=shape["semantic_dim"], value_dim=shape["value_dim"],
                statistics_dim=shape["statistics_dim"], hidden=info["hidden"],
            )
        else:
            raise ValueError("Writer artifact kind is unknown")
        model.load_state_dict(load_file(str(file), device="cpu"), strict=True)
        self.model = model.eval().requires_grad_(False)
        self.manifest = info


class FrozenCorpusAuthor:
    def __init__(self, *, source_writer: FrozenWriterArtifact,
                 pair_writer: FrozenWriterArtifact,
                 semantic_space: SemanticSpace, value_dim: int):
        if source_writer.manifest["kind"] != "source" or (
            pair_writer.manifest["kind"] != "pair"
        ):
            raise ValueError("Corpus author requires both accepted writer owners")
        if (
            source_writer.manifest["semantic_space"] != semantic_space.fingerprint
            or pair_writer.manifest["semantic_space"] != semantic_space.fingerprint
            or source_writer.manifest["layout"]["value_dim"] != value_dim
            or pair_writer.manifest["layout"]["value_dim"] != value_dim
        ):
            raise ValueError("Writer/key/value semantic spaces must match")
        self.source_writer = source_writer.model
        self.pair_writer = pair_writer.model
        self.space = semantic_space
        self.value_dim = value_dim
        self.points: dict[str, ReproducedPoint] = {}
        self.sources: dict[str, ReproducedSource] = {}
        self.pairs: list[ReproducedPair] = []

    @torch.no_grad()
    def add_sources(self, encoded: Tensor, valid_blocks: Tensor,
                    assignments: Sequence[SourceAssignment]) -> int:
        output = self.source_writer(
            encoded.detach(), mode="full", valid_blocks=valid_blocks,
        )
        added = 0
        for item in assignments:
            if (
                item.batch_index < 0 or item.batch_index >= encoded.shape[0]
                or item.output_slot < 0 or item.output_slot >= encoded.shape[1]
                or not bool(output.valid[item.batch_index, item.output_slot])
            ):
                raise ValueError("Corpus author has missing source-slot correspondence")
            if not item.identity or not item.content_id or not item.asset_ref:
                raise ValueError("Source authoring requires stable literal evidence identity")
            if item.identity in self.points:
                raise ValueError("Original source assignment was regenerated twice")
            b, slot = item.batch_index, item.output_slot
            key = output.key[b, slot].float().cpu().numpy().copy()
            point = ReproducedPoint(
                source_assignment=item.identity,
                scope=item.scope, footprint=item.footprint,
                key=key, value=output.value[b, slot].float().cpu().numpy().copy(),
                strength=float(output.strength[b, slot]),
                lifetime_logit=float(output.lifetime_logit[b, slot]),
                original_source=item.original_source,
                assignment_distance=item.original_distance,
                output_slot=slot,
            )
            existing = self.sources.get(item.content_id)
            if existing is not None:
                if (
                    existing.scope != item.scope or
                    existing.footprint != item.footprint or
                    existing.asset_ref != item.asset_ref or
                    not np.allclose(existing.key, key, atol=1e-6)
                ):
                    raise ValueError("Source chunk has conflicting regenerated key/provenance")
            else:
                self.sources[item.content_id] = ReproducedSource(
                    item.content_id, item.scope, item.footprint,
                    item.evidence_span, item.asset_ref, key,
                )
            self.points[item.identity] = point
            added += 1
        return added

    @torch.no_grad()
    def add_pairs(self, assignments: Sequence[PairAssignment]) -> int:
        seen = {tuple(sorted((p.left_assignment, p.right_assignment)))
                for p in self.pairs}
        device = next(self.pair_writer.parameters()).device

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=device)[None]

        for item in assignments:
            names = tuple(sorted((item.left_assignment, item.right_assignment)))
            if names in seen or names[0] == names[1] or (
                names[0] not in self.points or names[1] not in self.points
            ):
                raise ValueError("Pair authoring requires distinct existing point assignments")
            left, right = (self.points[x] for x in names)
            if left.scope != item.scope or right.scope != item.scope or (
                any(r.scope != item.scope for r in item.footprint.ranges)
            ):
                raise ValueError("Pair original support scope mismatches endpoints")
            stats = item.statistics.detach().float().to(device)
            if stats.ndim != 1:
                raise ValueError("Pair source-window stats must be one bounded vector")
            answer = self.pair_writer(
                tensor(left.key), tensor(right.key),
                tensor(left.value), tensor(right.value), stats[None],
            )
            self.pairs.append(ReproducedPair(
                names[0], names[1], item.scope, item.footprint,
                item.prior,
                answer.value[0].cpu().numpy().copy(),
                float(answer.strength[0]),
                float(answer.lifetime_logit[0]),
                tuple(float(x) for x in answer.dependence_logits[0]),
            ))
            seen.add(names)
        return len(assignments)

    def publish(self, output: str | Path, *,
                expected_points: set[str], expected_pairs: set[tuple[str, str]],
                expected_sources: set[str]) -> RebuildResult:
        # An empty target/assignment is never treated as valid partial coverage.
        return regenerate_bank(
            output, self.space, value_dim=self.value_dim,
            sources=list(self.sources.values()),
            points=list(self.points.values()), pairs=self.pairs,
            expected_sources=expected_sources,
            expected_assignments=expected_points,
            expected_pairs=expected_pairs,
        )
