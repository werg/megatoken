"""Causal retrieval-pretraining utilities with teacher targets, local feature utility,
selected sparse KB leaves and independent full-KV / LoRA dropout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from megatoken.generation.embedding import EmbeddingSpec
from megatoken.memory.contracts import Collection
from megatoken.memory.optimizer import RowID, SparseRowAdam
from megatoken.memory.reader import (
    build_memory_groups,
    local_source_value_loss,
    selected_evidence,
)


@dataclass(frozen=True)
class RetrievalWeights:
    embedding: float = 1.0
    ranking: float = 0.25
    source_value: float = 0.1
    key_anchor: float = 0.01
    null: float = 0.1
    memory_regularization: float = 1e-3

    def __post_init__(self):
        if any(
            x < 0
            for x in (
                self.embedding,
                self.ranking,
                self.source_value,
                self.key_anchor,
                self.null,
                self.memory_regularization,
            )
        ):
            raise ValueError("Retrieval loss weights must be nonnegative")


@dataclass(frozen=True)
class LocalRetrievalTarget:
    teacher_query: Tensor
    useful_source: Tensor
    annotated_complete: Tensor
    source_features: Tensor
    key_teacher: Tensor | None = None


def query_embedding_loss(
    realizer, raw: Tensor, target: Tensor, spec: EmbeddingSpec
) -> tuple[Tensor, int]:
    query = realizer(raw, spec)
    teacher = F.normalize(target.detach().float(), dim=-1)
    if query.shape != teacher.shape:
        raise ValueError("Teacher query target does not match selected raw embedding R")
    return (1 - (query * teacher).sum(-1)).sum(), query.shape[0]


def annotated_ranking_loss(
    scores, positive: Tensor, complete: Tensor, *, eta: float = 1.0
) -> tuple[Tensor, int]:
    positives = positive.bool().detach() & scores.valid
    mass = scores.weights.masked_fill(~positives, 0).sum(-1)
    selected = positives.any(-1)
    # A positive masked by ANN/eligibility is a retrieval miss, not negative
    # supervision. Reserve the null loss for annotated empty support only.
    explicit_null = complete.detach().bool() & ~positive.detach().bool().any(-1)
    losses = torch.where(
        selected,
        -eta * mass.clamp_min(1e-8).log(),
        -eta * scores.null_weight.clamp_min(1e-8).log(),
    )
    valid = selected | explicit_null
    return losses[valid].sum(), int(valid.sum())


class LocalMemoryWindow:
    """A selected-row optimizer bridge, not a dense bank or a second trainer."""

    def __init__(self, sparse: SparseRowAdam, layout: str, device: torch.device):
        self.sparse, self.layout, self.device = sparse, layout, device

    def field(self, collection: Collection, kind: str, record_id: int, name: str) -> Tensor:
        leaf = self.sparse.select(RowID(collection, kind, record_id, name, self.layout))
        if leaf.device != self.device:
            # Device selection retains differentiability; leaves keep one CPU FP32
            # storage owner and receive gradient through a transfer node.
            return leaf.to(self.device)
        return leaf

    def source_loss(
        self,
        query: Tensor,
        ids: Sequence[int],
        target: LocalRetrievalTarget,
        *,
        eligible: Tensor,
        threshold: float = 0.25,
    ) -> tuple[Tensor, int]:
        batch = query.shape[0]
        keys = (
            torch.stack([self.field(Collection.SOURCE, "record", i, "key") for i in ids], 0)[
                None
            ].expand(batch, -1, -1)
            if ids
            else query.new_empty(batch, 0, query.shape[-1])
        )
        scores = selected_evidence(query, keys, masks=eligible, threshold=threshold)
        return local_source_value_loss(
            scores,
            target.source_features,
            positive=target.useful_source,
            complete_annotations=target.annotated_complete,
        )

    def memory_groups(
        self,
        query: Tensor,
        ids: Sequence[int],
        *,
        anchor: int,
        eligible: Tensor,
        scopes: Sequence[str],
        groups: Sequence[str],
        versions: Sequence[str],
        threshold: float = 0.25,
        max_groups: int = 4,
    ):
        batch = query.shape[0]
        keys = (
            torch.stack([self.field(Collection.MEMORY, "record", i, "key") for i in ids], 0)[
                None
            ].expand(batch, -1, -1)
            if ids
            else query.new_empty(batch, 0, query.shape[-1])
        )
        values = (
            torch.stack([self.field(Collection.MEMORY, "record", i, "value") for i in ids], 0)[
                None
            ].expand(batch, -1, -1)
            if ids
            else query.new_empty(batch, 0, self.sparse.bank.value_dim)
        )
        strengths = (
            torch.stack(
                [self.field(Collection.MEMORY, "record", i, "strength").reshape(()) for i in ids], 0
            )[None].expand(batch, -1)
            if ids
            else query.new_empty(batch, 0)
        )
        lifetimes = (
            torch.stack(
                [
                    self.field(Collection.MEMORY, "record", i, "lifetime_logit").reshape(())
                    for i in ids
                ],
                0,
            )[None].expand(batch, -1)
            if ids
            else query.new_empty(batch, 0)
        )
        return build_memory_groups(
            queries=query,
            keys=keys,
            semantics=values,
            strengths=strengths,
            lifetime_logits=lifetimes,
            record_ids=[str(x) for x in ids],
            scopes=scopes,
            groups=groups,
            versions=versions,
            eligible=eligible,
            anchor=anchor,
            threshold=threshold,
            max_groups=max_groups,
        )

    def advance(self, *, step: int, max_norm: float = 1.0) -> int:
        return self.sparse.step(step=step, max_norm=max_norm)


def parameter_ownership(model):
    """Auditable no-alias ownership across model, realizer, projector and bank."""
    named = list(model.named_parameters(remove_duplicate=False))
    owners: dict[int, list[str]] = {}
    for name, tensor in named:
        owners.setdefault(id(tensor), []).append(name)
    collisions = {key: names for key, names in owners.items() if len(names) != 1}
    if collisions:
        raise ValueError(f"Model has duplicated registered parameter owners: {collisions}")
    return {name for name, _ in named}
