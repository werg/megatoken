"""Semantic relation loss on sampled output peers, against detached teacher encodings.

For each annotated pair (left, right) the sampled facets of one side are pulled
toward the E-teacher encoding of the other side, in a *fixed* feature space: the
facet block mean-pooled over its facets and unit-normalized. Nothing in the
loss is trainable. A learned projection shared by both sides with no negative
examples can drive the loss to zero by mapping every input to one constant
vector (a grouped meridian carries one episode, so there is no in-batch
contrast to prevent that); a fixed map cannot, so only the sampled facets move.

Teacher target encodings remain ONLY in the loss branch after prospective P.
Actual acoustic/visual timing and preservation are checked on sampled decoded
outputs in the separate concordance evaluation module.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

KINDS = frozenset(
    {
        "same_spoken_content",
        "translation_of",
        "describes",
        "synchronized_with",
        "first_frame_of",
        "last_frame_of",
        "edit_of",
        "preserves",
    }
)


@dataclass(frozen=True, slots=True)
class SelectedRelation:
    kind: str
    left: int
    right: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.left < 0 or self.right < 0 or self.left == self.right:
            raise ValueError("Invalid supervised pair relation")


def _pooled(value: Tensor) -> Tensor:
    """[batch, facets, width] -> unit-length pooled vector [batch, width]."""
    return F.normalize(value.float().mean(1), dim=-1)


def semantic_relation_loss(
    selected: Tensor, teachers: Tensor, relations: tuple[SelectedRelation, ...]
) -> tuple[Tensor, int]:
    """Mean over relations of the two cross-cosine distances; and the relation count.

    ``selected`` and ``teachers`` are [batch, tracks, facets, width] blocks of the
    sampled and the independently encoded facets. Teachers never take gradients;
    no teacher peer can enter a current P query.
    """
    if selected.shape != teachers.shape or selected.ndim != 4:
        raise ValueError("Relation supervision needs sampled and E-teacher raw groups")
    teachers = teachers.detach()
    total = selected.sum() * 0
    for rel in relations:
        if max(rel.left, rel.right) >= selected.shape[1]:
            raise ValueError("Relation index is outside this jointly selected meridian")
        left, right = _pooled(selected[:, rel.left]), _pooled(selected[:, rel.right])
        tleft, tright = _pooled(teachers[:, rel.left]), _pooled(teachers[:, rel.right])
        total = (
            total
            + ((1 - (left * tright).sum(-1)).mean() + (1 - (right * tleft).sum(-1)).mean()) * 0.5
        )
    return total, len(relations)
