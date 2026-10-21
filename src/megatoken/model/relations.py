"""Bounded, independently counted semantic relation loss on sampled output peers.

Teacher target encodings remain ONLY in the loss branch after prospective P.
Never force distinct speech/text/image/video raw facet tensors to equality.
Actual acoustic/visual timing and preservation are checked on sampled decoded
outputs in the separate concordance evaluation module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
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
MODALITIES = ("text", "audio", "image", "video")


@dataclass(frozen=True, slots=True)
class SelectedRelation:
    kind: str
    left: int
    right: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS or self.left < 0 or self.right < 0 or self.left == self.right:
            raise ValueError("Invalid supervised pair relation")


class RelationProjection(nn.Module):
    """Per-modality learned semantic projection, NOT an alternative core.

    The corresponding E target lives in the supervised loss only. Pooling and
    separate modality mappings avoid the invalid full-facet equality objective.
    """

    def __init__(self, width: int, hidden: int | None = None):
        super().__init__()
        shared = hidden or min(128, width)
        self.projectors = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, shared),
                    nn.SiLU(),
                    nn.Linear(shared, shared),
                )
                for modality in MODALITIES
            }
        )

    def project(self, value: Tensor, modality: str) -> Tensor:
        if modality not in self.projectors or value.ndim != 3:
            raise ValueError("Relation projection needs one event and known modality")
        layer = self.projectors[modality]
        features = value.mean(1).to(next(layer.parameters()).dtype)
        return F.normalize(layer(features).float(), dim=-1)

    def forward(
        self,
        selected: Tensor,
        teachers: Tensor,
        modalities: tuple[str, ...],
        relations: tuple[SelectedRelation, ...],
    ) -> tuple[Tensor, int]:
        if selected.shape != teachers.shape or selected.ndim != 4:
            raise ValueError("Relation supervision needs sampled and E-teacher raw groups")
        if len(modalities) != selected.shape[1]:
            raise ValueError("Missing modality identity for relation participants")
        zero = selected.sum() * 0
        total = zero
        count = 0
        for rel in relations:
            if max(rel.left, rel.right) >= len(modalities):
                raise ValueError("Relation index is outside this jointly selected meridian")
            # Directional contrast to detached counterpart representations.
            # Each term supervises a free P sample against an independently
            # encoded target. No teacher peer can enter a current P query.
            left = self.project(selected[:, rel.left], modalities[rel.left])
            right = self.project(selected[:, rel.right], modalities[rel.right])
            with torch.no_grad():
                tleft = self.project(teachers[:, rel.left].detach(), modalities[rel.left])
                tright = self.project(teachers[:, rel.right].detach(), modalities[rel.right])
            total = (
                total
                + ((1 - (left * tright).sum(-1)).mean() + (1 - (right * tleft).sum(-1)).mean())
                * 0.5
            )
            count += 1
        return total, count
