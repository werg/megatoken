"""Writing an entry by reading one item alone: the ingest and key heads.

An item is one physical record (a compressed text piece, an image, a speech or
video clip), already encoded to its native event block. Ingesting it runs C over
the item's own events under the full (bidirectional) mask, so nothing but the
item is read and nothing in the future is reachable. That is what makes it legal
at serving time, unlike the hindsight head, which reads a training sequence's
future. Two heads read the resulting C states:

- the **ingest realizer** (the hindsight realizer's architecture, its own
  weights) writes the entry's value in the bank's FP4 format;
- the **key realizer** writes a unit key in the query space, so the queries P
  already asks can match it.

Where the entry sits and how it is written into a base is decided by
``NativeTransducer.ingest`` (``model/transducer.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import METADATA_TRACK, RESERVED_TRACKS
from .core import MODALITY_IDS, ROLE_IDS, FacetedCore
from .state import KVHistory

KEY_QUERIES = 4  # attention-pooling queries of the key realizer


@dataclass(frozen=True)
class IngestItem:
    """One item to write: its native event block, kind and the metadata it is filed under."""

    raw: Tensor  # [n_facets, facet_dim], as the item is observed
    modality: str  # the event modality of ``raw`` (text, compressed_text, audio, image, video)
    metadata: bytes  # ``metadata_payload`` of the item: filed where a session with it starts
    document: str = ""  # parts of one document share this name...
    part: int = 0  # ...and are numbered from 0 in reading order

    def __post_init__(self) -> None:
        if self.modality not in MODALITY_IDS or self.part < 0 or (self.part and not self.document):
            raise ValueError("An ingest item needs a known modality; later parts name a document")


@dataclass(frozen=True)
class IngestedRows:
    """Edge fields of n items, ready for a knowledge base (bias zero: see ``NativeTransducer``)."""

    departure: Tensor  # [n, location_dim]
    key: Tensor  # [n, query_dim]
    value: Tensor  # [n, value_dim]
    destination: Tensor  # [n, location_dim]


class KeyRealizer(nn.Module):
    """One item's C states -> a unit key: attention pooling with learned queries, then a projection."""

    def __init__(self, core: FacetedCore, query_dim: int) -> None:
        super().__init__()
        width = core.config.facet_dim
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.pool = nn.Parameter(torch.randn(KEY_QUERIES, width) * 0.02)
        self.address = nn.Linear(width, width, bias=False)
        self.payload = nn.Linear(width, width, bias=False)
        self.project = nn.Linear(KEY_QUERIES * width, query_dim, bias=False)

    def forward(self, states: Tensor) -> Tensor:
        """``states`` [items, slots, width] -> unit keys [items, query_dim]."""
        if states.ndim != 3 or states.shape[2] != self.width:
            raise ValueError("Key realizer reads [items, slots, native width] states")
        dtype = self.pool.dtype
        normalized = self.norm(states.to(dtype))
        logits = torch.einsum("qw,nsw->nqs", self.pool, self.address(normalized)) / math.sqrt(
            self.width
        )
        pooled = logits.float().softmax(-1).to(dtype) @ self.payload(normalized)  # [n, q, w]
        return F.normalize(self.project(pooled.flatten(1)).float(), dim=-1, eps=1e-8)


class IngestHeads(nn.Module):
    """C over each item alone, then the ingest and key realizers."""

    def __init__(self, value_realizer: nn.Module, core: FacetedCore, query_dim: int) -> None:
        super().__init__()
        self.value = value_realizer
        self.key = KeyRealizer(core, query_dim)

    def states(self, core: FacetedCore, raws: Tensor, modalities: Sequence[str]) -> Tensor:
        """C's final states of every item read alone: [items, n_facets, width]."""
        c = core.config
        if raws.ndim != 3 or raws.shape[1:] != (c.n_facets, c.facet_dim):
            raise ValueError("Ingest reads one native event block per item")
        if len(modalities) != raws.shape[0]:
            raise ValueError("Every ingested item needs a modality")
        items, device = raws.shape[0], raws.device
        empty = raws.new_empty(items, c.kv_heads, 0, c.attention_head_dim)
        one = torch.zeros(items, 1, dtype=torch.long, device=device)
        return core.central(
            raws,
            KVHistory(empty, empty),
            torch.full_like(one, ROLE_IDS["document"]),
            torch.ones(items, c.n_facets, dtype=torch.long, device=device),
            track_ids=one,
            modality_ids=torch.tensor([[MODALITY_IDS[m]] for m in modalities], device=device),
            history_track_ids=one.new_empty(items, 0),
            # The whole item sees the whole item; the attention sink slot is added by attention.
            attention_mask=torch.ones(
                items, c.n_facets, c.n_facets, dtype=torch.bool, device=device
            ),
        )

    def forward(
        self, core: FacetedCore, raws: Tensor, modalities: Sequence[str]
    ) -> tuple[Tensor, Tensor]:
        """Values [items, value_dim] (FP4-exact) and unit keys [items, query_dim]."""
        final = self.states(core, raws, modalities)
        items, slots, width = final.shape
        own = torch.arange(items * slots, device=final.device) // slots
        visible = own[None] == torch.arange(items, device=final.device)[:, None]
        # Keep each record's readout palette local to that item. Passing the
        # per-event track IDs lets HindsightRealizer form one routing context
        # for each item's C states instead of pooling unrelated records into a
        # single batch-wide source context.
        metadata_track = core.config.max_tracks + RESERVED_TRACKS.index(METADATA_TRACK)
        event_track_ids = torch.full(
            (items,), metadata_track, device=final.device, dtype=torch.long
        )
        return (
            self.value(
                final.reshape(items * slots, width),
                visible,
                event_track_ids=event_track_ids,
            ),
            self.key(final),
        )
