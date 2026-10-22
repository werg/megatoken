"""The source path of a lookup: the embedding head's query, the search, observing what it finds.

At a lookup boundary the query block P sampled on the query track also goes
through the embedding head's query role, giving a 768-wide query in the teacher's
space. The mounted collections are searched with it, and the items chosen are
observed, one meridian each, right after the knowledge event: as ordinary events
of their own medium on the knowledge track, with the role of a document. Text is
read as its bytes, media and long text as their stored teacher features. Training
and serving run this same code; no teacher runs here.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from megatoken.generation.embedding import EmbeddingSpec
from megatoken.retrieval.session import Retrieved, Search, SourceSession
from megatoken.teacher import EMBEDDING_WIDTH

from .config import KNOWLEDGE_TRACK
from .core import ROLE_IDS
from .multitrack import MultiTrackState, ObservedEvent


def source_query(multi, raw: Tensor) -> Tensor:
    """The sampled [1, facets, width] query block -> the [768] unit query (differentiable)."""
    spec = EmbeddingSpec(multi.generator.config.media_extractor, "query", EMBEDDING_WIDTH)
    return multi.generator.embedding_realizer(raw, spec)[0]


def source_event(sources: SourceSession, entry: Retrieved, device: torch.device) -> ObservedEvent:
    """The event an item is observed as, in its own medium."""
    item, role = entry.item, ROLE_IDS["document"]
    if item.text is not None:
        return ObservedEvent(KNOWLEDGE_TRACK, "text", item.text, role_id=role)
    features = torch.from_numpy(sources.collections[entry.collection].features(item))
    return ObservedEvent(
        KNOWLEDGE_TRACK,
        item.modality,
        features.to(device)[None],
        role_id=role,
        extractor=item.record.source.extractor,
    )


def read_sources(
    multi,
    state: MultiTrackState,
    sources: SourceSession,
    raw: Tensor,
    *,
    lookup: int,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[MultiTrackState, list[Retrieved]] | None:
    """Search with the query read from ``raw`` and observe what is chosen.

    Returns the state after the observations and the items chosen, or None if
    cancelled midway. Nothing is recorded in ``sources`` except the latest search,
    which training reads for its losses; the caller commits the items with the lookup.
    """
    query = source_query(multi, raw)
    candidates = sources.candidates(query.detach().float().cpu().numpy())
    sources.last = Search(query, candidates)
    chosen = sources.choose(candidates, lookup)
    for entry in chosen:
        if cancelled is not None and cancelled():
            return None
        event = source_event(sources, entry, state.device)
        state = multi.tracks.observe_group(state, [event], encoder_grad=False)
    return state, chosen
