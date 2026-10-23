"""Knowledge bases built while training, for the reading-from-a-base stream.

Episodes of this stream hold only a question and an answer; their reference
items (document pieces, images, clips) are declared for a base instead of
being in the context, so the model can answer only by finding them. The
trainer reads each episode's items alone with the *current* ingest heads, writes
them into a small base of their own and mounts it beside the trained base.

The schedule is deliberately simple. An episode's base is built when the episode
is first drawn. The bases of the last ``capacity`` episodes stay mounted as well,
so the right items must be found among others (and one episode's question
cannot be answered from another's items by accident of timing alone). Every
``refresh_every`` optimizer steps all of them are dropped and rebuilt on demand,
so no entry is older than the heads that wrote it by more than that. These bases
are fixed once written, belong to this process, and are neither checkpointed nor
synced: only the trained base is.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from megatoken.data.knowledge import declared_items, declared_knowledge, episode_metadata
from megatoken.data.relations import local_relations
from megatoken.kb import add_items
from megatoken.memory.knowledge import KnowledgeBase
from megatoken.memory.transducer import Mounted

CAPACITY = 8  # episodes whose bases are mounted together
REFRESH_EVERY = 1024  # optimizer steps after which every entry is rewritten


class ReadingBase:
    """The ingested bases of the episodes recently drawn, rebuilt every ``refresh_every`` steps."""

    def __init__(
        self, native: Any, *, capacity: int = CAPACITY, refresh_every: int = REFRESH_EVERY
    ) -> None:
        if capacity < 1 or refresh_every < 1:
            raise ValueError("A reading base keeps at least one episode and refreshes sometime")
        self.native, self.capacity, self.refresh_every = native, capacity, refresh_every
        self.segments: OrderedDict[str, KnowledgeBase] = OrderedDict()
        self.built: int | None = None  # the step the current entries were written at

    def mount(self, multi: Any, episode: Any, assets: Any, step: int) -> tuple[Mounted, ...] | None:
        """The bases to search for this episode: the trained base and the episodes' ingested ones.

        ``None`` when the episode declares no items (it searches the trained base alone).
        """
        if not declared_items(episode):
            return None
        if self.built is None or step - self.built >= self.refresh_every:
            self.segments.clear()
            self.built = step
        if episode.episode_id in self.segments:
            self.segments.move_to_end(episode.episode_id)
        else:
            self.segments[episode.episode_id] = self._ingest(multi, episode, assets)
            while len(self.segments) > self.capacity:
                self.segments.popitem(last=False)
        return self.native.bank.mount(
            {f"reading:{name}": (base, 1.0) for name, base in self.segments.items()}
        )

    def _ingest(self, multi: Any, episode: Any, assets: Any) -> KnowledgeBase:
        bank = self.native.bank
        base = KnowledgeBase(
            bank.geometry, {}, name=episode.episode_id, writable=True, device=bank.device
        )
        items = declared_knowledge(episode, assets, episode_metadata(episode))
        names = {item.source for item in items}
        links = [(f"{episode.episode_id}:{row['left']}", f"{episode.episode_id}:{row['right']}")
                 for row in local_relations(episode)]
        # Only declared references enter normal search. Future episode targets stay out.
        links = [(a, b) for a, b in links if a in names and b in names]
        add_items(base, multi.generator, multi, items, relations=links)
        base.writable = False  # fixed from here on
        return base
