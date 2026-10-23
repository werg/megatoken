"""Native C/P/R integration of the continuous-state transducer.

Only ordinary events drive the boundary scheduler. A retrieval commits one P
query event followed by one observed knowledge event, both through native
C/KV. Hindsight runs in a separate, training-only C pass whose activations are
never inserted into a causal state or exposed by the serving interface.

The transducer is registered once on the generator (``model.transducer``); it
holds the bank's dense controls and its own heads, never a second reference
to the core's parameters.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from megatoken.generation.embedding import EmbeddingRealizer, EmbeddingSpec
from megatoken.memory.transducer import Lookup, Mounted, SearchAgenda, TransducerBank
from megatoken.memory.transducer_packing import pack_fp4, straight_through
from megatoken.retrieval.session import SourceSession

from .adapters import ADAPTER_DROPOUT, AdapterMixture
from .config import KNOWLEDGE_TRACK, METADATA_TRACK, RESERVED_TRACKS, TRANSDUCER_QUERY_TRACK
from .core import MODALITY_IDS, ROLE_IDS, FacetedCore
from .expert_pools import expert_scope
from .ingest import IngestedRows, IngestHeads, IngestItem
from .multitrack import MultiTrackState, ObservedEvent
from .sources import read_sources
from .state import KVHistory, RealizationContext

QUERY_SPACE = "transducer-v10-query"
METADATA_FIELDS = ("request", "task", "topic", "language", "modality", "date", "version")
# Results that arrive from outside the model close the turn that asked for them.
SOURCE_ROLES = frozenset(ROLE_IDS[name] for name in ("tool", "document", "environment"))


def metadata_payload(metadata: dict[str, object]) -> bytes:
    """The compact metadata prefix; missing fields are explicit nulls, not a mode."""
    normalized = {key: metadata.get(key) for key in METADATA_FIELDS}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()


def completed_turn(roles: Sequence[int], complete: Sequence[bool], role: str) -> bool:
    """Whether one ordinary meridian is a finished message of ``role``.

    Every event has that role and is the last part of its message.
    """
    return all(r == ROLE_IDS[role] and done for r, done in zip(roles, complete, strict=True))


def real_boundary(roles: Sequence[int], complete: Sequence[bool]) -> bool:
    """Whether a lookup is due after one ordinary meridian, from its roles alone.

    A completed user message or assistant turn (every event has that role, each the
    last part of its message), or the arrival of a tool, document or environment
    result. A user message is finished when the question is whole, so the lookup that
    follows it sees what was asked. Training and serving both ask this, so they plan
    the same lookups.
    """
    return (
        completed_turn(roles, complete, "user")
        or completed_turn(roles, complete, "assistant")
        or any(role in SOURCE_ROLES for role in roles)
    )


def event_attention_mask(*, groups: Tensor, mode: str) -> Tensor:
    """A [batch, query-event, key-event] mask from each event's retrieval group.

    ``full`` sees everything; ``group-inverse`` sees the event's own retrieval
    group and every later one. Sink slots are appended by attention itself, so
    no row is ever empty.
    """
    if groups.ndim != 2 or groups.dtype != torch.long:
        raise ValueError("Every event needs an int64 [batch, events] retrieval group")
    if bool((groups < 0).any()):
        raise ValueError("Event chronology cannot be negative")
    if mode == "full":
        return torch.ones((*groups.shape, groups.shape[1]), device=groups.device, dtype=torch.bool)
    if mode == "group-inverse":
        return groups[:, None, :] >= groups[:, :, None]
    raise ValueError(f"Unknown track attention mode {mode!r}")


class WideKnowledgeReadout(nn.Module):
    """Mix wide values, route rows through the core's shared U bank, then gather facets."""

    def __init__(self, core: FacetedCore, bank: TransducerBank) -> None:
        super().__init__()
        object.__setattr__(self, "_expert_pools", core.expert_pools)
        self.geometry = bank.geometry
        self.width = core.config.facet_dim
        self.facets = core.config.n_facets
        if self.geometry.value_dim % self.width:
            raise ValueError("Wide values must divide into native facet-width rows")
        self.row_count = self.geometry.value_dim // self.width
        self.row_identifiers = nn.Parameter(torch.randn(self.row_count, self.width) * 0.02)
        self.row_adapter = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )
        self.facet_queries = nn.Parameter(torch.randn(self.facets, self.width) * 0.02)
        self.key_norm = nn.LayerNorm(self.width)
        self.sink_key = nn.Parameter(torch.zeros(self.width))
        self.sink_value = nn.Parameter(torch.zeros(self.width))
        self.condition = nn.Linear(
            self.geometry.query_dim + self.geometry.location_dim, self.width, bias=False
        )
        self.route_input = nn.Linear(self.width, self.width)

    def forward(
        self, value: Tensor, query: Tensor, location: Tensor, *, core: FacetedCore
    ) -> Tensor:
        """[value], [query], [location] -> one [1, facets, width] knowledge block."""
        g = self.geometry
        if (
            value.shape != (g.value_dim,)
            or query.shape != (g.query_dim,)
            or location.shape != (g.location_dim,)
        ):
            raise ValueError("Knowledge readout requires complete value/query/location vectors")
        dtype = self.row_identifiers.dtype
        condition = self.condition(torch.cat((query, location)).to(dtype)[None])[0]
        rows = value.to(dtype).reshape(1, self.row_count, self.width)
        rows = self.row_adapter(rows + self.row_identifiers[None]) + condition
        # The shared U bank is registered once, on the core; this only routes through it.
        routed_condition = self.route_input(condition[None])[0]
        context = routed_condition[None, None].expand(1, self.facets, -1)
        routed = core.apply_u(rows, RealizationContext(context), level=1).to(dtype)
        keys = torch.cat((self.key_norm(routed), self.sink_key[None, None]), dim=1)
        values = torch.cat((routed, self.sink_value[None, None]), dim=1)
        queries = self.facet_queries[None] + condition
        weights = torch.softmax(
            (queries @ keys.transpose(1, 2)).float() / math.sqrt(self.width), dim=-1
        )
        return (weights.to(values.dtype) @ values).to(core.initial_facets.dtype)


class HindsightRealizer(nn.Module):
    """Future C states of one group -> 64 rows -> a value in the bank's exact FP4 format."""

    def __init__(self, core: FacetedCore, bank: TransducerBank) -> None:
        super().__init__()
        object.__setattr__(self, "_expert_pools", core.expert_pools)
        self.width = core.config.facet_dim
        self.value_dim = bank.geometry.value_dim
        if self.value_dim % self.width:
            raise ValueError("Hindsight rows must match the wide value geometry")
        self.rows = self.value_dim // self.width
        self.queries = nn.Parameter(torch.randn(self.rows, self.width) * 0.02)
        self.norm = nn.LayerNorm(self.width)
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)
        self.sink_key = nn.Parameter(torch.zeros(self.width))
        self.sink_value = nn.Parameter(torch.zeros(self.width))
        self.output_norm = nn.LayerNorm(self.width)
        self.output = nn.Linear(self.width, self.width, bias=False)

    def forward(
        self, states: Tensor, visible: Tensor, *, event_track_ids: Tensor | None = None
    ) -> Tensor:
        """``states`` [slots, width]; ``visible`` [lookups, slots] -> [lookups, value_dim].

        A lookup whose row of ``visible`` is empty reads the sink alone: its
        group lies past the end of the sequence.
        """
        if states.ndim != 2 or states.shape[1] != self.width:
            raise ValueError("Hindsight states must be [slots, native width]")
        if visible.ndim != 2 or visible.shape[1] != states.shape[0] or visible.dtype != torch.bool:
            raise ValueError("Hindsight visibility must be [lookups, slots] booleans")
        if visible.shape[0] == 0:
            return states.new_empty((0, self.value_dim))
        dtype = self.queries.dtype
        pools = self._expert_pools
        source_tracks = None
        if event_track_ids is None:
            state_context = states.float().mean(0, keepdim=True)
        else:
            if event_track_ids.ndim != 1 or event_track_ids.numel() < 1:
                raise ValueError("Hindsight event tracks must be one ID per source event")
            if states.shape[0] % event_track_ids.numel():
                raise ValueError("Hindsight source rows must divide evenly across events")
            facets_per_event = states.shape[0] // event_track_ids.numel()
            state_context = (
                states.float()
                .reshape(event_track_ids.numel(), facets_per_event, self.width)
                .mean(1)
            )
            source_tracks = event_track_ids.to(device=states.device, dtype=torch.long)
        # Route each event's source states together, keyed by its stable track;
        # route each hindsight query's output rows separately below.
        with pools.routing_epoch(
            state_context,
            track_ids=source_tracks,
            role="r",
            adapter_weights=None,
        ):
            normalized = self.norm(states.to(dtype))
            keys = torch.cat((self.key(normalized), self.sink_key[None]), dim=0)
            values = torch.cat((self.value(normalized), self.sink_value[None]), dim=0)
        logits = (self.queries @ keys.T).float() / math.sqrt(self.width)  # [rows, slots + 1]
        allowed = torch.cat((visible, visible.new_ones(visible.shape[0], 1)), dim=1)
        logits = logits[None].masked_fill(~allowed[:, None], float("-inf"))
        read = torch.softmax(logits, dim=-1).to(dtype) @ values  # [lookups, rows, width]
        output_rows = read + self.queries[None]
        with pools.routing_epoch(output_rows.float().mean(1), role="other", adapter_weights=None):
            raw = self.output(self.output_norm(output_rows)).float()
        raw = raw.reshape(visible.shape[0], self.value_dim)
        return straight_through(raw, pack_fp4(raw.detach()).decode(self.value_dim))


@dataclass(frozen=True)
class HindsightValues:
    """One h_t per lookup, with whether it came from the sink alone."""

    values: Tensor  # [lookups, value_dim]
    sink_only: Tensor  # [lookups] bool


@dataclass(frozen=True)
class OrdinaryEvents:
    """The recorded ordinary (observed or generated) events of one sequence."""

    raw: Tensor  # [1, events, facets, width]
    roles: Tensor  # [1, events]
    modalities: tuple[str, ...]
    tracks: tuple[str, ...]
    meridians: Tensor  # [1, events] native positions
    groups: Tensor  # [1, events] retrieval group of each event


class NativeTransducer(nn.Module):
    """Query/readout/hindsight heads and the bank, around one shared core owner."""

    def __init__(
        self, core: FacetedCore, bank: TransducerBank, *, training_heads: bool = True
    ) -> None:
        super().__init__()
        # Not a submodule: the generator already owns and optimizes the core.
        object.__setattr__(self, "_core", core)
        self.bank = bank
        c = core.config
        g = bank.geometry
        self.query_realizer = EmbeddingRealizer(c.n_facets, c.facet_dim, g.query_dim)
        self.prefix_head = nn.Linear(c.facet_dim, g.query_dim)
        self.readout = WideKnowledgeReadout(core, bank)
        # Training-only: a served model neither ships nor counts it.
        self.hindsight_realizer = HindsightRealizer(core, bank) if training_heads else None
        # Written from one item alone, so served too: it adds files to a base while serving.
        self.ingest = IngestHeads(HindsightRealizer(core, bank), core, g.query_dim)

    @property
    def core(self) -> FacetedCore:
        return self._core

    # ------------------------------------------------------------ hindsight

    def hindsight_pass(
        self, events: OrdinaryEvents, reads: Sequence[int | None], *, mode: str
    ) -> HindsightValues:
        """One training-only C pass over ordinary events; h_t for every lookup.

        ``reads[t]`` is the retrieval group lookup t reads (None: past the end).
        Nothing here touches KVHistory, a causal state or the serving path.
        """
        if not self.training or self.hindsight_realizer is None:
            raise RuntimeError("Hindsight may not execute on the serving path")
        if mode not in ("full", "group-inverse"):
            raise ValueError("Hindsight reads a full or group-inverse pass")
        c = self.core.config
        raw = events.raw
        if raw.ndim != 4 or raw.shape[0] != 1 or raw.shape[2:] != (c.n_facets, c.facet_dim):
            raise ValueError("Hindsight requires one sequence of native event blocks")
        count = raw.shape[1]
        if not count:
            raise ValueError("No ordinary events for hindsight training")
        for tensor in (events.roles, events.meridians, events.groups):
            if tensor.shape != (1, count):
                raise ValueError("Every ordinary event needs role, meridian and group")
        if len(events.modalities) != count or len(events.tracks) != count:
            raise ValueError("Every ordinary event needs a modality and a track")
        if any(track in RESERVED_TRACKS for track in events.tracks):
            raise ValueError("Query, knowledge, hindsight and metadata events are not ordinary")
        registry: list[str] = []
        for track in events.tracks:
            if track not in registry:
                registry.append(track)
        if len(registry) > c.max_tracks:
            raise ValueError("Too many tracks for configured session capacity")
        device = raw.device
        track_ids = torch.tensor([[registry.index(t) for t in events.tracks]], device=device)
        modality_ids = torch.tensor([[MODALITY_IDS[m] for m in events.modalities]], device=device)
        mask = event_attention_mask(groups=events.groups, mode=mode)
        facet_mask = mask.repeat_interleave(c.n_facets, 1).repeat_interleave(c.n_facets, 2)
        empty = raw.new_empty(1, c.kv_heads, 0, c.attention_head_dim)
        with expert_scope(None):
            final = self.core.central(
                raw.flatten(1, 2),
                KVHistory(empty, empty),
                events.roles,
                events.meridians.repeat_interleave(c.n_facets, dim=1),
                track_ids=track_ids,
                modality_ids=modality_ids,
                history_track_ids=track_ids.new_empty(1, 0),
                attention_mask=facet_mask,
            )[0]
        slot_groups = events.groups[0].repeat_interleave(c.n_facets)
        read = torch.tensor([-1 if r is None else r for r in reads], device=device)
        visible = (slot_groups[None] == read[:, None]) & (read[:, None] >= 0)
        # Future evidence must never inherit an adapter mixture from the live
        # session that happened to call this training-only path.
        event_tracks = torch.tensor(
            [registry.index(track) for track in events.tracks], device=device, dtype=torch.long
        )
        with expert_scope(None):
            values = self.hindsight_realizer(final, visible, event_track_ids=event_tracks)
        return HindsightValues(values, ~visible.any(dim=1))

    # -------------------------------------------------------------- lookups

    def begin(
        self,
        multi,
        state: MultiTrackState,
        payload: bytes,
        *,
        encoder_grad: bool = False,
        mounted: tuple[Mounted, ...] | None = None,
    ) -> tuple[MultiTrackState, SearchAgenda]:
        """Commit the metadata prefix as the first ordinary event; seed the agenda.

        ``encoder_grad`` is as for any observed event: only bootstrap training
        lets E learn from what it observes. ``mounted`` are the knowledge bases
        the session searches (default: the trained bank alone).
        """
        if state.batch_size != 1:
            raise ValueError("A transducer session is one independent sequence")
        state = multi.tracks.observe_group(
            state,
            [ObservedEvent(METADATA_TRACK, "text", payload, role_id=ROLE_IDS["system"])],
            encoder_grad=encoder_grad,
        )
        return state, self.bank.initial_agenda(self._prefix(state), mounted)

    def _prefix(self, state: MultiTrackState) -> Tensor:
        """The query-width vector a committed metadata event maps to under E."""
        pooled = state.final[0, 0].mean(dim=0).to(self.prefix_head.weight.dtype)
        weights = self._event_adapter_weights("e", state, METADATA_TRACK)
        track = self.core.config.max_tracks + RESERVED_TRACKS.index(METADATA_TRACK)
        track_ids = torch.full((state.batch_size,), track, device=state.device, dtype=torch.long)
        with (
            self.core.expert_pools.routing_epoch(
                pooled[None], track_ids=track_ids, role="e", adapter_weights=weights
            ),
            expert_scope(weights),
        ):
            return self.prefix_head(pooled[None])[0].float()

    def _event_adapter_weights(
        self, owner: str, state: MultiTrackState, track: str
    ) -> Tensor | None:
        """One committed owner's mixture for one reserved transducer track."""
        if state.adapters is None:
            return None
        track_id = self.core.config.max_tracks + RESERVED_TRACKS.index(track)
        ids = torch.full((state.batch_size, 1), track_id, device=state.device, dtype=torch.long)
        plan = self.core.adapters.plan(owner, state.adapters, ids)
        return None if plan is None else plan.weights[0, :, 0]

    # --------------------------------------------------------------- ingest

    @torch.no_grad()
    def ingest_rows(self, multi, items: Sequence[IngestItem]) -> IngestedRows:
        """Entries for ``items`` read alone: each item's value, key, departure and destination.

        Every part of a document must be in the call, in any order. The first
        part (and any standalone item) departs from where its metadata starts a
        session, through the same metadata path a session's prefix takes; part
        ``i`` departs from part ``i - 1``'s destination, so reading on is one
        more step. A destination is the content location of the value, as for a
        hindsight write-out. The edge's bias is zero, the same as a write-out into an
        empty bank, so a lookup scores it by key and location alone and the base's
        blend weight is the only prior on it.
        """
        if not items:
            raise ValueError("Nothing to ingest")
        device = self.bank.device
        raws = torch.stack([item.raw for item in items]).to(device)
        # Ingested library entries are written outside any live session scope.
        ingest_tracks = torch.full(
            (len(items),),
            self.core.config.max_tracks + RESERVED_TRACKS.index(METADATA_TRACK),
            device=device,
            dtype=torch.long,
        )
        with (
            self.core.expert_pools.routing_epoch(
                raws.mean(1), track_ids=ingest_tracks, role="r", adapter_weights=None
            ),
            expert_scope(None),
        ):
            values, keys = self.ingest(self.core, raws, [item.modality for item in items])
        destinations = self.bank.content_location(values)
        starts: dict[bytes, Tensor] = {}
        previous = {
            (item.document, item.part): row for row, item in enumerate(items) if item.document
        }
        departures = []
        for item in items:
            if item.part:
                if (item.document, item.part - 1) not in previous:
                    raise ValueError(
                        f"Part {item.part - 1} of {item.document!r} is not in this call"
                    )
                departures.append(destinations[previous[(item.document, item.part - 1)]])
                continue
            if item.metadata not in starts:
                starts[item.metadata] = self.metadata_start(multi, item.metadata)
            departures.append(starts[item.metadata])
        return IngestedRows(torch.stack(departures), keys, values, destinations)

    def metadata_start(self, multi, payload: bytes) -> Tensor:
        """Where a session that opens with this metadata starts its search."""
        state = multi.tracks.initial_state(1, device=self.bank.device)
        state = multi.tracks.observe_group(
            state,
            [ObservedEvent(METADATA_TRACK, "text", payload, role_id=ROLE_IDS["system"])],
            encoder_grad=False,
        )
        return self.bank.metadata_start(self._prefix(state))

    def mixture(
        self, found: Lookup, *, overlay: float, training: bool, generator: torch.Generator | None
    ) -> AdapterMixture:
        """The adapter mixture one lookup asks for, from the value its knowledge event reads.

        Its strength is one minus the sink's share of that value. A hindsight
        overlay is knowledge too, so only the bank's own part can be the sink.
        Training drops a lookup's whole mixture with probability ``ADAPTER_DROPOUT``.
        """
        strength = 1 - (1 - overlay) * found.sink_weight
        if training and float(torch.rand((), generator=generator)) < ADAPTER_DROPOUT:
            strength = strength * 0
        return self.core.adapters.select(found.value, strength)

    def boundary(
        self,
        multi,
        state: MultiTrackState,
        agenda: SearchAgenda,
        *,
        training: bool = False,
        hindsight: Tensor | None = None,
        overlay: float = 0.0,
        generator: torch.Generator | None = None,
        train_rows: bool | None = None,
        effort: int = 6,
        cancelled: Callable[[], bool] | None = None,
        sources: SourceSession | None = None,
    ) -> tuple[MultiTrackState, Lookup | None]:
        """One serialized query and knowledge commit; the agenda changes only on success.

        The query is P's own reparameterized sample on the reserved query
        track, realized by an embedding R. A cancellation or a failure leaves
        both the agenda and the returned state untouched.

        With ``sources``, the same query block is read by the embedding head's
        query role and the mounted collections are searched; the items chosen are
        observed after the knowledge event (``model.sources``).
        """
        if state.batch_size != 1 or state.device != self.bank.device:
            raise ValueError("Native lookup is one session on the bank/model device")
        if agenda.lookups >= self.bank.geometry.max_lookups or (cancelled and cancelled()):
            return state, None
        working = agenda.copy()
        mean = multi.tracks.predict_group(state, tracks=(TRANSDUCER_QUERY_TRACK,), effort=effort)
        # The policy's own reparameterized draw; ``generator`` drives only exploration.
        query_ids = self.core.config.max_tracks + RESERVED_TRACKS.index(TRANSDUCER_QUERY_TRACK)
        query_tracks = torch.full(
            (state.batch_size, 1), query_ids, device=state.device, dtype=torch.long
        )
        if training:
            prior_plan = self.core.adapters.plan("p", state.adapters, query_tracks)
            prior_weights = None if prior_plan is None else prior_plan.weights[0, :, 0]
            with (
                self.core.expert_pools.routing_epoch(
                    mean[:, 0].mean(1),
                    track_ids=query_tracks[:, 0],
                    role="p",
                    stage=self.core.config.prediction_layers,
                    adapter_weights=prior_weights,
                ),
                expert_scope(prior_weights),
            ):
                selected = multi.generator.facet_distribution(mean).sample()
        else:
            selected = mean
        query_plan = self.core.adapters.plan("r", state.adapters, query_tracks)
        query_weights = None if query_plan is None else query_plan.weights[0, :, 0]
        with (
            self.core.expert_pools.routing_epoch(
                selected[:, 0].mean(1),
                track_ids=query_tracks[:, 0],
                role="r",
                adapter_weights=query_weights,
            ),
            expert_scope(query_weights),
        ):
            query = self.query_realizer(
                selected[:, 0], EmbeddingSpec(QUERY_SPACE, "query", self.bank.geometry.query_dim)
            )[0]
        assistant = torch.full((1, 1), ROLE_IDS["assistant"], dtype=torch.long, device=state.device)
        queried = multi.tracks.advance_group(
            state, selected, tracks=(TRANSDUCER_QUERY_TRACK,), role_ids=assistant
        )
        found = self.bank.lookup(
            query,
            working,
            training=training,
            hindsight=hindsight,
            overlay=overlay,
            generator=generator,
            train_rows=train_rows,
        )
        if found is None:
            return state, None
        readout_weights = self._event_adapter_weights("r", state, KNOWLEDGE_TRACK)
        knowledge_track = self.core.config.max_tracks + RESERVED_TRACKS.index(KNOWLEDGE_TRACK)
        knowledge_tracks = torch.full(
            (state.batch_size,), knowledge_track, device=state.device, dtype=torch.long
        )
        with (
            self.core.expert_pools.routing_epoch(
                query[None] if query.ndim == 1 else query,
                track_ids=knowledge_tracks,
                role="r",
                adapter_weights=readout_weights,
            ),
            expert_scope(readout_weights),
        ):
            block = self.readout(found.value, query, found.landing, core=self.core)
        # From the knowledge event on, every meridian runs under this lookup's mixture.
        queried = replace(
            queried,
            adapters=self.mixture(found, overlay=overlay, training=training, generator=generator),
        )
        committed = multi.tracks.advance_group(
            queried,
            block[:, None],
            tracks=(KNOWLEDGE_TRACK,),
            role_ids=torch.full_like(assistant, ROLE_IDS["document"]),
        )
        read = None
        if sources is not None:
            read = read_sources(
                multi,
                committed,
                sources,
                selected[:, 0],
                lookup=agenda.lookups,
                cancelled=cancelled,
            )
            if read is None:
                return state, None
            committed = read[0]
        if cancelled and cancelled():
            return state, None
        self.bank.commit(working, found)
        agenda.publish(working)
        if read is not None:
            sources.commit(read[1])
        return committed, found
