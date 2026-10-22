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
from megatoken.memory.transducer import Lookup, SearchAgenda, TransducerBank
from megatoken.memory.transducer_packing import pack_fp4, straight_through

from .adapters import ADAPTER_DROPOUT, AdapterMixture
from .config import KNOWLEDGE_TRACK, METADATA_TRACK, RESERVED_TRACKS, TRANSDUCER_QUERY_TRACK
from .core import MODALITY_IDS, ROLE_IDS, FacetedCore
from .multitrack import MultiTrackState, ObservedEvent
from .state import KVHistory, RealizationContext

QUERY_SPACE = "transducer-v10-query"
METADATA_FIELDS = ("request", "task", "topic", "language", "modality", "date", "version")
# Results that arrive from outside the model close the turn that asked for them.
SOURCE_ROLES = frozenset(ROLE_IDS[name] for name in ("tool", "document", "environment"))


def metadata_payload(metadata: dict[str, object]) -> bytes:
    """The compact metadata prefix; missing fields are explicit nulls, not a mode."""
    normalized = {key: metadata.get(key) for key in METADATA_FIELDS}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()


def real_boundary(roles: Sequence[int], complete: Sequence[bool]) -> bool:
    """Whether a lookup is due after one ordinary meridian, from its roles alone.

    A completed assistant turn (every event assistant, each the last part of its
    message) or the arrival of a tool, document or environment result. Training
    and serving both ask this, so they plan the same lookups.
    """
    turn = all(
        role == ROLE_IDS["assistant"] and done for role, done in zip(roles, complete, strict=True)
    )
    return turn or any(role in SOURCE_ROLES for role in roles)


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
    """Mix wide values first; route 64 rows through C's shared U; gather 32 facets."""

    def __init__(self, core: FacetedCore, bank: TransducerBank) -> None:
        super().__init__()
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
        condition = self.condition(torch.cat((query, location)).to(dtype))
        rows = value.to(dtype).reshape(1, self.row_count, self.width)
        rows = self.row_adapter(rows + self.row_identifiers[None]) + condition
        # The shared U bank is registered once, on the core; this only routes through it.
        context = self.route_input(condition)[None, None].expand(1, self.facets, -1)
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

    def forward(self, states: Tensor, visible: Tensor) -> Tensor:
        """``states`` [slots, width]; ``visible`` [lookups, slots] -> [lookups, value_dim].

        A lookup whose row of ``visible`` is empty reads the sink alone: its
        group lies past the end of the sequence.
        """
        if states.ndim != 2 or states.shape[1] != self.width:
            raise ValueError("Hindsight states must be [slots, native width]")
        if visible.ndim != 2 or visible.shape[1] != states.shape[0] or visible.dtype != torch.bool:
            raise ValueError("Hindsight visibility must be [lookups, slots] booleans")
        dtype = self.queries.dtype
        normalized = self.norm(states.to(dtype))
        keys = torch.cat((self.key(normalized), self.sink_key[None]), dim=0)
        values = torch.cat((self.value(normalized), self.sink_value[None]), dim=0)
        logits = (self.queries @ keys.T).float() / math.sqrt(self.width)  # [rows, slots + 1]
        allowed = torch.cat((visible, visible.new_ones(visible.shape[0], 1)), dim=1)
        logits = logits[None].masked_fill(~allowed[:, None], float("-inf"))
        read = torch.softmax(logits, dim=-1).to(dtype) @ values  # [lookups, rows, width]
        raw = self.output(self.output_norm(read + self.queries[None])).float()
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
        values = self.hindsight_realizer(final, visible)
        return HindsightValues(values, ~visible.any(dim=1))

    # -------------------------------------------------------------- lookups

    def begin(
        self, multi, state: MultiTrackState, payload: bytes, *, encoder_grad: bool = False
    ) -> tuple[MultiTrackState, SearchAgenda]:
        """Commit the metadata prefix as the first ordinary event; seed the agenda.

        ``encoder_grad`` is as for any observed event: only bootstrap training
        lets E learn from what it observes.
        """
        if state.batch_size != 1:
            raise ValueError("A transducer session is one independent sequence")
        state = multi.tracks.observe_group(
            state,
            [ObservedEvent(METADATA_TRACK, "text", payload, role_id=ROLE_IDS["system"])],
            encoder_grad=encoder_grad,
        )
        prefix = self.prefix_head(state.final[0, 0].mean(dim=0).to(self.prefix_head.weight.dtype))
        return state, self.bank.initial_agenda(prefix.float())

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
    ) -> tuple[MultiTrackState, Lookup | None]:
        """One serialized query and knowledge commit; the agenda changes only on success.

        The query is P's own reparameterized sample on the reserved query
        track, realized by an embedding R. A cancellation or a failure leaves
        both the agenda and the returned state untouched.
        """
        if state.batch_size != 1 or state.device != self.bank.device:
            raise ValueError("Native lookup is one session on the bank/model device")
        if agenda.lookups >= self.bank.geometry.max_lookups or (cancelled and cancelled()):
            return state, None
        working = agenda.copy()
        mean = multi.tracks.predict_group(state, tracks=(TRANSDUCER_QUERY_TRACK,), effort=effort)
        # The policy's own reparameterized draw; ``generator`` drives only exploration.
        selected = multi.generator.facet_distribution(mean).sample() if training else mean
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
        if cancelled and cancelled():
            return state, None
        self.bank.commit(working, found)
        agenda.publish(working)
        return committed, found
