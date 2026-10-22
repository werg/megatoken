"""Native C/P/R integration of the continuous-state transducer.

Only ordinary events drive the boundary scheduler. A retrieval commits one P
query event followed by one observed knowledge event, both using native C/KV.
Hindsight runs in a separate, training-only C pass whose activations are never
inserted into a causal state or exposed by the serving interface.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from megatoken.memory.transducer import Lookup, SearchAgenda, TransducerBank
from megatoken.memory.transducer_packing import pack_fp4, straight_through, unit

from .config import KNOWLEDGE_TRACK, RETRIEVAL_QUERY_TRACK
from .core import ROLE_IDS, FacetedCore
from .multitrack import MultiTrackCore, MultiTrackState, ObservedEvent
from .state import KVHistory, RealizationContext


def event_attention_mask(
    *,
    groups: Tensor,
    meridians: Tensor,
    tracks: Tensor,
    mode: str = "causal",
) -> Tensor:
    """A native [batch, query-event, key-event] mask from recorded event metadata.

    Track identity is validated but does not restrict cross-track observation.
    Under inverse masking, each group sees itself and later groups, never
    earlier ones. C/P sink slots are appended after this mask, not stored.
    """
    if groups.shape != meridians.shape or tracks.shape != groups.shape:
        raise ValueError("Every event needs group, meridian and track metadata")
    if groups.ndim != 2 or groups.dtype != torch.long or (
        meridians.dtype != torch.long or tracks.dtype != torch.long
    ):
        raise ValueError("Event metadata must be [batch, events] int64 tensors")
    if bool((groups < 0).any()) or bool((meridians < 0).any()):
        raise ValueError("Event chronology cannot be negative")
    if mode == "full":
        return torch.ones(
            (*groups.shape, groups.shape[1]), device=groups.device, dtype=torch.bool
        )
    if mode == "group-inverse":
        return groups[:, None, :] >= groups[:, :, None]
    if mode == "causal":
        return meridians[:, None, :] <= meridians[:, :, None]
    raise ValueError(f"Unknown track attention mode {mode!r}")


class WideKnowledgeReadout(nn.Module):
    """Mix wide values before readout; run all rows through C's existing shared U."""

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
        self.sink_key = nn.Parameter(torch.zeros(self.width))
        self.sink_value = nn.Parameter(torch.zeros(self.width))
        self.condition = nn.Linear(
            bank.geometry.query_dim + bank.geometry.location_dim,
            self.width,
            bias=False,
        )
        self.route_input = nn.Linear(self.width, self.width)

    def forward(
        self, value: Tensor, query: Tensor, location: Tensor, *, core: FacetedCore
    ) -> Tensor:
        if value.shape != (self.geometry.value_dim,) or query.shape != (
            self.geometry.query_dim,
        ) or location.shape != (self.geometry.location_dim,):
            raise ValueError("Knowledge readout requires complete value/query/state vectors")
        condition = self.condition(torch.cat((query, location)).float())
        rows = value.float().reshape(1, self.row_count, self.width)
        rows = self.row_adapter(rows + self.row_identifiers[None])
        rows = rows + condition[None, None]
        # Shared U is registered ONLY on the original FacetedCore owner.
        context_rows = self.route_input(
            condition[None, None].expand(1, self.facets, -1)
        )
        routed = core.apply_u(rows, RealizationContext(context_rows), level=1)
        sink = self.sink_value[None, None]
        keys = torch.cat((routed, self.sink_key[None, None]), dim=1)
        scores = (
            (self.facet_queries[None] + condition[None, None]) @ keys.transpose(1, 2)
        ) / math.sqrt(self.width)
        weights = torch.softmax(scores.float(), dim=-1)
        values = torch.cat((routed, sink), dim=1)
        block = weights @ values
        return block.to(core.initial_facets.dtype)


class HindsightRealizer(nn.Module):
    """Group-future C states -> 64 rows -> the bank's exact FP4 decoded format."""

    def __init__(self, core: FacetedCore, bank: TransducerBank) -> None:
        super().__init__()
        self.width = core.config.facet_dim
        self.value_dim = bank.geometry.value_dim
        if self.value_dim % self.width:
            raise ValueError("Hindsight rows must match the wide value geometry")
        self.queries = nn.Parameter(torch.randn(self.value_dim // self.width, self.width) * 0.02)
        self.sink_key = nn.Parameter(torch.zeros(self.width))
        self.sink_values = nn.Parameter(torch.zeros(self.value_dim // self.width, self.width))
        self.output = nn.Linear(self.width, self.width, bias=False)

    def forward(self, future: Tensor | None) -> Tensor:
        """Sink-only read is defined when G(t+gap) falls beyond the sequence."""
        count = self.queries.shape[0]
        if future is not None:
            if future.ndim != 2 or future.shape[1] != self.width:
                raise ValueError("Hindsight future must be [events*facets, native width]")
            sources = future
            sink_key = self.sink_key[None].to(sources.dtype)
            keys = torch.cat((sources, sink_key), dim=0)
            logits = (self.queries @ keys.transpose(0, 1)) / math.sqrt(self.width)
            weights = torch.softmax(logits.float(), dim=-1)
            values = torch.cat((sources, self.sink_values.mean(0, keepdim=True)), dim=0)
            result = weights @ values
            result = result + weights[:, -1:] * (
                self.sink_values - self.sink_values.mean(0, keepdim=True)
            )
        else:
            result = self.sink_values
        raw = self.output(result).reshape(self.value_dim)
        packed = pack_fp4(raw.detach())
        return straight_through(raw, packed).reshape(count * self.width)


@dataclass(frozen=True)
class HindsightValues:
    values: tuple[Tensor, ...]
    sink_only: tuple[bool, ...]
    stage: str


class NativeTransducer(nn.Module):
    """A single owner for native query/readout/hindsight and the sparse bank."""

    def __init__(self, core: FacetedCore, bank: TransducerBank) -> None:
        super().__init__()
        self.core = core
        self.bank = bank
        self.tracks = MultiTrackCore(core)
        self.query_head = nn.Linear(core.config.facet_dim, bank.geometry.query_dim)
        self.prefix_head = nn.Linear(core.config.facet_dim, bank.geometry.query_dim)
        self.query_log_std = nn.Parameter(torch.full((bank.geometry.query_dim,), -3.0))
        self.readout = WideKnowledgeReadout(core, bank)
        self.hindsight_realizer = HindsightRealizer(core, bank)

    def hindsight_pass(
        self,
        raw: Tensor,
        roles: Tensor,
        groups: Tensor,
        meridians: Tensor,
        tracks: Tensor,
        *,
        step: int,
    ) -> HindsightValues:
        """Training-only inverse C pass with no state/history publication."""
        if not self.training:
            raise RuntimeError("Hindsight may not execute on the serving path")
        if raw.ndim != 4 or raw.shape[:2] != groups.shape or raw.shape[2:] != (
            self.core.config.n_facets,
            self.core.config.facet_dim,
        ) or roles.shape != groups.shape:
            raise ValueError("Hindsight requires recorded ordinary native event blocks")
        if not raw.shape[1]:
            raise ValueError("No ordinary events for hindsight training")
        stage = "full" if step < 4096 else "gap1" if step < 8192 else "gap2"
        event_mask = event_attention_mask(
            groups=groups,
            meridians=meridians,
            tracks=tracks,
            mode="full" if stage == "full" else "group-inverse",
        )
        n = self.core.config.n_facets
        facet_mask = event_mask.repeat_interleave(n, dim=1).repeat_interleave(n, dim=2)
        empty = KVHistory(
            raw.new_empty(raw.shape[0], self.core.config.kv_heads, 0,
                          self.core.config.attention_head_dim),
            raw.new_empty(raw.shape[0], self.core.config.kv_heads, 0,
                          self.core.config.attention_head_dim),
        )
        # A separate C computation: no call to write() or advance_group().
        final = self.core.central(
            raw.flatten(1, 2),
            empty,
            roles,
            torch.zeros(raw.shape[0], device=raw.device, dtype=torch.long),
            track_ids=tracks,
            history_track_ids=torch.empty(
                raw.shape[0], 0, device=raw.device, dtype=torch.long
            ),
            attention_mask=facet_mask,
        ).reshape_as(raw)
        all_groups = sorted(set(groups[0].tolist()))
        if groups.shape[0] != 1:
            raise ValueError("Hindsight reference processes one causally ordered session")
        produced = []
        sink_only = []
        for current in all_groups:
            to_read = current + (1 if stage == "gap2" else 0)
            available = groups[0] == to_read
            source = final[0, available].reshape(-1, self.core.config.facet_dim) if bool(
                available.any()
            ) else None
            produced.append(self.hindsight_realizer(source))
            sink_only.append(source is None)
        return HindsightValues(tuple(produced), tuple(sink_only), stage)

    def boundary(
        self,
        state: MultiTrackState,
        agenda: SearchAgenda,
        *,
        training: bool = False,
        hindsight: Tensor | None = None,
        overlay: float = 0.0,
        exploration_seed: int = 0,
    ) -> tuple[MultiTrackState, Lookup | None]:
        """One serialized query and one knowledge commit, with atomic agenda publication."""
        if state.batch_size != 1 or state.device != self.bank.device:
            raise ValueError("Native lookup is one session on the bank/model device")
        if agenda.lookups >= self.bank.geometry.max_lookups:
            return state, None
        predicted = self.tracks.predict_group(state, tracks=(RETRIEVAL_QUERY_TRACK,))
        if training:
            selected = predicted + torch.randn_like(predicted) * self.query_log_std.exp()[
                None, None, None
            ]
        else:
            selected = predicted
        query = unit(self.query_head(selected[0, 0]).mean(dim=0))
        queried = self.tracks.advance_group(
            state,
            selected,
            tracks=(RETRIEVAL_QUERY_TRACK,),
            role_ids=torch.full(
                (1, 1), ROLE_IDS["assistant"], dtype=torch.long, device=state.device
            ),
        )
        found = self.bank.lookup(
            query, agenda, training=training, hindsight=hindsight, overlay=overlay,
            exploration_seed=exploration_seed
        )
        if found is None:
            return state, None
        realized = self.readout(found.value, query, agenda.current, core=self.core)
        committed = self.tracks.advance_group(
            queried,
            realized[:, None],
            tracks=(KNOWLEDGE_TRACK,),
            role_ids=torch.full(
                (1, 1), ROLE_IDS["assistant"], dtype=torch.long, device=state.device
            ),
        )
        self.bank.commit(agenda, found)
        return committed, found

    def start(
        self, metadata: dict[str, object]
    ) -> tuple[MultiTrackState, SearchAgenda, Lookup]:
        """Commit a complete metadata prefix and immediately perform the first lookup."""
        keys = ("request", "task", "topic", "language", "modality", "date", "version")
        normalized = {key: metadata.get(key) for key in keys}
        serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
        state = self.tracks.initial_state(1, device=self.bank.device)
        state = self.tracks.observe_group(
            state,
            [ObservedEvent("__metadata__", "text", serialized, role_id=ROLE_IDS["user"])],
        )
        query_prefix = self.prefix_head(state.raw[0, 0]).mean(0)
        agenda = self.bank.initial_agenda(query_prefix)
        state, result = self.boundary(state, agenda, training=False)
        if result is None:
            raise RuntimeError("The first metadata boundary must be permitted")
        return state, agenda, result
