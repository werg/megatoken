"""One shared E/C/P/R model over causally ordered meridians of facet blocks.

This is the text-first reference implementation of the multitrack execution
geometry. It deliberately reuses the existing central, predictor, realizer,
expert bank, and Gaussian owners: no per-track transformer or hidden KV exists.

A meridian groups *selected* events, with bidirectional live attention inside
that group. Prospective prediction reads only earlier committed meridians.
The initial reference path has a uniform nonempty event count across a batch;
the public serving/tape interface below handles one independent session.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call

from megatoken.contracts import DECODER_LOSSES, LOSS_NAMES
from megatoken.controls import encode_controls
from megatoken.generation.latents import DiagonalGaussian
from megatoken.generation.structure import ActionSpec
from megatoken.generation.targets import PreparedAction, PreparedTargets
from megatoken.io import strict_json_loads
from megatoken.model.context import DEFAULT_CONTEXT_POLICY, ContextPolicy
from megatoken.tensors import byte_tensor, is_host

from .adapters import OWNERS, AdapterMixture, AdapterPlan
from .attention import AttentionFeatures, join_attention
from .config import RESERVED_TRACKS, SYSTEM_TRACK_PURPOSES
from .core import EFFORTS, MODALITY_IDS, P_HISTORY_READ_STAGES, ROLE_IDS, FacetedCore
from .expert_pools import expert_scope
from .media import REALIZERS, MediaOutputSpec, MediaTrainTarget, validate_public_media_spec
from .relations import KINDS, SelectedRelation, semantic_relation_loss
from .routing import RowRoute
from .state import KVHistory, assert_cache_version

if TYPE_CHECKING:
    from megatoken.generation.policy import FacetedGenerator

GROUP_TAPE_VERSION = "fmt-multitrack-facets-v16"


@dataclass(frozen=True)
class ObservedEvent:
    """One available input to commit in a shared meridian.

    Text is its exact bytes. Audio, image, video and compressed text are ordered
    pinned teacher hidden states ``[1, T, 512]`` (``megatoken.teacher``), never
    raw media.
    """

    track: str
    modality: str
    payload: bytes | Tensor
    role_id: int = ROLE_IDS["user"]
    extractor: str = ""
    valid_mask: Tensor | None = None
    # False for a part of a longer message whose later parts have not arrived.
    terminal: bool = True
    event_id: str = ""

    def __post_init__(self) -> None:
        if self.modality not in MODALITY_IDS:
            raise ValueError("Unsupported input modality")
        if self.role_id not in ROLE_IDS.values():
            raise ValueError("Unknown event role")
        if self.modality == "text":
            if not isinstance(self.payload, bytes) or self.extractor or self.valid_mask is not None:
                raise TypeError("Text events are exact bytes")
            return
        if not isinstance(self.payload, Tensor) or not self.extractor:
            raise TypeError("Media events are pinned teacher hidden states")
        if self.payload.ndim != 3 or self.payload.shape[0] != 1 or self.payload.shape[1] < 1:
            raise ValueError("Media features must be one ordered [1, tokens, width] row")


@dataclass(frozen=True)
class MultiTrackState:
    """One shared history and the last *complete* meridian of selected events."""

    raw: Tensor  # [B, T, facets, width]
    final: Tensor | None
    history: KVHistory
    history_tracks: Tensor  # [B, retained events]; equality, never ordinal geometry
    meridian_sizes: tuple[int, ...]
    meridians_seen: Tensor  # [B]; next causal meridian index, also the RoPE clock
    events_seen: Tensor  # [B]; selected events, independent of meridian count
    frontier_tracks: tuple[str, ...]
    frontier_modalities: tuple[str, ...]
    track_registry: tuple[str, ...]
    frontier_roles: Tensor  # [B, T]
    model_version: int
    routing_version: int
    geometry: tuple[int, ...]
    # The latest lookup's adapter mixture; None until the first lookup.
    adapters: AdapterMixture | None = None
    frontier_event_ids: tuple[str, ...] = ()
    track_descriptions: tuple[tuple[str, str, int], ...] = ()
    policy_history: tuple[str, ...] | None = None
    history_event_ids: tuple[str, ...] = ()
    history_event_tracks: tuple[str, ...] = ()
    history_positions: tuple[int, ...] = ()
    context_seen: int = 0
    context_policy: ContextPolicy = ContextPolicy()
    context_dropped: tuple[tuple[str, str], ...] = ()

    @property
    def batch_size(self) -> int:
        return self.raw.shape[0]

    @property
    def device(self) -> torch.device:
        return self.raw.device

    @property
    def frontier_count(self) -> int:
        return self.raw.shape[1]

    def detach(self) -> MultiTrackState:
        return replace(
            self,
            raw=self.raw.detach(),
            final=None if self.final is None else self.final.detach(),
            history=self.history.detach(),
            adapters=None if self.adapters is None else self.adapters.detach(),
        )


class MultiTrackCore:
    """Stateless geometry adapter around one existing FacetedCore parameter owner."""

    def __init__(self, core: FacetedCore) -> None:
        self.core = core

    def initial_state(
        self,
        batch_size: int,
        *,
        device: str | torch.device | None = None,
    ) -> MultiTrackState:
        state = self.core.initial_state(batch_size, device=device)
        return MultiTrackState(
            raw=state.facets[:, None],
            final=None,
            history=state.history,
            history_tracks=torch.empty(batch_size, 0, dtype=torch.long, device=state.device),
            meridian_sizes=(),
            meridians_seen=state.records_seen,
            events_seen=state.records_seen,
            frontier_tracks=(),
            frontier_modalities=(),
            track_registry=(),
            frontier_roles=state.role_ids[:, None],
            model_version=state.model_version,
            routing_version=state.routing_version,
            geometry=state.geometry,
            policy_history=() if getattr(self, "record_policy_context", False) else None,
            context_policy=ContextPolicy(
                local_events=self.core.config.context_local_events,
                cross_events=self.core.config.context_cross_events,
            ),
        )

    def _validate(self, state: MultiTrackState) -> None:
        core = self.core
        assert_cache_version(state, core.model_version, core.config.routing_version)
        if state.geometry != core.config.state_geometry:
            raise ValueError("Multitrack state belongs to another model geometry")
        f, d = core.config.n_facets, core.config.facet_dim
        if state.raw.ndim != 4 or state.raw.shape[2:] != (f, d):
            raise ValueError("Frontier must contain whole canonical facet blocks")
        if state.raw.device != core.initial_facets.device:
            raise ValueError("Multitrack state is on a different device")
        if state.final is not None and state.final.shape != state.raw.shape:
            raise ValueError("Frontier features must retain every selected facet")
        if state.meridians_seen.shape != (state.batch_size,):
            raise ValueError("Meridian chronology has the wrong batch shape")
        if state.events_seen.shape != (state.batch_size,):
            raise ValueError("Selected-event chronology has the wrong batch shape")
        if state.frontier_modalities and len(state.frontier_modalities) != state.frontier_count:
            raise ValueError("Frontier modality metadata must match selected events")
        if len(set(state.track_registry)) != len(state.track_registry) or any(
            name not in state.track_registry and name not in RESERVED_TRACKS
            for name in state.frontier_tracks
        ):
            raise ValueError("Track registry is inconsistent with the shared context")
        if state.frontier_roles.shape != state.raw.shape[:2]:
            raise ValueError("Frontier role metadata has the wrong shape")
        if sum(state.meridian_sizes) * f != state.history.slots:
            raise ValueError("History and retained meridian sizes disagree")
        if state.history_tracks.shape != (state.batch_size, sum(state.meridian_sizes)):
            raise ValueError("Retained track metadata must match whole KV event blocks")
        if len(state.history_event_ids) != sum(state.meridian_sizes):
            raise ValueError("Retained evidence addresses must match whole KV event blocks")
        if (
            type(state.context_seen) is not int
            or state.context_seen < 0
            or not isinstance(state.context_policy, ContextPolicy)
            or len(state.history_event_tracks) != len(state.history_event_ids)
            or len(state.history_positions) != len(state.history_event_ids)
            or any(
                a >= b
                for a, b in zip(state.history_positions, state.history_positions[1:], strict=False)
            )
            or any(
                position < 0 or position >= state.context_seen
                for position in state.history_positions
            )
        ):
            raise ValueError("Retained context metadata must preserve original event positions")
        if state.history.slots > core.config.max_history_chunks * f:
            raise ValueError("History exceeds its full-block capacity")
        if state.adapters is not None and state.adapters.logits.shape != (
            state.batch_size,
            len(OWNERS),
            core.config.adapter_experts,
        ):
            raise ValueError("Adapter mixture has the wrong batch or expert geometry")
        if state.final is None and state.meridian_sizes:
            raise ValueError("Only initial state may lack processed features")

    @staticmethod
    def _names(tracks: Sequence[str], count: int) -> tuple[str, ...]:
        names = tuple(tracks)
        if count < 1 or len(names) != count or any(not s for s in names):
            raise ValueError("One meridian needs nonempty track addresses")
        return names

    @staticmethod
    def _event_ids(tracks: Sequence[str], event_ids: Sequence[str] | None) -> tuple[str, ...]:
        # Unique-track callers retain a convenient implicit address. Peers on
        # the same track must explicitly name the distinct selected events.
        ids = tuple(tracks if event_ids is None else event_ids)
        if (
            len(ids) != len(tracks)
            or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x for x in ids)
        ):
            raise ValueError("Every group member requires a unique nonempty event identity")
        return ids

    def _track_indices(
        self, state: MultiTrackState, names: Sequence[str]
    ) -> tuple[Tensor, tuple[str, ...]]:
        registry = list(state.track_registry)
        # Handles serve equality and attachment only; no learned table reads them.
        registry.extend(sorted(set(names) - set(registry)))
        indices = [registry.index(name) for name in names]
        values = torch.tensor([indices], device=state.device, dtype=torch.long).expand(
            state.batch_size, -1
        )
        return values, tuple(registry)

    def describe_track(
        self, state: MultiTrackState, address: str, purpose: str, revision: int = 0
    ) -> MultiTrackState:
        """Install an available causal description; labels/addresses are never encoded."""
        if not address or not purpose.strip() or type(revision) is not int or revision < 0:
            raise ValueError("A track description needs a purpose and nonnegative revision")
        descriptions = {a: (p, r) for a, p, r in state.track_descriptions}
        if address in descriptions:
            previous, old_revision = descriptions[address]
            if revision < old_revision or (revision == old_revision and purpose != previous):
                raise ValueError("Changing a track purpose requires a new causal revision")
        descriptions[address] = (purpose, revision)
        return replace(
            state, track_descriptions=tuple((a, p, r) for a, (p, r) in descriptions.items())
        )

    def description_controls(self, state: MultiTrackState, tracks: Sequence[str]) -> Tensor | None:
        descriptions = {a: p for a, p, _ in state.track_descriptions}
        # System roles have explicit meaning and do not occupy semantic slots.
        descriptions.update(SYSTEM_TRACK_PURPOSES)
        payloads = [descriptions.get(a, "").encode() for a in tracks] * state.batch_size
        encoded = encode_controls(self.core, payloads, encoder_grad=False)
        return None if encoded is None else encoded.reshape(state.batch_size, len(tracks), -1)

    def set_context_policy(self, state, policy: ContextPolicy):
        """Change public visibility without inventing or restoring missing context."""
        self._validate(state)
        for _, event_id in policy.pins:
            if state.history_event_ids.count(event_id) != 1:
                raise ValueError("Pinned context must identify one retained earlier event")
        if policy.max_pins > 8:
            raise ValueError("Context pins exceed the host allowance")
        return replace(state, context_policy=policy)

    def context_visibility(self, state, tracks):
        return tuple(
            state.context_policy.visible(
                track,
                state.history_event_tracks,
                state.history_positions,
                state.history_event_ids,
                state.context_seen,
            )
            for track in tracks
        )

    def context_view(self, state, tracks):
        """Gather the union of readable blocks once, with a per-reader mask.

        Projected keys retain their original temporal rotations. No reconstruction
        or positional renumbering happens when the storage is gathered.
        """
        visible = self.context_visibility(state, tracks)
        kept = tuple(
            i for i in range(len(state.history_event_ids)) if any(row[i] for row in visible)
        )
        facets = self.core.config.n_facets
        slots = torch.tensor(
            [i * facets + j for i in kept for j in range(facets)],
            dtype=torch.long,
            device=state.device,
        )
        history = state.history
        if len(kept) != len(state.history_event_ids):
            history = KVHistory(
                history.keys.index_select(2, slots),
                history.values.index_select(2, slots),
                None if history.mask is None else history.mask.index_select(1, slots),
            )
        ids = torch.tensor(kept, dtype=torch.long, device=state.device)
        history_tracks = state.history_tracks.index_select(1, ids).repeat_interleave(facets, dim=1)
        mask = None
        if any(not row[i] for row in visible for i in kept):
            mask = torch.tensor(
                [[row[i] for i in kept] for row in visible],
                dtype=torch.bool,
                device=state.device,
            )[None].expand(state.batch_size, -1, -1)
            mask = mask.repeat_interleave(facets, 1).repeat_interleave(facets, 2)
            mask = torch.cat(
                (
                    mask,
                    torch.ones(
                        state.batch_size,
                        len(tracks) * facets,
                        len(tracks) * facets,
                        dtype=torch.bool,
                        device=state.device,
                    ),
                ),
                dim=-1,
            )
        return history, history_tracks, mask

    def _history_for(self, state: MultiTrackState, tracks: Sequence[str]):
        """Reclaim unneeded whole groups before the one final KV write.

        Group retention is conservative: if any peer is needed, retain all peers.
        A hard global cap still bounds memory, including pinned/locally wanted
        groups; pressure that would discard a pin is rejected before publication.
        """
        new_events = len(tracks)
        budget = self.core.config.max_history_chunks
        if new_events > budget:
            raise ValueError("One meridian exceeds the global historical event capacity")
        names = (*state.history_event_tracks, *tracks)
        policy = state.context_policy
        wanted = set()
        for track in set(names):
            local, _ = policy.window(track)
            own = [i for i, name in enumerate(names) if name == track]
            if local:
                wanted.update(own[-local:])
        cross = max((policy.cross_events, *(row[2] for row in policy.windows)))
        wanted.update(
            i
            for i, position in enumerate(state.history_positions)
            if position >= state.context_seen + new_events - cross
        )
        pinned = {event for _, event in policy.pins}
        wanted.update(i for i, event in enumerate(state.history_event_ids) if event in pinned)
        groups, start = [], 0
        for size in state.meridian_sizes:
            indices = tuple(range(start, start + size))
            if any(i in wanted for i in indices):
                groups.append(indices)
            start += size
        count = sum(map(len, groups))
        pressure = set()
        while count + new_events > budget:
            removable = next(
                (
                    i
                    for i, group in enumerate(groups)
                    if not any(state.history_event_ids[j] in pinned for j in group)
                ),
                None,
            )
            if removable is None:
                raise ValueError(
                    "Context budget cannot retain pinned evidence and the incoming group"
                )
            removed = groups.pop(removable)
            pressure.update(removed)
            count -= len(removed)
        kept = tuple(i for group in groups for i in group)
        history = state.history
        if len(kept) != len(state.history_event_ids):
            f = self.core.config.n_facets
            slots = torch.tensor(
                [i * f + j for i in kept for j in range(f)], device=state.device, dtype=torch.long
            )
            history = KVHistory(
                history.keys.index_select(2, slots),
                history.values.index_select(2, slots),
                None if history.mask is None else history.mask.index_select(1, slots),
            )
        dropped = tuple(
            (event, "global_budget" if i in pressure else "window_expired")
            for i, event in enumerate(state.history_event_ids)
            if i not in kept
        )
        return history, tuple(map(len, groups)), kept, dropped

    def advance_group(
        self,
        state: MultiTrackState,
        raw: Tensor,
        *,
        tracks: Sequence[str],
        event_ids: Sequence[str] | None = None,
        role_ids: Tensor,
        modalities: Sequence[str] | None = None,
        track_semantics: Tensor | None = None,
    ) -> MultiTrackState:
        """Jointly integrate selected blocks and commit one complete meridian."""
        self._validate(state)
        c = self.core.config
        if (
            raw.ndim != 4
            or raw.shape[0] != state.batch_size
            or raw.shape[2:]
            != (
                c.n_facets,
                c.facet_dim,
            )
        ):
            raise ValueError("Raw group must be [batch, events, facets, width]")
        events = raw.shape[1]
        if events > c.max_history_chunks:
            raise ValueError("One meridian exceeds the global historical event capacity")
        names = self._names(tracks, events)
        identities = self._event_ids(names, event_ids)
        modality_names = tuple(modalities) if modalities is not None else ("text",) * events
        if len(modality_names) != events or any(x not in MODALITY_IDS for x in modality_names):
            raise ValueError("One supported modality is required per selected event")
        if (
            role_ids.shape != (state.batch_size, events)
            or role_ids.dtype != torch.long
            or role_ids.device != state.device
        ):
            raise ValueError("Every group member needs one device-local int64 role")
        if raw.device != state.device or not raw.is_floating_point():
            raise ValueError("Raw group has the wrong device or dtype")
        if is_host(raw) and not bool(torch.isfinite(raw).all()):
            raise ValueError("Raw selected facets must be finite")
        # Visibility applies before C; physical reclamation follows the read.
        track_ids, registry = self._track_indices(state, names)
        modality_ids = torch.tensor(
            [[MODALITY_IDS[x] for x in modality_names]], device=state.device, dtype=torch.long
        ).expand(state.batch_size, -1)
        semantics = self.description_controls(state, names)
        if track_semantics is not None:
            semantics = track_semantics if semantics is None else semantics + track_semantics
        adapter_plan = self.core.adapters.plan("c", state.adapters, track_ids, semantics)
        read_history, read_tracks, read_mask = self.context_view(state, names)
        final = self.core.central(
            raw.flatten(1, 2),
            read_history,
            role_ids,
            state.meridians_seen,
            track_ids=track_ids,
            modality_ids=modality_ids,
            history_track_ids=read_tracks,
            attention_mask=read_mask,
            track_semantics=semantics,
            adapters=adapter_plan,
        )
        history, retained, kept, dropped = self._history_for(state, names)
        next_history = self.core.central.write(
            final,
            history,
            state.meridians_seen,
            adapters=adapter_plan,
            track_ids=track_ids,
            track_semantics=semantics,
        )
        old_tracks = state.history_tracks.index_select(
            1, torch.tensor(kept, dtype=torch.long, device=state.device)
        )
        next_history_tracks = torch.cat((old_tracks, track_ids), dim=1)
        shape = (state.batch_size, events, c.n_facets, c.facet_dim)
        policy_history = state.policy_history
        if policy_history is not None:
            from megatoken.model.policy_context import MAX_CONTEXT_BYTES, capture

            entry = capture(
                state, raw, names, identities, role_ids, modality_names, track_semantics
            )
            if sum(map(len, policy_history)) + len(entry) > MAX_CONTEXT_BYTES:
                raise ValueError("Policy collection history exhausted; finish this bounded episode")
            policy_history = (*policy_history, entry)
        return MultiTrackState(
            raw=raw,
            final=final.reshape(shape),
            history=next_history,
            history_tracks=next_history_tracks,
            meridian_sizes=(*retained, events),
            meridians_seen=state.meridians_seen + 1,
            events_seen=state.events_seen + events,
            frontier_tracks=names,
            frontier_modalities=modality_names,
            track_registry=registry,
            frontier_roles=role_ids,
            model_version=state.model_version,
            routing_version=state.routing_version,
            geometry=state.geometry,
            adapters=state.adapters,
            frontier_event_ids=identities,
            track_descriptions=state.track_descriptions,
            history_event_ids=(*(state.history_event_ids[i] for i in kept), *identities),
            history_event_tracks=(*(state.history_event_tracks[i] for i in kept), *names),
            history_positions=(
                *(state.history_positions[i] for i in kept),
                *range(state.context_seen, state.context_seen + events),
            ),
            context_seen=state.context_seen + events,
            context_policy=state.context_policy,
            context_dropped=dropped,
            policy_history=policy_history,
        )

    def observe_group(
        self,
        state: MultiTrackState,
        events: Sequence[ObservedEvent],
        *,
        encoder_grad: bool = True,
    ) -> MultiTrackState:
        """Encode heterogeneous available events and integrate their facets once."""
        self._validate(state)
        if state.batch_size != 1:
            raise ValueError("Heterogeneous event serving requires one independent session")
        if not events:
            raise ValueError("At least one event is required")
        tracks = tuple(event.track for event in events)
        self._names(tracks, len(tracks))
        identities = self._event_ids(tracks, tuple(e.event_id or e.track for e in events))
        track_ids, _ = self._track_indices(state, tracks)
        semantics = self.description_controls(state, tracks)
        adapters = self.core.adapters.plan("e", state.adapters, track_ids, semantics)
        raws = []
        with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
            for index, event in enumerate(events):
                weights = None if adapters is None else adapters.weights[0, :, index]
                with expert_scope(weights):
                    if event.modality == "text":
                        values, lengths = byte_tensor([event.payload], state.device)
                        encoded = self.core.encode_chunk(
                            values,
                            lengths=lengths,
                            track_ids=track_ids[:, index],
                            track_semantics=None if semantics is None else semantics[:, index],
                        )
                    else:
                        if event.payload.device != state.device:
                            raise ValueError("Media event must be on the model device")
                        encoded = self.core.encode_media(
                            event.payload,
                            event.valid_mask,
                            extractor=event.extractor,
                            track_ids=track_ids[:, index],
                            track_semantics=None if semantics is None else semantics[:, index],
                        )
                raws.append(encoded)
        return self.advance_group(
            state,
            torch.stack(raws, dim=1),
            tracks=tracks,
            event_ids=identities,
            modalities=tuple(event.modality for event in events),
            role_ids=torch.tensor(
                [[event.role_id for event in events]], dtype=torch.long, device=state.device
            ),
        )

    def _predict_seed(self, state: MultiTrackState) -> Tensor:
        if state.final is None:
            # Match singleton startup: the sentinel is processed locally, not
            # appended to history and not exposed as an observation.
            final = self.core.central(
                state.raw[:, 0],
                state.history,
                state.frontier_roles[:, 0],
                state.meridians_seen,
            )
            return final
        return state.final.mean(dim=1)

    def _predict_seeds(self, state, tracks):
        if state.final is None:
            return self._predict_seed(state)[:, None].expand(-1, len(tracks), -1, -1)
        visible = self.context_visibility(state, tracks)
        retained = {event: i for i, event in enumerate(state.history_event_ids)}
        seeds, startup = [], None
        for row in visible:
            indices = [
                i
                for i, event in enumerate(state.frontier_event_ids)
                if event in retained and row[retained[event]]
            ]
            if indices:
                seeds.append(state.final[:, indices].mean(1))
            else:
                if startup is None:
                    initial = self.initial_state(state.batch_size, device=state.device)
                    startup = self._predict_seed(initial)
                seeds.append(startup)
        return torch.stack(seeds, dim=1)

    def predict_group(
        self,
        state: MultiTrackState,
        *,
        tracks: Sequence[str],
        event_ids: Sequence[str] | None = None,
        control: Tensor | None = None,
        effort: int = 6,
        modalities: Sequence[str] | None = None,
    ) -> Tensor:
        """Read-only *joint* prediction from all previously committed tracks.

        Live prediction-query attention spans all requested tracks. Pair-expert
        topology remains local to each 32-facet block, using the existing bank.
        """
        self._validate(state)
        if effort not in EFFORTS:
            raise ValueError(f"Effort must be one of {EFFORTS}")
        events = len(self._names(tracks, len(tracks)))
        self._event_ids(tracks, event_ids)
        if events > self.core.config.max_history_chunks:
            raise ValueError("One meridian exceeds the global historical event capacity")
        batch, f, d = state.batch_size, self.core.config.n_facets, self.core.config.facet_dim
        if control is not None and (
            control.shape != (batch, events, self.core.config.control_dim)
            or control.device != state.device
        ):
            raise ValueError("Expected one control vector per requested event")
        descriptions = self.description_controls(state, tracks)
        if descriptions is not None:
            control = descriptions if control is None else control + descriptions
        base = self._predict_seeds(state, tracks)
        track_ids, _ = self._track_indices(state, tracks)
        adapters = self.core.adapters.plan("p", state.adapters, track_ids, control)
        first_weights = None if adapters is None else adapters.weights[0]
        modality_names = tuple(modalities) if modalities is not None else ("text",) * events
        if len(modality_names) != events or any(x not in MODALITY_IDS for x in modality_names):
            raise ValueError("Unsupported prediction modality descriptor")
        modality_ids = torch.tensor(
            [[MODALITY_IDS[x] for x in modality_names]], device=state.device, dtype=torch.long
        ).expand(batch, -1)
        facets = base.reshape(batch * events, f, d)
        with self.core.expert_pools.routing_epoch(
            base.mean(2),
            track_ids=track_ids,
            track_semantics=control,
            role="p",
            adapter_weights=first_weights,
        ):
            with expert_scope(None if first_weights is None else first_weights.flatten(0, 1)):
                facets = self.core.remix(facets).reshape(batch, events, f, d)
            facets = facets + self.core.central.modality_embedding(modality_ids)[:, :, None]
            if control is not None:
                with expert_scope(first_weights):
                    facets = (
                        facets + self.core.control_injection(control.to(facets.dtype))[:, :, None]
                    )
        facets = facets + self.core.effort_embedding.weight[EFFORTS.index(effort)]
        stages = tuple(range(1, self.core.config.prediction_layers + 1))
        nrows = sum(self.core._row_counts(f, effort))
        with self.core.expert_pools.routing_epoch(
            facets.mean(2),
            track_ids=track_ids,
            track_semantics=control,
            role="p",
            adapter_weights=first_weights,
        ):
            routes = self.core.p_router.compile(
                facets.mean(dim=2).reshape(batch * events, d),
                stages,
                nrows,
                epoch="prediction",
            )
        self.core._record_routing_loss(routes.balance_loss, routes.stabilization_loss)
        read_view = self.context_view(state, tracks)
        for stage in stages:
            route = routes.at(stage)
            with self.core.expert_pools.routing_epoch(
                facets.mean(2),
                track_ids=track_ids,
                track_semantics=control,
                role="p",
                stage=stage,
                adapter_weights=None if adapters is None else adapters.weights[stage - 1],
            ):
                facets = self.core.run_pair_stage(
                    facets,
                    stage,
                    route,
                    effort,
                    lambda value, stage=stage, route=route: self._stage(
                        value, stage, route, effort, state, track_ids, adapters, read_view
                    ),
                )
        return facets

    def modality_logits(
        self,
        state: MultiTrackState,
        tracks: Sequence[str],
        control: Tensor | None = None,
        legal: Tensor | None = None,
    ) -> Tensor:
        """Choose output kinds before content, using only shared available state."""
        self._validate(state)
        count = len(self._names(tracks, len(tracks)))
        purpose = self.description_controls(state, tracks)
        if purpose is not None:
            control = purpose if control is None else control + purpose
        base = self._predict_seeds(state, tracks).mean(2)
        if control is None:
            control = base.new_zeros(state.batch_size, count, self.core.config.control_dim)
        logits = self.core.modality_head(torch.cat((base, control.to(base.dtype)), -1)).float()
        if legal is not None:
            if (
                legal.shape != logits.shape
                or legal.dtype != torch.bool
                or not bool(legal.any(-1).all())
            ):
                raise ValueError("Each event requires a nonempty legal modality mask")
            logits = logits.masked_fill(~legal, -torch.inf)
        return logits

    def _stage(
        self,
        facets: Tensor,
        stage: int,
        route: RowRoute,
        effort: int,
        state: MultiTrackState,
        track_ids: Tensor,
        adapters: AdapterPlan | None,
        read_view=None,
    ) -> Tensor:
        core = self.core
        batch, events, f, d = facets.shape
        wrapper = core.stages[stage - 1]
        facets = facets + core.stage_codes[stage - 1]
        joined = facets.reshape(batch, events * f, d)
        addresses = core.central.slot_codes().repeat(events, 1)
        slot_tracks = track_ids.repeat_interleave(f, dim=1)
        time = state.meridians_seen if stage in P_HISTORY_READ_STAGES else None
        weights = None if adapters is None else adapters.weights[stage - 1]
        slot_weights = None if weights is None else weights.repeat_interleave(f, dim=1)
        with expert_scope(slot_weights):
            current_kv = core.central.memory(
                joined,
                temporal_positions=time,
                slot_codes=addresses,
                track_ids=slot_tracks,
            )
        read_history, read_tracks, read_mask = read_view or (
            state.history,
            state.history_tracks.repeat_interleave(f, dim=1),
            None,
        )
        history = (
            AttentionFeatures(
                read_history.keys,
                read_history.values,
                read_history.mask,
                read_tracks,
            )
            if time is not None
            else None
        )
        keys = join_attention(current_kv, history)
        with expert_scope(slot_weights):
            mixed = wrapper.attention.read(
                joined,
                keys,
                temporal_positions=time,
                slot_codes=addresses,
                query_track_ids=slot_tracks,
                query_key_mask=read_mask if time is not None else None,
            )
        joined = joined + wrapper.attention_gate.sigmoid() * mixed
        with expert_scope(None if weights is None else weights.flatten(0, 1)):
            individual = core.pair_updates(
                joined.reshape(batch * events, f, d), stage, route, effort
            )
        individual = individual.reshape(batch, events, f, d)
        return individual


@dataclass(frozen=True)
class GroupActionTape:
    """All raw draws for one generated meridian and its deterministic text outputs."""

    version: str
    meridian_id: int
    tracks: tuple[str, ...]
    roles: tuple[int, ...]
    controls: tuple[bytes, ...]
    spec_fingerprints: tuple[str, ...]
    codec_fingerprint: str
    facet_shape: tuple[int, int, int]
    raw_f32: bytes
    behavior_log_prob: float
    payloads: tuple[bytes, ...]
    emitted_ids: tuple[tuple[int, ...], ...]
    valid: tuple[bool, ...]
    # Whether each event ends its message (the decoder's own terminal decision).
    terminal: tuple[bool, ...]
    modalities: tuple[str, ...] = ()
    effort: int = 6
    event_log_probs: tuple[float, ...] = ()
    renderer_seeds: tuple[int, ...] = ()
    pack_fingerprints: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    modality_masks: tuple[tuple[bool, ...], ...] = ()
    modality_log_probs: tuple[float, ...] = ()
    render_status: str = "rendered"
    track_descriptions: tuple[tuple[str, str, int], ...] = ()
    policy_context: tuple[str, ...] | None = None
    context_policy_json: str = DEFAULT_CONTEXT_POLICY

    def __post_init__(self) -> None:
        ContextPolicy.from_json(self.context_policy_json)
        if self.render_status not in {"selected", "rendered"}:
            raise ValueError("Unknown grouped rendering state")
        if self.version != GROUP_TAPE_VERSION:
            raise ValueError("Incompatible grouped tape version")
        n, f, d = self.facet_shape
        if n < 1 or f < 1 or d < 1 or len(self.raw_f32) != n * f * d * 4:
            raise ValueError("Tape must contain every complete raw FP32 event")
        lengths = (
            len(self.tracks),
            len(self.modalities),
            len(self.roles),
            len(self.controls),
            len(self.spec_fingerprints),
            len(self.payloads),
            len(self.emitted_ids),
            len(self.valid),
            len(self.terminal),
            len(self.renderer_seeds),
            len(self.pack_fingerprints),
        )
        if (
            any(k != n for k in lengths)
            or len(self.event_ids) != n
            or len(set(self.event_ids)) != n
            or any(not isinstance(x, str) or not x for x in self.event_ids)
        ):
            raise ValueError("Grouped tape fields must agree on exact event identities")
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in self.renderer_seeds):
            raise ValueError("Renderer noise seeds must match group identities")
        if any(not isinstance(value, str) or not value for value in self.pack_fingerprints):
            raise ValueError("Grouped tape realizer identities must cover every event")
        if not math.isfinite(self.behavior_log_prob):
            raise ValueError("Behavior log probability must be finite")
        if self.effort not in EFFORTS:
            raise ValueError("Grouped action tape contains an unsupported effort")
        if len(self.event_log_probs) != n or any(
            not math.isfinite(value) for value in self.event_log_probs
        ):
            raise ValueError("Every selected event needs a finite policy density")
        if not math.isclose(
            sum(self.event_log_probs) + sum(self.modality_log_probs),
            self.behavior_log_prob,
            rel_tol=1e-5,
            abs_tol=1e-3,
        ):
            raise ValueError("Per-event policy density does not sum to the grouped action")
        if len(self.modality_masks) != n or len(self.modality_log_probs) != n:
            raise ValueError("Every event requires its modality choice probability and mask")
        for modality, mask, probability in zip(
            self.modalities, self.modality_masks, self.modality_log_probs, strict=True
        ):
            if len(mask) != 4 or any(type(x) is not bool for x in mask) or not any(mask):
                raise ValueError("Invalid legal modality mask")
            if (
                modality not in ("text", "audio", "image", "video")
                or not mask[MODALITY_IDS[modality]]
            ):
                raise ValueError("Selected modality is illegal")
            if not math.isfinite(probability) or probability > 0:
                raise ValueError("Modality choice requires a finite log probability")
        if not np.isfinite(np.frombuffer(self.raw_f32, dtype="<f4")).all():
            raise ValueError("Raw facet tape contains nonfinite coordinates")

    def raw(self, device: torch.device) -> Tensor:
        return (
            torch.from_numpy(np.frombuffer(self.raw_f32, dtype="<f4").copy())
            .to(device=device)
            .reshape(1, *self.facet_shape)
        )

    def single_raw(self, device: torch.device) -> Tensor:
        """The raw facets of a one-event tape: ``[1, facets, width]``."""
        if len(self.tracks) != 1:
            raise ValueError("Only a single-event tape has one raw facet block")
        return self.raw(device)[:, 0]

    def to_json(self) -> str:
        value = dict(
            version=self.version,
            meridian_id=self.meridian_id,
            tracks=self.tracks,
            event_ids=self.event_ids,
            render_status=self.render_status,
            track_descriptions=self.track_descriptions,
            policy_context=self.policy_context,
            context_policy_json=self.context_policy_json,
            modality_masks=self.modality_masks,
            modality_log_probs=self.modality_log_probs,
            modalities=self.modalities,
            roles=self.roles,
            controls=[base64.b64encode(b).decode("ascii") for b in self.controls],
            spec_fingerprints=self.spec_fingerprints,
            codec_fingerprint=self.codec_fingerprint,
            facet_shape=self.facet_shape,
            raw_f32=base64.b64encode(self.raw_f32).decode("ascii"),
            behavior_log_prob=self.behavior_log_prob,
            effort=self.effort,
            event_log_probs=self.event_log_probs,
            renderer_seeds=self.renderer_seeds,
            pack_fingerprints=self.pack_fingerprints,
            payloads=[base64.b64encode(b).decode("ascii") for b in self.payloads],
            emitted_ids=self.emitted_ids,
            valid=self.valid,
            terminal=self.terminal,
        )
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, serialized: str) -> GroupActionTape:
        value = strict_json_loads(serialized)
        if value.get("version") != GROUP_TAPE_VERSION:
            raise ValueError("Unknown grouped tape version")

        def decode(b):
            return base64.b64decode(b, validate=True)

        return cls(
            version=value["version"],
            meridian_id=int(value["meridian_id"]),
            tracks=tuple(value["tracks"]),
            event_ids=tuple(value["event_ids"]),
            render_status=value["render_status"],
            track_descriptions=tuple(tuple(x) for x in value["track_descriptions"]),
            policy_context=None
            if value["policy_context"] is None
            else tuple(value["policy_context"]),
            context_policy_json=value["context_policy_json"],
            modality_masks=tuple(tuple(x) for x in value["modality_masks"]),
            modality_log_probs=tuple(value["modality_log_probs"]),
            roles=tuple(value["roles"]),
            modalities=tuple(value["modalities"]),
            controls=tuple(map(decode, value["controls"])),
            spec_fingerprints=tuple(value["spec_fingerprints"]),
            codec_fingerprint=value["codec_fingerprint"],
            facet_shape=tuple(value["facet_shape"]),
            raw_f32=decode(value["raw_f32"]),
            behavior_log_prob=float(value["behavior_log_prob"]),
            effort=int(value["effort"]),
            event_log_probs=tuple(map(float, value["event_log_probs"])),
            renderer_seeds=tuple(value["renderer_seeds"]),
            pack_fingerprints=tuple(value["pack_fingerprints"]),
            payloads=tuple(map(decode, value["payloads"])),
            emitted_ids=tuple(tuple(s) for s in value["emitted_ids"]),
            valid=tuple(value["valid"]),
            terminal=tuple(value["terminal"]),
        )


@dataclass(frozen=True)
class GroupActionMember:
    """A supervised view of a member, never an independently replayable action.

    Its parent retains the complete group, legal masks and joint probability.
    Training may align this member's representation with its exact physical
    record without splitting the causal commit or inventing a new policy draw.
    """

    group: GroupActionTape
    index: int

    def __post_init__(self):
        if not 0 <= self.index < len(self.group.tracks):
            raise ValueError("Native member index is outside its selected group")

    def single_raw(self, device):
        return self.group.raw(device)[:, self.index]

    @property
    def facet_shape(self):
        return (1, *self.group.facet_shape[1:])

    @property
    def codec_fingerprint(self):
        return self.group.codec_fingerprint

    @property
    def native_tape(self):
        return self.group

    def __getattr__(self, name):
        if name in {
            "tracks",
            "event_ids",
            "roles",
            "controls",
            "modalities",
            "payloads",
            "emitted_ids",
            "valid",
            "terminal",
        }:
            return (getattr(self.group, name)[self.index],)
        raise AttributeError(name)


@dataclass(frozen=True)
class GroupSupervisedResult:
    """One differentiable multitrack transaction with independent loss units."""

    state: MultiTrackState
    losses: dict[str, tuple[Tensor, int]]
    metrics: dict[str, Tensor | int | float]


@dataclass(frozen=True)
class SelectedGroup:
    """Immutable selection snapshot; rendering and delivery are later outcomes."""

    before: MultiTrackState
    state: MultiTrackState
    raw: Tensor
    specs: tuple[ActionSpec | MediaOutputSpec, ...]
    control: Tensor | None
    tape: GroupActionTape
    log_prob: Tensor


@dataclass(frozen=True)
class GroupGeneration:
    state: MultiTrackState
    tape: GroupActionTape
    log_prob: Tensor
    media_outputs: tuple[Tensor | None, ...] = ()


class _GroupedTeacherReadout(nn.Module):
    """One direct R readout for predictions and E reconstruction.

    Functional parameter substitution on this wrapper freezes all R/U owners
    during recurrent encoder-only reconstruction, without running C.
    """

    def __init__(self, core: FacetedCore, decoder: nn.Module, media_realizers: nn.Module) -> None:
        super().__init__()
        self.core = core
        self.decoder = decoder
        self.media_realizers = media_realizers

    def losses_for_selected(
        self,
        raw: Tensor,
        plans: Sequence[PreparedTargets | PreparedAction | MediaTrainTarget],
        control: Tensor | None = None,
        mixture: AdapterMixture | None = None,
        track_ids: Tensor | None = None,
    ) -> dict[str, tuple[Tensor, int]]:
        if raw.ndim != 4 or raw.shape[1] != len(plans):
            raise ValueError("Realization needs one canonical facet block per target")
        zero = raw.float().sum() * 0
        adapter_plan = (
            self.core.adapters.plan("r", mixture, track_ids, control)
            if mixture is not None
            else None
        )
        names = (*DECODER_LOSSES, "audio", "visual")
        sums = {name: zero for name in names}
        counts = {name: 0 for name in names}
        for index, plan in enumerate(plans):
            selected = raw[:, index]
            context = self.core.prepare_realization(selected)
            weights = None if adapter_plan is None else adapter_plan.weights[0, :, index]
            track = None if track_ids is None else track_ids[:, index]
            with self.core.expert_pools.routing_epoch(
                selected.mean(1),
                track_ids=track,
                track_semantics=None if control is None else control[:, index],
                role="R",
                stage=0,
                adapter_weights=weights,
            ):
                with expert_scope(weights):
                    if isinstance(plan, MediaTrainTarget):
                        family = "audio" if plan.modality == "audio" else "visual"
                        sums[family] = sums[family] + self.media_realizers.loss(context, plan)
                        counts[family] += 1
                        continue
                    if isinstance(plan, PreparedTargets):
                        parts = self.decoder.teacher_raw(self.core, context, plan)
                    else:
                        parts = self.decoder(self.core, context, plan.spec, teacher=plan).losses
            for name in DECODER_LOSSES:
                value, count = parts[name]
                sums[name] = sums[name] + value
                counts[name] += count
        return {name: (sums[name], counts[name]) for name in names}

    def forward(
        self,
        raw: Tensor,
        plans: tuple[PreparedTargets | PreparedAction | MediaTrainTarget, ...],
        control: Tensor | None = None,
        mixture: AdapterMixture | None = None,
        track_ids: Tensor | None = None,
    ) -> dict[str, tuple[Tensor, int]]:
        return self.losses_for_selected(raw, plans, control, mixture, track_ids)


class MultiTrackGenerator:
    """One shared P/C/Gaussian, with modality-specific deterministic realization."""

    @classmethod
    def of(cls, generator: FacetedGenerator) -> MultiTrackGenerator:
        """The one multitrack view of a model, so per-version caches outlive a window.

        The view is kept on the model itself: a weak-keyed table whose values hold the model
        strongly would keep every model (and its device memory) alive for the whole process.
        """
        view = generator.__dict__.get("_multitrack_view")
        if view is None:
            view = generator.__dict__["_multitrack_view"] = cls(generator)
        return view

    def __init__(self, generator: FacetedGenerator) -> None:
        self.generator = generator
        self.tracks = MultiTrackCore(generator.core)
        self._frozen: (
            tuple[int, tuple[tuple[str, int, tuple[int, ...]], ...], dict[str, Tensor]] | None
        ) = None

    def _frozen_readout(self, readout: nn.Module) -> dict[str, Tensor]:
        """Detached views of every readout parameter, built once per model version.

        The views alias the live parameters, so optimizer updates show through;
        only a new version (or new parameter objects) needs a new dict.
        """
        version = self.generator.core.model_version
        parameters = tuple(readout.named_parameters())
        # Media conditioners and shape buckets may be bound after this wrapper
        # has already run once. Include parameter identities and shapes so a
        # late registration cannot silently fall back to live R parameters in
        # functional_call's non-strict mapping.
        signature = tuple(
            (name, id(parameter), tuple(parameter.shape)) for name, parameter in parameters
        )
        if self._frozen is None or self._frozen[0] != version or self._frozen[1] != signature:
            self._frozen = (
                version,
                signature,
                {name: parameter.detach() for name, parameter in parameters},
            )
        return self._frozen[2]

    @staticmethod
    def _sample_policy(
        distribution: DiagonalGaussian,
        names: Sequence[str],
        generator: torch.Generator | None,
    ) -> Tensor:
        """Assign RNG draws by canonical event identity, not physical slot order."""
        order = sorted(range(len(names)), key=lambda index: names[index])
        sorted_distribution = DiagonalGaussian(
            distribution.mean[:, order], distribution.log_scale[:, order]
        )
        selected = sorted_distribution.sample(generator)
        inverse = [order.index(index) for index in range(len(order))]
        return selected[:, inverse]

    def _controls(
        self, payloads: Sequence[bytes], state: MultiTrackState, tracks: Sequence[str]
    ) -> Tensor | None:
        from megatoken.controls import semantic_lifecycle_controls, separate_event_references

        core = self.generator.core
        separated = [separate_event_references(payload) for payload in payloads]
        descriptions = {a: (p, r) for a, p, r in state.track_descriptions}
        separated = [
            (semantic_lifecycle_controls(value, descriptions), references)
            for value, references in separated
        ]
        ids, _ = self.tracks._track_indices(state, tracks)
        plan = core.adapters.plan("p", state.adapters, ids)
        weights = None if plan is None else plan.weights[0].flatten(0, 1)
        with expert_scope(weights):
            encoded = encode_controls(
                core, [value[0] for value in separated], encoder_grad=False, track_ids=ids.flatten()
            )
        if not any(value[1] for value in separated):
            return None if encoded is None else encoded[None].to(state.device)
        rows = []
        visibility = self.tracks.context_visibility(state, tracks)
        for row, (_, references) in zip(visibility, separated, strict=True):
            if not references:
                rows.append(core.initial_facets.new_zeros(1, core.config.control_dim))
                continue
            indices = []
            for address in references:
                if state.history_event_ids.count(address) != 1:
                    raise ValueError("Reference must identify one retained earlier event")
                index = state.history_event_ids.index(address)
                if not row[index]:
                    raise ValueError(
                        "Reference is outside this track's context window; request a context handoff"
                    )
                indices.extend(
                    range(index * core.config.n_facets, (index + 1) * core.config.n_facets)
                )
            # Select actual available evidence from the existing bounded history.
            # Never encode an arbitrary address as meaning or keep a second media memory.
            values = state.history.values[:, :, indices].mean(2).flatten(1)
            rows.append(core.reference_projection(values))
        selected = torch.stack(rows, dim=1)
        return selected if encoded is None else selected + encoded[None].to(state.device)

    def _prior(
        self,
        mean: Tensor,
        state: MultiTrackState,
        tracks: Sequence[str],
        control: Tensor | None = None,
    ):
        """The scale projection uses the same causal P mixture as prediction."""
        ids, _ = self.tracks._track_indices(state, tracks)
        plan = self.generator.core.adapters.plan("p", state.adapters, ids, control)
        weights = None if plan is None else plan.weights[-1]
        with (
            self.generator.core.expert_pools.routing_epoch(
                mean.mean(2),
                track_ids=ids,
                track_semantics=control,
                role="p",
                stage=self.generator.config.prediction_layers,
                adapter_weights=weights,
            ),
            expert_scope(weights),
        ):
            return self.generator.facet_distribution(mean)

    @staticmethod
    def _specs(
        specs: Sequence[ActionSpec | MediaOutputSpec] | None, count: int, capacity: int
    ) -> tuple[ActionSpec | MediaOutputSpec, ...]:
        if specs is None:
            return (ActionSpec(max_tokens=min(512, capacity)),) * count
        values = tuple(specs)
        if len(values) != count or any(
            not isinstance(spec, (ActionSpec, MediaOutputSpec)) for spec in values
        ):
            raise ValueError("One valid output specification is needed per track")
        return values

    @staticmethod
    def _spec_id(spec: ActionSpec | MediaOutputSpec) -> str:
        if isinstance(spec, ActionSpec):
            return spec.fingerprint
        value = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode()
        return "media:" + hashlib.sha256(value).hexdigest()

    @staticmethod
    def _pack_ids(
        generator: FacetedGenerator, plans: Sequence[ActionSpec | MediaOutputSpec]
    ) -> tuple[str, ...]:
        """Bind live raw actions to exact pretrained generator+decoder weights."""
        identities = []
        for plan in plans:
            if isinstance(plan, ActionSpec):
                identities.append("text")
            else:
                identities.append(generator.media_realizers.resident(plan.modality).identity)
        return tuple(identities)

    @staticmethod
    def _renderer_seeds(
        raw: Tensor, names: Sequence[str], meridian_id: int, spec_fingerprints: Sequence[str]
    ) -> tuple[int, ...]:
        """Materialization RNG is stable under pack loading or track permutation."""
        seeds = []
        for idx, (name, spec_id) in enumerate(zip(names, spec_fingerprints, strict=True)):
            identity = hashlib.sha256(
                str(meridian_id).encode()
                + b":"
                + name.encode()
                + b":"
                + spec_id.encode()
                + b":"
                + raw[0, idx]
                .detach()
                .float()
                .contiguous()
                .cpu()
                .numpy()
                .astype("<f4", copy=False)
                .tobytes()
            ).digest()
            seeds.append(int.from_bytes(identity[:8], "big") % (2**63))
        return tuple(seeds)

    def _realize(
        self,
        raw: Tensor,
        specs: Sequence[ActionSpec | MediaOutputSpec],
        control: Tensor | None,
        *,
        seeds: Sequence[int] | None = None,
        mixture: AdapterMixture | None = None,
        track_ids: Tensor | None = None,
    ) -> tuple[
        tuple[bytes, ...],
        tuple[tuple[int, ...], ...],
        tuple[bool, ...],
        tuple[bool, ...],
        tuple[Tensor | None, ...],
    ]:
        payloads, ids, validity, terminals, media = [], [], [], [], []
        if seeds is not None and len(seeds) != len(specs):
            raise ValueError("Renderer seeds must identify each selected event")
        adapter_plan = (
            self.generator.core.adapters.plan("r", mixture, track_ids, control)
            if mixture is not None
            else None
        )
        for index, spec in enumerate(specs):
            selected = raw[:, index]
            context = self.generator.core.prepare_realization(selected)
            weights = None if adapter_plan is None else adapter_plan.weights[0, :, index]
            track = None if track_ids is None else track_ids[:, index]
            with self.generator.core.expert_pools.routing_epoch(
                selected.mean(1),
                track_ids=track,
                track_semantics=None if control is None else control[:, index],
                role="R",
                stage=0,
                adapter_weights=weights,
            ):
                with expert_scope(weights):
                    if isinstance(spec, ActionSpec):
                        decoded = self.generator.decoder(self.generator.core, context, spec)
                        payloads.append(decoded.payload)
                        ids.append(decoded.emitted_ids)
                        validity.append(decoded.valid)
                        terminals.append(decoded.terminal)
                        media.append(None)
                        continue
                    validate_public_media_spec(spec)
                    materializer = None
                    if seeds is not None:
                        materializer = torch.Generator(device=raw.device)
                        materializer.manual_seed(seeds[index])
                    tensor = self.generator.media_realizers.realize(
                        context, spec, generator=materializer
                    )
                    native = (
                        tensor[0]
                        .detach()
                        .float()
                        .contiguous()
                        .cpu()
                        .numpy()
                        .astype("<f4", copy=False)
                    )
                    # Large media is a digest-bound output, not a multi-MB tape field.
                    payloads.append(
                        json.dumps(
                            {
                                "kind": "media-sha256-v1",
                                "modality": spec.modality,
                                "realizer": self.generator.media_realizers.resident(
                                    spec.modality
                                ).identity,
                                "sha256": hashlib.sha256(native.tobytes()).hexdigest(),
                                "shape": list(native.shape),
                                "dtype": "f32le",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    )
                    ids.append(())
                    validity.append(True)
                    terminals.append(True)
                    media.append(tensor)
        return tuple(payloads), tuple(ids), tuple(validity), tuple(terminals), tuple(media)

    def supervised_meridian(
        self,
        state: MultiTrackState,
        *,
        tracks: Sequence[str],
        event_ids: Sequence[str] | None = None,
        plans: Sequence[PreparedTargets | PreparedAction | MediaTrainTarget],
        control_payloads: Sequence[bytes] | None = None,
        effort: int = 6,
        teacher_forced_action: bool = False,
        generator: torch.Generator | None = None,
        relations: Sequence[tuple[str, int, int, float]] = (),
        semantic_relations: Sequence[SelectedRelation] = (),
        evidence_relations: Sequence[tuple[str, int, Tensor]] = (),
        modality_masks: Tensor | None = None,
        full_group: GroupActionTape | None = None,
        plan_indices: Sequence[int] | None = None,
    ) -> GroupSupervisedResult:
        """One production grouped objective with bootstrap or live recurrent state.

        Every target's raw facets are encoded *after* P selects its prior.
        Executed-action continuations use genuine saved native draws where
        present and detached E(recorded bytes) only as an explicit fallback.
        """
        self.tracks._validate(state)
        if state.batch_size != 1:
            raise ValueError("Grouped supervised meridians require one independent episode")
        if full_group is not None:
            state = self.tracks.set_context_policy(
                state, ContextPolicy.from_json(full_group.context_policy_json)
            )
        names = self.tracks._names(tracks, len(tracks))
        identities = self.tracks._event_ids(names, event_ids)
        plans = tuple(plans)
        indices = tuple(range(len(names))) if plan_indices is None else tuple(plan_indices)
        if (
            not plans
            or len(plans) != len(indices)
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(names) for i in indices)
            or (full_group is None and indices != tuple(range(len(names))))
        ):
            raise ValueError("Target plans must map to distinct complete-group members")
        if any(
            not isinstance(plan, (PreparedTargets, PreparedAction, MediaTrainTarget))
            for plan in plans
        ):
            raise ValueError("Every track requires one prepared text or media target")
        modes = {plan.mode for plan in plans}
        if len(modes) != 1 or not modes.issubset({"bootstrap", "recurrent"}):
            raise ValueError("Peers must share one representation-training mode")
        mode = modes.pop()
        if any(isinstance(p, PreparedTargets) and len(p.payloads) != 1 for p in plans):
            raise ValueError("Each grouped target plan must describe one event")
        modalities = tuple(
            plan.modality if isinstance(plan, MediaTrainTarget) else "text" for plan in plans
        )
        roles = torch.tensor(
            [[p.roles[0] if isinstance(p, PreparedTargets) else p.role_id for p in plans]],
            dtype=torch.long,
            device=state.device,
        )
        if full_group is not None:
            if full_group.tracks != names or full_group.event_ids != identities:
                raise ValueError("Recorded selection membership differs from supervised group")
            if tuple(full_group.modalities[i] for i in indices) != modalities:
                raise ValueError("Target modalities differ from their recorded group members")
            modalities = full_group.modalities
            roles = torch.tensor([full_group.roles], dtype=torch.long, device=state.device)
        controls = tuple(control_payloads) if control_payloads is not None else (b"",) * len(names)
        if len(controls) != len(names) or any(not isinstance(x, bytes) for x in controls):
            raise ValueError("Every grouped event needs one causal control payload")
        core = self.generator.core
        core.routing_aux_loss(clear=True)
        control = self._controls(controls, state, names)
        mean = self.tracks.predict_group(
            state,
            tracks=names,
            event_ids=identities,
            control=control,
            effort=effort,
            modalities=modalities,
        )
        prior = self._prior(mean, state, names, control)
        sampled = self._sample_policy(prior, identities, generator)
        # Only recurrent continuation needs C(sampled); R(sampled) does not.
        selected = (
            self.tracks.advance_group(
                state,
                sampled,
                track_semantics=control,
                tracks=names,
                event_ids=identities,
                role_ids=roles,
                modalities=modalities,
            )
            if mode == "recurrent" and not teacher_forced_action and full_group is None
            else None
        )
        full_sampled = sampled
        sampled = sampled[:, indices]
        target_control = None if control is None else control[:, indices]
        target_names = tuple(names[i] for i in indices)
        track_ids, _ = self.tracks._track_indices(state, target_names)
        semantics = self.tracks.description_controls(state, target_names)
        encoder_plan = core.adapters.plan("e", state.adapters, track_ids, semantics)
        encoded = []
        for index, plan in enumerate(plans):
            weights = None if encoder_plan is None else encoder_plan.weights[0, :, index]
            with expert_scope(weights):
                if isinstance(plan, MediaTrainTarget):
                    current = core.encode_media(
                        plan.source,
                        extractor=plan.extractor,
                        track_ids=track_ids[:, index],
                        track_semantics=None if semantics is None else semantics[:, index],
                    )
                else:
                    payload = (
                        plan.payloads[0] if isinstance(plan, PreparedTargets) else plan.payload
                    )
                    values, lengths = byte_tensor([payload], state.device)
                    current = core.encode_chunk(
                        values,
                        lengths,
                        track_ids=track_ids[:, index],
                        track_semantics=None if semantics is None else semantics[:, index],
                    )
            encoded.append(current)
        target_raw = torch.stack(encoded, dim=1)
        q = DiagonalGaussian(
            target_raw.float(),
            torch.full_like(target_raw.float(), math.log(core.config.codec_noise_std)),
        )
        codec_sample = q.sample(generator)
        readout = _GroupedTeacherReadout(
            core, self.generator.decoder, self.generator.media_realizers
        )
        prior_terms = readout.losses_for_selected(
            sampled, plans, target_control, state.adapters, track_ids
        )

        if mode == "bootstrap":
            with core.discard_routing_losses():
                recon_terms = readout(
                    codec_sample, plans, target_control, state.adapters, track_ids
                )
            continuation = target_raw
            correspondence = (
                q.detach()
                .kl(DiagonalGaussian(prior.mean[:, indices], prior.log_scale[:, indices]))
                .sum()
            )
        else:
            # Direct R is frozen for this encoder-only recurrent auxiliary;
            # only the encoder-produced canonical facets receive its gradient.
            frozen = self._frozen_readout(readout)
            with core.discard_routing_losses():
                recon_terms = functional_call(
                    readout,
                    frozen,
                    (
                        codec_sample,
                        plans,
                        None if target_control is None else target_control.detach(),
                        None if state.adapters is None else state.adapters.detach(),
                        track_ids,
                    ),
                )
            correspondence = mean.float().sum() * 0
            if teacher_forced_action:
                actual = []
                for index, plan in enumerate(plans):
                    tape = (
                        plan.native_tapes[0]
                        if isinstance(plan, PreparedTargets)
                        else plan.native_tape
                        if isinstance(plan, (PreparedAction, MediaTrainTarget))
                        else None
                    )
                    if tape is None:
                        actual.append(target_raw[:, index].detach())
                    else:
                        actual.append(tape.single_raw(state.device))
                continuation = torch.stack(actual, dim=1)
            else:
                continuation = sampled
        # Ordinary recurrence keeps its one committed C pass for the next P.
        # Neither prior nor reconstruction realization processes C.
        next_state = (
            self.commit_tape(state, full_group)
            if full_group is not None
            else selected
            if mode == "recurrent" and not teacher_forced_action
            else self.tracks.advance_group(
                state,
                continuation,
                track_semantics=control,
                tracks=names,
                event_ids=identities,
                role_ids=roles,
                modalities=modalities,
            )
        )
        zero = mean.float().sum() * 0
        sums = dict.fromkeys(LOSS_NAMES, zero)
        counts = dict.fromkeys(LOSS_NAMES, 0)
        sums["lexical"], counts["lexical"] = prior_terms["lexical"]
        sums["reconstruction"], counts["reconstruction"] = recon_terms["lexical"]
        for family in ("audio", "visual"):
            sums[f"{family}_prediction"], counts[f"{family}_prediction"] = prior_terms[family]
            sums[f"{family}_reconstruction"], counts[f"{family}_reconstruction"] = recon_terms[
                family
            ]
        for name in ("structure", "binding", "placement", "terminal"):
            a, a_count = prior_terms[name]
            b, b_count = recon_terms[name]
            sums[name], counts[name] = a + b, a_count + b_count
        # Relation targets are out-of-band and do not add peer input to P/R.
        # Compare only coarse pooled semantic directions, not complete Z or
        # unrelated modality codec latents. All peers are sampled predictions.
        for relation_kind, left, right, strength in relations:
            if (
                relation_kind not in KINDS
                or not 0 <= left < len(names)
                or not 0 <= right < len(names)
                or left == right
                or not 0 < strength <= 10
            ):
                raise ValueError("Unrecognized or invalid jointly selected relation")
            semantic_a = full_sampled[:, left].float().mean(dim=1)
            semantic_b = full_sampled[:, right].float().mean(dim=1)
            alignment_loss = 1 - torch.nn.functional.cosine_similarity(
                semantic_a,
                semantic_b,
                dim=-1,
                eps=1e-6,
            )
            sums["relation"] = sums["relation"] + strength * alignment_loss.sum()
            counts["relation"] += 1
        semantic_relations = tuple(semantic_relations)
        target_relations = tuple(
            SelectedRelation(rel.kind, indices.index(rel.left), indices.index(rel.right))
            for rel in semantic_relations
            if rel.left in indices and rel.right in indices
        )
        relation_loss, relation_count = semantic_relation_loss(
            sampled, target_raw, target_relations
        )
        for kind, index, evidence in evidence_relations:
            if (
                kind not in KINDS
                or not 0 <= index < len(names)
                or evidence.shape != full_sampled[:, index].shape
            ):
                raise ValueError("Invalid relation to earlier available evidence")
            current = torch.nn.functional.normalize(full_sampled[:, index].float().mean(1), dim=-1)
            previous = torch.nn.functional.normalize(evidence.detach().float().mean(1), dim=-1)
            relation_loss = relation_loss + (1 - (current * previous).sum(-1)).sum()
            relation_count += 1
        modality_targets = torch.tensor(
            [[MODALITY_IDS[m] for m in modalities]], device=state.device
        )
        if full_group is not None:
            modality_masks = torch.tensor(
                [full_group.modality_masks], device=state.device, dtype=torch.bool
            )
        elif modality_masks is None:
            supported = [
                True,
                *(
                    REALIZERS[m] in self.generator.media_realizers.packs
                    for m in ("audio", "image", "video")
                ),
            ]
            modality_masks = torch.tensor(
                [[supported] * len(names)], device=state.device, dtype=torch.bool
            )
        if not modality_masks.gather(-1, modality_targets[..., None]).all():
            raise ValueError("Supervised modality is illegal under the public capability mask")
        modality_logits = self.tracks.modality_logits(state, names, control, modality_masks)
        sums["modality"] = torch.nn.functional.cross_entropy(
            modality_logits.flatten(0, 1), modality_targets.flatten(), reduction="sum"
        )
        counts["modality"] = len(names)
        sums["relation_semantic"] = relation_loss
        counts["relation_semantic"] = relation_count
        sums["facet_prediction"] = correspondence
        counts["facet_prediction"] = target_raw.numel() if mode == "bootstrap" else 0

        alignment = zero
        native_count = 0
        if mode == "recurrent":
            for index, plan in enumerate(plans):
                tape = (
                    plan.native_tapes[0]
                    if isinstance(plan, PreparedTargets)
                    else plan.native_tape
                    if isinstance(plan, (PreparedAction, MediaTrainTarget))
                    else None
                )
                if tape is not None:
                    actual = tape.single_raw(state.device)
                    alignment = alignment + (target_raw[:, index].float() - actual).square().sum()
                    native_count += 1
        sums["encoder_alignment"] = alignment
        counts["encoder_alignment"] = native_count * core.config.n_facets * core.config.facet_dim
        balance, z_loss = core.routing_aux_loss(clear=True)
        sums["router_balance"] = balance * len(plans)
        sums["router_z"] = z_loss * len(plans)
        counts["router_balance"] = counts["router_z"] = len(plans)

        def planned(plan, name):
            if not isinstance(plan, MediaTrainTarget):
                return plan.denominators[name]
            # A resolved media target carries no prepared denominators; they
            # mirror training.meridians.MediaPlan.denominators.
            family = "audio" if plan.modality == "audio" else "visual"
            if name in (
                f"{family}_prediction",
                f"{family}_reconstruction",
                "router_balance",
                "router_z",
            ):
                return 1
            if (
                name == "encoder_alignment"
                and plan.mode == "recurrent"
                and plan.native_tape is not None
            ):
                return core.config.n_facets * core.config.facet_dim
            if name == "facet_prediction" and plan.mode == "bootstrap":
                return core.config.n_facets * core.config.facet_dim
            return 0

        expected = {name: sum(planned(plan, name) for plan in plans) for name in LOSS_NAMES}
        expected["modality"] += len(names)
        expected["relation"] += len(relations)
        expected["relation_semantic"] += len(target_relations) + len(evidence_relations)
        if expected != counts:
            raise RuntimeError(f"Grouped loss plan drift: expected {expected}, observed {counts}")
        losses = {name: (sums[name], counts[name]) for name in LOSS_NAMES}
        metrics: dict[str, Tensor | int | float] = {
            "modality_events": (modality_masks.sum(-1) > 1).sum().detach(),
            "modality_correct": (
                (modality_logits.argmax(-1) == modality_targets) & (modality_masks.sum(-1) > 1)
            )
            .sum()
            .detach(),
            "facet_examples": len(names),
            "prior_examples": len(names),
            "reconstruction_examples": len(names),
            "reconstruction_direct_examples": len(names),
            "reconstruction_contextual_examples": 0,
            "prior_lexical_positions": counts["lexical"],
            "reconstruction_lexical_positions": counts["reconstruction"],
            "prior_lexical_loss_sum": sums["lexical"].detach(),
            "reconstruction_lexical_loss_sum": sums["reconstruction"].detach(),
            "facet_prediction_loss_sum": correspondence.detach(),
            "facet_prediction_coordinates": counts["facet_prediction"],
            "encoder_alignment_loss_sum": alignment.detach(),
            "encoder_alignment_coordinates": counts["encoder_alignment"],
            "relation_semantic_loss_sum": relation_loss.detach(),
            "relation_semantic_pairs": relation_count,
            "prediction_facet_norm_sum": mean.detach().float().flatten(2).norm(dim=-1).sum(),
            "encoded_facet_norm_sum": target_raw.detach().float().flatten(2).norm(dim=-1).sum(),
            "facet_alignment_squared_error_sum": (
                target_raw.detach().float() - mean[:, indices].detach().float()
            )
            .square()
            .sum(),
            "facet_alignment_coordinates": target_raw.numel(),
            "teacher_forced_action_continuations": int(
                mode == "recurrent" and teacher_forced_action
            ),
        }
        return GroupSupervisedResult(next_state, losses, metrics)

    @torch.no_grad()
    def select_group(
        self,
        state: MultiTrackState,
        *,
        tracks: Sequence[str],
        event_ids: Sequence[str] | None = None,
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        modality_options: Sequence[Sequence[ActionSpec | MediaOutputSpec]] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> SelectedGroup:
        self.tracks._validate(state)
        if state.batch_size != 1:
            raise ValueError("Grouped serving requires one independent session")
        names = self.tracks._names(tracks, len(tracks))
        identities = self.tracks._event_ids(names, event_ids)
        if event_ids is None:
            identities = tuple(
                f"selected:{int(state.meridians_seen[0])}:{name}" for name in identities
            )
        payloads = tuple(controls) if controls is not None else (b"",) * len(names)
        if len(payloads) != len(names) or any(not isinstance(x, bytes) for x in payloads):
            raise ValueError("Every track needs a byte-exact causal control")
        plans = self._specs(specs, len(names), self.generator.config.max_capacity)
        pack_fingerprints = self._pack_ids(self.generator, plans)
        for plan in plans:
            if isinstance(plan, MediaOutputSpec):
                validate_public_media_spec(plan)
        core = self.generator.core
        core.routing_aux_loss(clear=True)
        encoded = self._controls(payloads, state, names)
        choices = (
            tuple((p,) for p in plans)
            if modality_options is None
            else tuple(tuple(x) for x in modality_options)
        )
        if len(choices) != len(names) or any(not x for x in choices):
            raise ValueError("Each event needs at least one supported output option")
        masks = torch.zeros(1, len(names), 4, dtype=torch.bool, device=state.device)
        for index, options in enumerate(choices):
            seen = set()
            for option in options:
                kind = "text" if isinstance(option, ActionSpec) else option.modality
                if kind in seen:
                    raise ValueError("Only one output specification per modality is allowed")
                seen.add(kind)
                if isinstance(option, MediaOutputSpec):
                    validate_public_media_spec(option)
                    self.generator.media_realizers.resident(kind)  # readiness before selection
                masks[0, index, MODALITY_IDS[kind]] = True
        logits = self.tracks.modality_logits(state, names, encoded, masks)
        # Canonical event order assigns categorical draws reproducibly too.
        order = sorted(range(len(identities)), key=lambda i: identities[i])
        selected_kinds = torch.empty(len(names), dtype=torch.long, device=state.device)
        for index in order:
            selected_kinds[index] = torch.multinomial(
                logits[0, index].softmax(-1), 1, generator=generator
            )[0]
        modality_densities = (
            logits.log_softmax(-1).gather(-1, selected_kinds[None, :, None]).squeeze(-1)
        )
        plans = tuple(
            next(
                p
                for p in options
                if MODALITY_IDS["text" if isinstance(p, ActionSpec) else p.modality] == int(kind)
            )
            for options, kind in zip(choices, selected_kinds, strict=True)
        )
        pack_fingerprints = self._pack_ids(self.generator, plans)
        modalities = tuple("text" if isinstance(x, ActionSpec) else x.modality for x in plans)
        mean = self.tracks.predict_group(
            state,
            tracks=names,
            event_ids=identities,
            control=encoded,
            effort=effort,
            modalities=modalities,
        )
        gaussian = self._prior(mean, state, names, encoded)
        raw = self._sample_policy(gaussian, identities, generator)
        roles = (ROLE_IDS["assistant"],) * len(names)
        seeds = self._renderer_seeds(
            raw,
            identities,
            int(state.meridians_seen[0]),
            tuple(self._spec_id(spec) for spec in plans),
        )
        committed = self.tracks.advance_group(
            state,
            raw,
            track_semantics=encoded,
            tracks=names,
            event_ids=identities,
            role_ids=torch.tensor([roles], device=state.device, dtype=torch.long),
            modalities=modalities,
        )
        event_densities = gaussian.elementwise_log_prob(raw).sum(dim=(-1, -2))
        density = event_densities.sum() + modality_densities.sum()
        tape = GroupActionTape(
            version=GROUP_TAPE_VERSION,
            meridian_id=int(state.meridians_seen[0]),
            tracks=names,
            event_ids=identities,
            render_status="selected",
            track_descriptions=tuple(d for d in state.track_descriptions if d[0] in names),
            policy_context=state.policy_history,
            context_policy_json=state.context_policy.to_json(),
            modality_masks=tuple(tuple(row) for row in masks[0].tolist()),
            modality_log_probs=tuple(float(x) for x in modality_densities[0]),
            roles=roles,
            controls=payloads,
            modalities=tuple("text" if isinstance(x, ActionSpec) else x.modality for x in plans),
            spec_fingerprints=tuple(self._spec_id(x) for x in plans),
            codec_fingerprint=self.generator.codec.fingerprint,
            facet_shape=tuple(raw.shape[1:]),
            raw_f32=raw[0]
            .detach()
            .float()
            .contiguous()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes(),
            behavior_log_prob=float(density),
            effort=effort,
            event_log_probs=tuple(float(x) for x in event_densities[0]),
            renderer_seeds=seeds,
            pack_fingerprints=pack_fingerprints,
            payloads=(b"",) * len(names),
            emitted_ids=((),) * len(names),
            valid=(False,) * len(names),
            terminal=(False,) * len(names),
        )
        core.routing_aux_loss(clear=True)
        return SelectedGroup(
            state,
            committed,
            raw.detach().clone(),
            plans,
            None if encoded is None else encoded.detach().clone(),
            tape,
            density,
        )

    @torch.no_grad()
    def render_selected(self, selected: SelectedGroup) -> GroupGeneration:
        """Render from the selected snapshot, independent of later session input."""
        self.tracks._validate(selected.before)
        tape = selected.tape
        output, emitted, valid, terminal, media = self._realize(
            selected.raw,
            selected.specs,
            selected.control,
            seeds=tape.renderer_seeds,
            mixture=selected.before.adapters,
            track_ids=self.tracks._track_indices(selected.before, tape.tracks)[0],
        )
        self.tracks._validate(selected.before)
        rendered = replace(
            tape,
            payloads=output,
            emitted_ids=emitted,
            valid=valid,
            terminal=terminal,
            render_status="rendered",
        )
        return GroupGeneration(selected.state, rendered, selected.log_prob, media)

    @torch.no_grad()
    def generate_group(self, state: MultiTrackState, **kwargs) -> GroupGeneration:
        return self.render_selected(self.select_group(state, **kwargs))

    def commit_tape(self, state: MultiTrackState, tape: GroupActionTape) -> MultiTrackState:
        """Reconstruct one complete committed selection under current weights.

        No prediction, decoder or random draw is repeated. The public purpose
        revision and event controls are restored before the shared C transition.
        Failed rendering and empty physical output still have a selected state.
        """
        state = self.tracks.set_context_policy(
            state, ContextPolicy.from_json(tape.context_policy_json)
        )
        for address, purpose, revision in tape.track_descriptions:
            state = self.tracks.describe_track(state, address, purpose, revision)
        control = self._controls(tape.controls, state, tape.tracks)
        return self.tracks.advance_group(
            state,
            tape.raw(state.device),
            tracks=tape.tracks,
            event_ids=tape.event_ids,
            track_semantics=control,
            role_ids=torch.tensor([tape.roles], device=state.device, dtype=torch.long),
            modalities=tape.modalities,
        )

    def score_group(self, state: MultiTrackState, tape: GroupActionTape) -> torch.Tensor:
        """Current joint policy density of an immutable historical selection.

        Unlike exact replay this is differentiable, accepts changed weights and
        never renders. The recorded capability mask is part of the public state;
        renderer success, playback and completion timing are not policy draws.
        """
        self.tracks._validate(state)
        if state.context_policy.to_json() != tape.context_policy_json:
            raise ValueError("Policy scoring requires the recorded context window policy")
        if state.batch_size != 1 or int(state.meridians_seen[0]) != tape.meridian_id:
            raise ValueError("Policy scoring requires the original causal meridian")
        if tape.codec_fingerprint != self.generator.codec.fingerprint:
            raise ValueError("Policy scoring codec identity mismatch")
        for address, purpose, revision in tape.track_descriptions:
            state = self.tracks.describe_track(state, address, purpose, revision)
        encoded = self._controls(tape.controls, state, tape.tracks)
        masks = torch.tensor([tape.modality_masks], dtype=torch.bool, device=state.device)
        kinds = torch.tensor([[MODALITY_IDS[m] for m in tape.modalities]], device=state.device)
        if not masks.gather(-1, kinds[..., None]).all():
            raise ValueError("Recorded modality is illegal under its public capability mask")
        logits = self.tracks.modality_logits(state, tape.tracks, encoded, masks)
        categorical = logits.log_softmax(-1).gather(-1, kinds[..., None]).sum()
        mean = self.tracks.predict_group(
            state,
            tracks=tape.tracks,
            event_ids=tape.event_ids,
            control=encoded,
            effort=tape.effort,
            modalities=tape.modalities,
        )
        prior = self._prior(mean, state, tape.tracks, encoded)
        return prior.elementwise_log_prob(tape.raw(state.device)).sum() + categorical

    @torch.no_grad()
    def replay_group(
        self,
        state: MultiTrackState,
        tape: GroupActionTape,
        *,
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        effort: int = 6,
        atol: float = 1e-4,
        rtol: float = 1e-5,
    ) -> GroupGeneration:
        if tape.render_status != "rendered":
            raise ValueError("Replay of rendered output requires a completed rendering outcome")
        self.tracks._validate(state)
        if state.context_policy.to_json() != tape.context_policy_json:
            raise ValueError("Group replay requires the recorded context window policy")
        if state.batch_size != 1 or int(state.meridians_seen[0]) != tape.meridian_id:
            raise ValueError("Group replay requires the original causal meridian")
        if effort != tape.effort:
            raise ValueError("Grouped replay effort must equal the recorded selection effort")
        if tape.codec_fingerprint != self.generator.codec.fingerprint:
            raise ValueError("Group replay codec identity mismatch")
        plans = self._specs(specs, len(tape.tracks), self.generator.config.max_capacity)
        modality_names = tuple("text" if isinstance(x, ActionSpec) else x.modality for x in plans)
        expected_pack_ids = self._pack_ids(self.generator, plans)
        if tape.pack_fingerprints != expected_pack_ids:
            raise ValueError("Grouped tape realizer identity mismatch")
        if (
            tuple(self._spec_id(x) for x in plans) != tape.spec_fingerprints
            or modality_names != tape.modalities
        ):
            raise ValueError("Grouped tape output specifications differ")
        core = self.generator.core
        core.routing_aux_loss(clear=True)
        encoded = self._controls(tape.controls, state, tape.tracks)
        mean = self.tracks.predict_group(
            state,
            tracks=tape.tracks,
            event_ids=tape.event_ids,
            control=encoded,
            effort=effort,
            modalities=tape.modalities,
        )
        raw = tape.raw(state.device)
        if tuple(raw.shape[1:]) != tuple(mean.shape[1:]):
            raise ValueError("Grouped tape raw shape differs from model geometry")
        gaussian = self._prior(mean, state, tape.tracks, encoded)
        event_densities = gaussian.elementwise_log_prob(raw).sum(dim=(-1, -2))
        masks = torch.tensor([tape.modality_masks], dtype=torch.bool, device=state.device)
        kinds = torch.tensor([[MODALITY_IDS[m] for m in tape.modalities]], device=state.device)
        logits = self.tracks.modality_logits(state, tape.tracks, encoded, masks)
        modality_densities = logits.log_softmax(-1).gather(-1, kinds[..., None]).squeeze(-1)
        if not torch.allclose(
            modality_densities[0],
            torch.tensor(tape.modality_log_probs, device=state.device),
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError("Grouped modality choice replay mismatch")
        density = event_densities.sum() + modality_densities.sum()
        if not torch.allclose(
            event_densities[0].float(),
            torch.tensor(tape.event_log_probs, device=state.device),
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError("Grouped per-event policy density replay mismatch")
        if not torch.isclose(
            density.float(),
            torch.tensor(tape.behavior_log_prob, device=state.device),
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError("Grouped policy density replay mismatch")
        seeds = self._renderer_seeds(
            raw,
            tape.event_ids,
            tape.meridian_id,
            tape.spec_fingerprints,
        )
        if tape.renderer_seeds != seeds:
            raise ValueError("Grouped tape renderer noise identity mismatch")
        output, emitted, valid, terminal, media = self._realize(
            raw,
            plans,
            encoded,
            seeds=seeds,
            mixture=state.adapters,
            track_ids=self.tracks._track_indices(state, tape.tracks)[0],
        )
        committed = self.tracks.advance_group(
            state,
            raw,
            track_semantics=encoded,
            tracks=tape.tracks,
            event_ids=tape.event_ids,
            role_ids=torch.tensor([tape.roles], device=state.device, dtype=torch.long),
            modalities=tape.modalities,
        )
        if (output, emitted, valid, terminal) != (
            tape.payloads,
            tape.emitted_ids,
            tape.valid,
            tape.terminal,
        ):
            raise ValueError("Grouped realization replay mismatch")
        core.routing_aux_loss(clear=True)
        return GroupGeneration(committed, tape, density, media)
