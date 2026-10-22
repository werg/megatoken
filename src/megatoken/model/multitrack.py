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
import weakref
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
from megatoken.tensors import byte_tensor, is_host

from .attention import AttentionFeatures, join_attention
from .config import RESERVED_TRACKS
from .core import EFFORTS, MODALITY_IDS, P_HISTORY_READ_STAGES, ROLE_IDS, FacetedCore
from .media import MediaOutputSpec, MediaTrainTarget, validate_public_media_spec
from .relations import KINDS, SelectedRelation, semantic_relation_loss
from .routing import RowRoute
from .state import KVHistory, assert_cache_version

if TYPE_CHECKING:
    from megatoken.generation.policy import FacetedGenerator

GROUP_TAPE_VERSION = "fmt-multitrack-facets-v12"


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
        if (
            len(state.track_registry) > core.config.max_tracks
            or len(set(state.track_registry)) != len(state.track_registry)
            or any(
                name not in state.track_registry and name not in RESERVED_TRACKS
                for name in state.frontier_tracks
            )
        ):
            raise ValueError("Track registry is inconsistent with the shared context")
        if state.frontier_roles.shape != state.raw.shape[:2]:
            raise ValueError("Frontier role metadata has the wrong shape")
        if sum(state.meridian_sizes) * f != state.history.slots:
            raise ValueError("History and retained meridian sizes disagree")
        if state.history_tracks.shape != (state.batch_size, sum(state.meridian_sizes)):
            raise ValueError("Retained track metadata must match whole KV event blocks")
        if state.history.slots > core.config.max_history_chunks * f:
            raise ValueError("History exceeds its full-block capacity")
        if state.final is None and state.meridian_sizes:
            raise ValueError("Only initial state may lack processed features")

    @staticmethod
    def _names(tracks: Sequence[str], count: int) -> tuple[str, ...]:
        names = tuple(tracks)
        if (
            count < 1
            or len(names) != count
            or len(set(names)) != count
            or any(not s for s in names)
        ):
            raise ValueError("One meridian needs unique nonempty event track identities")
        return names

    def _track_indices(
        self, state: MultiTrackState, names: Sequence[str]
    ) -> tuple[Tensor, tuple[str, ...]]:
        registry = list(state.track_registry)
        indices = []
        for name in names:
            if name in RESERVED_TRACKS:
                # Internal query/knowledge/hindsight/metadata never consume user slots.
                indices.append(self.core.config.max_tracks + RESERVED_TRACKS.index(name))
                continue
            if name not in registry:
                if len(registry) >= self.core.config.max_tracks:
                    raise ValueError("Too many tracks for configured session capacity")
                registry.append(name)
            indices.append(registry.index(name))
        values = torch.tensor([indices], device=state.device, dtype=torch.long).expand(
            state.batch_size, -1
        )
        return values, tuple(registry)

    def _history_for(
        self, state: MultiTrackState, new_events: int
    ) -> tuple[KVHistory, tuple[int, ...]]:
        """Evict complete oldest meridians, never a fraction of a selected event."""
        budget = self.core.config.max_history_chunks
        if new_events > budget:
            raise ValueError("One meridian exceeds the global historical event capacity")
        sizes = list(state.meridian_sizes)
        count = sum(sizes)
        history = state.history
        to_drop = 0
        while count + new_events > budget:
            oldest = sizes.pop(0)
            count -= oldest
            to_drop += oldest * self.core.config.n_facets
        if to_drop:
            history = KVHistory(
                history.keys[:, :, to_drop:],
                history.values[:, :, to_drop:],
                None if history.mask is None else history.mask[:, to_drop:],
            )
        return history, tuple(sizes)

    def advance_group(
        self,
        state: MultiTrackState,
        raw: Tensor,
        *,
        tracks: Sequence[str],
        role_ids: Tensor,
        modalities: Sequence[str] | None = None,
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
        # C reads the full causal pre-state. Whole-meridian eviction is done
        # at publication, after this incoming group has read older context.
        track_ids, registry = self._track_indices(state, names)
        modality_ids = torch.tensor(
            [[MODALITY_IDS[x] for x in modality_names]], device=state.device, dtype=torch.long
        ).expand(state.batch_size, -1)
        final = self.core.central(
            raw.flatten(1, 2),
            state.history,
            role_ids,
            state.meridians_seen,
            track_ids=track_ids,
            modality_ids=modality_ids,
            history_track_ids=state.history_tracks.repeat_interleave(c.n_facets, dim=1),
        )
        history, retained = self._history_for(state, events)
        next_history = self.core.central.write(final, history, state.meridians_seen)
        kept = sum(retained)
        old_tracks = state.history_tracks[:, -kept:] if kept else state.history_tracks[:, :0]
        next_history_tracks = torch.cat((old_tracks, track_ids), dim=1)
        shape = (state.batch_size, events, c.n_facets, c.facet_dim)
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
        raws = []
        with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
            for event in events:
                if event.modality == "text":
                    values, lengths = byte_tensor([event.payload], state.device)
                    encoded = self.core.encode_chunk(values, lengths=lengths)
                else:
                    if event.payload.device != state.device:
                        raise ValueError("Media event must be on the model device")
                    encoded = self.core.encode_media(
                        event.payload, event.valid_mask, extractor=event.extractor
                    )
                raws.append(encoded)
        return self.advance_group(
            state,
            torch.stack(raws, dim=1),
            tracks=tracks,
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
            return self.core.remix(final)
        return self.core.remix(state.final.mean(dim=1))

    def predict_group(
        self,
        state: MultiTrackState,
        *,
        tracks: Sequence[str],
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
        if events > self.core.config.max_history_chunks:
            raise ValueError("One meridian exceeds the global historical event capacity")
        batch, f, d = state.batch_size, self.core.config.n_facets, self.core.config.facet_dim
        if control is not None and (
            control.shape != (batch, events, self.core.config.control_dim)
            or control.device != state.device
        ):
            raise ValueError("Expected one control vector per requested event")
        base = self._predict_seed(state)
        track_ids, _ = self._track_indices(state, tracks)
        modality_names = tuple(modalities) if modalities is not None else ("text",) * events
        if len(modality_names) != events or any(x not in MODALITY_IDS for x in modality_names):
            raise ValueError("Unsupported prediction modality descriptor")
        modality_ids = torch.tensor(
            [[MODALITY_IDS[x] for x in modality_names]], device=state.device, dtype=torch.long
        ).expand(batch, -1)
        facets = base[:, None].expand(-1, events, -1, -1).clone()
        facets = facets + self.core.central.modality_embedding(modality_ids)[:, :, None]
        if control is not None:
            facets = facets + self.core.control_injection(control.to(facets.dtype))[:, :, None]
        # Track names are request descriptors, not ordinal positions.
        # Attention separately learns equality of stable session track IDs.
        if events > 1:
            track_bytes = [b"<track>\n" + name.encode("utf-8") for name in tracks]
            track_control = encode_controls(self.core, track_bytes, encoder_grad=False)
            assert track_control is not None
            offsets = self.core.control_injection(track_control.to(facets.dtype))
            facets = facets + offsets[None, :, None]
        facets = facets + self.core.effort_embedding.weight[EFFORTS.index(effort)]
        stages = tuple(range(1, self.core.config.prediction_layers + 1))
        nrows = sum(self.core._row_counts(f, effort))
        routes = self.core.p_router.compile(
            facets.mean(dim=2).reshape(batch * events, d),
            stages,
            nrows,
            epoch="prediction",
        )
        self.core._record_routing_loss(routes.balance_loss, routes.stabilization_loss)
        for stage in stages:
            route = routes.at(stage)
            facets = self.core.run_pair_stage(
                facets,
                stage,
                route,
                effort,
                lambda value, stage=stage, route=route: self._stage(
                    value, stage, route, effort, state, track_ids
                ),
            )
        return facets

    def _stage(
        self,
        facets: Tensor,
        stage: int,
        route: RowRoute,
        effort: int,
        state: MultiTrackState,
        track_ids: Tensor,
    ) -> Tensor:
        core = self.core
        batch, events, f, d = facets.shape
        wrapper = core.stages[stage - 1]
        facets = facets + core.stage_codes[stage - 1]
        joined = facets.reshape(batch, events * f, d)
        addresses = core.central.slot_codes().repeat(events, 1)
        slot_tracks = track_ids.repeat_interleave(f, dim=1)
        time = state.meridians_seen if stage in P_HISTORY_READ_STAGES else None
        current_kv = core.central.memory(
            joined,
            temporal_positions=time,
            slot_codes=addresses,
            track_ids=slot_tracks,
        )
        history = (
            AttentionFeatures(
                state.history.keys,
                state.history.values,
                state.history.mask,
                state.history_tracks.repeat_interleave(f, dim=1),
            )
            if time is not None
            else None
        )
        keys = join_attention(current_kv, history)
        mixed = wrapper.attention.read(
            joined,
            keys,
            temporal_positions=time,
            slot_codes=addresses,
            query_track_ids=slot_tracks,
        )
        joined = joined + wrapper.attention_gate.sigmoid() * mixed
        individual = core.pair_updates(joined.reshape(batch * events, f, d), stage, route, effort)
        return individual.reshape(batch, events, f, d)


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

    def __post_init__(self) -> None:
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
        if any(k != n for k in lengths) or len(set(self.tracks)) != n:
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
            sum(self.event_log_probs), self.behavior_log_prob, rel_tol=1e-5, abs_tol=1e-3
        ):
            raise ValueError("Per-event policy density does not sum to the grouped action")
        if not np.isfinite(np.frombuffer(self.raw_f32, dtype="<f4")).all():
            raise ValueError("Raw facet tape contains nonfinite coordinates")

    def raw(self, device: torch.device) -> Tensor:
        return (
            torch.from_numpy(np.frombuffer(self.raw_f32, dtype="<f4").copy())
            .to(device=device)
            .reshape(1, *self.facet_shape)
        )

    def to_json(self) -> str:
        value = dict(
            version=self.version,
            meridian_id=self.meridian_id,
            tracks=self.tracks,
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
class GroupSupervisedResult:
    """One differentiable multitrack transaction with independent loss units."""

    state: MultiTrackState
    losses: dict[str, tuple[Tensor, int]]
    metrics: dict[str, Tensor | int | float]


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
    ) -> dict[str, tuple[Tensor, int]]:
        if raw.ndim != 4 or raw.shape[1] != len(plans):
            raise ValueError("Realization needs one canonical facet block per target")
        zero = raw.float().sum() * 0
        names = (*DECODER_LOSSES, "audio", "visual")
        sums = {name: zero for name in names}
        counts = {name: 0 for name in names}
        for index, plan in enumerate(plans):
            selected = raw[:, index]
            context = self.core.prepare_realization(selected)
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
    ) -> dict[str, tuple[Tensor, int]]:
        return self.losses_for_selected(raw, plans, control)


class MultiTrackGenerator:
    """One shared P/C/Gaussian, with modality-specific deterministic realization."""

    _shared: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    @classmethod
    def of(cls, generator: FacetedGenerator) -> MultiTrackGenerator:
        """The one multitrack view of a model, so per-version caches outlive a window."""
        if generator not in cls._shared:
            cls._shared[generator] = cls(generator)
        return cls._shared[generator]

    def __init__(self, generator: FacetedGenerator) -> None:
        self.generator = generator
        self.tracks = MultiTrackCore(generator.core)
        self._frozen: tuple[int, dict[str, Tensor]] | None = None

    def _frozen_readout(self, readout: nn.Module) -> dict[str, Tensor]:
        """Detached views of every readout parameter, built once per model version.

        The views alias the live parameters, so optimizer updates show through;
        only a new version (or new parameter objects) needs a new dict.
        """
        version = self.generator.core.model_version
        if self._frozen is None or self._frozen[0] != version:
            self._frozen = (
                version,
                {name: parameter.detach() for name, parameter in readout.named_parameters()},
            )
        return self._frozen[1]

    @staticmethod
    def _sample_policy(
        distribution: DiagonalGaussian,
        names: Sequence[str],
        generator: torch.Generator | None,
    ) -> Tensor:
        """Assign RNG draws by canonical track identity, not physical slot order."""
        order = sorted(range(len(names)), key=lambda index: names[index])
        sorted_distribution = DiagonalGaussian(
            distribution.mean[:, order], distribution.log_scale[:, order]
        )
        selected = sorted_distribution.sample(generator)
        inverse = [order.index(index) for index in range(len(order))]
        return selected[:, inverse]

    def _controls(self, payloads: Sequence[bytes], device: torch.device) -> Tensor | None:
        encoded = encode_controls(self.generator.core, payloads, encoder_grad=False)
        return None if encoded is None else encoded[None].to(device)

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
        for index, spec in enumerate(specs):
            selected = raw[:, index]
            context = self.generator.core.prepare_realization(selected)
            if isinstance(spec, ActionSpec):
                decoded = self.generator.decoder(self.generator.core, context, spec)
                payloads.append(decoded.payload)
                ids.append(decoded.emitted_ids)
                validity.append(decoded.valid)
                terminals.append(decoded.terminal)
                media.append(None)
            else:
                validate_public_media_spec(spec)
                materializer = None
                if seeds is not None:
                    materializer = torch.Generator(device=raw.device)
                    materializer.manual_seed(seeds[index])
                tensor = self.generator.media_realizers.realize(
                    context, spec, generator=materializer
                )
                native = (
                    tensor[0].detach().float().contiguous().cpu().numpy().astype("<f4", copy=False)
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
        plans: Sequence[PreparedTargets | PreparedAction | MediaTrainTarget],
        control_payloads: Sequence[bytes] | None = None,
        effort: int = 6,
        teacher_forced_action: bool = False,
        generator: torch.Generator | None = None,
        relations: Sequence[tuple[str, int, int, float]] = (),
        semantic_relations: Sequence[SelectedRelation] = (),
    ) -> GroupSupervisedResult:
        """One production grouped objective with bootstrap or live recurrent state.

        Every target's raw facets are encoded *after* P selects its prior.
        Executed-action continuations use genuine saved native draws where
        present and detached E(recorded bytes) only as an explicit fallback.
        """
        self.tracks._validate(state)
        if state.batch_size != 1:
            raise ValueError("Grouped supervised meridians require one independent episode")
        names = self.tracks._names(tracks, len(tracks))
        plans = tuple(plans)
        if len(plans) != len(names) or any(
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
        controls = tuple(control_payloads) if control_payloads is not None else (b"",) * len(names)
        if len(controls) != len(names) or any(not isinstance(x, bytes) for x in controls):
            raise ValueError("Every grouped event needs one causal control payload")
        core = self.generator.core
        core.routing_aux_loss(clear=True)
        control = self._controls(controls, state.device)
        mean = self.tracks.predict_group(
            state,
            tracks=names,
            control=control,
            effort=effort,
            modalities=modalities,
        )
        prior = self.generator.facet_distribution(mean)
        sampled = self._sample_policy(prior, names, generator)
        # Only recurrent continuation needs C(sampled); R(sampled) does not.
        selected = (
            self.tracks.advance_group(
                state,
                sampled,
                tracks=names,
                role_ids=roles,
                modalities=modalities,
            )
            if mode == "recurrent"
            else None
        )
        encoded = []
        for plan in plans:
            if isinstance(plan, MediaTrainTarget):
                current = core.encode_media(plan.source, extractor=plan.extractor)
            else:
                payload = plan.payloads[0] if isinstance(plan, PreparedTargets) else plan.payload
                values, lengths = byte_tensor([payload], state.device)
                current = core.encode_chunk(values, lengths)
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
        prior_terms = readout.losses_for_selected(sampled, plans, control)

        if mode == "bootstrap":
            with core.discard_routing_losses():
                recon_terms = readout(codec_sample, plans, control)
            continuation = target_raw
            correspondence = q.detach().kl(prior).sum()
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
                        None if control is None else control.detach(),
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
                        if isinstance(plan, PreparedAction)
                        else None
                    )
                    if tape is None:
                        actual.append(target_raw[:, index].detach())
                    else:
                        actual.append(
                            torch.as_tensor(
                                tape.facet_array().copy(),
                                dtype=torch.float32,
                                device=state.device,
                            ).reshape_as(target_raw[:, index])
                        )
                continuation = torch.stack(actual, dim=1)
            else:
                continuation = sampled
        # Ordinary recurrence keeps its one committed C pass for the next P.
        # Neither prior nor reconstruction realization processes C.
        next_state = (
            selected
            if mode == "recurrent" and not teacher_forced_action
            else self.tracks.advance_group(
                state,
                continuation,
                tracks=names,
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
            semantic_a = sampled[:, left].float().mean(dim=1)
            semantic_b = sampled[:, right].float().mean(dim=1)
            alignment_loss = 1 - torch.nn.functional.cosine_similarity(
                semantic_a,
                semantic_b,
                dim=-1,
                eps=1e-6,
            )
            sums["relation"] = sums["relation"] + strength * alignment_loss.sum()
            counts["relation"] += 1
        semantic_relations = tuple(semantic_relations)
        relation_loss, relation_count = semantic_relation_loss(
            sampled, target_raw, semantic_relations
        )
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
                    if isinstance(plan, PreparedAction)
                    else None
                )
                if tape is not None:
                    actual = torch.as_tensor(
                        tape.facet_array().copy(),
                        dtype=torch.float32,
                        device=state.device,
                    ).reshape_as(target_raw[:, index])
                    alignment = alignment + (target_raw[:, index].float() - actual).square().sum()
                    native_count += 1
        sums["encoder_alignment"] = alignment
        counts["encoder_alignment"] = native_count * core.config.n_facets * core.config.facet_dim
        balance, z_loss = core.routing_aux_loss(clear=True)
        sums["router_balance"] = balance * len(names)
        sums["router_z"] = z_loss * len(names)
        counts["router_balance"] = counts["router_z"] = len(names)

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
            if name == "facet_prediction" and plan.mode == "bootstrap":
                return core.config.n_facets * core.config.facet_dim
            return 0

        expected = {name: sum(planned(plan, name) for plan in plans) for name in LOSS_NAMES}
        expected["relation"] += len(relations)
        expected["relation_semantic"] += len(semantic_relations)
        if expected != counts:
            raise RuntimeError(f"Grouped loss plan drift: expected {expected}, observed {counts}")
        losses = {name: (sums[name], counts[name]) for name in LOSS_NAMES}
        metrics: dict[str, Tensor | int | float] = {
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
                target_raw.detach().float() - mean.detach().float()
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
    def generate_group(
        self,
        state: MultiTrackState,
        *,
        tracks: Sequence[str],
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> GroupGeneration:
        self.tracks._validate(state)
        if state.batch_size != 1:
            raise ValueError("Grouped serving requires one independent session")
        names = self.tracks._names(tracks, len(tracks))
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
        encoded = self._controls(payloads, state.device)
        modalities = tuple("text" if isinstance(x, ActionSpec) else x.modality for x in plans)
        mean = self.tracks.predict_group(
            state,
            tracks=names,
            control=encoded,
            effort=effort,
            modalities=modalities,
        )
        gaussian = self.generator.facet_distribution(mean)
        raw = self._sample_policy(gaussian, names, generator)
        roles = (ROLE_IDS["assistant"],) * len(names)
        seeds = self._renderer_seeds(
            raw,
            names,
            int(state.meridians_seen[0]),
            tuple(self._spec_id(spec) for spec in plans),
        )
        output, emitted, valid, terminal, media = self._realize(
            raw,
            plans,
            encoded,
            seeds=seeds,
        )
        committed = self.tracks.advance_group(
            state,
            raw,
            tracks=names,
            role_ids=torch.tensor([roles], device=state.device, dtype=torch.long),
            modalities=modalities,
        )
        event_densities = gaussian.elementwise_log_prob(raw).sum(dim=(-1, -2))
        density = event_densities.sum()
        tape = GroupActionTape(
            version=GROUP_TAPE_VERSION,
            meridian_id=int(state.meridians_seen[0]),
            tracks=names,
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
            payloads=output,
            emitted_ids=emitted,
            valid=valid,
            terminal=terminal,
        )
        core.routing_aux_loss(clear=True)
        return GroupGeneration(committed, tape, density, media)

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
        self.tracks._validate(state)
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
        encoded = self._controls(tape.controls, state.device)
        mean = self.tracks.predict_group(
            state,
            tracks=tape.tracks,
            control=encoded,
            effort=effort,
            modalities=tape.modalities,
        )
        raw = tape.raw(state.device)
        if tuple(raw.shape[1:]) != tuple(mean.shape[1:]):
            raise ValueError("Grouped tape raw shape differs from model geometry")
        gaussian = self.generator.facet_distribution(mean)
        event_densities = gaussian.elementwise_log_prob(raw).sum(dim=(-1, -2))
        density = event_densities.sum()
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
            tape.tracks,
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
        )
        committed = self.tracks.advance_group(
            state,
            raw,
            tracks=tape.tracks,
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
