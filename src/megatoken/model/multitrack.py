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
from torch import Tensor

from megatoken.controls import encode_controls
from megatoken.generation.latents import DiagonalGaussian
from megatoken.generation.structure import ActionSpec
from megatoken.io import strict_json_loads
from megatoken.tensors import byte_tensor

from .attention import AttentionFeatures, join_attention
from .core import EFFORTS, MODALITY_IDS, P_HISTORY_READ_STAGES, ROLE_IDS, FacetedCore
from .media import MediaOutputSpec
from .routing import RowRoute
from .state import KVHistory, RealizationContext, assert_cache_version

if TYPE_CHECKING:
    from megatoken.generation.policy import FacetedGenerator

GROUP_TAPE_VERSION = "fmt-multitrack-facets-v7"



@dataclass(frozen=True)
class ObservedEvent:
    """One available text or media input to commit in a shared meridian."""

    track: str
    modality: str
    payload: bytes | Tensor
    role_id: int = ROLE_IDS["user"]
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if self.modality not in {"text", "audio", "image", "video"}:
            raise ValueError("Unsupported input modality")
        if self.modality == "text" and not isinstance(self.payload, bytes):
            raise TypeError("Text events must contain bytes")
        if self.modality != "text" and not isinstance(self.payload, Tensor):
            raise TypeError("Media events must contain a tensor")
        if self.role_id not in ROLE_IDS.values():
            raise ValueError("Unknown event role")


@dataclass(frozen=True)
class MultiTrackState:
    """One shared history and the last *complete* meridian of selected events."""

    raw: Tensor  # [B, T, facets, width]
    early: Tensor | None
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
            early=None if self.early is None else self.early.detach(),
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
            early=None,
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
        if (state.early is None) != (state.final is None):
            raise ValueError("Early and final central caches must be paired")
        if state.early is not None and (
            state.early.shape != state.raw.shape or state.final.shape != state.raw.shape
        ):
            raise ValueError("Frontier features must retain every selected facet")
        if state.meridians_seen.shape != (state.batch_size,):
            raise ValueError("Meridian chronology has the wrong batch shape")
        if state.events_seen.shape != (state.batch_size,):
            raise ValueError("Selected-event chronology has the wrong batch shape")
        if state.frontier_modalities and len(state.frontier_modalities) != state.frontier_count:
            raise ValueError("Frontier modality metadata must match selected events")
        if (len(state.track_registry) > core.config.max_tracks
                or len(set(state.track_registry)) != len(state.track_registry)
                or any(name not in state.track_registry for name in state.frontier_tracks)):
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
        if count < 1 or len(names) != count or len(set(names)) != count or any(not s for s in names):
            raise ValueError("One meridian needs unique nonempty event track identities")
        return names

    def _track_indices(
        self, state: MultiTrackState, names: Sequence[str]
    ) -> tuple[Tensor, tuple[str, ...]]:
        registry = list(state.track_registry)
        indices = []
        for name in names:
            if name not in registry:
                if len(registry) >= self.core.config.max_tracks:
                    raise ValueError("Too many tracks for configured session capacity")
                registry.append(name)
            indices.append(registry.index(name))
        values = torch.tensor([indices], device=state.device, dtype=torch.long).expand(
            state.batch_size, -1
        )
        return values, tuple(registry)

    def _history_for(self, state: MultiTrackState, new_events: int) -> tuple[KVHistory, tuple[int, ...]]:
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
        if raw.ndim != 4 or raw.shape[0] != state.batch_size or raw.shape[2:] != (
            c.n_facets,
            c.facet_dim,
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
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Raw selected facets must be finite")
        # C reads the full causal pre-state. Whole-meridian eviction is done
        # at publication, after this incoming group has read older context.
        track_ids, registry = self._track_indices(state, names)
        modality_ids = torch.tensor(
            [[MODALITY_IDS[x] for x in modality_names]], device=state.device, dtype=torch.long
        ).expand(state.batch_size, -1)
        early, final = self.core.central(
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
            early=early.reshape(shape),
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
                    payload = event.payload
                    assert isinstance(payload, Tensor)
                    # A single media event has a single batch dimension.
                    if payload.device != state.device or payload.shape[0] != 1:
                        raise ValueError("Media event must be one batch row on the model device")
                    encoded = self.core.encode_media(
                        event.modality, payload, sample_rate=event.sample_rate
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

    def observe_text_group(
        self,
        state: MultiTrackState,
        payloads: Sequence[bytes],
        *,
        tracks: Sequence[str],
        roles: Sequence[int] | None = None,
        encoder_grad: bool = True,
    ) -> MultiTrackState:
        """Observe simultaneous text events; they share one C/KV transition."""
        payloads, tracks = tuple(payloads), tuple(tracks)
        if len(payloads) != len(tracks):
            raise ValueError("Payloads and tracks must correspond")
        roles = tuple(roles) if roles is not None else (ROLE_IDS["user"],) * len(payloads)
        if len(roles) != len(payloads):
            raise ValueError("One role is required per observed event")
        return self.observe_group(
            state,
            tuple(
                ObservedEvent(track, "text", payload, role_id=role)
                for track, payload, role in zip(tracks, payloads, roles, strict=True)
            ),
            encoder_grad=encoder_grad,
        )

    def _predict_seed(self, state: MultiTrackState) -> Tensor:
        if state.final is None:
            # Match singleton startup: the sentinel is processed locally, not
            # appended to history and not exposed as an observation.
            _, final = self.core.central(
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
            from megatoken.controls import encode_controls

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
            allow_exploration=False,
        )
        self.core._record_routing_loss(routes.balance_loss, routes.stabilization_loss)
        for stage in stages:
            facets = self._stage(facets, stage, routes.at(stage), effort, state, track_ids)
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
            joined, temporal_positions=time, slot_codes=addresses,
            track_ids=slot_tracks,
        )
        history = (
            AttentionFeatures(
                state.history.keys, state.history.values, state.history.mask,
                state.history_tracks.repeat_interleave(f, dim=1),
            )
            if time is not None
            else None
        )
        keys = join_attention(current_kv, history)
        mixed = wrapper.attention.read(
            joined, keys, temporal_positions=time, slot_codes=addresses,
            query_track_ids=slot_tracks,
        )
        joined = joined + wrapper.attention_gate.sigmoid() * mixed
        individual = joined.reshape(batch * events, f, d)
        begin = 0
        for update, count in enumerate(core._row_counts(f, effort)):
            selected = RowRoute(
                route.expert_ids[:, begin : begin + count],
                route.confidence[:, begin : begin + count],
                route.eligible,
                route.palette_probabilities,
                route.balance_loss,
                route.stabilization_loss,
            )
            core.p_bank.record_routes(
                selected.expert_ids,
                route.eligible if update == 0 else None,
                gradient_enabled=torch.is_grad_enabled(),
            )
            individual = wrapper.update(
                individual,
                core.pair_left[stage - 1, update, :count],
                core.pair_right[stage - 1, update, :count],
                selected,
                core.p_bank,
                update,
                core.pair_reduction_indices[stage - 1, update, EFFORTS.index(effort)],
                core.pair_reduction_mask[stage - 1, update, EFFORTS.index(effort)],
                core.pair_reduction_degree[stage - 1, update, EFFORTS.index(effort)],
            )
            begin += count
        return individual.reshape(batch, events, f, d)

    def realization_context(self, state: MultiTrackState, event: int) -> RealizationContext:
        self._validate(state)
        if state.early is None or state.final is None or not 0 <= event < state.frontier_count:
            raise ValueError("Realization requires a selected event from a committed meridian")
        return RealizationContext(state.raw[:, event], state.early[:, event], state.final[:, event])


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
    modalities: tuple[str, ...] = ()
    effort: int = 6
    event_log_probs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.version != GROUP_TAPE_VERSION:
            raise ValueError("Incompatible grouped tape version")
        n, f, d = self.facet_shape
        if not self.modalities:
            object.__setattr__(self, "modalities", ("text",) * n)
        if n < 1 or f < 1 or d < 1 or len(self.raw_f32) != n * f * d * 4:
            raise ValueError("Tape must contain every complete raw FP32 event")
        lengths = (len(self.tracks), len(self.modalities), len(self.roles), len(self.controls),
                   len(self.spec_fingerprints), len(self.payloads),
                   len(self.emitted_ids), len(self.valid))
        if any(k != n for k in lengths) or len(set(self.tracks)) != n:
            raise ValueError("Grouped tape fields must agree on exact event identities")
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
        return torch.from_numpy(np.frombuffer(self.raw_f32, dtype="<f4").copy()).to(
            device=device
        ).reshape(1, *self.facet_shape)

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
            payloads=[base64.b64encode(b).decode("ascii") for b in self.payloads],
            emitted_ids=self.emitted_ids,
            valid=self.valid,
        )
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, serialized: str) -> GroupActionTape:
        value = strict_json_loads(serialized)
        if value.get("version") != GROUP_TAPE_VERSION:
            raise ValueError("Unknown grouped tape version")
        decode = lambda b: base64.b64decode(b, validate=True)
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
            payloads=tuple(map(decode, value["payloads"])),
            emitted_ids=tuple(tuple(s) for s in value["emitted_ids"]),
            valid=tuple(value["valid"]),
        )


@dataclass(frozen=True)
class GroupSupervisedResult:
    """One differentiable multitrack transaction with independent loss units."""

    state: MultiTrackState
    loss: Tensor
    components: dict[str, Tensor]


@dataclass(frozen=True)
class GroupGeneration:
    state: MultiTrackState
    tape: GroupActionTape
    log_prob: Tensor
    media_outputs: tuple[Tensor | None, ...] = ()


class MultiTrackGenerator:
    """One shared P/C/Gaussian, with modality-specific deterministic realization."""

    def __init__(self, generator: FacetedGenerator) -> None:
        self.generator = generator
        self.tracks = MultiTrackCore(generator.core)

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

    def _realize(
        self, state: MultiTrackState, specs: Sequence[ActionSpec | MediaOutputSpec],
        control: Tensor | None,
    ) -> tuple[tuple[bytes, ...], tuple[tuple[int, ...], ...],
               tuple[bool, ...], tuple[Tensor | None, ...]]:
        payloads, ids, validity, media = [], [], [], []
        for index, spec in enumerate(specs):
            context = self.tracks.realization_context(state, index)
            if isinstance(spec, ActionSpec):
                decoded = self.generator.decoder(self.generator.core, context, spec)
                payloads.append(decoded.payload)
                ids.append(decoded.emitted_ids)
                validity.append(decoded.valid)
                media.append(None)
            else:
                vector = None if control is None else control[:, index]
                tensor = self.generator.media_realizers.realize(context, spec, vector)
                payloads.append(tensor[0].detach().float().contiguous().cpu().numpy()
                                .astype("<f4", copy=False).tobytes())
                ids.append(())
                validity.append(True)
                media.append(tensor)
        return tuple(payloads), tuple(ids), tuple(validity), tuple(media)

    def supervised_group(
        self,
        state: MultiTrackState,
        *,
        tracks: Sequence[str],
        targets: Sequence[bytes | Tensor],
        specs: Sequence[ActionSpec | MediaOutputSpec],
        controls: Sequence[bytes] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> GroupSupervisedResult:
        """Bootstrap one whole meridian, sharing P/C and modality E/R parameters.

        The objective is deliberately explicit. Predict from causal history,
        score the realized targets, reconstruct from independent noisy encoder
        facets, and align the detached encoder distribution with the prior.
        This is the supervised bootstrap path; recurrent teacher/action replay
        requires the separate real-trajectory continuation rules.
        """
        self.tracks._validate(state)
        names = self.tracks._names(tracks, len(tracks))
        if len(targets) != len(names) or len(specs) != len(names):
            raise ValueError("One target and realization spec is required per track")
        if state.batch_size != 1:
            raise ValueError("Grouped bootstrap currently supports one independent session")
        control_bytes = tuple(controls) if controls is not None else (b"",) * len(names)
        if len(control_bytes) != len(names):
            raise ValueError("One causal control is required per track")
        selected_specs = self._specs(specs, len(names), self.generator.config.max_capacity)
        for spec, target in zip(selected_specs, targets, strict=True):
            if isinstance(spec, ActionSpec) and not isinstance(target, bytes):
                raise TypeError("Text targets must contain bytes")
            if isinstance(spec, MediaOutputSpec) and not isinstance(target, Tensor):
                raise TypeError("Media targets must be tensors")
        core = self.generator.core
        core.routing_aux_loss(clear=True)
        control = self._controls(control_bytes, state.device)
        modal_names = tuple(
            "text" if isinstance(x, ActionSpec) else x.modality for x in selected_specs
        )
        mean = self.tracks.predict_group(
            state, tracks=names, control=control, effort=effort, modalities=modal_names
        )
        distribution = self.generator.facet_distribution(mean)
        sampled = self._sample_policy(distribution, names, generator)
        roles = torch.full(
            (1, len(names)), ROLE_IDS["assistant"], dtype=torch.long, device=state.device
        )
        predicted = self.tracks.advance_group(
            state, sampled, tracks=names, role_ids=roles, modalities=modal_names
        )

        def losses(realized: MultiTrackState) -> Tensor:
            values = []
            for i, (spec, target) in enumerate(zip(selected_specs, targets, strict=True)):
                ctx = self.tracks.realization_context(realized, i)
                if isinstance(spec, ActionSpec):
                    if spec.schemas:
                        raise ValueError("Typed tool targets require structured target planning")
                    assert isinstance(target, bytes)
                    plan = self.generator.prepare_target(
                        target_bytes=target, representation_mode="bootstrap"
                    )
                    items = self.generator.decoder.teacher_raw(core, ctx, plan)
                    active = [
                        item / max(1, count) for item, count in items.values() if count
                    ]
                    values.append(torch.stack(active).sum())
                else:
                    assert isinstance(target, Tensor)
                    cond = None if control is None else control[:, i]
                    out = self.generator.media_realizers.realize(ctx, spec, cond)
                    if target.shape != out.shape or target.device != out.device:
                        raise ValueError("Media target must match realized shape/device")
                    values.append((out.float() - target.float()).square().mean())
            return torch.stack(values).mean()

        main_loss = losses(predicted)
        # The encoder owns one canonical raw representation per target regardless
        # of how many source codec frames or visual patches it consumed.
        encodings = []
        for spec, target in zip(selected_specs, targets, strict=True):
            if isinstance(spec, ActionSpec):
                ids, lengths = byte_tensor([target], state.device)
                encoded = core.encode_chunk(ids, lengths)
            else:
                encoded = core.encode_media(spec.modality, target)
            encodings.append(encoded)
        encoded_raw = torch.stack(encodings, dim=1)
        tau = self.generator.config.codec_noise_std
        posterior = DiagonalGaussian(
            encoded_raw.float(), torch.full_like(encoded_raw, math.log(tau))
        )
        # KL is already averaged over batch, events, facets and coordinates.
        # Dividing again by one group's latent size almost removes its gradient
        # at production width (32 x 512), defeating bootstrap prior alignment.
        kl = posterior.detach().kl(distribution).mean()
        noise = torch.randn(
            encoded_raw.shape, device=state.device, dtype=torch.float32, generator=generator
        )
        reconstructed = self.tracks.advance_group(
            state, encoded_raw + tau * noise, tracks=names, role_ids=roles,
            modalities=modal_names
        )
        recon_loss = losses(reconstructed)
        # The real live state is the clean teacher-forced encoding, not either
        # alternate sampled/reconstruction branch used to score the objective.
        next_state = self.tracks.advance_group(
            state, encoded_raw, tracks=names, role_ids=roles,
            modalities=modal_names
        )
        router_balance, router_stabilization = core.routing_aux_loss(clear=True)
        routing = router_balance * 0.005 + router_stabilization * 1e-4
        total = main_loss + recon_loss + 0.1 * kl + routing
        return GroupSupervisedResult(
            next_state,
            total,
            {"prediction": main_loss, "reconstruction": recon_loss,
             "facet_kl": kl, "routing": routing},
        )

    @torch.no_grad()
    def generate_group(
        self, state: MultiTrackState, *, tracks: Sequence[str],
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        controls: Sequence[bytes] | None = None, effort: int = 6,
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
        core = self.generator.core
        core.routing_aux_loss(clear=True)
        encoded = self._controls(payloads, state.device)
        modalities = tuple(
            "text" if isinstance(x, ActionSpec) else x.modality for x in plans
        )
        mean = self.tracks.predict_group(
            state, tracks=names, control=encoded, effort=effort, modalities=modalities
        )
        gaussian = self.generator.facet_distribution(mean)
        raw = self._sample_policy(gaussian, names, generator)
        roles = (ROLE_IDS["assistant"],) * len(names)
        committed = self.tracks.advance_group(
            state, raw, tracks=names,
            role_ids=torch.tensor([roles], device=state.device, dtype=torch.long),
            modalities=modalities,
        )
        output, emitted, valid, media = self._realize(committed, plans, encoded)
        event_densities = gaussian.elementwise_log_prob(raw).sum(dim=(-1, -2))
        density = event_densities.sum()
        tape = GroupActionTape(
            version=GROUP_TAPE_VERSION,
            meridian_id=int(state.meridians_seen[0]),
            tracks=names,
            roles=roles,
            controls=payloads,
            modalities=tuple(
                "text" if isinstance(x, ActionSpec) else x.modality for x in plans
            ),
            spec_fingerprints=tuple(self._spec_id(x) for x in plans),
            codec_fingerprint=self.generator.codec.fingerprint,
            facet_shape=tuple(raw.shape[1:]),
            raw_f32=raw[0].detach().float().contiguous().cpu().numpy()
                .astype("<f4", copy=False).tobytes(),
            behavior_log_prob=float(density),
            effort=effort,
            event_log_probs=tuple(float(x) for x in event_densities[0]),
            payloads=output, emitted_ids=emitted, valid=valid,
        )
        core.routing_aux_loss(clear=True)
        return GroupGeneration(committed, tape, density, media)

    @torch.no_grad()
    def replay_group(
        self, state: MultiTrackState, tape: GroupActionTape, *,
        specs: Sequence[ActionSpec | MediaOutputSpec] | None = None,
        effort: int = 6, atol: float = 1e-4, rtol: float = 1e-5,
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
        if (tuple(self._spec_id(x) for x in plans) != tape.spec_fingerprints
                or modality_names != tape.modalities):
            raise ValueError("Grouped tape output specifications differ")
        core = self.generator.core
        core.routing_aux_loss(clear=True)
        encoded = self._controls(tape.controls, state.device)
        mean = self.tracks.predict_group(
            state, tracks=tape.tracks, control=encoded, effort=effort,
            modalities=tape.modalities
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
            atol=atol, rtol=rtol,
        ):
            raise ValueError("Grouped per-event policy density replay mismatch")
        if not torch.isclose(
            density.float(), torch.tensor(tape.behavior_log_prob, device=state.device),
            atol=atol, rtol=rtol,
        ):
            raise ValueError("Grouped policy density replay mismatch")
        committed = self.tracks.advance_group(
            state, raw, tracks=tape.tracks,
            role_ids=torch.tensor([tape.roles], device=state.device, dtype=torch.long),
            modalities=tape.modalities,
        )
        output, emitted, valid, media = self._realize(committed, plans, encoded)
        if (output != tape.payloads or emitted != tape.emitted_ids or valid != tape.valid):
            raise ValueError("Grouped realization replay mismatch")
        core.routing_aux_loss(clear=True)
        return GroupGeneration(committed, tape, density, media)

    def generate_text_group(
        self, state: MultiTrackState, *, tracks: Sequence[str],
        controls: Sequence[bytes] | None = None,
        specs: Sequence[ActionSpec] | None = None,
        effort: int = 6, generator: torch.Generator | None = None,
    ) -> GroupGeneration:
        return self.generate_group(
            state, tracks=tracks, specs=specs, controls=controls,
            effort=effort, generator=generator,
        )

    def replay_text_group(
        self, state: MultiTrackState, tape: GroupActionTape, *,
        specs: Sequence[ActionSpec] | None = None, effort: int = 6,
        atol: float = 1e-4, rtol: float = 1e-5,
    ) -> GroupGeneration:
        return self.replay_group(
            state, tape, specs=specs, effort=effort, atol=atol, rtol=rtol,
        )
