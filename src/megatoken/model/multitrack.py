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
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from megatoken.controls import encode_controls
from megatoken.generation.structure import ActionSpec
from megatoken.tensors import byte_tensor

from .attention import AttentionFeatures, join_attention
from .core import EFFORTS, P_HISTORY_READ_STAGES, ROLE_IDS, FacetedCore
from .routing import RowRoute
from .state import KVHistory, RealizationContext, assert_cache_version

if TYPE_CHECKING:
    from megatoken.generation.policy import FacetedGenerator

GROUP_TAPE_VERSION = "fmt-multitrack-facets-v1"


@dataclass(frozen=True)
class MultiTrackState:
    """One shared history and the last *complete* meridian of selected events."""

    raw: Tensor  # [B, T, facets, width]
    early: Tensor | None
    final: Tensor | None
    history: KVHistory
    meridian_sizes: tuple[int, ...]
    meridians_seen: Tensor  # [B]; logically committed meridian index
    frontier_tracks: tuple[str, ...]
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
            meridian_sizes=(),
            meridians_seen=state.records_seen,
            frontier_tracks=(),
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
        if state.frontier_roles.shape != state.raw.shape[:2]:
            raise ValueError("Frontier role metadata has the wrong shape")
        if sum(state.meridian_sizes) * f != state.history.slots:
            raise ValueError("History and retained meridian sizes disagree")
        if state.history.slots > core.config.max_history_chunks * f:
            raise ValueError("History exceeds its full-block capacity")
        if state.final is None and state.meridian_sizes:
            raise ValueError("Only initial state may lack processed features")

    @staticmethod
    def _names(tracks: Sequence[str], count: int) -> tuple[str, ...]:
        names = tuple(tracks)
        if len(names) != count or len(set(names)) != count or any(not s for s in names):
            raise ValueError("One meridian needs unique nonempty event track identities")
        return names

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
        names = self._names(tracks, events)
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
        history, retained = self._history_for(state, events)
        early, final = self.core.central(
            raw.flatten(1, 2),
            history,
            role_ids,
            state.meridians_seen,
        )
        next_history = self.core.central.write(final, history, state.meridians_seen)
        shape = (state.batch_size, events, c.n_facets, c.facet_dim)
        return MultiTrackState(
            raw=raw,
            early=early.reshape(shape),
            final=final.reshape(shape),
            history=next_history,
            meridian_sizes=(*retained, events),
            meridians_seen=state.meridians_seen + 1,
            frontier_tracks=names,
            frontier_roles=role_ids,
            model_version=state.model_version,
            routing_version=state.routing_version,
            geometry=state.geometry,
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
        self._validate(state)
        if state.batch_size != 1:
            raise ValueError("Text observation API currently serves one session")
        payloads = tuple(payloads)
        self._names(tracks, len(payloads))
        if any(not isinstance(value, bytes) for value in payloads):
            raise TypeError("Text observations must be raw bytes")
        roles = tuple(roles) if roles is not None else (ROLE_IDS["user"],) * len(payloads)
        if len(roles) != len(payloads) or any(role not in ROLE_IDS.values() for role in roles):
            raise ValueError("Unknown observation role")
        values, lengths = byte_tensor(payloads, state.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
            raw = self.core.encode_chunk(values, lengths=lengths)
        return self.advance_group(
            state,
            raw.reshape(1, len(payloads), self.core.config.n_facets, -1),
            tracks=tracks,
            role_ids=torch.tensor([roles], device=state.device, dtype=torch.long),
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
    ) -> Tensor:
        """Read-only *joint* prediction from all previously committed tracks.

        Live prediction-query attention spans all requested tracks. Pair-expert
        topology remains local to each 32-facet block, using the existing bank.
        """
        self._validate(state)
        if effort not in EFFORTS:
            raise ValueError(f"Effort must be one of {EFFORTS}")
        events = len(self._names(tracks, len(tracks)))
        batch, f, d = state.batch_size, self.core.config.n_facets, self.core.config.facet_dim
        if control is not None and (
            control.shape != (batch, events, self.core.config.control_dim)
            or control.device != state.device
        ):
            raise ValueError("Expected one control vector per requested event")
        base = self._predict_seed(state)
        facets = base[:, None].expand(-1, events, -1, -1).clone()
        if control is not None:
            facets = facets + self.core.control_injection(control.to(facets.dtype))[:, :, None]
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
            facets = self._stage(facets, stage, routes.at(stage), effort, state)
        return facets

    def _stage(
        self,
        facets: Tensor,
        stage: int,
        route: RowRoute,
        effort: int,
        state: MultiTrackState,
    ) -> Tensor:
        core = self.core
        batch, events, f, d = facets.shape
        wrapper = core.stages[stage - 1]
        facets = facets + core.stage_codes[stage - 1]
        joined = facets.reshape(batch, events * f, d)
        addresses = core.central.slot_codes().repeat(events, 1)
        time = state.meridians_seen if stage in P_HISTORY_READ_STAGES else None
        current_kv = core.central.memory(
            joined, temporal_positions=time, slot_codes=addresses
        )
        history = (
            AttentionFeatures(state.history.keys, state.history.values, state.history.mask)
            if time is not None
            else None
        )
        keys = join_attention(current_kv, history)
        mixed = wrapper.attention.read(
            joined, keys, temporal_positions=time, slot_codes=addresses
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

    def __post_init__(self) -> None:
        if self.version != GROUP_TAPE_VERSION:
            raise ValueError("Incompatible grouped tape version")
        n, f, d = self.facet_shape
        if n < 1 or f < 1 or d < 1 or len(self.raw_f32) != n * f * d * 4:
            raise ValueError("Tape must contain every complete raw FP32 event")
        lengths = (len(self.tracks), len(self.roles), len(self.controls),
                   len(self.spec_fingerprints), len(self.payloads),
                   len(self.emitted_ids), len(self.valid))
        if any(k != n for k in lengths) or len(set(self.tracks)) != n:
            raise ValueError("Grouped tape fields must agree on exact event identities")
        if not math.isfinite(self.behavior_log_prob):
            raise ValueError("Behavior log probability must be finite")
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
            roles=self.roles,
            controls=[base64.b64encode(b).decode("ascii") for b in self.controls],
            spec_fingerprints=self.spec_fingerprints,
            codec_fingerprint=self.codec_fingerprint,
            facet_shape=self.facet_shape,
            raw_f32=base64.b64encode(self.raw_f32).decode("ascii"),
            behavior_log_prob=self.behavior_log_prob,
            payloads=[base64.b64encode(b).decode("ascii") for b in self.payloads],
            emitted_ids=self.emitted_ids,
            valid=self.valid,
        )
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, serialized: str) -> GroupActionTape:
        value = json.loads(serialized)
        if value.get("version") != GROUP_TAPE_VERSION:
            raise ValueError("Unknown grouped tape version")
        decode = lambda b: base64.b64decode(b, validate=True)
        return cls(
            version=value["version"],
            meridian_id=int(value["meridian_id"]),
            tracks=tuple(value["tracks"]),
            roles=tuple(value["roles"]),
            controls=tuple(map(decode, value["controls"])),
            spec_fingerprints=tuple(value["spec_fingerprints"]),
            codec_fingerprint=value["codec_fingerprint"],
            facet_shape=tuple(value["facet_shape"]),
            raw_f32=decode(value["raw_f32"]),
            behavior_log_prob=float(value["behavior_log_prob"]),
            payloads=tuple(map(decode, value["payloads"])),
            emitted_ids=tuple(tuple(s) for s in value["emitted_ids"]),
            valid=tuple(value["valid"]),
        )


@dataclass(frozen=True)
class GroupGeneration:
    state: MultiTrackState
    tape: GroupActionTape
    log_prob: Tensor


class MultiTrackGenerator:
    """Text-first grouped serving, sharing a FacetedGenerator's trained weights.

    The existing decoder, Gaussian scale, and all core parameters are reused.
    This wrapper does not register, clone, or introduce another model.
    """

    def __init__(self, generator: FacetedGenerator) -> None:
        self.generator = generator
        self.tracks = MultiTrackCore(generator.core)

    def _controls(self, payloads: Sequence[bytes], device: torch.device) -> Tensor | None:
        encoded = encode_controls(self.generator.core, payloads, encoder_grad=False)
        return None if encoded is None else encoded[None].to(device)

    @staticmethod
    def _specs(specs: Sequence[ActionSpec] | None, count: int, capacity: int) -> tuple[ActionSpec, ...]:
        if specs is None:
            return (ActionSpec(max_tokens=min(512, capacity)),) * count
        values = tuple(specs)
        if len(values) != count:
            raise ValueError("One public output specification is required per event")
        return values

    @torch.no_grad()
    def generate_text_group(
        self,
        state: MultiTrackState,
        *,
        tracks: Sequence[str],
        controls: Sequence[bytes] | None = None,
        specs: Sequence[ActionSpec] | None = None,
        effort: int = 6,
        generator: torch.Generator | None = None,
    ) -> GroupGeneration:
        if state.batch_size != 1:
            raise ValueError("Grouped text serving requires one independent session")
        names = self.tracks._names(tracks, len(tracks))
        controls = tuple(controls) if controls is not None else (b"",) * len(names)
        if len(controls) != len(names):
            raise ValueError("Controls must correspond to output events")
        specifications = self._specs(specs, len(names), self.generator.config.max_capacity)
        self.generator.core.routing_aux_loss(clear=True)
        mean = self.tracks.predict_group(
            state, tracks=names, control=self._controls(controls, state.device), effort=effort
        )
        distribution = self.generator.facet_distribution(mean)
        raw = distribution.sample(generator)
        roles = (ROLE_IDS["assistant"],) * len(names)
        committed = self.tracks.advance_group(
            state, raw, tracks=names,
            role_ids=torch.tensor([roles], device=state.device, dtype=torch.long),
        )
        decoded = [
            self.generator.decoder(
                self.generator.core, self.tracks.realization_context(committed, index), spec
            )
            for index, spec in enumerate(specifications)
        ]
        density = distribution.log_prob(raw).sum()
        raw_bytes = raw[0].detach().float().contiguous().cpu().numpy().astype(
            "<f4", copy=False
        ).tobytes()
        tape = GroupActionTape(
            version=GROUP_TAPE_VERSION,
            meridian_id=int(state.meridians_seen[0]),
            tracks=names,
            roles=roles,
            controls=controls,
            spec_fingerprints=tuple(spec.fingerprint for spec in specifications),
            codec_fingerprint=self.generator.codec.fingerprint,
            facet_shape=tuple(raw.shape[1:]),
            raw_f32=raw_bytes,
            behavior_log_prob=float(density),
            payloads=tuple(v.payload for v in decoded),
            emitted_ids=tuple(v.emitted_ids for v in decoded),
            valid=tuple(v.valid for v in decoded),
        )
        self.generator.core.routing_aux_loss(clear=True)
        return GroupGeneration(committed, tape, density)

    @torch.no_grad()
    def replay_text_group(
        self,
        state: MultiTrackState,
        tape: GroupActionTape,
        *,
        specs: Sequence[ActionSpec] | None = None,
        effort: int = 6,
        atol: float = 1e-4,
        rtol: float = 1e-5,
    ) -> GroupGeneration:
        if int(state.meridians_seen[0]) != tape.meridian_id:
            raise ValueError("Grouped tape belongs to another causal meridian")
        if tape.codec_fingerprint != self.generator.codec.fingerprint:
            raise ValueError("Grouped tape codec differs")
        specifications = self._specs(specs, len(tape.tracks), self.generator.config.max_capacity)
        if tuple(spec.fingerprint for spec in specifications) != tape.spec_fingerprints:
            raise ValueError("Grouped tape output specifications differ")
        mean = self.tracks.predict_group(
            state, tracks=tape.tracks, control=self._controls(tape.controls, state.device),
            effort=effort,
        )
        distribution = self.generator.facet_distribution(mean)
        raw = tape.raw(state.device)
        if tuple(raw.shape[1:]) != tuple(mean.shape[1:]):
            raise ValueError("Grouped tape raw shape differs from model geometry")
        density = distribution.log_prob(raw).sum()
        if not torch.isclose(
            density.float(), torch.tensor(tape.behavior_log_prob, device=state.device),
            atol=atol, rtol=rtol,
        ):
            raise ValueError("Grouped policy density replay mismatch")
        committed = self.tracks.advance_group(
            state, raw, tracks=tape.tracks,
            role_ids=torch.tensor([tape.roles], device=state.device, dtype=torch.long),
        )
        decoded = [
            self.generator.decoder(
                self.generator.core, self.tracks.realization_context(committed, index), spec
            )
            for index, spec in enumerate(specifications)
        ]
        if (
            tuple(v.payload for v in decoded) != tape.payloads
            or tuple(v.emitted_ids for v in decoded) != tape.emitted_ids
            or tuple(v.valid for v in decoded) != tape.valid
        ):
            raise ValueError("Grouped realization replay mismatch")
        self.generator.core.routing_aux_loss(clear=True)
        return GroupGeneration(committed, tape, density)
