"""Raw-only, causal multitrack serving for observed text/audio/visual evidence.

Every output event has an effective prompt before P samples its own Z.
No future target or verifier annotation is present in ServingSession.
All same-meridian media/text siblings are selected atomically and may attend
through the existing multitrack P; R only sees a given selected raw Z.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from megatoken.controls import resolve_event_control
from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec, stable_id
from megatoken.generation.structure import ActionSpec
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import GroupGeneration, MultiTrackGenerator, MultiTrackState
from megatoken.model.core import ROLE_IDS
from megatoken.tensors import byte_tensor


@dataclass(frozen=True, slots=True)
class ServingObservation:
    track_id: str
    modality: str
    payload: bytes | MediaRecord
    instruction: str
    role: str = "user"
    revision: int = 0
    sample_rate: int = 16000
    settings_json: str = "{}"
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not self.track_id or not self.instruction or self.role not in ROLE_IDS
                or self.modality not in {"text", "audio", "image", "video"}
                or (self.modality == "text") != isinstance(self.payload, bytes)):
            raise ValueError("Invalid causal observed event or model input type")
        if self.modality != "text" and (
            not isinstance(self.payload, MediaRecord)
            or self.payload.modality != self.modality
        ):
            raise ValueError("Media observation requires a matching immutable record")


@dataclass(frozen=True, slots=True)
class ServingOutputRequest:
    track_id: str
    modality: str
    instruction: str
    spec: ActionSpec | MediaOutputSpec
    revision: int = 0
    settings_json: str = "{}"
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.track_id or not self.instruction:
            raise ValueError("Every generated output requires an event request")
        if self.modality == "text" and not isinstance(self.spec, ActionSpec):
            raise ValueError("Text output needs an ActionSpec")
        if self.modality != "text" and (
            not isinstance(self.spec, MediaOutputSpec)
            or self.modality != self.spec.modality
        ):
            raise ValueError("Media output requires its exact native rendering spec")
        if isinstance(self.spec, MediaOutputSpec) and self.spec.extent_source == "teacher_only":
            raise ValueError("Free rollout cannot adopt a gold-only output duration or geometry")


@dataclass
class ServingSession:
    generator: MultiTrackGenerator
    tracks: tuple[TrackSpec, ...]
    control_seed: bytes = b""
    control_reference: bytes = b""
    reference_origin: str | None = None
    tools_json: str = "[]"
    state: MultiTrackState | None = None
    meridians: list[Meridian] = field(default_factory=list)
    # track revisions prevent an old edit request from being replayed accidentally.
    _revisions: dict[str, int] = field(default_factory=dict)
    _nonce: int = 0

    def __post_init__(self) -> None:
        if len({t.track_id for t in self.tracks}) != len(self.tracks):
            raise ValueError("Session has duplicate track IDs")
        if self.state is None:
            self.state = self.generator.tracks.initial_state(1)
        if self.state.batch_size != 1:
            raise ValueError("Serving requires one independent episode")
        if self.control_reference and self.reference_origin not in {
            "external_evidence", "earlier_history"
        }:
            raise ValueError("Serving control reference requires past or external evidence")
        self.generator.tracks._validate(self.state)

    @property
    def meridian_id(self) -> int:
        return int(self.state.meridians_seen[0])

    def _request(self, track: str, modality: str, role: str, instruction: str,
                 revision: int, settings: str, refs: tuple[str, ...]) -> TrackEvent:
        spec = next((t for t in self.tracks if t.track_id == track), None)
        if spec is None or spec.role != role:
            raise ValueError("Serving request track must match a declared session role")
        if spec.default_modality != modality:
            raise ValueError("Track modality differs from its declared stream")
        if revision < self._revisions.get(track, 0):
            raise ValueError("Serving edit revision cannot move backwards")
        record_id = stable_id(
            "live-event", track, str(self.meridian_id), str(self._nonce)
        )[:32]
        self._nonce += 1
        return TrackEvent(
            record_id, track, modality, self.meridian_id, role,
            instruction=instruction, request_revision=revision,
            settings_json=settings, reference_event_ids=refs,
        )

    def _append(self, events: tuple[TrackEvent, ...], state: MultiTrackState) -> None:
        self.meridians.append(Meridian(events[0].meridian_id, events))
        self._revisions.update({event.track_id: event.request_revision for event in events})
        self.state = state

    def observe(self, events: Sequence[ServingObservation],
                *, assets: TensorAssetStore | None = None) -> tuple[str, ...]:
        """Commit exactly one observed meridian from actual available source assets."""
        if not events:
            raise ValueError("Observed meridian must be nonempty")
        group = tuple(self._request(e.track_id, e.modality, e.role, e.instruction,
                                    e.revision, e.settings_json, e.references)
                      for e in events)
        if len({x.track_id for x in group}) != len(group):
            raise ValueError("Concurrent observed tracks must be unique")
        for event in group:
            resolve_event_control(self, event)  # validate causal references early
        core = self.generator.generator.core
        device = self.state.device
        raws = []
        with torch.no_grad():
            for event in events:
                if event.modality == "text":
                    ids, lengths = byte_tensor([event.payload], device)
                    raw = core.encode_chunk(ids, lengths=lengths)
                else:
                    if assets is None:
                        raise ValueError("Media serving requires a resolved immutable asset store")
                    ref = event.payload
                    assert isinstance(ref, MediaRecord)
                    sources = (ref.source, *ref.views)
                    views = tuple(torch.from_numpy(assets.load(x)).to(device)[None]
                                  for x in sources)
                    kinds = tuple(x.kind for x in sources)
                    extractors = tuple(x.extractor for x in sources)
                    masks = tuple(torch.from_numpy(assets.load_mask(x)).to(device)[None]
                                  for x in ref.source_masks)
                    if len(views) > 1 or masks:
                        raw = core.media_stems.encode_multiview(
                            core, event.modality, views, kinds, extractors,
                            masks=masks,
                            sample_rate=ref.source.sample_rate or event.sample_rate,
                        )
                    elif kinds[0] == "features":
                        raw = core.encode_media_features(
                            event.modality, views[0], extractor=extractors[0],
                        )
                    elif kinds[0] == "raw":
                        raw = core.encode_media(
                            event.modality, views[0],
                            sample_rate=ref.source.sample_rate or event.sample_rate,
                        )
                    else:
                        raise ValueError("Observed codec targets cannot be supplied as E input")
                raws.append(raw)
            state = self.generator.tracks.advance_group(
                self.state, torch.stack(raws, dim=1),
                tracks=tuple(e.track_id for e in events),
                role_ids=torch.tensor([[ROLE_IDS[e.role] for e in events]],
                                      device=device, dtype=torch.long),
                modalities=tuple(e.modality for e in events),
            )
        self._append(group, state)
        return tuple(e.record_id for e in group)

    def generate(self, events: Sequence[ServingOutputRequest], *,
                 effort: int = 6, generator: torch.Generator | None = None
                 ) -> tuple[GroupGeneration, tuple[str, ...]]:
        """Select whole group with event-local causal controls, then render Z only."""
        if not events:
            raise ValueError("Generated meridian cannot be empty")
        group = tuple(self._request(e.track_id, e.modality, "assistant",
                                    e.instruction, e.revision, e.settings_json,
                                    e.references) for e in events)
        if len({x.track_id for x in group}) != len(group):
            raise ValueError("Concurrent generated tracks must be unique")
        controls = tuple(resolve_event_control(self, event) for event in group)
        result = self.generator.generate_group(
            self.state, tracks=tuple(e.track_id for e in events),
            specs=tuple(e.spec for e in events), controls=controls,
            effort=effort, generator=generator,
        )
        self._append(group, result.state)
        return result, tuple(e.record_id for e in group)
