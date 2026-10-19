"""Production training windows for causally grouped text megatokens.

Meridian boundaries are semantic model steps. This adapter reuses the existing
prepared Episode/TrainingWindow, global loss ledger, optimizer and rank cursors.
It is deliberately text-first: media source assets must have a pinned decoder
before they may participate in production model training.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch

from megatoken.contracts import DEFAULT_LOSS_WEIGHTS, ROLE_TO_ID
from megatoken.data import stage_recipe
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import EncodedRecord
from megatoken.generation.targets import PreparedAction, PreparedTargets
from megatoken.model.media import MediaOutputSpec, MediaTrainTarget
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.tensors import byte_tensor
from megatoken.tool_schema import (
    compile_tool_spec,
    matching_schema,
    recent_copy_sources,
    supervised_bindings,
)
from megatoken.training.history import NativeHistory
from megatoken.training.stream import EpisodeWindow

@dataclass(frozen=True)
class MediaPlan:
    """Scored media megatoken with independently pinned source and target."""

    reference: MediaRecord
    spec: MediaOutputSpec
    mode: str
    facet_count: int
    facet_dim: int

    @property
    def modality(self) -> str:
        return self.reference.modality

    @property
    def scored_positions(self) -> int:
        return 0

    @property
    def denominators(self) -> dict[str, int]:
        counts = dict.fromkeys(DEFAULT_LOSS_WEIGHTS, 0)
        family = "audio" if self.modality == "audio" else "visual"
        counts[f"{family}_prediction"] = 1
        counts[f"{family}_reconstruction"] = 1
        counts["facet_prediction"] = (
            self.facet_count * self.facet_dim if self.mode == "bootstrap" else 0
        )
        counts["router_balance"] = counts["router_z"] = 1
        return counts

    def resolve(self, store: TensorAssetStore, device: torch.device) -> MediaTrainTarget:
        source = torch.from_numpy(store.load(self.reference.source)).to(device)[None]
        assert self.reference.target is not None
        target = torch.from_numpy(store.load(self.reference.target)).to(device)[None]
        if self.modality == "image":
            target = target[:, None]
        return MediaTrainTarget(
            self.modality, self.spec, source, target, self.mode,
            source_kind=self.reference.source.kind,
            extractor=self.reference.source.extractor,
        )


Plan = PreparedAction | PreparedTargets | MediaPlan


@dataclass(frozen=True)
class MeridianStep:
    """One all-observed or all-generated group, never a partial meridian."""

    meridian_id: int
    tracks: tuple[str, ...]
    roles: tuple[int, ...]
    offsets: tuple[int, ...]
    records: tuple[EncodedRecord, ...]
    plans: tuple[Plan, ...] = ()
    modalities: tuple[str, ...] = ()

    @property
    def generated(self) -> bool:
        return bool(self.plans)


@dataclass
class MeridianPreparedBatch:
    """One independent episode window, scored through the usual loss ledger."""

    windows: list[EpisodeWindow]
    prefix: tuple[MeridianStep, ...]
    steps: tuple[MeridianStep, ...]
    counts: dict[str, int]
    native_history: NativeHistory
    effort: int


def _target_plan(
    model: Any,
    item: EpisodeWindow,
    entry: EncodedRecord,
    index: int,
    beginning: int,
    native: NativeHistory,
    config: Any,
) -> Plan:
    record = entry.record
    if len(entry.token_ids) > config.generated_capacity:
        raise ValueError("Grouped target exceeds generated_capacity")
    terminal = record.part_index + 1 == record.part_count
    length = max(1, len(entry.token_ids))
    capacity = min(config.generated_capacity, 1 << (length - 1).bit_length())
    tape = native.paired_tape(index)
    if (
        item.episode.tools_json != "[]"
        and record.action_id
        and record.part_count == 1
    ):
        spec = compile_tool_spec(
            item.episode.tools_json, config.generated_capacity,
            wire_formats=("canonical", "openai", "anthropic"),
        )
        # Only records preceding the *whole meridian* can be public copies.
        earlier = [record.payload for record in item.episode.records[:beginning]]
        spec = dataclasses.replace(spec, copy_sources=recent_copy_sources(earlier))
        schema_index = matching_schema(spec, record.payload)
        if schema_index is not None:
            bindings = supervised_bindings(spec, schema_index, record.payload)
            for layout in range(len(spec.schemas[schema_index].orders)):
                options = dict(
                    spec=spec, schema_index=schema_index, bindings=bindings,
                    layout=layout, representation_mode=config.representation_training,
                )
                candidate = model.prepare_action_target(record.payload, **options)
                if candidate.payload == record.payload:
                    return (
                        model.prepare_action_target(
                            record.payload, **options, native_tape=tape,
                        )
                        if tape is not None else candidate
                    )
    return model.prepare_target(
        [entry.token_ids],
        free_mask=[entry.free_mask],
        capacity=capacity,
        terminal=[terminal],
        role_ids=[ROLE_TO_ID[record.role]],
        representation_mode=config.representation_training,
        native_tapes=[tape],
    )



def _media_plan(model: Any, entry: EncodedRecord, modality: str, config: Any) -> MediaPlan:
    reference = MediaRecord.from_bytes(entry.payload)
    if reference.modality != modality or entry.record.media_type != MEDIA_RECORD_TYPE:
        raise ValueError("Media event requires a matching canonical media descriptor")
    if reference.target is None:
        raise ValueError("Generated media events require a pinned raw target")
    source = reference.source
    if source.kind == "features":
        feature_id = (
            model.config.audio_feature_id if modality == "audio"
            else model.config.vision_feature_id
        )
        feature_width = (
            model.config.audio_feature_dim if modality == "audio"
            else model.config.vision_feature_dim
        )
        if source.extractor != feature_id or source.shape[-1] != feature_width:
            raise ValueError("Pinned media feature revision/width differs from model")
    target = reference.target
    if modality == "audio":
        spec = MediaOutputSpec("audio", sample_count=target.shape[0])
    elif modality == "image":
        spec = MediaOutputSpec("image", height=target.shape[1], width=target.shape[2])
    else:
        spec = MediaOutputSpec(
            "video", frames=target.shape[0],
            height=target.shape[2], width=target.shape[3],
        )
    return MediaPlan(
        reference, spec, config.representation_training,
        model.config.n_facets, model.config.facet_dim,
    )



def prepare_meridian_batch(
    model: Any,
    item: EpisodeWindow,
    native: NativeHistory,
    config: Any,
) -> MeridianPreparedBatch:
    """Build a loss-exact grouped plan without inspecting future target bytes.

    Target bytes are held in PreparedTargets/PreparedAction for supervised R,
    never passed to prospective P. All grouping metadata comes from the
    validated Episode meridians.
    """
    window, episode = item.window, item.episode
    if not episode.meridians:
        raise ValueError("Meridian preparation requires explicit causal groups")
    encoded = (*window.prefix, *window.records)
    if len(encoded) != window.end:
        raise ValueError("Window prefix and scored records have an inconsistent cursor")
    positions = {record.record_id: index for index, record in enumerate(episode.records)}
    prefix, steps = [], []
    counts = dict.fromkeys(DEFAULT_LOSS_WEIGHTS, 0)
    for meridian in episode.meridians:
        offsets = tuple(positions[event.record_id] for event in meridian.events)
        if offsets[-1] >= window.end:
            if offsets[0] < window.end:
                raise ValueError("Training window split a meridian")
            break
        if offsets[0] < window.start <= offsets[-1]:
            raise ValueError("Training window split a meridian")
        if len(offsets) > 1 and any(native.paired_tape(cursor) is not None for cursor in offsets):
            raise ValueError(
                "Individual native action tapes cannot certify a jointly selected "
                "meridian; record a grouped native tape or use causal text targets"
            )
        if len(offsets) > 1 and any(
            native.events_for_record(cursor, include_record=False) for cursor in offsets[1:]
        ):
            raise ValueError(
                "A zero-record native action inside an atomic meridian has "
                "ambiguous causal timing; place it between meridians"
            )
        modalities = tuple(
            event.modality if event.modality in {"audio", "image", "video"} else "text"
            for event in meridian.events
        )
        if any(len(encoded[i].payload) > config.max_record_bytes for i in offsets):
            raise ValueError("Grouped record exceeds max_record_bytes")
        records = tuple(encoded[i] for i in offsets)
        for event, record in zip(meridian.events, records, strict=True):
            if event.modality in {"audio", "image", "video"}:
                if record.record.media_type != MEDIA_RECORD_TYPE:
                    raise ValueError(
                        "Media events require a pinned canonical media asset descriptor"
                    )
                ref = MediaRecord.from_bytes(record.payload)
                if ref.modality != event.modality or (
                    event.asset_sha256 is not None
                    and event.asset_sha256 != ref.source.sha256
                ):
                    raise ValueError("Media event modality/source hash differs from its asset")
            elif record.record.media_type == MEDIA_RECORD_TYPE or (
                record.record.media_type.split("/", 1)[0] in {"audio", "image", "video"}
            ):
                raise ValueError("Media payload cannot be ingested on a text track")
        selected = tuple(record.record.policy_mask for record in records)
        if any(selected) and not all(selected):
            raise ValueError(
                "An observed/generated mixed meridian would leak same-step targets; "
                "split observations and predictions into consecutive meridians"
            )
        plans: tuple[Plan, ...] = ()
        if all(selected) and offsets[0] >= window.start:
            plans = tuple(
                _media_plan(model, record, modality, config)
                if modality in {"audio", "image", "video"}
                else _target_plan(model, item, record, index, offsets[0], native, config)
                for record, index, modality in zip(
                    records, offsets, modalities, strict=True
                )
            )
            for plan in plans:
                for name, count in plan.denominators.items():
                    counts[name] += count
        step = MeridianStep(
            meridian.meridian_id,
            tuple(event.track_id for event in meridian.events),
            tuple(ROLE_TO_ID[event.role] for event in meridian.events),
            offsets, records, plans, modalities,
        )
        (prefix if offsets[-1] < window.start else steps).append(step)
    if sum(len(x.records) for x in (*prefix, *steps)) != window.end:
        raise ValueError("Grouped window does not cover exactly its physical records")
    effort = config.effort
    if config.use_effort_mixture:
        mixture = stage_recipe(config.phase).effort_mix
        levels = sorted(mixture)
        effort = levels[int(torch.multinomial(torch.tensor([mixture[k] for k in levels]), 1))]
    return MeridianPreparedBatch([item], tuple(prefix), tuple(steps), counts, native, effort)


def _actual_raw(
    model: Any,
    step: MeridianStep,
    native: NativeHistory,
    store: TensorAssetStore,
    device: torch.device,
    *,
    encoder_grad: bool,
) -> torch.Tensor:
    """Rebuild selected raw blocks, preserving pinned modality and native identity."""
    values = []
    for cursor, entry, modality in zip(
        step.offsets, step.records, step.modalities, strict=True
    ):
        tape = native.paired_tape(cursor)
        if tape is not None:
            raw = torch.as_tensor(
                tape.facet_array().copy(), dtype=torch.float32, device=device,
            ).reshape(1, model.config.n_facets, model.config.facet_dim)
        else:
            ids, lengths = byte_tensor([entry.payload], device)
            with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
                raw = model.core.encode_chunk(ids, lengths)
        values.append(raw)
    return torch.stack(values, dim=1)


def _preceding_native(
    multi: MultiTrackGenerator,
    state: Any,
    native: NativeHistory,
    step: MeridianStep,
) -> tuple[Any, int]:
    """Replay zero-record decisions before each physical event, never its future tape."""
    count = 0
    for cursor in step.offsets:
        for tape in native.events_for_record(cursor, include_record=False):
            raw = torch.as_tensor(
                tape.facet_array().copy(), dtype=torch.float32, device=state.device,
            ).reshape(1, 1, *tape.facet_shape)
            state = multi.tracks.advance_group(
                state, raw, tracks=("_recorded_native_",),
                role_ids=torch.tensor([[tape.role_id]], dtype=torch.long, device=state.device),
                modalities=("text",),
            )
            count += 1
    return state, count


def forward_meridian_batch(
    model: Any,
    batch: MeridianPreparedBatch,
    ledger: Any,
    config: Any,
    device: torch.device,
    model_version: int,
) -> dict[str, float | torch.Tensor]:
    """Differentiate through scored meridians and rebuild the actual prefix.

    No second optimizer, stream cursor, history, or parameter owner is created.
    """
    from megatoken.training.engine import episode_control_payload

    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1, device=device)
    if state.model_version != model_version:
        # The core owner is versioned by the outer training loop.
        raise ValueError("Meridian trainer received a stale model version")
    payload = episode_control_payload(batch.windows[0].episode)
    losses = []
    metrics: dict[str, float | torch.Tensor] = {
        "grouped_training_windows": 1,
        "grouped_supervised_meridians": 0,
        "grouped_selected_events": 0,
        "grouped_native_replays": 0,
    }
    with torch.no_grad():
        for step in batch.prefix:
            state, replayed = _preceding_native(multi, state, batch.native_history, step)
            metrics["grouped_native_replays"] += replayed
            state = multi.tracks.advance_group(
                state, _actual_raw(model, step, batch.native_history, device, encoder_grad=False),
                tracks=step.tracks,
                role_ids=torch.tensor([step.roles], dtype=torch.long, device=device),
                modalities=("text",) * len(step.tracks),
            )
    for step in batch.steps:
        state, replayed = _preceding_native(multi, state, batch.native_history, step)
        metrics["grouped_native_replays"] += replayed
        if step.generated:
            teacher_forced = any(
                entry.record.action_id is not None
                and any(
                    later.result_of == entry.record.action_id
                    for later in batch.windows[0].episode.records[step.offsets[-1] + 1 :]
                )
                for entry in step.records
            )
            result = multi.supervised_meridian(
                state, tracks=step.tracks, plans=step.plans,
                control_payloads=(payload,) * len(step.tracks),
                effort=batch.effort,
                teacher_forced_action=teacher_forced,
            )
            state = result.state
            losses.append(ledger.backward_loss(result.losses))
            metrics["grouped_supervised_meridians"] += 1
            metrics["grouped_selected_events"] += len(step.tracks)
            for name, value in result.metrics.items():
                metrics[name] = metrics.get(name, 0) + (
                    value.detach() if isinstance(value, torch.Tensor) else value
                )
        else:
            state = multi.tracks.advance_group(
                state, _actual_raw(
                    model, step, batch.native_history, device,
                    encoder_grad=config.representation_training == "bootstrap",
                ),
                tracks=step.tracks,
                role_ids=torch.tensor([step.roles], dtype=torch.long, device=device),
                modalities=("text",) * len(step.tracks),
            )
    if losses:
        torch.stack(losses).sum().backward()
    return metrics
