"""Production training windows for causally grouped text megatokens.

Meridian boundaries are semantic model steps. This adapter reuses the existing
prepared Episode/TrainingWindow, global loss ledger, optimizer and rank cursors.
It is deliberately text-first: media source assets must have a pinned decoder
before they may participate in production model training.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

import torch

from megatoken.contracts import DEFAULT_LOSS_WEIGHTS, ROLE_TO_ID
from megatoken.data import stage_recipe
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import EncodedRecord
from megatoken.generation.targets import PreparedAction, PreparedTargets
from megatoken.model.media import AUDIO_SAMPLE_RATE, MediaOutputSpec, MediaTrainTarget
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
        valid = None
        if self.reference.target.kind == "codec_latents":
            target_asset = self.reference.target
            if self.modality == "video" and not target_asset.codec_id:
                # Legacy donor packs store time-major tensors; typed native packs
                # already use their own declared latent-coordinate ordering.
                target = target.permute(0, 2, 1, 3, 4).contiguous()
                if target_asset.valid_count:
                    valid = (torch.arange(target.shape[2], device=device)
                             [None, None, :, None, None] < target_asset.valid_count)
            elif self.modality == "audio" and target_asset.valid_count:
                valid = (torch.arange(target.shape[1], device=device)[None, :, None]
                         < target_asset.valid_count)
        if self.modality == "image" and self.reference.target.kind == "raw":
            target = target[:, None]
        views = []
        for asset in self.reference.source_views:
            data = torch.from_numpy(store.load(asset)).to(device)[None]
            data, mask = _view_tokens(data, asset)
            views.append((asset.extractor, data, mask))
        if self.reference.source.kind == "features":
            source_asset = self.reference.source
            source, mask = _view_tokens(source, source_asset)
            views.insert(0, (source_asset.extractor, source, mask))
        native_views = tuple(
            torch.from_numpy(store.load(asset)).to(device)[None]
            for asset in self.reference.views
        )
        source_masks = tuple(
            torch.from_numpy(store.load_mask(asset)).to(device)[None]
            for asset in self.reference.source_masks
        )
        target_mask = (
            torch.from_numpy(store.load_mask(self.reference.target_mask)).to(device)[None]
            if self.reference.target_mask is not None else None
        )
        return MediaTrainTarget(
            self.modality, self.spec, source, target, self.mode,
            source_kind=self.reference.source.kind,
            source_sample_rate=self.reference.source.sample_rate or AUDIO_SAMPLE_RATE,
            extractor=self.reference.source.extractor,
            target_kind=self.reference.target.kind, source_views=tuple(views),
            target_valid_mask=valid, target_mask=target_mask,
            views=native_views,
            view_kinds=tuple(asset.kind for asset in self.reference.views),
            view_extractors=tuple(asset.extractor for asset in self.reference.views),
            source_masks=source_masks,
        )


def _view_tokens(value: torch.Tensor, asset) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose channel-last codec features and mask codec padding, not native pixels."""
    if asset.kind == "codec_latents":
        if asset.modality == "image":
            value = value.flatten(2).transpose(1, 2)
            sites_per_time = value.shape[1]
        elif asset.modality == "video":
            value = value.permute(0, 1, 3, 4, 2).reshape(
                value.shape[0], -1, value.shape[2]
            )
            sites_per_time = asset.shape[-2] * asset.shape[-1]
        else:
            sites_per_time = 1
    else:
        sites_per_time = 1
    active = (asset.valid_count or asset.shape[0]) * sites_per_time
    mask = torch.arange(value.shape[1], device=value.device)[None] < active
    return value, mask


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
    requests: tuple[tuple[str, int], ...] = ()
    relations: tuple[tuple[str, int, int, float], ...] = ()
    semantic_relations: tuple[tuple[str, int, int], ...] = ()

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
        raise ValueError("Generated media events require a pinned media target")
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
    if target.kind == "codec_latents" and target.codec_id:
        if not target.profile or not target.geometry:
            raise ValueError("Native codec targets require exact generator pack and physical geometry")
        native_kw = dict(
            codec_id=target.codec_id, pack_id=target.profile,
            latent_shape=target.shape,
            sample_rate=target.sample_rate or AUDIO_SAMPLE_RATE,
            fps_num=target.fps_num, fps_den=target.fps_den,
            shape_provenance="teacher_only", extent_source="teacher_only",
        )
        geometry = target.geometry
        if modality == "audio":
            spec = MediaOutputSpec("audio", sample_count=geometry[0], **native_kw)
        elif modality == "image":
            spec = MediaOutputSpec("image", height=geometry[-2], width=geometry[-1], **native_kw)
        else:
            spec = MediaOutputSpec("video", frames=geometry[0], height=geometry[-2],
                                   width=geometry[-1], **native_kw)
        return MediaPlan(reference, spec, config.representation_training,
                         model.config.n_facets, model.config.facet_dim)
    pretrained = target.kind == "codec_latents"
    renderer = (
        "pocket" if modality == "audio" else
        "sana" if modality == "image" else "sana_video"
    ) if pretrained else "native"
    geometry = reference.output_geometry
    if pretrained and geometry is None and source.kind != "raw":
        raise ValueError(
            "Codec latent targets with cached feature inputs require actual "
            "displayed output geometry, not guessed codec compression ratios"
        )
    if geometry is not None and geometry.modality != modality:
        raise ValueError("Declared media geometry disagrees with generated modality")
    provenance = geometry.provenance if geometry is not None else (
        "teacher_only" if pretrained else "requested"
    )

    if modality == "audio":
        if geometry is not None:
            sample_count, sample_rate = geometry.sample_count, geometry.sample_rate
        elif pretrained:
            if source.sample_rate != 24000:
                raise ValueError("Pocket codec target needs declared native 24k audio geometry")
            sample_count, sample_rate = source.shape[0], source.sample_rate
        else:
            sample_count, sample_rate = target.shape[0], target.sample_rate
        spec = MediaOutputSpec(
            "audio", sample_count=sample_count, sample_rate=sample_rate,
            renderer=renderer, latent_shape=target.shape if pretrained else (),
            shape_provenance=provenance,
        )
    elif modality == "image":
        if geometry is not None:
            height, width = geometry.height, geometry.width
        elif pretrained:
            height, width = source.shape[-2:]
        else:
            height, width = target.shape[-2:]
        spec = MediaOutputSpec(
            "image", height=height, width=width, renderer=renderer,
            latent_shape=target.shape if pretrained else (),
            shape_provenance=provenance,
        )
    else:
        if geometry is not None:
            frames, height, width = geometry.frames, geometry.height, geometry.width
            fps_num, fps_den = geometry.fps_num, geometry.fps_den
        elif pretrained:
            frames, height, width = source.shape[0], *source.shape[-2:]
            fps_num, fps_den = source.fps_num, source.fps_den
        else:
            frames, height, width = target.shape[0], *target.shape[-2:]
            fps_num, fps_den = target.fps_num, target.fps_den
        spec = MediaOutputSpec(
            "video", frames=frames, height=height, width=width,
            renderer=renderer,
            latent_shape=((target.shape[1], target.shape[0], *target.shape[-2:])
                          if pretrained else ()),
            shape_provenance=provenance,
            fps_num=fps_num, fps_den=fps_den,
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
    if config.concordance_recipe is not None:
        selected_media = {
            record.record_id for record in episode.records[window.start:window.end]
            if record.policy_mask
        }
        count = sum(
            event.record_id in selected_media
            and event.modality in {"audio", "image", "video"}
            for meridian in episode.meridians for event in meridian.events
        )
        if count > config.max_media_events_per_window:
            raise ValueError("Prepared concordance window exceeds the event-level media cap")
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
                if ref.source.kind == "features":
                    identity, width = (
                        (model.config.audio_feature_id, model.config.audio_feature_dim)
                        if event.modality == "audio" else
                        (model.config.vision_feature_id, model.config.vision_feature_dim)
                    )
                    if ref.source.extractor != identity or ref.source.shape[-1] != width:
                        raise ValueError(
                            "Observed media feature revision/width differs from model"
                        )
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
        indices = {entry.record.logical_id: i for i, entry in enumerate(records)}
        same_step_relations = tuple(
            (relation.kind, indices[relation.left], indices[relation.right],
             relation.weight)
            for relation in episode.omni_relations
            if all(selected) and offsets[0] >= window.start
            and relation.left in indices and relation.right in indices
        )
        counts["relation"] += len(same_step_relations)
        semantic_relations = ()
        if plans:
            annotations = json.loads(episode.supervision.annotations_json)
            if not isinstance(annotations, dict):
                raise ValueError("Concordance annotations must be a JSON object")
            rows = annotations.get("relations", ())
            current = {entry.record.record_id: i for i, entry in enumerate(records)}
            semantic_relations = tuple(
                (relation["kind"], current[relation["left"]], current[relation["right"]])
                for relation in rows
                if relation["left"] in current and relation["right"] in current
            )
            counts["relation_semantic"] += len(semantic_relations)
        step = MeridianStep(
            meridian.meridian_id,
            tuple(event.track_id for event in meridian.events),
            tuple(ROLE_TO_ID[event.role] for event in meridian.events),
            offsets, records, plans, modalities,
            tuple((event.request, event.request_revision) for event in meridian.events),
            same_step_relations, semantic_relations,
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
            with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
                if modality in {"audio", "image", "video"}:
                    source = MediaRecord.from_bytes(entry.payload).source
                    signal = torch.from_numpy(store.load(source)).to(device)[None]
                    reference = MediaRecord.from_bytes(entry.payload)
                    if reference.views or reference.source_masks:
                        refs = (reference.source, *reference.views)
                        signals = tuple(
                            torch.from_numpy(store.load(ref)).to(device)[None] for ref in refs
                        )
                        masks = tuple(
                            torch.from_numpy(store.load_mask(ref)).to(device)[None]
                            for ref in reference.source_masks
                        )
                        raw = model.core.media_stems.encode_multiview(
                            model.core, modality, signals,
                            tuple(ref.kind for ref in refs),
                            tuple(ref.extractor for ref in refs),
                            masks=masks, sample_rate=source.sample_rate or AUDIO_SAMPLE_RATE,
                        )
                    elif reference.source_views:
                        views = []
                        feature_assets = ((source,) if source.kind == "features" else ()) + reference.source_views
                        for view in feature_assets:
                            tensor = torch.from_numpy(store.load(view)).to(device)[None]
                            tensor, mask = _view_tokens(tensor, view)
                            views.append((view.extractor, tensor, mask))
                        raw = model.core.encode_media_views(modality, tuple(views))
                    elif source.kind == "features":
                        if source.valid_count and source.valid_count < source.shape[0]:
                            tokens, mask = _view_tokens(signal, source)
                            raw = model.core.encode_media_views(
                                modality, ((source.extractor, tokens, mask),)
                            )
                        else:
                            raw = model.core.encode_media_features(
                                modality, signal, extractor=source.extractor
                            )
                    elif source.kind == "raw":
                        raw = model.core.encode_media(
                            modality, signal, sample_rate=source.sample_rate or 16000
                        )
                    else:
                        raise ValueError("Codec latents need a declared feature projection")
                else:
                    ids, lengths = byte_tensor([entry.payload], device)
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
    store = TensorAssetStore(config.media_root or config.dataset)
    state = multi.tracks.initial_state(1, device=device)
    if state.model_version != model_version:
        # The core owner is versioned by the outer training loop.
        raise ValueError("Meridian trainer received a stale model version")
    payload = episode_control_payload(batch.windows[0].episode)
    from megatoken.controls import causal_step_payloads, resolve_event_control
    from megatoken.model.relations import SelectedRelation

    episode = batch.windows[0].episode
    event_by_record = {
        event.record_id: event for meridian in episode.meridians for event in meridian.events
    }
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
                state,
                _actual_raw(
                    model, step, batch.native_history, store, device,
                    encoder_grad=False,
                ),
                tracks=step.tracks,
                role_ids=torch.tensor([step.roles], dtype=torch.long, device=device),
                modalities=step.modalities,
            )
    for step in batch.steps:
        state, replayed = _preceding_native(multi, state, batch.native_history, step)
        metrics["grouped_native_replays"] += replayed
        if step.generated:
            # Recorded edit chains must continue from the *recorded* base,
            # never a sampled media output that the next gold edit did not see.
            edited_versions = {entry.record.artifact_version for entry in step.records
                               if entry.record.artifact_version}
            recorded_edit_base = any(
                later.base_artifact_version in edited_versions
                and later.edit_continuation == "recorded"
                for later in batch.windows[0].episode.records[step.offsets[-1] + 1:]
            )
            teacher_forced = recorded_edit_base or any(
                entry.record.action_id is not None
                and any(
                    later.result_of == entry.record.action_id
                    for later in batch.windows[0].episode.records[step.offsets[-1] + 1 :]
                )
                for entry in step.records
            )
            resolved = tuple(
                plan.resolve(store, device) if isinstance(plan, MediaPlan) else plan
                for plan in step.plans
            )
            result = multi.supervised_meridian(
                state, tracks=step.tracks, plans=resolved,
                control_payloads=(
                    tuple(resolve_event_control(episode, event_by_record[rec.record.record_id])
                          for rec in step.records)
                    if any(event_by_record[rec.record.record_id].instruction or
                           event_by_record[rec.record.record_id].reference_event_ids
                           for rec in step.records)
                    else causal_step_payloads(
                        episode, step.meridian_id, payload, step.tracks, step.requests,
                    )
                ),
                effort=batch.effort,
                teacher_forced_action=teacher_forced,
                relations=step.relations,
                semantic_relations=tuple(SelectedRelation(*rel) for rel in step.semantic_relations),
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
                    model, step, batch.native_history, store, device,
                    encoder_grad=config.representation_training == "bootstrap",
                ),
                tracks=step.tracks,
                role_ids=torch.tensor([step.roles], dtype=torch.long, device=device),
                modalities=step.modalities,
            )
    if losses:
        torch.stack(losses).sum().backward()
    return metrics
