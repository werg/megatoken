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
from megatoken.data.media_assets import (
    DESCRIPTOR_MODALITY,
    FEATURE_MODALITIES,
    MEDIA,
    MEDIA_RECORD_TYPE,
    MediaRecord,
    TensorAssetStore,
)
from megatoken.data.schema import EncodedRecord
from megatoken.generation.targets import PreparedAction, PreparedTargets
from megatoken.model.media import MediaOutputSpec, MediaTrainTarget
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.model.transducer import metadata_payload, real_boundary
from megatoken.retrieval.annotation import RetrievalAnnotation
from megatoken.tensors import byte_tensor
from megatoken.tool_schema import (
    compile_tool_spec,
    matching_schema,
    recent_copy_sources,
    supervised_bindings,
)
from megatoken.training.history import NativeHistory
from megatoken.training.retrieval import mount, retrieval_terms
from megatoken.training.stream import EpisodeWindow
from megatoken.training.text_anchors import anchor_loss
from megatoken.training.transducer import Meridian


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
        source, target = self.reference.source, self.reference.target
        assert target is not None
        mask = None
        if target.valid_count:
            # Only the leading real latent frames are supervised; codec padding is not.
            mask = torch.arange(target.shape[0], device=device)[None] < target.valid_count
        return MediaTrainTarget(
            self.modality,
            self.spec,
            torch.from_numpy(store.load(source)).to(device)[None],
            source.extractor,
            torch.from_numpy(store.load(target)).to(device)[None],
            mask,
            target.extractor,
            self.mode,
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
    # What the episode says about its first lookup (reading from a collection).
    retrieval: RetrievalAnnotation | None = None


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
    tape = native.paired_tape(index)
    if item.episode.tools_json != "[]" and record.action_id and record.part_count == 1:
        spec = compile_tool_spec(
            item.episode.tools_json,
            config.generated_capacity,
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
                    spec=spec,
                    schema_index=schema_index,
                    bindings=bindings,
                    layout=layout,
                    representation_mode=config.representation_training,
                )
                candidate = model.prepare_action_target(record.payload, **options)
                if candidate.payload == record.payload:
                    return (
                        model.prepare_action_target(
                            record.payload,
                            **options,
                            native_tape=tape,
                        )
                        if tape is not None
                        else candidate
                    )
    return model.prepare_target(
        [entry.token_ids],
        free_mask=[entry.free_mask],
        terminal=[terminal],
        role_ids=[ROLE_TO_ID[record.role]],
        representation_mode=config.representation_training,
        native_tapes=[tape],
    )


def _media_record(model: Any, entry: EncodedRecord, modality: str) -> MediaRecord:
    """The canonical media descriptor of an event, from this model's own teacher."""
    if entry.record.media_type != MEDIA_RECORD_TYPE:
        raise ValueError("Media events require a pinned canonical media descriptor")
    reference = MediaRecord.from_bytes(entry.payload)
    if reference.modality != DESCRIPTOR_MODALITY[modality]:
        raise ValueError("Media event modality differs from its descriptor")
    if reference.source.extractor != model.config.media_extractor:
        raise ValueError("Media features come from a different teacher than this model's")
    return reference


def _media_plan(model: Any, entry: EncodedRecord, modality: str, config: Any) -> MediaPlan:
    reference = _media_record(model, entry, modality)
    if reference.output_geometry is None:
        raise ValueError("Generated media events require a realization target")
    return MediaPlan(
        reference,
        MediaOutputSpec.from_geometry(reference.output_geometry),
        config.representation_training,
        model.config.n_facets,
        model.config.facet_dim,
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
    selected = {
        record.record_id
        for record in episode.records[window.start : window.end]
        if record.policy_mask
    }
    media_events = sum(
        event.record_id in selected and event.modality in MEDIA
        for meridian in episode.meridians
        for event in meridian.events
    )
    if media_events > config.max_media_events_per_window:
        raise ValueError("Prepared window exceeds the media event cap")
    if not episode.meridians:
        raise ValueError("Meridian preparation requires explicit causal groups")
    encoded = (*window.prefix, *window.records)
    if len(encoded) != window.end:
        raise ValueError("Window prefix and scored records have an inconsistent cursor")
    positions = {record.record_id: index for index, record in enumerate(episode.records)}
    prefix, steps = [], []
    counts = dict.fromkeys(DEFAULT_LOSS_WEIGHTS, 0)
    relation_rows = None  # the episode's annotations, parsed at the first supervised meridian
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
        modalities = tuple(event.modality for event in meridian.events)
        if any(len(encoded[i].payload) > config.max_record_bytes for i in offsets):
            raise ValueError("Grouped record exceeds max_record_bytes")
        records = tuple(encoded[i] for i in offsets)
        for event, record in zip(meridian.events, records, strict=True):
            if event.modality in FEATURE_MODALITIES:
                reference = _media_record(model, record, event.modality)
                if event.asset_sha256 is not None and event.asset_sha256 != reference.source.sha256:
                    raise ValueError("Media event source hash differs from its descriptor")
                if event.modality not in MEDIA and record.record.policy_mask:
                    raise ValueError("Compressed text is input-only and cannot be generated")
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
                if modality in MEDIA
                else _target_plan(model, item, record, index, offsets[0], native, config)
                for record, index, modality in zip(records, offsets, modalities, strict=True)
            )
            for plan in plans:
                for name, count in plan.denominators.items():
                    counts[name] += count
        indices = {entry.record.logical_id: i for i, entry in enumerate(records)}
        same_step_relations = tuple(
            (relation.kind, indices[relation.left], indices[relation.right], relation.weight)
            for relation in episode.omni_relations
            if all(selected)
            and offsets[0] >= window.start
            and relation.left in indices
            and relation.right in indices
        )
        counts["relation"] += len(same_step_relations)
        semantic_relations = ()
        if plans:
            if relation_rows is None:
                annotations = json.loads(episode.supervision.annotations_json)
                if not isinstance(annotations, dict):
                    raise ValueError("Concordance annotations must be a JSON object")
                relation_rows = annotations.get("relations", ())
            current = {entry.record.record_id: i for i, entry in enumerate(records)}
            semantic_relations = tuple(
                (relation["kind"], current[relation["left"]], current[relation["right"]])
                for relation in relation_rows
                if relation["left"] in current and relation["right"] in current
            )
            counts["relation_semantic"] += len(semantic_relations)
        step = MeridianStep(
            meridian.meridian_id,
            tuple(event.track_id for event in meridian.events),
            tuple(ROLE_TO_ID[event.role] for event in meridian.events),
            offsets,
            records,
            plans,
            modalities,
            tuple((event.request, event.request_revision) for event in meridian.events),
            same_step_relations,
            semantic_relations,
        )
        (prefix if offsets[-1] < window.start else steps).append(step)
    if sum(len(x.records) for x in (*prefix, *steps)) != window.end:
        raise ValueError("Grouped window does not cover exactly its physical records")
    effort = config.effort
    if config.use_effort_mixture:
        mixture = stage_recipe(config.phase).effort_mix
        levels = sorted(mixture)
        effort = levels[int(torch.multinomial(torch.tensor([mixture[k] for k in levels]), 1))]
    retrieval = RetrievalAnnotation.of(episode)
    # The annotated lookup is the window's first; a window that starts mid-episode replays
    # its prefix without gradient and does not score it.
    if retrieval is not None and not prefix and steps:
        counts["query_alignment"] += 1
        counts["retrieval_ranking"] += int(bool(retrieval.positives))
    return MeridianPreparedBatch(
        [item], tuple(prefix), tuple(steps), counts, native, effort, retrieval
    )


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
    for cursor, entry, modality in zip(step.offsets, step.records, step.modalities, strict=True):
        tape = native.paired_tape(cursor)
        if tape is not None:
            raw = torch.as_tensor(
                tape.facet_array().copy(),
                dtype=torch.float32,
                device=device,
            ).reshape(1, model.config.n_facets, model.config.facet_dim)
        else:
            with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
                if modality in FEATURE_MODALITIES:
                    source = MediaRecord.from_bytes(entry.payload).source
                    features = torch.from_numpy(store.load(source)).to(device)[None]
                    raw = model.core.encode_media(features, extractor=source.extractor)
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
                tape.facet_array().copy(),
                dtype=torch.float32,
                device=state.device,
            ).reshape(1, 1, *tape.facet_shape)
            state = multi.tracks.advance_group(
                state,
                raw,
                tracks=("_recorded_native_",),
                role_ids=torch.tensor([[tape.role_id]], dtype=torch.long, device=state.device),
                modalities=("text",),
            )
            count += 1
    return state, count


METADATA_KEEP = 0.5


def _metadata(episode: Any, seed: int | None) -> bytes:
    """The compact metadata prefix available before the first event.

    In training each field is kept or dropped at random, so a session opened
    with partial or no metadata (as serving may) is in distribution; held-out
    evaluation (``seed=None``) sees every field.
    """
    # Declared session tracks only: the modalities later events happen to use are future.
    modalities = sorted({track.default_modality for track in episode.tracks})
    fields = {"task": episode.stream, "modality": modalities, "version": "v10"}
    if seed is not None:
        keep = torch.rand(len(fields), generator=torch.Generator().manual_seed(seed + 1))
        fields = {
            k: v for (k, v), kept in zip(fields.items(), keep < METADATA_KEEP, strict=True) if kept
        }
    # An episode read from a collection always shows its question, which the first lookup uses.
    if (retrieval := RetrievalAnnotation.of(episode)) is not None:
        fields["request"] = retrieval.query_text
    return metadata_payload(fields)


def _boundary_meridian(step: MeridianStep) -> Meridian:
    """Whether a boundary follows is decided by record roles and part indices alone."""
    complete = [entry.record.part_index + 1 == entry.record.part_count for entry in step.records]
    return Meridian(step.tracks, step.roles, step.modalities, real_boundary(step.roles, complete))


def _transducer_sequence(
    model: Any,
    batch: MeridianPreparedBatch,
    trainer: Any,
    store: TensorAssetStore,
    device: torch.device,
    model_version: int,
    seed: int,
):
    """Plan the window's lookups and, in training, run its hindsight pass once."""
    steps = (*batch.prefix, *batch.steps)
    meridians = [_boundary_meridian(step) for step in steps]
    sources = None
    if batch.retrieval is not None:
        sources = mount(batch.retrieval, store.root, model.config.media_extractor)
    sequence = trainer.sequence(
        meridians,
        step=model_version,
        generator=torch.Generator().manual_seed(seed),
        sources=sources,
    )
    sequence.hindsight(
        meridians,
        lambda index: _actual_raw(
            model, steps[index], batch.native_history, store, device, encoder_grad=False
        ),
    )
    return sequence


def forward_meridian_batch(
    model: Any,
    batch: MeridianPreparedBatch,
    ledger: Any,
    config: Any,
    device: torch.device,
    model_version: int,
    *,
    text_anchors=None,
    differentiate: bool = True,
    transducer: Any,
) -> dict[str, float | torch.Tensor]:
    """Differentiate through scored meridians and rebuild the actual prefix.

    No second optimizer, stream cursor, history, or parameter owner is created.
    Held-out evaluation runs the identical forward with ``differentiate=False``
    (under ``torch.no_grad``). The transducer looks up at the window's planned
    boundaries; ``transducer`` is the run's ``TransducerTrainer`` (in evaluation
    its lookups read the bank alone and record nothing).
    """
    from megatoken.training.engine import episode_control_payload

    multi = MultiTrackGenerator.of(model)
    store = batch.windows[0].assets
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
    seed = int(torch.randint(0, 2**62, (1,)))
    sequence = _transducer_sequence(model, batch, transducer, store, device, model_version, seed)
    state = sequence.begin(
        multi,
        state,
        _metadata(episode, seed if transducer.native.training else None),
        encoder_grad=config.representation_training == "bootstrap",
    )
    metrics["transducer_lookups"] = sequence.planned
    meridian_index = 0
    with torch.no_grad():
        for step in batch.prefix:
            state = sequence.before_meridian(multi, state, meridian_index, effort=batch.effort)
            meridian_index += 1
            state, replayed = _preceding_native(multi, state, batch.native_history, step)
            metrics["grouped_native_replays"] += replayed
            state = multi.tracks.advance_group(
                state,
                _actual_raw(
                    model,
                    step,
                    batch.native_history,
                    store,
                    device,
                    encoder_grad=False,
                ),
                tracks=step.tracks,
                role_ids=torch.tensor([step.roles], dtype=torch.long, device=device),
                modalities=step.modalities,
            )
    for step in batch.steps:
        if text_anchors is not None:
            anchors, global_count, weight = text_anchors
            for entry, modality in zip(step.records, step.modalities, strict=True):
                key = anchors.key(entry.payload) if modality == "text" else None
                if key not in anchors.assets:
                    continue
                if global_count < 1:
                    raise RuntimeError("Text anchor denominator is missing")
                anchor = anchor_loss(model, entry.payload, anchors.features(key, device))
                losses.append(anchor * weight / global_count)
                metrics["text_anchor_mse_sum"] = (
                    metrics.get("text_anchor_mse_sum", 0.0) + anchor.detach()
                )
                metrics["text_anchors"] = metrics.get("text_anchors", 0) + 1
        state = sequence.before_meridian(multi, state, meridian_index, effort=batch.effort)
        meridian_index += 1
        state, replayed = _preceding_native(multi, state, batch.native_history, step)
        metrics["grouped_native_replays"] += replayed
        if step.generated:
            # Recorded edit chains must continue from the *recorded* base,
            # never a sampled media output that the next gold edit did not see.
            edited_versions = {
                entry.record.artifact_version
                for entry in step.records
                if entry.record.artifact_version
            }
            recorded_edit_base = any(
                later.base_artifact_version in edited_versions
                and later.edit_continuation == "recorded"
                for later in batch.windows[0].episode.records[step.offsets[-1] + 1 :]
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
                state,
                tracks=step.tracks,
                plans=resolved,
                control_payloads=(
                    tuple(
                        resolve_event_control(episode, event_by_record[rec.record.record_id])
                        for rec in step.records
                    )
                    if any(
                        event_by_record[rec.record.record_id].instruction
                        or event_by_record[rec.record.record_id].reference_event_ids
                        for rec in step.records
                    )
                    else causal_step_payloads(
                        episode,
                        step.meridian_id,
                        payload,
                        step.tracks,
                        step.requests,
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
            source_raw = _actual_raw(
                model,
                step,
                batch.native_history,
                store,
                device,
                encoder_grad=config.representation_training == "bootstrap",
            )
            state = multi.tracks.advance_group(
                state,
                source_raw,
                tracks=step.tracks,
                role_ids=torch.tensor([step.roles], dtype=torch.long, device=device),
                modalities=step.modalities,
            )
    if batch.retrieval is not None and not batch.prefix:
        if sequence.first_search is None:
            raise RuntimeError("The annotated first lookup did not run")
        terms, retrieved = retrieval_terms(
            sequence.first_search, sequence.sources, batch.retrieval, store
        )
        losses.append(ledger.backward_loss(terms))
        for name, value in retrieved.items():
            metrics[name] = metrics.get(name, 0) + value
    if differentiate and losses:
        torch.stack(losses).sum().backward()
    return metrics
