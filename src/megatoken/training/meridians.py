"""Production training windows for causally grouped text megatokens.

Meridian boundaries are semantic model steps. This adapter reuses the existing
prepared Episode/TrainingWindow, global loss ledger, optimizer and rank cursors.
It is deliberately text-first: media source assets must have a pinned decoder
before they may participate in production model training.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from megatoken.contracts import DEFAULT_LOSS_WEIGHTS, ROLE_TO_ID
from megatoken.data import stage_recipe
from megatoken.data.knowledge import episode_metadata
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
from megatoken.model.transducer import completed_turn, metadata_payload, real_boundary
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
from megatoken.training.transducer import (
    INGEST_KEY_WEIGHT,
    INGEST_VALUE_WEIGHT,
    Meridian,
)


@dataclass(frozen=True)
class MediaPlan:
    """Scored media megatoken with independently pinned source and target."""

    reference: MediaRecord
    spec: MediaOutputSpec
    mode: str
    facet_count: int
    facet_dim: int
    native_tape: object | None = None

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
        counts["encoder_alignment"] = (
            self.facet_count * self.facet_dim
            if self.mode == "recurrent" and self.native_tape is not None
            else 0
        )
        counts["router_balance"] = counts["router_z"] = 1
        return counts

    def resolve(self, store: TensorAssetStore, device: torch.device) -> MediaTrainTarget:
        source, target = self.reference.source, self.reference.target
        assert target is not None
        if self.reference.rendered is not None:
            store.load(self.reference.rendered)  # verify the actual decoded bytes before scoring
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
            native_tape=self.native_tape,
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
    evidence_relations: tuple[tuple[str, int, str], ...] = ()
    modality_choices: tuple[tuple[str, ...], ...] = ()

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
    # What the episode says about the lookup after its question (reading from a collection).
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
    from megatoken.data.context import validate_context_announcements

    validate_context_announcements(episode)
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
    counts["policy_outcome"] = len(native.policy_entries(window.start, window.end))
    relation_rows = None  # the episode's annotations, parsed at the first supervised meridian
    for meridian in episode.meridians:
        offsets = tuple(positions[event.record_id] for event in meridian.events)
        if offsets[-1] >= window.end:
            if offsets[0] < window.end:
                raise ValueError("Training window split a meridian")
            break
        if offsets[0] < window.start <= offsets[-1]:
            raise ValueError("Training window split a meridian")
        if (
            len(offsets) > 1
            and any(native.paired_tape(cursor) is not None for cursor in offsets)
            and native.group_for_records(offsets) is None
        ):
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
                dataclasses.replace(
                    _media_plan(model, record, modality, config),
                    native_tape=native.paired_tape(index),
                )
                if modality in MEDIA
                else _target_plan(model, item, record, index, offsets[0], native, config)
                for record, index, modality in zip(records, offsets, modalities, strict=True)
            )
            native_group = native.group_for_records(offsets)
            # A collected group records its complete joint selection even when
            # failed or empty outputs have no physical record. Modality choices
            # remain labels for every selected peer; payload losses below only
            # have targets for materialized records.
            counts["modality"] += (
                len(native_group.tracks) if native_group is not None else len(plans)
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
        evidence_relations = ()
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
            evidence_relations = tuple(
                (relation["kind"], current[now], earlier)
                for relation in relation_rows
                for now, earlier in (
                    (relation["left"], relation["right"]),
                    (relation["right"], relation["left"]),
                )
                if now in current and earlier in positions and positions[earlier] < offsets[0]
            )
            counts["relation_semantic"] += len(evidence_relations)
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
            evidence_relations,
            tuple(
                tuple(json.loads(event.settings_json).get("allowed_modalities", ()))
                for event in meridian.events
            ),
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
    if retrieval is not None and _question_scored(prefix, steps):
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
    descriptions=(),
    prefer_native=True,
) -> torch.Tensor:
    """Rebuild selected raw blocks, preserving pinned modality and native identity."""
    from megatoken.controls import encode_controls

    purposes = {t.track_id: t.description for t in descriptions}
    values = []
    for track, cursor, entry, modality in zip(
        step.tracks, step.offsets, step.records, step.modalities, strict=True
    ):
        tape = native.paired_tape(cursor) if prefer_native else None
        if tape is not None:
            raw = tape.single_raw(device)
        else:
            with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
                semantic = encode_controls(
                    model.core, [purposes.get(track, "").encode()], encoder_grad=False
                )
                if modality in FEATURE_MODALITIES:
                    source = MediaRecord.from_bytes(entry.payload).source
                    features = torch.from_numpy(store.load(source)).to(device)[None]
                    raw = model.core.encode_media(
                        features, extractor=source.extractor, track_semantics=semantic
                    )
                else:
                    ids, lengths = byte_tensor([entry.payload], device)
                    raw = model.core.encode_chunk(ids, lengths, track_semantics=semantic)
        values.append(raw)
    return torch.stack(values, dim=1)


def _modality_masks(model, step, device):
    from megatoken.model.media import REALIZERS

    names = ("text", "audio", "image", "video")
    supported = {
        name for name in names if name == "text" or REALIZERS[name] in model.media_realizers.packs
    }
    choices = step.modality_choices or ((),) * len(step.tracks)
    rows = []
    for allowed in choices:
        if any(name not in names for name in allowed) or len(allowed) != len(set(allowed)):
            raise ValueError(
                "Public modality constraints require distinct supported modality names"
            )
        admitted = supported.intersection(allowed) if allowed else supported
        if not admitted:
            raise ValueError("Public output constraints allow no installed decoder")
        rows.append([name in admitted for name in names])
    return torch.tensor([rows], device=device, dtype=torch.bool)


def _earlier_evidence(model, source, native, store, device, descriptions):
    step, index = source
    single = dataclasses.replace(
        step,
        tracks=(step.tracks[index],),
        offsets=(step.offsets[index],),
        records=(step.records[index],),
        modalities=(step.modalities[index],),
        roles=(step.roles[index],),
        plans=(),
    )
    return _actual_raw(
        model,
        single,
        native,
        store,
        device,
        encoder_grad=False,
        descriptions=descriptions,
        prefer_native=False,
    )[:, 0]


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
            state = multi.commit_tape(state, tape)
            count += 1
    return state, count


METADATA_KEEP = 0.5


def _metadata(episode: Any, seed: int | None) -> bytes:
    """The compact metadata prefix available before the first event.

    In training each field is kept or dropped at random, so a session opened
    with partial or no metadata (as serving may) is in distribution; held-out
    evaluation (``seed=None``) sees every field.
    """
    fields = episode_metadata(episode)
    if seed is not None:
        keep = torch.rand(len(fields), generator=torch.Generator().manual_seed(seed + 1))
        fields = {
            k: v for (k, v), kept in zip(fields.items(), keep < METADATA_KEEP, strict=True) if kept
        }
    return metadata_payload(fields)


def _complete(step: MeridianStep) -> list[bool]:
    """Which events are the last part of their message."""
    return [entry.record.part_index + 1 == entry.record.part_count for entry in step.records]


def _asked(step: MeridianStep) -> bool:
    """Whether the step finishes a user message."""
    return completed_turn(step.roles, _complete(step), "user")


def _boundary_meridian(step: MeridianStep) -> Meridian:
    """Whether a boundary follows is decided by record roles and part indices alone."""
    return Meridian(
        step.tracks,
        step.roles,
        step.modalities,
        real_boundary(step.roles, _complete(step)),
        _asked(step),
    )


def _question_scored(prefix: Sequence[MeridianStep], steps: Sequence[MeridianStep]) -> bool:
    """Whether the window scores the lookup that follows its first finished user message.

    That lookup falls right before the next meridian. A window that starts after it
    replays it without gradient and does not score it.
    """
    meridians = (*prefix, *steps)
    asked = next((i for i, step in enumerate(meridians) if _asked(step)), None)
    return asked is not None and len(prefix) <= asked + 1 < len(meridians)


def _transducer_sequence(
    model: Any,
    batch: MeridianPreparedBatch,
    trainer: Any,
    store: TensorAssetStore,
    device: torch.device,
    model_version: int,
    seed: int,
    mounted: Any = None,
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
        mounted=mounted,
        sources=sources,
    )
    sequence.hindsight(
        meridians,
        lambda index: _actual_raw(
            model,
            steps[index],
            batch.native_history,
            store,
            device,
            encoder_grad=False,
            descriptions=batch.windows[0].episode.tracks,
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
    multi = MultiTrackGenerator.of(model)
    store = batch.windows[0].assets
    state = multi.tracks.initial_state(1, device=device)
    if state.model_version != model_version:
        # The core owner is versioned by the outer training loop.
        raise ValueError("Meridian trainer received a stale model version")
    from megatoken.controls import resolve_event_control
    from megatoken.data.context import apply_context_announcement
    from megatoken.model.relations import SelectedRelation

    episode = batch.windows[0].episode
    for track in episode.tracks:
        if track.description:
            state = multi.tracks.describe_track(state, track.track_id, track.description)
    event_by_record = {
        event.record_id: event for meridian in episode.meridians for event in meridian.events
    }
    earlier_steps = {
        entry.record.record_id: (step, index)
        for step in (*batch.prefix, *batch.steps)
        for index, entry in enumerate(step.records)
    }
    revisions = {}
    for revision in episode.track_revisions:
        revisions.setdefault(revision.record_id, []).append(revision)

    def announce(state, step):
        for record in step.records:
            for revision in revisions.get(record.record.record_id, ()):
                state = multi.tracks.describe_track(
                    state, revision.track_id, revision.purpose, revision.revision
                )
        return state

    def context_before(state, step):
        for track, entry in zip(step.tracks, step.records, strict=True):
            state = apply_context_announcement(
                multi, state, track, entry.record.payload, entry.record.role
            )
        return state

    losses = []
    from megatoken.training.outcome import policy_outcome

    window = batch.windows[0].window
    outcome_metrics = {}
    for tape, credit in batch.native_history.policy_entries(window.start, window.end):
        terms, measured = policy_outcome(multi, tape, credit, device=device)
        losses.append(ledger.backward_loss(terms))
        for name, value in measured.items():
            outcome_metrics[name] = (
                outcome_metrics.get(name, 0) + value.detach()
                if isinstance(value, torch.Tensor)
                else outcome_metrics.get(name, 0) + value
            )
    metrics: dict[str, float | torch.Tensor] = {
        "grouped_training_windows": 1,
        "grouped_supervised_meridians": 0,
        "grouped_selected_events": 0,
        "grouped_native_replays": 0,
        **outcome_metrics,
    }
    seed = int(torch.randint(0, 2**62, (1,)))
    # A reading-from-a-base episode searches the ingested bases of its declared items too.
    mounted = (
        None
        if transducer.reading is None
        else transducer.reading.mount(multi, episode, store, model_version)
    )
    sequence = _transducer_sequence(
        model, batch, transducer, store, device, model_version, seed, mounted
    )
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
            state = context_before(state, step)
            state = sequence.before_meridian(multi, state, meridian_index, effort=batch.effort)
            meridian_index += 1
            state, replayed = _preceding_native(multi, state, batch.native_history, step)
            metrics["grouped_native_replays"] += replayed
            native_group = batch.native_history.group_for_records(step.offsets)
            state = (
                multi.commit_tape(state, native_group)
                if native_group is not None
                else multi.tracks.advance_group(
                    state,
                    native_group.raw(device)
                    if native_group is not None
                    else _actual_raw(
                        model,
                        step,
                        batch.native_history,
                        store,
                        device,
                        encoder_grad=False,
                        descriptions=episode.tracks,
                    ),
                    tracks=step.tracks if native_group is None else native_group.tracks,
                    event_ids=tuple(r.record.record_id for r in step.records)
                    if native_group is None
                    else native_group.event_ids,
                    role_ids=torch.tensor(
                        [step.roles if native_group is None else native_group.roles],
                        dtype=torch.long,
                        device=device,
                    ),
                    modalities=step.modalities if native_group is None else native_group.modalities,
                )
            )
            state = announce(state, step)
    for step in batch.steps:
        state = context_before(state, step)
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
            native_group = batch.native_history.group_for_records(step.offsets)
            if native_group is not None:
                for address, purpose, revision in native_group.track_descriptions:
                    state = multi.tracks.describe_track(state, address, purpose, revision)
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
            # Semantic executions are teacher-conditioned demonstrations:
            # predict from the committed prefix, score, then commit the actual
            # target representation before the next dependent meridian.
            # Ordinary grouped episodes retain their sampled recurrence.
            semantic_teacher = (
                episode.stream == "semantic_execution"
                and json.loads(episode.supervision.annotations_json)
                .get("semantic_execution", {})
                .get("continuation")
                == "teacher_conditioned"
            )
            teacher_forced = (
                semantic_teacher
                or json.loads(episode.supervision.annotations_json).get("continuation")
                == "recorded"
                or recorded_edit_base
                or any(
                    entry.record.action_id is not None
                    and any(
                        later.result_of == entry.record.action_id
                        for later in batch.windows[0].episode.records[step.offsets[-1] + 1 :]
                    )
                    for entry in step.records
                )
            )
            resolved = tuple(
                plan.resolve(store, device) if isinstance(plan, MediaPlan) else plan
                for plan in step.plans
            )
            plan_indices = (
                tuple(
                    getattr(batch.native_history.paired_tape(cursor), "index", 0)
                    for cursor in step.offsets
                )
                if native_group is not None
                else None
            )
            full_tracks = step.tracks if native_group is None else native_group.tracks
            full_event_ids = (
                tuple(r.record.record_id for r in step.records)
                if native_group is None
                else native_group.event_ids
            )
            full_controls = (
                tuple(
                    resolve_event_control(episode, event_by_record[rec.record.record_id])
                    for rec in step.records
                )
                if native_group is None
                else native_group.controls
            )
            result = multi.supervised_meridian(
                state,
                tracks=full_tracks,
                event_ids=full_event_ids,
                plans=resolved,
                control_payloads=full_controls,
                full_group=native_group,
                plan_indices=plan_indices,
                effort=batch.effort,
                teacher_forced_action=teacher_forced,
                relations=(
                    tuple(
                        (kind, plan_indices[left], plan_indices[right], weight)
                        for kind, left, right, weight in step.relations
                    )
                    if plan_indices is not None
                    else step.relations
                ),
                semantic_relations=tuple(
                    SelectedRelation(
                        kind,
                        plan_indices[left] if plan_indices is not None else left,
                        plan_indices[right] if plan_indices is not None else right,
                    )
                    for kind, left, right in step.semantic_relations
                ),
                evidence_relations=tuple(
                    (
                        kind,
                        plan_indices[index] if plan_indices is not None else index,
                        _earlier_evidence(
                            model,
                            earlier_steps[record],
                            batch.native_history,
                            store,
                            device,
                            episode.tracks,
                        ),
                    )
                    for kind, index, record in step.evidence_relations
                ),
                modality_masks=_modality_masks(model, step, device)
                if native_group is None
                else None,
            )
            state = result.state
            losses.append(ledger.backward_loss(result.losses))
            metrics["grouped_supervised_meridians"] += 1
            metrics["grouped_selected_events"] += len(
                step.tracks if native_group is None else native_group.tracks
            )
            for name, value in result.metrics.items():
                metrics[name] = metrics.get(name, 0) + (
                    value.detach() if isinstance(value, torch.Tensor) else value
                )
        else:
            native_group = batch.native_history.group_for_records(step.offsets)
            source_raw = (
                native_group.raw(device)
                if native_group is not None
                else _actual_raw(
                    model,
                    step,
                    batch.native_history,
                    store,
                    device,
                    encoder_grad=config.representation_training == "bootstrap",
                    descriptions=episode.tracks,
                )
            )
            state = (
                multi.commit_tape(state, native_group)
                if native_group is not None
                else multi.tracks.advance_group(
                    state,
                    source_raw,
                    tracks=step.tracks if native_group is None else native_group.tracks,
                    event_ids=tuple(r.record.record_id for r in step.records)
                    if native_group is None
                    else native_group.event_ids,
                    role_ids=torch.tensor(
                        [step.roles if native_group is None else native_group.roles],
                        dtype=torch.long,
                        device=device,
                    ),
                    modalities=step.modalities if native_group is None else native_group.modalities,
                )
            )
        state = announce(state, step)
    if batch.retrieval is not None and _question_scored(batch.prefix, batch.steps):
        if sequence.question_search is None:
            raise RuntimeError("The annotated lookup after the question did not run")
        terms, retrieved = retrieval_terms(
            sequence.question_search, sequence.sources, batch.retrieval, store
        )
        losses.append(ledger.backward_loss(terms))
        for name, value in retrieved.items():
            metrics[name] = metrics.get(name, 0) + value
    for name, loss in sequence.ingest_losses().items():
        weight = INGEST_VALUE_WEIGHT if name == "ingest_value" else INGEST_KEY_WEIGHT
        losses.append(loss * weight)
        metrics[f"{name}_loss"] = loss.detach()
    if differentiate and losses:
        torch.stack(losses).sum().backward()
    metrics["context_windows"] = 1
    metrics["context_retained_events_sum"] = len(state.history_event_ids)
    metrics["context_cache_bytes_sum"] = sum(
        value.numel() * value.element_size()
        for value in (state.history.keys, state.history.values)
    )
    return metrics
