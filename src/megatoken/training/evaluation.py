"""Fixed-harness diagnostics separating next-facet prediction and reconstruction."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import encode_controls, resolve_event_control
from megatoken.data import encode_record
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.knowledge import episode_metadata
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord
from megatoken.generation.structure import ActionSpec
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.training.engine import episode_control_payload, ingest_records
from megatoken.training.history import NativeHistory, replay_raw_tape


def wilson_interval(
    successes: int, attempts: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if attempts == 0:
        return (0.0, 1.0)
    if not 0 <= successes <= attempts:
        raise ValueError("Success count must be between zero and attempts")
    rate = successes / attempts
    scale = 1 + z * z / attempts
    center = (rate + z * z / (2 * attempts)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / attempts + z * z / (4 * attempts * attempts)) / scale
    return max(0.0, center - radius), min(1.0, center + radius)


@torch.no_grad()
def evaluate_records(
    model: Any, dataset: Any, codec: Any, config: Any, device: torch.device, *, episodes: int
) -> dict[str, Any]:
    """Causal CE and actual samples under the phase's representation mode.

    The target is never supplied as a control, sampled prefix, or exact-length
    hint. The sample limit is the configured public capacity for every target.
    Bootstrap measures noisy raw-codec reconstruction and teacher-forces E.
    Recurrent auxiliary reconstruction follows E -> frozen C -> frozen R from
    the same detached causal context; it does not replace the live prediction.
    Recurrent evaluation keeps sampled facets live between consecutive targets.
    """
    was_training = model.training
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    exact = attempted = legal = 0
    episodes = min(episodes, len(dataset))
    try:
        for episode in dataset.iter_episodes():
            if counts["episodes"] >= episodes:
                break
            counts["episodes"] += 1
            state = model.core.initial_state(1, device=device)
            native = NativeHistory.from_episode(episode, model)
            control_payload = episode_control_payload(episode)
            control = encode_controls(model.core, [control_payload], encoder_grad=False)
            for cursor, record in enumerate(episode.records):
                encoded = encode_record(record, codec)
                scored = record.policy_mask and len(encoded.token_ids) <= config.generated_capacity
                for tape in native.events_for_record(cursor, include_record=not scored):
                    state = replay_raw_tape(model, state, tape)
                if not scored:
                    if not native.has_generated_record(cursor):
                        state = ingest_records(model, state, [encoded], device)
                    continue
                sample = model.sample(
                    state,
                    spec=ActionSpec(max_tokens=config.generated_capacity),
                    effort=config.effort,
                    control_payload=control_payload,
                )
                attempted += 1
                legal += int(sample.tape.valid)
                exact += int(sample.tape.valid and sample.payload == record.payload)
                # Both diagnostics start from the same causal state. The
                # supervised branch then advances under the configured mode.
                plan = model.prepare_target(
                    target_ids=encoded.token_ids,
                    free_mask=encoded.free_mask,
                    terminal=record.part_index + 1 == record.part_count,
                    representation_mode=config.representation_training,
                    native_tapes=[native.paired_tape(cursor)],
                )
                step = model.supervised_step(
                    state,
                    plan=plan,
                    effort=config.effort,
                    control=control,
                    role_ids=torch.tensor([ROLE_TO_ID[record.role]], device=device),
                )
                state = step.state
                for name, (numerator, denominator) in step.losses.items():
                    sums[name] += float(numerator)
                    counts[name] += denominator
        return {
            "evaluation_mode": "model_only_diagnostic",
            "episodes": episodes,
            "representation_mode": config.representation_training,
            "prior_lexical_ce": sums["lexical"] / max(1, counts["lexical"]),
            "reconstruction_lexical_ce": (
                sums["reconstruction"] / counts["reconstruction"]
                if counts["reconstruction"]
                else None
            ),
            "facet_kl_per_coordinate": (
                sums["facet_prediction"] / counts["facet_prediction"]
                if counts["facet_prediction"]
                else None
            ),
            "encoder_alignment_mse": (
                sums["encoder_alignment"] / counts["encoder_alignment"]
                if counts["encoder_alignment"]
                else None
            ),
            "prior_scored_positions": counts["lexical"],
            "generated_chunks": attempted,
            "prior_exact_chunks": exact,
            "prior_exact_rate": exact / max(1, attempted),
            "prior_exact_wilson95": wilson_interval(exact, attempted),
            "valid_generation_rate": legal / max(1, attempted),
            "sample_capacity_limit": config.generated_capacity,
            "note": "Untrained/small-model diagnostics, not an agent benchmark score",
        }
    finally:
        model.train(was_training)


@torch.inference_mode()
def evaluate_served_records(
    model: Any,
    dataset: Any,
    config: Any,
    device: torch.device,
    *,
    episodes: int,
    seed: int = 17,
    collections: dict[str, str] | None = None,
    bases: dict[str, str] | None = None,
    trained_weight: float = 1.0,
    max_sources: int = 2,
    threshold: float | None = None,
    chunking: ChunkingConfig | None = None,
    output_specs: dict[str, ActionSpec | MediaOutputSpec] | None = None,
) -> dict[str, Any]:
    was_training = model.training
    runtime = None
    try:
        runtime = MultiTrackRuntime(
            MultiTrackGenerator(model),
            chunking=chunking,
            collections=collections,
            output_specs=output_specs,
        )
        return _evaluate_served_runtime(
            runtime,
            model,
            dataset,
            config,
            device,
            episodes=episodes,
            seed=seed,
            bases=bases,
            trained_weight=trained_weight,
            max_sources=max_sources,
            threshold=threshold,
            collections=collections,
        )
    finally:
        try:
            if runtime is not None:
                runtime.close()
                for collection in runtime.memory.collections.values():
                    collection.close()
        finally:
            model.train(was_training)


def _evaluate_served_runtime(
    runtime: MultiTrackRuntime,
    model: Any,
    dataset: Any,
    config: Any,
    device: torch.device,
    *,
    episodes: int,
    seed: int,
    bases: dict[str, str] | None,
    trained_weight: float,
    max_sources: int,
    threshold: float | None,
    collections: dict[str, str] | None,
) -> dict[str, Any]:
    """Score sampled outputs after real observations and session retrieval.

    The episode is played through :class:`MultiTrackRuntime`: all observed
    events in a causal group commit together, and policy targets are sampled
    from the previous committed state. Media observations use checked prepared
    features. Generated media needs a supplied capability or an explicitly
    requested public extent; hidden target geometry never becomes a request.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if max_sources < 1:
        raise ValueError("max_sources must be positive")
    mounted_bases: dict[str, float] = {}
    for name, path in (bases or {}).items():
        runtime.memory.open_base(name, path)
        mounted_bases[name] = 1.0

    exact = attempted = text_attempted = valid = 0
    completed_episodes = skipped_media = skipped_mixed = 0
    decoded_media = 0
    lookups = collection_retrievals = 0
    for episode_index, episode in enumerate(dataset.iter_episodes()):
        if completed_episodes + skipped_media + skipped_mixed >= episodes:
            break
        if any(
            record.media_type not in {"text/plain; charset=utf-8", MEDIA_RECORD_TYPE}
            for record in episode.records
        ):
            skipped_media += 1
            continue
        records = {record.record_id: record for record in episode.records}
        event_by_id = {
            event.record_id: event for meridian in episode.meridians for event in meridian.events
        }
        media_specs = {}
        missing_public_spec = False
        for record in episode.records:
            if not record.policy_mask or record.media_type != MEDIA_RECORD_TYPE:
                continue
            media = MediaRecord.from_bytes(record.payload)
            if media.modality in runtime.output_specs:
                media_specs[record.record_id] = runtime.output_specs[media.modality]
            elif (
                media.output_geometry is not None
                and media.output_geometry.provenance == "requested"
            ):
                media_specs[record.record_id] = MediaOutputSpec.from_geometry(media.output_geometry)
            else:
                missing_public_spec = True
        if missing_public_spec:
            skipped_media += 1
            continue
        if episode.meridians:
            groups = [
                [(event.track_id, records[event.record_id]) for event in meridian.events]
                for meridian in episode.meridians
            ]
        else:
            groups = [[(record.role, record)] for record in episode.records]
        if any(
            any(record.policy_mask for _, record in group)
            and not all(record.policy_mask for _, record in group)
            for group in groups
        ):
            skipped_mixed += 1
            continue

        session = runtime.open(
            metadata=episode_metadata(episode),
            bases=mounted_bases,
            trained=trained_weight,
            effort=config.effort,
            collections=None,
            max_sources=max_sources,
            threshold=threshold,
        )
        for track in episode.tracks:
            if getattr(track, "description", ""):
                session.memory.state = runtime.generator.tracks.describe_track(
                    session.state, track.track_id, track.description
                )
        control_payload = episode_control_payload(episode)
        # An item may be returned at open, after a user observation, and after
        # a generated terminal response. Count the real source-session log.
        sources = session.memory.sources
        for group_index, group in enumerate(groups):
            if group[0][1].policy_mask:
                tracks = tuple(track for track, _ in group)
                specs = tuple(
                    media_specs[record.record_id]
                    if record.record_id in media_specs
                    else ActionSpec(
                        max_tokens=config.generated_capacity,
                        require_terminal=(record.part_index + 1 == record.part_count),
                    )
                    for _, record in group
                )
                rng = torch.Generator(device=device).manual_seed(
                    seed + episode_index * 1_000_003 + group_index
                )
                generated = runtime.generate(
                    session,
                    tracks=tracks,
                    event_ids=tuple(record.record_id for _, record in group),
                    specs=specs,
                    controls=(
                        tuple(
                            resolve_event_control(episode, event_by_id[record.record_id])
                            for _, record in group
                        )
                        if all(hasattr(track, "track_id") for track in episode.tracks)
                        else [control_payload] * len(group)
                    ),
                    effort=config.effort,
                    generator=rng,
                )
                for index, ((_, record), payload, is_valid) in enumerate(
                    zip(group, generated.tape.payloads, generated.tape.valid, strict=True)
                ):
                    attempted += 1
                    valid += int(is_valid)
                    if record.record_id in media_specs:
                        tensor = generated.media_outputs[index]
                        if is_valid and tensor is not None and torch.isfinite(tensor).all():
                            decoded_media += 1
                    else:
                        text_attempted += 1
                        exact += int(is_valid and payload == record.payload)
                runtime.release_generation(session, session.revision)
            else:
                stores = getattr(dataset, "stores", ())
                if getattr(dataset, "assets", None) is not None:
                    stores = (dataset.assets,)
                observed = []
                for track, record in group:
                    if record.media_type == MEDIA_RECORD_TYPE:
                        media = MediaRecord.from_bytes(record.payload)
                        store = next(
                            (s for s in stores if (s.root / media.source.relative_path).is_file()),
                            None,
                        )
                        if store is None:
                            raise ValueError(
                                "Evaluation media observation lacks its checked asset store"
                            )
                        source = torch.from_numpy(store.load(media.source)).to(device)[None]
                        observed.append(
                            ObservedEvent(
                                track,
                                "compressed_text" if media.modality == "text" else media.modality,
                                source,
                                role_id=ROLE_TO_ID[record.role],
                                extractor=media.source.extractor,
                                terminal=record.part_index + 1 == record.part_count,
                                event_id=record.record_id,
                            )
                        )
                    else:
                        observed.append(
                            ObservedEvent(
                                track,
                                "text",
                                record.payload,
                                role_id=ROLE_TO_ID[record.role],
                                terminal=record.part_index + 1 == record.part_count,
                                event_id=record.record_id,
                            )
                        )
                runtime.observe(session, tuple(observed), effort=config.effort)
            for _, record in group:
                for revision in getattr(episode, "track_revisions", ()):
                    if revision.record_id == record.record_id:
                        session.memory.state = runtime.generator.tracks.describe_track(
                            session.state, revision.track_id, revision.purpose, revision.revision
                        )
        completed_episodes += 1
        lookups += session.memory.agenda.lookups
        collection_retrievals += 0 if sources is None else len(sources.log)

    evaluated = completed_episodes
    if evaluated == 0 or attempted == 0:
        raise ValueError(
            "No generated text targets could be evaluated; "
            f"skipped_media={skipped_media}, skipped_mixed_meridians={skipped_mixed}"
        )
    report = {
        "evaluation_mode": "served_session_generation",
        "episodes": evaluated,
        "skipped_media_episodes": skipped_media,
        "skipped_mixed_meridian_episodes": skipped_mixed,
        "generated_chunks": attempted,
        "generated_text_chunks": text_attempted,
        "decoded_media_events": decoded_media,
        "media_quality_established": False,
        "valid_generation_rate": valid / max(1, attempted),
        "exact_chunk_rate": exact / max(1, text_attempted),
        "exact_chunks": exact,
        "session_lookups": lookups,
        "collection_items_retrieved": collection_retrievals,
        "mounted_knowledge_bases": sorted(mounted_bases),
        "mounted_collections": sorted(collections or {}),
        "trained_bank_weight": trained_weight,
        "max_sources_per_lookup": max_sources,
        "collection_score_threshold": threshold,
        "note": (
            "Generated text chunk agreement after served session lookups; this is a diagnostic, "
            "not a task benchmark. Media without public output bounds or mixed causal groups "
            "are counted as skipped; decoded media availability alone does not measure quality."
        ),
    }
    return report
