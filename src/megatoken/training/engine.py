"""A single optimizer engine for bootstrap through student-state distillation.

Windows are independent optimizer transactions: reconstruct the actual prefix
under current weights, retain gradients through selected writer/reader records,
and discard all runtime state after updating parameters. No stale cache survives
an update. Outcome RL uses the same optimizer/checkpoints but complete tapes.
"""

from __future__ import annotations

import dataclasses
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import build_control_payload, encode_controls
from megatoken.data import PreparedDataset, WindowConfig, stage_recipe
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.media_assets import MediaRecord
from megatoken.io import content_hash, write_json
from megatoken.loading import build_model, load_codec, load_model_config, parameter_report
from megatoken.runtime import capture_rng, doctor, restore_rng, seed_everything
from megatoken.tensors import byte_tensor
from megatoken.tool_schema import (
    compile_tool_spec,
    matching_schema,
    recent_copy_sources,
    supervised_bindings,
)
from megatoken.training.checkpoint import load_checkpoint, resolve_checkpoint, save_checkpoint
from megatoken.training.config import DEFAULT_LOSS_WEIGHTS, TrainConfig
from megatoken.training.distributed import DistributedContext
from megatoken.training.diloco_client import DecoupledWorker
from megatoken.training.diloco_protocol import training_identity
from megatoken.training.history import NativeHistory, replay_raw_tape
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import (
    MeridianPreparedBatch,
    forward_meridian_batch,
    prepare_meridian_batch,
)
from megatoken.training.metrics import (
    Counters,
    JsonlLogger,
    dispatch_summary,
    materialize_scalars,
    rates,
)
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.schedule import PHASES
from megatoken.training.stream import (
    EpisodeWindow,
    MixtureWindowStream,
    IndependentWindowStream,
    WindowStream,
    compatible_batches,
)


@dataclass
class PreparedBatch:
    windows: list[EpisodeWindow]
    targets: dict[int, Any]
    counts: dict[str, int]
    effort: int = 6
    native_history: NativeHistory | None = None


def ingest_records(
    model: Any, state: Any, records: list[Any], device: torch.device, *, encoder_grad: bool = True
):
    payloads = [record.payload for record in records]
    ids, lengths = byte_tensor(payloads, device)
    roles = torch.tensor([ROLE_TO_ID[record.record.role] for record in records], device=device)
    return model.core.observe(
        ids, state, lengths=lengths, role_ids=roles, encoder_grad=encoder_grad
    )


def episode_control_payload(episode: Any) -> bytes:
    # Controls have explicit provenance and are never constructed from a
    # matching target or a hidden verifier annotation.
    return build_control_payload(
        seed=episode.control_seed,
        reference=episode.control_reference,
        reference_origin=episode.reference_origin,
        tools_json=episode.tools_json,
    )


def make_control(
    model: Any, windows: list[EpisodeWindow], device: torch.device, *, encoder_grad: bool = True
) -> torch.Tensor | None:
    payloads = [episode_control_payload(item.episode) for item in windows]
    return encode_controls(model.core, payloads, encoder_grad=encoder_grad)


# The trajectory-reference teacher is deliberately confined to supervised
# training. It is never serialized as a causal control or supplied to serving/RL.
_RECURRENT_REFERENCE_FRACTION = 0.5
_RECURRENT_REFERENCE_NOISE_STD = 0.1
_RECURRENT_REFERENCE_BYTES_PER_QUARTER = 1024


def _successful_trajectory_reference(episode: Any) -> bytes:
    """A bounded, chronological reference from a verified successful execution.

    The teacher sees future recorded content by design. Never include hidden
    supervision labels, rewards or verifier annotations. Four chronological
    slices retain both early instructions and later actions/observations without
    making control encoding proportional to a long full transcript.
    """
    records = episode.records
    frames = [b"<" + record.role.encode("ascii") + b">\n" + record.payload for record in records]
    sections = []
    for quarter in range(4):
        start = quarter * len(frames) // 4
        end = (quarter + 1) * len(frames) // 4
        chunk = b"\n".join(frames[start:end])
        limit = _RECURRENT_REFERENCE_BYTES_PER_QUARTER
        if len(chunk) > limit:
            chunk = chunk[: limit // 2] + chunk[-(limit - limit // 2) :]
        if chunk:
            sections.append(chunk)
    return b"\n".join(sections)


def make_training_control(
    model: Any, windows: list[EpisodeWindow], device: torch.device, *, recurrent: bool
) -> tuple[torch.Tensor | None, int]:
    """Mix causal controls with detached noisy successful-trajectory references.

    In recurrent phases, half of eligible verified-success windows use a
    teacher reference from the *recorded* trajectory. Its complete encoded
    vector is detached before Gaussian perturbation: training the predictor
    cannot push the byte/control encoder to memorize a future target.
    Remaining windows use the ordinary trainable, serving-compatible causal
    control. One choice and one noise draw is reused across a window's chunks.
    """
    if not recurrent:
        return make_control(model, windows, device, encoder_grad=True), 0
    eligible = [
        index
        for index, item in enumerate(windows)
        if item.episode.supervision.verified_success
        and any(record.policy_mask for record in item.episode.records)
    ]
    selection = torch.rand(len(eligible)).tolist()
    selected = [
        index
        for index, draw in zip(eligible, selection, strict=True)
        if draw < _RECURRENT_REFERENCE_FRACTION
    ]
    # Encode each window *once*: the teacher and causal rows are disjoint.
    selected_set = set(selected)
    causal_payloads = [
        b"" if index in selected_set else episode_control_payload(item.episode)
        for index, item in enumerate(windows)
    ]
    causal = encode_controls(model.core, causal_payloads, encoder_grad=False)
    if not selected:
        return causal, 0
    # Explicitly training-only: this adds the *current* successful trajectory's
    # reference, not an external/earlier-history reference permitted at inference.
    teacher_payloads = [
        episode_control_payload(windows[index].episode)
        + b"\n<reference>\n"
        + _successful_trajectory_reference(windows[index].episode)
        for index in selected
    ]
    with torch.no_grad():
        teacher = encode_controls(model.core, teacher_payloads, encoder_grad=False)
        assert teacher is not None
        teacher = teacher.detach().float()
        teacher = teacher + _RECURRENT_REFERENCE_NOISE_STD * torch.randn_like(teacher)
    if len(selected) == len(windows):
        return teacher, len(selected)
    if causal is None:
        control = teacher.new_zeros((len(windows), model.config.control_dim))
    else:
        control = causal.clone()
    indices = torch.tensor(selected, device=device, dtype=torch.long)
    return control.index_copy(0, indices, teacher.to(control.dtype)), len(selected)


def prepare_batches(
    model: Any, windows: list[EpisodeWindow], config: TrainConfig
) -> list[PreparedBatch]:
    result = []
    raw_windows, action_windows, meridian_windows = [], [], []
    native_histories: dict[int, NativeHistory] = {}
    for item in windows:
        if id(item.episode) not in native_histories:
            native_histories[id(item.episode)] = NativeHistory.from_episode(item.episode, model)
        if any(
            len(record.payload) > config.max_record_bytes
            for record in (*item.window.prefix, *item.window.records)
        ):
            raise ValueError(
                "A physical record exceeds max_record_bytes; prepare at the configured byte limit"
            )
        verified_streams = {
            "verified_trajectories",
            "verified_successful_multiturn",
            "tested_artifacts",
            "successful_replay",
            "long_verified_dependencies",
        }
        if (
            not config.allow_unverified_data
            and item.episode.stream in verified_streams
            and not item.episode.supervision.verified_success
        ):
            raise ValueError(
                f"Stream {item.episode.stream} requires independent success verification"
            )
        if item.episode.stream == "long_verified_dependencies" and not item.episode.dependencies:
            raise ValueError(
                "Long dependency data requires explicit causal writer/reader annotations"
            )
        if item.episode.meridians:
            meridian_windows.append(item)
        elif native_histories[id(item.episode)].has_events or (
            item.episode.tools_json != "[]"
            and any(record.record.action_id for record in item.window.records)
        ):
            action_windows.append(item)
        else:
            raw_windows.append(item)
    batches = compatible_batches(raw_windows, config.microbatch_episodes) + [
        [item] for item in action_windows
    ]
    for batch in batches:
        targets = {}
        counts = dict.fromkeys(DEFAULT_LOSS_WEIGHTS, 0)
        for offset in range(len(batch[0].window.records)):
            records = [item.window.records[offset] for item in batch]
            if not any(record.record.policy_mask for record in records):
                continue
            if not all(record.record.policy_mask for record in records):
                raise ValueError("Incompatible record ownership in a supervised microbatch")
            largest = max(len(record.token_ids) for record in records)
            if largest > config.generated_capacity:
                raise ValueError(
                    f"Target has {largest} codec positions, exceeding generated_capacity="
                    f"{config.generated_capacity}; prepare smaller physical records or "
                    "explicitly raise the trained capacity"
                )
            capacity = min(
                config.generated_capacity, max(1, 2 ** math.ceil(math.log2(max(1, largest))))
            )
            plan = None
            native_tapes = [
                native_histories[id(item.episode)].paired_tape(item.window.start + offset)
                for item in batch
            ]
            if (
                len(batch) == 1
                and batch[0].episode.tools_json != "[]"
                and records[0].record.action_id
                and records[0].record.part_count == 1
            ):
                item, record = batch[0], records[0]
                spec = compile_tool_spec(
                    item.episode.tools_json,
                    config.generated_capacity,
                    wire_formats=("canonical", "openai", "anthropic"),
                )
                earlier = [
                    entry.payload for entry in (*item.window.prefix, *item.window.records[:offset])
                ]
                spec = dataclasses.replace(spec, copy_sources=recent_copy_sources(earlier))
                schema_index = matching_schema(spec, record.payload)
                if schema_index is not None:
                    bindings = supervised_bindings(spec, schema_index, record.payload)
                    for layout in range(len(spec.schemas[schema_index].orders)):
                        options = {
                            "spec": spec,
                            "schema_index": schema_index,
                            "bindings": bindings,
                            "layout": layout,
                            "representation_mode": config.representation_training,
                        }
                        candidate = model.prepare_action_target(record.payload, **options)
                        # Preserve immutable original history. Serialization
                        # variants not represented by a public schema use raw CE.
                        if candidate.payload == record.payload:
                            # Native pairs must match the exact selected wire
                            # layout. Do not validate against rejected candidate
                            # serializations encountered while searching it.
                            plan = (
                                model.prepare_action_target(
                                    record.payload, **options, native_tape=native_tapes[0]
                                )
                                if native_tapes[0] is not None
                                else candidate
                            )
                            break
            if plan is None:
                plan = model.prepare_target(
                    [record.token_ids for record in records],
                    free_mask=[record.free_mask for record in records],
                    capacity=capacity,
                    terminal=[
                        record.record.part_index + 1 == record.record.part_count
                        for record in records
                    ],
                    representation_mode=config.representation_training,
                    native_tapes=native_tapes,
                )
            targets[offset] = plan
            for name, count in plan.denominators.items():
                counts[name] += count
        effort = config.effort
        if config.use_effort_mixture:
            mixture = stage_recipe(config.phase).effort_mix
            choices = sorted(mixture)
            effort = choices[
                int(torch.multinomial(torch.tensor([mixture[key] for key in choices]), 1))
            ]
        native = native_histories[id(batch[0].episode)]
        result.append(
            PreparedBatch(batch, targets, counts, effort, native if native.has_events else None)
        )
    for item in meridian_windows:
        result.append(
            prepare_meridian_batch(model, item, native_histories[id(item.episode)], config)
        )
    return result


def batch_counters(batches: list[PreparedBatch | MeridianPreparedBatch]) -> Counters:
    result = Counters()
    for batch in batches:
        for item in batch.windows:
            window = item.window
            prefix_tokens = sum(record.tokens for record in window.prefix)
            new_tokens = sum(record.tokens for record in window.records)
            result.add(
                Counters(
                    D=new_tokens,
                    I=new_tokens + prefix_tokens,
                    G=len(window.records),
                    source_bytes=sum(
                        len(record.payload) for record in (*window.prefix, *window.records)
                    ),
                    record_ingestions=len(window.prefix) + len(window.records),
                    reconstruction_tokens=prefix_tokens,
                    reconstruction_records=len(window.prefix),
                    scored_windows=int(bool(window.P)),
                    writer_reader_windows=int(bool(window.writer_reader_pairs)),
                )
            )
        if isinstance(batch, MeridianPreparedBatch):
            for step in batch.steps:
                for modality, record in zip(step.modalities, step.records, strict=True):
                    if modality not in {"audio", "image", "video"}:
                        continue
                    reference = MediaRecord.from_bytes(record.payload)
                    result.media_events += 1
                    asset = reference.target or reference.source
                    if modality == "audio" and asset.kind == "raw":
                        result.audio_samples += asset.shape[0]
                    elif modality == "video" and asset.kind == "raw":
                        result.video_frames += asset.shape[0]
                    elif modality == "image":
                        result.image_events += 1
            result.P += sum(
                plan.scored_positions for step in batch.steps for plan in step.plans
            )
            result.facet_encodes += sum(len(step.plans) for step in batch.steps)
            if batch.native_history.has_events:
                for step in (*batch.prefix, *batch.steps):
                    for cursor in step.offsets:
                        result.raw_facet_replays += len(
                            batch.native_history.events_for_record(
                                cursor, include_record=False,
                            )
                        ) + int(batch.native_history.paired_tape(cursor) is not None)
            if any(episode_control_payload(item.episode) for item in batch.windows):
                result.control_encodes += sum(len(step.plans) for step in batch.steps)
            continue
        result.P += sum(plan.scored_positions for plan in batch.targets.values())
        result.facet_encodes += sum(len(batch.windows) for _ in batch.targets)
        if batch.native_history is not None:
            window = batch.windows[0].window
            for cursor in range(window.end):
                events = batch.native_history.events_for_record(
                    cursor,
                    include_record=(
                        cursor < window.start or cursor - window.start not in batch.targets
                    ),
                )
                result.raw_facet_replays += len(events)
        if any(episode_control_payload(item.episode) for item in batch.windows):
            result.control_encodes += len(batch.windows)
    return result


def forward_batch(
    model: Any,
    batch: PreparedBatch | MeridianPreparedBatch,
    ledger: LossLedger,
    config: TrainConfig,
    device: torch.device,
    model_version: int,
) -> dict[str, float | torch.Tensor]:
    if isinstance(batch, MeridianPreparedBatch):
        return forward_meridian_batch(model, batch, ledger, config, device, model_version)
    state = model.core.initial_state(len(batch.windows), device=device, model_version=model_version)
    with torch.no_grad():
        for offset in range(len(batch.windows[0].window.prefix)):
            if batch.native_history is not None:
                for tape in batch.native_history.events_for_record(offset, include_record=True):
                    state = replay_raw_tape(model, state, tape)
                if batch.native_history.has_generated_record(offset):
                    continue
            state = ingest_records(
                model, state, [item.window.prefix[offset] for item in batch.windows], device
            )
    control, teacher_control_examples = make_training_control(
        model, batch.windows, device, recurrent=config.representation_training == "recurrent"
    )
    losses = []
    metrics: dict[str, float | torch.Tensor] = {
        "noisy_trajectory_control_examples": teacher_control_examples,
    }
    for offset in range(len(batch.windows[0].window.records)):
        records = [item.window.records[offset] for item in batch.windows]
        cursor = batch.windows[0].window.start + offset
        if batch.native_history is not None:
            for tape in batch.native_history.events_for_record(
                cursor, include_record=offset not in batch.targets
            ):
                state = replay_raw_tape(model, state, tape)
        if offset in batch.targets:
            roles = torch.tensor(
                [ROLE_TO_ID[record.record.role] for record in records], device=device
            )
            plan = batch.targets[offset]
            # The recorded tool result is only causally compatible with its
            # recorded action. Preserve that action's native raw tape when
            # available, otherwise teacher-force E(recorded action) solely for
            # the *successor* state. The sampled prior branch still receives
            # its own ordinary lexical supervision.
            teacher_forced_action = any(
                (action_id := item.window.records[offset].record.action_id) is not None
                and any(
                    future.result_of == action_id for future in item.episode.records[cursor + 1 :]
                )
                for item in batch.windows
            )
            if hasattr(plan, "spec"):
                step = model.supervised_action(
                    state,
                    plan=plan,
                    effort=batch.effort,
                    control=control,
                    teacher_forced_action=teacher_forced_action,
                )
            else:
                step = model.supervised_step(
                    state,
                    plan=plan,
                    effort=batch.effort,
                    control=control,
                    role_ids=roles,
                    teacher_forced_action=teacher_forced_action,
                )
            state = step.state
            losses.append(ledger.backward_loss(step.losses))
            for key, value in step.metrics.items():
                if isinstance(value, (float, int, torch.Tensor)):
                    metrics[key] = metrics.get(key, 0.0) + (
                        value.detach() if isinstance(value, torch.Tensor) else value
                    )
        else:
            if batch.native_history is not None and batch.native_history.has_generated_record(
                cursor
            ):
                continue
            state = ingest_records(
                model,
                state,
                records,
                device,
                encoder_grad=config.representation_training == "bootstrap",
            )
    if losses:
        torch.stack(losses).sum().backward()
    # Retain graph only inside this differentiable writer-to-reader window.
    return metrics


def _resume_compatible(old: dict[str, Any], config: TrainConfig, phase_transition: bool) -> None:
    current = asdict(config)
    critical = (
        "model",
        "model_overrides",
        "codec",
        "precision",
        "expert_backend",
        "engram_import",
        "weight_decay",
        "inner_optimizer",
        "decoupled",
    )
    if not phase_transition:
        critical += (
            "phase",
            "dataset",
            "seed",
            "microbatch_episodes",
            "max_history_tokens",
            "max_window_records",
            "scored_decisions",
            "generated_capacity",
            "accumulate_scored",
            "accumulate_ingested",
            "max_accumulation_windows",
            "use_stage_mixture",
            "use_effort_mixture",
            "effort",
            "loss_weights",
            "grad_clip",
            "budget",
            "deterministic",
            "gradient_checkpointing",
            "representation_mode",
        )
    for key in critical:
        if old.get(key) != current.get(key):
            raise ValueError(
                f"Resume configuration changed {key}; use an explicit phase transition "
                "only when the architecture/codec are unchanged"
            )


def train(
    config: TrainConfig, *, resume: str | None = None, phase_transition: bool = False
) -> dict[str, Any]:
    if config.phase == "R":
        raise ValueError("Outcome RL requires executed episodes: use the rl command")
    if phase_transition and not resume:
        raise ValueError("A phase transition requires the previous phase's checkpoint")
    if config.decoupled is not None and (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or (torch.distributed.is_available() and torch.distributed.is_initialized())
    ):
        raise ValueError("Decoupled learners require one independent process per sandbox, not torchrun")
    context = DistributedContext.initialize(config.device)
    output = Path(config.output)
    if config.decoupled is not None:
        output = output / f"worker-{config.decoupled.resolved_worker_id:02d}"
    checkpoint_store = None
    if config.decoupled is not None and config.decoupled.checkpoint_s3:
        from megatoken.training.diloco_storage import CheckpointStore

        checkpoint_store = CheckpointStore(
            config.decoupled.checkpoint_s3,
            config.decoupled.run_id,
            config.decoupled.resolved_worker_id,
        )
        if resume is None:
            restored = checkpoint_store.restore(output / "checkpoints")
            if restored is not None:
                resume = str(restored)
    logger = JsonlLogger(output / "metrics.jsonl", context.is_primary)
    diloco = None
    optimizer = None
    dataset = stream = None
    old_handlers = {}
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    try:
        diagnostics = doctor(
            require_cuda=config.require_blackwell, require_blackwell=config.require_blackwell
        )
        if not diagnostics["ok"]:
            raise RuntimeError("Runtime preflight failed: " + "; ".join(diagnostics["errors"]))
        if config.codec == "byte" and config.model != "tiny" and not config.allow_byte_codec:
            raise ValueError(
                "Large runs require the pinned raw codec; allow_byte_codec is only an explicit experiment"
            )
        if context.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        seed_everything(config.seed, config.deterministic)
        overrides = {
            **config.model_overrides,
            "expert_backend": config.expert_backend,
            "checkpoint_stages": config.gradient_checkpointing,
        }
        model_config = load_model_config(config.model, **overrides)
        if config.generated_capacity > model_config.max_capacity:
            raise ValueError("generated_capacity exceeds the model's maximum output positions")
        codec = load_codec(config.codec, vocab_size=model_config.vocab_size)
        model = build_model(model_config, codec, device=context.device, precision=config.precision)
        if config.pretrained_packs is not None:
            from megatoken.omni.loader import install_pretrained_packs
            install_pretrained_packs(
                model, config.pretrained_packs, device=context.device,
                dtype=torch.bfloat16 if config.precision == "bf16" else torch.float32,
            )
        payload = None
        if resume:
            payload, _ = load_checkpoint(resume, map_location="cpu")
            _resume_compatible(payload["train_config"], config, phase_transition)
            if payload["codec_fingerprint"] != codec.fingerprint:
                raise ValueError("Checkpoint codec identity differs")
            if payload["model_config"] != model_config.to_dict():
                raise ValueError("Checkpoint model configuration differs")
            # Dynamic temporary imports must be restored before optimizer owner
            # enumeration, or their FP32 masters and hooks would be missing.
            model.load_state_dict(payload["model"], strict=True)
        elif config.engram_import:
            from megatoken.assets import install_engram_asset

            install_engram_asset(model, config.engram_import)
        context.broadcast_model(model)
        phase = dataclasses.replace(
            PHASES[config.phase],
            budget=config.budget or PHASES[config.phase].budget,
            meter="G" if config.budget_meter == "G" else PHASES[config.phase].meter,
        )
        optimizer = MasterAdamW(
            model, phase.learning_rate(0), config.weight_decay,
            inner_optimizer=config.inner_optimizer,
        )
        if config.representation_training == "recurrent":
            # Contextual codec auxiliaries use frozen C/R with E gradients only.
            # Independent clipping prevents their norm scaling the main update.
            optimizer.isolate_gradient_parameters("encoder", model.core.encoder_parameters())
        windows_config = WindowConfig(
            max_window_tokens=config.max_history_tokens,
            max_decisions=config.scored_decisions,
            max_records=config.max_window_records,
            respect_dependencies=config.phase == "LC",
            max_dependency_tokens=config.max_history_tokens,
        )
        dataset = PreparedDataset(config.dataset)
        if dataset.manifest.get("codec_fingerprint", codec.fingerprint) != codec.fingerprint:
            raise ValueError("Prepared dataset uses a different codec")
        chunking = ChunkingConfig.from_identity(dataset.manifest["chunking"], codec)
        chunking_identity = chunking.identity(codec)
        if (
            payload is not None
            and not phase_transition
            and payload["chunking"] != chunking_identity
        ):
            raise ValueError("Exact resume changed the semantic chunking policy")
        data_rank = config.decoupled.resolved_worker_id if config.decoupled else context.rank
        data_world = config.decoupled.workers if config.decoupled else context.world_size
        if config.omni_recipe is not None:
            from megatoken.training.omni_stream import OmniWindowStream, load_omni_recipe
            stream = OmniWindowStream(
                config.dataset, codec, windows_config,
                load_omni_recipe(config.omni_recipe), "G",
                seed=config.seed, rank=data_rank, world_size=data_world,
                reduce_counts=context.sum_counts,
                independent=config.decoupled is not None,
            )
        elif config.use_stage_mixture:
            recipe = stage_recipe(config.phase)
            stream = MixtureWindowStream(
                config.dataset,
                codec,
                windows_config,
                recipe.mixture,
                phase.meter,
                seed=config.seed,
                rank=data_rank,
                world_size=data_world,
                reduce_counts=context.sum_counts,
                independent=config.decoupled is not None,
            )
        else:
            stream = (
                IndependentWindowStream(
                    dataset, codec, windows_config, seed=config.seed,
                    rank=data_rank, world_size=data_world,
                )
                if config.decoupled is not None
                else WindowStream(
                    dataset, codec, windows_config, seed=config.seed,
                    rank=context.rank, world_size=context.world_size,
                    reduce_counts=context.sum_counts,
                )
            )
        if config.decoupled is not None:
            diloco = DecoupledWorker(
                config.decoupled,
                model_config=model_config.to_dict(),
                codec_fingerprint=codec.fingerprint,
                masters=optimizer.masters,
                output=output / "diloco",
                token_budget=phase.budget,
                train_identity=training_identity(config),
                dataset_fingerprint=content_hash(dataset.manifest),
            )
            if payload is None:
                diloco.bootstrap(model, optimizer)
        global_step = 0
        consumed = 0
        counters = Counters()
        if payload is not None:
            optimizer.load_state_dict(payload["optimizer"])
            if diloco is not None:
                if phase_transition:
                    raise ValueError("Start a new coordinator run for a phase transition")
                diloco.resume(payload["diloco"], optimizer)
            global_step = payload["global_step"]
            if not phase_transition:
                if len(payload["rank_state"]) != context.world_size:
                    raise ValueError("Exact resume requires the checkpoint's rank count")
                rank_state = payload["rank_state"][context.rank]
                stream.load_state_dict(rank_state["cursor"])
                restore_rng(rank_state["rng"])
                consumed = payload["phase_consumed"]
                counters = Counters(**payload["counters"])
            else:
                seed_everything(config.seed + data_rank, config.deterministic)
        else:
            # Initialization is identical; sampling/exploration is rank-specific.
            seed_everything(config.seed + data_rank, config.deterministic)
        if config.compile:
            # Compile the reusable tensor operators, not Python action tapes,
            # data cursors, or mutable runtime history management.
            model.core.compile_kernels(mode=config.compile_mode)
        model.train()
        model.core.set_model_version(global_step)
        if context.is_primary:
            write_json(output / "config.json", asdict(config))
            write_json(output / "runtime.json", diagnostics)
            write_json(output / "parameters.json", parameter_report(model))
        logger.log(
            "start",
            phase=phase.name,
            budget=phase.budget,
            meter=phase.meter,
            world_size=data_world,
            mode="decoupled_diloco" if diloco is not None else "synchronous",
            model=model_config.name,
            served_parameters=parameter_report(model)["totals"]["served"],
        )
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, request_stop)
        last_saved = global_step if resume and not phase_transition else -1
        updates_this_run = 0

        def checkpoint(label: str, pinned: bool = False) -> str:
            nonlocal last_saved
            rank_state = context.all_gather_object(
                {"cursor": stream.state_dict(), "rng": capture_rng()}
            )
            name = f"{phase.name}-{global_step:09d}-{label}"
            if context.is_primary:
                save_checkpoint(
                    output / "checkpoints",
                    name,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        **({"diloco": diloco.state_dict()} if diloco is not None else {}),
                        "model_config": model_config.to_dict(),
                        "train_config": asdict(config),
                        "codec_fingerprint": codec.fingerprint,
                        "chunking": chunking_identity,
                        "rank_state": rank_state,
                        "global_step": global_step,
                        "phase_consumed": consumed,
                        "counters": asdict(counters),
                    },
                    {
                        "global_step": global_step,
                        "phase": phase.name,
                        "phase_consumed": consumed,
                        "dataset_fingerprint": stream.fingerprint,
                        "codec_fingerprint": codec.fingerprint,
                        "chunking_fingerprint": chunking_identity["fingerprint"],
                        "routing_version": model_config.routing_version,
                        "config_fingerprint": content_hash(asdict(config)),
                    },
                    pinned=pinned,
                    keep=config.keep_checkpoints,
                )
            context.barrier()
            if checkpoint_store is not None:
                checkpoint_store.upload(output / "checkpoints" / name)
            last_saved = global_step
            return str(output / "checkpoints" / name)

        last_checkpoint = str(resolve_checkpoint(resume)) if resume else ""
        if phase_transition:
            last_checkpoint = checkpoint("phase-start", pinned=True)
        while (
            consumed < phase.budget
            and (diloco is None or not diloco.finished)
            and (config.max_steps is None or updates_this_run < config.max_steps)
        ):
            started = time.perf_counter()
            selected = []
            selected_counts = {"I": 0, "P": 0, "G": 0}
            for _ in range(config.max_accumulation_windows):
                for _ in range(config.microbatch_episodes):
                    item = stream.next()
                    if item is None:
                        continue
                    selected.append(item)
                    selected_counts["I"] += item.window.I
                    selected_counts["G"] += item.window.G
                    # Main prediction CE and the byte-codec auxiliary are both
                    # actually evaluated in bootstrap and recurrent modes.
                    selected_counts["P"] += item.window.P * 2
                global_selected = context.sum_counts(selected_counts)
                if (
                    global_selected["P"] >= config.accumulate_scored
                    or global_selected["I"] >= config.accumulate_ingested
                    or (phase.meter == "G" and global_selected["G"] >= config.accumulate_events)
                ):
                    break
            preparation_error = None
            try:
                batches = prepare_batches(model, selected, config)
            except Exception as error:
                preparation_error = error
                batches = []
            rejected = context.sum_counts({"rejected_batches": int(preparation_error is not None)})
            if rejected["rejected_batches"]:
                raise ValueError(
                    "Distributed target preparation failed: "
                    + str(
                        preparation_error or "another rank rejected its targets or native history"
                    )
                ) from preparation_error
            local_counts = {
                key: sum(batch.counts[key] for batch in batches) for key in DEFAULT_LOSS_WEIGHTS
            }
            global_counts = context.sum_counts(local_counts)
            if not any(global_counts.values()):
                raise RuntimeError(
                    "Accumulation window contains no supervision; audit role/turn masks"
                )
            global_work = Counters(**context.sum_counts(asdict(batch_counters(batches))))
            global_work.validate()
            weights = dict(config.loss_weights)
            ledger = LossLedger(local_counts, global_counts, weights)
            optimizer.zero_grad()
            learning_rate = phase.learning_rate(
                diloco.global_tokens if diloco is not None else consumed
            )
            optimizer.set_lr(learning_rate)
            model.core.dispatch_metrics(reset=True)
            step_diagnostics: dict[str, float | torch.Tensor] = {}
            for batch in batches:
                diagnostics_batch = forward_batch(
                    model, batch, ledger, config, context.device, global_step
                )
                for key, value in diagnostics_batch.items():
                    step_diagnostics[key] = step_diagnostics.get(key, 0.0) + value
            ledger.finish()
            context.synchronize_gradients(optimizer.parameters)
            optimizer_started = time.perf_counter()
            step_metrics = optimizer.step(config.grad_clip)
            optimizer_seconds = time.perf_counter() - optimizer_started
            global_step += 1
            updates_this_run += 1
            consumed += getattr(global_work, phase.meter)
            counters.add(global_work)
            diloco_metrics = (
                diloco.after_step(
                    optimizer, steps=global_step, tokens=getattr(global_work, phase.meter)
                )
                if diloco is not None
                else {}
            )
            model.core.set_model_version(global_step)
            local_sums = ledger.sums
            merged_sums = context.all_gather_object(local_sums)
            sums = {
                key: sum(values.get(key, 0.0) for values in merged_sums) for key in global_counts
            }
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            elapsed = time.perf_counter() - started
            if global_step % config.log_every == 0:
                gathered = context.all_gather_object(materialize_scalars(step_diagnostics))
                diagnostics_global = {
                    key: sum(item.get(key, 0.0) for item in gathered)
                    for key in set().union(*(item.keys() for item in gathered))
                }
                for branch in ("prior", "reconstruction"):
                    positions = diagnostics_global.get(f"{branch}_lexical_positions", 0)
                    if positions:
                        diagnostics_global[f"{branch}_lexical_ce"] = (
                            diagnostics_global[f"{branch}_lexical_loss_sum"] / positions
                        )
                examples = diagnostics_global.get("facet_examples", 0)
                if examples:
                    for kind in ("prediction", "encoded"):
                        diagnostics_global[f"{kind}_facet_norm"] = (
                            diagnostics_global[f"{kind}_facet_norm_sum"] / examples
                        )
                coordinates = diagnostics_global.get("facet_alignment_coordinates", 0)
                if coordinates:
                    # This distance is a diagnostic, not the recurrent encoder
                    # objective: ordinary observations have no forecast label.
                    diagnostics_global["forecast_encoder_distance_mse"] = (
                        diagnostics_global["facet_alignment_squared_error_sum"] / coordinates
                    )
                for term, diagnostic in (
                    ("facet_prediction", "facet_kl_per_coordinate"),
                    ("encoder_alignment", "native_encoder_alignment_mse"),
                ):
                    if global_counts[term]:
                        diagnostics_global[diagnostic] = sums[term] / global_counts[term]
                dispatch = model.core.dispatch_metrics(reset=True)
                reduced = context.sum_tensors(
                    {
                        f"{family}/{key}": value
                        for family, values in dispatch.items()
                        for key, value in values.items()
                    }
                )
                dispatch = {
                    family: {key: reduced[f"{family}/{key}"] for key in values}
                    for family, values in dispatch.items()
                }
                logger.log(
                    "train",
                    global_step=global_step,
                    phase=phase.name,
                    phase_consumed=consumed,
                    phase_meter=phase.meter,
                    loss={k: sums[k] / v for k, v in global_counts.items() if v},
                    denominators=global_counts,
                    lr=learning_rate,
                    decoupled=diloco.metrics() if diloco is not None else None,
                    decoupled_step=diloco_metrics,
                    counters=asdict(counters),
                    step_counters=asdict(global_work),
                    optimizer_seconds=optimizer_seconds,
                    step_seconds=elapsed,
                    peak_memory_bytes=torch.cuda.max_memory_allocated()
                    if context.device.type == "cuda"
                    else 0,
                    batch_sizes=[len(batch.windows) for batch in batches],
                    efforts=[batch.effort for batch in batches],
                    generator=diagnostics_global,
                    experts=dispatch_summary(dispatch),
                    budget_overshoot=max(0, consumed - phase.budget),
                    **rates(global_work, elapsed),
                    **step_metrics,
                )
            if config.eval_every and global_step % config.eval_every == 0:
                from megatoken.training.evaluation import evaluate_records

                rng = capture_rng()
                if context.is_primary and (
                    diloco is None or config.decoupled.resolved_worker_id == 0
                ):
                    validation = PreparedDataset(config.dataset, split="validation")
                    try:
                        logger.log(
                            "evaluation",
                            global_step=global_step,
                            **evaluate_records(model, validation, codec, config, context.device),
                        )
                    finally:
                        validation.close()
                context.barrier()
                restore_rng(rng)
                model.train()
            if config.checkpoint_every and global_step % config.checkpoint_every == 0:
                last_checkpoint = checkpoint("periodic")
            stops = context.sum_counts({"stop": int(stop_requested)})
            if stops["stop"]:
                last_checkpoint = checkpoint("interrupted", pinned=True)
                break
        if (updates_this_run > 0 and consumed >= phase.budget) or last_saved != global_step:
            last_checkpoint = checkpoint(
                "phase-end" if consumed >= phase.budget else "run-end", pinned=True
            )
        result = {
            "global_step": global_step,
            "phase": phase.name,
            "phase_consumed": consumed,
            "complete": consumed >= phase.budget or (
                diloco is not None and diloco.finished
            ),
            "checkpoint": last_checkpoint,
            "counters": asdict(counters),
        }
        logger.log("finished", **result)
        return result
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if diloco is not None:
            diloco.close()
        if optimizer is not None:
            optimizer.close()
        if stream is not None:
            stream.close()
        if dataset is not None:
            dataset.close()
        context.close()
