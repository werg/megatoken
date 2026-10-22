"""A single optimizer engine for bootstrap through student-state distillation.

Windows are independent optimizer transactions: reconstruct the actual prefix
under current weights, retain gradients through selected writer/reader records,
and discard all runtime state after updating parameters. No stale cache survives
an update.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import build_control_payload
from megatoken.data import PreparedCorpus, WindowConfig, stage_recipe
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.media_assets import MediaRecord
from megatoken.io import content_hash, sha256_file, write_json
from megatoken.loading import (
    build_model,
    load_codec,
    load_model_config,
    parameter_report,
    restore_weights,
)
from megatoken.runtime import capture_rng, doctor, restore_rng, seed_everything
from megatoken.teacher import PINNED_TEACHER
from megatoken.tensors import byte_tensor
from megatoken.training.autoencode import (
    AutoencodePlan,
    FeatureSampler,
    autoencode_losses,
    embedding_center,
)
from megatoken.training.checkpoint import (
    CheckpointWriter,
    commit_checkpoint,
    load_checkpoint,
    resolve_checkpoint,
    snapshot_to_cpu,
    stage_checkpoint,
    transducer_attachment,
)
from megatoken.training.config import DEFAULT_LOSS_WEIGHTS, TrainConfig
from megatoken.training.diloco_client import DecoupledWorker
from megatoken.training.diloco_protocol import training_identity
from megatoken.training.distributed import DistributedContext
from megatoken.training.heldout import HeldOutEvaluator
from megatoken.training.history import NativeHistory
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
from megatoken.training.retrieval import close_mounted
from megatoken.training.router_balance import RouterBalanceWindow, balance_routers
from megatoken.training.stream import (
    EpisodeWindow,
    IndependentWindowStream,
    MixtureWindowStream,
    WindowStream,
)
from megatoken.training.transducer import TransducerTrainer
from megatoken.training.transducer_sync import (
    RankRounds,
    TransducerSync,
    install_published,
    resume_transducer,
)


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


def prepare_batches(
    model: Any, windows: list[EpisodeWindow], config: TrainConfig
) -> list[MeridianPreparedBatch]:
    """Only native grouped-meridian episodes are valid training examples.

    Earlier ungrouped/action fallback batching is removed, rather than silently
    applying a different loss/history schedule.
    """
    result: list[MeridianPreparedBatch] = []
    verified_streams = {
        "verified_trajectories",
        "verified_successful_multiturn",
        "tested_artifacts",
        "successful_replay",
        "long_verified_dependencies",
    }
    for item in windows:
        episode = item.episode
        if not episode.meridians:
            raise ValueError(
                "Training requires causal native meridians; reprepare this old ungrouped episode"
            )
        if any(
            len(record.payload) > config.max_record_bytes
            for record in (*item.window.prefix, *item.window.records)
        ):
            raise ValueError("A native record exceeds the configured original-byte limit")
        if (
            not config.allow_unverified_data
            and episode.stream in verified_streams
            and not episode.supervision.verified_success
        ):
            raise ValueError("Training stream requires independent verified success")
        if episode.stream == "long_verified_dependencies" and not episode.dependencies:
            raise ValueError("Long dependency stream requires causal source relationships")
        result.append(prepare_meridian_batch(model, item, NativeHistory.of(episode, model), config))
    return result


def window_config(config: TrainConfig) -> WindowConfig:
    """Optimizer window boundaries shared by training and held-out evaluation."""
    return WindowConfig(
        max_window_tokens=config.max_history_tokens,
        max_decisions=config.scored_decisions,
        max_records=config.max_window_records,
        respect_dependencies=config.phase == "LC",
        max_dependency_tokens=config.max_history_tokens,
    )


def batch_counters(batches: list[MeridianPreparedBatch]) -> Counters:
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
        for step in batch.steps:
            for modality, record in zip(step.modalities, step.records, strict=True):
                if modality not in {"audio", "image", "video"}:
                    continue
                result.media_events += 1
                geometry = MediaRecord.from_bytes(record.payload).output_geometry
                if geometry is None:
                    continue  # observed media: no physical output is realized
                if modality == "audio":
                    result.audio_samples += geometry.sample_count
                elif modality == "video":
                    result.video_frames += geometry.frames
                else:
                    result.image_events += 1
        result.P += sum(plan.scored_positions for step in batch.steps for plan in step.plans)
        result.facet_encodes += sum(len(step.plans) for step in batch.steps)
        if batch.native_history.has_events:
            for step in (*batch.prefix, *batch.steps):
                for cursor in step.offsets:
                    result.raw_facet_replays += len(
                        batch.native_history.events_for_record(
                            cursor,
                            include_record=False,
                        )
                    ) + int(batch.native_history.paired_tape(cursor) is not None)
        if any(episode_control_payload(item.episode) for item in batch.windows):
            result.control_encodes += sum(len(step.plans) for step in batch.steps)
    return result


# The decoupled settings that change what a resumed learner computes. Where the
# coordinator listens, timeouts, polling and storage do not.
_DECOUPLED_IDENTITY = ("run_id", "workers", "local_steps", "wire_dtype")


def _resume_compatible(old: dict[str, Any], config: TrainConfig, phase_transition: bool) -> None:
    current = asdict(config)
    for state in (old, current):
        decoupled = state.get("decoupled")
        state["decoupled"] = (
            None if decoupled is None else {key: decoupled[key] for key in _DECOUPLED_IDENTITY}
        )
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
            "datasets",
            "seed",
            "microbatch_episodes",
            "max_history_tokens",
            "max_window_records",
            "scored_decisions",
            "generated_capacity",
            "accumulate_scored",
            "accumulate_ingested",
            "accumulate_events",
            "max_accumulation_windows",
            "max_media_events_per_window",
            "max_record_bytes",
            "use_stage_mixture",
            "use_effort_mixture",
            "effort",
            "loss_weights",
            "grad_clip",
            "budget",
            "deterministic",
            "gradient_checkpointing",
            "representation_mode",
            "router_balance_mode",
            "trainable",
            "text_anchors",
            "text_anchor_weight",
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
    if phase_transition and not resume:
        raise ValueError("A phase transition requires the previous phase's checkpoint")
    if config.decoupled is not None and (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or (torch.distributed.is_available() and torch.distributed.is_initialized())
    ):
        raise ValueError(
            "Decoupled learners require one independent process per sandbox, not torchrun"
        )
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
    checkpoint_writer = None
    optimizer = None
    transducer = None
    dataset = stream = heldout = None
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
        packs_fingerprint = sha256_file(config.packs) if config.packs else None
        model = build_model(
            model_config,
            codec,
            device=context.device,
            precision=config.precision,
            packs=config.packs,
        )
        new_owners: tuple[str, ...] = ()
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
            new_owners = restore_weights(
                model, payload, packs_fingerprint, phase_transition=phase_transition
            )
        elif config.engram_import:
            from megatoken.assets import install_engram_asset

            install_engram_asset(model, config.engram_import)
        # Synchronous ranks are learners of one canonical bank merged on rank 0.
        transducer = TransducerTrainer(
            model.transducer,
            worker=config.decoupled.resolved_worker_id if config.decoupled else context.rank,
            provisional=config.decoupled is not None or context.world_size > 1,
        )
        rank_rounds = None
        if context.world_size > 1:
            rank_rounds = RankRounds(transducer, context)
        if resume:
            resume_transducer(transducer, rank_rounds, resolve_checkpoint(resume))
        context.broadcast_model(model)
        phase = config.schedule
        optimizer = MasterAdamW(
            model,
            phase.learning_rate(0),
            config.weight_decay,
            inner_optimizer=config.inner_optimizer,
        )
        if config.trainable == "media_bridge":
            # Every DiLoCo worker freezes the same owners, so their deltas are
            # exactly zero everywhere and the outer merge stays unbiased.
            bridge = {id(parameter) for parameter in model.media_bridge_parameters()}
            optimizer.freeze(
                [name for name, p in optimizer.model_parameters.items() if id(p) not in bridge]
            )
        elif config.representation_training == "recurrent":
            # Contextual codec auxiliaries use frozen C/R with E gradients only.
            # Independent clipping prevents their norm scaling the main update.
            optimizer.isolate_gradient_parameters("encoder", model.core.encoder_parameters())
        windows_config = window_config(config)
        dataset = PreparedCorpus(config.datasets)
        if dataset.codec_fingerprint != codec.fingerprint:
            raise ValueError("Prepared datasets use a different codec")
        if dataset.extractor not in (None, model_config.media_extractor):
            raise ValueError(
                f"Prepared features come from teacher {dataset.extractor!r}, "
                f"not this model's {model_config.media_extractor!r}"
            )
        text_anchors = None
        if config.text_anchors:
            from megatoken.training.text_anchors import TextAnchors

            text_anchors = TextAnchors(
                config.text_anchors, dataset.stores, extractor=model_config.media_extractor
            )
        sampler = center = None
        plan = AutoencodePlan.from_stage(stage_recipe(config.phase).constraints)
        if config.use_stage_mixture and plan.items:
            sampler = FeatureSampler(
                dataset,
                seed=config.seed,
                worker=config.decoupled.resolved_worker_id if config.decoupled else context.rank,
            )
            if not sampler:
                raise ValueError(
                    f"Stage {config.phase} trains encode-and-reconstruct on prepared "
                    "feature items, but the datasets have none"
                )
            center = embedding_center(sampler).to(context.device)
        chunking = ChunkingConfig.from_identity(dataset.chunking, codec)
        chunking_identity = chunking.identity(codec)
        if (
            payload is not None
            and not phase_transition
            and payload["chunking"] != chunking_identity
        ):
            raise ValueError("Exact resume changed the semantic chunking policy")
        data_rank = config.decoupled.resolved_worker_id if config.decoupled else context.rank
        data_world = config.decoupled.workers if config.decoupled else context.world_size
        if config.use_stage_mixture:
            mixture = stage_recipe(config.phase).mixture
            stream = MixtureWindowStream(
                config.datasets,
                codec,
                windows_config,
                mixture,
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
                    dataset,
                    codec,
                    windows_config,
                    seed=config.seed,
                    rank=data_rank,
                    world_size=data_world,
                )
                if config.decoupled is not None
                else WindowStream(
                    dataset,
                    codec,
                    windows_config,
                    seed=config.seed,
                    rank=context.rank,
                    world_size=context.world_size,
                    reduce_counts=context.sum_counts,
                )
            )
        if config.eval_every:
            heldout = HeldOutEvaluator(config, codec, windows_config, require=True)
        if config.decoupled is not None:
            diloco = DecoupledWorker(
                config.decoupled,
                model_config=model_config.to_dict(),
                codec_fingerprint=codec.fingerprint,
                masters=optimizer.masters,
                output=output / "diloco",
                token_budget=phase.budget,
                train_identity=training_identity(config),
                dataset_fingerprint=dataset.identity,
            )
            diloco.attach_transducer(TransducerSync())
            if payload is None:
                diloco.bootstrap(model, optimizer)
                # Start from the coordinator's canonical bank, not an empty one.
                fetched = diloco.fetch_transducer(-1)
                if fetched is not None:
                    install_published(transducer, diloco.transducer, fetched[2], fetched[1])
        global_step = 0
        consumed = 0
        counters = Counters()
        if payload is not None:
            optimizer.load_state_dict(payload["optimizer"], new=new_owners)
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
                if (sampler is None) != (rank_state["autoencode"] is None):
                    raise ValueError("Exact resume changed the encode-and-reconstruct sampler")
                if sampler is not None:
                    sampler.load_state_dict(rank_state["autoencode"])
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

        checkpoint_writer = CheckpointWriter()

        memory_attachment = transducer_attachment(
            (transducer if rank_rounds is None else rank_rounds).state_dict
        )

        def checkpoint(label: str, pinned: bool = False) -> str:
            """Snapshot at the step boundary; periodic writes overlap training.

            Pinned (phase/run boundary, interrupt) checkpoints are durable on return.
            """
            nonlocal last_saved
            rank_state = context.all_gather_object(
                {
                    "cursor": stream.state_dict(),
                    "rng": capture_rng(),
                    "autoencode": None if sampler is None else sampler.state_dict(),
                }
            )
            name = f"{phase.name}-{global_step:09d}-{label}"
            if rank_rounds is not None:
                rank_rounds.flush(global_step)
            if context.is_primary:
                # Bound host memory to one in-flight snapshot.
                checkpoint_writer.wait()
                root = output / "checkpoints"
                staged = stage_checkpoint(root, name)
                try:
                    extra = memory_attachment(staged)
                    payload = snapshot_to_cpu(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            **({"diloco": diloco.state_dict()} if diloco is not None else {}),
                            "model_config": model_config.to_dict(),
                            "train_config": asdict(config),
                            "codec_fingerprint": codec.fingerprint,
                            "packs_fingerprint": packs_fingerprint,
                            "chunking": chunking_identity,
                            "rank_state": rank_state,
                            "global_step": global_step,
                            "phase_consumed": consumed,
                            "counters": asdict(counters),
                        }
                    )
                    metadata = {
                        "global_step": global_step,
                        "phase": phase.name,
                        "phase_consumed": consumed,
                        "dataset_fingerprint": stream.fingerprint,
                        "codec_fingerprint": codec.fingerprint,
                        "packs_fingerprint": packs_fingerprint,
                        "chunking_fingerprint": chunking_identity["fingerprint"],
                        "routing_version": model_config.routing_version,
                        "config_fingerprint": content_hash(asdict(config)),
                    }
                except BaseException:
                    shutil.rmtree(staged, ignore_errors=True)
                    raise

                def write() -> None:
                    commit_checkpoint(
                        root,
                        name,
                        staged,
                        payload,
                        metadata,
                        extra=extra,
                        pinned=pinned,
                        keep=config.keep_checkpoints,
                    )
                    if checkpoint_store is not None:
                        checkpoint_store.upload(root / name, keep=config.keep_checkpoints)

                checkpoint_writer.submit(write)
                if pinned:
                    checkpoint_writer.wait()
            context.barrier()
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
            anchor_local = (
                0
                if text_anchors is None
                else sum(
                    text_anchors.key(entry.payload) in text_anchors.assets
                    for batch in batches
                    for step in batch.steps
                    for entry, modality in zip(step.records, step.modalities, strict=True)
                    if modality == "text"
                )
            )
            anchor_denom = context.sum_counts({"text_anchors": anchor_local})["text_anchors"]
            if not any(global_counts.values()):
                raise RuntimeError(
                    "Accumulation window contains no supervision; audit role/turn masks"
                )
            global_work = Counters(**context.sum_counts(asdict(batch_counters(batches))))
            global_work.validate()
            weights = dict(config.loss_weights)
            # The old per-operation router loss has been replaced.
            weights["router_balance"] = 0.0
            ledger = LossLedger(local_counts, global_counts, weights)
            optimizer.zero_grad()
            routers = balance_routers(model.core)
            router_replay = RouterBalanceWindow()
            learning_rate = phase.learning_rate(
                diloco.global_tokens if diloco is not None else consumed
            )
            optimizer.set_lr(learning_rate)
            model.core.dispatch_metrics(reset=True)
            step_diagnostics: dict[str, float | torch.Tensor] = {}
            for router in routers.values():
                router.balance_recorder = router_replay
            try:
                for batch in batches:
                    diagnostics_batch = forward_meridian_batch(
                        model,
                        batch,
                        ledger,
                        config,
                        context.device,
                        global_step,
                        text_anchors=(
                            None
                            if text_anchors is None
                            else (text_anchors, anchor_denom, config.text_anchor_weight)
                        ),
                        transducer=transducer,
                    )
                    for key, value in diagnostics_batch.items():
                        step_diagnostics[key] = step_diagnostics.get(key, 0.0) + value
                ledger.finish()
                # All ranks participate, including workers with zero eligible
                # assignments. The model's version remains fixed until step().
                step_diagnostics.update(
                    router_replay.backward(
                        context,
                        routers=routers,
                        coefficient=config.loss_weights["router_balance"],
                    )
                )
            finally:
                for router in routers.values():
                    router.balance_recorder = None
            step_diagnostics.update(transducer.after_backward(global_step))
            if sampler is not None:
                terms = autoencode_losses(
                    model,
                    sampler.draw(plan.items),
                    space_fingerprint=PINNED_TEACHER,
                    center=center,
                )
                totals = context.sum_counts({name: count for name, (_, count) in terms.items()})
                weights = {
                    "reconstruction": plan.reconstruction_weight,
                    "embedding": plan.embedding_weight,
                }
                loss = sum(
                    weights[name] * value / totals[name]
                    for name, (value, _) in terms.items()
                    if totals[name]
                )
                loss.backward()
                for name, (value, count) in terms.items():
                    if count:
                        step_diagnostics[f"autoencode_{name}"] = float(value.detach()) / count
            context.synchronize_gradients(optimizer.parameters)
            optimizer_started = time.perf_counter()
            step_metrics = optimizer.step(config.grad_clip)
            optimizer_seconds = time.perf_counter() - optimizer_started
            global_step += 1
            step_metrics.update(
                transducer.barrier(global_step, evolve=diloco is None and rank_rounds is None)
            )
            if rank_rounds is not None:
                step_metrics.update(rank_rounds.step(global_step))
            updates_this_run += 1
            consumed += getattr(global_work, phase.meter)
            counters.add(global_work)
            diloco_metrics = (
                diloco.after_step(
                    optimizer,
                    steps=global_step,
                    tokens=getattr(global_work, phase.meter),
                    transducer=transducer,
                )
                if diloco is not None
                else {}
            )
            ready = diloco_metrics.pop("transducer_ready", None)
            if ready is not None:
                install_published(transducer, diloco.transducer, ready[2], ready[1])
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
            if (
                config.eval_every
                and global_step % config.eval_every == 0
                and (diloco is None or config.decoupled.resolved_worker_id == 0)
            ):
                evaluation_started = time.perf_counter()
                local, failure = {}, None
                try:
                    local = heldout.sums(
                        model, transducer, global_step, rank=context.rank, world=context.world_size
                    )
                except Exception as error:
                    failure = error
                # A failing rank must not leave its peers waiting in the gather.
                gathered = context.all_gather_object(
                    (local, None if failure is None else str(failure))
                )
                if any(message for _, message in gathered):
                    raise RuntimeError(
                        "Held-out evaluation failed: " + next(m for _, m in gathered if m)
                    ) from failure
                logger.log(
                    "evaluation",
                    global_step=global_step,
                    phase=phase.name,
                    phase_consumed=consumed,
                    evaluation_seconds=time.perf_counter() - evaluation_started,
                    **heldout.report([sums for sums, _ in gathered]),
                )
            if transducer.switch_due(global_step):
                heldout = heldout or HeldOutEvaluator(config, codec, windows_config, require=False)
                loss = heldout.bank_only_loss(model, transducer, global_step)
                if rank_rounds is not None:
                    rank_rounds.record_validation(global_step, loss)
                elif diloco is None:
                    if transducer.record_validation(global_step, loss):
                        logger.log("transducer_switch", global_step=global_step)
                elif config.decoupled.resolved_worker_id == 0:
                    diloco.report_transducer_validation(global_step, loss)
                logger.log("transducer_validation", global_step=global_step, bank_only_loss=loss)
                model.train()
            if config.checkpoint_every and global_step % config.checkpoint_every == 0:
                last_checkpoint = checkpoint("periodic")
            stops = context.sum_counts({"stop": int(stop_requested)})
            if stops["stop"]:
                last_checkpoint = checkpoint("interrupted", pinned=True)
                break
        if diloco is not None and not stop_requested and updates_this_run > 0:
            # Hand the coordinator the last, partial window before the final checkpoint.
            diloco.finish(optimizer, transducer)
        if (updates_this_run > 0 and consumed >= phase.budget) or last_saved != global_step:
            last_checkpoint = checkpoint(
                "phase-end" if consumed >= phase.budget else "run-end", pinned=True
            )
        result = {
            "global_step": global_step,
            "phase": phase.name,
            "phase_consumed": consumed,
            "complete": consumed >= phase.budget or (diloco is not None and diloco.finished),
            "checkpoint": last_checkpoint,
            "counters": asdict(counters),
        }
        logger.log("finished", **result)
        return result
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if checkpoint_writer is not None:
            # A periodic write still in flight must land before the run returns.
            checkpoint_writer.close()
        if diloco is not None:
            diloco.close()
        if optimizer is not None:
            optimizer.close()
        if stream is not None:
            stream.close()
        if heldout is not None:
            heldout.close()
        if dataset is not None:
            dataset.close()
        close_mounted()
        context.close()
