"""Fresh-policy outcome RL over canonical facet actions.

This is a score-function update with independent same-task leave-one-out
baselines. It intentionally has no PPO clipping, stale rollout reuse, teacher
token KL, or invented categorical factors for deterministic decoded strings.
"""

from __future__ import annotations

import base64
import json
import signal
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import build_control_payload
from megatoken.data.chunking import ChunkingConfig, segment_bytes
from megatoken.environments import (
    FIXTURE_TOOLS,
    Environment,
    EnvironmentResult,
    environment_factory,
)
from megatoken.generation.structure import ActionSpec, ActionTape
from megatoken.io import content_hash, write_json
from megatoken.loading import build_model, load_codec
from megatoken.model.config import ModelConfig
from megatoken.runtime import capture_rng, restore_rng, seed_everything
from megatoken.tensors import byte_tensor
from megatoken.tool_schema import compile_tool_spec, recent_copy_sources
from megatoken.training.checkpoint import load_checkpoint, resolve_checkpoint, save_checkpoint
from megatoken.training.distributed import DistributedContext
from megatoken.training.evaluation import wilson_interval
from megatoken.training.losses import leave_one_out_advantages
from megatoken.training.metrics import JsonlLogger
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.schedule import PHASES


@dataclass(frozen=True)
class RLConfig:
    checkpoint: str
    output: str = "runs/rl"
    environment: str = "fixture"
    codec: str = "byte"
    device: str = "auto"
    task_seeds: int = 10_000
    rollouts_per_seed: int = 4
    max_steps_per_episode: int = 8
    max_updates: int | None = None
    max_task_groups: int | None = None
    max_latent_coordinates: int = 32_000_000_000
    max_total_episodes: int = 200_000
    capacity: int = 512
    seed: int = 17
    replay_atol: float = 1e-4
    replay_rtol: float = 1e-5
    checkpoint_every: int = 100
    use_tool_schema: bool = True
    require_verified: bool = True
    require_blackwell: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.task_seeds <= 50_000 or self.rollouts_per_seed < 2:
            raise ValueError(
                "RL requires 1–50,000 distinct seeds and at least two independent rollouts"
            )
        if self.task_seeds * self.rollouts_per_seed > self.max_total_episodes:
            raise ValueError("Requested rollouts exceed the total episode ceiling")
        if not 0 < self.max_latent_coordinates <= 32_000_000_000:
            raise ValueError("RL's latent-coordinate ceiling must be in (0, 32 billion]")
        if min(self.max_steps_per_episode, self.capacity, self.checkpoint_every) < 1:
            raise ValueError("Episode, capacity, and checkpoint limits must be positive")
        if self.max_updates is not None and self.max_updates < 1:
            raise ValueError("max_updates must be positive")
        if self.max_task_groups is not None and self.max_task_groups < 1:
            raise ValueError("max_task_groups must be positive")
        if min(self.replay_atol, self.replay_rtol) < 0:
            raise ValueError("Invalid replay tolerances")


@dataclass(frozen=True)
class RolloutStep:
    tape: ActionTape
    spec: ActionSpec
    observation: bytes
    action_id: str
    expected_state_id: str
    execution_id: str | None


@dataclass(frozen=True)
class Rollout:
    task_id: str
    rollout_id: str
    initial_observation: bytes
    chunking_identity: dict[str, Any]
    steps: tuple[RolloutStep, ...]
    reward: float
    verified_success: bool
    behavior_log_prob: float
    scored_positions: int
    latent_dimensions: int
    emitted_positions: int
    ingested_tokens: int
    source_bytes: int
    environment_seconds: float


def _ingest(model: Any, state: Any, payload: bytes, role: str, chunking: ChunkingConfig):
    device = state.device
    for chunk in segment_bytes(payload, model.codec, chunking):
        ids, lengths = byte_tensor([chunk.payload], device)
        state = model.core.observe(
            ids,
            state,
            lengths=lengths,
            role_ids=torch.tensor([ROLE_TO_ID[role]], device=device),
            encoder_grad=False,
        )
    return state


def _public_tools(task: dict[str, Any], observation: bytes, fallback: bool) -> str | None:
    if "tools" in task:
        return json.dumps(task["tools"])
    try:
        value = json.loads(observation)
        if isinstance(value, dict) and isinstance(value.get("tools"), list):
            return json.dumps(value["tools"])
    except (ValueError, UnicodeError):
        pass
    return json.dumps(FIXTURE_TOOLS) if fallback else None


@torch.no_grad()
def collect_rollout(
    model: Any,
    codec: Any,
    task: dict[str, Any],
    make_environment: Callable[[], Environment],
    config: RLConfig,
    *,
    chunking: ChunkingConfig | None = None,
) -> Rollout:
    chunking = (chunking or ChunkingConfig()).with_hard_limit(model.config.max_capacity)
    environment = make_environment()
    started = time.perf_counter()
    observed = environment.reset(task)
    environment_seconds = time.perf_counter() - started
    state = model.core.initial_state(1, device=next(model.parameters()).device)
    state = _ingest(model, state, observed.observation, "user", chunking)
    tools = _public_tools(task, observed.observation, config.environment == "fixture")
    histories = [observed.observation]
    initial = observed.observation
    steps = []
    log_prob = 0.0
    positions = 0
    latent_dimensions = emitted_positions = 0
    ingested = len(codec.encode(initial))
    byte_count = len(initial)
    invalid = 0
    rollout_id = uuid.uuid4().hex
    for index in range(config.max_steps_per_episode):
        if config.use_tool_schema and tools:
            spec = compile_tool_spec(tools, config.capacity)
            spec = replace(spec, copy_sources=recent_copy_sources(histories))
        else:
            spec = ActionSpec(max_tokens=config.capacity)
        public_control_seed = task.get("control_seed", "")
        if not isinstance(public_control_seed, str):
            raise ValueError("RL task control_seed must be public UTF-8 text")
        control_payload = build_control_payload(
            seed=public_control_seed.encode(), tools_json=tools or "[]"
        )
        sampled = model.sample(state, spec=spec, effort=6, rl=True, control_payload=control_payload)
        state = sampled.state
        action_id = f"{rollout_id}:{index}"
        expected_state = observed.state_id
        # Every selected facet action has already advanced recurrence. Wire
        # validity only determines whether its bytes can enter the executor.
        ingested += len(codec.encode(sampled.payload))
        byte_count += len(sampled.payload)
        histories.append(sampled.payload)
        if sampled.tape.valid:
            started = time.perf_counter()
            result = environment.step(
                sampled.payload, action_id=action_id, expected_state_id=expected_state
            )
            environment_seconds += time.perf_counter() - started
        else:
            invalid += 1
            # Validation failures have a deterministic, public feedback event.
            # The invalid bytes are never sent to the environment executor.
            result = EnvironmentResult(
                b'{"error":"invalid_deterministic_action"}', observed.state_id
            )
        state = _ingest(model, state, result.observation, "tool", chunking)
        ingested += len(codec.encode(result.observation))
        byte_count += len(result.observation)
        histories.append(result.observation)
        steps.append(
            RolloutStep(
                sampled.tape,
                spec,
                result.observation,
                action_id,
                expected_state,
                result.execution_id,
            )
        )
        log_prob += sampled.tape.behavior_log_prob
        positions += sampled.scored_positions
        latent_dimensions += sampled.tape.latent_dimensions
        emitted_positions += len(sampled.emitted_ids)
        observed = result
        tools = _public_tools({}, result.observation, False) or tools
        if result.done:
            break
    started = time.perf_counter()
    verified = environment.verify()
    environment_seconds += time.perf_counter() - started
    if config.require_verified and not verified.verified:
        raise ValueError("Outcome RL requires an independently verified task result")
    success = verified.verified and verified.reward >= 1.0
    # Secondary terms are bounded below the gap between failure and success.
    reward = float(success) + min(0.05, max(0.0, verified.reward) * 0.05 if not success else 0.0)
    reward -= min(0.05, 0.005 * max(0, len(steps) - 1)) + min(0.05, invalid * 0.01)
    return Rollout(
        str(task["task_id"]),
        rollout_id,
        initial,
        chunking.identity(codec),
        tuple(steps),
        reward,
        success,
        log_prob,
        positions,
        latent_dimensions,
        emitted_positions,
        ingested,
        byte_count,
        environment_seconds,
    )


def score_rollout(model: Any, rollout: Rollout) -> torch.Tensor:
    chunking = ChunkingConfig.from_identity(rollout.chunking_identity, model.codec)
    state = model.core.initial_state(1, device=next(model.parameters()).device)
    state = _ingest(model, state, rollout.initial_observation, "user", chunking)
    log_prob = state.facets.float().sum() * 0
    for step in rollout.steps:
        scored = model.score_tape(state, step.tape, spec=step.spec)
        log_prob = log_prob + scored.log_prob
        state = _ingest(model, scored.state, step.observation, "tool", chunking)
    return log_prob


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode()}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def write_rollout(handle: Any, rollout: Rollout, model_version: int) -> None:
    value = asdict(rollout)
    value.update(
        format_version=2,
        model_version=model_version,
        likelihood="canonical_facet_gaussian_density",
        replay="full_differentiable",
    )
    handle.write(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def train_rl(config: RLConfig, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tasks) < config.task_seeds:
        raise ValueError("Not enough distinct task seeds for requested RL run")
    identities = [str(task.get("task_id", "")) for task in tasks[: config.task_seeds]]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("RL task seeds need unique, nonempty task_id values")
    context = DistributedContext.initialize(config.device)
    optimizer = None
    old_handlers = {}
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    try:
        from megatoken.runtime import doctor

        diagnostics = doctor(require_blackwell=config.require_blackwell)
        if not diagnostics["ok"]:
            raise RuntimeError("RL runtime preflight failed: " + "; ".join(diagnostics["errors"]))
        payload, _ = load_checkpoint(config.checkpoint)
        model_config = ModelConfig.from_dict(payload["model_config"])
        if config.capacity > model_config.max_capacity:
            raise ValueError("RL capacity exceeds the model's maximum output positions")
        codec = load_codec(config.codec, vocab_size=model_config.vocab_size)
        if codec.fingerprint != payload["codec_fingerprint"]:
            raise ValueError("RL checkpoint and codec identities differ")
        chunking = ChunkingConfig.from_identity(payload["chunking"], codec)
        chunking = chunking.with_hard_limit(model_config.max_capacity)
        precision = payload["train_config"]["precision"]
        model = build_model(model_config, codec, device=context.device, precision=precision)
        model.load_state_dict(payload["model"], strict=True)
        model.core.set_model_version(payload["global_step"])
        model.eval()  # Rollout and replay use identical deterministic module behavior.
        if context.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        optimizer = MasterAdamW(model, PHASES["R"].learning_rate(0))
        optimizer.load_state_dict(payload["optimizer"])
        # Preserve all master/moment ownership from supervised training. Frozen
        # decoder owners retain grad=None, so Adam applies neither momentum nor
        # decay to the deterministic facet-to-action map during latent-only RL.
        model.freeze_decoder_for_rl()
        frozen_decoder = sorted(name for name, p in model.named_parameters() if not p.requires_grad)
        optimizer.freeze(frozen_decoder)
        seed_everything(config.seed + context.rank)
        factory = environment_factory(config.environment)
        output = Path(config.output)
        output.mkdir(parents=True, exist_ok=True)
        logger = JsonlLogger(output / "metrics.jsonl", context.is_primary)
        totals = {
            "P": 0,
            "Z": 0,
            "emitted_positions": 0,
            "D": 0,
            "I": 0,
            "collection_I": 0,
            "replay_I": 0,
            "source_bytes": 0,
            "episodes": 0,
            "successes": 0,
            "task_seeds": 0,
            "optimizer_updates": 0,
        }
        step = payload["global_step"]
        next_group = 0
        task_fingerprint = content_hash(tasks[: config.task_seeds])
        if "rl_config" in payload:
            if payload.get("frozen_decoder_parameters") != frozen_decoder:
                raise ValueError("RL deterministic decoder ownership changed during resume")
            mutable = {
                "checkpoint",
                "output",
                "max_updates",
                "max_task_groups",
                "checkpoint_every",
                "device",
            }
            for key, value in asdict(config).items():
                if key not in mutable and payload["rl_config"].get(key) != value:
                    raise ValueError(f"RL exact resume changed {key}")
            if task_fingerprint != payload["task_fingerprint"]:
                raise ValueError("RL task content or order changed during resume")
            if len(payload["rank_state"]) != context.world_size:
                raise ValueError("RL exact resume requires the same distributed rank count")
            restore_rng(payload["rank_state"][context.rank]["rng"])
            totals = dict(payload["rl_counters"])
            next_group = payload["next_task_group"]
        last_checkpoint = ""
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, request_stop)

        def checkpoint(label: str) -> str:
            ranks = context.all_gather_object({"rng": capture_rng()})
            name = f"R-{step:09d}-tasks{totals['task_seeds']:06d}-{label}"
            if context.is_primary:
                save_checkpoint(
                    output / "checkpoints",
                    name,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_config.to_dict(),
                        "train_config": {**payload["train_config"], "phase": "R"},
                        "codec_fingerprint": codec.fingerprint,
                        "chunking": chunking.identity(codec),
                        "global_step": step,
                        "rl_config": asdict(config),
                        "frozen_decoder_parameters": frozen_decoder,
                        "policy_likelihood": "canonical_facet_gaussian_density",
                        "rl_counters": totals,
                        "rank_state": ranks,
                        "task_fingerprint": task_fingerprint,
                        "next_task_group": next_group,
                    },
                    {
                        "global_step": step,
                        "phase": "R",
                        "routing_version": model_config.routing_version,
                        "codec_fingerprint": codec.fingerprint,
                        "chunking_fingerprint": chunking.identity(codec)["fingerprint"],
                    },
                    pinned=not label.startswith("periodic"),
                )
            context.barrier()
            return str(output / "checkpoints" / name)

        if "rl_config" not in payload:
            last_checkpoint = checkpoint("pre-rl")
        groups_this_run = updates_this_run = 0
        with (output / f"rollouts-rank{context.rank}.jsonl").open("a", encoding="utf-8") as journal:
            for group_start in range(next_group, config.task_seeds, context.world_size):
                if (
                    (config.max_updates is not None and updates_this_run >= config.max_updates)
                    or (
                        config.max_task_groups is not None
                        and groups_this_run >= config.max_task_groups
                    )
                    or totals["Z"] >= config.max_latent_coordinates
                    or totals["episodes"] >= config.max_total_episodes
                ):
                    break
                task_index = group_start + context.rank
                rollouts = []
                if task_index < config.task_seeds:
                    task = tasks[task_index]
                    for _ in range(config.rollouts_per_seed):
                        rollout = collect_rollout(
                            model, codec, task, factory, config, chunking=chunking
                        )
                        write_rollout(journal, rollout, step)
                        rollouts.append(rollout)
                ingestion = sum(r.ingested_tokens for r in rollouts)
                local = {
                    "P": sum(r.scored_positions for r in rollouts),
                    "Z": sum(r.latent_dimensions for r in rollouts),
                    "emitted_positions": sum(r.emitted_positions for r in rollouts),
                    "D": ingestion,
                    "I": 2 * ingestion,
                    "collection_I": ingestion,
                    "replay_I": ingestion,
                    "source_bytes": 2 * sum(r.source_bytes for r in rollouts),
                    "episodes": len(rollouts),
                    "successes": sum(r.verified_success for r in rollouts),
                    "task_seeds": int(bool(rollouts)),
                }
                global_counts = context.sum_counts(local)
                for key, value in global_counts.items():
                    totals[key] += value
                optimizer.zero_grad()
                mismatch = 0.0
                varied = False
                replay_failed = False
                if rollouts:
                    rewards = torch.tensor([[r.reward for r in rollouts]], device=context.device)
                    advantages = leave_one_out_advantages(rewards)[0]
                    varied = bool(advantages.abs().max() > 1e-8)
                    for rollout, advantage in zip(rollouts, advantages, strict=True):
                        candidate = score_rollout(model, rollout)
                        expected = candidate.new_tensor(rollout.behavior_log_prob)
                        difference = float((candidate.detach() - expected).abs())
                        mismatch = max(mismatch, difference)
                        if not torch.isclose(
                            candidate.detach(),
                            expected,
                            atol=config.replay_atol,
                            rtol=config.replay_rtol,
                        ):
                            replay_failed = True
                        if varied:
                            (-candidate * advantage / global_counts["episodes"]).backward()
                gate = context.sum_counts(
                    {"varied": int(varied), "replay_failed": int(replay_failed)}
                )
                if gate["replay_failed"]:
                    raise RuntimeError(
                        f"Same-weight action-tape replay failed before update; local max delta={mismatch}"
                    )
                varied_global = gate["varied"]
                if varied_global:
                    context.synchronize_gradients(optimizer.parameters)
                    optimizer.set_lr(PHASES["R"].learning_rate(totals["Z"]))
                    optimizer.step()
                    step += 1
                    totals["optimizer_updates"] += 1
                    updates_this_run += 1
                    model.core.set_model_version(step)
                next_group = group_start + context.world_size
                groups_this_run += 1
                logger.log(
                    "rl",
                    global_step=step,
                    counters=totals,
                    replay_max_absolute_error=mismatch,
                    replay="full_differentiable",
                    fresh_policy=True,
                    skipped_equal_reward_update=not bool(varied_global),
                    success_wilson95=wilson_interval(totals["successes"], totals["episodes"]),
                )
                if (
                    totals["Z"] >= config.max_latent_coordinates
                    or totals["episodes"] >= config.max_total_episodes
                ):
                    break
                if totals["task_seeds"] % config.checkpoint_every == 0:
                    last_checkpoint = checkpoint(f"periodic-{totals['task_seeds']}")
                if context.sum_counts({"stop": int(stop_requested)})["stop"]:
                    break
        last_checkpoint = (
            checkpoint("run-end") if groups_this_run else str(resolve_checkpoint(config.checkpoint))
        )
        result = {
            **totals,
            "global_step": step,
            "checkpoint": last_checkpoint,
            "Z_overshoot": max(0, totals["Z"] - config.max_latent_coordinates),
        }
        if context.is_primary:
            write_json(output / "summary.json", result)
        return result
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if optimizer is not None:
            optimizer.close()
        context.close()
