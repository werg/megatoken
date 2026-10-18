"""Fixed-harness diagnostics separating next-facet prediction and reconstruction."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.controls import encode_controls
from megatoken.data import encode_record
from megatoken.generation.structure import ActionSpec
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
    model: Any, dataset: Any, codec: Any, config: Any, device: torch.device
) -> dict[str, Any]:
    """Causal CE and actual samples under the phase's representation mode.

    The target is never supplied as a control, sampled prefix, or exact-length
    hint. The sample limit is the configured public capacity for every target.
    Bootstrap measures encoder reconstruction separately and teacher-forces E.
    Recurrent evaluation keeps sampled facets live between consecutive targets.
    """
    was_training = model.training
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    exact = attempted = legal = 0
    episodes = min(config.eval_episodes, len(dataset))
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
            "episodes": episodes,
            "representation_mode": config.representation_training,
            "prior_lexical_ce": sums["lexical"] / max(1, counts["lexical"]),
            "reconstruction_lexical_ce": (
                sums["reconstruction"] / counts["reconstruction"]
                if counts["reconstruction"]
                else None
            ),
            "facet_nll_per_coordinate": (
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
