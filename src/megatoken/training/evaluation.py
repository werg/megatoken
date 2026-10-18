"""Fixed-harness diagnostics which keep served-prior quality separate from q."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.data import encode_record
from megatoken.generation.structure import ActionSpec
from megatoken.tensors import byte_tensor
from megatoken.training.engine import ingest_records


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
    """Prior-conditioned CE and actual prior samples on the same fixed holdout.

    The target is never supplied as a control, sampled prefix, or exact-length
    hint. The sample limit is the configured public capacity for every target.
    q reconstruction metrics must be reported separately by the caller.
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
            control_payload = episode.control_seed
            if episode.control_reference:
                control_payload += b"\n<reference>\n" + episode.control_reference
            if episode.tools_json != "[]":
                control_payload += b"\n<tools>\n" + episode.tools_json.encode()
            control = None
            if control_payload:
                control_ids, control_lengths = byte_tensor([control_payload], device)
                control = model.core.encode_control(control_ids, lengths=control_lengths)
            for record in episode.records:
                encoded = encode_record(record, codec)
                if not record.policy_mask or len(encoded.token_ids) > config.generated_capacity:
                    state = ingest_records(model, state, [encoded], device)
                    continue
                sample = model.sample(
                    state,
                    spec=ActionSpec(max_tokens=config.generated_capacity),
                    effort=config.effort,
                    control=control,
                )
                attempted += 1
                legal += int(sample.tape.committed)
                exact += int(sample.tape.committed and sample.payload == record.payload)
                # State returned by sampling does not enter held-out teacher-
                # forced CE. Both branches start from the same actual prefix.
                plan = model.prepare_target(
                    target_ids=encoded.token_ids,
                    free_mask=encoded.free_mask,
                    terminal=record.part_index + 1 == record.part_count,
                )
                step = model.supervised_step(
                    state,
                    plan=plan,
                    prior_probability=1.0,
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
            "prior_lexical_ce": sums["lexical"] / max(1, counts["lexical"]),
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
