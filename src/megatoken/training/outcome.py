"""Verified joint-policy improvement over recorded serving decisions.

One denominator is one whole selected group, including failed or empty renders.
Tools and readiness schedules have no invented policy density. Costs are
explicit measured work/delay multiplied by frozen recipe coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from megatoken.model.policy_context import reconstruct


@dataclass(frozen=True)
class OutcomeCredit:
    reward: float
    baseline: float
    selected_events: int
    verification: str
    work_cost: float = 0.0
    delay_seconds: float = 0.0
    delay_cost: float = 0.0

    def __post_init__(self):
        if self.verification not in {"external_tests", "local_assertions", "publisher_tests"}:
            raise ValueError("Outcome learning requires independent executed verification")
        if (
            not all(
                math.isfinite(x)
                for x in (
                    self.reward,
                    self.baseline,
                    self.work_cost,
                    self.delay_seconds,
                    self.delay_cost,
                )
            )
            or not -1 <= self.reward <= 1
            or not -1 <= self.baseline <= 1
            or type(self.selected_events) is not int
            or self.selected_events < 1
            or min(self.work_cost, self.delay_seconds, self.delay_cost) < 0
        ):
            raise ValueError("Outcome credit needs bounded reward and measured nonnegative costs")

    @property
    def advantage(self):
        return (
            self.reward
            - self.baseline
            - self.work_cost * self.selected_events
            - self.delay_cost * self.delay_seconds
        )


def policy_outcome(multi, tape, credit: OutcomeCredit, *, device):
    if tape.policy_context is None or len(tape.policy_context) != tape.meridian_id:
        raise ValueError("Outcome learning needs the complete committed serving context")
    if credit.selected_events != len(tape.tracks):
        raise ValueError("Outcome work count differs from the complete selected decision")
    state = reconstruct(multi, tape.policy_context, device=device)
    from megatoken.model.context import ContextPolicy

    state = multi.tracks.set_context_policy(
        state, ContextPolicy.from_json(tape.context_policy_json)
    )
    log_prob = multi.score_group(state, tape)
    delta = log_prob - tape.behavior_log_prob
    # Bounded exponentiation avoids nonfinite importance weights while PPO's
    # sign-aware minimum still discourages harmful stale decisions.
    ratio = delta.clamp(-20, 20).exp()
    advantage = log_prob.new_tensor(credit.advantage)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    return {"policy_outcome": (-surrogate, 1)}, {
        "outcome_decisions": 1,
        "outcome_advantage_sum": advantage,
        "outcome_log_ratio_sum": delta.detach(),
    }
