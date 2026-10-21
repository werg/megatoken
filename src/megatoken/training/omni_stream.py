"""Probability-based omnimodal sampler over immutable prepared task families.

Unlike the legacy token-deficit stream, each draw is a task-family choice.
Expensive video events are capped later by accumulation; share is not claimed
to equal GPU compute. The cursor/RNG are checkpointable and rank-local.
"""

from __future__ import annotations

import random

from megatoken.training.stream import MixtureWindowStream


def load_omni_recipe(path: str) -> dict[str, float]:
    from megatoken.io import read_mapping

    spec = read_mapping(path)
    if spec.get("omni_recipe_version") != "concordant-v1":
        raise ValueError("Unsupported immutable omnimodal recipe")
    weights = spec.get("task_weights")
    if (
        not isinstance(weights, dict)
        or not weights
        or any(
            type(k) is not str or type(v) not in (int, float) or v <= 0 for k, v in weights.items()
        )
        or abs(sum(weights.values()) - 1) > 1e-6
    ):
        raise ValueError("Omni task probabilities must be positive and sum to one")
    return weights


class OmniWindowStream(MixtureWindowStream):
    def __init__(self, *args, seed: int, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.draws = random.Random(seed + 94139)
        self.coverage = {
            "tasks": 0,
            "joint": 0,
            "multisource": 0,
            "counterfactual": 0,
            "by_family": dict.fromkeys(self.mixture, 0),
        }

    def next(self):
        keys = tuple(sorted(self.mixture))
        name = self.draws.choices(keys, [self.mixture[k] for k in keys], k=1)[0]
        item = self.streams[name].next()
        if item is None:
            return None
        self.coverage["tasks"] += 1
        self.coverage["by_family"][name] += 1
        episode = item.episode
        outputs = [
            m
            for m in episode.meridians
            if len(m.events) >= 2
            and all(
                episode.records[
                    next(i for i, r in enumerate(episode.records) if r.record_id == e.record_id)
                ].policy_mask
                for e in m.events
            )
        ]
        if outputs:
            self.coverage["joint"] += 1
        observed = {
            e.track_id
            for m in episode.meridians
            for e in m.events
            if e.role in ("system", "user", "tool", "environment", "document")
        }
        if len(observed) >= 2:
            self.coverage["multisource"] += 1
        if episode.stream == "counterfactual" or any(
            "counterfactual" in tag for tag in episode.supervision.benchmark_tags
        ):
            self.coverage["counterfactual"] += 1
        return item

    def state_dict(self):
        state = super().state_dict()
        state["omni_sampler_version"] = 1
        state["omni_rng"] = self.draws.getstate()
        state["omni_coverage"] = self.coverage
        return state

    def load_state_dict(self, state):
        if state.get("omni_sampler_version") != 1:
            raise ValueError("Incomplete omni cursor/replay identity")
        super().load_state_dict(state)
        self.draws.setstate(state["omni_rng"])
        self.coverage = state["omni_coverage"]
