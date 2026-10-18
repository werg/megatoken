"""Deterministic streaming cursors and compatibility batching of episode windows."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from megatoken.data import Episode, PreparedDataset, TrainingWindow, WindowConfig, iter_windows
from megatoken.io import content_hash


@dataclass(frozen=True)
class EpisodeWindow:
    episode: Episode
    window: TrainingWindow

    @property
    def signature(self) -> tuple[Any, ...]:
        # Same timestep ownership, not same content or routes. Batching never
        # chooses expert assignments and does not alter episode histories.
        return (
            len(self.window.prefix),
            tuple(
                (
                    entry.record.role,
                    entry.record.policy_mask,
                    entry.record.part_index,
                    entry.record.part_count,
                    entry.record.action_id is not None,
                    entry.record.result_of is not None,
                )
                for entry in self.window.records
            ),
        )


class WindowStream:
    """A resumable bijective per-epoch shuffle requiring O(1) index memory.

    ``a * position + b mod N`` with a coprime multiplier is a permutation, so
    ranks never duplicate an episode within an epoch. Epoch parameters derive
    from a stable seed. The cursor includes the window offset within an episode.
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        codec: Any,
        config: WindowConfig,
        *,
        seed: int = 17,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Training split is empty")
        if len(dataset) < world_size:
            raise ValueError("Need at least one independent training episode per rank")
        self.dataset, self.codec, self.config = dataset, codec, config
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0
        self.position = rank
        self.window_offset = 0
        self._episode: Episode | None = None
        self._windows: list[TrainingWindow] | None = None
        self.fingerprint = content_hash(dataset.manifest)

    def _episode_index(self) -> int:
        size = len(self.dataset)
        rng = random.Random(f"megatoken:{self.seed}:{self.epoch}")
        multiplier = rng.randrange(1, max(2, size))
        while math.gcd(multiplier, size) != 1:
            multiplier = multiplier % size + 1
        offset = rng.randrange(size)
        return (multiplier * self.position + offset) % size

    def next(self) -> EpisodeWindow:
        if self._windows is None:
            self._episode = self.dataset[self._episode_index()]
            self._windows = list(iter_windows(self._episode, self.codec, self.config))
            if not self._windows:
                raise ValueError(f"Episode {self._episode.episode_id} produces no training windows")
        result = EpisodeWindow(self._episode, self._windows[self.window_offset])
        self.window_offset += 1
        if self.window_offset == len(self._windows):
            self.window_offset = 0
            self.position += self.world_size
            if self.position >= len(self.dataset):
                self.epoch += 1
                self.position = self.rank
            self._episode = None
            self._windows = None
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "position": self.position,
            "window_offset": self.window_offset,
            "dataset_fingerprint": self.fingerprint,
            "codec_fingerprint": self.codec.fingerprint,
        }

    def close(self) -> None:
        self.dataset.close()

    def load_state_dict(self, value: dict[str, Any]) -> None:
        for key, expected in (
            ("version", 1),
            ("seed", self.seed),
            ("rank", self.rank),
            ("world_size", self.world_size),
            ("dataset_fingerprint", self.fingerprint),
            ("codec_fingerprint", self.codec.fingerprint),
        ):
            if value.get(key) != expected:
                raise ValueError(
                    f"Data cursor mismatch for {key}; exact resume requires the same inputs/topology"
                )
        self.epoch, self.position = int(value["epoch"]), int(value["position"])
        self.window_offset = int(value["window_offset"])
        self._episode = self._windows = None
        if self.epoch < 0 or not 0 <= self.position < len(self.dataset) or self.window_offset < 0:
            raise ValueError("Invalid data cursor")


def compatible_batches(windows: list[EpisodeWindow], batch_size: int) -> list[list[EpisodeWindow]]:
    buckets: dict[tuple[Any, ...], list[EpisodeWindow]] = defaultdict(list)
    for window in windows:
        buckets[window.signature].append(window)
    return [
        items[start : start + batch_size]
        for items in buckets.values()
        for start in range(0, len(items), batch_size)
    ]


class MixtureWindowStream:
    """Deficit scheduling by transcript exposure or actual eligible policy positions."""

    def __init__(
        self,
        root: str,
        codec: Any,
        config: WindowConfig,
        mixture: dict[str, float],
        meter: str,
        *,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        if not mixture or abs(sum(mixture.values()) - 1.0) > 1e-6:
            raise ValueError("Stage stream shares must sum to one")
        self.mixture, self.meter = mixture, meter
        self.streams: dict[str, WindowStream] = {}
        self.exposure = dict.fromkeys(mixture, 0)
        for index, name in enumerate(sorted(mixture)):
            if mixture[name] <= 0:
                raise ValueError("Mixture shares must be positive")
            dataset = PreparedDataset(root, stream=name)
            if len(dataset) < world_size:
                raise ValueError(
                    f"Stage needs at least {world_size} episodes in stream {name!r}; "
                    "prepare the declared streams before launching this recipe"
                )
            self.streams[name] = WindowStream(
                dataset, codec, config, seed=seed + index * 1009, rank=rank, world_size=world_size
            )
        self.fingerprint = content_hash({k: s.fingerprint for k, s in self.streams.items()})

    def next(self) -> EpisodeWindow:
        name = min(self.streams, key=lambda key: self.exposure[key] / self.mixture[key])
        item = self.streams[name].next()
        # Never let an unscored window permanently starve every other stream.
        # P budgets still count zero; scheduling a context-only window incurs a
        # one-position virtual cost, reported separately from the training meter.
        self.exposure[name] += max(1, getattr(item.window, self.meter))
        return item

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": "mixture",
            "mixture": self.mixture,
            "meter": self.meter,
            "exposure": dict(self.exposure),
            "streams": {k: s.state_dict() for k, s in self.streams.items()},
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value.get("mixture") != self.mixture or value.get("meter") != self.meter:
            raise ValueError("Stage mixture changed during exact resume")
        for name, stream in self.streams.items():
            stream.load_state_dict(value["streams"][name])
        self.exposure = dict(value["exposure"])

    def close(self) -> None:
        for stream in self.streams.values():
            stream.close()
