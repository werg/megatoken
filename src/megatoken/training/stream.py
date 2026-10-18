"""Deterministic streaming cursors and compatibility batching of episode windows."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any

from megatoken.data import Episode, PreparedDataset, TrainingWindow, WindowConfig, iter_windows
from megatoken.io import content_hash


def _validate_topology(
    rank: int,
    world_size: int,
    reduce_counts: Callable[[dict[str, int]], dict[str, int]] | None,
) -> None:
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed rank/topology")
    if world_size > 1 and reduce_counts is None:
        raise ValueError("Distributed streams require a shared count reduction")


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
    """Synchronized episode blocks with explicit idle slots and lazy windows.

    Each block assigns one permuted episode to each rank. Ranks with fewer
    windows return None until the longest episode finishes, then all advance
    together. Padding is never retried locally or counted as training exposure.
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
        reduce_counts: Callable[[dict[str, int]], dict[str, int]] | None = None,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Training split is empty")
        _validate_topology(rank, world_size, reduce_counts)
        self.dataset, self.codec, self.config = dataset, codec, config
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0
        self.reduce_counts = reduce_counts or (lambda counts: counts)
        self.block_start = self.slot = self.block_length = self.local_length = 0
        self._episode: Episode | None = None
        self._windows: Iterator[TrainingWindow] | None = None
        self._opened = False
        self.fingerprint = content_hash(
            {"dataset": dataset.manifest, "scope": dataset.scope, "window": asdict(config)}
        )

    def _episode_index(self) -> int:
        size = len(self.dataset)
        rng = random.Random(f"megatoken:{self.seed}:{self.epoch}")
        multiplier = rng.randrange(1, max(2, size))
        while math.gcd(multiplier, size) != 1:
            multiplier = multiplier % size + 1
        offset = rng.randrange(size)
        return (multiplier * (self.block_start + self.rank) + offset) % size

    def _open_block(self) -> None:
        error = None
        episode, local_length = None, 0
        if self.block_start + self.rank < len(self.dataset):
            try:
                episode = self.dataset[self._episode_index()]
                # Counting releases each prefix slice immediately. Retaining a
                # list of every window would keep quadratic prefix references.
                local_length = sum(1 for _ in iter_windows(episode, self.codec, self.config))
                if local_length == 0:
                    raise ValueError(f"Episode {episode.episode_id} has no windows")
            except Exception as exc:
                error = exc
        counts = {
            f"rank:{rank}": local_length if rank == self.rank else 0
            for rank in range(self.world_size)
        }
        # The reduction is also the rendezvous for a resumed block. Gather its
        # cursor coordinates so every rank rejects a drifted checkpoint together,
        # before any rank starts returning windows from a different block.
        for field in ("epoch", "block_start", "slot", "block_length"):
            for rank in range(self.world_size):
                counts[f"cursor:{field}:{rank}"] = getattr(self, field) if rank == self.rank else 0
        counts["errors"] = int(error is not None)
        counts = self.reduce_counts(counts)
        if counts["errors"]:
            raise ValueError(
                f"Distributed window preparation failed: {error or 'another rank rejected its episode'}"
            ) from error
        for field in ("epoch", "block_start", "slot", "block_length"):
            if len({counts[f"cursor:{field}:{rank}"] for rank in range(self.world_size)}) != 1:
                raise ValueError(f"Synchronized data cursor differs across ranks for {field}")
        length = max(counts[f"rank:{rank}"] for rank in range(self.world_size))
        if length < 1 or self.block_length not in (0, length) or not 0 <= self.slot < length:
            raise ValueError("Invalid or changed synchronized window cursor")
        self.block_length = length
        self._episode, self.local_length = episode, local_length
        if episode is not None:
            self._windows = islice(
                iter_windows(episode, self.codec, self.config),
                min(self.slot, local_length),
                None,
            )
        self._opened = True

    def next(self) -> EpisodeWindow | None:
        if not self._opened:
            self._open_block()
        result = None
        if self.slot < self.local_length:
            result = EpisodeWindow(self._episode, next(self._windows))
        self.slot += 1
        if self.slot == self.block_length:
            self.slot = self.block_length = self.local_length = 0
            self.block_start += self.world_size
            if self.block_start >= len(self.dataset):
                self.epoch += 1
                self.block_start = 0
            self._episode = None
            self._windows = None
            self._opened = False
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "block_start": self.block_start,
            "slot": self.slot,
            "block_length": self.block_length,
            "dataset_fingerprint": self.fingerprint,
            "codec_fingerprint": self.codec.fingerprint,
        }

    def close(self) -> None:
        self.dataset.close()

    def _validate_state_dict(self, value: dict[str, Any]) -> tuple[int, int, int, int]:
        if not isinstance(value, dict):
            raise ValueError("Invalid data cursor; expected an object")
        for key, expected in (
            ("version", 2),
            ("seed", self.seed),
            ("rank", self.rank),
            ("world_size", self.world_size),
            ("dataset_fingerprint", self.fingerprint),
            ("codec_fingerprint", self.codec.fingerprint),
        ):
            actual = value.get(key)
            if actual != expected or isinstance(expected, int) and type(actual) is not int:
                raise ValueError(
                    f"Data cursor mismatch for {key}; exact resume requires the same inputs/topology"
                )
        fields = tuple(value.get(key) for key in ("epoch", "block_start", "slot", "block_length"))
        if any(type(item) is not int for item in fields):
            raise ValueError("Invalid data cursor; positions must be integer counts")
        epoch, block_start, slot, block_length = fields
        if (
            epoch < 0
            or not 0 <= block_start < len(self.dataset)
            or block_start % self.world_size
            or block_length < 0
            or not 0 <= slot < max(1, block_length)
        ):
            raise ValueError("Invalid data cursor")
        return epoch, block_start, slot, block_length

    def load_state_dict(self, value: dict[str, Any]) -> None:
        # Validate first: a failed restore cannot partially move an active stream.
        self.epoch, self.block_start, self.slot, self.block_length = self._validate_state_dict(
            value
        )
        self.local_length = 0
        self._episode = self._windows = None
        self._opened = False


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
        reduce_counts: Callable[[dict[str, int]], dict[str, int]] | None = None,
    ) -> None:
        _validate_topology(rank, world_size, reduce_counts)
        if not mixture or any(
            not isinstance(name, str)
            or not name
            or not isinstance(share, (int, float))
            or isinstance(share, bool)
            or not math.isfinite(share)
            or share <= 0
            for name, share in mixture.items()
        ):
            raise ValueError("Stage stream shares must be finite and positive")
        if abs(math.fsum(mixture.values()) - 1.0) > 1e-6:
            raise ValueError("Stage stream shares must sum to one")
        if meter not in {"D", "I", "P"}:
            raise ValueError("Window mixture meter must be D, I, or P")
        self.mixture, self.meter = dict(sorted(mixture.items())), meter
        self.reduce_counts = reduce_counts or (lambda counts: counts)
        self.streams: dict[str, WindowStream] = {}
        self.exposure = dict.fromkeys(self.mixture, 0)
        try:
            for index, name in enumerate(self.mixture):
                dataset = PreparedDataset(root, stream=name)
                try:
                    if not len(dataset):
                        raise ValueError(
                            f"Stage needs at least one episode in stream {name!r}; "
                            "prepare the declared streams before launching this recipe"
                        )
                    self.streams[name] = WindowStream(
                        dataset,
                        codec,
                        config,
                        seed=seed + index * 1009,
                        rank=rank,
                        world_size=world_size,
                        reduce_counts=reduce_counts,
                    )
                except BaseException:
                    dataset.close()
                    raise
        except BaseException:
            self.close()
            raise
        self.fingerprint = content_hash({k: s.fingerprint for k, s in self.streams.items()})

    def next(self) -> EpisodeWindow | None:
        name = min(self.streams, key=lambda key: self.exposure[key] / self.mixture[key])
        item = self.streams[name].next()
        # Never let an unscored window permanently starve every other stream.
        # P budgets still count zero; scheduling a context-only window incurs a
        # one-position virtual cost in this scheduling cursor. It never changes
        # the actual training meter, and padding itself contributes no charge.
        charge = 0 if item is None else getattr(item.window, self.meter)
        charge = self.reduce_counts({"exposure": charge})["exposure"]
        self.exposure[name] += max(1, charge)
        return item

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "kind": "mixture",
            "mixture": dict(self.mixture),
            "meter": self.meter,
            "exposure": dict(self.exposure),
            "streams": {k: s.state_dict() for k, s in self.streams.items()},
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value.get("version") != 2
            or value.get("kind") != "mixture"
            or value.get("mixture") != self.mixture
            or value.get("meter") != self.meter
        ):
            raise ValueError("Stage mixture changed during exact resume")
        states, exposure = value.get("streams"), value.get("exposure")
        if not isinstance(states, dict) or states.keys() != self.streams.keys():
            raise ValueError("Invalid mixture stream cursor")
        if (
            not isinstance(exposure, dict)
            or exposure.keys() != self.mixture.keys()
            or any(type(count) is not int or count < 0 for count in exposure.values())
        ):
            raise ValueError("Invalid mixture exposure cursor")
        # Validate every child before changing any cursor. An invalid later
        # stream or exposure cannot leave the earlier streams partly restored.
        for name, stream in self.streams.items():
            stream._validate_state_dict(states[name])
        for name, stream in self.streams.items():
            stream.load_state_dict(states[name])
        self.exposure = dict(exposure)

    def close(self) -> None:
        for stream in self.streams.values():
            stream.close()
