"""Held-out loss of the training objective, per stage stream, without an update.

The identical meridian forward that trains runs on a fixed selection of each
stage stream's validation split, in eval mode (so transducer lookups read the
bank alone and record nothing), under ``torch.no_grad`` and with no optimizer
step. Every term the forward scores (policy and reconstruction cross-entropy,
facet prediction, the audio and visual terms, ...) is reported as a mean over
its own denominator, per stream and pooled. Windows of episodes read from a
collection also report retrieval recall: the share of the annotated items in the
student's top ``k`` against the teacher's own query embedding of the same question.

The selection is the first ``eval_windows`` windows of the stream's validation
episodes in index order, so every evaluation of a run measures the same
windows (an episode with many windows can fill the selection). Ranks of a synchronous run take them in turns; a DiLoCo worker is its
own world of one and only one worker evaluates.

Windows of reading-from-a-base episodes (whose reference items are declared for a base
rather than placed in the context) are scored twice: with the base built from the
episode's own items mounted, as in training, and with nothing mounted. The two are
reported side by side as ``reading_base``; the difference is what the mounted base is
worth. Each window mounts only its own episode's items, so the numbers do not depend on
how many ranks share the windows.
"""

from __future__ import annotations

from typing import Any

import torch

from megatoken.data import PreparedCorpus, WindowConfig, iter_windows, stage_recipe
from megatoken.data.knowledge import declared_items
from megatoken.runtime import capture_rng, restore_rng
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.reading_base import ReadingBase
from megatoken.training.retrieval import RECALL_K
from megatoken.training.stream import EpisodeWindow

ALL = "all"  # the stream name when the whole corpus is one stream
_SEED_OFFSET = 7919  # evaluation draws never share the training run's random stream
# Entries that re-score windows the stream entries already count: kept out of the pooled loss.
READING = "reading-base:"
MOUNTED, UNMOUNTED = READING + "mounted", READING + "unmounted"

Sums = dict[str, dict[str, Any]]


class HeldOutEvaluator:
    """Validation corpora of the stage's streams, opened once and kept for the run."""

    def __init__(self, config: Any, codec: Any, window: WindowConfig, *, require: bool) -> None:
        self.config, self.codec, self.window = config, codec, window
        names = sorted(stage_recipe(config.phase).mixture) if config.use_stage_mixture else [None]
        self.corpora: dict[str, PreparedCorpus] = {}
        try:
            for name in names:
                self.corpora[name or ALL] = PreparedCorpus(
                    config.datasets, split="validation", stream=name
                )
        except BaseException:
            self.close()
            raise
        if require and not any(len(corpus) for corpus in self.corpora.values()):
            self.close()
            raise ValueError(
                "Held-out evaluation needs validation episodes in at least one stage stream "
                f"of {config.datasets}; prepare with a validation fraction"
            )

    def close(self) -> None:
        for corpus in self.corpora.values():
            corpus.close()

    def _selection(self, corpus: PreparedCorpus, rank: int, world: int):
        """This rank's share of the stream's first ``eval_windows`` windows, with their ordinals."""
        taken = 0
        for index in range(len(corpus)):
            episode, assets = corpus.fetch(index)
            for window in iter_windows(episode, self.codec, self.window):
                if taken >= self.config.eval_windows:
                    return
                if taken % world == rank:
                    yield taken, EpisodeWindow(episode, window, assets)
                taken += 1

    @torch.no_grad()
    def sums(self, model: Any, transducer: Any, step: int, *, rank: int, world: int) -> Sums:
        """Per stream: scored windows and each term's loss sum and denominator on this rank.

        Each window draws from its own seed (its ordinal in the selection), so its
        numbers do not depend on how many ranks share the windows.
        """
        from megatoken.training.engine import prepare_batches

        config = self.config
        rng = capture_rng()
        was_training = model.training
        model.eval()
        reading = transducer.reading
        result: Sums = {}

        def entry(name: str) -> dict[str, Any]:
            return result.setdefault(name, {"windows": 0, "sum": {}, "count": {}, "recall": {}})

        try:
            device = model.core.initial_facets.device
            for name, corpus in self.corpora.items():
                entry(name)
                for ordinal, item in self._selection(corpus, rank, world):
                    declared = bool(declared_items(item.episode))
                    # Scored with its own items mounted (and nothing else) and, if it
                    # declares any, again with nothing mounted.
                    for mounted in (True, False) if declared else (True,):
                        transducer.reading = (
                            ReadingBase(model.transducer, capacity=1) if mounted else None
                        )
                        torch.manual_seed(config.seed + _SEED_OFFSET + ordinal)
                        names = (
                            [UNMOUNTED] if not mounted else [name, MOUNTED] if declared else [name]
                        )
                        for batch in prepare_batches(model, [item], config):
                            ledger = LossLedger(
                                dict(batch.counts), dict(batch.counts), config.loss_weights
                            )
                            metrics = forward_meridian_batch(
                                model,
                                batch,
                                ledger,
                                config,
                                device,
                                step,
                                differentiate=False,
                                transducer=transducer,
                            )
                            ledger.finish()
                            for target in map(entry, names):
                                target["windows"] += 1
                                for term, value in ledger.sums.items():
                                    target["sum"][term] = target["sum"].get(term, 0.0) + float(
                                        value
                                    )
                                for term, count in batch.counts.items():
                                    target["count"][term] = target["count"].get(term, 0) + int(
                                        count
                                    )
                                for key, value in metrics.items():
                                    if key.startswith("retrieval_"):
                                        target["recall"][key] = (
                                            target["recall"].get(key, 0.0) + value
                                        )
        finally:
            transducer.reading = reading
            model.train(was_training)
            restore_rng(rng)
        return result

    def weighted(self, sums: dict[str, float], counts: dict[str, int]) -> float:
        """The training objective's weighted mean, as in training without router balance."""
        weights = {**self.config.loss_weights, "router_balance": 0.0}
        return sum(weights[term] * sums.get(term, 0.0) / n for term, n in counts.items() if n)

    def bank_only_loss(self, model: Any, transducer: Any, step: int) -> float:
        """Pooled held-out loss with every lookup reading the bank alone (value-switch rule)."""
        _, pooled = merge([self.sums(model, transducer, step, rank=0, world=1)])
        return self.weighted(pooled["sum"], pooled["count"])

    def report(self, gathered: list[Sums]) -> dict[str, Any]:
        """Per-stream and pooled means from every rank's ``sums``."""
        streams, pooled = merge(gathered)
        reading = {}
        if UNMOUNTED in streams:
            mounted, unmounted = (
                self.weighted(streams[key]["sum"], streams[key]["count"])
                for key in (MOUNTED, UNMOUNTED)
            )
            reading = {
                "reading_base": {
                    "windows": streams[MOUNTED]["windows"],
                    "mounted_loss": mounted,
                    "unmounted_loss": unmounted,
                    "gain": unmounted - mounted,
                }
            }
        return {
            **reading,
            "eval_windows_per_stream": self.config.eval_windows,
            "streams": {
                name: {
                    "windows": entry["windows"],
                    "loss": {t: entry["sum"][t] / n for t, n in entry["count"].items() if n},
                    "denominators": {t: n for t, n in entry["count"].items() if n},
                    "weighted_loss": self.weighted(entry["sum"], entry["count"]),
                    **recall_report(entry["recall"]),
                }
                for name, entry in streams.items()
            },
            **recall_report(pooled["recall"]),
            "pooled_loss": {t: pooled["sum"][t] / n for t, n in pooled["count"].items() if n},
            "pooled_weighted_loss": self.weighted(pooled["sum"], pooled["count"]),
        }


def merge(gathered: list[Sums]) -> tuple[Sums, dict[str, Any]]:
    """Add the ranks' sums per stream, and over every stream."""
    streams: Sums = {}
    pooled: dict[str, Any] = {"windows": 0, "sum": {}, "count": {}, "recall": {}}
    for sums in gathered:
        for name, entry in sums.items():
            target = streams.setdefault(name, {"windows": 0, "sum": {}, "count": {}, "recall": {}})
            for destination in (target,) if name.startswith(READING) else (target, pooled):
                destination["windows"] += entry["windows"]
                for field in ("sum", "count", "recall"):
                    for term, value in entry[field].items():
                        destination[field][term] = destination[field].get(term, 0) + value
    return streams, pooled


def recall_report(recall: dict[str, float]) -> dict[str, Any]:
    """Mean recall@k of the annotated items, student against teacher queries (if any)."""
    queries = recall.get("retrieval_queries", 0)
    if not queries:
        return {}
    return {
        "retrieval": {
            "queries": int(queries),
            "k": RECALL_K,
            "student_recall": recall["retrieval_student_recall"] / queries,
            "teacher_recall": recall["retrieval_teacher_recall"] / queries,
        }
    }
