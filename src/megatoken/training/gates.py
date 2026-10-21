"""Measured held-out acceptance gates for the retrieval-memory curriculum.

Every gate runs on held-out episodes through the *training* forward
(``forward_meridian_batch`` without differentiation), so the evaluated
computation is exactly the trained one. Results are written as evidence files,
and a phase receipt lists only gates whose measured value met its threshold.

Bootstrap gates (coherence_embedding -> progressive_warmup):

- ``coherence``: held-out per-token lexical CE <= ``max_lexical_ce`` and the
  terminal CE below ``max_terminal_ce`` (default: binary chance, ln 2).
- ``embedding_alignment``: mean cosine between E's realized document embeddings
  and the teacher's >= ``min_document_cosine``.
- ``query_alignment``: mean cosine between P's realized queries and the teacher's
  >= ``min_query_cosine``, and student recall@k of annotated gold sources in the
  teacher-keyed source catalog >= ``min_recall_ratio`` x the teacher's own recall@k.
- ``index_recall`` (with pinned centroids): IVF top-k overlap with exact top-k
  for the student's queries >= ``min_index_recall``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from megatoken.io import sha256_file

BOOTSTRAP_GATES = ("coherence", "embedding_alignment", "query_alignment")


@dataclass(frozen=True)
class GateThresholds:
    max_lexical_ce: float = 3.0
    max_terminal_ce: float = math.log(2)
    min_document_cosine: float = 0.75
    min_query_cosine: float = 0.75
    min_recall_ratio: float = 0.75
    min_index_recall: float = 0.95
    recall_k: int = 10


@dataclass
class HeldoutPass:
    losses: dict[str, float]
    counts: dict[str, int]
    queries: dict[tuple[str, str], np.ndarray]
    documents: dict[tuple[str, str], np.ndarray]
    episodes: int

    def per_token(self, name: str) -> float:
        count = self.counts.get(name, 0)
        return self.losses.get(name, 0.0) / count if count else math.nan


def heldout_forward(
    model: Any,
    config: Any,
    codec: Any,
    dataset: Any,
    *,
    recipe: Any,
    queries: dict,
    documents: dict,
    max_episodes: int,
    model_version: int,
) -> HeldoutPass:
    """Teacher-forced held-out pass through the training forward (no updates)."""
    import torch

    from megatoken.data import iter_windows
    from megatoken.training.engine import prepare_batches, window_config
    from megatoken.training.losses import LossLedger
    from megatoken.training.meridians import RetrievalProbe, forward_meridian_batch
    from megatoken.training.stream import EpisodeWindow

    config = dataclasses.replace(config, retrieval_kv_dropout=0.0, retrieval_lora_dropout=0.0)
    windows_config = window_config(config)
    probe = RetrievalProbe()
    losses: dict[str, float] = {}
    counts: dict[str, int] = {}
    episodes = 0
    was_training = model.training
    model.eval()
    try:
        for episode in islice(dataset.iter_episodes(), max_episodes):
            episodes += 1
            windows = [
                EpisodeWindow(episode, window)
                for window in iter_windows(episode, codec, windows_config)
            ]
            for batch in prepare_batches(model, windows, config):
                ledger = LossLedger(
                    expected_local=dict(batch.counts),
                    global_counts=dict(batch.counts),
                    weights=config.loss_weights,
                )
                aux = (
                    queries,
                    recipe,
                    max(1, len(queries)),
                    None,
                    documents,
                    max(1, len(documents)),
                )
                with torch.no_grad():
                    forward_meridian_batch(
                        model,
                        batch,
                        ledger,
                        config,
                        next(model.parameters()).device,
                        model_version,
                        aux,
                        differentiate=False,
                        probe=probe,
                    )
                ledger.finish()
                for name, value in ledger.sums.items():
                    losses[name] = losses.get(name, 0.0) + float(value)
                for name, count in batch.counts.items():
                    counts[name] = counts.get(name, 0) + int(count)
    finally:
        model.train(was_training)
    return HeldoutPass(
        losses,
        counts,
        {k: v.numpy() for k, v in probe.queries.items()},
        {k: v.numpy() for k, v in probe.documents.items()},
        episodes,
    )


def _unit(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value / max(np.linalg.norm(value), 1e-12)


def _mean_cosine(student: dict, teacher: dict) -> tuple[float, int]:
    keys = sorted(set(student) & set(teacher))
    if not keys:
        return math.nan, 0
    return float(np.mean([_unit(student[k]) @ _unit(teacher[k]) for k in keys])), len(keys)


def retrieval_recall(
    bank: Any,
    annotations: list,
    student: dict[tuple[str, str], np.ndarray],
    *,
    k: int,
    centroids: np.ndarray | None,
) -> dict[str, Any]:
    """Recall@k of annotated gold sources for student vs teacher query vectors."""
    from megatoken.memory.contracts import Collection, Visibility
    from megatoken.memory.disk_ivf import SQLiteDiskIVF
    from megatoken.memory.index import ExactIndex

    groups: dict[tuple[str, int], list] = {}
    for annotation in annotations:
        if (
            annotation.episode_id,
            annotation.event_id,
        ) in student and annotation.positive_content_ids:
            scope = annotation.query_teacher_inputs.ranges[0].scope
            groups.setdefault((scope, annotation.query.meridian), []).append(annotation)
    exact = ExactIndex(bank)
    ivf = SQLiteDiskIVF.published(bank, centroids) if centroids is not None else None
    student_recall, teacher_recall, index_overlap = [], [], []
    for (scope, clock), items in sorted(groups.items()):
        visibility = Visibility(scope, clock)
        search = dict(collection=Collection.SOURCE, scope=scope, visibility=visibility, limit=k)
        student_hits = exact.search_many(
            [student[(a.episode_id, a.event_id)] for a in items], **search
        )
        teacher_hits = exact.search_many([a.teacher_target for a in items], **search)
        for annotation, mine, reference in zip(items, student_hits, teacher_hits, strict=True):
            gold = set(annotation.positive_content_ids)
            student_recall.append(len(gold & {h.record.content_id for h in mine.hits}) / len(gold))
            teacher_recall.append(
                len(gold & {h.record.content_id for h in reference.hits}) / len(gold)
            )
            if ivf is not None:
                approximate = ivf.search(
                    student[(annotation.episode_id, annotation.event_id)], **search
                )
                index_overlap.append(
                    len(set(approximate.ids) & set(mine.ids)) / max(1, len(mine.ids))
                )
    return {
        "queries": len(student_recall),
        "k": k,
        "student_recall": float(np.mean(student_recall)) if student_recall else math.nan,
        "teacher_recall": float(np.mean(teacher_recall)) if teacher_recall else math.nan,
        "index_recall": float(np.mean(index_overlap)) if index_overlap else None,
        "nprobe": None if ivf is None else ivf.nprobe,
    }


def bootstrap_gates(
    *,
    checkpoint: str | Path,
    dataset: str | Path,
    query_sidecar: str | Path,
    document_sidecar: str | Path,
    recipe_path: str | Path,
    bank_path: str | Path,
    output: str | Path,
    centroids: str | Path | None = None,
    max_episodes: int = 1000,
    thresholds: GateThresholds | None = None,
    device: str = "cuda",
    mark_ready: bool = False,
) -> dict[str, Any]:
    """Measure the bootstrap gates on held-out data and write evidence + receipt.

    ``mark_ready`` records the measured readiness on the bank and writes the
    progressive_warmup receipt only when every bootstrap gate passed.
    """
    from megatoken.data import PreparedDataset
    from megatoken.data.document_embeddings import read_document_sidecar
    from megatoken.data.retrieval import read_sidecar
    from megatoken.loading import load_trained
    from megatoken.memory.recipe import RetrievalRecipe, open_bank_for_recipe, record_readiness
    from megatoken.training.config import TrainConfig

    thresholds = thresholds or GateThresholds()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = RetrievalRecipe.load(recipe_path)
    trained = load_trained(checkpoint, device=device)
    config = TrainConfig.from_dict(trained.payload["train_config"])
    annotations = read_sidecar(query_sidecar)
    queries = {(a.episode_id, a.event_id): a for a in annotations}
    documents = {(d.episode_id, d.event_id): d for d in read_document_sidecar(document_sidecar)}
    heldout = PreparedDataset(dataset, split="validation")
    try:
        measured = heldout_forward(
            trained.model,
            config,
            trained.codec,
            heldout,
            recipe=recipe,
            queries=queries,
            documents=documents,
            max_episodes=max_episodes,
            model_version=trained.payload["global_step"],
        )
    finally:
        heldout.close()
    document_cosine, document_count = _mean_cosine(
        measured.documents, {k: v.teacher_target for k, v in documents.items()}
    )
    query_cosine, query_count = _mean_cosine(
        measured.queries, {k: v.teacher_target for k, v in queries.items()}
    )
    with open_bank_for_recipe(recipe, bank_path, readonly=not mark_ready) as bank:
        recall = retrieval_recall(
            bank,
            annotations,
            measured.queries,
            k=thresholds.recall_k,
            centroids=None if centroids is None else np.load(centroids),
        )
        lexical = measured.per_token("lexical")
        terminal = measured.per_token("terminal")
        evidence = {
            "coherence": {
                "lexical_ce_per_token": lexical,
                "terminal_ce": terminal,
                "structure_ce": measured.per_token("structure"),
                "passed": (
                    lexical <= thresholds.max_lexical_ce and terminal < thresholds.max_terminal_ce
                ),
            },
            "embedding_alignment": {
                "document_cosine": document_cosine,
                "documents": document_count,
                "passed": document_count > 0 and document_cosine >= thresholds.min_document_cosine,
            },
            "query_alignment": {
                "query_cosine": query_cosine,
                "queries": query_count,
                **{key: recall[key] for key in ("k", "student_recall", "teacher_recall")},
                "passed": (
                    query_count > 0
                    and query_cosine >= thresholds.min_query_cosine
                    and recall["student_recall"]
                    >= thresholds.min_recall_ratio * recall["teacher_recall"]
                ),
            },
        }
        if recall["index_recall"] is not None:
            evidence["index_recall"] = {
                "ivf_exact_overlap": recall["index_recall"],
                "nprobe": recall["nprobe"],
                "passed": recall["index_recall"] >= thresholds.min_index_recall,
            }
        identity = {
            "checkpoint": str(checkpoint),
            "global_step": trained.payload["global_step"],
            "dataset": str(dataset),
            "episodes": measured.episodes,
            "query_sidecar_sha256": sha256_file(Path(query_sidecar)),
            "document_sidecar_sha256": sha256_file(Path(document_sidecar)),
            "recipe": recipe.fingerprint,
            "thresholds": asdict(thresholds),
        }
        paths = {}
        for gate, value in evidence.items():
            paths[gate] = str(output / f"{gate}.json")
            Path(paths[gate]).write_text(json.dumps({**identity, **value}, indent=2) + "\n")
        passed = tuple(gate for gate, value in evidence.items() if value["passed"])
        ready = set(BOOTSTRAP_GATES) <= set(passed)
        if mark_ready and ready:
            record_readiness(
                recipe,
                bank,
                coherence=True,
                embedding_alignment=True,
                query_alignment=True,
            )
    receipt = None
    if ready:
        receipt = output / "progressive_warmup.gate.json"
        receipt.write_text(
            json.dumps(
                {
                    "phase": "progressive_warmup",
                    "evidence": {gate: paths[gate] for gate in passed},
                    "model_snapshot": str(checkpoint),
                    "kb_snapshot": sha256_file(Path(bank_path)),
                    "corpus_manifest": recipe.fingerprint,
                    "successful": list(passed),
                },
                indent=2,
            )
            + "\n"
        )
    return {
        "passed": list(passed),
        "ready": ready,
        "receipt": None if receipt is None else str(receipt),
        "evidence": evidence,
    }
