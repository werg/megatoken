"""Training and measuring the query side of retrieval.

When an episode annotates its question (``megatoken.retrieval.annotation``), the
768-wide query the embedding head read from the query block of the lookup right after
the question (a finished user message is a lookup boundary) is
trained by two losses, one count each per annotated window:

- **query alignment**: the query matches the teacher's embedding of the question,
  scored by the centered alignment distance of the embedding head's own training
  (``generation.embedding.alignment_distance``), centered on the mean teacher
  embedding of the queries the collection was built to answer;
- **annotated ranking**: the softmax of the query's scores (cosine over a
  temperature) over the retrieved candidates and the annotated items, plus a null
  entry at the score threshold, puts its mass on the annotated items:
  ``-log(mass on the annotated items)``. A candidate that is not annotated is
  never labelled a negative; it only shares the softmax. Annotated items the
  index missed are scored anyway, so a miss still pulls the query toward them.
  An episode without annotated items has no ranking term.

The search result itself carries no gradient (top-k is discrete), and the
continuation loss reads the retrieved items like any observed events.

Retrieval quality is measured as recall@k of the annotated items for the
student's query against the teacher's own query embedding of the same question.
Collections are opened read-only and shared by path: every rank, DiLoCo worker
and evaluation reads the same files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from megatoken.data.media_assets import TensorAssetStore
from megatoken.generation.embedding import alignment_distance
from megatoken.retrieval.annotation import RetrievalAnnotation
from megatoken.retrieval.collection import SourceCollection
from megatoken.retrieval.session import Search, SourceSession

TEMPERATURE = 0.05
RECALL_K = 10

# One open handle per collection directory for the life of the process.
_MOUNTED: dict[Path, SourceCollection] = {}


def mount(annotation: RetrievalAnnotation, root: Path, teacher: str) -> SourceSession:
    """The collections an annotated episode names, as one source session."""
    root = Path(root).resolve()
    mounted = {}
    for relative in annotation.collections:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("A collection path escapes its prepared root")
        if path not in _MOUNTED:
            _MOUNTED[path] = SourceCollection.open(path, teacher=teacher)
        mounted[relative] = _MOUNTED[path]
    return SourceSession(mounted)


def close_mounted() -> None:
    for collection in _MOUNTED.values():
        collection.close()
    _MOUNTED.clear()


def _center(sources: SourceSession) -> np.ndarray:
    for collection in sources.collections.values():
        if (center := collection.query_center) is not None:
            return center
    raise ValueError("Annotated queries need a collection built with a query center")


def _recall(found: list[tuple[str, str]], positives: tuple[str, ...]) -> float:
    return len({item for _, item in found} & set(positives)) / len(positives)


def retrieval_terms(
    search: Search,
    sources: SourceSession,
    annotation: RetrievalAnnotation,
    assets: TensorAssetStore,
) -> tuple[dict[str, tuple[Tensor, int]], dict[str, float]]:
    """The annotated lookup's loss terms (sum, count) and its recall metrics."""
    query = search.query
    target = torch.from_numpy(assets.load(annotation.query)).to(query.device)
    center = torch.from_numpy(_center(sources)).to(query.device)
    terms = {"query_alignment": (alignment_distance(query[None], target[None], center).sum(), 1)}
    metrics: dict[str, float] = {}
    if annotation.positives:
        terms["retrieval_ranking"] = (_ranking(search, sources, annotation), 1)
        top = [(c.collection, c.hit.item_id) for c in search.candidates[:RECALL_K]]
        # Compare both queries against the same pre-search visibility. The
        # student's chosen sources may already have been committed by now.
        reference = sources.candidates(
            target.float().cpu().numpy(),
            RECALL_K,
            excluded=search.excluded,
            snapshots=search.snapshots or None,
        )
        metrics["retrieval_queries"] = 1.0
        metrics["retrieval_student_recall"] = _recall(top, annotation.positives)
        metrics["retrieval_teacher_recall"] = _recall(
            [(c.collection, c.hit.item_id) for c in reference], annotation.positives
        )
    return terms, metrics


def _ranking(search: Search, sources: SourceSession, annotation: RetrievalAnnotation) -> Tensor:
    query = search.query
    rows: dict[str, list[int]] = {}
    for candidate in search.candidates:
        rows.setdefault(candidate.collection, []).append(candidate.hit.row)
    annotated: dict[str, set[int]] = {}
    for name, collection in sources.collections.items():
        found = collection.find(annotation.positives)
        if found:
            annotated[name] = set(found.values())
            rows.setdefault(name, []).extend(found.values())
    if sum(len(found) for found in annotated.values()) < len(set(annotation.positives)):
        raise ValueError("An annotated item is missing from the mounted collections")
    keys, positive = [], []
    for name, members in rows.items():
        unique = list(dict.fromkeys(members))
        keys.append(torch.from_numpy(sources.collections[name].keys(unique)))
        positive += [row in annotated.get(name, ()) for row in unique]
    scores = torch.cat(keys).to(query.device).float() @ query / TEMPERATURE
    floor = max(sources.floor(name) for name in annotated)
    logits = torch.cat((scores, scores.new_tensor([floor / TEMPERATURE])))
    mass = torch.softmax(logits, dim=0)[:-1][torch.tensor(positive, device=query.device)].sum()
    return -mass.clamp_min(1e-8).log()
