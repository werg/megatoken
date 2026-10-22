"""The query-side losses, recall, and reading from a collection through the real engine."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from retrieval_support import COLLECTION, STREAM, reading_root
from test_distributed import _start_group
from training_fixture import tiny_train_config

from megatoken.data import PreparedCorpus
from megatoken.retrieval.annotation import RetrievalAnnotation
from megatoken.retrieval.collection import SourceCollection, text_item_id
from megatoken.retrieval.session import Search
from megatoken.teacher import PINNED_TEACHER
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train
from megatoken.training.retrieval import mount, retrieval_terms


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def annotated(tmp_path, *, floor=-1.0):
    root = reading_root(tmp_path)
    corpus = PreparedCorpus([str(root)], stream=STREAM)
    episode, assets = corpus.fetch(0)
    annotation = RetrievalAnnotation.of(episode)
    sources = mount(annotation, root, PINNED_TEACHER)
    sources.threshold = floor
    return annotation, sources, assets, root


def search_for(sources, query, *, depth=8):
    candidates = sources.candidates(query.detach().numpy(), depth)
    return Search(query, candidates)


def ranking(annotation, sources, assets, query, *, depth=8):
    terms, _ = retrieval_terms(search_for(sources, query, depth=depth), sources, annotation, assets)
    return terms


def positive_key(sources, annotation):
    collection = sources.collections[COLLECTION]
    (row,) = collection.find(annotation.positives).values()
    return torch.from_numpy(collection.keys([row])[0])


def test_ranking_puts_the_mass_on_annotated_items_and_never_on_a_negative_label(tmp_path):
    annotation, sources, assets, _ = annotated(tmp_path)
    good = positive_key(sources, annotation).clone().requires_grad_()
    wrong = torch.nn.functional.normalize(torch.randn(768), dim=0).requires_grad_()
    near, far = (
        ranking(annotation, sources, assets, q)["retrieval_ranking"] for q in (good, wrong)
    )
    assert near[1] == far[1] == 1 and near[0] < far[0]
    # The loss pulls a bad query toward the annotated item, never away from a retrieved one.
    far[0].backward()
    step = -wrong.grad
    assert float(step @ good.detach()) > 0
    # An annotated item the index did not return is still scored: a miss pulls the query.
    terms = ranking(annotation, sources, assets, wrong, depth=1)
    assert terms["retrieval_ranking"][0] > 0 and terms["retrieval_ranking"][1] == 1


def test_the_null_entry_is_the_score_threshold(tmp_path):
    annotation, sources, assets, _ = annotated(tmp_path)
    query = positive_key(sources, annotation)
    losses = []
    for floor in (-1.0, 0.0, 0.9, 0.999):
        sources.threshold = floor
        losses.append(float(ranking(annotation, sources, assets, query)["retrieval_ranking"][0]))
    # Raising the floor toward the best score leaves ever more of the mass to nothing.
    assert losses == sorted(losses) and losses[-1] > 2 * losses[0]


def test_an_episode_without_annotated_items_has_alignment_only_and_a_missing_item_is_an_error(
    tmp_path,
):
    annotation, sources, assets, _ = annotated(tmp_path)
    query = positive_key(sources, annotation)
    bare = replace(annotation, positives=())
    terms = ranking(bare, sources, assets, query)
    assert set(terms) == {"query_alignment"}
    # A query that is the target has no alignment distance left.
    target = torch.from_numpy(assets.load(annotation.query))
    aligned = ranking(annotation, sources, assets, target)["query_alignment"][0]
    far = ranking(annotation, sources, assets, query)["query_alignment"][0]
    assert float(aligned) < 1e-5 < float(far)
    unknown = replace(annotation, positives=(text_item_id(b"not in any collection"),))
    with pytest.raises(ValueError, match="missing"):
        ranking(unknown, sources, assets, query)


def test_recall_compares_the_student_with_the_teacher(tmp_path):
    annotation, sources, assets, _ = annotated(tmp_path)
    target = torch.from_numpy(assets.load(annotation.query))
    _, metrics = retrieval_terms(search_for(sources, target), sources, annotation, assets)
    # The teacher's own query finds the supporting passage; so does a query equal to it.
    assert metrics == {
        "retrieval_queries": 1.0,
        "retrieval_student_recall": 1.0,
        "retrieval_teacher_recall": 1.0,
    }
    blind = torch.nn.functional.normalize(torch.randn(768), dim=0)
    _, metrics = retrieval_terms(search_for(sources, blind, depth=1), sources, annotation, assets)
    assert metrics["retrieval_teacher_recall"] == 1.0 and metrics["retrieval_student_recall"] <= 1.0


def engine_config(tmp_path, **overrides):
    train_root = reading_root(tmp_path / "train", tasks=3)
    held_out = reading_root(tmp_path / "held-out", tasks=2, first=10, heldout=True)
    config = tiny_train_config(
        tmp_path / "run",
        phase="P1",
        representation_mode="recurrent",
        microbatch_episodes=1,
        accumulate_scored=1,
        max_accumulation_windows=1,
        eval_every=1,
        eval_windows=2,
        max_steps=2,
        **overrides,
    )
    return replace(config, datasets=[str(train_root), str(held_out)])


def events(config, name):
    lines = (Path(config.output) / "metrics.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["event"] == name]


def test_the_engine_trains_the_query_losses_and_reports_held_out_recall(tmp_path):
    config = engine_config(tmp_path)
    train(config)
    trains, evaluations = events(config, "train"), events(config, "evaluation")
    assert all("query_alignment" in e["loss"] and "retrieval_ranking" in e["loss"] for e in trains)
    assert all(e["denominators"]["query_alignment"] == 1 for e in trains)
    (last,) = [e for e in evaluations if e["global_step"] == 2]
    recall = last["retrieval"]
    assert recall["queries"] == 2 and recall["k"] == 10
    assert 0 <= recall["student_recall"] <= 1 and 0.5 <= recall["teacher_recall"] <= 1
    assert last["streams"]["all"]["retrieval"] == recall
    assert last["pooled_loss"]["query_alignment"] > 0


def _two_rank_worker(rank: int, rendezvous: str, config: dict) -> None:
    _start_group(rank, rendezvous)
    try:
        train(TrainConfig.from_dict(config))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable"
)
def test_two_ranks_share_read_only_collections_and_merge_their_denominators(tmp_path):
    config = engine_config(tmp_path)
    mp.spawn(
        _two_rank_worker,
        args=(str(tmp_path / "rendezvous"), asdict(config)),
        nprocs=2,
        join=True,
    )
    train_events = events(config, "train")
    # Each rank scored one annotated window in the first step; the merged denominator is both.
    assert train_events[0]["denominators"]["query_alignment"] == 2
    (last,) = [e for e in events(config, "evaluation") if e["global_step"] == config.max_steps]
    assert last["retrieval"]["queries"] == 2


def test_collections_stay_untouched_by_training(tmp_path):
    config = engine_config(tmp_path)
    path = Path(config.datasets[0]) / COLLECTION / "collection.sqlite"
    before = path.read_bytes()
    train(config)
    assert path.read_bytes() == before
    with SourceCollection.open(path.parent) as collection:
        assert collection.count == 6 * 3
