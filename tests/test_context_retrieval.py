"""Retrieval comparisons share the original availability and exclusion state."""

import torch
from test_retrieval_training import annotated

from megatoken.retrieval.collection import SourceCollection
from megatoken.retrieval.session import Search
from megatoken.training.retrieval import retrieval_terms


def test_teacher_comparison_keeps_the_precommit_source_snapshot(tmp_path):
    annotation, sources, assets, root = annotated(tmp_path)
    query = torch.from_numpy(assets.load(annotation.query))
    candidates = sources.candidates(query.numpy())
    search = Search(
        query,
        candidates,
        dict(sources.search_snapshots),
        {name: frozenset(rows) for name, rows in sources.observed.items()},
    )
    chosen = sources.choose(candidates, 0)
    assert any(item.item.item_id in annotation.positives for item in chosen)
    sources.commit(chosen)
    # New rows cannot displace the reference source when the teacher query is
    # evaluated later, and the student's own newly observed row stays eligible.
    with SourceCollection.open(root / annotation.collections[0], writable=True) as writer:
        for i in range(12):
            writer.add_text(f"Future evidence {i}".encode(), query.numpy(), origin=f"future-{i}")
        writer.reindex()
    _, measured = retrieval_terms(search, sources, annotation, assets)
    assert measured["retrieval_teacher_recall"] == 1
    assert measured["retrieval_student_recall"] == 1
