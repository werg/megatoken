"""Regeneration must be writer-only, complete, shadow-published and source-addressable."""
import json

import pytest
import torch

from megatoken.io import sha256_file
from megatoken.memory.authoring import (
    FrozenCorpusAuthor, FrozenWriterArtifact, PairAssignment, SourceAssignment,
)
from megatoken.memory.contracts import Collection, Footprint, SemanticSpace, SourceRange
from megatoken.memory.sqlite import SQLiteKB
from megatoken.memory.writers import RelationWriter, SourceWriter


def _artifact(path, model, *, kind, layout, space):
    from safetensors.torch import save_file

    path.mkdir()
    weights = path / "writer.safetensors"
    save_file({
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }, str(weights))
    (path / "manifest.json").write_text(json.dumps({
        "schema": "megatoken-writer-checkpoint-v1",
        "kind": kind, "mode": "full", "recipe": "accepted-recipe",
        "semantic_space": space.fingerprint,
        "native_model_snapshot": "frozen-native",
        "free_kb_snapshot": "frozen-free-bank",
        "layout": layout, "hidden": 32, "heads": 4,
        "weights_sha256": sha256_file(weights),
    }))
    return FrozenWriterArtifact(
        path, expected_recipe="accepted-recipe",
        expected_space=space.fingerprint,
        expected_model_snapshot="frozen-native",
        expected_free_bank="frozen-free-bank", kind=kind,
    )


def test_writer_only_shadow_rebuild_publishes_source_memory_and_pair(tmp_path):
    pytest.importorskip("safetensors.torch")
    torch.manual_seed(23)
    space = SemanticSpace(
        "space", "v1", "pinned", "teacher", 8, "document", "query", "ordered",
    )
    source = _artifact(
        tmp_path / "source-writer",
        SourceWriter(
            facets=4, facet_dim=8, semantic_dim=8,
            value_dim=16, hidden=32, heads=4, max_blocks=8,
        ), kind="source",
        layout={"facets": 4, "facet_dim": 8, "semantic_dim": 8,
                "value_dim": 16, "max_blocks": 8}, space=space,
    )
    pair = _artifact(
        tmp_path / "pair-writer",
        RelationWriter(
            key_dim=8, value_dim=16, statistics_dim=3, hidden=32,
        ), kind="pair",
        layout={"semantic_dim": 8, "value_dim": 16, "statistics_dim": 3},
        space=space,
    )
    author = FrozenCorpusAuthor(
        source_writer=source, pair_writer=pair,
        semantic_space=space, value_dim=16,
    )
    left = SourceRange("source-left", "v1", 0, 12, "sha-left", "team")
    right = SourceRange("source-right", "v1", 0, 12, "sha-right", "team")
    assignments = (
        SourceAssignment(
            "point-left", "chunk-left", "team", Footprint((left,)),
            left, "left.asset", "source-left", 0., 0, 0,
        ),
        SourceAssignment(
            "point-right", "chunk-right", "team", Footprint((right,)),
            right, "right.asset", "source-right", 0., 0, 1,
        ),
    )
    assert author.add_sources(
        torch.randn(1, 3, 4, 8), torch.ones(1, 3, dtype=torch.bool),
        assignments,
    ) == 2
    author.add_pairs((
        PairAssignment(
            "point-left", "point-right", "team",
            Footprint((left, right)), 0.7, torch.randn(3),
        ),
    ))
    output = tmp_path / "regenerated.sqlite"
    result = author.publish(
        output, expected_points={"point-left", "point-right"},
        expected_sources={"chunk-left", "chunk-right"},
        expected_pairs={("point-left", "point-right")},
    )
    assert result.source_coverage == result.point_coverage == result.pair_coverage == 1.
    with SQLiteKB(output, space, value_dim=16, readonly=True) as bank:
        assert len(bank.scan(Collection.SOURCE, "team")) == 2
        assert len(bank.scan(Collection.MEMORY, "team")) == 2
        assert len(bank.relations("team")) == 1


def test_missing_reproduced_point_blocks_publication(tmp_path):
    from megatoken.memory.reproduction import regenerate_bank
    space = SemanticSpace(
        "space", "v1", "pinned", "teacher", 8,
        "document", "query", "ordered",
    )
    output = tmp_path / "regenerated.sqlite"
    with pytest.raises(RuntimeError, match="missing point"):
        regenerate_bank(
            output, space, value_dim=16, points=[], pairs=[],
            expected_assignments={"must-exist"}, expected_pairs=set(),
        )
    assert not output.exists()
