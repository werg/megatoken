"""Source catalog imports require original bytes, context and learned-space pins."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from megatoken.memory.catalog_cli import build_source_catalog
from megatoken.memory.contracts import Collection, Footprint, SemanticSpace, SourceRange
from megatoken.memory.growth import GrowthPolicy
from megatoken.memory.recipe import RetrievalRecipe
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model.config import ModelConfig


def _source_setup(tmp_path):
    model = ModelConfig.tiny(
        semantic_dim=8, semantic_value_dim=16, max_kb_positions=2,
    )
    space = SemanticSpace(
        "teacher-shared", "v1", "offline", "pinned-checkpoint", 8,
        "document-format", "query-format", "ordered-block-context",
    )
    recipe = RetrievalRecipe(
        schema="megatoken-retrieval-recipe-v1",
        model=model, space=space,
        growth=GrowthPolicy(
            value_dim=16, seed=2, partition_id="pinned-source-partition",
        ),
        loss_weights={"native": 1.}, max_kb_positions=2,
        max_readout_groups=2,
    )
    root = tmp_path / "originals"
    root.mkdir()
    payload = b"The exact immutable versioned original bytes."
    (root / "doc.bin").write_bytes(payload)
    footprint = Footprint((
        SourceRange(
            "document-id", "v1", 0, len(payload),
            sha256(payload).hexdigest(), "team",
        ),
    ))
    entry = {
        "semantic_space_fingerprint": space.fingerprint,
        "scope": "team",
        "content_id": "document-id/chunk-0",
        "asset_ref": "doc.bin",
        "asset_sha256": sha256(payload).hexdigest(),
        "footprint": [asdict(r) for r in footprint.ranges],
        "evidence_span": asdict(footprint.ranges[0]),
        "key": [1.] + [0.] * 7,
    }
    return recipe, root, entry


def test_source_catalog_import_keeps_free_values_disabled_until_measured_readiness(tmp_path):
    recipe, root, entry = _source_setup(tmp_path)
    catalog = tmp_path / "source.jsonl"
    catalog.write_text(json.dumps(entry) + "\n")
    output = tmp_path / "catalog.sqlite"
    report = build_source_catalog(
        recipe, catalog=catalog, assets_root=root, output=output,
    )
    assert report["sources"] == 1
    assert not report["ready_for_free_allocation"]
    with SQLiteKB(output, recipe.space, value_dim=16, readonly=True) as bank:
        rows = bank.scan(Collection.SOURCE, "team")
        assert len(rows) == 1
        assert rows[0].value is None
        assert not bank.growth_state("readiness")["coherence"]


@pytest.mark.parametrize("corruption", ["asset_sha256", "semantic_space_fingerprint"])
def test_catalog_rejects_tampered_inputs_without_publishing_bank(tmp_path, corruption):
    recipe, root, entry = _source_setup(tmp_path)
    entry[corruption] = "wrong-revision"
    catalog = tmp_path / "source.jsonl"
    catalog.write_text(json.dumps(entry) + "\n")
    output = tmp_path / "catalog.sqlite"
    with pytest.raises(ValueError):
        build_source_catalog(
            recipe, catalog=catalog, assets_root=root, output=output,
        )
    assert not output.exists()
