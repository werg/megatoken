"""Offline centroid fitting and typed posting publication on a pinned source catalog."""
import numpy as np
import pytest

from megatoken.io import sha256_file
from megatoken.memory.centroids_cli import fit_centroids
from megatoken.memory.contracts import Collection, Footprint, SemanticSpace, SourceRange, Visibility
from megatoken.memory.disk_ivf import SQLiteDiskIVF
from megatoken.memory.growth import GrowthPolicy
from megatoken.memory.index_cli import publish_index
from megatoken.memory.recipe import RetrievalRecipe
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model.config import ModelConfig


def test_bounded_spherical_centroids_and_source_memory_postings(tmp_path):
    model = ModelConfig.tiny(
        semantic_dim=8, semantic_value_dim=16, max_kb_positions=2,
    )
    space = SemanticSpace(
        "source-space", "v1", "teacher", "pinned", 8,
        "doc", "query", "source-windows",
    )
    recipe = RetrievalRecipe(
        schema="megatoken-retrieval-recipe-v1",
        space=space, model=model, growth=GrowthPolicy(value_dim=16, seed=7),
        loss_weights={"native": 1.},
        max_kb_positions=2, max_readout_groups=2,
    )
    bank_path = tmp_path / "bank.sqlite"
    with SQLiteKB(bank_path, space, value_dim=16) as bank:
        for i in range(6):
            source = SourceRange(
                f"source-{i}", "v1", 0, 12, f"hash-{i}", "team",
            )
            key = np.eye(8, dtype=np.float32)[i]
            bank.insert_source(
                scope="team", footprint=Footprint((source,)), key=key,
                content_id=f"chunk-{i}", asset_ref=f"{i}.bin",
            )
            bank.insert_memory(
                scope="team", footprint=Footprint((source,)), key=key,
                value=np.zeros(16), assignment=f"chunk-{i}", birth_step=0,
            )
    output = tmp_path / "centroids.npy"
    receipt = fit_centroids(
        recipe, bank_path=bank_path, output=output,
        cells=3, max_samples=4, iterations=3, seed=19,
    )
    assert receipt["source_records"] == 6
    assert receipt["sample_count"] == 4
    centroids = np.load(output, allow_pickle=False)
    assert centroids.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1., atol=1e-5)
    with pytest.raises(FileExistsError):
        fit_centroids(
            recipe, bank_path=bank_path, output=output,
            cells=3, max_samples=4,
        )
    published = publish_index(
        recipe, bank_path=bank_path, centroid_path=output,
        centroid_sha256=sha256_file(output), nprobe=3,
    )
    assert published["postings"]["team:source"] == 6
    assert published["postings"]["team:memory"] == 6
    with SQLiteKB(bank_path, space, value_dim=16, readonly=True) as bank:
        index = SQLiteDiskIVF(bank, centroids, nprobe=3)
        retrieved = index.search(
            np.eye(8)[0], collection=Collection.SOURCE,
            scope="team", visibility=Visibility("team", 2), limit=6,
        )
        assert len(retrieved.ids) == 6
