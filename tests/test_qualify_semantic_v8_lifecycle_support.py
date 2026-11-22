"""Source-bound support-cohort lifecycle qualification controls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.source_identity_v8 import source_evidence_binding_v8

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_semantic_v8_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("qualify_semantic_v8_lifecycle_support", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Corpus:
    rows = []
    def __init__(self, roots, split=None):
        assert split is None
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return None
    def iter_episodes(self):
        return iter(self.rows)


def _binding(document, identity, member, component):
    side = document["sidecar"]
    return {
        "schema": "semantic-region-dataset-binding/1",
        "traceFormat": "semantic-trace/8",
        "episodeDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        **source_evidence_binding_v8(side),
        "registryDigest": identity["registryDigest"],
        "componentId": component["id"],
        "split": component["split"],
        "sourceId": member["id"],
        "family": member["family"],
        "sourceDigest": member["sourceDigest"],
        "coreDigest": member["coreDigest"],
    }


def _fixture(tmp_path, monkeypatch):
    ids = [f"support-{i:02d}" for i in range(64)]
    splits = ["train"] * 48 + ["validation"] * 12 + ["test"] * 4
    components = []
    sources = []
    members_by_id = {}
    components_by_id = {}
    for i in range(64):
        source_id, split = ids[i], splits[i]
        member = {"id": source_id, "family": f"family-{i:02d}",
                  "sourceDigest": f"{i + 1:064x}", "coreDigest": f"{i + 65:064x}"}
        component = {"id": f"component-{i:02d}", "split": split,
                     "sourceIds": [source_id], "members": [source_id]}
        sources.append(member)
        components.append(component)
        members_by_id[source_id], components_by_id[source_id] = member, component
    registry = {"schema": "semantic-region-family-registry/1",
                "sources": sources, "components": components}
    registry["digest"] = stable_algorithm_hash(registry)
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "semantic-v8-families.json").write_text(json.dumps(registry))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"schema": "semantic-v8-source-population/1",
        "registryDigest": registry["digest"], "sourceIds": ids}))
    sample_path = tmp_path / "sample.json"
    source_id = ids[0]
    member, component = members_by_id[source_id], components_by_id[source_id]
    identity = {
        "schema": "semantic-region-split-identity/1", "sourceId": source_id,
        "registryDigest": registry["digest"], "registry": registry,
        "componentId": component["id"], "members": component["members"],
        "split": component["split"], "sourceDigest": member["sourceDigest"],
        "coreDigest": member["coreDigest"],
    }
    document = {
        "public": {"schema": "semantic-execution/8", "traceFormat": "semantic-trace/8",
                   "timeline": [{"event": "sample"}]},
        "contexts": [{"visibleThrough": 0, "records": []}],
        "sidecar": {"sourceArtifact": {"reference": {"source": "const x=1", "entry": "x"}},
                    "splitIdentity": identity, "split": component["split"],
                    "sourceComponent": component["id"],
                    "familyRegistryDigest": registry["digest"]},
    }
    document["sidecar"]["datasetBinding"] = _binding(document, identity, member, component)
    sample_path.write_text(json.dumps(document))
    public_artifact = {"public": document["public"], "contexts": document["contexts"]}
    rows = []
    for i, (row_id, split) in enumerate(zip(ids, splits, strict=True)):
        semantic = {"sourceId": row_id, "datasetBinding": {"sourceId": row_id},
                    "publicArtifact": {"public": {}, "contexts": []}}
        if row_id == source_id:
            semantic["datasetBinding"] = document["sidecar"]["datasetBinding"]
            semantic["publicArtifact"] = public_artifact
        rows.append(SimpleNamespace(
            supervision=SimpleNamespace(annotations_json=json.dumps({"semantic_execution": semantic})),
            lineage=SimpleNamespace(frozen_split=split),
        ))
    Corpus.rows = rows
    import megatoken.data
    monkeypatch.setattr(megatoken.data, "PreparedCorpus", Corpus)
    return root, inventory, sample_path, ids, registry, document, identity, member, component


def test_support_lifecycle_binds_sample_to_exact_prepared_train_member(tmp_path, monkeypatch):
    root, inventory, sample, ids, registry, *_ = _fixture(tmp_path, monkeypatch)
    result = MODULE.validate_support_population(root, sample, inventory)
    assert result["registryDigest"] == registry["digest"]
    assert result["sourceCount"] == 64
    assert result["splitCounts"] == {"train": 48, "validation": 12, "test": 4}
    assert result["sampleSourceId"] == ids[0]
    assert result["sampleDatasetBinding"]["sourceId"] == ids[0]


@pytest.mark.parametrize("sample_index", [48, 60])
def test_support_lifecycle_rejects_validation_or_test_sample(tmp_path, monkeypatch, sample_index):
    root, inventory, sample, ids, *_ = _fixture(tmp_path, monkeypatch)
    value = json.loads(sample.read_text())
    value["sidecar"]["splitIdentity"]["sourceId"] = ids[sample_index]
    value["annotations"] = {"sourceId": ids[0]}
    sample.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        MODULE.validate_support_population(root, sample, inventory)


@pytest.mark.parametrize("field", ["public", "contexts", "sourceArtifact"])
def test_support_lifecycle_rejects_rehashed_sample_different_from_prepared(
    tmp_path, monkeypatch, field
):
    root, inventory, sample, _, registry, document, identity, member, component = _fixture(
        tmp_path, monkeypatch
    )
    if field == "public":
        document["public"]["timeline"].append({"event": "altered"})
    elif field == "contexts":
        document["contexts"].append({"visibleThrough": 1, "records": []})
    else:
        document["sidecar"]["sourceArtifact"]["reference"]["source"] = "const x=2"
    document["sidecar"]["datasetBinding"] = _binding(document, identity, member, component)
    sample.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="public/context/source bytes differ"):
        MODULE.validate_support_population(root, sample, inventory)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_support_lifecycle_rejects_missing_and_duplicate_prepared_sources(
    tmp_path, monkeypatch, mutation
):
    root, inventory, sample, *_ = _fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        Corpus.rows.pop()
        expected = "missing an inventory source"
    else:
        Corpus.rows[-1] = Corpus.rows[0]
        expected = "differs from its frozen inventory or split"
    with pytest.raises(ValueError, match=expected):
        MODULE.validate_support_population(root, sample, inventory)
