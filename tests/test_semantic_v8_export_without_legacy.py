"""Actual V8 preparation/export must stay independent of old wire clients.

The existing preparation helper injects only source/tokenizer audit boundaries;
native replay, physical records, masks, deduplication, shard export and reload
are real. This is not a production-tokenizer or fresh source-admission claim.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fixture",
    [
        "semantic-regions-v8.json.gz.b64",
        "semantic-protocol-region-v8.json.gz.b64",
        "semantic-callback-regions-v8.json.gz.b64",
    ],
    ids=["ordinary", "protocol", "callback"],
)
def test_actual_v8_preparation_exports_identical_shards_without_legacy(tmp_path, fixture):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        r"""
        import base64
        import gzip
        import hashlib
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        blocked = {
            "megatoken.data.algorithm_v7",
            "megatoken.data.detailed_v7",
            "megatoken.data.detailed_harness_v7",
            "megatoken.data.detailed_context_v7",
            "megatoken.data.semantic_harness_v6",
            "megatoken.data.semantic_contract",
            "megatoken.data.semantic_contract_v7",
            "megatoken.data.witness_v6",
            "megatoken.data.semantic_admission",
            "megatoken.data.semantic_corpus",
            "megatoken.training.semantic_budget",
        }
        class NoLegacyWire(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise AssertionError("Legacy wire dependency: " + fullname)

        destination, fixture, forbid_legacy = sys.argv[1:]
        if forbid_legacy == "1":
            sys.meta_path.insert(0, NoLegacyWire())

        from megatoken.data.storage import PreparedCorpus
        from tests.test_semantic_v8_training_policy import prepared_v8_fixture

        raw = (Path("tests/fixtures") / fixture).read_bytes()
        document = json.loads(gzip.decompress(base64.b64decode(raw)))
        row = document.get("examples", document.get("cases", [document]))[0]
        document = row.get("artifact", row)
        original_public = json.dumps(document["public"], sort_keys=True)
        document, codec, policy, prepared = prepared_v8_fixture(
            Path(destination), max_positions=1024, max_history=128, document=document,
        )
        assert json.dumps(document["public"], sort_keys=True) == original_public
        manifest = json.loads((prepared / "manifest.json").read_text())
        with PreparedCorpus([str(prepared)], split="train") as corpus:
            (episode,) = corpus.iter_episodes()
            semantic = json.loads(episode.supervision.annotations_json)["semantic_execution"]
            assert semantic["publicArtifact"] == {
                "public": document["public"], "contexts": document["contexts"],
            }
            # Original selected bytes and complete logical order survive physical
            # framing, deduplication, checksummed export and ordinary reload.
            targets = [
                b"".join(part.payload for part in parts)
                for parts in episode.iter_logical_records() if parts[0].policy_mask
            ]
            assert targets == [
                row["fields"]["line"].encode("utf-8")
                for row in document["public"]["timeline"] if row["mode"] == "generated"
            ]
            episode_hash = hashlib.sha256(
                json.dumps(episode.to_dict(), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        coverage = manifest["behavior_coverage"]
        for name in ("semantic_v8", "semantic_code_invocation"):
            scope = "train|semantic_execution|" + name
            assert coverage["episodes"][scope] == 1
            assert coverage["original_families"][scope] == 1
            assert len(coverage["original_family_ids"][scope]) == 1
        if "callback" in fixture:
            assert coverage["episodes"]["train|semantic_execution|semantic_callback"] == 1
        retained = ["manifest.json", manifest["index"]["path"]]
        retained.extend(item["path"] for item in manifest["shards"])
        loaded = sorted(blocked.intersection(sys.modules))
        if forbid_legacy == "1":
            assert not loaded
        print(json.dumps({
            "files": {name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
                      for name in retained},
            "episodeHash": episode_hash,
            "targets": len(targets),
            "coverage": coverage,
            "legacyModulesLoaded": loaded,
        }))
        """
    )
    results = []
    for forbid_legacy in (False, True):
        destination = tmp_path / ("absent" if forbid_legacy else "available")
        result = subprocess.run(
            [sys.executable, "-c", script, str(destination), fixture, str(int(forbid_legacy))],
            cwd=root,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
        results.append(json.loads(result.stdout))
    assert not results[1].pop("legacyModulesLoaded")
    results[0].pop("legacyModulesLoaded")
    assert results[0] == results[1]
    assert results[1]["targets"] > 0
