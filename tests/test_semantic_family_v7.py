"""Independent complete-component split verification against fresh JS output."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.semantic_family_v7 import verify_family_registry_v7, verify_split_identity_v7


@pytest.fixture(scope="module")
def registry():
    producer = Path(__file__).resolve().parents[2] / "tracing-js-self"
    if not (producer / "src/semantic/family-registry.js").is_file():
        pytest.skip("Set up the pinned producer checkout for cross-repository fixtures")
    script = '''
import {buildSemanticFamilyRegistry,semanticSplitIdentity} from './src/semantic/family-registry.js';
const sources=[
 {id:'a',family:'alpha',source:'"use strict";function run(a){return a+1;}',entry:'run'},
 {id:'b',family:'beta',source:'"use strict";function go(value){return value+1;}',entry:'go'},
 {id:'c',family:'gamma',ancestors:['parent'],source:'"use strict";function run(a){return a*2;}',entry:'run'},
 {id:'d',family:'delta',ancestors:['parent'],source:'"use strict";function run(a){return a-2;}',entry:'run'}];
const registry=buildSemanticFamilyRegistry(sources,{seed:17,families:['alpha']});
const identity=semanticSplitIdentity(registry,sources[0]);
console.log(JSON.stringify({registry,identity}));
'''
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=producer,
        text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_python_rebuilds_js_core_alias_and_ancestry_components(registry):
    value = registry["registry"]
    assert verify_family_registry_v7(value) == value["digest"]
    assert len(value["components"]) == 2
    heldout = next(row for row in value["components"] if "a" in row["sourceIds"])
    assert heldout["sourceIds"] == ["a", "b"]
    assert heldout["split"] == "test"
    assert heldout["holdout"] == {"kind": "family", "keys": ["alpha"]}


@pytest.mark.parametrize("mutation", [
    lambda value: value["components"][0].update(split="train" if value["components"][0]["split"] != "train" else "test"),
    lambda value: value["sources"][0].update(source='"use strict";function run(a){return 99;}'),
    lambda value: value["components"][0]["sourceIds"].pop(),
    lambda value: value["holdouts"]["families"].append("unknown-family"),
    lambda value: value["sources"][0].update(extra="ignored-by-forger"),
])
def test_digest_refresh_does_not_authorize_registry_mutation(registry, mutation):
    value = copy.deepcopy(registry["registry"])
    mutation(value)
    value["digest"] = stable_algorithm_hash({key: item for key, item in value.items() if key != "digest"})
    with pytest.raises(ValueError):
        verify_family_registry_v7(value)


def test_bound_source_cannot_move_into_training(registry):
    identity = copy.deepcopy(registry["identity"])
    source = identity["registry"]["sources"][0]
    document = {"public": {"representation": "algorithm"}, "contexts": [], "sidecar": {
        "sourceArtifact": {"sidecar": {"referenceInput": {"source": source["source"], "entry": source["entry"]}, "sourceFamily": "alpha"}},
        "sourceFamily": "alpha", "coreDigest": source["coreDigest"], "splitIdentity": identity,
        "split": identity["split"], "sourceComponent": identity["componentId"], "familyRegistryDigest": identity["registryDigest"]}}
    document["sidecar"]["datasetBinding"] = {"schema": "semantic-dataset-binding/1", "representation": "algorithm",
        "episodeDigest": stable_algorithm_hash(document["public"]), "contextsDigest": stable_algorithm_hash([]),
        "registryDigest": identity["registryDigest"], "componentId": identity["componentId"], "split": identity["split"],
        "sourceId": identity["sourceId"], "sourceDigest": identity["sourceDigest"], "coreDigest": identity["coreDigest"]}
    assert verify_split_identity_v7(document, source_receipt={"registryDigest": identity["registryDigest"]})["split"] == "test"
    document["sidecar"]["split"] = "train"
    with pytest.raises(ValueError, match="binding"):
        verify_split_identity_v7(document)
