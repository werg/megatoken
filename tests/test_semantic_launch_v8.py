"""V8 launch must use actual ordinary supply and exact continuation identities.

Prepared fixtures inject source/tokenizer audit boundaries explicitly; native
replay, framing, export, both split gates and stream identities are real. The
gate-wiring test additionally injects repository/model/donor/checkpoint/coordinator assets.
Neither is a production launch readiness receipt or a model-quality result.
"""

import base64
import copy
import gzip
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from megatoken.data import PreparedCorpus
from megatoken.data.semantic_corpus_v8 import prepare_semantic_corpus_v8
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.io import content_hash, sha256_file
from megatoken.training.semantic_launch import (
    production_configuration,
    qualify_code_revisions,
    qualify_launch,
    read_launch_configuration,
)
from megatoken.training.semantic_launch_v8 import (
    qualify_checkpoint_lineage_v8,
    qualify_coordinator_lineage_v8,
    qualify_corpus_v8,
    qualify_policy_v8,
)
from tests.test_semantic_v8_training_policy import POLICY, prepared_v8_fixture


def production_config(root, policy=POLICY, datasets=None):
    template = Path(__file__).parents[1] / "configs/train/semantic-v8-production.example.yaml"
    value = yaml.safe_load(template.read_text())
    value.update(
        datasets=datasets or [str(root / "unprepared")],
        codec=str(root / "tokenizer.json"),
        output=str(root / "training-output"),
        hindsight_targets=str(root / "hindsight"),
        engram_imports=[str(root / "donor-1"), str(root / "donor-2")],
        semantic_v8_policy=policy,
    )
    value["decoupled"].update(
        run_id="v8-launch-control",
        coordinator_url="https://coordinator.example",
        worker_id=0,
        workers=1,
    )
    return value, production_configuration(value)


def identities(policy):
    return {
        "tracing-js-self": {"revision": policy["producerRevision"], "clean": True},
        "megatoken": {"revision": policy["consumerRevision"], "clean": True},
        "natlang": {"revision": "c" * 40, "clean": True},
    }


@pytest.fixture(scope="module")
def two_split_supply(tmp_path_factory):
    root = tmp_path_factory.mktemp("v8-launch-corpus")
    fixture = Path(__file__).parent / "fixtures/semantic-regions-v8.json.gz.b64"
    examples = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))["examples"]
    registry = json.loads((Path(__file__).parent / "fixtures/current-launch-registry-v8.json").read_text())
    documents, receipts = [], {}
    source_by_id = {source["id"]: source for source in registry["sources"]}
    component_by_source = {
        source_id: component
        for component in registry["components"]
        for source_id in component["sourceIds"]
    }
    for index, split in enumerate(("train", "validation")):
        source_id = f"source-{index}"
        source = source_by_id[source_id]
        component = component_by_source[source_id]
        assert component["split"] == split
        document, codec, _, prepared = prepared_v8_fixture(
            root / split,
            document=copy.deepcopy(examples[index]),
            max_positions=1024,
            max_history=128,
        )
        sidecar = document["sidecar"]
        with PreparedCorpus([str(prepared)], split=None) as corpus:
            (episode,) = corpus.iter_episodes()
            receipt = json.loads(episode.supervision.annotations_json)["semantic_execution"][
                "sourceAdmission"
            ]
        receipts[source_id] = receipt
        identity = sidecar["splitIdentity"]
        identity.update(
            registry=registry,
            registryDigest=registry["digest"],
            seed=registry["seed"],
            sourceId=source_id,
            family=source["family"],
            ancestors=source["ancestors"],
            entry=source["entry"],
            sourceDigest=source["sourceDigest"],
            coreDigest=source["coreDigest"],
            coreStatus=source["coreStatus"],
            componentId=component["id"],
            members=component["members"],
            split=component["split"],
            holdout=component["holdout"],
        )
        sidecar.update(
            split=component["split"],
            sourceComponent=component["id"],
            familyRegistryDigest=registry["digest"],
        )
        sidecar["datasetBinding"].update(
            registryDigest=registry["digest"],
            sourceId=source_id,
            family=source["family"],
            componentId=component["id"],
            split=component["split"],
            sourceDigest=source["sourceDigest"],
            coreDigest=source["coreDigest"],
        )
        receipts[source_id].update(
            registryDigest=registry["digest"],
            datasetBindingDigest=stable_algorithm_hash(sidecar["datasetBinding"]),
        )
        documents.append(document)
    destination = root / "both-splits"
    prepare_semantic_corpus_v8(
        documents,
        destination,
        codec=codec,
        producer_revision=POLICY["producerRevision"],
        consumer_revision=POLICY["consumerRevision"],
        source_verifier=lambda doc: receipts[doc["sidecar"]["datasetBinding"]["sourceId"]],
        tokenizer_evidence={
            "schema": "semantic-tokenizer-admission/1",
            "status": "passed",
            "identity": codec.fingerprint,
            "sha256": "40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60",
            "verification": "pinned-tokenizer-bytes",
        },
        max_positions=1024,
        max_history=128,
    )
    policy = {**POLICY, "registryDigest": registry["digest"]}
    value, config = production_config(root, policy, [str(destination)])
    model = SimpleNamespace(max_history_chunks=128, max_capacity=1024)
    return root, value, config, codec, model, destination


def test_real_production_v8_configuration_has_its_own_launch_authority(tmp_path, monkeypatch):
    value, config = production_config(tmp_path)
    assert config.semantic_experiment is None
    evidence = qualify_policy_v8(config, identities(POLICY))
    assert evidence["schema"] == "semantic-launch-authority/8"
    assert evidence["policy"] == POLICY and evidence["cohortDigest"] is None
    for name in ("tracing-js-self", "megatoken"):
        for change in ({"clean": False}, {"revision": "d" * 40}):
            altered = identities(POLICY)
            altered[name].update(change)
            with pytest.raises(ValueError, match="clean .* checkout"):
                qualify_policy_v8(config, altered)
    with pytest.raises(ValueError, match="production"):
        production_configuration({**value, "purpose": "diagnostic", "max_steps": 1})
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="independent learner"):
        production_configuration(value)


def test_supplied_repository_identities_cannot_stand_in_for_the_running_consumer(
    tmp_path, monkeypatch
):
    import megatoken.training.semantic_launch as launch

    roots = {name: tmp_path / name for name in identities(POLICY)}
    for path in roots.values():
        path.mkdir()
    monkeypatch.setattr(
        launch, "repository_identity", lambda path: {"clean": True, "revision": "a" * 40}
    )
    with pytest.raises(ValueError, match="running megatoken package"):
        qualify_code_revisions(roots)
    with pytest.raises(ValueError, match="distinct checkout"):
        qualify_code_revisions({name: roots["megatoken"] for name in roots})

    # Source-tree runs can use their actual checkout. Wheel-based CI runs must
    # continue to reject an unrelated checkout, since the imported package is
    # not executing from that checkout.
    running_file = Path(launch.__file__).resolve()
    running_package = running_file.parents[1]
    actual_checkout = running_file.parents[3]
    roots["megatoken"] = actual_checkout
    if (actual_checkout / "src/megatoken").resolve() == running_package:
        assert set(qualify_code_revisions(roots)) == set(roots)
    else:
        with pytest.raises(ValueError, match="running megatoken package"):
            qualify_code_revisions(roots)

    wheel_module = (
        tmp_path / "wheel/lib/python3.12/site-packages/megatoken/training/semantic_launch.py"
    )
    wheel_module.parent.mkdir(parents=True)
    monkeypatch.setattr(launch, "__file__", str(wheel_module))
    roots["megatoken"] = tmp_path / "separate-clean-checkout"
    roots["megatoken"].mkdir()
    with pytest.raises(ValueError, match="running megatoken package"):
        qualify_code_revisions(roots)

    # Exercise the clean-checkout success path in either source or wheel CI by
    # placing the imported module at the corresponding synthetic source path.
    simulated_checkout = tmp_path / "running-consumer"
    simulated_module = simulated_checkout / "src/megatoken/training/semantic_launch.py"
    simulated_module.parent.mkdir(parents=True)
    monkeypatch.setattr(launch, "__file__", str(simulated_module))
    roots["megatoken"] = simulated_checkout
    assert set(qualify_code_revisions(roots)) == set(roots)
    monkeypatch.setattr(launch, "repository_identity", lambda path: {"clean": False})
    with pytest.raises(ValueError, match="committed, clean"):
        qualify_code_revisions(roots)


@pytest.mark.parametrize(
    "suffix, text", [(".json", '{"phase":"SC","phase":"SE"}'), (".yaml", "phase: SC\nphase: SE\n")]
)
def test_launch_configuration_rejects_duplicate_keys(tmp_path, suffix, text):
    path = tmp_path / ("config" + suffix)
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate"):
        read_launch_configuration(path)
    report = qualify_launch(path, repositories={}, require_cuda=False)
    assert report["gates"]["configurationFile"]["status"] == "blocked"
    assert not report["preflightReady"] and report["optimizerSteps"] == 0


def test_v8_launch_retains_missing_assets_without_loading_legacy(tmp_path):
    value, _ = production_config(tmp_path)
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(value))
    script = r"""
import importlib.abc, json, sys
blocked = {
    "megatoken.data.semantic_corpus", "megatoken.data.semantic_contract",
    "megatoken.data.semantic_admission", "megatoken.data.witness_v6",
    "megatoken.data.semantic_harness_v6", "megatoken.training.semantic_budget",
    "megatoken.data.algorithm_v7", "megatoken.data.semantic_contract_v7",
}
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("legacy wire import: " + fullname)
sys.meta_path.insert(0, NoLegacy())
from megatoken.training.semantic_launch import qualify_launch
report = qualify_launch(sys.argv[1], repositories={}, require_cuda=False)
assert not blocked.intersection(sys.modules)
assert all("legacy wire" not in gate.get("reason", "") for gate in report["gates"].values())
print(json.dumps(report))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        env=os.environ,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["gates"]["configuration"]["status"] == "passed"
    assert report["gates"]["engineering"]["status"] == "blocked"
    assert report["gates"]["coordinator"]["status"] == "blocked"
    assert not report["preflightReady"] and not Path(value["output"]).exists()
    assert report["learnerStarted"] is False and report["coordinatorMutations"] == 0
    digest = report.pop("digest")
    assert digest == content_hash(report)


@pytest.mark.parametrize("overlap", ["same", "child", "parent", "symlink-child"])
def test_launch_output_cannot_overlap_raw_tokenizer_asset(tmp_path, overlap):
    value, _ = production_config(tmp_path)
    codec = tmp_path / "immutable-tokenizer"
    codec.mkdir()
    value["codec"] = str(codec)
    if overlap == "same":
        output = codec
    elif overlap == "child":
        output = codec / "runs" / "training"
    elif overlap == "parent":
        output = tmp_path
    else:
        alias = tmp_path / "tokenizer-alias"
        alias.symlink_to(codec, target_is_directory=True)
        output = alias / "training"
    value["output"] = str(output)
    config_path = tmp_path / f"launch-{overlap}.yaml"
    config_path.write_text(yaml.safe_dump(value))

    report = qualify_launch(config_path, repositories={}, require_cuda=False)

    gate = report["gates"]["outputLocation"]
    assert gate["status"] == "blocked"
    assert gate["reason"] == "Training output overlaps immutable input assets"
    assert report["optimizerSteps"] == 0 and report["learnerStarted"] is False
    if overlap == "child":
        assert not output.exists()


def test_actual_two_split_supply_derives_the_ordinary_stream_without_writes(two_split_supply):
    from megatoken.data.recipes import stage_recipe
    from megatoken.training.stream import MixtureWindowStream
    from megatoken.training.windows import window_config

    _, _, config, codec, model, destination = two_split_supply
    before = {str(path): sha256_file(path) for path in destination.rglob("*") if path.is_file()}
    report = qualify_corpus_v8(config, codec, model)
    ordinary_stream = MixtureWindowStream(
        config.datasets,
        codec,
        window_config(config),
        stage_recipe(config.phase).mixture,
        config.schedule.meter,
        seed=config.seed,
        rank=config.decoupled.resolved_worker_id,
        world_size=config.decoupled.workers,
        reduce_counts=lambda counts: counts,
        independent=True,
    )
    try:
        assert report["trainingLineage"]["streamDigest"] == ordinary_stream.fingerprint
    finally:
        ordinary_stream.close()
    assert report["episodes"] == {"train": 1, "validation": 1, "test": 0}
    assert report["coverage"] == {"train": "passed", "validation": "passed"}
    assert report["trainingLineage"]["preparedCorpusDigest"] == report["corpusIdentity"]
    changed = qualify_corpus_v8(replace(config, max_window_records=64), codec, model)
    assert changed["trainingLineage"]["streamDigest"] != report["trainingLineage"]["streamDigest"]
    assert before == {
        str(path): sha256_file(path) for path in destination.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="at least one episode per worker"):
        qualify_corpus_v8(
            replace(config, decoupled=replace(config.decoupled, workers=8)), codec, model
        )
    with pytest.raises(
        ValueError,
        match="semantic_(tool_feedback|async|callback|repeated_invocation|child|source_reread|recovery)",
    ):
        qualify_corpus_v8(replace(config, phase="SE"), codec, model)


def test_source_admitted_training_rows_do_not_replace_missing_validation(tmp_path):
    _, codec, policy, root = prepared_v8_fixture(tmp_path, max_positions=1024, max_history=128)
    _, config = production_config(tmp_path, policy, [str(root)])
    with pytest.raises(ValueError, match="no validation episodes"):
        qualify_corpus_v8(config, codec, SimpleNamespace(max_history_chunks=128, max_capacity=1024))


def test_empty_prepared_roots_are_bound_same_as_ordinary_training(tmp_path, two_split_supply):
    from megatoken.data.chunking import ChunkingConfig
    from megatoken.data.dedup import DedupStore
    from megatoken.data.storage import export_prepared

    _, _, config, codec, model, prepared = two_split_supply
    manifest = json.loads((prepared / "manifest.json").read_text())
    chunking = ChunkingConfig.from_identity(manifest["chunking"], codec)
    empty = tmp_path / "empty-root"
    store = DedupStore(tmp_path / "empty.sqlite")
    try:
        store.freeze()
        export_prepared(
            store,
            empty,
            codec=codec,
            source_receipts=[],
            pipeline_fingerprint="7" * 64,
            chunking=chunking,
        )
    finally:
        store.close()
    report = qualify_corpus_v8(
        replace(config, datasets=[*config.datasets, str(empty)]), codec, model
    )
    fingerprints = sorted(
        (manifest["fingerprint"], json.loads((empty / "manifest.json").read_text())["fingerprint"])
    )
    assert report["trainingLineage"]["preparedManifestFingerprints"] == fingerprints
    assert report["corpusIdentity"] == content_hash(fingerprints)
    assert report["episodes"] == {"train": 1, "validation": 1, "test": 0}


def test_actual_v8_launch_corpus_and_stream_run_with_legacy_clients_absent(tmp_path):
    script = r"""
import importlib.abc, json, sys
from pathlib import Path
blocked = {
    "megatoken.data.semantic_corpus", "megatoken.data.semantic_contract",
    "megatoken.data.semantic_admission", "megatoken.data.witness_v6",
    "megatoken.data.semantic_harness_v6", "megatoken.training.semantic_budget",
    "megatoken.data.algorithm_v7", "megatoken.data.semantic_contract_v7",
    "megatoken.training.semantic_v7",
}
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("legacy wire import: " + fullname)
sys.meta_path.insert(0, NoLegacy())
from tests.test_semantic_launch_v8 import two_split_supply, test_empty_prepared_roots_are_bound_same_as_ordinary_training
from megatoken.training.semantic_launch_v8 import qualify_corpus_v8
root = Path(sys.argv[1])
class Factory:
    def mktemp(self, name):
        destination = root / name
        destination.mkdir()
        return destination
supply = two_split_supply.__wrapped__(Factory())
_, _, config, codec, geometry, _ = supply
report = qualify_corpus_v8(config, codec, geometry)
test_empty_prepared_roots_are_bound_same_as_ordinary_training(root, supply)
assert not blocked.intersection(sys.modules)
print(json.dumps({"episodes":report["episodes"],"blocked":len(blocked),"optimizerSteps":report["optimizerSteps"]}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=os.environ,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "episodes": {"train": 1, "validation": 1, "test": 0},
        "blocked": 9,
        "optimizerSteps": 0,
    }


def test_resume_and_coordinator_require_the_actual_corpus_and_stream(two_split_supply):
    _, _, config, codec, model, _ = two_split_supply
    supply = qualify_corpus_v8(config, codec, model)
    lineage = supply["trainingLineage"]
    payload = {
        "train_config": {"phase": config.phase, "semantic_v8_policy": config.semantic_v8_policy},
        "codec_fingerprint": codec.fingerprint,
        "chunking": {"fingerprint": lineage["chunkingFingerprint"]},
        "semantic_training_v8": lineage,
    }
    metadata = {
        "dataset_fingerprint": lineage["streamDigest"],
        "semantic_v8_lineage_digest": lineage["digest"],
    }
    assert qualify_checkpoint_lineage_v8(payload, metadata, config=config, supply=supply) == lineage
    changed = copy.deepcopy(payload)
    changed["semantic_training_v8"]["streamDigest"] = "9" * 64
    changed_lineage = changed["semantic_training_v8"]
    changed_lineage["digest"] = stable_algorithm_hash(
        {k: v for k, v in changed_lineage.items() if k != "digest"}
    )
    changed_metadata = {
        "dataset_fingerprint": changed_lineage["streamDigest"],
        "semantic_v8_lineage_digest": changed_lineage["digest"],
    }
    with pytest.raises(ValueError, match="freshly qualified"):
        qualify_checkpoint_lineage_v8(changed, changed_metadata, config=config, supply=supply)
    with pytest.raises(ValueError, match="legacy semantic experiment"):
        qualify_checkpoint_lineage_v8(
            {**payload, "semantic_exposure": {}}, metadata, config=config, supply=supply
        )
    prior_stage = {"train_config": {"phase": "S"}}
    assert (
        qualify_checkpoint_lineage_v8(
            prior_stage, {}, config=config, supply=supply, coordinator_bootstrap=True
        )
        is None
    )
    with pytest.raises(ValueError, match="freshly qualified"):
        qualify_checkpoint_lineage_v8(prior_stage, {}, config=config, supply=supply)
    manifest = {
        "semantic_v8_policy": config.semantic_v8_policy,
        "semantic_v8_lineage": lineage,
        "chunking_identity": {"fingerprint": lineage["chunkingFingerprint"]},
    }
    arguments = dict(
        config=config, codec=codec, corpus_identity=supply["corpusIdentity"], lineage=lineage
    )
    assert qualify_coordinator_lineage_v8(manifest, **arguments) == lineage["digest"]
    for key, value in (
        ("semantic_v8_policy", {**POLICY, "registryDigest": "8" * 64}),
        ("semantic_v8_lineage", changed_lineage),
        ("chunking_identity", {}),
    ):
        with pytest.raises(ValueError, match="V8"):
            qualify_coordinator_lineage_v8({**manifest, key: value}, **arguments)


def test_coordinator_receives_verified_checkpoint_and_bootstrap_gate_results(
    tmp_path, two_split_supply, monkeypatch
):
    import megatoken.data.recipes as recipes
    import megatoken.training.semantic_launch as launch
    import megatoken.training.semantic_v8 as semantic_v8

    _, _, original, codec, geometry, destination = two_split_supply
    value, config = production_config(tmp_path, original.semantic_v8_policy, [str(destination)])
    for donor in config.engram_imports:
        Path(donor).mkdir()
        for name in ("engram.json", "descriptors.npz"):
            (Path(donor) / name).write_bytes(b"injected asset boundary")
    config_path = tmp_path / "launch.json"
    config_path.write_text(json.dumps(value))
    loading = ModuleType("megatoken.loading")
    loading.load_model_config = lambda *args, **kwargs: SimpleNamespace(
        **vars(geometry),
        vocab_size=codec.vocab_size,
        to_dict=lambda: {"vocab_size": codec.vocab_size},
    )
    assets = ModuleType("megatoken.assets")
    assets.load_engram_asset = lambda *args: None
    runtime = ModuleType("megatoken.runtime")
    runtime.doctor = lambda **kwargs: {"ok": True, "errors": [], "devices": []}
    for name, module in (
        ("megatoken.loading", loading),
        ("megatoken.assets", assets),
        ("megatoken.runtime", runtime),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(recipes, "validate_training_donors", lambda config: None)
    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", lambda *args, **kwargs: codec)
    monkeypatch.setattr(
        launch, "qualify_code_revisions", lambda repositories: identities(config.semantic_v8_policy)
    )
    monkeypatch.setattr(launch, "qualify_hindsight", lambda *args: {"injected": True})
    checkpoint = {
        "phase": "S",
        "parameterIdentity": "verified-parameters",
        "bootstrapStateSha256": "verified-state",
        "metadata": {"sha256": "verified-metadata"},
    }
    monkeypatch.setattr(launch, "qualify_checkpoint", lambda *args, **kwargs: checkpoint)
    receipt = tmp_path / "bootstrap.json"
    receipt.write_text('{"source":"verified-bootstrap"}')
    observed = {}

    def coordinator(*args, **kwargs):
        observed.update(kwargs)
        return {"injectedProvider": True}

    monkeypatch.setattr(launch, "qualify_coordinator", coordinator)
    report = qualify_launch(
        config_path,
        repositories={name: name for name in identities(POLICY)},
        bootstrap_checkpoint=tmp_path / "checkpoint",
        bootstrap_receipt=receipt,
        probe_coordinator=True,
        require_cuda=False,
    )
    assert report["preflightReady"], report["gates"]
    assert observed["expected_parameter_identity"] == checkpoint["parameterIdentity"]
    assert observed["bootstrap_state_sha256"] == checkpoint["bootstrapStateSha256"]
    assert observed["bootstrap_metadata_sha256"] == checkpoint["metadata"]["sha256"]
    assert observed["bootstrap_receipt"] == {"source": "verified-bootstrap"}
    assert (
        observed["semantic_v8_lineage"] == report["gates"]["corpus"]["evidence"]["trainingLineage"]
    )
    assert not Path(config.output).exists()


def test_actual_model_checkpoint_keeps_v8_lineage_and_rejects_rebound_stream(tmp_path):
    """Real tensor/checksum validation, with explicitly synthetic source lineage."""
    from dataclasses import asdict

    from training_fixture import tiny_train_config

    from megatoken.generation.codec import ByteCodec
    from megatoken.loading import build_model, load_model_config
    from megatoken.training.checkpoint import save_checkpoint, transducer_attachment
    from megatoken.training.semantic_launch import qualify_checkpoint
    from tests.test_semantic_v8_lineage import fixture

    codec = ByteCodec.byte_fallback()
    lineage, binding, metadata = fixture(codec)
    config = replace(
        tiny_train_config(tmp_path / "data"),
        phase="SC",
        codec="injected-tokenizer",
        semantic_v8_policy=binding["train_config"]["semantic_v8_policy"],
    )
    model_config = load_model_config(
        config.model,
        expert_backend=config.expert_backend,
        checkpoint_stages=config.gradient_checkpointing,
    )
    model = build_model(model_config, codec)
    payload = {
        **binding,
        "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
        "optimizer": {
            "format_version": 3,
            "master_weights": {
                name: parameter.detach().float().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
        },
        "model_config": model_config.to_dict(),
        "train_config": asdict(config),
        "rank_state": [{"cursor": {}, "rng": {}, "autoencode": None}],
        "global_step": 0,
        "phase_consumed": 0,
        "counters": {},
    }
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints",
        "v8",
        payload,
        metadata,
        attachments=transducer_attachment(lambda: {"bank": {}}),
    )
    files = {str(path): sha256_file(path) for path in checkpoint.rglob("*") if path.is_file()}
    report = qualify_checkpoint(
        checkpoint,
        config=config,
        codec=codec,
        cohort_digest=None,
        v8_supply={"trainingLineage": lineage},
    )
    assert report["semanticV8Lineage"] == lineage
    changed = {**lineage, "streamDigest": "7" * 64}
    with pytest.raises(ValueError, match="freshly qualified"):
        qualify_checkpoint(
            checkpoint,
            config=config,
            codec=codec,
            cohort_digest=None,
            v8_supply={"trainingLineage": changed},
        )
    with pytest.raises(ValueError, match="freshly qualified"):
        qualify_checkpoint(checkpoint, config=config, codec=codec, cohort_digest=None)
    assert files == {
        str(path): sha256_file(path) for path in checkpoint.rglob("*") if path.is_file()
    }
