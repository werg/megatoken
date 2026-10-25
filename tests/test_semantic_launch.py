"""Nonoptimizing preflight: corrupt/missing inputs cannot become launch readiness."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from training_fixture import tiny_train_config

from megatoken.assets import load_engram_asset
from megatoken.data import PreparedCorpus
from megatoken.generation.codec import ByteCodec
from megatoken.io import content_hash, sha256_file
from megatoken.loading import build_model, load_model_config
from megatoken.training.checkpoint import save_checkpoint, transducer_attachment
from megatoken.training.semantic_launch import (
    qualify_checkpoint,
    qualify_coordinator,
    qualify_hindsight,
    qualify_launch,
    unresolved_bindings,
)


def test_placeholder_plan_is_not_ready_and_never_creates_training_output(tmp_path):
    source = Path(__file__).parents[1] / "configs/semantic/train-seed-17.json"
    config = json.loads(source.read_text())
    config["output"] = str(tmp_path / "must-not-exist")
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(config))
    report = qualify_launch(path, repositories={}, require_cuda=False)
    assert report["schema"] == "semantic-launch-qualification/1"
    assert not report["preflightReady"]
    assert report["gates"]["configuration"]["status"] == "passed"
    assert report["gates"]["resolvedBindings"]["status"] == "blocked"
    assert report["gates"]["coordinator"]["status"] == "blocked"
    assert report["optimizerSteps"] == report["coordinatorMutations"] == 0
    assert report["learnerStarted"] is False and report["learnedSuccess"] is None
    assert report["trainingMemoryFit"] == report["recoveryUnderLoad"] == "unmeasured"
    assert not Path(config["output"]).exists()
    digest = report.pop("digest")
    assert digest == content_hash(report)


def test_unresolved_bindings_reveal_paths_not_secret_values():
    assert unresolved_bindings(
        {"coordinator": "https://bind-me.invalid", "nested": ["UNBOUND_ASSET"]}
    ) == [
        "config.coordinator",
        "config.nested[0]",
    ]


def test_invalid_json_still_produces_blocked_report(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"phase":"SC","phase":"SE"}')
    report = qualify_launch(path, repositories={})
    assert not report["preflightReady"]
    assert "duplicate" in report["gates"]["configurationFile"]["reason"]


def test_donor_validator_reads_actual_tensor_and_rejects_bad_geometry(tmp_path):
    model = load_model_config("tiny")
    destination = tmp_path / "synthetic-donor"
    destination.mkdir()
    file = destination / "descriptors.npz"
    np.savez(
        file,
        rows=np.array([7]),
        descriptors=np.ones((1, 128), dtype=np.float32),
        coverage=np.ones(1),
        support=np.ones(1),
    )
    metadata = {
        "format_version": 1,
        "exact_rows": model.exact_rows,
        "ngram_rows": model.ngram_rows,
        "descriptor_sha256": sha256_file(file),
    }
    (destination / "engram.json").write_text(json.dumps(metadata))
    _, tensors = load_engram_asset(model, destination)
    assert tensors["descriptors"].shape == (1, 128)
    # A self-consistent hash alone must not admit malformed descriptors.
    np.savez(
        file,
        rows=np.array([7]),
        descriptors=np.ones((1, 127)),
        coverage=np.ones(1),
        support=np.ones(1),
    )
    metadata["descriptor_sha256"] = sha256_file(file)
    (destination / "engram.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="descriptors"):
        load_engram_asset(model, destination)


def test_hindsight_qualification_reads_every_actual_window_and_tensor(tmp_path):
    from test_teacher_graph import prepared

    config, targets, _ = prepared(tmp_path)
    targets.close()
    model = load_model_config(config.model, **config.model_overrides)
    with PreparedCorpus(config.datasets) as corpus:
        identity = corpus.identity
    report = qualify_hindsight(config, ByteCodec.byte_fallback(), model, identity)
    assert report["windowsRead"] > 0
    encoding = Path(config.hindsight_targets) / "encoding.pt"
    encoding.write_bytes(encoding.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="encoding file"):
        qualify_hindsight(config, ByteCodec.byte_fallback(), model, identity)


def test_checkpoint_qualification_never_repairs_or_accepts_inference_only_state(tmp_path):
    config = tiny_train_config(tmp_path / "data")
    directory = save_checkpoint(
        tmp_path / "checkpoints",
        "inference-only",
        {"model": {"x": torch.ones(1)}},
        {},
        attachments=transducer_attachment(lambda: {"bank": {}}),
    )
    with pytest.raises(ValueError, match="complete current training checkpoint"):
        qualify_checkpoint(
            directory, config=config, codec=ByteCodec.byte_fallback(), cohort_digest=None
        )
    aside = directory.parent / ".superseded-inference-only-1"
    aside.mkdir()
    marker = aside / "retain"
    marker.write_text("do not repair")
    with pytest.raises(ValueError, match="requires recovery"):
        qualify_checkpoint(
            directory, config=config, codec=ByteCodec.byte_fallback(), cohort_digest=None
        )
    assert marker.read_text() == "do not repair"


def test_checkpoint_complete_fields_and_checksum_are_checked_without_install(tmp_path):
    config = tiny_train_config(tmp_path / "data")
    model_config = load_model_config(
        "tiny",
        expert_backend=config.expert_backend,
        checkpoint_stages=config.gradient_checkpointing,
    )
    model = build_model(model_config, ByteCodec.byte_fallback())
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    masters = {
        name: parameter.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "model": state,
        "optimizer": {"format_version": 3, "master_weights": masters},
        "model_config": model_config.to_dict(),
        "train_config": asdict(config),
        "codec_fingerprint": ByteCodec.byte_fallback().fingerprint,
        "rank_state": [{"cursor": {}, "rng": {}, "autoencode": None}],
        "global_step": 0,
        "phase_consumed": 0,
        "counters": {},
        "chunking": {},
    }
    directory = save_checkpoint(
        tmp_path / "checkpoints",
        "synthetic-fields",
        payload,
        {},
        attachments=transducer_attachment(lambda: {"bank": {}}),
    )
    report = qualify_checkpoint(
        directory, config=config, codec=ByteCodec.byte_fallback(), cohort_digest=None
    )
    assert report["globalStep"] == 0 and "no state installed" in report["scope"]
    with (directory / "state.pt").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        qualify_checkpoint(
            directory, config=config, codec=ByteCodec.byte_fallback(), cohort_digest=None
        )


def test_checkpoint_model_shape_and_finiteness_are_verified(tmp_path):
    config = tiny_train_config(tmp_path / "data")
    model_config = load_model_config(
        "tiny",
        expert_backend=config.expert_backend,
        checkpoint_stages=config.gradient_checkpointing,
    )
    model = build_model(model_config, ByteCodec.byte_fallback())
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    masters = {
        name: parameter.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "model": state,
        "optimizer": {"format_version": 3, "master_weights": masters},
        "model_config": model_config.to_dict(),
        "train_config": asdict(config),
        "codec_fingerprint": ByteCodec.byte_fallback().fingerprint,
        "rank_state": [{"cursor": {}, "rng": {}, "autoencode": None}],
        "global_step": 0,
        "phase_consumed": 0,
        "counters": {},
        "chunking": {},
    }

    def publish(name, broken):
        return save_checkpoint(
            tmp_path / name,
            "checkpoint",
            broken,
            {},
            attachments=transducer_attachment(lambda: {"bank": {}}),
        )

    valid = publish("valid", payload)
    report = qualify_checkpoint(
        valid, config=config, codec=ByteCodec.byte_fallback(), cohort_digest=None
    )
    assert report["parameterIdentity"]

    omitted = dict(payload)
    omitted["optimizer"] = dict(payload["optimizer"])
    omitted["optimizer"]["master_weights"] = dict(masters)
    omitted["optimizer"]["master_weights"].pop(next(iter(masters)))
    with pytest.raises(ValueError, match="exactly match source model optimizer owners"):
        qualify_checkpoint(
            publish("omitted-master", omitted),
            config=config,
            codec=ByteCodec.byte_fallback(),
            cohort_digest=None,
        )

    missing = dict(payload)
    missing["model"] = dict(payload["model"])
    missing["model"].pop(next(iter(missing["model"])))
    with pytest.raises(ValueError, match="model keys or tensor shapes"):
        qualify_checkpoint(
            publish("missing-key", missing),
            config=config,
            codec=ByteCodec.byte_fallback(),
            cohort_digest=None,
        )

    wrong_shape = dict(payload)
    wrong_shape["model"] = dict(payload["model"])
    key = next(iter(wrong_shape["model"]))
    wrong_shape["model"][key] = torch.zeros((1,), dtype=payload["model"][key].dtype)
    with pytest.raises(ValueError, match="model keys or tensor shapes"):
        qualify_checkpoint(
            publish("wrong-shape", wrong_shape),
            config=config,
            codec=ByteCodec.byte_fallback(),
            cohort_digest=None,
        )

    nonfinite = dict(payload)
    nonfinite["model"] = dict(payload["model"])
    key = next(key for key, value in nonfinite["model"].items() if value.is_floating_point())
    nonfinite["model"][key] = nonfinite["model"][key].clone()
    nonfinite["model"][key].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        qualify_checkpoint(
            publish("nonfinite", nonfinite),
            config=config,
            codec=ByteCodec.byte_fallback(),
            cohort_digest=None,
        )


def test_bootstrap_phase_transition_derives_added_donor_owner_layout(tmp_path):
    from dataclasses import replace

    from megatoken.training.diloco_protocol import parameter_identity
    from megatoken.training.semantic_launch import _validate_model_and_masters

    config = tiny_train_config(tmp_path / "data")
    model_config = load_model_config(
        "tiny",
        expert_backend=config.expert_backend,
        checkpoint_stages=config.gradient_checkpointing,
    )
    from megatoken.assets import install_engram_asset

    def donor(name, row):
        donor_dir = tmp_path / name
        donor_dir.mkdir()
        descriptors_path = donor_dir / "descriptors.npz"
        np.savez(
            descriptors_path,
            rows=np.array([row]),
            descriptors=np.ones((1, 128), dtype=np.float32),
            coverage=np.ones(1, dtype=np.float32),
            support=np.ones(1, dtype=np.float32),
        )
        (donor_dir / "engram.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "exact_rows": model_config.exact_rows,
                    "ngram_rows": model_config.ngram_rows,
                    "descriptor_sha256": sha256_file(descriptors_path),
                }
            )
        )
        return donor_dir

    first_donor, second_donor = donor("donor-one", 7), donor("donor-two", 8)
    destination_config = replace(config, engram_imports=[str(first_donor), str(second_donor)])

    # A transition from no imports installs one root map and then branch zero.
    source = build_model(model_config, ByteCodec.byte_fallback())
    no_import_payload = {
        "model": {name: value.detach().clone() for name, value in source.state_dict().items()},
        "optimizer": {
            "master_weights": {
                name: parameter.detach().float().clone()
                for name, parameter in source.named_parameters()
                if parameter.requires_grad
            }
        },
    }
    no_import_identity = _validate_model_and_masters(
        no_import_payload,
        model_config,
        ByteCodec.byte_fallback(),
        destination_config,
        coordinator_bootstrap=True,
    )
    expected_two_donors = build_model(model_config, ByteCodec.byte_fallback())
    install_engram_asset(expected_two_donors, first_donor)
    install_engram_asset(expected_two_donors, second_donor)
    assert no_import_identity == parameter_identity(
        {name: p for name, p in expected_two_donors.named_parameters() if p.requires_grad}
    )

    # A transition from one restored import appends exactly import_branches.0.
    source_with_one = build_model(model_config, ByteCodec.byte_fallback())
    install_engram_asset(source_with_one, first_donor)
    one_import_payload = {
        "model": {
            name: value.detach().clone() for name, value in source_with_one.state_dict().items()
        },
        "optimizer": {
            "master_weights": {
                name: parameter.detach().float().clone()
                for name, parameter in source_with_one.named_parameters()
                if parameter.requires_grad
            }
        },
    }
    one_import_identity = _validate_model_and_masters(
        one_import_payload,
        model_config,
        ByteCodec.byte_fallback(),
        destination_config,
        coordinator_bootstrap=True,
    )
    assert one_import_identity == parameter_identity(
        {name: p for name, p in expected_two_donors.named_parameters() if p.requires_grad}
    )
    with pytest.raises(ValueError, match="Exact worker resume cannot add donor owners"):
        _validate_model_and_masters(
            one_import_payload,
            model_config,
            ByteCodec.byte_fallback(),
            destination_config,
            coordinator_bootstrap=False,
        )


def test_coordinator_export_is_bootstrap_input_but_cannot_be_worker_resume(tmp_path):
    config = tiny_train_config(tmp_path / "data")
    model = load_model_config(
        "tiny",
        expert_backend=config.expert_backend,
        checkpoint_stages=config.gradient_checkpointing,
    )
    actual = build_model(model, ByteCodec.byte_fallback())
    model_state = {name: value.detach().clone() for name, value in actual.state_dict().items()}
    masters = {
        name: parameter.detach().float().clone()
        for name, parameter in actual.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "export_only": True,
        "master_weights": masters,
        "model": model_state,
        "model_config": model.to_dict(),
        "train_config": asdict(config),
        "global_step": 0,
        "codec_fingerprint": ByteCodec.byte_fallback().fingerprint,
    }
    path = save_checkpoint(
        tmp_path / "exports",
        "synthetic-export",
        payload,
        {},
        attachments=transducer_attachment(lambda: {"bank": {}}),
    )
    report = qualify_checkpoint(
        path,
        config=config,
        codec=ByteCodec.byte_fallback(),
        cohort_digest=None,
        coordinator_bootstrap=True,
    )
    assert "exact FP32 master layout" in report["scope"]
    assert report["parameterIdentity"]
    with pytest.raises(ValueError, match="complete current training checkpoint"):
        qualify_checkpoint(path, config=config, codec=ByteCodec.byte_fallback(), cohort_digest=None)


def test_coordinator_qualification_uses_only_get_and_checks_actual_run(tmp_path, monkeypatch):
    from dataclasses import replace

    from megatoken.training import diloco_client
    from megatoken.training.config import DecoupledConfig
    from megatoken.training.diloco_protocol import training_identity

    config = replace(
        tiny_train_config(tmp_path),
        decoupled=DecoupledConfig(
            "http://127.0.0.1:1", "qualification-test", workers=1, worker_id=0
        ),
    )
    model, codec = load_model_config("tiny"), ByteCodec.byte_fallback()
    actual = build_model(model, codec)
    masters = {
        name: parameter.detach().float().clone()
        for name, parameter in actual.named_parameters()
        if parameter.requires_grad
    }
    from megatoken.training.diloco_protocol import parameter_identity

    expected_identity = parameter_identity(masters)
    manifest = {
        "run_id": config.decoupled.run_id,
        "workers": 1,
        "model_config": model.to_dict(),
        "codec_fingerprint": codec.fingerprint,
        "token_budget": config.schedule.budget,
        "train_identity": training_identity(config),
        "dataset_fingerprint": "test-corpus",
        "parameter_identity": expected_identity,
        "fragments": [{"index": 0, "slices": [{"name": "test", "offset": 0, "length": 1}]}],
    }
    calls = []
    bootstrap_calls = []
    bootstrap_bytes = b"exact initialized coordinator bootstrap"

    def fetch_file(self, path, directory, *, json_body=None):
        bootstrap_calls.append((path, json_body))
        downloaded = Path(directory) / "bootstrap.download"
        downloaded.write_bytes(bootstrap_bytes)
        return downloaded, None

    def response(self, method, path, value=None):
        calls.append((method, path, value))
        return (
            manifest
            if path == "/v1/manifest"
            else {"run_id": config.decoupled.run_id, "finished": False, "versions": [0]}
        )

    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", "test-credential-with-enough-length")
    monkeypatch.setattr(diloco_client.Transport, "json", response)
    monkeypatch.setattr(diloco_client.Transport, "fetch_file", fetch_file)
    result = qualify_coordinator(
        config,
        model,
        codec,
        "test-corpus",
        expected_parameter_identity=expected_identity,
    )
    assert result["fragments"] == 1
    assert calls == [("GET", "/v1/manifest", None), ("GET", "/v1/status", None)]
    manifest["fingerprint"] = content_hash(manifest)
    checkpoint_sha = "a" * 64
    receipt = {
        "schema": "diloco-bootstrap-provenance/1",
        "source_checkpoint_state_sha256": checkpoint_sha,
        "source_checkpoint_metadata_sha256": "b" * 64,
        "parameter_identity": expected_identity,
        "coordinator_manifest_fingerprint": manifest["fingerprint"],
        "run_id": config.decoupled.run_id,
        "source_kind": "checkpoint",
        "coordinator_bootstrap_sha256": hashlib.sha256(bootstrap_bytes).hexdigest(),
    }
    result = qualify_coordinator(
        config,
        model,
        codec,
        "test-corpus",
        expected_parameter_identity=expected_identity,
        bootstrap_state_sha256=checkpoint_sha,
        bootstrap_metadata_sha256="b" * 64,
        bootstrap_receipt=receipt,
    )
    assert result["bootstrapProvenance"]["source_checkpoint_state_sha256"] == checkpoint_sha
    assert result["bootstrapArtifact"]["sha256"] == receipt["coordinator_bootstrap_sha256"]
    assert result["bootstrapArtifact"]["bytes"] == len(bootstrap_bytes)
    assert bootstrap_calls == [("/v1/bootstrap", None)]
    with pytest.raises(ValueError, match="requires its initialization provenance receipt"):
        qualify_coordinator(
            config,
            model,
            codec,
            "test-corpus",
            expected_parameter_identity=expected_identity,
            bootstrap_state_sha256=checkpoint_sha,
        )
    wrong_bootstrap_receipt = dict(receipt, coordinator_bootstrap_sha256="f" * 64)
    with pytest.raises(ValueError, match="Live coordinator bootstrap hash differs"):
        qualify_coordinator(
            config,
            model,
            codec,
            "test-corpus",
            expected_parameter_identity=expected_identity,
            bootstrap_state_sha256=checkpoint_sha,
            bootstrap_metadata_sha256="b" * 64,
            bootstrap_receipt=wrong_bootstrap_receipt,
        )
    missing_bootstrap_hash = dict(receipt)
    missing_bootstrap_hash.pop("coordinator_bootstrap_sha256")
    with pytest.raises(ValueError, match="Live coordinator bootstrap hash differs"):
        qualify_coordinator(
            config,
            model,
            codec,
            "test-corpus",
            expected_parameter_identity=expected_identity,
            bootstrap_state_sha256=checkpoint_sha,
            bootstrap_metadata_sha256="b" * 64,
            bootstrap_receipt=missing_bootstrap_hash,
        )
    manifest["run_id"] = "different-run"
    with pytest.raises(ValueError, match="run_id"):
        qualify_coordinator(
            config,
            model,
            codec,
            "test-corpus",
            expected_parameter_identity=expected_identity,
        )
    manifest["run_id"] = config.decoupled.run_id
    manifest["parameter_identity"] = "different-layout"
    with pytest.raises(ValueError, match="parameter identity"):
        qualify_coordinator(
            config,
            model,
            codec,
            "test-corpus",
            expected_parameter_identity=expected_identity,
        )


def test_coordinator_initialization_writes_receipt_only_after_success(tmp_path):
    import json

    from megatoken.cli import _initialize_coordinator_with_provenance

    root = tmp_path / "run"
    masters = {"tiny.weight": torch.ones(2, dtype=torch.float32)}
    source_state = tmp_path / "source-state.pt"
    source_metadata = tmp_path / "source-metadata.json"
    source_state.write_bytes(b"exact source state")
    source_metadata.write_bytes(b"exact source metadata")

    def successful_initialize(destination, *, run_id, **kwargs):
        assert run_id == "run-a"
        destination.mkdir()
        (destination / "bootstrap.pt").write_bytes(b"actual initialized bootstrap")
        return {"fingerprint": "manifest-digest-a"}

    manifest = _initialize_coordinator_with_provenance(
        successful_initialize,
        root,
        run_id="run-a",
        parameter_masters=masters,
        source_checkpoint_state_sha256=sha256_file(source_state),
        source_checkpoint_metadata_sha256=sha256_file(source_metadata),
        model_config={},
    )
    receipt = json.loads((root / "semantic-bootstrap-provenance.json").read_text())
    assert manifest["fingerprint"] == receipt["coordinator_manifest_fingerprint"]
    assert receipt["source_checkpoint_state_sha256"] == sha256_file(source_state)
    assert receipt["source_checkpoint_metadata_sha256"] == sha256_file(source_metadata)
    assert receipt["coordinator_bootstrap_sha256"] == sha256_file(root / "bootstrap.pt")
    from megatoken.training.diloco_protocol import parameter_identity

    assert receipt["parameter_identity"] == parameter_identity(masters)
    assert receipt["source_kind"] == "checkpoint"

    failed_root = tmp_path / "failed-run"

    def failed_initialize(destination, *, run_id, **kwargs):
        destination.mkdir()
        raise RuntimeError("initialization failed")

    with pytest.raises(RuntimeError, match="initialization failed"):
        _initialize_coordinator_with_provenance(
            failed_initialize,
            failed_root,
            run_id="run-b",
            parameter_masters=masters,
        )
    assert not (failed_root / "semantic-bootstrap-provenance.json").exists()
