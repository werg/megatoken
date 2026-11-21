"""Bounded normal-loader optimizer/checkpoint/resume smoke with injected audit boundaries."""

import json
import multiprocessing
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
from test_semantic_v8_training_policy import pinned_tokenizer_path, prepared_v8_fixture

from megatoken.generation.codec import ByteCodec
from megatoken.training.checkpoint import load_checkpoint, save_checkpoint, transducer_attachment
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train


def test_v8_normal_update_checkpoint_fresh_process_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    # The fixture constructs native V8 rows with explicitly injected source and
    # tokenizer audit boundaries. The ordinary trainer, model, optimizer and
    # checkpoint operations below execute normally.
    prepared_v8_fixture(tmp_path)
    prepared = tmp_path / "prepared-v8"
    registry = json.loads((prepared / "semantic-v8-families.json").read_text())
    manifest = json.loads((prepared / "manifest.json").read_text())
    pins = manifest["sources"][0]
    policy = {
        "schema": "semantic-training-policy/8",
        "producerRevision": pins["producerRevision"],
        "consumerRevision": pins["consumerRevision"],
        "registryDigest": registry["digest"],
    }
    import megatoken.training.semantic_v8 as semantic_v8

    # The normal training loader still validates each
    # manifest, source/native receipt, split, physical record and batch policy.
    monkeypatch.setattr(
        semantic_v8,
        "load_semantic_v8_codec",
        lambda _path, *, vocab_size: ByteCodec.byte_fallback(),
    )
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="injected-tokenizer",
        datasets=[str(prepared)],
        output=str(tmp_path / "fit"),
        device="cpu",
        max_steps=1,
        checkpoint_every=1,
        keep_checkpoints=1,
        microbatch_episodes=1,
        accumulate_scored=1,
        accumulate_ingested=1,
        max_accumulation_windows=1,
        allow_unverified_data=True,
        semantic_v8_policy=policy,
    )
    first = train(config)
    assert first["global_step"] == 1
    payload, checkpoint_metadata = load_checkpoint(first["checkpoint"])
    transducer_state = torch.load(
        Path(first["checkpoint"]) / checkpoint_metadata["transducer_checkpoint"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    assert payload["global_step"] == 1
    assert max(int(state["step"]) for state in payload["optimizer"]["moments"].values()) == 1

    config_path = tmp_path / "resume-config.json"
    config_path.write_text(json.dumps(config.__dict__, separators=(",", ":")))
    resume_code = """
import importlib.abc, json, sys, torch, traceback
blocked={
    "megatoken.data.algorithm_v7",
    "megatoken.data.detailed_v7",
    "megatoken.data.detailed_harness_v7",
    "megatoken.data.detailed_context_v7",
    "megatoken.data.semantic_harness_v6",
    "megatoken.data.semantic_contract",
    "megatoken.data.semantic_contract_v7",
    "megatoken.training.semantic_budget",
}
class NoLegacyWire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("Legacy wire dependency: " + fullname + "\\n" + "".join(traceback.format_stack(limit=18)))
sys.meta_path.insert(0, NoLegacyWire())
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train
import megatoken.training.semantic_v8 as semantic_v8
torch.set_num_threads(1)
semantic_v8.load_semantic_v8_codec=lambda _path, *, vocab_size: ByteCodec.byte_fallback()
config=TrainConfig.from_dict(json.loads(open(sys.argv[1]).read()))
result=train(config,resume=sys.argv[2])
assert not blocked.intersection(sys.modules)
print(json.dumps({'global_step':result['global_step'],'termination':result['termination']}))
"""
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, "-c", resume_code, str(config_path), first["checkpoint"]],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["global_step"] == 2

    forged_payload = {
        **payload,
        "semantic_exposure": {
            "schema": "semantic-exposure/4",
            "examples": 1,
            "generatedTokens": 8,
            "observedTokens": 13,
            "computeSeconds": 0.5,
        },
        "semantic_cohort_digest": "f" * 64,
    }
    forged = save_checkpoint(
        tmp_path / "forged-experiment-state",
        "forged",
        forged_payload,
        {
            key: value
            for key, value in checkpoint_metadata.items()
            if key
            not in {
                "transducer_checkpoint",
                "state_sha256",
                "checkpoint_name",
                "format_version",
                "pinned",
            }
        },
        attachments=transducer_attachment(lambda: transducer_state),
    )
    forged_code = resume_code.replace(
        "result=train(config,resume=sys.argv[2])\nassert not blocked.intersection(sys.modules)\nprint(json.dumps({'global_step':result['global_step'],'termination':result['termination']}))",
        "try:\n train(config,resume=sys.argv[2])\nexcept ValueError as error:\n assert 'nonsemantic checkpoint' in str(error), error\nelse:\n raise AssertionError('forged experiment state was accepted')\nassert not blocked.intersection(sys.modules)\nprint('forged-state-rejected')",
    )
    rejected = subprocess.run(
        [sys.executable, "-c", forged_code, str(config_path), str(forged)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert rejected.returncode == 0, rejected.stderr
    assert rejected.stdout.strip().splitlines()[-1] == "forged-state-rejected"


def test_v8_diloco_initialize_handshake_export_and_served_reload(tmp_path, monkeypatch):
    """Use pinned tokenizer bytes with explicit source-verifier fixture boundaries."""
    from argparse import Namespace
    from dataclasses import asdict

    from megatoken.cli import _diloco_initialize
    from megatoken.data import PreparedCorpus
    from megatoken.generation.codec import LING_VOCAB_SIZE
    from megatoken.io import content_hash
    from megatoken.loading import build_model, load_model_config
    from megatoken.served import ServedBundle, export_served
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.config import DecoupledConfig, TrainConfig
    from megatoken.training.diloco_client import DecoupledWorker
    from megatoken.training.diloco_export import export_coordinator
    from megatoken.training.diloco_protocol import training_identity
    from megatoken.training.diloco_server import serve_coordinator
    from megatoken.training.semantic_v8 import load_semantic_v8_codec

    torch.set_num_threads(1)
    tokenizer = pinned_tokenizer_path()
    codec = load_semantic_v8_codec(tokenizer, vocab_size=LING_VOCAB_SIZE)
    _, _, policy, prepared = prepared_v8_fixture(tmp_path, codec=codec)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = DecoupledConfig(
        coordinator_url=f"http://127.0.0.1:{port}",
        allow_private_http=True,
        workers=1,
        worker_id=0,
        run_id="semantic-v8-lifecycle",
        local_steps=1,
        io_workers=1,
        timeout_seconds=3,
    )
    token = "test-semantic-v8-diloco-token-" + "a" * 32
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec=str(tokenizer),
        model_overrides={"vocab_size": LING_VOCAB_SIZE},
        datasets=[str(prepared)],
        output=str(tmp_path / "worker-output"),
        device="cpu",
        max_steps=1,
        allow_unverified_data=True,
        semantic_v8_policy=policy,
        decoupled=settings,
    )
    config_path = tmp_path / "diloco-v8.json"
    config_path.write_text(json.dumps(asdict(config)))
    coordinator_root = tmp_path / "coordinator"
    _diloco_initialize(
        Namespace(
            config=str(config_path),
            root=str(coordinator_root),
            resume=None,
            quorum=1,
            fragment_mib=1,
            outer_lr=0.7,
            outer_momentum=0.9,
            grace_seconds=0.0,
            max_staleness=2,
        )
    )
    manifest = json.loads((coordinator_root / "manifest.json").read_text())
    assert manifest["semantic_v8_policy"] == policy
    assert (
        manifest["semantic_v8_lineage"]["preparedCorpusDigest"] == manifest["dataset_fingerprint"]
    )
    assert manifest["semantic_v8_lineage"]["streamDigest"] != manifest["dataset_fingerprint"]

    server = multiprocessing.get_context("spawn").Process(
        target=serve_coordinator, args=(str(coordinator_root), "127.0.0.1", port, token)
    )
    server.start()
    worker = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not server.is_alive():
                raise AssertionError(f"Coordinator exited during startup: {server.exitcode}")
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/manifest",
                    headers={"Authorization": f"Bearer {token}"},
                )
                with urllib.request.urlopen(request, timeout=1):
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise AssertionError("Coordinator did not become ready")
        model_config = load_model_config(
            config.model,
            **{
                **config.model_overrides,
                "expert_backend": config.expert_backend,
                "checkpoint_stages": config.gradient_checkpointing,
            },
        )
        model = build_model(model_config, codec, device="cpu", precision="fp32")
        masters = {
            name: parameter.detach().cpu().float()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        with PreparedCorpus(config.datasets) as corpus:
            dataset_fingerprint = corpus.identity
        from megatoken.training.diloco_protocol import parameter_identity

        expected = {
            "run_id": settings.run_id,
            "workers": settings.workers,
            "model_config": model_config.to_dict(),
            "codec_fingerprint": codec.fingerprint,
            "parameter_identity": parameter_identity(masters),
            "token_budget": config.schedule.budget,
            "train_identity": training_identity(config),
            "dataset_fingerprint": dataset_fingerprint,
            "semantic_v8_lineage": manifest["semantic_v8_lineage"],
            "semantic_v8_policy": policy,
        }
        differences = {
            key: (expected_value, manifest.get(key))
            for key, expected_value in expected.items()
            if expected_value != manifest.get(key)
        }
        assert not differences, differences
        worker = DecoupledWorker(
            settings,
            model_config=model_config.to_dict(),
            codec_fingerprint=codec.fingerprint,
            masters=masters,
            output=tmp_path / "worker-client",
            token_budget=config.schedule.budget,
            train_identity=training_identity(config),
            dataset_fingerprint=dataset_fingerprint,
            semantic_v8_lineage=manifest["semantic_v8_lineage"],
            semantic_v8_policy=policy,
        )
        assert worker.manifest["semantic_v8_lineage"] == manifest["semantic_v8_lineage"]
        from megatoken.data.semantic_json import stable_algorithm_hash

        substituted = {**manifest["semantic_v8_lineage"], "streamDigest": "e" * 64}
        substituted["digest"] = stable_algorithm_hash(
            {key: value for key, value in substituted.items() if key != "digest"}
        )
        for expected_lineage, expected_policy in ((substituted, policy), (None, None)):
            with pytest.raises(ValueError, match="differs from syncer"):
                DecoupledWorker(
                    settings,
                    model_config=model_config.to_dict(),
                    codec_fingerprint=codec.fingerprint,
                    masters=masters,
                    output=tmp_path / "rejected-worker",
                    token_budget=config.schedule.budget,
                    train_identity=training_identity(config),
                    dataset_fingerprint=dataset_fingerprint,
                    semantic_v8_lineage=expected_lineage,
                    semantic_v8_policy=expected_policy,
                )
    finally:
        if worker is not None:
            worker.close()
        server.terminate()
        server.join(timeout=30)
        if server.is_alive():
            server.kill()
            server.join()

    manifest_path = coordinator_root / "manifest.json"
    saved_manifest = json.loads(manifest_path.read_text())
    for mutation, message in (
        (
            lambda value: value.update(semantic_v8_policy={**policy, "registryDigest": "9" * 64}),
            "V8 training policy",
        ),
        (lambda value: value.update(dataset_fingerprint="9" * 64), "prepared corpus identity"),
        (lambda value: value.update(semantic_v8_lineage=None), "policy and training lineage"),
    ):
        changed = json.loads(json.dumps(saved_manifest))
        mutation(changed)
        changed["fingerprint"] = content_hash(
            {key: value for key, value in changed.items() if key != "fingerprint"}
        )
        manifest_path.write_text(json.dumps(changed))
        rejected_output = tmp_path / f"rejected-{message.replace(' ', '-')}"
        with pytest.raises(ValueError, match=message):
            export_coordinator(coordinator_root, rejected_output, config)
        assert not rejected_output.exists()
    manifest_path.write_text(json.dumps(saved_manifest))

    exported = export_coordinator(coordinator_root, tmp_path / "exports", config)
    payload, metadata = load_checkpoint(exported["checkpoint"])
    from megatoken.semantic_lineage_v8 import checkpoint_lineage_v8

    assert checkpoint_lineage_v8(payload, metadata) == manifest["semantic_v8_lineage"]
    transitioned_root = tmp_path / "resumed-coordinator"
    _diloco_initialize(
        Namespace(
            config=str(config_path),
            root=str(transitioned_root),
            resume=exported["checkpoint"],
            quorum=1,
            fragment_mib=1,
            outer_lr=0.7,
            outer_momentum=0.9,
            grace_seconds=0.0,
            max_staleness=2,
        )
    )
    resumed_manifest = json.loads((transitioned_root / "manifest.json").read_text())
    assert resumed_manifest["semantic_v8_policy"] == policy
    assert resumed_manifest["semantic_v8_lineage"] == manifest["semantic_v8_lineage"]
    served_dir = tmp_path / "served"
    export_served(exported["checkpoint"], served_dir)
    served = ServedBundle(served_dir, device="cpu")
    assert served.semantic_v8_lineage["training"] == manifest["semantic_v8_lineage"]
    bundle_code = """
import importlib.abc, json, sys, traceback
blocked={
    "megatoken.data.algorithm_v7", "megatoken.data.detailed_v7",
    "megatoken.data.detailed_harness_v7", "megatoken.data.detailed_context_v7",
    "megatoken.data.semantic_harness_v6", "megatoken.data.semantic_contract",
    "megatoken.data.semantic_contract_v7",
}
class NoLegacyWire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("Legacy wire dependency: " + fullname)
sys.meta_path.insert(0, NoLegacyWire())
from megatoken.served import ServedBundle
bundle=ServedBundle(sys.argv[1], device="cpu")
assert bundle.semantic_v8_lineage["training"] is not None
assert not blocked.intersection(sys.modules)
print(json.dumps({"served": True, "legacyModulesLoaded": 0}))
"""
    loaded = subprocess.run(
        [sys.executable, "-c", bundle_code, str(served_dir)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert loaded.returncode == 0, loaded.stderr
    assert json.loads(loaded.stdout.strip().splitlines()[-1]) == {
        "served": True,
        "legacyModulesLoaded": 0,
    }
    jsonl_code = """
import importlib.abc, io, json, runpy, sys
blocked={
    "megatoken.data.algorithm_v7", "megatoken.data.detailed_v7",
    "megatoken.data.detailed_harness_v7", "megatoken.data.detailed_context_v7",
    "megatoken.data.semantic_harness_v6", "megatoken.data.semantic_contract",
    "megatoken.data.semantic_contract_v7",
}
class NoLegacyWire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("Legacy wire dependency: " + fullname)
sys.meta_path.insert(0, NoLegacyWire())
script=sys.argv[1]
sys.argv=[script, "--served", sys.argv[2], "--device", "cpu", "--threads", "1"]
sys.stdin=io.TextIOWrapper(io.BytesIO(b'{\"id\":\"metrics\",\"op\":\"metrics\"}\\n'))
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as result:
    assert result.code == 0
assert not blocked.intersection(sys.modules)
"""
    jsonl = subprocess.run(
        [
            sys.executable,
            "-c",
            jsonl_code,
            str(Path(__file__).resolve().parents[1] / "scripts/serve_semantic_v8.py"),
            str(served_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert jsonl.returncode == 0, jsonl.stderr
    response = json.loads(jsonl.stdout.strip().splitlines()[-1])
    assert response["id"] == "metrics" and response["ok"] is True
