"""Content-bound V8 resume across storage moves, with explicit fixture admission."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from megatoken.generation.codec import ByteCodec
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig


def fixture_codec(path, *, vocab_size):
    """Test-only byte vocabulary; this is not production tokenizer admission."""
    assert Path(path).read_bytes() == b"explicit byte-fallback relocation fixture\n"
    return ByteCodec.byte_fallback()


def assert_same_state(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            assert_same_state(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_same_state(a, b, f"{path}[{index}]")
    else:
        assert type(left) is type(right) and left == right, path


def test_real_v8_next_update_survives_input_relocation_in_a_fresh_process(tmp_path, monkeypatch):
    """Three actual updates: one initial and two independent continuations."""
    from test_semantic_v8_training_policy import prepared_v8_fixture

    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    _, _codec, policy, prepared = prepared_v8_fixture(tmp_path)
    tokenizer = tmp_path / "tokenizer.fixture"
    tokenizer.write_bytes(b"explicit byte-fallback relocation fixture\n")
    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", fixture_codec)
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec=str(tokenizer),
        datasets=[str(prepared)],
        output=str(tmp_path / "initial"),
        device="cpu",
        precision="fp32",
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
    first_payload, first_metadata = load_checkpoint(first["checkpoint"])
    assert first_payload["global_step"] == 1
    baseline = train(replace(config, output=str(tmp_path / "baseline")), resume=first["checkpoint"])
    baseline_payload, baseline_metadata = load_checkpoint(baseline["checkpoint"])
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_data = relocated / "prepared"
    relocated_codec = relocated / "tokenizer.fixture"
    shutil.copytree(prepared, relocated_data)
    shutil.copyfile(tokenizer, relocated_codec)
    moved = replace(
        config,
        codec=str(relocated_codec),
        datasets=[str(relocated_data)],
        output=str(relocated / "continuation"),
    )
    config_path = relocated / "resume.json"
    config_path.write_text(json.dumps(asdict(moved)) + "\n")
    # The resumed process cannot accidentally fall back to the original paths.
    prepared.rename(tmp_path / "original-data-unavailable")
    tokenizer.rename(tmp_path / "original-codec-unavailable")
    assert not prepared.exists() and not tokenizer.exists()
    code = """
import json, sys, torch
from tests.test_semantic_v8_input_relocation import fixture_codec
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train
import megatoken.training.semantic_v8 as semantic_v8
torch.set_num_threads(1)
semantic_v8.load_semantic_v8_codec=fixture_codec
config=TrainConfig.from_dict(json.loads(open(sys.argv[1]).read()))
result=train(config,resume=sys.argv[2])
print(json.dumps({'checkpoint':result['checkpoint'],'global_step':result['global_step']}))
"""
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    process = subprocess.run(
        [sys.executable, "-c", code, str(config_path), first["checkpoint"]],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (tmp_path / "relocated-process.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout.strip().splitlines()[-1])
    actual, metadata = load_checkpoint(result["checkpoint"])
    assert actual["global_step"] == baseline_payload["global_step"] == 2
    keys = set(baseline_payload) - {"train_config"}
    assert set(actual) - {"train_config"} == keys
    for key in sorted(keys):
        assert_same_state(baseline_payload[key], actual[key], key)
    baseline_bank = torch.load(
        Path(baseline["checkpoint"]) / baseline_metadata["transducer_checkpoint"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    relocated_bank = torch.load(
        Path(result["checkpoint"]) / metadata["transducer_checkpoint"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    assert_same_state(baseline_bank, relocated_bank, "transducer")
    assert actual["train_config"] == asdict(moved)
    assert metadata["config_fingerprint"] != baseline_metadata["config_fingerprint"]
    for key in ("dataset_fingerprint", "codec_fingerprint", "semantic_v8_lineage_digest", "semantic_v8_position_digest"):
        assert metadata[key] == baseline_metadata[key], key
    assert actual["semantic_v8_position_exposure"]["previousDigest"] == first_payload["semantic_v8_position_exposure"]["digest"]
    report = {
        "schema": "semantic-v8-input-relocation-evidence/1",
        "executedRevision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "runtime": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu", "precision": "fp32"},
        "scope": "One controlled source with injected source and byte-tokenizer admission; no production-tokenizer, source, hosted, release or learned-quality qualification",
        "optimizerUpdatesExecuted": 3,
        "trajectories": "one initial update; two separate next updates from the same checkpoint",
        "originalInputPathsAbsent": True,
        "freshProcessContinuation": True,
        "comparedPayloadFields": sorted(keys),
        "transducerStateEqual": True,
        "firstCheckpointMetadata": first_metadata,
        "baselineCheckpointMetadata": baseline_metadata,
        "relocatedCheckpointMetadata": metadata,
        "positionExposure": actual["semantic_v8_position_exposure"],
        "learnedSuccess": None,
    }
    (tmp_path / "relocation-proof.json").write_text(json.dumps(report, indent=2) + "\n")
