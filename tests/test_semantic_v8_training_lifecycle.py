"""Bounded normal-loader optimizer/checkpoint/resume smoke with injected audit boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from test_semantic_v8_training_policy import (
    test_freshly_prepared_v8_supply_passes_complete_native_batch_admission as prepare_v8_fixture,
)

from megatoken.generation.codec import ByteCodec
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train


def test_v8_normal_update_checkpoint_fresh_process_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    # The shared policy test constructs the realistic V8 artifact and prepared
    # shard. Stub only the redundant eager model/batch setup so this test reaches
    # the ordinary trainer once; the admission and tokenizer verifier boundaries
    # are explicitly injected for this fixture because production tokenizer bytes
    # are not distributed with the repository test fixtures.
    with (
        patch("megatoken.loading.build_model", return_value=object()),
        patch("megatoken.training.engine.prepare_batches", return_value=[object()]),
    ):
        prepare_v8_fixture(tmp_path)
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

    # This is the only fake: the normal training loader still validates each
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
    payload, _ = load_checkpoint(first["checkpoint"])
    assert payload["global_step"] == 1
    assert max(int(state["step"]) for state in payload["optimizer"]["moments"].values()) == 1

    config_path = tmp_path / "resume-config.json"
    config_path.write_text(json.dumps(config.__dict__, separators=(",", ":")))
    resume_code = """
import json, sys, torch
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train
import megatoken.training.semantic_v8 as semantic_v8
torch.set_num_threads(1)
semantic_v8.load_semantic_v8_codec=lambda _path, *, vocab_size: ByteCodec.byte_fallback()
config=TrainConfig.from_dict(json.loads(open(sys.argv[1]).read()))
result=train(config,resume=sys.argv[2])
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
        check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["global_step"] == 2
