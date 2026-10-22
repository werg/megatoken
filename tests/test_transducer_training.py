"""The v10 transducer through the real training engine, checkpoints and DiLoCo HTTP."""

from __future__ import annotations

import json
import multiprocessing
import socket
import time
import urllib.error
import urllib.request

import pytest
import retrieval_fixture
import torch
from retrieval_fixture import retrieval_train_config

from megatoken.data import PreparedDataset
from megatoken.generation.codec import ByteCodec
from megatoken.io import content_hash
from megatoken.loading import build_model, load_model_config, load_trained
from megatoken.runtime import seed_everything
from megatoken.training import transducer_diloco
from megatoken.training.checkpoint import load_checkpoint, resolve_checkpoint
from megatoken.training.config import DecoupledConfig
from megatoken.training.diloco_protocol import training_identity
from megatoken.training.diloco_server import initialize_coordinator, serve_coordinator
from megatoken.training.engine import train
from megatoken.training.schedule import PHASES
from megatoken.training.transducer import HindsightSchedule

TRANSDUCER = {
    "transducer": True,
    "transducer_location_dim": 8,
    "transducer_query_dim": 16,
    "transducer_value_rows": 4,
    "transducer_neighbors": 4,
    "transducer_selected": 4,
    "transducer_exploration": 1,
    "transducer_max_lookups": 16,
    "transducer_max_edges": 256,
    "transducer_byte_budget": 10**6,
}


@pytest.fixture
def transducer_overrides(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(
        retrieval_fixture,
        "MODEL_OVERRIDES",
        {**retrieval_fixture.MODEL_OVERRIDES, **TRANSDUCER},
    )


def test_engine_trains_checkpoints_and_resumes_the_bank(tmp_path, transducer_overrides):
    config = retrieval_train_config(tmp_path, max_steps=2, checkpoint_every=1)
    first = train(config)
    _, metadata = load_checkpoint(first["checkpoint"])
    assert metadata["transducer_checkpoint"]["file"] == "transducer.pt"
    saved = torch.load(resolve_checkpoint(first["checkpoint"]) / "transducer.pt", weights_only=True)
    assert saved["sequences"] == 4  # two windows per step
    resumed = train(config, resume=first["checkpoint"])
    assert resumed["global_step"] == 4
    trained = load_trained(resumed["checkpoint"])
    assert trained.model.transducer.bank.bank_id == saved["bank"]["bank_id"]


def test_diloco_http_carries_transducer_rounds(tmp_path, monkeypatch, transducer_overrides):
    monkeypatch.setattr(transducer_diloco, "LOCAL_STEPS", 1)
    codec = ByteCodec.byte_fallback()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = "test-diloco-token-" + "b" * 32
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    settings = DecoupledConfig(
        coordinator_url=f"http://127.0.0.1:{port}",
        run_id="transducer-http-v1",
        workers=1,
        worker_id=0,
        local_steps=1,
        poll_steps=1,
        heartbeat_steps=1,
        io_workers=2,
        timeout_seconds=30,
        max_offline_steps=20,
    )
    config = retrieval_train_config(
        tmp_path, output=str(tmp_path / "runs"), max_steps=3, decoupled=settings
    )
    with PreparedDataset(config.dataset) as prepared:
        dataset_fingerprint = content_hash(prepared.manifest)
    seed_everything(config.seed, config.deterministic)
    model_config = load_model_config(
        config.model,
        **{
            **config.model_overrides,
            "expert_backend": config.expert_backend,
            "checkpoint_stages": config.gradient_checkpointing,
        },
    )
    model = build_model(model_config, codec, device="cpu", precision="fp32")
    root = tmp_path / "coordinator"
    initialize_coordinator(
        root,
        run_id=settings.run_id,
        model_state=model.state_dict(),
        masters={
            name: p.detach().float() for name, p in model.named_parameters() if p.requires_grad
        },
        model_config=model_config.to_dict(),
        codec_fingerprint=codec.fingerprint,
        train_identity=training_identity(config),
        dataset_fingerprint=dataset_fingerprint,
        workers=1,
        min_quorum=1,
        fragment_bytes=1 << 20,
        grace_seconds=0,
        token_budget=config.budget or PHASES[config.phase].budget,
        transducer={
            "geometry": dict(vars(model_config.transducer_geometry)),
            "evidence_from": HindsightSchedule().warmup_steps,
        },
    )
    server = multiprocessing.get_context("spawn").Process(
        target=serve_coordinator, args=(str(root), "127.0.0.1", port, token)
    )
    server.start()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        for _ in range(100):
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/status", headers=headers
                )
                with urllib.request.urlopen(request, timeout=1) as response:
                    assert response.status == 200
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise AssertionError("Coordinator failed to start")
        result = train(config)
        state, _ = load_checkpoint(result["checkpoint"])
        assert state["diloco"]["transducer"]["sequence"] >= 2
        for _ in range(50):
            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/status", headers=headers)
            with urllib.request.urlopen(request, timeout=5) as response:
                if json.loads(response.read())["transducer_version"] >= 1:
                    break
            time.sleep(0.1)
        else:
            raise AssertionError("The coordinator never merged a transducer round")
    finally:
        server.terminate()
        server.join(timeout=10)
        if server.is_alive():
            server.kill()
            server.join()
