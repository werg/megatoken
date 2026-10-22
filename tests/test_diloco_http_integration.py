"""Real localhost HTTP coordinator + complete Megatoken training/resume transaction."""

from __future__ import annotations

import multiprocessing
import socket
import time
import urllib.error
import urllib.request

import torch
from training_fixture import tiny_train_config

from megatoken.data import PreparedDataset
from megatoken.generation.codec import ByteCodec
from megatoken.io import content_hash
from megatoken.loading import build_model, load_model_config
from megatoken.runtime import seed_everything
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import DecoupledConfig
from megatoken.training.diloco_protocol import training_identity
from megatoken.training.diloco_server import initialize_coordinator, serve_coordinator
from megatoken.training.engine import train
from megatoken.training.schedule import PHASES
from megatoken.training.transducer import HindsightSchedule


def test_diloco_local_http_train_checkpoint_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = "test-diloco-token-" + "a" * 32
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    settings = DecoupledConfig(
        coordinator_url=f"http://127.0.0.1:{port}",
        run_id="integration-http-v1",
        workers=1,
        worker_id=0,
        local_steps=1,
        poll_steps=1,
        heartbeat_steps=1,
        io_workers=2,
        timeout_seconds=30,
        max_offline_steps=20,
    )
    config = tiny_train_config(
        tmp_path,
        output=str(tmp_path / "runs"),
        checkpoint_every=1,
        inner_optimizer="muon",
        decoupled=settings,
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
        target=serve_coordinator,
        args=(str(root), "127.0.0.1", port, token),
    )
    server.start()
    try:
        url = f"http://127.0.0.1:{port}/v1/status"
        for _ in range(100):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}),
                    timeout=1,
                ) as response:
                    assert response.status == 200
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise AssertionError("Coordinator failed to start")
        first = train(config)
        state, _ = load_checkpoint(first["checkpoint"])
        assert state["diloco"]["protocol"] == 1
        assert state["diloco"]["worker"] == 0
        assert first["global_step"] == 1
        resumed = train(config, resume=first["checkpoint"])
        assert resumed["global_step"] == 2
        state2, _ = load_checkpoint(resumed["checkpoint"])
        assert state2["diloco"]["local_steps"] == 2
    finally:
        server.terminate()
        server.join(timeout=10)
        if server.is_alive():
            server.kill()
            server.join()
