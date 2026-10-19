"""Real localhost HTTP coordinator + complete Megatoken training/resume transaction."""

from __future__ import annotations

import multiprocessing
import socket
import time
import urllib.error
import urllib.request

import torch

from megatoken.data import prepare_sources, write_demo_source
from megatoken.generation.codec import ByteCodec
from megatoken.io import content_hash
from megatoken.loading import build_model, load_model_config
from megatoken.runtime import seed_everything
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import DecoupledConfig, TrainConfig
from megatoken.training.diloco_protocol import training_identity
from megatoken.training.diloco_server import initialize_coordinator, serve_coordinator
from megatoken.training.engine import train
from megatoken.training.schedule import PHASES


def test_diloco_local_http_train_checkpoint_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    source = write_demo_source(tmp_path / "source.jsonl", count=24, max_record_bytes=128)
    dataset = prepare_sources([source], tmp_path / "data", codec)
    dataset_fingerprint = content_hash(dataset.manifest)
    dataset.close()
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
    config = TrainConfig(
        dataset=str(tmp_path / "data"),
        output=str(tmp_path / "runs"),
        model="tiny",
        codec="byte",
        device="cpu",
        precision="fp32",
        max_steps=1,
        microbatch_episodes=2,
        accumulate_scored=80,
        accumulate_ingested=1000,
        max_accumulation_windows=2,
        generated_capacity=128,
        max_history_tokens=1024,
        max_record_bytes=512,
        checkpoint_every=1,
        eval_every=0,
        gradient_checkpointing=False,
        use_stage_mixture=False,
        use_effort_mixture=False,
        inner_optimizer="muon",
        decoupled=settings,
    )
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
        masters={name: p.detach().float() for name, p in model.named_parameters() if p.requires_grad},
        model_config=model_config.to_dict(),
        codec_fingerprint=codec.fingerprint,
        train_identity=training_identity(config),
        dataset_fingerprint=dataset_fingerprint,
        workers=1,
        min_quorum=1,
        fragment_bytes=1 << 20,
        grace_seconds=0,
        token_budget=config.budget or PHASES[config.phase].budget,
    )
    server = multiprocessing.get_context("spawn").Process(
        target=serve_coordinator, args=(str(root), "127.0.0.1", port, token),
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
