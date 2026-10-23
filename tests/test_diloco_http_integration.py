"""Real localhost HTTP coordinator + complete Megatoken training/resume transaction."""

from __future__ import annotations

import json
import multiprocessing
import socket
import time
import urllib.error
import urllib.request
from argparse import Namespace
from dataclasses import asdict, replace

import pytest
import torch
from training_fixture import tiny_train_config

from megatoken.data import PreparedCorpus
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model, load_model_config
from megatoken.runtime import seed_everything
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import DecoupledConfig
from megatoken.training.diloco_protocol import training_identity
from megatoken.training.diloco_server import initialize_coordinator, serve_coordinator
from megatoken.training.engine import train
from megatoken.training.transducer import HindsightSchedule


@pytest.mark.parametrize("purpose", ["diagnostic", "production"])
def test_diloco_local_http_train_checkpoint_resume(tmp_path, monkeypatch, purpose):
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
    if purpose == "production":
        from test_teacher_graph import FrozenWriter, episode
        from training_fixture import prepare_dataset

        from megatoken.data import Lineage
        from megatoken.data.hindsight_corpus import prepare_corpus

        ep = episode()
        training = prepare_dataset(tmp_path / "source", [ep])
        heldout = replace(
            ep,
            episode_id="heldout",
            lineage=Lineage("joint", "v1", "heldout"),
            records=tuple(
                replace(r, payload=r.payload.replace(b"blue", b"red")) for r in ep.records
            ),
        )
        validation = prepare_dataset(tmp_path / "validation", [heldout], heldout=True)
        config = tiny_train_config(
            tmp_path / "production",
            purpose="production",
            datasets=[str(training), str(validation)],
            hindsight_targets=str(tmp_path / "targets"),
            max_window_records=8,
            max_record_bytes=16384,
            eval_every=1,
            checkpoint_every=1,
            inner_optimizer=None,
            use_effort_mixture=None,
            use_stage_mixture=None,
            representation_mode=None,
            effort=6,
            decoupled=settings,
        )
        prepare_corpus(config, tmp_path / "targets", FrozenWriter(), fit_windows=1)
    with PreparedCorpus(config.datasets) as prepared:
        dataset_fingerprint = prepared.identity
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
    if purpose == "production":
        from megatoken.cli import _diloco_initialize

        config_path = tmp_path / "production.json"
        config_path.write_text(json.dumps(asdict(config)))
        _diloco_initialize(
            Namespace(
                config=str(config_path),
                root=str(root),
                resume=None,
                quorum=1,
                fragment_mib=1,
                outer_lr=0.7,
                outer_momentum=0.9,
                grace_seconds=0.0,
                max_staleness=2,
            )
        )
    else:
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
            token_budget=config.schedule.budget,
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
        # A spawned interpreter imports the native tensor runtime before binding
        # its port. Bound actual readiness time rather than assuming a warm,
        # otherwise idle machine can start it within five seconds.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not server.is_alive():
                raise AssertionError(f"Coordinator exited during startup: {server.exitcode}")
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
            raise AssertionError("Coordinator did not become ready within 30 seconds")
        first = train(config)
        state, _ = load_checkpoint(first["checkpoint"])
        assert state["diloco"]["protocol"] == 1
        assert state["diloco"]["worker"] == 0
        assert first["global_step"] == 1
        if purpose == "production":
            assert state["train_config"]["inner_optimizer"] == "muon"
            assert state["train_config"]["use_effort_mixture"]
            metrics = [
                json.loads(line)
                for line in (tmp_path / "production/run/worker-00/metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            trained = next(row for row in metrics if row["event"] == "train")
            assert trained["generator"]["teacher_target_values"] > 0
            assert trained["generator"]["navigation_examples"] > 0
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
