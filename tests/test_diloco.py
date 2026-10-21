"""Crash, convergence, sharding and mixed-optimizer tests for Decoupled DiLoCo."""

from __future__ import annotations

import copy
import uuid

import pytest
import torch
from torch import nn

from megatoken.data import PreparedDataset, WindowConfig, prepare_sources, write_demo_source
from megatoken.generation.codec import ByteCodec
from megatoken.training.diloco_protocol import (
    install_fragment,
    load_tensors,
    merge_pseudogradients,
    outer_nesterov,
    pack_fragment,
    parameter_identity,
    plan_fragments,
    save_tensors,
)
from megatoken.training.diloco_server import Conflict, Coordinator, initialize_coordinator
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.stream import IndependentWindowStream


def _coordinator(tmp_path, *, workers=3, quorum=2, max_staleness=0):
    baseline = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    model = {"linear.weight": baseline.clone()}
    manifest = initialize_coordinator(
        tmp_path / "syncer",
        run_id="unit-run",
        model_state=model,
        masters=model,
        model_config={"test": True},
        codec_fingerprint="codec",
        train_identity="objective",
        workers=workers,
        min_quorum=quorum,
        fragment_bytes=1024,
        outer_lr=1,
        outer_momentum=0,
        grace_seconds=0,
        max_staleness=max_staleness,
    )
    return Coordinator(tmp_path / "syncer"), baseline, manifest


def _submit(coordinator, tmp_path, *, worker, session, delta, sequence=1, version=0):
    path = tmp_path / f"delta-{worker}.safetensors"
    save_tensors(path, {"delta": torch.full((64,), delta, dtype=torch.float32)})
    return coordinator.submit(
        0,
        worker,
        session,
        sequence,
        version,
        steps=10,
        tokens=100,
        incoming=path,
    )


def test_fragment_plan_handles_large_packed_experts_and_exact_reinstallation():
    model = nn.Module()
    model.packed = nn.Parameter(torch.arange(400, dtype=torch.float32).reshape(10, 10, 4))
    model.bias = nn.Parameter(torch.linspace(0, 1, 11))
    optimizer = MasterAdamW(model, lr=0.01, inner_optimizer="adamw")
    try:
        fragments = plan_fragments(optimizer.masters, target_bytes=1024)
        assert len(fragments) > 1
        snapshot = {name: p.detach().clone() for name, p in optimizer.masters.items()}
        for fragment in fragments:
            original = pack_fragment(optimizer.masters, fragment)
            install_fragment(optimizer, fragment, original + 7)
        for name, value in snapshot.items():
            torch.testing.assert_close(optimizer.masters[name], value + 7, rtol=0, atol=0)
            torch.testing.assert_close(model.state_dict()[name], value + 7, rtol=0, atol=0)
        assert parameter_identity(snapshot) == parameter_identity(optimizer.masters)
    finally:
        optimizer.close()


def test_quorum_merge_is_atomic_and_crash_idempotent(tmp_path):
    coordinator, initial, manifest = _coordinator(tmp_path)
    sessions = [str(uuid.uuid4()) for _ in range(3)]
    for i in range(3):
        coordinator.lease(i, sessions[i])
    assert (
        _submit(coordinator, tmp_path, worker=0, session=sessions[0], delta=1)["status"] == "queued"
    )
    assert coordinator.status()["versions"] == [0]
    assert _submit(coordinator, tmp_path, worker=1, session=sessions[1], delta=3)["version"] == 1
    state = load_tensors(coordinator._path(0))
    torch.testing.assert_close(state["theta"], initial.reshape(-1) - 2, rtol=0, atol=0)
    assert coordinator.status()["merges"] == 1
    # Simulate a crash after atomic publication but before pending cleanup.
    queued = coordinator._pending_dir(0) / "000.safetensors"
    save_tensors(
        queued,
        {"delta": torch.ones(64)},
        metadata={
            "worker": "0",
            "sequence": "1",
            "base_version": "0",
            "steps": "10",
            "tokens": "100",
        },
    )
    restarted = Coordinator(tmp_path / "syncer")
    assert not queued.exists()
    assert restarted.status()["versions"] == [1]
    assert (
        _submit(restarted, tmp_path, worker=0, session=sessions[0], delta=1)["status"]
        == "already_merged"
    )
    with pytest.raises(Conflict, match="Stale"):
        _submit(restarted, tmp_path, worker=2, session=sessions[2], delta=1, version=0)
    assert manifest["min_quorum"] == 2


def test_weighting_and_outer_momentum_use_finite_float32():
    delta = merge_pseudogradients(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        tokens=[10, 20],
        steps=[10, 10],
        staleness=[0, 0],
    )
    expected = (torch.tensor([1.0, 2.0]) + 4 * torch.tensor([3.0, 4.0])) / 5
    torch.testing.assert_close(delta, expected)
    theta, momentum = outer_nesterov(
        torch.tensor([10.0, 10.0]),
        torch.zeros(2),
        delta,
        lr=0.5,
        momentum=0.8,
    )
    torch.testing.assert_close(momentum, delta)
    torch.testing.assert_close(theta, 10 - 0.9 * delta)
    with pytest.raises(ValueError, match="Invalid outer gradient"):
        merge_pseudogradients([torch.tensor([float("nan")])], [1], [1], [0])


def test_hybrid_muon_only_selects_eligible_hidden_matrices_and_resumes():
    torch.manual_seed(23)
    model = nn.ModuleDict(
        {
            "hidden": nn.Linear(8, 8),
            "embed": nn.Embedding(16, 8),
            "layernorm": nn.LayerNorm(8),
        }
    ).bfloat16()
    optimizer = MasterAdamW(model, lr=0.003, inner_optimizer="muon")
    assert optimizer.muon_names == ["hidden.weight"]
    assert optimizer.optimizer is not None
    assert optimizer.muon_optimizer is not None
    try:
        for master in optimizer.masters.values():
            master.grad = torch.randn_like(master) * 0.03
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        replica = copy.deepcopy(model)
        recovered = MasterAdamW(replica, lr=0.003, inner_optimizer="muon")
        try:
            recovered.load_state_dict(state)
            for key in optimizer.masters:
                gradient = torch.randn_like(optimizer.masters[key])
                optimizer.masters[key].grad = gradient.clone()
                recovered.masters[key].grad = gradient.clone()
            optimizer.step()
            recovered.step()
            for key in optimizer.masters:
                torch.testing.assert_close(
                    optimizer.masters[key], recovered.masters[key], rtol=0, atol=0
                )
        finally:
            recovered.close()
    finally:
        optimizer.close()


def test_independent_shard_cursor_is_restartable_without_collectives(tmp_path):
    codec = ByteCodec.byte_fallback()
    source = write_demo_source(tmp_path / "source.jsonl", count=24, max_record_bytes=128)
    prepared = prepare_sources([source], tmp_path / "data", codec)
    prepared.close()
    ds0 = PreparedDataset(tmp_path / "data")
    ds1 = PreparedDataset(tmp_path / "data")
    ds2 = PreparedDataset(tmp_path / "data")
    config = WindowConfig(max_window_tokens=512, max_records=2)
    a = IndependentWindowStream(ds0, codec, config, seed=31, rank=0, world_size=2)
    b = IndependentWindowStream(ds1, codec, config, seed=31, rank=1, world_size=2)
    clone = IndependentWindowStream(ds2, codec, config, seed=31, rank=0, world_size=2)
    try:
        first = a.next()
        other = b.next()
        assert first.episode.episode_id != other.episode.episode_id
        state = a.state_dict()
        clone.load_state_dict(state)
        for _ in range(3):
            left, right = a.next(), clone.next()
            assert left.episode.episode_id == right.episode.episode_id
            assert left.window.start == right.window.start
    finally:
        a.close()
        b.close()
        clone.close()


def test_coordinator_strict_init_never_replaces_committed_run(tmp_path):
    coordinator, initial, _ = _coordinator(tmp_path, workers=1, quorum=1)
    with pytest.raises(FileExistsError):
        initialize_coordinator(
            tmp_path / "syncer",
            run_id="unit-run",
            model_state={"linear.weight": initial},
            masters={"linear.weight": initial},
            model_config={},
            codec_fingerprint="codec",
            train_identity="objective",
            workers=1,
            min_quorum=1,
        )
    session = str(uuid.uuid4())
    coordinator.lease(0, session)
    status = coordinator.heartbeat(0, session, steps=4, tokens=100)
    assert status["global_tokens"] == 100
    # Recover an older checkpoint; server never double-counts replayed windows.
    assert coordinator.heartbeat(0, session, steps=2, tokens=50)["global_tokens"] == 100
    assert coordinator.heartbeat(0, session, steps=5, tokens=120)["global_tokens"] == 120


def test_global_install_carries_local_progress_made_while_the_round_is_in_flight():
    from megatoken.training.diloco_client import DecoupledWorker

    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(6))
    optimizer = MasterAdamW(model, lr=0.01, inner_optimizer="adamw")
    try:
        (fragment,) = plan_fragments(optimizer.masters, target_bytes=1 << 20)
        worker = DecoupledWorker.__new__(DecoupledWorker)
        worker.fragments = (fragment,)
        worker.versions = [3]
        worker.awaiting_global = {0}
        # Anchor = the upload point; two more local steps then moved the weights.
        upload_point = torch.full((6,), 1.0)
        worker.baselines = [upload_point.clone()]
        install_fragment(optimizer, fragment, upload_point + 0.25)
        merged = torch.full((6,), 5.0)
        assert worker._install(0, (4, merged), optimizer)
        torch.testing.assert_close(pack_fragment(optimizer.masters, fragment), merged + 0.25)
        torch.testing.assert_close(model.weight.detach(), merged + 0.25)
        # The next pseudo-gradient is exactly the residual, against the new global.
        torch.testing.assert_close(
            worker.baselines[0] - pack_fragment(optimizer.masters, fragment),
            torch.full((6,), -0.25),
        )
        assert worker.versions == [4] and not worker.awaiting_global
        assert not worker._install(0, (4, merged), optimizer)
    finally:
        optimizer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned fragment packing needs CUDA")
def test_device_fragment_packing_matches_host_values():
    masters = {
        "a": torch.arange(12, dtype=torch.bfloat16, device="cuda").reshape(3, 4),
        "b": torch.linspace(-1, 1, 7, device="cuda"),
    }
    (fragment,) = plan_fragments(masters, target_bytes=1 << 20)
    packed = pack_fragment(masters, fragment)
    assert packed.dtype == torch.float32 and packed.is_pinned()
    expected = torch.cat(
        [
            masters[p.name].float().reshape(-1)[p.offset : p.offset + p.length].cpu()
            for p in fragment.slices
        ]
    )
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
