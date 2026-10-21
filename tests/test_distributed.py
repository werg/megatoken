"""Real two-process Gloo gates for the method-based data-parallel trainer."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from retrieval_fixture import retrieval_train_config
from torch import nn

from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig
from megatoken.training.distributed import DistributedContext
from megatoken.training.engine import train
from megatoken.training.losses import LossLedger
from megatoken.training.optimizer import MasterAdamW


def _start_group(rank: int, rendezvous: str) -> DistributedContext:
    torch.set_num_threads(1)
    # The filesystem rendezvous and Gloo traffic remain on this test machine.
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), GLOO_SOCKET_IFNAME="lo")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{rendezvous}",
        # Rank 1 waits in collectives while rank 0 alone writes checkpoints; leave
        # headroom for a loaded CI host instead of failing on a slow disk.
        timeout=timedelta(seconds=300),
    )
    return DistributedContext.initialize("cpu")


class _MaskedExperts(nn.Module):
    """Locally inactive owners exercise the same missing-gradient reduction path."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 3)
        self.experts = nn.ModuleList(nn.Linear(3, 2, bias=False) for _ in range(4))
        self.register_buffer("identity", torch.tensor([17, 23], dtype=torch.long))

    def forward(self, inputs: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        hidden = self.shared(inputs).tanh()
        output = inputs.new_zeros(*inputs.shape[:2], 2)
        for expert, module in enumerate(self.experts):
            selected = torch.where(assignments == expert)[0]
            if selected.numel():
                output = output.index_copy(0, selected, module(hidden.index_select(0, selected)))
        return output


def _data(dtype: torch.dtype):
    inputs = torch.linspace(-0.7, 0.9, 60, dtype=dtype).reshape(3, 5, 4)
    target = torch.cos(torch.arange(30, dtype=torch.float32) * 0.3).reshape(3, 5, 2)
    task_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 1, 1, 0]], dtype=torch.bool)
    calibration_mask = torch.tensor(
        [[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0]], dtype=torch.bool
    )
    return inputs, target, task_mask, calibration_mask, torch.tensor([0, 1, 2])


def _loss_terms(model: nn.Module, data, rows: list[int]):
    inputs, target, task_mask, calibration_mask, assignment = data
    selected = torch.tensor(rows, dtype=torch.long)
    output = model(inputs[selected], assignment[selected]).float()
    mask = task_mask[selected]
    calibration = calibration_mask[selected]
    return {
        "task": (
            ((output - target[selected]).square() * mask[..., None]).sum(),
            int(mask.sum()) * 2,
        ),
        "calibration": (
            ((output + 0.2).square() * calibration[..., None]).sum(),
            int(calibration.sum()) * 2,
        ),
    }


def _gradient_worker(rank: int, rendezvous: str, output_path: str) -> None:
    context = _start_group(rank, rendezvous)
    summaries = []
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(71 + rank)
            model = _MaskedExperts().to(dtype=dtype)
            context.broadcast_model(model)
            reference = copy.deepcopy(model)
            optimizer = MasterAdamW(model, lr=0.003, weight_decay=0.05)
            reference_optimizer = MasterAdamW(reference, lr=0.003, weight_decay=0.05)
            data = _data(dtype)
            weights = {"task": 1.0, "calibration": 0.3}
            try:
                # First update has uneven nonzero counts. In the next update,
                # rank 1 is completely idle but participates in all reductions.
                for partition in (([0, 1], [2]), ([0, 2], [])):
                    optimizer.zero_grad()
                    reference_optimizer.zero_grad()
                    rows = partition[rank]
                    local_terms = _loss_terms(model, data, rows) if rows else {}
                    local_counts = {
                        key: local_terms[key][1] if local_terms else 0 for key in weights
                    }
                    global_counts = context.sum_counts(local_counts)
                    ledger = LossLedger(local_counts, global_counts, weights)
                    if rows:
                        ledger.backward_loss(local_terms).backward()
                    ledger.finish()
                    # Several owners exceed this bucket: test chunked packed
                    # reductions and the small-owner concatenation path together.
                    context.synchronize_gradients(optimizer.parameters, bucket_bytes=32)
                    combined = _loss_terms(reference, data, partition[0] + partition[1])
                    expected_counts = {name: value[1] for name, value in combined.items()}
                    assert expected_counts == global_counts
                    reference_ledger = LossLedger(expected_counts, expected_counts, weights)
                    reference_ledger.backward_loss(combined).backward()
                    reference_ledger.finish()
                    grad_tolerance = (
                        dict(rtol=2e-5, atol=2e-7)
                        if dtype == torch.float32
                        else dict(rtol=0.02, atol=0.002)
                    )
                    for name, master in optimizer.masters.items():
                        expected = reference_optimizer.masters[name]
                        if expected.grad is None:
                            assert master.grad is None
                        else:
                            assert master.grad is not None and master.grad.dtype == torch.float32
                            torch.testing.assert_close(master.grad, expected.grad, **grad_tolerance)
                    assert optimizer.masters["experts.3.weight"].grad is None
                    optimizer.step(max_norm=0.5)
                    reference_optimizer.step(max_norm=0.5)
                    weight_tolerance = (
                        dict(rtol=2e-5, atol=2e-7)
                        if dtype == torch.float32
                        else dict(rtol=0.01, atol=0.001)
                    )
                    for name, master in optimizer.masters.items():
                        torch.testing.assert_close(
                            master, reference_optimizer.masters[name], **weight_tolerance
                        )
                    # All ranks must end the transaction with identical masters,
                    # including the rank that did no local backward at all.
                    gathered = context.all_gather_object(
                        {name: value.detach().clone() for name, value in optimizer.masters.items()}
                    )
                    for name in optimizer.masters:
                        torch.testing.assert_close(
                            gathered[0][name], gathered[1][name], rtol=0, atol=0
                        )
                    summaries.append({"dtype": str(dtype), "counts": global_counts})
            finally:
                optimizer.close()
                reference_optimizer.close()
        Path(output_path, f"gradients-{rank}.json").write_text(json.dumps(summaries))
    finally:
        dist.destroy_process_group()


def _engine_worker(rank: int, rendezvous: str, config_dict: dict, output_path: str) -> None:
    _start_group(rank, rendezvous)
    try:
        config = TrainConfig.from_dict(config_dict)
        first = train(config)
        resumed = train(config, resume=first["checkpoint"])
        transitioned = train(
            replace(config, phase="P1", budget=1, output=config.output + "-p1"),
            resume=resumed["checkpoint"],
            phase_transition=True,
        )
        Path(output_path, f"engine-{rank}.json").write_text(
            json.dumps(
                {
                    "first": first,
                    "resumed": resumed,
                    "transitioned": transitioned,
                }
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable"
)
def test_two_rank_variable_mask_gradient_and_master_update_matches_combined_batch(tmp_path):
    mp.spawn(
        _gradient_worker,
        args=(str(tmp_path / "gradient-rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = json.loads((tmp_path / "gradients-0.json").read_text())
    rank1 = json.loads((tmp_path / "gradients-1.json").read_text())
    assert rank0 == rank1 and len(rank0) == 4
    assert rank0[0]["counts"] == {"task": 20, "calibration": 10}


@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable"
)
def test_two_rank_full_engine_checkpoint_resume_and_phase_transition(tmp_path):
    torch.set_num_threads(1)
    config = retrieval_train_config(
        tmp_path,
        microbatch_episodes=1,
        accumulate_scored=1,
        max_accumulation_windows=1,
        gradient_checkpointing=True,
    )
    mp.spawn(
        _engine_worker,
        args=(str(tmp_path / "engine-rendezvous"), asdict(config), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = json.loads((tmp_path / "engine-0.json").read_text())
    rank1 = json.loads((tmp_path / "engine-1.json").read_text())
    assert rank0 == rank1
    assert [rank0[name]["global_step"] for name in ("first", "resumed", "transitioned")] == [
        1,
        2,
        3,
    ]
    for name in ("first", "resumed", "transitioned"):
        payload, metadata = load_checkpoint(rank0[name]["checkpoint"])
        assert len(payload["rank_state"]) == 2
        assert [state["cursor"]["rank"] for state in payload["rank_state"]] == [0, 1]
        assert payload["counters"]["P"] > 0
        assert (
            payload["counters"]["I"]
            == payload["counters"]["D"] + payload["counters"]["reconstruction_tokens"]
        )
        assert metadata["global_step"] == rank0[name]["global_step"]
    assert rank0["transitioned"]["complete"]
