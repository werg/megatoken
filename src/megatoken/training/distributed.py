"""Explicit data parallel reductions for recurrent, method-based model execution.

Every rank advances the same optimizer transaction. Empty local partitions take
part in denominator and gradient reductions, so variable episode lengths and
inactive experts never change the global objective or hang a collective.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    owns_group: bool = False

    @classmethod
    def initialize(cls, requested_device: str = "auto") -> DistributedContext:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but this PyTorch runtime has no usable GPU")
            device = torch.device("cuda", local if device.index is None else device.index)
            torch.cuda.set_device(device)
        owns = False
        if world > 1 and not dist.is_initialized():
            dist.init_process_group(
                "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=20)
            )
            owns = True
        if dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        return cls(rank, world, device, owns)

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def sum_counts(self, counts: dict[str, int]) -> dict[str, int]:
        keys = sorted(counts)
        tensor = torch.tensor([counts[k] for k in keys], dtype=torch.int64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return dict(zip(keys, tensor.cpu().tolist(), strict=True))

    def all_gather_object(self, value: Any) -> list[Any]:
        if self.world_size == 1:
            return [value]
        result: list[Any] = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def broadcast_object(self, value: Any) -> Any:
        """Rank 0's ``value`` on every rank."""
        if self.world_size == 1:
            return value
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def sum_tensors(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = sorted(values)
        if not keys:
            return {}
        flat = torch.cat(
            [values[key].detach().to(self.device, torch.float64).reshape(-1) for key in keys]
        )
        if self.world_size > 1:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        output, offset = {}, 0
        for key in keys:
            size = values[key].numel()
            output[key] = flat[offset : offset + size].reshape(values[key].shape)
            offset += size
        return output

    def broadcast_model(self, module: torch.nn.Module) -> None:
        if self.world_size > 1:
            for tensor in (*module.parameters(), *module.buffers()):
                dist.broadcast(tensor.detach(), src=0)

    def synchronize_gradients(
        self, parameters: list[torch.nn.Parameter], bucket_bytes: int = 32 << 20
    ) -> None:
        """Sum, do not average: losses already use the true global denominator."""
        if self.world_size == 1:
            return
        flags = torch.tensor(
            [p.grad is not None for p in parameters], dtype=torch.int32, device=self.device
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        flags_cpu = flags.cpu().tolist()
        bucket: list[torch.nn.Parameter] = []
        size = 0

        def reduce_bucket() -> None:
            if not bucket:
                return
            flat = torch.cat([p.grad.reshape(-1) for p in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            offset = 0
            for parameter in bucket:
                count = parameter.numel()
                parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
                offset += count
            bucket.clear()

        for parameter, used in zip(parameters, flags_cpu, strict=True):
            if not used:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            elif not parameter.grad.is_contiguous():
                # A flat view must alias the gradient, or the reduced values would be lost.
                parameter.grad = parameter.grad.contiguous()
            needed = parameter.numel() * parameter.element_size()
            if bucket and size + needed > bucket_bytes:
                reduce_bucket()
                size = 0
            # Split a large packed expert parameter rather than allocating a
            # second full 1.2-GB flattened copy just for communication.
            if needed > bucket_bytes:
                flat_gradient = parameter.grad.view(-1)
                chunk = max(1, bucket_bytes // parameter.element_size())
                for start in range(0, flat_gradient.numel(), chunk):
                    dist.all_reduce(flat_gradient[start : start + chunk], op=dist.ReduceOp.SUM)
            else:
                bucket.append(parameter)
                size += needed
        reduce_bucket()

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
