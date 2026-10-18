"""Exact masked categorical distributions with bounded logit memory.

The vocabulary multiplication is still full vocabulary. Tiling controls memory;
it does not pretend to remove the arithmetic. Backward recomputes logit tiles so
autograd never saves a length-by-vocabulary tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.autograd import Function


def _dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _gemm(left: Tensor, right: Tensor, dtype: torch.dtype, tensorcore: bool) -> Tensor:
    if (
        tensorcore
        and left.is_cuda
        and left.dtype in (torch.float16, torch.bfloat16)
        and right.dtype == left.dtype
    ):
        # PyTorch >=2.10: BF16/FP16 Tensor Core products with an FP32 output,
        # avoiding both an FP32 weight copy and a rounded BF16 logit tensor.
        return torch.mm(left, right, out_dtype=torch.float32)
    return left.to(dtype) @ right.to(dtype)


def _validate(hidden: Tensor, weight: Tensor, target: Tensor | None, valid: Tensor) -> None:
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(
            "categorical hidden/weight must have shapes [positions,rank]/[vocabulary,rank]"
        )
    if valid.shape != (weight.shape[0],) or valid.dtype != torch.bool:
        raise ValueError("valid vocabulary mask must be a bool vector")
    if not bool(valid.any()):
        raise ValueError("categorical distribution has empty support")
    if target is not None:
        if target.shape != (hidden.shape[0],) or target.dtype != torch.long:
            raise ValueError("one int64 target is required per free position")
        if target.numel() and (bool((target < 0).any()) or bool((target >= weight.shape[0]).any())):
            raise ValueError("categorical target outside vocabulary")
        if target.numel() and not bool(valid[target].all()):
            raise ValueError("categorical target outside masked support")


class _TiledCrossEntropy(Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, valid, position_tile, vocabulary_tile, tensorcore):
        _validate(hidden, weight, target, valid)
        dtype = _dtype(hidden)
        logz = torch.empty(hidden.shape[0], device=hidden.device, dtype=dtype)
        target_logit = torch.zeros_like(logz)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, hidden.shape[0], position_tile):
                end = min(start + position_tile, hidden.shape[0])
                h = hidden[start:end]
                z = torch.full((end - start,), -torch.inf, device=h.device, dtype=dtype)
                for row in range(0, weight.shape[0], vocabulary_tile):
                    upper = min(row + vocabulary_tile, weight.shape[0])
                    logits = _gemm(h, weight[row:upper].T, dtype, tensorcore)
                    logits = logits.masked_fill(~valid[row:upper][None], -torch.inf)
                    z = torch.logaddexp(z, torch.logsumexp(logits, dim=-1))
                    local_target = target[start:end] - row
                    selected = (local_target >= 0) & (local_target < upper - row)
                    local_logit = logits.gather(1, local_target.clamp(0, upper - row - 1)[:, None])[
                        :, 0
                    ]
                    target_logit[start:end] = torch.where(
                        selected, local_logit, target_logit[start:end]
                    )
                logz[start:end] = z
        ctx.save_for_backward(hidden, weight, target, valid, logz)
        ctx.position_tile = position_tile
        ctx.vocabulary_tile = vocabulary_tile
        ctx.tensorcore = tensorcore
        return logz - target_logit

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, target, valid, logz = ctx.saved_tensors
        dtype = _dtype(hidden)
        grad_hidden = torch.zeros_like(hidden, dtype=dtype) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight, dtype=dtype) if ctx.needs_input_grad[1] else None
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, hidden.shape[0], ctx.position_tile):
                end = min(start + ctx.position_tile, hidden.shape[0])
                h = hidden[start:end]
                scale = grad_output[start:end].to(dtype)
                for row in range(0, weight.shape[0], ctx.vocabulary_tile):
                    upper = min(row + ctx.vocabulary_tile, weight.shape[0])
                    w = weight[row:upper]
                    logits = _gemm(h, w.T, dtype, ctx.tensorcore).masked_fill(
                        ~valid[row:upper][None], -torch.inf
                    )
                    derivative = (logits - logz[start:end, None]).exp()
                    local_target = target[start:end] - row
                    selected = (local_target >= 0) & (local_target < upper - row)
                    locations = torch.arange(end - start, device=h.device)
                    derivative[locations, local_target.clamp(0, upper - row - 1)] -= selected.to(
                        dtype
                    )
                    derivative *= scale[:, None]
                    # Match ordinary mixed-precision GEMM backward: its input
                    # gradients have the operand dtype; accumulation across tiles
                    # and optimizer master weights remain FP32.
                    if (
                        ctx.tensorcore
                        and hidden.is_cuda
                        and hidden.dtype in (torch.float16, torch.bfloat16)
                    ):
                        derivative = derivative.to(hidden.dtype)
                    if grad_hidden is not None:
                        grad_hidden[start:end] += _gemm(derivative, w, dtype, ctx.tensorcore)
                    if grad_weight is not None:
                        grad_weight[row:upper] += _gemm(derivative.T, h, dtype, ctx.tensorcore)
        return (
            None if grad_hidden is None else grad_hidden.to(hidden.dtype),
            None if grad_weight is None else grad_weight.to(weight.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def tiled_cross_entropy(
    hidden: Tensor,
    weight: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    position_tile: int = 128,
    vocabulary_tile: int = 4096,
    tensorcore: bool = True,
) -> Tensor:
    """Unreduced FP32 CE; first-order backward uses at most one logit tile."""
    if position_tile <= 0 or vocabulary_tile <= 0:
        raise ValueError("tile sizes must be positive")
    return _TiledCrossEntropy.apply(
        hidden, weight, targets, valid, position_tile, vocabulary_tile, tensorcore
    )


@torch.no_grad()
def tiled_argmax(
    hidden: Tensor,
    weight: Tensor,
    valid: Tensor,
    *,
    position_tile: int = 128,
    vocabulary_tile: int = 4096,
    tensorcore: bool = True,
) -> Tensor:
    """Exact deterministic full-vocabulary argmax; equal logits prefer lower IDs."""
    if position_tile <= 0 or vocabulary_tile <= 0:
        raise ValueError("tile sizes must be positive")
    _validate(hidden, weight, None, valid)
    output = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.long)
    dtype = _dtype(hidden)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        for start in range(0, hidden.shape[0], position_tile):
            end = min(start + position_tile, hidden.shape[0])
            h = hidden[start:end]
            best = torch.full((end - start,), -torch.inf, dtype=dtype, device=h.device)
            ids = torch.zeros(end - start, dtype=torch.long, device=h.device)
            for row in range(0, weight.shape[0], vocabulary_tile):
                upper = min(row + vocabulary_tile, weight.shape[0])
                scores = _gemm(h, weight[row:upper].T, dtype, tensorcore).masked_fill(
                    ~valid[row:upper][None], -torch.inf
                )
                values, indices = scores.max(dim=-1)
                wins = values > best
                ids = torch.where(wins, indices + row, ids)
                best = torch.maximum(best, values)
            output[start:end] = ids
    return output


class TiedVocabulary(nn.Module):
    """One rank-factorized output matrix for logits and public value embeddings."""

    def __init__(
        self, width: int, vocabulary: int, rank: int, *, vocabulary_tile: int = 4096
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(width, rank, bias=False)
        self.embedding = nn.Embedding(vocabulary, rank)
        self.vocabulary_tile = vocabulary_tile
        nn.init.normal_(self.embedding.weight, std=rank**-0.5)

    def embed(self, ids: Tensor) -> Tensor:
        return torch.nn.functional.linear(self.embedding(ids), self.projection.weight.T)

    def loss(self, states: Tensor, ids: Tensor, valid: Tensor) -> Tensor:
        return tiled_cross_entropy(
            self.projection(states),
            self.embedding.weight,
            ids,
            valid,
            vocabulary_tile=self.vocabulary_tile,
        )

    def argmax(self, states: Tensor, valid: Tensor) -> Tensor:
        return tiled_argmax(
            self.projection(states),
            self.embedding.weight,
            valid,
            vocabulary_tile=self.vocabulary_tile,
        )
