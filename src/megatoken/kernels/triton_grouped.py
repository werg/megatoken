"""GPU-resident variable-row grouped GEMM with a complete autograd path.

Inspired by Triton's documented blocked matrix multiplication API:
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
No architecture-specific speed claim is made before benchmarking Blackwell.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError("Install the CUDA extra (including Triton) for this backend") from exc


@triton.jit
def _grouped_mm(
    X,
    W,
    OFFSETS,
    TILE_OFFSETS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    W_SE: tl.constexpr,
    W_SN: tl.constexpr,
    W_SK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BE: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    total_tiles = tl.load(TILE_OFFSETS + E)
    if tile_m < total_tiles:
        es = tl.arange(0, BE)
        starts = tl.load(TILE_OFFSETS + es, mask=es < E, other=2147483647)
        expert = tl.sum((starts <= tile_m).to(tl.int32)) - 1
        row_start = tl.load(OFFSETS + expert)
        row_end = tl.load(OFFSETS + expert + 1)
        first_tile = tl.load(TILE_OFFSETS + expert)
        rows = row_start + (tile_m - first_tile) * BM + tl.arange(0, BM)
        cols = tile_n * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for block in range(tl.cdiv(K, BK)):
            inner = block * BK + ks
            a = tl.load(
                X + rows[:, None] * K + inner[None, :],
                (rows[:, None] < row_end) & (inner[None, :] < K),
                other=0,
            )
            b = tl.load(
                W + expert * W_SE + cols[None, :] * W_SN + inner[:, None] * W_SK,
                (cols[None, :] < N) & (inner[:, None] < K),
                other=0,
            )
            acc += tl.dot(a, b, input_precision="ieee")
        tl.store(
            Y + rows[:, None] * N + cols[None, :],
            acc,
            (rows[:, None] < row_end) & (cols[None, :] < N),
        )


@triton.jit
def _grouped_weight_grad(
    X,
    DY,
    OFFSETS,
    DW,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    expert = tl.program_id(0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.program_id(2) * BK + tl.arange(0, BK)
    start = tl.load(OFFSETS + expert)
    end = tl.load(OFFSETS + expert + 1)
    rs = tl.arange(0, BR)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for block in range(tl.cdiv(end - start, BR)):
        rows = start + block * BR + rs
        dy = tl.load(
            DY + rows[None, :] * N + ns[:, None], (rows[None, :] < end) & (ns[:, None] < N), other=0
        )
        x = tl.load(
            X + rows[:, None] * K + ks[None, :], (rows[:, None] < end) & (ks[None, :] < K), other=0
        )
        acc += tl.dot(dy, x, input_precision="ieee")
    tl.store(
        DW + expert * N * K + ns[:, None] * K + ks[None, :],
        acc,
        (ns[:, None] < N) & (ks[None, :] < K),
    )


def _metadata(expert_ids: Tensor, experts: int, block_rows: int = 32) -> tuple[Tensor, Tensor]:
    # CUDA bincount may discover its output size through a host-visible maximum.
    # The expert universe is already known: fixed-size scatter keeps allocation
    # and group metadata independent of data-dependent CPU synchronization.
    counts = torch.zeros(experts, dtype=torch.long, device=expert_ids.device)
    counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
    zero = counts.new_zeros(1)
    offsets = torch.cat((zero, counts.cumsum(0)))
    tile_offsets = torch.cat((zero, ((counts + block_rows - 1) // block_rows).cumsum(0)))
    return offsets, tile_offsets


def _launch(x: Tensor, weights: Tensor, offsets: Tensor, tile_offsets: Tensor) -> Tensor:
    rows, inner = x.shape
    experts, output, _ = weights.shape
    y = x.new_empty((rows, output))
    if rows:
        _grouped_mm[(triton.cdiv(rows, 32) + experts, triton.cdiv(output, 64))](
            x,
            weights,
            offsets,
            tile_offsets,
            y,
            output,
            inner,
            experts,
            *weights.stride(),
            32,
            64,
            32,
            triton.next_power_of_2(experts),
            num_warps=4,
        )
    return y


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weights: Tensor, expert_ids: Tensor) -> Tensor:
        permutation = torch.argsort(expert_ids, stable=True)
        sorted_x = x[permutation].contiguous()
        offsets, tile_offsets = _metadata(expert_ids, weights.shape[0])
        ordered = _launch(sorted_x, weights, offsets, tile_offsets)
        ctx.save_for_backward(sorted_x, weights, permutation, offsets, tile_offsets)
        return torch.empty_like(ordered).index_copy_(0, permutation, ordered)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, weights, permutation, offsets, tile_offsets = ctx.saved_tensors
        dy = grad_output[permutation].contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            sorted_dx = _launch(dy, weights.transpose(-1, -2), offsets, tile_offsets)
            dx = torch.empty_like(sorted_dx).index_copy_(0, permutation, sorted_dx)
        if ctx.needs_input_grad[1]:
            experts, output, inner = weights.shape
            dw = torch.empty_like(weights, memory_format=torch.contiguous_format)
            _grouped_weight_grad[(experts, triton.cdiv(output, 32), triton.cdiv(inner, 32))](
                x,
                dy,
                offsets,
                dw,
                output,
                inner,
                32,
                32,
                32,
                num_warps=4,
            )
        return dx, dw, None


def grouped_linear_triton(x: Tensor, weights: Tensor, expert_ids: Tensor) -> Tensor:
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("Triton grouped GEMM supports FP32/FP16/BF16, not FP8")
    return _GroupedLinear.apply(x.contiguous(), weights, expert_ids)
