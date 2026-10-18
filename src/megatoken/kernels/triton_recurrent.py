"""Fused FP32 per-record recurrence with analytic backward, no temporal scan."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError("Install Triton to use the CUDA recurrence backend") from exc


@triton.jit
def _step(
    S,
    Q,
    K,
    V,
    A,
    B,
    NEXT_STATE,
    Y,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    index = tl.program_id(0)
    ks, vs = tl.arange(0, BK), tl.arange(0, BV)
    state = tl.load(
        S + index * DK * DV + ks[:, None] * DV + vs[None, :],
        (ks[:, None] < DK) & (vs[None, :] < DV),
        other=0,
    ).to(tl.float32)
    key = tl.load(K + index * DK + ks, ks < DK, other=0).to(tl.float32)
    query = tl.load(Q + index * DK + ks, ks < DK, other=0).to(tl.float32)
    value = tl.load(V + index * DV + vs, vs < DV, other=0).to(tl.float32)
    decay, beta = tl.load(A + index).to(tl.float32), tl.load(B + index).to(tl.float32)
    decayed = state * decay
    residual = value - tl.sum(key[:, None] * decayed, 0)
    updated = decayed + beta * key[:, None] * residual[None, :]
    output = tl.sum(query[:, None] * updated, 0)
    tl.store(
        NEXT_STATE + index * DK * DV + ks[:, None] * DV + vs[None, :],
        updated,
        (ks[:, None] < DK) & (vs[None, :] < DV),
    )
    tl.store(Y + index * DV + vs, output, vs < DV)


@triton.jit
def _backward(
    S,
    Q,
    K,
    V,
    A,
    B,
    DS_OUT,
    DY,
    DS,
    DQ,
    DKEY,
    DVALUE,
    DA,
    DB,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    index = tl.program_id(0)
    ks, vs = tl.arange(0, BK), tl.arange(0, BV)
    mask = (ks[:, None] < DK) & (vs[None, :] < DV)
    address = index * DK * DV + ks[:, None] * DV + vs[None, :]
    state = tl.load(S + address, mask, other=0).to(tl.float32)
    key = tl.load(K + index * DK + ks, ks < DK, other=0).to(tl.float32)
    query = tl.load(Q + index * DK + ks, ks < DK, other=0).to(tl.float32)
    value = tl.load(V + index * DV + vs, vs < DV, other=0).to(tl.float32)
    decay, beta = tl.load(A + index).to(tl.float32), tl.load(B + index).to(tl.float32)
    dout = tl.load(DS_OUT + address, mask, other=0).to(tl.float32)
    dy = tl.load(DY + index * DV + vs, vs < DV, other=0).to(tl.float32)
    decayed = state * decay
    residual = value - tl.sum(key[:, None] * decayed, 0)
    updated = decayed + beta * key[:, None] * residual[None, :]
    grad = dout + query[:, None] * dy[None, :]
    dr = beta * tl.sum(grad * key[:, None], 0)
    dt = grad - key[:, None] * dr[None, :]
    dq = tl.sum(updated * dy[None, :], 1)
    dk = beta * tl.sum(grad * residual[None, :], 1) - tl.sum(decayed * dr[None, :], 1)
    da = tl.sum(tl.sum(dt * state, 0), 0)
    db = tl.sum(tl.sum(grad * key[:, None] * residual[None, :], 0), 0)
    tl.store(DS + address, decay * dt, mask)
    tl.store(DQ + index * DK + ks, dq, ks < DK)
    tl.store(DKEY + index * DK + ks, dk, ks < DK)
    tl.store(DVALUE + index * DV + vs, dr, vs < DV)
    tl.store(DA + index, da)
    tl.store(DB + index, db)


class _DeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, query, key, value, decay, beta):
        inputs = tuple(x.float().contiguous() for x in (state, query, key, value, decay, beta))
        state, query, key, value, decay, beta = inputs
        output_state, output = torch.empty_like(state), torch.empty_like(value)
        dk, dv = state.shape[-2:]
        _step[(decay.numel(),)](
            *inputs,
            output_state,
            output,
            dk,
            dv,
            triton.next_power_of_2(dk),
            triton.next_power_of_2(dv),
        )
        ctx.save_for_backward(*inputs)
        return output_state, output

    @staticmethod
    def backward(ctx, grad_state, grad_output):
        inputs = ctx.saved_tensors
        grads = tuple(torch.empty_like(x) for x in inputs)
        state, _, _, _, decay, _ = inputs
        dk, dv = state.shape[-2:]
        _backward[(decay.numel(),)](
            *inputs,
            grad_state.contiguous(),
            grad_output.contiguous(),
            *grads,
            dk,
            dv,
            triton.next_power_of_2(dk),
            triton.next_power_of_2(dv),
        )
        return grads


def delta_rule_triton(
    state: Tensor, query: Tensor, key: Tensor, value: Tensor, decay: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    return _DeltaRule.apply(state, query, key, value, decay, beta)
