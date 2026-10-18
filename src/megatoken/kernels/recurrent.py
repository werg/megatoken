"""One committed-record gated delta-rule update with independent stream states.

This new implementation defines its numerical recurrence explicitly; no absent
v12 implementation is assumed. For each stream/head, with K-normalized key k:

    T = decay * S
    residual = v - k @ T
    S_next = T + beta * outer(k, residual)
    output = q @ S_next

Pooling facets does not advance time. Exactly one call advances each site per
committed physical record, while prediction uses the separate read-only path.
"""

from __future__ import annotations

import torch
from torch import Tensor


def delta_rule_reference(
    state: Tensor, query: Tensor, key: Tensor, value: Tensor, decay: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=state.device.type, enabled=False):
        state, query, key, value, decay, beta = [
            x.to(dtype) for x in (state, query, key, value, decay, beta)
        ]
        decayed = state * decay[..., None, None]
        residual = value - torch.einsum("...k,...kv->...v", key, decayed)
        updated = decayed + beta[..., None, None] * key[..., :, None] * residual[..., None, :]
        output = torch.einsum("...k,...kv->...v", query, updated)
    return updated, output


def delta_rule_step(
    state: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    backend: str = "torch",
) -> tuple[Tensor, Tensor]:
    if state.shape[:-2] != query.shape[:-1] or query.shape != key.shape:
        raise ValueError("Query/key must have one vector per independent recurrent state")
    if (
        state.shape[-2] != key.shape[-1]
        or state.shape[:-2] != value.shape[:-1]
        or state.shape[-1] != value.shape[-1]
    ):
        raise ValueError("GDN state/key/value geometry mismatch")
    if decay.shape != state.shape[:-2] or beta.shape != decay.shape:
        raise ValueError("Decay and write gates must have one scalar per stream/head")
    if backend == "torch":
        return delta_rule_reference(state, query, key, value, decay, beta)
    if backend != "triton":
        raise ValueError(f"Unknown recurrent backend: {backend}")
    if not state.is_cuda:
        raise RuntimeError("Triton recurrence requires CUDA")
    from .triton_recurrent import delta_rule_triton

    return delta_rule_triton(state, query, key, value, decay, beta)


def delta_rule_read(state: Tensor, query: Tensor) -> Tensor:
    with torch.autocast(device_type=state.device.type, enabled=False):
        return torch.einsum("...k,...kv->...v", query.float(), state.float())
