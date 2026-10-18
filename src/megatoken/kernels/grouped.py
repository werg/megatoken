"""Sorted expert dispatch without per-row weight replication.

``auto`` selects the GPU-resident Triton grouped GEMM on CUDA and the
portable differentiable ``torch`` reference on CPU. CUDA does not silently
fall back to the synchronizing reference implementation if Triton is unavailable.
The public API is identical on CPU, CUDA, training, and serving.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate(x: Tensor, weights: Tensor, expert_ids: Tensor) -> None:
    if x.ndim != 2 or weights.ndim != 3 or expert_ids.shape != x.shape[:1]:
        raise ValueError("Expected x[rows,in], weights[experts,out,in], ids[rows]")
    if x.shape[-1] != weights.shape[-1]:
        raise ValueError("Expert input dimension mismatch")
    if expert_ids.dtype != torch.long:
        raise TypeError("expert_ids must be int64")
    if x.device != weights.device or expert_ids.device != x.device:
        raise ValueError("Rows, weights, and assignments must be on the same device")


def grouped_linear_reference(x: Tensor, weights: Tensor, expert_ids: Tensor) -> Tensor:
    """Group sorted rows into GEMMs; never expand weights to [rows,out,in].

    Reading the count vector once synchronizes CUDA. This deliberate reference
    limitation is removed by the Triton backend's GPU-resident group metadata.
    """
    _validate(x, weights, expert_ids)
    if not x.shape[0]:
        return x.new_empty((0, weights.shape[1])) + weights.sum() * 0
    permutation = torch.argsort(expert_ids, stable=True)
    sorted_ids = expert_ids[permutation]
    sorted_x = x[permutation]
    counts = torch.bincount(sorted_ids, minlength=weights.shape[0])
    pieces: list[Tensor] = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            pieces.append(F.linear(sorted_x.narrow(0, start, count), weights[expert]))
            start += count
    ordered = torch.cat(pieces, dim=0)
    return torch.empty_like(ordered).index_copy(0, permutation, ordered)


def grouped_linear(
    x: Tensor, weights: Tensor, expert_ids: Tensor, backend: str = "auto"
) -> Tensor:
    if backend == "auto":
        backend = "triton" if x.is_cuda else "torch"
    if backend == "torch":
        return grouped_linear_reference(x, weights, expert_ids)
    if backend != "triton":
        raise ValueError(f"Unknown grouped GEMM backend: {backend}")
    _validate(x, weights, expert_ids)
    if not x.is_cuda:
        raise RuntimeError("The triton backend requires CUDA; choose torch for CPU reference")
    from .triton_grouped import grouped_linear_triton

    # Preserve autocast semantics. BF16 model weights avoid this conversion.
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        x, weights = x.to(dtype), weights.to(dtype)
    if x.dtype != weights.dtype:
        raise TypeError("Triton grouped GEMM needs matching activation and weight dtypes")
    return grouped_linear_triton(x, weights, expert_ids)
