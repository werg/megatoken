"""Packed v10 transducer fields, quantized and decoded a whole batch of rows at a time.

Values are E2M1 FP4 nibbles (two per byte) with one E4M3 scale per 16 elements
and one FP32 scale per row. Keys, departures and destinations are E4M3 bytes
with one FP32 scale per 128 elements. Everything here is vectorized over rows
and runs on whatever device the tensors live on; there is no per-row Python.

Stochastic rounding draws from a counter-based hash of (row seed, element), so
a row's rounding depends only on its seed: the same edge, field, update and
worker always round identically, on any device and in any batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import Tensor

FP4_GROUP = 16
FP8_GROUP = 128
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP8_MAX = 448.0
_M32 = 0xFFFFFFFF
_FIELD_CODES = {"departure": 1, "key": 2, "value": 3, "destination": 4, "bias": 5}


def rounding_seed(*parts: object) -> int:
    """A stable 32-bit seed for identities that are not plain integers."""
    digest = sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _mul32(x: Tensor, constant: int) -> Tensor:
    """(x * constant) mod 2**32 without leaving the exact int64 range."""
    low = (x * (constant & 0xFFFF)) & _M32
    high = ((x * (constant >> 16)) & 0xFFFF) << 16
    return (low + high) & _M32


def _mix32(x: Tensor) -> Tensor:
    x = x & _M32
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def row_seeds(ids: Tensor, field: str, step: int, worker: int) -> Tensor:
    """One rounding seed per edge for one (field, update, worker) publication."""
    code = _FIELD_CODES[field]
    seeds = _mix32(ids.to(torch.int64) ^ (code * 0x9E3779B1 & _M32))
    seeds = _mix32(seeds + (step & _M32))
    return _mix32(seeds + ((worker * 0x85EBCA77) & _M32))


def uniforms(seeds: Tensor, count: int) -> Tensor:
    """[rows, count] uniforms in (0, 1) from per-row seeds."""
    index = torch.arange(count, device=seeds.device, dtype=torch.int64)
    x = _mix32(seeds[:, None] + _mul32(index, 0x9E3779B9)[None])
    return (x.to(torch.float64) + 0.5).div(2.0**32).to(torch.float32)


def unit(values: Tensor, dim: int = -1) -> Tensor:
    """Normalize addresses onto the unit sphere; zero or nonfinite rows are errors."""
    norm = torch.linalg.vector_norm(values.float(), dim=dim, keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise FloatingPointError("Nonfinite transducer address")
    if bool((norm < 1e-12).any()):
        raise ValueError("Zero transducer address")
    return values / norm.to(values.dtype)


def packed_unit(values: Tensor) -> Tensor:
    """``unit`` without the host checks, for decoded rows validated when they were packed."""
    return values / torch.linalg.vector_norm(values.float(), dim=-1, keepdim=True).clamp_min(
        1e-12
    ).to(values.dtype)


_TABLES: dict[tuple[str, torch.device], Tensor] = {}


def _table(name: str, device: torch.device) -> Tensor:
    """Small constant lookup tables, built once per device (no per-call host copy)."""
    key = name, torch.device(device)
    table = _TABLES.get(key)
    if table is None:
        if name == "fp8":
            codes = torch.arange(256, dtype=torch.int16).to(torch.uint8)
            values = codes.view(torch.float8_e4m3fn).float()
            table = torch.unique(values[torch.isfinite(values)]).sort().values
        elif name == "fp4":
            table = torch.tensor(_FP4_LEVELS)
        else:  # signed E2M1 by nibble: bit 3 is the sign
            levels = torch.tensor(_FP4_LEVELS)
            table = torch.cat((levels, -levels))
        table = _TABLES[key] = table.to(device)
    return table


def _fp8_levels(device: torch.device) -> Tensor:
    return _table("fp8", device)


def _stochastic_fp8(normalized: Tensor, u: Tensor | None) -> Tensor:
    if u is None:
        return normalized.to(torch.float8_e4m3fn)
    levels = _fp8_levels(normalized.device)
    upper = torch.searchsorted(levels, normalized.contiguous()).clamp(1, levels.numel() - 1)
    low, high = levels[upper - 1], levels[upper]
    p = ((normalized - low) / (high - low)).clamp(0, 1)
    return torch.where(u < p, high, low).to(torch.float8_e4m3fn)


@dataclass(frozen=True)
class FP8Rows:
    codes: Tensor  # [rows, dim] float8_e4m3fn
    scales: Tensor  # [rows, ceil(dim / 128)] float32

    def decode(self) -> Tensor:
        rows, dim = self.codes.shape
        groups = self.scales.shape[1]
        values = self.codes.float()
        if dim % FP8_GROUP:
            values = torch.nn.functional.pad(values, (0, groups * FP8_GROUP - dim))
        scaled = values.reshape(rows, groups, FP8_GROUP) * self.scales[..., None]
        return scaled.reshape(rows, groups * FP8_GROUP)[:, :dim]


def pack_fp8(values: Tensor, seeds: Tensor | None = None) -> FP8Rows:
    """E4M3 rows with per-128 FP32 scales; unbiased rounding when seeds are given."""
    x = values.detach().float()
    if x.ndim != 2 or not bool(torch.isfinite(x).all()):
        raise ValueError("FP8 packing needs finite [rows, dim] values")
    rows, dim = x.shape
    groups = -(-dim // FP8_GROUP)
    padded = torch.nn.functional.pad(x, (0, groups * FP8_GROUP - dim))
    blocks = padded.reshape(rows, groups, FP8_GROUP)
    scales = (blocks.abs().amax(dim=2) / _FP8_MAX).clamp_min(1e-30)
    normalized = (blocks / scales[..., None]).clamp(-_FP8_MAX, _FP8_MAX).reshape(rows, -1)[:, :dim]
    u = None if seeds is None else uniforms(seeds, dim)
    return FP8Rows(_stochastic_fp8(normalized, u), scales)


@dataclass(frozen=True)
class FP4Rows:
    codes: Tensor  # [rows, ceil(dim / 2)] uint8, element 2i in the low nibble
    group_scales: Tensor  # [rows, ceil(dim / 16)] float8_e4m3fn
    row_scales: Tensor  # [rows] float32

    def decode(self, dim: int) -> Tensor:
        rows, groups = self.codes.shape[0], self.group_scales.shape[1]
        nibbles = torch.stack((self.codes & 15, self.codes >> 4), dim=-1).reshape(rows, -1)
        signed = _table("fp4-signed", self.codes.device)[nibbles.int()]
        if signed.shape[1] != groups * FP4_GROUP:
            signed = torch.nn.functional.pad(signed, (0, groups * FP4_GROUP - signed.shape[1]))
        grouped = signed.reshape(rows, groups, FP4_GROUP) * self.group_scales.float()[..., None]
        return (grouped * self.row_scales[:, None, None]).reshape(rows, groups * FP4_GROUP)[:, :dim]


def _round_up_e4m3(scales: Tensor) -> Tensor:
    """The smallest representable E4M3 value at or above each scale."""
    levels = _fp8_levels(scales.device)
    positive = levels[levels > 0]
    index = torch.searchsorted(positive, scales.contiguous()).clamp_max(positive.numel() - 1)
    return positive[index]


def pack_fp4(values: Tensor, seeds: Tensor | None = None) -> FP4Rows:
    """Signed E2M1 rows. Group scales round up, so no element is clipped."""
    x = values.detach().float()
    if x.ndim != 2 or not bool(torch.isfinite(x).all()):
        raise ValueError("FP4 packing needs finite [rows, dim] values")
    rows, dim = x.shape
    groups = -(-dim // FP4_GROUP)
    blocks = torch.nn.functional.pad(x, (0, groups * FP4_GROUP - dim)).reshape(
        rows, groups, FP4_GROUP
    )
    maximum = blocks.abs().amax(dim=(1, 2))
    row_scales = torch.where(maximum > 0, maximum / 6.0, torch.ones_like(maximum))
    group_scales = _round_up_e4m3(
        (blocks.abs().amax(dim=2) / (6.0 * row_scales[:, None])).clamp_min(2.0**-9)
    )
    normalized = (blocks.abs() / (row_scales[:, None, None] * group_scales[..., None])).reshape(
        rows, -1
    )
    normalized = normalized.clamp(max=6.0)
    levels = _table("fp4", x.device)
    upper = torch.searchsorted(levels, normalized.contiguous()).clamp(1, len(_FP4_LEVELS) - 1)
    low, high = levels[upper - 1], levels[upper]
    if seeds is None:
        choose_up = normalized - low >= high - normalized
    else:
        choose_up = uniforms(seeds, normalized.shape[1]) < (normalized - low) / (high - low)
    code = torch.where(choose_up, upper, upper - 1).to(torch.uint8)
    negative = torch.nn.functional.pad(x, (0, groups * FP4_GROUP - dim)) < 0
    code = (code | (negative.to(torch.uint8) << 3))[:, :dim]
    if dim % 2:
        code = torch.nn.functional.pad(code, (0, 1))
    pairs = code.reshape(rows, -1, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    return FP4Rows(packed, group_scales.to(torch.float8_e4m3fn), row_scales)


def fp8_nbytes(dim: int) -> int:
    return dim + 4 * -(-dim // FP8_GROUP)


def fp4_nbytes(dim: int) -> int:
    return -(-dim // 2) + -(-dim // FP4_GROUP) + 4


def straight_through(master: Tensor, decoded: Tensor) -> Tensor:
    """Forward reads exactly the packed row; backward reaches the FP32 master."""
    if master.shape != decoded.shape:
        raise ValueError("Master and packed field shapes disagree")
    return master + (decoded.to(master.dtype) - master).detach()
