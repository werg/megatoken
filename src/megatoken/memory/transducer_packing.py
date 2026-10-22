"""Packed v10 transducer fields: actual E2M1 nibbles and E4M3 bytes.

All quantization runs on CPU at publication barriers. Reads decode immutable
codes without changing scales or advancing any rounding generator.
"""

from __future__ import annotations

import base64
import math
import struct
from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import Tensor

_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def rounding_seed(*parts: object) -> int:
    digest = sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _finite(values: Tensor) -> Tensor:
    values = torch.as_tensor(values, dtype=torch.float32).detach().flatten().cpu()
    if not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("Packed fields require finite, nonempty FP32 vectors")
    return values


def unit(values: Tensor) -> Tensor:
    """Keep location and key addresses on the sphere; never hide NaNs."""
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite transducer address")
    norm = torch.linalg.vector_norm(values.float())
    if not bool(torch.isfinite(norm)) or float(norm) < 1e-12:
        raise ValueError("Zero transducer address")
    return values / norm.to(values.dtype)


def _fp8_bytes(x: Tensor) -> bytes:
    return x.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()


def _decode_fp8(payload: bytes) -> Tensor:
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(
        torch.float8_e4m3fn
    ).float()


@dataclass(frozen=True)
class PackedField:
    """Portable, precisely sized quantized field; neither codes nor scales are FP32 masters."""

    format: str
    size: int
    codes: bytes
    scales: bytes
    row_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.size < 1 or not math.isfinite(self.row_scale) or self.row_scale <= 0:
            raise ValueError("Invalid packed row geometry or scale")
        if self.format == "e2m1":
            if len(self.codes) != (self.size + 1) // 2:
                raise ValueError("FP4 row must pack two elements per byte")
            if len(self.scales) != (self.size + 15) // 16:
                raise ValueError("FP4 row requires one E4M3 scale per 16 values")
        elif self.format == "e4m3":
            if len(self.codes) != self.size:
                raise ValueError("FP8 row requires one byte per element")
            if len(self.scales) != 4 * ((self.size + 127) // 128):
                raise ValueError("FP8 row requires one FP32 scale per 128 values")
        else:
            raise ValueError("Unknown transducer packed field")

    @property
    def nbytes(self) -> int:
        return len(self.codes) + len(self.scales) + 4

    def decode(self, *, device: torch.device | str | None = None) -> Tensor:
        if self.format == "e4m3":
            scales = torch.tensor(
                struct.unpack("<" + "f" * (len(self.scales) // 4), self.scales)
            ).repeat_interleave(128)[: self.size]
            result = _decode_fp8(self.codes) * scales
        else:
            packed = torch.frombuffer(bytearray(self.codes), dtype=torch.uint8)
            code = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(-1)[: self.size]
            values = _FP4_LEVELS[(code & 7).long()]
            signs = torch.where(code & 8 != 0, -1.0, 1.0)
            scales = _decode_fp8(self.scales).repeat_interleave(16)[: self.size]
            result = values * signs * scales * self.row_scale
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite packed transducer field")
        return result.to(device=device) if device is not None else result

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "size": self.size,
            "codes": base64.b64encode(self.codes).decode("ascii"),
            "scales": base64.b64encode(self.scales).decode("ascii"),
            "row_scale": self.row_scale,
        }

    @classmethod
    def from_dict(cls, value: dict) -> PackedField:
        return cls(
            format=value["format"],
            size=int(value["size"]),
            codes=base64.b64decode(value["codes"], validate=True),
            scales=base64.b64decode(value["scales"], validate=True),
            row_scale=float(value["row_scale"]),
        )


def pack_fp8(values: Tensor, *, stochastic_seed: int | None = None) -> PackedField:
    """E4M3FN elements and per-128 FP32 scales, with optional unbiased rounding."""
    x = _finite(values)
    count = (x.numel() + 127) // 128
    padded = torch.nn.functional.pad(x, (0, count * 128 - x.numel()))
    groups = padded.reshape(count, 128)
    scales = (groups.abs().amax(dim=1) / 448.0).clamp_min(1e-30)
    normalized = (groups / scales[:, None]).clamp(-448.0, 448.0)
    if stochastic_seed is not None:
        # Stochastic FP8 rounding requires adjacent representable neighbours.
        # torch.nextafter does not support float8; use the sorted finite codebook.
        codes = torch.arange(256, dtype=torch.uint8)
        values8 = codes.view(torch.float8_e4m3fn).float()
        levels = torch.unique(values8[torch.isfinite(values8)]).sort().values
        flat = normalized.flatten()
        upper = torch.searchsorted(levels, flat).clamp(1, len(levels) - 1)
        low, high = levels[upper - 1], levels[upper]
        p = ((flat - low) / (high - low)).clamp(0, 1)
        rng = torch.Generator().manual_seed(stochastic_seed)
        normalized = torch.where(
            torch.rand(flat.shape, generator=rng) < p, high, low
        ).reshape_as(normalized)
    return PackedField(
        "e4m3",
        x.numel(),
        _fp8_bytes(normalized.flatten()[: x.numel()]),
        struct.pack("<" + "f" * count, *scales.tolist()),
    )


def pack_fp4(values: Tensor, *, stochastic_seed: int | None = None) -> PackedField:
    """Signed E2M1 codes, one E4M3 scale/16 and one FP32 row scale."""
    x = _finite(values)
    count = (x.numel() + 15) // 16
    groups = torch.nn.functional.pad(x, (0, count * 16 - x.numel())).reshape(count, 16)
    maximum = float(groups.abs().amax())
    row_scale = maximum / 6.0 if maximum else 1.0
    group_scale = (groups.abs().amax(dim=1) / (6 * row_scale)).clamp_min(2**-9)
    encoded_scales = _decode_fp8(_fp8_bytes(group_scale))
    normalized = (groups / (row_scale * encoded_scales[:, None])).abs().flatten()
    normalized = normalized.clamp(max=6.0)
    upper = torch.searchsorted(_FP4_LEVELS, normalized).clamp(1, 7)
    lower_values, upper_values = _FP4_LEVELS[upper - 1], _FP4_LEVELS[upper]
    if stochastic_seed is None:
        choose_up = normalized - lower_values >= upper_values - normalized
    else:
        rng = torch.Generator().manual_seed(stochastic_seed)
        choose_up = torch.rand(normalized.shape, generator=rng) < (
            (normalized - lower_values) / (upper_values - lower_values)
        )
    code = torch.where(choose_up, upper, upper - 1).to(torch.uint8)
    code |= (groups.flatten() < 0).to(torch.uint8) << 3
    code = code[: x.numel()]
    if code.numel() % 2:
        code = torch.cat((code, torch.zeros(1, dtype=torch.uint8)))
    bytes4 = code.reshape(-1, 2)
    packed = bytes4[:, 0] | (bytes4[:, 1] << 4)
    return PackedField("e2m1", x.numel(), packed.numpy().tobytes(), _fp8_bytes(group_scale),
                       row_scale)


def straight_through(master: Tensor, packed: PackedField) -> Tensor:
    """Forward/recomputation read exactly the packed row; backward targets the master."""
    decoded = packed.decode(device=master.device).to(master.dtype)
    if master.shape != decoded.shape:
        raise ValueError("Master and packed field shapes disagree")
    return master + (decoded - master).detach()
