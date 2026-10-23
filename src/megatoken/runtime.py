"""Runtime diagnostics and precision settings, with no GPU assumptions at import."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        # Deterministic cuBLAS needs a fixed workspace, read when the first CUDA matmul
        # creates its handle; without it deterministic algorithms raise on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if not torch.cuda.is_available():
        # The portable CPU reference does not depend on NNPACK CPU discovery
        # (which is unavailable in some containers with a restricted /proc).
        torch.backends.nnpack.set_flags(False)
    if torch.cuda.is_available():
        # TF32 is only a fallback for explicit FP32 matmuls. BF16 remains the
        # model's launch precision; FP8 is not silently enabled on Blackwell.
        torch.set_float32_matmul_precision("highest" if deterministic else "high")


def capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    values = state["numpy"]
    np.random.set_state((values[0], np.array(values[1], dtype=np.uint32), *values[2:]))
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError(
                "RNG checkpoint GPU count differs; exact resume needs the original topology"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass(frozen=True)
class DeviceReport:
    index: int
    name: str
    capability: tuple[int, int]
    memory_bytes: int
    bf16: bool
    blackwell: bool
    matmul_ok: bool


def doctor(
    require_cuda: bool = False, require_blackwell: bool = False, *, probe_cuda: bool = True
) -> dict[str, Any]:
    devices = []
    errors = []
    warnings = []
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 10):
        errors.append("Use PyTorch 2.10 or newer; the validated CPU reference uses 2.10")
    for index in range(
        torch.cuda.device_count() if probe_cuda or require_cuda or require_blackwell else 0
    ):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        with torch.cuda.device(index):
            bf16 = torch.cuda.is_bf16_supported()
            valid = False
            try:
                matrix = torch.randn(128, 128, device=f"cuda:{index}", dtype=torch.bfloat16)
                product = matrix @ matrix.T
                torch.cuda.synchronize(index)
                valid = bool(torch.isfinite(product).all())
                if not valid:
                    errors.append(f"CUDA device {index} produced nonfinite BF16 matrix products")
            except RuntimeError as error:
                errors.append(f"CUDA device {index} failed BF16 execution: {error}")
        blackwell = capability[0] in {10, 12}
        devices.append(
            asdict(
                DeviceReport(
                    index,
                    properties.name,
                    capability,
                    properties.total_memory,
                    bf16,
                    blackwell,
                    valid,
                )
            )
        )
        if not bf16:
            errors.append(f"CUDA device {index} lacks BF16 support")
        if require_blackwell and not blackwell:
            errors.append(f"CUDA device {index} is not a recognized Blackwell architecture")
    if (require_cuda or require_blackwell) and not devices:
        errors.append("No CUDA GPU is visible")
    if not devices:
        warnings.append(
            "CPU reference mode: CUDA kernels, memory fit, and throughput are unverified"
        )
    versions = {}
    for package in ("torch", "numpy", "tokenizers", "triton", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "ok": not errors,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": devices,
        "errors": errors,
        "warnings": warnings,
    }
