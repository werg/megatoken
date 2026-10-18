"""Measured end-to-end benchmarks, with separate transcript and policy meters.

Warmup, timed execution, and an optional extra profiler iteration are separate.
These synthetic workloads diagnose implementation cost, not learned capability.
"""

from __future__ import annotations

import importlib.metadata
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import torch

from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model, load_codec, load_model_config, parameter_report
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.audit import audit_parameters
from megatoken.training.config import DEFAULT_LOSS_WEIGHTS
from megatoken.training.metrics import dispatch_summary
from megatoken.training.optimizer import MasterAdamW


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _measure(times: dict[str, float], name: str, device: torch.device) -> Iterator[None]:
    _synchronize(device)
    start = time.perf_counter()
    with torch.profiler.record_function(f"fmt::{name}"):
        yield
    _synchronize(device)
    times[name] = times.get(name, 0.0) + time.perf_counter() - start


def benchmark(
    config: ModelConfig | str = "tiny",
    *,
    device: str | torch.device = "cpu",
    steps: int = 5,
    batch: int = 1,
    record_bytes: int = 256,
    capacity: int = 64,
    backward: bool = False,
    backend: str = "torch",
    compile_mode: str | None = None,
    codec: ByteCodec | str = "byte",
    warmup: int = 2,
    effort: int = 6,
    history_records: int = 0,
    profile: bool = False,
    core_only: bool = False,
    precision: str | None = None,
    representation_mode: str = "bootstrap",
    prediction_chunks: int = 1,
    seed: int = 17,
) -> dict[str, Any]:
    """Run a real workload and return a JSON-safe measured report.

    Full training uses batched supervised decisions with all-free target IDs,
    facet draws, live recurrent towers, backward, and FP32-master AdamW.
    Full inference samples complete action tapes, one independent
    request at a time; ``batch`` is explicitly reported as serial in this mode.
    ``core_only`` measures the byte encoder and core without inventing a codec
    or pretending that a facet loss is scored lexical policy work.
    """
    if (
        steps < 1
        or batch < 1
        or record_bytes < 1
        or capacity < 1
        or warmup < 0
        or history_records < 0
        or prediction_chunks < 1
    ):
        raise ValueError(
            "Invalid benchmark sizes; measured steps and work dimensions must be positive"
        )
    if effort not in (1, 2, 3, 6):
        raise ValueError("Unsupported predictive effort")
    if representation_mode not in {"bootstrap", "recurrent"}:
        raise ValueError("representation_mode must be bootstrap or recurrent")
    if prediction_chunks != 1 and (not backward or core_only):
        raise ValueError("prediction_chunks requires a full supervised backward benchmark")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but no CUDA device is available")
    if backend == "triton" and device.type != "cuda":
        raise ValueError("Triton measurements require an actual CUDA device")
    if isinstance(config, str):
        config = load_model_config(config)
    config = replace(config, expert_backend=backend)
    if capacity > config.max_capacity:
        raise ValueError("Benchmark capacity exceeds configured maximum")
    precision = precision or ("bf16" if device.type == "cuda" else "fp32")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    torch.manual_seed(seed)
    actual_codec: ByteCodec | None = None
    if not core_only:
        actual_codec = (
            codec
            if isinstance(codec, ByteCodec)
            else load_codec(codec, vocab_size=config.vocab_size)
        )
        if actual_codec.vocab_size != config.vocab_size:
            raise ValueError(
                "Full-vocabulary benchmark requires the real codec matching this model; use core_only otherwise"
            )
        model = build_model(config, actual_codec, device=device, precision=precision)
        core = model.core
    else:
        core = FacetedCore(config).to(
            device=device, dtype=torch.bfloat16 if precision == "bf16" else torch.float32
        )
        model = core
    model.train(backward)
    if compile_mode:
        core.compile_kernels(compile_mode)
    optimizer = MasterAdamW(model, lr=1e-5) if backward else None
    if optimizer is not None and representation_mode == "recurrent" and not core_only:
        optimizer.isolate_gradient_parameters("encoder", core.encoder_parameters())
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    template = b"def update_state(value):\n    return value + 1\n# preserve exact bytes and prior evidence\n"
    payload = (template * math.ceil(record_bytes / len(template)))[:record_bytes]
    byte_ids = torch.tensor(list(payload), device=device, dtype=torch.long)
    target_ids: tuple[int, ...] = ()
    target_payload = b""
    if actual_codec is not None:
        # Single-byte IDs deliberately fix P to capacity while transcript D/I
        # retain their own raw-codec retokenization. Emitted IDs remain exact.
        alphabet = b"result = value + 1\n"
        target_ids = tuple(
            actual_codec.encode(bytes([alphabet[i % len(alphabet)]]))[0] for i in range(capacity)
        )
        target_payload = actual_codec.decode(target_ids)
    request_batch = batch if backward or core_only else 1
    input_batch = byte_ids[None].expand(request_batch, -1).contiguous()
    plan = (
        model.prepare_target(
            target_ids=[target_ids] * batch,
            capacity=capacity,
            representation_mode=representation_mode,
        )
        if backward and actual_codec is not None
        else None
    )
    source_tokens = len(actual_codec.encode(payload)) if actual_codec is not None else None
    target_tokens = len(actual_codec.encode(target_payload)) if actual_codec is not None else None
    global_step = 0

    def one_iteration() -> tuple[dict[str, float], dict[str, int]]:
        nonlocal global_step
        times: dict[str, float] = {}
        counts = dict(
            D=0,
            I=0,
            P=0,
            Z=0,
            emitted_positions=0,
            source_bytes=0,
            record_ingestions=0,
            reconstruction_tokens=0,
            reconstruction_records=0,
            facet_encodes=0,
            generated_bytes=0,
            sampled_decisions=0,
            optimizer_steps=0,
        )
        core.set_model_version(global_step)
        if optimizer is not None:
            optimizer.zero_grad()
        requests = 1 if backward or core_only else batch
        for _ in range(requests):
            state = core.initial_state(request_batch, device=device)
            with _measure(times, "reconstruction", device), torch.no_grad():
                for _ in range(history_records):
                    state = core.observe(input_batch, state)
            counts["source_bytes"] += history_records * request_batch * record_bytes
            counts["record_ingestions"] += history_records * request_batch
            counts["reconstruction_records"] += history_records * request_batch
            if source_tokens is not None:
                counts["I"] += history_records * request_batch * source_tokens
                counts["reconstruction_tokens"] += history_records * request_batch * source_tokens
            with _measure(times, "forward", device), torch.set_grad_enabled(backward):
                state = core.observe(
                    input_batch, state, encoder_grad=representation_mode == "bootstrap"
                )
                counts["source_bytes"] += request_batch * record_bytes
                counts["record_ingestions"] += request_batch
                if source_tokens is not None:
                    counts["D"] += request_batch * source_tokens
                    counts["I"] += request_batch * source_tokens
                if core_only:
                    # A nontrivial synthetic facet objective exercises backward;
                    # it is explicitly NOT lexical pretraining or policy work.
                    objective = (
                        core.predict_facets(state, effort=effort).float()
                        * torch.linspace(-1, 1, config.facet_dim, device=device)
                    ).mean()
                elif backward:
                    objectives = []
                    for _ in range(prediction_chunks):
                        result = model.supervised_step(
                            state, plan=plan, effort=effort, generator=generator
                        )
                        state = result.state
                        objectives.append(
                            sum(
                                numerator
                                * (DEFAULT_LOSS_WEIGHTS[name] / denominator if denominator else 0)
                                for name, (numerator, denominator) in result.losses.items()
                            )
                        )
                    objective = sum(objectives) / prediction_chunks
                    counts["D"] += request_batch * target_tokens * prediction_chunks
                    counts["I"] += request_batch * target_tokens * prediction_chunks
                    counts["P"] += plan.scored_positions * prediction_chunks
                    counts["source_bytes"] += (
                        request_batch * len(target_payload) * prediction_chunks
                    )
                    counts["record_ingestions"] += request_batch * prediction_chunks
                    counts["facet_encodes"] += request_batch * prediction_chunks
                else:
                    from megatoken.generation.structure import ActionSpec

                    spec = ActionSpec(min_tokens=capacity, max_tokens=capacity, text_kind="bytes")
                    result = model.sample(state, spec=spec, effort=effort, generator=generator)
                    counts["P"] += result.scored_positions
                    counts["Z"] += result.tape.latent_dimensions
                    counts["emitted_positions"] += len(result.emitted_ids)
                    counts["generated_bytes"] += len(result.payload)
                    counts["sampled_decisions"] += 1
                    tokens = len(actual_codec.encode(result.payload))
                    counts["D"] += tokens
                    counts["I"] += tokens
                    counts["source_bytes"] += len(result.payload)
                    counts["record_ingestions"] += 1
            if backward:
                with _measure(times, "backward", device):
                    objective.backward()
        if optimizer is not None:
            with _measure(times, "optimizer", device):
                optimizer.step()
            counts["optimizer_steps"] += 1
        global_step += 1
        return times, counts

    try:
        for _ in range(warmup):
            one_iteration()
        core.dispatch_metrics(reset=True)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times: dict[str, float] = {}
        counts: dict[str, int] = {}
        latencies: list[float] = []
        started = time.perf_counter()
        for _ in range(steps):
            before = time.perf_counter()
            iteration_times, iteration_counts = one_iteration()
            latencies.append(time.perf_counter() - before)
            for name, value in iteration_times.items():
                times[name] = times.get(name, 0.0) + value
            for name, value in iteration_counts.items():
                counts[name] = counts.get(name, 0) + value
        _synchronize(device)
        elapsed = time.perf_counter() - started
        dispatch = dispatch_summary(core.dispatch_metrics())
        memory = (
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            if device.type == "cuda"
            else {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        )
        measured_profile = None
        if profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True
            ) as profiler:
                one_iteration()
            measured_profile = {
                "separate_untimed_iteration": True,
                "regions": [
                    {
                        "name": event.key,
                        "calls": event.count,
                        "cpu_total_seconds": event.cpu_time_total / 1e6,
                        "device_total_seconds": getattr(event, "device_time_total", 0.0) / 1e6,
                    }
                    for event in profiler.key_averages()
                    if event.key.startswith("fmt::")
                ],
                "note": "Region CPU/device times are inclusive and overlap their parent regions; do not add them together",
            }
        if actual_codec is None:
            for meter in ("D", "I", "reconstruction_tokens"):
                counts[meter] = None
        rates = {
            f"{key}_per_second": value / elapsed
            for key, value in counts.items()
            if value is not None
            and key
            in {
                "D",
                "I",
                "P",
                "Z",
                "emitted_positions",
                "source_bytes",
                "record_ingestions",
                "sampled_decisions",
            }
        }
        try:
            triton_version = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            triton_version = None
        return {
            "format_version": 1,
            "workload": "facet_core"
            if core_only
            else ("supervised_training" if backward else "served_prior_actions"),
            "configuration": {
                "model": config.to_dict(),
                "steps": steps,
                "warmup": warmup,
                "batch": batch,
                "batch_execution": "vectorized" if backward or core_only else "serial_requests",
                "record_bytes": record_bytes,
                "capacity": capacity,
                "effort": effort,
                "history_records": history_records,
                "backward": backward,
                "representation_mode": representation_mode if backward and not core_only else None,
                "prediction_chunks": prediction_chunks,
                "precision": precision,
                "compile_mode": compile_mode,
                "codec_fingerprint": actual_codec.fingerprint if actual_codec else None,
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "triton": triton_version,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU",
                "compute_capability": list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None,
            },
            "elapsed_seconds": elapsed,
            "phase_seconds": times,
            "latency_seconds": {
                "minimum": min(latencies),
                "maximum": max(latencies),
                "mean": sum(latencies) / len(latencies),
            },
            "counts": counts,
            "rates": rates,
            "dispatch": dispatch,
            "memory": memory,
            "parameters": audit_parameters(core) if core_only else parameter_report(model),
            "profile": measured_profile,
            "limitations": [
                "Fresh synthetic initialization: throughput evidence only, not trained-model quality or task success",
                "Warmup and optional profiler iteration excluded; reconstruction and optimizer are included in elapsed time",
                "Core-only mode has no lexical P or raw-codec D/I measurement",
                "Inference D is reported as a transcript-work equivalent; no learning exposure occurs under no_grad",
                "Padding is measured for the implemented 32-row Triton tiles; torch reference has no padded GEMM arithmetic",
            ],
        }
    finally:
        if optimizer is not None:
            optimizer.close()
