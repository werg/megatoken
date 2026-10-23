#!/usr/bin/env python3
"""Throughput probe: how fast does the model train and decode, and what does long history cost?

QUESTIONS THIS ANSWERS
  1. Training speed and memory.  For each model size (32x256, 32x512) how many bytes (and
     codec tokens) of text does ONE training step ingest per second, for a small grid of
     microbatch size (several independent texts at once) and window length (how many chunks
     in a row the gradient flows through)?  What is the biggest microbatch that still fits
     in memory (we report peak allocated memory, not just "it did not crash")?
  2. Decode speed.  How many milliseconds does it take to generate one chunk at effort 1, 3
     and 6 (effort = how many prediction rows the predictor works on; higher = slower)?
  3. History length.  The model keeps a bounded full-resolution key/value history of past
     chunks (config field ``max_history_chunks``, default 1,024).  We raise it stepwise
     (1,024 -> 4,096 -> 16,384) and measure memory and per-chunk latency, and check that the
     bound is enforced and costs what the architecture document says (32 MiB at 1,024 chunks
     for 32x512, 16 MiB for 32x256).
  4. Needle wiring check.  Plant a distinctive chunk d chunks back (random filler in
     between) and measure how much of the prediction stage's attention lands on that chunk
     compared with a uniform spread.  The model is UNTRAINED, so this is only a wiring
     check (long positions are reachable, RoPE does not collapse at long distance), NOT a
     quality measure.  We also report the same number for a control chunk that is not
     related to the query; the difference between the two is the only meaningful signal.
  5. Projection.  tokens-per-GPU-hour times your GPU-hours, scaled by an explicit
     ``--speed-ratio`` (cloud GPU speed divided by this machine's speed).  There is no
     default: you must supply the number.

HOW TO RUN
  Smoke (tiny model, CPU, under 90 s; checks the harness only, numbers are meaningless):
      python scripts/probes/probe_throughput.py --smoke --output /tmp/tp_smoke.json
  Realistic (needs the GPU; takes it exclusively via /tmp/megatoken-gpu.lock; ~10-30 min):
      python scripts/probes/probe_throughput.py --device cuda --presets 32x256,32x512 \
          --batches 1,4 --windows 1,4 --record-bytes 1024 --capacity 256 \
          --history-chunks 1024,4096,16384 --needle-distances 1,16,256,4096 \
          --gpu-hours 1000 --gpus 8 --speed-ratio 2.0 --output /tmp/tp.json
  (--speed-ratio 2.0 is only an example.)  Needs the real codec for the big presets
  (``--codec``, default /home/werg/megatoken-runs/assets/codec).

WHAT THE NUMBERS MEAN
  train[].source_bytes_per_s   bytes of text read by the encoder per second (all records of
                               all texts in the microbatch, warm-up excluded, bf16, GPU
                               synchronised around timing).  Higher is better.
  train[].tokens_per_s         the same work counted in codec tokens (the benchmark's "I").
  train[].peak_allocated_gib   largest GPU memory actually in use during timed steps
                               (weights + optimizer + activations).
  train max_fit                largest microbatch (doubling search) that ran without running
                               out of memory and stayed under ``--mem-cap-gib``.
  decode[].ms_per_chunk        wall time to generate one chunk (``--capacity`` positions).
  history[].predict_ms         time to predict the next chunk with N chunks of history.
  history[].observe_ms         time to ingest one chunk with N chunks of history.
  history[].kv_bytes           BF16 key+value bytes held; kv_bytes_documented is the formula
                               N x 32 slots x kv_heads x head_dim x 2 (K+V) x 2 bytes.
  history.bound                after writing 1.5x the limit, slots kept == limit?  and does
                               the KV size equal the documented 32 / 16 MiB?
  needle[].ratio_to_uniform    attention on the planted chunk's slots / (their uniform
                               share), averaged over the four prediction stages that read
                               history.  1.0 = no better than chance.
  needle[].control_ratio       the same for an unrelated chunk in the same place.

HEALTHY VS UNHEALTHY (thresholds marked [guess] are not derived from anything solid)
  * Training: the 32x512 preset should be well above the earlier real P1 run on this
    machine (~460 tokens/s at microbatch 1, see /home/werg/megatoken-runs/smoke-p1/prof.log).
    Throughput growing roughly linearly from microbatch 1 to 4 means the GPU was not yet
    saturated; flat growth means it is.  [guess] Peak memory above ~60% of free shared
    memory at microbatch 4 means other jobs will OOM you.  If memory is the limit: lower
    microbatch_episodes / max_history_tokens, or enable gradient_checkpointing.
  * Decode: ms_per_chunk should grow with effort (1 < 3 < 6).  Flat means launch overhead
    dominates (CPU-bound), a model-size-independent floor.
  * History: kv_bytes must equal the documented bytes exactly, and bound.enforced must be
    true.  predict_ms should grow sub-linearly in N up to a few thousand chunks (fused
    attention) [guess]; if it grows >= 4x per 4x N the attention path fell off the fused
    kernel.  If the bound is not enforced or storage_bytes >> kv_bytes, memory leaks past
    the documented figure (see the note printed with bound).
  * Needle: ratio_to_uniform alone means little on an untrained model.  Healthy wiring:
    ratio is finite and not exactly 0 at the largest distance, and does not fall to a
    constant for every distance beyond a few (a constant would suggest RoPE collapse).
    [guess] needle ratio > control ratio at short distance on most runs.
  * Projection: simply arithmetic; trust it only as far as the speed ratio and the measured
    training rate (which comes from a synthetic workload, not a real data mix).

KNOBS TO TURN IF UNHEALTHY
  gradient_checkpointing, microbatch_episodes, max_history_tokens, max_history_chunks
  (ModelConfig), expert_backend (triton vs torch), generated_capacity.

NOTES
  The training workload is the library's own synthetic benchmark (``megatoken.benchmark``):
  full recurrent supervised path (encoder, central stage, predictor, real-vocabulary loss,
  backward, FP32-master AdamW).  "Window length" is the number of chunks predicted in a row
  with gradient (``prediction_chunks``).  It excludes the data pipeline, the memory bank and
  retrieval, so it is an upper bound on real step speed.  The GPU is shared: timings are
  noisy when other jobs run; the preflight block records free memory and load.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import gc
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

GPU_LOCK = "/tmp/megatoken-gpu.lock"
DEFAULT_CODEC = "/home/werg/megatoken-runs/assets/codec"
DOC_BOUND_MIB = {"32x512": 32.0, "32x256": 16.0}  # docs/architecture.md, 1,024 chunks, BF16


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


@contextlib.contextmanager
def gpu_lock(enabled: bool):
    if not enabled:
        yield
        return
    with open(GPU_LOCK, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def preflight(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"torch": torch.__version__, "device": str(device)}
    try:
        meminfo = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
        }
        info["host_mem_available_gib"] = meminfo["MemAvailable"] / 2**30
        info["swap_used_gib"] = (meminfo["SwapTotal"] - meminfo["SwapFree"]) / 2**30
        info["load_average_1m"] = os.getloadavg()[0]
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        info.update(
            gpu=torch.cuda.get_device_name(device),
            gpu_free_gib=free / 2**30,
            gpu_total_gib=total / 2**30,
        )
    return info


def load_config(preset: str, **overrides: Any):
    from megatoken.loading import load_model_config

    return load_model_config(preset, **overrides)


def load_codec_for(preset: str, codec_arg: str, vocab_size: int):
    from megatoken.loading import load_codec

    if preset == "tiny" or codec_arg == "byte":
        return load_codec("byte")
    return load_codec(codec_arg, vocab_size=vocab_size)


# ---------------------------------------------------------------- training grid


def train_cell(
    preset: str, args: argparse.Namespace, device: torch.device, batch: int, window: int
) -> dict[str, Any]:
    from megatoken.benchmark import benchmark

    config = load_config(preset)
    codec = load_codec_for(preset, args.codec, config.vocab_size)
    cell: dict[str, Any] = {"preset": preset, "batch": batch, "window": window}
    try:
        report = benchmark(
            config,
            codec=codec,
            device=device,
            precision="bf16" if device.type == "cuda" else "fp32",
            backend="triton" if device.type == "cuda" else "torch",
            steps=args.steps,
            warmup=args.warmup,
            batch=batch,
            record_bytes=args.record_bytes,
            capacity=min(args.capacity, config.max_capacity),
            backward=True,
            representation_mode="recurrent",
            prediction_chunks=window,
            seed=args.seed,
        )
    except Exception as error:  # noqa: BLE001
        if not is_oom(error):
            raise
        cleanup()
        return {**cell, "status": "oom", "error": str(error).splitlines()[0][:200]}
    rates = report["rates"]
    peak = report["memory"]["peak_allocated_bytes"]
    cell.update(
        status="ok",
        source_bytes_per_s=rates["source_bytes_per_second"],
        tokens_per_s=rates.get("I_per_second"),
        scored_positions_per_s=rates.get("P_per_second"),
        step_seconds_mean=report["latency_seconds"]["mean"],
        phase_seconds=report["phase_seconds"],
        peak_allocated_gib=None if peak is None else peak / 2**30,
        peak_reserved_gib=report["memory"]["peak_reserved_bytes"]
        and (report["memory"]["peak_reserved_bytes"] / 2**30),
    )
    cleanup()
    return cell


def run_train(
    preset: str,
    args: argparse.Namespace,
    device: torch.device,
    batches: list[int],
    windows: list[int],
) -> dict[str, Any]:
    cells = []
    for window in windows:
        for batch in batches:
            cell = train_cell(preset, args, device, batch, window)
            print(f"  train {preset} batch={batch} window={window}: {brief(cell)}", flush=True)
            cells.append(cell)
    # Doubling search for the largest fitting microbatch at window 1 (GPU only).
    max_fit: dict[str, Any] | None = None
    if device.type == "cuda" and not args.no_fit_search:
        batch = max(batches) * 2
        best = max(
            (c for c in cells if c["status"] == "ok" and c["window"] == min(windows)),
            key=lambda c: c["batch"],
            default=None,
        )
        while batch <= args.max_batch:
            cell = train_cell(preset, args, device, batch, min(windows))
            print(f"  fit-search {preset} batch={batch}: {brief(cell)}", flush=True)
            cells.append(cell)
            if cell["status"] != "ok":
                break
            best = cell
            if cell["peak_allocated_gib"] and cell["peak_allocated_gib"] > args.mem_cap_gib:
                break
            batch *= 2
        if best is not None:
            max_fit = {
                "batch": best["batch"],
                "window": best["window"],
                "peak_allocated_gib": best["peak_allocated_gib"],
                "stopped_because": "oom_or_cap_or_max_batch",
            }
    return {"cells": cells, "max_fit": max_fit}


def brief(cell: dict[str, Any]) -> str:
    if cell["status"] != "ok":
        return cell["status"]
    peak = cell["peak_allocated_gib"]
    mem = "n/a" if peak is None else f"{peak:.1f} GiB"
    return f"{cell['source_bytes_per_s']:.0f} B/s, {cell['tokens_per_s']:.0f} tok/s, peak {mem}"


# ---------------------------------------------------------------- decode


def run_decode(preset: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from megatoken.generation.structure import ActionSpec
    from megatoken.loading import build_model

    config = replace(
        load_config(preset), expert_backend="triton" if device.type == "cuda" else "torch"
    )
    codec = load_codec_for(preset, args.codec, config.vocab_size)
    capacity = min(args.capacity, config.max_capacity)
    precision = "bf16" if device.type == "cuda" else "fp32"
    torch.manual_seed(args.seed)
    model = build_model(config, codec, device=device, precision=precision).eval()
    core = model.core
    template = b"def update_state(value):\n    return value + 1\n"
    payload = (template * math.ceil(args.record_bytes / len(template)))[: args.record_bytes]
    byte_ids = torch.tensor(list(payload), device=device)[None]
    spec = ActionSpec(min_tokens=capacity, max_tokens=capacity, text_kind="bytes")
    out = []
    with torch.no_grad():
        state = core.observe(byte_ids, core.initial_state(1, device=device))
        for effort in args.efforts:
            gen = torch.Generator(device=device).manual_seed(args.seed + 1)
            for _ in range(args.warmup):
                model.sample(state, spec=spec, effort=effort, generator=gen)
            sync(device)
            times = []
            for _ in range(args.steps):
                start = time.perf_counter()
                model.sample(state, spec=spec, effort=effort, generator=gen)
                sync(device)
                times.append(time.perf_counter() - start)
            row = {
                "preset": preset,
                "effort": effort,
                "capacity": capacity,
                "ms_per_chunk": 1000 * sum(times) / len(times),
                "ms_min": 1000 * min(times),
            }
            print(
                f"  decode {preset} effort={effort}: {row['ms_per_chunk']:.1f} ms/chunk", flush=True
            )
            out.append(row)
    del model, core, state
    cleanup()
    return {"rows": out}


# ---------------------------------------------------------------- history


def kv_nominal_bytes(config, chunks: int, itemsize: int = 2) -> int:
    return chunks * config.n_facets * config.kv_heads * config.attention_head_dim * 2 * itemsize


def build_core(config, device: torch.device, seed: int):
    from megatoken.model.core import FacetedCore

    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return FacetedCore(config).to(device=device, dtype=dtype).eval()


def synthetic_state(core, chunks: int, device: torch.device, scale: float = 1.0):
    """A state whose history holds ``chunks`` random chunks (cheap: no chunk is encoded)."""
    from megatoken.model.state import KVHistory

    c = core.config
    dtype = core.initial_facets.dtype
    shape = (1, c.kv_heads, chunks * c.n_facets, c.attention_head_dim)
    keys = (torch.randn(shape, device=device) * scale).to(dtype)
    values = torch.randn(shape, device=device).to(dtype)
    final = torch.randn(1, c.n_facets, c.facet_dim, device=device).to(dtype)
    state = core.initial_state(1, device=device)
    return dataclasses.replace(
        state,
        final=final,
        history=KVHistory(keys, values),
        records_seen=torch.full((1,), chunks, dtype=torch.long, device=device),
        last_record_ids=torch.full((1,), chunks - 1, dtype=torch.long, device=device),
    )


def timed(fn, device: torch.device, reps: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(device)
    return 1000 * (time.perf_counter() - start) / reps


def check_bound(preset: str, device: torch.device, seed: int) -> dict[str, Any]:
    """Write 1.5x the limit of chunks; verify slots are capped and bytes match the docs."""
    config = replace(
        load_config(preset), expert_backend="triton" if device.type == "cuda" else "torch"
    )
    core = build_core(config, device, seed)
    c = config
    limit = c.max_history_chunks
    state = core.initial_state(1, device=device)
    history, written, block = state.history, 0, min(limit, 128)
    dtype = core.initial_facets.dtype
    with torch.no_grad():
        while written < int(1.5 * limit):
            final = torch.randn(1, block * c.n_facets, c.facet_dim, device=device).to(dtype)
            pos = torch.full((1,), written, dtype=torch.long, device=device)
            history = core.central.write(final, history, pos)
            written += block
    nominal = kv_nominal_bytes(c, limit, history.keys.element_size())
    actual = (history.keys.numel() + history.values.numel()) * history.keys.element_size()
    storage = history.keys.untyped_storage().nbytes() + history.values.untyped_storage().nbytes()
    documented = DOC_BOUND_MIB.get(preset)
    result = {
        "preset": preset,
        "limit_chunks": limit,
        "chunks_written": written,
        "slots_kept": history.slots,
        "slots_expected": limit * c.n_facets,
        "kv_bytes": actual,
        "kv_bytes_documented_formula": nominal,
        "kv_mib": actual / 2**20,
        "documented_mib": documented,
        "storage_bytes_behind_views": storage,
        "enforced": history.slots == limit * c.n_facets,
        "matches_docs": actual == nominal
        and (
            documented is None or history.keys.element_size() != 2 or actual == documented * 2**20
        ),
        "note": (
            "storage_bytes_behind_views > kv_bytes means the trimmed cache is a view of a "
            "larger concatenation (core.py CentralProcessor.write); the extra block is "
            "freed once the next write replaces it, so the steady state is bounded but a "
            "transient peak of about one extra block per write exists."
        ),
    }
    del core, history
    cleanup()
    return result


def run_history(preset: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    rows = []
    bound = check_bound(preset, device, args.seed)
    print(f"  bound {preset}: {bound['kv_mib']:.2f} MiB enforced={bound['enforced']}", flush=True)
    for chunks in args.history_chunks:
        config = replace(
            load_config(preset),
            max_history_chunks=chunks,
            expert_backend="triton" if device.type == "cuda" else "torch",
        )
        row: dict[str, Any] = {"preset": preset, "chunks": chunks}
        try:
            core = build_core(config, device, args.seed)
            sync(device)
            if device.type == "cuda":
                resident = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                state = synthetic_state(core, chunks, device)
                chunk = torch.randint(0, 255, (1, args.record_bytes), device=device)
                row["predict_ms"] = timed(
                    lambda c=core, s=state: c.predict_facets(s, effort=1), device, args.steps
                )
                row["observe_ms"] = timed(
                    lambda c=core, s=state, b=chunk: c.observe(b, s), device, args.steps
                )
            history = state.history
            row["kv_bytes"] = (history.keys.numel() + history.values.numel()) * (
                history.keys.element_size()
            )
            row["kv_bytes_documented"] = kv_nominal_bytes(
                config, chunks, history.keys.element_size()
            )
            if device.type == "cuda":
                row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                row["peak_over_resident_gib"] = (
                    torch.cuda.max_memory_allocated(device) - resident
                ) / 2**30
            row["status"] = "ok"
        except Exception as error:  # noqa: BLE001
            if not is_oom(error):
                raise
            row.update(status="oom", error=str(error).splitlines()[0][:200])
        rows.append(row)
        print(
            f"  history {preset} N={chunks}: "
            + (
                f"predict {row['predict_ms']:.1f} ms, observe {row['observe_ms']:.1f} ms, "
                f"KV {row['kv_bytes'] / 2**20:.0f} MiB"
                if row["status"] == "ok"
                else "oom"
            ),
            flush=True,
        )
        state = core = history = None  # noqa: F841
        cleanup()
    return {"bound": bound, "rows": rows}


# ---------------------------------------------------------------- needle


class _AttentionRecorder:
    """Stand-in for torch.nn.functional inside megatoken.model.attention that records
    the attention mass on the first ``needle_slots`` keys of each history-reading read."""

    def __init__(self, real, needle_slots: int, live_slots: int):
        self.real, self.needle_slots, self.live_slots = real, needle_slots, live_slots
        self.records: list[dict[str, float]] = []
        self.active = False

    def __getattr__(self, name):
        return getattr(self.real, name)

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, **kwargs):
        slots = k.shape[2]
        if self.active and slots > self.live_slots + 1:
            groups = q.shape[1] // k.shape[1]
            keys = k.repeat_interleave(groups, dim=1) if groups > 1 else k
            scores = (q.float() @ keys.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
            weights = scores.softmax(-1)
            mass = weights[..., : self.needle_slots].sum(-1)  # [B, heads, queries]
            self.records.append(
                {"slots": float(slots), "mass": float(mass.mean()), "max": float(mass.max())}
            )
        return self.real.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kwargs)


def run_needle(preset: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    import megatoken.model.attention as attention_module

    distances = sorted(args.needle_distances)
    config = replace(
        load_config(preset),
        max_history_chunks=max(distances) + 2,
        expert_backend="triton" if device.type == "cuda" else "torch",
    )
    core = build_core(config, device, args.seed)
    c = config
    nbytes = min(args.record_bytes, 64)
    needle_bytes = (b"NEEDLE-7f3a9c-quartz-lantern-" * 8)[:nbytes]
    other_bytes = (b"zebra moon 1234 harbor thunder " * 8)[:nbytes]
    needle = torch.tensor(list(needle_bytes), device=device)[None]
    other = torch.tensor(list(other_bytes), device=device)[None]
    recorder = _AttentionRecorder(attention_module.F, c.n_facets, c.n_facets)
    rows = []
    with torch.no_grad():
        start = core.initial_state(1, device=device)
        planted = core.observe(needle, start)
        control = core.observe(other, start)
        scale = float(planted.history.keys.float().std())
        for d in distances:
            row: dict[str, Any] = {"preset": preset, "distance_chunks": d}
            for label, source in (("needle", planted), ("control", control)):
                filler = synthetic_state(core, d, device, scale=scale).history
                from megatoken.model.state import KVHistory

                keys = torch.cat(
                    (source.history.keys, filler.keys[:, :, : (d - 1) * c.n_facets]), 2
                )
                values = torch.cat(
                    (source.history.values, filler.values[:, :, : (d - 1) * c.n_facets]), 2
                )
                state = dataclasses.replace(
                    source,
                    history=KVHistory(keys, values),
                    records_seen=torch.full((1,), d, dtype=torch.long, device=device),
                )
                # The query chunk repeats the needle text, so the needle is what it "recalls".
                state = core.observe(needle, state)
                recorder.records = []
                attention_module.F = recorder
                recorder.active = True
                try:
                    core.predict_facets(state, effort=1)
                finally:
                    recorder.active = False
                    attention_module.F = recorder.real
                records = recorder.records
                total_slots = records[0]["slots"] if records else float("nan")
                uniform = c.n_facets / total_slots
                mass = sum(r["mass"] for r in records) / max(1, len(records))
                row[f"{label}_mass"] = mass
                row[f"{label}_ratio"] = mass / uniform
                row["uniform_share"] = uniform
                row["stages_read"] = len(records)
            row["ratio_to_uniform"] = row.pop("needle_ratio")
            row["control_ratio"] = row.pop("control_ratio")
            rows.append(row)
            print(
                f"  needle {preset} d={d}: ratio {row['ratio_to_uniform']:.2f} "
                f"(control {row['control_ratio']:.2f}, uniform = 1.00)",
                flush=True,
            )
    del core
    cleanup()
    return {
        "rows": rows,
        "caveat": "Untrained weights: a wiring check only. Compare needle with control.",
    }


# ---------------------------------------------------------------- projection


def project(train_by_preset: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.speed_ratio is None:
        raise SystemExit("--speed-ratio is required for the projection (no default is assumed)")
    if args.speed_ratio <= 0 or args.gpus < 1 or args.gpu_hours < 0:
        raise SystemExit("--speed-ratio and --gpus must be positive; --gpu-hours nonnegative")
    out = {}
    for preset, report in train_by_preset.items():
        ok = [c for c in report["cells"] if c["status"] == "ok"]
        if not ok:
            continue
        best = max(ok, key=lambda c: c["tokens_per_s"])
        seconds = 3600 * args.gpu_hours * args.gpus
        out[preset] = {
            "uses_cell": {"batch": best["batch"], "window": best["window"]},
            "local_tokens_per_s": best["tokens_per_s"],
            "local_bytes_per_s": best["source_bytes_per_s"],
            "assumed_speed_ratio": args.speed_ratio,
            "gpu_hours": args.gpu_hours,
            "gpus": args.gpus,
            "tokens": best["tokens_per_s"] * args.speed_ratio * seconds,
            "bytes": best["source_bytes_per_s"] * args.speed_ratio * seconds,
            "assumption": (
                f"cloud GPU = {args.speed_ratio}x this machine's measured speed, perfect "
                "scaling across GPUs, no data-loading or sync loss, synthetic workload"
            ),
        }
    return out


# ---------------------------------------------------------------- main


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="tiny model on CPU, harness check")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--presets", default="32x256,32x512")
    parser.add_argument(
        "--sections",
        default="train,decode,history,needle",
        help="comma list; projection is automatic",
    )
    parser.add_argument("--codec", default=DEFAULT_CODEC)
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--windows", default="1,4")
    parser.add_argument("--record-bytes", type=int, default=1024)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--efforts", default="1,3,6")
    parser.add_argument("--history-chunks", default="1024,4096,16384")
    parser.add_argument("--needle-distances", default="1,16,256,4096")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--mem-cap-gib", type=float, default=40.0)
    parser.add_argument("--no-fit-search", action="store_true")
    parser.add_argument("--gpu-hours", type=float)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--speed-ratio", type=float, help="cloud GPU speed / this machine's")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.speed_ratio is not None and args.speed_ratio <= 0:
        raise SystemExit("--speed-ratio must be positive")
    if args.smoke:
        torch.set_num_threads(2)
        args.device, args.presets = "cpu", "tiny"
        args.batches, args.windows = "1", "1"
        args.record_bytes, args.capacity = 32, 8
        args.steps, args.warmup = 1, 1
        args.efforts = "1"
        args.history_chunks = "4,8"
        args.needle_distances = "1,3"
        args.sections = "train,decode,history,needle"
        if args.gpu_hours is None:
            args.gpu_hours = 1.0
        if args.speed_ratio is None:
            args.speed_ratio = 1.0  # smoke only exercises the arithmetic
    # Spell the index out: model.core.initial_state compares torch.device("cuda") with
    # "cuda:0" by equality and rejects it (src bug, see report).
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    args.efforts = parse_ints(args.efforts)
    args.history_chunks = parse_ints(args.history_chunks)
    args.needle_distances = parse_ints(args.needle_distances)
    presets = [p for p in args.presets.split(",") if p]
    sections = set(args.sections.split(","))
    if args.gpu_hours is not None and args.speed_ratio is None:
        raise SystemExit("--speed-ratio is required when --gpu-hours is given (no default)")
    started = time.time()
    result: dict[str, Any] = {"args": {k: str(v) for k, v in vars(args).items()}}
    with gpu_lock(device.type == "cuda"):
        result["preflight"] = preflight(device)
        print("preflight:", json.dumps(result["preflight"]), flush=True)
        for section in ("train", "decode", "history", "needle"):
            if section not in sections:
                continue
            result[section] = {}
            for preset in presets:
                print(f"[{section}] {preset}", flush=True)
                if section == "train":
                    report = run_train(
                        preset, args, device, parse_ints(args.batches), parse_ints(args.windows)
                    )
                elif section == "decode":
                    report = run_decode(preset, args, device)
                elif section == "history":
                    report = run_history(preset, args, device)
                else:
                    report = run_needle(preset, args, device)
                result[section][preset] = report
        result["postflight"] = preflight(device)
    if args.gpu_hours is not None and "train" in result:
        result["projection"] = project(result["train"], args)
        for preset, row in result["projection"].items():
            print(
                f"projection {preset}: {row['tokens']:.3e} tokens for {args.gpu_hours} GPU-h x "
                f"{args.gpus} GPUs at speed ratio {args.speed_ratio}"
            )
    result["wall_seconds"] = time.time() - started
    text = json.dumps(result, indent=2, allow_nan=False, default=str)
    if args.output:
        args.output.write_text(text + "\n")
    print(
        f"done in {result['wall_seconds']:.0f} s"
        + (f"; wrote {args.output}" if args.output else "")
    )
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
