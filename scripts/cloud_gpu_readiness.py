"""Bounded real-data, GPU training, checkpoint, resume, and serving smoke run."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path("/work/out/readiness")


def _deadline(signum, frame):
    raise TimeoutError("Readiness run exceeded its 45 minute work budget")


def record(check: str, **values) -> None:
    with (ROOT / "report.jsonl").open("a") as handle:
        handle.write(json.dumps({"check": check, **values}, sort_keys=True) + "\n")


def main() -> None:
    signal.signal(signal.SIGALRM, _deadline)
    signal.alarm(45 * 60)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "system.txt").write_text(
        subprocess.run(
            [
                "bash",
                "-lc",
                "nvidia-smi; df -B1 / /work /tmp /dev/shm; lsblk -b -o NAME,SIZE,TYPE,MOUNTPOINT || true; free -b; nproc",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    record(
        "gpu",
        name=props.name,
        memory_bytes=props.total_memory,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )

    # Pin the download to the immutable revision resolved at run time.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "90")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info("HuggingFaceFW/fineweb-edu", timeout=45, token=False).sha
    t0 = time.monotonic()
    downloaded = Path(
        hf_hub_download(
            "HuggingFaceFW/fineweb-edu",
            "sample/10BT/000_00000.parquet",
            repo_type="dataset",
            revision=revision,
            local_dir="/work/readiness-download",
        )
    )
    elapsed = time.monotonic() - t0
    usage = shutil.disk_usage("/work")
    record(
        "download",
        revision=revision,
        bytes=downloaded.stat().st_size,
        seconds=elapsed,
        MBps=downloaded.stat().st_size / max(elapsed, 0.001) / 1e6,
        disk_free_after_bytes=usage.free,
        disk_total_bytes=usage.total,
    )

    a = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    b = a @ a.T
    torch.cuda.synchronize()
    record("bf16", finite=bool(torch.isfinite(b).all().item()), dtype=str(b.dtype))
    try:
        import triton

        from megatoken.runtime import doctor

        diagnostics = doctor()
        record(
            "runtime",
            triton=triton.__version__,
            doctor_ok=diagnostics["ok"],
            errors=diagnostics["errors"],
        )
        if not diagnostics["ok"]:
            raise RuntimeError(
                "Megatoken runtime doctor failed: " + "; ".join(diagnostics["errors"])
            )
    except Exception:
        record("runtime_error", traceback=traceback.format_exc())
        raise

    sys.path.insert(0, "/work/megatoken/tests")
    from training_fixture import tiny_train_config

    from megatoken.assets import bake_checkpoint
    from megatoken.benchmark import benchmark_served_session
    from megatoken.served import ServedBundle, export_served
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.engine import train

    # Keep bulky model states in scratch; only concise reports leave the sandbox.
    run = Path("/work/readiness-training")
    run.mkdir(parents=True, exist_ok=True)
    imports = [
        "/work/assets/deepseek-v4.1-flash-1017",
        "/work/assets/qwen3.8-flash-next",
    ]
    config = tiny_train_config(
        run,
        output=str(run / "runs"),
        device="cuda",
        precision="bf16",
        max_steps=1,
        checkpoint_every=1,
        log_every=1,
        expert_backend="triton",
        model_overrides={
            "ngram_rows": 2_500_000,
            "ngram_dim": 16,
            "exact_bigrams": True,
            "ngram_orders": (1, 2, 3, 4, 8),
        },
        engram_imports=imports,
    )
    first = train(config)
    if first["global_step"] != 1:
        raise RuntimeError(f"Expected one optimizer step, got {first['global_step']}")
    checkpoint = first["checkpoint"]
    state, _ = load_checkpoint(checkpoint, map_location="cpu")
    record(
        "train_checkpoint", global_step=first["global_step"], checkpoint_step=state["global_step"]
    )

    config = replace(config, max_steps=2)
    resumed = train(config, resume=checkpoint)
    if resumed["global_step"] != 2:
        raise RuntimeError(f"Resume did not advance to step 2: {resumed['global_step']}")
    record("resume", global_step=resumed["global_step"])

    baked = bake_checkpoint(resumed["checkpoint"], run / "baked")
    bundle_path = Path("/work/readiness-served")
    exported = export_served(baked["checkpoint"], bundle_path)
    bundle = ServedBundle(bundle_path, device="cuda")
    record(
        "served_export_load",
        model_version=exported["model_version"],
        device=str(next(bundle.model.parameters()).device),
    )
    session = benchmark_served_session(
        bundle=bundle_path,
        device="cuda",
        steps=1,
        warmup=0,
        capacity=4,
        record_bytes=8,
        effort=1,
        backend="triton",
    )
    record("served_session", **session)
    shutil.rmtree("/work/readiness-download", ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "failure.txt").write_text(traceback.format_exc())
        raise
