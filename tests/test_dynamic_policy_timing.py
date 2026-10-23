"""An input-processing timeout still includes its actual neural ingestion cost."""

import importlib.util
import time
from pathlib import Path

import torch

from megatoken.generation import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig


def test_full_execution_timer_includes_observation_ingestion(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_dynamic_policy.py"
    spec = importlib.util.spec_from_file_location("dynamic_policy_timing", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original = evaluator.MultiTrackRuntime.observe
    calls = []

    def delayed(self, *args, **kwargs):
        time.sleep(0.02)
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(evaluator.MultiTrackRuntime, "observe", delayed)
    torch.set_num_threads(1)
    model = build_model(ModelConfig.tiny(max_capacity=512), ByteCodec.byte_fallback())
    result = evaluator.attempt(
        model,
        index=0,
        profile="independent_cheap_reads",
        serial=True,
        decisions=1,
        timeout=0.001,
        seed=17,
    )
    assert len(calls) >= 2
    assert result["elapsed_seconds"] >= 0.02 * len(calls)
    assert result["selected_groups"] == 0 and not result["task_success"]
    assert result["timing_clock"] == "wall_perf_counter"
    assert result["format"] == "autonomous-delegation-attempt/v2"
