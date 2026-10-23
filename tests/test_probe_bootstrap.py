"""Smoke tests for scripts/probes/probe_bootstrap.py (key presence and finite numbers)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probes" / "probe_bootstrap.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("probe_bootstrap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def check_round_trip(run) -> None:
    trip = run["round_trip"]
    assert trip["chunks"] > 0 and trip["latent_rms"] > 0
    assert 0 <= trip["baseline_byte_accuracy"] <= 1
    tags = " ".join(level["tag"] for level in trip["by_noise"])
    assert "training_tau" in tags and "serving_prior_std" in tags
    assert any(level["sigma"] == 0.0 for level in trip["by_noise"])
    for level in trip["by_noise"]:
        for key in ("exact_match_rate", "byte_accuracy", "similarity", "length_match_rate"):
            assert 0 <= level[key] <= 1
        assert set(level["by_type"]) <= {"code", "json", "prose"} and level["by_type"]
        assert level["by_length"]


def check_experts(run) -> None:
    health = run["expert_health"]
    assert health["banks"] and health["routers"]
    for scope in ("banks", "routers"):
        for stats in health[scope].values():
            assert 0 <= stats["normalized_entropy"] <= 1 + 1e-9
            assert 0 <= stats["frac_below_1pct_of_avg"] <= 1
    exercised = [s for s in health["banks"].values() if s["exercised"]]
    assert exercised  # at least one shared bank was routed to
    zero = health["zero_gradient"]
    assert health["optimizer_steps_observed"] >= 1
    for part in zero.values():
        assert part["count"] == sum(len(v) for v in part["by_category"].values())
    assert health["trainable_parameters"] > 100


def test_full_mode_smoke(probe, tmp_path):
    from megatoken.training.optimizer import MasterAdamW

    original_step = MasterAdamW.step
    output = tmp_path / "full.json"
    probe.main(["--smoke", "--output", str(output), "--work-dir", str(tmp_path / "work")])
    assert MasterAdamW.step is original_step  # the instrumentation is always removed
    result = json.loads(output.read_text())
    assert finite_numbers(result)
    (run,) = result["runs"].values()
    assert run["training"]["mode"] == "full" and run["training"]["optimizer_steps"] == 2
    assert run["training"]["curve"]
    check_round_trip(run)
    check_experts(run)
    predictor = run["predictor"]
    for key in (
        "mse_model",
        "mse_repeat",
        "mse_running_mean",
        "ratio_vs_repeat",
        "ratio_vs_running_mean",
        "ratio_vs_constant",
        "kl_model_per_coordinate",
        "prior_sigma_mean",
    ):
        assert predictor[key] > 0 and math.isfinite(predictor[key])
    assert predictor["by_type"]
    assert set(run["assessment"]) >= {"round_trip", "predictor", "experts", "gradients"}


def test_codec_mode_smoke(probe, tmp_path):
    output = tmp_path / "codec.json"
    probe.main(
        [
            "--smoke",
            "--train-mode",
            "codec",
            "--chunk-caps",
            "24,32",
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    result = json.loads(output.read_text())
    assert finite_numbers(result)
    assert sorted(result["runs"]) == ["cap24_tau0.1", "cap32_tau0.1"]
    for run in result["runs"].values():
        assert run["predictor"] is None and "predictor_note" in run
        check_round_trip(run)
        check_experts(run)


def test_helpers(probe):
    import numpy as np

    even = probe.usage_stats(np.array([5, 5, 5, 5]))
    assert even["normalized_entropy"] == pytest.approx(1.0) and even["frac_below_1pct_of_avg"] == 0
    skewed = probe.usage_stats(np.array([1000, 1, 0, 0]))
    assert skewed["normalized_entropy"] < 0.2 and skewed["frac_below_1pct_of_avg"] == 0.75
    assert probe.usage_stats(np.zeros(4))["exercised"] is False
    assert probe.length_bucket(1) == "1-8" and probe.length_bucket(33) == "33-64"
    assert probe.length_bucket(5000).startswith(">")
    scores = probe.Scores()
    scores.add("code", b"abcd", b"abcd")
    scores.add("code", b"abcd", b"abxd")
    report = scores.report()
    assert report["exact_match_rate"] == 0.5 and report["byte_accuracy"] == pytest.approx(0.875)
    assert probe.lr_at(0, 100, 1.0, 0.1) == pytest.approx(0.1)
    assert probe.lr_at(50, 100, 1.0, 0.1) == 1.0


def test_builtin_corpus_is_deterministic_and_chunks_respect_cap(probe):
    first = probe.builtin_corpus(3, 2, 300, offset=0)
    second = probe.builtin_corpus(3, 2, 300, offset=0)
    assert first == second and {d.kind for d in first} == {"code", "json", "prose"}
    other = probe.builtin_corpus(3, 2, 300, offset=1)
    assert [d.payload for d in other] != [d.payload for d in first]
    for doc in first:
        chunks = probe.chunk_document(doc, 32)
        assert b"".join(chunks) == doc.payload and max(map(len, chunks)) <= 32


def test_text_dir_corpus_and_codec_smoke(probe, tmp_path):
    root = tmp_path / "text"
    root.mkdir()
    for index in range(6):
        (root / f"mod{index}.py").write_text(
            "".join(f"def f{index}_{k}(x):\n    return x + {k}\n\n" for k in range(30))
        )
        (root / f"note{index}.md").write_text("The quick brown fox jumps over the lazy dog. " * 20)
        (root / f"data{index}.json").write_text((f'{{"id": {index}, "tags": ["a", "b"]}}\n') * 12)
    (root / "image.bin").write_bytes(b"\x00\x01\x02")  # ignored: not text
    train, held = probe.directory_corpus(root, 3, 0.2, 100, 1 << 16)
    assert train and held and {d.name for d in train}.isdisjoint({d.name for d in held})
    assert {d.kind for d in train + held} == {"code", "json", "prose"}
    output = tmp_path / "dir.json"
    probe.main(
        [
            "--smoke",
            "--train-mode",
            "codec",
            "--text-dir",
            str(root),
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    result = json.loads(output.read_text())
    assert result["corpus"]["source"] == str(root) and finite_numbers(result)
    (run,) = result["runs"].values()
    check_round_trip(run)
