"""Bounded independent probes of ordinary V8 content identity, without training.

The collector control deliberately injects admission and planner boundaries.
It proves identity agreement at the actual collector and position call sites,
not source/native admission, real prepared data, optimizer or learned behavior.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.training.config import DecoupledConfig, TrainConfig
from megatoken.training.engine import _resume_compatible
from megatoken.training.positions_v8 import (
    empty_position_exposure,
    position_objective,
    restore_position_exposure,
)
from megatoken.training.target_inventory_v8 import training_config_identity_v8


def file_identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    assert revision == "b85a3301b6482bdf5a5a9ba79f7f356c5f824681"
    checks = []

    def passed(name, **detail):
        checks.append({"name": name, "status": "passed", **detail})

    def refuses(name, action, match):
        try:
            action()
        except ValueError as error:
            assert match in str(error), str(error)
            passed(name, refusal=str(error))
        else:
            raise AssertionError("Unexpected acceptance: " + name)

    policy = {
        "schema": "semantic-training-policy/8",
        "producerRevision": "1" * 40,
        "consumerRevision": "2" * 40,
        "registryDigest": "a" * 64,
    }
    original = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec="/unavailable/original/tokenizer.json",
        datasets=["/unavailable/original/prepared"],
        max_steps=1,
        semantic_v8_policy=policy,
    )
    moved = replace(
        original,
        codec="/unavailable/relocated/tokenizer.json",
        datasets=["/unavailable/relocated/prepared"],
    )
    codec = "c" * 64
    manifests = ["d" * 64, "b" * 64]
    model = SimpleNamespace(to_dict=lambda: {"width": 1})
    original_before = asdict(original)
    supplied_before = list(manifests)

    def identity(config=original, *, codec_fingerprint=codec, roots=manifests):
        return training_config_identity_v8(
            config,
            codec_fingerprint=codec_fingerprint,
            prepared_manifest_fingerprints=roots,
        )

    with ExitStack() as stack:
        for target in (
            "builtins.open",
            "io.open",
            "pathlib.Path.open",
            "pathlib.Path.read_text",
            "pathlib.Path.read_bytes",
            "pathlib.Path.stat",
        ):
            stack.enter_context(
                patch(target, side_effect=AssertionError("Unexpected filesystem I/O"))
            )
        expected = identity()
        assert identity(moved, roots=list(reversed(manifests))) == expected
        objective = position_objective(
            original,
            model,
            SimpleNamespace(fingerprint=codec),
            [0],
            prepared_manifest_fingerprints=manifests,
        )
        assert objective["trainingConfigDigest"] == stable_algorithm_hash(expected)
    assert asdict(original) == original_before and manifests == supplied_before
    passed("pure identity and position objective perform no hidden file reads")
    passed("copied locators and reordered distinct manifest fingerprints retain identity")
    passed("physical configuration and supplied fingerprints remain unchanged")
    for name, changed in (
        ("codec content", identity(codec_fingerprint="e" * 64)),
        ("prepared content", identity(roots=["e" * 64, "b" * 64])),
        ("omitted root", identity(roots=["b" * 64])),
        ("extra empty root identity", identity(roots=[*manifests, "e" * 64])),
        ("seed", identity(replace(original, seed=original.seed + 1))),
        (
            "source policy",
            identity(
                replace(original, semantic_v8_policy={**policy, "producerRevision": "3" * 40})
            ),
        ),
        (
            "window semantics",
            identity(replace(original, max_window_records=original.max_window_records + 1)),
        ),
    ):
        assert changed != expected
        passed(name + " changes the content/objective identity")
    for name, change in (
        ("empty fingerprints", {"roots": []}),
        ("duplicate fingerprints", {"roots": [manifests[0], manifests[0]]}),
        ("uppercase fingerprint", {"roots": ["A" * 64]}),
        ("non-string fingerprint", {"roots": [True]}),
        ("malformed codec fingerprint", {"codec_fingerprint": "not-a-digest"}),
    ):
        refuses(name, lambda change=change: identity(**change), "exact codec and distinct")

    _resume_compatible(asdict(original), moved, False)
    passed("matching ordinary V8 policy permits only the locator compatibility check")
    refuses(
        "missing original codec field cannot be excused by relocation",
        lambda: _resume_compatible(
            {key: value for key, value in asdict(original).items() if key != "codec"}, moved, False
        ),
        "changed codec",
    )
    ordinary = TrainConfig(purpose="diagnostic", phase="B0", model="tiny", max_steps=1)
    for field, value in (("codec", "relocated-codec"), ("datasets", ["relocated-data"])):
        refuses(
            "non-V8 " + field + " remains strict",
            lambda field=field, value=value: _resume_compatible(
                asdict(ordinary), replace(ordinary, **{field: value}), False
            ),
            "changed " + field,
        )
    distributed = replace(
        original,
        decoupled=DecoupledConfig("https://invalid.example", "controlled", workers=1, worker_id=0),
    )
    for field, value in (("codec", moved.codec), ("datasets", moved.datasets)):
        refuses(
            "decoupled V8 " + field + " remains strict",
            lambda field=field, value=value: _resume_compatible(
                asdict(distributed), replace(distributed, **{field: value}), False
            ),
            "changed " + field,
        )
    for name, candidate in (
        (
            "window configuration",
            replace(original, max_window_records=original.max_window_records + 1),
        ),
        (
            "registry policy",
            replace(original, semantic_v8_policy={**policy, "registryDigest": "f" * 64}),
        ),
    ):
        refuses(
            name + " still refuses exact resume",
            lambda candidate=candidate: _resume_compatible(asdict(original), candidate, False),
            "Resume configuration changed",
        )

    old_path_identity = asdict(original)
    for key in (
        "output",
        "max_steps",
        "checkpoint_every",
        "keep_checkpoints",
        "log_every",
        "device",
    ):
        old_path_identity.pop(key, None)
    old_objective = {**objective, "trainingConfigDigest": stable_algorithm_hash(old_path_identity)}
    assert old_objective != objective
    old_state = empty_position_exposure(
        registry_digest=policy["registryDigest"], objective=old_objective
    )
    old_payload = {"train_config": asdict(original), "semantic_v8_position_exposure": old_state}
    old_metadata = {"global_step": 0, "semantic_v8_position_digest": old_state["digest"]}
    retained_before = copy.deepcopy(old_payload)
    refuses(
        "historical path-bound position receipt is not adopted into the new objective",
        lambda: restore_position_exposure(
            old_payload,
            registry_digest=policy["registryDigest"],
            objective=objective,
            global_step=0,
            checkpoint_metadata=old_metadata,
        ),
        "objective",
    )
    assert old_payload == retained_before
    passed("refused historical position receipt remains unchanged")
    current_state = empty_position_exposure(
        registry_digest=policy["registryDigest"], objective=objective
    )
    current_payload = {
        "train_config": asdict(original),
        "semantic_v8_position_exposure": current_state,
    }
    metadata = {"global_step": 0, "semantic_v8_position_digest": current_state["digest"]}
    assert (
        restore_position_exposure(
            current_payload,
            registry_digest=policy["registryDigest"],
            objective=objective,
            global_step=0,
            checkpoint_metadata=metadata,
        )
        == current_state
    )
    passed("new content-bound empty history retains exact checkpoint metadata binding")
    refuses(
        "wrong checkpoint position digest is rejected",
        lambda: restore_position_exposure(
            current_payload,
            registry_digest=policy["registryDigest"],
            objective=objective,
            global_step=0,
            checkpoint_metadata={**metadata, "semantic_v8_position_digest": "f" * 64},
        ),
        "checkpoint metadata",
    )

    # Reuse the frozen existing collector control's explicitly injected input
    # boundaries, then execute the real collector again before/after relocation.
    from tests import test_semantic_v8_target_inventory as controlled

    module_from_spec = importlib.util.module_from_spec
    loaded = []

    def capture(spec):
        module = module_from_spec(spec)
        if spec.name == "inventory_control":
            loaded.append(module)
        return module

    with tempfile.TemporaryDirectory(prefix="v8-content-identity-review-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(importlib.util, "module_from_spec", capture)
            controlled.test_complete_inventory_keeps_program_identity_and_exact_prepared_targets(
                source, monkeypatch, None
            )
            assert len(loaded) == 1
            collector = loaded[0]
            config = TrainConfig(
                purpose="diagnostic",
                phase="SE",
                model="tiny",
                codec="controlled",
                datasets=[str(source)],
                max_steps=1,
                generated_capacity=128,
                microbatch_episodes=1,
                max_accumulation_windows=1,
                semantic_v8_policy=policy,
            )
            ids = [f"source-{index:02d}" for index in range(64)]
            before = collector.target_inventory(asdict(config), ids)
            destination = Path(directory) / "moved"
            shutil.copytree(source, destination)
            source.rename(Path(directory) / "original-unavailable")
            after = collector.target_inventory(
                asdict(replace(config, codec="relocated-controlled", datasets=[str(destination)])),
                ids,
            )
            assert before == after
            compared = position_objective(
                config,
                model,
                SimpleNamespace(fingerprint=controlled.CODEC),
                [0],
                prepared_manifest_fingerprints=[controlled.MANIFEST],
            )
            assert before["trainingConfigDigest"] == compared["trainingConfigDigest"]
            assert before["optimizerUpdates"] == 0
            assert before["trainEpisodeCount"] == before["windowCount"] == 48
    passed(
        "actual controlled 48-window collector output survives relocation with old root absent",
        optimizerUpdates=0,
        sourceAdmission="injected",
        planner="injected",
    )
    passed(
        "actual collector digest agrees with position objective call site",
        trainingConfigDigest=before["trainingConfigDigest"],
    )

    sources = [
        "src/megatoken/training/engine.py",
        "src/megatoken/training/positions_v8.py",
        "src/megatoken/training/target_inventory_v8.py",
        "scripts/qualify_semantic_v8_target_inventory.py",
        "tests/test_semantic_v8_target_inventory.py",
    ]
    report = {
        "schema": "semantic-v8-content-identity-independent-review/1",
        "reviewedRevision": revision,
        "reviewedTree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "pytest": pytest.__version__,
        },
        "checks": checks,
        "totals": {"passed": len(checks), "failed": 0, "skipped": 0},
        "optimizerUpdates": 0,
        "decoderForwardPasses": 0,
        "controlledCollectorInvocations": 3,
        "distinctControlledCollectorWindows": 48,
        "controlledCollectorWindowTraversals": 144,
        "fixtureBoundaries": "Collector admission, codec and planner are injected controlled boundaries; no real source/data or learned qualification",
        "sources": {name: file_identity(root / name) for name in sources},
        "probe": file_identity(Path(__file__)),
        "learnedSuccess": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": len(checks),
                "optimizerUpdates": 0,
                "reviewedRevision": revision,
            }
        )
    )


if __name__ == "__main__":
    main()
