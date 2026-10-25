#!/usr/bin/env python3
"""Materialize the frozen seed/cohort/budget training matrix; never start training."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import write_json
from megatoken.training.config import TrainConfig
from megatoken.training.semantic_budget import validate_stage_supply
from megatoken.training.semantic_evaluation import verify_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bindings",
        type=Path,
        required=True,
        help="cohorts, datasets, codec, hindsight_targets, engram_imports and decoupled bindings",
    )
    parser.add_argument("--engineering-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/semantic/experiment-plan-v4.json").read_text())
    bindings = json.loads(options.bindings.read_text())
    options.output.mkdir(parents=True, exist_ok=True)
    generated = []
    codec_path = Path(bindings["codec"])
    if codec_path.is_dir():
        metadata = json.loads((codec_path / "codec.json").read_text())
        tokenizer_path = codec_path / metadata["tokenizer_file"]
        if tokenizer_path.parent.resolve() != codec_path.resolve():
            raise ValueError("Tokenizer must stay within its immutable codec directory")
    else:
        tokenizer_path = codec_path
    codec = ByteCodec.from_tokenizer_json(tokenizer_path, expected_sha256=LING_TOKENIZER_SHA256)
    engineering = json.loads(options.engineering_receipt.read_text())
    supplies = []
    for cohort_name, cohort_plan in plan["cohorts"].items():
        cohort_path = Path(bindings["cohorts"][cohort_name]).resolve()
        cohort = json.loads(cohort_path.read_text())
        verify_cohort(cohort)
        if cohort["name"] != cohort_name or cohort["thresholds"] != plan["acceptanceThresholds"]:
            raise ValueError("Bound cohort differs from the frozen task/threshold declaration")
        for representation in cohort_plan["representations"]:
            supplies.append(
                validate_stage_supply(
                    bindings["datasets"][cohort_name][representation],
                    phase=cohort_plan["phases"][representation],
                    representation=representation,
                    codec=codec,
                    producer_revision=engineering["repositories"]["tracing-js-self"],
                    consumer_revision=engineering["repositories"]["megatoken"],
                )
            )
            for regime in plan["regimes"]:
                for seed in plan["seeds"]:
                    name = f"{cohort_name}-{representation}-{regime}-{seed}"
                    config = json.loads(
                        (root / f"configs/semantic/train-seed-{seed}.json").read_text()
                    )
                    config.update(
                        {
                            key: copy.deepcopy(bindings[key])
                            for key in ("codec", "hindsight_targets", "engram_imports", "decoupled")
                        }
                    )
                    config.update(
                        phase=cohort_plan["phases"][representation],
                        datasets=bindings["datasets"][cohort_name][representation],
                        output=str(options.output / "runs" / name),
                        microbatch_episodes=1,
                        max_accumulation_windows=1,
                        max_window_records=1,
                    )
                    config["decoupled"]["run_id"] = name
                    config["decoupled"]["workers"] = 1
                    config["decoupled"]["worker_id"] = 0
                    config["semantic_experiment"] = {
                        "schema": "semantic-training-experiment/4",
                        "regime": regime,
                        "limit": next(iter(plan["budgets"][regime].values())),
                        "representation": representation,
                        "cohort": str(cohort_path),
                        "engineering_receipt": str(options.engineering_receipt.resolve()),
                        "max_token_overshoot": 65535,
                    }
                    TrainConfig.from_dict(config, expand_data=False)
                    path = options.output / (name + ".json")
                    write_json(path, config)
                    generated.append(
                        {
                            "path": str(path),
                            "cohortDigest": cohort["digest"],
                            "seed": seed,
                            "regime": regime,
                            "representation": representation,
                        }
                    )
    write_json(
        options.output / "matrix.json",
        {
            "schema": "semantic-training-matrix/4",
            "runs": generated,
            "stageSupply": supplies,
            "optimizerSteps": 0,
        },
    )
    print(json.dumps({"configurations": len(generated), "optimizerSteps": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
