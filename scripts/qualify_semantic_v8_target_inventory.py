#!/usr/bin/env python3
"""Inventory positive-weight V8 target windows from the ordinary native planner.

This is a read-only preflight: it prepares the exact train windows and native
loss plans, but does not execute backward or change model weights.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from megatoken.data import PreparedCorpus, iter_windows
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.loading import build_model, load_model_config
from megatoken.training.config import TrainConfig
from megatoken.training.engine import batch_counters, prepare_batches, window_config
from megatoken.training.semantic_v8 import (
    load_semantic_v8_codec,
    merge_semantic_v8_target_exposures,
    preflight_semantic_training_v8,
    semantic_v8_target_exposure_from_batches,
    semantic_v8_target_step_summary,
)
from megatoken.training.stream import EpisodeWindow
from megatoken.training.target_inventory_v8 import inventory_episode_source_id


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text())


def target_inventory(config_source, expected_source_ids_source=None):
    config_raw = (
        read_json(config_source) if isinstance(config_source, (str, Path)) else config_source
    )
    config = TrainConfig.from_dict(config_raw)
    config_identity = asdict(config)
    for non_objective in (
        "output",
        "max_steps",
        "checkpoint_every",
        "keep_checkpoints",
        "log_every",
        "device",
    ):
        config_identity.pop(non_objective, None)
    require(config.semantic_v8_policy is not None, "Inventory requires an explicit V8 policy")
    require(
        config.use_stage_mixture is False,
        "Inventory currently requires the exact flat V8 support stream",
    )
    require(
        config.microbatch_episodes == 1
        and config.max_accumulation_windows == 1
        and config.decoupled is None
        and os.environ.get("WORLD_SIZE", "1") == "1",
        "Inventory requires one synchronous, ordinary optimizer window per update",
    )
    require(
        len(config.datasets) == 1,
        "Inventory requires one immutable prepared support root",
    )

    overrides = {
        **config.model_overrides,
        "expert_backend": config.expert_backend,
        "checkpoint_stages": config.gradient_checkpointing,
    }
    model_config = load_model_config(config.model, **overrides)
    codec = load_semantic_v8_codec(config.codec, vocab_size=model_config.vocab_size)
    require(
        config.generated_capacity <= model_config.max_capacity,
        "Configured generated capacity exceeds model",
    )
    supply = preflight_semantic_training_v8(
        config.datasets,
        codec=codec,
        policy=config.semantic_v8_policy,
        max_history=model_config.max_history_chunks,
        max_positions=model_config.max_capacity,
    )
    registry_digest = config.semantic_v8_policy["registryDigest"]
    require(
        supply["registryDigest"] == registry_digest,
        "Prepared support registry differs from policy",
    )

    prepared_root = Path(config.datasets[0]).resolve()
    manifest = read_json(prepared_root / "manifest.json")
    registry = read_json(prepared_root / "semantic-v8-families.json")
    require(
        registry.get("digest") == registry_digest,
        "Prepared registry digest differs from V8 policy",
    )
    registry_ids = {row["id"] for row in registry["sources"]}
    if expected_source_ids_source is not None:
        inventory = (
            read_json(expected_source_ids_source)
            if isinstance(expected_source_ids_source, (str, Path))
            else {
                "sourceIds": list(expected_source_ids_source),
                "schema": "semantic-v8-source-population/1",
                "registryDigest": registry_digest,
            }
        )
        expected_ids = inventory.get("sourceIds")
        require(
            inventory.get("schema") == "semantic-v8-source-population/1"
            and inventory.get("registryDigest") == registry_digest
            and isinstance(expected_ids, list)
            and len(expected_ids) == 64
            and len(set(expected_ids)) == 64
            and set(expected_ids) == registry_ids,
            "Support source inventory is invalid or differs from the prepared registry",
        )
    else:
        expected_ids = sorted(registry_ids)
    expected_ids = set(expected_ids)
    split_by_id = {}
    for component in registry["components"]:
        for source_id in component["sourceIds"]:
            require(
                source_id not in split_by_id,
                "Prepared registry assigns a source to multiple splits",
            )
            split_by_id[source_id] = component["split"]
    expected_train_ids = {source_id for source_id, split in split_by_id.items() if split == "train"}
    split_counts = {
        split: sum(value == split for value in split_by_id.values())
        for split in ("train", "validation", "test")
    }
    require(
        len(expected_ids) == 64
        and split_counts == {"train": 48, "validation": 12, "test": 4}
        and set(split_by_id) == expected_ids,
        "Inventory requires the exact 64-source, 48/12/4 registry",
    )

    model = build_model(
        model_config,
        codec,
        device="cpu",
        precision=config.precision,
        packs=config.packs,
    )
    effective_weights = dict(config.loss_weights)
    effective_weights["router_balance"] = 0.0
    windows_config = window_config(config)
    exposures = []
    observed_sources = set()
    observed_windows = set()
    update_window_count = 0
    one_pass_meter_count = 0
    prepared_policy_records = set()
    available_records = set()
    available_denominators = {}
    train_episode_count = 0
    with PreparedCorpus(config.datasets, split="train") as corpus:
        require(
            corpus.codec_fingerprint == codec.fingerprint,
            "Prepared corpus tokenizer differs",
        )
        require(
            len(corpus) == len(expected_train_ids),
            "Prepared training episode count differs from the frozen source split",
        )
        for index in range(len(corpus)):
            episode, assets = corpus.fetch(index)
            source_id = inventory_episode_source_id(episode, registry_digest=registry_digest)
            require(
                source_id in expected_train_ids,
                "Prepared training episode is outside the frozen train source inventory",
            )
            require(
                source_id not in observed_sources,
                "Prepared support has a duplicate training source",
            )
            require(
                episode.lineage.frozen_split == "train",
                "Inventory enumerated a nontraining episode",
            )
            observed_sources.add(source_id)
            train_episode_count += 1
            # Derive the target universe independently of the window iterator
            # and loss planner. Otherwise both inventory and optimizer census
            # could agree after silently dropping the same prepared target.
            prepared_policy_records.update(
                (source_id, record.logical_id, record.record_id)
                for record in episode.records
                if record.policy_mask
            )
            for window in iter_windows(episode, codec, windows_config):
                window_key = (source_id, window.index)
                require(
                    window_key not in observed_windows,
                    "Native window iterator duplicated a window",
                )
                observed_windows.add(window_key)
                item = EpisodeWindow(episode=episode, window=window, assets=assets)
                batches = prepare_batches(model, [item], config)
                update_window_count += int(any(any(batch.counts.values()) for batch in batches))
                if any(any(batch.counts.values()) for batch in batches):
                    one_pass_meter_count += sum(
                        getattr(batch_counters([batch]), config.schedule.meter) for batch in batches
                    )
                for batch in batches:
                    for step in batch.steps:
                        if not step.plans:
                            continue
                        for encoded, plan in zip(step.records, step.plans, strict=True):
                            record = encoded.record
                            if not record.policy_mask:
                                continue
                            available_records.add((source_id, record.logical_id, record.record_id))
                            for name, count in plan.denominators.items():
                                if count:
                                    available_denominators[name] = (
                                        available_denominators.get(name, 0) + count
                                    )
                exposures.append(
                    semantic_v8_target_exposure_from_batches(
                        batches,
                        registry_digest=registry_digest,
                        effective_loss_weights=effective_weights,
                    )
                )
    require(
        observed_sources == expected_train_ids,
        "Prepared support omitted or substituted a frozen training source",
    )
    require(observed_windows, "Prepared support has no train windows")
    require(
        prepared_policy_records == available_records,
        "Native plans omitted or invented a prepared policy target record",
    )
    phase_meter, phase_budget = config.schedule.meter, config.schedule.budget
    require(
        phase_meter == "P" and one_pass_meter_count <= phase_budget,
        "Measured one-pass window meter exceeds or differs from the exact phase budget",
    )

    merged = merge_semantic_v8_target_exposures(exposures, registry_digest=registry_digest)
    scored_sources = merged["sources"]
    positive_weight = semantic_v8_target_step_summary(merged)
    body = {
        "schema": "semantic-v8-positive-loss-target-inventory/1",
        "registryDigest": registry_digest,
        "sourceIds": sorted(expected_train_ids),
        "scoredSourceIds": sorted(scored_sources),
        "trainEpisodeCount": train_episode_count,
        "windowCount": len(observed_windows),
        "optimizerWindowCount": update_window_count,
        "phaseMeter": phase_meter,
        "phaseBudget": phase_budget,
        "onePassMeterCount": one_pass_meter_count,
        "preparedManifestFingerprint": manifest["fingerprint"],
        "codecFingerprint": codec.fingerprint,
        "trainingConfigDigest": stable_algorithm_hash(config_identity),
        "modelConfigDigest": stable_algorithm_hash(model_config.to_dict()),
        "windowConfig": asdict(windows_config),
        "effectiveLossWeights": effective_weights,
        "available": {
            "policyTargetRecordCount": len(available_records),
            "lossDenominators": dict(sorted(available_denominators.items())),
        },
        "positiveWeight": {
            key: positive_weight[key]
            for key in (
                "sourceCount",
                "logicalTargetCount",
                "physicalTargetRecordCount",
                "scoredOccurrences",
                "lossDenominators",
            )
        },
        "sources": scored_sources,
        "optimizerUpdates": 0,
        "learnedSuccess": None,
    }
    return {**body, "digest": stable_algorithm_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-source-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = target_inventory(args.config, args.expected_source_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "digest": result["digest"],
                "registryDigest": result["registryDigest"],
                "sourceCount": result["positiveWeight"]["sourceCount"],
                "windowCount": result["windowCount"],
                "optimizerWindowCount": result["optimizerWindowCount"],
                "phaseMeter": result["phaseMeter"],
                "phaseBudget": result["phaseBudget"],
                "onePassMeterCount": result["onePassMeterCount"],
                "logicalTargetCount": result["positiveWeight"]["logicalTargetCount"],
                "physicalTargetRecordCount": result["positiveWeight"]["physicalTargetRecordCount"],
                "scoredOccurrences": result["positiveWeight"]["scoredOccurrences"],
                "lossDenominators": result["positiveWeight"]["lossDenominators"],
                "optimizerUpdates": 0,
                "learnedSuccess": None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
