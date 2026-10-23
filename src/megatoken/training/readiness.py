"""Check actual production inputs and enabled objectives without allocating a model."""

from __future__ import annotations

import json
from pathlib import Path

from megatoken.data import PreparedCorpus
from megatoken.data.coverage import require_coverage
from megatoken.data.hindsight import HindsightTargets
from megatoken.data.recipes import stage_recipe
from megatoken.io import sha256_file, write_json
from megatoken.loading import load_codec, load_model_config
from megatoken.training.config import TrainConfig


def check(config):
    report = {"ready": False, "purpose": config.purpose, "checks": {}, "errors": []}
    targets = None
    try:
        paths = [
            *config.datasets,
            config.codec,
            config.packs,
            config.hindsight_targets,
            config.text_anchors,
            config.engram_import,
            *(config.engram_imports or []),
        ]
        if any(isinstance(path, str) and "PIN_" in path for path in paths):
            raise ValueError("Replace the training template's PIN_* paths with prepared artifacts")
        from megatoken.data.recipes import validate_training_donors

        validate_training_donors(config)
        report["checks"]["curriculum"] = stage_recipe(config.phase).training
        model = load_model_config(config.model, **config.model_overrides)
        codec = load_codec(config.codec, vocab_size=model.vocab_size)
        if config.codec == "byte" and config.model != "tiny" and not config.allow_byte_codec:
            raise ValueError("Production models require the pinned raw codec")
        if config.generated_capacity > model.max_capacity:
            raise ValueError("Generated capacity exceeds the model's maximum")
        with PreparedCorpus(config.datasets) as corpus:
            if corpus.codec_fingerprint != codec.fingerprint:
                raise ValueError("Corpus and training codec differ")
            report["checks"]["corpus"] = corpus.identity
            if config.use_stage_mixture:
                recipe = stage_recipe(config.phase)
                missing = [
                    stream
                    for stream, weight in recipe.mixture.items()
                    if weight > 0
                    and not any(
                        part.manifest["counts"].get(f"split:train|stream:{stream}", 0)
                        for part in corpus.parts
                    )
                ]
                if missing:
                    raise ValueError(f"Required stage streams are absent: {missing}")
                require_coverage(
                    corpus, recipe.constraints.get("required_behaviors", {}), recipe.mixture
                )
            from megatoken.training.autoencode import AutoencodePlan
            from megatoken.training.media_dependencies import required_media_modalities

            plan = AutoencodePlan.from_stage(stage_recipe(config.phase).constraints)
            active = (
                tuple(stage_recipe(config.phase).mixture)
                if config.use_stage_mixture
                else tuple({episode.stream for episode in corpus.iter_episodes()})
            )
            required = required_media_modalities(
                corpus,
                corpus.scored_media_modalities(active),
                autoencode_items=plan.items if config.use_stage_mixture else 0,
            )
            packs = (
                {}
                if not config.packs
                else json.loads(Path(config.packs).read_text()).get("packs", {})
            )
            names = {"audio": "pocket", "image": "sana", "video": "sana_video"}
            missing = [modality for modality in required if names[modality] not in packs]
            if missing:
                raise ValueError(f"Active objectives lack required media decoder packs: {missing}")
            if (
                config.use_stage_mixture
                and plan.items
                and not any(
                    corpus.feature_count("train", m) for m in ("text", "audio", "image", "video")
                )
            ):
                raise ValueError("Default encode-and-reconstruct objective has no feature items")
            report["checks"]["required_decoders"] = sorted(required)
            report["checks"]["stage_mixture"] = "preserved"
            if config.hindsight_targets:
                targets = HindsightTargets(
                    config.hindsight_targets,
                    config=config,
                    codec=codec,
                    value_dim=model.transducer_geometry.value_dim,
                )
                targets.validate_corpus(corpus.identity)
                report["checks"]["teacher"] = {
                    "manifest": targets.identity,
                    "encoding": targets.manifest["encoding_identity"],
                    "coverage": targets.manifest.get("coverage", {}),
                }
            else:
                report["checks"]["teacher"] = "bounded diagnostic: causal retrieval only"
            report["checks"]["navigation_weight"] = config.navigation_weight
            report["checks"]["memory_updates"] = (
                "enabled" if config.trainable == "all" else "frozen"
            )
            report["checks"]["teacher_overlay"] = (
                "supported text windows"
                if config.trainable == "all" and config.hindsight_targets
                else "disabled for frozen scope"
                if config.trainable != "all"
                else "causal retrieval only"
            )
        if config.eval_every:
            with PreparedCorpus(config.datasets, split="validation") as validation:
                if not len(validation):
                    raise ValueError("Configured evaluation has no held-out validation episodes")
                if config.use_stage_mixture:
                    recipe = stage_recipe(config.phase)
                    require_coverage(
                        validation,
                        recipe.constraints.get("required_behaviors", {}),
                        recipe.mixture,
                        split="validation",
                    )
                report["checks"]["validation_episodes"] = len(validation)
        for name, path in [("packs", config.packs), ("text_anchors", config.text_anchors)]:
            if path:
                report["checks"][name] = sha256_file(path)
                if name == "packs":
                    from megatoken.omni.loader import read_locked_manifest

                    read_locked_manifest(path)
        for path in [config.engram_import] if config.engram_import else config.engram_imports or []:
            metadata = json.loads((Path(path) / "engram.json").read_text())
            if sha256_file(Path(path) / "descriptors.npz") != metadata["descriptor_sha256"]:
                raise ValueError("Imported donor descriptor checksum differs")
            if metadata["exact_rows"] != model.exact_rows:
                raise ValueError("Imported donor key space differs from the training model")
            report["checks"].setdefault("donor_assets", []).append(
                sha256_file(Path(path) / "engram.json")
            )
        report["ready"] = True
    except (ValueError, OSError, KeyError, TypeError) as error:
        report["errors"].append(str(error))
    finally:
        if targets is not None:
            targets.close()
    return report


def run(args):
    try:
        report = check(TrainConfig.load(args.config))
    except (ValueError, OSError) as error:
        report = {"ready": False, "checks": {}, "errors": [str(error)]}
    if args.report:
        write_json(args.report, report)
    return report
