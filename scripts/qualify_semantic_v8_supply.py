#!/usr/bin/env python3
"""Qualify actual V8 source supply for the ordinary SE curriculum.

This bounded gate rechecks prepared artifacts, exact source population, both
SE corpus splits, and the existing production behavior gate. It does not build
a model, allocate an optimizer, train, or qualify a full launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import LING_TOKENIZER_SHA256, LING_VOCAB_SIZE, ByteCodec
from megatoken.io import strict_json_loads
from megatoken.training.semantic_v8 import preflight_semantic_training_v8

EXPECTED_SE_BEHAVIORS = frozenset(
    {
        "semantic_code_invocation",
        "semantic_tool_feedback",
        "semantic_async",
        "semantic_callback",
        "semantic_repeated_invocation",
        "semantic_child",
        "semantic_source_reread",
        "semantic_recovery",
    }
)
SPLITS = ("train", "validation", "test")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_expected_source_ids(path: str | Path, registry_digest: str) -> tuple[str, ...]:
    """Read the independently supplied complete current source inventory."""
    value = strict_json_loads(Path(path).read_bytes())
    if isinstance(value, list):
        source_ids = value
    elif isinstance(value, dict) and set(value) == {"schema", "registryDigest", "sourceIds"}:
        _require(
            value["schema"] == "semantic-v8-source-population/1"
            and value["registryDigest"] == registry_digest,
            "Expected source population is bound to another registry",
        )
        source_ids = value["sourceIds"]
    else:
        raise ValueError(
            "Expected source inventory must be a JSON array or bound population object"
        )
    _require(
        isinstance(source_ids, list)
        and bool(source_ids)
        and all(isinstance(item, str) and item for item in source_ids)
        and len(set(source_ids)) == len(source_ids),
        "Expected source inventory must contain unique nonempty source IDs",
    )
    return tuple(sorted(source_ids))


def validate_registry_population(registry, *, registry_digest: str, expected_source_ids):
    """Validate the full registry and return the frozen split of every source ID."""
    _require(
        isinstance(registry, dict)
        and registry.get("schema") == "semantic-region-family-registry/1"
        and registry.get("digest") == registry_digest
        and stable_algorithm_hash(
            {key: value for key, value in registry.items() if key != "digest"}
        )
        == registry_digest,
        "Prepared source registry does not match the requested digest",
    )
    sources, components = registry.get("sources"), registry.get("components")
    _require(
        isinstance(sources, list)
        and bool(sources)
        and len(sources) == len(expected_source_ids)
        and isinstance(components, list),
        "Prepared source registry must contain every expected source and its component",
    )
    source_ids = [row.get("id") for row in sources if isinstance(row, dict)]
    _require(
        len(source_ids) == len(sources)
        and all(isinstance(source_id, str) and source_id for source_id in source_ids),
        "Prepared source registry differs from the full expected source inventory",
    )
    source_id_set = set(source_ids)
    _require(
        len(source_ids) == len(source_id_set) and source_id_set == set(expected_source_ids),
        "Prepared source registry differs from the full expected source inventory",
    )
    source_splits: dict[str, str] = {}
    for component in components:
        _require(
            isinstance(component, dict)
            and isinstance(component.get("sourceIds"), list)
            and component.get("split") in SPLITS,
            "Prepared source registry contains a malformed component or split",
        )
        for source_id in component["sourceIds"]:
            _require(
                isinstance(source_id, str)
                and source_id in source_id_set
                and source_id not in source_splits,
                "Prepared source registry duplicates or invents a source assignment",
            )
            source_splits[source_id] = component["split"]
    _require(
        set(source_splits) == source_id_set
        and all(
            any(split == candidate for split in source_splits.values()) for candidate in SPLITS
        ),
        "Prepared source registry must assign every source once across train, validation, and test",
    )
    return source_splits


def validate_observed_population(observed, expected_split_by_source):
    """Require one actual prepared episode for each frozen source, in its split."""
    _require(isinstance(observed, list), "Observed prepared source population is malformed")
    observed_split_by_source = {}
    for row in observed:
        _require(
            isinstance(row, dict)
            and isinstance(row.get("sourceId"), str)
            and row.get("split") in SPLITS,
            "Prepared source population contains a malformed episode identity",
        )
        source_id, split = row["sourceId"], row["split"]
        _require(
            source_id in expected_split_by_source and source_id not in observed_split_by_source,
            "Prepared corpus contains an unregistered or duplicate source ID",
        )
        _require(
            expected_split_by_source[source_id] == split,
            "Prepared source episode differs from its frozen registry split",
        )
        observed_split_by_source[source_id] = split
    missing = set(expected_split_by_source) - set(observed_split_by_source)
    _require(not missing, f"Prepared corpus is missing frozen source IDs: {sorted(missing)}")
    return {
        split: sorted(
            source_id
            for source_id, assigned in observed_split_by_source.items()
            if assigned == split
        )
        for split in SPLITS
    }


def validate_behavior_census(census, required_behaviors=EXPECTED_SE_BEHAVIORS):
    """Require all canonical SE behaviors in both production data splits."""
    if isinstance(required_behaviors, dict):
        requirements = required_behaviors
        names = set(requirements)
    else:
        names = set(required_behaviors)
        requirements = {name: 1 for name in names}
    _require(
        names == EXPECTED_SE_BEHAVIORS
        and all(type(count) is int and count > 0 for count in requirements.values()),
        "SE behavior inventory differs from the eight production requirements",
    )
    for split in ("train", "validation"):
        row = census.get(split) if isinstance(census, dict) else None
        _require(
            isinstance(row, dict) and row.get("episodes", 0) > 0,
            f"Production SE supply has no {split} episodes",
        )
        behaviors = row.get("behaviors", {})
        missing = [
            name
            for name in sorted(names)
            if behaviors.get(name, {}).get("familyCount", 0) < requirements[name]
        ]
        _require(not missing, f"Production SE {split} behavior census is missing: {missing}")


def _git_environment():
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _checked_consumer_revision(expected_revision: str) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    env = _git_environment()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, env=env, timeout=10
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
        env=env,
        timeout=10,
    ).strip()
    _require(
        actual == expected_revision and not dirty,
        "Supply qualification requires the exact clean pinned consumer checkout",
    )
    import megatoken

    _require(
        Path(megatoken.__file__).resolve().is_relative_to(root / "src"),
        "Imported megatoken package is not from the pinned consumer checkout",
    )
    return {"revision": actual, "clean": True, "root": str(root)}


def _file_identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _collect_actual_census(corpus, source_split_by_id, required_behaviors):
    from megatoken.data.semantic_corpus_v8 import semantic_behaviors_v8

    observed = []
    entries: dict[str, dict[str, object]] = {}
    census = {
        split: {
            "episodes": 0,
            "sourceIds": [],
            "behaviors": defaultdict(lambda: {"sourceIds": set(), "families": set()}),
        }
        for split in SPLITS
    }
    for episode in corpus.iter_episodes():
        try:
            annotations = json.loads(episode.supervision.annotations_json)
            semantic = annotations["semantic_execution"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Prepared semantic episode has no readable source identity") from error
        source_id = semantic.get("sourceId")
        split = episode.lineage.frozen_split
        _require(
            isinstance(source_id, str)
            and source_id in source_split_by_id
            and split in SPLITS
            and source_id not in entries,
            "Prepared episode has an unknown, duplicate, or malformed source/split identity",
        )
        _require(
            source_split_by_id[source_id] == split,
            "Prepared source episode differs from its frozen registry split",
        )
        observed.append({"sourceId": source_id, "split": split})
        entries[source_id] = {"split": split, "behaviors": sorted(semantic_behaviors_v8(episode))}
        row = census[split]
        row["episodes"] += 1
        row["sourceIds"].append(source_id)
        for behavior in entries[source_id]["behaviors"]:
            row["behaviors"][behavior]["sourceIds"].add(source_id)
            row["behaviors"][behavior]["families"].add(episode.lineage.family_id)
    population_by_split = validate_observed_population(observed, source_split_by_id)
    normalized = {}
    for split, row in census.items():
        normalized[split] = {
            "episodes": row["episodes"],
            "sourceIds": sorted(row["sourceIds"]),
            "behaviors": {
                behavior: {
                    "sourceCount": len(values["sourceIds"]),
                    "sourceIds": sorted(values["sourceIds"]),
                    "familyCount": len(values["families"]),
                    "families": sorted(values["families"]),
                }
                for behavior, values in sorted(row["behaviors"].items())
            },
        }
    validate_behavior_census(normalized, required_behaviors)
    _require(
        population_by_split == {split: normalized[split]["sourceIds"] for split in SPLITS},
        "Prepared episode census differs from the frozen source population",
    )
    return normalized, entries


def qualify_semantic_v8_supply(
    *,
    prepared_roots,
    tokenizer_json,
    expected_source_ids_path,
    producer_revision,
    consumer_revision,
    registry_digest,
    max_history=None,
    max_positions=None,
):
    """Rebuild every prepared V8 episode and qualify only the production SE supply."""
    from megatoken.data import PreparedCorpus
    from megatoken.data.recipes import stage_recipe, validate_stage_supply

    _require(
        isinstance(producer_revision, str) and REVISION.fullmatch(producer_revision),
        "A pinned producer revision is required",
    )
    _require(
        isinstance(consumer_revision, str) and REVISION.fullmatch(consumer_revision),
        "A pinned consumer revision is required",
    )
    _require(
        isinstance(registry_digest, str) and HASH.fullmatch(registry_digest),
        "A pinned source registry digest is required",
    )
    _require(
        max_history is None or type(max_history) is int and max_history > 0,
        "Maximum history must be a positive integer when supplied",
    )
    _require(
        max_positions is None or type(max_positions) is int and max_positions > 0,
        "Maximum positions must be a positive integer when supplied",
    )
    roots = tuple(Path(root).resolve(strict=True) for root in prepared_roots)
    _require(
        bool(roots) and len(set(roots)) == len(roots),
        "Supply qualification requires distinct prepared shard roots",
    )
    inventory_path = Path(expected_source_ids_path).resolve(strict=True)
    expected_ids = read_expected_source_ids(inventory_path, registry_digest)
    expected_inventory = _file_identity(inventory_path)
    consumer = _checked_consumer_revision(consumer_revision)
    tokenizer_path = Path(tokenizer_json).resolve(strict=True)
    _require(tokenizer_path.is_file(), "The pinned tokenizer JSON file is required")
    codec = ByteCodec.from_tokenizer_json(
        tokenizer_path,
        expected_sha256=LING_TOKENIZER_SHA256,
        expected_vocab_size=LING_VOCAB_SIZE,
    )
    tokenizer_identity = _file_identity(tokenizer_path)
    policy = {
        "schema": "semantic-training-policy/8",
        "producerRevision": producer_revision,
        "consumerRevision": consumer_revision,
        "registryDigest": registry_digest,
    }
    preflight = preflight_semantic_training_v8(
        [str(root) for root in roots],
        codec=codec,
        policy=policy,
        max_history=max_history,
        max_positions=max_positions,
    )
    recipe = stage_recipe("SE")
    required = recipe.constraints.get("required_behaviors", {})
    _require(
        set(required) == EXPECTED_SE_BEHAVIORS,
        "Canonical SE behavior requirements changed; update this bounded qualifier deliberately",
    )
    all_corpus = PreparedCorpus([str(root) for root in roots], split=None)
    try:
        _require(len(all_corpus) > 0, "Prepared V8 corpus is empty")
        registries = []
        manifests = []
        for root in roots:
            registry_path = root / "semantic-v8-families.json"
            manifest_path = root / "manifest.json"
            registry = strict_json_loads(registry_path.read_bytes())
            registries.append(registry)
            raw_manifest = manifest_path.read_bytes()
            manifest = strict_json_loads(raw_manifest)
            manifests.append(
                {
                    "root": str(root),
                    "fingerprint": manifest["fingerprint"],
                    "sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "bytes": len(raw_manifest),
                }
            )
        registry = registries[0]
        _require(
            all(candidate == registry for candidate in registries),
            "Prepared roots do not share the exact full frozen registry",
        )
        source_split_by_id = validate_registry_population(
            registry, registry_digest=registry_digest, expected_source_ids=expected_ids
        )
        census, episode_identities = _collect_actual_census(
            all_corpus, source_split_by_id, required
        )
        corpus_identity = all_corpus.identity
    finally:
        all_corpus.close()

    stage_config = SimpleNamespace(phase="SE", use_stage_mixture=True)
    stage_checks = {}
    for split in ("train", "validation"):
        with PreparedCorpus([str(root) for root in roots], split=split) as corpus:
            _require(len(corpus) > 0, f"Production SE supply has no {split} episodes")
            validate_stage_supply(stage_config, corpus, split=split)
        stage_checks[split] = {"status": "passed", "requiredBehaviors": dict(required)}

    return {
        "schema": "semantic-v8-production-supply-qualification/1",
        "qualification": "ordinary-prepared-source-supply-only",
        "status": "passed",
        "phase": "SE",
        "producerRevision": producer_revision,
        "consumer": consumer,
        "contractDigest": CONTRACT_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "interoperabilityDigest": INTEROPERABILITY_DIGEST,
        "policy": policy,
        "sourceRegistry": registry,
        "sourceRegistryDigest": registry_digest,
        "expectedSourceInventory": expected_inventory,
        "expectedSourceIds": list(expected_ids),
        "population": {
            "count": len(episode_identities),
            "bySplit": {split: census[split]["sourceIds"] for split in SPLITS},
        },
        "preparedRoots": manifests,
        "corpusIdentity": corpus_identity,
        "codec": {
            **tokenizer_identity,
            "fingerprint": codec.fingerprint,
            "vocabularySize": LING_VOCAB_SIZE,
        },
        "geometry": {"maxHistory": max_history, "maxPositions": max_positions},
        "preflight": preflight,
        "behaviorCensus": census,
        "stageSupply": stage_checks,
        "optimizerSteps": 0,
        "learnedSuccess": None,
        "fullLaunchQualified": False,
        "modelConstructed": False,
        "optimizerAllocated": False,
        "scope": (
            "Every prepared episode was rebuilt and checked against its native V8/source receipts; "
            "the complete observed source population and both ordinary SE splits passed the existing "
            "stage-supply gate. No learner, optimizer, donor, or full launch was qualified."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--expected-source-ids", type=Path, required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--registry-digest", required=True)
    parser.add_argument("--max-history", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Qualification reports are immutable; choose a new output path")
    try:
        report = qualify_semantic_v8_supply(
            prepared_roots=args.prepared_root,
            tokenizer_json=args.tokenizer_json,
            expected_source_ids_path=args.expected_source_ids,
            producer_revision=args.producer_revision,
            consumer_revision=args.consumer_revision,
            registry_digest=args.registry_digest,
            max_history=args.max_history,
            max_positions=args.max_positions,
        )
        output = args.output.resolve()
        if any(output.is_relative_to(Path(root).resolve()) for root in args.prepared_root):
            parser.error("Qualification report must be outside prepared shard roots")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": report["status"],
                "qualification": report["qualification"],
                "registryDigest": report["sourceRegistryDigest"],
                "population": report["population"]["count"],
                "fullLaunchQualified": False,
                "optimizerSteps": 0,
                "report": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
