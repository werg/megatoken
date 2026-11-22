#!/usr/bin/env python3
"""Verify a direct V8 text release against its sources, data, checkpoint and weights.

This first profile accepts one prepared root and one ordinary, single-rank,
fresh-stage checkpoint. It performs no optimizer updates, model forward passes
or live-service recovery. The producer separately verifies the complete source
qualification journal and its selection; this verifier checks every supplied
source and refuses unbound input assets.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.io import content_hash, sha256_file, strict_json_loads

DESCRIPTOR_SCHEMA = "semantic-v8-release-inputs/1"
REPORT_SCHEMA = "semantic-v8-native-release-verification/1"
MAX_JSON_BYTES = 128 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.stat().st_size <= MAX_JSON_BYTES, "Release JSON exceeds the reader bound")
    return strict_json_loads(path.read_bytes())


def same_json(left, right):
    """Native canonical JSON equality preserves booleans and numeric types."""
    return content_hash(left) == content_hash(right)


def canonical_relative(value):
    require(
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and "\x00" not in value
        and ":" not in value,
        "Release paths must be canonical relative POSIX paths",
    )
    path = PurePosixPath(value)
    require(
        bool(path.parts)
        and not path.is_absolute()
        and str(path) == value
        and all(part not in {".", ".."} for part in path.parts),
        "Release paths must be canonical relative POSIX paths",
    )
    return path


def checked_root(value):
    root = Path(os.path.abspath(value))
    for path in [*reversed(root.parents), root]:
        require(not path.is_symlink(), "Release root has a symlink ancestor")
    require(root.is_dir(), "Release root must be an existing directory")
    return root


def bound_path(root, value, *, directory=False):
    relative = canonical_relative(value)
    current = root
    for part in relative.parts:
        current /= part
        require(not current.is_symlink(), "Release member has a symlink ancestor")
    require(
        current.is_dir() if directory else current.is_file(),
        "Release role does not name the required existing regular file or directory",
    )
    return current


def bundle_inventory(root):
    """Hash every original member, including hidden journal files, without repair."""
    rows = []
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in sorted([*directories, *files]):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            require(
                stat.S_ISREG(mode) or stat.S_ISDIR(mode),
                "Release bundle contains a symlink or nonregular member",
            )
            relative = path.relative_to(root).as_posix()
            canonical_relative(relative)
            if stat.S_ISREG(mode):
                rows.append(
                    {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                )
    return sorted(rows, key=lambda row: row["path"])


def immutable_checkpoint_metadata(directory):
    """Refuse resolver repair/pointers before calling any ordinary checkpoint loader."""
    root = checked_root(directory)
    require(
        root.name != "latest.json"
        and not (root / "latest.json").exists()
        and not any(root.glob(".superseded-*")),
        "Release checkpoint requires a direct committed directory without pointers or repair state",
    )
    bound_path(root, "state.pt")
    metadata = read_json(bound_path(root, "metadata.json"))
    require(isinstance(metadata, dict), "Release checkpoint metadata must be an object")
    attachment = metadata.get("transducer_checkpoint")
    require(
        isinstance(attachment, dict)
        and isinstance(attachment.get("file"), str)
        and len(canonical_relative(attachment["file"]).parts) == 1,
        "Release checkpoint must bind a local transducer attachment",
    )
    bound_path(root, attachment["file"])
    return metadata


def validate_descriptor(document, root):
    require(
        isinstance(document, dict)
        and set(document)
        == {
            "schema",
            "producerRevision",
            "consumerRevision",
            "contractDigest",
            "fixtureDigest",
            "interoperabilityDigest",
            "roles",
        }
        and document.get("schema") == DESCRIPTOR_SCHEMA,
        "Release descriptor has an unsupported or incomplete field set",
    )
    for name in ("producerRevision", "consumerRevision"):
        require(
            isinstance(document[name], str)
            and re.fullmatch(r"[0-9a-f]{40}", document[name]) is not None,
            "Release descriptor requires exact repository revisions",
        )
    for name, expected in (
        ("contractDigest", CONTRACT_DIGEST),
        ("fixtureDigest", FIXTURE_DIGEST),
        ("interoperabilityDigest", INTEROPERABILITY_DIGEST),
    ):
        require(document[name] == expected, "Release descriptor changed the active V8 contract")
    roles = document["roles"]
    require(
        isinstance(roles, dict)
        and set(roles)
        == {"qualification", "tokenizer", "prepared", "checkpoint", "recipe", "served"},
        "Release descriptor roles are incomplete or unknown",
    )
    prepared = roles["prepared"]
    require(
        isinstance(prepared, list)
        and len(prepared) == 1
        and isinstance(prepared[0], dict)
        and set(prepared[0]) == {"path", "sourceArtifacts"},
        "The direct V8 text release profile requires exactly one prepared root",
    )
    paths = {
        name: bound_path(root, roles[name], directory=name not in {"tokenizer", "recipe"})
        for name in ("qualification", "tokenizer", "checkpoint", "recipe", "served")
    }
    paths["prepared"] = bound_path(root, prepared[0]["path"], directory=True)
    values = list(paths.values())
    require(
        all(
            left != right and not left.is_relative_to(right) and not right.is_relative_to(left)
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        ),
        "Release asset roles must not alias or contain one another",
    )
    sources = prepared[0]["sourceArtifacts"]
    require(isinstance(sources, list) and bool(sources), "Release source membership is empty")
    case_ids, source_ids, source_paths = set(), set(), set()
    for row in sources:
        require(
            isinstance(row, dict)
            and set(row) == {"caseId", "sourceId", "path"}
            and isinstance(row["caseId"], str)
            and bool(row["caseId"])
            and isinstance(row["sourceId"], str)
            and bool(row["sourceId"]),
            "Release source membership row is malformed",
        )
        path = bound_path(root, row["path"])
        require(
            path.is_relative_to(paths["qualification"])
            and row["caseId"] not in case_ids
            and row["sourceId"] not in source_ids
            and path not in source_paths,
            "Release source membership is duplicated or outside qualification",
        )
        case_ids.add(row["caseId"])
        source_ids.add(row["sourceId"])
        source_paths.add(path)
    return paths


def release_recipe(raw, descriptor, root):
    from megatoken.training.config import TrainConfig

    roles = descriptor["roles"]
    require(
        isinstance(raw, dict)
        and raw.get("codec") == roles["tokenizer"]
        and raw.get("datasets") == [roles["prepared"][0]["path"]],
        "Release recipe codec/datasets must exactly name the canonical bundled roles",
    )
    require(
        raw.get("model", "32x512") in {"tiny", "32x256", "32x512"},
        "Release recipe cannot refer to an unbound model configuration",
    )
    config = TrainConfig.from_dict(raw)
    require(
        config.semantic_v8_policy is not None
        and config.semantic_v8_policy["producerRevision"] == descriptor["producerRevision"]
        and config.semantic_v8_policy["consumerRevision"] == descriptor["consumerRevision"],
        "Release recipe policy differs from the exact paired source revisions",
    )
    require(
        not any(
            (
                config.packs,
                config.text_anchors,
                config.hindsight_targets,
                config.engram_import,
                config.engram_imports,
                config.decoupled,
                config.use_stage_mixture,
                config.use_effort_mixture,
            )
        ),
        "This release profile refuses external assets, donor imports and distributed/mixture routes",
    )
    # Only verified physical locators change. The original recipe bytes and the
    # checkpoint's original physical configuration remain separately retained.
    return replace(
        config,
        codec=str(root / roles["tokenizer"]),
        datasets=[str(root / roles["prepared"][0]["path"])],
    )


def checked_repository(root, expected, *, consumer=False):
    root = Path(root).resolve()
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=root, env=env, text=True, timeout=30
        ).strip()

    require(
        git("rev-parse", "HEAD") == expected
        and Path(git("rev-parse", "--show-toplevel")).resolve() == root
        and not git("status", "--porcelain", "--untracked-files=all"),
        "Release verification requires the exact clean repository revisions",
    )
    if consumer:
        import megatoken

        require(
            Path(megatoken.__file__).resolve().is_relative_to(root / "src")
            and Path(__file__).resolve().is_relative_to(root / "scripts"),
            "Release verifier imports must belong to the declared native checkout",
        )
    return {"revision": expected, "tree": git("rev-parse", "HEAD^{tree}"), "clean": True}


def source_membership(descriptor, root, registry_digest):
    rows, identities = [], set()
    for member in descriptor["roles"]["prepared"][0]["sourceArtifacts"]:
        path = bound_path(root, member["path"])
        document = read_json(path)
        binding = document.get("sidecar", {}).get("datasetBinding", {})
        source_id = binding.get("sourceId")
        require(
            document.get("public", {}).get("schema") == "semantic-execution/8"
            and isinstance(source_id, str)
            and bool(source_id)
            and source_id not in identities
            and binding.get("registryDigest") == registry_digest
            and binding.get("split") in {"train", "validation", "test"},
            "Release source artifact identity, registry or unique membership differs",
        )
        require(
            member.get("sourceId") == source_id,
            "Release source artifact differs from its declared sourceId",
        )
        identities.add(source_id)
        rows.append(
            {
                **member,
                "split": binding["split"],
                "sha256": sha256_file(path),
                "datasetBindingDigest": stable_algorithm_hash(binding),
            }
        )
    require(any(row["split"] == "train" for row in rows), "Release has no training sources")
    return rows


def prepared_semantics(root):
    """Read actual rows, payloads, labels, masks and every stored scope position."""
    from megatoken.data.storage import PreparedDataset

    with PreparedDataset(root, split=None) as dataset:
        manifest = dataset.manifest
        require(
            not manifest.get("feature_counts")
            and not manifest.get("feature_extractors")
            and not manifest.get("scored_media_counts"),
            "Direct V8 text releases cannot hide unbound media or teacher features",
        )
        episodes, membership = [], []
        for index in range(len(dataset)):
            episode = dataset[index]
            require(
                all(record.media_type == "text/plain; charset=utf-8" for record in episode.records),
                "Direct V8 text release contains unsupported media records",
            )
            value = episode.to_dict()
            # JSON whitespace/order is serialization, not training semantics.
            annotations = strict_json_loads(episode.supervision.annotations_json)
            value["supervision"]["annotations_json"] = annotations
            value = strict_json_loads(json.dumps(value, allow_nan=False))
            semantic = annotations["semantic_execution"]
            episodes.append(value)
            membership.append(
                {
                    "sourceId": semantic["sourceId"],
                    "split": episode.lineage.frozen_split,
                    "episodeId": episode.episode_id,
                    "records": len(episode.records),
                    "policyRecords": sum(record.policy_mask for record in episode.records),
                }
            )
        connection = dataset.connection
        require(
            connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == len(dataset)
            and connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0
            and connection.execute("SELECT COUNT(*) FROM feature_sources").fetchone()[0] == 0,
            "Prepared release index has hidden episodes or external feature assets",
        )
        index_rows = [
            list(row)
            for row in connection.execute(
                "SELECT episode_id,split,family_id,stream,source_id,tokens,scored_positions,"
                "outcome,verified_success FROM episodes ORDER BY entry_id"
            )
        ]
        scopes = [
            list(row)
            for row in connection.execute(
                "SELECT p.scope,p.position,e.episode_id FROM positions p "
                "JOIN episodes e USING(entry_id) ORDER BY p.scope,p.position"
            )
        ]
        require(
            len(scopes) == connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0],
            "Prepared release index contains orphan scope positions",
        )
        return {
            "episodes": episodes,
            "membership": membership,
            "indexRows": index_rows,
            "scopePositions": scopes,
            "manifestSemantics": {
                name: manifest[name]
                for name in (
                    "schema_version",
                    "complete",
                    "codec_fingerprint",
                    "chunking",
                    "counts",
                    "feature_counts",
                    "feature_extractors",
                    "scored_media_counts",
                    "sources",
                    "pipeline_fingerprint",
                )
            },
        }


def compare_prepared_semantics(bundled, regenerated, sources):
    require(
        same_json(bundled, regenerated),
        "Bundled prepared episodes, tokens, masks, splits or index semantics differ from fresh sources",
    )
    membership = bundled["membership"]
    require(
        sorted((row["sourceId"], row["split"]) for row in membership)
        == sorted((row["sourceId"], row["split"]) for row in sources),
        "Prepared source membership differs from the complete supplied source list",
    )
    return {
        "sourceCount": len(sources),
        "episodeCount": len(membership),
        "recordCount": sum(row["records"] for row in membership),
        "policyRecordCount": sum(row["policyRecords"] for row in membership),
        "semanticDigest": stable_algorithm_hash(bundled),
        "completeEpisodeTokenMaskAndSplitEquivalence": True,
        "regeneratedMetadataByteEqualityRequired": False,
        "membership": membership,
    }


def prepared_source_order(bundled, sources):
    """Use actual index order only after proving exact complete source membership."""
    membership = bundled["membership"]
    source_ids = [row["sourceId"] for row in sources]
    prepared_ids = [row["sourceId"] for row in membership]
    require(
        bool(source_ids)
        and len(source_ids) == len(set(source_ids))
        and len(prepared_ids) == len(set(prepared_ids))
        and len(source_ids) == len(prepared_ids)
        and set(source_ids) == set(prepared_ids),
        "Prepared source order requires exact unique complete supplied membership",
    )
    by_source = {row["sourceId"]: row for row in sources}
    require(
        all(by_source[row["sourceId"]]["split"] == row["split"] for row in membership),
        "Prepared source order changed a supplied source split",
    )
    return [by_source[source_id] for source_id in prepared_ids]


def regenerate_prepared(
    root,
    descriptor,
    paths,
    *,
    bundled,
    sources,
    producer_root,
    consumer_root,
    model_config,
    timeout_seconds,
):
    sources = prepared_source_order(bundled, sources)
    require(
        type(timeout_seconds) is int and 1 <= timeout_seconds <= 900,
        "Source recheck timeout must be an integer from 1 through 900 seconds",
    )
    with tempfile.TemporaryDirectory(prefix="semantic-v8-release-recheck-") as temporary:
        destination = Path(temporary) / "prepared"
        command = [
            sys.executable,
            str(consumer_root / "scripts/prepare_semantic_v8.py"),
            "--output",
            str(destination),
            "--producer-root",
            str(producer_root),
            "--producer-revision",
            descriptor["producerRevision"],
            "--consumer-revision",
            descriptor["consumerRevision"],
            "--tokenizer-json",
            str(paths["tokenizer"]),
            "--max-positions",
            str(model_config.max_capacity),
            "--max-history",
            str(model_config.max_history_chunks),
            "--source-recheck-timeout-seconds",
            str(timeout_seconds),
        ]
        for member in sources:
            command.extend(["--input", str(root / member["path"])])
        env = {**os.environ, "PYTHONPATH": str(consumer_root / "src")}
        completed = subprocess.run(
            command,
            cwd=consumer_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=len(sources) * (timeout_seconds + 60) + 300,
        )
        require(
            completed.returncode == 0,
            "Fresh source/native release preparation failed: " + completed.stderr[-4000:],
        )
        result = strict_json_loads(completed.stdout)
        counts = result["report"]["counts"]
        require(
            counts.get("attempted") == counts.get("admitted") == len(sources)
            and counts.get("refused", 0) == 0
            and result["report"]["refusals"] == [],
            "Fresh release preparation omitted or refused a supplied source",
        )
        return prepared_semantics(destination), result


def assert_same_state(left, right, name):
    """Compare actual loaded checkpoint/export state, including tensor type and shape."""
    import torch

    if isinstance(left, torch.Tensor):
        require(
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu()),
            "Served state differs from the checkpoint: " + name,
        )
    elif isinstance(left, dict):
        require(
            isinstance(right, dict) and left.keys() == right.keys(),
            "Served state keys differ from the checkpoint: " + name,
        )
        for key in left:
            assert_same_state(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, (list, tuple)):
        require(
            type(left) is type(right) and len(left) == len(right),
            "Served state sequence differs from the checkpoint: " + name,
        )
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_same_state(a, b, f"{name}[{index}]")
    else:
        require(
            type(left) is type(right) and left == right,
            "Served state value differs from the checkpoint: " + name,
        )


def checkpoint_consistency(config, model_config, codec, paths, sources):
    from megatoken.data import PreparedCorpus
    from megatoken.data.chunking import ChunkingConfig
    from megatoken.semantic_lineage_v8 import build_training_lineage_v8, checkpoint_lineage_v8
    from megatoken.training.checkpoint import load_checkpoint
    from megatoken.training.config import TrainConfig
    from megatoken.training.positions_v8 import (
        compare_position_inventory,
        independent_position_inventory,
        position_exposure_report,
    )
    from megatoken.training.semantic_v8 import validate_semantic_v8_target_exposure
    from megatoken.training.stream import WindowStream, source_sampling_report
    from megatoken.training.target_inventory_v8 import training_config_identity_v8
    from megatoken.training.windows import window_config

    strict_metadata = immutable_checkpoint_metadata(paths["checkpoint"])
    payload, metadata = load_checkpoint(paths["checkpoint"], map_location="cpu")
    require(same_json(metadata, strict_metadata), "Checkpoint metadata changed during loading")
    stored = TrainConfig.from_dict(payload["train_config"])
    actual_recipe, saved_recipe = asdict(config), asdict(stored)
    for name in ("codec", "datasets", "output"):
        actual_recipe.pop(name)
        saved_recipe.pop(name)
    require(
        same_json(actual_recipe, saved_recipe),
        "Release recipe changed checkpoint training settings",
    )
    require(
        same_json(payload["model_config"], model_config.to_dict())
        and payload["codec_fingerprint"] == codec.fingerprint
        and payload["packs_fingerprint"] is None
        and "diloco" not in payload
        and type(payload["global_step"]) is int
        and payload["global_step"] > 0,
        "Release checkpoint geometry, codec, route or committed update count differs",
    )
    expected_metadata = {
        "config_fingerprint": content_hash(payload["train_config"]),
        "global_step": payload["global_step"],
        "phase": config.phase,
        "phase_consumed": payload["phase_consumed"],
        "codec_fingerprint": codec.fingerprint,
        "packs_fingerprint": None,
        "routing_version": model_config.routing_version,
    }
    require(
        same_json({name: metadata.get(name) for name in expected_metadata}, expected_metadata),
        "Release checkpoint metadata differs from its actual payload",
    )
    manifest = read_json(paths["prepared"] / "manifest.json")
    fingerprint = manifest["fingerprint"]
    with PreparedCorpus(config.datasets, split="train") as corpus:
        chunking = ChunkingConfig.from_identity(corpus.chunking, codec).identity(codec)
        require(
            same_json(payload["chunking"], chunking), "Release checkpoint changed prepared chunking"
        )
        stream = WindowStream(corpus, codec, window_config(config), seed=config.seed)
        ranks = payload["rank_state"]
        require(
            isinstance(ranks, list) and len(ranks) == 1 and ranks[0].get("autoencode") is None,
            "Release profile requires one ordinary rank without auxiliary data streams",
        )
        stream.load_state_dict(ranks[0]["cursor"])
        require(
            same_json(metadata["source_sampling"], source_sampling_report([ranks[0]["cursor"]])),
            "Release checkpoint sampling receipt differs from its actual cursor",
        )
        lineage = build_training_lineage_v8(
            policy=config.semantic_v8_policy,
            tokenizer_identity=codec.fingerprint,
            prepared_manifests=[fingerprint],
            corpus_identity=corpus.identity,
            stream_identity=stream.fingerprint,
            chunking=chunking,
        )
        require(
            same_json(checkpoint_lineage_v8(payload, metadata), lineage),
            "Release checkpoint does not bind these prepared inputs and training stream",
        )
        # The runtime's fixed modality order starts with text (0). Packs are
        # refused above; the actual served loader also checks its text-only set.
        inventory = independent_position_inventory(
            corpus.iter_episodes(),
            codec=codec,
            config=config,
            model_config=model_config,
            modalities=[0],
            prepared_manifest_fingerprint=fingerprint,
            expected_source_ids=[row["sourceId"] for row in sources if row["split"] == "train"],
        )
    require(
        inventory["unmeasuredPhysicalTargets"] == 0,
        "Release profile cannot grant positional credit to unsupported target objectives",
    )
    target = validate_semantic_v8_target_exposure(
        payload.get("semantic_v8_target_exposure"),
        registry_digest=config.semantic_v8_policy["registryDigest"],
    )
    position = payload.get("semantic_v8_position_exposure")
    comparison = compare_position_inventory(
        inventory, position, target_exposure=target, global_step=payload["global_step"]
    )
    require(
        metadata.get("semantic_v8_position_digest") == position["digest"]
        and not any(
            comparison[name]
            for name in (
                "unexpectedPhysicalTargets",
                "preparedOracleMismatches",
                "targetCensusWindowOrOccurrenceMismatches",
                "targetCensusWithoutPositionReceipt",
                "unmeasuredPhysicalTargets",
            )
        ),
        "Release checkpoint positions do not match independently reconstructed prepared data",
    )
    identity = training_config_identity_v8(
        config,
        codec_fingerprint=codec.fingerprint,
        prepared_manifest_fingerprints=[fingerprint],
    )
    return (
        payload,
        metadata,
        {
            "stateSha256": metadata["state_sha256"],
            "globalStep": payload["global_step"],
            "trainingLineage": lineage,
            "recipeObjectiveDigest": stable_algorithm_hash(identity),
            "preparedPositionInventoryDigest": inventory["digest"],
            "positionSummary": position_exposure_report(position),
            "positionComparison": comparison,
            "scope": "Every recorded prepared-text target is independently checked; missing or repeated targets do not imply a complete one-pass qualification",
        },
    )


def served_consistency(paths, payload, metadata):
    import torch

    from megatoken.data.chunking import ChunkingConfig
    from megatoken.loading import load_trained
    from megatoken.served import ServedBundle

    require(
        same_json(immutable_checkpoint_metadata(paths["checkpoint"]), metadata),
        "Checkpoint metadata changed before served comparison",
    )
    raw_manifest = read_json(bound_path(paths["served"], "manifest.json"))
    raw_config = read_json(bound_path(paths["served"], "model_config.json"))
    raw_lineage = read_json(bound_path(paths["served"], "semantic-v8-lineage.json"))
    expected_config = dict(payload["model_config"])
    expected_config["checkpoint_stages"] = False
    require(
        same_json(raw_config, expected_config),
        "Served configuration differs from the exact typed checkpoint configuration",
    )
    trained = load_trained(paths["checkpoint"], device="cpu", codec_path=paths["tokenizer"])
    bundle = ServedBundle(paths["served"], device="cpu", compile=False)
    manifest = bundle.manifest
    require(
        same_json(bundle.config.to_dict(), expected_config)
        and same_json(manifest, raw_manifest)
        and same_json(bundle.semantic_v8_lineage, raw_lineage)
        and manifest["checkpoint_sha256"] == metadata["state_sha256"]
        and type(manifest["model_version"]) is int
        and manifest["model_version"] == payload["global_step"]
        and manifest["precision"] == payload["train_config"]["precision"]
        and manifest["codec_file"] == "tokenizer.json"
        and manifest["output_modalities"] == ["text"]
        and manifest["tokenizer_sha256"] == sha256_file(paths["tokenizer"])
        and bundle.semantic_v8_lineage is not None
        and same_json(bundle.semantic_v8_lineage["training"], payload["semantic_training_v8"]),
        "Served configuration, tokenizer or lineage differs from the actual checkpoint",
    )
    expected_chunking = (
        ChunkingConfig.from_identity(payload["chunking"], trained.codec)
        .with_hard_limit(bundle.config.max_capacity)
        .identity(trained.codec)
    )
    require(
        same_json(manifest["chunking"], expected_chunking), "Served chunking changed the checkpoint"
    )
    expected = trained.model.served_state_dict()
    actual = bundle.model.state_dict()
    assert_same_state(expected, actual, "model")
    expected_bank = trained.model.transducer.bank.state_snapshot()
    exported = torch.load(paths["served"] / "transducer.pt", map_location="cpu", weights_only=True)
    require(
        isinstance(exported, dict) and set(exported) == {"bank"}, "Served bank envelope differs"
    )
    assert_same_state(expected_bank, exported["bank"], "exported.transducer.bank")
    # load_snapshot intentionally invalidates cached shortlists by incrementing
    # the bank version. Export contains the exact checkpoint-loader snapshot;
    # loading that export performs exactly one further invalidation, not a
    # learned-state update. Check the prescribed increment instead of ignoring it.
    require(
        type(expected_bank["version"]) is int and expected_bank["version"] >= 1,
        "Checkpoint-loaded bank has an invalid version",
    )
    expected_loaded_bank = {**expected_bank, "version": expected_bank["version"] + 1}
    assert_same_state(
        expected_loaded_bank,
        bundle.model.transducer.bank.state_snapshot(),
        "loaded.transducer.bank",
    )
    return {
        "modelSha256": manifest["model_sha256"],
        "configSha256": manifest["config_sha256"],
        "transducerSha256": manifest["transducer_sha256"],
        "checkpointSha256": manifest["checkpoint_sha256"],
        "modelVersion": manifest["model_version"],
        "exactModelTensorCount": len(expected),
        "exactModelTensorElements": sum(value.numel() for value in expected.values()),
        "actualModelTensorsEqual": True,
        "actualTransducerBankEqual": True,
        "exactExportedBankSnapshotEqual": True,
        "additionalLoadInvalidationVersionIncrement": 1,
        "actualTokenizerBytesEqual": True,
        "liveServiceAcceptanceOrRecovery": "unmeasured",
    }


def verify_release(bundle_root, descriptor_path, producer_root, consumer_revision, timeout_seconds):
    import torch

    from megatoken.loading import load_model_config
    from megatoken.training.semantic_v8 import (
        load_semantic_v8_codec,
        preflight_semantic_training_v8,
    )

    root = checked_root(bundle_root)
    initial_inventory = bundle_inventory(root)
    descriptor_path = Path(descriptor_path)
    if descriptor_path.is_absolute():
        descriptor_path = descriptor_path.relative_to(root)
    descriptor_file = bound_path(root, descriptor_path.as_posix())
    descriptor = read_json(descriptor_file)
    paths = validate_descriptor(descriptor, root)
    require(
        consumer_revision == descriptor["consumerRevision"],
        "Requested consumer revision differs from the release descriptor",
    )
    consumer_root = Path(__file__).resolve().parents[1]
    producer_root = Path(producer_root).resolve()
    native = checked_repository(consumer_root, consumer_revision, consumer=True)
    producer = checked_repository(producer_root, descriptor["producerRevision"])
    config = release_recipe(read_json(paths["recipe"]), descriptor, root)
    model_config = load_model_config(
        config.model,
        **{
            **config.model_overrides,
            "expert_backend": config.expert_backend,
            "checkpoint_stages": config.gradient_checkpointing,
        },
    )
    codec = load_semantic_v8_codec(paths["tokenizer"], vocab_size=model_config.vocab_size)
    sources = source_membership(descriptor, root, config.semantic_v8_policy["registryDigest"])
    preflight = preflight_semantic_training_v8(
        config.datasets,
        codec=codec,
        policy=config.semantic_v8_policy,
        max_history=model_config.max_history_chunks,
        max_positions=model_config.max_capacity,
    )
    bundled = prepared_semantics(paths["prepared"])
    regenerated, fresh = regenerate_prepared(
        root,
        descriptor,
        paths,
        bundled=bundled,
        sources=sources,
        producer_root=producer_root,
        consumer_root=consumer_root,
        model_config=model_config,
        timeout_seconds=timeout_seconds,
    )
    prepared = compare_prepared_semantics(bundled, regenerated, sources)
    payload, metadata, checkpoint = checkpoint_consistency(
        config, model_config, codec, paths, sources
    )
    served = served_consistency(paths, payload, metadata)
    require(
        bundle_inventory(root) == initial_inventory, "Release inputs changed during verification"
    )
    require(
        checked_repository(consumer_root, consumer_revision, consumer=True) == native
        and checked_repository(producer_root, descriptor["producerRevision"]) == producer,
        "Release source checkouts changed during verification",
    )
    result = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "descriptorSha256": sha256_file(descriptor_file),
        "producer": producer,
        "consumer": native,
        "contractDigest": CONTRACT_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "interoperabilityDigest": INTEROPERABILITY_DIGEST,
        "inputInventoryDigest": stable_algorithm_hash(initial_inventory),
        "inputFiles": initial_inventory,
        "sourceArtifacts": sources,
        "sourceRecheckAndNativePreparation": fresh,
        "nativePreflight": preflight,
        "preparedEquivalence": prepared,
        "checkpoint": checkpoint,
        "served": served,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "optimizerUpdatesPerformed": 0,
        "decoderForwardPassesPerformed": 0,
        "qualificationJournalAuthority": "Producer snapshot verification is required separately",
        "routerInternalPositionCoverage": "unmeasured",
        "acceptedHistoryRecovery": "unmeasured",
        "learnedSuccess": None,
        "defaultPromotion": False,
        "scope": "One prepared text root, ordinary single-rank fresh-stage checkpoint, exact source/data/objective and loaded served-state equivalence only",
    }
    return {**result, "digest": stable_algorithm_hash(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--consumer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-recheck-timeout-seconds", type=int, default=30)
    args = parser.parse_args()
    require(
        not args.output.resolve().is_relative_to(args.bundle_root.resolve()),
        "Release verification output must be outside the immutable input bundle",
    )
    require(not args.output.exists(), "Release verification output already exists")
    result = verify_release(
        args.bundle_root,
        args.descriptor,
        args.producer_root,
        args.consumer_revision,
        args.source_recheck_timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "digest": result["digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
