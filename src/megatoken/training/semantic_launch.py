"""Read-only qualification of the actual inputs to a semantic training run.

This does not construct a model, optimizer, learner, lease, or training output.
Passing is an input/runtime preflight, not evidence of training memory fit,
successful recovery under load, or learned task quality.
"""

from __future__ import annotations

import copy
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from megatoken.io import content_hash, sha256_file, strict_json_loads

SCHEMA = "semantic-launch-qualification/1"
REPOSITORIES = ("tracing-js-self", "megatoken", "natlang")


def file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Expected an immutable file: {source}")
    return {"path": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size}


def unresolved_bindings(value: Any, path: str = "config") -> list[str]:
    """Report paths, never credential values, for placeholders in launch input."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in unresolved_bindings(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in unresolved_bindings(v, f"{path}[{i}]")]
    if isinstance(value, str) and (
        "UNBOUND_" in value
        or "PIN_" in value
        or "REPLACE_" in value
        or (
            value.startswith(("http://", "https://"))
            and (urlsplit(value).hostname or "").endswith(".invalid")
        )
    ):
        return [path]
    return []


def repository_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve(strict=True)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        ).stdout.strip()

    return {
        "path": str(root),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": not bool(git("status", "--porcelain", "--untracked-files=normal")),
    }


def qualify_hindsight(config, codec, model_config, corpus_identity: str) -> dict[str, Any]:
    """Read every actual train window through the same target loader as training."""
    from megatoken.data.hindsight import HindsightTargets
    from megatoken.data.hindsight_corpus import source_windows

    targets = HindsightTargets(
        config.hindsight_targets,
        config=config,
        codec=codec,
        value_dim=model_config.transducer_geometry.value_dim,
    )
    try:
        targets.validate_corpus(corpus_identity)
        encoding = file_identity(targets.root / "encoding.pt")
        expected = content_hash(
            {
                "teacher": targets.manifest["teacher"],
                "projection": encoding["sha256"],
                "rows": model_config.transducer_value_rows,
                "width": model_config.facet_dim,
            }
        )
        if targets.manifest["encoding_identity"] != expected:
            raise ValueError("Hindsight encoding file differs from its declared identity")
        windows = unsupported = 0
        for episode, window, before, _ in source_windows(config, codec, model_config):
            targets.load(episode, window.start, window.end, before, device="cpu")
            windows += 1
            unsupported += before is None
        coverage = targets.manifest["coverage"]
        if not windows or windows != coverage["text_windows"] + coverage["unsupported_windows"]:
            raise ValueError("Hindsight window coverage differs from the actual corpus")
        if unsupported != coverage["unsupported_windows"]:
            raise ValueError("Hindsight unsupported window count differs from the actual corpus")
        return {
            "manifest": file_identity(targets.root / "manifest.json"),
            "index": file_identity(targets.root / "targets.sqlite"),
            "encoding": encoding,
            "windowsRead": windows,
            "unsupportedWindows": unsupported,
            "scope": "Every training window and each referenced target tensor; no teacher inference",
        }
    finally:
        targets.close()


def _reject_nonfinite_tensors(value: Any, *, chunk_elements: int = 1_000_000) -> None:
    """Scan serialized tensors without creating a checkpoint-sized temporary."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or not value.is_contiguous():
            raise ValueError("Checkpoint contains a noncontiguous or non-strided tensor")
        if value.is_floating_point() or value.is_complex():
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), chunk_elements):
                if not bool(torch.isfinite(flat[start : start + chunk_elements]).all()):
                    raise ValueError("Checkpoint contains nonfinite tensor values")
    elif isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite_tensors(child, chunk_elements=chunk_elements)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_nonfinite_tensors(child, chunk_elements=chunk_elements)


def _validate_model_and_masters(
    payload, model_config, codec, train_config, *, coordinator_bootstrap
):
    """Validate actual checkpoint tensors against a shape-only real model."""
    import torch

    from megatoken.loading import build_model
    from megatoken.training.diloco_protocol import parameter_identity

    _reject_nonfinite_tensors(payload)
    with torch.device("meta"):
        model = build_model(model_config, codec, device="meta")
        shape_state = {
            name: torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="meta")
            for name, tensor in payload["model"].items()
            if isinstance(name, str) and isinstance(tensor, torch.Tensor)
        }
    if len(shape_state) != len(payload["model"]):
        raise ValueError("Checkpoint model state must map names to tensors")
    # The frontend load hooks construct checkpoint-owned donor imports before strict
    # matching, exactly as normal restoration does, while all tensors remain meta.
    try:
        model.load_state_dict(shape_state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Checkpoint model keys or tensor shapes are invalid: {error}") from error

    source_expected = {name: parameter for name, parameter in model.named_parameters()}
    optimizer = payload.get("optimizer") or {}
    masters = (
        payload.get("master_weights")
        if payload.get("export_only")
        else optimizer.get("master_weights")
    )
    if not isinstance(masters, dict) or not masters:
        raise ValueError("Checkpoint requires actual FP32 master weights")
    source_owners = {name for name, parameter in source_expected.items() if parameter.requires_grad}
    if set(masters) != source_owners:
        raise ValueError(
            "Checkpoint master weights do not exactly match source model optimizer owners"
        )
    for name, tensor in masters.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("Checkpoint master weights must be FP32 tensors")
        if tuple(tensor.shape) != tuple(source_expected[name].shape):
            raise ValueError(f"Checkpoint master weight shape differs for {name}")
    destination_owners = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    # A phase transition may add configured donor owners after source restoration.
    # Derive their fixed projection names/shapes without allocating donor/model weights.
    requested_donors = (
        [train_config.engram_import]
        if train_config.engram_import
        else list(train_config.engram_imports or [])
    )
    table = model.core.frontend.ngrams
    restored_donors = (1 if table.import_map is not None else 0) + len(table.import_branches)
    if restored_donors > len(requested_donors):
        raise ValueError("Checkpoint carries unrequested donor imports; bake them explicitly first")
    if not coordinator_bootstrap and restored_donors != len(requested_donors):
        raise ValueError("Exact worker resume cannot add donor owners; use coordinator bootstrap")
    if coordinator_bootstrap:
        next_index = restored_donors
        has_root_import = table.import_map is not None
        for _ in requested_donors[restored_donors:]:
            if not has_root_import:
                owner = "core.frontend.ngrams.import_map.weight"
                has_root_import = True
            else:
                branch_index = next_index - 1
                owner = f"core.frontend.ngrams.import_branches.{branch_index}.import_map.weight"
            destination_owners[owner] = torch.empty(
                (model_config.ngram_dim, 128), dtype=torch.float32, device="meta"
            )
            next_index += 1
    layout = {
        name: torch.empty(tuple(parameter.shape), dtype=torch.float32, device="meta")
        for name, parameter in destination_owners.items()
    }
    return parameter_identity(layout)


def qualify_checkpoint(path, *, config, codec, cohort_digest, coordinator_bootstrap=False):
    """Validate a committed directory without invoking pointer repair or restoration."""
    import torch

    from megatoken.loading import load_model_config
    from megatoken.model import ModelConfig
    from megatoken.training.checkpoint import FORMAT_VERSION
    from megatoken.training.engine import _resume_compatible
    from megatoken.training.semantic_budget import restore_semantic_training_state

    directory = Path(path).resolve(strict=True)
    if not directory.is_dir() or not (directory / "state.pt").is_file():
        raise ValueError(
            "Pass the exact committed checkpoint directory, not latest.json or its root"
        )
    if directory.name.startswith(".") or any(directory.parent.glob(".superseded-*")):
        raise ValueError("Checkpoint publication requires recovery before read-only qualification")
    metadata = strict_json_loads((directory / "metadata.json").read_bytes())
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Checkpoint must use current format {FORMAT_VERSION}")
    state = file_identity(directory / "state.pt")
    if state["sha256"] != metadata["state_sha256"]:
        raise ValueError("Checkpoint state checksum mismatch")
    bank = metadata["transducer_checkpoint"]
    if not isinstance(bank["file"], str) or Path(bank["file"]).name != bank["file"]:
        raise ValueError("Checkpoint bank must be an adjacent file")
    bank_file = file_identity(directory / bank["file"])
    if bank_file["sha256"] != bank["sha256"]:
        raise ValueError("Checkpoint bank checksum mismatch")
    payload = torch.load(directory / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    required = {
        "model",
        "optimizer",
        "model_config",
        "train_config",
        "codec_fingerprint",
        "rank_state",
        "global_step",
        "phase_consumed",
        "counters",
        "chunking",
    }
    if coordinator_bootstrap:
        required = {"model", "model_config", "train_config", "codec_fingerprint", "global_step"}
    if payload.get("format_version") != FORMAT_VERSION or not required <= payload.keys():
        raise ValueError("Checkpoint is not a complete current training checkpoint")
    if not payload["model"] or (not coordinator_bootstrap and not payload["optimizer"]):
        raise ValueError("Checkpoint has no model or optimizer state")
    if payload["codec_fingerprint"] != codec.fingerprint:
        raise ValueError("Checkpoint tokenizer differs from the launch tokenizer")
    saved_model = ModelConfig.from_dict(payload["model_config"])
    explicit = {
        **config.model_overrides,
        "expert_backend": config.expert_backend,
        "checkpoint_stages": config.gradient_checkpointing,
    }
    requested_model = load_model_config(config.model, **explicit).to_dict()
    if any(saved_model.to_dict().get(key) != requested_model[key] for key in explicit):
        raise ValueError("Checkpoint model geometry differs from explicit launch overrides")
    if saved_model.vocab_size != codec.vocab_size:
        raise ValueError("Checkpoint model vocabulary differs from the tokenizer")
    if coordinator_bootstrap:
        if saved_model.to_dict() != requested_model:
            raise ValueError("Coordinator bootstrap checkpoint architecture changed")
    else:
        _resume_compatible(copy.deepcopy(payload["train_config"]), config, False)
    if config.packs:
        raise ValueError("Shape-only launch validation does not support configured realizer packs")
    parameter_identity = _validate_model_and_masters(
        payload,
        saved_model,
        codec,
        config,
        coordinator_bootstrap=coordinator_bootstrap,
    )
    for name in ("global_step",) if coordinator_bootstrap else ("global_step", "phase_consumed"):
        if type(payload[name]) is not int or payload[name] < 0:
            raise ValueError(f"Checkpoint has invalid {name}")
    if not coordinator_bootstrap and (
        len(payload["rank_state"]) != 1
        or not {"cursor", "rng", "autoencode"} <= payload["rank_state"][0].keys()
    ):
        raise ValueError("Semantic checkpoint must retain one rank's cursor and RNG state")
    if config.decoupled is not None and not coordinator_bootstrap and not payload.get("diloco"):
        raise ValueError("Checkpoint is missing coordinator worker recovery state")
    if not coordinator_bootstrap:
        restore_semantic_training_state(payload, cohort_digest=cohort_digest)
    return {
        "metadata": file_identity(directory / "metadata.json"),
        "state": state,
        "bank": bank_file,
        "phase": payload["train_config"]["phase"],
        "globalStep": payload["global_step"],
        "phaseConsumed": payload.get("phase_consumed"),
        "parameterIdentity": parameter_identity,
        "bootstrapStateSha256": state["sha256"] if coordinator_bootstrap else None,
        "scope": (
            "Checksums, strict model tensors, finite values and exact FP32 master layout; no state installed"
            if coordinator_bootstrap
            else "Checksums, required restart fields and configuration compatibility; no state installed"
        ),
    }


def qualify_coordinator(
    config,
    model_config,
    codec,
    corpus_identity,
    *,
    expected_parameter_identity: str,
    bootstrap_state_sha256: str | None = None,
    bootstrap_metadata_sha256: str | None = None,
    bootstrap_receipt: dict[str, Any] | None = None,
):
    """Only authenticated GET requests: never take a lease or register a learner."""
    from megatoken.training.diloco_client import Transport
    from megatoken.training.diloco_protocol import Fragment, training_identity

    settings = config.decoupled
    if settings is None:
        raise ValueError("Production semantic launch requires its coordinator")
    if urlsplit(settings.coordinator_url).query or urlsplit(settings.coordinator_url).fragment:
        raise ValueError(
            "Coordinator credentials belong in environment variables, not URL parameters"
        )
    transport = Transport(
        settings.coordinator_url,
        os.environ.get("MEGATOKEN_DILOCO_TOKEN", ""),
        settings.timeout_seconds,
        allow_http=settings.allow_private_http,
    )
    manifest = transport.json("GET", "/v1/manifest")
    status = transport.json("GET", "/v1/status")
    expected = {
        "run_id": settings.run_id,
        "workers": settings.workers,
        "model_config": model_config.to_dict(),
        "codec_fingerprint": codec.fingerprint,
        "token_budget": config.schedule.budget,
        "train_identity": training_identity(config),
        "dataset_fingerprint": corpus_identity,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Coordinator differs from launch input: {key}")
    if manifest.get("parameter_identity") != expected_parameter_identity:
        raise ValueError(
            "Coordinator parameter identity differs from the verified checkpoint layout"
        )
    bootstrap_provenance = None
    bootstrap_artifact = None
    if bootstrap_state_sha256 is not None:
        receipt = bootstrap_receipt
        if not isinstance(receipt, dict):
            raise ValueError(
                "New coordinator bootstrap requires its initialization provenance receipt"
            )
        if (
            receipt.get("schema") != "diloco-bootstrap-provenance/1"
            or receipt.get("source_checkpoint_state_sha256") != bootstrap_state_sha256
            or receipt.get("source_checkpoint_metadata_sha256") != bootstrap_metadata_sha256
            or receipt.get("parameter_identity") != expected_parameter_identity
            or receipt.get("coordinator_manifest_fingerprint") != manifest.get("fingerprint")
            or receipt.get("run_id") != settings.run_id
            or receipt.get("source_kind") != "checkpoint"
        ):
            raise ValueError(
                "Coordinator bootstrap provenance does not match the supplied checkpoint"
            )
        with tempfile.TemporaryDirectory(prefix="semantic-bootstrap-probe-") as temporary:
            downloaded, _ = transport.fetch_file("/v1/bootstrap", Path(temporary))
            if downloaded is None:
                raise ValueError("Coordinator returned no bootstrap artifact")
            live_bootstrap_sha256 = sha256_file(downloaded)
            if live_bootstrap_sha256 != receipt.get("coordinator_bootstrap_sha256"):
                raise ValueError(
                    "Live coordinator bootstrap hash differs from its initialization receipt"
                )
            bootstrap_artifact = {
                "sha256": live_bootstrap_sha256,
                "bytes": downloaded.stat().st_size,
                "scope": "Read-only /v1/bootstrap download; temporary file removed after hashing",
            }
        bootstrap_provenance = receipt
    fragments = [Fragment.from_dict(item) for item in manifest["fragments"]]
    if not fragments or [item.index for item in fragments] != list(range(len(fragments))):
        raise ValueError("Coordinator fragment topology is incomplete")
    if any(item.length <= 0 or not item.slices for item in fragments):
        raise ValueError("Coordinator contains an empty fragment")
    if status.get("run_id") != settings.run_id or status.get("finished") is not False:
        raise ValueError("Coordinator status is for a different or already finished run")
    if len(status.get("versions", [])) != len(fragments):
        raise ValueError("Coordinator published versions differ from fragment topology")
    return {
        "runId": settings.run_id,
        "manifestDigest": content_hash(manifest),
        "parameterIdentity": manifest["parameter_identity"],
        "bootstrapProvenance": bootstrap_provenance,
        "bootstrapArtifact": bootstrap_artifact,
        "fragments": len(fragments),
        "workerId": settings.resolved_worker_id,
        "status": status,
        "scope": "Read-only objective, corpus, architecture, exact master layout, bootstrap provenance and liveness checks",
    }


def qualify_launch(
    config_path: str | Path,
    *,
    repositories: dict[str, str | Path],
    resume: str | Path | None = None,
    bootstrap_checkpoint: str | Path | None = None,
    bootstrap_receipt: str | Path | None = None,
    probe_coordinator: bool = False,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Return every independent failure as a gate, without starting training."""
    from megatoken.data.recipes import validate_stage_start, validate_training_donors
    from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
    from megatoken.loading import load_model_config
    from megatoken.runtime import doctor
    from megatoken.training.config import TrainConfig
    from megatoken.training.semantic_budget import SemanticExperimentConfig, validate_stage_supply

    gates: dict[str, dict] = {}

    def check(name, action, dependencies=()):
        failed = [key for key in dependencies if gates[key]["status"] != "passed"]
        if failed:
            gates[name] = {
                "status": "blocked",
                "reason": "Prerequisites failed: " + ", ".join(failed),
            }
            return None
        try:
            value = action()
            gates[name] = {"status": "passed", "evidence": value}
            return value
        except Exception as error:
            gates[name] = {
                "status": "blocked",
                "errorType": type(error).__name__,
                "reason": str(error),
            }
            return None

    raw = check(
        "configurationFile",
        lambda: {
            "identity": file_identity(config_path),
            "value": strict_json_loads(Path(config_path).read_bytes()),
        },
    )
    value = None if raw is None else raw["value"]
    # The manifest records a hash of the configuration, not an uncontrolled copy
    # of potential credentials. TrainConfig rejects unknown secret fields.
    if raw is not None:
        gates["configurationFile"]["evidence"] = raw["identity"]

    def configuration():
        result = TrainConfig.from_dict(value)
        if result.purpose != "production" or result.semantic_experiment is None:
            raise ValueError("Launch qualification requires a production semantic experiment")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Semantic comparison requires one independent learner process")
        return result

    config = check("configuration", configuration, ("configurationFile",))

    def bindings():
        missing = unresolved_bindings(value)
        if missing:
            raise ValueError("Unresolved launch bindings: " + ", ".join(missing))
        return {"unresolved": []}

    check("resolvedBindings", bindings, ("configurationFile",))
    if config is not None:
        gates["configuration"]["evidence"] = {
            "resolvedConfigDigest": content_hash(asdict(config)),
            "phase": config.phase,
            "seed": config.seed,
            "output": str(Path(config.output).resolve()),
            "workingDirectory": str(Path.cwd().resolve()),
            "inputBindings": {
                "datasets": [str(Path(path).resolve()) for path in config.datasets],
                "codec": str(Path(config.codec).resolve()),
                "hindsightTargets": str(Path(config.hindsight_targets).resolve()),
                "donors": [str(Path(path).resolve()) for path in config.engram_imports or []],
            },
        }

    def code_revisions():
        if set(repositories) != set(REPOSITORIES):
            raise ValueError("Provide all three repository checkouts")
        identities = {name: repository_identity(repositories[name]) for name in REPOSITORIES}
        if any(not item["clean"] for item in identities.values()):
            raise ValueError(
                "Qualification requires committed, clean code in all three repositories"
            )
        return identities

    check("codeRevisions", code_revisions)
    experiment = None if config is None else SemanticExperimentConfig(**config.semantic_experiment)

    def engineering():
        digest = experiment.preflight(phase=config.phase)
        receipt = strict_json_loads(Path(experiment.engineering_receipt).read_bytes())
        references = receipt.get("evidence")
        if not isinstance(references, list) or not references:
            raise ValueError("Engineering receipt has no pinned verification evidence")
        verified = []
        for reference in references:
            identity = file_identity(
                Path(experiment.engineering_receipt).parent / reference["path"]
            )
            if identity["sha256"] != reference["sha256"]:
                raise ValueError("Engineering verification evidence checksum changed")
            verified.append(identity)
        return {
            "cohortDigest": digest,
            "cohort": file_identity(experiment.cohort),
            "receipt": file_identity(experiment.engineering_receipt),
            "artifactProducerRevisions": receipt["repositories"],
            "verifiedEvidence": verified,
        }

    evidence = check("engineering", engineering, ("configuration",))
    model_config = check(
        "modelGeometry",
        lambda: load_model_config(
            config.model,
            **{
                **config.model_overrides,
                "expert_backend": config.expert_backend,
                "checkpoint_stages": config.gradient_checkpointing,
            },
        ),
        ("configuration",),
    )
    if model_config is not None:
        gates["modelGeometry"]["evidence"] = model_config.to_dict()

    def tokenizer():
        source = Path(config.codec)
        if source.is_dir():
            metadata = strict_json_loads((source / "codec.json").read_bytes())
            candidate = source / metadata["tokenizer_file"]
            if candidate.parent.resolve() != source.resolve():
                raise ValueError("Tokenizer escaped its asset directory")
            source = candidate
        return ByteCodec.from_tokenizer_json(
            source,
            expected_sha256=LING_TOKENIZER_SHA256,
            expected_vocab_size=model_config.vocab_size,
        )

    codec = check("tokenizer", tokenizer, ("configuration", "modelGeometry"))
    if codec is not None:
        gates["tokenizer"]["evidence"] = {
            "fingerprint": codec.fingerprint,
            "sha256": LING_TOKENIZER_SHA256,
            "vocabularySize": codec.vocab_size,
        }
    supply = check(
        "corpus",
        lambda: validate_stage_supply(
            config.datasets,
            phase=config.phase,
            representation=experiment.representation,
            codec=codec,
            producer_revision=evidence["artifactProducerRevisions"]["tracing-js-self"],
            consumer_revision=evidence["artifactProducerRevisions"]["megatoken"],
            max_history=model_config.max_history_chunks,
            max_positions=model_config.max_capacity,
        ),
        ("engineering", "tokenizer", "modelGeometry"),
    )

    def donors():
        from megatoken.assets import load_engram_asset

        validate_training_donors(config)
        for root in config.engram_imports or []:
            load_engram_asset(model_config, root)
        return [
            {
                "metadata": file_identity(Path(root) / "engram.json"),
                "descriptors": file_identity(Path(root) / "descriptors.npz"),
            }
            for root in config.engram_imports or []
        ]

    check("donors", donors, ("configuration", "modelGeometry"))
    check(
        "hindsight",
        lambda: qualify_hindsight(config, codec, model_config, supply["corpusIdentity"]),
        ("configuration", "tokenizer", "modelGeometry", "corpus"),
    )

    def checkpoint():
        if resume is not None and bootstrap_checkpoint is not None:
            raise ValueError("Choose worker resume or new-coordinator bootstrap, not both")
        selected = resume if resume is not None else bootstrap_checkpoint
        if selected is None:
            validate_stage_start(config, None)
            if config.decoupled is not None:
                raise ValueError(
                    "Coordinator launch requires an exact worker-resume checkpoint or new-coordinator bootstrap checkpoint"
                )
            return {"mode": "fresh", "checkpoint": None}
        result = qualify_checkpoint(
            selected,
            config=config,
            codec=codec,
            cohort_digest=evidence["cohortDigest"],
            coordinator_bootstrap=bootstrap_checkpoint is not None,
        )
        validate_stage_start(config, result["phase"])
        return {
            "mode": "coordinator-bootstrap" if bootstrap_checkpoint is not None else "resume",
            **result,
        }

    check("checkpoint", checkpoint, ("engineering", "tokenizer"))

    def read_bootstrap_receipt():
        if bootstrap_checkpoint is None:
            if bootstrap_receipt is not None:
                raise ValueError(
                    "Bootstrap provenance receipt applies only to a new coordinator bootstrap"
                )
            return None
        if bootstrap_receipt is None:
            raise ValueError(
                "Supply the semantic-bootstrap-provenance.json emitted by coordinator initialization"
            )
        receipt_path = Path(bootstrap_receipt).resolve(strict=True)
        if not receipt_path.is_file():
            raise ValueError("Bootstrap provenance receipt must be a file")
        return strict_json_loads(receipt_path.read_bytes())

    check("bootstrapProvenance", read_bootstrap_receipt, ("checkpoint",))

    def output():
        target = Path(config.output).resolve()
        existing = target
        while not existing.exists():
            existing = existing.parent
        if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
            raise ValueError("Training output has no writable parent directory")
        for root in [*config.datasets, *config.engram_imports, config.hindsight_targets]:
            asset = Path(root).resolve()
            if target == asset or asset in target.parents or target in asset.parents:
                raise ValueError("Training output overlaps immutable input assets")
        return {"path": str(target), "existingParent": str(existing), "created": False}

    check("outputLocation", output, ("configuration",))

    def hardware():
        result = doctor(
            require_cuda=require_cuda, require_blackwell=config.require_blackwell, probe_cuda=True
        )
        if not result["ok"]:
            raise ValueError("; ".join(result["errors"]))
        if config.device.startswith("cuda") and not result["devices"]:
            raise ValueError("Configured CUDA device is unavailable")
        return result

    check("hardware", hardware, ("configuration",))
    if probe_coordinator:
        check(
            "coordinator",
            lambda: qualify_coordinator(
                config,
                model_config,
                codec,
                supply["corpusIdentity"],
                expected_parameter_identity=evidence["checkpoint"]["parameterIdentity"],
                bootstrap_state_sha256=(
                    evidence["checkpoint"]["bootstrapStateSha256"]
                    if bootstrap_checkpoint is not None
                    else None
                ),
                bootstrap_metadata_sha256=(
                    evidence["checkpoint"]["metadata"]["sha256"]
                    if bootstrap_checkpoint is not None
                    else None
                ),
                bootstrap_receipt=(
                    evidence["bootstrapProvenance"] if bootstrap_checkpoint is not None else None
                ),
            ),
            (
                "configuration",
                "resolvedBindings",
                "modelGeometry",
                "tokenizer",
                "corpus",
                "donors",
                "hindsight",
                "checkpoint",
                "bootstrapProvenance",
            ),
        )
    else:
        gates["coordinator"] = {
            "status": "blocked",
            "reason": "Authenticated read-only probe not requested",
        }
    report = {
        "schema": SCHEMA,
        "preflightReady": all(g["status"] == "passed" for g in gates.values()),
        "gates": gates,
        "optimizerSteps": 0,
        "learnerStarted": False,
        "learnedSuccess": None,
        "trainingMemoryFit": "unmeasured",
        "recoveryUnderLoad": "unmeasured",
        "coordinatorMutations": 0,
        "scope": "Input integrity and runtime preflight; not a training, throughput or recovery result",
    }
    return {**report, "digest": content_hash(report)}
