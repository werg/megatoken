"""Served-only exports and the loader of a served bundle."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from megatoken.controls import CONTROL_SCHEMA_VERSION
from megatoken.data.chunking import ChunkingConfig
from megatoken.io import fsync_directory, sha256_file, write_json
from megatoken.lifecycle import LIFECYCLE_VERSION
from megatoken.loading import build_model, load_codec, load_trained, parameter_report
from megatoken.model.config import ARCHITECTURE_VERSION, ModelConfig
from megatoken.model.multitrack import GROUP_TAPE_VERSION


def export_served(
    checkpoint: str | Path,
    output: str | Path,
    *,
    codec_path: str = "byte",
    packs: str | Path | None = None,
) -> dict[str, Any]:
    """Served bundle of a checkpoint; realizer packs stay external, bound by fingerprint."""
    trained = load_trained(checkpoint, codec_path=codec_path, packs=packs)
    model, codec, payload, metadata = (
        trained.model,
        trained.codec,
        trained.payload,
        trained.metadata,
    )
    config = ModelConfig.from_dict(payload["model_config"])
    chunking = ChunkingConfig.from_identity(payload["chunking"], codec)
    chunking = chunking.with_hard_limit(config.max_capacity)
    precision = payload["train_config"]["precision"]
    state = model.served_state_dict()
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("A served bundle is immutable; choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".served-", dir=destination.parent))
    try:
        # Contiguous clones avoid exporting transposed/tied aliases as separate
        # parameter owners. The served constructor restores declared sharing.
        save_file(
            {name: value.detach().cpu().contiguous().clone() for name, value in state.items()},
            str(temporary / "model.safetensors"),
        )
        served_config = config.to_dict()
        served_config["checkpoint_stages"] = False
        write_json(temporary / "model_config.json", served_config)
        bundled_codec = "byte"
        if codec_path != "byte":
            source = Path(codec_path)
            if source.is_dir():
                source_metadata = json.loads((source / "codec.json").read_text())
                source = source / source_metadata["tokenizer_file"]
            shutil.copyfile(source, temporary / "tokenizer.json")
            bundled_codec = "tokenizer.json"
        # The sparse bank is not a parameter: it ships beside the weights.
        torch.save({"bank": model.transducer.bank.state_snapshot()}, temporary / "transducer.pt")
        durable_files = [temporary / "model.safetensors", temporary / "transducer.pt"]
        if bundled_codec != "byte":
            durable_files.append(temporary / bundled_codec)
        for artifact in durable_files:
            with artifact.open("rb") as handle:
                os.fsync(handle.fileno())
        manifest = {
            "format_version": ARCHITECTURE_VERSION,
            "control_schema_version": CONTROL_SCHEMA_VERSION,
            "model_sha256": sha256_file(temporary / "model.safetensors"),
            "transducer_sha256": sha256_file(temporary / "transducer.pt"),
            "config_sha256": sha256_file(temporary / "model_config.json"),
            "codec_file": bundled_codec,
            "codec_fingerprint": codec.fingerprint,
            "chunking": chunking.identity(codec),
            "policy_likelihood": "modality_categorical_plus_canonical_facet_gaussian_density",
            "precision": precision,
            "model_version": payload["global_step"],
            "routing_version": config.routing_version,
            "tape_version": GROUP_TAPE_VERSION,
            "lifecycle_version": LIFECYCLE_VERSION,
            "output_modalities": sorted(
                ["text", *(p.modality for p in model.media_realizers.packs.values())]
            ),
            "event_addressing": "independent_peer_events",
            "checkpoint_sha256": metadata["state_sha256"],
            "parameters": parameter_report(model)["totals"]["served"],
            "training_heads": False,
            "retrospective_writes": True,
            "packs_fingerprint": payload["packs_fingerprint"],
        }
        if bundled_codec != "byte":
            manifest["tokenizer_sha256"] = sha256_file(temporary / bundled_codec)
        write_json(temporary / "manifest.json", manifest)
        fsync_directory(temporary)
        os.rename(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"bundle": str(destination), **manifest}


class ServedBundle:
    """A verified served bundle, loaded onto a device: the one place a bundle is read.

    Serving itself is the multitrack runtime (``MultiTrackRuntime``): this holds what it
    needs, the model with its codec, configuration, chunk policy and manifest.
    """

    def __init__(
        self,
        bundle: str | Path,
        *,
        packs: str | Path | None = None,
        device: str = "auto",
        compile: bool = False,
    ) -> None:
        """``packs`` is the locked realizer manifest the bundle was trained with."""
        directory = Path(bundle)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest.get("format_version") != ARCHITECTURE_VERSION
            or manifest.get("training_heads") is not False
        ):
            raise ValueError(
                f"Unsupported or training-only bundle: serving requires architecture "
                f"v{ARCHITECTURE_VERSION} weights"
            )
        if (None if packs is None else sha256_file(packs)) != manifest["packs_fingerprint"]:
            raise ValueError("Supply the realizer pack manifest the bundle was trained with")
        if (
            manifest.get("policy_likelihood")
            != "modality_categorical_plus_canonical_facet_gaussian_density"
        ):
            raise ValueError("Unsupported served action likelihood")
        if (
            manifest.get("lifecycle_version") != LIFECYCLE_VERSION
            or manifest.get("event_addressing") != "independent_peer_events"
        ):
            raise ValueError("Served bundle has incompatible lifecycle or event addressing")
        if manifest.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
            raise ValueError("Served bundle requires the current public request contract")
        if manifest.get("tape_version") != GROUP_TAPE_VERSION:
            raise ValueError("Served tape protocol differs from the current control-aware policy")
        if manifest.get("precision") not in {"fp32", "bf16"}:
            raise ValueError("Unsupported served precision")
        if sha256_file(directory / "model.safetensors") != manifest["model_sha256"]:
            raise ValueError("Served weights checksum mismatch")
        if sha256_file(directory / "model_config.json") != manifest["config_sha256"]:
            raise ValueError("Served configuration checksum mismatch")
        if sha256_file(directory / "transducer.pt") != manifest["transducer_sha256"]:
            raise ValueError("Served transducer bank checksum mismatch")
        self.config = ModelConfig.from_dict(
            json.loads((directory / "model_config.json").read_text())
        )
        if manifest.get("routing_version") != self.config.routing_version:
            raise ValueError("Served routing version differs from the model configuration")
        codec_name = manifest["codec_file"]
        if codec_name != "byte":
            if Path(codec_name).name != codec_name:
                raise ValueError("Unsafe served codec path")
            if sha256_file(directory / codec_name) != manifest["tokenizer_sha256"]:
                raise ValueError("Served tokenizer checksum mismatch")
        self.codec = load_codec(
            "byte" if codec_name == "byte" else directory / codec_name,
            vocab_size=self.config.vocab_size,
        )
        if self.codec.fingerprint != manifest["codec_fingerprint"]:
            raise ValueError("Served codec mapping differs from the training codec")
        self.chunking = ChunkingConfig.from_identity(manifest["chunking"], self.codec)
        if self.chunking.hard_max_tokens > self.config.max_capacity:
            raise ValueError("Served chunk policy exceeds the configured model capacity")
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.model = build_model(
            self.config,
            self.codec,
            device=self.device,
            precision=manifest["precision"],
            training_heads=False,
            packs=packs,
        )
        supported = sorted(
            ["text", *(p.modality for p in self.model.media_realizers.packs.values())]
        )
        if manifest.get("output_modalities") != supported:
            raise ValueError("Served output capabilities differ from installed decoders")
        self.model.load_state_dict(
            load_file(str(directory / "model.safetensors"), device=str(self.device)), strict=True
        )
        with (directory / "transducer.pt").open("rb") as handle:
            bank = torch.load(handle, map_location="cpu", weights_only=True)["bank"]
        self.model.transducer.bank.load_snapshot(bank)
        if parameter_report(self.model)["totals"]["served"] != manifest["parameters"]:
            raise ValueError("Served parameter ownership count differs from the bundle")
        self.model.core.set_model_version(manifest["model_version"])
        self.model.eval()
        if compile:
            self.model.core.compile_kernels()
        self.manifest = manifest
