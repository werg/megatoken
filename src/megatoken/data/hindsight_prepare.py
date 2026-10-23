"""Prepare teacher targets for the unchanged selected training corpus."""

from __future__ import annotations

from pathlib import Path

import torch

from megatoken.io import content_hash, sha256_file, write_json

from .hindsight import pool_bytes


class LocalWriter:
    """Load only the retrieved writer backbone, with no decoder or network dependency."""

    def __init__(self, directory, *, device="cpu", horizon=8192):
        import transformers
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        directory = Path(directory)
        if (directory / "model").is_dir():
            directory = directory / "model"
        weights = directory / "training.safetensors"
        self.identity = content_hash(
            {
                "weights": sha256_file(weights),
                "config": sha256_file(directory / "encoder_config" / "config.json"),
                "tokenizer": {
                    p.name: sha256_file(p)
                    for p in sorted((directory / "encoder_tokenizer").iterdir())
                    if p.is_file()
                },
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "horizon": horizon,
                "renderer": "document-newlines-v1",
                "pool": "utf8-start-v1",
            }
        )
        config = AutoConfig.from_pretrained(directory / "encoder_config", local_files_only=True)
        if getattr(config, "is_causal", True) is not False:
            raise ValueError("The teacher writer must retain its noncausal configuration")
        with safe_open(weights, framework="pt", device="cpu") as archive:
            keys = list(archive.keys())
            candidates = sorted({k.split("backbone.")[0] for k in keys if "backbone." in k})
            if len(candidates) != 1:
                raise ValueError("Expected exactly one saved teacher writer backbone")
            prefix = candidates[0] + "backbone."
            state = {k[len(prefix) :]: archive.get_tensor(k) for k in keys if k.startswith(prefix)}
        from transformers.initialization import no_init_weights

        # Keep real rotary buffers; meta construction leaves nonpersistent buffers empty.
        with no_init_weights():
            self.model = AutoModel.from_config(config, dtype=next(iter(state.values())).dtype)
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.to(device=device).eval().requires_grad_(False)
        for module in self.model.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory / "encoder_tokenizer", local_files_only=True, use_fast=True
        )
        self.horizon, self.device = horizon, device

    @torch.inference_mode()
    def encode(self, text, *, rows=64):
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.horizon,
            add_special_tokens=True,
        )
        offsets = encoded.pop("offset_mapping")
        features = {
            k: torch.tensor([v], device=self.device)
            for k, v in encoded.items()
            if k in {"input_ids", "token_type_ids"}
        }
        features["attention_mask"] = {"full_attention": None, "linear_attention": None}
        hidden = self.model(**features, use_cache=False).last_hidden_state[0]
        pooled, mask, extent = pool_bytes(text, offsets, hidden, rows)
        return pooled.cpu(), mask.cpu(), extent


def writer_inputs(directory):
    """Verified cloud inputs for the retrieved backbone; no runtime download on learners."""
    directory = Path(directory)
    if (directory / "model").is_dir():
        directory = directory / "model"
    files = [directory / "training.safetensors"]
    for name in ("encoder_config", "encoder_tokenizer"):
        files.extend(sorted((directory / name).rglob("*")))
    inputs = []
    for path in files:
        if path.is_file():
            if path.is_symlink():
                raise ValueError("Writer inputs must be regular pinned files")
            inputs.append(
                {
                    "path": "assets/hindsight-writer/" + path.relative_to(directory).as_posix(),
                    "source": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
    if not any(item["path"].endswith("training.safetensors") for item in inputs):
        raise ValueError("Retrieved writer checkpoint is missing")
    return {"inputs": inputs}


def run(args):
    from megatoken.training.config import TrainConfig

    if min(args.horizon, args.fit_windows) < 1:
        raise ValueError("Preparation limits must be positive")
    if args.cloud_inputs:
        write_json(args.cloud_inputs, writer_inputs(args.writer))
        return {"writer_inputs": args.cloud_inputs}
    if not args.train_config or not args.output:
        raise ValueError("Preparation requires --train-config and --output")
    config = TrainConfig.load(args.train_config)
    from megatoken.data.recipes import validate_teacher_encoding_source

    validate_teacher_encoding_source(config, args.encoding_from)
    from .hindsight_corpus import prepare_corpus

    writer = LocalWriter(args.writer, device=args.device, horizon=args.horizon)
    manifest = prepare_corpus(
        config,
        args.output,
        writer,
        fit_windows=args.fit_windows,
        encoding_from=args.encoding_from,
    )
    return {
        "windows": manifest["coverage"]["text_windows"],
        "train_config": str(Path(args.output) / "train.json"),
    }
