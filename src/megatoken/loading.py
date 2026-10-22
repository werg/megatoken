"""One model/codec construction path used by training, evaluation, and serving."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from megatoken.generation.codec import ByteCodec
from megatoken.io import read_mapping, sha256_file

if TYPE_CHECKING:
    import torch

    from megatoken.model.config import ModelConfig


def load_codec(path: str | Path, *, vocab_size: int | None = None) -> ByteCodec:
    if str(path) == "byte":
        return ByteCodec.byte_fallback()
    source = Path(path)
    if source.is_dir():
        metadata = json.loads((source / "codec.json").read_text())
        tokenizer = source / metadata["tokenizer_file"]
        if tokenizer.parent.resolve() != source.resolve():
            raise ValueError("Tokenizer path must remain within its codec asset directory")
        return ByteCodec.from_tokenizer_json(
            tokenizer,
            expected_sha256=metadata["tokenizer_sha256"],
            expected_vocab_size=vocab_size or metadata["vocab_size"],
        )
    return ByteCodec.from_tokenizer_json(
        source, expected_sha256=sha256_file(source), expected_vocab_size=vocab_size or 157_184
    )


def load_model_config(name_or_path: str, **overrides: Any) -> ModelConfig:
    from megatoken.model.config import ModelConfig

    if name_or_path == "tiny":
        return ModelConfig.tiny(**overrides)
    if name_or_path == "32x256":
        return ModelConfig.compact32(**overrides)
    if name_or_path == "32x512":
        return ModelConfig(**overrides)
    return ModelConfig.from_dict({**read_mapping(name_or_path), **overrides})


def build_model(
    config: ModelConfig,
    codec: ByteCodec,
    *,
    device: torch.device | str = "cpu",
    precision: str = "fp32",
    training_heads: bool = True,
    packs: str | Path | None = None,
):
    """The model, with the locked media realizers installed before any optimizer or restore."""
    import torch

    from megatoken.generation import FacetedGenerator
    from megatoken.model.core import FacetedCore

    if config.vocab_size != codec.vocab_size:
        raise ValueError(
            f"Model vocabulary {config.vocab_size} != codec vocabulary {codec.vocab_size}"
        )
    model = FacetedGenerator(FacetedCore(config), codec, training_heads=training_heads)
    if packs is not None:
        from megatoken.omni.loader import install_realizers

        install_realizers(model, str(packs), device=device)
    # Casts the trainable realizer conditions too; frozen donors keep their dtype.
    return model.to(device=device, dtype=torch.bfloat16 if precision == "bf16" else torch.float32)


@dataclass(frozen=True)
class TrainedModel:
    model: Any
    codec: ByteCodec
    payload: dict[str, Any]
    metadata: dict[str, Any]


# A phase transition may install realizers for the first time (text -> media);
# nothing else may appear or disappear between a checkpoint and its successor.
NEW_REALIZER_OWNERS = ("media_realizers.packs.",)


def restore_weights(
    model, payload: dict[str, Any], packs_fingerprint: str | None, *, phase_transition: bool
) -> tuple[str, ...]:
    """Load checkpoint weights; return the owner prefixes that are new in this phase."""
    new_owners: tuple[str, ...] = ()
    if payload["packs_fingerprint"] != packs_fingerprint:
        if not (phase_transition and payload["packs_fingerprint"] is None):
            raise ValueError("Realizer pack manifest differs from the checkpoint")
        new_owners = NEW_REALIZER_OWNERS
    restored = model.load_state_dict(payload["model"], strict=not new_owners)
    if restored.unexpected_keys or any(
        not key.startswith(new_owners) for key in restored.missing_keys
    ):
        raise ValueError("Checkpoint parameters differ beyond newly installed realizers")
    return new_owners


def load_trained(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    codec_path: str | Path | None = None,
    packs: str | Path | None = None,
) -> TrainedModel:
    """Rebuild a committed checkpoint's model with its exact codec and realizer packs.

    ``codec_path``/``packs`` relocate assets; their content must match the checkpoint.
    """
    from megatoken.model.config import ModelConfig
    from megatoken.training.checkpoint import load_checkpoint

    payload, metadata = load_checkpoint(checkpoint)
    config = ModelConfig.from_dict(payload["model_config"])
    train_config = payload["train_config"]
    codec = load_codec(codec_path or train_config["codec"], vocab_size=config.vocab_size)
    if codec.fingerprint != payload["codec_fingerprint"]:
        raise ValueError("Codec does not match the trained checkpoint")
    packs = packs or train_config["packs"]
    model = build_model(
        config, codec, device=device, precision=train_config["precision"], packs=packs
    )
    restore_weights(
        model,
        payload,
        None if packs is None else sha256_file(packs),
        phase_transition=False,
    )
    model.core.set_model_version(payload["global_step"])
    bank = metadata["transducer_checkpoint"]  # load_checkpoint verified it
    # Serving needs only the packed bank; optimizer and growth state stay behind.
    import torch

    from megatoken.training.checkpoint import resolve_checkpoint

    state = torch.load(
        resolve_checkpoint(checkpoint) / bank["file"], map_location="cpu", weights_only=True
    )
    model.transducer.bank.load_snapshot(state["bank"])
    return TrainedModel(model, codec, payload, metadata)


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    """Count actual unique owners, aliases, and served versus training-only weights."""
    from megatoken.generation.policy import TRAINING_ONLY_PREFIX

    seen: dict[int, str] = {}
    aliases = []
    owners = []
    totals = {"served": 0, "training_only": 0, "temporary_import": 0}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in seen:
            aliases.append({"alias": name, "owner": seen[identity]})
            continue
        seen[identity] = name
        if "import_map" in name or "descriptor_map" in name:
            category = "temporary_import"
        elif name.startswith(TRAINING_ONLY_PREFIX):
            category = "training_only"
        else:
            category = "served"
        totals[category] += parameter.numel()
        owners.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "parameters": parameter.numel(),
                "category": category,
                "dtype": str(parameter.dtype),
            }
        )
    return {
        "totals": {**totals, "all": sum(totals.values())},
        "unique_owners": len(owners),
        "aliases": aliases,
        "owners": owners,
        "bf16_served_weight_bytes": 2 * totals["served"],
        "note": "Actual implementation allocation; activations, gradients, optimizer, and caches are separate",
    }
