"""One model/codec construction path used by training, evaluation, and serving."""

from __future__ import annotations

import json
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
    if name_or_path in {"recurrent32", "compact32"}:
        raise ValueError(
            "Training/model configuration uses canonical preset names 32x512 or 32x256; "
            "recurrent32 and compact32 are command-line aliases only"
        )
    if name_or_path in {"v12", "v13", "reference"}:
        raise ValueError("The old carry architecture is incompatible; select 32x512 or 32x256")
    return ModelConfig.from_dict({**read_mapping(name_or_path), **overrides})


def build_model(
    config: ModelConfig,
    codec: ByteCodec,
    *,
    device: torch.device | str = "cpu",
    precision: str = "fp32",
    training_heads: bool = True,
):
    import torch

    from megatoken.generation import FacetedGenerator
    from megatoken.model.core import FacetedCore

    if config.vocab_size != codec.vocab_size:
        raise ValueError(
            f"Model vocabulary {config.vocab_size} != codec vocabulary {codec.vocab_size}"
        )
    model = FacetedGenerator(FacetedCore(config), codec, training_heads=training_heads)
    return model.to(device=device, dtype=torch.bfloat16 if precision == "bf16" else torch.float32)


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    """Count actual unique owners, aliases, and served versus training-only weights."""
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
