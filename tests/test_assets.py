from dataclasses import asdict

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from megatoken.assets import bake_checkpoint, extract_engram, install_engram_asset
from megatoken.data.chunking import ChunkingConfig
from megatoken.generation.codec import ByteCodec
from megatoken.inference import export_served
from megatoken.loading import build_model, load_model_config
from megatoken.runtime import capture_rng
from megatoken.training.checkpoint import load_checkpoint, save_checkpoint
from megatoken.training.optimizer import MasterAdamW

from retrieval_fixture import retrieval_train_config


def test_offline_engram_extract_install_checkpoint_bake_and_export(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    donor = tmp_path / "donor.safetensors"
    save_file({"embedding": torch.randn(256, 32)}, str(donor))
    asset = tmp_path / "engram"
    metadata = extract_engram(donor, "embedding", "byte", asset, model="tiny")
    assert metadata["covered_rows"] == 256 and metadata["descriptor_dim"] == 128
    second = tmp_path / "reproduced"
    extract_engram(donor, "embedding", "byte", second, model="tiny")
    with np.load(asset / "descriptors.npz") as first, np.load(second / "descriptors.npz") as again:
        np.testing.assert_array_equal(first["descriptors"], again["descriptors"])
    codec, config = ByteCodec.byte_fallback(), load_model_config("tiny")
    model = build_model(config, codec, precision="bf16")
    install_engram_asset(model, asset)
    optimizer = MasterAdamW(model, lr=1e-3)
    table = model.core.frontend.ngrams
    table.lookup(torch.arange(256)).float().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    train_config = retrieval_train_config(tmp_path / "cfg", engram_import=str(asset))
    source = save_checkpoint(
        tmp_path / "input",
        "before",
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": config.to_dict(),
            "train_config": asdict(train_config),
            "global_step": 1,
            "codec_fingerprint": codec.fingerprint,
            "chunking": ChunkingConfig().with_hard_limit(config.max_capacity).identity(codec),
            "rank_state": [{"rng": capture_rng()}],
        },
        {"global_step": 1, "phase": "B0"},
        pinned=True,
    )
    optimizer.close()
    with pytest.raises(RuntimeError, match="bake|import"):
        export_served(source, tmp_path / "unbaked")
    baked = bake_checkpoint(source, tmp_path / "baked")
    assert baked["max_lookup_error"] == 0 and baked["optimizer_rows_reset"]
    payload, _ = load_checkpoint(baked["checkpoint"])
    assert not any("import_map" in name for name in payload["model"])
    assert not any("import_map" in name for name in payload["optimizer"]["names"])
    export_served(baked["checkpoint"], tmp_path / "served")
    (asset / "descriptors.npz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        install_engram_asset(build_model(config, codec), asset)
