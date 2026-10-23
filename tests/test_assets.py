import hashlib
import json
from argparse import Namespace
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from training_fixture import tiny_train_config

from megatoken.assets import bake_checkpoint, extract_engram, install_engram_asset
from megatoken.cli import _diloco_initialize
from megatoken.data.chunking import ChunkingConfig
from megatoken.generation.codec import ByteCodec
from megatoken.io import sha256_file
from megatoken.loading import build_model, load_model_config
from megatoken.runtime import capture_rng
from megatoken.served import ServedBundle, export_served
from megatoken.training.checkpoint import load_checkpoint, save_checkpoint, transducer_attachment
from megatoken.training.config import DecoupledConfig
from megatoken.training.diloco_export import export_coordinator
from megatoken.training.optimizer import MasterAdamW
from tests.media_support import install_fake_realizers


@pytest.mark.parametrize("bad", ["foo", ("a", "b"), (), ["a", ""]])
def test_multiple_imports_requires_nonempty_ordered_path_list(bad):
    from megatoken.training.config import TrainConfig

    with pytest.raises(ValueError, match="engram_imports"):
        TrainConfig(engram_imports=bad)


def test_bootstrap_preserves_optimizer_ownership_of_all_imports():
    from megatoken.model.frontend import NativeNGramTable

    table = NativeNGramTable(load_model_config("tiny"))
    for _ in range(2):
        table.install_import(torch.tensor([7]), torch.ones(1, 128),
                             torch.ones(1), torch.ones(1))
    optimizer = MasterAdamW(table, lr=0.01)
    parameters = dict(table.named_parameters())
    state = {name: tensor.clone() for name, tensor in table.state_dict().items()}
    table.load_state_dict(state)
    assert all(parameter is parameters[name] for name, parameter in table.named_parameters())
    # This is the order used by learner bootstrap: optimizer exists before load.
    maps = [table.import_map.weight, table.import_branches[0].import_map.weight]
    before = [parameter.detach().clone() for parameter in maps]
    sum(parameter.square().sum() for parameter in maps).backward()
    optimizer.step()
    assert all(not torch.equal(parameter, initial)
               for parameter, initial in zip(maps, before, strict=True))


def test_profile_extraction_uses_pinned_donor_byte_vocabulary(tmp_path, monkeypatch):
    from megatoken import donors

    vocabulary = donors.DonorByteVocabulary(tuple(bytes([i]) for i in range(256)), "vocab-fp")
    embedding = tmp_path / "weights.safetensors"
    save_file({"embedding": torch.randn(256, 32)}, str(embedding))
    profile = donors.DonorProfile(
        "synthetic",
        "example/model",
        "revision",
        "test",
        256,
        32,
        "tok-sha",
        "config-sha",
        "index-sha",
        hashlib.sha256(embedding.read_bytes()).hexdigest(),
        embedding.stat().st_size,
        "embedding",
        "weights.safetensors",
    )
    asset = donors.DonorAsset(profile, embedding, vocabulary)
    monkeypatch.setattr(donors, "load_donor_profile", lambda *_: asset)
    destination = tmp_path / "asset"
    extract_engram(None, None, None, destination, donor_profile="synthetic", donor_root=tmp_path)
    metadata = json.loads((destination / "engram.json").read_text())
    assert metadata["repository"] == "example/model"
    assert metadata["revision"] == "revision"
    assert metadata["donor_vocabulary_fingerprint"] == "vocab-fp"


def test_native_hashed_asset_install_is_bound_to_model_hash_layout(tmp_path):
    config = load_model_config("tiny")
    codec = ByteCodec.byte_fallback()
    model = build_model(config, codec)
    destination = tmp_path / "native"
    destination.mkdir()
    row = config.exact_rows + 7
    descriptor = np.ones((1, 128), dtype=np.float32)
    with (destination / "descriptors.npz").open("wb") as handle:
        np.savez(
            handle,
            rows=np.asarray([row], dtype=np.int64),
            descriptors=descriptor,
            coverage=np.ones(1, dtype=np.float32),
            support=np.ones(1, dtype=np.float32),
        )
    metadata = {
        "format_version": 1,
        "exact_rows": config.exact_rows,
        "ngram_rows": config.ngram_rows,
        "ngram_orders": list(config.ngram_orders),
        "exact_bigrams": config.exact_bigrams,
        "row_space": "native_hashed",
        "algorithm": "native-context-to-byte-hash-v1",
        "hash_algorithm": "rolling-257-order-seeded-v1",
        "descriptor_sha256": sha256_file(destination / "descriptors.npz"),
    }
    (destination / "engram.json").write_text(json.dumps(metadata))
    install_engram_asset(model, destination)
    assert model.core.frontend.ngrams.has_import
    metadata["ngram_rows"] += 1
    (destination / "engram.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="native table size"):
        install_engram_asset(build_model(config, codec), destination)


def test_offline_engram_extract_install_checkpoint_bake_and_export(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    donor = tmp_path / "donor.safetensors"
    save_file({"embedding": torch.randn(256, 32)}, str(donor))
    asset = tmp_path / "engram"
    metadata = extract_engram(donor, "embedding", "byte", asset, model="tiny")
    donor2 = tmp_path / "donor2.safetensors"
    save_file({"embedding": torch.randn(256, 48)}, str(donor2))
    asset2 = tmp_path / "engram2"
    extract_engram(donor2, "embedding", "byte", asset2, model="tiny", seed=29)
    with np.load(asset2 / "descriptors.npz") as archive:
        reduced = {name: archive[name][:16].copy() for name in archive.files}
    with (asset2 / "descriptors.npz").open("wb") as handle:
        np.savez(handle, **reduced)
    metadata2 = json.loads((asset2 / "engram.json").read_text())
    metadata2["covered_rows"] = 16
    metadata2["descriptor_sha256"] = sha256_file(asset2 / "descriptors.npz")
    (asset2 / "engram.json").write_text(json.dumps(metadata2))
    assert metadata["covered_rows"] == 256 and metadata["descriptor_dim"] == 128
    second = tmp_path / "reproduced"
    extract_engram(donor, "embedding", "byte", second, model="tiny")
    with np.load(asset / "descriptors.npz") as first, np.load(second / "descriptors.npz") as again:
        np.testing.assert_array_equal(first["descriptors"], again["descriptors"])
    codec, config = ByteCodec.byte_fallback(), load_model_config("tiny")
    model = build_model(config, codec, precision="bf16")
    install_engram_asset(model, asset)
    install_engram_asset(model, asset2)
    optimizer = MasterAdamW(model, lr=1e-3, inner_optimizer="muon")
    table = model.core.frontend.ngrams
    table.lookup(torch.arange(256)).float().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    train_config = tiny_train_config(
        tmp_path / "cfg",
        engram_imports=[str(asset), str(asset2)],
        inner_optimizer="muon",
    )
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
            "packs_fingerprint": None,
            "chunking": ChunkingConfig().with_hard_limit(config.max_capacity).identity(codec),
            "rank_state": [{"rng": capture_rng()}],
        },
        {"global_step": 1, "phase": "B0"},
        pinned=True,
        attachments=transducer_attachment(lambda: {"bank": model.transducer.bank.state_snapshot()}),
    )
    optimizer.close()
    with pytest.raises(RuntimeError, match="bake|import"):
        export_served(source, tmp_path / "unbaked")
    baked = bake_checkpoint(source, tmp_path / "baked")
    assert baked["max_lookup_error"] == 0 and baked["optimizer_rows_reset"]
    payload, _ = load_checkpoint(baked["checkpoint"])
    assert not any("import_map" in name for name in payload["model"])
    assert not any("import_map" in name for name in payload["optimizer"]["master_weights"])
    export_served(baked["checkpoint"], tmp_path / "served")
    (asset / "descriptors.npz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        install_engram_asset(build_model(config, codec), asset)


def test_engram_extractor_rejects_rows_that_are_not_tokenizer_vocabulary(tmp_path):
    donor = tmp_path / "hashed-table.safetensors"
    save_file({"layers.1.engram.embed": torch.randn(512, 32)}, str(donor))
    with pytest.raises(ValueError, match="exactly match.*hashed n-gram"):
        extract_engram(
            donor,
            "layers.1.engram.embed",
            "byte",
            tmp_path / "rejected",
            model="tiny",
        )


def test_diloco_export_bake_preserves_global_masters_and_served_lookup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    donor = tmp_path / "donor.safetensors"
    save_file({"embedding": torch.randn(256, 32)}, str(donor))
    asset = tmp_path / "engram"
    extract_engram(donor, "embedding", "byte", asset, model="tiny")
    donor2 = tmp_path / "donor2.safetensors"
    save_file({"embedding": torch.randn(256, 64)}, str(donor2))
    asset2 = tmp_path / "engram2"
    extract_engram(donor2, "embedding", "byte", asset2, model="tiny", seed=29)
    with np.load(asset2 / "descriptors.npz") as archive:
        reduced = {name: archive[name][:16].copy() for name in archive.files}
    with (asset2 / "descriptors.npz").open("wb") as handle:
        np.savez(handle, **reduced)
    metadata2 = json.loads((asset2 / "engram.json").read_text())
    metadata2["covered_rows"] = 16
    metadata2["descriptor_sha256"] = sha256_file(asset2 / "descriptors.npz")
    (asset2 / "engram.json").write_text(json.dumps(metadata2))
    packs = tmp_path / "packs.json"
    packs.write_text('{"version": 1}')
    monkeypatch.setattr(
        "megatoken.omni.loader.install_realizers",
        lambda model, manifest, *, device: install_fake_realizers(model),
    )
    config = load_model_config("tiny")
    train = tiny_train_config(
        tmp_path / "fixture",
        output=str(tmp_path / "runs"),
        precision="bf16",
        packs=str(packs),
        engram_imports=[str(asset), str(asset2)],
        prepared_chunking=ChunkingConfig(
            target_tokens=128, soft_max_tokens=192, hard_max_tokens=256, max_bytes=8192
        ),
        decoupled=DecoupledConfig(coordinator_url="http://127.0.0.1:9", run_id="engram", workers=2),
    )
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(asdict(train)))
    root = tmp_path / "coordinator"
    _diloco_initialize(
        Namespace(
            config=str(config_path),
            root=str(root),
            resume=None,
            quorum=1,
            fragment_mib=1,
            outer_lr=0.7,
            outer_momentum=0.9,
            grace_seconds=0.0,
            max_staleness=2,
        )
    )
    exported = export_coordinator(root, tmp_path / "export", train)["checkpoint"]
    payload, _ = load_checkpoint(exported)
    assert payload["export_only"] and "optimizer" not in payload
    assert payload["chunking"]["config"]["target_tokens"] == 128
    master_name = "core.frontend.ngrams.embedding.weight"
    map_name = "core.frontend.ngrams.import_map.weight"
    branch_map_name = "core.frontend.ngrams.import_branches.0.import_map.weight"
    initial_map = payload["model"][map_name].clone()
    initial_branch_map = payload["model"][branch_map_name].clone()
    original_masters = payload["master_weights"]
    initial_table_master = original_masters[master_name].clone()
    assert all(value.dtype == torch.float32 for value in original_masters.values())
    with pytest.raises(RuntimeError, match="bake|import"):
        export_served(exported, tmp_path / "unbaked-served")

    # Submit real coordinator updates: change the imported map and place a tiny
    # FP32 residual on an unrelated table row below BF16 resolution.
    from megatoken.training.diloco_protocol import save_tensors
    from megatoken.training.diloco_server import Coordinator

    coordinator = Coordinator(root)
    session = "00000000-0000-4000-8000-000000000001"
    coordinator.lease(0, session)
    base_value = original_masters[master_name][1000, 0].abs()
    sub_bf16_shift = base_value * torch.finfo(torch.bfloat16).eps * 0.1
    table_delta = -sub_bf16_shift / (0.7 * (1 + 0.9))
    for fragment in coordinator.fragments:
        delta = torch.zeros(fragment.length, dtype=torch.float32)
        changed = False
        cursor = 0
        for part in fragment.slices:
            if part.name.endswith(".import_map.weight"):
                delta[cursor : cursor + part.length] = -1e-3
                changed = True
            elif part.name == master_name:
                offset = 1000 * config.ngram_dim
                if part.offset <= offset < part.offset + part.length:
                    delta[cursor + offset - part.offset] = table_delta
                    changed = True
            cursor += part.length
        if changed:
            update = tmp_path / f"fragment-{fragment.index}.safetensors"
            save_tensors(update, {"delta": delta})
            coordinator.submit(fragment.index, 0, session, 1, 0, 1, 1, update)
    assert any(version > 0 for version in coordinator.status()["versions"])
    coordinator.close()
    exported = export_coordinator(root, tmp_path / "updated-export", train)["checkpoint"]
    payload, _ = load_checkpoint(exported)
    original_masters = payload["master_weights"]
    assert not torch.equal(payload["model"][map_name], initial_map)
    assert not torch.equal(payload["model"][branch_map_name], initial_branch_map)
    assert not torch.equal(original_masters[master_name][1000, 0], initial_table_master[1000, 0])
    before = build_model(config, ByteCodec.byte_fallback(), precision="bf16", packs=packs)
    before.load_state_dict(payload["model"], strict=True)
    sample = torch.tensor([[97, 98, 99, 100]])
    expected = before.core.frontend(sample).detach()
    baked = bake_checkpoint(exported, tmp_path / "baked", packs=packs)
    baked_payload, _ = load_checkpoint(baked["checkpoint"])
    assert "optimizer" not in baked_payload
    assert "core.frontend.ngrams.import_map.weight" not in baked_payload["model"]
    assert "core.frontend.ngrams.import_map.weight" not in baked_payload["master_weights"]
    assert not any("import_branches" in name for name in baked_payload["model"])
    assert not any("import_branches" in name for name in baked_payload["master_weights"])
    touched = set(before.core.frontend.ngrams.import_rows.tolist())
    for branch in before.core.frontend.ngrams.import_branches:
        touched.update(branch.import_rows.tolist())
    old = original_masters[master_name]
    new = baked_payload["master_weights"][master_name]
    torch.testing.assert_close(
        new[list(touched)], baked_payload["model"][master_name][list(touched)].float()
    )
    untouched = sorted(set(range(old.shape[0])) - touched)
    assert not torch.equal(old[1000], old[1000].bfloat16().float())
    assert torch.equal(new[untouched], old[untouched])

    import shutil

    shutil.rmtree(asset)
    shutil.rmtree(asset2)
    bundle_path = tmp_path / "served"
    export_served(baked["checkpoint"], bundle_path, packs=packs)
    served = ServedBundle(bundle_path, packs=packs, device="cpu")
    actual = served.model.core.frontend(sample).detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # The baked DiLoCo export remains a valid seed for a fresh coordinator.
    next_config = replace(
        train,
        engram_import=None,
        engram_imports=None,
        decoupled=replace(train.decoupled, run_id="next"),
    )
    next_path = tmp_path / "next.json"
    next_path.write_text(json.dumps(asdict(next_config)))
    _diloco_initialize(
        Namespace(
            config=str(next_path),
            root=str(tmp_path / "next-coordinator"),
            resume=baked["checkpoint"],
            quorum=1,
            fragment_mib=1,
            outer_lr=0.7,
            outer_momentum=0.9,
            grace_seconds=0.0,
            max_staleness=2,
        )
    )


def test_coordinator_phase_start_installs_both_requested_imports_after_checkpoint_restore(tmp_path):
    from megatoken.training.engine import train as run_training

    base = tiny_train_config(tmp_path / "base", max_steps=1)
    previous = run_training(base)["checkpoint"]
    assets = []
    for i in range(2):
        donor = tmp_path / f"donor-{i}.safetensors"
        save_file({"embedding": torch.randn(256, 32)}, str(donor))
        destination = tmp_path / f"import-{i}"
        extract_engram(donor, "embedding", "byte", destination, model="tiny", seed=i)
        assets.append(str(destination))
    next_config = replace(base, phase="P1", engram_imports=assets,
        decoupled=DecoupledConfig(coordinator_url="http://127.0.0.1:9", workers=1, run_id="next"))
    config_path = tmp_path / "next.json"
    config_path.write_text(json.dumps(asdict(next_config)))
    root = tmp_path / "coordinator"
    _diloco_initialize(Namespace(config=str(config_path), root=str(root), resume=previous,
        quorum=1, fragment_mib=1, outer_lr=0.7, outer_momentum=0.9,
        grace_seconds=0.0, max_staleness=2))
    bootstrap = torch.load(root / "bootstrap.pt", weights_only=True)["model"]
    assert "core.frontend.ngrams.import_map.weight" in bootstrap
    assert "core.frontend.ngrams.import_branches.0.import_map.weight" in bootstrap
    # Restoring the same desired imports must retain the learned maps without appending.
    from megatoken.assets import install_configured_imports
    model = build_model(load_model_config("tiny"), ByteCodec.byte_fallback())
    model.load_state_dict(bootstrap)
    before = model.core.frontend.ngrams.import_map.weight.clone()
    assert install_configured_imports(model, next_config, resumed=True) == ()
    assert len(model.core.frontend.ngrams.import_branches) == 1
    torch.testing.assert_close(model.core.frontend.ngrams.import_map.weight, before)
