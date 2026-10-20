"""Pretrained checkpoint manifests must cover every loaded file and forbid network fallback."""
import json

import pytest

from megatoken.omni.loader import read_locked_manifest


def test_strict_lock_rejects_unlisted_donor_bytes(tmp_path):
    import hashlib

    snapshot = tmp_path / "sana"
    snapshot.mkdir()
    model = snapshot / "model_index.json"
    model.write_text('{"components":[]}', encoding="utf-8")
    manifest = tmp_path / "pack-lock.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "packs": {"sana": {
            "revision": "a" * 40,
            "snapshot": "sana",
            "files": {"sana/model_index.json": hashlib.sha256(model.read_bytes()).hexdigest()},
        }},
    }), encoding="utf-8")
    root, packs = read_locked_manifest(str(manifest))
    assert root == tmp_path
    assert packs["sana"]["snapshot"] == "sana"
    (snapshot / "unexpected.safetensors").write_bytes(b"hidden weights")
    with pytest.raises(ValueError, match="Unhashed"):
        read_locked_manifest(str(manifest))
    (snapshot / "unexpected.safetensors").unlink()
    model.write_text('{"components":[1]}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_locked_manifest(str(manifest))


def test_strict_lock_refuses_remote_pocket_resources(tmp_path):
    import hashlib

    pack = tmp_path / "pocket"
    pack.mkdir()
    source = pack / "local.yaml"
    source.write_text(
        "weights_path: hf://kyutai/pocket-tts/model.safetensors@mutable\n"
        "flow_lm:\n  lookup_table:\n    tokenizer_path: tokenizer.json\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "pack-lock.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "packs": {"pocket": {
            "revision": "b" * 40,
            "config": "pocket/local.yaml",
            "files": {"pocket/local.yaml": hashlib.sha256(source.read_bytes()).hexdigest()},
        }},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Remote Pocket"):
        read_locked_manifest(str(manifest))


def test_strict_lock_rejects_moving_revision(tmp_path):
    import hashlib

    pack = tmp_path / "sana"
    pack.mkdir()
    artifact = pack / "model_index.json"
    artifact.write_text("{}")
    manifest = tmp_path / "pack-lock.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "packs": {"sana": {
            "revision": "main",
            "snapshot": "sana",
            "files": {"sana/model_index.json": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        }},
    }))
    with pytest.raises(ValueError, match="revision"):
        read_locked_manifest(str(manifest))
