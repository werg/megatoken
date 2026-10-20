"""MiniCPM exported vision tower and trained merger have immutable provenance."""

import hashlib
import json

import pytest

from megatoken.omni.perception import verify_minicpm_export


def test_exported_minicpm_requires_exact_weights_and_revision(tmp_path):
    files = {}
    for name in ("vision_tower.safetensors", "merger.safetensors", "config.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "source_revision": "a" * 40,
        "sha256": files,
        "source_snapshot": "minicpm",
        "submodule_tensors": {"vision_tower": 2, "merger": 1},
    }
    (tmp_path / "export_receipt.json").write_text(json.dumps(receipt))
    checked = verify_minicpm_export(str(tmp_path))
    assert checked["source_revision"] == "a" * 40
    (tmp_path / "merger.safetensors").write_bytes(b"new random trained merger")
    with pytest.raises(ValueError, match="SHA256"):
        verify_minicpm_export(str(tmp_path))
    (tmp_path / "merger.safetensors").write_bytes(b"merger.safetensors")
    receipt["source_revision"] = "main"
    (tmp_path / "export_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="revision"):
        verify_minicpm_export(str(tmp_path))
