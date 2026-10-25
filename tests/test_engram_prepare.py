import json
from pathlib import Path

import pytest

from megatoken import engram_prepare


def test_dual_native_preparation_binds_corpus_and_preserves_donor_order(tmp_path, monkeypatch):
    from megatoken import cache, hub, native_engram, publish

    pin = tmp_path / "pin.json"
    document = {"repo": "test/data", "revision": "fixed-data-revision"}
    pin.write_text(json.dumps(document))
    monkeypatch.setattr(publish, "load_pin", lambda path: document)
    key_file = tmp_path / "keys"
    key_file.write_text("fixture key file")
    monkeypatch.setattr(cache, "key_path", lambda: key_file)
    monkeypatch.setattr(cache, "load_keys", lambda: "private-keys")
    monkeypatch.setattr(hub, "api", lambda: "api-client")

    def restore(*, api, keys, pin, dest):
        assert api == "api-client" and keys == "private-keys" and pin == document
        root = dest / "corpus"
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text("{}")

    monkeypatch.setattr(publish, "fetch", restore)

    class Corpus:
        identity = "verified-corpus-identity"

        def __init__(self, roots, *, split):
            assert [root.name for root in roots] == ["corpus"] and split == "train"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("megatoken.data.PreparedCorpus", Corpus)
    downloads = []

    def download(*, repo_id, revision, filename, local_dir, token):
        assert token is False
        downloads.append((repo_id, revision, filename))
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / filename).write_text(f"{repo_id}: {filename}")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    extracted = []

    def extract(name, metadata, roots, output, **kwargs):
        from megatoken.io import sha256_file
        from megatoken.loading import load_model_config

        extracted.append((name, kwargs))
        assert (metadata / "tokenizer.json").is_file()
        assert [root.name for root in roots] == ["corpus"]
        output.mkdir()
        (output / "descriptors.npz").write_bytes(b"test donor descriptors")
        target = load_model_config(kwargs["model"])
        report = {
            "algorithm": "native-context-to-byte-hash-v1",
            "donor_profile": name,
            "corpus_fingerprint": Corpus.identity,
            "corpus_split": "train",
            "max_contexts": kwargs["max_contexts"],
            "seed": kwargs["seed"],
            "ngram_rows": target.ngram_rows,
            "ngram_orders": list(target.ngram_orders),
            "exact_rows": target.exact_rows,
            "exact_bigrams": target.exact_bigrams,
            "hash_algorithm": "rolling-257-order-seeded-v1",
            "donor_metadata": {"revision": engram_prepare.DONOR_PROFILES[name].revision},
            "descriptor_sha256": sha256_file(output / "descriptors.npz"),
        }
        (output / "engram.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(native_engram, "extract_native_engram", extract)
    out = tmp_path / "assets"
    result = engram_prepare.prepare_donors(
        pin, out, tmp_path / "work", model="tiny", max_contexts=32, seed=19
    )
    assert [name for name, _ in extracted] == list(engram_prepare.SELECTED_DONORS)
    assert all(
        args
        == {
            "model": "tiny",
            "max_contexts": 32,
            "seed": 19,
            "split": "train",
            "cache_dir": tmp_path / "work" / "native-row-cache",
            "source_mode": "local",
        }
        for _, args in extracted
    )
    assert result["corpus_fingerprint"] == "verified-corpus-identity"
    assert result["data_revision"] == document["revision"]
    for name, asset in zip(engram_prepare.SELECTED_DONORS, result["assets"], strict=True):
        profile = engram_prepare.DONOR_PROFILES[name]
        assert (profile.repository, profile.revision, "tokenizer.json") in downloads
        assert "LICENSE" in (Path(asset["path"]) / "DONOR-LICENSE").read_text()
    assert json.loads((out / "preparation.json").read_text()) == result
    # A retry reuses verified finished assets, including when the second donor's
    # original attempt had not finished. Different requested extraction settings fail.
    resumed = engram_prepare.prepare_donors(
        pin, out, tmp_path / "work", model="tiny", max_contexts=32, seed=19
    )
    assert resumed == result and len(extracted) == 2
    with pytest.raises(ValueError, match="differs from this preparation"):
        engram_prepare.prepare_donors(
            pin, out, tmp_path / "work", model="tiny", max_contexts=64, seed=19
        )


def test_native_preparation_rejects_empty_context_budget(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        engram_prepare.prepare_donors(
            "unused", tmp_path / "out", tmp_path / "work", model="tiny", max_contexts=0, seed=17
        )
