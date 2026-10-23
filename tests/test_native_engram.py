import json
from types import SimpleNamespace

import torch

from megatoken import native_engram
from megatoken.donors import DonorByteVocabulary, DonorProfile
from megatoken.io import sha256_file
from megatoken.loading import load_model_config
from megatoken.native_donor_hash import NativeRowRef
from megatoken.native_engram import _original_text_contexts


def test_native_extraction_uses_real_context_and_emits_hashed_asset(tmp_path, monkeypatch):
    vocab = DonorByteVocabulary((b"a", b"b", b"c"), "vocab-fingerprint")
    profile = DonorProfile(
        "synthetic",
        "example/native",
        "pinned-revision",
        "test-license",
        3,
        2,
        "tokenizer-sha",
        "config-sha",
        "index-sha",
        "shard-sha",
        100,
        "embedding",
        "shard.safetensors",
    )
    donor_asset = SimpleNamespace(profile=profile, vocabulary=vocab)
    refs = (
        NativeRowRef(1, 2, 0, 0, 8, 8, "table", "table.safetensors", "a" * 64, 100, 2, "F32"),
        NativeRowRef(1, 3, 0, 1, 9, 9, "table", "table.safetensors", "a" * 64, 100, 2, "F32"),
    )
    addressor = SimpleNamespace(
        profile=donor_asset,
        max_context_tokens=3,
        native_memory="engram",
        compressed_token_map_sha256=None,
        address=lambda context: refs,
    )
    monkeypatch.setattr(native_engram, "_load_addressor", lambda *_: addressor)
    monkeypatch.setattr(native_engram, "_verify_remote_shards", lambda *args: None)
    monkeypatch.setattr(native_engram, "load_model_config", lambda _: load_model_config("tiny"))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "LICENSE").write_text("synthetic source license")

    class FakeTokenizer:
        @classmethod
        def from_file(cls, _path):
            return cls()

        def encode(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return SimpleNamespace(ids=[0, 1, 2], offsets=[(0, 1), (1, 2), (2, 3)])

    monkeypatch.setattr(native_engram, "Tokenizer", FakeTokenizer)

    class Corpus:
        identity = "corpus-fingerprint"

        def __init__(self, roots, split):
            assert roots == ["prepared"] and split == "train"

        def __enter__(self):
            record = SimpleNamespace(
                media_type="text/plain; charset=utf-8", payload=b"abc", role="user"
            )
            self.episodes = [SimpleNamespace(records=(record,))]
            return self

        def __exit__(self, *args):
            pass

        def iter_episodes(self):
            return iter(self.episodes)

    monkeypatch.setattr("megatoken.data.PreparedCorpus", Corpus)

    class FakeReader:
        requests = 0
        bytes_received = 0

        def __init__(self, url, tensor_name, **kwargs):
            assert "/pinned-revision/" in url
            assert tensor_name == "table"

        def read_rows(self, rows):
            return {row: torch.tensor([float(row), float(row + 1)]) for row in rows}

        def close(self):
            pass

    monkeypatch.setattr(native_engram, "SafetensorsRangeReader", FakeReader)
    output = tmp_path / "asset"
    report = native_engram.extract_native_engram(
        "deepseek-v4.1-flash-1017",
        tmp_path,
        ["prepared"],
        output,
        model="tiny",
        max_contexts=4,
        seed=9,
    )
    assert report["selected_contexts"] == 1
    assert report["corpus_fingerprint"] == "corpus-fingerprint"
    assert report["row_space"] == "native_hashed"
    metadata = json.loads((output / "engram.json").read_text())
    assert metadata["descriptor_sha256"] == sha256_file(output / "descriptors.npz")
    import numpy as np

    with np.load(output / "descriptors.npz") as archive:
        assert archive["rows"].tolist() == [_hash_row(b"abc", 3, load_model_config("tiny"))]
        assert archive["descriptors"].shape == (1, 128)


def _hash_row(suffix, order, config):
    value = order
    for byte in suffix[-order:]:
        value = (value * 257 + byte + 1) % (config.ngram_rows - config.exact_rows)
    return config.exact_rows + value


def test_native_contexts_use_original_utf8_offsets_and_skip_partial_characters():
    vocabulary = DonorByteVocabulary((b"x", b"y", b"z", b"a", b"b", b"c", b"d"), "fp")
    composed_source = "a\u0301bcdef"
    # Normalized token zero represents the original two-character sequence.
    contexts = list(
        _original_text_contexts(
            composed_source,
            [0, 1, 2, 3, 4, 5],
            [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
            3,
            vocabulary,
        )
    )
    assert contexts[0] == ((0, 1, 2), "a\u0301bc".encode("utf-8"))

    split_character_source = "éabcdef"
    byte_fallback = list(
        _original_text_contexts(
            split_character_source,
            [0, 1, 3, 4, 5, 6, 3, 4],
            [
                (0, 1),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
            ],
            3,
            vocabulary,
        )
    )
    assert byte_fallback[1] == ((3, 4, 5), b"abc")
