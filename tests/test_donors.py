from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from megatoken import donors
from megatoken.donors import DONOR_PROFILES, load_donor_profile, read_donor_vocabulary
from megatoken.generation.codec import _byte_alphabet


def byte_level_spellings() -> dict[str, int]:
    alphabet = _byte_alphabet()
    return {spelling: byte for spelling, byte in alphabet.items()}


def write_tokenizer(path: Path, *, deepseek_quirks: bool) -> int:
    tokens = byte_level_spellings()
    if deepseek_quirks:
        vocab = {"<｜begin▁of▁sentence｜>": 0, "<｜end▁of▁sentence｜>": 1, "<｜▁pad▁｜>": 2}
        vocab.update({spelling: byte + 3 for spelling, byte in tokens.items()})
        vocab["hello"] = 259
        added = [
            {"id": row, "content": spelling, "special": True}
            for row, spelling in enumerate(
                ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>")
            )
        ]
        added.extend(
            [
                {"id": 260, "content": "<|im_start|>", "special": True},
                {"id": 261, "content": "tool_call", "special": False},
                {"id": 262, "content": "🦄", "special": False},
            ]
        )
        rows = 300
    else:
        vocab = {spelling: byte for spelling, byte in tokens.items()}
        vocab["hello"] = 256
        added = [
            {"id": 257, "content": "<|im_start|>", "special": True},
            {"id": 270, "content": "tool_call", "special": False},
        ]
        rows = 320
    document = {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True},
        "added_tokens": added,
    }
    path.write_text(json.dumps(document, ensure_ascii=False))
    return rows


def test_deepseek_special_rows_are_masked_before_byte_unpacking(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    rows = write_tokenizer(tokenizer, deepseek_quirks=True)

    vocabulary = read_donor_vocabulary(tokenizer, vocab_size=rows)

    assert vocabulary.vocab_size == 300
    assert vocabulary.token_bytes(259) == b"hello"
    assert vocabulary.token_bytes(261) == b"tool_call"
    assert {0, 1, 2, 260, 262}.isdisjoint(vocabulary.valid_token_ids)
    assert len(vocabulary.valid_token_ids) == 258
    assert len({vocabulary.token_bytes(i) for i in vocabulary.valid_token_ids}) >= 257
    with pytest.raises(ValueError, match="control or padding"):
        vocabulary.token_bytes(0)


def test_qwen_padded_rows_keep_original_ids_and_added_lexical_spellings(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    rows = write_tokenizer(tokenizer, deepseek_quirks=False)

    vocabulary = read_donor_vocabulary(tokenizer, vocab_size=rows)

    assert vocabulary.vocab_size == 320
    assert vocabulary.token_bytes(256) == b"hello"
    assert vocabulary.token_bytes(270) == b"tool_call"
    assert vocabulary.token_bytes(65) == b"A"
    assert 257 not in vocabulary.valid_token_ids
    assert all(i not in vocabulary.valid_token_ids for i in range(271, 320))
    assert vocabulary.fingerprint == read_donor_vocabulary(tokenizer, vocab_size=rows).fingerprint


def test_donor_reader_checks_checksum_and_fallback_coverage(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    rows = write_tokenizer(tokenizer, deepseek_quirks=False)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        read_donor_vocabulary(tokenizer, vocab_size=rows, expected_sha256="0" * 64)

    document = json.loads(tokenizer.read_text())
    document["model"]["vocab"].pop("A")
    tokenizer.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="single-byte fallbacks"):
        read_donor_vocabulary(tokenizer, vocab_size=rows)


def test_profiles_pin_both_ordinary_input_tables() -> None:
    deepseek = DONOR_PROFILES["deepseek-v4.1-flash-1017"]
    qwen = DONOR_PROFILES["qwen3.8-flash-next"]
    assert deepseek.repository == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert deepseek.tensor_name == "embed.weight"
    assert deepseek.vocab_size == 129_280 and deepseek.hidden_size == 5_120
    assert qwen.repository == "Qwen/Qwen3.8-Flash-Next"
    assert qwen.tensor_name == "model.language_model.embed_tokens.weight"
    assert qwen.vocab_size == 248_320 and qwen.hidden_size == 2_560


def test_profile_loader_verifies_pinned_small_files_and_index(tmp_path: Path, monkeypatch) -> None:
    profile = DONOR_PROFILES["qwen3.8-flash-next"]
    tokenizer = tmp_path / "tokenizer.json"
    rows = write_tokenizer(tokenizer, deepseek_quirks=False)
    config = json.dumps(
        {"text_config": {"vocab_size": rows, "hidden_size": 8, "tie_word_embeddings": False}}
    ).encode()
    index = json.dumps({"weight_map": {"synthetic.embed": "synthetic.safetensors"}}).encode()
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "model.safetensors.index.json").write_bytes(index)
    (tmp_path / "synthetic.safetensors").touch()
    profile = replace(
        profile,
        name="synthetic",
        vocab_size=rows,
        hidden_size=8,
        tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config).hexdigest(),
        index_sha256=hashlib.sha256(index).hexdigest(),
        tensor_name="synthetic.embed",
        shard_name="synthetic.safetensors",
    )
    monkeypatch.setattr(donors, "DONOR_PROFILES", {"synthetic": profile})

    asset = load_donor_profile(profile.name, tmp_path)

    assert asset.profile == profile
    assert asset.embedding_file == tmp_path / profile.shard_name
    assert asset.vocabulary.vocab_size == rows
    assert asset.vocabulary.valid_token_ids
