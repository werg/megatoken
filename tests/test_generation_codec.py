import hashlib
import json

import pytest

from megatoken.generation.codec import ByteCodec, ByteFallbackCodec, _byte_alphabet


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        bytes(range(256)),
        b"a\x00b\xff\xc0\x80",
        "Grüße 中文 🛰 e\u0301".encode(),
        b"<|role_end|>",
    ],
)
def test_byte_codec_round_trip(raw):
    codec = ByteFallbackCodec()
    assert codec.decode(codec.encode(raw)) == raw
    assert codec.fingerprint == ByteCodec.byte_fallback().fingerprint


def test_noncanonical_ids_are_not_retokenized():
    pieces = [bytes([value]) for value in range(256)] + [b"ab"]
    codec = ByteCodec(pieces, merge_ranks={(b"a", b"b"): 0})
    assert codec.encode(b"ab") == [256]
    actual = (97, 98)
    assert codec.decode(actual) == b"ab"
    assert tuple(codec.encode(codec.decode(actual))) != actual


def test_raw_import_disables_normalizer_and_special_interpretation(tmp_path):
    byte_to_text = {value: text for text, value in _byte_alphabet().items()}
    vocabulary = {byte_to_text[value]: value for value in range(256)}
    vocabulary["ab"] = 256
    document = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 257,
                "content": "<special>",
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [["a", "b"]],
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        },
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    codec = ByteCodec.from_tokenizer_json(path, expected_sha256=sha, expected_vocab_size=260)
    for raw in ["e\u0301".encode(), b"<special>", b"ab", b"\xffab"]:
        assert codec.decode(codec.encode(raw)) == raw
    assert codec.encode(b"ab") == [256]
    assert codec.encode(b"\xffab") == [255, 97, 98]  # declared binary meter
    assert 257 not in codec.valid_token_ids and 259 not in codec.valid_token_ids
    with pytest.raises(ValueError, match="not a lexical"):
        codec.decode([257])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ByteCodec.from_tokenizer_json(path, expected_sha256="0" * 64)


def test_invalid_codecs_and_masks_fail_closed():
    with pytest.raises(ValueError, match="fallbacks"):
        ByteCodec([b"a"])
    codec = ByteCodec.byte_fallback()
    assert ord("\n") in codec.allowed_ids("ascii")
    assert ord(" ") not in codec.allowed_ids("identifier")
    assert codec.decode(codec.allowed_ids("identifier"))
