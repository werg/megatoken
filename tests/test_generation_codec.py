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


def tokenizer_document(*, merged_special=False, ignore_merges=False):
    byte_to_text = {value: text for text, value in _byte_alphabet().items()}
    vocabulary = {byte_to_text[value]: value for value in range(256)}
    vocabulary.update(ab=256, abc=257, bc=258)
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 256 if merged_special else 259,
                "content": "ab" if merged_special else "<special>",
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
            "merges": [["a", "b"], ["ab", "c"], ["b", "c"]],
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": ignore_merges,
        },
    }


def load_fixture(tmp_path, **kwargs):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(tokenizer_document(**kwargs)), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return ByteCodec.from_tokenizer_json(path, expected_sha256=sha, expected_vocab_size=261), path


def test_raw_import_disables_normalizer_and_special_interpretation(tmp_path):
    codec, path = load_fixture(tmp_path)
    for raw in ["e\u0301".encode(), b"<special>", b"ab", b"\xffab"]:
        assert codec.decode(codec.encode(raw)) == raw
    assert codec.encode(b"ab") == [256]
    assert codec.encode(b"\xffab") == [255, 97, 98]
    assert 259 not in codec.valid_token_ids and 260 not in codec.valid_token_ids
    with pytest.raises(ValueError, match="not a lexical"):
        codec.decode([259])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ByteCodec.from_tokenizer_json(path, expected_sha256="0" * 64)


@pytest.mark.parametrize("use_fast", [False, True])
@pytest.mark.parametrize("ignore_merges", [False, True])
def test_special_token_reusing_a_merge_row_keeps_fast_and_manual_raw_bytes(
    tmp_path, use_fast, ignore_merges
):
    codec, _ = load_fixture(tmp_path, merged_special=True, ignore_merges=ignore_merges)
    if use_fast:
        pytest.importorskip("tokenizers")
        assert codec._fast_tokenizer is not None
    else:
        codec._fast_tokenizer = None
    assert codec.encode(b"ab") == [97, 98]
    # The blocked intermediate "ab" cannot participate in a later "abc" merge.
    # The independent legal b+c merge must still be used in both backends.
    assert codec.encode(b"abc") == [97, 258]
    assert codec.encode(b"bc") == [258]
    for payload in [b"ababc", "é ab abc e\u0301".encode(), b"\xffab", b"<special>"]:
        ids = codec.encode(payload)
        assert 256 not in ids
        assert codec.decode(ids) == payload
    with pytest.raises(ValueError, match="not a lexical"):
        codec.decode([256])


def test_manual_codec_omits_unrepresentable_merge_outputs():
    pieces = [bytes([value]) for value in range(256)] + [None]
    codec = ByteCodec(pieces, merge_ranks={(b"a", b"b"): 0})
    assert codec.encode(b"ab") == [97, 98]
    assert codec.decode(codec.encode(b"abab")) == b"abab"


def test_invalid_codecs_and_masks_fail_closed():
    with pytest.raises(ValueError, match="fallbacks"):
        ByteCodec([b"a"])
    codec = ByteCodec.byte_fallback()
    assert ord("\n") in codec.allowed_ids("ascii")
    assert ord(" ") not in codec.allowed_ids("identifier")
    assert codec.decode(codec.allowed_ids("identifier"))
