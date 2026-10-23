"""Pinned ordinary-token embedding donors and their byte-lexicon readers.

This module deliberately does not relax ``ByteCodec``. Donor import needs only
the exact byte spelling of each lexical vocabulary row; it does not use the
donor's tokenizer to encode text. Donor control rows and padding remain holes at
their original IDs, so every donor vector still lines up with its source row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megatoken.generation.codec import _byte_alphabet


@dataclass(frozen=True)
class DonorProfile:
    """Pinned metadata for an ordinary input-token embedding table."""

    name: str
    repository: str
    revision: str
    license_id: str
    vocab_size: int
    hidden_size: int
    tokenizer_sha256: str
    config_sha256: str
    index_sha256: str
    shard_sha256: str
    shard_size: int
    tensor_name: str
    shard_name: str


@dataclass(frozen=True)
class DonorByteVocabulary:
    """Byte spellings for donor token IDs; ``None`` marks controls or padding."""

    pieces: tuple[bytes | None, ...]
    fingerprint: str

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    @property
    def valid_token_ids(self) -> tuple[int, ...]:
        return tuple(i for i, piece in enumerate(self.pieces) if piece)

    def token_bytes(self, token_id: int) -> bytes:
        if not 0 <= int(token_id) < self.vocab_size:
            raise ValueError(f"Donor token ID {token_id} is outside the vocabulary")
        piece = self.pieces[int(token_id)]
        if not piece:
            raise ValueError(f"Donor token ID {token_id} is a control or padding row")
        return piece


@dataclass(frozen=True)
class DonorAsset:
    """A pinned donor's local shard and byte vocabulary, ready for extraction."""

    profile: DonorProfile
    embedding_file: Path
    vocabulary: DonorByteVocabulary


DONOR_PROFILES: dict[str, DonorProfile] = {
    "deepseek-v4.1-flash-1017": DonorProfile(
        name="deepseek-v4.1-flash-1017",
        repository="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
        license_id="MIT",
        vocab_size=129_280,
        hidden_size=5_120,
        tokenizer_sha256="c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
        config_sha256="8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
        index_sha256="74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
        shard_sha256="43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7",
        shard_size=1_323_858_272,
        tensor_name="embed.weight",
        shard_name="model-00002-of-00048.safetensors",
    ),
    "qwen3.8-flash-next": DonorProfile(
        name="qwen3.8-flash-next",
        repository="Qwen/Qwen3.8-Flash-Next",
        revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
        license_id="Qwen Community License 1.0",
        vocab_size=248_320,
        hidden_size=2_560,
        tokenizer_sha256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        config_sha256="889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
        index_sha256="99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
        shard_sha256="2d651616c258aaeae8ba1c4631bcbc6e72f3dca8f2a2f3add3b00c4bb45f95c2",
        shard_size=2_974_788_136,
        tensor_name="model.language_model.embed_tokens.weight",
        shard_name="model-00130-of-00131.safetensors",
    ),
}


def read_donor_vocabulary(
    tokenizer_json: str | Path,
    *,
    vocab_size: int,
    expected_sha256: str | None = None,
) -> DonorByteVocabulary:
    """Read ByteLevel token spellings without applying donor text-tokenization rules.

    Special token IDs are removed before decoding, including specials that also
    appear in the base BPE vocabulary. Non-special added tokens are included only
    when their literal spelling belongs to the reversible ByteLevel alphabet.
    Missing and padded rows remain ``None`` at their original indices.
    """
    raw = Path(tokenizer_json).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Donor tokenizer SHA256 mismatch: got {digest}")
    document = json.loads(raw)
    model = document.get("model", {})
    if model.get("type") != "BPE" or document.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("Donor byte vocabulary requires ByteLevel BPE")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Donor tokenizer has no BPE vocabulary")
    added = document.get("added_tokens", [])
    special_ids = {int(token["id"]) for token in added if token.get("special", False)}
    highest = max([*(int(i) for i in vocab.values()), *(int(t["id"]) for t in added)])
    if highest >= vocab_size:
        raise ValueError("Donor tokenizer rows exceed the declared embedding vocabulary")

    alphabet = _byte_alphabet()

    def unpack(spelling: str) -> bytes | None:
        try:
            return bytes(alphabet[char] for char in spelling)
        except KeyError:
            # Some releases place non-byte protocol markers in added-token rows.
            # They are not byte evidence and must not be mistaken for UTF-8 text.
            return None

    pieces: list[bytes | None] = [None] * vocab_size
    base_rows: set[int] = set()
    for spelling, row_value in vocab.items():
        row = int(row_value)
        if row < 0 or row >= vocab_size:
            raise ValueError("Donor tokenizer has a negative or out-of-range vocabulary row")
        if row in base_rows:
            raise ValueError(f"Donor tokenizer maps multiple base tokens to row {row}")
        base_rows.add(row)
        if row not in special_ids:
            pieces[row] = unpack(spelling)
    added_rows: set[int] = set()
    for token in added:
        row = int(token["id"])
        if row < 0 or row >= vocab_size:
            raise ValueError("Donor tokenizer has a negative or out-of-range added-token row")
        if row in added_rows:
            raise ValueError(f"Donor tokenizer repeats added-token row {row}")
        added_rows.add(row)
        if token.get("special", False):
            pieces[row] = None
        else:
            added_piece = unpack(str(token["content"]))
            if row in base_rows and pieces[row] not in (None, added_piece):
                raise ValueError(f"Added token disagrees with base vocabulary row {row}")
            if pieces[row] is None:
                pieces[row] = added_piece

    valid = {piece for piece in pieces if piece}
    missing = [value for value in range(256) if bytes((value,)) not in valid]
    if missing:
        raise ValueError(f"Donor byte vocabulary lacks single-byte fallbacks: {missing[:8]}")

    fingerprint = hashlib.sha256()
    fingerprint.update(b"donor-byte-vocabulary-v1\0")
    fingerprint.update(digest.encode("ascii"))
    fingerprint.update(vocab_size.to_bytes(8, "little"))
    for piece in pieces:
        if piece is None:
            fingerprint.update((-1).to_bytes(4, "little", signed=True))
        else:
            fingerprint.update(len(piece).to_bytes(4, "little", signed=True))
            fingerprint.update(piece)
    return DonorByteVocabulary(tuple(pieces), fingerprint.hexdigest())


def load_donor_profile(name: str, donor_root: str | Path) -> DonorAsset:
    """Validate a pinned local snapshot and return its ordinary input embedding."""
    try:
        profile = DONOR_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown donor profile {name!r}; choose from {sorted(DONOR_PROFILES)}"
        ) from exc

    root = Path(donor_root)
    _read_pinned_file(root / "tokenizer.json", profile.tokenizer_sha256)
    raw_config = _read_pinned_file(root / "config.json", profile.config_sha256)
    raw_index = _read_pinned_file(root / "model.safetensors.index.json", profile.index_sha256)
    config = json.loads(raw_config)
    text_config = config.get("text_config", config)
    if int(text_config.get("vocab_size", -1)) != profile.vocab_size:
        raise ValueError("Donor config vocabulary size does not match the pinned profile")
    if int(text_config.get("hidden_size", -1)) != profile.hidden_size:
        raise ValueError("Donor config hidden size does not match the pinned profile")
    if text_config.get("tie_word_embeddings") is not False:
        raise ValueError("Donor profile requires a distinct ordinary input embedding table")
    index = json.loads(raw_index)
    if index.get("weight_map", {}).get(profile.tensor_name) != profile.shard_name:
        raise ValueError("Donor index does not map the pinned input tensor to its expected shard")

    vocabulary = read_donor_vocabulary(
        root / "tokenizer.json",
        vocab_size=profile.vocab_size,
        expected_sha256=profile.tokenizer_sha256,
    )
    return DonorAsset(profile, root / profile.shard_name, vocabulary)


def _read_pinned_file(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned donor file is missing: {path.name}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Pinned donor file checksum mismatch for {path.name}: got {digest}")
    return raw


def donor_profile_metadata(asset: DonorAsset) -> dict[str, Any]:
    """Small provenance record to store beside extracted descriptors."""
    profile = asset.profile
    return {
        "donor_kind": "ordinary_input_token_embeddings",
        "repository": profile.repository,
        "revision": profile.revision,
        "license_id": profile.license_id,
        "tensor_name": profile.tensor_name,
        "shard_name": profile.shard_name,
        "shard_sha256": profile.shard_sha256,
        "shard_size": profile.shard_size,
        "vocab_size": profile.vocab_size,
        "hidden_size": profile.hidden_size,
        "tokenizer_sha256": profile.tokenizer_sha256,
        "donor_vocabulary_fingerprint": asset.vocabulary.fingerprint,
    }
