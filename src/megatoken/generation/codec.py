"""Byte-exact output IDs and a pinned, raw (unnormalized) Ling token meter.

This module intentionally has no torch dependency: data workers can use the same
contract as serving without importing the model runtime. Output IDs are evidence;
preserve them for supervised targets and deterministic tape verification instead
of replacing them with an encode(decode(ids)) round trip.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

LING_REPOSITORY = "inclusionAI/Ling-3.0-tiny"
LING_REVISION = "9a98e35fe1c9ee255f78dd64771c7ae15a799481"
LING_TOKENIZER_SHA256 = "40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60"
LING_VOCAB_SIZE = 157_184
CODEC_VERSION = "raw-byte-bpe-v2;binary=single-byte-v1;special-merges=masked"


def _byte_alphabet() -> dict[str, int]:
    """The reversible GPT-2/ByteLevel alphabet, independently of UTF-8 decoding."""
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values.copy()
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(values, chars, strict=True)}


class ByteCodec:
    """A vocabulary with immutable exact byte pieces and deterministic encoding.

    ``pieces`` includes ``None`` for structural/special/padded rows which may not
    occur at a free lexical position. Every byte must have a one-byte fallback.
    A raw byte payload can always be represented, including invalid UTF-8 and NUL.
    """

    def __init__(
        self,
        pieces: Sequence[bytes | None],
        *,
        merge_ranks: dict[tuple[bytes, bytes], int] | None = None,
        split_pattern: str | None = None,
        fingerprint: str | None = None,
        tokenizer_document: dict | None = None,
    ) -> None:
        self._pieces = tuple(pieces)
        self._ids: dict[bytes, int] = {}
        for idx, piece in enumerate(self._pieces):
            if piece:
                self._ids.setdefault(piece, idx)
        missing = [b for b in range(256) if bytes([b]) not in self._ids]
        if missing:
            raise ValueError(f"codec lacks single-byte fallbacks: {missing[:8]}")
        self._byte_ids = tuple(self._ids[bytes([b])] for b in range(256))
        # An added special token may reuse a base BPE row. Such a row must
        # never be created as a merge result or used as an intermediate piece.
        self._merges = {
            pair: rank
            for pair, rank in (merge_ranks or {}).items()
            if pair[0] in self._ids and pair[1] in self._ids and pair[0] + pair[1] in self._ids
        }
        self._split_pattern = split_pattern
        self._fast_tokenizer = None
        if tokenizer_document is not None:
            from tokenizers import Tokenizer  # required: the pure-Python path is not equivalent

            # NFC changes bytes. Added-token interpretation also changes the
            # output alphabet. Raw FMT text has neither operation.
            document = dict(tokenizer_document)
            document.update(
                normalizer=None,
                added_tokens=[],
                post_processor=None,
                truncation=None,
                padding=None,
            )
            model = dict(document["model"])
            alphabet = _byte_alphabet()

            def legal_merge(merge):
                left, right = merge.split(" ") if isinstance(merge, str) else merge
                pair = bytes(alphabet[c] for c in left), bytes(alphabet[c] for c in right)
                return pair in self._merges

            model["merges"] = [merge for merge in model.get("merges", []) if legal_merge(merge)]
            # Direct multi-byte vocabulary lookup could bypass a removed
            # merge and return the masked protocol row again.
            model["ignore_merges"] = False
            document["model"] = model
            self._fast_tokenizer = Tokenizer.from_str(json.dumps(document))
        if fingerprint is None:
            digest = hashlib.sha256(CODEC_VERSION.encode())
            for piece in self._pieces:
                digest.update(
                    (-1 if piece is None else len(piece)).to_bytes(4, "little", signed=True)
                )
                if piece is not None:
                    digest.update(piece)
            for (left, right), rank in sorted(self._merges.items(), key=lambda item: item[1]):
                digest.update(rank.to_bytes(8, "little"))
                digest.update(len(left).to_bytes(4, "little") + left)
                digest.update(len(right).to_bytes(4, "little") + right)
            digest.update((split_pattern or "").encode())
            fingerprint = digest.hexdigest()
        self._fingerprint = fingerprint

    @classmethod
    def byte_fallback(cls) -> ByteCodec:
        """Offline bootstrap vocabulary: token ID equals its byte value."""
        return cls(tuple(bytes([b]) for b in range(256)))

    @classmethod
    def from_tokenizer_json(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_vocab_size: int = LING_VOCAB_SIZE,
    ) -> ByteCodec:
        raw = Path(path).read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ValueError(f"tokenizer SHA256 mismatch: got {sha256}")
        document = json.loads(raw)
        model = document["model"]
        if model["type"] != "BPE" or document.get("decoder", {}).get("type") != "ByteLevel":
            raise ValueError("raw codec supports a ByteLevel BPE mapping only")
        if (
            model.get("dropout")
            or model.get("continuing_subword_prefix")
            or model.get("end_of_word_suffix")
        ):
            raise ValueError("stochastic or affixed BPE is not a byte-exact raw codec")
        alphabet = _byte_alphabet()

        def unpack(text: str) -> bytes:
            try:
                return bytes(alphabet[char] for char in text)
            except KeyError as exc:
                raise ValueError("base BPE vocabulary contains a non-byte symbol") from exc

        vocab = model["vocab"]
        highest = max(
            [*vocab.values(), *[token["id"] for token in document.get("added_tokens", [])]]
        )
        if highest >= expected_vocab_size:
            raise ValueError("tokenizer rows exceed the configured output vocabulary")
        pieces: list[bytes | None] = [None] * expected_vocab_size
        for spelling, idx in vocab.items():
            pieces[idx] = unpack(spelling)
        for token in document.get("added_tokens", []):
            # Keep protocol tokens out of lexical distributions, including added
            # entries that reuse a row in the underlying BPE mapping.
            if token.get("special", False):
                pieces[token["id"]] = None
        merge_ranks = {}
        for rank, merge in enumerate(model.get("merges", [])):
            left, right = merge.split(" ") if isinstance(merge, str) else merge
            merge_ranks[(unpack(left), unpack(right))] = rank
        pre = document.get("pre_tokenizer", {})
        pretokenizers = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
        patterns = [
            step["pattern"]["Regex"]
            for step in pretokenizers
            if step.get("type") == "Split" and "Regex" in step.get("pattern", {})
        ]
        if len(patterns) > 1:
            raise ValueError("multiple raw BPE split patterns are not supported")
        byte_steps = [step for step in pretokenizers if step.get("type") == "ByteLevel"]
        if len(byte_steps) != 1 or byte_steps[0].get("add_prefix_space", False):
            raise ValueError("raw codec requires one ByteLevel step without prefix insertion")
        supported = {"Split", "ByteLevel"}
        if any(step.get("type") not in supported for step in pretokenizers):
            raise ValueError("unsupported pre-tokenizer could alter raw byte boundaries")
        if not patterns and byte_steps[0].get("use_regex", True):
            # Canonical ByteLevel splitting, retained without normalization.
            patterns = [
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            ]
        fingerprint = hashlib.sha256(
            (CODEC_VERSION + ";asset=" + sha256 + ";rows=" + str(expected_vocab_size)).encode()
        ).hexdigest()
        return cls(
            pieces,
            merge_ranks=merge_ranks,
            split_pattern=patterns[0] if patterns else None,
            fingerprint=fingerprint,
            tokenizer_document=document,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def vocab_size(self) -> int:
        return len(self._pieces)

    @property
    def valid_token_ids(self) -> tuple[int, ...]:
        return tuple(idx for idx, piece in enumerate(self._pieces) if piece)

    def token_bytes(self, token_id: int) -> bytes:
        if not 0 <= int(token_id) < len(self._pieces):
            raise ValueError(f"token ID {token_id} is outside the vocabulary")
        piece = self._pieces[int(token_id)]
        if not piece:
            raise ValueError(f"token ID {token_id} is not a lexical byte token")
        return piece

    def decode(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.token_bytes(idx) for idx in ids)

    def encode(self, payload: bytes | str) -> list[int]:
        raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        if not raw:
            return []
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            # Versioned explicit convention: never apply Unicode replacement or
            # run regex splitting on a made-up decoding of a binary payload.
            return [self._byte_ids[byte] for byte in raw]
        if self._fast_tokenizer is not None:
            ids = self._fast_tokenizer.encode(text, add_special_tokens=False).ids
            if self.decode(ids) != raw:
                raise ValueError("raw tokenizer unexpectedly changed input bytes")
            return ids
        if not self._merges:
            return [self._byte_ids[byte] for byte in raw]
        if self._split_pattern:
            try:
                import regex
            except ImportError as exc:
                raise RuntimeError(
                    "raw Ling metering needs tokenizers or regex; install megatoken[data]"
                ) from exc
            chunks = [
                match.group().encode("utf-8") for match in regex.finditer(self._split_pattern, text)
            ]
            if b"".join(chunks) != raw:
                raise ValueError("pre-tokenizer pattern did not cover every input byte")
        else:
            chunks = [raw]
        ids: list[int] = []
        for chunk in chunks:
            pieces = [bytes([value]) for value in chunk]
            while len(pieces) > 1:
                candidates = [
                    (self._merges.get((pieces[i], pieces[i + 1]), 1 << 60), i)
                    for i in range(len(pieces) - 1)
                ]
                rank, pos = min(candidates)
                if rank == 1 << 60:
                    break
                pieces[pos : pos + 2] = [pieces[pos] + pieces[pos + 1]]
            ids.extend(self._ids[piece] for piece in pieces)
        return ids

    def allowed_ids(self, kind: str = "bytes") -> tuple[int, ...]:
        """Masks are token-local; complete UTF-8 validation remains a draft gate."""
        if kind == "bytes" or kind == "utf8":
            return self.valid_token_ids
        if kind == "identifier":
            permitted = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-")
        elif kind == "ascii":
            permitted = set(range(32, 127)) | {9, 10, 13}
        else:
            raise ValueError(f"unknown lexical alphabet: {kind}")
        return tuple(
            i
            for i, piece in enumerate(self._pieces)
            if piece and all(b in permitted for b in piece)
        )
