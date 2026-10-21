"""Checksum-verified coarse text/document input to the unified EmbeddingGemma2 E.

These are *observations*, not lexical target bytes. One validated teacher
feature sequence becomes one canonical 32-facet event. Exact quotation/editing
uses the normal byte route instead of this lossy compression.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from megatoken.io import canonical_json, strict_json_loads

TEXT_HIDDEN_RECORD_TYPE = "application/vnd.megatoken.embeddinggemma2-hidden+json"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EXTRACTOR = re.compile(
    r"embeddinggemma2:[0-9a-f]{40}:language_model\.embedding_projection:input:"
    r"chat-template-v1\Z"
)


@dataclass(frozen=True, slots=True)
class TextHiddenReference:
    """A canonical, path-free CAS reference to one original-source teacher view."""

    modality: str
    extractor: str
    feature_sha256: str
    source_sha256: str
    teacher_input_sha256: str
    tokens: int
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or self.modality not in {"document", "text_summary"}:
            raise ValueError("Unknown compressed text/document input reference")
        if not _EXTRACTOR.fullmatch(self.extractor):
            raise ValueError("Document input needs the immutable 512d EmbeddingGemma2 tap")
        if any(
            not isinstance(value, str) or not _SHA.fullmatch(value)
            for value in (
                self.feature_sha256,
                self.source_sha256,
                self.teacher_input_sha256,
            )
        ):
            raise ValueError("Compressed text asset hashes must be original SHA256 digests")
        if type(self.tokens) is not int or not 1 <= self.tokens <= 8192:
            raise ValueError("Document teacher token budget must be within 8192")

    @classmethod
    def from_cache_descriptor(cls, descriptor: dict) -> TextHiddenReference:
        """Use only the *original* local embeddinggemma2 cache descriptor."""
        if not isinstance(descriptor, dict):
            raise ValueError("Teacher descriptor must be an object")
        source_kind = descriptor.get("modality")
        if (
            descriptor.get("format") != "megatoken.embeddinggemma2.hidden.v1"
            or source_kind not in {"text", "document"}
            or descriptor.get("hidden_tap") != "language_model.embedding_projection:input"
            or descriptor.get("preprocessing_id") != "chat-template-v1"
        ):
            raise ValueError("Not a pinned EmbeddingGemma2 text/document hidden cache")
        asset = descriptor.get("asset")
        if not isinstance(asset, dict):
            raise ValueError("Compressed input is missing its teacher feature asset")
        sha = asset.get("sha256")
        shape = asset.get("shape")
        extractor = asset.get("extractor")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or type(shape[0]) is not int
            or shape[1] != 512
            or descriptor.get("valid_tokens") != shape[0]
            or asset.get("kind") != "features"
            or not isinstance(sha, str)
            or not _SHA.fullmatch(sha)
            or asset.get("path") != f"teacher/{sha}.f32le"
            or extractor
            != (
                f"embeddinggemma2:{descriptor.get('teacher_revision', '')}:"
                "language_model.embedding_projection:input:chat-template-v1"
            )
        ):
            raise ValueError("Teacher features must be unpooled, ordered, local, and pinned")
        return cls(
            modality="document" if source_kind == "document" else "text_summary",
            extractor=extractor,
            feature_sha256=sha,
            source_sha256=descriptor.get("source_sha256", ""),
            teacher_input_sha256=descriptor.get("teacher_input_sha256", ""),
            tokens=shape[0],
        )

    def to_bytes(self) -> bytes:
        return canonical_json(asdict(self))

    @classmethod
    def from_bytes(cls, payload: bytes) -> TextHiddenReference:
        if not isinstance(payload, bytes) or len(payload) > 2048:
            raise ValueError("Compressed document record must be bounded canonical JSON")
        value = strict_json_loads(payload)
        if (
            not isinstance(value, dict)
            or set(value) != set(cls.__dataclass_fields__)
            or canonical_json(value) != payload
        ):
            raise ValueError("Compressed document input requires the canonical reference form")
        return cls(**value)

    def load(self, root: str | Path) -> np.ndarray:
        """Hash original teacher bytes on every read; no arbitrary file paths."""
        base = Path(root).resolve()
        path = base / "teacher" / f"{self.feature_sha256}.f32le"
        if path.is_symlink() or not path.resolve().is_relative_to(base) or not path.is_file():
            raise ValueError("Teacher document cache is missing or outside the asset root")
        raw = path.read_bytes()
        if (
            len(raw) != self.tokens * 512 * 4
            or hashlib.sha256(raw).hexdigest() != self.feature_sha256
        ):
            raise ValueError("Compressed document teacher cache checksum/width differs")
        values = np.frombuffer(raw, dtype="<f4").copy().reshape(self.tokens, 512)
        if not np.isfinite(values).all():
            raise ValueError("Compressed document teacher features must be finite")
        return values
