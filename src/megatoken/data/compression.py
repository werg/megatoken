"""Compressed reading: long reference text enters as EmbeddingGemma2 hidden states.

A reference document or a large tool result that would otherwise occupy many
byte-level chunks of context is read instead as the teacher's per-token hidden
states: one event per piece of up to 8192 teacher tokens, about 16x fewer
events than byte chunks of ~512 tokens. Such events are input-only; the model
never generates them. Preparation runs the teacher once and stores the
features as immutable assets; the episode carries a canonical descriptor
(``MediaRecord`` of modality ``text``) in place of the text bytes.

The prompt is the teacher's fixed document prompt, so a document always maps
to the same features and can be cached. Question-aware prompts are a later,
measured upgrade.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from .schema import PhysicalRecord, Provenance, Role, SourceSpan, split_payload, stable_id

DOCUMENT_PROMPT = "title: none | text: "


@dataclass(frozen=True, slots=True)
class CompressionPolicy:
    """Which logical fields are read compressed, and how they are split."""

    min_bytes: int = 8192
    roles: frozenset[str] = field(default_factory=lambda: frozenset({"tool", "document"}))
    # ~3-4 bytes per teacher token keeps a piece under 8192 tokens; a piece the
    # teacher still refuses is halved.
    piece_bytes: int = 24576
    prompt: str = DOCUMENT_PROMPT

    def __post_init__(self) -> None:
        if not 1 <= self.min_bytes or not 1024 <= self.piece_bytes:
            raise ValueError("Compression thresholds must be positive")
        if not self.roles <= {"tool", "document", "user", "system", "environment"}:
            raise ValueError("Only observed roles may be read compressed")


class DocumentCompressor:
    """Turns one long observed field into descriptor records backed by teacher features."""

    def __init__(self, teacher, store: TensorAssetStore, policy: CompressionPolicy | None = None):
        self.teacher, self.store = teacher, store
        self.policy = policy or CompressionPolicy()

    def applies(self, payload: bytes, *, role: str, score: bool) -> bool:
        return (
            not score
            and role in self.policy.roles
            and len(payload) >= self.policy.min_bytes
            and _is_text(payload)
        )

    def descriptors(self, payload: bytes) -> list[tuple[int, bytes, MediaRecord]]:
        """``(byte offset, piece, descriptor)`` covering every byte of ``payload``."""
        out = []
        pending = list(split_payload(payload, self.policy.piece_bytes))
        while pending:
            offset, piece = pending.pop(0)
            try:
                observation = self.teacher.encode(
                    "text", piece.decode("utf-8"), prompt=self.policy.prompt
                )
            except ValueError as error:
                if "8192" not in str(error) or len(piece) < 2:
                    raise
                halves = list(split_payload(piece, max(1, len(piece) // 2)))
                pending[:0] = [(offset + start, part) for start, part in halves]
                continue
            source = self.store.put(
                observation.features.numpy(),
                modality="text",
                kind="features",
                extractor=observation.extractor,
                source_family=hashlib.sha256(piece).hexdigest(),
            )
            out.append((offset, piece, MediaRecord("text", source, prompt=self.policy.prompt)))
        return out

    def records(
        self,
        payload: bytes,
        *,
        logical_id: str,
        role: Role,
        turn_index: int = 0,
        provenance: Provenance = "unknown",
        source_uri: str = "",
        source_field: str = "",
        transform: str = "identity",
        result_of: str | None = None,
        execution_id: str | None = None,
    ) -> tuple[PhysicalRecord, ...]:
        pieces = self.descriptors(payload)
        return tuple(
            PhysicalRecord(
                record_id=stable_id("compressed-v1", logical_id, str(index), piece),
                role=role,
                payload=descriptor.to_bytes(),
                turn_index=turn_index,
                logical_id=logical_id,
                part_index=index,
                part_count=len(pieces),
                provenance=provenance,
                source_span=SourceSpan(
                    source_uri, source_field, offset, offset + len(piece), transform
                ),
                result_of=result_of,
                execution_id=execution_id,
                media_type=MEDIA_RECORD_TYPE,
            )
            for index, (offset, piece, descriptor) in enumerate(pieces)
        )


def _is_text(payload: bytes) -> bool:
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
