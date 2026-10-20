"""Pinned original-source document embedding labels, never a separate vector ingester.

One data-prep row names a real native observed source event and the complete
teacher input footprint. The trained document R consumes E's raw facets from
that same actual original source event, not teacher vectors as P inputs.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from megatoken.memory.contracts import Footprint, SourceRange, Visibility


@dataclass(frozen=True)
class DocumentEmbeddingExample:
    episode_id: str
    event_id: str
    payload_sha256: str
    teacher_inputs: Footprint
    source_span: SourceRange
    teacher_target: tuple[float, ...]
    preprocessing_revision: str

    def __post_init__(self) -> None:
        if not self.episode_id or not self.event_id or not self.preprocessing_revision:
            raise ValueError("Document teacher must name a real prepared original event")
        if len(self.payload_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.payload_sha256
        ):
            raise ValueError("Document labels require exact SHA256 of source payload bytes")
        if self.source_span not in self.teacher_inputs.ranges:
            raise ValueError("Model source span must be part of complete teacher provenance")
        if not self.teacher_target or any(
            not math.isfinite(value) for value in self.teacher_target
        ) or sum(v * v for v in self.teacher_target) <= 1e-12:
            raise ValueError("Document teacher must supply a finite nonzero target vector")
        if len({r.scope for r in self.teacher_inputs.ranges}) != 1:
            raise ValueError("A document key may not mix causal source scopes")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["teacher_inputs"] = self.teacher_inputs.to_json()
        return result

    @classmethod
    def from_dict(cls, value: dict) -> DocumentEmbeddingExample:
        source = dict(value["source_span"])
        source["ancestors"] = tuple(source.get("ancestors", ()))
        return cls(
            episode_id=value["episode_id"], event_id=value["event_id"],
            payload_sha256=value["payload_sha256"],
            teacher_inputs=Footprint.from_json(value["teacher_inputs"]),
            source_span=SourceRange(**source),
            teacher_target=tuple(value["teacher_target"]),
            preprocessing_revision=value["preprocessing_revision"],
        )

    def validate_event(self, *, payload: bytes, clock: int) -> None:
        if sha256(payload).hexdigest() != self.payload_sha256:
            raise ValueError("Original prepared source event differs from document teacher input")
        if not Visibility(self.source_span.scope, clock).permits(self.teacher_inputs):
            raise ValueError("Document teacher consumed unavailable suffix or excluded source")
        if len(payload) != self.source_span.end - self.source_span.start:
            raise ValueError("Document byte offsets must identify exactly the source event")


def write_document_sidecar(path: str | Path,
                           examples: list[DocumentEmbeddingExample]) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identities = set()
    rows = []
    for entry in examples:
        key = (entry.episode_id, entry.event_id)
        if key in identities:
            raise ValueError("One original source event has duplicate document teacher labels")
        identities.add(key)
        rows.append(json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")))
    payload = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
    path.write_bytes(payload)
    manifest = {
        "format": "megatoken-document-embedding-v1",
        "entries": len(rows),
        "sha256": sha256(payload).hexdigest(),
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    return manifest


def read_document_sidecar(path: str | Path) -> list[DocumentEmbeddingExample]:
    path = Path(path)
    payload = path.read_bytes()
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    if manifest.get("format") != "megatoken-document-embedding-v1" or (
        manifest.get("sha256") != sha256(payload).hexdigest()
    ):
        raise ValueError("Document embedding sidecar checksum/format mismatch")
    result = [
        DocumentEmbeddingExample.from_dict(json.loads(line))
        for line in payload.splitlines()
    ]
    if len(result) != manifest.get("entries") or len({
        (x.episode_id, x.event_id) for x in result
    }) != len(result):
        raise ValueError("Incomplete or duplicated document embedding labels")
    return result
