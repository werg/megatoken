"""Checksummed, source-coordinate sidecars keyed by stable episode/event IDs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from megatoken.memory.contracts import Footprint, QueryBoundary


@dataclass(frozen=True)
class RetrievalExample:
    episode_id: str
    event_id: str
    query: QueryBoundary
    query_teacher_inputs: Footprint
    document_key_context: Footprint
    optimization_supervision: Footprint
    payload_support: Footprint
    positive_content_ids: tuple[str, ...]
    support_weights: tuple[float, ...]
    annotations_complete: bool
    source_snapshot: str
    split_family: str
    teacher_target: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or self.event_id != self.query.event_id:
            raise ValueError("Stable event IDs must match query receipts")
        if len(self.positive_content_ids) != len(self.support_weights):
            raise ValueError("Annotation IDs and weights must correspond")
        if any(x < 0 for x in self.support_weights):
            raise ValueError("Support weights must be nonnegative")
        if self.positive_content_ids and sum(self.support_weights) <= 0:
            raise ValueError("Annotated support requires positive target mass")

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "query_teacher_inputs": self.query_teacher_inputs.to_json(),
            "document_key_context": self.document_key_context.to_json(),
            "optimization_supervision": self.optimization_supervision.to_json(),
            "payload_support": self.payload_support.to_json(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RetrievalExample:
        data = dict(data)
        data["query"] = QueryBoundary(**data["query"])
        for field in ("query_teacher_inputs", "document_key_context",
                      "optimization_supervision", "payload_support"):
            data[field] = Footprint.from_json(data[field])
        for field in ("positive_content_ids", "support_weights"):
            data[field] = tuple(data[field])
        if data["teacher_target"] is not None:
            data["teacher_target"] = tuple(data["teacher_target"])
        return cls(**data)


def write_sidecar(path: str | Path, entries: list[RetrievalExample]) -> dict:
    """Write a deterministic portable JSONL sidecar and its content-addressed manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identifiers = set()
    lines = []
    for entry in entries:
        identity = (entry.episode_id, entry.event_id)
        if identity in identifiers:
            raise ValueError("Duplicate retrieval event within a sidecar")
        identifiers.add(identity)
        lines.append(json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")))
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    path.write_bytes(payload)
    result = {"format": "megatoken-retrieval-v1", "entries": len(entries),
              "sha256": sha256(payload).hexdigest()}
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(result, sort_keys=True) + "\n"
    )
    return result


def read_sidecar(path: str | Path) -> list[RetrievalExample]:
    path = Path(path)
    data = path.read_bytes()
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    if manifest.get("format") != "megatoken-retrieval-v1" or (
        sha256(data).hexdigest() != manifest.get("sha256")
    ):
        raise ValueError("Retrieval manifest format/checksum mismatch")
    entries = [RetrievalExample.from_dict(json.loads(line)) for line in data.splitlines()]
    if len(entries) != manifest["entries"]:
        raise ValueError("Retrieval sidecar count mismatch")
    return entries
