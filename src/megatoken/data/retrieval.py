"""Checksummed, source-coordinate sidecars keyed by stable episode/event IDs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import binascii
from hashlib import sha256
import json
import math
from pathlib import Path

from megatoken.memory.contracts import Footprint, QueryBoundary, Visibility


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
    main_request_b64: str | None = None
    pair_context_complete: bool = False
    pair_support_units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id or self.event_id != self.query.event_id:
            raise ValueError("Stable event IDs must match query receipts")
        if not self.source_snapshot or not self.split_family:
            raise ValueError("Retrieval supervision requires a pinned source snapshot/split")
        # Future labels may enter optimization supervision, but neither the
        # teacher's causal query nor a fetched source's complete contextual
        # key/support footprint may depend on unavailable evidence.
        scope = self.query_teacher_inputs.ranges[0].scope
        all_footprints = (
            self.query_teacher_inputs, self.document_key_context,
            self.payload_support, self.optimization_supervision,
        )
        if any(r.scope != scope for fp in all_footprints for r in fp.ranges):
            raise ValueError("Query, evidence and label footprints must share one scope")
        if not Visibility(scope, self.query.meridian).permits(self.query_teacher_inputs):
            raise ValueError("Query teacher sees future or invalid prefix evidence")
        eligibility = Visibility(scope, self.query.deadline)
        if not eligibility.permits(self.document_key_context) or not (
            eligibility.permits(self.payload_support)
        ):
            raise ValueError("Retrieved source context/support is not causally eligible")
        if self.main_request_b64 is not None:
            self.request_content()
        if self.pair_support_units:
            causal_sources = {
                span.source_id for span in self.query_teacher_inputs.ranges
            }
            if len(set(self.pair_support_units)) != len(self.pair_support_units) or (
                not set(self.pair_support_units).issubset(causal_sources)
            ):
                raise ValueError("Pair support units must be distinct causal prefix sources")
        if len(self.positive_content_ids) != len(self.support_weights):
            raise ValueError("Annotation IDs and weights must correspond")
        if len(set(self.positive_content_ids)) != len(self.positive_content_ids):
            raise ValueError("Duplicate useful-source annotation IDs are ambiguous")
        if any(not math.isfinite(x) or x < 0 for x in self.support_weights):
            raise ValueError("Support weights must be nonnegative")
        if self.positive_content_ids and sum(self.support_weights) <= 0:
            raise ValueError("Annotated support requires positive target mass")
        if self.teacher_target is not None:
            if not self.teacher_target or any(
                not math.isfinite(x) for x in self.teacher_target
            ) or not 1e-12 < math.sqrt(sum(x * x for x in self.teacher_target)) < float("inf"):
                raise ValueError("Teacher query vector must be finite and nonzero")

    def request_content(self) -> bytes:
        """Bytes of the explicit, causally visible main request; never teacher input."""
        if self.main_request_b64 is None:
            raise ValueError("Retrieval query training requires pinned main-request bytes")
        try:
            encoded = base64.b64decode(self.main_request_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 main-request payload") from exc
        if not encoded or sha256(encoded).hexdigest() != self.query.request_hash:
            raise ValueError("Main request does not match the causal request hash")
        return encoded

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
        for field in ("positive_content_ids", "support_weights", "pair_support_units"):
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
