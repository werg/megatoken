"""Versioned edit trajectories and atomic raw replay identity."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

@dataclass(frozen=True)
class ArtifactVersion:
    version: str
    parent: str
    content_sha256: str
    source_family: str
    request: str = ""
    preserve: tuple[str, ...] = ()
    provenance: str = "recorded"
    def __post_init__(self):
        if not self.version or not self.source_family or len(self.content_sha256) != 64:
            raise ValueError("Artifact needs identity, family and hash")
        if self.provenance not in ("recorded","student","teacher"):
            raise ValueError("Unknown artifact provenance")

class EditTrajectory:
    def __init__(self, initial: ArtifactVersion):
        if initial.parent:
            raise ValueError("Initial version cannot depend on a future parent")
        self.versions = [initial]
    def append(self, version: ArtifactVersion):
        if version.parent != self.versions[-1].version:
            raise ValueError("Edited output must descend from actual previous artifact")
        if version.version in {v.version for v in self.versions}:
            raise ValueError("Duplicate edit version")
        if version.source_family != self.versions[0].source_family:
            raise ValueError("Cross-family edit contamination")
        self.versions.append(version)
    def replay_identity(self):
        return sha256(json.dumps([asdict(v) for v in self.versions],sort_keys=True).encode()).hexdigest()

def event_seed(session_id, meridian_id, track_id, revision, renderer_id):
    raw=json.dumps((session_id,meridian_id,track_id,revision,renderer_id),
                   separators=(",",":")).encode()
    return int.from_bytes(sha256(raw).digest()[:8],"big")
