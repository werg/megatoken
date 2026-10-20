"""Explicit frozen/pretrained component registry; no silent codec substitution."""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class PackSpec:
    name: str
    role: str
    repository: str
    revision: str
    weight_sha256: str
    codec: str = ""
    codec_revision: str = ""
    feature_width: int = 0
    sample_rate: int = 0
    fps: str = ""
    conditioning: str = "raw_facets"
    def __post_init__(self):
        if not self.name or not self.repository or not self.revision or len(self.weight_sha256) != 64:
            raise ValueError("Pack needs pinned revision and SHA256")
        if any(ch not in "0123456789abcdef" for ch in self.weight_sha256):
            raise ValueError("Invalid pack weight digest")
        if self.role not in ("encoder", "generator", "codec", "decoder", "teacher"):
            raise ValueError("Invalid component role")
        if self.role == "generator" and (not self.codec or not self.codec_revision):
            raise ValueError("Generator must declare matching codec identity")
        if self.conditioning != "raw_facets":
            raise ValueError("Native pack cannot bypass raw facets")
    @property
    def identity(self):
        return sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

class PackRegistry:
    def __init__(self, specs=()):
        self._specs = {}
        for spec in specs:
            self.register(spec)
    def register(self, spec: PackSpec):
        if spec.name in self._specs:
            raise ValueError("Duplicate pack name")
        self._specs[spec.name] = spec
    def get(self, name):
        return self._specs[name]
    def verify_pair(self, generator: str, codec: str):
        gen, dec = self.get(generator), self.get(codec)
        if gen.role != "generator" or dec.role not in ("codec", "decoder"):
            raise ValueError("Invalid generator/codec pair roles")
        if gen.codec != dec.name or gen.codec_revision != dec.revision:
            raise ValueError("Checkpoint incompatible with declared decoder")
        return True
    def manifest(self):
        return {key: spec.identity for key, spec in sorted(self._specs.items())}
