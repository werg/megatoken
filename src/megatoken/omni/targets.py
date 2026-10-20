"""Codec-latent media asset manifest and immutable predecoder training targets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CodecLatentTarget:
    modality: str
    codec_name: str
    codec_revision: str
    codec_digest: str
    generator_name: str
    source_sha256: str
    latent_sha256: str
    shape: tuple[int, ...]
    valid_shape: tuple[int, ...]
    clock_rate_num: int = 0
    clock_rate_den: int = 1
    latent_scale: float = 1.
    latent_shift: float = 0.
    convention: str = "continuous"
    def __post_init__(self):
        if self.modality not in ("audio", "image", "video") or not self.codec_name or not self.codec_revision:
            raise ValueError("Unknown latent codec contract")
        if any(len(d) != 64 or any(c not in "0123456789abcdef" for c in d) for d in
               (self.codec_digest, self.source_sha256, self.latent_sha256)):
            raise ValueError("Codec/source/latent weight hashes must be pinned")
        if not self.shape or len(self.shape) != len(self.valid_shape) or any(
                not 0 < v <= s for v, s in zip(self.valid_shape, self.shape, strict=True)):
            raise ValueError("Invalid valid/padded codec latent geometry")
        if self.clock_rate_num < 0 or self.clock_rate_den < 1 or self.latent_scale <= 0:
            raise ValueError("Invalid latent normalization or sample clock")
        if self.convention not in ("continuous", "velocity", "noise", "flow"):
            raise ValueError("Undeclared codec latent convention")
    @property
    def identity(self):
        from dataclasses import asdict
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

def store_latents(root, values, *, expected_sha=None):
    """Write only immutable floating-point latent arrays with atomic replace."""
    import os
    import tempfile
    data = np.asarray(values, dtype="<f4")
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Latent tensor must be finite and nonempty")
    blob = data.tobytes(order="C")
    digest = sha256(blob).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Codec target data digest mismatch")
    path = Path(root)/"codec_latents"/(digest+".f32le")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Immutable latent target hash collision")
        return digest
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest

def load_latents(root, spec: CodecLatentTarget):
    path = Path(root)/"codec_latents"/(spec.latent_sha256+".f32le")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != spec.latent_sha256:
        raise ValueError("Corrupted pinned codec target")
    if len(raw) != 4 * int(np.prod(spec.shape)):
        raise ValueError("Codec geometry and bytes disagree")
    return np.frombuffer(raw, dtype="<f4").reshape(spec.shape).copy()
