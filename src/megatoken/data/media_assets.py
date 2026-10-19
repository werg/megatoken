"""Pinned, checksummed tensor assets for multimodal meridian records.

Model-independent storage. Data readers never download, decode arbitrary media
formats, or execute a pretrained model. Every media record is a canonical JSON
reference into immutable little-endian FP32 arrays under the prepared root.
External preprocessing is explicitly identified and frozen before training.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from megatoken.io import atomic_binary_writer, canonical_json, strict_json_loads

MEDIA_RECORD_TYPE = "application/vnd.megatoken.media-reference+json"
MEDIA_RECORD_VERSION = 1
_MAX_ELEMENTS = 16_000_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
Modal = Literal["audio", "image", "video"]


def _sha(value: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("Media assets require a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class TensorAsset:
    """One physically stored array, excluding an implicit batch dimension."""

    modality: Modal
    kind: Literal["raw", "features"]
    sha256: str
    shape: tuple[int, ...]
    extractor: str = ""
    sample_rate: int = 0
    fps_num: int = 0
    fps_den: int = 1

    def __post_init__(self) -> None:
        _sha(self.sha256)
        if self.modality not in ("audio", "image", "video"):
            raise ValueError("Unsupported media modality")
        if self.kind not in ("raw", "features"):
            raise ValueError("Media asset kind must be raw or features")
        if (
            not isinstance(self.shape, tuple)
            or any(type(n) is not int or n < 1 for n in self.shape)
            or not self.shape
            or math.prod(self.shape) > _MAX_ELEMENTS
        ):
            raise ValueError("Invalid or excessive media tensor shape")
        if self.kind == "features":
            if len(self.shape) != 2 or not self.extractor:
                raise ValueError("Cached features require tokens, width, and pinned extractor")
        else:
            target_ranks = {"audio": 1, "image": 3, "video": 4}
            if len(self.shape) != target_ranks[self.modality]:
                raise ValueError("Native media asset has the wrong tensor rank")
            if self.modality == "image" and self.shape[0] != 3:
                raise ValueError("Images must contain RGB channels")
            if self.modality == "video" and self.shape[1] != 3:
                raise ValueError("Videos must contain RGB channels")
            if self.extractor:
                raise ValueError("Native raw media cannot claim an external feature extractor")
        if (
            type(self.sample_rate) is not int or self.sample_rate < 0
            or type(self.fps_num) is not int or self.fps_num < 0
            or type(self.fps_den) is not int or self.fps_den < 1
        ):
            raise ValueError("Invalid sample-rate/fps metadata")
        if self.modality == "audio" and self.sample_rate != 16000:
            raise ValueError("Audio assets must use the pinned 16 kHz convention")
        if self.modality != "audio" and self.sample_rate:
            raise ValueError("Non-audio media must not have a sample rate")
        if self.modality == "video" and not self.fps_num:
            raise ValueError("Video tensor assets require a positive rational fps")
        if self.modality != "video" and self.fps_num:
            raise ValueError("Only video assets carry a frame-rate")

    @property
    def relative_path(self) -> Path:
        return Path("media") / f"{self.sha256}.f32le"

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> TensorAsset:
        fields = {name for name in cls.__dataclass_fields__}
        if set(mapping) - fields:
            raise ValueError("Unknown tensor asset metadata fields")
        value = dict(mapping)
        value["shape"] = tuple(value["shape"])
        return cls(**value)


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """Text-independent, auditable local input and optional output target."""

    modality: Modal
    source: TensorAsset
    target: TensorAsset | None = None
    version: int = MEDIA_RECORD_VERSION

    def __post_init__(self) -> None:
        if self.version != MEDIA_RECORD_VERSION:
            raise ValueError("Unsupported media reference version")
        if self.modality != self.source.modality or (
            self.target is not None and self.target.modality != self.modality
        ):
            raise ValueError("Media reference modality does not match source/target")
        if self.target is not None and self.target.kind != "raw":
            raise ValueError("Realization targets must contain native media arrays")

    def to_bytes(self) -> bytes:
        return canonical_json(asdict(self))

    @classmethod
    def from_bytes(cls, payload: bytes) -> MediaRecord:
        if not isinstance(payload, bytes) or len(payload) > 4096:
            raise ValueError("Media descriptor must be bounded canonical JSON bytes")
        value = strict_json_loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "modality", "source", "target", "version"
        }:
            raise ValueError("Media record has incomplete or unknown fields")
        if canonical_json(value) != payload:
            raise ValueError("Media descriptor must have canonical serialization")
        if not isinstance(value["source"], dict) or (
            value["target"] is not None and not isinstance(value["target"], dict)
        ):
            raise ValueError("Media tensor descriptors must be objects")
        return cls(
            modality=value["modality"],
            source=TensorAsset.from_dict(value["source"]),
            target=(
                None if value["target"] is None
                else TensorAsset.from_dict(value["target"])
            ),
            version=value["version"],
        )


class TensorAssetStore:
    """Immutable local asset store rooted at the prepared dataset directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(
        self,
        value: np.ndarray,
        *,
        modality: Modal,
        kind: Literal["raw", "features"] = "raw",
        extractor: str = "",
        sample_rate: int = 0,
        fps_num: int = 0,
        fps_den: int = 1,
    ) -> TensorAsset:
        values = np.asarray(value, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("Media tensor contains nonfinite values")
        values = np.ascontiguousarray(values)
        data = values.tobytes(order="C")
        digest = hashlib.sha256(data).hexdigest()
        ref = TensorAsset(
            modality, kind, digest, tuple(values.shape), extractor,
            sample_rate, fps_num, fps_den,
        )
        path = self.root / ref.relative_path
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing immutable media asset has changed")
            return ref
        with atomic_binary_writer(path) as output:
            output.write(data)
        return ref

    def load(self, asset: TensorAsset) -> np.ndarray:
        """Fail closed on stale or substituted files and invalid tensor contents."""
        path = self.root / asset.relative_path
        # Exactly the digest-derived relative path is used; never a URL or
        # untrusted input path embedded in the record.
        data = path.read_bytes()
        if len(data) != math.prod(asset.shape) * 4:
            raise ValueError("Media asset does not match its declared FP32 shape")
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError("Media asset SHA256 differs from its declared identity")
        result = np.frombuffer(data, dtype="<f4").copy().reshape(asset.shape)
        if not np.isfinite(result).all():
            raise ValueError("Media tensor is nonfinite")
        return result
