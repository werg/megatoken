"""Pinned, checksummed tensor assets for multimodal meridian records.

Model-independent storage. Data readers never download, decode media formats,
or execute a pretrained model. A media event is one canonical JSON record that
references immutable little-endian FP32 arrays under the prepared root:

- ``source``: the observed input, always the pinned EmbeddingGemma2 hidden
  states of the original media (``megatoken.teacher``);
- ``target``: optional supervision for realizing that media, always latents of
  the resident pretrained realizer's own codec (Pocket/Mimi, Sana DC-AE,
  Sana-Video VAE), plus the physical output geometry they decode to.

Both assets carry ``source_family`` = SHA256 of the original media file bytes,
written by ``megatoken.data.media_prepare`` from one decode of that file.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from megatoken.io import atomic_binary_writer, canonical_json, strict_json_loads
from megatoken.teacher import EMBEDDING_WIDTH, HIDDEN_WIDTH, MAX_TOKENS, validate_extractor

MEDIA_RECORD_TYPE = "application/vnd.megatoken.media-reference+json"
MEDIA_RECORD_VERSION = 6
_MAX_ELEMENTS = 16_000_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
Modal = Literal["text", "audio", "image", "video"]
# Realizable media: the modalities that may carry a codec-latent target.
MEDIA = ("audio", "image", "video")
# Event modality -> modality of the teacher features its descriptor holds. A
# ``compressed_text`` event is a long text (a reference document, a large tool
# result) read as EmbeddingGemma2 hidden states instead of byte by byte; it is
# input-only and never generated.
DESCRIPTOR_MODALITY = {
    "audio": "audio",
    "image": "image",
    "video": "video",
    "compressed_text": "text",
}
FEATURE_MODALITIES = tuple(DESCRIPTOR_MODALITY)
# Codec latent rank per modality: audio [frames, latent], image [C, h, w],
# video [frames, C, h, w] (time-major).
LATENT_RANK = {"audio": 2, "image": 3, "video": 4}


def _sha(value: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("Media assets require a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class TensorAsset:
    """One physically stored array, excluding an implicit batch dimension.

    ``extractor`` names what produced the array: the teacher identity for
    features and for the teacher's pooled embedding of the same input, the
    realizer pack identity (``<pack>:<checkpoint digest>``) for codec latents. ``valid_count`` > 0 marks only the leading frames of a
    padded latent sequence as real.
    """

    modality: Literal["text", "audio", "image", "video"]
    kind: Literal["features", "embedding", "codec_latents", "rendered"]
    sha256: str
    shape: tuple[int, ...]
    extractor: str
    source_family: str
    sample_rate: int = 0
    fps_num: int = 0
    fps_den: int = 1
    valid_count: int = 0
    normalization: str = ""

    def __post_init__(self) -> None:
        _sha(self.sha256)
        _sha(self.source_family)
        if self.modality not in ("text", *MEDIA):
            raise ValueError("Unsupported asset modality")
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(type(n) is not int or n < 1 for n in self.shape)
            or math.prod(self.shape) > _MAX_ELEMENTS
        ):
            raise ValueError("Invalid or excessive media tensor shape")
        if self.kind in ("features", "embedding"):
            validate_extractor(self.extractor)
            if self.kind == "embedding" and self.shape != (EMBEDDING_WIDTH,):
                raise ValueError("A teacher embedding is one 768-wide unit vector")
            if self.kind == "features" and (
                len(self.shape) != 2 or self.shape[1] != HIDDEN_WIDTH or self.shape[0] > MAX_TOKENS
            ):
                raise ValueError("Teacher features are [tokens <= 8192, 512]")
            # The teacher samples media by its own locked processor policy; its
            # token sequence carries no physical clock, padding or normalization.
            if (
                self.valid_count
                or self.normalization
                or self.sample_rate
                or self.fps_num
                or self.fps_den != 1
            ):
                raise ValueError("Teacher features are unpadded, unnormalized and clock-free")
        elif self.kind == "codec_latents":
            if (
                self.modality not in MEDIA
                or len(self.shape) != LATENT_RANK[self.modality]
                or ":" not in self.extractor
            ):
                raise ValueError("Codec latents need the realizer identity and native rank")
            if (self.modality == "audio") != (self.sample_rate == 24000):
                raise ValueError("Speech latents, and only they, are Pocket/Mimi 24 kHz")
            if (self.modality == "video") != (self.fps_num > 0) or self.fps_den < 1:
                raise ValueError("Video latents, and only they, carry a positive frame rate")
            if self.modality == "image" and self.valid_count:
                raise ValueError("Still-image latents have no frame axis to pad")
        elif self.kind == "rendered":
            if (
                self.modality not in MEDIA
                or ":" not in self.extractor
                or self.sha256 != self.source_family
            ):
                raise ValueError("Rendered media must bind its exact bytes and decoder identity")
        else:
            raise ValueError("Media assets require a supported extraction or rendering kind")
        if self.valid_count < 0 or self.valid_count > self.shape[0]:
            raise ValueError("Valid frames exceed the padded tensor")

    @property
    def relative_path(self) -> Path:
        return Path("media") / f"{self.sha256}.f32le"

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> TensorAsset:
        if set(mapping) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown tensor asset metadata fields")
        return cls(**{**mapping, "shape": tuple(mapping["shape"])})


@dataclass(frozen=True, slots=True)
class OutputGeometry:
    """Requested/predicted or teacher-only physical bounds, never semantic content."""

    modality: Modal
    sample_count: int = 0
    sample_rate: int = 0
    frames: int = 1
    height: int = 0
    width: int = 0
    fps_num: int = 0
    fps_den: int = 1
    provenance: Literal["requested", "predicted", "teacher_only"] = "teacher_only"

    def __post_init__(self) -> None:
        if self.modality == "audio":
            if not 1 <= self.sample_count <= 24000 * 120 or self.sample_rate != 24000:
                raise ValueError("Audio output geometry needs a 24 kHz sample count")
            if self.height or self.width or self.fps_num:
                raise ValueError("Audio output cannot contain visual geometry")
        elif self.modality in ("image", "video"):
            if not (
                8 <= self.height <= 2048 and 8 <= self.width <= 2048 and 1 <= self.frames <= 193
            ):
                raise ValueError("Visual output geometry is missing or excessive")
            if self.modality == "image" and (self.frames != 1 or self.fps_num):
                raise ValueError("Still image output cannot have a frame clock")
            if self.modality == "video" and self.fps_num <= 0:
                raise ValueError("Video output requires its physical frame rate")
            if self.sample_count or self.sample_rate:
                raise ValueError("Visual output cannot contain an audio sample clock")
        else:
            raise ValueError("Unknown output geometry modality")
        if self.fps_den <= 0:
            raise ValueError("Invalid output frame-rate denominator")
        if self.provenance not in ("requested", "predicted", "teacher_only"):
            raise ValueError("Undeclared output-geometry provenance")


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """One feature event: observed teacher features and, for media, an optional
    realization target. Text records (compressed reading) never have a target."""

    modality: Modal
    source: TensorAsset
    target: TensorAsset | None = None
    output_geometry: OutputGeometry | None = None
    # The teacher task prompt the source was encoded with (``title: ... | text: ``).
    prompt: str = ""
    # The teacher's pooled 768-d embedding of the same input, from the same
    # pass: the target for embedding distillation of this item's facets.
    embedding: TensorAsset = field(kw_only=True)
    rendered: TensorAsset | None = field(default=None, kw_only=True)
    transform_json: str = field(default="{}", kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(strict_json_loads(self.transform_json), dict):
            raise ValueError("Media transforms require a JSON object")
        if self.rendered is not None and (
            self.rendered.kind != "rendered"
            or self.rendered.modality != self.modality
            or self.rendered.source_family != self.source.source_family
        ):
            raise ValueError("Collected media must bind the actual decoded output")
        if not isinstance(self.prompt, str):
            raise ValueError("A teacher prompt is text")
        if self.modality not in ("text", *MEDIA):
            raise ValueError("Media records describe text, audio, image or video")
        if self.source.kind != "features" or self.source.modality != self.modality:
            raise ValueError("A media record observes teacher features of its own modality")
        if (
            self.embedding.kind != "embedding"
            or self.embedding.modality != self.modality
            or self.embedding.source_family != self.source.source_family
            or self.embedding.extractor != self.source.extractor
        ):
            raise ValueError("The teacher embedding must come from the same input and pass")
        if (self.target is None) != (self.output_geometry is None):
            raise ValueError("A realization target and its output geometry come together")
        if self.target is not None:
            if self.modality == "text":
                raise ValueError("Compressed text is input-only and has no realization target")
            if self.target.kind != "codec_latents" or self.target.modality != self.modality:
                raise ValueError("Realization targets are codec latents of the same modality")
            if self.target.source_family != self.source.source_family:
                raise ValueError("Source features and target latents describe different media")
            assert self.output_geometry is not None
            if self.output_geometry.modality != self.modality:
                raise ValueError("Output geometry modality differs from the record")

    def to_bytes(self) -> bytes:
        return canonical_json({"version": MEDIA_RECORD_VERSION, **asdict(self)})

    @classmethod
    def from_bytes(cls, payload: bytes) -> MediaRecord:
        if not isinstance(payload, bytes) or len(payload) > 16384:
            raise ValueError("Media descriptor must be bounded canonical JSON bytes")
        value = strict_json_loads(payload)
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "version",
                "modality",
                "source",
                "target",
                "output_geometry",
                "prompt",
                "embedding",
                "rendered",
                "transform_json",
            }
            or value["version"] != MEDIA_RECORD_VERSION
        ):
            raise ValueError("Media record has an unsupported version or fields")
        if canonical_json(value) != payload:
            raise ValueError("Media descriptor must have canonical serialization")
        return cls(
            modality=value["modality"],
            source=TensorAsset.from_dict(value["source"]),
            target=None if value["target"] is None else TensorAsset.from_dict(value["target"]),
            output_geometry=(
                None
                if value["output_geometry"] is None
                else OutputGeometry(**value["output_geometry"])
            ),
            prompt=value["prompt"],
            embedding=TensorAsset.from_dict(value["embedding"]),
            rendered=None
            if value["rendered"] is None
            else TensorAsset.from_dict(value["rendered"]),
            transform_json=value["transform_json"],
        )


class TensorAssetStore:
    """Immutable content-addressed asset store rooted at the prepared dataset directory.

    A file is read and checksummed when first loaded; its decoded array then stays
    in a small byte-bounded cache and is trusted only while the file's size,
    modification time and inode are what they were when it was verified.
    """

    CACHE_BYTES = 256 << 20

    def __init__(self, root: str | Path):
        self.root = Path(root)
        # sha256 -> (file identity when verified, decoded array), least recently used first.
        self._cache: OrderedDict[str, tuple[tuple[int, int, int], np.ndarray]] = OrderedDict()
        self._cached_bytes = 0

    def put(self, value: np.ndarray, **metadata: Any) -> TensorAsset:
        values = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
        if not np.isfinite(values).all():
            raise ValueError("Media tensor contains nonfinite values")
        data = values.tobytes(order="C")
        asset = TensorAsset(
            sha256=hashlib.sha256(data).hexdigest(), shape=tuple(values.shape), **metadata
        )
        path = self.root / asset.relative_path
        if path.exists():
            if self._read(path, asset) is None:
                raise ValueError("Existing immutable media asset has changed")
            return asset
        with atomic_binary_writer(path) as output:
            output.write(data)
        return asset

    def put_teacher(self, output: Any, *, source_family: str) -> tuple[TensorAsset, TensorAsset]:
        """Store one teacher pass (``TeacherOutput``): its features and pooled embedding."""
        common = dict(
            modality=output.modality if output.modality in MEDIA else "text",
            extractor=output.extractor,
            source_family=source_family,
        )
        features = self.put(output.features.numpy(), kind="features", **common)
        embedding = self.put(output.embedding.numpy(), kind="embedding", **common)
        return features, embedding

    def load(self, asset: TensorAsset) -> np.ndarray:
        """Fail closed on stale, substituted, or non-finite files; a private writable copy."""
        path = self.root / asset.relative_path
        entry = self._cache.get(asset.sha256)
        if entry is not None and entry[1].shape == asset.shape and self._identity(path) == entry[0]:
            self._cache.move_to_end(asset.sha256)
            return entry[1].copy()
        result = self._read(path, asset)
        if result is None:
            raise ValueError("Media asset differs from its declared shape or SHA256")
        self._remember(asset.sha256, self._identity(path), result)
        return result.copy()

    def copy_record(self, record: MediaRecord, destination: TensorAssetStore) -> None:
        """Verify and publish every tensor needed to replay one physical record."""
        for asset in (record.source, record.embedding, record.target, record.rendered):
            if asset is None:
                continue
            metadata = {
                key: value for key, value in asdict(asset).items() if key not in {"sha256", "shape"}
            }
            held = destination.put(self.load(asset), **metadata)
            if held != asset:
                raise ValueError("Published media asset changed its immutable identity")

    @staticmethod
    def _identity(path: Path) -> tuple[int, int, int] | None:
        if path.is_symlink():
            return None
        try:
            status = path.stat()
        except OSError:
            return None
        return status.st_size, status.st_mtime_ns, status.st_ino

    def _remember(self, sha256: str, identity: tuple[int, int, int] | None, array: np.ndarray):
        previous = self._cache.pop(sha256, None)
        if previous is not None:
            self._cached_bytes -= previous[1].nbytes
        if identity is None or array.nbytes > self.CACHE_BYTES:
            return
        self._cache[sha256] = (identity, array)
        self._cached_bytes += array.nbytes
        while self._cached_bytes > self.CACHE_BYTES:
            _, (_, evicted) = self._cache.popitem(last=False)
            self._cached_bytes -= evicted.nbytes

    @staticmethod
    def _read(path: Path, asset: TensorAsset) -> np.ndarray | None:
        # Only the digest-derived path is read; size is checked before reading.
        if path.is_symlink() or path.stat().st_size != math.prod(asset.shape) * 4:
            return None
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            return None
        result = np.frombuffer(data, dtype="<f4").reshape(asset.shape)
        if not np.isfinite(result).all():
            return None
        return result.copy()
