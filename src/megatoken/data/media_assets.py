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
MEDIA_RECORD_VERSION = 2
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
    kind: Literal["raw", "features", "codec_latents", "mask"]
    sha256: str
    shape: tuple[int, ...]
    extractor: str = ""
    sample_rate: int = 0
    fps_num: int = 0
    fps_den: int = 1
    # Exact clock/mask metadata prevents codec padding becoming real media.
    valid_count: int = 0
    time_offset_num: int = 0
    time_offset_den: int = 1
    normalization: str = ""
    artifact_version: str = ""
    source_family: str = ""
    # Pinned native codec identity and explicit valid-coordinate metadata.
    codec_id: str = ""
    preprocessing_id: str = ""
    normalization_json: str = "{}"
    geometry: tuple[int, ...] = ()
    clock_id: str = ""
    start_ms: int = 0
    end_ms: int = 0
    profile: str = ""

    def __post_init__(self) -> None:
        _sha(self.sha256)
        if self.modality not in ("audio", "image", "video"):
            raise ValueError("Unsupported media modality")
        if self.kind not in ("raw", "features", "codec_latents", "mask"):
            raise ValueError("Media asset kind must be raw, features or codec_latents")
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
        elif self.kind == "codec_latents":
            if self.codec_id:
                if not self.preprocessing_id or not self.geometry:
                    raise ValueError(
                        "Codec latents require pinned codec, preprocessor and decoded geometry"
                    )
                if len(self.shape) not in (2, 3, 4, 5):
                    raise ValueError("Unsupported codec latent coordinates")
                expected_rank = {"audio": 1, "image": 3, "video": 4}[self.modality]
                if len(self.geometry) != expected_rank or any(n < 1 for n in self.geometry):
                    raise ValueError("Codec latent geometry must describe decoded physical output")
            else:
                target_ranks = {"audio": 2, "image": 3, "video": 4}
                if len(self.shape) != target_ranks[self.modality] or not self.extractor:
                    raise ValueError("Codec latents need a pinned codec and native latent rank")
        elif self.kind == "mask":
            if len(self.shape) not in (1, 2, 3, 4, 5) or self.extractor or self.codec_id:
                raise ValueError("Masks require valid coordinates, without codec/feature weights")
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
            type(self.sample_rate) is not int
            or self.sample_rate < 0
            or type(self.fps_num) is not int
            or self.fps_num < 0
            or type(self.fps_den) is not int
            or self.fps_den < 1
        ):
            raise ValueError("Invalid sample-rate/fps metadata")
        if (
            self.modality == "audio"
            and self.sample_rate not in (16000, 24000)
            and self.sample_rate <= 0
        ):
            raise ValueError("Audio assets require a declared native clock")
        if (
            self.modality == "audio"
            and self.kind == "raw"
            and self.sample_rate not in (16000, 24000)
        ):
            raise ValueError("Raw speech must declare 16 or 24 kHz preprocessing")
        if self.valid_count < 0 or self.valid_count > self.shape[0]:
            raise ValueError("Valid data exceeds the padded tensor")
        if self.time_offset_num < 0 or self.time_offset_den < 1:
            raise ValueError("Invalid temporal origin")
        if (
            type(self.start_ms) is not int
            or type(self.end_ms) is not int
            or self.start_ms < 0
            or self.end_ms < self.start_ms
        ):
            raise ValueError("Media clock interval is invalid")
        if any(type(v) is not int or v < 1 for v in self.geometry):
            raise ValueError("Invalid decoded geometry")
        try:
            normalization = strict_json_loads(self.normalization_json)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid codec normalization metadata") from exc
        if not isinstance(normalization, dict):
            raise ValueError("Normalization must be a JSON object")
        if (
            self.kind == "codec_latents"
            and self.codec_id
            and (not self.clock_id or self.end_ms <= self.start_ms)
        ):
            raise ValueError("Codec target needs a real physical time interval")
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
        if "geometry" in value:
            value["geometry"] = tuple(value["geometry"])
        return cls(**value)


@dataclass(frozen=True, slots=True)
class OutputGeometry:
    """Requested/predicted or teacher-only physical bounds, never semantic content.

    Codec latent dimensions are not necessarily an invertible record of
    displayed frames/crops. Store the actual output dimensions explicitly.
    """

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
            if not 1 <= self.sample_count <= 24000 * 60 or self.sample_rate not in (16000, 24000):
                raise ValueError("Audio output geometry needs a native sample count/rate")
            if self.height or self.width or self.fps_num:
                raise ValueError("Audio output cannot contain video geometry")
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
                raise ValueError("Video/image output cannot contain audio sample clock")
        else:
            raise ValueError("Unknown output geometry modality")
        if self.fps_den <= 0:
            raise ValueError("Invalid output frame-rate denominator")


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """Text-independent, auditable local input and optional output target."""

    modality: Modal
    source: TensorAsset
    target: TensorAsset | None = None
    version: int = MEDIA_RECORD_VERSION
    source_views: tuple[TensorAsset, ...] = ()
    output_geometry: OutputGeometry | None = None
    views: tuple[TensorAsset, ...] = ()
    source_masks: tuple[TensorAsset, ...] = ()
    target_mask: TensorAsset | None = None

    def __post_init__(self) -> None:
        if self.output_geometry is not None:
            if (
                self.version != MEDIA_RECORD_VERSION
                or self.output_geometry.modality != self.modality
            ):
                raise ValueError("Output geometry requires a matching v2 media record")
            if self.target is None:
                raise ValueError("Physical output geometry cannot be attached to observation alone")
        if self.version not in (1, MEDIA_RECORD_VERSION):
            raise ValueError("Unsupported media reference version")
        if self.version == 1 and (
            self.source_views or (self.target is not None and self.target.kind == "codec_latents")
        ):
            raise ValueError("Multi-view/codec targets require media record v2")
        if any(v.modality != self.modality or v.kind == "raw" for v in self.source_views):
            raise ValueError("Additional input views must be matching features/codec latents")
        if self.modality != self.source.modality or (
            self.target is not None and self.target.modality != self.modality
        ):
            raise ValueError("Media reference modality does not match source/target")
        if self.target is not None and self.target.kind not in ("raw", "codec_latents"):
            raise ValueError("Realization targets must be raw or matching codec latents")
        if self.version == 1 and (self.views or self.source_masks or self.target_mask):
            raise ValueError("Masks and additional views require media record v2")
        if any(
            v.modality != self.modality or v.kind not in ("raw", "features", "codec_latents")
            for v in self.views
        ):
            raise ValueError("Supplementary input views must match the event modality")
        if self.source_masks and len(self.source_masks) != len(self.views) + 1:
            raise ValueError("Provide one valid mask for every declared source view")
        for asset in (*self.source_masks, *((self.target_mask,) if self.target_mask else ())):
            if asset.kind != "mask" or asset.modality != self.modality:
                raise ValueError("Declared masks must be pinned mask assets")

    def to_bytes(self) -> bytes:
        data = asdict(self)
        if self.output_geometry is None:
            data.pop("output_geometry")
        if not self.views:
            data.pop("views")
        if not self.source_masks:
            data.pop("source_masks")
        if self.target_mask is None:
            data.pop("target_mask")
        if self.version == 1:
            data.pop("source_views")
            for field in ("source", "target"):
                if data[field] is not None:
                    for extra in (
                        "valid_count",
                        "time_offset_num",
                        "time_offset_den",
                        "normalization",
                        "artifact_version",
                        "source_family",
                        "codec_id",
                        "preprocessing_id",
                        "normalization_json",
                        "geometry",
                        "clock_id",
                        "start_ms",
                        "end_ms",
                        "profile",
                    ):
                        data[field].pop(extra, None)
        return canonical_json(data)

    @classmethod
    def from_bytes(cls, payload: bytes) -> MediaRecord:
        if not isinstance(payload, bytes) or len(payload) > 16384:
            raise ValueError("Media descriptor must be bounded canonical JSON bytes")
        value = strict_json_loads(payload)
        required = {"modality", "source", "target", "version"}
        optional = {"source_views", "output_geometry", "views", "source_masks", "target_mask"}
        if (
            not isinstance(value, dict)
            or not required <= set(value)
            or set(value) - required - optional
        ):
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
            target=(None if value["target"] is None else TensorAsset.from_dict(value["target"])),
            version=value["version"],
            source_views=tuple(TensorAsset.from_dict(v) for v in value.get("source_views", ())),
            output_geometry=(
                None
                if value.get("output_geometry") is None
                else OutputGeometry(**value["output_geometry"])
            ),
            views=tuple(TensorAsset.from_dict(v) for v in value.get("views", ())),
            source_masks=tuple(TensorAsset.from_dict(v) for v in value.get("source_masks", ())),
            target_mask=(
                None
                if value.get("target_mask") is None
                else TensorAsset.from_dict(value["target_mask"])
            ),
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
        kind: Literal["raw", "features", "codec_latents", "mask"] = "raw",
        extractor: str = "",
        sample_rate: int = 0,
        fps_num: int = 0,
        fps_den: int = 1,
        valid_count: int = 0,
        time_offset_num: int = 0,
        time_offset_den: int = 1,
        normalization: str = "",
        artifact_version: str = "",
        source_family: str = "",
        codec_id: str = "",
        preprocessing_id: str = "",
        normalization_json: str = "{}",
        geometry: tuple[int, ...] = (),
        clock_id: str = "",
        start_ms: int = 0,
        end_ms: int = 0,
        profile: str = "",
    ) -> TensorAsset:
        values = np.asarray(value, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("Media tensor contains nonfinite values")
        values = np.ascontiguousarray(values)
        data = values.tobytes(order="C")
        digest = hashlib.sha256(data).hexdigest()
        ref = TensorAsset(
            modality,
            kind,
            digest,
            tuple(values.shape),
            extractor,
            sample_rate,
            fps_num,
            fps_den,
            valid_count,
            time_offset_num,
            time_offset_den,
            normalization,
            artifact_version,
            source_family,
            codec_id,
            preprocessing_id,
            normalization_json,
            tuple(geometry),
            clock_id,
            start_ms,
            end_ms,
            profile,
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

    def put_mask(
        self,
        mask: np.ndarray,
        *,
        modality: Modal,
        sample_rate: int = 0,
        fps_num: int = 0,
        fps_den: int = 1,
    ) -> TensorAsset:
        """Store a deterministic binary valid-coordinate mask, not codec padding."""
        values = np.asarray(mask)
        if values.dtype != np.bool_ and not np.isin(values, (0, 1)).all():
            raise ValueError("Valid-coordinate masks must be binary")
        return self.put(
            values.astype("<f4"),
            modality=modality,
            kind="mask",
            sample_rate=sample_rate,
            fps_num=fps_num,
            fps_den=fps_den,
        )

    def load_mask(self, asset: TensorAsset) -> np.ndarray:
        if asset.kind != "mask":
            raise ValueError("Expected a declared mask tensor")
        result = self.load(asset)
        if not np.isin(result, (0, 1)).all():
            raise ValueError("Valid-coordinate mask has nonbinary values")
        return result.astype(bool)
