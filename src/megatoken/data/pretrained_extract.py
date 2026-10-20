"""Offline-only pretrained media feature/codec bridges with explicit identities.

Model weights are never downloaded here and this module is not imported by
the trainer's runtime hot path. Each export is deterministic after a frozen
source waveform, crop/resize, model directory and preprocessor revision.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from megatoken.data.media_assets import MediaRecord, TensorAsset, TensorAssetStore


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_audio_rate(source: TensorAsset, expected: int) -> None:
    if source.modality != "audio" or source.kind != "raw" or source.sample_rate != expected:
        raise ValueError(f"Audio donor requires explicit {expected} Hz mono raw input")


def finite_features(value: Any, *, width: int = 0) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype="<f4")
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("Frozen perception view must contain ordered [tokens,dim]")
    if width and result.shape[-1] != width:
        raise ValueError("Frozen feature dimension differs from pinned model geometry")
    if not np.isfinite(result).all():
        raise ValueError("Donor supplied nonfinite feature values")
    return np.ascontiguousarray(result)


def extract_whistle(wave: np.ndarray, *, weights: str, expected_width: int) -> np.ndarray:
    """16k Whistle encoder, ~80 ms per frame; no ASR transcript injected into E."""
    import needle

    path = Path(weights)
    if not path.is_file() or path.suffix != ".cact":
        raise ValueError("Whistle requires an existing pinned local .cact file")
    if wave.ndim != 1 or wave.size == 0 or wave.size > 30 * 16000:
        raise ValueError("Whistle observations require one 16k mono <=30s chunk")
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) > 1.001:
        raise ValueError("Whistle waveform must be finite normalized 16k PCM")
    embedder = needle.Whistle(weights=str(path.resolve()))
    # Needle's public embed() is a flattened frame-major encoder vector.
    flat = np.asarray(embedder.embed(np.asarray(wave, dtype=np.float32)),
                      dtype=np.float32).reshape(-1)
    if expected_width < 1 or not flat.size or flat.size % expected_width:
        raise ValueError("Whistle flattened encoder frames do not match declared width")
    return finite_features(flat.reshape(-1, expected_width), width=expected_width)


def extract_siglip2(frames: np.ndarray, *, model_dir: str, device: str,
                    detail: bool = False) -> np.ndarray:
    """Pinned fallback spatial features; preserves ordered per-frame patches."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Siglip2VisionModel

    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("SigLIP view needs [frames,3,height,width]")
    model = Siglip2VisionModel.from_pretrained(model_dir, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model.eval().to(device)
    values = []
    with torch.inference_mode():
        for frame in frames:
            image = Image.fromarray(np.uint8(np.rint(np.clip(
                frame.transpose(1, 2, 0), 0, 1) * 255)), "RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()
                      if isinstance(tensor, torch.Tensor)}
            layer = model(**inputs).last_hidden_state
            values.append(finite_features(layer).copy())
    return finite_features(np.concatenate(values, 0))


def extract_minicpm_v47(frames: np.ndarray, *, model_dir: str, device: str,
                        downsample_mode: str = "16x") -> np.ndarray:
    """Vision tower/merger only. Never call a Qwen textual decoder for E.

    The v4.7 public get_image_features accepts pixel_values and target_sizes.
    Versions without that interface fail, rather than substituting text logits.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if downsample_mode not in {"4x", "16x"}:
        raise ValueError("MiniCPM vision downsampling must be 4x or 16x")
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("MiniCPM observation must be [frames,3,height,width]")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
    ).eval().to(device)
    if not hasattr(model, "get_image_features"):
        raise ValueError("Pinned checkpoint lacks the v4.7 native vision feature API")
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
    )
    pieces = []
    with torch.inference_mode():
        for frame in frames:
            image = Image.fromarray(np.uint8(np.rint(np.clip(
                frame.transpose(1, 2, 0), 0, 1) * 255)), "RGB")
            inputs = processor(images=image, return_tensors="pt")
            if "pixel_values" not in inputs or "target_sizes" not in inputs:
                raise ValueError("MiniCPM v4.7 processor must expose native image target sizes")
            pixel_values = inputs["pixel_values"]
            if isinstance(pixel_values, (tuple, list)):
                pixel_values = torch.cat(pixel_values, 0)
            target_sizes = inputs["target_sizes"]
            pixel_values = pixel_values.to(device)
            target_sizes = target_sizes.to(device)
            output = model.get_image_features(
                pixel_values=pixel_values,
                target_sizes=target_sizes,
                downsample_mode=downsample_mode,
            )
            values = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            if isinstance(values, (tuple, list)):
                values = torch.cat(values, dim=1)
            pieces.append(finite_features(values))
    return finite_features(np.concatenate(pieces, 0))


def extract_pocket_mimi(wave: np.ndarray, *, donor: Any) -> np.ndarray:
    """Encode 24k native continuous Mimi latents, NOT 16k semantic features."""
    import torch

    if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("Pocket Mimi requires finite nonempty mono waveform")
    if donor.sample_rate != 24000:
        raise ValueError("Pocket donor and Mimi sample clock must be 24 kHz")
    with torch.inference_mode():
        value = torch.as_tensor(wave.copy(), device=next(donor.mimi.parameters()).device)
        latents = donor.mimi.encode_to_latent(value.reshape(1, 1, -1))
    return finite_features(latents)


def sample_video_endpoints(frames: np.ndarray, *, displayed_indices: tuple[int, ...],
                           crop_id: str, spatial_mapping: dict[str, object]):
    """Extract actual displayed endpoints AFTER final crop/resize/time selection.

    The supplied tensor is the final visible target, not padded codec frames.
    Returns end frames and a frozen mapping for pixel/perceptual comparison.
    """
    if (frames.ndim != 4 or frames.shape[1] != 3 or not displayed_indices
            or len(displayed_indices) != len(frames)
            or any(a >= b for a, b in itertools.pairwise(displayed_indices))
            or not crop_id):
        raise ValueError("Video display indices/crop must match final sampled clip")
    mapping = {
        "first_displayed_index": displayed_indices[0],
        "last_displayed_index": displayed_indices[-1],
        "display_frame_count": len(frames),
        "crop_id": crop_id,
        "spatial_mapping": spatial_mapping,
    }
    return np.ascontiguousarray(frames[0]), np.ascontiguousarray(frames[-1]), mapping


def pinned_export(
    store: TensorAssetStore, record: MediaRecord, features: np.ndarray,
    *, extractor_id: str, feature_slot: str,
) -> MediaRecord:
    """Make a new immutable MediaRecord with primary or extra E views."""
    if record.source.kind not in {"raw", "features"}:
        raise ValueError("Media E requires raw signal or recognized frozen features")
    if feature_slot not in {"primary", "acoustic", "detail", "codec"}:
        raise ValueError("Unknown E feature-view slot")
    asset = store.put(
        finite_features(features),
        modality=record.modality, kind="features", extractor=extractor_id,
        sample_rate=record.source.sample_rate,
        fps_num=record.source.fps_num, fps_den=record.source.fps_den,
    )
    if feature_slot == "primary":
        return MediaRecord(record.modality, asset, record.target, views=record.views,
                           source_masks=record.source_masks, target_mask=record.target_mask)
    return MediaRecord(record.modality, record.source, record.target,
                       views=(*record.views, asset), target_mask=record.target_mask)


def record_native_codec_target(
    store: TensorAssetStore, record: MediaRecord, value: np.ndarray,
    *, codec_id: str, pack_id: str, preprocessing_id: str,
    normalization_json: str, geometry: tuple[int, ...], clock_id: str,
    start_ms: int, end_ms: int, valid_mask: np.ndarray | None = None,
) -> MediaRecord:
    """No guessed native latent shape/rate; caller records all transforms."""
    source = record.source
    codec = store.put(
        value, modality=record.modality, kind="codec_latents",
        codec_id=codec_id, profile=pack_id, preprocessing_id=preprocessing_id,
        normalization_json=normalization_json, geometry=geometry,
        clock_id=clock_id, start_ms=start_ms, end_ms=end_ms,
        sample_rate=source.sample_rate,
        fps_num=source.fps_num, fps_den=source.fps_den,
    )
    mask = None
    if valid_mask is not None:
        mask = store.put_mask(valid_mask, modality=record.modality,
                              sample_rate=source.sample_rate,
                              fps_num=source.fps_num, fps_den=source.fps_den)
    return MediaRecord(record.modality, source, codec, views=record.views,
                       source_masks=record.source_masks, target_mask=mask)
