"""Original media files -> prepared media records (the only media preparation path).

For one original file, the pinned EmbeddingGemma2 teacher reads the file with its
own locked processor (the observed source), and the modality's resident realizer
encodes one decode of the same file into its codec latents (the realization
target). Both assets carry ``source_family`` = SHA256 of the file bytes, so a
record's source and target describe the same media by construction.

Text anchors (``megatoken.training.text_anchors``) are prepared the same way:
teacher features of exact text bytes, ``source_family`` = SHA256 of those bytes.

Clips must already be cut to the event's extent (realizer limits: 120 s speech,
193 video frames, 2048 px). The teacher samples video at its processor's own
frame policy; long media should be prepared as several events.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from megatoken.data.media_assets import MediaRecord, OutputGeometry, TensorAsset, TensorAssetStore

REALIZER = {"audio": "pocket", "image": "sana", "video": "sana_video"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.clip(waveform.astype(np.float32), -1.0, 1.0)


def decode_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return pixels.transpose(2, 0, 1)


def decode_video(path: Path) -> tuple[np.ndarray, Fraction]:
    """Every displayed frame [F, 3, H, W] in [0, 1] and the stream's rational frame rate."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(path))
    frames = decoder.get_frames_in_range(0, len(decoder)).data
    rate = Fraction(decoder.metadata.average_fps).limit_denominator(1001)
    return frames.float().numpy() / 255.0, rate


class MediaPreparer:
    """Resident teacher and realizer codecs over one prepared root."""

    def __init__(self, manifest: str | Path, root: str | Path, *, device: str = "cuda"):
        from megatoken.omni.embeddinggemma2 import EmbeddingGemma2
        from megatoken.omni.loader import read_locked_manifest

        self.base, self.packs = read_locked_manifest(str(manifest))
        self.store = TensorAssetStore(root)
        self.device = device
        self.teacher = EmbeddingGemma2.from_pack(self.base, self.packs, device=device)
        self._realizers: dict[str, object] = {}

    def realizer(self, modality: str):
        from megatoken.omni.loader import build_realizer

        name = REALIZER[modality]
        if name not in self._realizers:
            if name not in self.packs:
                raise ValueError(f"Pack manifest does not lock the {name} realizer")
            # The facet condition is unused here; only the frozen codec encodes.
            self._realizers[name] = build_realizer(
                self.base, name, self.packs[name], facet_count=1, facet_width=8, device=self.device
            )
        return self._realizers[name]

    def _features(self, modality: str, source: str, family: str) -> TensorAsset:
        observation = self.teacher.encode(modality, source)
        return self.store.put(
            observation.features.numpy(),
            modality=modality,
            kind="features",
            extractor=observation.extractor,
            source_family=family,
        )

    def text(self, payload: bytes) -> TensorAsset:
        """Teacher features of exact text bytes (a byte-E anchor)."""
        return self._features("text", payload.decode("utf-8"), hashlib.sha256(payload).hexdigest())

    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord:
        """One media event; ``target`` adds the realizer latents and output geometry."""
        import torch

        path = Path(path)
        family = file_sha256(path)
        source = self._features(modality, str(path), family)
        if not target:
            return MediaRecord(modality, source)
        realizer = self.realizer(modality)
        if modality == "audio":
            waveform = decode_audio(path, realizer.sample_rate)
            latents = realizer.encode(torch.from_numpy(waveform)[None].to(self.device))
            geometry = OutputGeometry(
                "audio", sample_count=len(waveform), sample_rate=realizer.sample_rate
            )
            clock = dict(sample_rate=realizer.sample_rate)
        elif modality == "image":
            pixels = decode_image(path)
            latents = realizer.encode(torch.from_numpy(pixels)[None].to(self.device))
            geometry = OutputGeometry("image", height=pixels.shape[1], width=pixels.shape[2])
            clock = {}
        else:
            clip, rate = decode_video(path)
            latents = realizer.encode(torch.from_numpy(clip)[None].to(self.device))
            geometry = OutputGeometry(
                "video",
                frames=len(clip),
                height=clip.shape[2],
                width=clip.shape[3],
                fps_num=rate.numerator,
                fps_den=rate.denominator,
            )
            clock = dict(fps_num=rate.numerator, fps_den=rate.denominator)
        latent = self.store.put(
            latents[0].cpu().numpy(),
            modality=modality,
            kind="codec_latents",
            extractor=realizer.identity,
            source_family=family,
            **clock,
        )
        return MediaRecord(modality, source, latent, geometry)

    def video_endpoints(
        self, path: str | Path, *, sampled_source_indices: tuple[int, ...] | None = None
    ) -> tuple[MediaRecord, MediaRecord, dict]:
        """Opening and closing displayed frames of a clip as image events, with their audit."""
        from PIL import Image

        from megatoken.omni.endpoints import EndpointMap, displayed_endpoints

        clip, rate = decode_video(Path(path))
        opening, closing, audit = displayed_endpoints(
            clip,
            sampled_source_indices=sampled_source_indices or tuple(range(len(clip))),
            fps_num=rate.numerator,
            fps_den=rate.denominator,
        )
        records = []
        with tempfile.TemporaryDirectory() as scratch:
            for name, frame in (("opening", opening), ("closing", closing)):
                image = Path(scratch) / f"{name}.png"
                pixels = np.round(frame.transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(pixels).save(image)
                records.append(self.media("image", image))
        mapping = EndpointMap(
            video_sha256=file_sha256(Path(path)),
            opening_sha256=records[0].source.sha256,
            closing_sha256=records[1].source.sha256,
            **audit,
        )
        return records[0], records[1], {**asdict(mapping), "fingerprint": mapping.fingerprint}
