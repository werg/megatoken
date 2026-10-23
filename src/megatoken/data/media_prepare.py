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
import json
import subprocess
import tempfile
from dataclasses import asdict, replace
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

    def _features(self, modality: str, source: str, family: str) -> tuple[TensorAsset, TensorAsset]:
        return self.store.put_teacher(self.teacher.encode(modality, source), source_family=family)

    def text(self, payload: bytes) -> tuple[TensorAsset, TensorAsset]:
        """Teacher features and pooled embedding of exact text bytes (a byte-E anchor)."""
        return self._features("text", payload.decode("utf-8"), hashlib.sha256(payload).hexdigest())

    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord:
        """One media event; ``target`` adds the realizer latents and output geometry."""
        import torch

        path = Path(path)
        family = file_sha256(path)
        source, embedding = self._features(modality, str(path), family)
        if not target:
            return MediaRecord(modality, source, embedding=embedding)
        realizer = self.realizer(modality)
        if modality == "audio":
            waveform = decode_audio(path, realizer.sample_rate)
            latents = realizer.encode(torch.from_numpy(waveform)[None].to(self.device))
            geometry = OutputGeometry(
                "audio", sample_count=len(waveform), sample_rate=realizer.sample_rate
            )
            clock = dict(sample_rate=realizer.sample_rate)
        elif modality == "image":
            # Resized to the nearest geometry the realizer accepts; the stored
            # geometry is what the target latents really describe.
            pixels = realizer.fit(torch.from_numpy(decode_image(path))[None].to(self.device))
            latents = realizer.encode(pixels)
            geometry = OutputGeometry("image", height=pixels.shape[2], width=pixels.shape[3])
            clock = {}
        else:
            frames, rate = decode_video(path)
            # Resized, and trimmed by at most frame_step - 1 trailing frames.
            clip = realizer.fit(torch.from_numpy(frames)[None].to(self.device))
            latents = realizer.encode(clip)
            geometry = OutputGeometry(
                "video",
                frames=clip.shape[1],
                height=clip.shape[3],
                width=clip.shape[4],
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
        return MediaRecord(modality, source, latent, geometry, embedding=embedding)

    def audio_interval(
        self,
        path: str | Path,
        *,
        start_sample: int,
        end_sample: int,
        channel: int,
        target: bool = True,
    ) -> MediaRecord:
        """Extract the exact available original speaker channel before the teacher.

        The original clock, silence and source hash survive resampling. A
        channel is explicit: separated speakers are never silently mixed.
        """
        import soundfile as sf

        original = Path(path)
        if (
            type(start_sample) is not int
            or type(end_sample) is not int
            or not 0 <= start_sample < end_sample
        ):
            raise ValueError("Audio interval needs exact ordered original sample coordinates")
        with sf.SoundFile(original) as audio:
            if (
                end_sample > len(audio)
                or type(channel) is not int
                or not 0 <= channel < audio.channels
            ):
                raise ValueError(
                    "Audio interval or speaker channel is outside the original recording"
                )
            rate = audio.samplerate
            audio.seek(start_sample)
            waveform = audio.read(end_sample - start_sample, dtype="float32", always_2d=True)[
                :, channel
            ]
        with tempfile.TemporaryDirectory() as scratch:
            chunk = Path(scratch) / "available.wav"
            sf.write(chunk, waveform, rate, subtype="FLOAT")
            record = self.media("audio", chunk, target=target)
            transform = {
                "version": 1,
                "original_sha256": file_sha256(original),
                "channel": channel,
                "start_tick": start_sample,
                "end_tick": end_sample,
                "ticks_per_second": rate,
                "extracted_sha256": file_sha256(chunk),
                "feature_scope": "exact_available_interval",
            }
        return replace(record, transform_json=json.dumps(transform, sort_keys=True))

    def video_interval(
        self,
        path: str | Path,
        *,
        start_tick: int,
        end_tick: int,
        fps: Fraction,
        target: bool = True,
    ) -> MediaRecord:
        """Prepare only the available displayed interval, with exact source PTS.

        Resampling to the output codec's uniform clock retains a correspondence
        to every original displayed frame. No full-clip teacher features leak
        into a prefix example. Source ticks use the container's own time base.
        """
        from megatoken.data.media_timing import decode_video_interval, video_timeline

        original = Path(path)
        timeline = video_timeline(original)
        frames, indices = decode_video_interval(original, timeline, start_tick, end_tick, fps)
        height, width = frames.shape[-2:]
        pixels = np.round(frames.clip(0, 1) * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        with tempfile.TemporaryDirectory() as scratch:
            chunk = Path(scratch) / "available.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-c:v",
                    "ffv1",
                    str(chunk),
                ],
                input=pixels.tobytes(),
                check=True,
                capture_output=True,
            )
            record = self.media("video", chunk, target=target)
            retained = (
                len(indices) if record.output_geometry is None else record.output_geometry.frames
            )
            transform = {
                "version": 1,
                "original_sha256": file_sha256(original),
                "time_base_num": timeline.time_base.numerator,
                "time_base_den": timeline.time_base.denominator,
                "start_tick": start_tick,
                "end_tick": end_tick,
                "source_display_indices": list(indices),
                "source_pts": [timeline.pts[index] for index in indices],
                "retained_display_indices": list(indices[:retained]),
                "output_fps_num": fps.numerator,
                "output_fps_den": fps.denominator,
                "extracted_sha256": file_sha256(chunk),
                "feature_scope": "exact_available_interval",
            }
        return replace(record, transform_json=json.dumps(transform, sort_keys=True))

    def rendered(self, tensor, spec) -> MediaRecord:
        """Prepare the actual decoded selection, with a cryptographic output receipt.

        Reuse the ordinary media teacher/codec path, never a target substituted
        for the actor's output. Display quantization is an explicit transform;
        the original selected float tensor remains a checked shared asset.
        """
        import soundfile as sf
        from PIL import Image

        from megatoken.model.media import validate_public_media_spec

        validate_public_media_spec(spec)
        values = tensor.detach().float().cpu().contiguous().numpy()
        if values.shape[0] != 1:
            raise ValueError("Collected rendering must contain one event")
        values = values[0].astype("<f4", copy=False)
        identity = hashlib.sha256(values.tobytes()).hexdigest()
        decoder = self.realizer(spec.modality)
        rendered = self.store.put(
            values,
            modality=spec.modality,
            kind="rendered",
            extractor=decoder.identity,
            source_family=identity,
            sample_rate=spec.sample_rate,
            fps_num=spec.fps_num,
            fps_den=spec.fps_den,
        )
        with tempfile.TemporaryDirectory() as scratch:
            if spec.modality == "audio":
                path = Path(scratch) / "selected.wav"
                if values.shape != (spec.sample_count,):
                    raise ValueError("Selected speech geometry differs from the public request")
                sf.write(path, values, spec.sample_rate, subtype="FLOAT")
            else:
                if values.shape != (spec.frames, 3, spec.height, spec.width):
                    raise ValueError("Selected visual geometry differs from the public request")
                pixels = (
                    np.round(np.clip(values, 0, 1) * 255).astype(np.uint8).transpose(0, 2, 3, 1)
                )
                if spec.modality == "image":
                    path = Path(scratch) / "selected.png"
                    Image.fromarray(pixels[0]).save(path)
                else:
                    path = Path(scratch) / "selected.mkv"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-loglevel",
                            "error",
                            "-y",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            f"{spec.width}x{spec.height}",
                            "-framerate",
                            f"{spec.fps_num}/{spec.fps_den}",
                            "-i",
                            "pipe:0",
                            "-c:v",
                            "ffv1",
                            str(path),
                        ],
                        input=pixels.tobytes(),
                        check=True,
                    )
            record = self.media(spec.modality, path)
            transport_hash = file_sha256(path)
        # These features and codec targets were made from this exact selected
        # rendering's losslessly transported/display-quantized derivative.
        geometry = OutputGeometry(
            spec.modality,
            sample_count=spec.sample_count,
            sample_rate=spec.sample_rate,
            frames=spec.frames,
            height=spec.height,
            width=spec.width,
            fps_num=spec.fps_num,
            fps_den=spec.fps_den,
            provenance=spec.provenance,
        )
        return replace(
            record,
            source=replace(record.source, source_family=identity),
            embedding=replace(record.embedding, source_family=identity),
            target=replace(record.target, source_family=identity),
            output_geometry=geometry,
            rendered=rendered,
            transform_json=json.dumps(
                {
                    "version": 1,
                    "selected_sha256": identity,
                    "transport_sha256": transport_hash,
                    "display_quantization": "float32" if spec.modality == "audio" else "rgb24",
                },
                sort_keys=True,
            ),
        )

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
