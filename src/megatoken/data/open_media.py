"""Public media datasets -> media-identity and concordant task episodes.

Each reader unpacks rows of one pinned public dataset (local files only; fetching
is the cloud job's ``inputs``) into plain *clips*: a media file path plus the
labels the publisher supplies. Each builder turns a clip into an aligned bundle
whose artifacts are prepared by ``MediaPreparer`` (any object with the same
``media(modality, path)`` method works), then ``data/concordance.py`` compiles the
bundle into ordinary episodes whose stream is the task family named in
``stages.json`` (``transcription``, ``image_edit``, ...).

Dataset-level facts live in ``SOURCES``: the immutable revision every job input
URL must carry (checked by a test), the licence found on the dataset card, and
what is uncertain about it. Splits are frozen here, per original source family
(speaker, sentence, photo, recording), so no family straddles train/validation.

Procedural edits (``speech_edits``, ``video_edits``) are real, verified edits:
a deterministic ffmpeg program applied to a real recording, recorded with
``executed`` provenance. They stand in for human-made speech/video edit data,
which no open dataset provides.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import tarfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .concordance import (
    AlignedBundle,
    BundleArtifact,
    compile_task,
    standard_views,
)
from .media_assets import MEDIA_RECORD_TYPE, MediaRecord
from .schema import Episode, Lineage, Meridian, PhysicalRecord, TrackEvent, TrackSpec

VALIDATION_PERCENT = 5
MAX_SPEECH_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class OpenSource:
    source_id: str
    repository: str
    revision: str
    license: str
    caveat: str = ""


SOURCES = {
    source.source_id: source
    for source in (
        OpenSource(
            "librispeech_asr",
            "openslr/librispeech_asr",
            "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
            "CC-BY-4.0",
        ),
        OpenSource(
            "libritts_r",
            "mythicinfinity/libritts_r",
            "0d1718db351207d5a493135a62d5ed8cc4733830",
            "CC-BY-4.0",
        ),
        OpenSource(
            "fleurs",
            "google/fleurs",
            "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
            "CC-BY-4.0",
        ),
        OpenSource(
            "docci",
            "google/docci",
            "a0a43eaf34676ffd008fb6565dd8c2ba00d09100",
            "CC-BY-4.0",
            "Images and descriptions are fetched from the publisher's storage bucket "
            "(the Hub repository holds only a loading script); the job input pins their sha256.",
        ),
        OpenSource(
            "magicbrush",
            "osunlp/MagicBrush",
            "6c254560879df7851233ea0fd2ae2a62764d0ef2",
            "CC-BY-4.0",
            "Source images come from MS-COCO, whose Flickr photos carry mixed per-image "
            "Creative Commons licences (some NonCommercial/NoDerivatives); the card does not "
            "address this. Pilot scale only until cleared.",
        ),
        OpenSource(
            "rekadaily_processed",
            "RekaAI/RekaDaily-10k-processed",
            "4e7b797ffc1ca06ee7b0ac4c60d31dea8081a35b",
            "Apache-2.0",
            "First-person recordings of paid collectors; clips contain ambient audio and "
            "possibly incidental speech.",
        ),
    )
}


# The stage-plan streams the readers below emit (identity streams plus concordant task families).
STREAMS = frozenset(
    (
        "audio_identity",
        "image_identity",
        "video_identity",
        "transcription",
        "verbalization",
        "speech_translation",
        "image_description",
        "image_creation",
        "image_edit",
        "speech_edit",
        "av_creation",
        "av_edit",
    )
)


class Preparer(Protocol):
    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord: ...


class CachingPreparer:
    """Prepare each file once even when several bundles (or a voice reference) name it."""

    def __init__(self, inner: Preparer):
        self.inner = inner
        self._records: dict[tuple[str, str, bool], MediaRecord] = {}

    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord:
        key = (modality, str(path), target)
        if key not in self._records:
            self._records[key] = self.inner.media(modality, path, target=target)
        return self._records[key]


def frozen_split(family: str) -> str:
    bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:24]


# --- episodes -----------------------------------------------------------------------------


def identity_episode(record: MediaRecord, family: str | None = None) -> Episode:
    """Autoencode one prepared media event: a short instruction, then the media itself.

    A ``family`` (the original source family of the recording) freezes the split with that
    family, so the clip's identity and task episodes never straddle train and validation.
    """
    if record.target is None:
        raise ValueError("Identity episodes need a realization target")
    modality = record.modality
    records = (
        PhysicalRecord("prompt", "user", f"Reproduce this {modality}.".encode()),
        PhysicalRecord(
            "reproduction",
            "assistant",
            record.to_bytes(),
            turn_index=1,
            policy_mask=True,
            media_type=MEDIA_RECORD_TYPE,
            provenance="natural",
        ),
    )
    digest = record.source.source_family
    return Episode(
        f"{modality}-identity-{digest[:24]}",
        records,
        Lineage(
            "media-identity",
            "v1",
            family or digest,
            frozen_split=frozen_split(family) if family else None,
        ),
        stream=f"{modality}_identity",
        tracks=(TrackSpec("prompt", "user"), TrackSpec("reproduction", "assistant")),
        meridians=(
            Meridian(0, (TrackEvent("prompt", "prompt", "text", 0, "user"),)),
            Meridian(
                1,
                (
                    TrackEvent(
                        "reproduction",
                        "reproduction",
                        modality,
                        1,
                        "assistant",
                        # Event identity is the observed source, never the R target.
                        asset_sha256=record.source.sha256,
                    ),
                ),
            ),
        ),
    )


def _text(artifact_id: str, label: str, value: str, *, provenance: str = "natural"):
    return BundleArtifact(
        artifact_id, label, "text", payload=value.encode("utf-8"), provenance=provenance
    )


def _media(
    media: Preparer,
    artifact_id: str,
    label: str,
    modality: str,
    path: str | Path,
    *,
    parent_id: str | None = None,
    version: str = "v0",
    provenance: str = "natural",
) -> BundleArtifact:
    return BundleArtifact(
        artifact_id,
        label,
        modality,
        media=media.media(modality, path),
        parent_id=parent_id,
        version=version,
        provenance=provenance,
    )


def compile_bundle(
    source: OpenSource,
    bundle_id: str,
    family: str,
    artifacts: Sequence[BundleArtifact],
    *,
    preprocessing: str,
    tier: str = "publisher-annotated",
) -> list[Episode]:
    """Every standard task view the bundle's labels support."""
    bundle = AlignedBundle(
        bundle_id,
        Lineage(
            source.source_id,
            source.revision,
            family,
            frozen_split=frozen_split(family),
            source_uri=f"hf://datasets/{source.repository}",
        ),
        tuple(artifacts),
        source.license,
        preprocessing,
        tier,
    )
    return [compile_task(bundle, view) for view in standard_views(bundle)]


def speech_episodes(media: Preparer, clip: Mapping[str, Any]) -> list[Episode]:
    """LibriSpeech clip -> transcription episode and audio identity episode."""
    source = SOURCES["librispeech_asr"]
    speech = _media(media, _sha(clip["id"], "speech"), "source_speech", "audio", clip["audio"])
    bundle_id = f"librispeech:{clip['id']}"
    episodes = compile_bundle(
        source,
        bundle_id,
        f"librispeech:speaker:{clip['speaker']}",
        (speech, _text(_sha(clip["id"], "text"), "verbatim_transcript", clip["text"])),
        preprocessing="librispeech-clip-v1",
    )
    return [*episodes, identity_episode(speech.media, f"librispeech:speaker:{clip['speaker']}")]


def verbalization_episodes(
    media: Preparer, clip: Mapping[str, Any], reference: Mapping[str, Any] | None
) -> list[Episode]:
    """LibriTTS-R clip -> verbalization (written text -> speech), with a same-speaker voice sample."""
    source = SOURCES["libritts_r"]
    artifacts = [
        _text(_sha(clip["id"], "written"), "written_text", clip["written"]),
        _media(media, _sha(clip["id"], "speech"), "speech", "audio", clip["audio"]),
    ]
    if clip["spoken"] != clip["written"]:
        artifacts.append(_text(_sha(clip["id"], "spoken"), "spoken_form", clip["spoken"]))
    if reference is not None:
        artifacts.append(
            _media(
                media,
                _sha(reference["id"], "voice"),
                "voice_reference",
                "audio",
                reference["audio"],
            )
        )
    return compile_bundle(
        source,
        f"libritts_r:{clip['id']}",
        f"libritts_r:speaker:{clip['speaker']}",
        artifacts,
        preprocessing="libritts-r-clip-v1",
    )


def translation_episodes(media: Preparer, pair: Mapping[str, Any]) -> list[Episode]:
    """FLEURS sentence read in a source language and in English -> direct speech translation."""
    source = SOURCES["fleurs"]
    key = f"{pair['language']}:{pair['sentence']}"
    return compile_bundle(
        source,
        f"fleurs:{key}",
        f"fleurs:sentence:{pair['sentence']}",
        (
            _media(media, _sha(key, "src"), "source_speech", "audio", pair["source_audio"]),
            _text(_sha(key, "srct"), "verbatim_transcript", pair["source_text"]),
            _text(_sha(key, "tgt"), "translated_text", pair["target_text"]),
            _media(media, _sha(key, "tgts"), "translated_speech", "audio", pair["target_audio"]),
        ),
        preprocessing="fleurs-pair-v1",
    )


def image_episodes(media: Preparer, clip: Mapping[str, Any]) -> list[Episode]:
    """DOCCI photo -> image description, image creation and image identity episodes."""
    source = SOURCES["docci"]
    image = _media(media, _sha(clip["id"], "image"), "image", "image", clip["image"])
    episodes = compile_bundle(
        source,
        f"docci:{clip['id']}",
        f"docci:{clip['id']}",
        (
            image,
            _text(_sha(clip["id"], "caption"), "display_caption", clip["description"]),
            _text(
                _sha(clip["id"], "request"), "creation_request", first_sentence(clip["description"])
            ),
        ),
        preprocessing="docci-photo-v1",
    )
    return [*episodes, identity_episode(image.media, f"docci:{clip['id']}")]


def edit_episodes(
    media: Preparer,
    source: OpenSource,
    kind: str,
    clip: Mapping[str, Any],
    *,
    provenance: str,
    tier: str,
) -> list[Episode]:
    """An original recording, a plain-language instruction and the verified edited version.

    ``kind`` is ``image``, ``speech`` or ``av`` (the stream is ``<kind>_edit``).
    """
    modality = {"image": "image", "speech": "audio", "av": "video"}[kind]
    original = _media(
        media, _sha(clip["id"], "original"), f"original_{kind}", modality, clip["original"]
    )
    edited = _media(
        media,
        _sha(clip["id"], "edited"),
        f"edited_{kind}",
        modality,
        clip["edited"],
        parent_id=original.artifact_id,
        version="v1",
        provenance=provenance,
    )
    return compile_bundle(
        source,
        f"{source.source_id}:{kind}-edit:{clip['id']}",
        clip["family"],
        (
            original,
            edited,
            _text(_sha(clip["id"], "edit"), f"{kind}_edit_instruction", clip["instruction"]),
        ),
        preprocessing=f"{source.source_id}-{kind}-edit-v1",
        tier=tier,
    )


def av_episodes(media: Preparer, clip: Mapping[str, Any]) -> list[Episode]:
    """RekaDaily clip -> audiovisual creation, video identity and (procedural) video edit episodes."""
    source = SOURCES["rekadaily_processed"]
    video = _media(media, _sha(clip["id"], "video"), "video", "video", clip["video"])
    episodes = compile_bundle(
        source,
        f"rekadaily:{clip['id']}",
        f"rekadaily:recording:{clip['recording']}",
        (
            video,
            _media(media, _sha(clip["id"], "audio"), "soundtrack", "audio", clip["audio"]),
            _text(_sha(clip["id"], "caption"), "display_caption", clip["caption"]),
            _text(_sha(clip["id"], "request"), "creation_request", clip["request"]),
        ),
        preprocessing="rekadaily-clip-v1",
    )
    return [*episodes, identity_episode(video.media, f"rekadaily:recording:{clip['recording']}")]


# --- readers ------------------------------------------------------------------------------


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()), maxsplit=1)[0]


def _audio_seconds(payload: bytes) -> float:
    import soundfile

    info = soundfile.info(io.BytesIO(payload))
    return info.frames / info.samplerate


def _rows(paths: Sequence[Path], columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as parquet

    for path in paths:
        reader = parquet.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=64, columns=list(columns)):
            yield from batch.to_pylist()


def read_librispeech(parquets: Sequence[Path], out: Path, limit: int) -> Iterator[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for row in _rows(parquets, ("audio", "text", "speaker_id", "id")):
        if kept >= limit:
            return
        payload = row["audio"]["bytes"]
        if _audio_seconds(payload) > MAX_SPEECH_SECONDS:
            continue
        path = out / f"{row['id']}.flac"
        path.write_bytes(payload)
        kept += 1
        yield {"id": row["id"], "speaker": row["speaker_id"], "audio": path, "text": row["text"]}


def read_libritts(parquets: Sequence[Path], out: Path, limit: int) -> Iterator[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for row in _rows(parquets, ("audio", "text_original", "text_normalized", "speaker_id", "id")):
        if kept >= limit:
            return
        payload = row["audio"]["bytes"]
        if _audio_seconds(payload) > MAX_SPEECH_SECONDS:
            continue
        path = out / f"{row['id']}.wav"
        path.write_bytes(payload)
        kept += 1
        yield {
            "id": row["id"],
            "speaker": row["speaker_id"],
            "audio": path,
            "written": row["text_original"],
            "spoken": row["text_normalized"],
        }


def voice_references(clips: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Another utterance by the same speaker for each clip (None when the speaker has one)."""
    by_speaker: dict[str, list[Mapping[str, Any]]] = {}
    for clip in clips:
        by_speaker.setdefault(clip["speaker"], []).append(clip)
    return {
        clip["id"]: group[(group.index(clip) + 1) % len(group)]
        for group in by_speaker.values()
        if len(group) > 1
        for clip in group
    }


def _tsv(path: Path) -> list[list[str]]:
    # Columns: id, file name, raw transcription, normalized transcription, characters, samples, gender.
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE) if row]


def _extract(archive: Path, wanted: Mapping[str, str], out: Path) -> dict[str, Path]:
    """Stream one tar.gz once, writing just the named members (``wanted``: member -> output name)."""
    out.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = wanted.get(member.name)
            if name is None or not member.isfile():
                continue
            path = out / name
            path.write_bytes(tar.extractfile(member).read())
            found[member.name] = path
            if len(found) == len(wanted):
                break
    return found


def read_fleurs(
    root: Path,
    languages: Sequence[str],
    out: Path,
    per_language: int,
    *,
    target: str = "en_us",
    split: str = "train",
) -> Iterator[dict[str, Any]]:
    """Source-language -> English speech pairs of the same FLoRes sentence (``root/<lang>/...``)."""
    max_samples = int(MAX_SPEECH_SECONDS * 16000)

    def rows(language: str) -> dict[str, list[str]]:
        first: dict[str, list[str]] = {}
        for row in _tsv(root / language / f"{split}.tsv"):
            if int(row[5]) <= max_samples:
                first.setdefault(row[0], row)
        return first

    english = rows(target)
    chosen: dict[str, list[tuple[list[str], list[str]]]] = {}
    for language in languages:
        theirs = rows(language)
        shared = sorted(set(theirs) & set(english), key=lambda key: _sha(key))
        chosen[language] = [(theirs[key], english[key]) for key in shared[:per_language]]
    wanted_english = {
        f"{split}/{en[1]}": f"{target}-{en[1]}" for pairs in chosen.values() for _, en in pairs
    }
    english_files = _extract(root / target / f"{split}.tar.gz", wanted_english, out)
    for language, pairs in chosen.items():
        wanted = {f"{split}/{row[1]}": f"{language}-{row[1]}" for row, _ in pairs}
        files = _extract(root / language / f"{split}.tar.gz", wanted, out)
        for row, en in pairs:
            yield {
                "language": language,
                "sentence": row[0],
                "source_audio": files[f"{split}/{row[1]}"],
                "source_text": row[2],
                "target_audio": english_files[f"{split}/{en[1]}"],
                "target_text": en[2],
            }


def read_docci(
    descriptions: Path, archive: Path, out: Path, limit: int, *, max_side: int = 1024
) -> Iterator[dict[str, Any]]:
    """Training-split DOCCI photos, scaled to at most ``max_side`` pixels, with their descriptions."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    wanted = {}
    # JSONL lines end at "\n" only; str.splitlines would also cut at U+2028 or U+0085
    # inside a description.
    for line in descriptions.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] == "train":
            wanted[f"images/{row['image_file']}"] = row
    kept = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            row = wanted.get(member.name.removeprefix("./"))
            if row is None or not member.isfile():
                continue
            image = Image.open(io.BytesIO(tar.extractfile(member).read())).convert("RGB")
            image.thumbnail((max_side, max_side))
            path = out / f"{row['example_id']}.jpg"
            image.save(path, quality=95)
            yield {"id": row["example_id"], "image": path, "description": row["description"]}
            kept += 1
            if kept >= limit:
                return


def read_magicbrush(parquets: Sequence[Path], out: Path, limit: int) -> Iterator[dict[str, Any]]:
    """Edit turns: source image, instruction, edited image. A session's turns share its COCO id."""
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for row in _rows(parquets, ("img_id", "turn_index", "source_img", "instruction", "target_img")):
        if kept >= limit:
            return
        stem = f"{row['img_id']}-{row['turn_index']}"
        paths = {}
        for name, image in (("original", row["source_img"]), ("edited", row["target_img"])):
            paths[name] = out / f"{stem}-{name}.png"
            paths[name].write_bytes(image["bytes"])
        kept += 1
        yield {
            "id": stem,
            "family": f"magicbrush:coco:{row['img_id']}",
            "original": paths["original"],
            "edited": paths["edited"],
            "instruction": row["instruction"],
        }


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def _streams(path: Path) -> set[str]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {stream["codec_type"] for stream in json.loads(result.stdout)["streams"]}


def read_reka(
    archive: Path, out: Path, limit: int, *, min_seconds: float = 1.5, max_seconds: float = 8.0
) -> Iterator[dict[str, Any]]:
    """Short captioned clips from the start of a RekaDaily tar, as small video + soundtrack files.

    Video is scaled to fit 640x640 at 16 frames/s without sound; the soundtrack is 24 kHz mono
    (the speech realizer's rate). Clips without sound or without an activity label are skipped.
    """
    out.mkdir(parents=True, exist_ok=True)
    pending: dict[str, dict[str, Any]] = {}
    kept = 0
    with tarfile.open(archive, "r|") as tar:
        for member in tar:
            stem, _, extension = member.name.rpartition(".")
            if extension == "json":
                pending.setdefault(stem, {})["meta"] = json.loads(tar.extractfile(member).read())
            elif extension == "mp4":
                source = out / f"{stem}.source.mp4"
                source.write_bytes(tar.extractfile(member).read())
                pending.setdefault(stem, {})["source"] = source
            entry = pending.get(stem, {})
            if "meta" not in entry or "source" not in entry:
                continue
            pending.pop(stem)
            meta, source = entry["meta"], entry["source"]
            usable = (
                min_seconds <= meta["duration_s"] <= max_seconds
                and meta.get("activities")
                and meta.get("text")
                and "audio" in _streams(source)
            )
            if usable:
                video, audio = out / f"{stem}.mp4", out / f"{stem}.wav"
                _ffmpeg(
                    "-i",
                    str(source),
                    "-an",
                    "-vf",
                    "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-r",
                    "16",
                    str(video),
                )
                _ffmpeg("-i", str(source), "-vn", "-ac", "1", "-ar", "24000", str(audio))
                kept += 1
                yield {
                    "id": meta["clip_id"],
                    "recording": meta["video_id"],
                    "video": video,
                    "audio": audio,
                    "caption": meta["text"],
                    "request": f"Create a short first-person clip of: {meta['activities'][0]}.",
                }
            source.unlink()
            if kept >= limit:
                return


# --- procedural edits ---------------------------------------------------------------------

SPEECH_EDITS = (
    ("Make the speech 6 decibels louder.", ("-af", "volume=6dB,alimiter=limit=0.97")),
    ("Speak 25 percent faster without changing the pitch.", ("-af", "atempo=1.25")),
    ("Play the recording backwards.", ("-af", "areverse")),
)
VIDEO_EDITS = (
    ("Mirror the video horizontally.", ("-vf", "hflip")),
    ("Convert the video to black and white.", ("-vf", "hue=s=0")),
    ("Play the video at double speed.", ("-vf", "setpts=0.5*PTS")),
    ("Play the video in reverse.", ("-vf", "reverse")),
)


def _procedural(
    options: Sequence[tuple[str, tuple[str, ...]]], clip_id: str, original: Path, out: Path
) -> dict[str, Any]:
    """Apply the one edit this clip id selects (deterministically) and return its clip."""
    instruction, arguments = options[int(_sha(clip_id), 16) % len(options)]
    edited = out / f"{clip_id}.edited{original.suffix}"
    _ffmpeg("-i", str(original), *arguments, str(edited))
    return {"id": clip_id, "original": original, "edited": edited, "instruction": instruction}


def speech_edit(clip: Mapping[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    edit = _procedural(SPEECH_EDITS, clip["id"], Path(clip["audio"]), out)
    return {**edit, "family": f"librispeech:speaker:{clip['speaker']}"}


def video_edit(clip: Mapping[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    edit = _procedural(VIDEO_EDITS, clip["id"], Path(clip["video"]), out)
    return {**edit, "family": f"rekadaily:recording:{clip['recording']}"}
