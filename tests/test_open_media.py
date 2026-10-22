"""Open media datasets -> episodes whose streams are the stage plan's names (no network, tiny fixtures)."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import soundfile
from media_support import prepared_record
from PIL import Image

from megatoken import cloud
from megatoken.data.concordance import TASK_FAMILIES
from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.data.open_media import (
    SOURCES,
    STREAMS,
    CachingPreparer,
    av_episodes,
    edit_episodes,
    first_sentence,
    frozen_split,
    image_episodes,
    read_docci,
    read_fleurs,
    read_librispeech,
    read_libritts,
    read_magicbrush,
    read_reka,
    speech_edit,
    speech_episodes,
    translation_episodes,
    verbalization_episodes,
    video_edit,
    voice_references,
)
from megatoken.data.recipes import stage_recipe

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


class FakePreparer:
    """Stands in for MediaPreparer: a distinct prepared record per distinct file content."""

    def __init__(self, store):
        self.store, self.calls = store, []

    def media(self, modality, path, *, target=True):
        self.calls.append((modality, Path(path).name, target))
        seed = int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:6], 16)
        return prepared_record(self.store, modality, seed=seed, target=target)


@pytest.fixture
def media(tmp_path):
    return CachingPreparer(FakePreparer(TensorAssetStore(tmp_path / "assets")))


def _wave(seed: int, seconds: float = 0.5, rate: int = 16000, fmt: str = "FLAC") -> bytes:
    buffer = io.BytesIO()
    time = np.arange(int(seconds * rate)) / rate
    soundfile.write(buffer, 0.3 * np.sin(2 * np.pi * (200 + 40 * seed) * time), rate, format=fmt)
    return buffer.getvalue()


def _image(seed: int, size=(40, 30), fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (seed * 20 % 255, 80, 160)).save(buffer, format=fmt)
    return buffer.getvalue()


def _streams(episodes) -> dict[str, int]:
    result: dict[str, int] = {}
    for episode in episodes:
        result[episode.stream] = result.get(episode.stream, 0) + 1
    return result


def _generated(episode) -> list[MediaRecord]:
    return [
        MediaRecord.from_bytes(record.payload)
        for record in episode.records
        if record.policy_mask and record.media_type.endswith("media-reference+json")
    ]


def test_split_is_a_stable_function_of_the_family_and_roughly_five_percent():
    assert frozen_split("a") == frozen_split("a")
    share = sum(frozen_split(f"family-{i}") == "validation" for i in range(2000)) / 2000
    assert 0.02 < share < 0.09


def test_librispeech_clip_gives_transcription_and_identity_without_leaking_the_transcript(
    tmp_path, media
):
    table = pa.table(
        {
            "audio": [
                {"bytes": _wave(1), "path": "a.flac"},
                {"bytes": _wave(2, 30.0), "path": "b"},
            ],
            "text": ["HELLO WORLD", "TOO LONG"],
            "speaker_id": [11, 12],
            "id": ["11-1-0", "12-1-0"],
        }
    )
    parquet.write_table(table, tmp_path / "0000.parquet")
    clips = list(read_librispeech([tmp_path / "0000.parquet"], tmp_path / "wav", limit=5))
    assert [clip["id"] for clip in clips] == ["11-1-0"]  # the 30 s utterance is over the limit
    episodes = speech_episodes(media, clips[0])
    assert _streams(episodes) == {"transcription": 1, "audio_identity": 1}
    asr = next(e for e in episodes if e.stream == "transcription")
    assert [r.role for r in asr.records] == ["user", "assistant"]
    assert asr.records[1].payload == b"HELLO WORLD" and asr.records[1].policy_mask
    assert asr.lineage.family_id == "librispeech:speaker:11"
    assert asr.lineage.revision == SOURCES["librispeech_asr"].revision
    assert asr.lineage.frozen_split == frozen_split("librispeech:speaker:11")
    assert json.loads(asr.supervision.annotations_json)["source_license"] == "CC-BY-4.0"


def test_libritts_verbalization_uses_a_same_speaker_voice_sample_and_spoken_form(tmp_path, media):
    rows = [
        ("7-1-0", "7", "Dr. Who paid $5.", "Doctor Who paid five dollars."),
        ("7-1-1", "7", "Plain words.", "Plain words."),
        ("8-1-0", "8", "Alone.", "Alone."),
    ]
    parquet.write_table(
        pa.table(
            {
                "audio": [{"bytes": _wave(i, fmt="WAV"), "path": ""} for i in range(3)],
                "text_original": [r[2] for r in rows],
                "text_normalized": [r[3] for r in rows],
                "speaker_id": [r[1] for r in rows],
                "id": [r[0] for r in rows],
            }
        ),
        tmp_path / "t.parquet",
    )
    clips = list(read_libritts([tmp_path / "t.parquet"], tmp_path / "wav", limit=10))
    references = voice_references(clips)
    assert references["7-1-0"]["id"] == "7-1-1" and "8-1-0" not in references
    first = verbalization_episodes(media, clips[0], references["7-1-0"])
    alone = verbalization_episodes(media, clips[2], None)
    assert _streams(first) == _streams(alone) == {"verbalization": 1}
    plain = [event.track_id for event in first[0].meridians[0].events]
    assert plain == ["written_text", "voice_reference"]
    assert [event.track_id for event in first[0].meridians[1].events] == ["spoken_form", "speech"]
    assert [event.track_id for event in alone[0].meridians[1].events] == ["speech"]
    assert len(_generated(first[0])) == 1  # only the speech is a media target, the sample is input


def test_fleurs_pairs_the_same_sentence_in_two_languages(tmp_path, media):
    def language(name, rows):
        folder = tmp_path / "fleurs" / name
        folder.mkdir(parents=True)
        lines = [f"{i}\t{i}.wav\t{raw}\t{raw.lower()}\tx\t{n}\tMALE" for i, raw, n in rows]
        (folder / "train.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with tarfile.open(folder / "train.tar.gz", "w:gz") as tar:
            for i, _, _ in rows:
                data = _wave(int(i), fmt="WAV")
                info = tarfile.TarInfo(f"train/{i}.wav")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    language(
        "en_us", [("1", "Good morning.", 8000), ("2", "Thank you.", 8000), ("9", "Long.", 400000)]
    )
    language(
        "de_de", [("1", "Guten Morgen.", 8000), ("3", "Nur deutsch.", 8000), ("9", "Lang.", 400000)]
    )
    pairs = list(read_fleurs(tmp_path / "fleurs", ["de_de"], tmp_path / "out", per_language=5))
    assert [(p["sentence"], p["source_text"], p["target_text"]) for p in pairs] == [
        ("1", "Guten Morgen.", "Good morning.")
    ]
    episodes = translation_episodes(media, pairs[0])
    # The same labelled recording also supports plain transcription in the source language.
    assert _streams(episodes) == {"speech_translation": 1, "transcription": 1}
    episode = next(e for e in episodes if e.stream == "speech_translation")
    outputs = [event.track_id for event in episode.meridians[1].events]
    assert outputs == ["source_transcript", "translation", "translated_speech"]
    assert episode.lineage.family_id == "fleurs:sentence:1"
    assert len(_generated(episode)) == 1


def test_docci_photo_gives_description_creation_and_image_identity(tmp_path, media):
    descriptions = tmp_path / "descriptions.jsonlines"
    rows = [
        {
            "example_id": "train_0",
            "split": "train",
            "image_file": "train_0.jpg",
            "description": "A red cup. It sits on a table.",
        },
        {
            "example_id": "test_0",
            "split": "test",
            "image_file": "test_0.jpg",
            "description": "Held out.",
        },
    ]
    descriptions.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with tarfile.open(tmp_path / "images.tar.gz", "w:gz") as tar:
        for name in ("train_0", "test_0"):
            data = _image(3, size=(2000, 1000), fmt="JPEG")
            info = tarfile.TarInfo(f"images/{name}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (clip,) = read_docci(descriptions, tmp_path / "images.tar.gz", tmp_path / "out", limit=5)
    assert max(Image.open(clip["image"]).size) == 1024  # scaled to the largest side
    episodes = image_episodes(media, clip)
    assert _streams(episodes) == {"image_description": 1, "image_creation": 1, "image_identity": 1}
    creation = next(e for e in episodes if e.stream == "image_creation")
    request = creation.records[0]
    assert request.payload == b"A red cup." == first_sentence(clip["description"]).encode()
    assert len(_generated(creation)) == 1 and [r.role for r in creation.records] == [
        "user",
        "assistant",
        "assistant",
    ]


def test_magicbrush_turns_keep_a_session_in_one_family_and_edit_the_exact_original(tmp_path, media):
    parquet.write_table(
        pa.table(
            {
                "img_id": ["100", "100"],
                "turn_index": [1, 2],
                "source_img": [{"bytes": _image(1), "path": ""}, {"bytes": _image(2), "path": ""}],
                "instruction": ["add a hat", "make it red"],
                "target_img": [{"bytes": _image(2), "path": ""}, {"bytes": _image(3), "path": ""}],
            }
        ),
        tmp_path / "t.parquet",
    )
    clips = list(read_magicbrush([tmp_path / "t.parquet"], tmp_path / "out", limit=5))
    assert {clip["family"] for clip in clips} == {"magicbrush:coco:100"}
    episodes = [
        e
        for clip in clips
        for e in edit_episodes(
            media,
            SOURCES["magicbrush"],
            "image",
            clip,
            provenance="natural",
            tier="publisher-annotated",
        )
    ]
    assert _streams(episodes) == {"image_edit": 2}
    first = episodes[0]
    assert first.records[1].payload == b"add a hat"
    assert first.dependencies and first.dependencies[0].kind == "edit"
    # Turn 2's original is turn 1's edited image: the same prepared event, prepared once.
    assert media.inner.calls.count(("image", "100-2-original.png", True)) == 1


@ffmpeg
def test_procedural_speech_and_video_edits_are_deterministic_real_edits(tmp_path, media):
    audio = tmp_path / "clip.flac"
    audio.write_bytes(_wave(1, seconds=1.0))
    clip = {"id": "1-2-3", "speaker": "1", "audio": audio}
    edit = speech_edit(clip, tmp_path / "edits")
    again = speech_edit(clip, tmp_path / "again")
    assert edit["instruction"] == again["instruction"]
    assert soundfile.info(edit["edited"]).frames > 0
    (episode,) = edit_episodes(
        media, SOURCES["librispeech_asr"], "speech", edit, provenance="executed", tier="procedural"
    )
    assert episode.stream == "speech_edit"
    assert episode.records[2].provenance == "executed"
    assert episode.records[2].policy_mask and not episode.records[0].policy_mask

    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=8:duration=1",
            str(video),
        ],
        check=True,
    )
    cut = video_edit({"id": "c", "recording": "r", "video": video}, tmp_path / "videos")
    (episode,) = edit_episodes(
        media, SOURCES["rekadaily_processed"], "av", cut, provenance="executed", tier="procedural"
    )
    assert episode.stream == "av_edit" and episode.lineage.family_id == "rekadaily:recording:r"


@ffmpeg
def test_reka_clips_become_av_creation_and_video_identity(tmp_path, media):
    def clip_files(name, seconds, sound):
        path = tmp_path / f"{name}.mp4"
        inputs = ["-f", "lavfi", "-i", f"testsrc=size=128x96:rate=24:duration={seconds}"]
        if sound:
            inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", *inputs, "-shortest", str(path)], check=True
        )
        return path

    def meta(name, seconds, **extra):
        return {
            "clip_id": name,
            "video_id": "rec-" + name[:1],
            "duration_s": seconds,
            "text": "A dense caption.",
            "activities": ["Core Body Movements: Move Forward"],
            **extra,
        }

    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name, seconds, sound, extra in (
            ("a_clip", 2, True, {}),
            ("b_silent", 2, False, {}),
            ("c_short", 0.5, True, {}),
            ("d_unlabelled", 2, True, {"activities": []}),
            ("e_clip", 2, True, {}),
        ):
            for suffix, data in (
                ("json", json.dumps(meta(name, seconds, **extra)).encode()),
                ("mp4", clip_files(name, seconds, sound).read_bytes()),
            ):
                info = tarfile.TarInfo(f"{name}.{suffix}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    clips = list(read_reka(tar_path, tmp_path / "out", limit=5))
    assert [clip["id"] for clip in clips] == ["a_clip", "e_clip"]
    assert soundfile.info(clips[0]["audio"]).samplerate == 24000
    episodes = av_episodes(media, clips[0])
    assert _streams(episodes) == {"av_creation": 1, "video_identity": 1}
    creation = next(e for e in episodes if e.stream == "av_creation")
    assert creation.records[0].payload.startswith(b"Create a short first-person clip of: Core Body")
    assert len(_generated(creation)) == 2  # the video and its soundtrack


def test_every_stream_this_module_emits_is_in_the_stage_plan():
    assert STREAMS <= set(stage_recipe("P1").mixture)
    assert STREAMS <= TASK_FAMILIES | {"audio_identity", "image_identity", "video_identity"}


def test_cloud_job_inputs_carry_each_pinned_revision():
    job = cloud.validate(
        json.loads((Path(__file__).parents[1] / "configs/cloud/prepare-open-mix.json").read_text())
    )
    assert all(item["sha256"] != "0" * 64 for item in job["inputs"])
    urls = " ".join(item["url"] for item in job["inputs"])
    for source in SOURCES.values():
        if source.source_id != "docci":
            assert source.revision in urls, source.source_id
    assert "storage.googleapis.com/docci/data/docci_images.tar.gz" in urls
