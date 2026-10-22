"""Open media datasets -> episodes whose streams are the stage plan's names (no network, tiny fixtures)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import tarfile
import zipfile
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
    CLIP_READERS,
    CONTAINERS,
    SOURCES,
    STREAMS,
    TASKS,
    CachingPreparer,
    OpenSource,
    Shards,
    blob,
    build_episodes,
    edit_episodes,
    field_of,
    first_sentence,
    frozen_split,
    load_sources,
    probe,
    read_clips,
    source_episodes,
    speech_edit,
    video_edit,
    write_audio,
)
from megatoken.data.recipes import stage_recipe
from megatoken.data.sources import TermsReview

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


def make_source(source_id="unit", task="transcription", reader="parquet", **fields) -> OpenSource:
    return OpenSource(
        source_id=source_id,
        repository=f"org/{source_id}",
        revision="0" * 40,
        reader=reader,
        task=task,
        terms=TermsReview(True, "CC-BY-4.0", "https://example.org", "Reviewer", "2026-10-22"),
        **fields,
    )


def _wave(seed: int, seconds: float = 2.0, rate: int = 16000, fmt: str = "FLAC") -> bytes:
    buffer = io.BytesIO()
    time = np.arange(int(seconds * rate)) / rate
    soundfile.write(buffer, 0.3 * np.sin(2 * np.pi * (200 + 40 * seed) * time), rate, format=fmt)
    return buffer.getvalue()


def _image(seed: int, size=(64, 48), fmt: str = "PNG") -> bytes:
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


def _run(media, source, tmp_path, *, count=10, procedural=0, inputs=None):
    """Every episode of a source whose files sit in ``<inputs>/<source_id>`` (no Hub)."""
    return list(
        source_episodes(
            media, source, count=count, procedural=procedural, work=tmp_path / "work", inputs=inputs
        )
    )


def _tar(path: Path, members: dict[str, bytes], mode: str = "w") -> None:
    with tarfile.open(path, mode) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _local(tmp_path, source_id) -> Path:
    folder = tmp_path / "inputs" / source_id
    folder.mkdir(parents=True)
    return folder


def test_split_is_a_stable_function_of_the_family_and_roughly_five_percent():
    assert frozen_split("a") == frozen_split("a")
    share = sum(frozen_split(f"family-{i}") == "validation" for i in range(2000)) / 2000
    assert 0.02 < share < 0.09


def test_field_paths_follow_dicts_and_lists_and_join_composite_ids():
    row = {"conversations": [{"value": "<image>"}, {"value": "A cat."}], "id": 7, "turn": 2}
    assert field_of(row, "conversations.1.value") == "A cat."
    assert field_of(row, "conversations.5.value") is None
    assert field_of(row, ["id", "turn"]) == "7-2"
    assert field_of(row, ["id", "missing"]) is None
    assert blob({"bytes": b"x", "path": None}) == b"x" and blob({"bytes": None}) is None


def test_shards_list_local_files_in_a_fixed_order_and_slice(tmp_path):
    source = make_source(include=("data/*.parquet", "extra/**/*.tar"))
    folder = _local(tmp_path, "unit")
    for name in ("data/b.parquet", "data/a.parquet", "extra/x/y.tar", "data/skip.txt"):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"1")
    shards = Shards(source, tmp_path / "inputs")
    assert shards.names() == ["data/a.parquet", "data/b.parquet", "extra/x/y.tar"]
    assert Shards(source, tmp_path / "inputs", (1, 2)).names() == ["data/b.parquet"]
    with shards.open("data/a.parquet") as handle:
        assert handle.read() == b"1"


def test_shards_stream_from_the_pinned_hub_revision(tmp_path):
    source = make_source(include=("data/*.bin",))
    prefix = f"datasets/{source.repository}@{source.revision}/"
    (tmp_path / "data").mkdir()
    (tmp_path / "data/a.bin").write_bytes(b"remote")

    class FakeHub:
        def glob(self, pattern, detail=False):
            assert pattern.startswith(prefix)
            return [
                prefix + str(p.relative_to(tmp_path))
                for p in sorted(tmp_path.glob(pattern[len(prefix) :]))
            ]

        def open(self, path, mode, block_size=None):
            assert path.startswith(prefix) and mode == "rb"
            return (tmp_path / path[len(prefix) :]).open("rb")

    shards = Shards(source)
    shards._fs = FakeHub()
    assert shards.names() == ["data/a.bin"]
    with shards.open("data/a.bin") as handle:
        assert handle.read() == b"remote"


def test_speech_with_transcripts_gives_transcription_and_identity_and_cleans_up(tmp_path, media):
    source = make_source(
        "asr",
        columns={"audio": "audio", "text": "text", "id": "id", "family": "speaker_id"},
        include=("*.parquet",),
        options={"max_seconds": 5.0},
    )
    folder = _local(tmp_path, "asr")
    parquet.write_table(
        pa.table(
            {
                "audio": [
                    {"bytes": _wave(1), "path": "a.flac"},
                    {"bytes": _wave(2, 30.0), "path": "b"},  # over the length limit
                    {"bytes": b"not audio", "path": "c"},  # undecodable
                    {"bytes": _wave(3), "path": "d"},
                ],
                "text": ["HELLO WORLD", "TOO LONG", "BROKEN", ""],  # the last has no transcript
                "speaker_id": [11, 12, 13, 14],
                "id": ["11-1-0", "12-1-0", "13-1-0", "14-1-0"],
            }
        ),
        folder / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"transcription": 1, "audio_identity": 1}
    asr = next(e for e in episodes if e.stream == "transcription")
    assert [r.role for r in asr.records] == ["user", "assistant"]
    assert asr.records[1].payload == b"HELLO WORLD" and asr.records[1].policy_mask
    assert asr.lineage.family_id == "asr:11" and asr.lineage.revision == "0" * 40
    assert asr.lineage.frozen_split == frozen_split("asr:11")
    assert json.loads(asr.supervision.annotations_json)["source_license"] == "CC-BY-4.0"
    assert not list((tmp_path / "work").rglob("*.flac"))  # prepared clips are deleted


def test_machine_made_labels_are_recorded_with_their_provenance_and_tier(tmp_path, media):
    source = make_source(
        "pseudo",
        columns={"audio": "audio", "text": "text"},
        include=("*.parquet",),
        options={"provenance": {"text": "teacher"}, "tier": "model-labeled", "clean": "gigaspeech"},
    )
    folder = _local(tmp_path, "pseudo")
    parquet.write_table(
        pa.table(
            {
                "audio": [{"bytes": _wave(i), "path": ""} for i in range(3)],
                "text": [
                    "hello there <COMMA> general <PERIOD>",
                    "<MUSIC> <SIL>",  # a silence/music marker: the segment is dropped
                    "is it <QUESTIONMARK>",
                ],
            }
        ),
        folder / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    asr = [e for e in episodes if e.stream == "transcription"]
    assert [e.records[1].payload for e in asr] == [b"hello there, general.", b"is it?"]
    assert asr[0].records[1].provenance == "teacher"
    assert asr[0].supervision.verification == "model-labeled"


def test_verbalization_hears_the_previous_clip_of_the_same_speaker_and_keeps_spoken_form(
    tmp_path, media
):
    source = make_source(
        "tts",
        "verbalization",
        columns={
            "audio": "audio",
            "written": "text_original",
            "spoken": "text_normalized",
            "id": "id",
            "family": "speaker_id",
        },
        include=("*.parquet",),
    )
    rows = [
        ("7-1-0", "7", "Dr. Who paid $5.", "Doctor Who paid five dollars."),
        ("8-1-0", "8", "Alone.", "Alone."),
        ("7-1-1", "7", "Plain words.", "Plain words."),
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
        _local(tmp_path, "tts") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    speech = [e for e in episodes if e.stream == "verbalization"]
    assert _streams(episodes) == {"verbalization": 3, "audio_identity": 3}
    first, alone, second = ([event.track_id for event in e.meridians[0].events] for e in speech)
    assert first == ["written_text"] and alone == ["written_text"]  # nobody to imitate yet
    assert second == ["written_text", "voice_reference"]
    assert [event.track_id for event in speech[0].meridians[1].events] == ["spoken_form", "speech"]
    assert [event.track_id for event in speech[2].meridians[1].events] == ["speech"]
    assert len(_generated(speech[2])) == 1  # only the speech is a media target, the sample is input


def test_text_translation_of_speech_gives_translation_transcription_and_identity(tmp_path, media):
    source = make_source(
        "ast",
        "translation_text",
        columns={
            "audio": "audio",
            "text": "text",
            "translation": "translation_en",
            "id": "utt_id",
            "family": "original_audio_id",
        },
        include=("*.parquet",),
    )
    parquet.write_table(
        pa.table(
            {
                "audio": [{"bytes": _wave(1), "path": ""}],
                "text": ["Guten Morgen."],
                "translation_en": ["Good morning."],
                "utt_id": ["u1"],
                "original_audio_id": ["video-9"],
            }
        ),
        _local(tmp_path, "ast") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"speech_translation": 1, "transcription": 1, "audio_identity": 1}
    episode = next(e for e in episodes if e.stream == "speech_translation")
    assert [event.track_id for event in episode.meridians[1].events] == [
        "source_transcript",
        "translation",
    ]
    assert episode.lineage.family_id == "ast:video-9"
    assert _generated(episode) == []  # text-only target: no media is generated


def test_fleurs_pairs_the_same_sentence_in_two_languages(tmp_path, media):
    source = make_source(
        "fleurs",
        "translation_pair",
        "fleurs",
        options={"languages": ["de_de"], "target": "en_us"},
    )
    folder = _local(tmp_path, "fleurs")

    def language(name, rows):
        (folder / "data" / name / "audio").mkdir(parents=True)
        lines = [f"{i}\t{i}.wav\t{raw}\t{raw.lower()}\tx\t{n}\tMALE" for i, raw, n in rows]
        (folder / "data" / name / "train.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _tar(
            folder / "data" / name / "audio/train.tar.gz",
            {f"train/{i}.wav": _wave(int(i), fmt="WAV") for i, _, _ in rows},
            "w:gz",
        )

    language(
        "en_us", [("1", "Good morning.", 8000), ("2", "Thank you.", 8000), ("9", "Long.", 400000)]
    )
    language(
        "de_de", [("1", "Guten Morgen.", 8000), ("3", "Nur deutsch.", 8000), ("9", "Lang.", 400000)]
    )
    pairs = list(read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 5))
    assert [(p["id"], p["source_text"], p["target_text"]) for p in pairs] == [
        ("1", "Guten Morgen.", "Good morning.")
    ]
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    # The same labelled recording also supports plain transcription in the source language.
    assert _streams(episodes) == {"speech_translation": 1, "transcription": 1}
    episode = next(e for e in episodes if e.stream == "speech_translation")
    outputs = [event.track_id for event in episode.meridians[1].events]
    assert outputs == ["source_transcript", "translation", "translated_speech"]
    assert episode.lineage.family_id == "fleurs:1"
    assert len(_generated(episode)) == 1


def test_docci_photo_gives_description_creation_and_image_identity(tmp_path, media):
    source = make_source("docci", "captioned_image", "docci", options={"max_side": 1024})
    with pytest.raises(FileNotFoundError, match="not on the Hub"):
        list(read_clips(source, Shards(source), tmp_path / "none", 1))
    folder = _local(tmp_path, "docci")
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
    (folder / "docci_descriptions.jsonlines").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    _tar(
        folder / "docci_images.tar.gz",
        {
            f"images/{name}.jpg": _image(3, size=(2000, 1000), fmt="JPEG")
            for name in ("train_0", "test_0")
        },
        "w:gz",
    )
    (clip,) = read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 5)
    assert max(Image.open(clip["image"]).size) == 1024  # scaled to the largest side
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_description": 1, "image_creation": 1, "image_identity": 1}
    creation = next(e for e in episodes if e.stream == "image_creation")
    request = creation.records[0]
    assert request.payload == b"A red cup." == first_sentence(clip["description"]).encode()
    assert len(_generated(creation)) == 1 and [r.role for r in creation.records] == [
        "user",
        "assistant",
        "assistant",
    ]


def test_captioned_images_from_rows_pick_the_longest_caption_and_one_row_per_image(tmp_path, media):
    source = make_source(
        "coco",
        "captioned_image",
        columns={"image": "image", "caption": "captions", "id": "cocoid"},
        include=("*.parquet",),
        options={"unique": "cocoid", "min_chars": 8, "max_side": 32},
    )
    parquet.write_table(
        pa.table(
            {
                "image": [{"bytes": _image(i, (64, 48), "JPEG"), "path": ""} for i in range(4)],
                "captions": [
                    ["A cat.", "A small cat sits on a mat."],
                    ["Duplicate row of the first image, never read."],
                    ["Tiny"],  # too short
                    ["Two dogs run along a beach."],
                ],
                "cocoid": [1, 1, 2, 3],
            }
        ),
        _local(tmp_path, "coco") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    descriptions = [e for e in episodes if e.stream == "image_description"]
    assert [e.records[1].payload for e in descriptions] == [
        b"A small cat sits on a mat.",
        b"Two dogs run along a beach.",
    ]
    assert _streams(episodes) == {"image_description": 2, "image_creation": 2, "image_identity": 2}


def test_webdataset_samples_group_members_by_key_and_read_json_or_text(tmp_path, media):
    source = make_source(
        "wds",
        "captioned_image",
        "webdataset",
        columns={"image": "jpg", "caption": "json.prompt"},
        include=("*.tar",),
    )
    folder = _local(tmp_path, "wds")
    _tar(
        folder / "a.tar",
        {
            "s0.jpg": _image(1, fmt="JPEG"),
            "s0.json": json.dumps({"prompt": "A lighthouse at dusk."}).encode(),
            "s1.jpg": _image(2, fmt="JPEG"),
            "s1.json": json.dumps({"other": 1}).encode(),  # no caption: skipped
            "s2.json": json.dumps({"prompt": "Two boats on a lake."}).encode(),
            "s2.jpg": _image(3, fmt="JPEG"),
        },
    )
    records = list(CONTAINERS["webdataset"](source, Shards(source, tmp_path / "inputs")))
    assert [r["__key__"] for r in records] == ["s0", "s1", "s2"]
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_description": 2, "image_creation": 2, "image_identity": 2}


def test_archive_members_join_their_table_row_and_a_partner_member(tmp_path):
    source = make_source(
        "pairs",
        "video_edit",
        "archive",
        include=("edited/*.zip",),
        options={
            "table": {"path": "table.csv", "key": "{video}", "columns": ["instruct"]},
            "partner": "source/{name}",
        },
    )
    folder = _local(tmp_path, "pairs")
    (folder / "edited").mkdir()
    (folder / "source").mkdir()
    (folder / "table.csv").write_text("video,instruct\na.mp4,make it red\nb.mp4,make it blue\n")
    for name, files in (("edited", ("a.mp4", "b.mp4", "z.mp4")), ("source", ("a.mp4", "z.mp4"))):
        with zipfile.ZipFile(folder / name / "part_1.zip", "w") as archive:
            for file in files:
                archive.writestr(f"clips/{file}", f"{name}:{file}".encode())
    records = list(CONTAINERS["archive"](source, Shards(source, tmp_path / "inputs")))
    # b.mp4 has no source video and z.mp4 no instruction: only a.mp4 is a complete pair.
    assert [(r["__key__"], r["member"], r["partner"], r["instruct"]) for r in records] == [
        ("a.mp4", b"edited:a.mp4", b"source:a.mp4", "make it red")
    ]


def test_archive_of_images_matches_members_to_a_jsonl_table(tmp_path, media):
    source = make_source(
        "dense",
        "captioned_image",
        "archive",
        columns={"image": "member", "caption": "caption", "id": "__key__"},
        include=("images/*.zip",),
        options={
            "table": {"path": "captions.jsonl", "key": "{image_id}.png", "columns": ["caption"]}
        },
    )
    folder = _local(tmp_path, "dense")
    (folder / "images").mkdir()
    (folder / "captions.jsonl").write_text(
        "\n".join(
            json.dumps({"image_id": str(i), "caption": f"Caption number {i}, fairly long."})
            for i in (1, 2)
        )
    )
    with zipfile.ZipFile(folder / "images/0.zip", "w") as archive:
        for i in (1, 2, 3):
            archive.writestr(f"{i}.png", _image(i))
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_description": 2, "image_creation": 2, "image_identity": 2}


def test_uncaptioned_images_give_only_image_identity_and_skip_rows_without_one(tmp_path, media):
    source = make_source(
        "pictures",
        "picture",
        columns={"image": "image", "id": "id"},
        include=("*.parquet",),
    )
    parquet.write_table(
        pa.table(
            {
                "image": [{"bytes": _image(1), "path": ""}, None, {"bytes": b"junk", "path": ""}],
                "id": [1, 2, 3],
            }
        ),
        _local(tmp_path, "pictures") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_identity": 1}


def test_image_edits_keep_a_session_in_one_family_and_edit_the_exact_original(tmp_path, media):
    source = make_source(
        "magic",
        "image_edit",
        columns={
            "original": "source_img",
            "edited": "target_img",
            "instruction": "instruction",
            "id": ["img_id", "turn_index"],
            "family": "img_id",
            "guide": "guide",
        },
        include=("*.parquet",),
        options={
            "absent": ["guide"],
            "provenance": {"media": "teacher"},
            "tier": "model-generated",
        },
    )
    parquet.write_table(
        pa.table(
            {
                "img_id": ["100", "100", "101", "102"],
                "turn_index": [1, 2, 1, 1],
                "source_img": [{"bytes": _image(i), "path": ""} for i in (1, 2, 3, 4)],
                "instruction": ["add a hat", "make it red", "guided edit", "size mismatch"],
                "target_img": [
                    {"bytes": _image(2), "path": ""},
                    {"bytes": _image(3), "path": ""},
                    {"bytes": _image(4), "path": ""},
                    {"bytes": _image(5, size=(40, 40)), "path": ""},
                ],
                "guide": [
                    None,
                    {"bytes": None, "path": None},
                    {"bytes": _image(9), "path": ""},
                    None,
                ],
            }
        ),
        _local(tmp_path, "magic") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_edit": 2}
    first = episodes[0]
    assert first.records[1].payload == b"add a hat"
    assert first.lineage.family_id == "magic:100"
    assert first.dependencies and first.dependencies[0].kind == "edit"
    assert first.records[2].provenance == "teacher"
    assert first.supervision.verification == "model-generated"


def test_long_recordings_of_any_sound_are_cropped_to_a_central_excerpt(tmp_path, media):
    source = make_source(
        "music",
        "sound",
        columns={"audio": "audio", "id": "id"},
        include=("*.parquet",),
        options={"crop": True, "max_seconds": 2.0},
    )
    parquet.write_table(
        pa.table(
            {
                "audio": [
                    {"bytes": _wave(1, 6.0), "path": ""},
                    {"bytes": _wave(2, 0.1), "path": ""},
                ],
                "id": ["long", "tiny"],
            }
        ),
        _local(tmp_path, "music") / "0.parquet",
    )
    (clip,) = read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 5)
    assert soundfile.info(clip["audio"]).duration == pytest.approx(2.0, abs=0.01)
    assert _streams(_run(media, source, tmp_path, inputs=tmp_path / "inputs")) == {
        "audio_identity": 1
    }
    assert (
        write_audio(_wave(1, 6.0), tmp_path / "x", "s", max_seconds=2.0) is None
    )  # no crop: over the limit


def _video(path: Path, seconds: float, *, sound: bool, size="128x96") -> Path:
    inputs = ["-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={seconds}"]
    if sound:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", *inputs, "-shortest", str(path)], check=True
    )
    return path


@ffmpeg
def test_captioned_clips_with_sound_become_av_creation_and_video_identity(tmp_path, media):
    source = make_source(
        "reka",
        "av_caption",
        "webdataset",
        columns={
            "video": "mp4",
            "caption": "json.text",
            "request": "json.activities.0",
            "id": "json.clip_id",
            "family": "json.video_id",
        },
        include=("*.tar",),
        options={
            "request": "Create a short first-person clip of: {request}.",
            "min_seconds": 1.5,
            "max_seconds": 8.0,
        },
    )

    def meta(name, **extra):
        return {
            "clip_id": name,
            "video_id": "rec-" + name[:1],
            "text": "A dense caption.",
            "activities": ["Core Body Movements: Move Forward"],
            **extra,
        }

    members = {}
    for name, seconds, sound, extra in (
        ("a_clip", 2, True, {}),
        ("b_silent", 2, False, {}),
        ("c_short", 0.5, True, {}),
        ("d_unlabelled", 2, True, {"activities": []}),
        ("e_clip", 2, True, {}),
    ):
        members[f"{name}.json"] = json.dumps(meta(name, **extra)).encode()
        members[f"{name}.mp4"] = _video(tmp_path / f"{name}.mp4", seconds, sound=sound).read_bytes()
    _tar(_local(tmp_path, "reka") / "shard.tar", members)
    clips = list(read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 5))
    assert [clip["id"] for clip in clips] == ["a_clip", "e_clip"]
    assert soundfile.info(clips[0]["audio"]).samplerate == 24000
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs", procedural=1)
    assert _streams(episodes) == {"av_creation": 2, "video_identity": 2, "av_edit": 1}
    creation = next(e for e in episodes if e.stream == "av_creation")
    assert creation.records[0].payload.startswith(b"Create a short first-person clip of: Core Body")
    assert len(_generated(creation)) == 2  # the video and its soundtrack


@ffmpeg
def test_annotated_long_videos_are_cut_at_their_described_activities(tmp_path, media):
    source = make_source(
        "fine",
        "av_scenes",
        columns={"video": "mp4", "meta": "json", "id": "json.name", "family": "json.name"},
        include=("*.parquet",),
        options={"clips_per_video": 2, "min_seconds": 1.0, "max_seconds": 2.0},
    )

    def activity(start, end, text):
        return {
            "description": text,
            "timestamp": {"start_timestamp": start, "end_timestamp": end},
        }

    meta = {
        "name": "yt-1",
        "content_metadata": {
            "scenes": [
                {
                    "title": "Cooking pasta",
                    "activities": [
                        activity("00:00:00.000", "00:00:03.000", "A cook stirs a pot."),
                        activity("00:00:03.000", "00:00:03.400", "Too short."),
                        activity("00:00:04.000", "00:00:06.000", "The cook plates the pasta."),
                    ],
                }
            ]
        },
    }
    clip = _video(tmp_path / "long.mp4", 7, sound=True).read_bytes()
    silent = _video(tmp_path / "silent.mp4", 7, sound=False).read_bytes()
    parquet.write_table(
        pa.table({"mp4": [clip, silent], "json": [meta, {**meta, "name": "yt-2"}]}),
        _local(tmp_path, "fine") / "0.parquet",
    )
    clips = list(read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 10))
    assert [c["id"] for c in clips] == ["yt-1:0.0", "yt-1:4.0"]
    assert clips[0]["caption"] == "A cook stirs a pot."
    assert clips[0]["request"] == "Create a short clip of: Cooking pasta."
    assert {c["family"] for c in clips} == {"fine:yt-1"}
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"av_creation": 2, "video_identity": 2}


@ffmpeg
def test_silent_clips_and_instructed_video_edits(tmp_path, media):
    source = make_source(
        "open",
        "video",
        "archive",
        columns={"video": "member", "id": "__key__"},
        include=("*.zip",),
        options={"table": {"path": "t.csv", "key": "{video}", "columns": []}, "max_seconds": 1.0},
    )
    folder = _local(tmp_path, "open")
    (folder / "t.csv").write_text("video\nc.mp4\n")
    with zipfile.ZipFile(folder / "0.zip", "w") as archive:
        archive.writestr("dir/c.mp4", _video(tmp_path / "c.mp4", 3, sound=False).read_bytes())
    (clip,) = read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 5)
    assert 0.9 < probe(clip["video"])[0] < 1.2
    assert _streams(_run(media, source, tmp_path, inputs=tmp_path / "inputs")) == {
        "video_identity": 1
    }

    edits = make_source(
        "edits",
        "video_edit",
        "archive",
        columns={
            "edited": "member",
            "original": "partner",
            "instruction": "instruct",
            "id": "__key__",
        },
        include=("edited/*.zip",),
        options={
            "table": {"path": "t.csv", "key": "{video}", "columns": ["instruct"]},
            "partner": "source/{name}",
            "max_seconds": 1.0,
        },
    )
    folder = _local(tmp_path, "edits")
    (folder / "edited").mkdir()
    (folder / "source").mkdir()
    (folder / "t.csv").write_text("video,instruct\nv.mp4,make it grey\n")
    for name, size in (("edited", "64x48"), ("source", "64x48")):
        with zipfile.ZipFile(folder / name / "0.zip", "w") as archive:
            archive.writestr(
                "v.mp4", _video(tmp_path / f"{name}.mp4", 2, sound=False, size=size).read_bytes()
            )
    episodes = _run(media, edits, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"av_edit": 1}
    assert episodes[0].records[1].payload == b"make it grey"
    assert episodes[0].dependencies[0].kind == "edit"


@ffmpeg
def test_procedural_speech_and_video_edits_are_deterministic_real_edits(tmp_path, media):
    source = make_source("proc")
    audio = tmp_path / "clip.flac"
    audio.write_bytes(_wave(1, seconds=1.0))
    clip = {"id": "1-2-3", "family": "proc:1", "audio": audio}
    edit = speech_edit(clip, tmp_path / "edits")
    again = speech_edit(clip, tmp_path / "again")
    assert edit["instruction"] == again["instruction"]
    assert soundfile.info(edit["edited"]).frames > 0
    (episode,) = edit_episodes(
        media, source, "speech", edit, provenance="executed", tier="procedural"
    )
    assert episode.stream == "speech_edit" and episode.lineage.family_id == "proc:1"
    assert episode.records[2].provenance == "executed"
    assert episode.records[2].policy_mask and not episode.records[0].policy_mask

    video = _video(tmp_path / "clip.mp4", 1, sound=False, size="64x48")
    cut = video_edit({"id": "c", "family": "proc:r", "video": video}, tmp_path / "videos")
    (episode,) = edit_episodes(media, source, "av", cut, provenance="executed", tier="procedural")
    assert episode.stream == "av_edit" and episode.lineage.family_id == "proc:r"


def test_procedural_edits_follow_the_first_clips_of_speech_sources(tmp_path, media):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    source = make_source(
        "asr", columns={"audio": "audio", "text": "text", "id": "id"}, include=("*.parquet",)
    )
    parquet.write_table(
        pa.table(
            {
                "audio": [{"bytes": _wave(i), "path": ""} for i in range(3)],
                "text": ["one", "two", "three"],
                "id": ["a", "b", "c"],
            }
        ),
        _local(tmp_path, "asr") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs", procedural=2)
    assert _streams(episodes) == {"transcription": 3, "audio_identity": 3, "speech_edit": 2}
    assert not list((tmp_path / "work").rglob("*.edited.*"))


def test_build_episodes_stops_at_the_requested_count(tmp_path, media):
    source = make_source(
        "asr", columns={"audio": "audio", "text": "text", "id": "id"}, include=("*.parquet",)
    )
    parquet.write_table(
        pa.table(
            {
                "audio": [{"bytes": _wave(i), "path": ""} for i in range(4)],
                "text": list("abcd"),
                "id": list("wxyz"),
            }
        ),
        _local(tmp_path, "asr") / "0.parquet",
    )
    clips = list(read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 2))
    assert [clip["id"] for clip in clips] == ["w", "x"]
    assert list(read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 0)) == []
    assert _streams(build_episodes(media, source, clips, work=tmp_path)) == {
        "transcription": 2,
        "audio_identity": 2,
    }


def test_unapproved_sources_are_refused(tmp_path, media):
    refused = dataclasses.replace(make_source(), terms=TermsReview(False, "CC-BY-4.0"))
    with pytest.raises(ValueError, match="unapproved"):
        list(source_episodes(media, refused, count=1, work=tmp_path))


def test_every_stream_this_module_emits_is_in_the_stage_plan():
    assert STREAMS <= set(stage_recipe("P1").mixture)
    assert STREAMS <= TASK_FAMILIES | {"audio_identity", "image_identity", "video_identity"}


def test_manifest_sources_are_pinned_licensed_approved_and_readable():
    assert len(SOURCES) >= 40
    for source in SOURCES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", source.revision), source.source_id
        assert source.license and source.license not in ("", "unreviewed"), source.source_id
        assert source.terms.license_url.startswith("https://"), source.source_id
        # Approval is the maintainer's decision, recorded with who and when.
        source.terms.require_approval()
        assert source.task in TASKS and source.reader in {*CONTAINERS, *CLIP_READERS}
        assert set(source.streams) <= STREAMS and set(source.streams) <= set(
            stage_recipe("P1").mixture
        )
        if source.reader in CONTAINERS:
            assert source.include and TASKS[source.task].materialize, source.source_id
            roles = set(source.columns)
            needed = {
                "transcription": {"audio", "text"},
                "verbalization": {"audio", "written"},
                "translation_text": {"audio", "text", "translation"},
                "captioned_image": {"image", "caption"},
                "picture": {"image"},
                "image_edit": {"original", "edited", "instruction"},
                "sound": {"audio"},
                "av_caption": {"video", "caption"},
                "av_scenes": {"video", "meta"},
                "video": {"video"},
                "video_edit": {"original", "edited", "instruction"},
            }[source.task]
            assert needed <= roles, source.source_id
        if source.reader == "archive":
            assert "table" in source.options, source.source_id
        assert source.inventory.get("basis"), source.source_id


def test_every_stage_plan_media_stream_has_a_source():
    media_streams = {
        name
        for name in stage_recipe("P1").mixture
        if name.endswith(("_identity", "_creation", "_edit", "_translation", "_description"))
        or name in {"transcription", "verbalization"}
    }
    assert media_streams <= {stream for source in SOURCES.values() for stream in source.streams}


def test_the_manifest_loads_from_any_path_and_rejects_duplicates(tmp_path):
    raw = json.loads((Path(__file__).parents[1] / "configs/data/sources.media.json").read_text())
    path = tmp_path / "m.json"
    path.write_text(json.dumps({**raw, "sources": raw["sources"][:2]}))
    assert list(load_sources(path)) == [s["source_id"] for s in raw["sources"][:2]]
    path.write_text(json.dumps({**raw, "sources": [raw["sources"][0]] * 2}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_sources(path)
    path.write_text(json.dumps({**raw, "schema_version": 2}))
    with pytest.raises(ValueError, match="schema"):
        load_sources(path)


CLOUD = Path(__file__).parents[1] / "configs/cloud"
MEDIA_JOBS = sorted(path.stem for path in CLOUD.glob("prepare-media-*.json"))


def _job(name):
    return cloud.validate(json.loads((CLOUD / f"{name}.json").read_text()))


def _counts(job, flag="count"):
    return {
        key: int(value) for key, value in re.findall(rf"--{flag} (\w+)=(\d+)", job["command"][-1])
    }


def test_the_pilot_job_streams_media_from_the_hub_except_the_docci_files():
    job = _job("prepare-open-mix")
    assert all(item["sha256"] != "0" * 64 for item in job["inputs"])
    assert {item["path"].split("/")[1] for item in job["inputs"]} == {"docci", "hotpot"}
    assert "storage.googleapis.com/docci/data/docci_images.tar.gz" in " ".join(
        item["url"] for item in job["inputs"]
    )


def test_every_media_source_is_built_by_exactly_one_scale_job():
    built: dict[str, str] = {}
    for name in MEDIA_JOBS:
        job = _job(name)
        assert job["resources"]["gpu"] == 1 and job["outputs"] == ["out/prepared"]
        for source_id, count in _counts(job).items():
            assert source_id in SOURCES, (name, source_id)
            assert source_id not in built, f"{source_id} is built by {built[source_id]} and {name}"
            built[source_id] = name
            assert count > 0
            assert _counts(job, "procedural").get(source_id, 0) <= count
            # First-run caps stay within what the source holds.
            items = SOURCES[source_id].inventory.get("items")
            assert items is None or count <= items, (name, source_id)
        assert set(_counts(job, "procedural")) <= set(_counts(job)), name
    assert set(built) == set(SOURCES)


def test_job_option_overrides_name_real_sources_and_languages():
    for name in MEDIA_JOBS:
        command = _job(name)["command"][-1]
        for source_id, key, value in re.findall(r"--option '(\w+)\.(\w+)=(.*?)'", command):
            assert source_id in _counts(_job(name)) and key in SOURCES[source_id].options
            if key == "languages":
                assert set(json.loads(value)) <= set(SOURCES[source_id].options["languages"])


def test_sources_that_need_the_hub_account_to_accept_terms_run_in_their_own_job():
    for name in MEDIA_JOBS:
        sources = set(_counts(_job(name)))
        gated = {s for s in sources if SOURCES[s].gated}
        assert gated == (sources if name.endswith("-gated") else set()), name
    assert {s for s, source in SOURCES.items() if source.gated} >= {"finevideo", "emilia"}


def test_the_docci_files_are_job_inputs_of_the_images_job():
    job = _job("prepare-media-images")
    assert [item["path"] for item in job["inputs"]] == [
        "inputs/docci/docci_descriptions.jsonlines",
        "inputs/docci/docci_images.tar.gz",
    ]
    assert all(len(item["sha256"]) == 64 and item["sha256"] != "0" * 64 for item in job["inputs"])


@pytest.fixture
def prepare_script(monkeypatch):
    """scripts/prepare_open_media.py with the GPU preparer swapped for the file-hash fake."""
    path = Path(__file__).parents[1] / "scripts/prepare_open_media.py"
    spec = importlib.util.spec_from_file_location("prepare_open_media", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    class Fake:
        def __init__(self, packs, root, device="cpu"):
            self.inner = FakePreparer(TensorAssetStore(root))

        def media(self, modality, path, *, target=True):
            return self.inner.media(modality, path, target=target)

    monkeypatch.setattr(script, "MediaPreparer", Fake)
    return script


def test_the_prepare_script_builds_an_immutable_root_from_counts(
    tmp_path, monkeypatch, capsys, prepare_script
):
    shard = _local(tmp_path, "librispeech_asr") / "all/train.clean.100"
    shard.mkdir(parents=True)
    parquet.write_table(
        pa.table(
            {
                "audio": [{"bytes": _wave(i), "path": ""} for i in range(6)],
                "text": [f"TEXT {i}" for i in range(6)],
                "id": [f"1-{i}" for i in range(6)],
                "speaker_id": [1, 1, 2, 2, 3, 3],
            }
        ),
        shard / "0000.parquet",
    )
    root = tmp_path / "root"
    arguments = [
        *("--packs", "unused", "--codec", "byte", "--device", "cpu"),
        *("--work", str(tmp_path / "scratch"), "--inputs", str(tmp_path / "inputs")),
        *("--count", "librispeech_asr=5"),
    ]

    def run(*extra):
        monkeypatch.setattr("sys.argv", ["prepare_open_media", *arguments, *extra])
        prepare_script.main()

    run("--root", str(root))
    counts = json.loads((root / "manifest.json").read_text())["counts"]
    assert counts["stream:transcription"] == counts["stream:audio_identity"] == 5
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["transcription"] == 5
    with pytest.raises(SystemExit):  # prepared roots are immutable
        run("--root", str(root))
    with pytest.raises(SystemExit):  # unknown source
        run("--root", str(tmp_path / "new"), "--count", "nonexistent=1")
    with pytest.raises(SystemExit):  # malformed slice
        run("--root", str(tmp_path / "new"), "--shard-slice", "3/2")
