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
    identity_episode,
    is_web,
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


def _turns(episode):
    """(role, policy_mask, kind) of each record of an edit episode, text shortened to its words."""
    return [
        (
            r.role,
            r.policy_mask,
            "media" if r.media_type.endswith("media-reference+json") else r.payload.decode(),
        )
        for r in episode.records
    ]


def test_a_three_step_edit_session_is_one_multi_turn_episode(tmp_path, media):
    source = make_source(
        "sessions",
        "image_edit_chain",
        columns={
            "original": "source_img",
            "edited": "target_img",
            "instruction": "instruction",
            "id": "img_id",
            "family": "img_id",
        },
        include=("*.parquet",),
        options={
            "session": {"by": "img_id", "order": "turn_index"},
            "provenance": {"media": "teacher"},
            "tier": "model-generated",
        },
    )
    # Session 100 has three turns (each turn edits the previous result), delivered out of
    # order; session 101 has one; session 102's second result has another size, which ends it.
    chain = [_image(i) for i in (1, 2, 3, 4)]
    small = _image(5, size=(40, 40))
    sources = [chain[1], chain[0], chain[2], chain[0], chain[0], chain[1]]
    results = [chain[2], chain[1], chain[3], chain[3], chain[1], small]
    parquet.write_table(
        pa.table(
            {
                "img_id": ["100", "100", "100", "101", "102", "102"],
                "turn_index": [2, 1, 3, 1, 1, 2],
                "source_img": [{"bytes": data, "path": ""} for data in sources],
                "target_img": [{"bytes": data, "path": ""} for data in results],
                "instruction": ["make it red", "add a hat", "add a dog", "remove it", "blur", "x"],
            }
        ),
        _local(tmp_path, "sessions") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_edit": 3}
    long = next(e for e in episodes if e.lineage.family_id == "sessions:100")
    assert _turns(long) == [
        ("user", False, "media"),
        ("user", False, "add a hat"),
        ("assistant", True, "media"),
        ("user", False, "make it red"),
        ("assistant", True, "media"),
        ("user", False, "add a dog"),
        ("assistant", True, "media"),
    ]
    assert [len(m.events) for m in long.meridians] == [2, 1, 1, 1, 1, 1]
    # Each result is the exact parent of the next: three edit dependencies in a line.
    ids = [r.record_id for r in long.records]
    assert [(d.writer_id, d.reader_id) for d in long.dependencies] == [
        (ids[0], ids[2]),
        (ids[2], ids[4]),
        (ids[4], ids[6]),
    ]
    events = [e for m in long.meridians for e in m.events if e.role == "assistant"]
    assert [e.request_revision for e in events] == [1, 2, 3]
    assert [len(e.reference_event_ids) for e in events] == [1, 1, 1]
    assert events[1].reference_event_ids == (ids[2],)
    assert long.records[2].provenance == "teacher"
    assert long.supervision.verification == "model-generated"
    lengths = sorted(sum(r.policy_mask for r in e.records) for e in episodes)
    assert lengths == [1, 1, 3]  # the size mismatch ended session 102 after its first turn


def test_a_row_holding_a_whole_edit_chain_and_the_video_chain(tmp_path, media):
    source = make_source(
        "wholechain",
        "image_edit_chain",
        columns={"images": "images", "instruction": "turns", "id": "chain_id"},
        include=("*.parquet",),
        options={"max_steps": 2},
    )
    parquet.write_table(
        pa.table(
            {
                "chain_id": ["c1"],
                "images": [[{"bytes": _image(i), "path": ""} for i in (1, 2, 3, 4)]],
                "turns": [["add snow", "", "add a sun"]],
            }
        ),
        _local(tmp_path, "wholechain") / "0.parquet",
    )
    # The empty second request ends the usable chain after one turn.
    (episode,) = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert sum(r.policy_mask for r in episode.records) == 1
    assert episode.records[1].payload == b"add snow"


@ffmpeg
def test_video_edit_sessions_trim_every_version_alike(tmp_path, media):
    source = make_source(
        "vchain",
        "video_edit_chain",
        columns={"images": "videos", "instruction": "turns", "id": "chain_id"},
        include=("*.parquet",),
        options={"max_seconds": 1.0},
    )
    versions = [
        {"bytes": _video(tmp_path / f"{i}.mp4", 2 + i * 0.1, sound=False).read_bytes()}
        for i in range(3)
    ]
    parquet.write_table(
        pa.table({"chain_id": ["v"], "videos": [versions], "turns": [["grey", "mirror"]]}),
        _local(tmp_path, "vchain") / "0.parquet",
    )
    (episode,) = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert episode.stream == "av_edit"
    assert sum(r.policy_mask for r in episode.records) == 2
    assert [d.kind for d in episode.dependencies] == ["edit", "edit"]
    assert not list((tmp_path / "work").rglob("*.mp4"))


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
    assert _streams(episodes) == {
        "av_creation": 2,
        "video_identity": 2,
        "av_edit": 1,
        "edit_plan": 1,
    }
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
    streams = _streams(episodes)
    assert streams.pop("edit_plan", 0) <= 2  # only a sample of speech edits also give a plan
    assert streams == {"transcription": 3, "audio_identity": 3, "speech_edit": 2}
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
        if is_web(source):  # a published site: its table files are pinned by digest
            assert source.revision.startswith("web-"), source.source_id
            assert all(
                re.fullmatch(r"[0-9a-f]{64}", source.options["pins"].get(name, ""))
                for name in source.include
            ), source.source_id
        else:
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
                "image_edit_chain": {"instruction"},
                "video_edit_chain": {"instruction"},
                "image_questions": {"image"},
                "captioned_audio": {"audio", "caption"},
                "captioned_video": {"video", "caption"},
                "video_questions": {"video"},
                "audio_questions": {"audio"},
                "edit_judging": {"original", "candidates", "instruction"},
                "image_edit_plan": {"original", "edited", "instruction", "plan"},
            }[source.task]
            assert needed <= roles, source.source_id
            if source.task.endswith("_chain"):  # all versions in one field, or per-turn pairs
                assert "images" in roles or {"original", "edited"} <= roles, source.source_id
            if source.task == "image_questions":
                assert "turns" in roles or {"question", "answer"} <= roles, source.source_id
        if source.reader == "archive":
            assert "table" in source.options, source.source_id
        assert source.inventory.get("basis"), source.source_id


def test_every_stage_plan_media_stream_has_a_source():
    media_streams = {
        name
        for name in stage_recipe("P1").mixture
        if name.endswith(("_identity", "_creation", "_edit", "_translation", "_description"))
        or name in {"transcription", "verbalization", "edit_judging", "edit_plan"}
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


def test_the_same_media_through_two_sources_is_two_mergeable_episodes(tmp_path: Path) -> None:
    from megatoken.data import DedupStore
    from megatoken.generation.codec import ByteCodec

    record = prepared_record(TensorAssetStore(tmp_path), "image", seed=3)
    first, second = identity_episode(record, "coco:1"), identity_episode(record, "coco-recap:1")
    assert first.episode_id != second.episode_id
    with DedupStore(tmp_path / "dedup.sqlite") as store, store.transaction():
        assert store.add(first, ByteCodec.byte_fallback()) == "accepted_candidate"
        assert store.add(second, ByteCodec.byte_fallback()) == "duplicate"


def test_a_reused_clip_id_with_new_content_gives_new_episodes(tmp_path, media):
    from megatoken.data import DedupStore
    from megatoken.generation.codec import ByteCodec

    source = make_source(
        "asr",
        columns={"audio": "audio", "text": "text", "id": "id", "family": "speaker_id"},
        include=("*.parquet",),
    )
    folder = _local(tmp_path, "asr")
    audio = {"bytes": _wave(1), "path": "a.flac"}
    parquet.write_table(
        pa.table(
            {
                "audio": [audio, audio, audio],
                "text": ["HELLO WORLD", "HELLO THERE", "HELLO WORLD"],
                "speaker_id": [11, 11, 11],
                "id": ["11-1-0", "11-1-0", "11-1-0"],
            }
        ),
        folder / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    with DedupStore(tmp_path / "dedup.sqlite") as store, store.transaction():
        verdicts = [store.add(episode, ByteCodec.byte_fallback()) for episode in episodes]
    assert "accepted_candidate" in verdicts
    transcripts = {e.episode_id for e in episodes if e.stream == "transcription"}
    assert len(transcripts) == 2  # the repeated row merges; the new caption does not


def test_a_conversation_about_an_image_is_one_multi_turn_episode_in_every_stored_layout(
    tmp_path, media
):
    cauldron = make_source(
        "pairs",
        "image_questions",
        columns={"image": "images", "turns": "texts", "id": "uid"},
        include=("*.parquet",),
        options={"format": "pairs", "max_turns": 2},
    )
    llava = make_source(
        "chat",
        "image_questions",
        columns={"image": "image", "turns": "conversations", "id": "uid"},
        include=("*.parquet",),
        options={"format": "messages"},
    )
    lists = make_source(
        "lists",
        "image_questions",
        columns={"image": "image", "question": "questions", "answer": "answers", "id": "uid"},
        include=("*.parquet",),
        options={"format": "parallel"},
    )
    parquet.write_table(
        pa.table(
            {
                "uid": ["a", "b", "c"],
                # The second row shows two images and is skipped.
                "images": [
                    [{"bytes": _image(1), "path": ""}],
                    [{"bytes": _image(2), "path": ""}, {"bytes": _image(3), "path": ""}],
                    [{"bytes": _image(4), "path": ""}],
                ],
                "texts": [
                    [
                        {"user": "What is shown?", "assistant": "A red square."},
                        {"user": "What colour?", "assistant": "Red."},
                        {"user": "Third?", "assistant": "Cut off by max_turns."},
                    ],
                    [{"user": "Q", "assistant": "A"}],
                    [{"user": "Where?", "assistant": ""}],  # no answer: nothing to learn
                ],
            }
        ),
        _local(tmp_path, "pairs") / "0.parquet",
    )
    parquet.write_table(
        pa.table(
            {
                "uid": ["a"],
                "image": [{"bytes": _image(5), "path": ""}],
                "conversations": [
                    [
                        {"from": "human", "value": "<image>\nWhat is here?"},
                        {"from": "gpt", "value": "A circle."},
                        {"from": "human", "value": "Size?"},
                        {"from": "gpt", "value": "Small."},
                        {"from": "human", "value": "Dangling question"},
                    ]
                ],
            }
        ),
        _local(tmp_path, "chat") / "0.parquet",
    )
    parquet.write_table(
        pa.table(
            {
                "uid": ["a"],
                "image": [{"bytes": _image(6), "path": ""}],
                "questions": [["How many?", "Which side?"]],
                "answers": [["Two.", "Left."]],
            }
        ),
        _local(tmp_path, "lists") / "0.parquet",
    )
    first = _run(media, cauldron, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(first) == {"image_description": 1, "image_identity": 1}
    chat = next(e for e in first if e.stream == "image_description")
    assert [r.policy_mask for r in chat.records] == [False, False, True, False, True]
    assert [r.payload for r in chat.records[1:]] == [
        b"What is shown?",
        b"A red square.",
        b"What colour?",
        b"Red.",
    ]
    assert [len(m.events) for m in chat.meridians] == [2, 1, 1, 1]
    assert [
        e.request_revision for m in chat.meridians for e in m.events if e.role == "assistant"
    ] == [
        1,
        2,
    ]
    second = _run(media, llava, tmp_path, inputs=tmp_path / "inputs")
    reply = next(e for e in second if e.stream == "image_description")
    assert [r.payload for r in reply.records[1:]] == [
        b"What is here?",
        b"A circle.",
        b"Size?",
        b"Small.",
    ]
    third = _run(media, lists, tmp_path, inputs=tmp_path / "inputs")
    assert (
        sum(r.policy_mask for e in third if e.stream == "image_description" for r in e.records) == 2
    )


def test_recordings_with_descriptions_give_audio_description_and_identity(tmp_path, media):
    source = make_source(
        "captions",
        "captioned_audio",
        columns={"audio": "audio", "caption": "caption", "id": "id"},
        include=("*.parquet",),
        options={"pick": "first", "min_chars": 5},
    )
    parquet.write_table(
        pa.table(
            {
                "id": ["a", "b", "c"],
                "audio": [{"bytes": _wave(i), "path": ""} for i in range(3)],
                "caption": [["A dog barks twice.", "Dog."], ["x"], ["Rain falls on a roof."]],
            }
        ),
        _local(tmp_path, "captions") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"audio_description": 2, "audio_identity": 2}
    described = next(e for e in episodes if e.stream == "audio_description")
    assert [(r.role, r.policy_mask) for r in described.records] == [
        ("user", False),
        ("assistant", True),
    ]
    assert described.records[1].payload == b"A dog barks twice."


def test_an_original_of_another_size_follows_the_editing_model_when_the_shape_matches(tmp_path):
    from megatoken.data.open_media import write_image_chain, write_image_pair

    # MagicBrush: a 500 px photo edited by a model that returns 1024 px.
    small, big, wide = _image(1, (50, 50)), _image(2, (100, 100)), _image(3, (100, 60))
    pair = write_image_pair(small, big, tmp_path, "a")
    assert pair is not None and {Image.open(p).size for p in pair} == {(100, 100)}
    assert write_image_pair(small, wide, tmp_path, "b") is None  # another shape
    chain = write_image_chain([small, big, big, wide], tmp_path, "c")
    assert len(chain) == 3 and {Image.open(p).size for p in chain} == {(100, 100)}


def test_rows_linked_to_images_in_a_split_archive_become_edit_sessions(tmp_path, media):
    """SEED-Data-Edit multi-turn: jsonl rows name the source image and up to five edits, whose
    files sit in one tar.gz that was cut into parts."""
    source = make_source(
        "seed",
        "image_edit_chain",
        reader="linked",
        columns={"images": "images", "instruction": "texts", "id": "source_image"},
        include=("annotations/*.jsonl",),
        options={
            "linked": {
                "archives": ["images/multi_turn.tar.gz.part-*"],
                "multipart": True,
                "sequence": {
                    "images": ["source_image", "edit_image_{n}"],
                    "texts": ["instruction_{n}"],
                },
            },
            "provenance": {"media": "teacher"},
        },
    )
    folder = _local(tmp_path, "seed")
    (folder / "annotations").mkdir()
    (folder / "images").mkdir()
    rows = [
        {
            "source_image": "20240407_586P/Data/12/aa.jpg",
            "edit_image_1": "20240407_586P/Data/12/aa-edit1.JPG",
            "instruction_1": "Remove the light bulb from the picture",
            "edit_image_2": "20240407_586P/Data/12/aa-edit2.JPG",
            "instruction_2": "Change the yellow text in the picture to red",
        },
        {
            "source_image": "20240407_586P/Data/13/bb.jpg",
            "edit_image_1": "20240407_586P/Data/13/bb_edit1.jpg",
            "instruction_1": "Add character on the right",
        },
        {  # its edit is in no archive: skipped
            "source_image": "20240407_586P/Data/14/cc.jpg",
            "edit_image_1": "20240407_586P/Data/14/cc_edit1.jpg",
            "instruction_1": "Replace background",
        },
    ]
    (folder / "annotations/split_0001.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    members = {
        # Upper-case extensions, an extra leading directory and interleaved rows.
        "data/20240407_586P/Data/12/aa.jpg": _image(1),
        "data/20240407_586P/Data/13/bb.jpg": _image(2),
        "data/20240407_586P/Data/12/aa-edit1.jpg": _image(3),
        "data/20240407_586P/Data/13/bb_edit1.jpg": _image(4),
        "data/20240407_586P/Data/12/aa-edit2.jpg": _image(5),
        "data/20240407_586P/Data/14/cc.jpg": _image(6),
        "data/unrelated.jpg": _image(7),
    }
    whole = tmp_path / "all.tar.gz"
    _tar(whole, members, "w:gz")
    data = whole.read_bytes()
    cut = len(data) // 3
    (folder / "images/multi_turn.tar.gz.part-aa").write_bytes(data[:cut])
    (folder / "images/multi_turn.tar.gz.part-ab").write_bytes(data[cut:])
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_edit": 2}
    by_turns = sorted(sum(r.policy_mask for r in e.records) for e in episodes)
    assert by_turns == [1, 2]
    two = next(e for e in episodes if sum(r.policy_mask for r in e.records) == 2)
    assert [r.payload for r in two.records if r.media_type.startswith("text")] == [
        b"Remove the light bulb from the picture",
        b"Change the yellow text in the picture to red",
    ]
    assert [d.kind for d in two.dependencies] == ["edit", "edit"]


def test_multiple_choice_questions_about_recordings_list_the_options_in_the_question(
    tmp_path, media
):
    source = make_source(
        "choices",
        "audio_questions",
        columns={
            "audio": "audio",
            "question": "question",
            "choices": "options",
            "answer": "answer",
            "id": "id",
        },
        include=("*.parquet",),
        options={"format": "choices"},
    )
    parquet.write_table(
        pa.table(
            {
                "id": ["a", "b", "c"],
                "audio": [{"bytes": _wave(i), "path": ""} for i in range(3)],
                "question": ["What is heard?", "How many?", "Bad answer?"],
                "options": [["rain", "wind"], ["one", "two", "three"], ["x", "y"]],
                "answer": ["wind", "three", "zzz"],
            }
        ),
        _local(tmp_path, "choices") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    # The third row's answer index is outside its options: no turn, no episode.
    assert _streams(episodes) == {"audio_description": 2, "audio_identity": 2}
    first = next(e for e in episodes if e.stream == "audio_description")
    assert first.records[1].payload == b"What is heard?\nA. rain\nB. wind"
    assert first.records[2].payload == b"B. wind"


@ffmpeg
def test_captioned_clips_and_conversations_about_video_give_video_description(tmp_path, media):
    captions = make_source(
        "vcap",
        "captioned_video",
        "archive",
        columns={"video": "member", "caption": "caption", "id": "__key__"},
        include=("*.zip",),
        options={
            "table": {"path": "t.csv", "key": "{video}", "columns": ["caption"]},
            "max_seconds": 1.0,
        },
    )
    folder = _local(tmp_path, "vcap")
    (folder / "t.csv").write_text("video,caption\nc.mp4,A test pattern moves\n")
    with zipfile.ZipFile(folder / "0.zip", "w") as archive:
        archive.writestr("dir/c.mp4", _video(tmp_path / "c.mp4", 2, sound=False).read_bytes())
    episodes = _run(media, captions, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"video_description": 1, "video_identity": 1}
    described = next(e for e in episodes if e.stream == "video_description")
    assert described.records[1].payload == b"A test pattern moves"

    questions = make_source(
        "vqa",
        "video_questions",
        columns={"video": "video", "turns": "conversations", "id": "id"},
        include=("*.parquet",),
        options={"format": "messages", "max_seconds": 1.0},
    )
    parquet.write_table(
        pa.table(
            {
                "id": ["v"],
                "video": [{"bytes": _video(tmp_path / "q.mp4", 2, sound=False).read_bytes()}],
                "conversations": [
                    [
                        {"from": "human", "value": "<image>\nWhat moves?"},
                        {"from": "gpt", "value": "A pattern."},
                    ]
                ],
            }
        ),
        _local(tmp_path, "vqa") / "0.parquet",
    )
    episodes = _run(media, questions, tmp_path, inputs=tmp_path / "inputs")
    asked = next(e for e in episodes if e.stream == "video_description")
    assert [r.payload for r in asked.records[1:]] == [b"What moves?", b"A pattern."]
    assert not list((tmp_path / "work").rglob("*.mp4"))


def test_tables_may_be_json_arrays_with_path_keys_and_rows_sharing_a_key_may_merge(tmp_path, media):
    """LLaVA-Video-178K shape: a JSON array whose rows name a video by a path inside the tar,
    several rows (conversations) for the same video."""
    source = make_source(
        "videoqa",
        "image_questions",
        "archive",
        columns={"image": "member", "turns": "conversations", "id": "__key__"},
        include=("images/*.tar",),
        options={
            "format": "messages",
            "table": {
                "path": "qa.json",
                "key": "{video__name}",
                "columns": ["conversations"],
                "merge": ["conversations"],
            },
        },
    )
    folder = _local(tmp_path, "videoqa")
    (folder / "images").mkdir()
    rows = [
        {
            "video": "academic/clips/a.png",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is this?"},
                {"from": "gpt", "value": "A square."},
            ],
        },
        {
            "video": "other/path/a.png",
            "conversations": [
                {"from": "human", "value": "What colour?"},
                {"from": "gpt", "value": "Blue."},
            ],
        },
        {"video": "academic/clips/b.png", "conversations": []},
    ]
    (folder / "qa.json").write_text(json.dumps(rows))
    _tar(folder / "images/0.tar", {"x/a.png": _image(1), "x/b.png": _image(2)})
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_description": 1, "image_identity": 1}
    chat = next(e for e in episodes if e.stream == "image_description")
    assert [r.payload for r in chat.records[1:]] == [
        b"What is this?",
        b"A square.",
        b"What colour?",
        b"Blue.",
    ]


def test_json_lines_in_a_file_named_json_are_read_line_by_line(tmp_path):
    from megatoken.data.open_media import _table

    source = make_source("t")
    folder = _local(tmp_path, "t")
    (folder / "t.json").write_text(
        '{"location": "data_aug2/x_1.wav", "caption": "Slow piano"}\n'
        '{"location": "data/y.wav", "caption": "Fast drums"}\n'
    )
    rows = _table(
        Shards(source, tmp_path / "inputs"),
        {"path": "t.json", "key": "{location__name}"},
        ["caption"],
    )
    assert rows == {"x_1.wav": {"caption": "Slow piano"}, "y.wav": {"caption": "Fast drums"}}


def test_a_table_with_one_file_per_archive_is_read_with_its_archive(tmp_path, media):
    source = make_source(
        "parts",
        "captioned_image",
        "archive",
        columns={"image": "member", "caption": "caption", "id": "__key__"},
        include=("part_*.tar",),
        options={
            "table": {
                "path": "unused.csv",
                "key": "{clip_id}.png",
                "columns": ["caption"],
                "per_archive": {"pattern": r"part_(\d+)", "path": "caps_{0}.csv"},
            }
        },
    )
    folder = _local(tmp_path, "parts")
    for number in (1, 2):
        (folder / f"caps_{number:03d}.csv").write_text(
            f"clip_id,caption\nc{number},Caption of clip {number} here\n"
        )
        _tar(folder / f"part_{number:03d}.tar", {f"d/c{number}.png": _image(number)})
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    captions = sorted(e.records[1].payload for e in episodes if e.stream == "image_description")
    assert captions == [b"Caption of clip 1 here", b"Caption of clip 2 here"]


def test_webdataset_keys_may_hold_dots_when_asked(tmp_path):
    source = make_source(
        "dots", "av_caption", "webdataset", include=("*.tar",), options={"key_split": "last"}
    )
    _tar(
        _local(tmp_path, "dots") / "0.tar",
        {"d/abc.0.mp4": b"v0", "d/abc.0.json": b'{"caption": "a"}', "d/abc.1.mp4": b"v1"},
    )
    from megatoken.data.open_media import webdataset_records

    records = list(webdataset_records(source, Shards(source, tmp_path / "inputs")))
    assert [(r["__key__"], sorted(set(r) - {"__key__", "__shard__"})) for r in records] == [
        ("abc.0", ["json", "mp4"]),
        ("abc.1", ["mp4"]),
    ]


def test_webdataset_members_with_dotted_names_are_addressable_and_json_arrays_link(tmp_path, media):
    """cc12m-recaptioned shape: members ``<key>.image.jpg`` and ``<key>.synthetic_caption.txt``."""
    from megatoken.data.open_media import locate, webdataset_records

    assert locate({"image.jpg": b"x", "json": {"a.b": 3}}, "json.a.b") == 3
    assert locate({"image.jpg": b"x"}, "image.jpg") == b"x"
    assert locate({"a": {"b": [5, 6]}}, "a.b.1") == 6
    source = make_source(
        "cc",
        "captioned_image",
        "webdataset",
        columns={"image": "image.jpg", "caption": "synthetic_caption.txt", "id": "__key__"},
        include=("*.tar",),
    )
    _tar(
        _local(tmp_path, "cc") / "0.tar",
        {
            "k1.image.jpg": _image(1),
            "k1.original_caption.txt": b"short",
            "k1.synthetic_caption.txt": b"A long synthetic description of the image.",
        },
    )
    (record,) = webdataset_records(source, Shards(source, tmp_path / "inputs"))
    assert record["synthetic_caption.txt"].startswith("A long synthetic")
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_description": 1, "image_creation": 1, "image_identity": 1}

    # ShareGPT-4o-Image edits: a JSON array table; input images are a list; one archive.
    edits = make_source(
        "gpt4o",
        "image_edit_chain",
        reader="linked",
        columns={"images": "images", "instruction": "texts", "id": "output_image"},
        include=("edits.json",),
        options={
            "linked": {
                "archives": ["part_*.tar"],
                "sequence": {"images": ["input_image", "output_image"], "texts": ["input_prompt"]},
            }
        },
    )
    folder = _local(tmp_path, "gpt4o")
    (folder / "edits.json").write_text(
        json.dumps(
            [
                {
                    "input_prompt": "Remove the barrier",
                    "input_image": ["image/v2v_1.png"],
                    "output_image": "image/v2v_2.png",
                }
            ]
        )
    )
    _tar(folder / "part_0.tar", {"image/v2v_1.png": _image(1), "image/v2v_2.png": _image(2)})
    (episode,) = _run(media, edits, tmp_path, inputs=tmp_path / "inputs")
    assert episode.stream == "image_edit" and episode.records[1].payload == b"Remove the barrier"


def test_the_better_rated_of_two_candidate_edits_is_the_edit(tmp_path, media):
    source = make_source(
        "pref",
        "image_edit",
        columns={"original": "source", "edited": "left", "instruction": "instruction", "id": "key"},
        include=("*.parquet",),
        options={
            "choose": {"candidates": ["left", "right"], "scores": ["left_score", "right_score"]}
        },
    )
    parquet.write_table(
        pa.table(
            {
                "key": ["a", "b", "c"],
                "instruction": ["add a hat", "make it red", "unrated"],
                "source": [{"bytes": _image(i), "path": ""} for i in (1, 2, 3)],
                "left": [{"bytes": _image(i), "path": ""} for i in (2, 3, 4)],
                "right": [{"bytes": _image(i), "path": ""} for i in (5, 6, 7)],
                "left_score": [1.0, 9.0, None],
                "right_score": [8.0, 2.0, None],
            }
        ),
        _local(tmp_path, "pref") / "0.parquet",
    )
    from megatoken.data.open_media import Shards, _chosen, parquet_records

    records = list(parquet_records(source, Shards(source, tmp_path / "inputs")))
    chosen = [_chosen(source, record) for record in records]
    assert [c["bytes"] if c else None for c in chosen] == [_image(5), _image(3), None]
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"image_edit": 2}


def test_linked_rows_lose_a_leading_folder_and_a_shard_reads_only_its_archives(tmp_path, media):
    """Senorita: the table says ./style/19/h.png, the archive holds ./19/h.png in one of many tars."""
    source = make_source(
        "sen",
        "image_edit_chain",
        reader="linked",
        columns={"images": "images", "instruction": "texts", "id": "source_image"},
        include=("table.json",),
        options={
            "linked": {
                "archives": ["up/*.tar.gz"],
                "strip": r"^(?:\./)?[^/]+/",
                "sequence": {"images": ["source_image", "target_image"], "texts": ["instruction"]},
            }
        },
    )
    folder = _local(tmp_path, "sen")
    (folder / "up").mkdir()
    (folder / "table.json").write_text(
        json.dumps(
            [
                {
                    "source_image": f"./style/{n}/a.png",
                    "target_image": f"./style/{n}/a_org.png",
                    "instruction": f"edit {n}",
                }
                for n in (1, 2)
            ]
        )
    )
    for n in (1, 2):
        _tar(
            folder / "up" / f"{n}.tar.gz",
            {f"./{n}/a.png": _image(n), f"./{n}/a_org.png": _image(n + 4)},
            "w:gz",
        )
    everything = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert len(everything) == 2
    shard = Shards(source, tmp_path / "inputs", files=["up/2.tar.gz"])
    from megatoken.data.open_media import linked_records

    (record,) = linked_records(source, shard)
    assert record["texts"] == ["edit 2"] and len(record["images"]) == 2


def test_linked_rows_may_share_an_image_or_name_it_twice_and_a_repeated_member_is_ignored(
    tmp_path, media
):
    source = make_source(
        "shared",
        "image_edit_chain",
        reader="linked",
        columns={"images": "images", "instruction": "texts", "id": "id"},
        include=("table.json",),
        options={
            "linked": {
                "archives": ["up/*.tar"],
                "sequence": {"images": ["source_image", "target_image"], "texts": ["instruction"]},
            }
        },
    )
    folder = _local(tmp_path, "shared")
    (folder / "up").mkdir()
    (folder / "table.json").write_text(
        json.dumps(
            [
                {"id": "a", "source_image": "s.png", "target_image": "a.png", "instruction": "x"},
                {"id": "b", "source_image": "s.png", "target_image": "b.png", "instruction": "y"},
                {"id": "c", "source_image": "s.png", "target_image": "s.png", "instruction": "z"},
            ]
        )
    )
    _tar(folder / "up" / "0.tar", {"d1/s.png": _image(1), "a.png": _image(2), "b.png": _image(3)})
    _tar(folder / "up" / "1.tar", {"d2/s.png": _image(4)})
    from megatoken.data.open_media import linked_records

    records = list(linked_records(source, Shards(source, tmp_path / "inputs")))
    assert sorted(record["id"] for record in records) == ["a", "b", "c"]
    assert all(len(record["images"]) == 2 for record in records)


def test_a_message_list_ends_at_the_first_turn_that_does_not_alternate(tmp_path):
    from megatoken.data.open_media import conversation

    source = make_source(
        "chat", "audio_questions", options={"format": "messages"}, columns={"turns": "turns"}
    )
    messages = [
        {"role": "user", "content": "one?"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "stray"},
        {"role": "assistant", "content": "stray too"},
        {"role": "user", "content": "two?"},
        {"role": "assistant", "content": "second"},
    ]
    assert conversation(source, {"turns": messages}) == [("one?", "first")]


def test_audio_stored_as_decoded_samples_is_read_like_a_file(tmp_path, media):
    source = make_source(
        "notes",
        "sound",
        columns={"audio": "audio", "id": "id", "family": "instrument"},
        include=("*.parquet",),
    )
    samples = np.sin(2 * np.pi * 220 * np.arange(32000) / 16000).astype("float32") * 0.4
    parquet.write_table(
        pa.table(
            {
                "audio": [{"array": samples.tolist(), "path": "x.wav", "sampling_rate": 16000}],
                "id": ["a"],
                "instrument": ["flute"],
            }
        ),
        _local(tmp_path, "notes") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"audio_identity": 1}


def test_a_transcription_source_can_also_be_read_aloud_from_its_transcripts(tmp_path, media):
    source = make_source(
        "asr",
        columns={"audio": "audio", "text": "text", "id": "id"},
        include=("*.parquet",),
        options={"verbalize": True},
    )
    assert source.streams == (
        "audio_identity",
        "edit_plan",
        "speech_edit",
        "transcription",
        "verbalization",
    )
    parquet.write_table(
        pa.table(
            {
                "audio": [{"bytes": _wave(1), "path": "a.flac"}],
                "text": ["Hello, world."],
                "id": ["a"],
            }
        ),
        _local(tmp_path, "asr") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"transcription": 1, "verbalization": 1, "audio_identity": 1}
    said = next(e for e in episodes if e.stream == "verbalization")
    assert said.records[0].payload == b"Hello, world." and said.records[1].policy_mask
    assert make_source("plain").streams == (
        "audio_identity",
        "edit_plan",
        "speech_edit",
        "transcription",
    )


def test_a_parquet_table_with_nested_turns_links_to_its_images(tmp_path, media):
    """ImgEdit multi-turn: one row holds a list of turns, each naming its own input and output."""
    source = make_source(
        "imgedit",
        "image_edit_chain",
        reader="linked",
        columns={"images": "images", "instruction": "texts", "id": "data.0.output_images.0"},
        include=("Parquet/*.parquet",),
        options={
            "family_pattern": r"^[^/]+/(\d+_\d+)_\d+/",
            "linked": {
                "archives": ["Multiturn/results_a.tar.split.*"],
                "multipart": True,
                "sequence": {
                    "images": [
                        "data.0.input_images.0",
                        "data.0.output_images.0",
                        "data.1.output_images.0",
                    ],
                    "texts": ["data.0.prompt", "data.1.prompt"],
                },
            },
        },
    )

    def turn(prompt, origin, result):
        return {"input_images": [origin], "output_images": [result], "prompt": prompt}

    chain = "results_a/00087_00016_000166465"
    folder = _local(tmp_path, "imgedit")
    (folder / "Parquet").mkdir()
    (folder / "Multiturn").mkdir()
    parquet.write_table(
        pa.table(
            {
                "data": [
                    [
                        turn(
                            "replace the lemon with a mango",
                            f"{chain}/origin_0.png",
                            f"{chain}/result_0.png",
                        ),
                        turn("make it ripe", f"{chain}/origin_1.png", f"{chain}/result_1.png"),
                        turn("remove it", f"{chain}/origin_2.png", f"{chain}/result_2.png"),
                    ]
                ]
            }
        ),
        folder / "Parquet/a.parquet",
    )
    members = {
        f"mnt/data/code/{chain}/{name}.png": _image(number)
        for number, name in enumerate(
            ["origin_0", "result_0", "mask_0", "origin_1", "result_1", "origin_2", "result_2"]
        )
    }
    whole = tmp_path / "all.tar"
    _tar(whole, members)
    data = whole.read_bytes()
    cut = len(data) // 2
    (folder / "Multiturn/results_a.tar.split.000").write_bytes(data[:cut])
    (folder / "Multiturn/results_a.tar.split.001").write_bytes(data[cut:])
    (episode,) = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert episode.stream == "image_edit"
    assert episode.lineage.family_id == "imgedit:00087_00016"
    texts = [r.payload for r in episode.records if r.media_type.startswith("text")]
    assert texts == [b"replace the lemon with a mango", b"make it ripe"]


def test_samples_numbered_afresh_in_every_tar_get_distinct_ids(tmp_path, media):
    source = make_source(
        "wds",
        "picture",
        "webdataset",
        columns={"image": "jpg", "id": ["__shard__", "__key__"]},
        include=("*.tar",),
    )
    folder = _local(tmp_path, "wds")
    for number in (0, 1):
        _tar(folder / f"{number}.tar", {"000000000.jpg": _image(number + 1)})
    from megatoken.data.open_media import Shards, read_clips

    clips = list(read_clips(source, Shards(source, tmp_path / "inputs"), tmp_path / "out", 5))
    assert [clip["id"] for clip in clips] == ["0.tar-000000000", "1.tar-000000000"]
    assert len({clip["family"] for clip in clips}) == 2


def _judging_source(**extra):
    return make_source(
        "judge",
        "edit_judging",
        columns={
            "original": "source",
            "instruction": "instruction",
            "candidates": ["left", "right"],
            "id": "key",
        },
        include=("*.parquet",),
        options={
            "judge": {
                "scores": {
                    "instruction following": ["left_follow", "right_follow"],
                    "visual quality": ["left_quality", "right_quality"],
                },
                "outof": 4,
            },
            **extra,
        },
    )


def _judging_table(tmp_path, **columns):
    rows = {
        "key": ["a", "b"],
        "source": [{"bytes": _image(1), "path": ""}] * 2,
        "left": [{"bytes": _image(2), "path": ""}] * 2,
        "right": [{"bytes": _image(3), "path": ""}] * 2,
        "instruction": ["add snow", "make it night"],
        "left_follow": [1, 3],
        "right_follow": [4, 3],
        "left_quality": [2, 3],
        "right_quality": [3, 3],
    }
    parquet.write_table(pa.table({**rows, **columns}), _local(tmp_path, "judge") / "0.parquet")


def _is_media(record) -> bool:
    return record.media_type.endswith("media-reference+json")


def test_edit_judging_shows_the_original_and_candidates_and_supervises_the_verdict(tmp_path, media):
    _judging_table(tmp_path)
    episodes = _run(media, _judging_source(), tmp_path, inputs=tmp_path / "inputs")
    assert _streams(episodes) == {"edit_judging": 2}
    for episode in episodes:
        assert [r.policy_mask for r in episode.records].count(True) == 1
        assert not _is_media(episode.records[-1]) and episode.records[-1].policy_mask
        # The original and both candidates come before the verdict, none of them supervised.
        assert sum(_is_media(r) for r in episode.records) == 3
        assert not any(r.policy_mask for r in episode.records if _is_media(r))
    verdicts = {e.records[-1].payload.decode() for e in episodes}
    assert any("Verdict: the candidates are equally good." in v for v in verdicts)
    best = next(v for v in verdicts if "better edit" in v)
    assert "instruction following 1 of 4" in best and "instruction following 4 of 4" in best
    assert best.count("Candidate") == 3


def test_edit_judging_keeps_the_best_candidate_in_either_position(tmp_path, media):
    rows = {
        "key": [f"k{n}" for n in range(24)],
        "source": [{"bytes": _image(1), "path": ""}] * 24,
        "left": [{"bytes": _image(2), "path": ""}] * 24,
        "right": [{"bytes": _image(3), "path": ""}] * 24,
        "instruction": ["add snow"] * 24,
    }
    source = make_source(
        "pref",
        "edit_judging",
        columns={
            "original": "source",
            "instruction": "instruction",
            "candidates": ["left", "right"],
            "id": "key",
        },
        include=("*.parquet",),
        options={"judge": {"winner": 0}},
    )
    parquet.write_table(pa.table(rows), _local(tmp_path, "pref") / "0.parquet")
    texts = {
        e.records[-1].payload.decode()
        for e in _run(media, source, tmp_path, count=30, inputs=tmp_path / "inputs")
    }
    assert texts == {
        "Verdict: Candidate A is the better edit.",
        "Verdict: Candidate B is the better edit.",
    }


def test_an_edit_plan_is_written_before_the_result_it_describes(tmp_path, media):
    source = make_source(
        "plan",
        "image_edit_plan",
        columns={
            "original": "source",
            "edited": "result",
            "instruction": "request",
            "plan": "tool_calls",
            "id": "key",
        },
        include=("*.parquet",),
    )
    parquet.write_table(
        pa.table(
            {
                "key": ["p"],
                "source": [{"bytes": _image(1), "path": ""}],
                "result": [{"bytes": _image(2), "path": ""}],
                "request": ["make it warmer"],
                "tool_calls": [[{"tool": "adjust", "temperature": 12}, {"tool": "export"}]],
            }
        ),
        _local(tmp_path, "plan") / "0.parquet",
    )
    (episode,) = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert episode.stream == "edit_plan"
    plan = next(r for r in episode.records if r.policy_mask and not _is_media(r))
    assert plan.payload.decode() == '[{"tool": "adjust", "temperature": 12}, {"tool": "export"}]'
    assert [r.policy_mask for r in episode.records] == [False, False, True, True]
    assert [d.kind for d in episode.dependencies] == ["edit"]


# --- sources published as URLs (Pico-Banana's layout: tables and pictures on a web site) -------


@pytest.fixture
def pico_site():
    from web_support import Site

    pictures = {
        f"/images/{name}.png": _image(number)
        for number, name in enumerate(
            [
                "multi-turn/101_attemptB_turn1",
                "multi-turn/101_attemptB_turn2",
                "multi-turn/101_attemptB_turn3",
                "positive-edit/kf12os_retry2",
                "rejected-edit/14",
                "rejected-edit/kf12os_retry1",
                "positive-edit/1",
            ],
            1,
        )
    }
    pictures["/farm8/orig.jpg"] = _image(9, fmt="JPEG")
    pictures["/farm6/orig.jpg"] = _image(8, fmt="JPEG")
    site = Site(pictures)
    base = site.url
    multi = {
        "files": [
            {"id": "edit_turn1", "url": "images/multi-turn/101_attemptB_turn1.png"},
            {"id": "edit_turn2", "url": "images/multi-turn/101_attemptB_turn2.png"},
            {"id": "final_image", "url": "images/multi-turn/101_attemptB_turn3.png"},
            {"id": "original_input_image", "url": f"{base}/farm8/orig.jpg"},
        ],
        "metadata_edit_turn_prompts": [
            "Warm golden tone",
            "Zoom in on the car",
            "Comic book style",
        ],
    }
    broken = {
        **multi,
        "files": [
            *multi["files"][:3],
            {"id": "original_input_image", "url": f"{base}/gone/orig.jpg"},
        ],
    }
    preference = {
        "output_image": "images/positive-edit/kf12os_retry2.png",
        "rejected_images": [
            "nano-banana-300k/images/rejected-edit/14.png",
            "nano-banana-300k/images/rejected-edit/kf12os_retry1.png",
        ],
        "text": "Make the shirt look looser and billowed.",
        "edit_type": "Change the size/shape/orientation of an object",
        "open_image_input_url": f"{base}/farm6/orig.jpg",
        "summarized_text": "Billow the shirt.",
    }
    single = {
        "open_image_input_url": f"{base}/farm8/orig.jpg",
        "text": "Remove the red flag.",
        "output_image": "images/positive-edit/1.png",
        "edit_type": "Remove an existing object",
        "summarized_text": "Flag removed.",
    }
    for name, rows in (("multi", [multi, broken]), ("pref", [preference]), ("sft", [single])):
        site.files[f"/jsonl/{name}.jsonl"] = (
            "\n".join(json.dumps(r) for r in rows) + "\n"
        ).encode()
    yield site
    site.close()


def _web_source(site, name, task, transform, columns, **options):
    digest = hashlib.sha256(site.files[f"/jsonl/{name}.jsonl"]).hexdigest()
    return OpenSource(
        source_id=f"web_{name}",
        repository=site.url,
        revision="web-2026-10-23",
        reader="table",
        task=task,
        terms=TermsReview(True, "CC-BY-NC-ND-4.0", "https://example.org", "Reviewer", "2026-10-22"),
        include=(f"jsonl/{name}.jsonl",),
        columns=columns,
        options={
            "base": site.url,
            "transform": transform,
            "pins": {f"jsonl/{name}.jsonl": digest},
            **options,
        },
    )


def test_pico_multi_turn_sessions_are_fetched_and_a_dead_original_is_counted(
    tmp_path, media, pico_site
):
    from megatoken.data.fetch import FETCH_STATS

    source = _web_source(
        pico_site,
        "multi",
        "image_edit_chain",
        "pico_session",
        {
            "original": "original",
            "edited": "edited",
            "instruction": "instruction",
            "id": "id",
            "family": "original",
        },
        fetch={"fields": ["original", "edited.*"], "workers": 2},
    )
    episodes = _run(media, source, tmp_path)
    assert len(episodes) == 1, [(e.stream, len(e.records)) for e in episodes]
    (episode,) = episodes  # the session whose original photo is gone is skipped
    assert episode.stream == "image_edit"
    assert [r.policy_mask for r in episode.records].count(True) == 3
    assert [d.kind for d in episode.dependencies] == ["edit", "edit", "edit"]
    stats = FETCH_STATS["web_multi"]
    assert stats["dead"] == 1 and stats["http_404"] == 1 and stats["rows"] == 2
    assert stats["rows_with_dead_links"] == 1 and stats["fetched"] >= 4
    # The table was downloaded once, checked against its pin, and every picture asked for once.
    assert pico_site.hits["/jsonl/multi.jsonl"] >= 1
    assert pico_site.hits["/farm8/orig.jpg"] == 1


def test_pico_preference_rows_become_judging_episodes_with_the_chosen_edit_among_the_candidates(
    tmp_path, media, pico_site
):
    source = _web_source(
        pico_site,
        "pref",
        "edit_judging",
        "pico_preference",
        {
            "original": "original",
            "instruction": "instruction",
            "candidates": "candidates",
            "id": "id",
            "family": "original",
        },
        fetch={"fields": ["original", "candidates.*"]},
        judge={"winner": 0},
    )
    (episode,) = _run(media, source, tmp_path)
    assert episode.stream == "edit_judging"
    # The original and three candidates (one chosen, two rejected) are shown; the verdict follows.
    assert sum(r.media_type.endswith("media-reference+json") for r in episode.records) == 4
    verdict = episode.records[-1].payload.decode()
    assert re.fullmatch(
        r"Verdict: Candidate [A-C] is the best edit; the others are worse\.", verdict
    )
    # The rejected edits' relative paths lose their folder prefix when fetched.
    assert pico_site.hits["/images/rejected-edit/14.png"] == 1


def test_pico_single_edits_use_the_published_instruction(tmp_path, media, pico_site):
    source = _web_source(
        pico_site,
        "sft",
        "image_edit",
        "pico_edit",
        {"original": "original", "edited": "edited", "instruction": "instruction", "id": "id"},
        fetch={"fields": ["original", "edited"]},
    )
    (episode,) = _run(media, source, tmp_path)
    assert episode.stream == "image_edit"
    assert episode.records[1].payload == b"Remove the red flag."


def test_a_table_cut_short_by_the_server_is_resumed_and_a_wrong_pin_is_refused(tmp_path, pico_site):
    from web_support import Site

    from megatoken.data.fetch import download

    body = pico_site.files["/jsonl/multi.jsonl"] * 40
    site = Site({"/t.jsonl": body}, cut=("/t.jsonl",))
    try:
        digest = hashlib.sha256(body).hexdigest()
        path = download(
            f"{site.url}/t.jsonl", tmp_path / "t.jsonl", sha256=digest, sleep=lambda s: None
        )
        assert (
            path.read_bytes() == body and site.hits["/t.jsonl"] >= 3
        )  # HEAD, the cut GET, a resume
        again = download(f"{site.url}/t.jsonl", tmp_path / "t.jsonl", sha256=digest)
        assert again == path
        with pytest.raises(OSError, match="pinned"):
            download(f"{site.url}/t.jsonl", tmp_path / "other.jsonl", sha256="0" * 64)
    finally:
        site.close()


def test_every_picture_edit_program_runs_and_its_plan_is_the_program(tmp_path):
    from megatoken.data.open_media import IMAGE_EDITS, image_edit

    picture = tmp_path / "photo.jpg"
    picture.write_bytes(_image(3, size=(64, 48), fmt="JPEG"))
    seen = set()
    for number in range(400):
        edit = image_edit({"id": f"p{number}", "family": "f", "image": picture}, tmp_path / "e")
        seen.add(edit["instruction"])
        assert edit["plan"].startswith("```python\n") and edit["plan"].endswith("\n```")
        assert Image.open(edit["edited"]).size[0] > 0
        edit["edited"].unlink()
    assert seen == {instruction for instruction, _ in IMAGE_EDITS}
    gray = next(c for i, c in IMAGE_EDITS if i.startswith("Convert the picture to black"))
    assert "grayscale" in gray


@ffmpeg
def test_procedural_video_and_speech_edits_carry_their_ffmpeg_command_as_the_plan(tmp_path, media):
    from megatoken.data.open_media import plan_episodes

    source = make_source("proc")
    video = _video(tmp_path / "clip.mp4", 1, sound=False, size="64x48")
    edit = video_edit({"id": "c", "family": "proc:r", "video": video}, tmp_path / "videos")
    assert re.fullmatch(r"```sh\nffmpeg -i input\.mp4 -vf \S+ output\.mp4\n```", edit["plan"])
    clip = {
        "id": "c",
        "family": "proc:r",
        "frames": [edit["original"], edit["edited"]],
        "plans": [edit["plan"]],
        "instruction": edit["instruction"],
    }
    (episode,) = plan_episodes(media, source, "av", clip, provenance="executed", tier="procedural")
    assert episode.stream == "edit_plan"
    assert [r.policy_mask for r in episode.records] == [False, False, True, True]
    assert episode.records[2].payload.decode() == edit["plan"]


def test_picture_sources_make_executed_plans_for_the_clips_they_are_asked_to(tmp_path, media):
    pytest.importorskip("PIL")
    source = make_source(
        "pics", "picture", columns={"image": "image", "id": "id"}, include=("*.parquet",)
    )
    assert "edit_plan" in source.streams and "image_edit" not in source.streams
    parquet.write_table(
        pa.table(
            {
                "image": [{"bytes": _image(n, size=(64, 48)), "path": ""} for n in (1, 2)],
                "id": ["a", "b"],
            }
        ),
        _local(tmp_path, "pics") / "0.parquet",
    )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs", procedural=2)
    assert _streams(episodes) == {"image_identity": 2, "edit_plan": 2}
    plan = next(e for e in episodes if e.stream == "edit_plan")
    code = next(r for r in plan.records if r.policy_mask and not _is_media(r)).payload.decode()
    assert code.startswith("```python\nedited = ")
    assert [d.kind for d in plan.dependencies] == ["edit"]
    assert not list((tmp_path / "work").rglob("*.png"))


JARVIS_SETTINGS = """Temperature = 6450,
Tint = -11,
SaturationAdjustmentOrange = -5,
SaturationAdjustmentRed = 0,
Exposure2012 = -0.05,
Highlights2012 = -79,
ConvertToGrayscale = false,
OverrideLookVignette = true,
ToneCurveName = "Medium Contrast",
"""


def test_jarvisart_folders_become_a_plan_with_the_lightroom_settings_as_a_tool_call(
    tmp_path, media
):
    source = make_source(
        "jarvis",
        "image_edit_plan",
        reader="folders",
        columns={
            "original": "original",
            "edited": "edited",
            "instruction": "instruction",
            "plan": "plan",
            "id": "__key__",
        },
        include=("global/*/before.jpg",),
        options={
            "members": {
                "before": "before.jpg",
                "after": "processed.jpg",
                "settings": "config.lua",
                "request": "en/user_want_middle/user_prompt.txt",
                "reasoning": "en/user_want_middle/revised160_expert_cot.txt",
            },
            "transform": "jarvis_plan",
        },
    )
    folder = _local(tmp_path, "jarvis")
    for name in ("0001", "0002"):
        root = folder / "global" / name
        (root / "en/user_want_middle").mkdir(parents=True)
        (root / "before.jpg").write_bytes(_image(1, fmt="JPEG"))
        (root / "processed.jpg").write_bytes(_image(2, fmt="JPEG"))
        (root / "config.lua").write_text(JARVIS_SETTINGS)
        (root / "en/user_want_middle/user_prompt.txt").write_text("Make it warmer and sharper.")
        if name == "0001":
            (root / "en/user_want_middle/revised160_expert_cot.txt").write_text(
                "Alright, the user wants a warmer look. I will correct the white balance first."
            )
    episodes = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    (episode,) = episodes  # the second folder lacks its reasoning file
    plan = next(r for r in episode.records if r.policy_mask and not _is_media(r)).payload.decode()
    assert plan.startswith("Alright, the user wants a warmer look.")
    call = json.loads(plan.split("Tool call: ", 1)[1])
    assert call == {
        "tool": "lightroom.develop",
        "settings": {
            "Temperature": 6450,
            "Tint": -11,
            "SaturationAdjustmentOrange": -5,
            "Exposure2012": -0.05,
            "Highlights2012": -79,
            "OverrideLookVignette": True,
            "ToneCurveName": "Medium Contrast",
        },
    }


def test_a_judge_conversation_about_one_edit_becomes_pointwise_judging(tmp_path, media):
    """EditScore: the judge's prompt names the request; its reply is the verdict. Rows that judge
    one picture alone (naturalness) have no edit to look at and are skipped."""
    source = make_source(
        "editscore",
        "edit_judging",
        columns={
            "original": "images.0",
            "candidates": ["images.1"],
            "instruction": "conversations.0.value",
            "verdict": "conversations.1.value",
            "id": "key",
        },
        include=("*.parquet",),
        options={"clean": "image_tags"},
    )
    prompt = "<image>\n<image>\nYou are a professional digital artist.\nEditing instruction: Make it snow\n"
    reply = '{\n"reasoning" : "Snow appears on the roof.",\n"score" : [15, 22]\n}'
    pair = [
        {"bytes": _image(1), "path": "images/1_0.png"},
        {"bytes": _image(2), "path": "images/1_1.png"},
    ]
    rows = pa.table(
        {
            "key": ["1", "2"],
            "conversations": [
                [{"from": "human", "value": prompt}, {"from": "gpt", "value": reply}],
                [
                    {"from": "human", "value": "<image>\nRate naturalness."},
                    {"from": "gpt", "value": reply},
                ],
            ],
            "images": [pair, pair[:1]],
        }
    )
    parquet.write_table(rows, _local(tmp_path, "editscore") / "0.parquet")
    (episode,) = _run(media, source, tmp_path, inputs=tmp_path / "inputs")
    assert episode.stream == "edit_judging"
    texts = [r.payload.decode() for r in episode.records if not _is_media(r)]
    assert texts[0].startswith("You are a professional digital artist.")
    assert "<image>" not in texts[0] and "Editing instruction: Make it snow" in texts[0]
    assert texts[-1] == reply and episode.records[-1].policy_mask
    assert sum(_is_media(r) for r in episode.records) == 2  # the original and the one candidate


def test_a_flickr_original_is_fetched_from_open_images_when_it_holds_a_copy(
    tmp_path, media, pico_site
):
    from megatoken.data.open_media import _open_images_index

    _open_images_index.cache_clear()
    csv_text = (
        "ImageID,Subset,OriginalURL,OriginalLandingURL\n"
        f"abc123,train,https://farm3.staticflickr.com/5310/orig.jpg?zz=1,https://flickr.example/1\n"
        "def456,train,https://c1.staticflickr.com/1/67/other.jpg,https://flickr.example/2\n"
    )
    pico_site.files["/oi/index.csv"] = csv_text.encode()
    pico_site.files["/oi/train/abc123.jpg"] = _image(7, fmt="JPEG")
    source = _web_source(
        pico_site,
        "multi",
        "image_edit_chain",
        "pico_session",
        {
            "original": "original",
            "edited": "edited",
            "instruction": "instruction",
            "id": "id",
            "family": "original",
        },
        fetch={"fields": ["original", "edited.*"], "workers": 2},
        originals={
            "index": f"{pico_site.url}/oi/index.csv",
            "pin": hashlib.sha256(csv_text.encode()).hexdigest(),
            "url": f"{pico_site.url}/oi/train/{{id}}.jpg",
        },
    )
    # The table's first original is …/farm8/orig.jpg, whose file name Open Images lists; the
    # second row's (…/gone/orig.jpg) has the same name, so both come from Open Images.
    episodes = _run(media, source, tmp_path)
    assert len(episodes) == 2
    assert pico_site.hits["/oi/train/abc123.jpg"] == 1  # one copy, kept by content hash
    assert pico_site.hits["/farm8/orig.jpg"] == 0 and pico_site.hits["/gone/orig.jpg"] == 0
    _open_images_index.cache_clear()
