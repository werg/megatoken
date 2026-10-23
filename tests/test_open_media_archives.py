"""Archive layouts of the gated media repositories: zipped tables, tab-separated tables, tars that
list audio before its transcripts (no network, tiny fixtures)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile
from media_support import prepared_record
from PIL import Image

from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.open_media import (
    CachingPreparer,
    OpenSource,
    _json_array,
    source_episodes,
)
from megatoken.data.sources import TermsReview


class FakePreparer:
    def __init__(self, store):
        self.store = store

    def media(self, modality, path, *, target=True):
        seed = int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:6], 16)
        return prepared_record(self.store, modality, seed=seed, target=target)


@pytest.fixture
def media(tmp_path):
    return CachingPreparer(FakePreparer(TensorAssetStore(tmp_path / "assets")))


def make_source(source_id, task, reader, **fields) -> OpenSource:
    return OpenSource(
        source_id=source_id,
        repository=f"org/{source_id}",
        revision="0" * 40,
        reader=reader,
        task=task,
        terms=TermsReview(True, "CC-BY-4.0", "https://example.org", "Reviewer", "2026-10-23"),
        **fields,
    )


def _wave(seed: int, seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    time = np.arange(int(seconds * rate)) / rate
    soundfile.write(buffer, 0.3 * np.sin(2 * np.pi * (200 + 40 * seed) * time), rate, format="WAV")
    return buffer.getvalue()


def _image(seed: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (seed * 20 % 255, 80, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


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


def _run(media, source, tmp_path, count=10):
    return list(
        source_episodes(
            media, source, count=count, work=tmp_path / "work", inputs=tmp_path / "inputs"
        )
    )


def test_a_json_array_is_decoded_one_item_at_a_time_whatever_the_chunking():
    items = [{"a": i, "s": 'é"],\\,' + "x" * (i % 17), "n": [1, {"k": None}]} for i in range(200)]
    for text in (
        json.dumps(items),
        json.dumps(items, indent=2),
        json.dumps(items, separators=(",", ":")) + "\n",
    ):
        for chunk in (1, 5, 64, 1 << 20):
            stream = io.StringIO(text)
            stream.read(1)  # the "[" the caller has already consumed
            assert list(_json_array(stream, chunk)) == items
    stream = io.StringIO("[ ]")
    stream.read(1)
    assert list(_json_array(stream)) == []
    stream = io.StringIO('[{"a": 1}, {"a":')
    stream.read(1)
    with pytest.raises(json.JSONDecodeError):
        list(_json_array(stream, 4))


def test_a_zipped_json_table_is_read_one_kind_of_row_at_a_time(tmp_path, media):
    """SVIT shape: one zip holds a JSON array of several kinds of conversation (the id says
    which), each row naming a picture that sits in another zip."""
    source = make_source(
        "kinds",
        "image_questions",
        "archive",
        columns={"image": "member", "turns": "conversations", "id": "__key__"},
        include=("raw/images.zip",),
        options={
            "format": "messages",
            "table": {
                "path": "format/all.zip",
                "key": "{image__name}",
                "columns": ["conversations"],
                "merge": ["conversations"],
                "prefix": {"id": "kind_b_"},
            },
        },
    )
    folder = _local(tmp_path, "kinds")
    (folder / "raw").mkdir()
    (folder / "format").mkdir()

    def chat(question, answer):
        return [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]

    rows = [
        {"id": "kind_a_0", "image": "vg/VG_100K/1.png", "conversations": chat("Q-a", "A-a")},
        {"id": "kind_b_0", "image": "vg/VG_100K/1.png", "conversations": chat("Q-b", "A-b")},
        {"id": "kind_b_1", "image": "vg/VG_100K_2/1.png", "conversations": chat("Q-c", "A-c")},
        {"id": "kind_a_1", "image": "vg/VG_100K/2.png", "conversations": chat("Q-d", "A-d")},
    ]
    with zipfile.ZipFile(folder / "format/all.zip", "w") as archive:
        archive.writestr("all.json", json.dumps(rows))
    with zipfile.ZipFile(folder / "raw/images.zip", "w") as archive:
        archive.writestr("VG_100K/1.png", _image(1))
        archive.writestr("VG_100K/2.png", _image(2))
    episodes = _run(media, source, tmp_path)
    assert sorted(e.stream for e in episodes) == ["image_description", "image_identity"]
    chat_episode = next(e for e in episodes if e.stream == "image_description")
    assert [r.payload for r in chat_episode.records[1:]] == [b"Q-b", b"A-b", b"Q-c", b"A-c"]


def test_a_tab_separated_table_without_a_header_names_the_members_of_a_tar_gz(tmp_path, media):
    """GigaSpeech 2 shape: ``<key>\\t<text>`` lines, wavs in a nested folder of a tar.gz."""
    source = make_source(
        "tsv",
        "transcription",
        "archive",
        columns={"audio": "member", "text": "text", "id": "__key__"},
        include=("train/*.tar.gz",),
        options={
            "table": {
                "path": "train_refined.tsv",
                "names": ["key", "text"],
                "key": "{key}.wav",
                "columns": ["text"],
            },
            "family_pattern": r"^(\d+-\d+)-\d+\.wav$",
        },
    )
    folder = _local(tmp_path, "tsv")
    (folder / "train").mkdir()
    (folder / "train_refined.tsv").write_text(
        '0-1-0\tXIN CHAO "BAN" ,\n0-1-1\tTAM BIET\n0-2-0\tCHUA CO FILE\n', encoding="utf-8"
    )
    _tar(
        folder / "train/0.tar.gz",
        {
            "release1/train/0/1/0-1-0.wav": _wave(1),
            "release1/train/0/1/0-1-1.wav": _wave(2),
            "release1/train/0/1/0-1-9.wav": _wave(3),  # not in the table
        },
        "w:gz",
    )
    episodes = _run(media, source, tmp_path)
    asr = [e for e in episodes if e.stream == "transcription"]
    assert sorted(e.records[1].payload for e in asr) == [b"TAM BIET", b'XIN CHAO "BAN" ,']
    assert {e.lineage.family_id for e in asr} == {"tsv:0-1"}


def test_archives_listing_audio_before_transcripts_are_paired_through_scratch_disk(tmp_path, media):
    """WenetSpeech4TTS shape: ``wavs/<id>.wav`` first, ``txts/<id>.txt`` (the id, a tab and the
    transcript; then the time span) after them."""
    source = make_source(
        "wenet",
        "transcription",
        "webdataset",
        columns={"audio": "audio", "text": "text", "id": "__key__"},
        include=("*.tar.gz",),
        options={
            "spool": True,
            "transform": "wenetspeech4tts",
            "family_pattern": r"^(.+)_S\d+(?:-S\d+)?$",
        },
    )
    folder = _local(tmp_path, "wenet")
    _tar(
        folder / "Premium_0.tar.gz",
        {
            "Premium_0/wavs/X1_S00001-S00002.wav": _wave(1),
            "Premium_0/wavs/X1_S00003.wav": _wave(2),
            "Premium_0/wavs/X2_S00001.wav": _wave(3),
            "Premium_0/txts/X1_S00001-S00002.txt": "X1_S00001-S00002\t你好，世界\n[1.0, 3.0]\n".encode(),
            "Premium_0/txts/X1_S00003.txt": "X1_S00003\t再见\n[4.0, 5.0]\n".encode(),
            "Premium_0/txts/X2_S00001.txt": b"X2_S00001\t\n[0.0, 1.0]\n",  # no transcript
        },
        "w:gz",
    )
    episodes = _run(media, source, tmp_path)
    asr = [e for e in episodes if e.stream == "transcription"]
    assert sorted(e.records[1].payload.decode() for e in asr) == ["你好，世界", "再见"]
    assert {e.lineage.family_id for e in asr} == {"wenet:X1"}
    assert not list((tmp_path / "work").rglob("spool-*"))  # the scratch copy is gone
