"""``megatoken collection``: prepared roots and raw files into one searchable collection."""

from __future__ import annotations

import json

import pytest
import torch
from retrieval_support import FakePreparer, FakeTeacher, tiny_model
from test_multi_root_training import media_bridge_roots

from megatoken.cli import main
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.retrieval.collection import SourceCollection, text_item_id
from megatoken.teacher import PINNED_TEACHER


@pytest.fixture
def fake_preparer(monkeypatch):
    monkeypatch.setattr(
        "megatoken.retrieval.cli.make_preparer", lambda packs, root, device: FakePreparer(root)
    )


def files(tmp_path):
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "note.txt").write_text("The vault opens with key 31.")
    (folder / "manual.md").write_text("Section one. " * 1000)  # long: read as pieces
    (folder / "photo.png").write_bytes(b"png bytes")
    (folder / "talk.wav").write_bytes(b"wav bytes")
    return folder


def test_build_add_and_info_round_trip(tmp_path, fake_preparer, capsys):
    torch.set_num_threads(1)
    _, media_root = media_bridge_roots(tmp_path)
    output = tmp_path / "collection"
    assert (
        main(
            ["collection", "build", "--name", "mixed", "--output", str(output)]
            + ["--items", str(media_root), str(files(tmp_path)), "--packs", "locked.json"]
            + ["--threshold", "0.4", "--cells", "3"]
        )
        == 0
    )
    built = json.loads(capsys.readouterr().out)
    # Three prepared clips; a passage, the long manual in three pieces, an image and a recording.
    assert built["added"] == {"prepared": 3, "files": 1 + 3 + 1 + 1}
    assert built["modalities"] == {
        "audio": 2,
        "compressed_text": 3,
        "image": 2,
        "text": 1,
        "video": 1,
    }
    assert built["threshold"] == 0.4 and built["index"]["cells"] == 3
    assert built["indexed_items"] == built["items"] == 9

    # A later file joins the built collection and is found by its own embedding.
    extra = tmp_path / "extra.txt"
    extra.write_text("A new fact about the vault.")
    assert (
        main(["collection", "add", "--path", str(output), "--items", str(extra), "--packs", "x"])
        == 0
    )
    added = json.loads(capsys.readouterr().out)
    assert added["added"]["files"] == 1 and added["items"] == 10
    assert main(["collection", "info", "--path", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["items"] == 10
    with SourceCollection.open(output) as collection:
        key = FakeTeacher().encode(
            "text", "A new fact about the vault.", prompt="title: none | text: "
        )
        (hit,) = collection.search(key.embedding.numpy(), 1, probes=3)
        item = collection.item(hit.row)
        assert item.text == b"A new fact about the vault." and item.origin == str(extra)
        assert item.item_id == text_item_id(item.text)
        pieces = [i for i in map(collection.item, range(1, 11)) if i.modality == "compressed_text"]
        assert [i.origin.rpartition("#bytes=")[2] for i in pieces] == [
            "0-5000",
            "5000-10000",
            "10000-13000",
        ]
        # Every medium carries its original file reference.
        assert {i.origin for i in map(collection.item, range(1, 11)) if i.modality == "image"} >= {
            str(tmp_path / "files" / "photo.png")
        }


def test_raw_files_need_the_pack_manifest(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(
            ["collection", "build", "--name", "x", "--output", str(tmp_path / "c")]
            + ["--items", str(files(tmp_path))]
        )
    assert "--packs" in capsys.readouterr().err


def test_a_running_session_adds_files_and_finds_them(tmp_path):
    torch.set_num_threads(1)
    folder = tmp_path / "start"
    folder.mkdir()
    (folder / "first.txt").write_text("Opening hours are nine to five.")
    SourceCollection.create(tmp_path / "docs", name="docs", teacher=PINNED_TEACHER).close()
    preparer = FakePreparer(tmp_path / "scratch")
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(tiny_model()),
        collections={"docs": tmp_path / "docs"},
        preparer=preparer,
    )
    assert runtime.add_files("docs", [folder / "first.txt"]) == 1
    runtime.memory.collections["docs"].set_threshold(-1.0)
    session = runtime.open({}, effort=1)
    assert [e.item.origin for e in session.memory.sources.log] == [str(folder / "first.txt")]
    # Files added while sessions exist are found by the next lookup.
    (folder / "second.txt").write_text("Closed on public holidays.")
    assert runtime.add_files("docs", [folder / "second.txt"]) == 1
    from megatoken.model.multitrack import ObservedEvent

    runtime.observe(session, (ObservedEvent("main", "text", b"x", role_id=3),), effort=1)
    assert [e.item.origin for e in session.memory.sources.log] == [
        str(folder / "first.txt"),
        str(folder / "second.txt"),
    ]
    # Adding the same file again adds nothing.
    assert runtime.add_files("docs", [folder / "second.txt"]) == 0
