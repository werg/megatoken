"""Mounting knowledge bases per session and adding files to one while serving."""

from __future__ import annotations

import hashlib
import io
import json

import pytest
import torch
from media_support import prepared_record

from megatoken.contracts import ROLE_TO_ID
from megatoken.data.media_assets import TensorAssetStore
from megatoken.generation.codec import ByteCodec
from megatoken.kb import model_identity
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.transducer import metadata_payload
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.multitrack_service import serve_multitrack_jsonl
from megatoken.omni.embeddinggemma2 import TeacherOutput
from megatoken.teacher import HIDDEN_WIDTH, PINNED_TEACHER
from megatoken.training.meridians import _boundary_meridian
from megatoken.training.transducer import TransducerTrainer


class StubTeacher:
    """The teacher's two calls with deterministic features: one row per 64 bytes."""

    def text_pieces(self, text, *, prompt=""):
        return [text[i : i + 4000] for i in range(0, len(text), 4000)]

    def encode(self, modality, source, *, prompt=""):
        generator = torch.Generator().manual_seed(len(source))
        return TeacherOutput(
            torch.randn(max(1, len(source) // 64), HIDDEN_WIDTH, generator=generator),
            torch.nn.functional.normalize(torch.ones(768), dim=0),
            modality,
            PINNED_TEACHER,
        )


class StubPreparer:
    """The existing preparation path (teacher features into an asset store), minus the real teacher."""

    def __init__(self, root):
        self.store, self.teacher = TensorAssetStore(root), StubTeacher()

    def media(self, modality, path, *, target=True):
        seed = int(hashlib.sha256(str(path).encode()).hexdigest()[:6], 16)
        return prepared_record(self.store, modality, seed=seed, target=False)


def _model():
    torch.manual_seed(4)
    return build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).eval()


def _runtime(tmp_path):
    return MultiTrackRuntime(
        MultiTrackGenerator(_model()), preparer=lambda: StubPreparer(tmp_path / "prepared")
    )


def _tool():
    return (ObservedEvent("main", "text", b"result", role_id=ROLE_TO_ID["tool"]),)


def _files(tmp_path):
    (tmp_path / "notes.txt").write_text("The vault key is kept in the north cabinet. " * 20)
    (tmp_path / "photo.png").write_bytes(b"not read by the stub")
    return [tmp_path / "notes.txt", tmp_path / "photo.png"]


def test_files_added_while_serving_are_found_by_sessions_that_mount_the_base(tmp_path):
    runtime = _runtime(tmp_path)
    memory = runtime.memory
    memory.create_base("mine")
    early = runtime.open({"task": "q"}, bases={"mine": 1.0})  # opened before any file exists
    assert [m.name for m in early.memory.agenda.mounted] == ["trained", "mine"]
    added = memory.add_files(_files(tmp_path), base="mine").base
    assert len(added) == 2 and memory.bases["mine"].edge_count == 2
    # No weight changed; the next lookup of a session that already mounted the base sees them.
    lookup = memory.observe(early.memory, _tool())
    assert {(name, edge) for _, name, edge in lookup.pairs} == {("mine", e) for e in added}
    assert len(early.memory.agenda.shortlists[1][0]) == 2
    # The item's source is remembered per entry.
    assert set(memory.bases["mine"].sources.values()) == {"notes.txt", "photo.png"}
    late = runtime.open({"task": "q"}, bases={"mine": 1.0})
    assert {n for _, n, _ in memory.observe(late.memory, _tool()).pairs} == {"mine"}


def test_blend_weights_unknown_names_and_the_trained_weight_per_session(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.memory.create_base("mine")
    runtime.memory.add_files(_files(tmp_path), base="mine")
    off = runtime.open(bases={"mine": 0.0})
    assert [m.name for m in off.memory.agenda.mounted] == ["trained"]
    only = runtime.open(bases={"mine": 2.0}, trained=0.0)
    assert [(m.name, m.weight) for m in only.memory.agenda.mounted] == [("mine", 2.0)]
    with pytest.raises(ValueError, match="No open knowledge base named"):
        runtime.open(bases={"missing": 1.0})
    with pytest.raises(ValueError, match="already open"):
        runtime.memory.create_base("mine")
    runtime.memory.close_base("mine")
    with pytest.raises(ValueError, match="No open knowledge base"):
        runtime.open(bases={"mine": 1.0})


def test_adding_files_needs_a_writable_base_and_a_preparer(tmp_path):
    runtime = _runtime(tmp_path)
    memory = runtime.memory
    memory.create_base("mine")
    memory.add_files(_files(tmp_path), base="mine")
    memory.bases["mine"].save(tmp_path / "mine.kb")
    sealed = memory.open_base("sealed", tmp_path / "mine.kb")
    with pytest.raises(RuntimeError, match="read-only"):
        memory.add_files(_files(tmp_path), base="sealed")
    assert sealed.edge_count == 2
    bare = MultiTrackRuntime(MultiTrackGenerator(runtime.generator.generator))
    bare.memory.create_base("x")
    with pytest.raises(ValueError, match="media preparer"):
        bare.memory.add_files(_files(tmp_path), base="x")


def test_a_base_built_for_other_weights_is_refused(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.memory.create_base("mine")
    runtime.memory.add_files(_files(tmp_path), base="mine")
    runtime.memory.bases["mine"].save(tmp_path / "mine.kb")
    torch.manual_seed(77)
    other = MultiTrackRuntime(
        MultiTrackGenerator(build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).eval())
    )
    assert (
        model_identity(other.generator.generator)["weights"] != runtime.memory.identity["weights"]
    )
    with pytest.raises(ValueError, match="different weights"):
        other.memory.open_base("mine", tmp_path / "mine.kb")


def test_serving_looks_up_exactly_as_training_does_across_bases(tmp_path):
    runtime = _runtime(tmp_path)
    memory, model = runtime.memory, runtime.generator.generator
    memory.create_base("mine")
    memory.add_files(_files(tmp_path), base="mine")
    metadata = {"task": "q", "modality": ["text"]}
    served = memory.new_session(metadata, bases={"mine": 0.5}, effort=1)
    # The training path (an evaluation window) with the same bases mounted, step by step.
    trainer = TransducerTrainer(model.transducer)
    model.transducer.eval()
    mounted = model.transducer.bank.mount({"mine": (memory.bases["mine"], 0.5)})
    step = type("Step", (), {})()
    step.tracks, step.roles, step.modalities = ("main",), (ROLE_TO_ID["user"],), ("text",)
    step.records = (
        type("R", (), {"record": type("P", (), {"part_index": 0, "part_count": 1})()})(),
    )
    sequence = trainer.sequence([_boundary_meridian(step)], step=0, mounted=mounted)
    multi = runtime.generator
    with torch.inference_mode():
        state = sequence.begin(multi, multi.tracks.initial_state(1), metadata_payload(metadata))
        sequence.before_meridian(multi, state, 0, effort=1)
    torch.testing.assert_close(sequence.agenda.state, served.agenda.state)
    assert len(sequence.agenda.locations) == len(served.agenda.locations)
    assert sequence.agenda.versions == served.agenda.versions


# ------------------------------------------------------------------ the protocol


def _serve(model, requests, *, kb_root=None, **options):
    runtime = MultiTrackRuntime(MultiTrackGenerator(model), **options)
    results = io.StringIO()
    serve_multitrack_jsonl(runtime, requests(results), results, kb_root=kb_root)
    return [json.loads(line) for line in results.getvalue().splitlines()]


def _line(value):
    return json.dumps(value) + "\n"


def test_jsonl_protocol_creates_fills_saves_and_mounts_bases_inside_the_root(tmp_path):
    root = tmp_path / "kb"
    root.mkdir()
    _files(root)
    model = _model()

    def requests(results):
        yield _line({"op": "kb_create", "id": 1, "name": "mine"})
        yield _line(
            {"op": "add_files", "id": 2, "base": "mine", "paths": ["notes.txt", "photo.png"]}
        )
        yield _line({"op": "add_files", "id": 3, "base": "mine", "paths": ["../outside.txt"]})
        yield _line({"op": "open", "id": 4, "bases": {"mine": 1.0}, "trained_weight": 0.5})
        yield _line({"op": "kb_save", "id": 5, "name": "mine", "path": "mine.kb"})
        yield _line({"op": "kb_open", "id": 6, "name": "again", "path": "mine.kb"})
        yield _line({"op": "add_files", "id": 7, "base": "again", "paths": ["notes.txt"]})
        yield _line({"op": "kb_list", "id": 8})
        yield _line({"op": "kb_close", "id": 9, "name": "again"})
        yield _line({"op": "kb_nonsense", "id": 10, "name": "mine"})

    rows = _serve(model, requests, kb_root=root, preparer=lambda: StubPreparer(root / "prepared"))
    by_id = {row["id"]: row for row in rows}
    assert by_id[1]["ok"] and by_id[1]["writable"] and by_id[1]["edges"] == 0
    assert by_id[2]["ok"] and by_id[2]["base"]["added"] == 2 and by_id[2]["base"]["edges"] == 2
    assert not by_id[3]["ok"] and "outside the knowledge base root" in by_id[3]["error"]
    assert by_id[4]["ok"]
    assert by_id[5]["ok"] and (root / "mine.kb").exists()
    assert by_id[6]["ok"] and by_id[6]["edges"] == 2 and not by_id[6]["writable"]
    assert not by_id[7]["ok"] and "read-only" in by_id[7]["error"]
    assert set(by_id[8]["bases"]) == {"mine", "again"}
    assert by_id[9]["closed"] == "again"
    assert not by_id[10]["ok"] and "Unknown knowledge base operation" in by_id[10]["error"]


def test_knowledge_operations_are_off_without_a_root():
    model = _model()

    def requests(results):
        yield _line({"op": "kb_create", "id": 1, "name": "mine"})
        yield _line({"op": "kb_open", "id": 2, "name": "x", "path": "x.kb"})
        yield _line({"op": "open", "id": 3, "bases": {"mine": 1.0}})

    rows = _serve(model, requests)
    assert rows[0]["ok"]  # an in-memory user base needs no files
    assert not rows[1]["ok"] and "--kb-root" in rows[1]["error"]
    assert rows[2]["ok"]


# ------------------------------------------------------------------ the command line


def test_kb_build_and_inspect_write_a_base_that_the_model_it_was_built_for_mounts(tmp_path, capsys):
    from test_reading_base import reading_dataset
    from test_served import checkpoint_model

    from megatoken.cli import main
    from megatoken.served import ServedBundle, export_served

    _, checkpoint = checkpoint_model(tmp_path / "checkpoint")
    bundle = tmp_path / "served"
    export_served(checkpoint, bundle)
    dataset = reading_dataset(tmp_path / "corpus", 3)
    out = tmp_path / "demo.kb"
    # Built from the exported bundle ...
    argv = ["--name", "demo", "--items", str(dataset), "--device", "cpu"]
    assert main(["kb", "build", *argv, "--model", str(bundle), "--output", str(out)]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["edges"] == 9 and built["output"] == str(out)  # 3 episodes x 3 items
    # ... and from the checkpoint it came from: the same weights, so the same identity.
    twin = tmp_path / "twin.kb"
    twin_argv = ["--model", str(checkpoint), "--codec", "byte", "--output", str(twin)]
    assert main(["kb", "build", *argv, *twin_argv]) == 0
    capsys.readouterr()
    assert main(["kb", "inspect", str(out)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["edges"] == 9 and shown["sources"] == 9 and shown["name"] == "demo"
    assert main(["kb", "inspect", str(twin)]) == 0
    assert json.loads(capsys.readouterr().out)["identity"] == shown["identity"]
    # The model it was built for opens it; the identity it carries is that model's.
    runtime = MultiTrackRuntime(MultiTrackGenerator(ServedBundle(bundle, device="cpu").model))
    base = runtime.memory.open_base("demo", out)
    assert base.edge_count == 9 and runtime.memory.identity == shown["identity"]
    session = runtime.open({"task": "base_reading"}, bases={"demo": 1.0})
    lookup = runtime.memory.observe(session.memory, _tool())
    assert {name for _, name, _ in lookup.pairs} == {"demo"}
    # Files and roots are not mixed in one build.
    with pytest.raises(ValueError, match="not both"):
        mixed = ["--name", "x", "--items", str(dataset), str(out), "--model", str(bundle)]
        main(["--debug", "kb", "build", *mixed])


def test_failure_in_later_ingest_group_publishes_nothing(tmp_path, monkeypatch):
    from megatoken.data.knowledge import PreparedFile, knowledge_items
    from megatoken.kb import add_items

    runtime = _runtime(tmp_path)
    base = runtime.memory.create_base("atomic")
    preparer = StubPreparer(tmp_path / "assets")
    items = knowledge_items([PreparedFile(p, preparer) for p in _files(tmp_path)])
    native = runtime.memory.native
    original = native.ingest_rows
    calls = 0

    def fail_later(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second group failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "ingest_rows", fail_later)
    with pytest.raises(RuntimeError, match="second group failed"):
        add_items(base, runtime.generator.generator, runtime.generator, items, batch=1)
    assert calls == 2 and base.edge_count == 0 and not base.sources
    assert base.version == 0 and base.table.next_id == 1
