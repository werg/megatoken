import base64
import io
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from megatoken.generation import ActionSpec, ByteCodec
from megatoken.generation.codec import _byte_alphabet
from megatoken.inference import InferenceEngine, export_served, serve_jsonl
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.tensors import byte_tensor
from megatoken.training.checkpoint import save_checkpoint


def checkpoint_model(root: Path, codec=None):
    torch.set_num_threads(2)
    torch.manual_seed(901)
    codec = codec or ByteCodec.byte_fallback()
    config = ModelConfig.tiny(max_capacity=8, vocab_size=codec.vocab_size)
    model = build_model(config, codec)
    checkpoint = save_checkpoint(
        root,
        "fixture",
        {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "codec_fingerprint": codec.fingerprint,
            "global_step": 7,
            "train_config": {"precision": "fp32"},
        },
        {"global_step": 7},
        pinned=True,
    )
    model.core.set_model_version(7)
    return model, checkpoint


@pytest.fixture
def bundle(tmp_path):
    model, checkpoint = checkpoint_model(tmp_path / "checkpoint")
    path = tmp_path / "served"
    metadata = export_served(checkpoint, path)
    return model, path, metadata


def test_export_removes_training_heads_and_preserves_policy_exactly(bundle):
    original, directory, metadata = bundle
    weights = load_file(str(directory / "model.safetensors"))
    assert not any(name.startswith("posterior.") for name in weights)
    assert metadata["training_heads"] is False and metadata["donor_assets_required"] is False
    engine = InferenceEngine(directory, device="cpu")
    assert engine.model.posterior is None
    session = engine.new_session()
    engine.observe(session, b"actual \xff observation", role="tool")
    state = original.core.initial_state(1)
    raw, lengths = byte_tensor([b"actual \xff observation"], "cpu")
    with torch.no_grad():
        state = original.core.ingest(raw, state, lengths=lengths, role_ids=torch.tensor([3]))
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4, max_repairs=0)
    expected = original.sample(state, spec=spec, generator=torch.Generator().manual_seed(887))
    actual = engine.generate(session, spec=spec, seed=887)
    assert expected.tape == actual.tape
    assert torch.equal(expected.state.facets, actual.state.facets)
    assert torch.equal(expected.state.carry, actual.state.carry)
    assert session.records[-1] == ("assistant", actual.payload)
    assert len(session.events) == 2 and len(session.tapes) == 1


@pytest.mark.parametrize(
    "name,error",
    [("model.safetensors", "weights checksum"), ("model_config.json", "configuration checksum")],
)
def test_modified_served_files_fail_integrity_checks(bundle, name, error):
    _, directory, _ = bundle
    path = directory / name
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match=error):
        InferenceEngine(directory, device="cpu")


def test_codec_mapping_and_export_immutability_are_enforced(bundle, tmp_path):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["codec_fingerprint"] = "wrong-codec"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mapping differs"):
        InferenceEngine(directory)
    _, checkpoint = checkpoint_model(tmp_path / "another-checkpoint")
    with pytest.raises(FileExistsError, match="immutable"):
        export_served(checkpoint, directory)


def write_tokenizer(path: Path):
    alphabet = {value: text for text, value in _byte_alphabet().items()}
    vocabulary = {alphabet[value]: value for value in range(256)}
    vocabulary["ab"] = 256
    document = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {"type": "NFC"},
        "post_processor": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [["a", "b"]],
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        },
    }
    path.write_text(json.dumps(document))


def test_loaded_tokenizer_is_bundled_and_verified_without_external_assets(tmp_path):
    asset = tmp_path / "original-tokenizer.json"
    write_tokenizer(asset)
    codec = ByteCodec.from_tokenizer_json(asset, expected_vocab_size=260)
    _, checkpoint = checkpoint_model(tmp_path / "checkpoint", codec)
    directory = tmp_path / "served"
    manifest = export_served(checkpoint, directory, codec_path=str(asset))
    asset.unlink()
    engine = InferenceEngine(directory, device="cpu")
    assert engine.codec.fingerprint == codec.fingerprint
    assert engine.codec.encode(b"ab") == [256]
    assert engine.codec.decode(engine.codec.encode("e\u0301")) == "e\u0301".encode()
    assert engine.codec.vocab_size == 260
    assert manifest["codec_file"] == "tokenizer.json"
    path = directory / "tokenizer.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="tokenizer checksum"):
        InferenceEngine(directory)


def test_unbaked_import_is_rejected_at_export(tmp_path):
    model, checkpoint = checkpoint_model(tmp_path / "base")
    model.core.frontend.ngrams.install_import(
        torch.tensor([1]), torch.ones(1, 128), torch.ones(1), torch.ones(1)
    )
    imported = save_checkpoint(
        tmp_path / "imported",
        "fixture",
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "codec_fingerprint": model.codec.fingerprint,
            "global_step": 8,
            "train_config": {"precision": "fp32"},
        },
        {"global_step": 8},
        pinned=True,
    )
    with pytest.raises(RuntimeError, match="bake"):
        export_served(imported, tmp_path / "invalid-served")


def test_session_model_ownership_and_roles_are_checked(bundle):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    with pytest.raises(ValueError, match="Unknown"):
        engine.observe(session, b"x", role="future")
    session.model_id = "another-model"
    with pytest.raises(ValueError, match="another served model"):
        engine.generate(session)


def test_jsonl_sessions_keep_exact_bytes_and_survive_request_errors(bundle, monkeypatch):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    new_session = engine.new_session

    def stable_session():
        session = new_session()
        session.session_id = "session"
        return session

    monkeypatch.setattr(engine, "new_session", stable_session)
    requests = [
        {"id": 1, "op": "open"},
        {
            "id": 2,
            "op": "observe",
            "session": "session",
            "role": "tool",
            "payload_b64": base64.b64encode(b"\x00\xff").decode(),
        },
        {"id": 3, "op": "observe", "session": "session", "payload_b64": "!not-base64"},
        {
            "id": 4,
            "op": "generate",
            "session": "session",
            "kind": "ascii",
            "min_tokens": 2,
            "max_tokens": 2,
            "max_repairs": 0,
            "seed": 89,
        },
        {"id": 5, "op": "status", "session": "session"},
        {"id": 6, "op": "close", "session": "session"},
        {"id": 7, "op": "status", "session": "session"},
    ]
    output = io.StringIO()
    serve_jsonl(engine, io.StringIO("\n".join(map(json.dumps, requests))), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [item["id"] for item in responses] == list(range(1, 8))
    assert [item["ok"] for item in responses] == [True, True, False, True, True, True, False]
    generated = responses[3]
    assert base64.b64decode(generated["payload_b64"]) == generated["text"].encode()
    assert engine.codec.decode(generated["emitted_ids"]) == generated["text"].encode()
    assert generated["tape"]["committed"] is True
    assert responses[4]["records"] == 2 and responses[4]["events"] == 2


def test_jsonl_session_limit_and_malformed_lines_are_bounded(bundle):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    output = io.StringIO()
    serve_jsonl(
        engine, io.StringIO('{"op":"open"}\n{"op":"open"}\nnot json\n'), output, max_sessions=1
    )
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is False and "limit" in responses[1]["error"]["message"]
    assert responses[2]["ok"] is False


def test_jsonl_invalid_tool_schema_is_a_request_error_and_preserves_the_session(
    bundle, monkeypatch
):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    monkeypatch.setattr(engine, "new_session", lambda: session)
    requests = [
        {"op": "open"},
        {
            "op": "generate",
            "session": session.session_id,
            "tools": [{"name": "broken", "parameters": {"type": "not-a-json-type"}}],
        },
        {"op": "status", "session": session.session_id},
    ]
    output = io.StringIO()
    serve_jsonl(engine, io.StringIO("\n".join(map(json.dumps, requests))), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [response["ok"] for response in responses] == [True, False, True]
    assert responses[2]["records"] == 0 and responses[2]["events"] == 0
