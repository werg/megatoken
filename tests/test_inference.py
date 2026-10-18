import base64
import io
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from megatoken.controls import build_control_payload, encode_controls
from megatoken.data.chunking import ChunkingConfig, segment_bytes
from megatoken.generation import ActionSpec, ByteCodec
from megatoken.generation.codec import _byte_alphabet
from megatoken.generation.structure import TAPE_VERSION, ActionTape
from megatoken.inference import InferenceEngine, export_served, serve_jsonl
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.tensors import byte_tensor
from megatoken.training.checkpoint import save_checkpoint


def checkpoint_model(root: Path, codec=None, *, max_capacity=8):
    torch.set_num_threads(2)
    torch.manual_seed(901)
    codec = codec or ByteCodec.byte_fallback()
    config = ModelConfig.tiny(max_capacity=max_capacity, vocab_size=codec.vocab_size)
    model = build_model(config, codec)
    checkpoint = save_checkpoint(
        root,
        "fixture",
        {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "codec_fingerprint": codec.fingerprint,
            "chunking": ChunkingConfig().with_hard_limit(config.max_capacity).identity(codec),
            "global_step": 7,
            "train_config": {"precision": "fp32"},
        },
        {"global_step": 7},
        pinned=True,
    )
    model.core.set_model_version(7)
    return model, checkpoint


@pytest.fixture
def bundle(tmp_path, request):
    model, checkpoint = checkpoint_model(
        tmp_path / "checkpoint", max_capacity=getattr(request, "param", 8)
    )
    path = tmp_path / "served"
    metadata = export_served(checkpoint, path)
    return model, path, metadata


def test_export_preserves_the_canonical_facet_policy_and_recurrent_state_exactly(bundle):
    original, directory, metadata = bundle
    weights = load_file(str(directory / "model.safetensors"))
    assert not any(
        any(old in name for old in ("posterior", "carry", "alignment")) for name in weights
    )
    assert metadata["training_heads"] is False and metadata["donor_assets_required"] is False
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    payload = b"actual \xff observation"
    engine.observe(session, payload, role="tool")
    state = original.core.initial_state(1)
    chunks = segment_bytes(payload, original.codec, engine.chunking)
    with torch.no_grad():
        for chunk in chunks:
            raw, lengths = byte_tensor([chunk.payload], "cpu")
            state = original.core.observe(raw, state, lengths=lengths, role_ids=torch.tensor([3]))
    assert b"".join(event.payload for event in session.events) == payload
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    expected = original.sample(state, spec=spec, generator=torch.Generator().manual_seed(887))
    actual = engine.generate(session, spec=spec, seed=887)
    assert expected.tape == actual.tape
    for left, right in (
        (expected.state.facets, actual.state.facets),
        (expected.state.early, actual.state.early),
        (expected.state.final, actual.state.final),
        (expected.state.history.keys, actual.state.history.keys),
        (expected.state.history.values, actual.state.history.values),
    ):
        assert torch.equal(left, right)
    assert session.records[-1] == ("assistant", actual.payload)
    assert len(session.events) == len(chunks) + 1 and len(session.tapes) == 1
    assert metadata["chunking"] == engine.chunking.identity(engine.codec)
    assert metadata["policy_likelihood"] == "canonical_facet_gaussian_density"
    assert metadata["tape_version"] == TAPE_VERSION


def test_observed_chat_examples_are_one_authoritative_event_before_chunking(bundle):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    payload = b"<|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello<|im_end|>"
    expected = segment_bytes(
        payload, engine.codec, engine.chunking, content_kind="text", chat_template="none"
    )
    engine.observe(
        session, payload, role="tool", content_kind="text", filename="example.txt", locale="en"
    )
    assert [event.payload for event in session.events] == [part.payload for part in expected]
    assert all(role == "tool" for role, _ in session.records)
    assert b"".join(event.payload for event in session.events) == payload


def test_serving_uses_the_training_control_bytes_and_retains_them_for_replay(bundle):
    original, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    tools = [{"name": "read", "parameters": {"type": "object", "properties": {}}}]
    seed, reference = b"policy \xff", b"known evidence \x00"
    payload = build_control_payload(
        seed=seed,
        reference=reference,
        reference_origin="external_evidence",
        tools_json=json.dumps(tools, sort_keys=True, separators=(",", ":")),
    )
    expected = original.sample(
        original.core.initial_state(1),
        spec=spec,
        control_payload=payload,
        generator=torch.Generator().manual_seed(318),
    )
    actual = engine.generate(
        session,
        spec=spec,
        seed=318,
        control_seed=seed,
        control_reference=reference,
        reference_origin="external_evidence",
        tools_json=json.dumps(tools, indent=2),
    )
    assert expected.tape == actual.tape
    assert actual.tape.control_payload == payload
    assert torch.equal(expected.state.facets, actual.state.facets)
    # The selected action writes once immediately; controls write no extra record.
    assert actual.state.history.slots == engine.config.history_slots
    assert actual.state.records_seen.item() == 1
    assert len(session.events) == len(session.records) == 1


def test_invalid_reference_provenance_cannot_advance_a_served_session(bundle):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    before = session.state
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="causal"):
        engine.generate(
            session, control_reference=b"hidden target", reference_origin="current_target"
        )
    assert session.state is before
    assert not session.records and not session.tapes and not session.events
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_served_tape_protocol_is_explicit(bundle):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["tape_version"] = "legacy-protocol"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tape protocol"):
        InferenceEngine(directory, device="cpu")


def test_legacy_bundle_format_is_rejected_before_weight_loading(bundle):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["format_version"] = 2
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="v4 contextual realization"):
        InferenceEngine(directory, device="cpu")


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
            "chunking": ChunkingConfig()
            .with_hard_limit(model.config.max_capacity)
            .identity(model.codec),
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
    assert generated["tape"]["valid"] is True
    assert generated["scored_positions"] == 0
    assert generated["latent_dimensions"] == engine.config.n_facets * engine.config.facet_dim
    assert generated["likelihood"] == "canonical_facet_gaussian_density"
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


@pytest.mark.parametrize("bundle", [64], indirect=True)
@pytest.mark.parametrize("field", ["tools", "tools_json"])
def test_jsonl_prior_and_decoder_receive_the_same_public_tools_and_controls(
    bundle, monkeypatch, field
):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    monkeypatch.setattr(engine, "new_session", lambda: session)
    controls, arguments = [], []
    predict, sample = engine.model.core.predict_facets, engine.model.sample

    def capture_predict(state, control=None, effort=6, **kwargs):
        controls.append(control.detach().clone())
        return predict(state, control=control, effort=effort, **kwargs)

    def capture_sample(*args, **kwargs):
        arguments.append(kwargs)
        return sample(*args, **kwargs)

    monkeypatch.setattr(engine.model.core, "predict_facets", capture_predict)
    monkeypatch.setattr(engine.model, "sample", capture_sample)
    tools = [{"name": "read", "parameters": {"type": "object", "properties": {}}}]
    request = {
        "op": "generate",
        "session": session.session_id,
        "control_seed": "write a caf\u00e9 summary",
        "control_reference": "earlier evidence",
        "reference_origin": "earlier_history",
        "seed": 209,
        field: tools if field == "tools" else json.dumps(tools, indent=4),
    }
    requests = [{"op": "open"}, request]
    output = io.StringIO()
    serve_jsonl(engine, io.StringIO("\n".join(map(json.dumps, requests))), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert all(response["ok"] for response in responses), responses
    payload = build_control_payload(
        seed=request["control_seed"].encode(),
        reference=request["control_reference"].encode(),
        reference_origin=request["reference_origin"],
        tools_json=json.dumps(tools),
    )
    assert len(arguments) == len(controls) == 1
    assert arguments[0]["control_payload"] == payload
    assert arguments[0]["generator"].initial_seed() == 209
    assert arguments[0]["spec"].schemas
    with torch.no_grad():
        expected = encode_controls(engine.model.core, [payload], encoder_grad=False)
    torch.testing.assert_close(controls[0], expected, atol=0, rtol=0)
    assert session.tapes[0].control_payload == payload


def test_jsonl_invalid_control_fields_preserve_the_session(bundle, monkeypatch):
    _, directory, _ = bundle
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    monkeypatch.setattr(engine, "new_session", lambda: session)
    invalid_fields = [
        {"control_seed": 19},
        {"control_reference": "future target", "reference_origin": "current_target"},
        {"control_reference": "unattributed"},
        {"tools": [], "tools_json": "[]"},
    ]
    requests = [
        {"op": "open"},
        *({"op": "generate", "session": session.session_id, **fields} for fields in invalid_fields),
        {"op": "status", "session": session.session_id},
    ]
    output = io.StringIO()
    serve_jsonl(engine, io.StringIO("\n".join(map(json.dumps, requests))), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [response["ok"] for response in responses] == [True, False, False, False, False, True]
    assert responses[-1]["records"] == responses[-1]["events"] == 0


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


def test_served_chunk_policy_is_checked_before_observations(bundle):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["chunking"]["config"]["target_tokens"] = 1
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="[Cc]hunk"):
        InferenceEngine(directory, device="cpu")


@pytest.mark.parametrize("bundle", [64], indirect=True)
def test_cli_generation_shares_tool_and_raw_control_inputs_with_python(
    bundle, tmp_path, capsys, monkeypatch
):
    from megatoken.cli import main
    from megatoken.tool_schema import compile_tool_spec

    _, directory, _ = bundle
    tools_text = json.dumps(
        [{"name": "read", "parameters": {"type": "object", "properties": {}}}],
        indent=2,
    )
    tools_path = tmp_path / "tools.json"
    seed_path, reference_path = tmp_path / "control.bin", tmp_path / "reference.bin"
    tools_path.write_text(tools_text, encoding="utf-8")
    seed_bytes, reference_bytes = b"policy \xff\x00", b"earlier evidence \xfe"
    seed_path.write_bytes(seed_bytes)
    reference_path.write_bytes(reference_bytes)
    reads, original_read = [], Path.read_text

    def count_tool_reads(path, *args, **kwargs):
        if path == tools_path:
            reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_tool_reads)
    assert (
        main(
            [
                "generate",
                "--model",
                str(directory),
                "--device",
                "cpu",
                "--prompt",
                "choose",
                "--tools",
                str(tools_path),
                "--control-seed-file",
                str(seed_path),
                "--control-reference-file",
                str(reference_path),
                "--reference-origin",
                "earlier_history",
                "--max-tokens",
                "64",
                "--seed",
                "317",
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    tape = ActionTape.from_json(json.dumps(response["tape"]))
    assert reads == [tools_path]
    assert tape.control_payload == build_control_payload(
        seed=seed_bytes,
        reference=reference_bytes,
        reference_origin="earlier_history",
        tools_json=tools_text,
    )
    assert tape.control_payload
    # The numeric CLI seed still selects the same Gaussian draw as Python;
    # raw control-file bytes are independent model inputs, including non-UTF8.
    engine = InferenceEngine(directory, device="cpu")
    session = engine.new_session()
    engine.observe(session, b"choose")
    expected = engine.generate(
        session,
        spec=compile_tool_spec(tools_text, 64),
        seed=317,
        control_seed=seed_bytes,
        control_reference=reference_bytes,
        reference_origin="earlier_history",
        tools_json=tools_text,
    )
    assert tape == expected.tape
