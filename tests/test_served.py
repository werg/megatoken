"""The served bundle and the one serving runtime over it: exports, loading, the protocol."""

import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from megatoken import multitrack_service
from megatoken.controls import build_control_payload, encode_controls
from megatoken.data.chunking import ChunkingConfig, segment_bytes
from megatoken.generation import ActionSpec, ByteCodec
from megatoken.generation.codec import _byte_alphabet
from megatoken.loading import build_model
from megatoken.model.config import ARCHITECTURE_VERSION, ModelConfig
from megatoken.model.multitrack import (
    GROUP_TAPE_VERSION,
    GroupActionTape,
    MultiTrackGenerator,
    ObservedEvent,
)
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.multitrack_service import MultiTrackService, serve_multitrack_jsonl
from megatoken.served import ServedBundle, export_served
from megatoken.tool_schema import compile_tool_spec, recent_copy_sources
from megatoken.training.checkpoint import save_checkpoint, transducer_attachment


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
            "train_config": {"precision": "fp32", "packs": None},
            "packs_fingerprint": None,
        },
        {"global_step": 7},
        pinned=True,
        attachments=transducer_attachment(lambda: {"bank": model.transducer.bank.state_snapshot()}),
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


def served(directory, **options):
    """The serving runtime over an exported bundle: the one program everything serves with."""
    bundle = ServedBundle(directory, device="cpu")
    runtime = MultiTrackRuntime(
        MultiTrackGenerator(bundle.model), chunking=bundle.chunking, **options
    )
    return bundle, runtime


def reply(spec, **options):
    """A one-output generation request for the assistant's track."""
    return {"track": "assistant", "modality": "text", **spec, **options}


def ask(service, **request):
    return service.handle(request)


def test_export_preserves_the_canonical_facet_policy_and_recurrent_state_exactly(bundle):
    original, directory, metadata = bundle
    weights = load_file(str(directory / "model.safetensors"))
    assert not any(
        any(old in name for old in ("posterior", "carry", "alignment")) for name in weights
    )
    assert metadata["training_heads"] is False and metadata["packs_fingerprint"] is None
    # The adapter mixture's experts and controllers are served weights.
    adapters = {f"core.adapters.{name}" for name in original.core.adapters.state_dict()}
    assert adapters <= weights.keys()
    loaded, runtime = served(directory)
    # The sparse bank ships beside the weights and is the trained one.
    assert loaded.model.transducer.bank.bank_id == original.transducer.bank.bank_id
    # The same program over the checkpoint's own model must do exactly what the bundle does:
    # lookups, adapter mixture and all.
    reference = MultiTrackRuntime(MultiTrackGenerator(original.eval()), chunking=loaded.chunking)
    payload = b"actual \xff observation"
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    results, sessions = [], []
    for program in (runtime, reference):
        session = program.open(effort=1)
        program.observe_text(session, "tool", payload, role="tool", effort=1)
        results.append(
            program.generate(
                session,
                tracks=("assistant",),
                specs=(spec,),
                effort=1,
                generator=torch.Generator().manual_seed(887),
            )
        )
        sessions.append(session)
    actual, expected = results
    assert actual.tape == expected.tape
    for left, right in (
        (actual.state.raw, expected.state.raw),
        (actual.state.final, expected.state.final),
        (actual.state.history.keys, expected.state.history.keys),
        (actual.state.history.values, expected.state.history.values),
    ):
        assert torch.equal(left, right)
    # Every observed chunk and the answer are in the session's text history, in order.
    assert b"".join(sessions[0].history[:-1]) == payload
    assert sessions[0].history[-1] == actual.tape.payloads[0]
    assert metadata["chunking"] == loaded.chunking.identity(loaded.codec)
    assert (
        metadata["policy_likelihood"]
        == "modality_categorical_plus_canonical_facet_gaussian_density"
    )
    assert metadata["tape_version"] == GROUP_TAPE_VERSION


def test_a_served_generation_replays_from_its_group_tape(bundle):
    _, directory, _ = bundle
    _, runtime = served(directory)
    session = runtime.open(effort=1)
    runtime.observe_text(session, "user", b"hello there", effort=1)
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    before = session.state
    result = runtime.generate(
        session,
        tracks=("assistant",),
        specs=(spec,),
        effort=1,
        generator=torch.Generator().manual_seed(11),
    )
    tape = GroupActionTape.from_json(result.tape.to_json())
    assert tape == result.tape and tape.valid == (True,) and tape.tracks == ("assistant",)
    replayed = runtime.generator.replay_group(before, tape, specs=(spec,), effort=1)
    assert replayed.tape == tape


def test_observed_chat_examples_are_one_authoritative_event_before_chunking(bundle):
    _, directory, _ = bundle
    loaded, runtime = served(directory)
    session = runtime.open(effort=1)
    payload = b"<|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello<|im_end|>"
    expected = segment_bytes(
        payload, loaded.codec, loaded.chunking, content_kind="text", chat_template="none"
    )
    count = runtime.observe_text(
        session,
        "tool",
        payload,
        role="tool",
        effort=1,
        content_kind="text",
        filename="example.txt",
        locale="en",
    )
    assert count == len(expected) and session.history == [part.payload for part in expected]
    assert b"".join(session.history) == payload
    # Only the last chunk ends the message; each chunk is one meridian on the track.
    assert [tracks for _, tracks in session.events] == [("tool",)] * count


def test_a_user_message_ends_with_its_last_chunk_and_a_lookup_follows(bundle):
    _, directory, _ = bundle
    loaded, runtime = served(directory)
    session = runtime.open(effort=1)
    payload = bytes(range(97, 117))
    chunks = segment_bytes(payload, loaded.codec, loaded.chunking, chat_template="none")
    assert len(chunks) > 1
    lookups = [session.memory.agenda.lookups]
    for index, chunk in enumerate(chunks):
        event = ObservedEvent("user", "text", chunk.payload, terminal=index == len(chunks) - 1)
        runtime.observe(session, (event,), effort=1)
        lookups.append(session.memory.agenda.lookups)
    # Nothing is asked until the message is whole; then the lookup has read all of it.
    assert lookups == [1] * len(chunks) + [2]


def test_text_beyond_the_chunk_policy_is_refused_as_one_event(bundle):
    _, directory, _ = bundle
    loaded, runtime = served(directory)
    session = runtime.open(effort=1)
    before = session.state
    huge = b"x" * (loaded.chunking.max_bytes + 1)
    with pytest.raises(ValueError, match="chunk policy"):
        runtime.observe(session, (ObservedEvent("user", "text", huge),))
    assert session.state is before and not session.history


def test_serving_uses_the_training_control_bytes_and_retains_them_for_replay(bundle):
    original, directory, _ = bundle
    _, runtime = served(directory)
    session = runtime.open(effort=1)
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    tools = [{"name": "read", "parameters": {"type": "object", "properties": {}}}]
    seed, reference = b"policy \xff", b"known evidence \x00"
    payload = build_control_payload(
        seed=seed,
        reference=reference,
        reference_origin="external_evidence",
        tools_json=json.dumps(tools, sort_keys=True, separators=(",", ":")),
    )
    expected = MultiTrackRuntime(MultiTrackGenerator(original.eval()))
    expected_session = expected.open(effort=1)
    kwargs = dict(tracks=("assistant",), specs=(spec,), effort=1)
    want = expected.generate(
        expected_session,
        controls=(payload,),
        generator=torch.Generator().manual_seed(318),
        **kwargs,
    )
    got = runtime.generate(
        session,
        controls=(payload,),
        generator=torch.Generator().manual_seed(318),
        **kwargs,
    )
    assert got.tape == want.tape and got.tape.controls == (payload,)
    # The selected action writes once; the control writes no event of its own.
    assert torch.equal(got.state.raw, want.state.raw)
    assert session.events[-1][1] == ("assistant",)


# ---------------------------------------------------------------- the protocol


def test_requests_keep_exact_bytes_and_survive_errors(bundle):
    _, directory, _ = bundle
    loaded, runtime = served(directory)
    service = MultiTrackService(runtime)
    opened = ask(service, op="open", effort=1)
    sid = opened["session"]
    assert opened["ok"]
    observed = ask(
        service,
        op="observe",
        session=sid,
        effort=1,
        events=[
            {
                "track": "tool",
                "role": "tool",
                "payload_b64": base64.b64encode(b"\x00\xff").decode(),
            }
        ],
    )
    assert observed["ok"]
    bad = ask(
        service,
        op="observe",
        session=sid,
        events=[{"track": "tool", "role": "tool", "payload_b64": "!not-base64"}],
    )
    assert not bad["ok"]
    generated = ask(
        service,
        op="generate",
        id=4,
        session=sid,
        seed=89,
        effort=1,
        outputs=[reply({"kind": "ascii", "min_tokens": 2, "max_tokens": 2})],
    )
    assert generated["ok"] and generated["id"] == 4
    (answer,) = generated["outputs"]
    assert base64.b64decode(answer["payload_b64"]) == answer["text"].encode()
    assert loaded.codec.decode(answer["emitted_ids"]) == answer["text"].encode()
    assert answer["valid"] is True and generated["tape"]["valid"] == [True]
    assert generated["tape"]["tracks"] == ["assistant"]
    status = ask(service, op="status", session=sid)
    assert status["ok"] and status["track_registry"]
    assert ask(service, op="close", session=sid)["closed"] == sid
    assert not ask(service, op="status", session=sid)["ok"]


def test_a_text_message_is_cut_by_the_bundles_chunk_policy(bundle):
    _, directory, _ = bundle
    loaded, runtime = served(directory)
    service = MultiTrackService(runtime)
    sid = ask(service, op="open", effort=1)["session"]
    text = "x " * 80
    expected = segment_bytes(
        text.encode(), loaded.codec, loaded.chunking, content_kind="text", chat_template="none"
    )
    answer = ask(
        service,
        op="observe_text",
        session=sid,
        track="user",
        text=text,
        content_kind="text",
        effort=1,
    )
    assert answer["ok"] and answer["chunks"] == len(expected)
    assert ask(service, op="observe_text", session=sid, track="user", role="future", text="x")[
        "error"
    ].startswith("ValueError: Unknown")
    assert not ask(service, op="observe_text", session=sid, track="user", text="a", payload_b64="")[
        "ok"
    ]


def test_the_session_limit_and_malformed_lines_are_bounded(bundle):
    _, directory, _ = bundle
    _, runtime = served(directory)
    output = io.StringIO()
    serve_multitrack_jsonl(
        runtime,
        io.StringIO('{"op":"open","effort":1}\n{"op":"open","effort":1}\nnot json\n[1]\n'),
        output,
        max_sessions=1,
    )
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[0]["ok"] is True
    assert responses[1]["ok"] is False and "capacity" in responses[1]["error"]
    assert responses[2]["ok"] is False and responses[3]["ok"] is False


@pytest.mark.parametrize("bundle", [64], indirect=True)
@pytest.mark.parametrize("field", ["tools", "tools_json"])
def test_prior_and_decoder_receive_the_same_public_tools_and_controls(bundle, monkeypatch, field):
    _, directory, _ = bundle
    _, runtime = served(directory)
    service = MultiTrackService(runtime)
    sid = ask(service, op="open", effort=1)["session"]
    controls, specs = [], []
    tracks = runtime.generator.tracks
    predict, realize = tracks.predict_group, runtime.generator._realize

    def capture_predict(state, **kwargs):
        controls.append(kwargs["control"].detach().clone())
        return predict(state, **kwargs)

    def capture_realize(raw, plans, *args, **kwargs):
        specs.extend(plans)
        return realize(raw, plans, *args, **kwargs)

    monkeypatch.setattr(tracks, "predict_group", capture_predict)
    monkeypatch.setattr(runtime.generator, "_realize", capture_realize)
    tools = [{"name": "read", "parameters": {"type": "object", "properties": {}}}]
    output = reply(
        {
            "max_tokens": 64,
            "control_seed": "write a café summary",
            "control_reference": "earlier evidence",
            "reference_origin": "earlier_history",
            field: tools if field == "tools" else json.dumps(tools, indent=4),
        }
    )
    answer = ask(service, op="generate", session=sid, seed=209, effort=1, outputs=[output])
    assert answer["ok"], answer
    payload = build_control_payload(
        seed=output["control_seed"].encode(),
        reference=output["control_reference"].encode(),
        reference_origin=output["reference_origin"],
        tools_json=json.dumps(tools),
    )
    assert len(controls) == 1 and specs[0].schemas
    with torch.no_grad():
        expected = encode_controls(runtime.generator.generator.core, [payload], encoder_grad=False)
    torch.testing.assert_close(controls[0].reshape(expected.shape), expected, atol=0, rtol=0)
    assert GroupActionTape.from_json(json.dumps(answer["tape"])).controls == (payload,)


@pytest.mark.parametrize("bundle", [64], indirect=True)
def test_a_tool_call_may_copy_from_the_text_the_session_holds(bundle):
    _, directory, _ = bundle
    _, runtime = served(directory)
    session = runtime.open(effort=1)
    runtime.observe_text(session, "user", b"read the file notes.txt", effort=1)
    assert session.history == [b"read the file notes.txt"]
    tools = [{"name": "read", "parameters": {"type": "object", "properties": {}}}]
    output = reply({"max_tokens": 64, "tools": tools})
    spec, _ = multitrack_service._output(output, 64, session.history)
    assert any(source.payload == b"read the file notes.txt" for source in spec.copy_sources)


def test_invalid_control_fields_preserve_the_session(bundle):
    _, directory, _ = bundle
    _, runtime = served(directory)
    service = MultiTrackService(runtime)
    sid = ask(service, op="open", effort=1)["session"]
    before = ask(service, op="status", session=sid)
    rng = torch.random.get_rng_state().clone()
    invalid_fields = [
        {"control_seed": 19},
        {"control_reference": "future target", "reference_origin": "current_target"},
        {"control_reference": "unattributed"},
        {"tools": [], "tools_json": "[]"},
        {"tools": [{"name": "broken", "parameters": {"type": "not-a-json-type"}}]},
        {"control_b64": "AA==", "control_seed": "also"},
        {"temperature": 0.7},
        {"max_tokens": None},
    ]
    for fields in invalid_fields:
        answer = ask(service, op="generate", session=sid, outputs=[reply(fields)])
        assert answer["ok"] is False, fields
    assert ask(service, op="status", session=sid) == before
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("bundle", [64], indirect=True)
def test_cli_generation_shares_tool_and_raw_control_inputs_with_python(
    bundle, tmp_path, capsys, monkeypatch
):
    from megatoken.cli import main

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
    argv = [
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
        "--effort",
        "1",
        "--seed",
        "317",
    ]
    assert main(argv) == 0
    response = json.loads(capsys.readouterr().out)
    tape = GroupActionTape.from_json(json.dumps(response["tape"]))
    assert reads == [tools_path]
    payload = build_control_payload(
        seed=seed_bytes,
        reference=reference_bytes,
        reference_origin="earlier_history",
        tools_json=tools_text,
    )
    assert tape.controls == (payload,) and payload
    assert response["track"] == "assistant" and "valid" in response
    # The numeric CLI seed still selects the same Gaussian draw as Python;
    # raw control-file bytes are independent model inputs, including non-UTF8.
    _, runtime = served(directory)
    session = runtime.open(effort=1)
    runtime.observe_text(session, "assistant", b"choose", effort=1)
    expected = runtime.generate(
        session,
        tracks=("assistant",),
        specs=(
            replace(
                compile_tool_spec(tools_text, 64),
                copy_sources=recent_copy_sources([b"choose"]),
            ),
        ),
        controls=(payload,),
        effort=1,
        generator=torch.Generator().manual_seed(317),
    )
    assert tape == expected.tape


def test_generate_is_a_one_shot_session_and_checks_its_mounts(bundle, tmp_path, capsys):
    from megatoken.cli import main

    _, directory, _ = bundle
    prompt = tmp_path / "prompt.txt"
    prompt.write_bytes(b"What is the code word?")
    argv = ["generate", "--model", str(directory), "--device", "cpu", "--prompt-file", str(prompt)]
    assert main([*argv, "--effort", "1", "--max-tokens", "4"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["tape"]["tracks"] == ["assistant"] and "retrieved" not in answer
    with pytest.raises(ValueError, match="NAME="):
        main(["--debug", *argv, "--collection", "main"])
    with pytest.raises(ValueError, match="NAME="):
        main(["--debug", *argv, "--base", "mine"])
    with pytest.raises(RuntimeError, match="object"):
        main(["--debug", *argv, "--metadata", "[1]"])


def test_a_malformed_request_never_ends_the_serving_loop(bundle, monkeypatch):
    _, directory, _ = bundle
    _, runtime = served(directory)
    service = MultiTrackService(runtime)
    sid = ask(service, op="open", effort=1)["session"]
    generate = {"op": "generate", "session": sid, "effort": 1}
    original = runtime.generate

    def nan_density(session, **kwargs):
        result = original(session, **kwargs)
        if kwargs["generator"].initial_seed() == 99:
            result = replace(result, log_prob=torch.tensor(float("nan")))
        return result

    monkeypatch.setattr(runtime, "generate", nan_density)
    outputs = [reply({"min_tokens": 1, "max_tokens": 2})]
    lines = [
        {**generate, "id": 1, "outputs": [reply({"terminal": 2})]},  # an out-of-range flag
        {**generate, "id": 2, "outputs": [reply({"kind": ["ascii"]})]},
        {**generate, "id": 3, "outputs": [reply({"max_tokens": None})]},
        {**generate, "id": 4, "outputs": outputs, "seed": 99},  # a NaN density is not strict JSON
        {**generate, "id": 5, "outputs": outputs, "seed": 1},  # the loop is still serving
    ]
    output = io.StringIO()
    service.serve(io.StringIO("\n".join(map(json.dumps, lines))), output)
    responses = [json.loads(text) for text in output.getvalue().splitlines()]
    assert [r["ok"] for r in responses] == [False, False, False, False, True]
    assert responses[0]["error"].startswith("ValueError")
    assert "strict JSON" in responses[3]["error"]
    assert [r.get("id") for r in responses] == [1, 2, 3, 4, 5]


def test_served_tape_protocol_is_explicit(bundle):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["tape_version"] = "legacy-protocol"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tape protocol"):
        ServedBundle(directory, device="cpu")


def test_legacy_bundle_format_is_rejected_before_weight_loading(bundle):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["format_version"] = 2
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"requires architecture v{ARCHITECTURE_VERSION}"):
        ServedBundle(directory, device="cpu")


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
        ServedBundle(directory, device="cpu")


def test_codec_mapping_and_export_immutability_are_enforced(bundle, tmp_path):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["codec_fingerprint"] = "wrong-codec"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mapping differs"):
        ServedBundle(directory)
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
    engine = ServedBundle(directory, device="cpu")
    assert engine.codec.fingerprint == codec.fingerprint
    assert engine.codec.encode(b"ab") == [256]
    assert engine.codec.decode(engine.codec.encode("e\u0301")) == "e\u0301".encode()
    assert engine.codec.vocab_size == 260
    assert manifest["codec_file"] == "tokenizer.json"
    path = directory / "tokenizer.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="tokenizer checksum"):
        ServedBundle(directory)


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
            "train_config": {"precision": "fp32", "packs": None},
            "packs_fingerprint": None,
        },
        {"global_step": 8},
        pinned=True,
        attachments=transducer_attachment(lambda: {"bank": model.transducer.bank.state_snapshot()}),
    )
    with pytest.raises(RuntimeError, match="bake"):
        export_served(imported, tmp_path / "invalid-served")


def test_served_chunk_policy_is_checked_before_observations(bundle):
    _, directory, _ = bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["chunking"]["config"]["target_tokens"] = 1
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="[Cc]hunk"):
        ServedBundle(directory, device="cpu")


def test_bundle_requires_the_pack_manifest_it_was_trained_with(bundle, tmp_path):
    _, directory, _ = bundle
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    with pytest.raises(ValueError, match="pack manifest the bundle was trained with"):
        ServedBundle(directory, packs=lock, device="cpu")


def test_cli_serves_exported_model_and_persists_private_memory_on_eof(bundle, tmp_path, monkeypatch):
    import io
    import sys
    from types import SimpleNamespace

    from megatoken import cli
    from megatoken.kb import model_identity
    from megatoken.memory.knowledge import KnowledgeBase

    original, directory, _ = bundle
    output = io.StringIO()
    root = tmp_path / "private"
    root.mkdir()

    def requests():
        yield json.dumps(dict(op="open", scope="alice",
                              write_policy={"lag": 4}, effort=1)) + "\n"
        opened = json.loads(output.getvalue().splitlines()[-1])
        assert opened["ok"] and opened["memory_writes"] and opened["overlay"]
        for text in ("fact", "result"):
            yield json.dumps(dict(op="observe_text", session=opened["session"],
                                  track="main", text=text, effort=1)) + "\n"

    monkeypatch.setattr(sys, "stdin", requests())
    monkeypatch.setattr(sys, "stdout", output)
    cli._serve(SimpleNamespace(model=str(directory), packs=None, device="cpu", compile=False,
                              collection=[], kb_root=str(root), max_sessions=2))
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 3 and all(row["ok"] for row in responses)
    saved = KnowledgeBase.load(next((root / "scopes").glob("*.kb")), geometry=original.transducer.bank.geometry,
                               identity=model_identity(original))
    assert saved.scope == "alice" and saved.edge_count > 0
