import json
from dataclasses import replace

import pytest
import torch
from jsonschema.exceptions import ValidationError

from megatoken.generation import (
    ActionSpec,
    ByteCodec,
    CopySource,
    FacetedGenerator,
    FieldSpec,
    SchemaSpec,
)
from megatoken.generation.feasibility import wire_witness_tokens
from megatoken.generation.structure import ActionTape, Observation, TapeMismatch
from megatoken.model import FacetedCore, ModelConfig
from megatoken.tensors import byte_tensor


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(617)
    return FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback())


def assert_state_equal(first, second):
    assert torch.equal(first.facets, second.facets)
    assert torch.equal(first.records_seen, second.records_seen)
    assert torch.equal(first.last_record_ids, second.last_record_ids)
    for name in ("final",):
        a, b = getattr(first, name), getattr(second, name)
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    # Derived attention intermediates may choose a different reduction under
    # no_grad collection and differentiable replay. The taped raw input above
    # remains exact; the recomputed causal cache agrees to FP32 roundoff.
    torch.testing.assert_close(first.history.keys, second.history.keys, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(first.history.values, second.history.values, rtol=1e-6, atol=1e-6)
    if first.history.mask is None or second.history.mask is None:
        assert first.history.mask is second.history.mask
    else:
        assert torch.equal(first.history.mask, second.history.mask)


def assert_replay_density_close(actual, expected, latent_dimensions):
    # Recomputed no_grad/autograd attention and the FP32 density reduction may
    # differ by a few rounding steps. Bound their average error per coordinate,
    # rather than putting an architecture-dependent absolute cap on the sum.
    torch.testing.assert_close(
        actual / latent_dimensions,
        expected / latent_dimensions,
        rtol=0,
        atol=16 * torch.finfo(torch.float32).eps,
    )


def predictor_parameters(model):
    excluded = {id(parameter) for parameter in model.deterministic_decoder_parameters()}
    return [parameter for parameter in model.core.parameters() if id(parameter) not in excluded]


def has_gradient(parameters):
    return any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in parameters
    )


def invalid_materialization_spec(model):
    """A feasible public schema whose actual lexical choice needs JSON escaping.

    Opposite tab/newline vocabulary rows beat every zero row for either sign of
    the projected state; a tie selects tab. Both escaped wires exceed the valid
    printable-character witness. The projection stays live for history-gradient
    tests, and no decoder or feasibility gate is mocked.
    """
    schema = SchemaSpec(
        "escaped_string",
        (FieldSpec("value", kind="ascii", min_tokens=1, max_tokens=1, allow_copy=False),),
    )
    witness_tokens = len(model.codec.encode(schema.serialize({"value": b"a"})))
    assert wire_witness_tokens(schema, model.codec) == witness_tokens
    assert model.codec.allowed_ids("ascii")[0] == ord("\t")
    assert all(
        len(model.codec.encode(schema.serialize({"value": value}))) > witness_tokens
        for value in (b"\t", b"\n")
    )
    with torch.no_grad():
        model.decoder.vocabulary.embedding.weight.zero_()
        model.decoder.vocabulary.embedding.weight[ord("\t"), 0] = 1
        model.decoder.vocabulary.embedding.weight[ord("\n"), 0] = -1
    return ActionSpec(allow_text=False, schemas=(schema,), max_tokens=witness_tokens)


def test_batched_two_pass_supervision_counts_free_positions_and_commits_once(model):
    state = model.core.initial_state(2)
    plan = model.prepare_target(
        [tuple(b"a\xff"), tuple(b"xyz")],
        free_mask=[(True, False), (True, True, False)],
        capacity=8,
        terminal=[False, True],
    )
    result = model.supervised_step(
        state, plan=plan, role_ids=torch.tensor([2, 4]), generator=torch.Generator().manual_seed(42)
    )
    assert result.losses["lexical"][1] == result.losses["reconstruction"][1] == 3
    assert plan.scored_positions == 6
    assert result.losses["structure"][1] == result.losses["terminal"][1] == 4
    assert result.losses["facet_prediction"][1] == 2 * model.latent_dimensions
    assert all(result.losses[name][1] == count for name, count in plan.denominators.items())
    assert result.state.final is not None
    assert state.final is None
    assert result.state.records_seen.tolist() == [1, 1]
    assert result.state.role_ids.tolist() == [2, 4]
    assert state.records_seen.tolist() == [0, 0]
    sum(value / count for value, count in result.losses.values() if count).backward()
    assert has_gradient(model.facet_distribution.parameters())
    assert has_gradient(model.decoder.vocabulary.parameters())
    assert has_gradient(model.core.encoder_parameters())
    assert has_gradient(model.core.realizer_parameters())


def test_detached_facet_target_reconstruction_and_prior_have_distinct_gradients(model):
    plan = model.prepare_target(target_bytes=b"learn", capacity=8)
    result = model.supervised_step(model.core.initial_state(1), plan=plan)
    result.losses["facet_prediction"][0].backward(retain_graph=True)
    assert not has_gradient(model.core.encoder_parameters())
    assert not has_gradient(model.decoder.parameters())
    assert has_gradient(predictor_parameters(model))
    assert has_gradient(model.facet_distribution.parameters())
    model.zero_grad(set_to_none=True)
    result.losses["reconstruction"][0].backward(retain_graph=True)
    assert has_gradient(model.core.encoder_parameters())
    assert has_gradient(model.decoder.parameters())
    assert not has_gradient(model.facet_distribution.parameters())
    assert not has_gradient(predictor_parameters(model))
    model.zero_grad(set_to_none=True)
    result.losses["lexical"][0].backward()
    assert has_gradient(predictor_parameters(model))
    assert has_gradient(model.facet_distribution.parameters())
    assert not has_gradient(model.core.encoder_parameters())


def test_served_predictive_facets_do_not_see_target_bytes_or_length(model, monkeypatch):
    captured = []
    original = model.decoder.teacher_raw

    def capture(core, raw, plan):
        captured.append(raw.facets.detach().clone())
        return original(core, raw, plan)

    monkeypatch.setattr(model.decoder, "teacher_raw", capture)
    state = model.core.initial_state(1)
    for target in (b"a", b"different"):
        plan = model.prepare_target(target_bytes=target, capacity=16)
        model.supervised_step(state, plan=plan, generator=torch.Generator().manual_seed(122))
    assert torch.equal(captured[0], captured[2])  # both prior samples
    assert not torch.equal(captured[1], captured[3])  # content-dependent canonical encodings


def test_one_gaussian_draw_and_no_stochastic_downstream_work(model):
    state = model.core.initial_state(1)
    generator = torch.Generator().manual_seed(37)
    expected_generator = torch.Generator().manual_seed(37)
    torch.randn((1, model.config.n_facets, model.config.facet_dim), generator=expected_generator)
    global_rng = torch.random.get_rng_state().clone()
    result = model.sample(state, spec=ActionSpec(min_tokens=4, max_tokens=4), generator=generator)
    assert torch.equal(generator.get_state(), expected_generator.get_state())
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert result.scored_positions == result.tape.scored_positions == 0
    assert result.latent_dimensions == result.tape.latent_dimensions == model.latent_dimensions
    raw = torch.from_numpy(result.tape.facet_array().copy()).reshape(1, *result.tape.facet_shape)
    decoded = model.decode_facets(raw, spec=ActionSpec(min_tokens=4, max_tokens=4))
    torch.manual_seed(9981)
    again = model.decode_facets(raw, spec=ActionSpec(min_tokens=4, max_tokens=4))
    assert decoded.payload == again.payload == result.payload
    assert decoded.emitted_ids == again.emitted_ids == result.emitted_ids
    assert decoded.trace == again.trace == result.tape.trace


def test_generated_facets_commit_raw_without_encoding_decoded_bytes(model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("generated bytes were re-encoded into canonical state")

    monkeypatch.setattr(model.core, "encode_chunk", forbidden)
    state = model.core.initial_state(1)
    spec = ActionSpec(min_tokens=4, max_tokens=4)
    sampled = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(9))
    raw = torch.from_numpy(sampled.tape.facet_array().copy()).reshape(1, *sampled.tape.facet_shape)
    expected = model.core.advance_raw(state, raw, role_ids=torch.tensor([2]))
    assert_state_equal(sampled.state, expected)
    scored = model.score_tape(state, sampled.tape, spec=spec)
    assert_state_equal(sampled.state, scored.state)
    assert torch.equal(sampled.log_prob, scored.log_prob)


def test_teacher_processes_clean_canonical_target_from_original_history(model):
    values, lengths = byte_tensor([b"earlier history"], "cpu")
    state = model.core.observe(values, model.core.initial_state(1), lengths=lengths)
    plan = model.prepare_target(target_bytes=b"observed", role_ids=[4])
    result = model.supervised_step(state, plan=plan)
    values, lengths = byte_tensor(plan.payloads, state.device)
    canonical = model.core.encode_chunk(values, lengths=lengths)
    expected = model.core.advance_raw(state, canonical, role_ids=torch.tensor([4]))
    assert_state_equal(result.state, expected)
    assert result.state.records_seen.item() == state.records_seen.item() + 1


def test_full_facet_tape_roundtrip_density_and_integrity(model):
    state = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=7)
    sampled = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(13), rl=True)
    restored = ActionTape.from_json(sampled.tape.to_json())
    assert restored == sampled.tape
    assert not hasattr(restored, "root") and not hasattr(restored, "regional")
    assert not hasattr(restored, "attempts") and not hasattr(restored, "committed")
    scored = model.score_tape(state, restored, spec=spec)
    assert torch.equal(sampled.log_prob, scored.log_prob)
    assert_state_equal(sampled.state, scored.state)
    assert model.codec.decode(sampled.emitted_ids) == sampled.payload
    (-scored.log_prob).backward()
    assert has_gradient(predictor_parameters(model))
    with pytest.raises(TapeMismatch):
        model.score_tape(state, replace(restored, payload=b"forged"), spec=spec)
    with pytest.raises(TapeMismatch):
        model.score_tape(state, replace(restored, trace=()), spec=spec)
    with pytest.raises(TapeMismatch):
        model.score_tape(state, restored, spec=replace(spec, min_tokens=0))
    with pytest.raises(ValueError, match="complete raw facet"):
        replace(restored, facets=restored.facets[:-1])
    legacy = json.loads(restored.to_json())
    legacy["version"] = "fmt-augmented-policy-v1"
    with pytest.raises(ValueError, match="legacy"):
        ActionTape.from_json(json.dumps(legacy))
    with pytest.raises(ValueError, match="sampling controls"):
        model.sample(state, temperature=0.7)
    with pytest.raises(ValueError, match="effort 6"):
        model.sample(state, rl=True, effort=3)


def test_invalid_materialization_still_advances_and_replays_raw_computation(model, monkeypatch):
    # An impossible fixed schema now fails feasibility *before* sampling.
    schema = SchemaSpec("integer", (FieldSpec("value", kind="json", fixed_json="0"),))
    state = model.core.initial_state(1)
    with pytest.raises(ValueError, match="no public output mode fits"):
        model.sample(
            state,
            spec=ActionSpec(allow_text=False, schemas=(schema,), max_tokens=1),
            generator=torch.Generator().manual_seed(5),
        )

    # Actual lexical invalidity remains a first-class sampled outcome. Force
    # one invalid UTF-8 byte, and verify that its *raw* facets still advance
    # causal state and replay exactly.
    spec = ActionSpec(text_kind="utf8", min_tokens=1, max_tokens=1)
    monkeypatch.setattr(
        model.decoder.vocabulary,
        "argmax",
        lambda states, allowed: torch.full(
            (states.shape[0],), 255, device=states.device, dtype=torch.long
        ),
    )
    result = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(5))
    assert not result.tape.valid and "UnicodeDecodeError" in result.tape.error
    assert result.state.final is not None and state.final is None
    assert result.state.records_seen.item() == 1
    assert not torch.equal(state.facets, result.state.facets)
    replay = model.score_tape(state, result.tape, spec=spec)
    assert torch.equal(result.log_prob, replay.log_prob)
    assert_state_equal(result.state, replay.state)


def test_invalid_json_escaping_advances_and_replays_raw_computation(model):
    spec = invalid_materialization_spec(model)
    state = model.core.initial_state(1)
    result = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(5))
    assert not result.tape.valid and "including fixed syntax" in result.tape.error
    assert result.payload in {
        spec.schemas[0].serialize({"value": value}) for value in (b"\t", b"\n")
    }
    assert result.state.final is not None and state.final is None
    assert result.state.records_seen.item() == 1
    assert not torch.equal(state.facets, result.state.facets)
    replay = model.score_tape(state, result.tape, spec=spec)
    assert torch.equal(result.log_prob, replay.log_prob)
    assert_state_equal(result.state, replay.state)


def test_replay_reconstructs_state_and_candidate_density_depends_on_history(model):
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    initial = model.core.initial_state(1)
    prompt = Observation(b"choose the next chunk", role_id=1)
    factual, _ = model.replay(initial, [prompt])
    first = model.sample(factual, spec=spec, generator=torch.Generator().manual_seed(1))
    second = model.sample(first.state, spec=spec, generator=torch.Generator().manual_seed(2))
    state, probabilities = model.replay(
        initial, [prompt, first.tape, second.tape], specs=[spec, spec]
    )
    assert_state_equal(state, second.state)
    assert_replay_density_close(probabilities[1], second.log_prob, model.latent_dimensions)
    raw = torch.from_numpy(second.tape.facet_array().copy()).reshape(1, *second.tape.facet_shape)
    isolated = model.facet_distribution(model.core.predict_facets(initial)).log_prob(raw).sum()
    assert not torch.allclose(isolated, second.log_prob)


def test_freezing_preserves_entire_encoder_and_action_map_under_latent_update(model):
    names = model.freeze_decoder_for_rl()
    frozen_ids = {id(parameter) for parameter in model.deterministic_decoder_parameters()}
    assert names and all(
        not parameter.requires_grad for parameter in model.deterministic_decoder_parameters()
    )
    assert all(parameter.requires_grad for parameter in predictor_parameters(model))
    assert all(parameter.requires_grad for parameter in model.facet_distribution.parameters())
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if id(parameter) in frozen_ids
    }
    state = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=3)
    sampled = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(93))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.1)
    scored = model.score_tape(state, sampled.tape, spec=spec)
    (-scored.log_prob / model.latent_dimensions).backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in before:
            assert parameter.grad is None
            assert torch.equal(parameter, before[name])
    candidate = model.score_tape(state, sampled.tape, spec=spec)
    assert candidate.payload == sampled.payload
    assert not torch.equal(candidate.log_prob, sampled.log_prob)


def test_typed_identifiers_copies_and_placements_have_two_pass_counts(model):
    schema = SchemaSpec(
        "write",
        (
            FieldSpec("path", kind="identifier", min_tokens=1, max_tokens=8, identifier="p"),
            FieldSpec("same_path", kind="identifier", min_tokens=1, max_tokens=8, bound_to="p"),
            FieldSpec("content", kind="utf8", max_tokens=20),
        ),
    )
    spec = ActionSpec(
        schemas=(schema,),
        copy_sources=(CopySource("earlier", b"copied"),),
        max_tokens=128,
        require_terminal=False,
    )
    plan = model.prepare_action_target(
        b'{"path":"x","same_path":"x","content":"!copied?"}', spec=spec, bindings={"content": 1}
    )
    assert plan.scored_positions == 6  # unique identifier + two free bytes, each in two passes
    result = model.supervised_action(model.core.initial_state(1), plan=plan)
    assert result.losses["binding"][1] == 4
    assert result.losses["placement"][1] == 2
    assert result.losses["lexical"][1] == result.losses["reconstruction"][1] == 3
    sum(loss / count for loss, count in result.losses.values() if count).backward()
    assert model.decoder.binding_query.weight.grad.abs().sum() > 0
    assert model.decoder.identifier_slots.weight.grad.abs().sum() > 0
    assert model.decoder.schema_codes.weight.grad.abs().sum() > 0  # downstream conditioning


def test_fixed_copy_identifier_and_json_types_are_preserved(model):
    schema = SchemaSpec(
        "pair",
        (
            FieldSpec("id", kind="identifier", min_tokens=2, max_tokens=2, identifier="i"),
            FieldSpec("ref", kind="identifier", min_tokens=2, max_tokens=2, bound_to="i"),
            FieldSpec("v", kind="json", fixed_json='{"n":2,"ok":true}'),
        ),
        layouts=(("id", "ref", "v"), ("v", "id", "ref")),
    )
    spec = ActionSpec(
        allow_text=False, schemas=(schema,), copy_sources=(CopySource("id", b"ab"),), max_tokens=128
    )
    plan = model.prepare_action_target(
        b'{"id":"ab","ref":"ab","v":{"n":2,"ok":true}}', spec=spec, bindings={"id": 1}, layout=1
    )
    assert plan.scored_positions == 0
    result = model.supervised_action(model.core.initial_state(1), plan=plan)
    assert result.losses["lexical"][1] == result.losses["reconstruction"][1] == 0
    raw = torch.randn(1, model.config.n_facets, model.config.facet_dim)
    decoded = model.decoder(model.core, raw, spec, teacher=plan)
    parsed = json.loads(decoded.payload)
    assert parsed["id"] == parsed["ref"] == "ab"
    assert parsed["v"]["ok"] is True
    assert model.codec.decode(decoded.emitted_ids) == plan.payload


def test_nested_json_and_encoded_content_validate_declared_types(model):
    schema = SchemaSpec(
        "call",
        (
            FieldSpec("name", kind="json", fixed_json='"edit"'),
            FieldSpec(
                "arguments",
                kind="json",
                max_tokens=80,
                json_schema='{"type":"object","properties":{"n":{"type":"integer"},"ok":{"type":"boolean"}},"required":["n","ok"]}',
            ),
        ),
    )
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=128)
    plan = model.prepare_action_target(b'{"name":"edit","arguments":{"n":2,"ok":true}}', spec=spec)
    result = model.supervised_action(model.core.initial_state(1), plan=plan)
    assert result.losses["lexical"][1] == len(b'{"n":2,"ok":true}')
    with pytest.raises(ValidationError):
        model.prepare_action_target(b'{"name":"edit","arguments":{"n":"bad","ok":true}}', spec=spec)
    encoded = SchemaSpec(
        "encoded",
        (
            FieldSpec(
                "arguments",
                kind="json",
                max_tokens=100,
                json_schema=json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "string",
                                "contentMediaType": "application/json",
                                "contentSchema": {"type": "integer"},
                            }
                        },
                        "required": ["data"],
                    }
                ),
            ),
        ),
    )
    public = ActionSpec(allow_text=False, schemas=(encoded,), max_tokens=128)
    assert model.prepare_action_target(b'{"arguments":{"data":"2"}}', spec=public)
    with pytest.raises(ValidationError):
        model.prepare_action_target(b'{"arguments":{"data":"false"}}', spec=public)
    with pytest.raises(ValidationError):
        FieldSpec("fixed", kind="json", fixed_json="true", json_schema='{"type":"integer"}')


def test_total_output_limit_and_shared_identifier_minima(model):
    schema = SchemaSpec(
        "pair",
        (
            FieldSpec("a", kind="identifier", min_tokens=0, max_tokens=3, identifier="id"),
            FieldSpec("b", kind="identifier", min_tokens=2, max_tokens=3, bound_to="id"),
        ),
    )
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=32)
    decoded = model.decode_facets(torch.randn(1, 4, 32), spec=spec)
    assert decoded.valid
    parsed = json.loads(decoded.payload)
    assert 2 <= len(parsed["a"]) <= 3 and parsed["a"] == parsed["b"]
    with pytest.raises(ValueError, match="including fixed syntax"):
        model.prepare_action_target(b'{"a":"ab","b":"ab"}', spec=replace(spec, max_tokens=4))
    with pytest.raises(ValueError, match="no public output mode"):
        model.sample(model.core.initial_state(1), spec=replace(spec, max_tokens=3))


def test_empty_terminal_chunk_is_a_real_raw_facet_event(model):
    result = model.sample(
        model.core.initial_state(1),
        spec=ActionSpec(min_tokens=0, max_tokens=0, require_terminal=True),
    )
    assert result.payload == b"" and result.scored_positions == 0 and result.tape.terminal
    assert result.tape.valid and result.state.final is not None
    assert result.state.records_seen.item() == 1


def test_metrics_are_detached_and_report_both_decoder_passes(model):
    plan = model.prepare_target(
        target_bytes=[b"abc", b"de"], capacity=4, free_mask=[(True, False, True), (True, True)]
    )
    output = model.supervised_step(model.core.initial_state(2), plan=plan)
    metrics = output.metrics
    assert metrics["prior_examples"] == metrics["reconstruction_examples"] == 2
    assert (
        metrics["prior_lexical_positions"] + metrics["reconstruction_lexical_positions"]
        == plan.scored_positions
    )
    torch.testing.assert_close(
        metrics["prior_lexical_loss_sum"], output.losses["lexical"][0].detach()
    )
    torch.testing.assert_close(
        metrics["reconstruction_lexical_loss_sum"], output.losses["reconstruction"][0].detach()
    )
    assert all(
        not value.requires_grad for value in metrics.values() if isinstance(value, torch.Tensor)
    )


def test_served_export_contains_all_shared_owners_and_is_identical(model):
    served = FacetedGenerator(FacetedCore(model.config), model.codec, training_heads=False)
    weights = model.served_state_dict()
    assert set(weights) == {n for n in model.state_dict() if "hindsight_realizer" not in n}
    assert not any(
        "posterior" in name or "regional" in name or "root_prior" in name or "carry" in name
        for name in weights
    )
    served.load_state_dict(weights, strict=True)
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    original = model.sample(
        model.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(15)
    )
    exported = served.sample(
        served.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(15)
    )
    assert original.tape == exported.tape
    with pytest.raises(RuntimeError, match="served model"):
        served.supervised_step(served.core.initial_state(1), target_bytes=b"x")


def test_noncanonical_teacher_ids_remain_actual_ce_targets():
    pieces = [bytes([value]) for value in range(256)] + [b"ab"]
    codec = ByteCodec(pieces, merge_ranks={(b"a", b"b"): 0})
    model = FacetedGenerator(FacetedCore(ModelConfig.tiny(vocab_size=257)), codec)
    plan = model.prepare_target([97, 98])
    assert plan.payloads == (b"ab",) and plan.ids == ((97, 98),)
    assert codec.encode(b"ab") == [256]
    output = model.supervised_step(model.core.initial_state(1), plan=plan)
    assert output.losses["lexical"][1] == output.losses["reconstruction"][1] == 2


def test_small_predictor_and_shared_reconstruction_can_learn_a_byte_task():
    torch.set_num_threads(1)
    torch.manual_seed(947)
    model = FacetedGenerator(FacetedCore(ModelConfig.tiny(max_capacity=4)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=0)
    plan = model.prepare_target(target_bytes=b"AAAA", capacity=4)
    prior_losses, reconstruction_losses = [], []
    for step in range(22):
        result = model.supervised_step(
            model.core.initial_state(1),
            plan=plan,
            generator=torch.Generator().manual_seed(step + 1),
        )
        prior = result.losses["lexical"][0] / 4
        reconstruction = result.losses["reconstruction"][0] / 4
        loss = (
            prior
            + reconstruction
            + 0.05 * result.losses["facet_prediction"][0] / model.latent_dimensions
        )
        prior_losses.append(float(prior.detach()))
        reconstruction_losses.append(float(reconstruction.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    assert sum(prior_losses[-3:]) / 3 < prior_losses[0] * 0.25
    assert sum(reconstruction_losses[-3:]) / 3 < reconstruction_losses[0] * 0.25
    spec = ActionSpec(text_kind="ascii", min_tokens=4, max_tokens=4)
    sampled = model.sample(
        model.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(600)
    )
    assert sampled.payload == b"AAAA"
    values, lengths = byte_tensor([b"AAAA"], "cpu")
    encoded = model.core.encode_chunk(values, lengths=lengths)
    assert model.decode_facets(encoded, spec=spec).payload == b"AAAA"


def test_recurrent_mode_reconstructs_directly_with_frozen_readout_and_two_passes(
    model, monkeypatch
):
    plan = model.prepare_target(target_bytes=b"follow", representation_mode="recurrent")
    calls = []
    original = model.decoder.teacher_raw

    def capture(core, raw, plan):
        calls.append(raw)
        return original(core, raw, plan)

    monkeypatch.setattr(model.decoder, "teacher_raw", capture)
    result = model.supervised_step(model.core.initial_state(1), plan=plan)
    assert len(calls) == 2
    assert result.state.facets is calls[0].facets
    assert calls[1].facets.shape == calls[0].facets.shape
    assert calls[1].facets is not result.state.facets
    assert plan.scored_positions == 12
    assert result.losses["reconstruction"][1] == 6
    assert result.losses["facet_prediction"][1] == result.losses["encoder_alignment"][1] == 0
    result.losses["reconstruction"][0].backward(retain_graph=True)
    assert has_gradient(model.core.encoder_parameters())
    assert not has_gradient(predictor_parameters(model))
    assert not has_gradient(model.decoder.parameters())
    assert not has_gradient(model.facet_distribution.parameters())
    assert all(parameter.grad is None for parameter in model.core.central_parameters())
    assert all(parameter.grad is None for parameter in model.core.realizer_parameters())
    model.zero_grad(set_to_none=True)
    result.losses["lexical"][0].backward()
    assert has_gradient(predictor_parameters(model))
    assert not has_gradient(model.core.encoder_parameters())
    with pytest.raises(ValueError, match="differs from the prepared"):
        model.supervised_step(
            model.core.initial_state(1), plan=plan, representation_mode="bootstrap"
        )


def test_later_token_loss_flows_through_live_recurrent_facet_tower(model):
    plan = model.prepare_target(target_bytes=b"tower", representation_mode="recurrent")
    first = model.supervised_step(model.core.initial_state(1), plan=plan)
    first.state.facets.retain_grad()
    second = model.supervised_step(first.state, plan=plan)
    second.losses["lexical"][0].backward()
    assert first.state.facets.grad is not None and first.state.facets.grad.norm() > 0
    assert has_gradient(predictor_parameters(model))
    assert not has_gradient(model.core.encoder_parameters())


def test_recurrent_observations_detach_encoder_but_preserve_body_context_gradient(model):
    initial = model.core.initial_state(1)
    first_bytes, lengths = byte_tensor([b"system context"], initial.device)
    state = model.core.observe(
        first_bytes, initial, lengths=lengths, role_ids=torch.tensor([0]), encoder_grad=False
    )
    next_bytes, lengths = byte_tensor([b"user asks"], initial.device)
    state = model.core.observe(
        next_bytes, state, lengths=lengths, role_ids=torch.tensor([1]), encoder_grad=False
    )
    assert state.history.slots > 0
    plan = model.prepare_target(target_bytes=b"answer", representation_mode="recurrent")
    result = model.supervised_step(state, plan=plan)
    result.losses["lexical"][0].backward()
    assert not has_gradient(model.core.encoder_parameters())
    assert has_gradient(predictor_parameters(model))


def test_structured_recurrent_counts_cover_live_and_encoder_only_contextual_decoders(model):
    schema = SchemaSpec("text", (FieldSpec("value", kind="ascii", max_tokens=8),))
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=32)
    plan = model.prepare_action_target(
        b'{"value":"ok"}', spec=spec, representation_mode="recurrent"
    )
    result = model.supervised_action(model.core.initial_state(1), plan=plan)
    assert plan.scored_positions == 4
    assert result.losses["lexical"][1] == result.losses["reconstruction"][1] == 2
    assert result.losses["facet_prediction"][1] == result.losses["encoder_alignment"][1] == 0
    assert result.losses["structure"][1] == 8
    assert result.losses["terminal"][1] == 2


def test_nonempty_control_bytes_roundtrip_and_train_only_the_served_control_path(model):
    from megatoken.controls import build_control_payload

    model.freeze_decoder_for_rl()
    state = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=3)
    payload = build_control_payload(
        seed=b"style\x00\xff",
        reference=b"earlier observed evidence",
        reference_origin="earlier_history",
    )
    result = model.sample(
        state, spec=spec, control_payload=payload, generator=torch.Generator().manual_seed(738)
    )
    restored = ActionTape.from_json(result.tape.to_json())
    assert restored.control_payload == payload
    replay = model.score_tape(state, restored, spec=spec)
    explicit = model.score_tape(state, restored, spec=spec, control_payload=payload)
    assert_replay_density_close(result.log_prob, replay.log_prob, model.latent_dimensions)
    torch.testing.assert_close(replay.log_prob, explicit.log_prob, rtol=0, atol=0)
    assert_state_equal(result.state, replay.state)
    (-replay.log_prob / model.latent_dimensions).backward()
    assert not has_gradient(model.core.control_encoder.parameters())
    assert has_gradient(model.core.control_injection.parameters())
    assert not has_gradient(model.core.encoder_parameters())
    assert not has_gradient(model.decoder.parameters())
    with pytest.raises(TapeMismatch, match="original causal bytes"):
        model.score_tape(state, restored, spec=spec, control_payload=b"future target bytes")
    with pytest.raises(ValueError, match="causal"):
        build_control_payload(reference=b"target", reference_origin="future_target")
    with pytest.raises(TypeError, match="controls"):
        model.replay(state, [restored], specs=[spec], controls=[payload])


def test_fixed_facets_have_identical_decoder_output_under_different_causal_controls(
    model, monkeypatch
):
    from megatoken.generation.latents import DiagonalGaussian

    state = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=3, max_tokens=3)
    raw = torch.linspace(-1, 1, model.latent_dimensions).reshape(
        1, model.config.n_facets, model.config.facet_dim
    )
    monkeypatch.setattr(DiagonalGaussian, "sample", lambda self, generator=None: raw)
    first = model.sample(state, spec=spec, control_payload=b"public condition A")
    second = model.sample(state, spec=spec, control_payload=b"different public condition B")
    assert first.tape.control_payload != second.tape.control_payload
    assert first.payload == second.payload
    assert first.emitted_ids == second.emitted_ids
    assert first.tape.trace == second.tape.trace
    assert torch.equal(first.state.facets, second.state.facets)
    assert not torch.allclose(first.log_prob, second.log_prob)
    assert model.decode_mean(state, spec=spec, control_payload=b"condition").valid
    with pytest.raises(TypeError, match="control"):
        model.sample(state, spec=spec, control=torch.zeros(1, model.config.control_dim))


@pytest.mark.parametrize(
    "bad",
    [None, "plain text", [], {}, {"$bytes": 1}, {"$bytes": "AA==", "extra": 1}, {"$bytes": "AB=="}],
)
def test_v4_tape_control_payload_requires_an_exact_serialized_byte_field(model, bad):
    result = model.sample(
        model.core.initial_state(1),
        spec=ActionSpec(max_tokens=0),
        control_payload=b"control\x00\xff",
        generator=torch.Generator().manual_seed(44),
    )
    encoded = json.loads(result.tape.to_json())
    encoded["control_payload"] = bad
    with pytest.raises(ValueError, match="control_payload"):
        ActionTape.from_json(json.dumps(encoded))
    encoded.pop("control_payload")
    with pytest.raises(ValueError, match="control_payload"):
        ActionTape.from_json(json.dumps(encoded))
    with pytest.raises(TypeError, match="immutable bytes"):
        replace(result.tape, control_payload="not bytes")


def test_schema_with_fixed_wire_envelope_larger_than_limit_is_not_selected(model):
    from megatoken.generation.feasibility import wire_witness_tokens

    schema = SchemaSpec(
        "long-tool",
        (
            FieldSpec(
                "name",
                kind="json",
                fixed_json='"ridiculously_long_tool_name"',
                max_tokens=128,
            ),
            FieldSpec("arguments", kind="json", fixed_json="{}", max_tokens=128),
        ),
    )
    witness = wire_witness_tokens(schema, model.codec)
    assert witness is not None and witness > 8
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=8)
    with pytest.raises(ValueError, match="no public output mode"):
        model.sample(model.core.initial_state(1), spec=spec)


def test_public_required_json_argument_has_constructible_wire_witness(model):
    from megatoken.generation.feasibility import wire_witness_tokens

    schema = SchemaSpec(
        "read",
        (
            FieldSpec("name", kind="json", fixed_json='"read"', max_tokens=128),
            FieldSpec(
                "arguments",
                kind="json",
                json_schema=(
                    '{"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}'
                ),
                max_tokens=128,
            ),
        ),
    )
    witness = wire_witness_tokens(schema, model.codec)
    assert witness == len(model.codec.encode(b'{"name":"read","arguments":{"path":""}}'))


def test_packed_fp32_tape_is_byte_exact_and_accepts_legacy_v4_float_lists(model):
    result = model.sample(
        model.core.initial_state(1),
        spec=ActionSpec(max_tokens=0),
        generator=torch.Generator().manual_seed(117),
    )
    tape = result.tape
    assert isinstance(tape.facets, bytes)
    assert len(tape.facets) == 4 * tape.latent_dimensions
    restored = ActionTape.from_json(tape.to_json())
    assert restored.facets == tape.facets
    assert restored == tape
    legacy = json.loads(tape.to_json())
    legacy["facets"] = tape.facet_array().tolist()
    decoded = ActionTape.from_json(json.dumps(legacy))
    assert isinstance(decoded.facets, tuple)
    assert decoded.facet_array().tobytes() == tape.facets
    torch.testing.assert_close(
        model.score_tape(
            model.core.initial_state(1),
            decoded,
            spec=ActionSpec(max_tokens=0),
        ).log_prob,
        result.log_prob,
        rtol=0,
        atol=1e-5,
    )


def test_owner_groups_partition_every_parameter_exactly_once(model):
    owners = model.parameter_owners()
    assert set(owners) == {"E", "C", "P", "R"}
    seen = {}
    for owner, parameters in owners.items():
        for parameter in parameters:
            assert id(parameter) not in seen, (owner, seen[id(parameter)])
            seen[id(parameter)] = owner
    assert seen.keys() == {id(p) for p in model.parameters()}
    owner_of = lambda p: seen[id(p)]  # noqa: E731
    transducer = model.transducer
    assert owner_of(transducer.bank.modulation) == "P"
    assert owner_of(next(transducer.query_realizer.parameters())) == "P"
    assert owner_of(transducer.readout.row_identifiers) == "E"
    assert (
        owner_of(
            model.embedding_realizer.query.weight
            if hasattr(model.embedding_realizer.query, "weight")
            else next(model.embedding_realizer.query.parameters())
        )
        == "R"
    )


def test_rl_freeze_leaves_p_side_retrieval_trainable(model):
    model.freeze_decoder_for_rl()
    # Latent RL adapts how P queries the transducer; the knowledge readout stays pinned.
    assert model.transducer.query_realizer.query.weight.requires_grad
    assert model.transducer.bank.modulation.requires_grad
    assert not model.transducer.readout.row_identifiers.requires_grad
