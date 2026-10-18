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
from megatoken.generation.structure import ActionTape, TapeMismatch
from megatoken.model.config import ModelConfig
from megatoken.model.core import FacetedCore


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(617)
    return FacetedGenerator(
        FacetedCore(ModelConfig.tiny(max_capacity=32)), ByteCodec.byte_fallback()
    )


def test_batched_supervision_counts_only_free_positions_and_commits_once(model):
    state = model.core.initial_state(2)
    plan = model.prepare_target(
        [tuple(b"a\xff"), tuple(b"xyz")],
        free_mask=[(True, False), (True, True, False)],
        capacity=8,
        repair=True,
        terminal=[False, True],
    )
    result = model.supervised_step(
        state,
        plan=plan,
        role_ids=torch.tensor([2, 4]),
        prior_probability=1,
        generator=torch.Generator().manual_seed(42),
    )
    assert result.losses["lexical"][1] == plan.scored_positions == 6
    assert all(result.losses[name][1] == count for name, count in plan.denominators.items())
    assert torch.equal(result.state.records_seen, state.records_seen + 1)
    assert torch.equal(state.records_seen, torch.zeros_like(state.records_seen))
    total = sum(value / max(count, 1) for value, count in result.losses.values())
    total.backward()
    assert model.root_prior.statistics.weight.grad.abs().sum() > 0
    assert model.vocabulary.embedding.weight.grad.abs().sum() > 0
    assert (
        model.core.u_always.up.weight.grad is not None
        if hasattr(model.core.u_always, "up")
        else True
    )


def test_latent_loss_stop_gradient_directions_are_isolated(model):
    state = model.core.initial_state(1)
    plan = model.prepare_target(target_bytes=b"learn", capacity=8)
    result = model.supervised_step(state, plan=plan, prior_probability=0)
    result.losses["prior_fit"][0].backward(retain_graph=True)
    assert all(
        parameter.grad is None or not parameter.grad.any()
        for parameter in model.posterior.parameters()
    )
    assert model.root_prior.statistics.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    result.losses["posterior_pressure"][0].backward()
    assert all(
        parameter.grad is None or not parameter.grad.any()
        for parameter in model.root_prior.parameters()
    )
    assert all(
        parameter.grad is None or not parameter.grad.any()
        for parameter in model.regional_prior.parameters()
    )
    assert model.posterior.root.statistics.weight.grad.abs().sum() > 0


def test_prior_growth_does_not_see_target_or_exact_length(model, monkeypatch):
    captured = []
    original = model.realizer.finish_growth

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value.detach().clone())
        return value

    monkeypatch.setattr(model.realizer, "finish_growth", capture)
    state = model.core.initial_state(1)
    for target in (b"a", b"different"):
        plan = model.prepare_target(target_bytes=target, capacity=16)
        model.supervised_step(
            state, plan=plan, prior_probability=1, generator=torch.Generator().manual_seed(122)
        )
    assert captured[0].shape[1] == captured[1].shape[1] == 16
    assert torch.equal(captured[0], captured[1])


def test_served_sampling_never_calls_target_posterior(model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("target posterior reached served branch")

    monkeypatch.setattr(model.posterior, "encode", forbidden)
    result = model.sample(
        model.core.initial_state(1),
        spec=ActionSpec(text_kind="ascii", min_tokens=1, max_tokens=4, max_repairs=0),
    )
    assert result.tape.committed


def test_full_repair_tape_replay_and_serialization_match(model):
    state = model.core.initial_state(1)
    spec = ActionSpec(
        text_kind="ascii", min_tokens=4, max_tokens=8, max_repairs=2, repair_probability=1
    )
    sampled = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(13), rl=True)
    assert len(sampled.tape.attempts) == 3
    assert sampled.scored_positions == 3 * len(sampled.tape.attempts[0].draws[0].ids)
    restored = ActionTape.from_json(sampled.tape.to_json())
    assert restored == sampled.tape
    scored = model.score_tape(state, restored, spec=spec)
    assert torch.equal(scored.log_prob, sampled.log_prob)
    assert torch.equal(scored.state.carry, sampled.state.carry)
    assert torch.equal(scored.state.facets, sampled.state.facets)
    assert model.codec.decode(sampled.emitted_ids) == sampled.payload
    scored.log_prob.backward()
    assert model.root_prior.statistics.weight.grad.abs().sum() > 0


def test_tampered_and_incomplete_tapes_are_rejected(model):
    state = model.core.initial_state(1)
    spec = ActionSpec(
        text_kind="ascii", min_tokens=1, max_tokens=4, max_repairs=1, repair_probability=1
    )
    result = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(12))
    with pytest.raises(TapeMismatch):
        model.score_tape(state, replace(result.tape, choices=result.tape.choices[:-1]), spec=spec)
    with pytest.raises(TapeMismatch):
        model.score_tape(state, replace(result.tape, attempts=result.tape.attempts[:1]), spec=spec)
    with pytest.raises(TapeMismatch):
        model.score_tape(state, replace(result.tape, payload=b"forged"), spec=spec)
    with pytest.raises(TapeMismatch):
        model.score_tape(state, result.tape, spec=replace(spec, min_tokens=0))
    with pytest.raises(ValueError, match="temperature 1"):
        model.sample(state, temperature=0.7)
    with pytest.raises(ValueError, match="effort 6"):
        model.sample(state, rl=True, effort=3)


def test_typed_identifiers_copy_placements_and_repair_are_scored_once(model):
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
        max_tokens=32,
        max_repairs=0,
        require_terminal=False,
    )
    plan = model.prepare_action_target(
        b'{"path":"x","same_path":"x","content":"!copied?"}',
        spec=spec,
        bindings={"content": 1},
        capacity=16,
        repair=True,
    )
    assert plan.scored_positions == 1 + 2 + 2  # identifier once, two free positions per pass
    state = model.core.initial_state(1)
    result = model.supervised_action(state, plan=plan, prior_probability=1)
    assert result.losses["binding"][1] == 2
    assert result.losses["placement"][1] == 1
    assert result.losses["lexical"][1] == 5
    assert result.state.records_seen.item() == 1
    sum(loss / max(count, 1) for loss, count in result.losses.values()).backward()
    assert model.binding_query.weight.grad.abs().sum() > 0
    assert model.identifier_slots.weight.grad.abs().sum() > 0


def test_typed_teacher_path_is_the_same_candidate_likelihood(model):
    schema = SchemaSpec(
        "pair",
        (
            FieldSpec("id", kind="identifier", min_tokens=1, max_tokens=4, identifier="id"),
            FieldSpec("ref", kind="identifier", min_tokens=1, max_tokens=4, bound_to="id"),
            FieldSpec("value", kind="utf8", max_tokens=12),
        ),
        layouts=(("id", "ref", "value"), ("value", "id", "ref")),
    )
    spec = ActionSpec(
        schemas=(schema,),
        copy_sources=(CopySource("earlier", b'q"\n'),),
        max_tokens=32,
        max_repairs=0,
    )
    plan = model.prepare_action_target(
        json.dumps({"id": "a", "ref": "a", "value": 'q"\n!'}).encode(),
        spec=spec,
        bindings={"value": 1},
        capacity=16,
        layout=1,
    )
    state = model.core.initial_state(1)
    # Forced public actions remain valid samples of the augmented support. Their
    # density must be scored by the exact same served program.
    teacher = model._execute(state, spec=spec, teacher=plan, prior_probability=0)
    replay = model.score_tape(state, teacher.tape, spec=spec)
    assert torch.equal(teacher.log_prob, replay.log_prob)
    assert torch.equal(teacher.state.carry, replay.state.carry)
    assert model.codec.decode(replay.emitted_ids) == plan.payload


def test_nested_json_values_and_fixed_syntax_preserve_tool_types(model):
    schema = SchemaSpec(
        "call",
        (
            FieldSpec("name", kind="json", fixed_json='"edit"'),
            FieldSpec(
                "arguments",
                kind="json",
                min_tokens=1,
                max_tokens=24,
                json_schema='{"type":"object","properties":{"n":{"type":"integer"},"ok":{"type":"boolean"}},"required":["n","ok"]}',
            ),
        ),
    )
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=32, max_repairs=0)
    plan = model.prepare_action_target(
        b'{"name":"edit","arguments":{"n":2,"ok":true}}', spec=spec, capacity=32
    )
    result = model.supervised_action(model.core.initial_state(1), plan=plan, prior_probability=1)
    assert result.losses["lexical"][1] == len(b'{"n":2,"ok":true}')
    assert json.loads(plan.payload)["arguments"]["ok"] is True
    with pytest.raises(ValidationError):
        model.prepare_action_target(
            b'{"name":"edit","arguments":{"n":"bad","ok":true}}', spec=spec, capacity=32
        )


def test_failed_attempts_are_kept_without_factual_commit(model):
    schema = SchemaSpec(
        "number",
        (
            FieldSpec(
                "value", kind="json", min_tokens=1, max_tokens=1, json_schema='{"type":"integer"}'
            ),
        ),
    )
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=1, max_repairs=1)
    state = model.core.initial_state(1)
    for seed in range(10):
        result = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(seed))
        if not result.tape.committed:
            break
    assert not result.tape.committed
    assert len(result.tape.attempts) == 2 and result.scored_positions == 2
    assert result.state.records_seen.item() == state.records_seen.item()
    replay = model.score_tape(state, result.tape, spec=spec)
    assert torch.equal(result.log_prob, replay.log_prob)
    assert torch.equal(result.state.carry, replay.state.carry)


def test_replay_reconstructs_state_and_recomputes_candidate_routes(model):
    initial = model.core.initial_state(1)
    spec = ActionSpec(text_kind="ascii", min_tokens=1, max_tokens=4, max_repairs=0)
    first = model.sample(initial, spec=spec, generator=torch.Generator().manual_seed(1))
    second = model.sample(first.state, spec=spec, generator=torch.Generator().manual_seed(2))
    state, probabilities = model.replay(initial, [first.tape, second.tape], specs=[spec, spec])
    assert torch.equal(state.carry, second.state.carry)
    assert torch.equal(probabilities[1], second.log_prob)
    isolated = model.score_tape(initial, second.tape, spec=spec)
    assert not torch.allclose(isolated.log_prob, second.log_prob)
    with torch.no_grad():
        model.core.conditioning_injection.weight.add_(0.05)
    candidate = model.score_tape(initial, first.tape, spec=spec)
    assert not torch.allclose(candidate.log_prob, first.log_prob)


def test_empty_terminal_action_is_a_real_committed_boundary(model):
    state = model.core.initial_state(1)
    spec = ActionSpec(min_tokens=0, max_tokens=0, max_repairs=0, require_terminal=True)
    result = model.sample(state, spec=spec)
    assert result.payload == b"" and result.scored_positions == 0 and result.tape.terminal
    assert result.state.records_seen.item() == 1


def test_tight_capacity_reserves_repeated_identifier_once_per_occurrence(model):
    schema = SchemaSpec(
        "pair",
        (
            FieldSpec("a", kind="identifier", min_tokens=1, max_tokens=8, identifier="id"),
            FieldSpec("b", kind="identifier", min_tokens=1, max_tokens=8, bound_to="id"),
        ),
    )
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=2, max_repairs=0)
    result = model.sample(
        model.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(2)
    )
    parsed = json.loads(result.payload)
    assert len(parsed["a"]) == 1 and parsed["a"] == parsed["b"]
    assert result.scored_positions == 1


def test_infeasible_schemas_are_masked_before_the_capacity_choice(model):
    schemas = (
        SchemaSpec("too-large", (FieldSpec("value", kind="json", fixed_json='"far too large"'),)),
        SchemaSpec("fits", (FieldSpec("value", kind="json", fixed_json="0"),)),
    )
    spec = ActionSpec(allow_text=False, schemas=schemas, max_tokens=1, max_repairs=0)
    state = model.core.initial_state(1)
    for seed in range(3):
        sampled = model.sample(state, spec=spec, generator=torch.Generator().manual_seed(seed))
        assert sampled.payload == b'{"value":0}' and sampled.scored_positions == 0
        assert next(choice.value for choice in sampled.tape.choices if choice.name == "schema") == 1
        replay = model.score_tape(state, sampled.tape, spec=spec)
        torch.testing.assert_close(sampled.log_prob, replay.log_prob, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no output mode or schema fits"):
        model.sample(state, spec=replace(spec, schemas=schemas[:1]))


def test_multiple_shared_identifiers_reserve_the_intersected_minima(model):
    schema = SchemaSpec(
        "two-ids",
        (
            FieldSpec("a", kind="identifier", min_tokens=0, max_tokens=3, identifier="a"),
            FieldSpec("a_ref", kind="identifier", min_tokens=2, max_tokens=3, bound_to="a"),
            FieldSpec("b", kind="identifier", min_tokens=0, max_tokens=3, identifier="b"),
            FieldSpec("b_ref", kind="identifier", min_tokens=2, max_tokens=3, bound_to="b"),
        ),
    )
    spec = ActionSpec(allow_text=False, schemas=(schema,), max_tokens=8, max_repairs=0)
    output = model.sample(
        model.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(19)
    )
    parsed = json.loads(output.payload)
    assert len(parsed["a"]) == len(parsed["b"]) == 2
    assert parsed["a"] == parsed["a_ref"] and parsed["b"] == parsed["b_ref"]
    assert output.scored_positions == 4


def test_fixed_json_must_satisfy_its_declared_schema():
    with pytest.raises(ValidationError):
        FieldSpec("fixed", kind="json", fixed_json="true", json_schema='{"type":"integer"}')


def test_prior_and_posterior_metrics_remain_detached_tensors(model):
    plan = model.prepare_target(
        target_bytes=[b"abc", b"de"],
        capacity=4,
        free_mask=[(True, False, True), (True, True)],
        repair=True,
    )
    output = model.supervised_step(model.core.initial_state(2), plan=plan, prior_probability=0.5)
    metrics = output.metrics
    assert metrics["prior_examples"] + metrics["posterior_examples"] == 2
    assert (
        metrics["prior_lexical_positions"] + metrics["posterior_lexical_positions"]
        == plan.scored_positions
    )
    torch.testing.assert_close(
        metrics["prior_lexical_loss_sum"] + metrics["posterior_lexical_loss_sum"],
        output.losses["lexical"][0].detach(),
    )
    assert all(
        not value.requires_grad for value in metrics.values() if isinstance(value, torch.Tensor)
    )


def test_served_export_has_no_target_posterior_and_is_byte_identical(model):
    served = FacetedGenerator(FacetedCore(model.config), model.codec, training_heads=False)
    weights = model.served_state_dict()
    assert not any(name.startswith("posterior.") for name in weights)
    served.load_state_dict(weights, strict=True)
    assert served.posterior is None
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4, max_repairs=0)
    original = model.sample(
        model.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(15)
    )
    exported = served.sample(
        served.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(15)
    )
    assert original.tape == exported.tape
    with pytest.raises(RuntimeError, match="served-only"):
        served.supervised_step(served.core.initial_state(1), target_bytes=b"x")


def test_encoded_openai_arguments_validate_the_inner_schema(model):
    schema = SchemaSpec(
        "function",
        (
            FieldSpec(
                "function",
                kind="json",
                min_tokens=1,
                max_tokens=32,
                json_schema=json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "arguments": {
                                "type": "string",
                                "contentMediaType": "application/json",
                                "contentSchema": {"type": "integer"},
                            }
                        },
                        "required": ["arguments"],
                    }
                ),
            ),
        ),
    )
    spec = ActionSpec(schemas=(schema,), max_tokens=32)
    good = model.prepare_action_target(b'{"function":{"arguments":"2"}}', spec=spec)
    assert good.payload
    with pytest.raises(ValidationError):
        model.prepare_action_target(b'{"function":{"arguments":"false"}}', spec=spec)
    with pytest.raises(ValidationError):
        model.prepare_action_target(b'{"function":{"arguments":"invalid json"}}', spec=spec)


def test_small_served_prior_can_learn_a_byte_task():
    torch.set_num_threads(2)
    torch.manual_seed(947)
    model = FacetedGenerator(
        FacetedCore(ModelConfig.tiny(max_capacity=4)), ByteCodec.byte_fallback()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=0)
    plan = model.prepare_target(target_bytes=b"AAAA", capacity=4)
    losses = []
    for step in range(18):
        state = model.core.initial_state(1)
        result = model.supervised_step(
            state, plan=plan, prior_probability=1, generator=torch.Generator().manual_seed(step + 1)
        )
        loss = result.losses["lexical"][0] / 4
        losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    assert sum(losses[-3:]) / 3 < losses[0] * 0.25
    spec = ActionSpec(text_kind="ascii", min_tokens=4, max_tokens=4, max_repairs=0)
    output = model.sample(
        model.core.initial_state(1), spec=spec, generator=torch.Generator().manual_seed(600)
    )
    assert output.payload == b"AAAA"
