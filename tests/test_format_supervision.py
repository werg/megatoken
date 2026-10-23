"""Learn the public choice between ordinary text and a tool, without label masks."""

from dataclasses import asdict, replace

import pytest
import torch

from megatoken.generation import ActionSpec, ByteCodec, FacetedGenerator, FieldSpec, SchemaSpec
from megatoken.model import FacetedCore, ModelConfig
from megatoken.training.config import TrainConfig
from megatoken.training.engine import _resume_compatible


@pytest.fixture
def model():
    torch.manual_seed(913)
    torch.set_num_threads(1)
    return FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback())


def public_spec():
    return ActionSpec(
        schemas=(SchemaSpec("answer", (FieldSpec("value", min_tokens=1, max_tokens=4),)),),
        max_tokens=64,
    )


def test_text_format_loss_is_the_actual_serving_choice_and_updates_its_logits(model):
    spec = public_spec()
    raw = torch.randn(1, model.config.n_facets, model.config.facet_dim)
    plain = model.prepare_target(target_bytes=b"hi")
    labeled = model.prepare_target(target_bytes=b"hi", mode_specs=[spec])
    old = model.decoder.teacher_raw(model.core, raw, plain)["structure"]
    new = model.decoder.teacher_raw(model.core, raw, labeled)["structure"]
    header = model.decoder.header(model.core.prepare_realization(raw))
    expected = torch.nn.functional.cross_entropy(
        model.decoder.logits(header, model.decoder.mode_codes).float(), torch.tensor([0])
    )
    torch.testing.assert_close(new[0] - old[0], expected)
    assert new[1] == old[1] + 1
    assert labeled.denominators["structure"] == plain.denominators["structure"] + 2
    gradient = torch.autograd.grad(new[0] - old[0], model.decoder.mode_codes.weight)[0]
    assert torch.linalg.vector_norm(gradient) > 0
    result = model.supervised_step(model.core.initial_state(1), plan=labeled)
    assert all(result.losses[name][1] == count for name, count in labeled.denominators.items())


def test_tool_format_competes_with_text_instead_of_a_target_derived_singleton(model):
    spec = public_spec()
    plan = model.prepare_action_target(b'{"value":"a"}', spec=spec)
    raw = torch.randn(1, model.config.n_facets, model.config.facet_dim)
    losses = model.decoder(model.core, raw, spec, teacher=plan).losses
    forced = replace(plan, spec=replace(spec, allow_text=False))
    old = model.decoder(model.core, raw, forced.spec, teacher=forced).losses
    header = model.decoder.header(model.core.prepare_realization(raw))
    expected = torch.nn.functional.cross_entropy(
        model.decoder.logits(header, model.decoder.mode_codes).float(), torch.tensor([1])
    )
    torch.testing.assert_close(losses["structure"][0] - old["structure"][0], expected)
    assert losses["structure"][1] == old["structure"][1]
    gradient = torch.autograd.grad(
        losses["structure"][0] - old["structure"][0], model.decoder.mode_codes.weight
    )[0]
    assert torch.linalg.vector_norm(gradient) > 0


def test_impossible_tool_wire_is_not_a_competing_format(model):
    spec = replace(public_spec(), max_tokens=1)
    plan = model.prepare_target(target_bytes=b"x", mode_specs=[spec])
    raw = torch.randn(1, model.config.n_facets, model.config.facet_dim)
    labeled = model.decoder.teacher_raw(model.core, raw, plan)["structure"]
    old = model.decoder.teacher_raw(model.core, raw, replace(plan, mode_specs=()))["structure"]
    torch.testing.assert_close(labeled[0], old[0])
    with pytest.raises(ValueError, match="text-permitting"):
        model.prepare_target(
            target_bytes=b"x", mode_specs=[replace(public_spec(), allow_text=False)]
        )


def test_old_checkpoint_cannot_silently_change_training_objective():
    config = TrainConfig(purpose="diagnostic", max_steps=1)
    old = asdict(config)
    old.pop("format_supervision_version")
    with pytest.raises(ValueError, match="format_supervision_version"):
        _resume_compatible(old, config, False)
    _resume_compatible(old, replace(config, format_supervision_version=1), False)
    _resume_compatible(old, config, True)


def test_imported_envelopes_preserve_serving_tool_indices_for_both_target_kinds(model):
    import json
    from types import SimpleNamespace

    from megatoken.tool_schema import compile_tool_spec
    from megatoken.training.meridians import _target_plan

    tools = json.dumps(
        [
            {
                "name": name,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            }
            for name in ("first", "second")
        ]
    )
    item = SimpleNamespace(episode=SimpleNamespace(tools_json=tools, records=[]))
    native = SimpleNamespace(paired_tape=lambda _: None)
    config = SimpleNamespace(
        generated_capacity=128, representation_training="bootstrap", format_supervision_version=2
    )
    expected = compile_tool_spec(tools, 128).schemas
    for payload, action_id in [
        (b'{"name":"second","arguments":{}}', "action"),
        (b"plain answer", ""),
    ]:
        record = SimpleNamespace(
            payload=payload, part_index=0, part_count=1, action_id=action_id, role="assistant"
        )
        entry = SimpleNamespace(
            record=record, token_ids=tuple(payload), free_mask=(True,) * len(payload)
        )
        plan = _target_plan(model, item, entry, 0, 0, native, config)
        spec = plan.spec if action_id else plan.mode_specs[0]
        assert spec.schemas[: len(expected)] == expected
        assert spec.allow_text
        if action_id:
            assert plan.schema_index == 1


def test_text_format_labels_cannot_override_public_length_bounds(model):
    with pytest.raises(ValueError, match="public output support"):
        model.prepare_target(
            target_bytes=b"too long", mode_specs=[replace(public_spec(), max_tokens=1)]
        )
