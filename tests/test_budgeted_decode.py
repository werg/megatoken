"""Regression gates for tool-wire capacity and batched lexical realization."""

import pytest
import torch

from megatoken.generation import ActionSpec, ByteCodec, FacetedGenerator, FieldSpec, SchemaSpec
from megatoken.generation.feasibility import wire_overhead_tokens, wire_witness_tokens
from megatoken.model import FacetedCore, ModelConfig


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(473)
    return FacetedGenerator(FacetedCore(ModelConfig.tiny()), ByteCodec.byte_fallback())


def _schema():
    return SchemaSpec(
        "run",
        (
            FieldSpec("name", kind="json", fixed_json='"run"'),
            FieldSpec("args", kind="ascii", min_tokens=1, max_tokens=64),
        ),
    )


def test_schema_capacity_reserves_json_wire_not_only_field_tokens(model, monkeypatch):
    schema = _schema()
    codec = model.codec
    overhead = wire_overhead_tokens(schema, codec)
    fixed = len(codec.encode(b'"run"'))
    assert overhead + fixed + 1 == wire_witness_tokens(schema, codec)
    assert overhead > 0

    # A minimum-size witness exists, but only three free bytes fit. Force the
    # length head to choose the longest publicly admissible free span.
    budget = overhead + fixed + 3
    assert budget <= model.config.max_capacity

    from megatoken.generation.decoder import _Program

    original = _Program.choose

    def choose_largest_length(self, name, logits, allowed, **kwargs):
        if name == "length/args" and self.teacher is None:
            allowed = tuple(allowed)
            return original(self, name, logits, (allowed[-1],), **kwargs)
        return original(self, name, logits, allowed, **kwargs)

    monkeypatch.setattr(_Program, "choose", choose_largest_length)
    monkeypatch.setattr(
        model.decoder.vocabulary,
        "argmax",
        lambda states, allowed: torch.full(
            (states.shape[0],), ord("a"), device=states.device, dtype=torch.long
        ),
    )
    output = model.decode_mean(
        model.core.initial_state(1),
        spec=ActionSpec(allow_text=False, schemas=(schema,), max_tokens=budget),
        effort=1,
    )
    assert output.valid, output.error
    assert output.payload == b'{"name":"run","args":"aaa"}'
    assert len(output.emitted_ids) == budget
    assert dict(output.field_ids)["args"] == (ord("a"),) * 3


def test_minimum_valid_wire_rejected_when_even_envelope_cannot_fit(model):
    schema = _schema()
    minimum = wire_witness_tokens(schema, model.codec)
    assert minimum is not None and minimum > 0
    with pytest.raises(ValueError, match="no public output mode fits"):
        model.decode_mean(
            model.core.initial_state(1),
            spec=ActionSpec(allow_text=False, schemas=(schema,), max_tokens=minimum - 1),
            effort=1,
        )


def test_final_free_spans_share_one_lexical_readback_per_alphabet(model, monkeypatch):
    schema = SchemaSpec(
        "two",
        (
            FieldSpec("first", kind="ascii", min_tokens=2, max_tokens=2),
            FieldSpec("second", kind="ascii", min_tokens=3, max_tokens=3),
        ),
    )
    seen = []

    def sampled(states, valid):
        seen.append(states.shape[0])
        return torch.full((states.shape[0],), ord("z"), dtype=torch.long, device=states.device)

    monkeypatch.setattr(model.decoder.vocabulary, "argmax", sampled)
    result = model.decode_mean(
        model.core.initial_state(1),
        spec=ActionSpec(allow_text=False, schemas=(schema,), max_tokens=128),
        effort=1,
    )
    assert result.valid, result.error
    assert result.payload == b'{"first":"zz","second":"zzz"}'
    assert seen == [5]
    assert dict(result.field_ids)["first"] == (ord("z"),) * 2
    assert dict(result.field_ids)["second"] == (ord("z"),) * 3
