"""Causal control framing, batch neutrality, and encoder ownership contracts."""

import json

import pytest
import torch

from megatoken.controls import build_control_payload, encode_controls
from megatoken.model import FacetedCore, ModelConfig


@pytest.fixture
def core():
    torch.set_num_threads(1)
    torch.manual_seed(608)
    return FacetedCore(ModelConfig.tiny())


def test_control_framing_preserves_seed_reference_bytes_and_canonicalizes_only_tools():
    seed, reference = b"policy \xff\n  literal", b"earlier \x00 evidence\n"
    tools = [{"parameters": {"type": "object"}, "name": "caf\u00e9", "description": "read"}]
    expected_tools = (
        '[{"description":"read","name":"caf\u00e9","parameters":{"type":"object"}}]'
    ).encode()
    expected = seed + b"\n<reference>\n" + reference + b"\n<tools>\n" + expected_tools
    kwargs = {"seed": seed, "reference": reference, "reference_origin": "earlier_history"}
    assert build_control_payload(**kwargs, tools_json=json.dumps(tools, indent=4)) == expected
    assert build_control_payload(**kwargs, tools_json=expected_tools.decode()) == expected
    assert build_control_payload(seed=seed, tools_json=" [ ] ") == seed
    assert build_control_payload() == b""


@pytest.mark.parametrize(
    "origin", [None, "target", "current_target", "future_history", "verifier_outcome"]
)
def test_current_or_future_reference_origins_are_rejected(origin):
    with pytest.raises(ValueError, match="causal"):
        build_control_payload(reference=b"answer", reference_origin=origin)


@pytest.mark.parametrize("origin", ["external_evidence", "earlier_history"])
def test_only_explicit_causal_references_are_accepted(origin):
    assert build_control_payload(reference=b"known", reference_origin=origin) == (
        b"\n<reference>\nknown"
    )


@pytest.mark.parametrize(
    "tools", ["{}", "null", "[NaN]", "[Infinity]", "[1e999]", '[{"name":1,"name":2}]']
)
def test_control_tools_reject_ambiguous_nonfinite_or_nonlist_json(tools):
    with pytest.raises(ValueError):
        build_control_payload(tools_json=tools)


def test_control_payload_and_batch_types_are_explicit():
    with pytest.raises(TypeError, match="bytes"):
        build_control_payload(seed="not raw bytes")
    with pytest.raises(TypeError, match="JSON string"):
        build_control_payload(tools_json=[])
    with pytest.raises(ValueError, match="empty control batch"):
        encode_controls(None, [])
    with pytest.raises(TypeError, match="bytes"):
        encode_controls(None, ["not raw bytes"])
    # No module or ML work is needed for a completely absent control.
    assert encode_controls(None, [b"", b""]) is None


def test_mixed_empty_control_is_neutral_and_nonempty_matches_single_example(core, monkeypatch):
    original = core.encode_control

    def with_bias(*args, **kwargs):
        # Neutrality must survive a future biased encoder, not depend on the
        # accidental zero response of today's bias-free source projections.
        return original(*args, **kwargs) + 0.75

    monkeypatch.setattr(core, "encode_control", with_bias)
    with torch.no_grad():
        mixed = encode_controls(core, [b"", b"causal tool policy"])
        nonempty = encode_controls(core, [b"causal tool policy"])
        assert torch.equal(mixed[0], torch.zeros_like(mixed[0]))
        torch.testing.assert_close(mixed[1:], nonempty, atol=1e-6, rtol=1e-6)
        alone = core.predict_facets(core.initial_state(1), effort=1)
        together = core.predict_facets(core.initial_state(2), control=mixed, effort=1)[:1]
    torch.testing.assert_close(together, alone, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("encoder_grad", [False, True])
def test_recurrent_control_keeps_control_encoder_trainable_and_isolates_frontend(
    core, encoder_grad
):
    controls = encode_controls(core, [b"", b"known instructions"], encoder_grad=encoder_grad)
    controls.square().sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in core.control_encoder.parameters()
    )
    encoder_gradients = [parameter.grad for parameter in core.encoder_parameters()]
    if encoder_grad:
        assert any(gradient is not None and gradient.norm() > 0 for gradient in encoder_gradients)
    else:
        assert all(gradient is None for gradient in encoder_gradients)


def test_an_absent_control_row_cannot_train_the_encoder(core):
    controls = encode_controls(core, [b"", b"another episode has a control"])
    controls[0].sum().backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in core.parameters()
    )
