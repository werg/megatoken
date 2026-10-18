from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from megatoken.model.config import ModelConfig
from megatoken.model.frontend import ByteFrontend
from megatoken.model.ingestion import ChunkEncoder


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(731)


def test_pyramid_halves_positions_and_bounds_every_attention_source():
    encoder = ChunkEncoder(ModelConfig.tiny()).eval()
    reductions, attention_sizes = [], []

    def record_reduction(module, inputs, output):
        reductions.append((inputs[0].shape[1], output[0].shape[1]))

    def record_attention(module, inputs, output):
        attention_sizes.append((inputs[0].shape[1], inputs[1].shape[1]))

    handles = [encoder.reduction.register_forward_hook(record_reduction)]
    handles += [
        module.register_forward_hook(record_attention)
        for module in (encoder.reduction.attention, encoder.facet_attention)
    ]
    output = encoder(torch.arange(1031).remainder(256)[None])
    for handle in handles:
        handle.remove()
    assert output.shape == (1, 4, 32)
    assert reductions == [(129, 65), (65, 33), (33, 17), (17, 9), (9, 5), (5, 3)]
    assert set(attention_sizes) == {(8, 8), (4, 4)}
    assert sum(after for _, after in reductions) < 2 * reductions[0][0]


@pytest.mark.parametrize("mode", ["fp32", "bf16", "autocast"])
def test_padding_and_unequal_batch_membership_do_not_change_facets(mode):
    encoder = ChunkEncoder(ModelConfig.tiny()).eval()
    if mode == "bf16":
        encoder.bfloat16()
    context = torch.autocast("cpu", dtype=torch.bfloat16) if mode == "autocast" else nullcontext()
    lengths = torch.tensor([0, 1, 31, 32, 33, 64, 65, 257, 1025])
    ids = torch.randint(256, (len(lengths), 1071))
    with context:
        together = encoder(ids, lengths)
        for row, length in enumerate(lengths.tolist()):
            alone = encoder(ids[row : row + 1, :length])
            torch.testing.assert_close(
                alone,
                together[row : row + 1],
                rtol=1e-5 if mode == "fp32" else 2e-2,
                atol=1e-6 if mode == "fp32" else 2e-3,
            )
    assert torch.isfinite(together).all()
    assert together.dtype == (torch.float32 if mode == "fp32" else torch.bfloat16)


def test_native_binary_unicode_order_case_and_final_byte_all_reach_the_encoder():
    encoder = ChunkEncoder(ModelConfig.tiny()).eval()
    native = bytes(range(256)) + "日本語 e\u0301 🙂\r\n".encode()
    ids = torch.tensor(list(native))[None]
    base = encoder(ids)
    reordered = ids.clone()
    reordered[:, [3, 4]] = reordered[:, [4, 3]]
    altered_zero = ids.clone()
    altered_zero[0, 0] = 1
    altered_case = ids.clone()
    altered_case[0, 65] = 97
    altered_tail = ids.clone()
    altered_tail[0, -1] = 33
    for alternative in (reordered, altered_zero, altered_case, altered_tail):
        assert not torch.allclose(base, encoder(alternative), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_every_real_byte_has_gradient_and_padding_has_none(dtype):
    encoder = ChunkEncoder(ModelConfig.tiny()).to(dtype=dtype)
    ids = torch.randint(256, (2, 521))
    lengths = torch.tensor([3, 513])
    activations = []

    def retain_byte_activations(module, inputs, output):
        output.retain_grad()
        activations.append(output)

    handle = encoder.bytes.register_forward_hook(retain_byte_activations)
    facets = encoder(ids, lengths)
    (facets.float() * torch.randn_like(facets.float())).sum().backward()
    handle.remove()
    gradients = activations[0].grad.float().abs().sum(-1)
    mask = torch.arange(ids.shape[1])[None] < lengths[:, None]
    assert (gradients[mask] > 0).all()
    assert gradients[~mask].count_nonzero() == 0
    for parameter in (
        encoder.ngram_mixture.weight,
        encoder.ngrams.embedding.weight,
        encoder.reduction.ordered_merge.weight,
        encoder.reduction.attention.query.weight,
        encoder.reduction.ffn.gate_up.weight,
        encoder.output_projection.weight,
    ):
        assert parameter.grad is not None and parameter.grad.float().norm() > 0
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )


def test_empty_record_has_no_fake_byte_contribution_and_output_scale_is_unconstrained():
    encoder = ChunkEncoder(ModelConfig.tiny())
    empty = encoder(torch.empty((1, 0), dtype=torch.long))
    padded = encoder(torch.tensor([[0, 255, 64], [255, 0, 11]]), torch.tensor([0, 0]))
    torch.testing.assert_close(padded, empty.expand_as(padded))
    padded.square().sum().backward()
    assert encoder.bytes.weight.grad.count_nonzero() == 0
    assert encoder.ngrams.embedding.weight.grad.count_nonzero() == 0
    before = encoder(torch.tensor([[65, 66, 67]])).detach()
    with torch.no_grad():
        encoder.output_projection.weight.mul_(3)
    after = encoder(torch.tensor([[65, 66, 67]])).detach()
    torch.testing.assert_close(after, 3 * before)


def test_stride_features_are_causal_and_ngram_import_has_one_owner():
    encoder = ChunkEncoder(ModelConfig.tiny()).bfloat16().eval()
    prefix = torch.tensor([[65, 0, 255, 17, 93, 65, 0, 255]])
    left = torch.cat((prefix, torch.zeros_like(prefix)), -1)
    right = torch.cat((prefix, torch.full_like(prefix, 255)), -1)
    torch.testing.assert_close(
        encoder.source_features(left).features[:, :1],
        encoder.source_features(right).features[:, :1],
        rtol=0,
        atol=0,
    )
    encoder.ngrams.install_import(
        torch.tensor([0, 65, 255]), torch.randn(3, 128), torch.ones(3), torch.ones(3)
    )
    before = encoder(left)
    report = encoder.ngrams.bake()
    assert report.max_lookup_error == 0
    torch.testing.assert_close(encoder(left), before, rtol=0, atol=0)
    names = [name for name, _ in encoder.named_parameters() if "ngrams.embedding.weight" in name]
    assert names == ["ngrams.embedding.weight"]


def test_production_source_and_facet_geometry_and_invalid_native_inputs():
    geometry = replace(ModelConfig(), ngram_rows=2048, exact_bigrams=False)
    encoder = ChunkEncoder(geometry)
    assert encoder(torch.randint(256, (2, 2053))).shape == (2, 32, 512)
    assert encoder.source_features(torch.zeros((1, 9), dtype=torch.long)).features.shape == (
        1,
        2,
        256,
    )
    frontend = ByteFrontend(ModelConfig.tiny())
    with pytest.raises(TypeError, match="integer"):
        frontend(torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="\\[0,255\\]"):
        frontend(torch.tensor([[256]]))
    with pytest.raises(TypeError, match="integer byte counts"):
        frontend(torch.zeros((1, 1), dtype=torch.long), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="record length"):
        frontend(torch.zeros((1, 1), dtype=torch.long), torch.tensor([2]))
    with pytest.raises(ValueError, match="at least one record"):
        frontend(torch.empty((0, 1), dtype=torch.long))
