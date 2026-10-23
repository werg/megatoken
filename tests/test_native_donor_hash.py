from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from megatoken.donors import DONOR_PROFILES, DonorAsset, DonorByteVocabulary
from megatoken.native_donor_hash import NativeDonorAddressor


def deepseek_addressor() -> NativeDonorAddressor:
    profile = DONOR_PROFILES["deepseek-v4.1-flash-1017"]
    vocabulary = DonorByteVocabulary((None,) * profile.vocab_size, "fixture")
    config = {
        "text_config": {
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384_006_168, 384_016_682],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16_000_000,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 99_092,
        }
    }
    index = {
        "layers.1.engram.embed.weight": "model-00047-of-00048.safetensors",
        "layers.1.engram.embed.scale": "model-00047-of-00048.safetensors",
        "layers.14.engram.embed.weight": "model-00048-of-00048.safetensors",
        "layers.14.engram.embed.scale": "model-00048-of-00048.safetensors",
    }
    # Official DeepSeek compressed IDs for tokenizer rows 2, 3, 4 and 5 are
    # 2, 3, 4 and 5. Other rows are unused by this fixed parity context.
    compressed_ids = tuple(range(profile.vocab_size))
    return NativeDonorAddressor(
        profile=DonorAsset(profile, Path("."), vocabulary),
        max_context_tokens=4,
        native_memory="engram",
        _index=index,
        _config=config,
        _deepseek_compressed_ids=compressed_ids,
        _deepseek_compressed_vocab_size=99_092,
    )


def qwen_addressor() -> NativeDonorAddressor:
    profile = DONOR_PROFILES["qwen3.8-flash-next"]
    vocabulary = DonorByteVocabulary((None,) * profile.vocab_size, "fixture")
    config = {
        "text_config": {
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "vocab_size": 248_320,
            "eos_token_id": 248_044,
            "ple_embed_dim": 2_560,
            "split_ngram_parts": 128,
        }
    }
    part_files = {
        part: f"model-{(5 if part < 2 else 6 + (part - 2) // 4):05d}-of-00131.safetensors"
        for part in range(128)
    }
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    index = {f"{prefix}.shard_{part}.weight": shard for part, shard in part_files.items()}
    return NativeDonorAddressor(
        profile=DonorAsset(profile, Path("."), vocabulary),
        max_context_tokens=3,
        native_memory="ple",
        _index=index,
        _config=config,
    )


def test_deepseek_addresses_match_pinned_official_reference_fixture() -> None:
    # Values were produced by the official inference/engram.py at the pinned
    # source revision in native_donor_hash.DEEPSEEK_REFERENCE.
    expected = [
        22_259,
        23_259_088,
        41_371_678,
        58_272_182,
        67_624_233,
        92_075_454,
        96_301_072,
        117_124_931,
        142_558_994,
        150_547_688,
        162_989_874,
        181_633_912,
        192_289_736,
        213_618_501,
        237_612_242,
        249_163_395,
        261_693_639,
        281_928_537,
        299_139_980,
        315_746_102,
        326_816_926,
        343_423_276,
        352_597_450,
        380_883_595,
    ]
    addressor = deepseek_addressor()

    refs = addressor.address([3, 4, 5], layer_id=1)

    assert [ref.global_row for ref in refs] == expected
    assert [(ref.order, ref.head) for ref in refs] == [
        (order, head) for order in (2, 3, 4) for head in range(8)
    ]
    assert all(ref.dtype == "F8_E4M3" and ref.scale_dtype == "F8_E8M0" for ref in refs)
    assert all(
        ref.scale_row == ref.row and ref.scale_width == 8 and ref.block_size == 32 for ref in refs
    )
    assert all(ref.component_index == i for i, ref in enumerate(refs))


def test_deepseek_returns_separate_layer_maps_and_rejects_unknown_layer() -> None:
    addressor = deepseek_addressor()

    refs = addressor.address([3, 4, 5])

    assert len(refs) == 48
    assert [ref.layer_id for ref in refs] == [1] * 24 + [14] * 24
    assert refs[0].shard_name == "model-00047-of-00048.safetensors"
    assert refs[-1].shard_name == "model-00048-of-00048.safetensors"
    expected_layer14 = [
        738_671,
        18_973_607,
        47_256_193,
        55_444_531,
        75_539_291,
        83_005_306,
        103_863_108,
        114_293_665,
        129_221_833,
        159_698_187,
        173_401_037,
        186_352_206,
        207_680_784,
        209_960_455,
        235_097_182,
        243_476_921,
        259_483_245,
        282_696_594,
        300_752_079,
        308_490_318,
        320_863_922,
        351_297_015,
        353_876_790,
        382_776_724,
    ]
    assert [ref.global_row for ref in refs[24:]] == expected_layer14
    with pytest.raises(ValueError, match="no table"):
        addressor.address([3, 4, 5], layer_id=2)


def test_qwen_addresses_match_pinned_official_row_fixture() -> None:
    refs = qwen_addressor().address([3, 4, 5])
    expected = [
        8_807_125,
        30_450_454,
        51_272_170,
        64_422_588,
        81_408_710,
        113_737_508,
        122_230_591,
        146_888_097,
        171_649_727,
        195_298_022,
        212_887_723,
        231_715_632,
        253_540_006,
        275_364_398,
        289_305_919,
        319_013_232,
    ]
    expected_shards = [
        "model-00006-of-00131.safetensors",
        "model-00008-of-00131.safetensors",
        "model-00010-of-00131.safetensors",
        "model-00011-of-00131.safetensors",
        "model-00013-of-00131.safetensors",
        "model-00016-of-00131.safetensors",
        "model-00017-of-00131.safetensors",
        "model-00020-of-00131.safetensors",
        "model-00022-of-00131.safetensors",
        "model-00025-of-00131.safetensors",
        "model-00026-of-00131.safetensors",
        "model-00028-of-00131.safetensors",
        "model-00030-of-00131.safetensors",
        "model-00033-of-00131.safetensors",
        "model-00034-of-00131.safetensors",
        "model-00037-of-00131.safetensors",
    ]

    assert [ref.global_row for ref in refs] == expected
    assert [ref.shard_name for ref in refs] == expected_shards


def test_qwen_addresses_match_official_transformers_reference(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config_module = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    addressor = qwen_addressor()
    refs = addressor.address([3, 4, 5])

    class RowIdEmbedding:
        def __init__(self, num_embeddings, embedding_dim):
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = torch.empty(0)

        def __call__(self, indices):
            return (
                indices.to(torch.float64).unsqueeze(-1).expand(*indices.shape, self.embedding_dim)
            )

    monkeypatch.setattr(modeling.nn, "Embedding", RowIdEmbedding)
    config = config_module.Qwen4ExpTextConfig(
        vocab_size=248_320,
        hidden_size=2_560,
        num_hidden_layers=48,
        ple_layer_ids=[2],
        ple_embed_dim=2_560,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        eos_token_id=248_044,
    )
    reference = modeling.Qwen4ExpTextNGramEmbedding(config, 2_560, layer_idx=1, ple_layer_index=0)
    ref_output = reference(torch.tensor([[3, 4, 5]]), None)
    actual_rows = [int(ref_output[0, -1, head * 160].item()) for head in range(16)]

    assert [ref.global_row for ref in refs] == actual_rows
    assert [(ref.order, ref.head) for ref in refs] == [
        (order, head) for order in (2, 3) for head in range(8)
    ]
    assert all(ref.dtype == "BF16" and ref.scale_tensor_name is None for ref in refs)
    assert [ref.row for ref in refs] == [ref.global_row % 2_500_012 for ref in refs]


def test_qwen_eos_boundary_matches_official_transformers_reference(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config_module = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    addressor = qwen_addressor()
    refs = addressor.address([248_044, 5])
    expected = [
        15_389_869,
        39_778_609,
        55_713_969,
        62_213_332,
        88_817_728,
        118_483_999,
        133_731_511,
        155_458_159,
        179_763_390,
        197_956_758,
        205_378_969,
        220_499_474,
        242_466_248,
        265_658_744,
        293_662_119,
        315_720_898,
    ]
    assert [ref.global_row for ref in refs] == expected

    class RowIdEmbedding:
        def __init__(self, num_embeddings, embedding_dim):
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = torch.empty(0)

        def __call__(self, indices):
            return (
                indices.to(torch.float64).unsqueeze(-1).expand(*indices.shape, self.embedding_dim)
            )

    monkeypatch.setattr(modeling.nn, "Embedding", RowIdEmbedding)
    config = config_module.Qwen4ExpTextConfig(
        vocab_size=248_320,
        hidden_size=2_560,
        num_hidden_layers=48,
        ple_layer_ids=[2],
        ple_embed_dim=2_560,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        eos_token_id=248_044,
    )
    reference = modeling.Qwen4ExpTextNGramEmbedding(config, 2_560, layer_idx=1, ple_layer_index=0)
    ref_output = reference(torch.tensor([[248_044, 5]]), None)
    actual_rows = [int(ref_output[0, -1, head * 160].item()) for head in range(16)]

    assert [ref.global_row for ref in refs] == actual_rows


def test_collision_identity_uses_actual_order_key_not_full_context():
    addressor = deepseek_addressor()
    first = addressor.lookup_key([1, 2, 4, 5], order=2)
    second = addressor.lookup_key([9, 2, 4, 5], order=2)
    assert first == second == (5, 4)


def test_collision_identity_collapses_deepseek_normalized_aliases():
    addressor = deepseek_addressor()
    compressed = list(addressor._deepseek_compressed_ids)
    compressed[10] = compressed[5]
    addressor = replace(addressor, _deepseek_compressed_ids=tuple(compressed))

    assert addressor.lookup_key([1, 2, 3, 5], order=2) == addressor.lookup_key(
        [1, 2, 3, 10], order=2
    )


def test_collision_identity_reproduces_qwen_eos_reset():
    addressor = qwen_addressor()
    eos = 248_044
    assert (
        addressor.lookup_key([eos, 3, 4, 5], order=3)
        == addressor.lookup_key([3, 4, 5], order=3)
        == (5, 4, 3)
    )
