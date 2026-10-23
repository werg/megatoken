"""Pinned address functions for native DeepSeek Engram and Qwen PLE tables.

The returned references identify donor rows; they do not load model weights.
All references preserve donor-specific hash component order and tensor layout.
DeepSeek's compression and row hashes follow its pinned official inference code.
Qwen's PLE addresses follow the pinned Transformers Qwen4-Exp reference code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np

from megatoken.donors import DonorAsset, load_donor_profile

DEEPSEEK_REFERENCE = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/"
    "1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py"
)
QWEN_REFERENCE = (
    "https://github.com/huggingface/transformers/blob/"
    "5c65f7c651a646bca8a6f85ae3cfa866ee6a606f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
)
DEEPSEEK_COMPRESSED_MAP_SHA256 = "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"


@dataclass(frozen=True)
class NativeRowRef:
    """One ordered donor hash component and its physical tensor row."""

    layer_id: int
    order: int
    head: int
    component_index: int
    global_row: int
    row: int
    tensor_name: str
    shard_name: str
    shard_sha256: str
    shard_size: int
    width: int
    dtype: str
    scale_tensor_name: str | None = None
    scale_row: int | None = None
    scale_width: int | None = None
    scale_dtype: str | None = None
    block_size: int | None = None


@dataclass(frozen=True)
class NativeDonorAddressor:
    """A donor's reference hash contract plus row-address lookup."""

    profile: DonorAsset
    max_context_tokens: int
    native_memory: Literal["engram", "ple"]
    _index: dict[str, str]
    _config: dict
    _deepseek_compressed_ids: tuple[int, ...] = ()
    _deepseek_compressed_vocab_size: int = 0
    compressed_token_map_sha256: str | None = None

    def address(
        self,
        context_ids: tuple[int, ...] | list[int],
        *,
        layer_id: int | None = None,
    ) -> tuple[NativeRowRef, ...]:
        """Return source rows for a chronological context ending at the current token.

        Contexts shorter than the configured maximum are padded at the start using
        the donor's reference pad convention. No tokens are invented after the
        current token. The result is stable in layer, n-gram order, then head order.
        """
        if not context_ids:
            raise ValueError("A native n-gram lookup needs a current token ID")
        ids = tuple(int(token_id) for token_id in context_ids)
        if any(token_id < 0 or token_id >= self.profile.vocabulary.vocab_size for token_id in ids):
            raise ValueError("Native donor context contains an out-of-range token ID")
        if self.native_memory == "engram":
            return self._deepseek_address(ids, layer_id)
        return self._qwen_address(ids, layer_id)

    def lookup_key(self, context_ids: tuple[int, ...] | list[int], order: int) -> tuple[int, ...]:
        """Return the exact ordered donor IDs hashed for one n-gram order."""
        ids = tuple(int(token_id) for token_id in context_ids)
        if not ids or any(
            token_id < 0 or token_id >= self.profile.vocabulary.vocab_size for token_id in ids
        ):
            raise ValueError("Native donor context contains no valid token sequence")
        if not 2 <= order <= self.max_context_tokens:
            raise ValueError("Native lookup order is outside the donor's n-gram range")
        if self.native_memory == "engram":
            args = self._config["text_config"]
            compressed = self._deepseek_compressed_ids
            pad = compressed[int(args["engram_pad_token_id"])]
            tokens = [
                compressed[token_id] for token_id in reversed(ids[-self.max_context_tokens :])
            ]
            tokens.extend([pad] * (self.max_context_tokens - len(tokens)))
            return tuple(tokens[:order])

        args = self._config["text_config"]
        eos = int(args["eos_token_id"])
        prior = list(ids[:-1])
        if eos in prior:
            last_eos = len(prior) - 1 - prior[::-1].index(eos)
            prior = prior[last_eos + 1 :]
        tokens = [ids[-1], *reversed(prior[-(self.max_context_tokens - 1) :])]
        tokens.extend([eos] * (self.max_context_tokens - len(tokens)))
        return tuple(tokens[:order])

    def _deepseek_address(
        self, ids: tuple[int, ...], layer_id: int | None
    ) -> tuple[NativeRowRef, ...]:
        args = self._config["text_config"]
        layers = tuple(int(value) for value in args["engram_layer_ids"])
        if layer_id is not None and layer_id not in layers:
            raise ValueError(f"DeepSeek Engram has no table for layer {layer_id}")
        selected_layers = (layer_id,) if layer_id is not None else layers
        max_ngram = int(args["engram_max_ngram_size"])
        compressed = self._deepseek_compressed_ids
        pad = compressed[int(args["engram_pad_token_id"])]
        recent_ids = ids[-max_ngram:]
        tokens = [compressed[token_id] for token_id in reversed(recent_ids)]
        tokens.extend([pad] * (max_ngram - len(tokens)))

        refs = []
        for layer_pos, current_layer in enumerate(layers):
            if current_layer not in selected_layers:
                continue
            table_rows = int(args["engram_num_embeddings"][layer_pos])
            primes = _deepseek_primes(args, layers)
            offsets = _deepseek_offsets(primes)
            multipliers = _deepseek_multipliers(
                current_layer, max_ngram, self._deepseek_compressed_vocab_size
            )
            table = f"layers.{current_layer}.engram.embed.weight"
            scale = f"layers.{current_layer}.engram.embed.scale"
            weight_shard = self._index.get(table)
            scale_shard = self._index.get(scale)
            if weight_shard is None or scale_shard != weight_shard:
                raise ValueError(
                    f"Pinned DeepSeek index is missing the Engram table pair for layer {current_layer}"
                )
            expected_shard = DEEPSEEK_NATIVE_SHARDS.get(weight_shard)
            if expected_shard is None:
                raise ValueError(f"DeepSeek native table shard {weight_shard} is not pinned")
            rolling = tokens[0] * multipliers[0]
            local_component = 0
            for order in range(2, max_ngram + 1):
                rolling ^= tokens[order - 1] * multipliers[order - 1]
                order_primes = primes[layer_pos][order - 2]
                for head, prime in enumerate(order_primes):
                    row = offsets[layer_pos][local_component] + rolling % prime
                    if row >= table_rows:
                        raise ValueError("DeepSeek Engram hash points beyond its declared table")
                    refs.append(
                        NativeRowRef(
                            layer_id=current_layer,
                            order=order,
                            head=head,
                            component_index=layer_pos
                            * (max_ngram - 1)
                            * int(args["engram_n_heads"])
                            + (order - 2) * int(args["engram_n_heads"])
                            + head,
                            global_row=row,
                            row=row,
                            tensor_name=table,
                            shard_name=weight_shard,
                            shard_sha256=expected_shard[0],
                            shard_size=expected_shard[1],
                            width=int(args["engram_head_dim"]),
                            dtype="F8_E4M3",
                            scale_tensor_name=scale,
                            scale_row=row,
                            scale_width=int(args["engram_head_dim"]) // 32,
                            scale_dtype="F8_E8M0",
                            block_size=32,
                        )
                    )
                    local_component += 1
        return tuple(refs)

    def _qwen_address(self, ids: tuple[int, ...], layer_id: int | None) -> tuple[NativeRowRef, ...]:
        args = self._config["text_config"]
        actual_layer = int(args["ple_layer_ids"][0]) - 1
        if layer_id is not None and layer_id != actual_layer:
            raise ValueError(f"Qwen PLE has no table for layer {layer_id}")
        ngram_size = int(args["ngram_size"])
        eos = int(args["eos_token_id"])
        # Transformers' PLE uses EOS for missing history and resets preceding
        # context at EOS boundaries. An EOS in the current position itself does
        # not erase context for that same position.
        prior = list(ids[:-1])
        if eos in prior:
            last_eos = len(prior) - 1 - prior[::-1].index(eos)
            prior = prior[last_eos + 1 :]
        recent = [ids[-1], *reversed(prior[-(ngram_size - 1) :])]
        recent.extend([eos] * (ngram_size - len(recent)))
        heads_per = int(args["heads_per_ngram"])
        head_count = (ngram_size - 1) * heads_per
        prime_sizes, offsets = _qwen_head_layout(int(args["ngram_vocab_size_base"]), head_count, 0)
        multipliers = _qwen_multipliers(
            int(args["vocab_size"]),
            ngram_size,
            0,
            int(args.get("seed", 1234)),
        )
        total_rows = sum(prime_sizes)
        parts = int(args["split_ngram_parts"])
        part_rows = (total_rows + parts - 1) // parts
        prefix = f"model.language_model.layers.{actual_layer}.ple.ple_embedding.ngram_embedding"
        refs = []
        rolling = recent[0] * multipliers[0]
        for order in range(2, ngram_size + 1):
            rolling ^= recent[order - 1] * multipliers[order - 1]
            head_start = (order - 2) * heads_per
            for relative_head in range(heads_per):
                head = head_start + relative_head
                global_row = offsets[head] + rolling % prime_sizes[head]
                part = global_row // part_rows
                row = global_row % part_rows
                tensor = f"{prefix}.shard_{part}.weight"
                shard = self._index.get(tensor)
                if shard is None:
                    raise ValueError(f"Pinned Qwen index is missing native PLE tensor {tensor}")
                expected = QWEN_NATIVE_SHARDS.get(shard)
                if expected is None:
                    raise ValueError(f"Qwen native table shard {shard} is not pinned")
                refs.append(
                    NativeRowRef(
                        layer_id=actual_layer,
                        order=order,
                        head=relative_head,
                        component_index=(order - 2) * heads_per + relative_head,
                        global_row=global_row,
                        row=row,
                        tensor_name=tensor,
                        shard_name=shard,
                        shard_sha256=expected[0],
                        shard_size=expected[1],
                        width=int(args["ple_embed_dim"]) // head_count,
                        dtype="BF16",
                    )
                )
        return tuple(refs)


def load_native_donor(name: str, donor_root: str | Path) -> NativeDonorAddressor:
    """Validate pinned small metadata and build an address-only native donor reader."""
    asset = load_donor_profile(name, donor_root)
    root = Path(donor_root)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((root / "config.json").read_text())
    if name == "deepseek-v4.1-flash-1017":
        args = config["text_config"]
        if args.get("engram_num_embeddings") != [384_006_168, 384_016_682]:
            raise ValueError("DeepSeek Engram table dimensions do not match the pinned profile")
        compressed_ids, compressed_size = _deepseek_compressed_token_ids(root / "tokenizer.json")
        if compressed_size != int(args["engram_compressed_vocab_size"]):
            raise ValueError(
                "DeepSeek compressed tokenizer size differs from the native table profile"
            )
        map_digest = hashlib.sha256(
            b"".join(token_id.to_bytes(4, "little") for token_id in compressed_ids)
        ).hexdigest()
        if map_digest != DEEPSEEK_COMPRESSED_MAP_SHA256:
            raise ValueError(
                "DeepSeek compressed token map differs from the pinned official reference"
            )
        _deepseek_primes(args, tuple(args["engram_layer_ids"]))
        return NativeDonorAddressor(
            asset,
            max_context_tokens=int(args["engram_max_ngram_size"]),
            native_memory="engram",
            _index=index,
            _config=config,
            _deepseek_compressed_ids=tuple(compressed_ids),
            _deepseek_compressed_vocab_size=compressed_size,
            compressed_token_map_sha256=map_digest,
        )
    if name == "qwen3.8-flash-next":
        args = config["text_config"]
        if int(args["split_ngram_parts"]) != 128 or int(args["ngram_size"]) != 3:
            raise ValueError("Qwen PLE layout differs from the pinned profile")
        primes, _ = _qwen_head_layout(
            int(args["ngram_vocab_size_base"]),
            (int(args["ngram_size"]) - 1) * int(args["heads_per_ngram"]),
            0,
        )
        if (sum(primes) + int(args["split_ngram_parts"]) - 1) // int(
            args["split_ngram_parts"]
        ) != 2_500_012:
            raise ValueError("Qwen PLE checkpoint shard row size differs from the pinned profile")
        return NativeDonorAddressor(
            asset,
            max_context_tokens=int(args["ngram_size"]),
            native_memory="ple",
            _index=index,
            _config=config,
        )
    raise ValueError(f"No native n-gram address implementation for donor {name!r}")


def _deepseek_compressed_token_ids(tokenizer_json: Path) -> tuple[list[int], int]:
    try:
        from tokenizers import Regex, Tokenizer, normalizers
    except ImportError as exc:
        raise RuntimeError("DeepSeek native address setup requires the tokenizers package") from exc
    backend = Tokenizer.from_file(str(tokenizer_json))
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), "\ue000"),
            normalizers.Strip(),
            normalizers.Replace("\ue000", " "),
        ]
    )
    keys: dict[str, int] = {}
    mapping = []
    for token_id in range(backend.get_vocab_size(with_added_tokens=True)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
            if key is None:
                raise ValueError(
                    f"DeepSeek tokenizer has no spelling for byte-fallback row {token_id}"
                )
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        mapped = keys.get(key)
        if mapped is None:
            mapped = len(keys)
            keys[key] = mapped
        mapping.append(mapped)
    return mapping, len(keys)


@cache
def _deepseek_multipliers(
    layer_id: int, max_ngram_size: int, compressed_vocab_size: int
) -> tuple[int, ...]:
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    generator = np.random.default_rng(10007 * layer_id)
    values = generator.integers(0, multiplier_bound, size=max_ngram_size, dtype=np.int64)
    return tuple(int(value * 2 + 1) for value in values)


def _deepseek_primes(
    args: dict, layer_ids: tuple[int, ...]
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return _deepseek_primes_cached(
        int(args["engram_max_ngram_size"]),
        int(args["engram_n_heads"]),
        int(args["engram_vocab_size"]),
        layer_ids,
    )


@cache
def _deepseek_primes_cached(
    max_ngram: int,
    heads: int,
    base: int,
    layer_ids: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    seen: set[int] = set()
    layers = []
    for _ in layer_ids:
        orders = []
        for _ in range(max_ngram - 1):
            values = []
            current = base - 1
            for _ in range(heads):
                current = _next_prime(current, seen)
                seen.add(current)
                values.append(current)
            orders.append(tuple(values))
        layers.append(tuple(orders))
    return tuple(layers)


@cache
def _deepseek_offsets(
    primes: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[tuple[int, ...], ...]:
    result = []
    for layer in primes:
        offsets = []
        offset = 0
        for order in layer:
            for prime in order:
                offsets.append(offset)
                offset += prime
        result.append(tuple(offsets))
    return tuple(result)


@cache
def _qwen_head_layout(
    base: int, head_count: int, layer_index: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sizes = tuple(
        _nth_prime_after(base - 1, layer_index * head_count + head + 1)
        for head in range(head_count)
    )
    offsets = []
    current = 0
    for size in sizes:
        offsets.append(current)
        current += size
    return sizes, tuple(offsets)


@cache
def _qwen_multipliers(
    vocab_size: int, ngram_size: int, layer_index: int, seed: int
) -> tuple[int, ...]:
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(vocab_size, 1)) // 2)
    base_seed = seed + 10007 * layer_index
    gamma = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    return tuple(
        2 * (_splitmix64((base_seed + gamma * (i + 1)) & mask) % half_bound) + 1
        for i in range(ngram_size)
    )


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _next_prime(start: int, seen: set[int] | None = None) -> int:
    candidate = start + 1
    seen = seen or set()
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


@cache
def _nth_prime_after(start: int, count: int) -> int:
    current = start
    for _ in range(count):
        current = _next_prime(current)
    return current


# SHA256/size values are the small official Hugging Face LFS metadata records;
# no table shards are part of the source tree or fetched by this module.
DEEPSEEK_NATIVE_SHARDS: dict[str, tuple[str, int]] = {
    "model-00047-of-00048.safetensors": (
        "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5",
        101_535_150_936,
    ),
    "model-00048-of-00048.safetensors": (
        "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd",
        101_537_926_640,
    ),
}

QWEN_NATIVE_SHARDS: dict[str, tuple[str, int]] = {
    "model-00005-of-00131.safetensors": (
        "54fdc7adf431d6b94d9292fc8bd7938d906b0cffe0997d76ac1a176413aa80f2",
        3_355_888_504,
    ),
    "model-00006-of-00131.safetensors": (
        "8277a48e1fd7ab7ff7f0282befcc6eb4e09ab2e0780c00af6685e198faba233e",
        3_200_016_032,
    ),
    "model-00007-of-00131.safetensors": (
        "0fba369354e45b43efbe2773d63b7cb20caedb56ff9ed80e076bbcdaf560666a",
        3_200_016_032,
    ),
    "model-00008-of-00131.safetensors": (
        "024f59b6f2cb27ab79a4267cff9603c398ef1a7ef9ba76009fd34a0a84a9455e",
        3_200_016_032,
    ),
    "model-00009-of-00131.safetensors": (
        "23fbef154f7f68e5b6f555492832c78822a827554757797daf91de48df06b496",
        3_200_016_032,
    ),
    "model-00010-of-00131.safetensors": (
        "b3a303529a190c6661870d5abebf0ac18d36922a1da8db737d19f55bd1031fe8",
        3_200_016_032,
    ),
    "model-00011-of-00131.safetensors": (
        "d1215943b48481febb33a1129c96613dec5f364613a2cc3ba1e27c23be0ca78a",
        3_200_016_032,
    ),
    "model-00012-of-00131.safetensors": (
        "f63d649ea84867ec75884dace5ea4d8c723a6560463674a6ee0ce621b4b9dbc1",
        3_200_016_032,
    ),
    "model-00013-of-00131.safetensors": (
        "5e0f0fb335f4c19a0158f60294e5b5c30bdfc344b003633c9c42e5e638a6808e",
        3_200_016_032,
    ),
    "model-00014-of-00131.safetensors": (
        "5ee73bd21c73b2610d2a30924f7acd24e63b1ca29e0a1f1c87f0f6a9648e347a",
        3_200_016_032,
    ),
    "model-00015-of-00131.safetensors": (
        "b35421bf21724db705a38ee9f779de07f9f1711be964049942960c5e3dddb1c2",
        3_200_016_032,
    ),
    "model-00016-of-00131.safetensors": (
        "73c4b1bd1f3752d70f0161ec8c8a7701f0538e1ae722daf16a905e04a666e69f",
        3_200_016_032,
    ),
    "model-00017-of-00131.safetensors": (
        "ee60c1269974cc797bad8bfc9a0fac72437b323ceafcb225cef6980ecffce1b3",
        3_200_016_032,
    ),
    "model-00018-of-00131.safetensors": (
        "72cf3696aa1af4611b2da2b013868f3724082362899d9dce3e9fec908d441fc2",
        3_200_016_032,
    ),
    "model-00019-of-00131.safetensors": (
        "bdad649be407bd8033a4abd6a24bf434ed6a173807892d18cc1c9771031bfd0c",
        3_200_016_032,
    ),
    "model-00020-of-00131.safetensors": (
        "2258dad55787922f1d4429aea528495511f58447baa9b4f1b7f92f58ba7110b0",
        3_200_016_032,
    ),
    "model-00021-of-00131.safetensors": (
        "775952ac572fa4ff7e5e316ecb60483d4dd96f4369d6a7a403fa31d7f9d1a6d0",
        3_200_016_032,
    ),
    "model-00022-of-00131.safetensors": (
        "7d205070bb898c59d23bc379652607911f39e219627028a29d82e8e8fecab7c0",
        3_200_016_032,
    ),
    "model-00023-of-00131.safetensors": (
        "598d4f772853233b88911fdb1a49f8865e3c732e93b17aea44b65f455e69e9dc",
        3_200_016_032,
    ),
    "model-00024-of-00131.safetensors": (
        "09e534942e7d4eb31803b347e14dd70090f16f85c89b0ad766bdace135806f12",
        3_200_016_032,
    ),
    "model-00025-of-00131.safetensors": (
        "482d18f0e515c23c6933593d0cce289770600c6fa539168bd304e2169cc8a2be",
        3_200_016_032,
    ),
    "model-00026-of-00131.safetensors": (
        "ac479fe29ed3e03b7c3523974371d68dace22b7744ff9236f9429cdbb9241da0",
        3_200_016_032,
    ),
    "model-00027-of-00131.safetensors": (
        "4981d4e636723dbb063f04854e4362c1fe341d9ba80b5bb4231bb0d5ee8ab266",
        3_200_016_032,
    ),
    "model-00028-of-00131.safetensors": (
        "1b45a957b1f3572d325b0795f549fb772c160bd5c5167f94a347b96e3c7b978d",
        3_200_016_032,
    ),
    "model-00029-of-00131.safetensors": (
        "40d2f2291ff4800373087cd00f1f95c3bd69331c32f39cfdd5d43516764b03ec",
        3_200_016_032,
    ),
    "model-00030-of-00131.safetensors": (
        "893a9b59e412a3242252274770efe0794ce8b216b2b3907b6854b7eaa6ddd334",
        3_200_016_032,
    ),
    "model-00031-of-00131.safetensors": (
        "9b6bf01d60f21bed295b9b1d43501593b4ce7ece28aaf75618b4fb58039c0d04",
        3_200_016_040,
    ),
    "model-00032-of-00131.safetensors": (
        "7453fcb5872160888730439ac71a5279d348686d68c6ba2d62ff12461443dc69",
        3_200_016_040,
    ),
    "model-00033-of-00131.safetensors": (
        "5b2f577ce40932031c4749f9177dab2119ff118f9f9b9df3d0d5e57c8e41d128",
        3_200_016_040,
    ),
    "model-00034-of-00131.safetensors": (
        "35b5522668652e8cc7d8eacb0d45c59eb9ed9786e9102a9b71503f746f50da64",
        3_200_016_040,
    ),
    "model-00035-of-00131.safetensors": (
        "4fe611218d36ff7c06e4c24f7252215fdddc879ac3de2b1d569727ed31f82db1",
        3_200_016_040,
    ),
    "model-00036-of-00131.safetensors": (
        "237789c0cef556ef22c9f1ca99b7baf71b33a56b70c5828380e1008e02510649",
        3_200_016_040,
    ),
    "model-00037-of-00131.safetensors": (
        "73b53c94d23589bdfde55b20538a515525f22fd37c7cdf24e85738ed771218e1",
        1_742_347_160,
    ),
}
