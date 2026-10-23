# Donor embedding and native n-gram comparison — 2026-10-23

## Scope and answer

The two exact public checkpoints matching the requested names are [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). DeepSeek's official change log says the 1017 model replaced the September V4.1 Flash release on 2026-10-17; `deepseek-flash` is a moving API alias, while the Hugging Face repository is the weight source. Qwen's official repository identifies Flash-Next as the open-weight model released 2026-08-26; the Qwen3.8-Flash API is a served product based on that checkpoint with extra production features. These names are verified; “Qwen 3.8 Flash Next” is not a guessed alternative.

Both checkpoints provide a conventional input token embedding table and substantial native n-gram memory. These are different assets and must not be conflated. The ordinary input table is an optional baseline donor that maps tokens to exact byte unigram/bigram rows. The required native-memory path samples real donor contexts, reads their hashed table rows, joins the ordered donor components, projects once per donor, and aggregates each context onto Megatoken's hashed byte suffix rows for configured higher n-gram orders. This path is implemented and passed live extraction for both pinned donors. Production preparation caches the referenced shards in CoreWeave and verifies their checksums before bounded local selected-row reads; small checks can instead use bounded HTTP ranges. Hash parity, source layout, and dequantization are checked independently of model-quality evaluation.

## Verified checkpoint metadata

Metadata was read from official Hugging Face repository files at the pinned commits below. Safetensors files were not downloaded: for each relevant shard, I requested only the first 8 bytes and the bounded safetensors header, which exposed the tensor names, dimensions, and data type.

| Donor | Exact repository and revision | License | Ordinary input embedding | Approximate stored table size |
|---|---|---|---|---|
| DeepSeek update | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), commit `1bae0e1088993b9b82de08f59de5be5a34e42170` | MIT | `embed.weight`, shard `model-00002-of-00048.safetensors`, BF16 `[129280, 5120]` | 1,323,827,200 bytes tensor payload (about 1.23 GiB) |
| Qwen open weights | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540), commit `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Qwen Community License 1.0 | `model.language_model.embed_tokens.weight`, shard `model-00130-of-00131.safetensors`, BF16 `[248320, 2560]` | 1,271,398,400 bytes tensor payload (about 1.18 GiB) |

Both configs say `tie_word_embeddings: false`; the output head is a separate table. For this transfer, use the **input** embedding: it is the vector looked up for each input token. Do not silently substitute `lm_head.weight`, which is used to score output tokens. Qwen's separate output table is `lm_head.weight` in `model-00131-of-00131.safetensors`, also `[248320, 2560]` BF16. DeepSeek has its own head naming/layout; the official model implementation separately constructs `embed` and `head`.

Qwen's license broadly allows use, modification, fine-tuning and distribution with its notice retained. Two additional clauses matter for commercial deployment: products above 100 million monthly active users or US$20 million monthly revenue must prominently display the model name; and a business that provides Model-as-a-Service or an AI Work Assistant needs a separate Qwen license for commercial use (internal use is excepted under the stated conditions). Check the [license file](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE) before any such deployment. DeepSeek's checkpoint carries the [MIT license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE).

## Raw byte compatibility with the current importer

I tested the strict production byte codec and a separate donor-only reader against each official tokenizer JSON. The donor reader preserves the full model vocabulary row count and original token IDs, masks declared special rows before byte decoding, and leaves padded or non-byte protocol rows empty. It does not change `ByteCodec`'s stricter raw-text tokenization rules.

- **Qwen ordinary token embeddings: compatible with the donor reader.** The tokenizer has 248,320 model rows, 248,044 BPE entries, 21 special additions, and 12 additional byte-spellable protocol additions. The reader maps 248,056 lexical rows and verifies all 256 one-byte fallbacks. The tokenizer declares NFC normalization, but donor transfer reads stored ByteLevel spellings; it does not encode fresh prose through the donor normalizer.
- **DeepSeek ordinary token embeddings: the production codec correctly rejects it as a full raw-text codec.** Three declared special spellings inside the base vocabulary (`<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, and `<｜▁pad▁｜>`) contain characters outside the byte alphabet. The donor reader masks those special IDs before decoding and omits additional non-byte protocol spellings; it maps 128,008 lexical rows in the full 129,280-row model vocabulary and verifies all 256 byte fallbacks. DeepSeek's multiple Split pre-tokenizer rules and prefix-space metadata remain unsupported for raw-text tokenization, which is immaterial to this vocabulary-only baseline.

The pinned donor profiles carry the official repository/revision, license label, tokenizer/config/index hashes, source tensor and shard, vocabulary/width, and the large shard's SHA256 and byte size. Their reader returns a separate byte-vocabulary object, preserving row IDs while skipping controls and padding. The ordinary embedding extraction verifies the source shard digest and hidden width before creating its baseline map. The Qwen input shard is 2,974,788,136 bytes; DeepSeek's is 1,323,858,272 bytes. Only metadata and bounded headers were read; no model weights were downloaded.

## Native n-gram memory: separate, much larger, incompatible key spaces

### DeepSeek Engram

The 1017 config declares two Engram layers (1 and 14), a maximum n-gram length of 4, 8 hash heads, and 16 million as the base hash vocabulary. It stores two actual tables:

- `layers.1.engram.embed.weight`, `model-00047-of-00048.safetensors`, FP8 E4M3 `[384006168, 256]`.
- `layers.14.engram.embed.weight`, `model-00048-of-00048.safetensors`, FP8 E4M3 `[384016682, 256]`.

Each table's FP8 values occupy about 98.3 GB (about 91.5 GiB), plus a scale tensor of about 3.1 GB. The model card describes Engram as conditional memory with roughly 196B parameters overall. These 384-million-row tables are **not** token-ID indexed. The official [`inference/engram.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py) maps tokenizer IDs through a normalization/compression map, constructs rolling hashes over preceding 2-, 3-, and 4-token contexts, and sends each context to multiple disjoint prime-sized hash buckets per head and layer. `inference/model.py` then combines fetched rows through a learned projection and a stream-dependent gate. These are not token embedding vectors; the ordinary table is a separate optional baseline.

The required native transfer reproduces those context hashes, reads 24 ordered table rows per context at each layer, dequantizes the FP8 values with eight E8M0 scales (one scale for each 32-value block), joins the vectors and projects once per donor. The weight tensors are `[384006168, 256]` and `[384016682, 256]`; their scale tensors are `[384006168, 8]` and `[384016682, 8]`. Each scale entry multiplies one 32-value block as the official DeepSeek reader does. The two table-plus-scale payloads are 101,377,628,352 and 101,380,404,048 bytes; their whole safetensors shards are 101,535,150,936 and 101,537,926,640 bytes. It then aggregates the result onto Megatoken's hashed byte suffix rows. The donor's learned projection and stream-dependent gate are not part of the table rows, so the transfer initializes feature rows rather than transplanting a working Engram layer.

### Qwen PLE / N-gram Embedding

Qwen's official [release README](https://github.com/QwenLM/Qwen3.8-Flash-Next) describes a local-context lookup and confirms 51B additional n-gram embedding parameters. Its pinned config specifies n-gram size 3 (bigrams/trigrams), base bucket size 20,000,000, 8 heads per n-gram order, 128 split parts, and a single configured PLE layer. The official safetensors index contains 128 tensors named `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight` through `shard_127.weight`; inspected headers show BF16 `[2500012, 160]` per shard. Together they are 320,001,536 rows × 160 values, or 51.2B BF16 parameters and about 102.4 GB (95.4 GiB) of element data. This is a second real native n-gram memory, distinct from `embed_tokens.weight`.

Those table rows are organized across n-gram/head bucket parts rather than ordinary token IDs. The [Hugging Face Transformers reference implementation](https://github.com/huggingface/transformers/blob/5c65f7c651a646bca8a6f85ae3cfa866ee6a606f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) specifies the lookup: layer-seeded multipliers mix current and preceding token IDs with rolling XORs; each bigram/trigram order maps through eight distinct prime-sized heads and a per-head row offset. EOS resets available context. The config calls the PLE layer one-indexed; weights place it at `layers.1`. The required native transfer reads 16 ordered rows per real context, joins and projects them once per donor, then maps the byte span to Megatoken's hashed suffix rows. The donor's learned PLE projection and convolution are not reproduced by the table rows alone; this initializes feature rows rather than copying a working PLE layer.

## Multi-donor implication and recommendation

Each donor is extracted into its own independently projected map. The model can mount multiple donor maps at once; it does not average vectors across donors. Each map records source revision, input tensor, tokenizer fingerprint, coverage, and support, so the training path can use one donor's descriptor without allowing a larger vocabulary to dominate another donor.

Required implementation and validation:

1. The ordinary input-embedding baseline is optional and creates its own independent map; it is never averaged with native memory or another donor.
2. Native Engram/PLE transfer is the required production route. It reproduces each hash function, reads the official tensor/shard layout and DeepSeek's FP8 scales, maps real context spans to source bytes, and writes each donor to an independent map. Fixed parity cases against the reference implementations and pinned source metadata checks are required. Normal model evaluation follows extraction; a separate quality experiment is not a prerequisite.

### Parity evidence and validation still pending

The DeepSeek compressed-ID map was computed independently with the vendor's pinned reference helper and the implementation in `native_donor_hash.py`. Both contain 129,280 entries, compress to 99,092 IDs, and have the same SHA256 (`c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481`) when each entry is serialized as a little-endian 32-bit integer. This full-map check catches normalization, partial-UTF8 fallback, or whitespace behavior changes. Fixed row fixtures cover both DeepSeek tables and Qwen's reference model class. A small cloud extraction and range-read verification remains separate from these address-only checks.

## Primary sources

- DeepSeek's official change log, including the 2026-10-17 update: [DeepSeek API updates](https://api-docs.deepseek.com/updates/).
- DeepSeek's official 1017 card and technical report: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
- DeepSeek's reference hash/layout implementation: [`inference/engram.py` at the pinned commit](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py) and [model lookup/projection](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py).
- Qwen's official release announcement and architecture summary: [Qwen3.8-Flash-Next repository](https://github.com/QwenLM/Qwen3.8-Flash-Next).
- Qwen's official model card, config, tokenizer, weight index and license: [Qwen3.8-Flash-Next on Hugging Face](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).
- Qwen PLE dimensions and hash implementation: [Transformers Qwen4-Exp configuration](https://github.com/huggingface/transformers/blob/5c65f7c651a646bca8a6f85ae3cfa866ee6a606f/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py) and [model implementation](https://github.com/huggingface/transformers/blob/5c65f7c651a646bca8a6f85ae3cfa866ee6a606f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py). This is the model-library reference implementation, not a file in Qwen's model card repository.

## Confidence and limits

**High confidence** in release identities, immutable revisions, licensing text, ordinary embedding names/shapes/dtypes, large-shard SHA256 values, Qwen's 128 PLE tensor shapes, DeepSeek's two Engram tensor shapes, and donor reader coverage: these were checked against official repository metadata and bounded safetensors headers. The official DeepSeek reference hash code and Transformers Qwen4-Exp reference implementation provide both address algorithms. **High confidence** that neither native table can be transplanted row-for-row into exact byte keys. Native import correctness still needs implementation and parity validation. No weights were downloaded and no training/cloud job was started.
