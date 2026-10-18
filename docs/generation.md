# Generation, exact bytes, and policy likelihood

The same `FacetedGenerator` owns supervised training, parallel generation,
augmented action scoring, and recurrent replay. `FacetedCore` owns factual
ingestion, memory, deterministic routing programs, and the shared P/U experts.
The generator calls those modules; it does not maintain a second serving model.

This is an implementation built from the supplied v13 design. The separately
mentioned v12.1 foundation and full shape registry were not present in the
repository. Their undisclosed serialization details are not silently inferred.
The implementation has explicit, versioned contracts and reports actual storage
rather than adding unused parameters to reproduce a proposed ledger.

## Public interfaces

```python
import torch
from megatoken.generation import ByteCodec, FacetedGenerator, ActionSpec
from megatoken.model.config import ModelConfig
from megatoken.model.core import FacetedCore

config = ModelConfig.tiny(max_capacity=32)
model = FacetedGenerator(FacetedCore(config), ByteCodec.byte_fallback())
state = model.core.initial_state(batch_size=2)

plan = model.prepare_target(
    target_bytes=[b"first continuation", b"second continuation"],
    capacity=32,
    terminal=[False, True],
)
result = model.supervised_step(state, plan=plan, prior_probability=0.5)
# Every value is an unweighted SUM and its actual denominator.
loss = sum(value / max(count, 1) for value, count in result.losses.values())
loss.backward()
next_state = result.state

one_episode = model.core.initial_state(batch_size=1)
spec = ActionSpec(text_kind="ascii", max_tokens=32, max_repairs=1)
sample = model.sample(one_episode, spec=spec,
                      generator=torch.Generator().manual_seed(42))
rescored = model.score_tape(one_episode, sample.tape, spec=spec)
torch.testing.assert_close(sample.log_prob, rescored.log_prob, rtol=0, atol=0)
assert model.codec.decode(sample.emitted_ids) == sample.payload
```

The simple loss reduction above is for a single local batch. Production training
uses the trainer's globally aggregated denominators over accumulation and
replicas, applies the configured weights once, and then steps the optimizer.
`PreparedTargets.denominators` is available before any model forward, so gradient
accumulation does not have to retain all microbatch graphs just to discover a
normalizer. Role IDs default to assistant (2); pass document (4) or the appropriate
causal record role when training a different emission owner.

`prepare_target` also accepts actual `target_ids`. It preserves their original
segmentation and decodes them to obtain the bytes to commit. `free_mask` has one
entry per actual target ID; padding, copied fields, and unrelated observations
do not become extra lexical targets. Raw supervision is batched over independent
episodes with a common generated capacity. Typed actions and policy replay
currently execute one episode at a time; the trainer accumulates those examples.

## The raw Ling codec

The offline bootstrap is a complete 256-row vocabulary where token ID equals byte
value. It supports NUL, arbitrary binary payloads, invalid UTF-8, and empty input.
It is deliberately not advertised as the production tokenizer.

Production imports the output mapping from:

| Asset | Pin |
| --- | --- |
| Repository | `inclusionAI/Ling-3.0-tiny` |
| Revision | `9a98e35fe1c9ee255f78dd64771c7ae15a799481` |
| File | `tokenizer.json` |
| File SHA256 | `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60` |
| Output rows | 157,184 |
| Lexically emittable base rows | 156,891 |
| FMT raw codec version | `raw-byte-bpe-v1;binary=single-byte-v1` |

The asset can be obtained through the repository's acquisition tooling, without
downloading teacher weights. The importer verifies its bytes before constructing
the model vocabulary:

```python
from megatoken.generation.codec import ByteCodec, LING_TOKENIZER_SHA256

codec = ByteCodec.from_tokenizer_json(
    "artifacts/codec/tokenizer.json",
    expected_sha256=LING_TOKENIZER_SHA256,
    expected_vocab_size=157_184,
)
```

The original tokenizer applies NFC normalization. That would change decomposed
Unicode and violate byte fidelity. FMT explicitly disables normalization,
postprocessing, and added-special-token interpretation while retaining the raw
ByteLevel BPE vocabulary, merges, and pre-tokenizer boundaries. A string that
looks like `<|role_end|>` remains ordinary text. Added protocol tokens and padded
rows have no lexical emission bytes and are excluded from the probability
normalizer as well as from sampling.

Valid UTF-8 uses this raw BPE convention. Non-UTF-8 payloads use an explicit
single-byte fallback convention, including when part of the payload could have
been merged. The codec fingerprint binds the asset hash, output row count, and
convention version. Checkpoints and action tapes carry it. The fast path uses
the Rust `tokenizers` library; a Python BPE implementation provides a readable
reference. Both paths were checked against the pinned asset on combining
characters, multibyte Unicode, special-looking strings, NUL, and binary data.

Input representations remain native bytes. This imports neither a teacher input
embedding nor a teacher transformer. The source asset is documented on its
[publisher file page](https://huggingface.co/inclusionAI/Ling-3.0-tiny/blob/9a98e35fe1c9ee255f78dd64771c7ae15a799481/tokenizer.json).

## Ordered generation boundaries

| Boundary | Executed decisions and work |
| --- | --- |
| Before stage 7 | Fix an allowed effort from causal task information. Initial outcome RL requires effort 6. |
| Stage 7 | Draw the causal root Gaussian; execute the core update; choose mode, available schema, and size bucket. |
| Stage 8 | Choose feasible capacity within the bucket; begin parallel growth through scale 64. |
| Stage 10 | Choose historical bindings and early identifier values; draw regional innovations conditioned on those actual ancestors. |
| After stage 10 | Compile future routes once from the actual chosen ancestors; complete the entire capacity workspace. |
| Stage 15 | Choose feasible lengths, layout, and copy insertion placements. |
| Stage 16 | Complete the final plan, choose terminal status, refine, and draw free lexical positions in parallel. |
| Repair | Validate the whole action; preserve each failed attempt and every stochastic repair decision; run bounded parallel replacement passes. |
| Commitment | Reconcile private carry from the actual augmented actions; ingest a validated physical output record exactly once. |

The exact length never reduces growth before stage 15. The shared U always FFN
also performs the final dense lexical refinement. Local block attention and a
separately normalized pooled global branch provide hybrid output attention;
plan and source cross-attention remain separate. Growth and repair reuse their
attention modules. None of these served modules receives the target posterior's
hidden activations.

An empty validated terminal action is still a chronological record boundary: it
has zero lexical positions, an explicit role, and one factual recurrent step.
An invalid action receives no factual record and is not executable. Its sampled
attempts still affect private carry, which allows later recovery from the actual
branch.

## Typed tools, identifiers, and exact copies

`ActionSpec` contains only constraints available before the current output:
schemas, actual earlier copy sources, length limits, and repair policy. Its
fingerprint is checked during replay. Schemas must not be inferred from the
target and then passed in as conditioning information. The target may select a
supervised label among schemas that were already available.

```python
from megatoken.generation import ActionSpec, CopySource, FieldSpec, SchemaSpec

schema = SchemaSpec("write", (
    FieldSpec("path", kind="identifier", min_tokens=1, max_tokens=64,
              identifier="path_id"),
    FieldSpec("same_path", kind="identifier", min_tokens=1, max_tokens=64,
              bound_to="path_id"),
    FieldSpec("content", kind="utf8", max_tokens=256),
))
spec = ActionSpec(
    schemas=(schema,),
    copy_sources=(CopySource("earlier-source", b"exact earlier text"),),
    max_tokens=512,
    require_terminal=False,
)
```

An early identifier is scored once. All bound occurrences reuse its actual IDs
and bytes. A historical copy binding is chosen before regional innovations;
stage 15 chooses its insertion point within the generated field. Copied tokens
occupy capacity positions and have no lexical CE. Copy sources can preserve
their original emitted IDs; the codec verifies that those IDs decode to the
declared bytes.

`kind="json"` represents a raw JSON value, including nested objects, arrays,
numbers, booleans, and strings. `json_schema` supplies its public JSON Schema as
a JSON string. `fixed_json` provides deterministic field syntax, such as a fixed
tool name or wire-format tag. Standard tool schema compilation lives in
`megatoken.tool_schema`; canonical, OpenAI, and Anthropic envelopes preserve their
actual parameter types.

The cached validator rejects invalid UTF-8, nonstandard JSON constants, duplicate
object keys, and failed parameter schemas. References must be bundled locally.
FMT explicitly validates `contentSchema` when `contentMediaType` is
`application/json`; unlike ordinary annotation-only treatment, this checks the
decoded argument string in OpenAI-style tool calls before commitment.

`prepare_action_target(payload, spec=..., bindings=..., layout=...)` returns a
`PreparedAction` with all loss denominators. Positive binding labels index the
already available copy sources from one; zero means generate. The preparation
method explicitly canonicalizes deterministic fixed JSON syntax and exposes the
result as `plan.payload`. The trainer only uses this typed path when that result
matches the actual record bytes; otherwise it retains the raw masked reconstruction
path. Preparation excludes fixed syntax when a causal public schema supplies it;
unsupported legacy envelopes retain their raw-text targets. This auxiliary objective
is separate from exact augmented action likelihood. Original records are never silently
rewritten in recurrent history.

`supervised_action` trains mode/schema, identifier, binding, placement, length,
terminal, and lexical heads through their ordinary factorization. Optional repair
supervision uses a genuinely sampled draft as input, retaining exact bound fields
and scoring only replacement free positions. It does not reveal replacement
targets through draft embeddings.

JSON escaping is deterministic. Unchanged lexical pieces retain their original
emitted segmentation. When a raw string piece needs escaping, the serializer
creates the corresponding deterministic escape IDs. The action tape keeps both
the original sampled IDs and the final emitted IDs. Always verify
`codec.decode(tape.emitted_ids) == tape.payload`; canonical retokenization is not
the action likelihood.

## Posterior, prior, and loss accounting

The target posterior has its own byte embedding and content encoder. The served
prior sees causal state, control, and the actual sampled ancestors. Gaussian
means, bounded log scales, densities, and KL statistics remain FP32 even after
converting the model to BF16.

`prior_fit` computes `KL(stop_gradient(q) || p)`. `posterior_pressure` computes
`KL(q || stop_gradient(p))`: both use the same KL orientation, with different gradient
recipients. Tiny statistics heads are recomputed on detached ancestor values for these
auxiliary objectives so a reparameterized ancestor cannot introduce an accidental
gradient into the other network. Lexical paths and scored served likelihoods retain normal gradients
through causal state. Content alignment excludes private carry.

Priors train from the first batch. `prior_probability` selects actual prior draws
for reconstruction, separately from posterior draws. The result reports both
lexical loss sums and position counts, so good posterior reconstruction cannot
hide a weak served prior. Repair's additional lexical pass is included in P.
Numerical diagnostics in `SupervisedResult.metrics` remain detached scalar
tensors until transaction logging; static counts remain ordinary numbers. This
avoids a host synchronization for every loss or diagnostic. Sparse free-position
indices come from the prepared host plan, so selecting scored rows does not need
to discover a dynamic mask size on the GPU.

Loss keys are `lexical`, `structure`, `binding`, `placement`, `terminal`,
`prior_fit`, `posterior_pressure`, `content_alignment`, `router_balance`, and
`router_z`. P and U router objectives are unweighted before the trainer applies
the curriculum weights. Structural counts include each explicitly evaluated
categorical head, even when public constraints give it probability one. Fixed
serialized punctuation does not acquire a categorical factor.

## Full vocabulary without a full logit matrix

`TiedVocabulary` uses one rank projection and one vocabulary embedding, including
for draft/repair embeddings. Cross-entropy and sampling scan the complete masked
vocabulary in position and vocabulary tiles. Forward stores the partition
function, targets, and factors. A custom first-order backward recomputes each
tile and accumulates hidden/weight gradients. No length-by-vocabulary matrix is
saved for backward.

CPU and FP64 tests compare the mathematical reference with dense masked
cross-entropy, including both gradients. On CUDA, BF16/FP16 operands use
`torch.mm(..., out_dtype=torch.float32)` so the expensive products use Tensor
Cores while normalizers and accumulated gradients remain FP32. Backward uses
ordinary mixed-precision derivative tiles. Passing `tensorcore=False` selects the
FP32 reference for numerical diagnosis. See the installed PyTorch 2.10
[`torch.mm` contract](https://docs.pytorch.org/docs/stable/generated/torch.mm.html).

Gumbel-max sampling scans every allowed row. It uses neither top-k nor top-p
truncation and does not pretend the multiplication has disappeared. The
augmented policy API uses temperature 1. The CUDA numerical-parity test is an
explicit hardware gate; a CPU test pass is not a Blackwell throughput result.

## Complete tapes and recurrent policy gradients

`ActionTape` is immutable and JSON serializable. It stores root and regional
values, all categorical choices, early identifier draws, every draft and failed
repair attempt, exact final IDs/bytes, terminal/commit status, and behavior log
density. The tape contains no authoritative expert route or hidden-state cache.

`sample` and `score_tape` run the same ordered event interpreter. Scoring consumes
the recorded values, renormalizes over the current feasible support, and rejects
missing, extra, out-of-support, or inconsistent events. Continuous innovation
densities are part of the likelihood. Deterministic serialization is not an
additional token distribution. The final repaired string is not scored as an
ordinary autoregressive sequence.

`replay` reconstructs from actual observations and complete preceding tapes.
Use `Observation(payload, role_id)` to preserve user, system, tool, document, or
environment ownership; a bare byte payload is a tool observation for backward
compatibility. Candidate weights rebuild parameter-dependent state and compute
their own routes. The helper retains the full differentiable short-episode
graph. Adding explicit detach boundaries is a truncated, generally biased
recurrent policy-gradient approximation and must be reported as such.

The RL trainer collects independent same-task rollouts under the current policy,
verifies the executed outcome, and reconstructs every candidate trajectory with
gradients before updating. A same-weight likelihood gate runs before Adam.
Equal-reward groups skip the optimizer entirely, preserving both weights and
momentum; the episode ceiling still applies when public fixed arguments make
free lexical P zero. Full checkpoints retain task content/order identity, the
next task-group cursor, each rank's RNG, optimizer masters/moments, and cumulative
meters. Integration tests compare uninterrupted and resumed weights exactly.
RL episode termination belongs to the environment's `done` response or the configured
step limit; a raw tape's terminal flag does not end the task by itself.

D/I use the raw codec's canonical offline token-reference count of actual record
bytes, including both collection and candidate replay in I. P counts the exact
sampled free IDs, including failed drafts and repair attempts. Noncanonical
emitted segmentations remain in the tape for policy likelihood and never change
the definition of the offline source-volume meter.

## Export and verification

```python
served = FacetedGenerator(FacetedCore(model.config), model.codec,
                          training_heads=False)
served.load_state_dict(model.served_state_dict(), strict=True)
assert served.posterior is None
```

Served export rejects an unbaked engram adapter. After baking, it includes no
posterior parameters and needs no donor asset. Strict loading and same-seed
byte/tape parity are covered by tests. Supervised calls on a served-only instance
fail clearly.

`megatoken.inference.export_served` writes an immutable, atomic bundle containing
served safetensors, configuration, a manifest with checksums, and the actual raw
tokenizer asset when applicable. `InferenceEngine` verifies the bundle and uses
strict loading with `training_heads=False`; observations and outputs use the same
role-aware physical-record ingestion and shared policy as training.

The CLI's `serve --model BUNDLE` command offers a local JSON-lines protocol over
standard input/output. Each line is a complete request. Responses echo an
optional request `id` and contain `ok`. Start with `open`, then use the returned
session identifier in subsequent requests:

| Operation | Request fields | Result |
| --- | --- | --- |
| `open` | Optional `id` | New session and model version. |
| `observe` | `session`, `role`, and exactly one of `text` or `payload_b64` | Ingest one actual causal record and return the record count. |
| `generate` | `session`; optional `seed`, `kind`, `min_tokens`, `max_tokens`, `max_repairs`, `effort` | Complete augmented tape, actual emitted IDs, bytes as `payload_b64`, and commit/terminal status. |
| `status` | `session` | Counts of records, decisions, and replay events. |
| `close` | `session` | Release that session. |

Generation may instead receive public `tools` and permitted `wire_formats`
(`canonical`, `openai`, `anthropic`). Copy sources come only from that session's
actual previous records. A `text` response is included only when the generated
bytes decode as valid UTF-8; base64 is always authoritative. Malformed requests
receive an error response without erasing the session. The protocol starts no
network listener and enforces a configurable session limit.

Run the focused suite with:

```bash
pytest -q tests/test_generation_codec.py tests/test_generation_distributions.py tests/test_generation_policy.py tests/test_inference.py tests/test_rl_integration.py
```

It covers exact binary/UTF-8 paths, special/padded vocabulary masks, noncanonical
IDs, tiled gradients, FP32 Gaussian statistics, causal target isolation,
capacity-before-length ordering, once-only commitment, typed tools, bound IDs,
copies, repair accounting, failed attempts, replay/carry parity, candidate state
dependence, and served-only export. A small integrated task demonstrates that
prior-driven training can reduce a byte objective and generate its learned
answer. It is an engineering learnability check, not evidence of general agent
quality. Large-model throughput, Blackwell kernel parity, and downstream task
success remain explicit run-time evaluation gates.
