# Semantic execution episodes from tracing-js-self

This repository now has a versioned adapter for the `semantic-execution/1` export
format owned by [werg/tracing-js-self](https://github.com/werg/tracing-js-self/blob/main/docs/TRAINING_DATA_PLAN.md).
The adapter lives in `src/megatoken/data/semantic_execution.py`, with a
JSONL converter at `scripts/import_semantic_execution.py`. It is deliberately
not a replacement for the tracer's execution engine or a JavaScript reference oracle.

## Convert a diagnostic corpus

Generate examples in `tracing-js-self`:

```sh
npm ci
npm run training:generate -- --seed 42 --count 20 \
  --output training-local.jsonl --manifest training-manifest.json
```

On this repository's checkout, convert with a pinned raw-byte reference codec:

```sh
python scripts/import_semantic_execution.py \
  --input /path/to/training-local.jsonl \
  --output /path/to/semantic-episodes.jsonl \
  --tokenizer-json /path/to/pinned/tokenizer.json --allow-local
```

For parser-only tests, use `--diagnostic-byte` instead of
`--tokenizer-json`. A debug byte codec is *not* the reference production codec.

A native Episode contains one `PhysicalRecord` per trace group, explicit
`Meridian` and `TrackEvent` identities, one of the fixed eight tracks,
`Dependency` edges between committed producers and consumers, and
`LogicalAction`/`result_of` IDs for eligible request/observation pairs.
Only the two initial observed records, previous committed generated records,
and later legal observations may be used as conditioning. The JavaScript
reference result, full journal, arrival buffers, archive hashes and private
`sidecar` never become `PhysicalRecord` bytes, controls or the episode prefix.

## Prepare as native megatoken data

The converter outputs native Episode schema-v3 JSONL. In a pinned local source
manifest for `megatoken prepare`, declare its adapter as `native`, stream as
`semantic_execution`, and set `options.preserve_native_stream=true`. Set an
immutable source revision and the actual file SHA256; record the true license
and required human source-terms approval. The native preparation adapter
preserves exact trace records, meridians, dependencies, physical loss masks,
and supervision; it must not rechunk or silently promote local diagnostics.

Training may then select the prepared dataset with `use_stage_mixture=false`
(the existing stage mixture does **not** assign a sampling weight to this
experimental stream). The default verification gate refuses `locally_verified`
episodes. For isolated diagnostic trainer tests only, opt into
`allow_unverified_data=true`; this is not an admission claim. For a
production verified stream, a separate independent admission certifier,
operation-level evidence, and free-running-model correctness checks are still
required.

## Complete host requests and fragmented causal records

The initial supported external codec is `canonical-json/1` with the exact
`demo.readAsync/1`, `demo.readNumber/1`, and `demo.store/1` signatures.
A request record carries `codec`, `signature`, `operation`, `payload`
(the complete serialized JSON arguments), and `arguments` (the matching
materialized array). Requests with inconsistent bytes, unknown schema versions,
unresolved values, duplicate effects, or an observation without a preceding
committed request are rejected.

A request or source can span physical `value-open`, `value-part`, and
`value-close` chunks. Import preserves one model meridian per physical
chunk, groups their `PhysicalRecord` parts under one logical identity, and
uses the final close event as the producer. Every part of a fragmented request
belongs to the same `LogicalAction`, with its hash over the complete original
physical bytes; a delivered environment result refers to the completed action
and its execution ID. Observed source and input fragments stay user-context
records even if they occupy more than two meridians.

The dedicated [pinned tracer contract workflow](../.github/workflows/semantic-contract.yml)
generates real training-js/1 traces from a fixed tracer commit, converts them
through this adapter, and verifies exact record payload bytes, native
meridians, and `admitted=0`. It uses a byte-fallback debug codec, so passing
this workflow is a schema/visibility regression check, **not** proof of actual
production-codec fit, JavaScript conformance, or learned-model correctness.

## Guarantees and gates

- Strict validation rejects unknown grammar versions, improper loss masks,
  mixed/duplicate tracks, same-group/future dependencies, fragmented-value
  interleaving, action-result mismatch, and output chunks over the actual
  pinned codec's `max_positions`. The adapter does not clip payloads.
- A fragment is not available to a dependent group until its closing marker
  has committed. Source and initial graph are observed, and later trace
  decisions are generated.
- `semantic_execution` is a verification-required training stream. The
  trainer's explicit teacher-conditioned mode scores P on the prior committed
  meridians and then uses the **recorded target representation** for the next
  demonstration prediction. This is intentionally distinct from sampled
  free-running continuations.
- An unadmitted tracer artifact is refused by default and requires the
  explicit `allow_local=True` diagnostic switch. Even the label
  `admitted` cannot set a success supervision claim unless an independent
  `admission_verifier` callback approves the export. The CLI has no such
  callback: its outputs always remain unadmitted.
- Full free-running serving correctness still requires the tracer's
  generated-state controller, reference comparison and actual model rollout
  harness. The adapter does not run JavaScript, compute a correct branch,
  repair model-generated values or infer training improvements from self-replay.

These are bounded initial integration contracts. Broad async/host semantics,
code-on-call, reference-codec-aware source bundling, finite-context continuation
packets, and all additional corpus admission gates remain controlled by
`tracing-js-self`'s implementation ledger. Do not train an unadmitted corpus
as a verified-success stream merely because this adapter accepted its schema.
