# ADR 0006 — Retain full per-megatoken KV at every age

Status: **accepted**, 2026-10-18. Architecture version **6**.

## Decision

A selected megatoken produces exactly `n_facets` projected final-stage keys
and values. The central processor appends that complete block once after its
final residual stage. Every retained megatoken keeps **all 32 slots** in both
production profiles. The history limit is a strict FIFO of whole megatokens:
when the `max_history_chunks` limit is exceeded, drop the oldest complete
32-slot K/V block. No slot averaging, old-record summarization, re-projection,
or age-dependent resolution change is permitted.

The following v5 interfaces are removed rather than silently ignored:
`recent_history_chunks`, `archived_slots_per_chunk`, the optional
`history_slots < n_facets` projection, and `KVHistory.archived_chunks`.
Saved v5 configuration, cache, and checkpoint geometry must fail version
validation rather than acquire different semantics without notice.

## Rationale

A compressed historical megatoken is a different learned representation of
the same causal evidence. It makes the input to attention depend on age,
mixes originally distinct facet meanings, and adds a second history transition
that has no corresponding ordinary recurrent step. The user-specified
architecture instead needs one unambiguous full-resolution input and KV
projection for each selected megatoken. Deterministic truncation of whole
records remains an explicit finite-memory limit; it does not distort the
remaining records.

RoPE is applied once at the original megatoken index when the record is
written. Keys and values of retained records never change on later appends.
Partial episode resets preserve per-row validity masks. Training gradients
through retained KV blocks remain available within untruncated towers.

## Quantified tradeoff and gates

- Main `32x512`: 32 facets, two 128-wide KV heads, K+V, BF16 = **32 KiB**
  per megatoken; **32 MiB** at the 1,024-megatoken bound.
- Compact `32x256`: the same 32 facets and two 64-wide KV heads = **16 MiB**
  at that bound.
- The former v5 64-recent/960-archived geometry stored **3.875 MiB** in the
  main preset; this change is approximately **8.26×** its maximum historical
  KV slots and historical QK/AV arithmetic.
- C/P history attention remains deliberately restricted to the already
  specified C readers and P3/P6/P9/P12. This decision does not increase the
  number of history-reading layers.
- Test that all retained keys/values remain bitwise stable, FIFO discards
  whole records, masked reset rows cannot read past evidence, and gradient
  credit reaches early retained writers. Benchmark full-bound BF16 device fit,
  append allocations, attention latency, and long-context capability before
  launching the full production curriculum.

## Complexity audit: other non-obvious mechanisms

These existing mechanisms are **not** alternate historical archival and have
not been removed by this change. Their costs and purposes should remain
visible during the first pilot:

1. **Three temporary central KV writers.** C1–C3 each generate a 32-slot,
   separately gated short-lived KV for the following central block. These are
   discarded after the central pass; only C4 writes persistent history. They
   increase projection work and parameters. Test whether their independent
   projections beat simply attending to preceding central activations.
2. **Four designated historical reads in P.** P3/P6/P9/P12 read long KV;
   eight other predictor stages read only local facets. This saves attention
   work but is a deliberately heterogeneous routing of context. Treat the
   schedule as an explicit quality/latency choice rather than assuming all
   prediction stages can use long history.
3. **Predictive effort levels 1/2/3/6 and expert routing palettes.** These
   change the amount of sparse computation, require route bookkeeping and
   parity tests, and may complicate serving latency. Their value is
   conditional compute, not state compression.
4. **Two realizer supervision routes.** A raw-only ingestion reconstruction
   path and the ordinary contextual C→R path serve different initialization
   and recurrent objectives. Track both losses and their quality gap;
   do not mistake raw-only reconstruction for served inference quality.
5. **Bounded FIFO history and whole-episode resets.** Even without archive
   compression, a 1,024-megatoken limit drops older evidence and masked
   batched resets retain physically allocated but unreadable slots until
   eviction. These are explicit memory and batching contracts, not
   reconciliation. A ring-buffer implementation can reduce append copying
   later, but must preserve gradient and replay semantics.
6. **Hierarchical byte ingestion and deterministic typed actions.** Byte
   reduction intentionally compresses *within a chunk*, not historical
   megatokens. Schema validation, exact copies, and action tapes are
   task-facing contracts, not hidden recurrent carries.

The first four are reasonable future ablation targets. This PR removes only
the hidden full-to-coarse KV path and its optional initial slot pooling, to
keep the experiment interpretable.
