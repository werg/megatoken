# ADR 0004: reuse central processing for contextual realization

Status: superseded by [ADR 0007](0007-direct-realization.md) for architecture v8.

Historical context follows; it is not the current execution contract.

## Decision

Both byte ingestion and next-chunk prediction produce the same 32-facet input
format. A selected raw tensor is preserved exactly. The central processor then
reads that tensor and prior history, performs four residual stages, and writes
one projected KV entry. Its early and final activations are cached for both the
realizer and the next prediction. They are derived activations, not a second
independent recurrent state.

The normal realizer consumes the raw facets and those contextual activations.
It starts with all 32 facet identities, expands output positions, and incorporates
the later core features during refinement. It does not re-run the central core.
The next predictor starts with a non-residual slot/channel remix of the processed
state and performs twelve residual sparse stages before producing the next
Gaussian mean. The only served random choice is the draw over that complete
32-facet prediction. Prospective control enters prediction, not the historical
processing of an already selected input.

Ingestion retains a narrow byte/ngram hierarchy and adds two dense refinement
blocks at facet resolution. The production geometry stays 32 x 512, with
32 x 256 as the smaller preset. Packing originally used 256 reference-codec tokens with a soft limit of 512.
[ADR 0005](0005-recursive-parser-chunking.md) supersedes those preparation values
with a 512-token target and 768-token soft limit; the hard output capacity remains 1,024.

## Why

Realization is normally followed by another recurrent step. Reusing central
processing avoids discarding useful contextual computation or duplicating it
inside a separate decoder tower. The earlier implementation expanded outputs
before the final prediction stage and fused late core features; version 3 moved
all realization after the final prediction and removed its learned source-feature
path. Version 4 restores a useful contextual readout without restoring private
carry, decoded-byte reconciliation, multiple sampled control decisions, or a
separate learned observation writer.

The state boundary and the decoder boundary need not coincide. The recurrent
input is still the exact selected raw tensor. The effective deterministic
realizer is a function of that tensor and its causal history through the shared
central processor. Public schema and exact-copy inputs remain explicit.

## Execution and memory

Observed bytes are encoded and processed immediately. Generated raw facets are
also processed immediately, before contextual decoding, and that processed
result is returned as the next state. The next predictor reuses it. No deferred
input is silently consumed by a subsequent observation. Pure prediction does not
write history. Alternate teacher-forcing or reconstruction branches fork from
the same pre-action state and never append hypothetical samples to the live
trajectory.

The core retains one bounded projected KV history. Early/final feature caches are
invalidated with the model version and reconstructed from ordered observations
and native raw-facet tapes. The implementation reuses those activations; it does
not claim concurrent GPU execution merely because two branches have independent
work. CUDA overlap is a separate measured optimization.

## Training ownership

Bootstrap combines noisy direct ingestion-to-realizer reconstruction, causal
prediction-to-encoder distribution matching, and output supervision through the
contextual realizer. The encoder target is a diagonal Gaussian with controlled
positive noise, so the correspondence KL includes its variance rather than
fitting only a noiseless point.

Recurrent training feeds selected raw predictions forward through live towers.
It trains ingestion through a separate reconstruction branch using the same
contextual processing/readout with their parameters held fixed for that branch.
Gradients through the branch's input still train the encoder. Native facet
following is permitted only for a saved raw tensor paired with the exact bytes
it produced; unrelated observed bytes are not regressed toward a causal forecast.
The encoder's losses cannot change central, predictor, memory, or realizer
parameters through hidden shared owners or gradient clipping.

For latent-action outcome RL, the complete conditional decoding map must be
fixed. Freeze ingestion, central processing, its KV/history projections, and
realization. The predictor, its remix, controls, and Gaussian scale remain
trainable. Consequently the small dense central processor has independent
parameter ownership from the large sparse prediction bank. Operator code remains
shared; weight tying is used only where coupled training is intended.

## Consequences

- The raw state and the completed prediction remain directly comparable.
- Central processing and KV writing work without any byte ingestion or output.
- The last output still needs central processing because its contextual readout
  uses those features, even if there will be no subsequent prediction.
- A raw-only decoder entry remains available for bootstrap and explicit
  inspection. It is not a substitute for the normal contextual serving path.
- Output replay requires the correct causal history as well as the raw sample.
- Version 3 checkpoints and tapes are not silently reinterpreted as version 4.
- Numerical correctness, execution costs, reconstruction, and recurrent quality
  have separate validation. CPU checks do not establish Blackwell throughput or
  production model quality.
