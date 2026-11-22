# Native V8 data preparation

The paired changes in [producer PR #47](https://github.com/werg/tracing-js-self/pull/47) and [native PR #38](https://github.com/werg/megatoken/pull/38) deliver a bounded V8 data-preparation pipeline. This repository supplies independent replay, original-source admission, causal-context, tokenizer/mask, split and physical-packing checks.

The producer PR links the shared data-preparation scope and all follow-ups. Current native and paired-data checks qualify their actual source pair. Existing optimizer/resume smoke tests exercise loader and serialization wiring; model performance is outside this milestone.

The regenerated primitive-conversion fixture preserves every original input, disposition, boundary and outcome. Its Promise-key capability metadata now matches the composed producer; public data and contexts are unchanged. Native admission now recognizes the already implemented `promise-adoption/1` capability while retaining unknown, duplicate and mismatched-inventory refusals. [Independent hosted regeneration](https://github.com/werg/tracing-js-self/actions/runs/45061368646).

## Native follow-ups

- [megatoken #39](https://github.com/werg/megatoken/issues/39) produces the next reproducible corpus through ordinary admission and preparation.
- [megatoken #40](https://github.com/werg/megatoken/issues/40) is dormant until data preparation/generation is complete. It is not a dependency of dataset acceptance.

Broader semantic and host profiles, future recovery and production migration are mapped in the producer's [follow-up index](https://github.com/werg/tracing-js-self/pull/47). The earlier seventeen-package program is not this PR's merge gate.

[Previous implementation ledger](https://github.com/werg/megatoken/blob/33031866fc7e2d20156a454064aeafd4ef79ccf3/docs/language-coverage-v8-cutover-status.md) retains historical results and identities. Original data, regression controls and substantive evidence remain in their existing locations.
