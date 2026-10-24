# Semantic execution: version-three native contract

The **only supported** public execution format is `semantic-execution/3`, with
`semantic-trace/3` and `semantic-context/3`. Private supervision uses
`semantic-supervision/2`. Source-language support (`training-js/1`) and
input/capability admission remain separate contracts. Old version-two episodes
are rejected; there is no fallback parser, converter, or dual writer.

`scripts/import_semantic_execution.py` checks typed public records, declared
observed/generated boundaries, closed fragments, deterministic external request
arguments, existing observation associations, terminal completion, raw codec
capacity, and episode budgets. It produces native `Episode` physical records
with the exact input UTF-8 payload and policy loss masks, without importing the
tracer's private sidecar. The import is **diagnostic only** unless an authorized
independent admission verifier accepts the source execution receipts.

## Local contract

```sh
python -m pytest -q tests/test_semantic_execution.py
python scripts/import_semantic_execution.py \
  --input /tmp/witness-v3.jsonl --output /tmp/native-v3.jsonl \
  --diagnostic-byte --allow-local
```

Use the pinned production tokenizer for actual token-position accounting, not
the byte fallback. Native `Episode` windows preserve observed data as
non-policy input and generated complete records as policy targets. Fragment
frames do not become additional guest semantics.

The source of truth for selected execution and witness arithmetic consistency
is the paired tracer's typed `SemanticController`. The native importer does not
invent a completed result from a partial selected rollout. Planned full
model-selected recurrence, continuation, and paired model comparisons are not
equivalent to a passed JSON import.

See the paired `werg/tracing-js-self` redesign plan and draft PR #26.
