# Canonical V8 behavior census

The ordinary prepared-data coverage census reads the admitted V8 public artifact. It derives stage behavior labels from the actual source, invocation, and causal records. It does not grant admission to an artifact or turn a diagnostic dataset into a production dataset.

| Behavior | Required evidence |
| --- | --- |
| `semantic_code_invocation` | An observed source publication with its exact byte digest, followed by an observed invocation bound to that source and session. |
| `semantic_repeated_invocation` | Distinct root call invocations for the same published program and source digest. Callbacks and descendant call edges do not inflate this count. |
| `semantic_async` | Canonical generated detailed await and resume records linked to the same continuation, invocation, session, and track in causal order. |
| `semantic_tool_feedback` | A canonical generated detailed request, its exact observed result, and subsequent generated work on the same invocation, session, and track. |
| `semantic_source_reread` | An observed eviction after invocation, followed by an observed reread of the exact published source bytes and digest on the same session and track. |
| `semantic_child` | Still absent from ordinary V8 records. A JavaScript parent invocation ID and a detached native coordinator snapshot are insufficient. Admitted data must persist actual typed creation, join, and native completion evidence before this production requirement can be met. |

The census accepts the canonical V8 `sequence` and `track` fields. Alternate legacy field names, duplicated identities, generated source publications, substituted source bytes, cross-session evidence, and invalid ordering cannot establish coverage. Full artifact validation remains the admission authority; these conservative checks also protect direct census callers.

`tests/test_semantic_v8_behavior_census.py` retains four controls over real V8 fixtures and targeted malformed mutations. Its source-reread fixture is a direct producer `RegionExecutionHarness` export and passes independent native replay, context, and continuation checks. These are protocol and coverage controls, not production-token admission or learned-quality evidence.

The production recipes and their required behaviors remain unchanged. The next data increment must persist source-qualified native orchestration records and qualify the exact final producer/consumer/tokenizer pair. The main remaining-work ledger is [tracing-js-self PR 47](https://github.com/werg/tracing-js-self/pull/47).
