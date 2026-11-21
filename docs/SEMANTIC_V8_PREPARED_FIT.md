# Fitting an admitted V8 prefix without losing its identity

The bounded decoder experiment reads one assigned training episode from the
ordinary prepared corpus. Its checkpoint and served export now retain the same
prepared manifest identities, corpus, tokenizer, frozen source membership and
physical chunking policy. The experiment records its own selected-prefix stream
identity, including the exact target hashes, window settings and episode ID.
It does not claim that its decoder-only updates are an ordinary resumable run.

Previously, enabling the ordinary V8 policy in the fitter made checkpoint export
fail because the checkpoint did not include the required training lineage.
Constructing a fresh default chunking policy also changed the model's serving
history when the admitted records used a smaller physical token budget. The
fitter now passes the retained policy and expected identity to actual inference.

Token accounting reports the selected generated prefix separately from the full
admitted episode. A one-record diagnostic can stop after its actual decoder emits
and independently acknowledges that exact nonterminal record. The report marks
this as an accepted-prefix result; complete source execution remains false until
an explicit end is accepted. Inference receives only initial public observations,
and no expected target is supplied to selection or used to repair model output.

The focused regressions cover selected-prefix identity, token accounting,
nonterminal success classification and full checkpoint/export/reload lineage.
Structural test fixtures explicitly inject source and tokenizer verification.
These tests do not replace a command-level run with the pinned tokenizer and
fresh producer/native source admission. Generalization, full source completion,
production training and default routing remain separate gates.

## Actual accepted-state recovery

When the reloaded fitted model produces an acknowledged prefix, the diagnostic
saves the complete native model checkpoint before probing any further output.
It then starts the ordinary JSONL service in a fresh Python process with the
same served weights, tokenizer, training identity and generation limits. The
service regenerates every accepted selection from its saved random state,
compares its original token and accounting evidence, independently replays the
semantics and reconstructs the exact retained model continuation.

The collector requires the resulting checkpoint to match the original in full,
including acknowledged public history, random state, attempt budget, cache and
session evidence. For a nonterminal prefix it also compares the next actual
selection, including rejected output, with the uninterrupted process. This probe
is retained and measured but never acknowledged. A restore that performs host
work, omits an accepted selection or changes the next output remains failed.

All requests, original responses, standard output, standard error and the final
qualification summary are retained. Tests of the collector explicitly inject
its process boundary; the fitted command itself uses the unmodified service and
actual exported tensors. An accepted-prefix replay remains an in-sample recovery
diagnostic. Original-source correspondence must be checked separately by the
producer, and completing an entire program or generalizing to held-out sources
is not implied.
