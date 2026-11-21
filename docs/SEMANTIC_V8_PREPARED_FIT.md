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
