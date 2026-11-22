# Rejected model fragments retain their exact bytes

The original actual step-348 evaluation, producer run
<https://github.com/werg/tracing-js-self/actions/runs/45046754653>, produced no
complete source programs. Eight of its 16 cases also exposed an evidence bug:
the native exception retained an invalid final fragment and its token IDs, but
the aggregate payload stopped before that fragment. The producer correctly
refused to authenticate an aggregate that disagreed with the retained bytes.

Native source `6d5ea69109ce26c2baa04a24575f675174066480`, based on
`cdd01ac0dcd65d92e948cec39f7b5c156800206e`, records original raw bytes and explicit
shape/runtime/token/UTF-8 validation facts before rejection. A runtime-invalid
UTF-8 fragment can retain aggregate provenance only when its single expected
text output and original token IDs reproduce exactly the retained bytes. A
malformed output count, nonbyte payload, wrong track, invalid token or token/byte
disagreement clears the aggregate authority, retains the raw rejected native
output arrays, and remains rejected. No bytes are repaired or substituted.

The patch preserves the failed branch, its original token counts and its lack
of acknowledgement. It neither commits a record nor grants source admission.
An invalid sequence can have measured model-emission provenance while still
failing the execution grammar. The original evaluation's eight unvalidated
selections remain unvalidated in their immutable reports.

## Bounded verification

- **17 native test cases passed**, zero failures/errors/skips, in 26.95 seconds.
  These include first and later invalid fragments, eight malformed/disagreement
  conditions, and all eight retained invalid-byte cases from the original run.
  Existing original-output, explicit commit, retry, callback, terminal close,
  complete acknowledgement, training/serving-prefix, and real runtime controls
  remained in the same file.
- **Two existing producer boundary controls passed**, zero failures/errors/skips,
  at original producer `9cf091e6ccc5add960ef0897576d0c391b199a65`. They check that
  authenticated partial binary failures retain their measured cost without
  inventing a complete UTF-8 record or successful source execution, and that
  unverified or mismatched identities cannot gain model qualification.
- Ruff **0.17.6** passed `check src tests scripts` and formatting validation for
  all **626 Python files** in that scope.

`native-controls.xml`, `frozen-producer-controls.xml`, `proof.json` and the Ruff
outputs retain these exact observations. The original eight-case fixture keeps
its old original token IDs and raw bytes. The boundary regression deliberately
uses the existing explicit fake runtime and byte-fallback codec; its test IDs
are not relabelled as the original model's IDs. The separate producer audit
verified all original token IDs against the real retained tokenizer. These are
engineering controls, not a new model-quality measurement or the full paired
release gate.

The original checkpoint's training lineage and contract remain unchanged.
Running its model with this serving correction requires a separately identified
inference revision compatible with the old contract. Current broader language
code must not bypass an old contract mismatch. Any follow-up uses the same full
16-case population and budget and retains these original failures separately.
