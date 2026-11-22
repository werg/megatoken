# Exact native commit recovery: local CPU evidence

The complete retained regression run passed **83 pytest cases and eight pytest
subtests**, with zero failures or skips. The separate focused run passed **46
cases**, with zero skips. The focused manifest overlaps the complete manifest;
these counts must not be summed as independent controls. See `scope.json` for
the exact JUnit entry count, runtime versions, command and implementation blob
identities.

The complete run used local candidate
`4117413b5b91024fb45e4eee04dc083526d0b4c0`, with its checkout clean at invocation
and all executable/test files unchanged throughout the run. This is a local
candidate identity, not a claimed hosted GitHub run. Publication is matched by
the source-file Git blob and SHA-256 map in `scope.json`. The parent branch was
the previously isolated native request/inventory foundation `dbc89dc`; the
target-inventory files are not part of this recovery implementation increment.

The new path independently binds exact request and reply bytes to the full
native checkpoints before and after a commit. Recovery requires the latest known
outcome, the same execution ownership and model, and independent reconstruction
of every retained native session. It compares unchanged sibling state and all
attempt accounting, never returns a saved selection as reusable model state,
and never dispatches host work.

The review found a second necessary boundary: after a successful commit, a later
selection, cancellation or child bootstrap may change native state without
creating another commit intent. A synced immutable supersession marker must
therefore precede every later mutating RPC. Recovery refuses an earlier commit
after that marker, even when the newer operation never received an outcome.
The native generation is fixed at startup, and existing journal occupancy blocks
fresh mutations until explicit recovery succeeds. Ordinary restore is disabled
in this bounded durable mode. Read-only requests and process resource cleanup do
not create application-level mutation acknowledgements.

The evidence includes actual tiny CPU native semantic and coordinator state
restoration, a fresh subprocess, inherited descriptor exclusion after the parent
closes every copy, unknown-outcome and lost-sync-acknowledgement controls,
rehashed sibling deletion refusal, exact JSONL frames, generation substitution,
and subsequent commit recovery after an earlier state was superseded.

The model's lexical outputs or lifecycle choices are supplied explicitly in the
recovery controls. No learned-success, source-population, training, tokenizer,
GPU, distributed ownership or automatic composite host/history recovery claim is
made. General serving, checkpoint, closed-session and orchestration regressions
are included to protect the existing opt-out path.
