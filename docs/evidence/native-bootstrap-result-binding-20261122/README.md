# Native durable bootstrap result binding

This correction is based on the exact native bootstrap candidate tree `d7c33b85315c10ba095819adf1b0c58420bab4d5` (published source commit `97cfd78c886e039b9cff6b9e677ec01265896235`). The local source commit was `2843e9f02e32d4a8103f42c516c05f0e2463351b`. The correction is confined to `src/megatoken/semantic_commit_recovery_v8.py` and `tests/test_semantic_commit_recovery_v8.py`.

## Reproduced gap and correction

The completed bootstrap slot was rehashed after changing the stored response child's `parent`, execution invocation ID, or lifecycle `status`, while leaving the native postcheckpoint unchanged. A second call to `durable_bootstrap_child()` returned the altered response together with the unchanged postcheckpoint. The validator compared only address and `bootstrap_ready` against the response itself.

The corrected bridge requires the exact appended bootstrap operation shape and binds its `randomBefore` to the precheckpoint's random state. It checks the response track against the original create track, allowing only `bootstrap_ready` and `status` to change. For an already completed slot it also compares the returned track with the live native track registry. Direct bridge calls now check process fencing before returning a completed-slot receipt.

Focused adversarial controls rehash the response mutations and also tamper with `randomBefore`; a direct fenced duplicate-read control checks the method boundary. The original candidate is retained unchanged; this correction is a separate branch.

## Qualification

Command:

```sh
PYTHONPATH=src:. /workspace/scratch/05d92519587f/native-recovery-venv/bin/python -m pytest -q \
  tests/test_semantic_commit_recovery_v8.py \
  tests/test_semantic_request_journal_v8.py \
  --junitxml=docs/evidence/native-bootstrap-result-binding-20261122/junit.xml
```

Result: 59 tests passed, 0 failed, 0 errors, 0 skipped in 110.80 seconds. Runtime was Python 3.12.14, pytest 9.2.0, and torch 2.14.1+cpu. The repository declares pytest below 9; this run used the preexisting recovery environment with pytest 9.2.0, so it does not qualify the locked hosted runtime. See `raw.log` and `junit.xml` for the complete output.

This is a native journal/bridge correctness regression suite, not a hosted paired whole-driver gate and not evidence of distributed lease takeover or unknown remote outcome recovery.
