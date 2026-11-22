# Native child-bootstrap recovery qualification

This evidence records the local native recovery cohort for the published
child-bootstrap source tree. It qualifies the bounded native journal and replay
behavior exercised by these tests; it does not qualify the complete producer
driver or a distributed host/native protocol.

## Exact source

The published native candidate is commit
`97cfd78c886e039b9cff6b9e677ec01265896235`, tree
`d7c33b85315c10ba095819adf1b0c58420bab4d5`. The local tested commit is
`2843e9f02e32d4a8103f42c516c05f0e2463351b`, also tree
`d7c33b85315c10ba095819adf1b0c58420bab4d5`. Thus the tested file contents are
identical to the published commit even though its local commit metadata differs.
Both are based on native main commit `10d1f5fd611fbe4fbc3f19b6463f597d90946f7c`.

The exact four changed paths relative to that base are:

| Path | SHA-256 |
| --- | --- |
| `src/megatoken/semantic_commit_recovery_v8.py` | `4fedfebf6bd3fd9d5fccf254920576b9baa606ce7696c9ba9c43129822743c22` |
| `src/megatoken/semantic_request_journal_v8.py` | `31c481b56327873c47d5974895d39ffbe9473db8cfa32ea65c5aa119b77f59ed` |
| `scripts/serve_semantic_v8.py` | `de2c52102171d3bcda3e7d31cc613782e514d9925f8efbb33a7aae3f64b337e0` |
| `tests/test_semantic_commit_recovery_v8.py` | `6c23ff1db5d5829aad0c8f81b64cef96730a5560274c20ad7047be3e7b38eb4f` |

## Cohort and runtime

The exact command was:

```sh
PYTHONPATH=src ../native-recovery-venv/bin/python -m pytest -q \
  tests/test_semantic_commit_recovery_v8.py \
  tests/test_semantic_request_journal_v8.py \
  --junitxml=docs/evidence/native-child-bootstrap-20261122/pytest-junit.xml
```

It completed on 2026-11-22 in 82.61 seconds with **54 passed, 0 failures,
0 errors, and 0 skipped**. The raw pytest output and machine-readable JUnit
report are retained beside this document as `pytest-raw.log` and
`pytest-junit.xml`.

Runtime details: Linux 6.18.57 x86_64, Python 3.12.14, PyTorch 2.14.1+cpu,
pytest 9.2.0, and Ruff 0.17.6. The repository declares pytest `>=8.3,<9` for
development, while this environment supplied 9.2.0; the observed passing result
is reported without implying execution under the declared pytest range. Ruff
checks passed for all four changed paths, and `git diff --check` was clean.

## Focused attempts and corrected gaps

The first new lost-reply probe exposed a test assumption: the native
`track_create` record is serialized JSON text, not a mapping. The validation
was corrected to parse that exact text before checking its operation name; the
durable lost-reply and repeated-recovery probe then passed.

The adversarial review then found that configured JSONL dispatch could invoke
raw `bootstrap-orchestration-child`, superseding create recovery without a
bootstrap journal slot. Guarded sessions now refuse that route and require
`durable-bootstrap-child`. Review also found that an existing supersession
marker was accepted without checking which mutation it represented. The
journal now checks the exact mutation digest and operation, and bootstrap
causal checks require either an unsuperseded create before admission or a
create marker bound to the exact bootstrap request during recovery.

The retained cohort covers successful durable bootstrap with a lost response,
fresh-process recovery and repeat recovery, exact duplicate response, unknown
intent refusal, stale-create recovery refusal, wrong owner/model/generation/
ordinal/create-frame and forged-child refusals, a later-mutation fence,
unrelated-session mutation refusal, and independent reexecution mismatch.
Actual fixture replay preserved the complete session accounting, attempts, and
global random state. Recovery reports zero host dispatches.

## Scope limits

These are native local controls using a real native runtime with controlled
fixture output. They do not claim actual learned model generation, a hosted
whole-driver gate, or complete producer/native end-to-end qualification. The
bounded behavior remains specific to an exact committed `track_create` followed
by its immediately adjacent child-bootstrap request under the same owner,
generation, and model identity. Unknown outcomes remain non-retryable.
