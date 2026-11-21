# Live native V8 model-session recovery

The V8 JSONL service now accepts `checkpoint` and `restore` operations. These are
live inference checkpoints, separate from optimizer/training checkpoints. The
service remains explicitly selected; this increment does not switch the default
semantic service or qualify a production release.

## Saved authority

A checkpoint covers the whole model adapter because its random stream and
consumed attempt allowance are shared across its sessions. It binds the loaded
model, tokenizer, ordinary training lineage, semantic contract, generation settings, runtime configuration,
complete acknowledged public histories and contexts, observation-ingestion
cursors, completed invocations, original generated fragments, and failed attempts.
It retains Python, NumPy, CPU Torch and available CUDA random-generator states.

The checkpoint records active sessions in `sessions` and closed sessions in
`closedSessions`. A closed session that was opened keeps its final acknowledged
history, replay entries, continuation fingerprint, completion markers, and
original opening receipt. Restore processes sessions in opening order, replays
each closed session's accepted selections, checks its final continuation, and
closes it before opening the next. This honors the configured active-session
limit while preserving closed-session attempts and costs. A close issued before
that session opened creates a tombstone with no model history or fabricated
opening cost. Closed IDs remain unavailable after restore; repeating `cancel`
is idempotent and does not reopen or reset a session. If building a closed cut
fails, the candidate and session remain untouched. The global attempt sequence
and its consumed limit span both active and closed sessions.

Opening receipts use ECMAScript number equivalence: JSON `1` and `1.0` preserve
the same receipt identity. Metadata token and byte counts must still be
nonnegative safe integers; booleans, fractional counts, and negative or nonfinite
opening times refuse recovery before native model state is allocated.

Only committed model continuations are saved. An outstanding candidate remains in
attempt evidence and consumes its original allowance, but its speculative state
is omitted. Restore marks it abandoned; it cannot become an acknowledged action.
The next prediction receives a new selection identity. Retained child invocations
and callbacks share their original accepted history and completed-invocation set.

## Reconstruction and verification

Restore requires a fresh empty runtime with exactly matching model/tokenizer,
semantic contract and generation/runtime settings. It opens each original session
under its recorded random state. For every acknowledged selection it reconstructs
the exact earlier public prefix, runs the actual loaded model under the recorded
random state, and compares the original token IDs, fragment boundaries, bytes,
selection identity and causal identities. The ordinary native commit validator
then checks the regenerated candidate against the acknowledged public transition.

The result must reproduce a fingerprint of the complete committed model state:
selected tensors, cached keys/values, retrieval continuation and scheduler,
track budgets, retained context copies, routing, and logical event counters.
Process-local locks, memory allocation IDs and monotonic timestamps are not model
continuation identities. No pickle, executable object, or input-supplied tensor
cache is loaded. A mismatch refuses recovery and releases all newly allocated
sessions. The pre-restore random state is reinstated on failure.

After a successful reconstruction, the checkpoint's latest random state is
restored, including random work consumed by rejected or unacknowledged attempts.
Historical attempt and token accounting is preserved. The transport measures the
restore RPC's compute time separately; regenerated historical selections do not
become additional successful tasks.

## Service calls

The ordinary served bundle's validated `semantic_v8_lineage_digest` is the
expected `trainingIdentity` in initialization. The service loads this digest from
the bundle and requires the caller to supply the same value; leaving it out,
substituting a different digest or trying to reset it to null refuses before a
model session opens. The digest binds prepared source/data/policy/codec identities
to the training checkpoint and exported model resources. It is retained in every
selection and live checkpoint, and is checked against the loaded bundle before
restoration. A diagnostic bundle without ordinary V8 lineage uses explicit null.
This identity records provenance and never establishes learned execution quality.

Observation and context splitting uses the served bundle's retained physical
chunk policy. It does not expand chunks to the model's larger position capacity.
Generated fragment budgets are capped by that policy as well. Consequently a
model with 512 positions can consume data prepared in 128-token records without
changing those record boundaries or the intervening retrieval history.

The producer freezes the expected value with `--training-identity HASH` in the
experiment command, passes it through the process selector and checks both the
initialization receipt and original selection evidence. Driver recovery also
binds the entire externally expected initialization receipt.

```json
{"id":"save-1","op":"checkpoint"}
```

Persist the returned checkpoint as immutable JSON. Start the same served bundle
in a fresh process, then supply that checkpoint:

```json
{"id":"restore-1","op":"restore","checkpoint":{"...":"saved checkpoint"}}
```

The producer's process selector exposes the corresponding `checkpoint()` and
`restore({checkpoint})` operations. Its wrapper binds native state and attested
identity together with transport attempt/cost history, validates the recovery
receipt, and closes an uncertain transport if recovery is not acknowledged.
Message-size and request-time limits still apply; reconstruction costs grow with
the number of acknowledged predictions. Identical cache bits are required, so
hardware or nondeterministic execution differences may conservatively refuse a
restore instead of claiming an unproved equivalent continuation.

## Host effects and remaining integration

The native adapter executes model computations only. Recovery never calls an
external service or reissues a saved request. Public observations preserve the
native validator's effect/callback obligations. Resuming the producer's actual
host resources still requires its durable request journal, retained-handler
reattachment and external completion recovery. An uncertain dispatched effect
must not be dispatched again. Full producer-driver recovery, real host adapters,
end-to-end crash recovery, installed-package absence tests, and default-service
retirement remain separate open gates.

## Executable evidence

The serving recovery tests cover acknowledged-cache restoration, consumed limits,
failed and pending candidates, tampered/rehashed public histories and cursors,
token changes, cache mismatches, missing receipts, partial-allocation cleanup,
child/callback continuation, and JSONL recovery in a fresh process.

Multi-session controls verify that a committed session can be reexecuted while
a pending sibling is cancelled, with both attempt records and opening receipts
retained and the first session able to continue. Rehashed closed-cut changes
refuse atomically, a closed identifier cannot be selected again, and repeated
close leaves its tombstone unchanged. A separate two-session run with the actual
tiny CPU runtime samples the model's own candidate fragments and checks that the
surviving session emits the same next fragments after restore. None of those
actual-model candidates passed semantic acknowledgement; this tests lifecycle
recovery only. Positive acknowledged-record recovery still uses an explicitly
supplied decoder unless a separately pinned trained run supplies the accepted
model record.

One test uses the actual tiny CPU model and verifies that a fresh process restores
its committed initial cache and random stream and emits the same next tokens as
an uninterrupted process. Those random outputs are not labeled successful
semantic execution. Accepted-record and retained-callback recovery controls use
an explicitly supplied decoder.

The [ordinary pinned-asset lifecycle run](evidence/semantic-v8-ordinary-a4d0a56-7da2150/README.md)
also uses a real source-qualified prepared family, the actual pinned tokenizer,
two ordinary optimizer updates and a normally exported serving bundle. It verifies
expected-lineage refusal controls and exact next-token recovery in a fresh native
process. Its generated record was rejected for exhausting the fragment budget;
zero model records were acknowledged. The paired producer study retains that
failure and its measured costs. Trained accepted-record recovery, real external
host recovery and qualification of later combined revisions remain open.

The [window-parity controls](evidence/semantic-v8-window-parity-2026-11-20/README.md)
add acknowledged split-record reconstruction through the actual CPU model after
a failed candidate. All model-state fields and the next prediction match the
ordinary trainer, including across optimizer windows. Its decoder is explicitly
supplied prepared bytes; that accepted-record control remains distinct from a
trained model selecting a correct record unaided.
