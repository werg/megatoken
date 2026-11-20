# Live native V8 model-session recovery

The V8 JSONL service now accepts `checkpoint` and `restore` operations. These are
live inference checkpoints, separate from optimizer/training checkpoints. The
service remains explicitly selected; this increment does not switch the default
semantic service or qualify a production release.

## Saved authority

A checkpoint covers the whole model adapter because its random stream and
consumed attempt allowance are shared across its sessions. It binds the loaded
model, tokenizer, semantic contract, generation settings, runtime configuration,
complete acknowledged public histories and contexts, observation-ingestion
cursors, completed invocations, original generated fragments, and failed attempts.
It retains Python, NumPy, CPU Torch and available CUDA random-generator states.

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

One test uses the actual tiny CPU model and verifies that a fresh process restores
its committed initial cache and random stream and emits the same next tokens as
an uninterrupted process. Those random outputs are not labeled successful
semantic execution. Accepted-record and retained-callback recovery controls use
an explicitly supplied decoder. Qualification with a trained model, production
tokenizer/source assets and real external host recovery is still required.
