# Native commit recovery under execution ownership

This optional trusted-local Linux path retains an exact native commit request,
the exact successful inner reply bytes, and the complete adapter checkpoints
immediately before and after the commit. It covers `commit` and
`commit-orchestration`. It does not make a saved selection reusable: restoring an
uncommitted selection still abandons that candidate.

## Required ownership

Start `scripts/serve_semantic_v8.py` with all of these arguments together:

```
--request-journal-dir /private/native-journal
--execution-lock-fd 3
--execution-lock-path /private/selected-orchestration-execution-locks/KEY.lock
--execution-session-id SESSION
--execution-stream-id STREAM
--execution-session-generation 1
```

The producer acquires its stable Linux execution lock before creating the native
child and duplicates its held descriptor into child FD 3. The native process
independently verifies the session/stream filename binding, owner-private
directories, ordinary file ownership, mode, single link, stable inode and
nonblocking exclusive kernel lock. It retains a duplicate throughout its life.
Closing the parent's descriptor cannot release an outliving child's copy.
Neither side unlinks the lock file or explicitly unlocks the shared description.

Every operation through the configured JSONL dispatcher verifies ownership.
Ordinary unjournaled commit operations are refused. An uncertain durable write
fences all subsequent stateful operations in that process; metrics remain
read-only. This local exclusion applies only when every participant uses this
path. It does not provide distributed ownership or defend against a malicious
owner of the private storage directory.

## Request and result

The outer operation is `durable-commit` or `recover-commit`. Its only arguments
are `identity`, `requestBase64` and `preCheckpointDigest`, in addition to the
normal outer `id` and `op` fields.

`requestBase64` is canonical RFC 4648 base64 encoding of the exact original
single JSONL commit frame, including one trailing newline. Duplicate JSON keys,
nonfinite numbers, invalid UTF-8, multiple lines, missing newlines and unexpected
request fields fail before dispatch. The inner request's operation and session
must match the identity. The request journal's immutable identity is:

```json
{
  "schema": "semantic-native-request-identity/1",
  "session_id": "session-a",
  "stream_id": "stream-a",
  "session_generation": 1,
  "ordinal": 1,
  "operation": "commit",
  "model_identity": {"checkpoint": "SHA256", "tokenizer": "SHA256"}
}
```

A loaded training-lineage hash adds the `training` field to `model_identity`.
The identity must exactly match the loaded model and inherited ownership. The
producer's supplied checkpoint digest must match the entire native adapter
immediately before the commit, including the exact target's pending selection.
Any pending sibling selection makes this bounded recovery path unavailable.

The journal persists a request envelope containing the exact encoded frame and
precheckpoint digest before calling the ordinary native commit. Its successful
outcome contains both full checkpoints and exact inner reply bytes. The inner
reply has the usual `{id, ok: true, result}` structure and one trailing newline.
An independent durable readback is required before returning to the caller.

The returned `semantic-native-durable-commit/8` receipt contains:

| Field | Meaning |
| --- | --- |
| `identity`, `requestId`, `requestSha256` | Exact admitted slot and original frame binding |
| `preCheckpointDigest` | Producer-supplied original whole-adapter checkpoint digest |
| `postCheckpointDigest`, `postCheckpoint` | Complete native state after that exact commit |
| `responseBase64`, `responseSha256` | Exact original successful inner JSONL reply bytes |
| `recovered` | Whether a fresh adapter was independently reconstructed |
| `restoreReceipt` | The native whole-adapter restoration receipt, or null on original execution |

The slot ID is SHA-256 of canonical sorted JSON
`{schema:"semantic-native-request-slot/1",scope:S,ordinal:N}`. `S` is SHA-256 of
canonical sorted JSON `{session_id,stream_id,session_generation}`. These fields
contain only ASCII identifiers and safe integers, so ordinary producer stable
JSON and native journal canonical JSON agree. These digests provide integrity,
not signature authority.

## Recovery requirements

Recovery independently rereads the complete journal and accepts only a completed
latest slot. Any later intent or outcome, including an unknown later request,
prevents restoring an older state. A never-admitted request is not executed by
recovery; an admitted request without a known durable outcome is never retried.
The producer remains responsible for deciding the current execution generation.
The native bridge freezes that configured generation (default 1), refuses a
request from any other generation, and independently checks journal occupancy
before the first mutating RPC. If the scope already has any retained request,
even an unknown one, a fresh process must recover explicitly before it may
initialize, select or create work. A different generation must be configured
deliberately; a changed request field cannot switch generations.
Ordinary checkpoint `restore` RPC is unavailable while this bridge is configured;
the caller must use explicit `recover-commit`. The producer also refuses fresh
initialization when its journal already contains an intent for this generation.

After a successful durable commit or recovery, every later mutating application
RPC first publishes and independently rereads an immutable supersession marker.
The marker binds the original request/outcome and the next request's canonical
digest. Selection, rejection, completion, cancellation, initialization, restore,
coordinator creation and child bootstrap all invalidate the older recoverable
state before they run. Lost marker acknowledgement prevents the new mutation.
A fresh process refuses that earlier commit even if no later commit intent was
ever written. A later durable commit can establish a new eligible complete
poststate. Checkpoint, status and metrics reads do not invalidate it.

End-of-process cleanup only releases in-memory resources and emits no new
application acknowledgement. It deliberately does not supersede a retained
checkpoint, so a clean process replacement preserves the same recovery semantics
as abrupt process loss. An explicit cancellation RPC is different: it publishes
the supersession marker before changing state and cannot be undone by recovering
an older successful commit.

The saved precheckpoint must match the producer's original digest. The saved
transition must retain the complete session inventory, sibling sessions, closed
session tombstones, openings, settings, identity and random state. Its attempt
history may change only the exact selected attempt from awaiting to committed.
The exact reply must match the committed public/context state or the recorded
native coordinator decision. Thus rehashing a postcheckpoint cannot excuse
dropping a sibling or inventing a different native outcome.

A fresh entire adapter then reconstructs the saved postcheckpoint through the
ordinary native restore path. Model emissions and accepted history are checked
again; no host work is issued. The bridge compares the complete resulting
checkpoint, including siblings, cost and random state, with the retained
postcheckpoint. Same-process retry similarly compares the entire current adapter
and refuses if it has advanced. It never rolls back an active adapter.

## Limits and evidence

This establishes explicit native commit reconciliation only. It does not seal
the producer's host/history cut, automatically resume a composite execution,
retry unknown host work, transfer ownership, prove distributed atomicity, or
demonstrate learned task quality. Source/native admission and optimizer exposure
remain separate gates.

`tests/test_semantic_commit_recovery_v8.py` covers exact repeated replies, fresh
subprocess restore under an inherited descriptor, durable outcome acknowledgement
loss, commit-before-outcome failure, altered frame/model/stream/checkpoint
refusal, pending and advanced siblings, later known and unknown intents,
rehashed sibling deletion, raw JSONL framing, kernel contention/inode changes,
and actual tiny CPU native semantic/coordinator restoration. Supplied lexical
outputs and controlled lifecycle records make these engineering checks only.
