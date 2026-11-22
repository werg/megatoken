# Checkpoint-bound native commit readback v8

The optional request bridge extends the durable request journal for one narrow case:
a successful native `commit` response paired with the full native post-commit
checkpoint. The server stores one canonical record containing the exact JSON reply bytes,
the request identity and digest, and the checkpoint captured immediately after commit.
The existing checkpoint format then replays committed native work through its own verifier.

A fresh process that sees a completed commit does not dispatch it again. It restores the
stored checkpoint into the already loaded native adapter, captures the reconstructed
checkpoint again, and compares every retained field before returning the exact stored
reply bytes. The reply's selection ID and public digest must match a committed attempt
and the restored public session. The selected model and tokenizer identities are checked
before dispatch and again against the checkpoint.

## Deliberate limits

This path accepts only successful `commit` RPCs. It refuses to return a stored response
if the checkpoint contains any unacknowledged selection, including pending work from
another session. A select reply cannot be recovered through this path: native checkpoint
restore deliberately abandons uncommitted selections, so the reply would describe state
that does not exist after restore. An intent without an outcome remains unknown and is
never dispatched again. A commit that was never admitted may still fail when its earlier
selection is absent; the bridge will not recreate that selection.

The request journal's completed status alone is not enough to return bytes. The stored
checkpoint must restore and re-capture exactly. This keeps response readback separate
from proof that a new adapter has the matching post-request state.

## Server launch contract

The native server can receive a branded execution lease on file descriptor 4 with
`--execution-lock-fd 4`, `--execution-lock-session SESSION`,
`--execution-lock-stream STREAM`, and `--execution-lock-path PATH`. The supervisor
must acquire the lease before spawning, pass the same open-file description on fd 4,
and retain its lease. The server opens the expected path separately to prove that it
already contends on the exact inode, then verifies that relocking the inherited descriptor
succeeds. This rejects an unlocked file and a separately opened descriptor. File
descriptor 3 remains available for process transport.

To enable native request journal readback, add `--request-journal DIR` and send
commit requests with a `nativeRequest` object using schema
`semantic-native-request-identity/1`, exact session and stream IDs, positive session
generation and ordinal, operation `commit`, and checkpoint/tokenizer model hashes
(plus a training hash when applicable). The server verifies the lock namespace and loaded
model before dispatch.

The current JavaScript native-process selector can pass an acquired lease to fd 4, but it
does not persist an exact commit frame, request ordinal, or request identity before sending.
It therefore does not enable journal recovery. The commit bridge remains an explicit
native-server capability for a caller that has independently retained and can resend the
exact request frame and identity. No generic selector retry, stale-request reconstruction,
or automatic recovery claim follows from this implementation. Unknown requests stop the
server after an explicit recovery-required response, and the selector treats that category
as an acknowledgement-unknown terminal state.

The bridge requires a trusted local POSIX filesystem and the existing verified native
checkpoint implementation. It makes no learned-quality or source-admission claim.
