# Native request journal v8

## Scope

The request journal is a POSIX local-storage primitive for recording a native request
before dispatch and retaining its exact response bytes before the caller receives them.
Each request slot is scoped by session ID, stream ID, session generation, and ordinal.
The slot also binds the operation, model identity, and exact request bytes.

A successful first execution follows this order:

1. Acquire a nonblocking exclusive kernel lock for the session/stream/generation scope.
2. Publish the immutable intent with file and directory synchronization.
3. Invoke the supplied callback once.
4. Publish the exact returned response bytes with file and directory synchronization.
5. Return those same bytes.

The lock file is stable and is never unlinked or stolen by age or PID. Directories and
records must be owned by the current user and private to that user; symlinks, changed
slot contents, malformed history, and unavailable locking fail closed. The implementation
requires POSIX `flock`.

## Readback states

- `never-admitted`: no intent exists at the requested next slot.
- `unknown`: a durable intent exists but no durable outcome was found. The callback is
  never retried for this slot.
- `completed`: exact response bytes and their digest were read back from a durable,
  immutable outcome record.

If the request identity or bytes differ for an existing slot, the call raises a conflict.
A later ordinal cannot pass an earlier unknown or incomplete request.

## Boundary

This module does not connect the journal to the native server, checkpoints, or adapter
state. A stored response is not proof that a fresh native process has the corresponding
post-request model state. It does not authorize applying an old response to a new adapter,
committing it, replaying a child action, or transferring ownership. A process that loses
its response acknowledgement can inspect the retained bytes, but the original caller's
acknowledgement is still unknown. Native dispatch and state reconciliation require a
separate protocol that binds a durable request ID to the exact post-state and prevents an
old process from continuing while a replacement is active.

The optional [native commit recovery bridge](SEMANTIC_NATIVE_COMMIT_RECOVERY_V8.md)
now supplies that separate protocol for exact commit acknowledgements under an
inherited local execution lock. Selection replies and unknown outcomes remain
ineligible. The generic journal continues to grant storage evidence only.

The subprocess tests intentionally terminate at four boundaries: before intent link,
after intent publication while the callback is running, after outcome publication but
before acknowledgement, and while a live process holds the scope lock. They verify storage
classification and duplicate-dispatch prevention only.
