# Native V8 coordinator and typed child execution

This is a bounded implementation of the existing production cutover plan's
coordinator parity work. It keeps coordinator and child work in one actual
native `TrackSession`, with separate runtime track addresses and one shared
causal model continuation. It does not change the default selected server,
assert production curriculum coverage, or issue a source-admission receipt.

## Ingress and acknowledged actions

After normal V8 model initialization, `open_orchestration(session, public,
maxActive=1, childBudget=128, origin="model")` accepts an independently checked
V8 public cut consisting only of observed program publications and declared
root call invocations. The bounded profile admits one to sixteen root proposals;
it does not admit undeclared callbacks, descendants, module initialization, or
replacement task revisions inside an existing coordinator. The ordinary V8
adapter's callback behavior remains available outside this coordinator profile.

Each declared child binds the exact source digest, invocation and input graph,
realm and completion policy, capability grants, and owning coordinator. The
coordinator has only the union of the offered program grants. Selected creation
cannot expand its grant set, substitute an input/source, change its completion
destination, reuse an invocation, or exceed the declared child and remaining
parent work budget. These observed inputs authorize the offered work; they are
not evidence that independent source admission or production tokenization ran.

The JSONL server exposes these operations, with hyphens or underscores:

| Operation | Authority and resulting behavior |
| --- | --- |
| `open-orchestration` | Check observed ingress and open the real coordinator track. |
| `select-orchestration` | Produce original coordinator bytes and selection evidence on a private native branch. |
| `commit-orchestration` | Require exact original bytes and evidence, validate the complete action, then atomically adopt the native branch and resulting typed transition. |
| `bootstrap-orchestration-child` | Host acknowledgement publishes the reserved child's exact V8 source and input before it can run. |
| `status-orchestration` | Read current typed children, dependency receipts, delivered results, publication status, and pending selection identity. |
| `cancel-orchestration` | Cancel the coordinator and live children, discard any pending candidate, retain acknowledged history, and reject further bootstrap or selection. |

A coordinator may select `track_create`, `track_join`, `track_progress`,
`track_complete`, or `track_cancel` using the existing neutral lifecycle tool
names. Creating a child returns its reserved address and exact completion
dependency. Reservation is separate from bootstrap so a crash or checkpoint
between them cannot make an unobserved child runnable. A join must refer to a
dependency issued for that coordinator's own unfinished child. With one active
slot, joining frees that slot for the bootstrapped child.

Child `select` and `commit` use the ordinary V8 semantic protocol and independent
public-prefix checks, routed to the child's actual native track. Every candidate
remains private until the exact semantic acknowledgement. There is no native
host-adapter submission in the coordinator path. The producer's separately
validated selected-effect boundary remains responsible for any external work.

## Exact result delivery

`complete` requires the acknowledged terminal guest record. The adapter
independently replays the V8 public history to derive the exact completion kind
and tagged value. A caller-supplied result must match; it cannot replace the
replayed result. Delivery also retains the replayed heap, so a reference-valued
result has its original object meaning available to the coordinator.

One typed completion receipt satisfies the issued child dependency and wakes a
waiting parent. An exact historical completion retry binds both its original public cut and
contexts. It returns the recorded receipt even after another child advances,
while another candidate is pending, or after the coordinator/session closes.
It does not reopen the session, publish another observation, wake a task or
consume another child execution. Changed cut, context or result data is refused. Progress must publish the exact
kind/value of a delivered, not-yet-published child result. Coordinator completion
requires every declared child result to have been delivered and published.

## Checkpoint and failure semantics

An orchestration checkpoint adds one ordered operation history to the existing
V8 public cut, original attempt evidence, RNG records, and complete native
continuation fingerprint. It interleaves coordinator acknowledgements,
bootstrap, guest semantic acknowledgements, exact result delivery, and
cancellation. Reserved children and parents waiting on completion are valid
checkpoint cuts.

Recovery constructs a staged fresh runtime. It reproduces the initial ingress,
reexecutes original selections, compares exact emitted bytes/token IDs and all
deterministic evidence, and acknowledges operations in the retained order.
Changed bindings, invented dependencies, changed receipts, omitted or reordered
acknowledgements, duplicate operations, and a changed final continuation are
refused before the staged runtime becomes live. Unacknowledged branches are
excluded; their consumed attempts remain in the ledger. Coordinator acknowledgements
have exact historical retry receipts. The ordinary guest semantic `commit`
keeps its existing fail-closed behavior after acknowledgement loss; this slice
does not claim that every native acknowledgement API is idempotent.

Native observation ingestion can use randomness. A coordinator's selection and
its later acknowledgement therefore retain separate RNG boundaries. Bootstrap,
completion, and cancellation also retain their own RNG cuts. Other sessions or
rejected candidates may consume randomness between these calls without making
recovery silently choose a different native continuation.

Controlled lifecycle records require the explicit constructor authority
`allow_controlled_lifecycle=True`; the ordinary server does not enable it.
Their evidence says `modelSelected: false`, and a checkpoint cannot restore
that authority into an adapter without the same explicit permission. Restoring
an accepted actual-model selection reexecutes that model; it never replaces
failed output with a controlled record.

Cancellation is a terminal admission boundary for this bounded native cohort.
It preserves earlier acknowledged work and prevents another child bootstrap or
selection. It does not undo a remote effect already admitted elsewhere, restore
remote handles, or turn an unknown effect outcome into a successful result.

## Evidence and remaining qualification

`tests/test_semantic_v8_orchestration_authority.py` runs without the ML runtime.
It checks real V8 source/input binding, independent terminal replay, parent
scope and grants, reserved readiness, exact dependencies, completion/progress,
and a fresh process with eleven legacy modules blocked. Those controls establish
protocol authority only.

`tests/test_semantic_v8_orchestration.py` requires torch and the actual shared
native recurrent runtime. It covers private candidates, exact and repeated
acknowledgements, reserved and waiting reconstruction, original child track
routing, exact reference-valued completion, terminal cancellation, rehashed
checkpoint tampering, interleaved sessions with a rejected attempt between
selection and acknowledgement, a fresh process with old clients absent, and actual model
failure bytes without fallback. Lexical child outputs and coordinator fixtures
in the successful engineering controls are explicitly supplied; they do not
prove learned task success. These tests are included in the existing hosted
`test_*v8*.py` selection and must pass there before shared-model recovery is
credited. The local data worker lacks torch; its absence is retained as a
qualification limitation, not converted into a skip.

Source-qualified lifecycle collection, real `semantic_child` census records,
repeated-invocation/source-reread coverage, full production launch evidence,
producer orchestration driver parity, dynamic descendants/callback ownership,
remote resource restoration, and default package/catalog/release cutover remain
separate mapped work. No stage requirement or behavior label is weakened by this
implementation.
