# Native replay of returning host callbacks

The native V8 request controller now recognizes `owned-graph/1` only when the
pinned host-operation declaration selects that codec. It passes the current
selected heap to the independent host materializer, which checks the operation's
arity, Promise return, payload budget, canonical bytes, complete reachable object
snapshot, and permission to carry callback markers. A closure marker must name
an existing guest closure; a data node cannot impersonate one. Stale or missing
objects, unknown callbacks, custom prototypes, accessors, and graph arguments
sent to the existing JSON-only request path are refused before effects or
dispatch intents are added.

The materializer uses the neutral typed value and graph schema. Native V8 no
longer reaches the legacy V6 witness module through host operations. The absence
test now blocks that module as well as the older V6/V7 wire modules.

## Retained source examples

`tests/fixtures/semantic-callback-return-v8.json.gz.b64` retains the public source
publication and causal contexts of four ordinary original-source runs:

| Example | Generated records | Behavior checked independently by native replay |
| --- | ---: | --- |
| Async callback | 16 | Child Promise completion before the parent receives its result |
| Shared receiver and argument aliases | 33 | Receiver identity, aliased arguments, and retained captured cells |
| Reentrant host callback | 17 | A callback requests another host callback before its parent resolves |
| Throwing callback | 8 | Guest throw reaches the rejected parent result |

All 74 records replay against their exact earlier contexts, including native
checkpoint reconstruction. Tests also reconstruct the final child and parent
prediction boundaries without giving the verifier a future target. These are
public semantic replay checks; tokenizer/mask/source admission and learned model
quality remain separate gates. The pre-fix consumer refused all four examples at
their first owned graph request.

The observed result validator accepts the producer's existing
`guest-callback-throw` provenance only for rejected results. Fulfilled results
still cannot carry an error origin.

## Explicit remaining boundaries

Module initialization, original-source certification of arbitrary retained
sessions, recovery without subsequently retained public observations, and model
execution remain separate gates. A parentless callback requires an independently
observed handler event. The controls below do not promote selected records to
original-source or training admission.

## Retained call and callback initialization

Native replay now initializes ordinary retained calls and closure-backed calls
from the independently reconstructed realm. Heap objects, captured cells,
scheduler counters and deterministic random state survive across calls. Each
invocation gets its own frames, effects, jobs, parameter cells and completion.
Fresh sibling realms retain no cells from one another. Parameter cells use the
session's actual invocation order, including fresh sibling realms, instead of
guessing an ordinal from the spelling of a callback identifier. A source-site
entry and its unambiguous declared function name initialize the same parameters.

Before input import, replay checks the realm policy, source revision, random
seed, task revision, running turn, parent identity and retained function site.
Input graphs cannot overwrite a retained object or introduce unknown references,
and receivers must refer to a retained or newly imported object. A missing
captured cell refuses instead of inventing a binding.

The first retained fixture contained three selected-control episodes, 26
generated records in total: repeated closure calls followed by an ordinary
retained call, a callback whose identity contains no digits, and two independent
fresh realms. They were generated and replayed by clean producer
`1965c3767cfb4ff1719c90ecf3a0d71ff3142e39`, including JSON checkpoint restoration.
Native tests check every earlier prediction boundary and six malformed admission
variants. These controls are **selected public replay engineering evidence**, not
original-source or tokenizer admission. Their receipt keeps `admitted:false` and
`sourceBinding:"untested"`.

Reproduce the fixture with:

```sh
node scripts/generate_retained_invocations_v8.mjs /path/to/clean/tracing-js-self
```

Validation for this increment: 11 retained-invocation controls pass, together
with 65 existing callback identity/return, detailed-state, mixed host/Promise
and realm-scheduler controls. These are separate selected test sets; neither
count establishes a complete language or release gate.

## Host-owned event handlers and exact delivery authority

Native replay now supports host-owned handlers independently of guest host
requests. Registration creates a handler obligation without fabricating a guest
dispatch or callback capability. A static handler may be registered before any
invocation; its arrivals may use fresh or retained realms. A retained closure
handler preserves the closure's code and captured cells. A handler's lifetime
owner alone does not make it a guest callback registration: an externally
registered handler can have a live owning invocation and use a fresh event
realm. An actual pending guest callback capability still requires its unique
dispatched request and exact registration identity.

Every event is checked against the registered program, track, derived realm,
trigger and next available event invocation identity before it marks delivery.
The next public observation must activate that exact callback, preserving its
entry, closure, parent, task revision, receiver, bindings and ingress key. It
cannot substitute an ordinary call or inject extra argument aliases. Host
delivery never interrupts a running realm turn; closed subscriptions and
duplicate event identities refuse.

Closing a host-owned handler needs no guest callback to revoke. Queued inputs
stay private until delivery or retirement, so the close receipt first publishes
the discarded event identities. Native replay adds each unique, previously
unseen discarded identity to the closed handler and never executes those
inputs. Existing delivered identities cannot be reused as discarded inputs.

The first host-handler increment pinned producer
`4bff56f65f452fa659e4fdccc451269b71c0114d`. It contained **seven selected-control
episodes and 52 generated records**, including four new host-handler episodes:
retained closure delivery, static fresh/retained delivery, and actual buffered
inputs retired before a later ordinary call. The generator checks producer
public replay and JSON checkpoint restoration for every episode. All prediction
boundaries are then reconstructed independently by Python native replay. These
remain selected-public controls, with source and tokenizer admission untested.

Earlier malformed controls assumed that a handler close could be observed on a
worker track, or that an event activation could add argument aliases. The
producer always emits host-owned close receipts and exact event activations.
The positive controls now follow those real routes; separate negative controls
retain the previous mutations as refusals.

Validation of the integrated host-handler increment: **103 tests pass** across
host handlers, retained invocations, callback identity/returns/boundaries,
observation transactions, recovery and realm scheduling. Ruff and whitespace
checks pass. No external adapter, optimizer or model decoder runs in these
selected-control tests.

## External effect reconciliation and subscription reattachment

The native consumer now reconstructs the public effect recovery receipts. A
receipt requires the exact outstanding effect, invocation and track, plus a
nonempty external evidence identifier. A fulfilled or rejected disposition must
be followed immediately by the matching observed result. No generated prediction
may occur between those two observations. Result delivery keeps the normal
graph, Promise and callback-retirement checks.

A `not-dispatched` disposition returns the request to eligibility while retaining
its existing attempt count. The next dispatch must use exactly the next attempt;
reusing or skipping a number refuses. A `lost` disposition leaves the result
unresolved, blocks automatic retry and refuses a later invented result. Rejected
observations leave all state, outstanding dispatches and obligations unchanged.

An evidence identifier is an opaque reference supplied by the embedding. The
consumer checks its presence and causal effect binding; it does not authenticate
the external service's claim. The same identifier may refer to independently
verified batch evidence, so the consumer does not invent a uniqueness rule.

Reattachment requires the exact retained handler, track and realm, with explicit
evidence. The public history may still show the handler as active when the
private native handle was lost; that is a valid restoration boundary. Closing
or closed handlers and revoked guest callback capabilities cannot be revived.
Reattachment does not issue a new guest request, create a dispatch or repeat the
original registration.

`scripts/generate_recovery_v8.mjs` retains twelve producer-generated recovery
and callback-registration controls, with 54 selected records. These include
fulfilled and rejected recovery, an explicitly
unresolved lost result, verified non-dispatch followed by attempt 2, and an
existing subscription reattached after JSON restoration, plus a returning owned
callback restored from an independently read durable input journal. The first three start
with one actual controlled adapter call, then prove recovery does not call it
again. The retry control imports the documented boundary before the adapter;
only its verified second attempt makes one actual adapter call. Subscription
recovery registers once and reattaches once, then delivers two callbacks without
repeating registration. Every context and prediction boundary is independently
reconstructed in Python, including its saved continuation.

The returning callback control writes one real durable arrival while the parent
is suspended, restores the queue with a new journal reader, and restores again
between the child's return and final completion. Two arguments and the receiver
retain the same cyclic object, the input retains its value captured before a
later external mutation, and the closure retains its earlier captured cell.
Restoration does not recreate a dead native adapter continuation. Completing the
child still leaves the parent effect unresolved; only a separate external
reconciliation receipt supplies the parent outcome. A third independent-reader
restore checks the completed result. The adapter is called once throughout.

Native replay also refuses canonical native callback event counters above the
JavaScript safe integer limit. Oversized decimal names cannot silently avoid the
counter check or alter the handler's seen events.

The retained-handler controls now also include a nonnull lifetime owner with
fresh event realms: eight selected episodes, 58 records in total. This guards
against confusing lifetime ownership with a guest registration effect. The
two current fixtures were regenerated from clean producer
`9fc193538c4e5c2e512f9fef767d30ed9e437c5a`, with checked producer replay and
JSON restoration before export. The
recovery and retained controls remain selected public replay evidence;
`admitted:false`, untested source binding, masks and tokenizer are preserved.

Reproduce the recovery controls with:

```sh
node scripts/generate_recovery_v8.mjs /path/to/clean/tracing-js-self
```

Validation of the integrated recovery increment: **178 tests pass** across
external recovery, handlers, retained calls, callback results and identity,
observation transactions, cancellation, realm scheduling and mixed host/Promise
composition. The regenerated fixtures also pass the 121-test focused subset;
these overlapping counts are not additive. Ruff and whitespace checks pass.

The durable owned-return follow-up passes **109 tests** covering recovery,
handlers, callback results and retained invocations, including the new cyclic
receiver/input alias checks, missing-parent-result check, unknown returned
identity refusal and two callback event counter overflow cases. This focused
count overlaps the earlier integrated run and does not broaden admission claims.

## Exact callback registration and retirement after a late result

Results and fulfilled/rejected reconciliation now require the selected callback
registration's observed handler. A durable dispatch receipt alone cannot invent
that registration. Resultless `not-dispatched` or `lost` evidence may still apply
before registration, so a verified pre-adapter failure can remain eligible for
an authorized retry. Malformed result authority refuses before state changes.

Registration binds the exact subscription identity, selected entry, owner,
realm, track, closure, lifetime and queue limit. Two simultaneous requests may
register the same closure under separate subscription identities. A separate
host handler using that closure does not satisfy either guest registration. This
also avoids treating an unrelated host registration as a pending guest request
merely because they share an owner and closure.

Notification subscription numbering comes only from selected notification
requests. Ordinary invocation names and returning callbacks do not advance it.
The fixture includes two ordinary calls whose names resemble automatically
assigned callback names, and a returning callback that performs its own listener
registration. Both adapters in the latter case run once, and producer replay,
restoration and native replay agree on the separate identities.

A late result can settle its effect without reviving a revoked subscription or
reopening a closed handler. Fulfilled subscription results preserve lost private
handles until explicit reattachment. Failed registration and call-lifetime
completion retire their capabilities; closing and closed remain distinct. A
real controlled adapter result arriving after explicit retirement is included,
along with refusal and atomicity tests for missing registration evidence.

The consumer's hosted workflow now pins the same producer revision as these
fixtures. The shared contract compares byte-for-byte at that exact pair; earlier
hosted failures against the stale producer pin do not establish a semantic
failure of the updated pair.

Validation for this increment: **172 focused tests pass** across recovery,
handler ownership, retained calls, callback results and callback identity.
The shared contract comparison, Ruff and whitespace checks pass. These counts
overlap earlier runs and remain engineering evidence rather than source or
model-quality admission.

## Native action receipts preserve their exact selected request

An optional native action receipt on a dispatch now binds its invocation,
effect, operation, payload, owner track and selected-record digest to the
independently reconstructed request. Its action key must be a nonempty string.
An explicit null receipt, missing binding, nonfinite value, negative zero,
cyclic object or excessive nesting refuses before dispatch or state mutation.

Replay stores the complete plain JSON receipt on the intent, including extension
fields such as a signature, model track or external authority metadata. Those
fields are retained as observed data: replay does not authenticate a signature
or establish that a model produced the request. The embedding remains
responsible for that evidence. This fixes continuation contexts that previously
lost a real dispatch's native action metadata.

The added producer-pinned control obtains its receipt through the actual
before-dispatch hook, verifies that the adapter receives its action key, and
checks public replay and JSON restoration. Independent tests cover altered or
missing bindings, malformed receipts, extension preservation and context
substitution. The receipt and artifact remain explicitly source-uncertified.

Validation of the native action increment: **128 focused tests pass**, including
34 action-receipt controls, recovery and handler ownership tests. Ruff and
whitespace checks pass. The full hosted test selection is run separately against
the exact published producer/consumer pair with the pinned training dependencies
and tokenizer.

## JSON value types remain exact across replay and saved checkpoints

Authority comparisons now use the existing ECMAScript-compatible canonical JSON
representation. Python considers `true` equal to `1` and `false` equal to `0`
inside dictionaries, which previously let a context or rehashed checkpoint
replace observed native action metadata without detection. Committed contexts,
prediction contexts and restored checkpoints now distinguish those JSON types.
The same checks cover their saved state, region cursor and obligations; a
checkpoint's own integrity hash does not replace independent prefix authority.

The controls first publish a valid dispatch with matching metadata and recompute
its context prefix hashes, then change only the submitted context or checkpoint.
All 16 boolean/number substitutions across the four receipt boundaries were
incorrectly accepted before the fix. They now refuse without changing the
submitted data. Eight positive controls preserve ECMAScript number equivalence:
Python representations `1` and `1.0` describe the same JSON number and remain
interchangeable. These checks continue to preserve metadata, without claiming
that its signature or external authority is authentic.

Validation of this runtime increment: **170 focused tests pass** across native
action receipts, external recovery, replay and checkpoints, prediction prefixes,
and mixed host/Promise composition. Ruff and whitespace checks pass. The full
CI selection now collects 579 tests, including the 24 new type-boundary controls.

The first full installed-dependency test run remains recorded at producer
`9fc193538c4e5c2e512f9fef767d30ed9e437c5a` and consumer
`f25fa325def722c5ec7d6ed7f9300ac235cf6e1a`: 553 of 555 tests passed, two stale
test setups failed, and none were skipped. The separate test-only correction
`a2734bba6f23f2af29f3bc698d538e75d0d52223` makes the callback counter's safe-integer
boundary explicit and reconstructs a pending guest callback before its handler
receipt. Its two affected test modules pass all 22 controls. That historical run
is not relabeled as evidence for the later runtime change; the complete test
selection must run again on its published revision.
