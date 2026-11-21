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

Host-owned closure handlers without a guest owner resolve and validate their
retained function before an explicit unsupported-profile refusal. Their full
registration, event delivery, parentless callback, and close lifecycle is
separate work; the ordinary returning callback examples above do not depend on
that profile. Module initialization, original-source certification of arbitrary
retained sessions, and model execution remain separate gates.

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

The retained fixture contains three selected-control episodes, 26 generated
records in total: repeated closure calls followed by an ordinary retained call,
a callback whose identity contains no digits, and two independent fresh realms.
They are generated and replayed by clean producer
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
