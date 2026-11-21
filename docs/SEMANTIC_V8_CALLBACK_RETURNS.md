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

The supported retained callback activation is `activation:"callback"`. A
closure-backed `activation:"call"` first checks immutable function identity,
then refuses the unimplemented retained-call profile instead of silently
discarding captured cells and heap state. Host-owned closure handlers without a
guest owner similarly resolve and validate their retained function before an
explicit unsupported-profile refusal. Their full registration, event delivery,
parentless callback, and close lifecycle is separate work; the ordinary
returning callback examples above do not depend on that profile.
