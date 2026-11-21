# Native v8 model serving

`scripts/serve_semantic_v8.py` loads a served Megatoken checkpoint and asks its
real `MultiTrackRuntime` to generate one public `semantic-trace/8` record at a
time. It accepts only a committed public prefix, the earlier generated records'
contexts, and the independently reconstructed current context. It never accepts
the next expected target, source-oracle output, or private admission evidence.

```sh
PYTHONPATH=src python scripts/serve_semantic_v8.py \
  --served /path/to/served-checkpoint --device cpu --threads 1 \
  --max-tokens 512 --max-fragments 64 \
  --profile-output /path/to/new-native-v8-profile.json
```

## Request and acknowledgement protocol

The process reads one JSON object per line and writes one reply per line. Every
request needs a string `id` and an `op`. Replies preserve the ID and contain
either `{"ok":true,"result":...}` or `{"ok":false,"error":...}`. Requests are
limited to 40 MiB. Public prefixes are limited to 32 MiB and 16,384 rows. The
maximum selected record is 65,536 bytes, with at most 64 physical fragments.

| Operation | Required fields | Effect |
| --- | --- | --- |
| `initialize` | `seed`, `checkpointIdentity`, `tokenizerIdentity` | Attest the loaded checkpoint and codec. Seeds are 17, 29, or 43. |
| `select` | `session`, `public`, `contexts`, `context` | Independently replay the committed prefix, check the complete current context, and stage one generated candidate. |
| `commit` | `session`, `public`, `contexts`, `invocationId`, `record`, `selection` | Independently replay the appended candidate and advance the native continuation only if the original selection matches. |
| `reject` | `session`, `selectionId`, `reason` | Discard a candidate and permit a retry against the same public state. |
| `complete` | `session`, `public`, `contexts`, `invocationId` | Check that this invocation's explicit terminal record was acknowledged. The native session remains available for its caller and other callbacks. An optional `completion` is accepted but is not authority. |
| `cancel` | `session`, `reason` | Close the whole native session, including a pending candidate. The caller also uses this after a complete episode. The session ID stays closed and cannot be selected again; repeating close is idempotent. |
| `metrics` or `status` | none | Return measured attempt evidence and cumulative costs. |

On `select`, `contexts` contains exactly the contexts for previously committed
generated records. The separate `context` is the current prediction boundary;
it must not be included in `contexts` until `commit`. After acknowledgement,
later host observations may extend the prefix. The adapter refuses replacing
old rows, importing teacher-generated history, or selecting again while a
candidate awaits acknowledgement.

The response's `selection` binds `native-semantic-v8/1`, the checkpoint and codec
identities, session, invocation, prefix digest, current-context digest, and
original `payloadHex`. Each fragment retains its original `emittedTokenIds`
(also available as `tokenIds`), exact payload bytes, terminal flag, native
meridian, and selected event ID. The adapter does not normalize or repair model
output. Invalid UTF-8 and exhausted fragment budgets are failures with retained
emission evidence. Failed and rejected attempts remain in metrics.

Generation runs on a private branch of the native state. A successful `commit`
adopts that branch; rejection leaves the preceding acknowledged continuation
unchanged. This profile disables native memory writes and external source
collections. It preserves the loaded checkpoint's normal native transducer.
Completion is per invocation: ending a child callback does not erase its
parent's continuation or a retained handler's later callbacks.

Live checkpoints keep active model sessions under `sessions` and closed-session
records under `closedSessions`. A closed, previously opened session retains its
verified acknowledged history and opening receipt so recovery can recheck its
accepted selections without making it active again. Restore rebuilds closed
sessions in opening order and closes each before opening the next, which keeps
the configured active-session limit in force. A close before any model session
opens saves only a tombstone and creates no opening or token cost. The whole
adapter checkpoint preserves the shared attempt allowance and costs across both
groups.

A retained closure binds its invocation and handler registration to one
immutable function site in the published program. Both the requested entry
and the closure's saved code reference must resolve uniquely, and their source
starts must match. An unambiguous declared name and `function:<start>` may name
the same function; an ambiguous name is rejected even if the other spelling is
unique. Handler registration also checks that the retained realm belongs to the
selected source. These checks run before input graph import, cell allocation,
or handler publication. They prevent public observations from redirecting saved
captures to a different function; they do not expand the native module or host
callback profiles.

## Framing and measurement

V8 preparation and serving use `semantic-execution/8` session metadata and the
same `semantic-trace/8` instruction. Observed public rows and the current context
use the same canonical UTF-8 JSON and `make_records` physical splitter as
training. Published programs and invocation inputs have the user role; host
observations have the environment role; complete current contexts have the
unscored system role. The adapter checks that all physical chunks of the current
context remain visible before generation.

Metrics retain generated tokens, original fragments, observation/context tokens
and bytes, per-fragment instruction tokens and bytes, session metadata tokens
and bytes, preparation/generation wall times, and native context/cache snapshots.
A rejected attempt's repeated observations and context count again because they
were really ingested by its discarded branch. Session metadata counts once per
opened session. These token categories describe external textual framing;
internal learned controls, track-description embeddings, retrieval-query work,
and tensor operations are not converted into fictitious text tokens. Their cost
is included in the runtime's wall time and memory measurements.

Every RPC reply includes synchronized request wall time and process peak resident
memory. CUDA runs synchronize at request boundaries. The optional immutable
profile also records cold checkpoint load time and final attempt evidence.
`complete` does not attest that future events cannot arrive, and `cancel` does
not claim semantic success. The producer controls the whole execution outcome.

## What the checks establish

```sh
PYTHONPATH=src python -m unittest \
  tests.test_prediction_prefix_v8 tests.test_semantic_model_v8
```

The protocol tests use an explicitly fake decoder to test exact framing,
fragment evidence, rejection/retry, independent semantic rejection, callback
lifetime, and acknowledgement. A separate test generates with a real tiny
native model, verifies unchanged checkpoint weights, and checks that an
unacknowledged candidate does not mutate the accepted model continuation.
These checks establish transport and runtime behavior. They do not establish
learned execution quality, generalization, external-model comprehension, or
native admission; metrics explicitly report zero optimizer steps and no learned
success claim.
