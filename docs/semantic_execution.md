# Semantic execution: production v4 contract

Megatoken consumes source, invocation arguments, selected semantic records, and
delivered environmental results as one causal native episode. The serving adapter
uses the same dynamic track registry, context policy, physical tokenizer, and
committed-action executor as ordinary native generation. The paired JavaScript
controller remains the authority for selected guest computations; the Python
consumer independently checks transport, ownership, effects, scheduling, physical
framing, and model visibility.

No optimizer, fine-tuning, overfitting, or learned-success run is needed to validate
these engineering interfaces. Controlled selections are labeled controlled;
initialized model checks establish inference plumbing and failure handling only.

## Exact version and asset identities

| Artifact | Required version or identity |
| --- | --- |
| Public envelope | `semantic-execution/4`, `selected-harness/2` |
| Witness, source context, selected continuation | `semantic-trace/4`, `semantic-context/4`, `semantic-continuation/4` |
| Private source supervision | `semantic-supervision/2` |
| Model prefix and selected response | `selected-inference-prefix/2`, `selected-model-output/2` |
| Native lifecycle binding | `semantic-execution-binding/4`; native lifecycle version 2 |
| Runtime inference checkpoint | `semantic-model-checkpoint/4`; declared observed-bootstrap equivalence |
| Training restart format | 17, including mandatory experiment exposure counters |
| Tokenizer | `inclusionAI/Ling-3.0-tiny`, revision `9a98e35fe1c9ee255f78dd64771c7ae15a799481` |
| Tokenizer file SHA-256 | `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60` |
| Native codec fingerprint | `7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2` |

The exact shared declarations are packaged in
`src/megatoken/data/contracts/semantic-harness-v4.json` and
`src/megatoken/data/contracts/host-operations-v4.json`. Python verifies the
declaration and fixture digests, independently checks all valid and invalid
fixtures, and consumes the same descriptor bytes as the producer. Dedicated CI
pins the compatible producer commit and compares both files byte for byte.
Every admitted corpus and evaluation receipt also records the exact producer,
consumer, catalog, schema, fixture, and tokenizer identities it used.

Changed contracts reject earlier saved artifacts. Regenerate prepared data and
recollect affected model actions. Grouped witness exports remain explicit offline
diagnostics; they cannot pass production admission even if a caller supplies an
`admitted` string. The byte tokenizer is useful for focused tests but is refused
by production native admission.

## Public observations, generated labels, and private evidence

Programs include immutable source/digest, entry points, source spans, declared
capabilities, dependency revisions, module profile, and original-source metadata.
Invocations carry explicit identity, realm policy, receiver, closure, activation
kind, parent invocation, revision, input graph, bindings, trigger, ingress key,
deadline, and completion policy. Fresh and retained realms are distinct. An
invocation cannot overwrite a retained allocation or impersonate another owner.

Only semantic records and explicit coordinator lifecycle decisions receive
generation loss. Source, invocation arguments, externally supplied state,
dispatch/result receipts, callbacks, scheduling, source rereads, cancellation,
resource disposition, and runtime-allocated identities remain observed input.
Later use of an observed address in a generated lifecycle argument still carries
action/binding loss. Private reference outcomes, journals, certificates, lineage,
and admission receipts live in supervision or sidecars, never prediction records.

`semantic_harness_v4.py` imports discovered tracks rather than reserving eight
lanes. It checks code before activation, operation/source-site grants, evaluated
arguments against the selected heap, exact one-time dispatch/result correlation,
turn ownership, Promise adoption and settlement, FIFO ready continuations/jobs,
callback lifetime, cleanup, terminal completion, cancellation, and reconciled
uncertain outcomes. Root JavaScript call return and awaited task completion are
kept separate. Native completion receipts resolve returned Promises under the
declared policy and preserve cleanup rejection.

The capability materializer supports declared canonical JSON and owned graphs.
Owned graphs preserve aliases, cycles, holes, special scalars, and explicitly
owned callbacks. JavaScript integer-index key ordering is applied to the selected
heap before exact snapshot comparison. Callback markers are allowed only by the
operation's slots/graph policy and must reference an already owned closure.
Payload, node, property, and callback budgets are enforced. An observed result
graph is legal only for an operation declaring `owned-graph/1` results; scalar
operations cannot introduce reference identities.

## The committed model request reaches the real broker

`scripts/serve_selected_semantic.py --served BUNDLE` serves JSON-lines RPCs around
an actual immutable served model. `select` may produce up to 64 real native text
fragments for one logical witness, bounded by 64 KiB. No effect is dispatchable
before the terminal fragment. The response identifies the exact committed event
IDs, meridians, owner, tokenizer, and checked source reads.

The JavaScript controller validates and commits the selected request. Its
`beforeDispatch` hook then calls `commit-request` with the original bytes,
selection provenance, and trusted source-validation receipt. Megatoken locates
the saved native events, checks exact consecutive fragments and source/heap
materialization, and invokes `MultiTrackRuntime.dispatch_semantic_requests` on
that existing action. The returned descriptor binds the one JS broker dispatch.
There is no second generated tool envelope. The action route is recorded before
submission, and identical retries return the original receipt. A submission with
an uncertain outcome cannot be silently issued again.

Results re-enter as observed input. Awaited work becomes eligible only after its
typed dependency has been delivered and active capacity is available. Invalid
model text, missing termination, excess bytes, invalid UTF-8, or mismatched
requests retain their original bytes and selection evidence; the adapter never
replaces them with a teacher answer.

RPC operations are `initialize`, `select`, `commit-request`, `complete`, `cancel`,
`checkpoint`, `restore`, and the pure-control `outcome`. Coordinator execution adds
`open-orchestration`, `step-orchestration`, `status-orchestration`, and
`cancel-orchestration`. `initialize` must precede
active work, accepts evaluation seeds 17/29/43, compares the requested checkpoint
against the actually loaded `model_sha256`, seeds Python/NumPy/Torch, and returns
`semantic-inference-identity/4` with actual checkpoint and tokenizer identities.
Evaluation requires that attestation for model-origin results.

The request registry retains exact selected bytes after subsequent prefix
acknowledgement. An asynchronous dispatch can therefore commit its original
request after the pending feedback queue has advanced, without gaining authority
from an unverified client receipt. Every native RPC also reports elapsed work and
the native process's measured lifetime peak resident memory, including refusals.

### Free-running native coordinator evaluation

`open-orchestration` receives immutable child source/input proposals and allocates
one real coordinator. `step-orchestration` samples and commits its lifecycle action
through the same model session used for each child's semantic selections. Native
creation acknowledgement, bootstrap, joins, child return/throw, and parent progress
all preserve their actual original event IDs and bytes. The JavaScript runner uses
only the declared fixture broker for controlled engineering runs, or the original
model selections for fixed-weight and trained runs. Expected outcomes remain in
the independent scorer.

Engineering injection requires the explicit server flag
`--controlled-lifecycle-fixtures`, a coordinator declared `origin: controlled`, and
the separately gated `select-controlled` operation. It uses a distinct committed
fixture-action registry and reports `modelSelected: false`; it cannot impersonate
model or source-teacher authority. Model-origin requests reject those fixture
fields. Invalid coordinator or guest selections retain original fragments and
cancel the actual native descendants through the observed cancellation boundary.

The scorer checks `semantic-orchestration-evidence/4`: source-only proposals,
native allocation ownership, creation before child source/input, scoped completion
dependencies, completion before exact parent progress, child semantic replay, and
zero remaining work. Callback completion acknowledgements are an explicit
same-realm barrier: a newly eligible callback cannot consume the sole active slot
before the previous callback's native completion has actually arrived.

The paired tracer's `test/training-orchestration-evaluation.test.js` is the actual
process integration gate when `MEGATOKEN_PYTHON`, `MEGATOKEN_REPOSITORY`, and
`MEGATOKEN_ORCHESTRATION_SERVED` are set. Set
`MEGATOKEN_ORCHESTRATION_EVIDENCE_DIR` to retain its original evidence. It drives
explicit and compact async/callback/throwing children, coordinator budget
cancellation, invalid guest bytes, and an actual initialized-model rejection.
`scripts/validate_semantic_orchestration.py` independently checks these receipts
with the production tokenizer. Their original source-family splits remain intact;
engineering fixtures are not relabeled held-out test cases for a quality report.

## Dynamic children, external ingress, and restart

Selected `track_create`/`track_assign` actions use a typed execution binding.
Registration of source and invocation context is inert. The runtime first
publishes creation acknowledgement and its runtime-issued completion dependency,
then source, then invocation. Only successful publication marks the child ready.
Active capacity may keep a correctly bootstrapped child waiting. Failed bootstrap
cancels its allocated reservation and emits a separate failure receipt; a retry
cannot allocate a replacement identity invisibly.

`track_join` suspends the actual owning parent using that issued dependency.
Creation acknowledgement does not complete the dependency. The child's later
return, throw, failure, or cancellation is delivered separately to its parent,
and only then can the parent wake. Both guest parents and ordinary coordinator
tracks are supported. A coordinator does not invent a JavaScript parent frame.
Capabilities can only narrow, closed work cannot be resurrected implicitly, and
parent cancellation cascades to descendants. Lifetime addresses are independent
of the simultaneous active-track bound.

Callback ingress acknowledges the suspended parent's exact already-selected await
from `realmCommitted` before allocating its child. A typed callback-completion
dependency transfers capacity even if the awaited native Promise has already
settled. Persistent callbacks arriving after their registration call completed
instead require a matching live subscription receipt. A separate waiting native
subscription owner preserves the registration's provenance and grants, owns the
later callback tasks, and retires only after observed unsubscribe and all delivered
children finish. The completed registration invocation remains completed.
An exported owned closure can also be called later without an event subscription.
That path requires the original native completion graph or a validated committed
callback-graph request to prove the closure actually escaped to the host. The
observed external callback gets a fresh execution task and an explicit ingress
receipt retaining its original invocation, realm, closure and escape evidence.
An unrelated closure in the same heap does not acquire that authority.

Inference checkpoints pin model/tokenizer identities, exact public prefixes,
selected state references, action receipts, rejection evidence, waiting Promise
dependencies, lifecycle ownership, and registered/published child context. They
include children allocated before their first model selection and retain
completion deliveries. Restore preserves addresses and starts from an explicit
observed checkpoint bootstrap. It does **not** claim bit-identical KV or sampled
output replay. Native resources and uncertain effects remain subject to the
paired harness's explicit reattachment/reconciliation authority; model restore
dispatches zero native effects.

Source visibility is checked using the actual native `ContextPolicy`, rather
than counting bytes. Every generated guest record must see every physical source
fragment. Native observed context announcements preserve source visibility across
long local histories; eviction shrinks visibility, and a digest-checked reread
arrives as a new observation before continuation. Logical episodes may span more
records than resident history only when their verified context boundaries keep
the necessary source and current state available.

Retained activation and resumed work receive an observed
`semantic-selected-bootstrap/4` when another owner has advanced the shared realm
or the necessary selected history has been evicted. This copies the controller's
current committed heap, cells, locals, frames, pending continuations and completion;
it is bound to the exact public same-realm prefix digest and causal cutoff. Source
and all state fragments must be visible together before another model selection.
The adapter never reconstructs this state from a completed teacher execution.
The shared public `model-context` receipt carries the same selected state,
`realmPrefixDigest`, and `visibleThrough`. Its cutoff is exactly the preceding
public sequence, its digest covers the same-realm committed prefix, and a checked
source reread must immediately precede it. The JavaScript replay and source gate
recompute the complete state; the native consumer checks the independent causal,
identity, data-graph, scheduling, and initial-input constraints. Source collection
and serving emit these observations at retained activation, resumed realm turns,
and every 32 selected records during a long segment. They receive zero loss.

Preparation and serving share the same pure bounded-history retention function.
The data visibility gate checks actual resident events, rather than assuming all
earlier serialized records remain readable. Each prediction must see its verified
invocation-state snapshot and every subsequent same-realm record, or the complete
earlier public state history when no snapshot exists, plus all source fragments.
A small window which retains code but loses a captured-cell mutation is rejected.
The prepared geometry records both physical token capacity and resident history;
training refuses an incompatible smaller model before optimizer setup.

## Collecting admitted training data

The first component gate uses the real production tokenizer and physical
chunk planner:

```sh
python scripts/admit_semantic_harness.py \
  --input teacher.json --output native-receipt.json \
  --tokenizer-json /path/to/tokenizer.json
```

Its `semantic-native-admission/4` receipt proves exact logical-to-physical bytes,
one logical action across fragments, observed/generated masks, causal result
ordering, and actual source visibility. It explicitly reports
`componentGate: native-consumer-only`; it cannot certify source semantics by itself.

`scripts/prepare_semantic_corpus.py` re-executes the pinned producer's independent
source/reference/replay gate, repeats native admission, checks all stored and
fresh identities, preserves source/license/trust ancestry, deduplicates whole
episodes, freezes whole-family splits, and writes prepared native data plus
private audits. The CLI accepts JSON or JSONL inputs. Unsupported/refused cases
remain in the attempted denominator with reasons. `scripts/validate_semantic_corpus.py`
applies the same path to every required case in a producer manifest and fails if
any required case is rejected.

```sh
python scripts/validate_semantic_corpus.py \
  --corpus /path/to/teacher-corpus --output /path/to/prepared \
  --tracer-root /path/to/pinned/tracing-js-self \
  --tokenizer-json /path/to/tokenizer.json
```

The source producer collects actual effectful, Promise, callback, repeated
ingress, and long-session source executions. Controlled continuity fixtures are
additional engineering checks; they do not become teacher data by relabeling.

Tracks teaching has its own concrete native collection recipe:

```sh
python scripts/collect_semantic_lifecycle.py \
  --input independently-admitted-child.json --output /path/to/prepared-tracks \
  --tracer-root /path/to/pinned/tracing-js-self \
  --producer-revision EXACT_PRODUCER_SHA --consumer-revision EXACT_CONSUMER_SHA \
  --tokenizer-json /path/to/tokenizer.json
```

This runs a declared coordinator policy through the real native lifecycle
executor: observed proposal, generated create/bind, creation receipt, child
code/input, generated join, admitted guest execution, actual correlated child
completion, parent wake/progress, and coordinator completion. A separate trusted
teacher-action type records orchestration provenance without manufacturing a
sampled model tape. Each child is independently source/native admitted again;
private component receipts and all source-family ancestry remain attached.
The wrapper retains the child's frozen split; children from different splits
cannot be joined into one orchestration episode.
Collected guest bytes are replayed without repeating native effects. Multiple
children require unique invocation IDs at source collection. The recipe accepts
one root activation per child; multi-event handler sessions are collected by the
source producer as separate episodes.

The complete production collection recipe uses sixteen authored algorithm
families with paired explicit/compact projections. Every family is retained
under the declared split seed. Generate each representation with the tracer's
`generate-effectful-training.mjs --recipe training-v4 --representation explicit|compact`
and the real native consumer. Then prepare all experiment arms together:

```sh
python scripts/prepare_semantic_experiments.py \
  --explicit-corpus /path/to/recipe-explicit --compact-corpus /path/to/recipe-compact \
  --tracer-root /path/to/pinned/tracing-js-self \
  --producer-revision EXACT_PRODUCER_SHA --consumer-revision EXACT_CONSUMER_SHA \
  --tokenizer-json /path/to/tokenizer.json --output /path/to/experiments
```

This rechecks source and native evidence, creates five prepared representation
arms and four actual lifecycle collections (train/validation in each interactive
representation), and verifies every required behavior in both training and
validation. The output includes prepared data bindings and a nonoptimizing
stage-supply receipt. A corpus with missing stages is refused.

## Training configuration and fair evaluation

`SE` requires actual code/invocation, tool-feedback, async, callback, repeated
invocation, typed-child, source-reread, and recovery supply. Missing stages fail
before training. `SC` and `SO` preserve the separate historical pure
common-support controls. Templates for seeds 17, 29, and 43 retain the normal
production tokenizer, teacher-memory, donor, checkpoint, and held-out requirements.
They contain explicit asset bindings to fill; no sample optimizer run is part
of preparation or engineering acceptance.

`configs/semantic/experiment-plan-v4.json` freezes separate common-support and
effectful cohorts, explicit/compact representations, the pure outcome-only
control, all three seeds, fixed-example/fixed-token/measured-compute budgets,
whole-task/trace thresholds, and zero-tolerance hard gates.
`scripts/plan_semantic_training.py` binds actual prepared data and frozen cohorts
to the complete 45-configuration matrix without starting training. The training
preflight requires a pinned complete three-repository engineering receipt with
zero optimizer steps. It checks the exact dataset representation, tokenizer,
schema/catalog/fixture digests and producer/consumer pins before model or optimizer
setup; each native training window also checks its actual teaching labels.
The semantic comparison matrix deliberately uses one metered worker and one
atomic window per budget decision. It does not claim unimplemented global
budget synchronization between independent workers. Restart format 17 preserves consumed full episodes,
generated/observed token exposure, and active compute time. Fixed-token windows
retain a declared maximum 65,535-token overshoot for a single 64-KiB logical
action; receipts report the actual overshoot. Exported budget receipts bind the
served weights digest to the exact training checkpoint and frozen cohort.

Compact data requires the producer's certified compact projection. It cannot be
created by deleting arbitrary semantic work. Outcome-only labels use the same
independently admitted source/input and final observable graph, produce one
`semantic-outcome/4` answer, and cannot hide effects, Promise work, or callbacks.
The serving `outcome` operation directly samples that answer from source/input;
it does not run the semantic witness interface behind a different label.
Active invocations lock the declared representation, which also survives
checkpoint restoration. Compact serving instructions preserve the producer's
eight-operator/32-node expression limit and all causal boundaries.

`scripts/evaluate_semantic.py` freezes cohort declarations, invokes a production
runner, and compares exact completion, state, ordered effects, callbacks, and
cleanup. Expected results remain outside the runner's request. A correct final
answer with a false intermediate computation fails the trace gate; the first
controller inconsistency is retained. Reports preserve measured token exposure,
latency, compute, memory, dispatch, ready-queue, resource-leak, failure, seed,
representation, checkpoint, and tokenizer evidence. Controlled and initialized
fixed-weight reports have `learnedSuccess: null`. Model-origin results require
the real seeded inference attestation. Comparisons require every planned seed
and a matching frozen budget/cohort declaration.

Actual inference reports meter the original emitted token IDs for every retained
fragment, including invalid output, and verify those IDs decode to its exact raw
bytes. Fragment event IDs deduplicate repeated failure/selection receipts.
`canonicalGeneratedTokens` separately reports native dataset encoding costs;
these can differ from original inference costs at tokenizer fragmentation
boundaries. Controlled engineering costs use that explicitly labeled canonical
basis. Learned comparisons require the frozen test-only cohort; validation and
test cases cannot silently be pooled into a learned result.

Outcome-only costs count the original generated record bytes, including invalid
bytes retained in a refusal, and only the canonical source/invocation observations
actually served. Controlled selection plans and expected outcomes are removed
from model-origin runner requests. Native RPC replies measure their process peak
resident memory and elapsed work, including failures. The cross-process evaluator
reports the sum of runner/native peak memory as a documented upper bound rather
than silently omitting model memory.

The historical runbook does not contain a measured frozen baseline. A new
common-support cohort may preserve that runbook and explicitly mark historical
measurements pending. Learned comparisons remain blocked until the actual
historical receipt, matching cohort and all three seeds are supplied; no measured
historical result is invented during engineering preparation.

## Engineering validation commands

The dedicated semantic workflows exercise shared positive/negative schemas,
the real production tokenizer, source-certified preparation, typed lifecycle
collection, a two-invocation async selected bridge, Promise cleanup, retained
events, checkpoint recovery, actual native inference boundaries, and frozen
controlled evaluation. They do not instantiate an optimizer.
The controlled evaluation command uses `--require-engineering-success`, so a
wrong task result, invalid trace, nonzero hard gate or leaked resource fails CI.

```sh
python -m pytest -q \
  tests/test_semantic_data_pipeline.py tests/test_semantic_runtime_integration.py \
  tests/test_semantic_model_adapter.py tests/test_semantic_harness_v4.py \
  tests/test_semantic_orchestration.py \
  tests/test_semantic_contract_v4.py tests/test_semantic_host_contract.py \
  tests/test_selected_semantic_bridge_v4.py tests/test_semantic_harness_lifecycle_v4.py \
  tests/test_semantic_execution.py tests/test_training_stream.py \
  tests/test_checkpoint_async.py tests/test_checkpoint_replace.py \
  tests/test_track_context.py tests/test_context_preparation.py
python -m pytest -q tests/test_meridian_training.py \
  -k semantic_execution_continues_from_teacher_recorded_semantics
```

Cross-repository engineering receipts and compatible Git pins are the publication
gate. Free-running trained-model quality is a later empirical result, never
inferred from a schema check, controlled trajectory, teacher-forced loss, or
unchanged initialized weights.
