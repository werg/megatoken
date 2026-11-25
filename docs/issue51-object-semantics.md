# Issue 51: object existence and mutation-sensitive enumeration

Current verified scope is in **Completed follow-up** immediately below.
Earlier checkpoints describe only their recorded code and inputs.

## Completed follow-up — retained object operations and recovery

The paired implementation verified here is producer `a72267e26ab7b7fc5f22134abee315fd7ed6e64b` and consumer
`053e9c78cad1dbb5313012421ab2987526fe6dab`, both clean when the recorded checks began. Documentation commits
may advance the draft branches after these exact implementation and data pins.

The receipt records the canonical generation source revision
`76fd24a91ad7e535aac33af542b481e1337838df` and generated artifact revision
`5553de1a10761626c5604a57453291039d73683d` separately from the verified producer
revision above. The source fixture generator and unmodified standard contract
gate both have successful, hashed evidence logs. When the verified producer
revision exactly matches the generated artifact revision, the receipt uses
those recorded generation and gate results without claiming a second rerun.
For a later code-only producer revision, it records fresh source-check and
gate runs. Original generation status and all ten generation logs are preserved
separately as historical evidence.

### What this follow-up establishes

- Seven Proxy reflection operations now share the ordinary descriptor, prototype
  and extensibility rules: `getPrototypeOf`, `setPrototypeOf`, `isExtensible`,
  `preventExtensions`, `getOwnPropertyDescriptor`, `ownKeys` and
  `defineProperty`. Object and Reflect callers retain their different behavior.
  Callbacks use the selected handler and method; invariants inspect the target
  after callback effects. Definition callbacks receive a fresh descriptor object
  while the original descriptor snapshot remains authenticated through calls,
  returns and checkpoints.
- Transparent nested Proxy targets and `Proxy.revocable` retain distinct object,
  handler, target and revoker identities. Repeated revocation has no second
  effect. Revoked get, set, has and delete operations derive the correct
  TypeError, message and standard prototype from their source operands.
- A handler's own `has` getter is an explicit continuation. It runs once with
  the handler receiver. Its returned closure remains selected across handler
  mutation or revocation; a nullish result falls back to the captured target.
  Getter throws preserve identity. Every exposed selection/call/return boundary
  is checked on restoration.
- Standard Error objects use shared ordinary descriptors and prototype identity,
  including their inherited names, own messages and ordinary `instanceof`
  behavior. Invalid `in` checks retain operand effects and reject an invalid
  right-hand side before key conversion. Errors caught inside Promise reactions
  remain ordinary shared objects. Self-resolution rejection creates its derived
  TypeError after the ordinary invocation stack has emptied, through an internal
  allocation path that public records cannot invoke.
- Derived-constructor arrows retain the live receiver and `new.target` before
  and after `super()`, including pre-initialization ReferenceErrors and later
  scheduled calls. Promise reactions and adoption getters preserve nested call
  scopes, captured values and authentic parameter bindings across returns and
  suspended checkpoints.
- Ordinary `for…in` checks captured candidates against live property state,
  including prototype and enumerability changes under the pinned engine.
  Stable transparent Proxy enumeration retains Proxy identity and target state.
- Map and Set operations retain SameValueZero equality and ordered tombstones.
  Live collection iterators observe supported mutations without restarting;
  completed iterators stay complete. Array/string spread retains evaluated
  values before later effects. Nested ordinary array binding, defaults, rest
  and abrupt outer-iterator cleanup retain their order and identities.

Preparation also preserves lone UTF-16 code units through JSON serialization
and tokenization, keeping these valid JavaScript string examples admissible.

### Verification on the recorded revisions

| Check | Recorded result |
| --- | --- |
| Standard unmodified `npm run contracts:check` | Passed; 174 protocol source cases, 6 region examples, 4 host source cases |
| Complete producer test selection | 3462 passed, 98 skipped, zero failures; 458 test modules |
| Producer reference fixture | Passed |
| Paired native-source checks | 9 selected titles passed with 0 selected failures/skips/cancellations; all 130 tests in 9 files ran (130 passed, 0 skipped, 0 failed/cancelled) at the recorded producer/consumer pins |
| Independent Python regression selection | 550 passed, 0 skipped across 26 test modules |
| Python Ruff lint and format checks | 2 steps passed |
| Complete frozen issue51 cohort checks | 134 passed, 0 skipped |
| Actual pinned-tokenizer preparation | 121 attempted, 121 admitted, zero refusals and zero optimizer steps |

The producer command uses Node v24.22.3 with experimental VM
modules and 4 concurrent test modules. The full command and individual test
results are retained in the receipt and logs. The Python selection and its
exact command are recorded separately; these counts are tests, not percentages
of JavaScript or independent source families.

The canonical population preserves all **93 previous
source inputs** byte-for-byte for source, entry, arguments, family and ancestry.
The issue51 cohort contains **122 declared originals: 121
supported and 1 refused**, preserving all **122
previous cohort inputs** in the same fields. Earlier negative controls were
promoted only after their unchanged original source passed native execution,
source observation checks and fresh replay. The cohort remains one declared
ancestry group; additional examples do not become independent families.

A separate historical correction run on producer `5553de1a10761626c5604a57453291039d73683d`
and consumer `6bb4574931c4fdf19588a86aef96577b22ff6541` recorded 3,558 producer tests
(3,429 passed, 31 failed, 98 skipped), an interrupted native run with 465 passed
and 16 failed among 481 completed of 546 collected, 125 passed and 9 failed
cohort tests, and preparation that admitted 111 of 121 attempts with 10 refusals.
These results are retained as historical correction evidence and are not final
passing verification. The receipt records the archive manifest hash and hashes
for its small evidence files; the large cohort export, preparation JSONL and
prepared output remain outside repository evidence. The failed initial paired
test-title filtering attempt is also preserved as historical runner-selection
evidence and is not counted as a semantic result. Its raw failure-detail counter
is 18, while the TAP summary records the authoritative 9 distinct failed tests.

A separate full-files paired native run on producer
`a72267e26ab7b7fc5f22134abee315fd7ed6e64b` and consumer
`80280e425b0402bb4ee5f18e42e27087fa91dea5` completed 130 tests: 129 passed,
one failed, zero skipped or cancelled. The selected nine native checks had
eight passes and one failure in the Promise self-resolution adoption replay:
intrinsic Error allocation occurred without an active ordinary invocation,
and the consumer rejected it with `active invocation required`. This is
historical correction evidence, not a final passing result. Its manifest and
five hash-verified files are preserved under
`historical/finish-paired-self-resolution-80280e4/`.

The receipt records operation and method counts for every accepted source.
Method counts distinguish the operation and Object/Reflect caller where the
record provides them. They describe exercised transitions, not general method
coverage percentages.

Preparation used the pinned Ling tokenizer with 8192 positions and 1024 history.
The preparation report confirms zero optimizer steps and no learning claim.
The full manifest/report fingerprint and tokenizer identity are retained.

- Contract digest: `ef501aa5c4e5a41b77a7c7c58e0f38dc2ee6d0ae3ef1aa6b6c91e304ddbf6c2f`.
- Fixture digest: `56f7f36d3b25b3b11a208be3de337e637b9782e6759faea1e1ec107b0a90f0ce`.
- Interoperability digest: `aa3c9077e4627e90d81ea7534635ffc3b168051db1c02463c3b5fe9e989f1677`.
- Frozen registry digest: `14348b06d4d128fc94ce7056fb20f52e87b237b5e6da17b00479ee940b64eb2f`.
- Preparation manifest: `82042270808558a4e78df0a9015ae2c3a5db5f58c39bde47917a5864a8b8463f`.

See [the exact verification receipt](evidence/issue51-20261124/finish-verification.json)
and [the preparation result](evidence/issue51-20261124/finish-preparation.json).
Successful logs and their SHA256 hashes accompany them. Earlier checkpoint
receipts below remain historical evidence for their recorded revisions.

### Explicit remaining boundaries

Issue #51 is an ongoing umbrella and remains open. This follow-up does not
claim every Proxy/constructor/iterator combination. Remaining bounded cases
include general callable Proxy apply/construct, nested target invariant
dispatch, effectful Proxy enumeration, trap getters beyond the own `has`
path, non-callable trap-getter results, effectful descriptor or own-key list
conversion, broader exotic inheritance and asynchronous callbacks, general
nested custom iterables, assignment destructuring, and implicit captured
builtin iterator bindings. Native engine-specific Error stacks remain outside
the standard Error property profile.

Only the recorded Node 24 environment was checked here. The separate combined CI gate file was compared byte-for-byte with
historical baseline `f7696e63258913e7bb8a450c67a708b70c3232ee`; the baseline and
current hashes and original definition values are in the receipt. The gate has
`run: false`: this is a pin-preservation check, not a passing combined CI
execution. Neither draft PR is merged.

### Reproduction

Check out the producer and consumer revisions above. Regenerate the paired
fixtures with the repository generators only when intentionally changing their
inputs; the saved receipts apply to the committed artifacts at these pins.

```sh
# In the producer:
npm run contracts:check
node --experimental-vm-modules --test --test-concurrency=4 test/*.test.js
node fixtures/reference/reference-fixture.mjs
node tools/export-issue51-evidence.mjs --out /tmp/issue51-finish.json

# Create the preparation input from the same frozen export:
python - /tmp/issue51-finish.json /tmp/issue51-artifacts.jsonl <<'PYINPUT'
import json, sys
with open(sys.argv[2], "w") as output:
    for row in json.load(open(sys.argv[1]))["cases"]:
        if row["artifact"] is not None:
            print(json.dumps(row["artifact"]), file=output)
PYINPUT

# In the consumer:
PYTHONPATH=src python -m pytest -q tests/test_issue51_object_source_v8.py
PYTHONPATH=src python scripts/prepare_semantic_v8.py \
  --input /tmp/issue51-artifacts.jsonl --output /tmp/issue51-prepared \
  --producer-root /path/to/tracing-js-self \
  --producer-revision a72267e26ab7b7fc5f22134abee315fd7ed6e64b \
  --consumer-revision 053e9c78cad1dbb5313012421ab2987526fe6dab \
  --tokenizer-json /path/to/pinned/tokenizer.json \
  --max-positions 8192 --max-history 1024 --source-recheck-timeout-seconds 90
```

## Historical work record


This is an implementation checkpoint for
[tracing-js-self #51](https://github.com/werg/tracing-js-self/issues/51), coordinated
with the producer in the sibling `tracing-js-self` checkout. The issue remains an
ongoing umbrella: these changes do not complete every object, Proxy, construction,
collection, or source-projection operation it describes.

## Implemented native checks

`property.has` uses the same ordinary descriptor/prototype lookup as property
reads, without invoking a getter or inventing its returned value. Its exact row
is `{kind, object, key, result}`, with `kind: "property.has"` and a tagged boolean
result. Own and inherited properties, nonenumerable properties, Symbols,
accessors, sparse array holes, null prototypes, live deletions, and restored
prototype identities are checked independently. Unknown identities, undeclared
Symbols, numeric keys that have not undergone property-key conversion, forged
results, and extra receiver fields are refused before committing any state.

Known implicit Object/Array/Function intrinsic members can establish existence
without materializing callable values. Initialized Object.prototype state takes
precedence over the implicit default, so deleting or adding one of its properties
changes subsequent existence checks. A callable's implicit `prototype` property
depends on construction metadata and remains refused when that property is not
represented. A Promise with an implicit prototype requires its intrinsic group
to be initialized before an inherited existence result can be accepted.

Ordinary for-in retains the existing captured candidate list and cursor shape.
On each active step it checks that the captured prototype identities and the
enumerability of surviving captured descriptors have not changed. It then skips
candidates that no longer exist in the live prototype lookup. This permits
property additions and deletions while preserving the original candidate order:

- Newly added names do not join an already captured candidate list.
- Deleting a future own or inherited property skips it if it is now absent.
- Deleting an own shadow still yields its candidate if an inherited property
  remains, including an implicit intrinsic such as `toString`.
- Readding a captured name preserves its original position; inspecting an
  accessor's existence does not call it.
- A hidden shadow excluded at creation does not become a new candidate when
  deleted later.
- The cursor advances over skipped names, survives JSON restoration, and stays
  exhausted after completion even if later mutations would be unsupported for an
  active cursor.

Changing a surviving captured property's enumerability or replacing a captured
prototype still produces an explicit refusal during active enumeration. The
policy matches the independently executed Node loops covered by these tests;
this is not a claim to model every permitted mutation behavior of every engine.
If initialized Object.prototype has enumerable string properties at cursor
creation, its identity must be explicit in the captured chain; an implicit
terminal prototype is refused rather than silently omitting those names. Names
added after creation still do not extend the candidate list, but can supply the
inherited presence of an existing candidate whose own property was deleted.

## Verification and retained evidence

The existing property, Proxy, construction and enumeration test selection passed
51 tests before implementation. The first published implementation checkpoint
(`cc90ac78b26a01b6e2da68277863ad2f78b30d2d`, matching the tested local tree) passed
the following selection of 64 tests:

```sh
PYTHONPATH=src python -m unittest \
  tests.test_property_presence_v8 tests.test_enumeration_v8 \
  tests.test_protocol_properties tests.test_proxy_v8 tests.test_construction_v8 -q
```

After those checkpoints, these 60 existing protocol/native tests also passed:

```sh
PYTHONPATH=src python -m unittest \
  tests.test_protocol_boundaries_v8 tests.test_protocol_v8 tests.test_native_v8 \
  tests.test_native_detail_regressions_v8 tests.test_native_array_boundaries_v8 -q
```

Additional controls subsequently cover implicit intrinsic fallback after
deletion and refusal of uninitialized Promise inheritance. The focused property
existence/enumeration selection passed 24 tests after those refinements, and the
combined selection passed 66 tests at the tree published as `591eda5`. Tests
invoke the available Node executable for independent original-loop comparisons;
they report skips if Node is unavailable. They do not train a model.
The observed runtime was Node v24.22.3 with Python 3.12.14. The subsequent
implicit-prototype enumeration guard passed the combined 68-test selection;
source-artifact integration is recorded separately when its final inputs arrive.

## Continue from this checkpoint

1. Keep the producer and consumer on their current work branches. The root task
   coordinates incremental remote commits and PR progress notes; GitHub-created
   commits can have different hashes from temporary local commits. Inspect
   the current branch before making another local commit.
2. Regenerate and copy the producer's current contract and affected fixtures
   together. Do not patch historical source receipts or their recorded source
   identities to make them appear newly executed.
3. Reexecute original source cases for property existence and structural
   enumeration mutations through the producer's ordinary source adapter. Pass
   the produced artifacts through `validate_v8_artifact` in a fresh Python
   process, checking replay, every recorded context and continuation parity.
   Until this is recorded, unit tests and native loop probes do not establish
   end-to-end source admission or preparation.
4. Run the focused commands above after syncing the current contract. Add
   corrupted result/key/cursor controls to the original artifact population,
   then exercise the ordinary admission/preparation route if source evidence
   satisfies its existing gates.

The next original-source integration artifacts and their exact producer/consumer
identities are pending at this checkpoint. Existing broader limitations include
Proxy/exotic enumeration, mutation of captured prototypes or enumerability, and
implicit intrinsic prototype enumeration when enumerable intrinsic properties
are not represented in the captured chain (explicitly refused at creation).
Continue those as concrete semantic
work rather than interpreting unsupported cases as successful guest exceptions.

## Checkpoint — clean original-source admission and ordinary preparation

The producer at `c3e1a55f19eb04fc17957dd8fdea01031ca35ecd` passed the full
`npm run contracts:check` on Node v24.22.3. Its byte-identical current contract
was synchronized in consumer `c3df6a32436233f714b365c3a7b2f3e5ce967b5c`; all
68 focused property/Proxy/construction/enumeration tests passed.

The clean exporter produced 23 originals: 20 supported and 3 explicit refusal
controls, with no unexpected failures. Consumer commit
`4c8ee3143171407dd670d44abcb86fb143344c34` retains that complete compressed
evidence and independent native tests. The 25 tests in
`tests/test_issue51_object_source_v8.py` passed: native replay, every context,
continuation, forged operation refusal and rehashed cursor refusal. The single
frozen registry includes *all* 23 source identities and the same ancestor.

The ordinary preparation command completed against the clean producer and
consumer implementation revisions above:

```sh
# Producer checkout at c3e1a55, consumer checkout at c3df6a3.
node tools/export-issue51-evidence.mjs --out /tmp/issue51-cohort.json
# In the consumer checkout, extract only the supported originals.
python -c 'import json; d=json.load(open("/tmp/issue51-cohort.json")); print("\n".join(json.dumps(x["artifact"]) for x in d["cases"] if x["artifact"] is not None))' > /tmp/issue51-artifacts.jsonl
PYTHONPATH=src python scripts/prepare_semantic_v8.py \
  --input /tmp/issue51-artifacts.jsonl --output /tmp/issue51-prepared \
  --producer-root /path/to/tracing-js-self \
  --producer-revision c3e1a55f19eb04fc17957dd8fdea01031ca35ecd \
  --consumer-revision c3df6a32436233f714b365c3a7b2f3e5ce967b5c \
  --tokenizer-json /path/to/pinned/tokenizer.json --source-recheck-timeout-seconds 90
```

Receipt (original clean revisions; not a claim about subsequent commits):

- 20 attempted / 20 admitted, 0 preparation refusals.
- Contract digest `9e5946f6745460ad45698eed882641cbc8bb171feb58a98ee53b60984ba23658`.
- Fixture digest `56f7f36d3b25b3b11a208be3de337e637b9782e6759faea1e1ec107b0a90f0ce`.
- Interoperability digest `813034bfdedd09a779cc6f45109099fe8231621baac8b770f23379329fbd9218`.
- Registry digest `e24d9baecb211cf77479b5d81daab0a775936dd6dc6a05c365bc41d8570b22f6`.
- Manifest fingerprint `5d279e992f64ecd1ab2b3e1089d7356670614c0b7af2afe5c06ee0769d1fc5ec`.
- Actual inclusionAI/Ling-3.0-tiny tokenizer SHA256
  `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60`,
  codec identity `7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2`.
- 8192 maximum positions, 1024 maximum history; zero optimizer steps,
  no learning claim. The three deliberately refused controls are not included
  in the input artifact stream; they remain in the frozen registry.

The native checker already understands the existing conversion and property
records; no parallel builtin-specific native evaluator is needed for Reflect
object-key conversion. Producer commits `5fea23aa` and `152b8f50` implement
retained runtime keys and independently observe direct Reflect calls. New
source fixtures must be rechecked and synchronized before describing their
expanded population as ordinarily prepared. See the paired PRs for the latest
exact revision receipts.


## Checkpoint 5 — full expanded baseline verified

The unmodified `npm run contracts:check` passed at producer
`87f92b7bb1c4bca5af5830fa9d178c609ca8e53f` with Node v24.22.3. It independently
rechecked all 74 canonical protocol sources, 6 region examples and 4 host
examples. The expanded interoperability digest is
`809f6b29878f3f8321e082463acb825e76946b2371f5d8e2f5cfc0c91df60f66`.

The complete object cohort now retains 26 declared originals: 23 successful
source exports and 3 explicit refusal controls. Its frozen registry digest is
`342b81dd173f2b713d3f9bd38bfd16d3bbe36ef973ba3d08acde68c979b53daa`.
Consumer `ce04621455a3e06a22714347e49293a854ea77e1` passed all 31 cohort tests,
including independent replay/context/continuation checks and atomic rejection
of forged Reflect conversion results, Promise receivers and restored cursors.

Ordinary preparation on these exact clean revisions admitted all 23 supported
originals with zero preparation refusals, using the pinned Ling tokenizer,
8192 positions and 1024 history. The manifest fingerprint is
`e5d21b943228999bd39b9882a64c53e261e03037227b0288893d3b075bbe8664`.
The operation performs zero optimizer steps and makes no learning claim.
The three source refusal identities remain in the declared registry.

Evidence: `docs/evidence/issue51-20261124/reflect-contracts.log` and
`docs/evidence/issue51-20261124/reflect-preparation.json`. These receipts describe
the exact revisions above; later Proxy, error and construction work must receive
its own regenerated contract, original-source and consumer evidence.

The next implementation work is prioritized by semantic uncertainty: retained
Proxy traps with late invariant checks, genuine intrinsic-error identity and
partial effects, and constructor arrows retaining live `this`/`new.target`.
The issue remains open; this checkpoint is not complete #51 language coverage.


## Checkpoint 6 — Proxy faults, constructor captures and shared object state

The completed paired implementation is producer `6b0d8f5bd18de39c85046babed02b4afba0578fd` and
consumer `13ce89795a4625c0e124171f61a0e8bbd45d88e1`. The full unmodified `npm run contracts:check`
passed after regenerating all affected fixtures. It independently rechecked
**93 canonical protocol sources, 6 region examples and 4 host sources**. All
74 prior canonical source inputs retain their exact source, entry, arguments,
family and ancestry. Their generated operation counts are unchanged.

### Behavior established

- Source-qualified `in` and `delete` expressions use Proxy `has` and
  `deleteProperty` traps while retaining the selected synchronous function, its
  arguments and effective receiver even if the handler changes during the call.
  Results use Boolean normalization. Target descriptors and extensibility are
  checked after the trap returns, including traps that freeze a property,
  prevent extensions or delete the property first. Strict false deletion and
  invariant failures create distinct operation-derived TypeErrors. The source
  adapter authenticates the caller's strictness, the retained return token,
  fresh allocation, source origin and derived message. A guest throw retains
  its own identity. Frozen `get`/`set` invariants also agree between the public
  JS and independent Python machines; this is not a broader source-qualified
  frozen-get/set diagnostic claim.
- Derived-constructor arrows captured before `super()` retain the live receiver
  cell and actual `new.target` when invoked after initialization. Private access
  continues to operate on that initialized original instance. They keep this
  receiver when the constructor returns a substitute object. This also works when the arrow later serves as a Proxy
  `has` trap or a primitive-conversion callback.
- Invalid `in` derives a TypeError before attempting property-key conversion,
  preserving earlier operand effects and caught error identity/name. The former
  refusal control is now accepted without changing its source or identity.
  Intrinsic Error brands are paired with retained allocation records; ordinary
  writes to own `name` and `message` remain possible. Restoration verifies the
  allocation registry and active call's callee, receiver and captured aliases.
- Promise property operations retain fresh Symbol identities, registered Symbol
  aliases and Symbols returned by captured helper functions. Matching Symbol
  descriptions do not merge distinct identities.

### Evidence and preparation

The complete frozen issue cohort contains **48 declared originals: 42 supported
and 6 explicit refusals**, with zero unexpected export failures. All 26 original
cohort inputs retain their source, entry, arguments, family and ancestry. The
registry includes every supported and refused member under
`issue51-original-objects`.

| Verification | Result |
| --- | --- |
| Producer integration tests | 120 passed |
| Independent Python regression and recovery tests | 323 passed |
| Whole-cohort replay, context, continuation and forgery tests | 55 passed |
| Actual pinned-tokenizer preparation | 42 attempted, 42 admitted, 0 refusals |

The generator-close, enumeration and controlled host-recovery archives were
also regenerated from the clean producer. The first two have source checks;
the controlled host-recovery scenario proves one dispatch and recovery without
redispatch, and is not promoted to source-qualified evidence.

Ordinary preparation ran on the exact clean revisions above with the pinned
Ling tokenizer, 8192 positions and 1024 history. It performs **zero optimizer
steps** and makes no learning claim. The six refused source identities remain
in the registry but have no artifact in the preparation input stream.

- Core contract: `39a27911db67db3d1848a503a4d743f3c1cc9b699b6d2404d363dc37ce8d0a9d`.
- Fixture: `56f7f36d3b25b3b11a208be3de337e637b9782e6759faea1e1ec107b0a90f0ce`.
- Interoperability: `62e5f6862fc1498a3b56b823b5d7f5a6573a09bb0831075786d0ea92226a22f7`.
- Frozen registry: `04f60979f58d56b3949fca79f0b9ebd7f4da6f5b7a10f241d84b08ecba814e3e`.
- Preparation manifest: `3b6bd2bae822c1b417f6ca3f0de8c8dbeabc42e7fca3fa362f3b9953177987ef`.
- Tokenizer SHA256: `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60`.

The durable receipt is
`docs/evidence/issue51-20261124/objects-verification.json`, with the full standard
contract, producer, native and cohort test outputs alongside it. The separate
`objects-preparation.json` records the exact clean-revision preparation result.
These receipts supersede Checkpoint 5 for current scope; older receipts remain
historical evidence for their own revisions.

### Reproduction

```sh
# Producer at 6b0d8f5bd18de39c85046babed02b4afba0578fd, Node v24.22.3.
npm run contracts:check
node tools/export-issue51-evidence.mjs --out /tmp/issue51-objects.json
# Consumer at 13ce89795a4625c0e124171f61a0e8bbd45d88e1; extract non-null artifacts as JSONL.
python - /tmp/issue51-objects.json /tmp/issue51-artifacts.jsonl <<'PY'
import json, sys
with open(sys.argv[2], "w") as output:
    for row in json.load(open(sys.argv[1]))["cases"]:
        if row["artifact"] is not None:
            print(json.dumps(row["artifact"]), file=output)
PY
PYTHONPATH=src python -m pytest -q tests/test_issue51_object_source_v8.py
PYTHONPATH=src python scripts/prepare_semantic_v8.py \
  --input /tmp/issue51-artifacts.jsonl --output /tmp/issue51-prepared \
  --producer-root /path/to/tracing-js-self \
  --producer-revision 6b0d8f5bd18de39c85046babed02b4afba0578fd \
  --consumer-revision 13ce89795a4625c0e124171f61a0e8bbd45d88e1 \
  --tokenizer-json /path/to/pinned/tokenizer.json \
  --source-recheck-timeout-seconds 90
```

### Explicit remaining boundaries

Reading the initial invalid-`in` error `message`, Error descriptor/prototype/
`instanceof` operations, and invocation of a derived arrow while `this` is
uninitialized are not source-qualified here. The invalid-`in` thrown outcome,
TypeError identity and name are admitted. Constructor arrows in scheduled
jobs or unauthenticated adoption continuations are refused. Proxy revocation,
nested targets or handlers, trap getters, callable targets, strict false-set
caller errors, Proxy enumeration and other trap families remain outside this
bounded profile. Captured prototype replacement and enumerability mutation,
broader collections, destructuring and abrupt iterator cleanup remain issue51
work. The standard contract gate and paired native evidence are complete for
this checkpoint; this is not a claim that the separate full combined CI gate
or the entire issue51 language scope is complete. Neither paired draft is merged.
