# Issue 51: object existence and mutation-sensitive enumeration

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
python -c 'import json; d=json.load(open("/tmp/issue51-cohort.json")); print("\\n".join(json.dumps(x["artifact"]) for x in d["cases"] if x["artifact"] is not None))' > /tmp/issue51-artifacts.jsonl
PYTHONPATH=src python scripts/prepare_semantic_v8.py \\
  --input /tmp/issue51-artifacts.jsonl --output /tmp/issue51-prepared \\
  --producer-root /path/to/tracing-js-self \\
  --producer-revision c3e1a55f19eb04fc17957dd8fdea01031ca35ecd \\
  --consumer-revision c3df6a32436233f714b365c3a7b2f3e5ce967b5c \\
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
