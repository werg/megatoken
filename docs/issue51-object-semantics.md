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
