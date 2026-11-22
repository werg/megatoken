# Actual V8 optimizer continuation after input relocation

The source fix is native `b85a3301b6482bdf5a5a9ba79f7f356c5f824681`
(tree `8c92b6c9278d9885430b276278563762dc95e660`). The final input controls
and independent review are published at
`4f3a42c004444fbd1fe032238d2105aca3b5ace8`
(tree `325505734c9dcbc49d43512bdcff7552dbdc8bbb`). All four production
source/collector files are byte-identical between those revisions. This evidence
commit changes no executed source or tests.

## Results and exact populations

| Run | Result | Scope |
| --- | --- | --- |
| Initial actual continuation before fix, retained at `8227656c` | One failure, zero errors/skips, 14.95 s | Initial and original-location next updates succeeded; the fresh relocated process refused the changed codec locator before optimization. |
| First positive check during implementation | One pass, zero failures/errors/skips, 19.175 s | Early working-tree check, superseded by the exact published-source regression. |
| Published `b85a3301` regression | 98 passed, zero failures/errors/skips, 107.664 s | Real relocation, 55 position controls, ordinary lifecycle, target inventory/qualifiers and three ordinary non-V8 trajectory controls. |
| Initial additional refusal controls | 11 passed, two failed, zero errors/skips | Both failures were test expectation mistakes: the shard error had a capital letter, and missing policy refused during configuration construction. Production refused both unsafe inputs. Original test bytes and full raw evidence are retained. |
| Corrected refusal-only controls | 13 passed, zero failures/errors/skips, 8.473 s | Test-only corrections; the positive relocation case was deselected. |
| Final complete input module at `4f3a42c0` | 14 passed, zero failures/errors/skips, 25.949 s | Positive real continuation and all 13 refusal cases together on the published tree. |
| Independent source/identity review | 29 named checks passed, zero failures/skips | Separate controlled probe, zero model forward passes or optimizer updates; not 29 additional pytest cases. |

The 98-case and final 14-case populations overlap in one relocation case:
**111 distinct pytest cases are covered across those two runs**, not one
111-case run. Earlier checks overlap and are not added. `case-populations.json`
retains every test identity, outcome and original JUnit suite metadata. The
independent review lives in `../v8-input-relocation-independent-2026-11-22`.

Runtime: Python 3.12.14, Torch 2.10.0+cpu, pytest 9.2.0, CPU/fp32.
Pytest is outside the repository's declared `>=8.3,<9` range. These are local
checks with the actual runtime recorded, not a fully locked hosted gate.

## Defect and change

V8 exact resume and its position objective included physical codec/data paths.
An otherwise identical prepared corpus and tokenizer therefore could not move
into a restored release directory. The original failure used the real trainer,
optimizer and checkpoint loader, not a mocked resume-compatibility predicate.

A shared pure identity function now binds the verified codec fingerprint and
sorted, distinct prepared-manifest fingerprints. Physical `train_config` paths
remain recorded unchanged for provenance. The ordinary V8 compatibility check
relaxes only the two locators and only under an unchanged explicit V8 policy;
non-V8 and distributed/decoupled training retain strict locator checks. The
position objective and independent target collector use the same content-bound
identity, without reopening old paths.

Before model construction/restoration, the engine reseals every admitted
prepared manifest, including empty roots, checks the aggregate dataset identity,
and compares codec, prepared inputs, corpus identity and chunking to the saved
checkpoint lineage. Existing complete lineage, stream/cursor and RNG checks
remain. A changed path cannot authorize changed data. Old path-bound position
receipts fail the new objective comparison; they are not migrated or relabeled.
Both already-running hosted fits remain frozen at their original code.

## Actual positive continuation

The final positive case performs one initial optimizer update and two separate
next updates from that checkpoint. The second continuation runs in a fresh Python
process after the original prepared-data and tokenizer paths have been removed.
All checkpoint payload fields other than physical `train_config` are compared
recursively, with exact tensor shape/dtype/value equality. This includes model,
optimizer, RNG/cursor state, counters, lineage, target exposure and position
history. The separate transducer state also matches exactly. The checkpoint
configuration fingerprint changes as expected with recorded physical paths;
content/lineage/position fingerprints remain equal and the prior position digest
links to the initial checkpoint.

There are **three actual updates in this positive experiment**, not one three-step
trajectory. The final refusal module fixture executes one other initial update
shared by its cases: **four total actual updates in the final 14-case run**.
Those updates are not combined with earlier runs or the frozen full campaigns.

This positive experiment uses one controlled source and explicitly injected
source/byte-tokenizer admission, while exercising the real training/model,
checkpoint, optimizer and fresh-process continuation machinery. It does not
qualify production tokenizer/source admission. A separate refusal case exercises
the actual production tokenizer loader against unpinned bytes. Broader existing
production-tokenizer lifecycle cases are part of the 98-case regression with
their own scopes.

## Refusal boundaries

Eleven cases require refusal before model construction for a resealed changed
manifest, changed index, changed shard, changed family registry, missing empty
root, extra empty root, duplicate root, changed consumer policy, changed codec
fingerprint, changed window limit and changed lexical-loss weight. Two additional
cases refuse unpinned production tokenizer bytes and removal of the explicit V8
policy. No failed continuation creates a checkpoint.

The 29-check independent probe additionally verifies no hidden filesystem reads,
input-order invariance, malformed/duplicate fingerprints, seed/policy/window
binding, strict non-V8/decoupled behavior, refusal of old position history and
checkpoint metadata seals. Three controlled collector invocations traverse 48
fixture windows each (144 traversals, zero optimizer work). The full original and
relocated collector outputs agree with the original directory absent, and the
collector/objective configuration digests match.

## Retained evidence and remaining work

Raw log/XML bytes, the original red test record and the initial mistaken control
expectations are preserved. Gzipped proof JSONs retain full position receipts and
checkpoint metadata; `source-and-scope.json` records their decompressed hashes,
source blobs, runtime and reproducible final command. `files.json` seals every
file other than itself. Checkpoint/model weights are not included here.

This closes the ordinary V8 content-identity prerequisite demonstrated by these
bounded tests. It does not close the full training, serving or release work
package. Production distributed relocation, complete source/assets/recipe
qualification, a direct V8 archive and fresh restore, accepted live-history
continuation/no-repeat rollback, final paired installed-absence/default gates
and learned quality remain open. A served model reload with zero accepted
records must remain distinct from accepted application-history restoration.
