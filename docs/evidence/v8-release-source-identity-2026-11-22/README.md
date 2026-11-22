# Explicit expected source identity in native release descriptors

Source revision: `c253a1e7770ee569d1b2022efadb471a1cc98da2`, tree
`9e7f6b12dcd26232ffeba70869c502d4190b4951`, on
`codex/native-v8-release-composition-20261122`. Its parent is the retained
composition proof `6d0bf38a3a43d402d134acbc6c671d830f1f4bac`. The local source
commit `36898d695db33c300828b7835bd038c1bc9c21a9` has the identical tree.
`source-map.json` records all three changed paths and their old/new blob IDs.

## Corrected boundary

Each descriptor source row now has exactly `caseId`, `sourceId` and `path`.
The expected source ID must be a nonempty string and unique across the release.
It must exactly equal the supplied artifact's dataset-binding source ID before
native preflight or fresh preparation starts. The expected value is retained in
the report; the artifact cannot silently supply a replacement identity.

The case label remains independent. The producer can legitimately construct a
physical source identity by appending a digest to a candidate label. The
positive control preserves such an identity and separately retains the label.
`positive-identity.json` records an actual call to the producer's unchanged
identity-construction function and an independent native digest calculation:
both yield `declared-case:95a869417b705f27c48736609f928aa1f1aaa2c3`.
This was an identity calculation, with zero source replays and zero updates.
The producer checkout had unrelated release-tool edits, retained in the report;
this calculation is not a clean paired-release qualification.

The producer still must bind this expected identity to the frozen original
task, source, arguments, family and ancestry. Native replay receives the
artifact itself and does not replace that selection check. This increment
does not equate every case label with a source ID or rewrite existing IDs.

## Retained failure and focused result

The historical read-only membership review on `5ecdbaa9842ab086c312a7d0be281780091df065`
is copied unchanged as `historical-5ecd-membership-review.json`. It showed that
the unchanged rectangular-capacity artifact could be accepted with a
rectangle-perimeter case label by the native membership/order readers. It
performed no source replay, optimization or whole-release verification.

For a reproducible regression, `red-candidate.patch` is the exact change against
the parent's tree with the new descriptor shape, fixtures and tests present
but the artifact identity comparison still absent. Its focused test reached a
controlled preflight sentinel with a different physical source. `red.log` and
`red.xml` retain **one expected failure**, zero errors/skips, in **0.351 seconds**.
The earlier repository and codec setup is explicitly controlled; no fresh
source process ran. This is a boundary regression, not a failed real package.

With the exact source revision above, the whole affected release-verifier
module passes **69 cases**, zero failures/errors/skips, in **13.496 seconds**.
That includes the prior 62 cases and seven new cases covering missing, empty,
non-string, duplicate and extra source-row fields, the early mismatch refusal,
and the independent digest-suffixed identity. The duplicate check uses distinct
case labels and paths so another uniqueness rule cannot mask it. The mismatch
check exercises the top-level verifier and checks unchanged input bytes.
The existing two-source prepared-order and omission/duplication controls remain
in this run. Raw output and JUnit are `controls.log` and `controls.xml`.

The module includes one existing controlled byte-codec checkpoint/export test
that performs **one real optimizer update** and checks actual checkpoint,
position history and served tensors. It does not use a production tokenizer or
perform fresh source replay. The new identity controls perform no updates.
The previously recorded 398-case composition union was not rerun or relabeled.

Ruff formatting and checks pass (`ruff.log`). The local runtime is Python
3.12.14, Torch 2.10.0+cpu and pytest 9.2.0. Pytest 9.2.0 remains outside the
declared >=8.3,<9 requirement; the future hosted environment pins 8.4.2.
`qualification.json` records the command, exact counts and raw-file hashes.

## Remaining qualification

A fresh, complete source campaign with the producer's task-binding check,
production tokenizer, ordinary checkpoint, matching served model, and actual
package/restore round trip remains pending. This proof does not qualify that
pair, learned quality, complete training exposure, accepted service history or
default cutover. No historical fit, artifact, source registry or main branch
was changed by this increment.
