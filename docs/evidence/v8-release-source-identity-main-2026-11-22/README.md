# Expected source identity on the current native main baseline

This is the narrow adoption of three previously reviewed files onto exact native
main `ddda9a7339297e6074765a001743e2bb569e0eb9`. Source commit
`74cb4c5117d554b40882a2bf415f1863fd1c32a7`, tree
`110ba702d2c17513936cd0803820e024d2165876`, is on the isolated branch
`codex/native-v8-release-source-identity-main-20261122`. Local source commit
`eb5cf94d9b4e31b0c17c1b462fa68bc9818c2f83` has the same tree.

## Exact scope

Only the release verifier, its test module and its documentation changed. All
three blobs are exact copies from source commit
`c253a1e7770ee569d1b2022efadb471a1cc98da2`; their baseline blobs matched the
parent of that increment before adoption. Each source row requires exactly
`caseId`, `sourceId` and `path`, unique nonempty expected source IDs, and an exact
match to the artifact's dataset-binding source ID before preflight or fresh
source preparation. The case label remains independent and identities are not
rewritten.

Current main's two additional Promise tests, workflow peer pin and four
frozen-contract evidence files are byte-identical before and after this change.
The Promise tests were not rerun here. No other runtime, payload, checkpoint,
prepared-order or recovery source changed. `source-map.json` records the three
old/new blobs and all six specifically preserved main files.

The earlier candidate branch at
`200da926eef6c90670e5837797c1e72028912741` remains unchanged. Its 105 evidence
files absent from this main baseline were deliberately not copied in this
source increment. Their exact paths and blobs are retained in the map for a
separate evidence-only adoption. The full read-only comparison is retained as
`prior-candidate-comparison.json`: 1,111 identical files, five differing common
files, four main-only evidence files and 105 earlier-candidate-only evidence
files. Main's extra Promise tests account for one of the five common differences;
the peer pin accounts for another, and the three release files are the remainder.

## Focused qualification

The complete affected verifier/order test module passes **69 cases**, with zero
failures, errors or skips, in **13.616 seconds**. That is the prior 62 cases plus
seven explicit source-identity controls. Raw output, JUnit, the exact command
and runtime are retained in `controls.log`, `controls.xml` and
`qualification.json`. Ruff formatting and checks pass (`ruff.log`).

One existing controlled byte-codec test performs one real optimizer update and
checks the checkpoint, position history, export and served tensors. The new
identity checks perform zero updates. This test module performs no fresh source
replay and does not qualify a production-tokenizer package/restore round trip.
No tests outside this affected module ran; earlier large cohorts retain their
own exact source identities and limits.

The existing available runtime is Python 3.12.14, Torch 2.10.0+cpu and pytest
9.2.0. Pytest 9.2.0 is outside the declared >=8.3,<9 requirement. No environment
was modified to hide that difference; the final hosted check must still use its
pinned pytest 8.4.2 environment.

## Paired producer requirement

Producer `7c20124f3d89dfe1a8d5e892238e7b7f92589720` emits two-field source rows.
Those rows intentionally do not satisfy this native revision's three-field
contract. The matching producer change must populate `sourceId` from the
independently verified `taskArtifactBinding.membership.sourceId`, require
non-null unique membership in this prepared release profile, and preserve
`caseId` separately. It must not infer a physical source ID from the case label
or rewrite a registry.

That producer already independently binds the original task/source/registry and
the recorded native admission, and rebuilds the descriptor from a fresh campaign
snapshot on package and restore. This added native assertion does not imply that
the current package path still accepts the earlier task substitution. Keep the
module-capable producer authority and complete the small paired descriptor
change before launching the final package gate.

No main branch, running workflow, historical fit or original artifact was changed.
