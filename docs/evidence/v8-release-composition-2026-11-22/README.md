# Current native V8 release composition and bounded controls

The frozen source is `5ecdbaa9842ab086c312a7d0be281780091df065`
(local `f95604bff5f2ae856c38f116618113f99e7d25c0`, identical tree
`3d09866da2fbcabdc2f1eb00d03be9caa17f1232`), on isolated branch
`codex/native-v8-release-composition-20261122`. Main and both historical full
training campaign branches were not changed.

## Exact composition

The base is unified native `15844c00c095849755ca6f0e1d207f7db79a9264`.
Three separately published increments copied only explicit reviewed paths:

| Increment | Published composition | Exact imported source | Paths |
| --- | --- | --- | --- |
| Promise adoption | `918f70278364105389d4d420b0dd2fd0182d0f8c` | `03a9e33866207b5b0c4105297c6050dacadcf85f` | 14 |
| Content-bound ordinary resume | `8b2103f4fca6abb80dc99b9ad73bc6eee07e8e9d` | `7741c003c8887da0000f7912ab6d6cdbee9ec3a1` | 35 |
| Strict release verifier | `6c70b40a437ee986b42c2d86b8a450d48fe32fa2` | `3fb3370dd3c10dbc9309fd2870760cf87751b418` | 31 |

All existing destination blobs matched their declared starting versions; new
paths were absent. The original 1,102 unchanged baseline files, all baseline
evidence, and all seven named guard/recovery implementation and test paths are
preserved exactly. The normalized child-bootstrap guard and current causal
create API were not replaced by older variants. Promise payloads and public
contract stay exact to the qualified source. The ordinary resume change stays
exact to its four `b85a330` runtime files and final `4f3a42c` tests/proof.

The source-order red control is separately published at
`1da339f97509dfdd67c27233488064a845e5c8e7`. It reads real two-source prepared
rows, presents the descriptor in reverse order, and demonstrates that the old
verifier supplied the wrong input order. Its subprocess is intercepted to
inspect arguments; it is not fresh source qualification and performs no updates.
The direct correction at `5ecdbaa` changes three verifier source/test/doc paths.
It checks complete unique source sets, lengths and splits before reconstruction,
then sends original source documents in the actual prepared index order. It
neither rewrites nor sorts the stored corpus. All 62 verifier cases pass,
including six refusal variants for duplicated, missing, invented or differently
split membership. The earlier 55-case source proof remains historical evidence.

## Qualified union and retained failed attempts

The exact final test population was independently collected at `5ecdbaa`:
**398 pytest cases**. Every one has a retained passing execution, with no missing
or unexpected identities. The original recovery/Promise run additionally
reported **99 passed subtests**. This is a qualified union, not one clean
398-case execution:

| Execution | Exact source | Passed cases | Failures | Setup errors | Skips |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial combined selection | `6c70b40` | 316 | 0 | 75 | 0 |
| Failed-case retry plus the complete current verifier module | `5ecdbaa` | 102 | 1 | 0 | 0 |
| Normal pytest entrypoint for the coordinator/export/reload case | `5ecdbaa` | 1 | 0 | 0 | 0 |

Twenty-one passing verifier cases overlap the first two executions and are
counted once. All 75 initial errors contain `No space left on device` during
pytest temporary-directory setup, before those test bodies ran. Completed,
clean worktrees were removed separately to recover space; original hosted ZIPs,
extractions, unique full-training checkpoint and active source/proof paths were
preserved. The next failure came from invoking pytest programmatically through
stdin: multiprocessing tried to reopen `<stdin>` in the coordinator child.
The unchanged test passed with the normal `python -m pytest` entrypoint.
Both failed attempts are retained in full; failures were not relabeled as passes.

Between `6c70b40` and `5ecdbaa`, the only runtime/test changes are the verifier
script and its test module; the other change is its documentation and new red
proof. Every other runtime and test blob remains identical. The complete new
verifier module passed on the final source. `qualified-union.json` retains the
case-level attribution, raw JUnit attributes and this source-equivalence boundary.
The JUnit initial `tests=490` attribute includes 99 subtests alongside the 391
ordinary testcase elements; the ordinary cases and subtests are kept separate.

Local runtime is Python 3.12.14, Torch 2.10.0+cpu and pytest 9.2.0. The latter
is outside the declared `>=8.3,<9` test range and differs from the future hosted
lock (pytest 8.4.2). The pinned production tokenizer was supplied for the
coordinator/export/reload and policy controls, with no tokenizer-availability
skips. Controlled source admission and byte codecs remain explicit fixtures
where the individual tests declare them.

## Retained proofs and remaining qualification

Original logs/XML are retained, some with deterministic lossless gzip. Their
uncompressed byte counts and SHA256 values are in `raw-artifacts.json`; no
original content was rewritten. Selected actual controlled proofs are retained
for complete prepared positions with a stage transition, fresh-process ordinary
resume and independently repeated next steps after moving identical inputs.
Their separate trajectories do not form one combined training run. The release
verifier itself performs zero optimizer updates or decoder forwards.

The source map, final collection, selectors, case attribution, import denial
probe, Ruff result and original red/green order logs are indexed by `files.json`.
The import probe covers fresh module imports only and is not complete physical
removal of older implementations. No final-pair source population or production
release assets were generated here, and no complete optimizer exposure,
learned quality, default cutover or accepted live-history release claim is made.

`release-invocation-plan.md` specifies the separate exact64, production-tokenizer,
two-update package/restore gate. It remains an unexecuted plan and must wait for
a frozen producer policy and exact native source pin.
