# Bounded fresh V8 package and restore qualification

Status: invocation plan only. It has not executed and grants no source, training,
release or recovery credit. Compose and freeze the producer, native policy and
native source first. Native candidate `5ecdbaa9842ab086c312a7d0be281780091df065`
contains the current Promise interpreter, preserved unified recovery guards,
content-bound input identity and strict source-order release verifier. A final
producer revision and a clean checkout of the exact native policy revision are
required before any assets are generated.

## Runtime and source identity

Use a dedicated hosted workflow with a finite 60-minute job bound, `ubuntu-24.04`,
Python 3.12.14, the producer's pinned Node release, and its hash-locked native
requirements (including Torch 2.10.0 and pytest 8.4.2). Local composition controls
use pytest 9.2.0 and are not evidence for that fully locked hosted environment.
Install the native checkout without resolving new dependencies. Verify both
clean Git heads and trees, compare producer/native V8 contract bytes, and retain
the exact dependency/runtime report. Acquire the production tokenizer at the
native declared repository/revision, then verify both its raw SHA256 and codec
identity against the exact producer policy.

## Source and prepared data

Reuse the established exact64 qualification workflow through ordinary preparation:

1. Run `tools/qualify-semantic-v8.mjs --population se-support --inputs all` with
   the exact native checkout, Python, tokenizer and native policy. Require all
   64 original sources and native receipts to pass, with the complete declared
   population and committed attempt journal retained.
2. Require the frozen registry to contain exactly 48 training, 12 validation and
   4 test sources; no duplicate case, source, component or split membership.
3. Run the production-tokenizer measurement and native packing pass over all
   64 sources. Its preparation counts remain separate from optimizer exposure.
4. Run native `prepare_semantic_v8.py` on the ORIGINAL `row.sourceArtifact`
   documents, in the established sorted source-ID input order, at 128 positions
   and 128 history chunks. Retain original/admitted documents and native receipts
   separately. Require the strict native supply report for the exact64 registry.

Do not replace the original preparation documents with post-admission copies.
The latter can have identical public records while changing the complete
source-document identity retained in preparation receipts. Package regeneration
must recover the actual prepared index order from its exact source membership,
not assume that campaign journal order is preparation order.

## Two ordinary updates and matching served export

Choose the same source-qualified training child-teacher sample used by the
existing support lifecycle driver. Invoke:

```sh
node tools/qualify-ordinary-v8-lifecycle.mjs \
  --prepared-root "$PREPARED" \
  --source-inventory "$SOURCE_INVENTORY" \
  --sample-source "$ORIGINAL_TRAIN_SAMPLE" \
  --out "$LIFECYCLE" \
  --megatoken "$NATIVE" --python python \
  --tokenizer "$TOKENIZER" --native-policy "$NATIVE_POLICY"
```

Do not add `--full-target-coverage`. Require exactly one initial update and one
fresh-process resumed update, `optimizerUpdateCounts` equal to
`{lifecycle: 2, coverage: 0, total: 2}`, both retained checkpoint/position receipts,
`fullTargetCoverageRequested: false` and `fullTargetCoverageQualified: false`.
The prepared-position inventory remains complete; the two-update observed
subset remains partial. The matching served export must come from the actual
second checkpoint, not either historical 348-update trajectory.

## Actual package and restore

Use these exact lifecycle roles:

| Role | Existing output |
| --- | --- |
| Original recipe | `$LIFECYCLE/training-config.json` |
| Concrete committed checkpoint | `$LIFECYCLE/fit/checkpoints/SE-000000002-periodic` |
| Matching served model | `$LIFECYCLE/served` |
| Prepared corpus | The same exact64 `$PREPARED` directory |
| Qualification | The original complete qualification directory, including its lock |
| Tokenizer | The verified production tokenizer file |

```sh
python tools/package-semantic-release-v8.py \
  --producer-root "$PRODUCER" --megatoken "$NATIVE" --python python \
  --qualification "$QUALIFICATION" --prepared "$PREPARED" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$LIFECYCLE/fit/checkpoints/SE-000000002-periodic" \
  --served "$LIFECYCLE/served" --recipe "$LIFECYCLE/training-config.json" \
  --out "$PACKAGE" --source-recheck-timeout-seconds 30

python tools/restore-semantic-release-v8.py \
  --producer-root "$PRODUCER" --megatoken "$NATIVE" --python python \
  --archive "$EXACT_CREATED_ARCHIVE" \
  --expected-archive-sha256 "$INDEPENDENT_ARCHIVE_SHA256" \
  --out "$FRESH_RESTORE" --evidence-out "$RESTORE_EVIDENCE" \
  --source-recheck-timeout-seconds 30
```

Read the actual archive path from the package result and independently hash it;
do not guess a filename. Restore into an absent, separate destination. Both
commands must verify the complete immutable campaign snapshot, exact source and
asset inventories, original and portable recipe binding, all-source native
repreparation, complete prepared semantic/token/mask/index equivalence, actual
checkpoint/position/cursor/lineage identity and loaded model/transducer state.
The package/restore stages perform zero optimizer updates and decoder forwards.

The original native report precedes its own archival inclusion and the final
outer manifest. Keep its original input inventory intact. Independently verify
all final archive members and compare the explicitly named semantic summaries
after restore; do not compare physical temporary paths, timings or the earlier
whole-input inventory digest to the later expanded payload.

## Retention and claim boundaries

Upload all original stage logs, JUnit reports, campaign journal, original
source/admitted/native receipts, tokenizer identity, prepared data, both
checkpoints, served bundle, package/archive receipts, created archive and restore
proof. Set `include-hidden-files: true`; the live campaign lock is part of the
snapshot protocol and must survive transport. Record artifact byte count and
SHA256, workflow attempt, exact pair and runtime. Keep qualification,
preparation, two-update training, package and restore denominators separate.

This bounded result can establish fresh source/data/checkpoint/export/archive
consistency under the frozen pair. It cannot establish complete optimizer
exposure, learned quality, default cutover, portable container execution or
accepted live-history recovery. Historical full-pass artifacts remain unchanged
and retain their original source and runtime identities.
