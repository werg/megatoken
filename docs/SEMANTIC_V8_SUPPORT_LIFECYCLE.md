# Ordinary optimizer qualification on the frozen V8 support corpus

The producer's ordinary lifecycle command can now train from the exact prepared
64-source support directory. It uses the normal trainer, two optimizer updates
across a fresh process, ordinary export and the real JSONL serving/restoration
collector. It is a bounded execution diagnostic, not a learned-quality study.

## Canonical inputs and refusal boundaries

The prepared directory must already pass ordinary V8 admission and the strict
production behavior-supply gate. Its family registry and immutable source-ID
inventory must describe exactly 64 distinct sources, assigned once each to the
frozen 48/12/4 train/validation/test split. The support inventory schema is
`semantic-v8-source-population/1`; its registry digest must equal the recomputed
complete registry digest. The collector uses the native preparation module's
canonical source-binding verifier, not a recursive metadata search.

The selected serving sample must belong to training. Its source/corpus/family,
component/split, public/context digests and dataset binding must verify against
the prepared registry. Its complete prepared public/context artifact must match,
including the source sidecar for a lifecycle teacher. Every prepared episode
must name an expected source exactly once in its declared frozen split.

The normal checkpoint collector still checks step 1 to step 2 progression,
optimizer moments, changed master weights, training lineage, served lineage and
fresh-process next-selection equality. No decoder or accepted selection is
injected. Source supply, optimizer execution and learned execution remain
separate claims.

## Reproduce through the producer

From a clean, exactly pinned producer checkout with its matching native checkout:

```sh
node tools/qualify-ordinary-v8-lifecycle.mjs \
  --prepared-root /absolute/path/to/v8-support-prepared \
  --source-inventory /absolute/path/to/v8-support-source-ids.json \
  --sample-source /absolute/path/to/source-qualified-training-child-teacher.json \
  --out /absolute/path/to/new-support-lifecycle-output \
  --megatoken /absolute/path/to/megatoken \
  --python python \
  --tokenizer /absolute/path/to/pinned-tokenizer.json \
  --native-policy /absolute/path/to/exact-pair-policy.json
```

Output must be new and outside both source checkouts and the prepared directory.
The producer snapshots the selected admitted artifact and inventory before using
them. It derives the training policy from the same complete support registry and
calls this native collector with `--expected-source-ids`. Support mode refuses a
second synthetic source selection or an extra orchestration-teacher flag.

## Evidence and remaining work

[Validation run 44987781224](https://github.com/werg/megatoken/actions/runs/44987781224)
checked the two candidate Python files exactly as committed in validation
revision `a0cfb1d2c2fc152a1716e655619c7ef93c65d82b`. Ruff and formatting passed;
all 11 focused source-binding/supply controls passed, zero skipped. The first
format gate found an unused fixture loop variable; its failure is retained.
The real paired 64-source optimizer route remains pending at this commit.

Two optimizer updates do not prove that every behavior target was consumed,
accepted-program recovery or learned usefulness. Reports retain
`fullLaunchQualified: false` and `learnedSuccess: null`. Production launch
recipes, durable external model asset publication, multi-worker/distributed
recovery, held-out quality/cost and default cutover require their own evidence.

## Scored-target exposure

The collector now also requires the actual optimizer target-exposure census in
both checkpoints, an exact parent digest from step one to resumed step two, and
preserved cumulative source/target/record counts. [Definitions, proof and limits](SEMANTIC_V8_TARGET_EXPOSURE.md).
