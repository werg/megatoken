# Operational semantic execution qualification

This runbook uses the **sole canonical source-grounded v4 witness**.
All evaluation and training use `semantic-trace/4` selected execution.
Different source families or seeds are useful, but changing the trace
representation is not an experimental parameter.

## Source/native preparation

Generate the selected trace corpus with the pinned
`tracing-js-self/tools/generate-effectful-training.mjs`, supplying the
real Megatoken native importer and tokenizer and requiring native
admission. For the complete `training-v4` or `benchmark-v1` curriculum,
prepare its single source-manifest:

```sh
python scripts/prepare_semantic_experiments.py \
  --corpus /work/semantic-recipe \
  --recipe benchmark-v1 \
  --tracer-root /work/tracing-js-self \
  --tokenizer-json /work/tokenizer.json \
  --producer-revision TRACER_COMMIT \
  --consumer-revision MEGATOKEN_COMMIT \
  --output /work/benchmark-prepared
```

This re-executes the independent source verifier and native admission
for every case. It retains family/split coverage and normalizes no
alternate teacher formats. A refusal remains a refusal. The
`semantic-experiment-preparation/4` receipt records the accepted
source cases, lifecycle runs and every required stage-supply check.
The `benchmark-recipe-preparation.json` report is an optional,
read-only coverage summary, not different learner data.

## Training preflight

`scripts/plan_semantic_training.py` constructs the canonical seed
and budget matrix using `configs/semantic/experiment-plan-v4.json`.
Bind a frozen cohort, actual admitted dataset, tokenizer identity,
source and consumer Git commits, original native engineering receipt,
immutable teacher structure targets, donor imports, and coordinator
endpoint. Do not substitute historical experiment settings for
current data.

```sh
python scripts/plan_semantic_training.py \
  --bindings /work/semantic-experiment/dataset-bindings.json \
  --engineering-receipt /work/cross-repo-engineering.json \
  --output /work/semantic-training-plan
```

The supplied bindings must also contain the referenced frozen
cohorts and production launch inputs; the dataset-bindings file
created by preparation alone is **not sufficient** to start training.

`scripts/qualify_semantic_launch.py` performs the existing
read-only launch qualification, including immutable checkpoint and
coordinator checks. It does not run an optimizer, create missing
checkpoint bytes or silently admit a corpus.

## Controlled versus learned evidence

The real selected bridge, native input/callback delivery and source
conformance are checked before any learned-model claim. Controlled
selections exercise the whole stack but have `learnedSuccess: null`.
Initialized fixed weights demonstrate that an actual model was called,
not that it can execute JavaScript.

For a held-out evaluation use the frozen cohort and the current
`scripts/evaluate_semantic.py` selected-runner path, with
`representation="compact"`. The success criterion is correct
completion and reachable state plus ordered effects, with no
unauthorized effects, future-observation leakage or untracked
fallback. Reasoning-trace spelling and intermediate derivation
bytes are not an equality target.

Use `scripts/analyze_semantic_evaluation.py` for family-level
success, trace validity, empirical cost percentiles and explicit
missing-data accounting. No previous output-only or explicit
ablation corpus is prepared or served.

The original multi-arm follow-up receipts can be recovered
from their pinned historical Git revision if needed for historical
audit. They are deliberately not runnable inputs to the current
semantic pipeline.
