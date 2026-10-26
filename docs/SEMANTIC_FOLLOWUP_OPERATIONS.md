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

First freeze each held-out cohort from the same admitted recipe. These
commands use the default test-only split; validation families remain
available for training-time model selection.

```sh
node /work/tracing-js-self/tools/freeze-selected-evaluation.mjs \
  /work/semantic-recipe /work/common-support-declaration.json --common-support
node /work/tracing-js-self/tools/freeze-selected-evaluation.mjs \
  /work/semantic-recipe /work/effectful-declaration.json
python scripts/evaluate_semantic.py freeze \
  --declaration /work/common-support-declaration.json --output /work/common-support-cohort.json
python scripts/evaluate_semantic.py freeze \
  --declaration /work/effectful-declaration.json --output /work/effectful-cohort.json
```

Preparation writes `/work/benchmark-prepared/dataset-bindings.json` with
only `datasets` and `codec`. Combine it with the frozen cohort paths and
the actual production inputs before invoking the planner. In the example
below, `/work/production-launch.json` is an existing deployment-specific
training configuration containing the immutable `hindsight_targets` path,
ordered `engram_imports` paths, and full `decoupled` coordinator settings.
Use the real imported assets and coordinator URL; the `PIN_*` values in
the checked-in seed templates are unconfigured declarations.

```sh
python - <<'PY'
import json
from pathlib import Path

prepared = json.loads(Path('/work/benchmark-prepared/dataset-bindings.json').read_text())
production = json.loads(Path('/work/production-launch.json').read_text())
bindings = {
    'datasets': prepared['datasets'],
    'codec': prepared['codec'],
    'cohorts': {
        'common-support': '/work/common-support-cohort.json',
        'effectful-tracks-events': '/work/effectful-cohort.json',
    },
    **{key: production[key] for key in ('hindsight_targets', 'engram_imports', 'decoupled')},
}
Path('/work/complete-semantic-bindings.json').write_text(json.dumps(bindings, indent=2) + '\n')
PY
```

```sh
python scripts/plan_semantic_training.py \
  --bindings /work/complete-semantic-bindings.json \
  --engineering-receipt /work/cross-repo-engineering.json \
  --output /work/semantic-training-plan
```

The planner checks the frozen cohort identities, stage supply, tokenizer,
and configuration shape. The preparation file alone cannot satisfy this
interface. Keep the combined binding file and `matrix.json` with the
engineering receipt so every planned run has inspectable inputs.

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
