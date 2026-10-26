# Canonical semantic execution

Megatoken consumes **one** model-facing JavaScript execution representation:
`semantic-execution/5` / `semantic-trace/5` in the `selected-harness/2`
envelope. It is produced by the source-grounded `tracing-js-self` teacher
and interpreted by the selected state controller. There is no grouped
v3 importer, explicit step-for-step learner, or output-only answer mode.

A trace is a **reasoning witness**, not a bijective mapping of interpreter
instructions. Pure evaluated source terms have ground values at their
execution point, e.g. `next = (current + 1) -> 8`. Source-bound cells keep
the actual source name, while anonymous pure intermediates need no name.
Observable input, output, mutation, effects, event/callback arrivals,
Promise/await state and completion must remain causally exact.

## Authority boundaries

The selected JavaScript controller validates local computation against
**previously committed public state**. Only observed environment records
can provide native results and external arrivals. A generated request
contains evaluated arguments and a source-grounded authorized effect
site; Megatoken independently materializes its payload and checks its
native event ownership before dispatching exactly once.

The private interpreter journal, exact replay, independent JavaScript/V8
reference, and pinned native effect receipts exist **solely to verify
correctness**. They do not become additional learner targets. Native
meridian/tape framing is a transport and loss-mask mechanism, not another
trace grammar.

The native importer is
`megatoken.data.semantic_harness_v5.adapt_selected_harness_v5`.
`megatoken.data.semantic_request` validates the authorized external
request without importing a historical group-shaped corpus.
`SelectedSemanticInference.select` always selects exactly one complete
`semantic-trace/5` record with the source-grounded compact expression
budget (up to eight operators and 32 expression nodes). Obsolete
`explicit`, `outcome-only` and unknown representations fail closed.

## Provenance and admission

A locally verified trace is **not** admitted model training data. Actual
source, effects and whole-task completion require independent reference
agreement; public witness replay and a source-bound private certificate
must also pass. Admitted native episodes additionally need the production
tokenizer, visibility/loss-mask checks, native capability validation and
whole-source-family split integrity.

The shared contract is checked byte-for-byte between the pinned
tracer and Megatoken:
`src/megatoken/data/contracts/semantic-harness-v5.json` and
`src/megatoken/data/contracts/host-operations-v4.json`.
Changing the witness or fixture digest requires regenerating the
producer declaration, updating the paired consumer, and recollecting
teacher data. Old model checkpoints and data are never silently
reinterpreted.

The v5 execution contract binds an optional unsigned 32-bit seed to program
publication and carries the committed random-generator state in every selected
context snapshot. Each standalone random witness must match the next seeded
draw, and a retained realm must preserve the same seed. A snapshot's random
state is checked against the program seed and all preceding selected draws;
well-formed but forged state is rejected.

## Prepare one source curriculum

From the Megatoken checkout, with a pinned producer checkout adjacent:

```sh
python scripts/validate_semantic_corpus.py \
  --corpus /work/semantic-teacher \
  --output /work/semantic-prepared \
  --tracer-root /work/tracing-js-self \
  --tokenizer-json /work/tokenizer.json
```

For a complete source recipe (`training-v4` or `benchmark-v1`):

```sh
python scripts/prepare_semantic_experiments.py \
  --corpus /work/semantic-recipe \
  --recipe training-v4 \
  --tracer-root /work/tracing-js-self \
  --tokenizer-json /work/tokenizer.json \
  --producer-revision TRACER_COMMIT \
  --consumer-revision MEGATOKEN_COMMIT \
  --output /work/semantic-experiment
```

This prepares ordinary/pure source calculations, effectful traces and
real train/validation lifecycle child data using the **same witness**.
The source-family split and ancestry are frozen before variants or
callbacks are collected. The preparation receipts prove software and
native-data gates; they do not prove learned correctness or perform
optimization.

The `SC` and `SE` training stages distinguish **pure execution from
effectful interactive execution**, not alternative trace representations.
`configs/semantic/experiment-plan-v4.json` contains the currently
supported cohorts and fixed example/token/compute budgets.
`scripts/plan_semantic_training.py` binds immutable admitted dataset,
tokenizer, seed, donor, engineering and checkpoint inputs; it does not
launch an optimizer. A training run may claim success only after its
held-out free-running model evaluation against exact observable results
and effects.

## Verify

```sh
python -m pytest -q tests/test_semantic_contract_v5.py \
  tests/test_semantic_harness_v5.py \
  tests/test_semantic_data_pipeline.py \
  tests/test_semantic_model_adapter.py \
  tests/test_semantic_orchestration.py
```

Dedicated CI checks the real tokenizer, producer/native contract parity,
actual source-certified episodes, controlled shared-process coordinator
execution, failed native requests and replay/checkpoint safety.
For a useful read-only success/cost breakdown use
`scripts/analyze_semantic_evaluation.py`.

Historical multi-arm runs remain accessible through their original Git
commits, not a second executable producer or runtime adapter. The current
code must not load, serve or silently convert them.
