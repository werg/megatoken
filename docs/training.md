# Training and operations

## One representation, two directions of training

The recurrent input is a `[batch, 32, 512]` facet tensor. The core applies six residual
body stages, a non-residual slot/channel remix, and ten residual prediction stages.
The **completed prediction**, after those ten stages, is the Gaussian mean. One sampled
raw facet tensor drives the deterministic realizer and the next recurrent iteration.
The byte encoder `E(bytes)` builds the same shape through ordered byte/n-gram processing
and repeated halving. See [architecture.md](architecture.md) for the tensor graph.

`TrainConfig.representation_mode` selects `bootstrap` or `recurrent`. The value `auto`
resolves to bootstrap for B0 and recurrent for every other supervised phase. Checked-in
production YAMLs make the choice explicit. Prepared loss plans record their mode;
forward execution rejects a different mode or a changed denominator.

### Bootstrap: the byte representation initially provides the target

For a target chunk `x` following the current input/context:

```python
processed = core.process(state, control=control, effort=effort)
prior = facet_distribution(processed.facets)
z = prior.sample(generator)
e = core.encode_chunk(target_bytes, lengths)

facet_prediction = -prior.log_prob(e.detach())
lexical = decoder.teacher(z, target)
reconstruction = decoder.teacher(e, target)
next_state = core.replace_facets(processed, e)
```

This pseudocode names the three objectives; raw and structured training use the same
prepared target machinery. Gaussian NLL trains the completed predictor toward the
encoded target, including its learned scale. Reconstruction trains E and the shared
decoder. Teacher forcing substitutes E as the next input, without an extra learned
writer. The next chunk can also train that earlier E through the live bootstrap window.
The target is never supplied to the current prediction or used as a control.

### Recurrent towers: the byte representation follows the predictor

```python
processed = core.process(state, control=control, effort=effort)
prior = facet_distribution(processed.facets)
z = prior.sample(generator)
e = core.encode_chunk(target_bytes, lengths)

lexical = decoder.teacher(z, target)
encoder_alignment = (e.float() - prior.mean.detach()).square().sum()
next_state = core.replace_facets(processed, z)
```

The sampled facets remain live through consecutive predictions. Later decoder losses
can train earlier recurrent computation. There is one decoder pass, no reconstruction
objective, and no E-target objective pulling the predictor toward a byte representation.
The follower target is the completed prediction mean, avoiding arbitrary Gaussian noise
in the encoder's target. Only E receives encoder-alignment gradients.

Observed byte inputs and the frontend features used for causal controls are detached
from main-model losses in recurrent mode. The control encoder and the body remain
trainable. Encoder and main-model owners also have independent gradient clipping: a
large E alignment error cannot reduce a core update through a shared clipping norm.
Their parameters retain ordinary independent FP32 Adam moments in one optimizer.

Default loss weights are defined once in `megatoken.contracts`: lexical 1.0,
reconstruction 1.0, encoder alignment 1.0, facet prediction 0.1, structure 0.25,
binding 0.5, placement 0.25, terminal 0.1, router balance 0.005, and router z-loss 1e-4.
An inactive objective has a zero denominator. These weights are starting settings;
held-out causal generation and executed tasks determine whether they should change.

## What one optimizer transaction does

Training, serving, collection, and replay use the same `build_control_payload` and
`encode_controls` helpers for public seed bytes, causal references, and tool schemas.
Only tool JSON is canonicalized; seed/reference bytes remain exact. References require
`external_evidence` or `earlier_history` provenance. Empty control rows are neutral even
when batched with nonempty controls.

The JSONL generation request accepts `control_seed`, `control_reference`, and
`reference_origin`; its `tools` also condition the core. The numeric `seed` remains the
random generator seed. Python callers use the matching `InferenceEngine.generate`
arguments. Version-3 action tapes store the source control bytes for every draw. Replay
re-encodes those bytes under the current weights instead of trusting stale encoded
features, and rejects an explicitly different control. Control changes occur upstream
of the one facet draw; deterministic realization of a fixed raw tensor is unchanged.

The trainer selects independent episode windows and plans the eligible loss denominators.
Compatible raw-record event schedules batch together without inspecting target content
or expert choices. Typed complete actions and windows containing native action tapes use
single-episode execution so their causal event sequence remains explicit.

Each window reconstructs its complete earlier prefix under current weights with no
backward graph. Offline external records use the shared E observation path. Native
student/generated history replays its recorded raw facets, including invalid and empty
output events, at the recorded physical-record cursor. A matching generated record is
not re-encoded. Recorded facets for a future scored target are never inserted into its
prefix. Old native metadata without a causal cursor fails explicitly.

The graph remains live through selected observations and predictions in the new window.
After backward, global gradient reduction, and one optimizer update, runtime state is
discarded. Both facets and historical KV are rebuilt after parameter changes; stale
model or routing versions fail explicitly. The prefix boundary is the explicit TBPTT
boundary. Encoder teacher anchoring at an offline window boundary is a training choice,
not a second network state or a serving reconciliation step.

The KV cache contains processed earlier input bodies. `state.facets` is the current
unprocessed input. `process` consumes it once, stores body context, and predicts the
next input. `observe` consumes a pending prior input through the same body, then installs
E of the new observation. This preserves system→user and generated action→tool chronology.
A historical native tape uses `advance_raw`, which shares that body transition but skips
prediction work whose output would be discarded.

`max_history_tokens` bounds the differentiable **new-window** transcript. It does not
truncate the full reconstructed episode prefix. `scored_decisions` limits the number
of scored logical groups and `max_window_records` limits physical records. Logical tool
actions remain indivisible. LC extends a selected window to an annotated reader, within
its explicit hard dependency budget. A logical action with multiple physical chunks can
therefore produce more facet predictions than its one decision count.

Production tower starting limits are 4 scored decisions in P1, 8 in M1/M2/S/D, and 16
in LC; B0 begins with 2 teacher-forced decisions. The accompanying record limits and
history budgets still apply. These are configurable graph bounds, not automatically
ramped schedules or a claim that every source supplies that many consecutive targets.
Use the recurrent benchmark and a bounded phase pilot when increasing them.

Accumulation targets `accumulate_scored` or `accumulate_ingested` globally, with
`max_accumulation_windows` as a hard iteration bound. Complete windows can overshoot a
budget; the actual work is reported. `max_steps` limits additional optimizer updates in
this invocation and is separate from the phase's work ceiling.

## Distributed data and gradients

Ranks traverse a common seeded episode permutation in synchronized blocks. Each rank
owns one episode per block. If its episode has fewer windows, it returns explicit idle
slots until the longest episode finishes. A final incomplete block also uses idle slots.
No rank independently rolls over an epoch and repeats another rank's work. A preparation
failure is propagated collectively before peers continue.

Mixture exposure is reduced across ranks before choosing the next stream. Every rank
therefore selects the same stratum, including when its last local window was idle or
unscored. Empty strata fail; nonempty strata with fewer episodes than ranks are valid.
Versioned cursors include topology, seed, epoch/block/slot, dataset scope, window policy,
and codec identity. Resume inside a padded tail preserves the subsequent global order.

For each loss, the objective is the sum of eligible numerators divided by its global
eligible count. Local batch means are never averaged. Lexical positions, structural
choices, facet coordinates, and routing examples use their own denominators. Gradient
accumulation and rank count preserve that main objective. The nonlinear router load
auxiliary depends on its execution batch, is normalized across examples, and is reported
separately. Gradients are summed once after local accumulation, including parameter
owners used on only one rank.

## Precision, ownership, and kernels

Production defaults to BF16 model weights and tensor arithmetic. AdamW has FP32 masters,
moments, and accumulation buffers; post-accumulate hooks move each BF16 leaf gradient
into its FP32 buffer before the next backward. Normalization statistics and Gaussian
probability calculations use FP32. Sampled raw facets retain FP32 coordinates so tape
serialization/replay does not round the policy action. Historical KV uses the model's
arithmetic precision.

Adam defaults are β₁=0.9, β₂=0.95, ε=1e-8, matrix weight decay 0.1, and clipping at 1.0
per active gradient domain. Embeddings, lookups, norms, and biases do not decay.
Lookup/control owners use half the base learning rate. Repeated module calls share one
parameter owner, one FP32 master, and one optimizer slot. Non-finite gradients abort
before an update.

```bash
megatoken inspect --model recurrent32 --codec artifacts/codec --owners
megatoken --json-output runs/recurrent-benchmark.json benchmark \
  --model recurrent32 --codec artifacts/codec --device cuda --precision bf16 \
  --backward --representation-mode recurrent --prediction-chunks 8 \
  --batch 8 --capacity 256 --history-records 128 --steps 20
```

The primary model has 477,498,706 permanent served parameters and no private training
heads. `inspect` performs a meta-device owner census without allocating full weight
storage. Activations, gradients, optimizer state, and KV are separate from weight counts.

Experts use a PyTorch reference or explicit Triton grouped kernels with row and weight
gradients. Attention uses SDPA and unmasked live paths preserve FlashAttention eligibility.
Full-vocabulary CE is tiled with recomputation in backward; deterministic argmax scans
all legal rows. There is no vocabulary top-k surrogate in the supervised objective.
The removed GDN/matrix recurrence has no leftover backend or kernel switch.

Blackwell image, compilation, hardware tests, measured work, and multi-node launch are
covered in [blackwell.md](blackwell.md). CPU correctness is not GPU throughput evidence.

## Work meters and diagnostics

| Meter | Meaning |
|---|---|
| U | Unique accepted transcript after deduplication, under the declared raw codec |
| D | Transcript in newly selected learning windows, including actual observations |
| I | D plus no-loss prefix reconstruction exposure |
| P | Actual supervised free lexical positions evaluated by the deterministic decoder |
| Z | Coordinates of stochastic facet actions evaluated in outcome RL |
| T | Newly generated teacher output, including rejected attempts |

B0 counts both prediction and reconstruction lexical passes in P. Recurrent stages count
one pass. Fixed syntax and declared copies contribute transcript exposure and no free
lexical CE. Actual emitted IDs can partition bytes differently from the reference codec;
scoring retains emitted IDs rather than retokenizing them. Generated raw facet replay is
recorded separately from byte encoding. Outcome RL scores a latent density with P=0;
its output token count is not a lexical policy likelihood.

`metrics.jsonl` records per-term sums/denominators, causal and reconstruction lexical CE,
encoder alignment MSE, predicted/encoded facet norms, D/I/P/Z rates, source bytes, physical
record exposure, prefix reconstruction, facet/control encoding, raw tape replay, expert
eligible/routed/learning rows, dispatch padding, optimizer time, gradient norms, and peak
memory. Learning-row counts exclude no-grad reconstruction and checkpoint recomputation.
Training diagnostics use detached device scalars and are materialized at logging boundaries.

Evaluation follows the checkpoint's representation mode. Bootstrap reconstruction and
facet NLL are inactive in recurrent evaluation; their report fields are null, not false
zero losses. Recurrent evaluation reports encoder alignment separately from causal CE.
Generated exactness and validity remain distinct from independently verified task success.

## Checkpoints and transitions

`output/checkpoints/latest.json` points to an immutable directory containing `state.pt`
and `metadata.json`. The metadata hashes the payload and the pointer hashes the metadata.
Loading uses PyTorch's restricted weights-only format. Checkpoints include model/config,
codec and semantic chunk identities, FP32 optimizer state, global step, phase counters,
and per-rank RNG and cursor. Checkpoint and served format version 2 contain model
architecture version 3; older architecture weights are rejected, not guessed into shape.

Periodic retention is bounded by `keep_checkpoints`. Phase boundaries, run-end snapshots,
pre-RL state, and explicit baked checkpoints are retained. SIGINT/SIGTERM finishes the
current transaction before saving. A hard kill can lose work after the last saved
transaction; an unfinished temporary directory is never loaded as a valid checkpoint.

```bash
megatoken train --config configs/train/b0.yaml --resume runs/b0/checkpoints
megatoken train --config configs/train/p1.yaml \
  --resume runs/b0/checkpoints --phase-transition
```

Exact resume requires the same rank count, dataset/scope, chunk policy, codec, seed,
batch/window policy, objective, representation mode, precision, backend, and architecture.
Operational output/logging/checkpoint limits and additional max steps may change. A
phase transition preserves optimizer moments but resets the new phase's cursor and
meters, applies its objective/clipping policy, and saves an entry checkpoint. Pipeline
execution pauses on unfinished phases instead of silently advancing.

The regression suite compares uninterrupted and resumed CPU model, optimizer, RNG, and
cursor state. It does not promise bitwise equality across different CUDA builds, hardware,
precision, or world sizes. Raw taped FP32 facet coordinates are preserved exactly;
derived floating-point context and density comparisons use explicit numerical tolerances.

## Optional native-table import and bake

Cold start needs no external weights. For an explicitly selected local donor embedding,
the offline extractor aggregates fixed 128D projected descriptors for exact native byte
keys. It reads tensors in chunks and records the seed, donor/tensor/codec hashes, coverage,
and support. Support counts distinct vocabulary tokens, not inferred corpus frequency.
Only exact unigrams/bigrams receive imports; longer orders remain native hashed features.

```bash
megatoken engram extract --embedding-file /path/to/selected-embeddings.safetensors \
  --tensor-name model.embed_tokens.weight --donor-codec /path/to/raw-donor-tokenizer.json \
  --output artifacts/engram --model recurrent32
megatoken engram bake --checkpoint runs/p1/checkpoints \
  --codec artifacts/codec --output runs/baked
```

Set `engram_import` for the initial run. The temporary map is installed or restored before
optimizer owner enumeration. Baking folds its contribution into the native table, rebases
only affected FP32 masters, clears affected row moments, and removes temporary owners.
Export requires a completed bake. No donor model runs during training or serving.

## Fresh executed outcome RL

RL draws one raw Gaussian facet action per chunk and runs the fixed deterministic decoder.
The byte encoder and every decoder owner, including shared U experts, are frozen during
latent-only optimization. Their model weights, FP32 masters, and Adam moments remain
unchanged. This is necessary because changing a deterministic mapping while scoring only
its input density would not represent the same policy-gradient objective.

Collection retains every raw action, invalid or valid. Invalid actions receive no external
execution; their attempted computation remains in state and an explicit public error can
follow. Valid actions are sent to the actual environment with an idempotency key and state
lease. Observations use the same semantic chunk policy as training and serving. A complete
raw tape includes the action, codec/spec identities, effort, role, exact payload/IDs,
validity, terminal flag, causal control bytes, behavior density, and deterministic decode trace.

Before updating, every rank checks fresh same-weight replay against behavior density.
Candidate history is rebuilt under current weights with recorded raw actions; old expert
routes are not forced onto it. Same-task leave-one-out advantages are computed from
independently verified outcomes. Equal reward groups skip the optimizer entirely, avoiding
a momentum or weight-decay-only update. There is no PPO reuse of stale rollouts.

Effort is fixed at 6 for RL. `max_task_groups` bounds additional collection groups and
`max_updates` bounds actual optimizer updates. Episode and latent-coordinate ceilings
still apply; complete groups may cause a reported Z overshoot. RL checkpoints include
task-content hashes, next task group, RNG, meters, frozen owner names, and optimizer state.
Tool `terminal=False` does not finish an environment episode; environment `done` and the
configured step limit govern episode completion.

Task files are JSON arrays or JSONL objects. Fixture example:

```json
{"task_id":"edit-0001","prompt":"Change answer.txt to blue, then finish.","initial_files":{"answer.txt":"red"},"expected_files":{"answer.txt":"blue"}}
```

The fixture executes `list`, `read`, `write`, and `finish` in memory and verifies the final
file map. Hidden expectations and rewards never enter model controls or observations.
A real isolated environment service implements:

| Endpoint | Request | Required response |
|---|---|---|
| `POST /reset` | `task` | `session_id`, `observation_base64`, `state_id` |
| `POST /step` | `session_id`, `action_base64`, `action_id`, `expected_state_id` | `observation_base64`, `state_id`, `execution_id`, `done` |
| `POST /verify` | `session_id` | `observation_base64`, `state_id`, `done`, `reward`, `verified` |

The service enforces idempotency, state leases, and actual verification. Remote transport
uses HTTPS; local development may use loopback HTTP. Credentials are read from a named
environment variable without printing their values. Teacher serving and environment
execution are provisioned separately from student training.

## Decisions before scaling

Use frozen held-out families and executed tasks. Track causal generation, encoder
alignment error and coordinate norms, exact bytes/copies, structured validity, task
success, expert learning coverage, and end-to-end throughput from the first stage.
Bootstrap reconstruction alone does not establish a useful recurrent model. Increasing
facet width, cache capacity, or tower length needs measured quality and memory evidence.

Recipes include history and data-quality recommendations. Runtime checks cover source
acceptance, frozen splits, causal masks, required streams, verified-success strata,
explicit dependency annotations, accounting, and restart contracts. They do not turn a
source label into measured long-context coverage or automatically advance curricula.
`WindowAudit` checks actual writer/reader windows and exposure. Use those measurements
and fixed task outcomes to decide when to extend the next pilot.
