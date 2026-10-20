> **BREAKING MIGRATION NOTICE:** This document retains historical v8 training/background details. Its old `configs/train/*.yaml` stage commands and any optional-memory instructions are **not supported**. Only the phase-gated retrieval-memory curriculum described in [README.md](../README.md) and [retrieval-memory-implementation-status.md](retrieval-memory-implementation-status.md) may launch training.\n\n# Training and operations

> **Implementation boundary (v8):** The stages described below still support
> legacy one-event episodes; the new grouped `MultiTrackGenerator`
> now also supports grouped text, audio, image and video B0/recurrent
> meridians using checksummed media references and separate audio/visual
> loss denominators. Opt-in G budgets count physical megatokens, and frozen
> donor feature caches may feed trainable E projections. The current native
> media realizers are **untrained PCM/pixel baselines**, not high-quality
> pretrained generators; media evaluation and grouped RL remain future work.
> See [the runnable media pilot](omni-media-pipeline.md) and
> [revised omnimodal curriculum](omnimodal-architecture-and-training.md) for the
> proposed OM0–OM5 training strategy and concrete multi-track task families.
> Do not treat those proposed phases or mixtures as an already implemented recipe.

## Two representation-training modes

The recurrent input is a `[batch, 32, 512]` raw facet tensor, or `[batch, 32, 256]`
in the smaller preset. E builds the same format through ordered byte/ngram processing,
repeated halving, and two dense facet refinement blocks. Four independently owned
central stages process every selected input, use three temporary KV read/write
transitions, then publish only the final rotated KV entry. A
non-residual remix and twelve sparse prediction stages produce the next Gaussian mean.
The realizer reads canonical raw facets directly; only future P uses
C's final features and historical KV. See [architecture.md](architecture.md) for the tensor graph.

`TrainConfig.representation_mode` selects `bootstrap` or `recurrent`. The value `auto`
resolves to bootstrap for B0 and recurrent for every other supervised phase. Production
YAMLs make the choice explicit. Prepared loss plans record their mode, exact masks,
optional native labels, and denominators; forward rejects a different mode or denominator.

### Bootstrap: train a noisy byte codec and its causal predictor

Let `e = E(x)` encode the next exact byte chunk. The codec target distribution is
`q(z | x) = Normal(e, tau**2 I)`, with positive fixed `tau = codec_noise_std` (default 0.1).
It is a controlled noisy codec, with no private posterior network or learned q head.
The causal prior has its own learned scale initialized by `prior_noise_std` (default 0.1).

Bootstrap combines three objectives:

- Draw `z` from the causal prior and score the target with direct
  `R(z)`, without a C pass for output decoding.
- Reconstruct from noisy `E(target) → R`; bootstrap trains both E and R.

- Minimize `KL(stop_gradient(q) || p)` over all facet coordinates. The analytic Gaussian
  KL includes q's `tau**2` variance. It trains the causal prior without moving E to fit
  an unrelated forecast, and does not collapse to point-target Gaussian NLL.

The live bootstrap trajectory teacher-forces the clean encoded target, processed by C
from the same original pre-action context. The sampled reconstruction/prediction branches
never append their hypothetical histories to that trajectory. Later targets can train
an earlier teacher-forced E through the selected live window. The target does not enter
its own causal prediction or prospective controls.

### Recurrent towers: keep predictions live and train ingestion on its own branch

Recurrent phases return the selected prior draw after its central pass for
ordinary continuation, preserving full-facet gradients across adjacent chunks.
**Offline executed tool traces are different:** when a recorded action has a
dependent tool observation, the successor state uses that recorded action's
exact native facet tape when available, otherwise a detached E(recorded bytes)
approximation. Prior samples still receive direct lexical loss, but are
not allowed to inherit an observation from a different unexecuted tool call.
Live sampled actions may continue only through their own executed observations.

Every ordinary target also receives an E-only codec auxiliary. Noisy
`E(x)` goes directly to R with functionally detached R/U owners.
No C is run, history is not accessed, and gradients reach only E.
There is no recurrent KL or forecast MSE pulling E toward unrelated
future raw samples.

For a genuine saved native pair, an additional encoder alignment term can fit E to the
recorded raw tensor that produced those exact bytes. Eligibility requires a valid v8
tape, matching codec and facet geometry, exact original emitted IDs and payload, and
ownership of that one whole physical record. Teacher corrections, unrelated observed
bytes, invalid draws, and empty draws cannot borrow an alignment target. A native label
is supervision only; it is excluded from the prefix before its own prediction.

Observed byte inputs and frontend features used by prospective controls are detached
from main-model losses in recurrent mode. Control projection, the predictor, C, memory,
and R still learn from their own main objectives. E has an independent gradient-clipping
domain, so a large auxiliary reconstruction or native alignment error cannot scale the
main-model update. FP32 Adam moments remain owned once per parameter.

#### Detached noisy successful-trajectory control (recurrent training only)

To retain the future-informed training signal for a **controlled** agent, recurrent
supervision now has one additional *conditioning source*, not another decoder pass
or latent distribution. For verified successful recorded episodes, **50% of eligible
training windows** replace their ordinary causal control vector with an embedding
of their own successful trajectory reference. The same existing control encoder
receives a bounded, role-framed sketch (four chronological quarters, up to 1 KiB
per quarter) of actual record bytes; it never receives supervision labels,
verification outcomes, rewards, or hidden tests. The resulting 128D (or preset
control-width) vector is **fully detached**, then receives fresh elementwise
Gaussian noise of standard deviation **0.1**. One sampled vector is held fixed
across all scored chunks of that window and injected at the ordinary pre-P
control-injection site.

The other half of eligible windows, every non-verified episode, and all bootstrap
examples retain the existing **causal public control**. Its shared E frontend is
detached during recurrent training, but its control head and P control projection
continue to learn. Therefore the teacher does not train its own encoder through
future-target gradients; its use trains P to respond to a useful reference vector,
while causal-control examples continue to train the shared encoder. This is the
same model and output objective, with no extra KV writes.

**The successful-trajectory reference is knowingly noncausal:** it includes target
and later observation bytes and is **never** a permissible serving, evaluation,
collection, or RL control source for that same trajectory. No such reference is
added to `Episode.control_reference`, the causal payload helper, physical history,
or the action tape. Deployment still takes only explicitly sourced seed/reference
bytes and tools. An earlier completed successful execution may be supplied as a
causal reference for a **different** run. Report held-out prior generation using
retained external references separately from causal seed-only and fallback controls;
a teacher-conditioned training loss is not a measure of autonomous causal quality.

| Objective | Bootstrap | Recurrent |
|---|---|---|
| Prior draw -> direct R target CE | Main-model gradients; causal prefix E may learn | Main-model gradients; observed E detached |
| Noisy E target reconstruction | Direct R; trains E and R | Direct R with R/U frozen; trains E only |
| Analytic q-to-prior Gaussian KL | Prior gradients; q detached | Inactive |
| Exact native raw/byte pair alignment | Inactive | E only, when a genuine pair exists |
| Live next input | Clean E target, processed once | Selected prior draw except teacher-consistent offline tool continuations |

Both modes evaluate two free lexical CE passes per target. Fixed syntax and explicit
copies remain excluded from free lexical masks. Default objective weights are defined
once in `megatoken.contracts`: lexical 1.0, reconstruction 1.0, native encoder alignment
1.0, facet-distribution KL 0.1, structure 0.25, binding 0.5, placement 0.25, terminal 0.1,
router balance 0.005, and router z-loss 1e-4. Inactive objectives have zero denominators.
These settings require held-out reconstruction, causal generation, and executed-task
pilots before treating them as quality recommendations.

## What one optimizer transaction does

Training, serving, collection, and replay use the same `build_control_payload` and
`encode_controls` helpers for public seed bytes, causal references, and tool schemas.
The supervised, detached trajectory-reference teacher above is an explicit
training-only exception, never passed to `build_control_payload` as external evidence.
Only tool JSON is canonicalized; seed/reference bytes remain exact. References require
`external_evidence` or `earlier_history` provenance. Empty control rows are neutral even
when batched with nonempty controls.

The JSONL generation request accepts `control_seed`, `control_reference`, and
`reference_origin`; its `tools` also condition the core. The numeric `seed` remains the
random generator seed. Python callers use the matching `InferenceEngine.generate`
arguments. Version-8 action tapes store the source control bytes for every draw. Replay
re-encodes those bytes under the current weights instead of trusting stale encoded
features, and rejects an explicitly different control. Control changes occur upstream
of the one facet draw. For fixed raw facets and explicit output controls, realization is
independent of history, KV and cached final C features.

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

The KV cache already contains the selected current input. `state.facets` retains its
exact raw values, while `state.final` caches its one C pass for future P. `observe`
encodes and processes the new observation immediately; `advance_raw` processes a saved
or sampled raw tensor immediately. Both append one KV record. `predict_facets` reads the
cached features without changing history, and `prepare_realization(state)`
exposes raw facets directly to R. Replaying a historical native draw therefore needs no
E, decoder, random draw, or discarded prediction. Its prospective controls are relevant
when scoring its density, but not when reconstructing fixed-action C/KV history.

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
megatoken inspect --model 32x512 --codec artifacts/codec --owners
megatoken --json-output runs/recurrent-benchmark.json benchmark \
  --model 32x512 --codec artifacts/codec --device cuda --precision bf16 \
  --backward --representation-mode recurrent --prediction-chunks 8 \
  --batch 8 --capacity 256 --history-records 128 --steps 20
```

The two production presets are `32x512` and `32x256`; both use all 32 facet slots.
Run `inspect` for the current actual unique-owner count, including E, independent dense
C, sparse P, and R. There are no private posterior or alignment heads. `inspect` performs a meta-device owner census without allocating full weight
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

Both bootstrap and recurrent stages count their prediction and reconstruction lexical
passes in P. A loss weight of zero does not erase an actually evaluated pass. Fixed
syntax and declared copies contribute transcript exposure and no free lexical CE.
Actual emitted IDs can partition bytes differently from the reference codec;
scoring retains emitted IDs rather than retokenizing them. Generated raw facet replay is
recorded separately from byte encoding. Outcome RL scores a latent density with P=0;
its output token count is not a lexical policy likelihood.

`metrics.jsonl` records per-term sums/denominators, causal and reconstruction lexical CE,
genuine-pair encoder alignment MSE, predicted/encoded facet norms, D/I/P/Z rates, source
bytes, physical record exposure, prefix reconstruction, facet/control encoding, raw tape replay, expert
eligible/routed/learning rows, dispatch padding, optimizer time, gradient norms, and peak
memory. Learning-row counts exclude no-grad reconstruction and checkpoint recomputation.
Training diagnostics use detached device scalars and are materialized at logging boundaries.
`forecast_encoder_distance_mse` is only a representation-distance diagnostic;
`native_encoder_alignment_mse` reports the actual alignment loss on eligible saved pairs.

Evaluation follows the checkpoint's representation mode and separates causal CE from
noisy-codec reconstruction CE. Gaussian correspondence KL is active only in bootstrap;
native alignment is reported only when genuine paired labels exist. Inactive report
fields are null. Generated exactness and validity remain distinct from independently
verified task success.

## Checkpoints and transitions

`output/checkpoints/latest.json` points to an immutable directory containing `state.pt`
and `metadata.json`. The metadata hashes the payload and the pointer hashes the metadata.
Loading uses PyTorch's restricted weights-only format. Checkpoints include model/config,
codec and semantic chunk identities, FP32 optimizer state, global step, phase counters,
and per-rank RNG and cursor. Checkpoint, served bundle, and model architecture use version 8.
Single-event and grouped tapes use their respective v8 identities.
Earlier R parameters, C caches, checkpoint weights and tapes are incompatible
with direct realization and cannot be silently loaded.

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
  --output artifacts/engram --model 32x512
megatoken engram bake --checkpoint runs/p1/checkpoints \
  --codec artifacts/codec --output runs/baked
```

Set `engram_import` for the initial run. The temporary map is installed or restored before
optimizer owner enumeration. Baking folds its contribution into the native table, rebases
only affected FP32 masters, clears affected row moments, and removes temporary owners.
Export requires a completed bake. No donor model runs during training or serving.

## Fresh executed outcome RL

RL draws one raw Gaussian facet action per chunk and runs the fixed deterministic decoder.
The complete conditional decoding map is frozen: E, all four dense C stages, their
role/position inputs and history/KV projections, and R including U experts. These owners
have independent parameters from the twelve sparse P stages. The predictor, remix,
prospective-control projection, and Gaussian scale remain trainable. Frozen model weights,
FP32 masters, and Adam moments remain unchanged. This is necessary because changing a deterministic mapping while scoring only
its input density would not represent the same policy-gradient objective.

Collection retains every raw action, invalid or valid. Invalid actions receive no external
execution; their attempted computation remains in state and an explicit public error can
follow. Valid actions are sent to the actual environment with an idempotency key and state
lease. Observations use the same semantic chunk policy as training and serving. A complete
raw tape includes the action, codec/spec identities, effort, role, exact payload/IDs,
validity, terminal flag, causal control bytes, behavior density, and deterministic decode trace.

Before updating, every rank checks fresh same-weight replay against behavior density.
Candidate history is rebuilt from frozen E/C and recorded raw actions, giving the same
conditional R map. Current P routes are recomputed rather than copied from the old tape. Same-task leave-one-out advantages are computed from
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

Use frozen held-out families and executed tasks. Track causal generation, noisy encoder
reconstruction, genuine-pair alignment, and coordinate norms, exact bytes/copies, structured validity, task
success, expert learning coverage, and end-to-end throughput from the first stage.
Bootstrap reconstruction alone does not establish a useful recurrent model. Increasing
facet width, cache capacity, or tower length needs measured quality and memory evidence.

Recipes include history and data-quality recommendations. Runtime checks cover source
acceptance, frozen splits, causal masks, required streams, verified-success strata,
explicit dependency annotations, accounting, and restart contracts. They do not turn a
source label into measured long-context coverage or automatically advance curricula.
`WindowAudit` checks actual writer/reader windows and exposure. Use those measurements
and fixed task outcomes to decide when to extend the next pilot.
