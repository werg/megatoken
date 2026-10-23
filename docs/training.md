> **BREAKING MIGRATION NOTICE:** This document retains historical v8 training/background details. Its old `configs/train/*.yaml` stage commands are **not supported**. Only the phase-gated curriculum described in [README.md](../README.md), with the memory mechanism in [continuous-state-transducer-plan.md](continuous-state-transducer-plan.md), may launch training.

# Training and operations

> **Implementation boundary (architecture v11):** The stages described below
> still support legacy one-event episodes. The grouped `MultiTrackGenerator`
> also supports grouped text, audio, image and video bootstrap/recurrent
> meridians. Media enters only as pinned EmbeddingGemma2 hidden states and leaves
> only through the frozen pretrained Pocket, Sana and Sana-Video realizers with
> trainable `FacetCondition`s; audio and visual losses have separate global
> denominators; a stage's budget is counted in its own meter (media-bridge: `G`
> events). Media is part of the stage
> plan: a short media-bridge stage (MB) after the codec bootstrap, explicit media
> and long-document-reading shares in base pretraining, and replay in later
> stages, chained by `megatoken diloco export` then `megatoken diloco init
> --resume`. See [media.md](media.md) for the contract and
> configs, [diloco-coreweave.md](diloco-coreweave.md) for the coordinator
> workflow, and [the concordant task roadmap](pretrained-omnimodal-bottleneck-plan.md)
> for task families and data. No media-quality or GPU-scale training result is
> established.

## Two representation-training modes

The recurrent input is a `[batch, 32, 512]` raw facet tensor, or `[batch, 32, 256]`
in the smaller preset. E builds the same format through ordered byte/ngram processing,
repeated halving, and two facet-refinement blocks. Four central stages process every
selected input and use three temporary KV read/write transitions before publishing
only the final rotated KV entry. Block-local structure remains distinct; eligible
projection and FFN matrices use globally shared shape pools across E/C/P/U/R. A
non-residual remix and twelve sparse prediction stages produce the next Gaussian mean.
The realizer reads canonical raw facets directly; only future P uses
C's final features and historical KV. See [architecture.md](architecture.md) for the tensor graph.

### Shared expert ownership

Eligible learned projections, SwiGLU operations, and supported ungrouped
`Conv1d` sites route through globally registered expert pools. A deterministic
fold and track/role/stage codes create
a compact bootstrap context; a shared bootstrap selector chooses one small
router expert, which emits the routing feature. Each shared shape pool then
selects its eligible expert palette once for the current routing epoch. Every
compatible site reuses that cached palette, with a deterministic site-prefix
phase distributing its rows across the palette. Learned selection happens
once for the bootstrap and once per active shape pool per epoch, instead of at
every individual projection. Pools group linear sites by padded input/output
shape, SwiGLU sites by hidden width, and convolution taps by the equivalent
linear input/output shape, so compatible sites share expert values and routing
heads. Convolution bias is added once after tap accumulation. Sites retain
local biases, while the pool is referenced without registering duplicate
parameters under each site. Optimizer,
checkpoint, and synchronous or DiLoCo synchronization therefore see every
global pool parameter once. Retrieval-conditioned adapter factors are stored
with shared expert values and follow the same one-owner rule.

The trainer clips gradients from global pools in a separate `shared` domain.
This keeps each pool's combined gradient in one independently clipped domain,
regardless of how many sites use it. P/U selectors, role-local adapter
controllers, and the rest of the model remain in the model domain; recurrent
encoder reconstruction retains its own encoder domain. Router balance diagnostics
report the shared bootstrap selector, one selector per active shape pool, and the
explicit P/U and adapter controllers.
Sites sharing a selector contribute to that selector's combined assignment
counts instead of creating repeated copies of the same balance term.

In the media-bridge stage, the shared expert pools, bootstrap router, all shared
pool selector heads, and their adapters remain frozen while media-local encoding
and projection gains, the
central modality embedding, and realizer conditions train. The E-only recurrent
encoder-reconstruction branch uses detached functional copies of R/U and the
shared pools. It sends gradients through that fixed readout into E-produced
facets without changing readout weights. Shared pools can still receive gradients
from the ordinary prior path that consumes E output; those gradients are clipped
once in the `shared` domain, while E-private gradients use the separate `encoder`
domain. Other core owners do not receive gradients from the reconstruction branch.
Main recurrent and replay objectives update shared pools under their normal scope.

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
from main-model losses in recurrent mode. Control projection, the predictor, C,
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

A JSONL generation output accepts `control_seed`, `control_reference`, and
`reference_origin`; its `tools` also condition the core. The numeric `seed` remains the
random generator seed. Python callers use the matching `controls` of
`MultiTrackRuntime.generate`. Group action tapes store the source control bytes for every draw. Replay
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

## Prepared roots, stage budgets and held-out evaluation

**Several prepared roots.** Each preparation script (text and code sources, media
identities, concordant media tasks, long-document reading) writes its own immutable
root with its own media tensors. A run lists them all in `datasets: [root, ...]`
and reads them as one corpus (`PreparedCorpus`). With `use_stage_mixture: true`
every stage stream is read from whichever roots contain it; a stream present in
several roots is one stream, their episodes concatenated and drawn in proportion to
size. With `use_stage_mixture: false` the whole concatenation is one stream (smoke
runs, B0 bootstrap). Each episode is always paired with the asset store of the root
it came from, so media tensors, encode-and-reconstruct items and held-out windows
resolve against their own root; text anchors are looked up in all roots. Roots must
share the codec, the semantic chunking policy and the teacher that produced their
features (and the model's teacher), the same root may not be listed twice, and a
stage stream missing from every root fails at launch. Roots are ordered by manifest
fingerprint, so listing order and mount paths do not change the data order, the
cursor identity or the DiLoCo shards. Changing the set of roots invalidates an exact
resume.

**Stage budgets.** A run's budget is its stage's limit in the stage's own meter, as
declared in `stages.json` (B0: `I`, MB: `G`, P1/M1/M2/LC: `D`, S: `P`). The same
meter drives the learning-rate schedule, DiLoCo's global token budget and the
`phase_consumed` counter. `budget` may only shorten it (a smoke run of a few
thousand `D` tokens), never exceed the stage limit. `accumulate_events` still sets
the per-step target for the `G`-metered stage; every stage also accumulates to
`accumulate_scored` / `accumulate_ingested`.

**Held-out evaluation.** With `eval_every: N` the run scores, every N optimizer steps,
the first `eval_windows` windows of each stage stream's validation episodes (the whole
corpus is one stream without a stage mixture) with the identical training forward:
eval mode, no gradient, no optimizer step, no bank writes, a fixed per-window seed.
The `evaluation` log event holds, per stream, each loss term's mean over its own
denominator (policy and reconstruction cross-entropy, facet prediction, the audio and
visual terms) and the training objective's weighted sum, plus the pooled values. A
stream without validation episodes reports zero windows; a run with none at all
fails at launch. Synchronous ranks take the windows in turns and merge their sums;
under DiLoCo only worker 0 evaluates its own weights between syncs.

**Reading from a collection.** The `collection_reading` stream
(`scripts/prepare_collection_reading.py`) teaches the query side of retrieval
([media.md](media.md#retrieving-the-original-items)). Its episodes hold only the
question and the scored answer; the supporting items sit, with every other task's items, in
a source collection inside the prepared root (`collections/train`, `collections/validation`,
each built from its own split's tasks). The episode's annotations name the collection, the
teacher's embedding of the question and the annotated items. A finished user message is a
lookup boundary, so the lookup right after the question has read it. Each window that scores
that lookup (it falls before one of the window's own meridians, not in the replayed prefix)
scores it with two losses, one count each:
`query_alignment` (the embedding head's query matches the teacher's embedding of the
question, by the centered alignment distance, centered on the collection's stored mean query
embedding) and `retrieval_ranking` (`-log` of the softmax mass the annotated items take among
the retrieved candidates, the annotated items and a null entry at the score threshold;
retrieved items that are not annotated are never labelled negatives). The retrieved items
enter the sequence, so the continuation loss trains their use. Collections are opened
read-only in every process and shared by path, so ranks and DiLoCo workers read the same
files and no collection state is synchronized; the only trained parameter is the existing
embedding head. The held-out `evaluation` event reports `retrieval`: the share of annotated
items in the top 10 for the student's query and for the teacher's own query embedding of the
same question (`student_recall`, `teacher_recall`), per stream and pooled. The stream has a
1% share in P1, M1, M2, LC and S.

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
Adapter factors live beside each shared expert in the global pools. Four compact
role controllers (E/C/P/R) emit adapter preferences from one shared router feature
per lookup; their heads remain private to their role owners. The media-bridge stage
freezes the global pools, shared bootstrap router, and all adapter factors, while
training the bridge's local components. Optimizer-step-global balance covers the
explicit P/U selectors, the four role controllers, the shared bootstrap selector,
and one selector per active shape pool. Shared selectors are counted once even when
many routed sites use them; their balance losses are summed as separate normalized
terms, each normalized by that selector's own hard assignment counts and soft
probabilities.
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
Run `inspect` for the current actual unique-owner count, including E-private
frontend and block structure, the four C block sites, sparse P, R/U, and the
globally shared expert pools. Shared pool values are counted once regardless of
how many E/C/P/U/R sites use them. There are no private posterior or alignment
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
| T | Newly generated teacher output, including rejected attempts |

Both bootstrap and recurrent stages count their prediction and reconstruction lexical
passes in P. A loss weight of zero does not erase an actually evaluated pass. Fixed
syntax and declared copies contribute transcript exposure and no free lexical CE.
Actual emitted IDs can partition bytes differently from the reference codec;
scoring retains emitted IDs rather than retokenizing them. Generated raw facet replay is
recorded separately from byte encoding.

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

## Donor memory imports and bake

Cold start needs no external weights. The ordinary input-token embedding is an optional
baseline: it maps donor token vectors to exact byte unigram/bigram rows. For either pinned
baseline profile, run the extraction where the full indexed donor shard is already stored;
the extractor verifies its repository revision and shard checksum. This baseline does not
import DeepSeek Engram or Qwen PLE memory.

```bash
megatoken engram extract --donor-profile deepseek-v4.1-flash-1017 \
  --donor-root /path/to/pinned-donor-snapshot --output artifacts/deepseek-baseline
megatoken engram extract --donor-profile qwen3.8-flash-next \
  --donor-root /path/to/pinned-donor-snapshot --output artifacts/qwen-baseline
```

The required production import uses the native hashed context tables from both donors.
The cloud preparation recipe selects real text contexts from a verified prepared corpus,
reads only the addressed donor rows by HTTP range, and writes one independent asset map per
donor. Use [the DiLoCo preparation workflow](diloco-coreweave.md#run-a-phase) and its
[training example](../configs/train/diloco-engram-import.example.yaml). The source row IDs
are never copied into Megatoken's byte hash space; each donor is hashed and projected with
its own reference mapping.

Set `engram_import` for one map or ordered `engram_imports` for multiple independent maps.
Temporary maps are installed or restored before optimizer owner enumeration. Baking folds
the imported contributions into the native table, rebases affected FP32 masters, clears
affected row moments, and removes temporary owners. DiLoCo exports must be baked before
creating a served bundle or using the export as the next phase seed. No donor model runs
during training or serving.

## Outcome RL

Not implemented: the RL stage and its code were removed. Under the recorded
future latent-action RL constraint, freeze the conditional decoding map and all
shared expert matrix values and router/pool heads, because those shared values
also serve E, C, P, U, and R. P-specific route-program parameters and other
private distribution parameters may still change its density, but P cannot
update shared projection or FFN matrices used by the frozen conditional map.
The constraint is recorded in
[decisions/0004-contextual-realization.md](decisions/0004-contextual-realization.md).

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

