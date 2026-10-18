# Training and operations

## What one optimizer transaction does

The trainer selects independent episode windows and plans the exact loss denominators.
It combines compatible event schedules into padded batches; target bytes and expert
routes are not used to decide compatibility. Typed complete actions currently use
single-episode execution so each action can have its own schema, copy sources, and
identifier program. Raw document and code records use the batched path.

Every window begins by reconstructing its complete earlier actual prefix with the
current model under `no_grad`. It then retains the graph through the new window's
observations and supervised decisions, so a later reader can train an earlier writer.
After accumulated gradients and one optimizer update, that runtime state is discarded.
Recurrent, historical, and carry caches are never reused across model versions.

`max_history_tokens` is the differentiable **new-window** token budget. It is not a claim
that the complete reconstructed episode prefix is that short. `max_window_records` and
`scored_decisions` provide additional soft boundaries. Logical actions are indivisible;
LC can extend a window through an annotated reader up to the explicit dependency limit.
The data runbook explains record splitting and the source of writer/reader annotations.
Profile the actual episodes, including reconstruction, when sizing GPU memory and time.

The engine targets `accumulate_scored` or `accumulate_ingested` across all ranks, with
`max_accumulation_windows` as a hard bound on collection iterations. A complete window
can overshoot a token budget; this is reported, rather than dropping the final action
or secretly splitting a dependency. `max_steps` limits additional updates in the current
invocation and is independent of the phase's token budget.

For each loss term, the objective is the sum of its eligible numerators divided by its
global eligible count. It is not an average of local batch means. Each term has its own
denominator: lexical positions, structural choices, latent dimensions, and routing
examples need different normalization. Gradient accumulation and rank count do not
change the main token objective. The router auxiliary uses an eligible, local execution
batch's load statistic, then normalizes across examples; its nonlinear load statistic
is intentionally partition-dependent and is reported separately.

## Precision, ownership, and optimization

The default production configuration is BF16 model weights and arithmetic. AdamW keeps
FP32 masters and moments. Post-accumulate hooks move each BF16 leaf gradient into its
FP32 accumulation buffer before the next microbatch. Sensitive normalization,
Gaussian probability calculations, recurrent state updates, logits, and statistics
remain FP32. CUDA uses maintained fused AdamW; CPU uses the foreach reference.

Defaults are β₁=0.9, β₂=0.95, ε=1e-8, matrix weight decay 0.1, and global gradient clipping
at 1.0. Embeddings, lookup weights, norms, biases, and sensitive recurrent gates do not
decay. Lookup/control/prior parameters use half the base learning rate; GDN gate owners
use one quarter. Parameters shared across repeated module calls get one master and one
optimizer slot. Non-finite gradients fail before an optimizer update.

`megatoken inspect --model v13 --codec artifacts/codec --owners` performs a meta-device
shape census without allocating the full weight storage. The primary model has
444,427,723 served parameters, 254,592 posterior/alignment parameters used only in
training, and optionally 2,048 temporary import-map parameters. No parameter padding is
used to make the total equal a historical document.

Expert execution supports a PyTorch reference and explicit Triton grouped kernels.
The kernels preserve eligible routing and include gradients with respect to both rows
and weights. The recurrent kernel has an analytic backward checked against the PyTorch
reference. Full-vocabulary CE is tiled and recomputed in backward so saved state does
not contain a full length-by-vocabulary tensor. Sampling also uses the complete legal
vocabulary; a faster top-k surrogate is not substituted in RL likelihoods.

See [blackwell.md](blackwell.md) for measured workload commands, compilation boundaries,
hardware tests, distributed launch, and the limitations of CPU validation.

## Independent work meters

| Meter | What it counts |
|---|---|
| U | Unique accepted transcript after deduplication, measured by the declared raw codec |
| D | Transcript in newly selected learning windows, including actual observations |
| I | D plus no-loss reconstruction; RL additionally reports collection and replay ingestion separately |
| P | Actual scored free lexical positions, including applicable identifiers and repair draws |
| T | Newly generated teacher output, including rejected attempts |

Reference transcript tokenization is distinct from actual emitted IDs. For example,
two sampled IDs may concatenate to bytes that the reference codec represents as one
token: that is two eligible policy positions and one reference transcript token.
Likelihoods preserve the sampled IDs. Declared copies and fixed schema syntax contribute
transcript bytes and ingestion work but no free lexical CE. Legacy raw-text reconstruction
without a matching public schema retains its lexical targets. Failed repair attempts
contribute P and teacher attempts contribute T even when nothing is accepted.

`metrics.jsonl` records phase progress, per-term loss/denominators, prior and posterior
lexical CE separately, source bytes/s, D/s, I/s, P/s, actual record ingestions, no-grad
reconstruction, posterior/control encoding, expert eligible/routed/learning rows,
dispatch padding, optimizer time, gradient norm, and peak memory. Learning rows exclude
no-grad reconstruction and checkpoint recomputation. Utilization during all execution
is a separate diagnostic from exposure that receives learning credit.

Collection keeps accepted and failed episodes and teacher input/cache/output counters.
Outcome RL journals complete tapes, public observations, action/execution identities,
behavior likelihood, independently verified reward, and environment time. It reports
Wilson intervals for observed success counts. Neither nominal source sizes nor an
ordinary dense-model FLOP estimate replaces these counters.

## Checkpoints and transitions

`output/checkpoints/latest.json` points to an atomically committed directory containing
`state.pt` and `metadata.json`. The metadata hashes the payload, and the pointer hashes
the metadata. Loading uses PyTorch's restricted weights-only format. Every checkpoint
contains the model, actual model configuration, codec identity, FP32 optimizer state,
global step, phase counters, and per-rank RNG/data cursor state. Model and routing
versions travel with the checkpoint. Checkpoint directories are immutable.

Periodic retention is bounded by `keep_checkpoints`. Phase boundaries, run-end snapshots,
pre-RL state, and explicit baked checkpoints are retained. SIGINT/SIGTERM asks each
rank to finish its current optimizer transaction and save a checkpoint. A hard kill can
lose work since the last committed checkpoint; an unfinished temporary directory is
never treated as a valid checkpoint.

```bash
# Exact recovery, with the same topology and objective:
megatoken train --config configs/train/b0.yaml --resume runs/b0/checkpoints

# New stage, retaining the learned model and optimizer:
megatoken train --config configs/train/p1.yaml \
  --resume runs/b0/checkpoints --phase-transition
```

Exact recovery requires the same rank count, dataset identity, codec, seed, batch/window
policy, objective, precision, and architecture. Change operational limits such as output
directory, logging, checkpoint frequency, or additional max steps as needed. Changing
model/backend/objective requires an explicit new experiment; the code does not silently
reinterpret an old cursor. A stage transition resets the phase's cursor and progress,
keeps optimizer moments, and saves its entry checkpoint.

The CPU regression suite compares uninterrupted and resumed full-model runs bit for bit,
including model, optimizer, RNG, and data cursors. That evidence does not establish
bitwise equivalence across different CUDA releases, hardware, or world sizes.

## Optional native-table import and bake

The default cold start requires no external weights. For an explicitly selected local
donor embedding file, an offline extractor aggregates fixed 128D projected descriptors
for exact native byte keys. It reads one tensor by chunks, uses a recorded seed and
per-token normalization, and records donor/tensor/codec hashes. Coverage and support
are fixed coefficients; support counts distinct donor vocabulary tokens, not an inferred
corpus frequency. Only exact unigrams/bigrams receive imports. Hashed longer orders
remain explicitly hashed native features.

```bash
megatoken engram extract --embedding-file /path/to/selected-embeddings.safetensors \
  --tensor-name model.embed_tokens.weight --donor-codec /path/to/raw-donor-tokenizer.json \
  --output artifacts/engram --model v13
```

Set `engram_import: artifacts/engram` in the initial training configuration. The temporary
map is installed before optimizer creation, and dynamically restored before optimizer
ownership checks during resume. No donor model call occurs in training or serving.

```bash
megatoken engram bake --checkpoint runs/p1/checkpoints \
  --codec artifacts/codec --output runs/bake
```

Baking writes a new checkpoint; it leaves the source checkpoint intact. It folds the
static contribution into the native table, checks exact lookup parity, rebases FP32
master rows, resets only affected row moments, and removes the map and optimizer hooks.
Export refuses an unbaked active import. Subsequent exact resume can retain the historical
`engram_import` configuration path: restored baked state does not reread the donor asset.

## Executed RL and environment integration

The initial RL implementation uses full differentiable short-episode replay. Each task
has independent rollouts from a fresh environment reset and the same current policy.
Same-task leave-one-out advantages supply a baseline. The scored action contains every
latent draw, categorical choice, lexical ID, and repair draw. Candidate state is rebuilt
from real history; behavior expert selections are not forced onto candidate weights.

Before updating, the trainer compares fresh same-weight replay against recorded behavior
likelihood. A failure on any rank prevents the update on every rank. Equal-reward groups
perform no optimizer step, so momentum and weight decay cannot create a fake RL update.
The policy remains at temperature 1 with full legal support. `max_task_groups` and
`max_updates` respectively bound additional collection groups and actual optimizer
updates in an invocation. Episode and P ceilings still apply; complete groups may cause
a reported P overshoot. RL checkpoints save task-content hashes, the next task group,
rank RNG, meters, and optimizer state for exact recovery.

Episodes end when the environment returns `done` or the configured step limit is reached.
Tool schemas constrain action `terminal=False`; a raw tape's terminal flag does not
submit or terminate the environment task by itself.

Task files are JSON arrays or JSONL objects. Fixture example:

```json
{"task_id":"edit-0001","prompt":"Change answer.txt to blue, then finish.","initial_files":{"answer.txt":"red"},"expected_files":{"answer.txt":"blue"}}
```

The fixture executes `list`, `read`, `write`, and `finish` against an in-memory file map
and verifies the final map against hidden expectations. Verifier targets and rewards
are metadata; they never enter model controls or observations. It is useful for
correctness gates, not a substitute for repository/build/container evaluations.

For real tasks, provision a separate isolated service implementing these JSON endpoints:

| Endpoint | Request | Required response |
|---|---|---|
| `POST /reset` | `task` | `session_id`, `observation_base64`, `state_id` |
| `POST /step` | `session_id`, `action_base64`, `action_id`, `expected_state_id` | `observation_base64`, `state_id`, `execution_id`, `done` |
| `POST /verify` | `session_id` | `observation_base64`, `state_id`, `done`, `reward`, `verified` |

The service must enforce action idempotency and state leases, and perform actual
verification. Remote transport uses HTTPS; local development can use loopback HTTP.
Credentials can be read from a named environment variable through `HTTPEnvironment`;
the client never prints their values. Teacher serving and environment execution are
separate resource pools from student training. No cloud allocation or paid teacher
generation is started merely by installing or testing this package.

## Quality decisions before scaling

Use fixed held-out families and tasks. Track served-prior lexical loss and actual samples
from the first stage, including structured validity and task execution. Posterior
reconstruction alone cannot establish a usable served policy. Compare prior and
posterior loss, inspect expert learning coverage, and profile true record lengths,
copy rates, effort, reconstruction, and repair.

The recipe JSON includes curriculum recommendations and data quality targets. Current
runtime enforcement covers declared source acceptance, immutable family splits, exact
causal masks, available stage streams, verified-success streams, explicit dependency
annotations, loss accounting, and restart contracts. It does not assert that a source
name proves the recommended long-context fraction, or automatically promote a phase
because its posterior loss fell. `WindowAudit` exposes measured writer/reader and long
exposure checks for the actual prepared windows; use it and fixed task results to approve
your experiment-specific curriculum changes.
