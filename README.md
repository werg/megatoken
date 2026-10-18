# Megatoken

A from-scratch, causal faceted model and its complete training system. Native bytes enter
a persistent factual state; a latent prior plans the next chunk; a parallel realizer
produces exact output IDs and typed actions. One implementation handles supervised
training, served generation, and differentiable replay of executed actions.

The primary architecture has **444,427,723 served parameters** and **254,592 training-only
parameters**, measured from actual unique owners. The default is native random
initialization. An optional offline embedding import can warm-start exact byte keys and
be baked into the native table before export. No donor model is needed at inference.

This repository contains executable software, tests, launch recipes, and a small offline
integration corpus. It does not contain pretrained weights or a downloaded production
corpus. Model quality, Blackwell throughput, and the largest memory configurations must
be measured on the training hardware. The implementation does not claim to reproduce an
unavailable historical codebase or its exact parameter ledger.

## Start with the complete offline run

Python 3.11 or 3.12 is supported. The CPU reference is tested with PyTorch 2.10.0.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train,data,dev]'

megatoken doctor
pytest -q
megatoken smoke --output runs/first-smoke
```

`smoke` prepares independent fixture episodes, runs B0 → P1 → M1 → M2 → LC → S → D,
preserves the optimizer through every transition, performs fresh executed outcome RL,
exports the served model, and generates a final chunk. It uses the tiny architecture
and an explicit 256-byte test codec; those assets are not substitutes for the production
vocabulary or training data. Each smoke output directory is immutable: use a new name
for another run. The RL fixture tests structural policy learning with public constant
arguments; it is not a coding benchmark.

For just a short supervised run:

```bash
megatoken demo --output artifacts/demo.jsonl --count 32
megatoken prepare --demo artifacts/demo.jsonl --output artifacts/demo --codec byte
megatoken train --config configs/train/smoke.yaml
```

## First Blackwell launch

The supplied Dockerfile uses NVIDIA's `26.09-py3` PyTorch image and preserves its
framework build. That release documents CUDA 13.4.1 and PyTorch 2.14.0a0. A production
operator can provide an immutable, validated image digest through `BASE_IMAGE`.
See [NVIDIA's release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-09.html)
and the [Blackwell runbook](docs/blackwell.md).

```bash
mkdir -p artifacts runs
docker build -t megatoken:train .
docker run --rm --gpus all --ipc=host \
  --mount type=bind,source="$PWD/artifacts",target=/workspace/megatoken/artifacts \
  --mount type=bind,source="$PWD/runs",target=/workspace/megatoken/runs \
  -it megatoken:train bash

# Inside the training environment:
megatoken doctor --require-blackwell
megatoken codec --output artifacts/codec
megatoken inspect --model v13 --codec artifacts/codec
python scripts/validate_blackwell.py --output runs/hardware-gate --codec artifacts/codec
```

`megatoken codec` downloads only the pinned tokenizer metadata, verifies its SHA256,
removes normalization and special-token interpretation from the raw codec, and records
the resulting identity.
The mapping comes from
[`inclusionAI/Ling-3.0-tiny` at an immutable revision](https://huggingface.co/inclusionAI/Ling-3.0-tiny/tree/9a98e35fe1c9ee255f78dd64771c7ae15a799481).
The full output allocation is 157,184 rows; 156,891 rows are legal free lexical pieces.
Special and padded rows are masked, rather than trained as invented words.

Prepare a bootstrap pilot with a capacity compatible with B0:

```bash
megatoken demo --output artifacts/bootstrap.jsonl --count 10000
megatoken prepare --demo artifacts/bootstrap.jsonl \
  --codec artifacts/codec --max-record-tokens 256 --output artifacts/prepared
NPROC_PER_NODE=8 scripts/launch.sh configs/train/b0.yaml --max-steps 100
```

This fixture pilot establishes runtime operation. Prepare the real bootstrap and phase
streams before a budgeted training run. `--max-steps` limits additional optimizer updates
for this invocation; it does not mark the phase complete. Resume the pilot with the same
configuration and its last checkpoint:

```bash
NPROC_PER_NODE=8 scripts/launch.sh configs/train/b0.yaml \
  --resume runs/b0/checkpoints
```

The default CUDA path uses maintained PyTorch BF16 operators and SDPA. Explicit Triton
grouped-expert and recurrence backends are included with forward/backward parity gates.
Enable them after the hardware gate succeeds:

```bash
NPROC_PER_NODE=8 scripts/launch.sh configs/train/b0.yaml \
  --output runs/b0-triton --expert-backend triton --recurrent-backend triton --compile
```

Backend changes start a separately named experiment. Exact resume rejects changing
the model, precision, codec, backend, objective, or data cursor contract. Multi-node
torchrun and Slurm examples are in [docs/blackwell.md](docs/blackwell.md).

## Data preparation and collection

The [data runbook](docs/data.md) documents source adapters, actual task execution, family
lineage, acceptance filters, train/validation/test separation, and source acquisition.
Preparation uses streaming readers and an external SQLite index/deduplication store;
it does not load the complete corpus into RAM. Input and output manifests carry source
revisions, hashes, codecs, and transformation provenance. Interrupted preparation can
resume without reassigning already frozen family splits.

`configs/data/sources.v13.json` contains pinned publisher metadata for 19 selected source
strata. It is an acquisition template, not an assertion that the complete inventory is
downloaded, licensed for your use, or already joined to its original ancestors. Supply
your approved source manifest, declared field mappings, and any required parent joins:

```bash
megatoken prepare --sources configs/data/sources.approved.json \
  --codec artifacts/codec --max-record-tokens 4096 --output artifacts/production
```

Prepared outputs are immutable; use distinct directories for pilots, production data,
and revised mixtures. Set each phase YAML's `dataset` accordingly. Every positive-weight
stream in that phase must exist and have enough independent episodes for all ranks.
The reader refuses missing streams rather than silently renormalizing the recipe.

The teacher collector calls an explicitly configured OpenAI-compatible endpoint and
executes complete actions through either the included in-memory fixture or a versioned
isolated HTTP environment. It stores actual observations, idempotency keys, state leases,
teacher usage, rejections, and independent verification. Student-state distillation
uses an exported frozen student bundle, preserving its actual tape and then obtaining
fresh environment observations before teacher correction. See `megatoken collect --help`
and the [collection instructions](docs/data.md#run-a-budgeted-collection-pilot).

## Training stages

The canonical budgets, streams, and effort probabilities live once in
[`src/megatoken/data/stages.json`](src/megatoken/data/stages.json). The shared trainer
selects the phase's meter, deficit-schedules its stream mixture, samples predictive
effort, and advances the learning rate by consumed work.

| Stage | Purpose | Default ceiling | Entry point |
|---|---|---:|---|
| B0 | Native initialization, codec/structure skills, prior learning | 250M I | `configs/train/b0.yaml` |
| P1 | Broad document, code, synthetic, chat, and trajectory pretraining | 400B D | `configs/train/p1.yaml` |
| M1 | Connected repositories, issues, PRs, logs, and retention replay | 60B D | `configs/train/m1.yaml` |
| M2 | Verified trajectories and grounded artifacts | 40B D | `configs/train/m2.yaml` |
| LC | Differentiable long writer-to-reader dependencies | 20B D | `configs/train/lc.yaml` |
| S | Successful actions, recovery, exact instructions, and replay | 2B P | `configs/train/s.yaml` |
| D | Executed frozen-student corrections and successful replay | 1B P | `configs/train/d.yaml` |
| R | Fresh-policy outcome RL with complete replay | 1B P; 200K episodes | `megatoken rl` |

These are configurable experiment ceilings, not promises of useful data supply or a
requirement to exhaust every budget before assessing quality. Recipe history schedules
and quality quotas are visible planning recommendations. The checked-in YAMLs select
explicit, conservative graph limits; they do not automatically ramp history length or
certify quality thresholds. Use measured prior-generation and task results to choose
longer capacities, context profiles, and further training.

Transitioning to another phase preserves FP32 optimizer state and resets that phase's
data cursor and meters:

```bash
NPROC_PER_NODE=8 scripts/launch.sh configs/train/p1.yaml \
  --resume runs/b0/checkpoints --phase-transition
```

Or run complete supervised phases in sequence under torchrun:

```bash
torchrun --standalone --nproc-per-node=8 -m megatoken pipeline --configs \
  configs/train/b0.yaml configs/train/p1.yaml configs/train/m1.yaml \
  configs/train/m2.yaml configs/train/lc.yaml configs/train/s.yaml configs/train/d.yaml
```

The pipeline pauses if a phase stops at its update limit or a signal. It does not
silently move an unfinished phase into the next stage. A pipeline resume recognizes
the checkpoint's phase and skips earlier configurations.

For RL, supply distinct task IDs and a separately provisioned environment service:

```bash
megatoken rl --checkpoint runs/d/checkpoints --codec artifacts/codec \
  --tasks configs/tasks/rl.jsonl --environment https://your-isolated-environment.example \
  --output runs/r --task-seeds 10000 --rollouts-per-seed 4 --require-blackwell
```

The fixture is also available with `--environment fixture`. Its interface is documented
in [docs/training.md](docs/training.md). RL records fresh behavior tapes, rebuilds
candidate state from actual observations, checks same-weight likelihood parity before
an update, and uses same-task leave-one-out advantages. There is no PPO reuse of stale
rollouts. Equal-reward groups do not apply a momentum or weight-decay-only update.

## Shared model and training architecture

- **Factual state:** ordered byte features and exact/hashed native n-grams, six factual
  stages, four gated record-time recurrent writes, bounded historical memory, and
  literal byte access. Only actual committed records enter this state.
- **Predictive work:** 16 reusable stage modules with globally shared sparse P and U
  banks, deterministic eligible expert subsets, separate historical/live attention,
  a served hierarchical Gaussian prior, and private predictive carry.
- **Realization:** capacity-first parallel growth, exact-length selection after growth,
  tiled full-vocabulary loss and sampling, typed deterministic syntax, exact copies,
  early identifier bindings, and bounded repair.
- **One probability implementation:** the same action executor samples, supervises,
  and scores complete augmented tapes. Repairs and stochastic ancestors remain in the
  likelihood; rejected output never becomes factual history.
- **One optimizer implementation:** BF16 model arithmetic with FP32 master weights,
  accumulation, AdamW moments, normalization statistics, and recurrent state. Variable
  batches divide summed losses by the real global denominators. Distributed gradients
  are summed once after local accumulation, including experts used on only one rank.
- **One restart contract:** atomic checksummed model/config/codec/optimizer/RNG/cursor
  checkpoints. Runtime caches are reconstructed under the current model version after
  every optimizer update. Windows retain gradients through selected earlier writers.

Detailed causal and probability contracts are in [docs/generation.md](docs/generation.md).
Training, resume semantics, and operational metrics are in [docs/training.md](docs/training.md).

## Evaluate and serve

```bash
megatoken evaluate --checkpoint runs/s/checkpoints --codec artifacts/codec \
  --dataset artifacts/production --episodes 128
megatoken --json-output runs/training-benchmark.json benchmark \
  --model v13 --codec artifacts/codec --device cuda --precision bf16 \
  --backward --batch 8 --capacity 256 --steps 20

megatoken export --checkpoint runs/d/checkpoints --codec artifacts/codec --output artifacts/served
megatoken generate --model artifacts/served --prompt 'Explain the next step.' --max-tokens 256
megatoken serve --model artifacts/served
```

The local JSONL server supports independent `open`, `observe`, `generate`, `status`, and
`close` sessions. Exact bytes are returned as base64 alongside emitted IDs and the full
action tape. Observations from executed tools return through `observe`; the server does
not execute model-produced commands on the host. It is a simple stateful integration
API, not a claim of production-scale continuous batching.

`evaluate` reports held-out raw-record prior CE, generated-byte exactness, validity, and
confidence intervals. Its sampling spec permits arbitrary bytes, so reported validity
is not a tool-schema or task-success score. These diagnostics are separate from posterior
reconstruction loss and actual task success. The repository's tests prove specific
software contracts; they do not prove the trained model will be strong. Run the hardware
gate, a bounded B0 pilot, and fixed executed task evaluations before expanding a training
allocation.

## Development

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q
python -m build
```

CPU CI runs lint, tests, package builds, and CLI checks. The manual Blackwell workflow
uses a provisioned self-hosted runner, builds the actual container, checks CUDA kernels
and replay, and saves measured results. CUDA tests skip explicitly when no GPU is present.
The [initial validation report](docs/validation.md) records the installed-package test
results, complete eight-stage smoke run, parameter census, and outstanding hardware gates.

Keep new source formats in `data/adapters.py`, new environments behind the shared
`Environment` protocol, and new training objectives over the existing policy and
optimizer. Avoid copying models or training loops into stage-specific scripts.
