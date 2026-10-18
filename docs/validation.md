# Architecture v4 implementation validation

Validation date: **2026-10-18**, for the unmodified v4 main-branch build.
The installed wheel passed **304 CPU tests with 15 CUDA-only skips** before the
short-call and causal-training fixes in PR #2. These historical receipts do not
establish validation of the changed branch. The PR requires a fresh installed
CPU suite and the Blackwell gate; quality and throughput remain unmeasured.

[Machine-readable receipts](validation_receipts.json) retain the test counts,
runtime, source checksums, actual parameter censuses, and smoke results.

## Executed gates

| Gate | Observed result |
| --- | --- |
| Installed-package full suite | **304 passed, 15 skipped** in **49.86 seconds** |
| Ruff lint and format | Passed for source, tests, and scripts |
| Shell entrypoint syntax | Passed for torchrun/Slurm launch scripts |
| Wheel and source distribution | Built successfully; wheel installed without a source-path override |
| Source/wheel identity | All **72 Python/JSON package files** matched byte-for-byte |
| Installed CLI runtime preflight | Passed in CPU reference mode |
| Parameter ownership | **266 unique owners**, no aliases, for both production presets |
| All-stage offline workflow | Prepared data; B0 through D updates; executed R collection/replay; exported and reloaded v4 bundle; generated valid bytes |
| Independent implementation review | C/KV chronology, R/P cache reuse, functional gradient isolation, and RL freeze/replay reviewed without an additional blocker |

The reference host has Python **3.12.14**, PyTorch **2.10.0+cpu**, NumPy **2.5.4**,
tokenizers **0.23.3**, and safetensors **0.8.0**. There is no CUDA device, Triton
runtime, or Docker runtime available for local hardware validation. Local
loopback access was enabled for real HTTP-environment and multiprocess Gloo
tests. The [CI workflow](../.github/workflows/ci.yml) also runs the installed
package on Python 3.11 and 3.12; remote results belong to their corresponding
commit and are separate from this local receipt.

The 15 skips are one CUDA vocabulary parity case, four grouped-expert cases,
two complete model/policy cases, and eight grouped-query attention cases with
production widths, masking, and long history. No skipped case supplies GPU
correctness evidence.

Reproduce the package gate with:

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
bash -n scripts/launch.sh scripts/slurm.sh
python -m build --no-isolation
python -m pip install --no-deps --force-reinstall dist/megatoken-0.1.0-py3-none-any.whl
python -m pytest -q --junitxml=validation/contextual32-v4-installed-tests.xml
megatoken --help
megatoken doctor
```

## Measured parameter allocation

Both production presets have E2 local refinement, C4 shared central processing,
a non-residual remix, and P12 residual prediction. The realizer uses every raw
facet plus cached early/final C features. The smaller preset changes width, not
architecture generation. Their actual instantiated owner counts are:

| Preset | Raw facets | Served parameters | BF16 weight bytes |
| --- | --- | ---: | ---: |
| `32x512` | 32 × 512 | **486,174,561** | **972,349,122** |
| `32x256` | 32 × 256 | **265,020,129** | **530,040,258** |

Both censuses have zero permanent training-only parameters, zero temporary-import
parameters, and no registration aliases. The primary central processor itself
has **8,660,489 parameters**, compared with **301,989,888** in its shared sparse P
bank. Independent C ownership makes the conditional decoding map freezable in RL
while retaining the large predictor's trainable capacity. Operator code is shared
without tying these owners together.

These byte totals cover BF16 weights only. Gradients, FP32 optimizer masters and
moments, activation graphs, KV caches, and decoder workspaces require additional
memory. A meta-device census establishes allocation, not device fit.

Reproduce the census using the pinned production codec:

```bash
megatoken codec --output artifacts/codec
megatoken --json-output validation/32x512-v4-parameters.json \
  inspect --model 32x512 --codec artifacts/codec --owners
megatoken --json-output validation/32x256-v4-parameters.json \
  inspect --model 32x256 --codec artifacts/codec --owners
```

The production vocabulary has 157,184 allocated rows. Both presets allow 1,024
output positions and retain 32 projected KV slots per selected input. At 1,024
cached chunks, the new recent-plus-coarse archive uses about 3.875 MiB of projected
KV for the primary episode and 1.938 MiB for the smaller preset, rather than
retaining full-resolution KV for all older records. These figures exclude central/raw caches and training
graphs. Arithmetic and memory scope are documented in
[architecture.md](architecture.md).

The pinned tokenizer revision is `inclusionAI/Ling-3.0-tiny` at
`9a98e35fe1c9ee255f78dd64771c7ae15a799481`. The byte-preserving production codec
fingerprint is:

```text
7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2
```

It is recorded with prepared data, checkpoints, served bundles, and native tapes.
Tokenizer assets and model binaries are produced by explicit commands, not
committed into the code repository.

## What the tests establish

### Ingestion, central reuse, and prediction

[Ingestion tests](../tests/test_model_ingestion.py) check ordered byte/ngram
features, exponential pair reduction, and the two local facet refinements.
They cover binary/Unicode/empty input, all-real-byte gradients, zero padding
gradients, unequal-batch invariance in FP32/BF16, and unconstrained raw output
scale. The uniform-length fast path performs one terminal facet readout with
output and all-parameter-gradient parity to explicit lengths.

[Core tests](../tests/test_model_core.py) verify that every selected observation
or raw draw triggers exactly one C computation and immediate KV write. Cached
prediction and realization preparation trigger no additional C call. Changing
C-final features while keeping raw facets fixed changes P; changing raw facets
while keeping C-final/history fixed introduces no skip around C. Other gates
cover exact FP32 selected-input identity, immediate record counts, startup
sentinels, bounded eviction, partial resets, explicit detach boundaries,
version/geometry rejection, live earlier-input/history gradients, and C/P
activation-checkpoint parity.

[Contextual generation tests](../tests/test_generation_context.py) check actual
output sensitivity to C features, R/P reuse of the same selected state, and
information flow from every one of 32 facet slots even into short outputs.
[Policy tests](../tests/test_generation_policy.py) cover one full-facet Gaussian
draw, deterministic structure and lexical materialization, public exact copies,
identifier bindings, output capacity, and invalid/empty selected actions.

### Bootstrap, recurrent objectives, and ownership

Gaussian tests verify the analytic noisy-codec `KL(q || p)`, including variance,
and controlled positive initial scale. Bootstrap's detached target distribution,
noisy direct reconstruction, contextual prior realization, and alternative
teacher-forcing branch are exercised separately. Target bytes do not condition
the causal prior.

Recurrent supervision keeps the selected prediction live across the next C/P
iteration. E's auxiliary reconstructs through contextual C/R with those weights
and source history detached. A frozen `functional_call` retains derivatives into
E without updating C/KV/U/R, and disables unsafe checkpoint recomputation for
that branch. Auxiliary router losses do not leak into main objectives.

[Real-trainer v4 tests](../tests/test_training_v4.py) use multiple targets with
activation checkpointing enabled. A later target has nonzero gradient to the
first selected input, while auxiliary forks never append to live history.
Multiplying the E auxiliary by **1,000×** leaves main gradients and independently
clipped Adam updates unchanged. Bootstrap and recurrent training both count two
actual decoder CE passes per eligible target. Native following has a nonzero
alignment denominator only when an exact tape/byte/ID/role pair exists; ordinary
offline targets receive no fabricated facet label.

[Optimizer and integration tests](../tests/test_training_math.py) verify FP32
masters/moments with BF16 parameters, variable-mask global normalization,
phase transitions, current-work meters, and exact model/optimizer/RNG/cursor
resume. The package exposes disjoint E, C, P, and core-owned R groups.

### Native history, controls, and outcome RL

[Native-history tests](../tests/test_training_history.py) preserve the original
interleaving of observed records, invalid draws, valid empty draws, and valid
nonempty actions. A scored target's own tape cannot appear in its prefix.
Fixed-action history replay runs C without re-encoding decoded bytes or
recomputing P/control. Density replay instead re-encodes the tape's public
control source and checks the conditional deterministic materialization.

The `fmt-canonical-facets-v4` protocol retains exact FP32 raw coordinates and
control bytes. Checkpoint/bundle format 4 and explicit model geometry reject
older incompatible assets. Controls share one byte-framing implementation across
training, evaluation, serving, and student collection. Tests check neutral absent
controls, mixed batches, public schema forwarding, hidden verifier exclusion,
and unchanged serving sessions after request errors.

[RL tests](../tests/test_rl_integration.py) execute fixture actions with verified
outcomes. A real predictor update changes the latent density while frozen E/C/KV/
U/R weights, replayed raw/early/final features, KV, and decoded bytes/IDs remain
unchanged under matched execution modes. The frozen FP32 masters and moments are
preserved too. Replay mismatch fails before Adam; exact RL resume restores task
cursor and RNG. Equal-reward groups skip the optimizer, including momentum and
weight-decay-only updates. Lexical decoding contributes zero P; Gaussian work
uses Z.

Raw saved facets are exact. Derived floating-point activations and densities use
backend-appropriate tolerances when execution modes differ; the tests do not
assume that no-grad and autograd select numerically identical attention kernels.

### Data, distributed execution, assets, and serving

[Semantic chunk tests](../tests/test_data_chunking.py) cover the shared 256-token
packing target, 512 soft limit, and 1,024 hard limit. Original bytes and codec-ID
slices survive paragraph/list/code boundaries, Unicode, CRLF, oversized units,
and forced cuts. Native generated IDs and boundaries stay immutable.

Data preparation and reader tests exercise streaming local datasets/Parquet,
lineage and family splits, exact/near deduplication, quarantine, causal
execution links, interrupted preparation, shard integrity, and separate
U/D/I/P/T/Z meters. Fixture Hub resolution does not validate production source
yield. Teacher/student collection tests retain executed observations, charge
rejected attempts, and rerun changed actions against actual environment state.

The [window stream](../tests/test_training_stream.py) is checked on unequal
per-episode window counts, synchronized epoch/block advancement, idle ranks,
mixture exposure, cursor drift, and collective preparation errors. Real
[two-process Gloo tests](../tests/test_distributed.py) compare uneven-mask,
idle-rank, and sparse-expert gradients/master updates with a combined-batch
reference, then exercise full trainer checkpoint/resume and phase transition.
They supply CPU distributed evidence; NCCL remains a GPU gate.

Asset tests exercise optional n-gram descriptor import, exact rounding-aware
bake, optimizer-state cleanup, and export. Codec tests preserve byte fallback
when special-token rows reuse ordinary BPE merges. Serving tests cover bundled
checksums, independent sessions, observations, native replay, strict ownership,
and JSONL request limits. CLI tests also distinguish canonical model names from
legacy CLI aliases and literal served-bundle paths.

## Completed installed-package smoke

```bash
python -m megatoken --json-output validation/contextual32-v4-smoke.json \
  smoke --output runs/contextual32-v4-final-smoke
```

The [receipt](validation_receipts.json) records the following actual work:

| Observed work | Result |
| --- | --- |
| B0, P1, M1, M2, LC, S, D | One optimizer update each; supervised global step **7** |
| Per-supervised-phase meters | `D = 398`, `I = 398`, `P = 222` |
| Fresh executed R | **8 episodes** across **2 task seeds**, **0 fixture successes** |
| R optimizer work | **0 updates**, correctly skipped for equal-reward groups |
| R policy work | Lexical `P = 0`; Gaussian `Z = 1,024`; 256 emitted positions |
| R transcript/ingestion | `D = 5,032`; collection `I = 5,032`; replay `I = 5,032` |
| Served reload and generation | Format 4 / model version 7; `sample_valid = true` |

The smoke's R phase checks collection, execution, replay, accounting, and the
zero-advantage skip path. The separate RL integration regression above supplies
evidence for a real optimizer update and a fixed conditional action map. The
fixture's success count is not a model-quality estimate. Phase transitions kept
FP32 optimizer state; all supervised stages now exercised the same two-pass P
accounting. A repeated smoke needs a fresh output directory.

This run uses the tiny model with an explicit byte fallback codec and a reduced
128-position chunk/output cap. It does not train the production allocation or
measure full-vocabulary GPU memory use.

## Remaining hardware and learned-quality gates

Run the [Blackwell gate](blackwell.md) on the target machine before a long
allocation:

```bash
megatoken doctor --require-blackwell
megatoken codec --output artifacts/codec
python scripts/validate_blackwell.py --output runs/hardware-gate --codec artifacts/codec
```

The gate checks BF16 device execution, grouped expert and GQA values/gradients,
complete tape replay, and actual training/inference costs. The production codec
is required for a full-vocabulary measurement; a core-only probe cannot establish
that memory fit. Record performance at the intended batch, output capacities,
and history distribution after the numerical gate. No concurrency or speedup is
claimed solely from caching C features.

Production corpus acquisition, paid teacher collection, a Blackwell/Docker/NCCL
run, large-profile device fit, and learned capability have not been established
by these CPU receipts. The code and commands are ready for an initial B0 pilot;
its reconstruction, prior output loss, representation norms, validity, and
held-out task results determine whether to scale the configured budgets.
