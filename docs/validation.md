# Recurrent32 v3 implementation validation

Validation date: **2026-10-18**. This report covers the current
`architecture_version=3` implementation, its data and training contracts, and the
software checks used before a production run. The installed-package suite passed
with **277 tests passed and 15 GPU-only skips**. Parameter totals are taken from
the saved v3 census files; tiny-model fixture results do not measure trained capability
or Blackwell performance.

## Final validation receipts

| Check | Current receipt |
| --- | --- |
| Complete suite against the rebuilt and installed wheel | **277 passed, 15 skipped** in 61.92 seconds; [retained receipt](validation_receipts.json) |
| Offline preparation, all training stages, served export, and generation | Completed from the final source tree; [smoke receipt](validation_receipts.json), with observed results below |

The suite includes the final causal-control and version-3 tape changes. Ruff lint
and formatting, shell syntax, the wheel/source-distribution build, installed CLI
diagnostics, pinned codec construction, and served generation also passed. The
wheel's Python modules were compared byte-for-byte with the source tree. The 15
skips cover one CUDA vocabulary parity case, four grouped-expert cases, two
model/policy cases, and eight GQA/masked/history attention cases. They supply no
GPU parity evidence.

The installed-package gate builds both distributions, installs the wheel, and
runs without a source-path override:

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
bash -n scripts/launch.sh scripts/slurm.sh
python -m build --no-isolation
python -m pip install --no-deps --force-reinstall dist/megatoken-0.1.0-py3-none-any.whl
python -m pytest -q --junitxml=validation/recurrent32-installed-tests.xml
megatoken --help
megatoken doctor
```

The reference CPU environment uses Python **3.12.14**, PyTorch **2.10.0+cpu**,
NumPy **2.5.4**, tokenizers **0.23.3**, safetensors **0.8.0**, datasets **4.8.5**,
and PyArrow **23.0.1**. No CUDA device or Docker runtime is available in this
local environment. The [CI workflow](../.github/workflows/ci.yml) additionally
configures Python 3.11 and 3.12 runs; its remote execution results are separate
from the local receipt above.

## Current architecture and allocation

The primary recurrent input is **32 raw facets of width 512**. The compact
configuration retains 32 facets at width 256. Both use six residual body stages,
a nonresidual remix, and **ten residual prediction stages after the remix**.
The completed predicted facets become the next recurrent input. Historical
context retains causal body K/V with 32 slots per processed chunk and a default
limit of 1,024 chunks. The [architecture document](architecture.md) describes
these state transitions and their arithmetic.

The saved implementation censuses report:

| Configuration | Recurrent facets | Total parameters | BF16 served-weight bytes |
| --- | --- | ---: | ---: |
| [recurrent32 v3](validation_receipts.json) | 32 × 512 | **477,498,706** | **954,997,412** |
| [compact32 v3](validation_receipts.json) | 32 × 256 | **265,990,610** | **531,981,220** |

All parameters in these default censuses are served parameters; additional
training-only and temporary-import allocations are zero. The byte totals cover
BF16 weights alone. Gradients, FP32 optimizer masters and moments, activations,
K/V history, workspaces, and allocator overhead require additional memory.
These censuses establish allocation, not device fit or throughput.

Both configurations use 157,184 vocabulary rows and a 1,024-position output
capacity. All production stage profiles, including B0, retain that capacity.
The pinned `inclusionAI/Ling-3.0-tiny` tokenizer revision is
`9a98e35fe1c9ee255f78dd64771c7ae15a799481`. The current byte-preserving codec
fingerprint, recorded in both census files, is:

```text
7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2
```

The codec identity travels with prepared data, checkpoints, served bundles, and
native action tapes. Tokenizer assets and model weights are acquired or generated
through the explicit asset commands and are not committed as model binaries.

## What the current tests establish

The [test suite](../tests) exercises concrete numerical, causal, and operational
contracts. The final receipt above determines which checks completed in the
reported environment.

### Byte ingestion, recurrence, and generation

[Encoder tests](../tests/test_model_ingestion.py) exercise ordered byte/ngram
features followed by hierarchical pair reductions. They check exponentially
growing receptive fields, bounded local attention, unequal-batch and padding
invariance in FP32/BF16, binary and Unicode inputs, empty input, and gradients
from every real byte with no contribution from padding. The final raw facet
projection can represent unconstrained scale.

[Core tests](../tests/test_model_core.py) check the production 32-facet geometry,
the internal nonresidual remix, and use of the completed prediction as the next
input. They verify that a selected raw facet tensor is preserved exactly;
consecutive observations process each preceding input once; historical K/V
contains causal body context; and later predictions propagate gradients to
earlier inputs and history. Reset, detach, model-version validation, bounded
history, deterministic routes, and checkpoint recomputation are also exercised.

[Generation tests](../tests/test_generation_policy.py) check one full-facet
Gaussian action with deterministic downstream materialization, full-vocabulary
losses, legal output IDs, structured fields, exact copies, identifier bindings,
and total output capacity. Invalid materializations and valid empty outputs
still advance raw recurrent state. Same-weight tape replay must reproduce the
recorded coordinates, state, and Gaussian density. Pure decoding inspection does
not advance a served session or re-encode decoded bytes as a substitute action.

### Phase-specific gradients and optimizer behavior

B0 bootstrap tests distinguish prior learning toward detached encoder targets
from encoder reconstruction learning. Bootstrap evaluates two decoder
cross-entropy passes, so actual lexical `P` counts both. Later recurrent phases
evaluate one prior-decoder pass and carry live predicted facets between outputs.
Their target encoder learns alignment toward the **detached completed prior
prediction**; target bytes and target length cannot condition that prediction.
Observation and control paths detach the byte encoder in recurrent phases while
preserving gradients through the trainable control encoder and causal body.
These directions are tested in
[generation](../tests/test_generation_policy.py),
[controls](../tests/test_controls.py), and the core suite.

[Optimizer tests](../tests/test_training_math.py) verify FP32 master weights and
moments with BF16 parameters, exact next-update restoration, and independent
encoder/main-model gradient clipping. A large encoder-following gradient must
not rescale the main model's update through a shared clipping denominator.
[Training integration](../tests/test_training_integration.py) covers real
parameter updates, variable-mask loss normalization, objective validation,
phase transitions, current-work meters, and exact checkpoint/optimizer/RNG/data
resume across the supervised stages.

### Public controls and immutable native history

Training, serving, and student collection share one control payload builder.
Tests preserve seed/reference bytes exactly, canonicalize only the public tool
JSON, reject noncausal reference origins, and keep absent controls neutral in
both single-example and mixed batches. A declared external reference is still
an operator-provided provenance assertion; the implementation cannot infer
whether arbitrary supplied evidence was honestly labeled.

The native protocol is **`fmt-canonical-facets-v3`**. Each tape owns the complete
FP32 facet draw and its exact source-byte `control_payload`, which replay encodes
under the current weights. Legacy versions and malformed or altered identities
fail explicitly. [Serving tests](../tests/test_inference.py) check that prior
controls match training and that request errors leave sessions intact.
[Collection tests](../tests/test_data_cli.py) exercise public policy/tool
forwarding, hidden verifier exclusion, valid-empty continuation, and complete
tape preservation through actual episode and ledger export.

[Native-history tests](../tests/test_training_history.py) preserve the ordered
interleaving of observed records and raw native draws. Valid nonempty draws own
one unchanged assistant record; invalid and valid empty draws own no record and
may share a cursor. Tests reject altered cursors, records, IDs, geometry, and
facets; compare replay with live context; preserve earlier-history gradients;
and exclude the target draw from its own conditioning in training and evaluation.

### Preparation, windows, and distributed scheduling

[Semantic chunk tests](../tests/test_data_chunking.py) cover the shared
256-token target, 512-token soft limit, and 1,024-token hard limit. They preserve
bytes and original reference-ID slices through paragraph, list, and code
boundaries, including Unicode, CRLF, oversized units, and forced cuts. Native
generated records keep immutable boundaries and fail explicitly if oversized.
Preparation/checkpoint/serving identities include the chunker, configuration,
codec, and parser-runtime identity.

Data tests also cover family splits before descendant generation, exact/near
deduplication, benchmark quarantine, causal action/result links, policy/recovery
masks, interrupted preparation, indexed shard integrity, and separate
`U/D/I/P/T/Z` accounting. The [reader tests](../tests/test_data_readers.py) use
real multi-row-group Parquet and the installed datasets streaming library over
local files. Hub resolution is redirected to a fixture; this does not validate
network acquisition or production-source yield. Teacher collection tests charge
rejected and interrupted requests and re-execute changed student actions against
their actual environment state.

[Window-stream tests](../tests/test_training_stream.py) use actual prepared
SQLite/JSONL datasets and synchronized thread reductions. Three episodes with
1, 3, and 2 windows across two ranks are visited exactly once per epoch. The
suite checks synchronized block slots, idle ranks, resume inside padded tails,
identical mixture choices from global `D/I/P` charges, atomic cursor restoration,
collective preparation failures, and collective rejection of rank cursor drift.

Separate [two-process Gloo tests](../tests/test_distributed.py) compare distributed
FP32/BF16 gradients and master updates with a combined-batch reference. Cases
include uneven masks, experts active on only one rank, a completely idle rank,
and reduction buckets split across parameter boundaries. Full engine restart
and phase transition check rank agreement on cursors and master weights.

### Executed RL, assets, and packaging

[RL integration tests](../tests/test_rl_integration.py) execute fixture actions
and use independently verified rewards. They check predictor updates while the
byte encoder and deterministic decoder remain frozen, complete raw-facet replay
with history gradients, rejection of replay mismatch before Adam, and exact
restoration of weights, optimizer, RNG, and task cursor. Equal-reward groups
perform no momentum or weight-decay-only update. RL samples Gaussian coordinates
under `Z`; deterministic lexical decoding contributes zero policy `P`.

[Asset tests](../tests/test_assets.py) exercise optional descriptor extraction,
installation, exact bake parity, optimizer-state cleanup, and served export.
[CLI tests](../tests/test_cli.py) include operation without importing PyTorch for
non-ML commands. Export/serving tests check checksums, bundled codec and chunking
identity, independent sessions, strict model ownership, and bounded JSONL request
handling. The installed-wheel gate adds package layout and entrypoint coverage
beyond tests run directly against the source tree.

## Completed all-stage smoke

The final source tree completed the following command:

```bash
PYTHONPATH=src python -m megatoken --debug \
  --json-output validation/recurrent32-smoke.json \
  smoke --output runs/recurrent32-final-smoke
```

The [saved receipt](validation_receipts.json) records preparation of
a local fixture corpus, every stage, served export, and generation after reload:

| Observed work | Result |
| --- | --- |
| B0, P1, M1, M2, LC, S, D | One optimizer update each; supervised global step **7** |
| Fresh executed R | **8 episodes**, **5 independently verified fixture successes** |
| R optimizer work | **2 updates**; final global step **9** |
| R policy meters | Lexical `P = 0`; Gaussian `Z = 1,024` |
| R deterministic emitted positions | **431** |
| R transcript exposure | `D = 5,237` |
| R ingestion | Collection `I = 5,237`; replay `I = 5,237`; total `I = 10,474` |
| Served reload and generation | Bundle loaded; sampled output reported `sample_valid = true` |

B0 recorded `P = 222`; each later supervised phase recorded `P = 111` for the
same fixture target counts, exercising the two-pass/bootstrap and
one-pass/recurrent accounting split. Phase transitions retained optimizer state.
The exported bundle identifies the version-3 tape protocol and model version 9.
The source-run smoke and the installed-wheel suite are separate validation
receipts. A repeat smoke must use a fresh output directory.

This smoke uses the tiny configuration and explicit byte fallback codec. It
checks end-to-end execution and phase compatibility. It does not train the
recurrent32 allocation, validate the production vocabulary's GPU memory use,
or estimate task quality.

## Blackwell and production-data gates

The 15 CUDA-only cases skipped in the CPU run require a provisioned GPU. Local evidence does not include Blackwell
throughput, CUDA/Triton parity, maximum-profile memory fit, NCCL execution, or a
Docker image build.

Use the commands and environment requirements in the
[Blackwell runbook](blackwell.md). On the target machine, retain diagnostics and
run the numerical gates before measuring throughput:

```bash
python -c 'import json; from megatoken.runtime import doctor; print(json.dumps(doctor(require_cuda=True, require_blackwell=True), indent=2))'
nvidia-smi
pytest -q tests/test_kernels_grouped.py tests/test_runtime.py tests/test_model_policy_integration.py
pytest -q
megatoken codec --output artifacts/codec
python scripts/validate_blackwell.py --output runs/hardware-gate --codec artifacts/codec
```

The hardware gate exercises BF16 execution, grouped expert/GQA values and
gradients, complete tape replay, and measured training/inference work for the
selected backends. Supplying the pinned codec enables the full recurrent32
lexical benchmark: by default it uses four live recurrent prediction chunks,
256 output positions, and eight history records. Use `--capacity 1024
--prediction-chunks 8` to request the larger capacity/tower gate. A primary
core-only probe cannot establish full-vocabulary fit or throughput. Retain
measurements at the intended batch, capacity, and history distribution. The
manual CI hardware job additionally builds the actual container and retains its
results.

Production corpus acquisition, source approvals and ancestry joins, paid teacher
collection, an external task service, and full-budget training have not been
performed during this implementation. The pinned inventory and fixture pipelines
provide executable starting points. Supply reviewed data and task environments,
then run a bounded B0 transaction, checkpoint/resume check, and fixed executed
evaluation on the chosen hardware before scaling the curriculum.
