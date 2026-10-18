# Initial implementation validation

Validation date: **2026-10-18**. This report describes software checks performed for
the initial implementation. It is not a trained-model quality or GPU performance report.

## Verified results

| Check | Observed result |
|---|---|
| Complete suite against the built and installed wheel | **142 passed, 6 skipped** in 54.22 seconds |
| Ruff lint and formatting | Passed |
| Shell entrypoint syntax | Passed for `launch.sh` and `slurm.sh` |
| Python package build | Wheel and source distribution built successfully |
| Installed CLI | Help, runtime doctor, pinned codec construction, and primary parameter census succeeded |
| Complete offline training smoke | B0, P1, M1, M2, LC, S, D, and fresh executed R completed |
| Served export and generation | Export loaded successfully and the final sampled chunk committed |

The final full-suite command ran without a source-path override after installing the
rebuilt wheel:

```bash
python -m build --no-isolation
python -m pip install --no-deps --force-reinstall dist/megatoken-0.1.0-py3-none-any.whl
python -m pytest -q --junitxml=validation/cpu-tests.xml
```

The local reference environment used Python **3.12.14**, PyTorch **2.10.0+cpu**, NumPy
**2.5.4**, tokenizers **0.23.3**, safetensors **0.8.0**, datasets **4.8.5**, and PyArrow
**23.0.1**. The CI workflow also configures a separate Python 3.11 run. There was no
CUDA device or Docker runtime in the local validation environment.

## What the integration tests exercise

- **Preparation and recovery:** immutable source and shard identities, family-level
  splits, deduplication and benchmark quarantine, exact-byte handling, action masking,
  stable interruption/resume, and real multi-row-group Parquet files. The Hugging Face
  reader test uses the actual datasets library over local Parquet while redirecting
  Hub resolution to the fixture; it does not claim to download the production corpus.
  It caught a datasets `.skip()` row-group resume issue. The reader now resumes with
  deterministic iterator slicing, at the cost of scanning preceding remote rows.
- **Causality and shared model ownership:** routing eligibility, expert sharing,
  record ordering, state reset and version checks, separate factual and private state,
  recurrent gradients, and exact owner accounting.
- **Generation and probability:** legal vocabulary masking, full-vocabulary loss and
  gradients, prior and posterior gradient direction, structured capacity feasibility,
  exact copies and identifiers, repair tapes, served-prior learning, and identical
  likelihoods under same-weight sample/replay.
- **Supervised training:** real parameter updates, variable-mask loss denominators,
  learning-rate work meters, deterministic stream state, atomic checkpoint integrity,
  exact optimizer/RNG/data resume, phase transitions, and writer-to-reader gradients.
- **Distributed training:** real two-process Gloo runs in FP32 and BF16; summed
  objectives match the combined-batch result, including experts used on only one
  rank, a completely idle rank, and gradient buckets split across tensor boundaries.
  Full training and restart preserve per-rank cursors and equal master weights.
- **Executed RL:** actual fixture edits with mixed independently verified outcomes,
  a real policy update, complete structural and variable-length raw action replay,
  history gradients, rejection of tampered tapes before Adam, exact interrupted
  resume, and no momentum or weight-decay-only update for equal-reward groups.
- **Deployment:** checksummed served-only export, bundled codec identity, posterior-free
  loading, stateful generation, independent sessions, and isolated JSONL request errors.

These checks are implemented in the repository's [test suite](../tests). Each one
establishes a specific contract; the fixture rewards and tiny-model outputs do not
estimate production task performance.

## Complete smoke run

The following command was run from the final source tree:

```bash
PYTHONPATH=src python -m megatoken --debug smoke --output runs/final-smoke
```

The seven supervised phases each performed an optimizer update, retaining optimizer
state across phase transitions. The final supervised global step was 7. Executed RL
then collected **8 episodes**, recorded **2 verified fixture successes**, performed
**1 optimizer update**, and saved global step **8**. The served bundle loaded and a
sampled chunk committed successfully.

The RL fixture uses public constant arguments, so its update trains structural choices
and records **zero free lexical P positions**. Episode limits bound this run. Separate
tests exercise variable-length lexical action replay. The smoke uses the tiny model
and explicit byte test codec, not the primary vocabulary or production data.

## Primary allocation and codec

The installed CLI instantiated the primary architecture on the meta device and counted
actual unique parameter owners with the pinned codec:

| Allocation | Parameters |
|---|---:|
| Served | 444,427,723 |
| Training only | 254,592 |
| Total | 444,682,315 |
| Temporary import, disabled by default | 0 |

There are **258 unique owners** and no aliased parameter entries in this census.
Served BF16 weights alone require **888,855,446 bytes**; this excludes gradients,
optimizer state, activations, runtime caches, and allocator overhead.

The pinned `inclusionAI/Ling-3.0-tiny` tokenizer revision is
`9a98e35fe1c9ee255f78dd64771c7ae15a799481`. Its metadata was fetched and checked locally,
then the byte-preserving raw codec was constructed successfully. The resulting
allocation has **157,184 rows**, including **156,891 legal free lexical rows**. The
codec fingerprint is
`7fa783ecf258c08fa6f9cd50b213185e09146c579967d839cb9de26ea59f2f95`.
Tokenizer assets and model weights are intentionally generated or downloaded by the
operator rather than committed to Git.

## Hardware and data gates still to run

The **six skipped tests** comprise one CUDA vocabulary parity test, two grouped expert
kernel cases, one recurrent kernel case, and two CUDA model/policy integration cases.
No Blackwell throughput, largest-profile memory fit, CUDA replay parity, or container
build result is claimed from this CPU environment.

On a provisioned Blackwell machine, run:

```bash
megatoken doctor --require-blackwell
megatoken codec --output artifacts/codec
python scripts/validate_blackwell.py --output runs/hardware-gate --codec artifacts/codec
```

The [Blackwell runbook](blackwell.md) covers the container, reference and Triton
backends, measured training/inference work, launchers, and multi-node operation.
The workflow's manual Blackwell job builds the container and retains hardware evidence.

The pinned source inventory is an acquisition template. Full production data acquisition,
source approval and ancestor joins, paid teacher collection, an external task service,
and full-budget training were not performed during implementation. The software paths
and fixtures are present; operators supply their reviewed corpora and task environments.
Use a bounded B0 pilot and fixed executed task evaluations before scaling a run.
