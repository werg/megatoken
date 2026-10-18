# Blackwell execution and measurement

The implementation starts with resident BF16 weights, FP32 optimizer masters and
moments, raw recurrent facets, and bounded BF16 causal KV context. The same core
serves training and inference: four dense central blocks supply both contextual
realization and the next predictor, whose non-residual remix precedes twelve
residual sparse stages. Cached central features avoid a second C pass for either
consumer. The portable PyTorch backend is the CPU/numerical reference.
Production CUDA recipes explicitly select Triton grouped expert forward/backward
GEMMs; the `auto` setting uses Triton on CUDA and torch on CPU. Missing Triton on
CUDA is an error rather than a silent fallback to the synchronizing reference.

CPU tests verify the mathematical and causal implementation. They do not establish
Blackwell kernel compatibility, GPU memory fit, speed, or trained capability. Run
the hardware gates below on the exact image and GPU configuration used for training.

## 1. Pin and inspect the execution environment

The repository Dockerfile uses `nvcr.io/nvidia/pytorch:26.09-py3`. NVIDIA's
[26.09 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-09.html)
identify CUDA 13.4.1 and a PyTorch 2.14 development build. Installation preserves
the image's constrained framework package and records the complete package set in
`/opt/megatoken-environment.txt`. CPU CI separately pins the tested PyTorch 2.10.0
reference. These are distinct environments, each with its own validation evidence.
A GPU name alone is insufficient:
NVIDIA lists B200/GB200 as compute capability 10.0, B300/GB300 as 10.3, and several
RTX Blackwell products as 12.0, with GB10/DGX Spark at 12.1. Select builds for the actual device, and record the
driver, PyTorch CUDA build, Triton version, and compute capability in every run.
See NVIDIA's [compute capability table](https://developer.nvidia.com/cuda/gpus).

CUDA 12.8 introduced native SM100 compilation. NVIDIA's
[Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
explains compatible cubin/PTX selection and the restrictions on architecture-specific
targets such as `sm_100a`. Use the CUDA version in the tested container, rather than
assuming an arbitrary older wheel includes compatible kernels. This project does
not hard-code `sm_100a`, enable FP8, or require architecture-specific tensor-memory
assembly to execute its BF16 reference.

Run the diagnostic inside the allocated container:

```bash
python -c 'import json; from megatoken.runtime import doctor; print(json.dumps(doctor(require_cuda=True, require_blackwell=True), indent=2))'
nvidia-smi
```

The diagnostic executes a BF16 matrix multiplication on each visible GPU and fails
if requested CUDA/Blackwell support is absent, execution fails, or the result is nonfinite. Retain its JSON next to the training
configuration and checkpoint. An image can include CUDA libraries while the host
driver or container device configuration still prevents execution.

For Docker, install/configure the host NVIDIA Container Toolkit using NVIDIA's
[installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The repository image can then be tested with GPU devices exposed:

```bash
docker build -t megatoken:local .
docker run --rm --gpus all --ipc=host megatoken:local \
  python -c 'import json; from megatoken.runtime import doctor; print(json.dumps(doctor(require_cuda=True), indent=2))'
```

Pin the image by digest after validation; `docker build --build-arg
BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:YOUR_VALIDATED_DIGEST .` accepts that exact
base. Data manifests, tokenizer assets, and
output directories should be mounted explicitly for training; a container image
does not provide accepted training data or a tokenizer asset by itself.

## 2. Validate numerical and causal behavior before throughput tuning

Run the complete test suite on the GPU machine and confirm the CUDA tests execute:

```bash
pytest -q tests/test_kernels_grouped.py tests/test_runtime.py tests/test_model_policy_integration.py
pytest -q
```

The grouped tests compare values and gradients for sorted, uneven assignments and
unused experts, including the primary pair/output input widths of 1,024 and 512.
The CPU temporal-KV suite checks three stage-local C writes/reads, the final-only
history publication, relative/late-time RoPE, slot identities, fixed P3/6/9/12
historical readers, whole-record FIFO eviction, gradients, and v5 checkpoint rejection.
Repeat checkpointed and uncheckpointed parity tests under the new architecture.
Runtime tests compare unmasked Flash GQA with the math reference at 32 facets,
four query heads, two KV heads, and head widths 128/64, including the full
32-slot history bound (32 MiB BF16 KV per main-model episode). Separate masked-source cases include completely empty records. CPU
tests verify raw next-input identity, the internal nonresidual remix, observation
chronology, earlier-input/cache credit, checkpointed gradients, deterministic
route replay, BF16 paths, actual parameter ownership, and engram bake parity. The hardware run
must retain these results, then run an integrated BF16 training transaction and
same-weight raw-facet tape replay using the chosen backend.

`scripts/validate_blackwell.py --codec artifacts/codec --output runs/hardware-gate`
runs those parity tests and measures both expert backends. Its full-vocabulary
primary workload uses a four-chunk recurrent tower at 256 output positions with
eight prefix records. Pass `--capacity 1024 --prediction-chunks 8` for the maximum
output and a longer tower. Without the real codec, its primary result is explicitly
core-only. These short gates precede the full workload sweep below.

`ModelConfig(expert_backend="auto")` dispatches to Triton on CUDA and to torch on
CPU; `expert_backend="triton"` fails fast when run without CUDA. Production GPU
training should use the explicit Triton setting after passing the parity gate.
Compare actual assignment distributions,
including one record, eight independent records, and larger compatible batches.
The PyTorch implementation sorts rows and reads the group-count vector on the host;
that synchronization is a known reference-path cost. The Triton implementation
keeps group metadata on the device and performs each grouped matrix multiplication
without per-row replication of expert weights.

There are two sequential P updates at each stage. Combining them into one dispatch
would change the model: the second update reads the first update's result. Eligible
sets are compiled from fixed stage/scale descriptors and causal epoch inputs. Do
not change an episode's routes to improve a batch's utilization.

## 3. Benchmark the workload you intend to train

The callable benchmark is useful before and after the CLI launch wrappers:

```python
import json
from megatoken.benchmark import benchmark

report = benchmark(
    config="32x512",
    codec="artifacts/codec",
    device="cuda",
    precision="bf16",
    steps=20,
    warmup=5,
    batch=8,
    record_bytes=2048,
    capacity=256,
    history_records=32,
    effort=6,
    backward=True,
    representation_mode="recurrent",
    prediction_chunks=4,
    backend="torch",
    profile=True,
)
print(json.dumps(report, indent=2))
```

This runs the complete recurrent supervised path: observed-byte encoding, four
live full-facet predictions and deterministic realizations, full-vocabulary losses,
encoder alignment to detached completed predictions, backward through the tower,
and an FP32-master AdamW update with independent encoder clipping. The selected
facets remain the next inputs. It still evaluates
synthetic inputs and a fresh model, so it measures implementation work rather than
learning progress. The real tokenizer must match the configured vocabulary. The
benchmark will not silently substitute a 256-row output head for 32x512.

Set `backward=False` for sampled raw-facet actions, deterministic byte
materialization, and learned recurrent advancement; use `prediction_chunks=1`
for that inference measurement. To measure B0, select `representation_mode="bootstrap"`:
its prediction and reconstruction decoder passes both contribute to lexical P,
and teacher forcing installs the encoded target as the next input.
The current general action-tape interface samples one episode at a time; this
benchmark labels `batch > 1` inference as `serial_requests`. Raw supervised work
and core-only ingestion use vectorized independent episodes. The report therefore
does not imply a batched inference scheduler that has not been implemented.

Set `core_only=True` to measure the recurrent core without a codec asset. That mode
reports source bytes and actual records, with `D` and `I` unavailable and `P=0`.
Backward uses a causal predicted-facet objective after ingestion. It is useful for
operator validation and does not replace the real lexical training objective. Use full-vocabulary measurements before making capacity
or throughput decisions.

Every measured report includes:

- Actual source bytes, physical records, reconstructed records, raw-codec `D/I`,
  and scored lexical `P`. Prefix reconstruction contributes to `I` and elapsed
  time, and is excluded from `D`.
- Forward, backward, reconstruction, and optimizer wall time. CUDA synchronization
  bounds these measurements. Warmup is excluded.
- Per-expert routed rows, eligible inclusions, gradient-enabled row assignments,
  active expert count, entropy, and 32-row tile padding. Gradient-enabled assignments
  are exposure, not proof of a meaningful task gradient; inspect optimizer gradient
  statistics separately.
- Actual CUDA peak allocated and reserved memory. CPU reports leave these GPU
  measurements unavailable instead of estimating them.
- Optional profiler regions from a separate additional iteration. Inclusive CPU
  and device times overlap their parents and must not be summed as disjoint time.

The source contains `fmt::experts/p` and `fmt::experts/u` profiler ranges, in addition
to phase ranges. PyTorch's [profiler documentation](https://docs.pytorch.org/docs/2.10/profiler.html)
describes CPU/CUDA activities, shape recording, and memory profiling. Measure
profile overhead separately, as this benchmark does.

Sweep capacities 64/256/512/1024, efforts 1/2/3/6, short and long physical records,
and different prefix lengths. Record memory and latency at each point. The
raw recurrent sample occupies 64 KiB per episode in primary and 32 KiB in compact
when stored in FP32. Historical BF16 K/V grows by 32 KiB per processed chunk in
primary, up to 32 MiB at the default 1,024-chunk limit; compact uses half. These
figures exclude activations, temporary concatenations, and autograd history.
Attention reads the retained facet slots, so its work grows with this context:
maximum-history QK/AV products alone cost about 34.36 GFLOPs per primary iteration.
Prefix reconstruction and training graphs add their own measured cost. See
[architecture and arithmetic](architecture.md) for the precise 32-facet geometry.

## 4. Compile only the appropriate numerical boundary

`core.compile_kernels(mode="default")` compiles tensor-only stage/encoder/output
FFNs and control encoding. It uses `nn.Module.compile`, which preserves
module parameter identity and normal state-dict names; see the
[PyTorch module API](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html).
The immutable raw-facet state, bounded KV history, action-tape control flow,
and routing epochs stay explicit. Compare `default` with a measured alternative only after parity tests.

Whole-policy CUDA graph capture is not claimed. Dynamic capacities, per-request
schemas, raw-facet tape serialization, prefix replay loops, and the eager
reference dispatch have different capture constraints. Do not report a dense-FFN
compile result as compilation of the complete serving policy.

## 5. Launch data parallel training

Launch one training process per GPU with `torchrun`. The trainer uses explicit
NCCL gradient reductions on CUDA, shared global loss denominators, and identical
optimizer transaction boundaries. Empty local partitions still participate in the
collectives. It does not use expert parallelism or parameter offload.

Use the repository training CLI with its concrete phase configuration, for example:

```bash
NPROC_PER_NODE=8 scripts/launch.sh configs/train/b0.yaml

# Continue from an actual B0 checkpoint; phase transitions preserve model/codec.
NPROC_PER_NODE=8 scripts/launch.sh configs/train/p1.yaml \
  --resume runs/b0/checkpoints/YOUR_CHECKPOINT --phase-transition
```

Use the checkpoint path printed by the trainer. PyTorch's
[torchrun guide](https://docs.pytorch.org/docs/2.10/elastic/run.html) documents local
process counts, rendezvous configuration, and multi-node launch semantics. Every
rank must see the same immutable prepared-data manifest and codec fingerprint.

For Slurm, request the required nodes and GPUs, then launch **one torchrun agent per
node** using `srun --ntasks-per-node=1`. Give all agents the same rendezvous ID and
reachable coordinator address; use `--nnodes`, `--nproc-per-node`, `--rdzv-backend=c10d`,
and `--rdzv-endpoint` explicitly. `torchrun` creates the per-GPU workers. Avoid
launching one Slurm task per GPU and then spawning another full set of GPU workers
inside each task. SchedMD's [srun reference](https://slurm.schedmd.com/srun.html)
defines task/resource allocation and failure exit behavior. Use the cluster's
supported networking and CPU-affinity settings rather than copying an interface
name from a different system.

`scripts/slurm.sh` implements this topology. Submit it from the shared repository
root inside the cluster's configured container/runtime, for example:

```bash
sbatch --nodes=2 --gpus-per-node=8 scripts/slurm.sh configs/train/b0.yaml
```

Resource names, account/partition, container integration, and network selection
remain cluster configuration. Set `RDZV_PORT` if the default port 29500 is already
allocated; every node inherits the same coordinator and job-specific rendezvous ID.
Do not depend on the copied Slurm batch script's filesystem location: the launcher
uses `SLURM_SUBMIT_DIR`, or an explicit shared `MEGATOKEN_ROOT`.

The prepared corpus can use the same semantic chunking policy for every
production phase. For example, after collecting reviewed local source records:

```bash
megatoken prepare --demo artifacts/bootstrap.jsonl --codec artifacts/codec \
  --chunk-target-tokens 256 --chunk-soft-max-tokens 512 --max-record-tokens 1024 \
  --output artifacts/prepared
NPROC_PER_NODE=8 scripts/launch.sh configs/train/b0.yaml --max-steps 100
```

The bootstrap/demo input is only an execution fixture; use accepted corpus
manifests for a substantive training run. This short pilot verifies the actual
configuration, loss paths, checkpoint/resume, and measured GPU work before a long
allocation.

The phase YAML files are launch profiles. Shared recipes provide phase budgets,
stream proportions, and effort probabilities. Their history limits are explicit
static window bounds; the JSON history curriculum is not automatically applied by
the launcher. All checked-in production profiles, including B0, allow 1,024 output
positions. Actual decoder workspaces follow each target's capacity. Preparation
uses a 256-token target, 512-token soft limit, and 1,024-token hard limit, shared
across the phases. A smaller B0 output cap is an explicit experiment requiring
compatible preparation. Native executed record boundaries cannot be silently
shortened to make a profile fit. Adjust preprocessing and configuration together.

Begin with a short multi-GPU transaction and a resume/replay test. Confirm rank
agreement on codec/data fingerprints, loss denominators, completed windows, routing
version, checkpoint step, and nonfinite-gradient handling. Save the precise
environment and measured B0 gate results before committing the bulk stage budget.

## 6. Preserve state and optimizer correctness while tuning

Selected raw facets, central feature caches, and historical KV belong to one
model version. Every optimizer update increments that version. Reconstruct the
ordered observations and raw sampled actions under the new weights. Both
`observe` and `advance_raw` run four central blocks with three temporary
read/write KV transitions and publish **only** the final RoPE-rotated KV once.
The resulting early/final features serve realization and the next prediction. `predict_facets` is read-only and reuses those features. This eager
chronology also handles invalid or empty selected actions. Actual environment
responses enter through a later observation. Prefix replay costs real time and
`I`, even under `no_grad`.

Preserve causal controls with that history. Training and serving use the shared
byte framing for policy seeds, references, and canonical tool JSON. Every v4
action tape retains its exact `control_payload`; density scoring re-encodes those
bytes under the scoring weights. Rebuilding a fixed raw history needs only the
observations, selected draws, and record roles: central processing does not inject
prospective control. A reference must have an `external_evidence` or
`earlier_history` origin. In JSONL serving, `control_seed` and `control_reference`
are UTF-8 text, `reference_origin` declares that provenance, and `tools` feeds both
the decoder constraints and prior control. The numeric `seed` remains the draw's
RNG seed. Completely absent controls skip encoding; empty rows in mixed batches
receive exact zero so batching cannot introduce an unintended control.

Bootstrap jointly trains noisy encoder reconstruction, a KL toward the detached
encoder distribution, and contextual realization of prior samples. Teacher-forced
next inputs fork from the pre-action state. Recurrent phases retain live sampled
facets across their tower. Their encoder-only reconstruction branch differentiates
through central processing and realization with those parameters held fixed, plus
native-pair alignment when an exact facet/byte pair exists. The main recurrent
path keeps central/predictor/realizer gradients live while observation and control
paths detach E. Both modes evaluate two decoder CE passes per target; benchmark
that work and retain the representation mode in checkpoints and receipts.

For latent outcome RL, freeze the complete conditional decoding map: E, central
processing and its shared KV projector, and R including core-owned U weights.
The predictor, remix, controls, and Gaussian scale remain trainable. Reusing
central features does not by itself implement concurrent CUDA execution; overlap
is a separate profiling and scheduling task.

Engram descriptors are optional offline assets. Import learning adds only the
temporary bias-free descriptor map. Bake with the actual inference rounding path,
check exact lookup and forward parity, remove the temporary map and its optimizer
slots, and rebase/reset affected native-table moments through the optimizer adapter.
Resume the documented reduced-learning-rate transition from the pre-bake checkpoint.
No donor backbone is needed for an ordinary post-bake served model.

Keep quality measurements beside speed: served-prior loss, byte/action validity,
same-weight replay, successful executed tasks, dependent late-context decisions,
and retention. A throughput increase is useful only if the causal policy and these
training objectives still operate correctly.

## 7. Run the optional hardware workflow

CPU CI lints, runs the full suite, builds the package, and inspects the installed
CLI on Python 3.11/3.12. CUDA tests skip when no GPU exists. The separate manual
`workflow_dispatch` hardware job requires a Linux self-hosted runner labelled
`blackwell`, a current GitHub runner supporting the pinned actions, Docker, and
the NVIDIA Container Toolkit. It builds this repository's Dockerfile and runs:

```bash
python scripts/validate_blackwell.py --output validation
```

The gate fails if Blackwell BF16 execution or Triton is unavailable, executes
grouped/GQA forward and backward parity plus complete raw-facet tape replay,
and measures both backends on full tiny generator training/inference and the
primary core. It saves diagnostic JSON, JUnit results, real dispatch
counts, optimizer time, and measured device memory as workflow artifacts. Optional
`--compile` includes the tensor compilation setting.

To replace the primary core-only probe with a complete 32x512 generator benchmark,
pass `--codec /path/to/pinned/codec`; the workflow's `codec_directory` input mounts
an existing absolute runner path read-only. Without that asset, the primary probe
has no lexical `D/I/P` interpretation and cannot establish full-vocabulary memory
fit or throughput. A short successful gate also does not establish performance at
the actual batch, capacity, and history distribution; retain a subsequent report
for that workload before a long training allocation.
