# Blackwell execution and measurement

The implementation starts with resident BF16 weights, FP32 optimizer masters and
moments, FP32 recurrent states, and the same causal model for training and serving.
The portable PyTorch backend is the reference. The optional Triton backend contains
real grouped expert forward/backward GEMMs and a fused record recurrence with an
analytic backward. Selecting it is an explicit configuration choice. It has no
silent fallback that could make a CUDA benchmark measure a different backend.

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
RTX Blackwell products as 12.0. Select builds for the actual device, and record the
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
if requested CUDA/Blackwell support is absent. Retain its JSON next to the training
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
pytest -q tests/test_kernels_grouped.py tests/test_kernels_recurrent.py
pytest -q
```

The grouped tests compare values and gradients for sorted, uneven assignments and
unused experts. The recurrence tests compare both returned outputs and all six
input gradients. The CPU suite also verifies earlier-writer credit, FP32 independent
recurrent states, deterministic route replay, factual/predictive separation, BF16
core paths, actual parameter ownership, and engram bake parity. The hardware run
must retain these results, then run an integrated BF16 training transaction and
same-weight action-tape replay using the chosen backend.

`ModelConfig(expert_backend="triton", recurrent_backend="triton")` enables both CUDA
operators. Keep `torch` if a gate fails. Compare actual assignment distributions,
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
    config="v13",
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
    backend="torch",
    profile=True,
)
print(json.dumps(report, indent=2))
```

This runs the complete supervised generator path: causal ingestion, served-prior
draws, ordered decisions, capacity growth, full-vocabulary lexical loss, actual
commit ingestion, backward, and an FP32-master AdamW update. It still evaluates
synthetic inputs and a fresh model, so it measures implementation work rather than
learning progress. The real tokenizer must match the configured vocabulary. The
benchmark will not silently substitute a 256-row output head for v13.

Set `backward=False` for actual sampled, validated, committed prior actions.
The current general action-tape interface samples one episode at a time; this
benchmark labels `batch > 1` inference as `serial_requests`. Raw supervised work
and core-only ingestion use vectorized independent episodes. The report therefore
does not imply a batched inference scheduler that has not been implemented.

Set `core_only=True` to measure factual ingestion without a codec asset. That mode
reports source bytes and actual records, with `D` and `I` unavailable and `P=0`.
Its synthetic facet objective can exercise backward, but it is not a lexical
pretraining objective. Use full-vocabulary measurements before making capacity
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

Sweep capacities 64/256/1024/4096, efforts 1/2/3/6, short and long physical records,
and empty/partially filled/full historical caches. Record memory and latency at
each point. Context exposure is compressed through bounded historical slots and
record-time recurrence; 256K transcript tokens do not imply full 256K token attention.

## 4. Compile only the appropriate numerical boundary

`core.compile_kernels(mode="default")` compiles tensor-only dense FFNs, control
encoding, and carry reconciliation. It uses `nn.Module.compile`, which preserves
module parameter identity and normal state-dict names; see the
[PyTorch module API](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html).
The immutable factual state, action-tape control flow, and routing epochs stay
explicit. Compare `default` with a measured alternative only after parity tests.

Whole-policy CUDA graph capture is not claimed. Dynamic capacities, per-request
schemas, host action-tape serialization, variable retained histories, and the eager
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

The phase YAML files are launch profiles. Shared recipes provide phase budgets,
stream proportions, and effort probabilities. Their history limits are explicit
static window bounds; the JSON history curriculum is not automatically applied by
the launcher. B0 starts with 256 output positions and requires data prepared to
that bound. Subsequent profiles accept up to 4096 positions per physical record.
Native executed record boundaries cannot be silently shortened to make a profile
fit. Adjust preprocessing and validated configuration together.

Begin with a short multi-GPU transaction and a resume/replay test. Confirm rank
agreement on codec/data fingerprints, loss denominators, completed windows, routing
version, checkpoint step, and nonfinite-gradient handling. Save the precise
environment and measured B0 gate results before committing the bulk stage budget.

## 6. Preserve state and optimizer correctness while tuning

Runtime recurrent, historical, source, and carry caches belong to one model version.
Every optimizer update increments that version. Reconstruct caches from actual
records and complete action tapes; do not recycle parameter-dependent caches across
an update. Prefix replay costs real time and `I`, even under `no_grad`.

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
grouped/recurrent forward and backward parity plus complete action-tape replay,
and measures both backends on full tiny generator training/inference and the
primary factual core. It saves diagnostic JSON, JUnit results, real dispatch
counts, optimizer time, and measured device memory as workflow artifacts. Optional
`--compile` includes the tensor compilation setting.

To replace the primary core-only probe with a complete v13 generator benchmark,
pass `--codec /path/to/pinned/codec`; the workflow's `codec_directory` input mounts
an existing absolute runner path read-only. Without that asset, the primary probe
has no lexical `D/I/P` interpretation and cannot establish full-vocabulary memory
fit or throughput. A short successful gate also does not establish performance at
the actual batch, capacity, and history distribution; retain a subsequent report
for that workload before a long training allocation.
