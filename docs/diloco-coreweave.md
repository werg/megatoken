# Decoupled DiLoCo: coordinator, learners and CoreWeave GPU sandboxes

DiLoCo is the **target training methodology** for megatoken. It runs the same
mandatory retrieval-memory architecture as the torchrun trainer: every learner
loads the identical pinned recipe and query/document sidecars, and the coordinator
rejects any worker whose sidecar or recipe content differs. All neural parameters
(E/C/P/R, native KB KV projector, memory LoRA experts) synchronize as fragments.

**Phase coverage:** every phase. The coherence + embedding phase (`B0`/`P1`)
has no live bank and syncs only fragments. For live free-KB phases (`M1`+), set
`retrieval_bank` in the config before `megatoken diloco init`; the coordinator
copies that gate-ready bank and becomes its single owner (see below and
[configs/train/retrieval-free-kb-diloco.example.yaml](../configs/train/retrieval-free-kb-diloco.example.yaml)).

### Free-KB row sync

Free-KB rows are sparse, dynamically allocated SQLite records, so they use
their own protocol (`megatoken.training.diloco_kb`) next to fragments:

- **Ownership:** the coordinator holds `kb/canonical.sqlite` and alone grows
  rows and assigns IDs. Learners never grow locally; they forward query demand.
- **Learners:** each downloads the published snapshot to
  `<output>/worker-NN/diloco/kb.sqlite`, trains it with local sparse Adam, and
  every `local_steps` uploads `baseline - current` for each touched row field
  (baseline = the row's anchor: its last installed canonical value, or the
  value it was last uploaded at) plus its demand.
- **Merge:** per row field, contributions from the learners that touched it
  are weighted like fragments and applied with per-row outer Nesterov momentum.
  Rows retired on the canonical bank are skipped. Demand is recorded, growth
  runs (`retrieval_growth_budget` per merge), and rows, version and the
  per-worker sequence ledger commit in one SQLite transaction.
- **Staleness/duplicates:** bases older than `max_staleness` get HTTP 409 and
  the learner reinstalls; replayed sequences are idempotent.
- **Overlapped rounds:** learners never pause for a round. Installing a newer
  canonical version (rows or fragments) carries over the progress trained
  since the anchor: `new = global + (current - anchor)`. The steps run while a
  round was in flight are kept and become exactly the next upload, so nothing
  is double-counted or discarded. For rows, only rows the change set actually
  replaced get the residual re-applied.
- **Incremental pulls:** triggers on the canonical bank log every changed
  primary key under the version being produced. Whenever the coordinator
  reports a newer version, a learner POSTs its base version and the row IDs it trained; the
  coordinator returns only rows changed since that base plus those trained rows
  (with postings and `meta`). The learner upserts them in one transaction, which
  leaves its synchronized tables identical to the canonical bank while keeping
  its own sparse Adam moments. Transfer cost scales with the merge, not the bank.
- **Snapshots:** a stripped full snapshot (no coordinator state or learner
  moments) is built lazily, only for bootstrap and for learners whose base
  predates the retained change history (`retained_versions`, default 1024;
  HTTP 410 triggers the fallback).
- **Endpoints:** `GET /v1/kb?after=VERSION` (snapshot), `POST /v1/kb/changes`,
  `POST /v1/kb/updates`; `status` reports `kb_version`. Uploads are capped by
  `max_kb_update_bytes` (256 MiB).

Known limit: outer momentum is applied only to rows touched in a merge.

Its goal is useful multi-GPU pretraining with no GPU-to-GPU
interconnect. The design follows [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428):
each learner runs local steps, partitions its trainable FP32 weights into
independent fragments, and sends pseudo-gradients to a persistent coordinator.
The coordinator merges fragment contributions after a quorum and a bounded
grace interval, using token-weighted updates and Nesterov outer momentum.
Workers poll for newer fragment versions while continuing to train. This is
a practical asynchronous implementation, not a reproduction of the paper's
proprietary orchestration infrastructure or its adaptive performance controller.

## Prerequisites

- A CoreWeave account with GPU-sandbox access and a quota of at least eight
  concurrent GPUs for an eight-worker run. Serverless GPU capacity currently
  offers **one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) per sandbox**.
  [CoreWeave GPU docs](https://docs.coreweave.com/products/sandboxes/serverless/gpu-sandboxes).
- A persistent coordinator host with public/private HTTPS ingress, a **durable
  filesystem** for `--root`, and enough storage for the global FP32 weights,
  momentum, bootstrap, and queued worker updates. Budget **tens of GiB**
  for ~450M parameters and eight workers; monitor actual usage.
- Immutable prepared input dataset, pinned codec, and CUDA-sm_120-capable
  training image. The existing Dockerfile uses NVIDIA NGC CUDA/PyTorch. The
  container now includes optional S3 data and checkpoint dependencies.
- A private S3-compatible bucket with workload credentials to persist worker
  checkpoints (strongly recommended). CoreWeave AI Object Storage uses the
  `https://cwobject.com` endpoint and virtual-hosted addressing.
- An unguessable token called `MEGATOKEN_DILOCO_TOKEN`, with the **same** value
  on the coordinator host and every learner. Keep it, and any
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` for the
  checkpoint bucket, out of command lines and logs. Use short-lived
  credentials or workload identity instead where available.

## Configure

Edit `configs/train/retrieval-coherence-diloco-coreweave.example.yaml`:

- `decoupled.run_id`: unique immutable run ID. Never reuse for a changed
  objective, dataset, or experimental restart.
- `decoupled.coordinator_url`: resolvable **HTTPS** endpoint. The Python
  syncer serves HTTP bound on localhost; terminate TLS at a reverse proxy.
  `allow_private_http: true` is only for explicitly secured private networks.
- `workers: 8` with stable IDs `0..7` injected as `MEGATOKEN_WORKER_ID`.
- `inner_optimizer: muon`: applies PyTorch Muon to eligible hidden 2D
  matrices. Embeddings, norms, biases, controls, and packed rank-3 expert
  tensors use AdamW. FP32 masters/momenta survive checkpoints.
- `local_steps: 256`: per-fragment local optimizer updates between submissions.
  Start with 256, then benchmark 64/128/512 for wall-clock loss and network use.
- `decoupled.checkpoint_s3: s3://YOUR-BUCKET/megatoken-checkpoints`:
  enable remote atomic checkpoint publication and automatic resume. Worker
  objects are separated by `run_id/worker-XX/`.

If your prepared artifacts are not baked into the image, stage them to the
paths in the example config. **The coordinator initialization host must also
be able to read the codec path**; it does not need a GPU. The example uses
`/work/artifacts/codec` and `/work/artifacts/prepared`. Copy those files
there on the host or mount the same prepared inputs.

Data S3 layout:

```text
s3://YOUR-BUCKET/megatoken-input/
  codec/codec.json
  codec/<pinned tokenizer>
  prepared/manifest.json
  prepared/<prepared data referenced by manifest>
```

Prepare these once using the existing `megatoken data` pipeline. The staging
script validates object sizes and required manifests, while the existing
dataset/codec code validates domain-specific identities during training.
Make all workers use the **same immutable S3 prefix**.

## Start a persistent coordinator

Install the package with `pip install -e '.[train,storage]'` on the
coordinator host, then:

```bash
export MEGATOKEN_DILOCO_TOKEN="$(cat /secure/path/diloco-token)"
# Stage codec + prepared data at /work/artifacts before initializing.
megatoken diloco init \
  --config configs/train/retrieval-coherence-diloco-coreweave.example.yaml \
  --root /persistent/megatoken/diloco-coherence \
  --quorum 4 --fragment-mib 64 \
  --outer-lr 0.7 --outer-momentum 0.9 \
  --max-staleness 2 --grace-seconds 8

megatoken diloco serve --root /persistent/megatoken/diloco-coherence \
  --host 127.0.0.1 --port 8823
```

Publish port 8823 behind a properly authenticated HTTPS ingress or mesh with
TLS termination, rate/size limits, and a firewall restricting access to the
workers. Do **not** expose bare HTTP on a public interface. Configure the
coordinator as a persistent service managed by systemd/Kubernetes with durable
storage; do not run it inside an ephemeral worker sandbox. The syncer
authoritatively persists every fragment's global FP32 parameters, momentum,
deduplicating sequence numbers, and high-water token counters. Its in-memory
quorum grace timer is restored conservatively, not precisely, after a restart.

The `init` command deliberately refuses to replace any existing coordinator
directory. To begin a new phase/experiment, choose a new run ID and root.
`--resume` warm-starts the coordinator's global FP32 **weights** from a
previous coordinator export (`exports/latest.json`) or a training checkpoint;
it does not transplant the previous outer optimizer history. See "Phase
transitions" below.

Inspect authenticated status:

```bash
curl -fsS -H "Authorization: Bearer $MEGATOKEN_DILOCO_TOKEN" \
  https://YOUR-COORDINATOR/v1/status
```

Status contains global token high water, active leases, per-fragment versions,
pending submissions, and accepted merges. Worker JSONL metrics include
per-step wall time, per-domain gradient norms, local/global tokens, pending
uploads, fragment versions, and coordinator-visible merge progress.

## Running learners on GPU sandboxes

The existing `Dockerfile` builds a Blackwell-compatible CUDA/PyTorch image
(CLI, learner, S3 storage library, source, tests, staging script). Build it
and push it to a registry visible to the sandboxes, using an immutable image
digest in production.

GPU jobs run through the single-sandbox job runner in
[`src/megatoken/cloud.py`](../src/megatoken/cloud.py): a
`megatoken.cloud-job/1` file declares the image, resources (exactly one GPU per
sandbox), lifetime, sha256-pinned inputs, the command and the output
directories (an example is [configs/cloud/qa-p1.json](../configs/cloud/qa-p1.json)).

```bash
python -m megatoken.cloud launch JOB.json        # start the sandbox, push a clean committed snapshot and inputs
python -m megatoken.cloud logs SANDBOX_ID --follow
python -m megatoken.cloud collect SANDBOX_ID --dest DIR   # pull and sha256-verify outputs, then release
python -m megatoken.cloud stop SANDBOX_ID
```

Credentials come from the environment or a mode-600 `MEGATOKEN_CW_ENV_FILE`
(see the module docstring). **A multi-worker DiLoCo launcher built on
`megatoken.cloud` does not exist yet.** The former per-worker launcher script was
removed, and nothing in this repository currently starts N learner sandboxes
for one coordinator. Until one exists, each learner must be started by hand
or by your own tooling as a job whose command is `megatoken train` with the
shared config, a unique `MEGATOKEN_WORKER_ID` (`0..workers-1`) and the shared
`MEGATOKEN_DILOCO_TOKEN`; the protocol requirements below are unchanged.

Learners send only outbound HTTPS; no worker needs inbound networking. All
workers may start and finish independently, and a failed worker does not block
fragment updates as long as the configured quorum remains available.

The sandbox lifetime limit is explicit and cannot be extended on a running
sandbox; the CoreWeave maximum is 30 days. If a run exceeds a sandbox
lifetime, launch a replacement with the same worker ID and S3 checkpoint
namespace. The coordinator's lease can take up to three minutes to expire
before a *different* worker session may reuse the slot.

## Recovery and integrity

- The coordinator's root directory **must persist** across process/host
  restarts. Fragment commits are atomic, and an independent publication step
  is reconciled after a crash. A worker sequence is deduplicated before a
  pseudo-gradient can be applied again.
- Worker checkpoints preserve the model, hybrid optimizer, exact data cursor,
  RNG, local counters, fragment baseline tensors, versions, pending
  submissions, and the actual pending delta payloads. S3 latest pointers are
  published only after every checkpoint file (weights, metadata and, for
  free-KB phases, the learner's `kb.sqlite` and its manifest) is uploaded.
  Restores verify each file's checksum before publishing locally.
- Periodic checkpoints are snapshotted to host memory at the step boundary and
  written and uploaded on a background lane while training continues (one in
  flight at a time). Phase/run-end and interrupt checkpoints are durable
  before the trainer returns.
- On a replacement sandbox, the trainer restores its last committed S3
  checkpoint automatically when `checkpoint_s3` is configured and
  `--resume` is omitted. Explicit `--resume` selects a local checkpoint.
- The syncer enforces per-worker leases, maximum allowed staleness,
  fragment shape/finiteness checks, model/codec/optimizer ownership identity,
  and a shared training-objective fingerprint. A stale submission must
  download a newer global fragment rather than being silently applied.
- If a checkpoint precedes already-credited training tokens, coordinator
  accounting uses the worker's reported **high-water marks**, preventing
  duplicate token credit from replay. Exact once-only dataset processing across
  uncommitted worker crashes is not guaranteed: deduplication is for optimizer
  contributions and counters, not for raw example delivery.
- Sandbox scratch storage is **ephemeral**. Without `checkpoint_s3` (or an
  equivalent persistent per-worker volume), a sandbox replacement cannot
  recover its inner optimizer or exact stream cursor. Do not treat HTTP
  coordinator state alone as a complete restartable run.
- Runtime faults and HTTP retry schedules are logged; a missing quorum cannot
  be repaired algorithmically. The operator must restore enough workers.

## Final model and evaluation

The coordinator owns global trainable FP32 fragments, while learner
checkpoints own inner optimizer state and non-trainable buffers. For a
coherent **offline** snapshot, stop the coordinator service first and run:

```bash
megatoken diloco export \
  --config configs/train/retrieval-coherence-diloco-coreweave.example.yaml \
  --root /persistent/megatoken/diloco-coherence \
  --output /persistent/megatoken/exports
megatoken evaluate \
  --checkpoint /persistent/megatoken/exports/latest.json \
  --dataset /work/artifacts/prepared \
  --codec /work/artifacts/codec
```

Exports do not contain the workers' inner optimizer states or data cursors, so
they are not restartable local-training checkpoints; their `global_step`
reflects the highest fragment outer version, not a synchronized learner step.
They do carry the coordinator's global FP32 `master_weights`, which is what
seeds the next phase.

### Phase transitions

A phase transition is a **new coordinator** initialized from the previous
phase's export. Workers never resume across phases; they start fresh under the
new run ID with their own inner-optimizer state.

```bash
megatoken diloco export --config PREV.yaml --root PREV_ROOT --output exports
megatoken diloco init --config NEXT.yaml --root NEW_ROOT --resume exports/latest.json
```

Owners that are new in the next phase (for example realizer conditions when
packs are first installed) start from initialization; owners frozen in the
previous phase keep their exported weights. The media phases (bridge, joint,
tasks) chain this way; see [media.md](media.md).

## Validation checklist before a large-budget run

1. Run the CPU test suite, including `tests/test_diloco.py`, and
   `ruff check/format --check`. The existing synchronous two-rank Gloo
   suite must remain green.
2. On a Blackwell sandbox, run the repository's hardware gate
   `scripts/validate_blackwell.py` and a short two-worker real-training
   test. Confirm sm_120 kernels and Muon/AdamW moments are finite.
3. Kill and restart the syncer during a quorum merge; verify no version
   applies a worker's sequence twice and public fragments recover from
   authoritative FP32 state.
4. Kill a learner with a pending submission; restart from S3 and validate
   the optimizer/cursor and no double-credited token budget. Simulate
   late, duplicate, malformed, and stale uploads.
5. Compare one GPU, synchronous DDP if networking allows, and 8-worker
   DiLoCo at **equal global data tokens and held-out loss**, accounting
   for WAN throughput, S3 checkpoint overhead, staleness, and quorum delays.

CPU/unit tests do not establish GPU parity or reproduce full-scale convergence.
The original single-machine and torchrun paths remain available for
baselines.

## References

- [Decoupled DiLoCo (Douillard et al., 2026)](https://arxiv.org/abs/2604.21428)
- [CoreWeave serverless GPU sandboxes](https://docs.coreweave.com/products/sandboxes/serverless/gpu-sandboxes)
- [Sandbox configuration and lifetime](https://docs.coreweave.com/products/sandboxes/client/guides/sandbox-configuration)
- [PyTorch Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
