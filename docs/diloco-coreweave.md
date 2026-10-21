# Decoupled DiLoCo on CoreWeave GPU sandboxes

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
independent fragments, and sends pseudo-gradients to a coordinator.
The coordinator merges fragment contributions after a quorum and a bounded
grace interval, using token-weighted updates and Nesterov outer momentum.
Workers poll for newer fragment versions while continuing to train. This is
a practical asynchronous implementation, not a reproduction of the paper's
proprietary orchestration infrastructure or its adaptive performance controller.

## Prerequisites

- CoreWeave sandbox access. Credentials live in a mode-600 file named by
  `MEGATOKEN_CW_ENV_FILE` (default `~/.config/megatoken/coreweave.env`;
  `CWSANDBOX_AUTH=wandb` with `WANDB_API_KEY`, or `CWSANDBOX_API_KEY`). They are
  read by the launcher on your workstation and never enter a sandbox.
- A client venv with `pip install -e '.[cloud]'` (`cwsandbox` only; the launcher
  otherwise uses the standard library). The repository must be a **clean,
  committed** working tree: every sandbox runs that one commit.
- GPU quota for N concurrent learners. Serverless GPU capacity offers **one
  NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) per sandbox**, so each
  learner is one sandbox. [CoreWeave GPU docs](https://docs.coreweave.com/products/sandboxes/serverless/gpu-sandboxes).
- A CUDA-sm_120-capable image (the repository `Dockerfile` builds on NVIDIA NGC
  CUDA/PyTorch; [configs/cloud/qa-p1.json](../configs/cloud/qa-p1.json) uses an
  NGC PyTorch image) and prepared data, built in the cloud as described below.
- Optional: `decoupled.checkpoint_s3` (remote learner checkpoints) needs your own
  S3-compatible bucket, `pip install -e '.[storage]'` in the learner image
  (`pip` field of the run file) and AWS credentials that you supply to the
  learners yourself; the launcher does not deliver them.

## Run a phase

There is one way to run a DiLoCo phase on CoreWeave:
[`src/megatoken/cloud_diloco.py`](../src/megatoken/cloud_diloco.py). Write a run
file (`megatoken.cloud-diloco/1`, JSON):

```json
{
  "schema": "megatoken.cloud-diloco/1",
  "id": "qa-p1",
  "image": "nvcr.io/nvidia/pytorch:26.09-py3",
  "lifetime_hours": 24,
  "pip": ["pyarrow"],
  "workers": 4,
  "config": "configs/train/qa-pilot-p1-diloco.yaml",
  "data": {"peer": "<sandbox id>", "prefixes": ["out/corpus/datasets/rc-train"]},
  "coordinator": {"resources": {"cpu": "8", "memory": "32Gi"}, "args": ["--quorum", "3"]},
  "learners": {"resources": {"cpu": "8", "memory": "32Gi", "gpu": 1}}
}
```

- `config` is a normal `TrainConfig` file in the repo with a `decoupled` block.
  Its `coordinator_url` is filled in inside each sandbox
  (`python -m megatoken.cloud_diloco derive`), and `decoupled.workers` **must
  equal** `workers` or the sandbox fails. Use
  [configs/train/retrieval-coherence-diloco-coreweave.example.yaml](../configs/train/retrieval-coherence-diloco-coreweave.example.yaml)
  as the template; its paths (dataset, codec, sidecars) must point at where the
  data lands under `/work`.
- `workers` is 1..64. Learner resources must request one GPU.
- `coordinator.args` are passed to `megatoken diloco init`: `--quorum`,
  `--fragment-mib`, `--outer-lr`, `--outer-momentum`, `--max-staleness`,
  `--grace-seconds` (defaults 4, 64, 0.7, 0.9, 2, 8.0). Coordinator resources
  default to 8 CPUs and 32Gi.
- `decoupled.run_id` is the unique immutable run ID. Never reuse it for a changed
  objective, dataset or experimental restart.
- `inner_optimizer: muon` applies PyTorch Muon to eligible hidden 2D matrices;
  embeddings, norms, biases, controls and packed rank-3 expert tensors use
  AdamW. FP32 masters and momenta survive checkpoints.
- `local_steps` is the number of per-fragment local optimizer updates between
  submissions. The example starts at 256; benchmark other values for wall-clock
  loss and network use.
- The run `id` must be new locally (`launch` refuses an id whose run record
  still exists).

Then:

```bash
python -m megatoken.cloud_diloco launch RUN.json [--repo .]
python -m megatoken.cloud_diloco status RUN_ID
python -m megatoken.cloud_diloco finish RUN_ID --dest DIR
```

`launch` starts the coordinator, waits until it answers `/v1/status`, then starts
the N learners, and prints the run record. If any step of the launch fails it
stops every sandbox it started. `status` reports the coordinator's authenticated
status (global token high water, active leases, per-fragment versions, pending
submissions, accepted merges) and which learners have finished. Learner JSONL
metrics include per-step wall time, per-domain gradient norms, local/global
tokens, pending uploads, fragment versions and merge progress.

`finish` waits for every learner, then collects each learner's outputs
(`DIR/learner-N`) and logs. Every collect includes the sandbox's job log
(`.cloud/job-final.log`), because sandbox logs vanish once a sandbox stops. If
any learner failed, `finish` stops all sandboxes, **refuses to export** and
reports the failed learners. Otherwise it signals the coordinator, which stops
serving and runs `megatoken diloco export` into `out/export`; that export is
collected into `DIR/coordinator` and all sandboxes are stopped. The export (the
global FP32 masters, `export_only`) is the seed of the next phase.

Status of scale: the path has been smoke-tested end to end on CoreWeave with two
GPU learners, a CPU coordinator and a tiny model, with one outer merge applied
and the export collected and loadable. **No large-scale run has been done.**

## How it works

**Roles.** The coordinator is a CPU sandbox running `megatoken diloco init`
then `megatoken diloco serve` on port 8080. Each learner is a GPU sandbox
running `megatoken train` with `MEGATOKEN_WORKER_ID` set to `0..N-1`. Learners
start only once the coordinator answers `/v1/status`, and all roles run one code
snapshot (one commit, recorded in the run record). Learners send only outbound
HTTPS; no learner needs inbound networking.

**Endpoint and secrets.** The coordinator's port is published as a public HTTPS
endpoint that admits only holders of the sandbox's create-only share token
(header `X-Sandbox-Share-Token`; requests without it get 401). Behind that, the
DiLoCo protocol still requires its own bearer token. The launcher mints a fresh
DiLoCo bearer token per run, keeps it locally in mode 600 under
`~/.local/state/megatoken/cloud/secrets`, and delivers it and the coordinator's
share token to the sandboxes as mode-600 files in `$MEGATOKEN_CW_SECRETS`
(`MEGATOKEN_DILOCO_TOKEN`; in learners also `MEGATOKEN_DILOCO_GATEWAY_TOKEN`,
sent as the gateway header). Secrets never appear in the sandbox spec,
environment spec or command lines, and CoreWeave credentials never enter a
sandbox. The DiLoCo client reads response headers case-insensitively, since the
HTTPS gateway is HTTP/2 and lowercases header names.

**Data is built once, in the cloud.** Uploads from a workstation are uplink-bound
(measured about 0.5 MB/s), while sandbox-to-sandbox HTTPS measured about
850 MB/s. So the run's `data` is one of:

- `{"peer": "<sandbox id>", "prefixes": [...]}`: a finished
  [`megatoken.cloud`](../src/megatoken/cloud.py) job whose job file set
  `serve_port` and `serve_outputs`. Its outputs are served read-only over its
  share-token endpoint, and every sandbox of the run fetches the files under
  the listed prefixes from it, verified by sha256, landing at the same relative
  path under `/work`. The peer must have finished (its outputs ready) before
  `launch`.
- `{"inputs": [...]}`: ordinary job inputs, for small data. Each has a `path`,
  a `sha256` and either a pinned `url` or a local `source` file that the
  launcher pushes (slow for anything large).

Build the data with a normal `megatoken.cloud` job
(`python -m megatoken.cloud launch JOB.json`, see
[configs/cloud/qa-p1.json](../configs/cloud/qa-p1.json)); do not `collect` it
until the DiLoCo run no longer needs its outputs, since collecting releases it.

**Lifetime.** `lifetime_hours` applies to every sandbox of the run and cannot
be extended on a running sandbox; the platform maximum is 720 hours (30 days).

## Recovery and integrity

- The coordinator authoritatively persists every fragment's global FP32
  parameters, momentum, deduplicating sequence numbers and high-water token
  counters under its root. Fragment commits are atomic, and an independent
  publication step is reconciled after a crash. A worker sequence is
  deduplicated before a pseudo-gradient can be applied again. In the sandbox
  flow that root is on the coordinator sandbox's ephemeral `/work`: the
  launcher does not restart or replace a coordinator, so losing that sandbox
  loses the phase's global state. The in-memory quorum grace timer is restored
  conservatively, not precisely, after a restart.
- Worker checkpoints preserve the model, hybrid optimizer, exact data cursor,
  RNG, local counters, fragment baseline tensors, versions, pending
  submissions, and the actual pending delta payloads. With `checkpoint_s3`, S3 latest
  pointers are published only after every checkpoint file (weights, metadata and, for
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
- Sandbox scratch storage is **ephemeral**. Without `checkpoint_s3` (your own
  bucket; see Prerequisites), a replaced learner cannot recover its inner
  optimizer or exact stream cursor. The launcher does not start replacement
  learners: a replacement is a manual launch with the same worker ID, and the
  coordinator's lease can take up to three minutes to expire before a
  *different* worker session may reuse the slot. Do not treat HTTP coordinator
  state alone as a complete restartable run.
- All workers may start and finish independently, and a failed worker does not
  block fragment updates as long as the configured quorum remains available.
  `finish`, however, refuses to export a phase in which any learner failed.
- Runtime faults and HTTP retry schedules are logged; a missing quorum cannot
  be repaired algorithmically. The operator must restore enough workers.

## Final model and evaluation

The coordinator owns global trainable FP32 fragments, while learner
checkpoints own inner optimizer state and non-trainable buffers. `finish` gives
the coherent **offline** snapshot: it stops the coordinator's server before
running `megatoken diloco export`, and collects the result at
`DIR/coordinator/out/export/` (with `latest.json`). To re-export or evaluate
elsewhere, the same command is `megatoken diloco export --config CFG --root ROOT
--output DIR`, which needs a stopped syncer. Evaluate the collected export with:

```bash
megatoken evaluate \
  --checkpoint DIR/coordinator/out/export/latest.json \
  --dataset PREPARED_DATASET --codec CODEC_DIR
```

Exports do not contain the workers' inner optimizer states or data cursors, so
they are not restartable local-training checkpoints; their `global_step`
reflects the highest fragment outer version, not a synchronized learner step.
They do carry the coordinator's global FP32 `master_weights`, which is what
seeds the next phase.

### Phase transitions

A phase transition is a **new coordinator** initialized from the previous
phase's export. Workers never resume across phases; they start fresh under the
new run ID with their own inner-optimizer state. `init` refuses to replace an
existing coordinator directory, so every phase is a new run (new run `id` and
`decoupled.run_id`). `--resume` warm-starts the global FP32 **weights** from the
previous export (or a training checkpoint); it does not transplant the previous
outer optimizer history.

The previous phase's `finish` already produced the export. The launcher has no
dedicated option for seeding a phase from it: `coordinator.args` is passed
verbatim to `megatoken diloco init`, so `["--resume", "<path>/latest.json"]` works
if that path exists inside the coordinator sandbox. The only delivery
mechanism is the run's `data` (its files reach every sandbox of the run), for
example a peer job serving the export, with the export's files beside
`latest.json` at the same relative paths. Seeding a phase this way has not been
exercised end to end. Outside the launcher the command is
`megatoken diloco init --config NEXT.yaml --root NEW_ROOT --resume <export>/latest.json`.

Owners that are new in the next phase (for example realizer conditions when
packs are first installed) start from initialization; owners frozen in the
previous phase keep their exported weights. The media phases (bridge, joint,
tasks) chain this way; see [media.md](media.md).

## Validation checklist before a large-budget run

1. Run the CPU test suite, including `tests/test_diloco.py`, and
   `ruff check/format --check`. The existing synchronous two-rank Gloo
   suite must remain green.
2. On a Blackwell sandbox, run the repository's hardware gate
   `scripts/validate_blackwell.py` (a `megatoken.cloud` job) and a short
   two-learner `cloud_diloco` run on a real config, not only the tiny-model
   smoke test. Confirm sm_120 kernels and Muon/AdamW moments are finite.
3. Restart the syncer during a quorum merge (the sandbox flow does not
   restart a coordinator, so exercise it with a local coordinator root); verify no version
   applies a worker's sequence twice and public fragments recover from
   authoritative FP32 state.
4. Kill a learner with a pending submission; if `checkpoint_s3` is
   configured against your own bucket, restart it from there and validate the
   optimizer/cursor and no double-credited token budget. Simulate
   late, duplicate, malformed, and stale uploads.
5. Compare one GPU, synchronous DDP if networking allows, and multi-learner
   DiLoCo at **equal global data tokens and held-out loss**, accounting
   for WAN throughput, any `checkpoint_s3` overhead, staleness, and quorum delays.

CPU/unit tests do not establish GPU parity or reproduce full-scale convergence.
The original single-machine and torchrun paths remain available for
baselines.

## References

- [Decoupled DiLoCo (Douillard et al., 2026)](https://arxiv.org/abs/2604.21428)
- [Run a DiLoCo phase (source docstring)](../src/megatoken/cloud_diloco.py)
- [CoreWeave serverless GPU sandboxes](https://docs.coreweave.com/products/sandboxes/serverless/gpu-sandboxes)
- [Sandbox configuration and lifetime](https://docs.coreweave.com/products/sandboxes/client/guides/sandbox-configuration)
- [PyTorch Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
