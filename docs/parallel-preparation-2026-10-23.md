# Parallel preparation follow-up, 2026-10-23

This supplements `handover-2026-10-23.md`. The user authorized recovery, quality review,
production donor extraction, and a queued GPU training/export/serving check in parallel.
Long cloud jobs below are running or queued, not completed results.

## Recover existing preparation

Healthy chat ingestion continues in its original eight cloud sandboxes. Only the
local main collection controller (PID 1918599) is paused: the original failure handler
would otherwise destroy a failed sandbox before its data could be recovered.
Do not resume it until the independent recovery controller verifies all eight jobs.

The corrected streaming JSON reader and transient reader retries are committed in
`78ed285`; its exact snapshot is stored under
`/home/werg/megatoken-runs/cloud/recovery-snapshot/`.
The failed AFM chat portion is accepting rows through this reader. Coding pilot
portions 0–11 are also being repaired sequentially while its original ingestion
continues separately. Existing plans, source cursors, codecs, and shard fingerprints
are preserved. A shard fingerprint legitimately differs from other shards: recovery
recomputes each against its own sources and settings.

Independent detached recovery monitors live in
`/home/werg/megatoken-runs/cloud/recovery/`:

- Chat: PID 4075592, `controller-chat-state.json`, `controller-chat.log`.
- Coding pilot: PID 4076229, `controller-coding-pilot-state.json`,
  `controller-coding-pilot.log`.

Check `controller-pids.json` and live command lines for newer PIDs before signalling.
Chat completion must never wait for coding completion. Recovery waits for original
writers, checks source cursors and SQLite integrity, preserves original status and
output-manifest backups, then regenerates checksummed outputs using the original
output scope. It only marks success after verification. The coding pilot's original
output scope does not include raw shard databases. Its old collection watcher died;
a replacement collector must start after verified repair.

## Quality review and Korean correction

Review artifacts are outside the repository in
`/home/werg/megatoken-runs/cloud/data-quality-review/`.
The bounded accepted-episode sample showed no orphan observations or incorrect
ordering. Invalid examples found in rejected rows were already excluded. This is a
sample, not proof about every accepted episode.

A confirmed selection error used `conversations` (English) instead of
`conversations_ko` in `heegyu/glaive-function-calling-v2-ko` at revision
`c8b54e70bfbdd2fd6be037557e6feaf1abf14f5d`.
The correction script `scripts/prepare_korean_correction.py` creates a separate
addendum, preserving separate system prompts and valid function schemas, retaining
unusual descriptions as unscored context, and linking each translation to the exact
original English family keys. It does not change active source plans or databases.

A real ten-row preflight adapted all ten rows and preserved all nine tool calls and
nine results. Four focused tests pass. The full correction expects 15,170 source
rows; inspect the generated receipt and checksum before import. Builder-emitted
rows can exceed native-import accepted rows because the normal scored-target quality
check still applies. This addendum is not part of the main data pin until a global
family-dedup preparation explicitly includes it. The full correction is now running at low priority on existing chat sandbox
`bd5d3655-8d06-44e5-a487-6b53b27ba811`, remote PID 9364. Output is
`/work/korean-correction-out`; log and exit marker are
`/work/.cloud/korean-correction.log` and `.exit`. A separate collector must compress
and retrieve the addendum, then verify its JSONL hash and counts. Chat controller
release is gated on the local quality-review `korean-correction-collected.json`
record having `verified: true`, to prevent collection from stopping this machine
before the separate addendum is safe. Current execution status and collection
details are recorded in `korean-correction-source-plan.json` outside the repo.

## Production donor extraction

Both DeepSeek-V4.1-Flash-1017 and Qwen3.8-Flash-Next native n-gram memories are required
production inputs, not an optional quality experiment. The CPU extraction requests
65,536 contexts per donor from verified data revision
`ba529dbe8451422e9201475f8fa0d393777a4abe`. This trial corpus has three agentic
sources; it does not cover every source in the still-running full chat build.

The initial production attempt failed because safetensors tried mapping a roughly
100 GB source shard into the runtime address space. Commit `2ce0092` replaces
whole-file mappings with bounded local reads after full download/hash verification.
A regression uses a sparse 100 GB file and verifies that selecting two rows reads
only those rows. Fourteen relevant range/extraction tests pass.

Restarted cloud sandbox: `06807d47-0ac6-4398-af1d-a742d18a562a`.
Detached collector PID: 3655286. Exact snapshot: `2ce0092`.
Local state/logs: `/home/werg/megatoken-runs/cloud/production-donors/`.
The previous failed attempt is preserved in `attempt-before-bounded-local-reader/`.
Large source tables remain cloud-side; only compact prepared assets are collected.

## Queued GPU verification

Detached controller PID 2695021 lives in
`/home/werg/megatoken-runs/cloud/gpu-readiness/` and uses snapshot `78ed285`.
It replaces the former idle disk-only retry loop; do not start a duplicate.
At last check, no GPU had been allocated. Creation refusals included the account's
concurrent-sandbox quota and runner capacity; these do not establish a GPU shortage.
It retries for up to six hours and records each refusal category.

Once allocated, the bounded one-GPU job checks actual GPU operation, real download
and disk throughput, training, checkpoint resume, both native donor assets, projection
baking, served export/load, and a served-session benchmark. CPU donor checks already
passed earlier, but they are not GPU evidence. Results remain pending.

## Shared workspace

Substantial unrelated model/data-schema changes are active in the shared worktree.
Do not stage, revert, or include them in these commits. Cloud launches above use exact
committed archives. Merge remote changes in an isolated worktree if needed to publish
these fixes without disturbing the concurrent edits.

## Verified update at 13:28 UTC

The Korean correction completed and was collected with an independently verified
checksum: 15,170 input rows, 15,129 emitted episodes, 41 rejected rows (24 English
adaptation and 17 Korean adaptation schema errors). Every emitted episode retains
its system context and links to its English family. The local JSONL is
198,945,524 bytes, SHA256
`932ab4f77611e3add72f3d62932ab42fc8b22823841d71f1a014a329d28dfee8`.
It is stored in the external quality-review `korean-correction-corpus/` directory;
the compressed archive is 34,409,950 bytes. The atomic collection marker now has
`verified: true`; this collection dependency is satisfied. It still needs inclusion
in global family dedup before becoming part of the main data pin.

GPU request `ef290158-a13f-440c-8b66-809cafa39879` was accepted at 13:14:11 UTC.
At 13:26:41 UTC it remained in startup, without evidence that the workload ran;
this is not yet GPU validation. The controller owns its 15-minute startup timeout
and cleanup. See external `gpu-readiness/startup-wait.json` and subsequent logs.

## Monitoring and donor completion, 13:35 UTC

Both production donor extractions completed successfully and were collected:
`production-donors/controller.json` has stage `complete`, collection status 0,
eight output files totaling 35,557,705 bytes. Each donor used the requested 65,536
selected contexts from the verified trial corpus. DeepSeek produced 41,317 covered
byte-memory rows; Qwen produced 33,146. These are populated destination rows, not
counts of all memories in either donor model. Descriptor shapes, checksums,
licenses, expected row-space settings, and finite floating values were checked
locally. Verification record:
`/home/werg/megatoken-runs/cloud/production-donors/verified-assets.json`.
The GPU readiness job still uses the earlier small real-donor assets; do not imply
that this pending check validates training with the newly extracted larger assets.

Read-only progress monitoring is running every five minutes for up to 36 hours:
`/home/werg/megatoken-runs/cloud/progress-monitor/monitor.py`.
Its current PID is stored in `progress-monitor/process.json`; check the command
line before signalling. It uses the committed integration worktree for cloud
client imports; do not remove that worktree while monitoring is active.
Each sandbox inspection runs in a separate process with a 75-second timeout and at
most four concurrent inspections. It checks actual source cursor/database activity,
cloud disk space, repair outcomes, donor output, GPU state, and controller liveness.
Episode counts are sampled every third check. Counts remain candidates before
cross-shard global deduplication. Latest snapshot, history, and alerts are
`progress-monitor/latest.json`, `history.jsonl`, and `events.jsonl`.
Monitoring does not alter cloud jobs or send external messages. Recovery, collection,
and GPU retry controllers retain ownership of their jobs and cleanup.

At 13:32 UTC, the chat shards contained 3,727,381 candidate episodes, and all eight
chat machines had recent database activity. At 13:34 UTC the data repairs were still
running. GPU request `ef290158-a13f-440c-8b66-809cafa39879` timed out before startup
and was stopped; replacement `988dd7d8-b445-4a53-93b8-2095d50551c1` was accepted at
13:30:48 UTC and remains in startup. No GPU training result exists yet.

Recovery monitor PIDs changed once more after the safety checks were finalized:
chat 4109862, coding pilot 4110870. Prefer the current external
`recovery/controller-pids.json` over any historical PID in this document.

## GPU result and retry, 15:31 UTC

At 15:23 UTC, chat preparation contained 4,988,553 candidate episodes before global
deduplication. Four chat machines had reached outputs-ready. The recovered worker's
original exit code remains 1 pending the final audited reconciliation; all its
source portions are complete. Coding repair continues independently.

The first running GPU check used an RTX PRO 6000 Blackwell Server Edition with
101,975,851,008 bytes GPU memory. Real download of 2,152,819,114 bytes took about
3.25 seconds; BF16 math, runtime checks, and the first optimizer/checkpoint step
passed. Resume failed: dynamically reconstructed additional donor buffers were
on CPU while their learned map and model were on CUDA. The failed job was collected
and stopped; its evidence is preserved in external
`gpu-readiness/attempt-before-device-fix/`.

`_NGramImport` now initializes its buffers on the learned map's device, preserving
FP32 frozen descriptors. A new restore regression reproduces the failure with the
meta device without needing a GPU, checks CPU value parity/gradients, and offers
explicit opt-in CUDA coverage. Focused frontend/native import checks: 16 passed,
one optional CUDA case skipped. The readiness wrapper also now requests exactly
one resumed update: its update limit applies per invocation, so the expected
checkpoint step is 2, not 3.

A new detached GPU retry controller is running (current PID in external
`gpu-readiness/process.json`) with exact isolated snapshot `61533b1`, containing
only the donor-placement and resume-limit fixes on the prior GPU snapshot line.
It now uses both newly completed production donor assets. Original failure reports
are preserved; the corrected GPU training/resume/export/serving result is pending.
Do not remove `gpu-readiness/device-fix-code/` while this controller is active.

## Smaller cloud requests and local GPU use

The user authorized appropriate local GPU work and requested smaller cloud resource
requests when capacity is exhausted. This preference is recorded in `AGENTS.md`.
The bounded two-donor checkpoint restore regression passed on the local GB10, along
with its CPU and meta cases. Existing local processes were left running.

The readiness job now requests 4 CPU / 16 GiB RAM / 1 GPU, with an explicit fallback
to 2 CPU / 16 GiB RAM / 1 GPU. The memory floor and GPU count are preserved. Large
preparation jobs do not inherit these small-job sizes.

The reusable `megatoken.cloud_capacity.start_with_fallbacks` launcher preserves the
workload, inputs, snapshot, and secrets, advancing only through job-declared resource
alternatives after runner-capacity rejection or a startup timeout. The existing
cloud launcher owns failed-start cleanup. Account quotas, authentication errors,
gateway failures, and upload errors propagate without reducing requirements.
The external readiness retry controller now calls this helper, and its production
asset job uses the same smaller alternatives. No already-running job was resized,
and the successful GPU check was not relaunched.

The prior corrected cloud GPU check has passed: optimizer step, checkpoint resume
to step 2, bake/export/load, and served-session inference, using both completed
production donor assets. Its reports remain in external `gpu-readiness/collected`.
