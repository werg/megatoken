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
