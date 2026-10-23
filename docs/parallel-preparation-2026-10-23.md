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

- Chat: PID 3772602, `controller-chat-state.json`, `controller-chat.log`.
- Coding pilot: PID 3773018, `controller-coding-pilot-state.json`,
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
family-dedup preparation explicitly includes it. Current execution status and cloud
location are recorded in `korean-correction-source-plan.json` outside the repo.

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
