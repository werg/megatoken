# Local smoke results, 23 October 2026

The user authorized these local checks, fixes, appropriate use of the shared GB10,
and internal-drive cleanup. Other users' GPU processes and source changes were
preserved. Evidence is under `/home/werg/megatoken-runs/local-smokes-20261023-01/`.

## Results

| Check | Evidence and result |
| --- | --- |
| Long session | `serve-soak-fixed.json`: 150 GPU chat rounds completed with output cleanup, including 15 side-task lifecycles. A separate session deliberately retained outputs, reached the expected 64-output bound, and recovered after release. Retained attention history stayed within eight events. |
| Exact restart | `gpu-restart/acceptance.json`: two uninterrupted GPU optimizer updates exactly matched one update followed by resume. Weights, optimizer, random/cursor state, consumption counters, and final step matched without numerical tolerance. |
| Older information | `old-fact/acceptance.json`: a fact left the four-event attention history after 16 unrelated messages. Explicit retrieval returned its exact original bytes; a nonexistent source was refused. This checks source retrieval, not learned recall accuracy. |
| Mixed GPU inputs | `mixed-final/acceptance.json`: four input episodes cover English, Korean, linked tool calls/results, and a real recorded media observation. Both completed production donor assets were installed. Four updates plus one resumed update reached step five; every source was actually sampled. |
| Baking and serving | The mixed run baked 61,176 populated native-memory rows with zero lookup difference, removed both temporary donor maps, exported, loaded on CUDA, and generated through the public service after Korean input. Media observations have text targets in this diagnostic; it does not establish decoded media quality. |
| Damaged inputs and interrupted work | Local checks cover corrupted cache chunks and teacher targets, interrupted transfers and checkpoint publication, unsafe output paths, original checksum verification, and split preservation. The final combined selected suite passed 160 cases. A separate asynchronous/runtime selection passed 13 cases, including delayed reads, cancellation, and late rendering. |

Scripts are diagnostic and bounded. They do not establish natural answer quality,
learned memory recall, speech intelligibility, or the quality of the eventual full
training run. Decoder-path checks from earlier work remain separate evidence.

## Corrections

The long-session test's random bundle limited whole input records to 64 bytes,
which could not hold an atomic context-control announcement when a side task
completed. Its test bundle now allows 128 positions. The serving probe's written
finding also names the current `selection_revision` response field correctly.

The first replay check exposed invalid-output delivery handling. The current
concurrent source changes already contain its correction; the recheck passed.
The full source snapshots identify the actual code used rather than attributing
that correction to these new files.

The mixed run uncovered that the earlier Korean correction corpus used episode
format 3 while current training requires format 5. The explicit converter
`python -m megatoken.data.legacy_linear` upgrades only automatically grouped linear
transcripts, preserving records, tool actions, original provenance, and family
links. It refuses native action tapes, simultaneous groups, and custom track
controls rather than inventing their meaning.

The entire Korean addendum was converted into `korean-current/`: all 15,129
originally emitted episodes were retained. Input SHA256:
`932ab4f77611e3add72f3d62932ab42fc8b22823841d71f1a014a329d28dfee8`.
Output SHA256:
`27243ef09c340d6ffbf78a44b080c871a8e67ec2ad8b5f9ca91618583e4d23c6`.
The final mixed run used this converted corpus. It remains a separate addendum;
this check does not merge it into the main prepared-data publication.

The converter also accepts `--prepared-input OLD_ROOT --codec ORIGINAL_CODEC
--output NEW_ROOT`. It verifies the original manifest, index, shards, and indexed
record checksums; copies checked media tensors; retains resolved families and
train/evaluation splits; then publishes a fresh prepared dataset. Tests prove a
held-out example stays out of training and source files remain unchanged. An old
cloud build needs this explicit conversion after it finishes before current
training can consume it. No existing data pin or immutable root was relabeled.

## Reproduction

Use one frozen source copy for an entire run. For example:

```bash
python scripts/accept_local_smokes.py --check restart --device cuda --root NEW_RESTART_ROOT
python scripts/accept_local_smokes.py --check memory --device cpu --root NEW_MEMORY_ROOT
python scripts/accept_local_smokes.py --check mixed --device cuda --root NEW_MIXED_ROOT \
  --korean CURRENT_KOREAN_JSONL --media PREPARED_RECORDINGS --donors DONOR_ASSET_PARENT
python scripts/probes/probe_serve_soak.py --device cuda --requests 150 --output NEW_RESULT_JSON
```

The original source snapshots and per-file overlays are retained with the run.
The final preserved source archive is `source-final.tar.gz`, SHA256
`b726eb6da3db815791296d7eb640bbd0c01bcf7a7af2df0cf71946a773888e64`.
The shared working tree contains other concurrent changes; smoke evidence applies
to these identified snapshots, not every later edit to that tree.

## Internal-drive cleanup

All 152,643 available extracted sound-data outputs were rehashed and matched their
recorded checksums, totaling 31,381,755,078 bytes. The previously missing cloud final
log is recorded in the old rescue receipt; the prepared data itself is complete.
Only the duplicate rescue transport archive was removed:
`/home/werg/megatoken-runs/cloud/media-sound-5-rescue.tar` (31,663,073,280 bytes).
The extracted dataset is retained. `cleanup-archive-verification.json` and
`cleanup-receipt.json` record the checks and removal. This freed approximately
31.7 GB on the internal NVMe drive; immediately afterward about 106 GiB was free.
