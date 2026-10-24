# Internal-drive cleanup, 24 October 2026

**Subsequent user instruction:** remove all Megatoken checkpoints and retain the
hindsight-kb downloads. The retention described in the first cleanup below is now
historical; the subsequent removal is recorded at the end of this document.

The user requested freeing local storage. Only older intermediate checkpoints from
two completed Megatoken diagnostic runs and dangling package-cache entries were
removed. Original data, downloaded donor assets, serving exports, the newest saved
checkpoint in each run, and unrelated projects were retained.

Removed four unpinned periodic checkpoints:

- `dynamic-acceptance-local-08/training/checkpoints/B0-000000002-periodic`
- `dynamic-acceptance-local-08/training/checkpoints/B0-000000004-periodic`
- `dynamic-acceptance-local-11/training/checkpoints/B0-000000002-periodic`
- `dynamic-acceptance-local-11/training/checkpoints/B0-000000004-periodic`

These directories totaled 32,698,274,010 bytes (about 32.7 GB). Before removal,
the latest pointer, checkpoint metadata, saved-state checksum, and memory attachment
checksum were verified for both retained checkpoints. No active command, open file,
or memory mapping referenced the removed checkpoint directories. Their metadata
and expected checksums are retained in the cleanup receipt. These are historical
artifacts; checksum verification does not imply compatibility with current training.

Retained newest checkpoints: update 6 for run 08 and update 5 for run 11, together
with both serving exports. Post-cleanup checks verified the pointers, metadata,
files, and sizes again. `uv cache prune --offline` separately reported removing
1,719 dangling cache files totaling 63.1 MiB, using its normal in-use checks.

At final measurement, the internal drive had 103,473,414,144 bytes free
(about 96.4 GiB). Other work was also changing disk usage, so the total drive-space
change must not be attributed entirely to this cleanup.

Detailed receipts are outside the repository at
`/home/werg/megatoken-runs/storage-cleanup-20261024/`, including retained-checkpoint
verification, the exact removal plan, the cleanup result, and final storage summary.
No cloud workload was launched or restarted.

## Complete removal of diagnostic checkpoints

The user subsequently authorized removing every Megatoken checkpoint because
production training has not started, while preserving anything downloaded from
hindsight-kb. The project run tree, repository artifacts, project scratch workspace,
and project-specific temporary test directories were inventoried using checkpoint
and serving-export metadata. No candidate was referenced by an active process,
open file, or memory mapping.

Removed 242 checkpoint directories and 49 derived serving-export directories,
totaling 52,786,056,310 bytes (about 52.8 GB). Obsolete latest-checkpoint pointers
were removed as well. Source data, donor downloads, decoder packs, source snapshots,
and diagnostic reports were retained. A second scan found no remaining checkpoints
or serving exports in those project scopes.

The downloaded writer in
`/home/werg/megatoken-runs/hindsight-kb-writer/` was explicitly excluded. Its seven
files total 1,063,026,862 bytes. Every file was hashed before and after removal;
all bytes and checksums are unchanged. The separate `~/hindsight-kb` project was
not modified.

The internal drive had 157,106,704,384 bytes free immediately afterward, about
146.3 GiB. Exact deletion metadata, preserved-download checksums, active-reference
checks, and the final scan are retained in
`/home/werg/megatoken-runs/storage-cleanup-20261024/remove-project-checkpoints/`.
