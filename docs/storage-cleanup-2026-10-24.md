# Internal-drive cleanup, 24 October 2026

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
