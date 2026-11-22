# Router count receipt validation — 22 November 2026

The original reader accepted internally contradictory router count histories. This increment rejects those contradictions and preserves valid receipts from actual training. It changes receipt validation, its regression tests, and the scope explanation only. Router capture, optimizer execution, loss semantics, bootstrap recovery, policy pins, and the two frozen hosted coverage campaigns are unchanged.

## Exact revisions and regression evidence

| Role | Native revision | Result |
| --- | --- | --- |
| Actual optimizer runtime and original reader | `2f3e6cf04aad60473e8ecc67e846e81e5b9dec83` | Controlled training, checkpoint resume and stage transition executed before the reader patch |
| New negative regressions on the unchanged runtime | `1aafb80b4ac0c3da7de71969e107d7aa7382ab8e` | 12 expected rejections failed; 6 controls passed; 0 skips |
| Strengthened reader and scope documentation | `19d011247e9f1883db632121ca6d8c591abc875d` | Same 18 controls passed; 0 skips |
| Broader receipt controls on the strengthened reader | same revision | 54 passed, 1 full-optimization fixture deselected, 0 skips |

The twelve red failures are retained in `red.log` and `red.xml`; they are expected `DID NOT RAISE` failures. The synthetic states do not show that the trainer actually lost or invented counts. They demonstrate that a coherent resealing of contradictory metadata was accepted by the original validator.

The reader now requires empty count maps before the first audited update, at least one assignment for every recorded row, no assignments without rows, valid per-update increments even after earlier adapter fan-out, and a monotonic bounded resume history. A resumed history cannot remove or invent selectors once auditing has started, reduce counts, change them without a new receipt, or retroactively invent more than one audited update. Explicit partial histories remain partial.

## Actual execution and read-only revalidation

Before the patch, 47 distinct selected tests were exercised against exact runtime `2f3e6cf`. The initial invocation passed 46 and skipped one test because its pinned production tokenizer path was absent. A separate invocation of that same test with the verified tokenizer path passed, resolving the sole skip. Both original JUnit files are retained; the initial skip has not been erased or presented as a zero-skip initial run.

The actual small controlled runs produced three optimizer updates for a complete fixture, one update in a subsequent training stage, and a separate two-update checkpoint/resume lifecycle. Source and tokenizer admission are injected for these byte-codec fixtures. These are six actual updates, not the production 348-update support pass and not evidence of learned behavior.

The stricter reader then revalidated those unchanged artifacts without optimization. The audit independently rebuilt both prepared position inventories, checked four position states against their checkpoint metadata, reconstructed both logged target-presence histories with their original seals, and reproduced the exact complete-position and bounded-resume comparisons. It separately applied all six recorded router-count deltas to counter-only shadow states and recovered the original cumulative maps exactly. The stage archive remains intact and a fresh stage begins with empty counts. Counter-only shadows grant no text-position or optimizer credit.

| Actual receipt | Audited optimizer updates | Recorded rows | Recorded expert assignments |
| --- | ---: | ---: | ---: |
| Complete controlled fixture | 3 | 15,810 | 16,003 |
| Subsequent stage | 1 | 7,575 | 7,668 |
| Resume lifecycle, first checkpoint | 1 | 7,575 | 7,668 |
| Resume lifecycle, final cumulative checkpoint | 2 | 12,523 | 12,655 |

The first and final resume receipts overlap; their totals must not be added. Rows and assignments differ because an adapter row can select multiple experts. All four states retain `routerInternalPositionCoverage: unmeasured`.

The initial read-only audit helper omitted an actual model checkpoint-stage override and was correctly refused by exact objective comparison. `audit-helper-correction.json` and the original helper bytes record this audit-only correction. No training checkpoint or receipt bytes were altered. The final helper reconstructs the trainer's actual overrides and retains its own hash in the report.

## Scope and reproducibility

These counts cover forward routes recorded for router balancing. They exclude reconstruction readouts inside `discard_routing_losses`, no-gradient work, auxiliary routes after recorder detachment, and routes excluded by the existing replay guards. Complete per-update accounting means every optimizer update has a receipt from that recorder. It is not an independent exhaustive reconstruction of every model-internal router row; that coverage remains unmeasured.

`source-map.json` lists the three exact changed source blobs. `actual-complete-proof.json.gz` and `actual-resume-proof.json.gz` preserve the original proof JSON bytes. `actual-inputs-runtime-reports.tar.gz` preserves the two prepared fixtures plus configuration, runtime reports, training metrics and checkpoint metadata; its exact members and hashes are in `actual-archive-members.json`. The non-tensor final resume payload is separately exported with the original tensor-checkpoint byte count and SHA-256. Tensor checkpoint files, the intentionally forged negative-test checkpoint, and duplicate pytest symlink aliases are excluded from Git.

The read-only helper was executed using the original local test roots and checkpoint. It loads the original checkpoint to verify the exported proof and requires those local tensor bytes to run unchanged. The retained non-tensor export, proof JSON and metrics allow direct inspection of every receipt checked by the audit without retaining predictions, activations or model weights.

The exact commands and source identities are in `commands.json`. `file-manifest.json` covers every retained file except itself. Neither hosted campaign has been rerun, retargeted, cancelled or relabeled by this validation work.
