# Causal V8 training coverage

The ordinary V8 census derives behavior from native-replayed public records and
checks that their complete logical payloads, order, roles and target masks are
the records actually prepared for training. It does not trust curriculum labels
in annotations. The shared logical renderer is also used by preparation, so the
census cannot silently check a different spelling of the model's observations.

Detailed operations live inside generated `semantic` records. Scanning only the
outer kind missed real requests and suspension/resumption. The census now
decodes those details and requires causal evidence:

| Behavior | Required history |
| --- | --- |
| Code invocation | A checked invocation and generated training targets. |
| Repeated invocation | More than one checked invocation of the same published program in the same session. |
| Tool feedback | A request, its matching observed result, and later generated work in that invocation. |
| Async | A checked await and matching resumed continuation in that invocation. |
| Callback | A natively checked callback activation. Ordinary callback work does not count as a typed child task. |
| Source reread | An eviction followed by a read of the same session, track, program and source digest. Initial reads do not qualify. |
| Recovery/cancellation | The existing checked reconciliation, reattachment or cancellation observations. |

Native replay independently requires cache observations to name an already
published program and its exact source digest. A reread must reproduce that
program's original source bytes. Recomputing prefix hashes after changing the
source or inventing a program does not make the observation valid.

Preparation uses complete targets without custom byte spans or recorded token
mask overrides. The census requires that same shape. Clearing the actual token
loss through `free_spans` or `emitted_free_mask` while leaving the coarse
`policy_mask` true must not earn coverage. The ordinary episode validator also
retains physical part, meridian and role integrity checks.

## Evidence and boundaries

The paired `semantic-source-recovery-v8.json.gz.b64` fixture was freshly
collected and rechecked at clean producer
`a494f007f7373fa4d96b76be88c5610fb815d6e8`, tree
`ec26fb944e9061c0478771ad7cc4bfa3860c86ed`. It preserves the original counter
source and inputs, adding only the explicit checkpoint-at-first-await projection
option. Its 20 public records include a real checkpoint, eviction, exact source
reread, reconciliation and resumed execution; it dispatches each of its two
effects once. The frozen 66-case population is unchanged.

The final local selection has **30 passes, zero failures or skips**: 27 census
and native cache controls plus three existing ordinary prepare/export/reload
controls with eleven legacy imports blocked. The export controls inject their
source/tokenizer audit boundary explicitly. The source-side fixture selection
has three passes, including fresh recollection/recheck and altered-source
refusal. Source receipts retain their original environment; tests recollect
instead of rebinding old receipts to new code.

Independent before/after probes retain two reproduced defects. First, clearing
target byte spans changed eligible byte-codec positions from 1,199 to zero while
the earlier census still reported all six behaviors. Second, edited source-read
bytes or an unpublished cache identity still passed native replay after prefix
rehashing. Both now refuse; the unchanged original retains its six behaviors
and 1,199 eligible positions. These are bounded replay/census defects; no
complete production source-admission bypass was demonstrated.

The broader local policy selection has four passes and three missing-PyTorch
failures, with six readiness cases unselected. Those failures remain recorded
and require hosted execution. Byte fallback in focused tests qualifies framing
and masks, not production tokenizer admission or learned task success.

Typed child orchestration currently has a separate coordinator history. That
history does not manufacture guest training records or `semantic_child` coverage.
Source-certified child data, full production SE supply, real production assets
and the complete launch/lifecycle gates remain open.

Reproduction:

```sh
PYTHONPATH=src python -m pytest -q tests/test_semantic_v8_behavior_census.py tests/test_semantic_v8_export_without_legacy.py
```

The exact failed attempts, source hashes and independent probes are retained in
`docs/evidence/v8-behavior-census-2026-11-22/`.
