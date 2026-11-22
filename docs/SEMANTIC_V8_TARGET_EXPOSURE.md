# What the ordinary V8 optimizer actually scores

The trainer now records which admitted source targets contribute positive-weight
loss denominators to successful optimizer updates. A prepared corpus proves
available supply; this census measures the selected training windows.

## Count definitions

- **Sources:** distinct frozen training source identities with a scored target.
- **Logical targets:** distinct source/target identities, independent of framing.
- **Physical records:** distinct source/target/physical-record identities.
- **Scored occurrences:** every scored occurrence in selected windows, including
  repeated windows and duplicate occurrences on different data ranks.
- **Loss denominators:** the summed actual prepared denominators for objectives
  with positive effective optimizer weight. A zero denominator, masked target,
  observed-only operation or zero-weight objective adds no target exposure.

Unique coverage is deduplicated. Scored occurrences and denominator sums are not.
The obsolete per-operation router-balance objective has zero effective weight;
the separate global replacement does not become a source-target exposure claim.
These counters do not measure the magnitude or usefulness of a learned update.

## Actual optimizer and checkpoint path

The census consumes ordinary prepared Meridian batches after target planning,
rather than reconstructing exposure from source text or declared behavior labels.
Each occurrence must carry the exact frozen registry, a training split, source
identity, logical target and physical record. Read-only Mapping denominators
are accepted; malformed or negative counts are refused.

The distributed gather, merged census and candidate next state are validated
before optimizer mutation. The new state becomes current only after the
optimizer step and required synchronization/DILoCo work succeeds. An unscored
sampling round does not earn an optimizer-update credit.

Checkpoints retain the full census, global-step bounds, cumulative counts and a
digest of the immediately preceding census state. Same-V8 resume must retain the
census and its exact registry/global-step identity. A fresh non-V8 stage boundary
has a separate guarded initialization path; this helper is not evidence that
every ordinary cross-stage policy is already qualified.

The two-update lifecycle collector verifies step one, fresh-process step two,
the exact parent digest, preserved source/target/record identities and cumulative
counts. It reports both checkpoint digests, unique coverage, actual occurrences
and positive-weight denominators. Support mode also restricts observed sources
to the exact immutable 64-source inventory. Hash links are integrity checks,
not signatures or independent proof that arbitrary supplied data is true.

## Complete support-target coverage gate

The fixed support population has 64 sources: 48 training, 12 validation and four
test sources. `scripts/qualify_semantic_v8_target_inventory.py` independently
enumerates every ordinary training window and its native loss plans. It binds
the resulting inventory to the exact prepared manifest, tokenizer, frozen registry,
model, resolved objective and window settings. It performs **zero optimizer updates**.

Run that preflight with the ordinary training configuration and immutable source
inventory:

```sh
python scripts/qualify_semantic_v8_target_inventory.py \
  --config training-config.json \
  --expected-source-ids support-source-ids.json \
  --output target-inventory.json
```

The program ID comes from the verified semantic annotation and its source/registry/
split binding. The prepared lineage adapter name is shared by many programs and
is not a program identity. Observed records with no loss plan earn no target credit.
The report separates all planned windows from windows that make an optimizer update,
and independently measures the complete pass against the configured phase budget.
It also derives every policy-masked physical target directly from the complete
prepared episodes and requires that exact set in the plans. A window iterator or
planner cannot silently omit a prepared target and shrink both sides of the later
coverage comparison. Observations without policy masks are excluded from that set.

A separate fresh ordinary training invocation uses the same objective and inputs,
with `max_steps` set to `optimizerWindowCount` from that inventory. Writing the final
checkpoint at that same interval avoids serializing the model after every window.
Output location, checkpoint/log cadence and the update bound are excluded from the
objective identity; source data, masks, weights, model and window settings are not.

```sh
python -m megatoken train --config coverage-training-config.json
python scripts/qualify_semantic_v8_target_coverage.py \
  --checkpoint final-coverage-checkpoint \
  --prepared prepared-support \
  --expected-source-ids support-source-ids.json \
  --inventory target-inventory.json \
  --output target-coverage-evidence.json
```

The final collector rebuilds the inventory and compares the actual checkpoint
census. Success requires every positive-weight physical and logical target, exact
scored occurrence counts, exact window sets, exact per-loss denominators and the
complete bounded optimizer-update count. A key-presence match alone can pass
`allPositiveWeightTargetsExposed` while failing `exactOnePassCensus`; the complete
gate requires both. Validation/test data are not optimized. Per-loss denominator
counts overlap and must not be summed into a distinct-token count. Token/mask-position
coverage and learned success remain unmeasured by this gate.

The producer's ordinary support lifecycle retains its bounded two-update route
by default and compares that partial census with the complete inventory. Select
`--full-target-coverage` to additionally perform the fresh complete optimizer pass.
That selection is valid only with the explicit prepared support corpus, immutable
source inventory and admitted sample. Reports distinguish the request, status and
qualification: a bounded lifecycle can pass while
`fullTargetCoverageQualified` remains false and `targetCoverageStatus` is
`not-requested`. The dedicated full-coverage workflow supplies the explicit flag
and requires the final coverage evidence plus `fullTargetCoverageRequested: true`,
`fullTargetCoverageQualified: true` and `targetCoverageStatus: passed`. Omitting the
full pass cannot satisfy that gate. Its target step is bounded at two hours inside
a three-hour dedicated job; routine source lifecycles keep their existing budget.

Ordinary models retain at most eight admissions for immutable prepared episodes.
The entire episode, codec instance and fingerprint, current registry/policy bytes,
geometry and required split form the reuse key. The registry is reread for every
window. Reuse avoids repeating complete native input validation for adjacent
windows but never retains model activations, generated state or loss plans. Fresh
models and resumed processes start with no retained admission results.

## Hosted evidence

[Run 44990761001](https://github.com/werg/megatoken/actions/runs/44990761001)
tested candidate `910d45bf059c53ef1b7a537d81274a8afad659b9` after Ruff formatting.
**57 controls passed, zero failed or skipped**, including real optimizer updates,
fresh-process resume, the exact pinned production tokenizer, bounded coordinator
initialization/export/reload, and lifecycle report refusal controls. This commit
contains the exact formatted source that ran. The lifecycle test was unchanged
by formatting, as checked by the retained diff step.

Earlier failures are retained: immutable denominator mappings, a brittle
negative-test subprocess replacement, an unused test variable, and a workflow
context error before jobs could start. The first probe's three missing-tokenizer
skips were resolved by provisioning and verifying the exact production asset.

## Limits and next acceptance work

The historic 57-case gate does not exercise a real multi-rank training run or prove
distributed failure recovery. Complete support-target exposure is credited only
when the separate actual full-window gate above passes on the stated pair; two
optimizer updates alone cannot establish it. Full production recipes/assets,
token/mask-position coverage, held-out learned programs and cost remain separate
gates. Diagnostic coverage reports retain `fullLaunchQualified: false` and
`learnedSuccess: null` even when every required target has contributed.
