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

The actual 64-source paired optimizer route needs new-head evidence. This
57-case gate does not exercise a real multi-rank training run or prove distributed
failure recovery. Full production recipes/assets, complete required-behavior
target coverage, held-out learned programs and cost remain separate gates.
Reports retain `allTargetsExposed: false`, `fullLaunchQualified: false` and
`learnedSuccess: null`; two updates cannot establish those broader claims.
