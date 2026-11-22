# Native V8 cutover implementation status

## Current continuation — 22 November 2026

The separately frozen production support corpus passed original source checking,
independent native admission, exact tokenization/packing, ordinary preparation
and all eight required behaviors in both training and validation at producer
`1e71e5a9385de8f1425593787c3261c20c3a0160` / native
`b276dc3c14075206c1550cbaacb3f456712788b3`. The population is exactly
64 sources: 48 train, 12 validation and 4 test. Preparation performs no optimizer
updates. [Retained support evidence](https://github.com/werg/tracing-js-self/tree/2a34f4ac26a15fe32e7735974afb36a0670f1982/docs/evidence/v8-support-qualified1e71-2026-11-22).

This increment extends the ordinary lifecycle collector to use that actual
prepared population. It verifies a complete immutable source inventory, the
canonical registry/source/dataset binding of the serving sample, exact prepared
public/context bytes, and the frozen split before accepting two optimizer updates,
fresh-process resume, export and service restoration. It does not synthesize a
second sample or count matching source-ID text as source authority.

The exact formatted implementation was checked on validation commit
`a0cfb1d2c2fc152a1716e655619c7ef93c65d82b`: Ruff lint and formatting passed,
and all 11 source-binding/supply controls passed with zero skips in
[run 44987781224](https://github.com/werg/megatoken/actions/runs/44987781224).
The formatter left both files unchanged. These are boundary controls; the
new full 64-source optimizer route is not yet qualified. Recurring paired native
qualification now uses producer `c640f2b1dfed112d5e2bc5f60195234a40c3dec1`.
See [support lifecycle scope and command](SEMANTIC_V8_SUPPORT_LIFECYCLE.md).

**Remaining:** run the paired support corpus through those normal commands on
this exact new executable pair; qualify production recipes/assets and exposure
to all required targets; retain distributed checkpoint/export lineage; measure
held-out learned execution and cost; finish paired language/recovery scope and
operational/default cutover. All 17 acceptance packages remain open in
[producer PR 47](https://github.com/werg/tracing-js-self/pull/47).

The sections below retain historical implementation checkpoints and apply only
to their recorded revisions. Their old pending items are superseded by the
current coordinated PR ledger where later evidence exists.

## Anonymous generator admission — 21 November 2026

The 34-source producer qualification at `aa07a2b0960e24dfa312494dc949b9fc5b98942b`
with consumer `8772c0d9d3be16c848365594d0fe49273e328090` passed all source checks
but admitted only 33 sources. An anonymous generator callback has an empty
function name, which the native validator incorrectly treated as a nonempty
heap identity. The validator now separates function-name metadata from instance
identities and applies the producer's 4096 UTF-16 code-unit metadata bound.
Empty object/instance identities and over-limit metadata still refuse.
Thirty-three focused generator/protocol/native-close tests and 68 subtests pass;
the affected Ruff lint and format checks also pass.

Recurring V8 CI now pins producer `f04f1beb09e94824a15994625e6e43ec326c53e9`,
including both new map/finally and mapped-generator cases. The original failed
admission remains failed; the corrected exact pair requires its own receipt.
The previous pair's ordinary two-update/resume/export/service-restoration run
passed with zero acknowledged records. Actual-model accepted-record recovery,
full module semantics, learned quality and all 17 acceptance packages remain open.

The coordinated implementation is tracked in the [main producer PR 47](https://github.com/werg/tracing-js-self/pull/47).
Its description and checked-in ledger retain all 17 requested work packages,
their remaining acceptance conditions, and the two original plans. This branch
starts at `24f4a7c16edf4603cbf5ea6863db1f6a8cc67e37`.

## Continuation: durable input and complete recovery evidence

Consumer `4918b3d1c65b8113f9334a8afdc1daf9f129a80d` reconstructs and compares
all deterministic acknowledged selection fields and session-opening metadata.
Rehashed token accounting, visible-context and opening-metadata substitutions
refuse. Only wall-clock timings and the expected replay acknowledgement-state
transition are excluded. Fifteen focused CPU recovery/serving/identity/lineage
tests pass, including an actual fresh JSONL service.

The paired producer `52ea51c92500288535d9845ec31392fb6ce8cc51` introduces
per-arrival durable registered host input and later-tail recovery. The consumer
mirrors contract digest
`a3ef5979c34f7151495b78bef19315baf6342fd659bb06a597fffba0817242e0`;
public fixture and interoperability digests remain unchanged. The compressed
region fixture expands byte-for-byte to the producer's original bytes. Twenty-six
focused native/region/recovery tests and fifteen subtests pass. Producer host and
driver validation occurs before model state opens; the journal is private input
evidence and does not enter model context.

These increments do not qualify accepted semantic records from an ordinarily
trained model, arbitrary callback return continuations, complete edition
coverage, distributed/deployment recovery, learned quality or default cutover.
All seventeen acceptance packages remain open in the main producer ledger.

## Delivered increments

- Extracted canonical ECMAScript JSON and typed value/operation validation into
  neutral modules. The V7 client reuses those semantics and retains its wire.
- Added a direct V8 detail codec and switched native V8 replay and model selection
  to it. Canonical bytes, roles and typed validation remain checked.
- Native V8 prefix validation now checks V8 observation envelopes directly. It
  no longer manufactures V6 envelopes or invokes the V6 envelope validator.
- Import-blocking tests verify that direct V8 detail and observation modules load
  and execute with V6/V7 wire modules unavailable.

Validation: 39 existing focused detail/algorithm/native/prefix tests passed;
12 direct V8 detail/prefix tests passed. These populations overlap. The first
direct-prefix run had one error-message assertion failure, corrected without
weakening the rejection. Ruff and whitespace checks passed for changed files.

## Ordinary V8 preparation and training admission

The normal trainer now accepts an explicit `semantic-training-policy/8` with exact
producer/consumer revisions and a complete family-registry digest. The preparation
CLI rechecks source and family/split binding in the pinned producer, verifies raw
tokenizer bytes, runs native replay/context/mask checks, and exports ordinary
prepared shards. The normal loader rebuilds native records and compares the full
normalized episode, including source lineage. Resume keeps the policy pinned.

Source-admitted targets remain teacher-conditioned, with model execution outcome
`unmeasured`; no preparation step marks learned success. Missing or unknown semantic
schemas, unadmitted V8 targets, byte-fallback recipes and wrong tokenizer files are
refused even when the general unverified-data option is enabled.

Validation: five focused controls pass, including prepared-shard reconstruction
and the normal `prepare_batches` path. The positive unit fixture explicitly injects
the source/tokenizer verification boundary; it does not prove a real paired-source
or production-tokenizer run. Ruff and format checks pass. Full ordinary optimizer,
export/reload and actual pinned-asset qualification remain open.

## Direct V8 effect receipts and bounded optimizer lifecycle

Native effect replay now records the canonical V8 request bytes and direct V8
observations. Its result journal uses the producer's V8 lifecycle contract;
request/result handling no longer imports the V6 witness renderer. The paired
contract and region, protocol and callback fixtures were regenerated by producer
revision `31ecf909df31262182ac0bb9664c563aa72919f3`. The callback generator checks
source admission freshly and retains no fabricated historical producer pin.

Validation: 53 focused tests pass across detail, native replay, prediction
prefixes, protocols, training policy and training lifecycle. This includes an
import guard that proves the actual relative V6 import is blocked, plus a normal
CPU optimizer update, saved optimizer moments, and a fresh-process resume that
advances from step one to step two. This is a bounded diagnostic with explicitly
injected source/tokenizer verification boundaries, not production asset
qualification or evidence of learned usefulness. Ruff and format checks pass for
the added import-absence test.

## Remaining native work

- [x] Remove the V6 witness dependency from native V8 effect receipts.
- [ ] Remove old wire dependencies from remaining replay helpers.
- [ ] Own the complete direct V8 public/state/continuation contract independently.
- [ ] Match producer changes for Promise assimilation, modules and other language families.
- [ ] Compose protocol work with host effects, retained callbacks, child work and recovery.
- [ ] Integrate strict V8 preparation/admission into ordinary recipes and the trainer.
- [ ] Bind data/policy/codec identities on load, resume, stage transition and export.
- [x] Exercise a bounded normal-loader optimizer update and fresh-process checkpoint resume.
- [ ] Qualify optimizer/resume/export/reload through normal commands with actual pinned assets.
- [x] Implement direct V8 native model-session checkpoint/reconstruction and producer transport operations.
- [ ] Qualify full producer/host live-session recovery with durable external obligations and no redispatch.
- [ ] Regenerate/requalify paired artifacts on exact final heads and pass installed absence tests.
- [ ] Complete final default routing, legacy removal, deployment and whole-release rollback gates.
- [ ] Run the separately declared held-out model, comprehension and efficiency studies.

Technical cutover, edition completeness and learned adoption are all **incomplete**.
Existing production routes remain available until their V8 replacements are
qualified. No old receipt is relabeled as proof for a new implementation.

## Live native model-session checkpoint/reconstruction

The V8 service now saves acknowledged histories, random state, consumed attempts
and a complete model-continuation fingerprint. A fresh runtime regenerates and
compares every accepted original model fragment, independently validates its
public commit, and requires identical model/cache/retrieval/track continuation.
Unacknowledged candidates remain failed attempt evidence and never become
restored history. Failed reconstruction releases allocated resources and restores
the prior RNG state. The producer transport preserves its own cost/attempt
accounting and verifies the exact restore receipt.

Evidence: 18 serving/recovery tests passed in an isolated checkout before the
partial-allocation cleanup control was added. All six final recovery controls
now pass, including that cleanup case. The one deselected existing test
was the separately tracked synthetic training-admission migration. The actual
tiny CPU model continues with identical next tokens after fresh-process recovery;
accepted-record/callback controls use an explicitly supplied decoder. This is
not a trained-model quality claim. Full host-resource recovery and production
asset qualification remain open. See [live recovery](semantic-v8-live-restore.md).
