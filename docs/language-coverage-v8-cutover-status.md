# Native V8 cutover implementation status

The coordinated implementation is tracked in the [main producer PR 47](https://github.com/werg/tracing-js-self/pull/47).
Its description and checked-in ledger retain all 17 requested work packages,
their remaining acceptance conditions, and the two original plans. This branch
starts at `24f4a7c16edf4603cbf5ea6863db1f6a8cc67e37`.

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
checkpoint/resume/export/reload and actual pinned-asset qualification remain open.

## Remaining native work

- [ ] Remove old wire dependencies from effect receipts and remaining replay helpers.
- [ ] Own the complete direct V8 public/state/continuation contract independently.
- [ ] Match producer changes for Promise assimilation, modules and other language families.
- [ ] Compose protocol work with host effects, retained callbacks, child work and recovery.
- [ ] Integrate strict V8 preparation/admission into ordinary recipes and the trainer.
- [ ] Bind data/policy/codec identities on load, resume, stage transition and export.
- [ ] Exercise actual optimization, checkpoint/resume/export/reload through normal commands.
- [ ] Implement live selected-session restoration with acknowledged history and no redispatch.
- [ ] Regenerate/requalify paired artifacts on exact final heads and pass installed absence tests.
- [ ] Complete final default routing, legacy removal, deployment and whole-release rollback gates.
- [ ] Run the separately declared held-out model, comprehension and efficiency studies.

Technical cutover, edition completeness and learned adoption are all **incomplete**.
Existing production routes remain available until their V8 replacements are
qualified. No old receipt is relabeled as proof for a new implementation.
