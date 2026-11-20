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
