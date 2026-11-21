# V8 production-launch implementation evidence

The ordinary launch qualifier now accepts a production V8 training policy and
keeps the original production recipe, donor, teacher, repository, checkpoint,
hardware and authenticated coordinator requirements. Its direct corpus route
checks every admitted row, both real training/validation splits and the actual
ordinary stream identity. The shared window configuration was moved without
changing its behavior. No legacy experiment or wire validator is required on
the V8 path.

The checkpoint/coordinator check uses the actual earlier gate results; the
previous code looked for those results inside the unrelated engineering receipt.
Same-stage resume must match freshly derived V8 corpus and stream lineage.
Previous-stage bootstrap validates its checkpoint but contributes weights rather
than inventing a V8 continuation. Destination coordinator identity is checked
independently against the current corpus and stream.

## Results and limitations

The complete local new test file reports **11 passes, one failure, zero skips**
in17.60seconds. The failure is the mandatory actual-model checkpoint control:
this environment has no `torch` installation. That control and the existing
checkpoint/coordinator suite must execute in hosted CI before actual-model
qualification is credited. The failed local attempt remains preserved; it is
not relabeled as a pass or optional skip.

The11 passing controls cover real native replay, physical preparation/export,
both production split gates, missing validation, insufficient worker supply,
changed stream identity, checkpoint/coordinator lineage controls, YAML/JSON
duplicate-key rejection and code/checkouts. A fresh process successfully runs
the actual V8 corpus and ordinary stream with nine legacy modules blocked.
The preparation fixtures explicitly inject source/tokenizer audit boundaries;
the gate-wiring control additionally injects repository identities and expensive
asset/provider boundaries. These are implementation controls, not source
qualification or production readiness receipts.

Independent review reproduced two additional defects and confirmed the repairs:

- Valid empty prepared roots contributed to ordinary training identity but were
  omitted from preflight lineage. Preflight now uses the exact same complete
  root inventory, and a real empty-root regression is retained.
- Caller-supplied clean checkouts could be mistaken for the executing consumer.
  Qualification now requires distinct repository roots and binds the actual
  imported Megatoken package to the supplied consumer checkout. The code itself
  must be committed and clean. A nested directory cannot masquerade as a
  repository root.

Before and after probes, failed development attempts and final logs/XML are
retained unchanged. `manifest.json` records hashes of the implementation files
and every compressed evidence file. The source/tokenizer and provider-injection
boundaries remain explicit in both tests and manifests.

No learner or optimizer was started. Required production SE child/repeated
invocation/source-reread supply remains unsatisfied by the existing small V8
population. Complete asset-bound launch, memory fit, recovery under load, learned
quality and default cutover are separate open acceptance gates.
