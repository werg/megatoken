# Exact support-target coverage: canonical source identity

The previous actual 64-source campaign at producer `0694fa9280712fa1364ead46a477c757a0ca4bfa`
and native `dbc89dcd52cb93d1d9fdca2989fab64b85e5d427` failed before its first optimizer update:
[run 45005718379, job 137156743990](https://github.com/werg/tracing-js-self/actions/runs/45005718379/job/137156743990).
All 64 source/native/token/packing and strict 48/12/4 preparation gates passed.
The complete failing inventory-stage log is retained alongside this note. The complete
original job and input artifacts are in GitHub artifact `14747074516` (archive SHA-256
`aa58746ea26d49a67333f7e7812e1cad585c6fbc2d0093592d6fb39dc1c3f792`).

The inventory incorrectly used `episode.lineage.source_id`, which is the shared
prepared-data adapter name `semantic-execution-v8`. The ordinary trainer's scored
census uses `sourceId` in the `semantic_execution` supervision annotation, bound to
the exact frozen registry and split by `datasetBinding`. Inventory now uses that
same identity, including on unscored windows, after complete native preflight.
It checks the source, registry and training split again instead of falling back
to the adapter name. Distinct source programs may legitimately share adapter lineage.

Twenty target-inventory controls passed locally with zero skips, including two
different programs with shared lineage and refused source/registry/split substitutions.
This is an implementation control result. It does not by itself establish complete
optimizer coverage. Full ordinary optimizer exposure, exact window/occurrence/loss
denominator census, token-position coverage and learned success remain separate gates.

A direct run against all retained prepared episodes then reached a second actual
planner boundary: an observation-only step contains records but has no loss plans.
The inventory's available-target loop incorrectly zipped those unequal sequences.
It now skips steps without loss plans, exactly as the optimizer exposure collector
does, retaining strict record/plan pairing whenever a loss plan exists. The complete
local failure is retained as `observation-only-inventory-failure.log`; no optimizer
work had started in that attempt either.

The full training split contains 348 ordinary windows across 48 sources. A direct
measurement found adjacent windows from one immutable source each spent about
2.5 seconds repeating complete native admission. The model now keeps at most eight
complete input-admission results, keyed by the full immutable episode value, codec
instance and fingerprint, exact registry/policy bytes, geometry and expected split.
Every window still rereads the registry and builds its own model/target/loss plans.
No activations, weights or decoded model state are reused. A fresh model/resume has
no admission cache. Nonstandard episode or codec objects continue to validate on
every call. Changed source proofs, physical bytes, registry, policy, geometry and
split all refuse after a positive cached admission; another codec instance performs
fresh native admission even when its fingerprint matches.

All 53 focused inventory/census/admission/policy controls passed locally with the
pinned tokenizer and zero skips. Complete local logs and JUnit cases are retained.
The initial control test changed payload length without updating its byte span;
that test-construction failure is retained separately and corrected to change bytes
at the same length, so the cache-refusal check reaches actual admission.

The complete retained 64-source prepared corpus was then inventoried twice, once
without admission reuse and once with it. The complete inventory JSON bytes match
exactly (SHA-256 `2891c81f9021640049ee8ac9ea5408aa2bf9e8793c2ef7cac5016bb6d05a70ae`).
Both found all 48 training sources, 348 optimizer-bearing windows, 696 logical
targets and 720 physical targets/occurrences. Positive-weight denominators were
lexical 26,027, reconstruction 26,027, structure 2,880, terminal 1,440 and router_z
720; the measured P budget was 52,054 against a 200,000,000 limit. These overlapping
per-loss normalization counts are not a distinct-token total. Full compressed
inventory, both logs, source provenance and explicit zero-optimizer scope are retained.
This control uses inputs admitted by the previous exact pair and is not fresh
source qualification or evidence that the new optimizer has consumed them.

Hosted native `9c68b14` passed 60 focused controls with zero skips, plus Ruff and
formatting, in [run 45007600568](https://github.com/werg/megatoken/actions/runs/45007600568).
The complete successful job log is retained as `native9c68-hosted-pass.log.gz`.

The optimized native `a1d7f83dc46e4829ca4226e4040519cd5835f365` subsequently passed
**81 controls with zero failures or skips**, Ruff and formatting in
[run 45007944778, job 137163626029](https://github.com/werg/megatoken/actions/runs/45007944778/job/137163626029).
This includes the canonical pinned tokenizer, actual ordinary optimizer/resume
controls, all source-binding and admission-cache refusals and the inherited native
request-journal controls. The complete compressed job log, raw JUnit, runtime log
and exact source/native/tokenizer identity report are retained with the `nativea1d7-`
prefix. The original artifact is `14748548491`, archive SHA-256
`17b62d0edc16aecaee4f779b7804f50b8657e2d6b04eb4f29f3ea89b65c93cd7`.
This focused gate is separate from the actual complete 348-window optimizer pass.

A further completeness control derives all policy-masked physical targets directly
from the admitted prepared episodes, independently of the window iterator and loss
planner. Their exact source/logical/physical identity set must equal the planned
set. This closes a hole where the planner and later optimizer census could both
drop the same target and still agree. The actual retained corpus has 720 prepared
training policy targets and the same 720 planned targets, with no omissions or
inventions. Twenty-three inventory controls pass with zero skips, including
counterexamples that drop a real prepared target or plan an unprepared target.
The full controls log and JUnit are retained with the `prepared-target-inventory-`
prefix. This additional refusal check does not change valid inventory JSON or
the training objective; it does not itself establish any optimizer exposure.
