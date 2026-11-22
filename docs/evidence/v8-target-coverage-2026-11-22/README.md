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
