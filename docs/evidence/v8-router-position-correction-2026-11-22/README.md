# V8 router position validator evidence

This folder records focused controls for the bounded router counts added to V8
optimizer position history. It does not qualify complete hidden-router
coverage, a production launch, full support-target exposure, or learned quality.

## Source map and results

| Log | Exact source tree | Scope and result |
| --- | --- | --- |
| `position-router-resume-controls.log` | `/workspace/scratch/9faed68167fb/router-audit-native`, commit `f36b6578055c2051a15a291f310e08c2e204438e`, based on `6f5259fcba8fd68866d628424b3f7df91615972b`; includes preserved-author commits `be59c7be6d503d98ff8a196bb2193618df522dce` and `f36b6578055c2051a15a291f310e08c2e204438e` | Contradictory resealed counts, impossible count increments, bounded history, exact resume, and stage transition; 22 passed, 33 deselected. |
| `adapter-router-balance.log` | Same `f36b6578055c2051a15a291f310e08c2e204438e` tree | Actual adapter router balance accounting control; 1 passed. |
| `native-tokenizer-output-overlap.log` | `/workspace/scratch/9faed68167fb/route-readiness-native`, commit `39ecf7e22d5619d7b9bb9ed45370ded1e5661c31` | Output path overlap with the pinned tokenizer (same, nested, parent, symlink); 4 passed, 12 deselected. |
| `native-production-readiness-tests.log` | Same `39ecf7e22d5619d7b9bb9ed45370ded1e5661c31` tree | V8 and legacy semantic launch-preflight controls; 28 passed. |
| `native-position-audit-tests.log` | Same `39ecf7e22d5619d7b9bb9ed45370ded1e5661c31` tree | Earlier position, lifecycle, adapter, and model controls; 94 passed, 1 skipped because the separately provisioned pinned tokenizer was unavailable. This predates the router aggregate correction and is retained as prior evidence only. |
| `producer-position-coverage-tests.log` | `/workspace/scratch/9faed68167fb/bootstrap-producer`, commit `b4157952ed0fc28c5ae1eb8893cd9035ce25bbdb` | Producer position-coverage gate; 18 passed. |

The router correction is limited to the recorder’s count invariants: selector
maps must be internally consistent, rows cannot exceed assignments, histories
cannot introduce impossible increments, and resume cannot change selector
membership or cumulative counts without an audited optimizer update. The
separate router test confirms the existing adapter balancing path still runs.

Router internal position coverage remains explicitly **unmeasured**. These
aggregates cover only forward routes enrolled in the balancing recorder; they
do not account for all routing operations or recover hidden routing rows from
prepared text. The source route matrix and ordinary V8 launch example also
remain open for complete current-pair production readiness: the matrix is
partial, and the checked-in production YAML still has unresolved asset and
coordinator bindings. Full current-pair lifecycle and target-position evidence
must remain scoped to their own exact runs.

## Published correction and preserved original history

This composition also preserves the original external commits `1aafb80` and
`19d0112`, plus their complete red/green and unchanged actual-run validation
archive at `c489ff8`, as merge history rather than replacing their provenance.
See [the original retained correction](../v8-router-receipt-validation-2026-11-22/README.md).
The isolated test tree above independently composes those same executable
changes with native close support. Its 22-case subset and one adapter test
overlap the original correction's manifests; they must not be added as new
language populations or complete optimizer-exposure evidence.
