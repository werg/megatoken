# Actual support inventory identity correction — 22 November 2026

Native foundation dbc89dcd52cb93d1d9fdca2989fab64b85e5d427 passed 54 controls with zero skips and clean Ruff check/format in run 45005083938. Full log: ../v8-foundation-dbc-2026-11-22/foundation-pass.log.

Producer 0694fa9280712fa1364ead46a477c757a0ca4bfa / native dbc89dc then exercised the actual 64-source support registry in producer run 45005718379. All 64 original-source/native receipts, causal token/packing checks and strict ordinary prepared-supply qualification passed. The inventory stopped before optimizer execution because PreparedCorpus lineage.source_id is the generic semantic-execution-v8 label. The source-specific ID is stored in supervision.annotations_json.semantic_execution.sourceId and bound to the frozen registry, component, split and datasetBinding.

This correction reads the actual source-bound annotation and cross-checks it against registry membership, datasetBinding identities and lineage.family_id/frozen_split. The training exposure collector already uses semantic_execution.sourceId. The added controls model the actual generic-lineage shape and reject changed binding/split identities. Runtime qualification of this correction is pending; no 48-source optimizer coverage or learned quality is claimed from the failed run.

Full failed Actions job is retained verbatim in first-actual-corpus-failure.log, including immutable inputs and the failing stage. The detailed inventory stage log is in that run's uploaded lifecycle artifact.
