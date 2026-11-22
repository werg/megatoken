# Causal bootstrap predecessor controls

This test-only increment is based on the published causal native candidate `56cce3b80423b0cb70c3a096d867561066e3a211`, whose exact tree is `96ac16250743ac46d3f4b97d231bfcea8cc2a223`. It preserves the causal `durable-bootstrap-child` / `recover-bootstrap-child` API, the four raw operation alias controls, router and close evidence, and does not change production code.

The ten added test leaves cover predecessor identity substitution for session, stream, generation, ordinal, and model identity (five cases); changed exact create request bytes or create precheckpoint digest (two); a nonadjacent intervening native journal request; a missing create outcome; and a mutated live precheckpoint that differs from the acknowledged create poststate. Each verifies refusal before child bootstrap is dispatched; unchanged-state cases also compare the full adapter checkpoint.

## Focused run

```sh
PYTHONPATH=src:. /workspace/scratch/05d92519587f/native-recovery-venv/bin/python -m pytest -q \
  tests/test_semantic_bootstrap_recovery_v8.py \
  -k 'same_adjacent_create_identity or exact_original_create_bytes or latest_completed_native_request or completed_create_outcome or precheckpoint_must_equal' \
  --junitxml=docs/evidence/native-bootstrap-predecessor-controls-20261122/junit.xml
```

Result: 10 passed, 17 deselected, 0 failures, 0 errors, and 0 skips in 29.73 seconds. Runtime was Python 3.12.14, pytest 9.2.0, and torch 2.14.1+cpu. The repository declares pytest below 9, so this focused local run is not evidence for the locked hosted environment. Raw output and JUnit are retained in this directory.

## Coordinator continuation digest review

On a real create→bootstrap runtime, the coordinator `continuationDigest` changed from `c3d2097c9daa60f4750d916c2bcb5ec1290e47f5ea91ad4f59036e8867c2e474` to `be69d490007cb39bf4d3d601888799847ad1781bb39333d99842b45910d7601b`. A rehashed outcome with the post digest replaced by the pre digest was rejected during fresh-process recovery when independent full-state bootstrap reexecution differed; the bridge fenced and returned no receipt. The exact probe output is in `continuation-digest-probe.log`.

The structural bootstrap checkpoint validator does not independently require the continuation digest to change. The observed full recovery boundary catches a stale digest through native reexecution and complete checkpoint comparison, so this increment adds no production check and makes no standalone-snapshot validation claim.
