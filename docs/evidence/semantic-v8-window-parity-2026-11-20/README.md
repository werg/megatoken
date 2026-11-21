# Training-window continuation parity

An actual recurrent-state comparison found a second training/serving mismatch
after the physical chunk-budget fix. A retained-callback example agreed for its
first 13 predictions and diverged on prediction 14, the first target in its
second optimizer window. The earlier raw bytes still matched, but the completed
contextual tensor differed. The original failure is retained in
`before-pytest.log.gz.b64`; it is not relabeled as a passing run.

When an ordinary training window rebuilt its earlier prefix, it treated previous
generated records as observations and omitted their original request controls.
Those controls had conditioned both live generation and the original supervised
update. The fix resolves the same earlier public controls when reconstructing
those generated records. Saved native action tapes continue to use their own
bound controls. No future target or private source evidence enters the model.

## What is exercised

The tests use the actual CPU model, grouped trainer, runtime, encoding, retrieval,
context eviction and semantic acknowledgement. They compare every field of the
model state, including raw and final tensors, key/value caches, routing mixtures,
event positions, track descriptions, visible-history policy and ownership. They
also compare original emitted token IDs/bytes, prediction masks, decoder bounds,
terminal flags and roles, and request controls at every physical prediction.

| Case | Compared physical predictions | Boundaries exercised |
| --- | ---: | --- |
| A record split into three fragments, followed by the next record | 4 distinct targets | A deliberately invalid UTF-8 attempt is refused; the accepted record is reconstructed from a checkpoint through the real model; a later training window reconstructs its prefix. |
| Async host counter | 11 | Dispatch/result observations, two host results, recurrent representation, effort 2 and eviction at the 128-event history limit. |
| Retained callback session | 20 | Two callback invocations, child completion, host observations, parent resumption, eviction and a second optimizer window. |

Prepared target event IDs hash bytes already known to the data pipeline. Live
event IDs name attempts before their output exists. The test therefore establishes
a one-to-one address mapping from identical original emitted fragments. It uses
that mapping only while comparing ownership; it never renames stored state or
changes any model input. Observation/context addresses and all controls compare
exactly without that mapping.

The lexical decoder is explicitly controlled to emit the prepared target
fragments. The source/tokenizer audit boundaries use the declared unit fixture;
the resulting data still pass the ordinary strict prepared-data and batch
validators. These are engineering controls, not independently source-admitted
real-tokenizer runs or evidence of learned execution. The separate
[ordinary pinned-asset lifecycle receipt](../semantic-v8-ordinary-a4d0a56-7da2150/README.md)
records actual trained-token output and its failed execution outcome.

## Verification

`callback-after-pytest.log` records the repaired callback case passing.
`existing-controls-pytest.log` records three existing controls passing, including
native replay's preservation of its own controls. `final-pytest.log.gz.b64` retains a
six-test run with five passes and one test-setup failure: the small example has
exactly 64 records, so its initial 64-record limit correctly failed the new guard
requiring multiple windows. Reducing that test-only limit to 16 produces two
scored windows; `split-final-pytest.log` records that corrected case passing.
Thus all six distinct controls pass in the final relevant runs; this is not
presented as one all-green combined invocation. `summary.json` binds the tested
implementation and final regression source bytes.

Failure logs are gzip-compressed and base64-encoded to retain their original
bytes, including pytest's trailing whitespace. Their decoded SHA-256 values are
in `summary.json`. Decode a log with
`base64 -d before-pytest.log.gz.b64 | gzip -d`.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:tests:. python -m pytest \
  tests/test_semantic_v8_continuation_parity.py \
  tests/test_causal_student_training.py::test_causal_training_prediction_matches_actual_serving \
  tests/test_training_history.py::test_evaluation_replays_native_prefix_and_excludes_native_target_facets \
  tests/test_training_history.py::test_fixed_native_replay_retains_central_gradients_and_ignores_prospective_controls \
  -q
```

This receipt does not replace final combined-pair qualification, training with a
diverse corpus, trained-model accepted-record recovery or held-out usefulness
studies. It closes the reproduced loss of prior request controls at a training
window boundary and adds explicit coverage of the causal histories above.
