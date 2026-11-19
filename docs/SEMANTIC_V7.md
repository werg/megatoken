# Native semantic execution v7

This candidate implements the independent native consumer and training/serving
integration for `werg/tracing-js-self` PR #45. The companion changes are in
[PR #35](https://github.com/werg/megatoken/pull/35). It does not promote v7 to the
production default or claim a learned-execution improvement.

## Public data and model input

Every new public episode declares `semantic-execution/7`, `semantic-trace/7`,
`semantic-context/7`, and either `algorithm` or `detailed` representation.
The algorithm profile checks compact operations, canonical JavaScript primitive
semantics, source-bound effects, immutable program/input publications, live
state, and causal continuations. The detailed profile independently checks the
new `detail {"fields":...,"kind":...}` spelling, public envelopes, ownership,
heap changes, callbacks, rejection/cleanup, source reads, and full selected
prefixes. It does not import the old witness decoder.

Both profiles use one native `assistant` track. Public semantic track and
invocation identities stay in the checked payloads. Training and serving use
the same observed bytes, roles, task metadata, physical fragmentation, current
context, and selection controls. Native transport does not create guest
execution boundaries or publish old lifecycle/context-window announcements.
Only generated selected records receive prediction loss. Context, source,
input, dispatch receipts, and environment responses are observations with zero
loss. Actual retained native context is checked before generation; a Python
definition cache alone does not count as visible model input.

## Admission and prepared training

The paired producer workflow checks out the exact consumer revision in its
`contracts/semantic-native-policy-v7.json`, installs its hash-locked native
environment, downloads the exact Ling tokenizer, and runs both source and
native admission. `scripts/admit_semantic_v7.py` refuses a dirty or mismatched
consumer checkout. Native receipts bind public data, contexts, tokenizer,
contract/fixture digests, actual encoded training bytes and controls, and frozen
dataset membership. Source/reference/journal replay remains a separate producer
authority; the native receipt never claims to replace it.

Prepare an admitted producer packet stream in a new output directory:

```sh
python scripts/prepare_semantic_v7.py \
  --input /path/to/v7-admitted/admitted.jsonl --output /path/to/new-v7-prepared \
  --tracer-root /path/to/tracing-js-self \
  --producer-revision EXACT_PRODUCER_COMMIT --consumer-revision EXACT_CONSUMER_COMMIT \
  --tokenizer-json /path/to/tokenizer.json --max-positions 1024 --max-history 1024
python scripts/check_semantic_v7_training.py \
  --prepared /path/to/new-v7-prepared \
  --producer-revision EXACT_PRODUCER_COMMIT --consumer-revision EXACT_CONSUMER_COMMIT \
  --tokenizer-json /path/to/tokenizer.json --max-positions 1024 --max-history 1024
```

The outputs are ordinary native shards, an index, a manifest, a complete frozen
family registry, and admission/refusal reports. Declared relatives, identical
normalized cores, and exact source aliases share one split. Subset exports
retain the whole original registry. Training checks every source/component
binding against that registry, both admission receipts, and the actual stored
index split. Changing annotation and lineage fields together cannot move a row.

Training configuration must set `semantic_v7_policy` with exactly `schema:
"semantic-training-policy/7"`, the reviewed 40-character `producerRevision` and
`consumerRevision`, and the reviewed 64-character `registryDigest`. These are
explicit run inputs. Training does not infer trusted pins from incoming rows.
The model context geometry must accommodate the admitted preparation geometry.
The historical `semantic_experiment` option cannot be combined with v7 policy.
The prepared supply is checked before model/optimizer setup and each batch is
checked again at the actual training entry. Resume cannot silently change the
policy. Older prepared artifacts must be regenerated and independently admitted.

## Actual model serving

`megatoken.algorithm_model_adapter_v7.AlgorithmModelAdapterV7` wraps the actual
`MultiTrackRuntime` with independent algorithm replay. Open with a checked
program/input, select original model bytes, record dispatch, and deliver only
already observed results. Checkpoint/restore replays public history without
dispatching a service.

`megatoken.detailed_model_adapter_v7.DetailedModelAdapterV7` supplies the same
native runtime integration for detailed records. Its caller provides the live
source controller's observation validation, expected current continuation, and
record-commit callbacks. The adapter validates sequence and invocation ownership,
publishes the complete current context, and decodes the original model output
before committing it. Private training evidence is never accepted as model input.
Restore accepts public timeline/contexts and validation callbacks; it cannot
dispatch an effect. Rejected output after sampling invalidates the session, so
uncommitted sampled bytes cannot contaminate a retry.

Tests compare actual runtime event bytes, roles and controls to native training
records for compact execution and a later callback invocation. CPU model tests
exercise real generation and finite generated-token loss. They are engineering
checks, with no optimizer run or claim of useful model accuracy. Full serving
cost and external comprehension qualification remain separate release gates.
