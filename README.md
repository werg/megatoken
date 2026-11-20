# Megatoken — causal faceted generation with a continuous-state transducer

This project is a software prototype, not a pretrained model. Training quality, performance and release fitness have **not** been established.

The [semantic execution v7 integration](docs/SEMANTIC_V7.md) adds independent
compact and detailed trace consumers, native admission, matching model-serving
inputs, and prepared training with frozen semantic-family splits. It is a
candidate integration; serving-cost and comprehension qualification remain open.

## Native semantic execution v8

The v8 candidate adds independent replay and physical-record packing for mixed
algorithm, region, detailed, and protocol records. The
[producer training and dataset runbook](https://github.com/werg/tracing-js-self/blob/baeaa136bf701da5def14e3946bae1baa877331d/docs/SEMANTIC_V8_TRAINING.md)
describes source collection, frozen family splits, and native-admitted exports.

The qualified implementation is pinned to consumer
`6adf99f1301557ee5b95269581a3acd4162496b4` and producer
`b66c19be960f114b69787746b2487e6f23a73577`. Its final focused consumer suite
passed **214 tests with no failures or skips**, including live source-authority
checks, replay and checkpoint restoration, protocol boundaries, and property
semantics. Ruff checks and formatting passed for all 524 checked files.

At these same revisions, **23 of 23 interoperability cases** and **11 of 11 gallery
cases** passed source, independent native, and tokenizer qualification. The
[prepared dataset report](https://github.com/werg/tracing-js-self/blob/baeaa136bf701da5def14e3946bae1baa877331d/coverage/language-expansion/dataset-v8-b66c19b/README.md)
retains those qualification reports and the independently verified export:
**34 episodes, 898 complete examples, and 29,374 generated tokens**, including
898 end-of-sequence tokens. Every exported episode received fresh native
admission after its split was frozen. The export performed **zero optimizer
steps**.

| Split | Episodes | Complete examples |
| --- | ---: | ---: |
| Train | 23 | 523 |
| Validation | 7 | 239 |
| Test | 4 | 136 |
| Total | 34 | 898 |

The [producer regression report](https://github.com/werg/tracing-js-self/blob/baeaa136bf701da5def14e3946bae1baa877331d/coverage/language-expansion/producer-regression-b66c19b/README.md)
retains full producer tests, contract checks, and reference-fixture results.
Hosted checks on the measured consumer commit also passed:

- [CI](https://github.com/werg/megatoken/actions/runs/44740013999).
- [Source-certified semantic corpus and production native gates](https://github.com/werg/megatoken/actions/runs/44740013895).
- [Pinned selected semantic harness bridge](https://github.com/werg/megatoken/actions/runs/44740013827).

These hosted runs cover the general repository and existing v6/native bridge
pipelines; the v8 source, native, and token qualification is recorded separately
above. Later documentation commits do not change the measured implementation
revisions. Model-execution qualification, end-to-end serving costs, and default
promotion remain separate gates.

This increment adds bounded `for...in` replay for ordinary objects and arrays,
including inherited enumerable keys and nonenumerable shadowing, together with
`Object.hasOwn`. It checks stable key, descriptor, and prototype snapshots before
advancing enumeration. Promise executor and join capabilities keep their own
settlement authority; scheduler paths requiring unrepresented Promise
assimilation or cleanup jobs are refused before committing state. Observed
results, fresh data graphs, effect status, and receipts commit together, so a
refused delivery leaves the previous state intact.

The native host-request profile still rejects calls requiring the
`owned-graph/1` argument format. Tests of fresh result graphs cover the result
transition itself; end-to-end admission of those graph-based host requests
remains unsupported.

Run admission from the exact clean consumer checkout selected by the producer's
`contracts/semantic-native-policy-v8.json`. Supply the matching clean producer
checkout and revision so the live source authority can re-execute and verify the
source-bound artifact. The production tokenizer is the pinned Ling tokenizer
with SHA-256 `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60`;
the qualified environment uses `tokenizers==0.23.3`. Keep input and receipt files
outside the producer checkout so it stays clean.

```sh
python -m pip install -e '.[data]' 'tokenizers==0.23.3'
PYTHONPATH=src python scripts/admit_semantic_v8.py \
  --input /path/to/source-bound-v8.json --output /path/to/native-receipt.json \
  --consumer-revision EXACT_CONSUMER_COMMIT \
  --producer-root /path/to/clean/tracing-js-self \
  --producer-revision EXACT_PRODUCER_COMMIT \
  --tokenizer-json /path/to/pinned-Ling-tokenizer.json
```

Check the receipt's `admitted` field and all six checks: replay, contexts,
continuation, masks, source binding, and tokenizer. Exit status zero means a
receipt was written; it does not by itself mean admission. The Python APIs
`megatoken.data.native_v8.validate_v8_artifact` and
`prepare_native_v8_training` provide replay checks and physical packing,
respectively; neither grants admission or replaces live source verification.
Missing source authority or tokenizer checks remain untested.

## One architecture

One selected raw-facet representation powers ordinary E/C/P/R generation, and one learned memory sits beside it: the [Continuous State Transducer](docs/continuous-state-transducer-plan.md) (v10, implemented, not yet trained). At chosen boundaries the model emits a query, searches a sparse bank of location-to-location edges with an agenda-wide neighbourhood search, and admits what it finds as extra attention slots ("knowledge megatokens"). Edges carry FP8 routing vectors and 32,768-wide FP4 values; a momentum-free local-scale optimizer trains them, hindsight writes fill them, and error-driven growth adds and splits them. The transducer is always on: there is no switch and no older retrieval stack (the v9 source catalog, search index and embedding-alignment sidecars were removed). The implementation and executed checks define current behavior; design documents describe intended work.

The bank travels with every checkpoint (`transducer.pt` beside the weights). Under DiLoCo the coordinator owns the canonical bank and merges each learner's 32-step round of sparse changes; synchronous multi-GPU ranks share it the same way, merged on rank 0.

Tracks have stable session addresses and explicit purposes. Their addresses affect
attachment and attention equality, rather than learned expert preferences. One track
can carry successive modalities or several separately addressed output peers. See
[the current implementation status](docs/multi-track-implementation.md) for the
lifecycle tools, bounded scheduling, asynchronous rendering, and remaining training
and media work.

Architecture v13 changes parameter ownership and artifact meaning. Rebuild prepared
episodes and model-bound knowledge bases and train fresh weights; old slot-based
checkpoints and old grouped tapes cannot resume exactly. Original media and compatible
checksummed feature assets remain reusable. The launch commands below still require
real prepared supply and the matching decoder identities.

## Prepare, then train

Python 3.11/3.12 and the tested native PyTorch runtime are expected.

```bash
python -m pip install -e '.[train,data,dev]'
megatoken doctor
```

The package defines one curriculum with one artifact/compute binding file per stage in
`configs/train`. Training-method decisions come from `data/stages.json`, including the
optimizer, effort mixture, representation, trainable weights, decision horizon and learning-rate
schedule. Production uses independent DiLoCo learners throughout. Resolve the `PIN_*` artifact
and coordinator bindings before launch; missing supply is an error.

```bash
python -m megatoken.cloud_diloco launch configs/cloud/train-default.example.json
```

The checked-in launch templates require teacher hindsight and structural memory guidance.
Prepare targets once for the exact corpus and window settings, preserving every data root and
stage mixture, then check the emitted configuration before launch:

```bash
python -m megatoken prepare-hindsight \
  --train-config /path/to/resolved-train.yaml --writer /path/to/retrieved-writer \
  --output /path/to/hindsight --device cuda
python -m megatoken train-check --config /path/to/hindsight/train.json
```

The default cloud handoff starts at the initial representation stage:
[configs/cloud/train-default.example.json](configs/cloud/train-default.example.json).
Later stage jobs select the next stage binding and restore the preceding export.
It delivers the prepared teacher artifact to learners and its identity to the coordinator.
[Default memory setup and remaining inputs](docs/training.md#default-memory-recipe-and-launch-readiness)
explain the single-node and cloud paths. A checked template with unfilled artifact paths reports
**not ready**; enabling an objective never substitutes for preparing its data.

There is **no** implicit launcher default or `megatoken smoke`.

**DiLoCo is the target training methodology.** `megatoken diloco init|serve|export` run decoupled learners ([configs/train/pretraining.example.yaml](configs/train/pretraining.example.yaml), [docs/diloco-coreweave.md](docs/diloco-coreweave.md)). Every phase runs decoupled: neural parameters sync as fragments and the coordinator merges the transducer bank's rounds. Each phase is a new coordinator started from the previous phase's export. For hardware-specific setup, see [Blackwell](docs/blackwell.md).

## Serve

```bash
megatoken export --checkpoint runs/b0/checkpoints/latest.json --output served/b0 --codec byte
megatoken generate --model served/b0 --prompt "Hello" --max-tokens 32   # one-shot session
megatoken serve --model served/b0                                       # JSON-lines service
```

There is one serving program, the multitrack runtime ([generation](docs/generation.md#export-and-serving),
[protocol](docs/multi-track-implementation.md#group-native-jsonl-protocol)). `generate` is a
one-shot session of the protocol `serve` speaks, and the frozen student that collects rollouts for
training runs it too, so lookups, the adapter mixture and mounted collections and knowledge bases behave
the same when serving, collecting and training. Its tape is the group tape (`fmt-multitrack-facets-v16`).

New models support [longer local and shorter cross-track context](docs/track-context.md),
with physical cache reclamation, explicit evidence rereads, and recorded training
policies. Initial window defaults remain experimental.

For dynamic workers, grouped media collection, timed translation, executed voice/tool
traces, and their preparation and evaluation gates, use the
[training runbook](docs/dynamic-training-runbook.md). The runbook distinguishes
executable preparation paths from data acquisition and learned-quality results.

## Validation contract

Before training, run the bounded correctness checks and the actual-input readiness check.
Do not gate the first bounded tranche on comparison studies, whole-corpus teacher preparation,
or learned-quality results that require training. Monitor quality and throughput during the run.

## Media: one input path, one realization path

Media has exactly one path in and one path out ([docs/media.md](docs/media.md)). Every image, audio or video event enters as the **512-wide EmbeddingGemma2 hidden states** (not its pooled 768-dimensional embedding), extracted offline by the pinned frozen teacher and projected by a trainable `MediaEncoder` into the same E reductions as text; text enters as bytes, except long reference documents and large tool results, which can be read compressed through the same teacher ([docs/compressed-reading.md](docs/compressed-reading.md)); its pooled 768-dimensional embedding is the target of the encoder's embedding head. Media leaves through frozen pretrained realizers (Pocket for 24 kHz speech, Sana for images, Sana-Video for video) with trainable `FacetCondition`s, trained on the donors' codec latents. All pretrained weights come from one locked pack manifest written by [scripts/acquire_packs.py](scripts/acquire_packs.py) from the pinned [inventory](configs/packs/inventory.json); install with `pip install -e '.[train,media]'`. Media is part of the stage plan (`src/megatoken/data/stages.json`): a short [media-bridge stage](configs/train/media-bridge.example.yaml) after the codec bootstrap, explicit media and long-document shares in base pretraining, and replay in later stages, each stage a new DiLoCo coordinator started from the previous export. No media-quality or GPU-training results are implied by these contracts.
