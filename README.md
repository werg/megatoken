# Megatoken — causal faceted generation with a continuous-state transducer

This project is a software prototype, not a pretrained model. Training quality, performance and release fitness have **not** been established.

The [semantic execution v7 integration](docs/SEMANTIC_V7.md) adds independent
compact and detailed trace consumers, native admission, matching model-serving
inputs, and prepared training with frozen semantic-family splits. It is a
candidate integration; serving-cost and comprehension qualification remain open.

## Native semantic execution v8

Current work in [PR #38](https://github.com/werg/megatoken/pull/38) and
[producer PR #47](https://github.com/werg/tracing-js-self/pull/47) is **V8 data
preparation**: independent source admission, causal contexts and targets,
component splits, tokenizer/mask checks and physical packing. The
[data-preparation status](docs/language-coverage-v8-cutover-status.md) defines
the bounded milestone and links deferred work. Model quality and production
cutover are later milestones.

### Earlier bounded baseline

The bounded v8 candidate adds independent replay and physical-record packing
for mixed algorithm, region, detailed, and protocol records, together with
actual native model serving and a reproducible fitting diagnostic. This is the
delivery scope of [consumer PR #37](https://github.com/werg/megatoken/pull/37)
and [producer PR #46](https://github.com/werg/tracing-js-self/pull/46). The
[producer training and dataset runbook](https://github.com/werg/tracing-js-self/blob/64239894b36607de1265b67c337f537f3fc9a163/docs/SEMANTIC_V8_TRAINING.md)
describes source collection, frozen family splits, and native-admitted exports.

For that earlier baseline, the measured implementation was pinned to consumer
`39c4036564428e44cc89caf261f517d856482695` and producer
`012c9fe49a082319bf98ac0c571596da925741a4`. The retained
[native regression](docs/evidence/semantic-v8-final-39c4036/README.md) passed
**244 tests with zero failures, errors, or skips**, including live source-authority
checks, replay and checkpoint restoration, protocol boundaries, native serving,
and property semantics. Ruff checks and formatting passed for all 532 checked
Python files. Subsequent consumer changes through `6164754` affect documentation
and the CI producer pin; the implementation, tests, and dependencies are unchanged.

At this exact producer/consumer pair, the
[complete qualification](https://github.com/werg/tracing-js-self/blob/64239894b36607de1265b67c337f537f3fc9a163/coverage/language-expansion/native-v8-012c9fe-all/README.md)
retains **665 of 665 expansion cases**, **26 of 26 interoperability cases**, and
**11 of 11 gallery cases** passing source checks, independent native admission,
exact production-tokenizer measurement, and native physical packing. The
populations overlap and are reported separately. The independent reference
benchmark passed 624 equivalent-program executions and all 52 distinguishing
pairs. Supplementary Number, cleanup, custom-iterator, and iterable-binding
results retain their own inventories and expected refusals in that evidence index.

The [prepared dataset report](https://github.com/werg/tracing-js-self/blob/64239894b36607de1265b67c337f537f3fc9a163/coverage/language-expansion/dataset-v8-012c9fe/README.md)
retains **37 episodes, 1,021 complete examples, and 33,850 generated tokens**,
including 1,021 end-of-sequence tokens. Every exported episode received fresh
native admission after its global split was frozen. All 34 family components
remain isolated across splits. The export performed **zero optimizer steps**.

| Split | Episodes | Complete examples | Generated tokens, including EOS |
| --- | ---: | ---: | ---: |
| Train | 26 | 646 | 21,555 |
| Validation | 7 | 239 | 8,056 |
| Test | 4 | 136 | 4,239 |
| Total | 37 | 1,021 | 33,850 |

An independent second export produced byte-identical examples and token counts;
its separate receipt does not double the dataset. The earlier
[34-episode export](https://github.com/werg/tracing-js-self/blob/baeaa136bf701da5def14e3946bae1baa877331d/coverage/language-expansion/dataset-v8-b66c19b/README.md)
remains historical evidence for its original revisions.

All four hosted workflows passed at consumer
`61647547912dc3cb397ed69376bd911a9c69052b`:

- [Python 3.11/3.12 CI](https://github.com/werg/megatoken/actions/runs/44785537294).
- [Independent v8 source and native admission](https://github.com/werg/megatoken/actions/runs/44785537371).
- [Source-certified semantic corpus and production native gates](https://github.com/werg/megatoken/actions/runs/44785537396).
- [Pinned selected semantic harness bridge](https://github.com/werg/megatoken/actions/runs/44785537301).

The conditional Blackwell job did not run and is not counted as passing.
These links describe exact recorded runs; later commits retain their own hosted
checks. The bounded candidate does not promote the default representation.
External-model comprehension, broadly generalizing learned execution, controlled
cost/latency/cache comparisons at matched usefulness, and coordinated default
promotion are follow-up research outside these PRs' completion scope. Their
quality and cost requirements remain necessary before any future promotion.
The [bounded model experiment](docs/SEMANTIC_V8_BOUNDED_FIT.md) records its actual
in-sample successes, failed held-out-input attempts, and evidence limitations.

The [consumer v8 workflow](.github/workflows/semantic-v8.yml) checks each pull
request's exact consumer commit against a pinned clean producer. It runs the
native tests with the live source checker installed, rejects skipped checks,
and recollects the complete interoperability and gallery populations. Each case
must pass source and native admission, exact production-tokenizer measurement,
and native physical packing. The run retains both revision identities, its
native policy, test logs, source artifacts, admission receipts and cost reports.
The test policy changes only the consumer revision under test; all contract,
fixture, interoperability and tokenizer requirements remain pinned.

This increment adds bounded `for...in` replay for ordinary objects and arrays,
including inherited enumerable keys and nonenumerable shadowing, together with
`Object.hasOwn`. It checks stable key, descriptor, and prototype snapshots before
advancing enumeration. Promise executor and join capabilities keep their own
settlement authority; scheduler paths requiring unrepresented Promise
assimilation or cleanup jobs are refused before committing state. Observed
results, fresh data graphs, effect status, and receipts commit together, so a
refused delivery leaves the previous state intact.

An executor throw before any resolving call now consumes the shared resolving
pair as an implicit rejection. Later calls through either retained resolver
leave the promise and its original rejection evidence unchanged. Checkpoint
restoration rejects missing or contradictory first-call state. The public
golden tests also require every protocol case declared by the shared contract,
so a newly added case cannot be omitted from the consumer fixture silently.

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
