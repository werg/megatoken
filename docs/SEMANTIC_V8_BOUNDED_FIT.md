# Bounded native v8 training and execution

A small CPU experiment exercised actual Megatoken training, full checkpoint
save/reload, and two successive native model selections. After correcting a
host-envelope bug, the unchanged fitted checkpoint completed its training input
on all three declared seeds in a frozen JavaScript/process study with fresh
source authority. All three runs at the input excluded from optimization failed.
Both inputs belong to the same program and family. The zero-update model
completed none of its six runs. This establishes narrow in-sample execution
through the real model, independent native replay, and private source comparison.
It does not establish generalization, external-model comprehension, broad language
coverage, or a favorable serving-cost comparison.

The result was reproduced without retraining at the final contract/frontend
pair: producer `012c9fe` and native consumer `39c4036`. The producer's
[final-revision evidence and archive index](https://github.com/werg/tracing-js-self/tree/6bfa59ee4eaa0b3405cad6b1bd5bb3a7003a8108/coverage/language-expansion/native-v8-fit-39c4036)
contain every new declared outcome, emitted fragment, admission receipt,
study, measured cost, and the passing 244-test native gate. The
[earlier experiment packet](https://github.com/werg/tracing-js-self/tree/e91ba6c78f1d3e53947bfdf84c6549ba8e7ddd99/coverage/language-expansion/native-v8-fit-23cd478)
preserves the original training, parity evidence, and confounded evaluations.
Both packets name and hash a separate 43 MB model/checkpoint archive, but that
archive is absent from the repositories and has no published retrieval URL.
The [independent audit](https://github.com/werg/tracing-js-self/tree/ff0aa41700818efca02acb9865a22bc918345646/coverage/language-expansion/native-v8-fit-39c4036/independent-audit)
reproduces all twelve private-source outcomes and decodes all fifteen original
prediction fragments. It could not reload the historical weights or regenerate
their neural emissions. These retained-output checks do not establish independent
reproduction of the historical model run.

## Completion scope

[Consumer PR #37](https://github.com/werg/megatoken/pull/37) and
[producer PR #46](https://github.com/werg/tracing-js-self/pull/46) deliver the
bounded candidate, its checked native serving and fitting path, and the complete
declared engineering qualification. External-model comprehension, generalized
held-out execution, a statistically powered notation comparison, controlled
cost/latency/cache comparisons at matched usefulness, and default promotion are
separate follow-up studies outside these PRs' completion criteria. They require
independent response collection or new frozen model experiments and budgets;
this sandbox does not supply those study resources or the missing historical
checkpoint archive. Their unmeasured or unsuccessful outcomes remain explicit,
and the existing quality and cost gates still apply before future promotion.

The completed source/native populations and dataset are linked from the
[repository overview](../README.md#native-semantic-execution-v8). A data export
with zero optimizer steps establishes training-data admission; it adds no model
successes to this experiment. The default representation remains unchanged.

## Exact experiment

The program was:

```javascript
"use strict";function run(x){return x+1;}
```

The training invocation supplied `6` and returned `7`. A separate input supplied
`8` and should return `9`. Each reference execution consists of one pure
`algorithm` record and one explicit detailed `end` record. The held-out case was
not included in the optimizer objective or early-stopping decision.

Both source artifacts received fresh independent native admission with all six
checks passing: replay, contexts, continuation, masks, live source binding, and
production tokenizer. The source checkout was
[`c65a905cd01ea7b9e8f211b19c02414b1a5dbb0d`](https://github.com/werg/tracing-js-self/commit/c65a905cd01ea7b9e8f211b19c02414b1a5dbb0d).
The measured training checkout was
[`be453ab44847ea0b84f262909f31ba5c247851fa`](https://github.com/werg/megatoken/commit/be453ab44847ea0b84f262909f31ba5c247851fa).
These pins describe this experiment, not later implementation changes.

| Setting | Measured value |
| --- | --- |
| Model | Native `tiny`, 4,051,928 served parameters |
| Native history / physical capacity | 128 history chunks / 512 tokens |
| Precision and device | FP32, CPU, one PyTorch thread |
| Python / PyTorch | 3.12.14 / 2.10.0+cpu |
| Initialization seed | 17 |
| Optimizer | `MasterAdamW`, learning rate 0.02, weight decay 0 |
| Actual optimizer updates | 256 |
| Updated owners | 33 decoder parameter owners |
| Frozen owners | Native encoder, recurrence, prior, and memory bank |
| Supervised target budget | 74 native generated tokens per update across two records |
| Inference bound | 128 tokens per physical fragment, one fragment per logical record |
| Optimizer wall time | 285.39 seconds |
| Fit and intermediate-evaluation wall time | 321.93 seconds |

The tokenizer was the production Ling asset, with SHA-256
`40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60`
and codec fingerprint
`7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2`.
No byte-fallback tokenizer was used for this experiment.

`scripts/fit_semantic_v8.py` captures causal pre-target states by running the
production grouped trainer without gradients. It then updates only decoder
owners through the actual grouped supervision method, with genuine sampled
prior features. The scored records are the admitted generated targets;
observations and contexts are unscored. Full checkpoints retain model weights,
the actual optimizer state, and the complete native transducer attachment.
Diagnostic optimizer updates are recorded explicitly; the script does not
represent this bounded experiment as an ordinary resumable production run.

Serving receives only initial public program and invocation observations. Every
later context is reconstructed from actual committed model output. Neither the
expected target nor a repaired candidate is passed to inference. The native
adapter retains original emitted token IDs and bytes, stages a candidate on a
separate continuation, and advances only after independent acknowledgement.

## The envelope defect and its correction

The original diagnostic assembled an algorithm row with `entry: "run"`. The
public contract requires an object containing the program definition digest,
region identity and digest, exact visible cutoff, and complete shared-state
digest. The native artifact checker was validating the algorithm's values but
was not validating this algorithm entry object. As a result, it accepted a
noncanonical envelope, changing the next context's prefix digest and therefore
the model's next conditioning input.

The original training report and every failed intermediate attempt were
preserved. At update 96 the model already emitted the complete first algorithm
record correctly, but the malformed host envelope confounded evaluation of the
next record. Those original second-record failures are not clean measurements
of learned execution quality.

The correction, published in
[`23cd478`](https://github.com/werg/megatoken/commit/23cd478), adds the public
`region_entry_v8` host-metadata constructor and independent native checks of the
entire algorithm entry. A whole algorithm must also be the untouched invocation
entry; a newly hashed descriptor cannot authorize replaying it after another
generated record. Tests reject missing, string, stale, forged, extra-field, and
wrong-type entries, as well as repeated whole-algorithm execution.

The same fitted checkpoint was evaluated again after this transport correction.
There were **no additional optimizer updates**. The correction does not alter
the selected record's text or tokens; it constructs and validates the host-owned
envelope separately.

## Corrected in-process outcomes

| Evaluation | Generated records accepted | Outcome |
| --- | ---: | --- |
| Zero-update model, training input | 0 | Failed to terminate its bounded first candidate |
| Zero-update model, held-out input | 0 | Failed its bounded first candidate |
| Fitted model, original incorrect host envelope | First algorithm could be accepted | Second-record evaluations were confounded by changed prefix framing |
| Same fitted model, corrected envelope, `x = 6` | 2 of 2 | Exact algorithm result `7`, then exact explicit `end`; native completion succeeded |
| Same fitted model, corrected envelope, `x = 8` | 0 | Malformed algorithm JSON; independent validation rejected it |

The corrected in-process run therefore completed **one of two cases**, consisting
of **one in-sample success and zero held-out successes**. It must not be presented
as a 50% generalization score. The held-out malformed output and its original
61 emitted tokens remain part of the evidence.

The first trainer/serving prediction boundary was checked on the zero-update
checkpoint. After the corrected successful run, both prediction boundaries were
checked on the fitted checkpoint. Native raw state, all history key/value
tensors, event tracks, track purposes, and request controls matched exactly;
the maximum raw-state difference was zero at both boundaries. The first-boundary
case is also a persistent regression in `tests/test_semantic_model_v8.py`.

## Measured framing and serving work

These are measurements from the corrected **in-process** native replay, before
the separate frozen JavaScript/process study. They describe one CPU run and
include no performance comparison against another representation.

| Measured category | In-sample success | Held-out failure |
| --- | ---: | ---: |
| Native prediction attempts | 2 | 1 |
| Accepted records | 2 | 0 |
| Original emitted text tokens | 74 | 61 |
| Published observation tokens | 1,011 | 1,011 |
| Repeated current-context tokens | 1,570 | 772 |
| Request-instruction tokens | 150 | 75 |
| Session-metadata tokens | 42 | 42 |
| Total measured textual input tokens | 2,773 | 1,900 |
| Native inference-loop wall time | 3.428 seconds | 1.989 seconds |

For the successful run, the first prediction received 772 context tokens and
the second received another 798. The second prediction did not reingest the
published observations, but it did ingest the complete new current context.
The 74 generated tokens alone are therefore not a complete serving-cost measure.
Native physical boundaries use separate terminal flags: these 74 tokens are
61 algorithm-text tokens and 13 end-record-text tokens, with no added EOS token.

The reported input categories count actual framing text with the declared
production tokenizer. Internal learned controls, track descriptions, native
retrieval work, and tensor operations are not assigned artificial text-token
counts. Their computation is included in wall time and runtime memory evidence.
The serving bridge also retains physical-fragment boundaries, per-attempt
timings, context/cache snapshots, and rejected-attempt costs. The completed
frozen process study below adds process initialization, RPC, JavaScript
controller, and process memory evidence.

## Earlier source-authoritative process study

That study's producer was
[`9d4f6b3fcff97e7f915ae9c7228f7fd75aaaec3c`](https://github.com/werg/tracing-js-self/commit/9d4f6b3fcff97e7f915ae9c7228f7fd75aaaec3c);
the native consumer was
[`23cd47816f9b99bbdb2ec9a32dd455daf68b732b`](https://github.com/werg/megatoken/commit/23cd47816f9b99bbdb2ec9a32dd455daf68b732b).
The frozen settings were CPU, 128 tokens, one physical fragment, and effort 1.
Each model was declared on inputs `6` and `8` with seeds `17`, `29`, and `43`
before inference. Each rollout allowed four predictions, eight generated
records, one outstanding request, zero stale retries, and a 30-second timeout.

| Model / input | Completed and source-validated | Prediction attempts | Verified emitted tokens | Unknown-cost attempts |
| --- | ---: | ---: | ---: | ---: |
| Zero-update, training input `6` | 0 / 3 | 3 | 234 | 0 |
| Zero-update, held-out input `8` | 0 / 3 | 3 | 110 | 2 |
| Fitted, training input `6` | 3 / 3 | 6 | 222 | 0 |
| Fitted, held-out input `8` | 0 / 3 | 3 | 183 | 0 |

All twelve rollouts settled, and both processes shut down gracefully with exit
code zero. Every fitted training-input rollout generated both exact records;
every fitted held-out rollout generated malformed algorithm JSON and failed
on its first attempt. The combined fitted rate of 3/6 mixes in-sample and
held-out inputs; it is not a 50% generalization score. There are zero held-out
successes and only one successful distinct source/input case.

The first JavaScript/process study had used the older private source artifacts.
Its source-identity recheck rejected the authority for the executing producer
environment: three fitted public completions yielded **zero qualified
executions**. Both original reports remain archived unchanged. Fresh source
artifacts and independent native receipts were then generated at those
study revisions. Their public observations and contexts are identical to the
training inputs; only their private authority changed. All six admission gates
passed for both inputs. The new study was frozen and rerun with the same fitted
weights and **no additional optimizer updates**.

| Measured across six rollouts | Zero-update | Fitted |
| --- | ---: | ---: |
| Verified generated tokens | 344 | 405 |
| Native-reported emitted token IDs | 564 | 405 |
| Native-reported tokens on unverified attempts | 220 | 0 |
| Observation / context / instruction tokens | 6,066 / 4,632 / 450 | 6,066 / 7,026 / 675 |
| Text input tokens excluding session metadata | 11,148 | 13,767 |
| Sum of rollout elapsed time | 28.861 seconds | 32.811 seconds |
| Reported native RPC compute time | 19.020 seconds | 23.413 seconds |
| Native peak process memory | 532,385,792 bytes | 524,713,984 bytes |

The two unknown-cost baseline attempts retain 110 original emitted IDs each,
fragment bytes, and timing. An independent production-codec audit reproduces
the exact fragment bytes from all 15 current-study emissions. The two affected
fragments nevertheless contain invalid UTF-8, so the native text/provenance
gate rejects them. The verified total remains 344 plus two unknown-cost
attempts; the separately retained native count is 564. This distinction does
not imply that attempt costs were omitted from the logs.

Successful rollouts ingest observations once, then contexts of 772 and 798
tokens. Their post-generation cache snapshots contain 15 events / 7,680 bytes
after the algorithm and 20 events / 10,240 bytes after the end. The textual
input total excludes session metadata and does not turn nontext native controls
or learned tensor work into artificial token counts. Elapsed time includes
initialization on the first row and controller work; these are not six
independent cold starts. No controlled v6/v8 speed or memory comparison was
performed.

## Revalidation at the final contract and frontend

After the remaining contract/frontend changes were published, the same saved
models were run again at producer
[`012c9fe49a082319bf98ac0c571596da925741a4`](https://github.com/werg/tracing-js-self/commit/012c9fe49a082319bf98ac0c571596da925741a4)
and native consumer
[`39c4036564428e44cc89caf261f517d856482695`](https://github.com/werg/megatoken/commit/39c4036564428e44cc89caf261f517d856482695).
Their contract digest is
`344f7e99aeed605a1965c70f0e0d8829707d6891437ba43dd743fa304cd777dc`.
Fresh source artifacts and native receipts again passed all six admission
gates. The original training observations and contexts match those new
artifacts exactly. Both checkout identities and both served-model hashes were
unchanged before and after the run.

The inputs, three seeds, settings, limits, and saved checkpoints were unchanged,
with **zero additional optimizer updates**. All twelve declared rollouts
settled and both processes exited gracefully. The fitted model again completed
all three training-input runs and none of the three held-out runs. The baseline
completed none of six runs. The final result therefore preserves the same
in-sample feasibility claim and zero held-out successes.

| Measured at final revisions, across six rollouts | Zero-update | Fitted |
| --- | ---: | ---: |
| Completed and source-validated executions | 0 / 6 | 3 / 6, all in sample |
| Actual prediction attempts | 6 | 9 |
| Verified emitted tokens | 344 | 405 |
| Native-reported token IDs / unknown-cost attempts | 564 / 2 | 405 / 0 |
| Text input tokens excluding session metadata | 11,148 | 13,767 |
| Sum of rollout elapsed time | 39.517 seconds | 73.656 seconds |
| Reported native RPC compute time | 27.111 seconds | 60.500 seconds |
| Native peak process memory | 527,986,688 bytes | 528,179,200 bytes |

These are the actual new timings, not the earlier study's measurements. They
are not a controlled speed comparison. The fresh audit again matches all 15
original emitted fragments byte for byte with the production codec. The same
two baseline fragments contain invalid UTF-8, with their 110 native-reported
token IDs each retained separately from the verified cost total. No failed
attempt or earlier report was replaced.

The full workflow-selected native gate at these exact clean revisions passed
244 tests across 27 modules with zero failures, errors, or skips, including the
live source-authority checks. Full Ruff lint and formatting checks passed for
all 532 Python files, as did `git diff --check`. Runtime evidence records Python
3.12.14, PyTorch 2.10.0+cpu, tokenizers 0.23.3, Ruff 0.17.6, and Node v24.22.3.
The final evidence packet retains all logs, JUnit results, settings, and clean
before/after identities alongside the source and model reports.

## Immutable model identities and evidence

| Artifact | SHA-256 |
| --- | --- |
| Zero-update served model | `2a7eefffb21e8901150e6932ccd3d7f048fa2efb0ed4d9ebc9072feab43f0452` |
| Fitted served model | `fce915db7e521f905bea44cb559c05b3fa1cce25403621e3019b1351cf7ce2a3` |
| Zero-update full checkpoint | `5a221c7debde04a73168e0f0c0abe5fa6e51ff114e0ef73b2b647b403d81bccf` |
| Fitted full checkpoint | `3bb46a876f1bf2a4fa3b744f251385bbebdc97c7af9bca4c9c74e9074543846c` |
| Unchanged served transducer attachment | `f355a3f3e4b8fabb84c5f97b2d9b596052b8ebfd12985105f2c0c7dd54ff8ba5` |

The original experiment output contained `baseline/`, `fitted/`, both full checkpoints,
`baseline-evaluation.json`, `progress.json`, and the original `report.json`.
The historical weight/checkpoint archive containing those model bundles is not
available from the repository; its recorded identities above remain useful for
checking a future recovered copy, but hashes cannot reconstruct its contents.
The original report retains its original failed outcome; it was not overwritten
to conceal the envelope defect. `corrected-envelope-development.json` records
the unchanged-weight reevaluation. `first-boundary-parity.json` and
`complete-boundary-parity-development.json` record the real native state checks.
`artifact-index.json` binds the retained files by SHA-256 and byte size. The
corrected development reports are distinct from the later clean-checkout
frozen JavaScript studies. All six full process-study reports, including the
failed source-authority run, are retained byte for byte alongside the inputs,
native receipts, settings, and private scoring cases.

## Repeating the bounded procedure

Obtain current source artifacts and fresh native receipts from exact clean
producer and consumer checkouts. Pass the original `.source.json` bytes named
by the receipt's `sourceVerification.artifactSha256`; an `.admitted.json` file
with subsequently appended receipt metadata has a different file identity and
is correctly refused by the trainer.

```sh
PYTHONPATH=src python scripts/fit_semantic_v8.py \
  --artifact /path/to/fit-increment.source.json \
  --admission /path/to/fit-increment.native.json \
  --heldout /path/to/heldout-increment.source.json \
  --tokenizer-json /path/to/pinned-Ling-tokenizer.json \
  --output /path/to/new-fit-output \
  --seed 17 --max-steps 256 --max-tokens 128 --evaluate-every 32
```

The corrected script can stop when its actual in-sample execution succeeds at
an evaluation boundary. The historical measured run used all 256 updates
because its host-envelope defect prevented that success check. To request the
full bounded update count before the first fitted evaluation, use
`--evaluate-every 256`. A new run needs fresh admission for its own clean consumer
revision and retains its own checkpoint identities and outcomes.
