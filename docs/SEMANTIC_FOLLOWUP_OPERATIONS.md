# Qualify inputs and measure serving before a training run

The follow-up tools run without an optimizer. They make missing production
inputs and measured runtime costs reviewable; they do not establish learned
JavaScript execution quality.

## Preparing a source benchmark recipe

`scripts/prepare_semantic_experiments.py` keeps `training-v4` as its default and
also accepts `--recipe benchmark-v1` for paired explicit/compact source
manifests. Benchmark preparation validates the full family inventory, the
existing seed-17 family hash split for every declaration and case, normalized
AST uniqueness against both the benchmark catalogue and `training-v4`, role
coverage, and the declared train/validation lifecycle children. It then uses
the same independent source verifier, production native consumer, outcome-only
common-support preparation, lifecycle collector and stage-supply checks as the
v4 path.

```sh
python scripts/prepare_semantic_experiments.py \
  --recipe benchmark-v1 \
  --explicit-corpus /work/benchmark-explicit \
  --compact-corpus /work/benchmark-compact \
  --tracer-root /work/checkouts/tracing-js-self \
  --tokenizer-json /work/tokenizer.json \
  --producer-revision TRACER_COMMIT \
  --consumer-revision MEGATOKEN_COMMIT \
  --output /work/benchmark-prepared
```

The native prepared-data manifests and the existing
`semantic-experiment-preparation/4` receipt remain unchanged. Benchmark-only
family, split, role and prepared-case counts are written separately to
`benchmark-recipe-preparation.json`. Preparation is not training and records
zero optimizer steps.

## Actual launch inputs

First materialize the existing experiment matrix with
`scripts/plan_semantic_training.py`. Replace every placeholder with the actual
immutable corpus, production tokenizer, ordered donor imports, hindsight
targets and coordinator endpoint. An engineering-completion receipt alone
does not supply these inputs.

Run from the working directory in which the learner would be started:

```sh
python scripts/qualify_semantic_launch.py \
  --config /work/experiments/common-support-explicit-fixed-examples-17.json \
  --repositories-root /work/checkouts \
  --bootstrap-checkpoint /work/checkpoints/previous-stage \
  --bootstrap-receipt /work/diloco/run-17/semantic-bootstrap-provenance.json \
  --probe-coordinator \
  --output /work/qualification/run-17-before-launch.json
```

The checkout parent must contain `tracing-js-self`, `megatoken` and `natlang`.
All three must be committed and clean. Their current revisions are recorded
separately from the revisions that produced the unchanged corpus and original
engineering evidence. Input reuse never silently replaces its producer pins.

There are two different checkpoint inputs. `--bootstrap-checkpoint DIR` checks
the previous stage's actual model and FP32 master weights used by
`megatoken diloco init --config CONFIG --root NEW_RUN --resume DIR`. Starting a
new coordinator is the existing production stage-transition path; its learners
start from that coordinator. `--resume DIR` instead checks a complete checkpoint
for restarting a worker in the same run, including its optimizer and coordinator
state. These options are mutually exclusive. The checker does not initialize a
coordinator. A new-coordinator bootstrap also requires the receipt emitted by
that exact `diloco init` invocation. The receipt binds the source checkpoint
state and metadata hashes, initialized `bootstrap.pt` hash, parameter-layout
identity, run ID and the coordinator manifest fingerprint. Qualification
compares the receipt's source state hash and parameter layout with the supplied
checkpoint, and its manifest fingerprint with the live manifest. Missing or
mismatched provenance blocks readiness. This local receipt is an auditable
chain of file identities, not a cryptographic attestation against an operator
who can rewrite both files. Because the manifest fingerprint covers the run
configuration and layout rather than the serialized bootstrap weights, the
read-only new-coordinator probe also downloads `/v1/bootstrap` through the
existing authenticated transport, hashes it against the receipt, records its
SHA-256 and byte size, then deletes the temporary download. A same-run worker
resume checks its verified master layout against the live coordinator's
`parameter_identity`; it does not mistake the current coordinator weights for
the earlier checkpoint's values and does not fetch the bootstrap artifact.

The checker reads and verifies:

- The strict production configuration, frozen cohort and completed engineering receipt.
- The exact production tokenizer and the model's declared physical geometry.
- Every prepared semantic record, its source/native admission, visibility and
  physical limits, and required behavior supply in both train and validation.
- The selected donor identities, checksums, actual descriptor tensors and
  compatibility with the model's byte/hash layout. The loader and learner share
  the same checks.
- The hindsight index, encoding identity, corpus and window settings; every
  training window is read through the actual target loader, including its target
  tensor checksums. This performs no new teacher inference.
- An exact committed checkpoint directory when `--resume DIR` is given:
  checksums, strict state-dict keys and shapes against a meta-device instance of
  the actual model, finiteness of real checkpoint tensors in bounded chunks,
  the complete source optimizer-owner set and each FP32 master's shape, explicit model overrides, optimizer-state
  presence, cursor/RNG presence, experiment counters and configuration compatibility.
  A checkpoint root, `latest.json`, incomplete publication or inference-only
  bundle is refused. Qualification never repairs or rewrites checkpoint directories.
  A coordinator bootstrap also admits the current coordinator-export format,
  checks its actual FP32 masters and exact architecture, and verifies the
  canonical preceding-stage rule. Realizer-pack configurations are explicitly
  blocked by this shape-only model check until their dynamic shapes can be
  verified without loading those assets. Configured donor imports are restored
  through the model's real state-dict hooks; during a new-coordinator stage
  transition, the configured destination donor owners are included in the
  shape-only layout before computing the expected live master-layout identity.
- A writable output parent that does not overlap immutable data assets. The
  training output is not created.
- The runtime's CUDA/BF16 probe, and an authenticated read of the coordinator's
  manifest and status when `--probe-coordinator` is present. Objective, run,
  worker topology, architecture, corpus, tokenizer, budget and the exact
  checkpoint-derived master layout must agree. New coordinator bootstraps must
  also pass the separate provenance-receipt gate. No
  lease, worker registration, gradient or update is sent. The learner still
  verifies actual parameter ownership when it starts.

Coordinator credentials come from `MEGATOKEN_DILOCO_TOKEN` and, if needed,
`MEGATOKEN_DILOCO_GATEWAY_TOKEN`. They are not written into reports. Endpoint
query credentials are rejected. The existing transport's HTTPS and redirect
rules remain in force.

The draft `semantic-launch-qualification/1` report has a digest, concrete paths,
per-check evidence or blocking reason, `optimizerSteps: 0`, and
`learnerStarted: false`. Output paths are immutable: choose a fresh report path
for another check. Exit status is **0** for passing preflight and **2** for a
blocked preflight. Omitting the coordinator probe leaves that check explicitly
blocked. `--allow-cpu` permits qualification of CPU execution only; it does not
provide a GPU result.

`preflightReady` means the checks above passed at the time they ran. It is not
a durable authorization to spend compute, proof that all tensors can be
installed into a learner, or proof of recovery under load. `trainingMemoryFit`
and `recoveryUnderLoad` remain `unmeasured`. The real learner rechecks mutable
coordinator state and assets before use. This separation is intentional: a
read-only input check cannot manufacture an end-to-end recovery result.

## Measure the selected serving process

The actual selected model server can write a separate measured profile while
leaving its v4 JSON-line protocol unchanged:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/serve_selected_semantic.py \
  --served /work/served/checkpoint-17 --device cpu --max-tokens 512 \
  --profile-origin fixed --profile-output /work/profiles/checkpoint-17.json
```

Pass those arguments through the existing selected-process launcher in the
tracer or Natlang pilot. The profile is finalized when the launcher closes
stdin and the runtime closes. Normal replies remain on stdout. Use a new output
path for each run.

`semantic-serving-profile/1` binds the loaded model and manifest hashes and
tokenizer identity, plus the actual seed/checkpoint initialization receipt. It
reports model/runtime cold load time, the first call and empirical median and
95th percentile for each operation, and the same statistics after the first
call of that operation. A single call has no invented warm measurement. It also
records original failed output evidence, active and retained invocation counts,
track states, pending rendering/group state before shutdown, and whether the
runtime closed. Reports retain at most 100,000 per-call entries and explicitly
report truncation; a truncated report is not complete.

Active-work peaks are sampled at RPC boundaries, not continuously. Successful
RPC counts mean the server returned a protocol response; a response can still
contain a semantically invalid model record. Original bytes remain with the
caller for selected-controller validation. Completion, rejection and cancellation
release the generation buffers for that invocation while retaining its causal
history and idempotent action receipts. A pending asynchronous action retains
its original fragments until terminal cleanup, including when its native
commitment arrives after a later prefix acknowledgement.

For CUDA, measurements synchronize the selected device and include its measured
allocated/reserved peaks since before loading the model. CPU reports use `null`
for GPU memory. Host memory is this process's lifetime peak resident memory,
not GPU memory or a sum across workers. RPC time excludes transport/queue time;
the caller's end-to-end latency remains a separate measurement. Serving memory
never stands in for training memory.

Use `controlled` for any replayed selections or controlled lifecycle actions,
`fixed` for initialized weights, and `trained` only for an actual trained model.
The server rejects controlled selections under a fixed/trained profile label.
`declaredOrigin` records the caller's description; it does not attest that a
checkpoint was trained. Per-call `selectionAuthority` records the actual runtime
selection receipt when present. The model identity and initialization receipt
are checked independently of that caller declaration.
Even a trained serving profile has `learnedSuccess: null`: task success comes
from the independently checked evaluation report, not the timing tool.

## Verification

```sh
python -m pytest -q tests/test_semantic_launch.py tests/test_semantic_profile.py tests/test_assets.py
```

The tests use explicitly synthetic local assets to exercise corruption,
missing-input, restart-field and read-only coordinator checks. A real tiny
served bundle exercises the profiling path. None of those fixtures can satisfy
the production donor identity or production launch gates.
