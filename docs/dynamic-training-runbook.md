# Dynamic tracks and corresponding media: preparation and training

This is the operational companion to the [design and source review](dynamic-tracks-and-concordant-training-plan.md).
The implementation supports dynamic track lifecycles, per-event modality choices,
joint outputs, concurrent tools, source publication, and recorded outcome supervision.
Those capabilities still need representative data and trained weights. Passing the
checks below establishes software behavior, not voice quality or task success.

## 1. Establish the runnable baseline

Install the dependencies described in the README. Use one unchanged source tree
through preparation and training: preparation identities include the reader code,
and changes require preparing the corpus again.

```bash
python -m pytest -q tests
python scripts/accept_dynamic_tracks.py --root /absolute/new/acceptance-run --device cpu
```

The acceptance command creates its own authored, executed text tasks. It prepares
and samples them through the ordinary optimizer, restarts from a checkpoint,
exports a served model, and generates through the public service. Inspect
`acceptance.json`, preparation coverage, training logs, and `generated.json` in the
chosen directory. The command refuses to overwrite an existing run. Its successful
completion does not establish learned delegation, natural conversation, media
synchronization, distributed accelerator performance, or production quality.

## 2. Prepare timed source material

Keep original recordings and annotations together. Record the original conversation
family and freeze its split before creating variants. Represent source intervals in
exact sample/frame coordinates; distinguish content time from the time the source
became available. Preserve independent speaker channels and their overlap. A target
translation has its own speech clock. Joint audiovisual targets require an explicit
shared alignment annotation.

Start with [the local timed-source example](../configs/data/timed-media.example.jsonl).
Replace its illustrative paths and checksum with real local sources; remove tasks
for which the source has no annotation. Text, speech translation, conversation,
audiovisual understanding, and jointly generated peers are compiled from the same
corresponding source artifacts. Audio/video features are extracted from the selected
source intervals rather than from a whole-recording representation cropped later.

```bash
python scripts/acquire_packs.py --root /absolute/packs
python scripts/verify_packs.py --manifest /absolute/packs/lock.json --rollout --prepare
python scripts/prepare_timed_media.py \
  --manifest /absolute/timed-sources.jsonl \
  --packs /absolute/packs/lock.json \
  --codec /absolute/codec.json \
  --terms-json /absolute/reviewed-source-terms.json \
  --root /absolute/new/timed-corpus --device cuda
```

The terms document uses the existing `TermsReview` fields: an explicit approval,
license and license URL, reviewer, review date, notes, redistribution group, and its
basis. A catalog source can instead be selected with `--source-id`. These commands
do not approve a source or manufacture missing annotations. Use `--pilot` to limit
preparation to the first two recordings before scaling. Archive source checksums,
pack identities, preparation receipts, prepared tensors, and the prepared corpus
together. The committed source receipt prevents silently replacing a preparation.

The meeting-prefix preparation command is another local reader for original AMI
headset channels and publisher word timings. It makes causal speaker continuations;
it does not provide executed assistant tools or certify word timing independently.
Natural meeting overlap must not be relabeled as executed full-duplex tool use.
Its `--codec` argument must match training (`byte` is an explicit pilot choice),
and `--terms-json` records the reviewed source terms before preparation begins.

## 3. Collect actual execution

The frozen student used by `megatoken data collect` now accepts grouped output
requests, preserves same-track peers, and saves whole selections even when some
renderings are empty or fail. Supply `--student-bundle` for the model under test,
and `--student-packs` plus `--student-media-root` for generated media. The pack
identities must match the served model. Run `megatoken data collect --help` for the
task, teacher, environment, budget, and recovery arguments. Task rows may provide
`outputs` or serialized `outputs_json` requests. Each collected selection gets
session-specific event addresses so repeated request labels cannot collide.

Use real compatible tool adapters and record call/result addresses, immutable
snapshot identities, start/completion receipts, failures, cancellations, delivered
output, and source readiness. Selected content and completed rendering are different
facts. Input arriving during rendering must remain in the subsequent history.
Cancelled delivery does not undo an executed tool or previously delivered speech.

Import independently recorded native episodes with the checked importer:

```bash
python scripts/prepare_execution_traces.py \
  --episodes /absolute/executed-episodes.jsonl \
  --source-assets /absolute/recorded-assets \
  --terms /absolute/reviewed-source-terms.json \
  --codec /absolute/codec.json \
  --output /absolute/new/execution-corpus
```

The default destination is the full-duplex execution stream. Admission requires
actual linked tool execution, overlapping speaker audio, and versioned source
arrival/readiness; moving an unrelated source into this stream also requires verified
success. Referenced tensor assets are checksum-verified and copied. Native policy
tapes, when present, must agree with the recorded events, codec, and media receipts.
Episodes without a complete recorded selection context can provide supervised
targets but cannot supply verified policy-outcome training.

## 4. Admit data to the ordinary training recipe

Use fresh prepared roots in the existing stage launch configuration. Keep matching
codec, decoder pack, and preparation identities. Do not resume old slot-addressed
weights as the new architecture. The model chooses output modality from the legal
installed capabilities; the semantic track purpose supplies meaning independently
of its opaque address.

The checked recipes include authored execution, collected full-duplex execution,
timed conversation, and streaming translation. These have real local preparation
producers, rather than fictitious downloaded dataset entries. Their presence in a
recipe does not mean data has been supplied. Preparation writes observed behavior
coverage by split and original family. Training checks required streams and behavior
families; held-out checks require their own validation supply. Family identities are
deduplicated across prepared roots, so copying a corpus does not increase coverage.

Check the receipt for the exact behaviors requested by the selected stage. Build
missing examples; do not disable the production checks to substitute a text-only
pilot. Preserve causality and serial dependencies where independence is unknown.
The design/source review contains the remaining acquisition inventory and dataset
shortlist; it does not claim those sources were downloaded or trained.

Grouped training includes every selected peer in the joint policy and future
history, including peers without a valid decoded payload. Only actual payloads
receive payload reconstruction loss. Modality accuracy counts all selected peers.
Structural and verified outcome-only decisions count toward the whole-decision
progress measure, including a failed last selection with no following record.
The existing shared loss ledger handles reduction, accumulation, restart, and
distributed denominators. Verified outcome supervision is integrated with this
training path; a separate fully frozen-decoder latent-only RL stage is not delivered.

## 5. Evaluate the trained result

Use held-out original families and fixed arrival schedules. Evaluate the exported
runtime with its own generated decisions, and separately report diagnostics that
request particular output tracks. Prepared media observations can now enter generated
evaluation. Supply public output bounds with `megatoken evaluate --output-specs`
(a JSON mapping from modality to the corresponding public serving request) when
the corpus does not contain a public request. Hidden target dimensions are never
converted into a public generation budget. Install matching packs with `--packs`.

Report generated tool/task success, source availability and attribution, interruption
handling, selected modality accuracy, rendering failures, latency, memory, and useful
parallel work against a serial baseline. State the data, hardware, sample count, and
denominator for each result. A finite rendered tensor establishes decoding only.
Use independent speech recognition, translation, perceptual, and time-local
audiovisual checks plus reviewed examples for quality. The generated evaluator
explicitly leaves media quality unestablished; the synthetic decoder tests cannot
replace those measurements.

The full-duplex integration regression exercises listening during blocked speech
rendering, actual snapshot tool reads, publication of a searchable source, interruption,
late completion, and a later reference to admitted evidence. It verifies protocol
behavior with a test decoder. Retain that gate while adding real recorded sessions
and real-weight held-out evaluations.
