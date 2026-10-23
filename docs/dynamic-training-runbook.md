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

The frozen student used by `megatoken collect` now accepts grouped output
requests, preserves same-track peers, and saves whole selections even when some
renderings are empty or fail. Supply `--student-bundle` for the model under test,
and `--student-packs` plus `--student-media-root` for generated media. The pack
identities must match the served model. Run `megatoken collect --help` for the
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
training path. The student stage additionally supports `trainable: latent_policy`:
it updates only prediction-local parameters, modality decisions, and the Gaussian
while freezing encoding, memory, shared expert banks and all realization weights.
This keeps a recorded latent action's meaning fixed during outcome improvement.
Joint supervised correction can still use `trainable: all`; its historical latent
outcome term is an auxiliary credit signal, not a policy-improvement guarantee
while the realization mapping changes. These are explicit modes of the existing
optimizer, rather than a second training engine.

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

## 6. Reproduce bounded real-weight acceptance and inventory evidence

Use the same frozen source copy for the actor's preparation, training, collection,
restart and exported evaluation. A concurrent checkout edit can change parameter
ownership or preparation identities while a long-running Python process retains
its earlier imports. Preserve a complete source archive and its checksums with
the run; copying only its task generator does not identify the training software.

```bash
python scripts/accept_prepared_media.py \
  --data /absolute/prepared-recordings --packs /absolute/packs/lock.json \
  --codec /absolute/codec --root /absolute/new/media-acceptance --device cuda

python scripts/accept_student_outcomes.py \
  --bundle /absolute/actor/served --packs /absolute/packs/lock.json \
  --root /absolute/new/student-outcomes --device cuda

python scripts/accept_duplex_runtime.py \
  --bundle /absolute/media-acceptance/served --packs /absolute/packs/lock.json \
  --recording-data /absolute/prepared-recordings \
  --root /absolute/new/combined-protocol --device cuda
```

The outcome command expects the actor's training recipe and checkpoint beside its
export. It collects without teacher intervention, verifies each resulting task
state, then runs the ordinary optimizer, restart and export. Failed attempts are
retained. Arbitrary byte output is preserved exactly in physical records; the
teacher's textual transport labels any non-UTF-8 content explicitly as base64.
Teacher-repaired success never supplies positive outcome credit to earlier
student actions. This command uses the restricted latent-policy mode and checks
restart under that same freezing contract.

The combined command requires independently recorded, overlapping speaker channels.
It uses the real speech decoder and actually selected read calls, immutable read
snapshots, out-of-order completions, atomic document publication, listening during
pending rendering, cancellation and partial delivery acknowledgements. It prepares
a native training episode with exact action/result bindings, the cancelled
selection, media assets and verified read-outcome supervision. Both the arrival
schedule and the public read requests are host-controlled; this is a protocol
experiment, not evidence that the model learned delegation, free-form tool
arguments, intelligible speech, or an appropriate conversational answer. Playback
acknowledgements describe host-reported sample delivery, not a physical loudspeaker
measurement. Small-history models re-read admitted evidence before referring to it
after its earlier attention entries have been evicted.

Freeze the complete catalog and measured supply report:

```bash
python scripts/audit_training_inventory.py \
  --prepared /absolute/prepared-recordings \
  --checkpoint-metadata /absolute/run/checkpoints/STEP/metadata.json \
  --access-receipts /absolute/source-access-receipts.json \
  --output /absolute/run/source-inventory.json
```

The default catalog includes every checked-in source manifest. Unknown access stays
unknown; a public card does not prove recording access. A downloaded receipt needs
locally verified original-file checksums. Prepared supply requires checking the
shards and all referenced tensors. Preparation counts remain separate by root and
split, so copies cannot be added together as new original families. Checkpoint
reports count actually returned, non-padding training windows per source across
ranks; virtual mixture charges and context-only padding do not count. These counts
survive restart. Older checkpoints without this evidence are explicitly unmeasured,
and counts from successive checkpoints must not be summed as separate exposure.

Required training streams, held-out behavior coverage, and declared decoder packs
are checked before loading model and donor weights or allocating optimizer state.
The real decoder objects are then checked again before optimizer ownership is
enumerated. The production recipe remains incomplete until representative supply
and generated held-out quality satisfy the earlier gates.

### Local evidence and remaining release gates

The 23 October local experiments retained these results under
`/home/werg/megatoken-runs/`:

| Experiment | What was actually checked |
| --- | --- |
| `dynamic-acceptance-local-11` | The larger text model trained for four optimizer updates, resumed for one more in a fresh process, exported, and generated through the public service. Its declared experiment geometry bounds history and n-gram storage; this is not a production training run. |
| `dynamic-ami-acceptance-02` | Eighteen updates including restart on one original meeting family, followed by export and generation of three speech chunks using the real speech decoder. Evaluation used training material. |
| `dynamic-translation-acceptance-01` | Preparation of four real recordings with human translations, sampled training, restart and export across speech reconstruction, transcription and speech-to-text translation. |
| `dynamic-duplex-runtime-04`, `dynamic-duplex-training-01`, `dynamic-duplex-runtime-06` | Controlled real-decoder conversation/tool/source/interruption execution; native preparation and three further optimizer updates including restart; export; then successful repetition of the controlled protocol with the updated served weights. |
| `dynamic-student-outcomes-08` | Four student-only tasks, all independently verified failures, retaining eight selected decisions. Restricted latent-policy optimization, restart and export completed; this establishes failure-credit integration, not task success. |

The broad CPU regression selection, excluding concurrently developed experimental
probes, passed 2,489 cases and skipped 25 optional or
accelerator cases. Two distributed tests exposed timing assumptions on the busy
shared machine: a five-second spawned-server startup assumption and a live-lease
assertion after deliberately stopping heartbeats. The readiness bound now covers
native-runtime startup and reports child failure; lease recovery checks the actual
renewal and accepted upload. Both affected tests passed their subsequent run.
A separate accelerator regression selection passed 66 cases, including actual
CUDA snapshot and kernel checks. Subsequent focused checks cover the restricted
policy update, unchanged realized bytes, private checkpoint mappings, binary
student output, and process-isolated restart. These overlapping runs must not be
summed as independent test coverage. Source archives and experiment receipts retain
which source copy each run used.

This is not complete production acceptance. Still required are representative
training and validation families for every mandatory behavior, a sustained learned
wait/emit translation sequence, quality-matched learned delegation comparisons,
independent decoded speech/translation/audiovisual checks with shifted and
mismatched negative examples, human-reviewed outputs, and the resulting calibrated
production mixture and training run. The controlled protocol must not stand in for
learned scheduling or natural assistant conversation. No coherent speech,
translation quality, or audiovisual synchronization result is claimed here.


### Further held-out runs and failure analysis (23 October)

The later experiments below use frozen source copies and retain failed attempts.
They extend the earlier evidence; they do not establish production acceptance.
All paths are under `/home/werg/megatoken-runs/`.

| Experiment | What the evidence establishes | What it does not establish |
| --- | --- | --- |
| `dynamic-streaming-prepared-01`, `dynamic-streaming-acceptance-01` | Forty prepared task views from twenty original French speakers with human English translations; sixteen speakers for training and four for validation. Source arrivals, waiting and continuation remain in one causal history. GPU optimization, fresh-process restart, export and held-out generation completed. | Generated lifecycle calls and translation text were not reliable. Recorded source-arrival replay is not autonomous scheduling. |
| `dynamic-paired-translation-prepared-03`, `dynamic-paired-translation-acceptance-02` | Forty task views from ten French/English sentence families, eight training and two validation. Joint speech/text and actual committed-text-then-speech variants survive duplicate filtering. Both source and translated speech retain their own clock. | Speaker identities were unavailable; no speaker-disjoint claim. Two validation families do not establish broad translation competence. |
| `dynamic-ami-disjoint-prepared-01`, `dynamic-ami-disjoint-acceptance-01`, `dynamic-duplex-runtime-07` | Original overlapping headset channels and publisher word times from separate training and validation meetings with different speakers. Six task views, real optimization/restart/export and a controlled speech/tool/source/interruption protocol completed. | Natural human overlap does not supply natural assistant decision timing. Host-controlled requests do not prove learned full-duplex behavior. |
| `dynamic-delegation-comparison-03` | Actual execution of serial and delegated read/write programs on thirty-two original task families under slow-independent, cheap-independent and dependent conditions. Ninety-six matched comparisons verify identical task results and measure actual execution time. | The programs are scripted. Their speed differences are not a learned model benefit. |
| `dynamic-autonomous-trained-01` | Student-only generated execution on held-out tasks, with actual tools and no substituted teacher responses or prerecorded results. | None of twelve attempts passed the independent task verifier. Autonomous delegation remains failed. |
| `dynamic-visual-bridge-acceptance-02`, `dynamic-visual-joint-acceptance-01` | Ten task views from two meeting families. Restricted media-connection training, joint training, restart, export and actual image/video decoding completed. Generated first/last images were compared with the generated video's endpoints. | Endpoint pixel differences are diagnostics, not established semantic consistency; the generated caption was invalid UTF-8. No audio/video synchronization claim. |
| `dynamic-paired-independent-quality-01`, `dynamic-translation-path-independent-quality-02` | Independent recognition of the exact sampled speech, human-reference comparisons, deliberately unrelated caption controls, and archived sampled bytes. The recognizer separated all four unrelated captions from their matching publisher recordings in the English calibration sample. | Generated captions were invalid UTF-8 and generated translation speech had poor agreement with the human references. Human review and useful translation quality remain unestablished. |
| `dynamic-translation-path-comparison-02` | Same served weights and held-out sentence families compared direct joint generation with actual committed-text-then-speech generation, including first-output and completion times. | Both paths failed text quality. There is no quality-matched latency winner. |

The first duplicate-filter implementation discarded sequential translation views
because their transcript bytes matched a joint view. Admission now keeps different
causal training contracts while still linking their original families for split
safety. Merely renaming a track or changing physical chunk cuts does not manufacture
a new contract. All forty paired-translation views are retained in the third
preparation above, with no quarantined rows or discarded duplicates.

The current training uses the same explicit public requests as serving, and
teaches both ordinary-text and canonical tool choices against their actual legal
alternatives. No legacy objective or request mode remains. Imported tool calls
must be normalized before preparing training data. Selected native actions always
retain their complete explicit event-to-record bindings.

The current model architecture and checkpoints are version 16, prepared episodes
and datasets are version 6, and grouped recorded actions are version 19. Older
artifacts are rejected. There are no schema migrations, inferred saved geometry,
legacy data cursors, alternate old objectives, or dense-layer compatibility views
for shared expert modules. Reprepare original sources, recollect native actions,
rebuild model-bound retrieval data, and train new weights. Ordinary exact restart
continues to require the unchanged current contract and saved objective.


The bounded learned scheduling experiment is reproducible with:

```bash
python scripts/accept_delegation_choice.py --root /absolute/new/scheduling-run \
  --families 64 --steps 64
```

This application asks the model to choose between two fixed, actually executed
procedures. It constrains output to those two tools and tests held-out original
input families. It reports the fraction whose chosen procedure both verifies and
has lower measured execution cost; higher is better. Always-serial and
always-delegate baselines must accompany that result. This is a test of bounded
cost choice, not free-form worker creation, arithmetic or general scheduling.
The training budget and source archive are part of the experiment receipt.

The source inventory records listed, accessible, downloaded, prepared and sampled
supply separately. The newer local inventory is `dynamic-source-inventory-02.json`
with supporting `dynamic-source-access-receipts-02.json`. It includes locally
checked original CoVoST, FLEURS and separate-meeting AMI assets. The held-out video
acquisition is explicitly only the first eight MiB of its original container, not
a complete original-video download. Inventory entries and task-view counts must
not be added together as if they were independent original recordings.

Completion still requires representative supply for every mandatory behavior,
successful learned wait/emit and autonomous tool execution, quality-matched neural
delegation benefits, adequate decoded speech/translation/audiovisual agreement,
and human-reviewed outputs. A substantive production run must satisfy those gates;
none of these short diagnostic runs substitutes for it.


### Latest learned-choice result and timing correction

`dynamic-learned-delegation-choice-03` completed sixty-four GPU optimizer updates,
two more updates after a fresh-process restart, export, and student-only evaluation
on twelve held-out tasks from four original input families. The model selected a
verified, lower-cost procedure in eight attempts. Always choosing the serial
procedure also succeeds on this cost-choice criterion in eight attempts. Thus the
bounded learning acceptance check fails. All twelve selected fixed inner programs
returned independently verified task results; those fixed programs perform the
arithmetic and file operations, so this is not evidence of neural arithmetic or
free-form tool competence.

The original execution helper now uses a deterministic logical clock for recorded
causal schedules. Its `elapsed_seconds` annotation must not be interpreted as
wall-clock performance. `dynamic-delegation-comparison-03` and the learned run's
`evaluation-wall-clock-01` explicitly measure elapsed time with a monotonic clock
and retain the simulated schedule values separately. These replace older reports
that did not distinguish the two clocks.

On those twelve small tasks, actual model choice plus selected-program execution
averaged approximately 0.671 seconds; the predefined serial procedure alone
averaged approximately 0.039 seconds. Both measurements exclude the private
counterfactual reference probes; the serial baseline requires no neural procedure
selection. Lower time is better at verified task results. This experiment provides
no net latency benefit. Exact per-attempt measurements and the calculation are in
`dynamic-learned-delegation-choice-03/net-completion-time.json`.

The final local evidence index is `dynamic-wider-completion-evidence-02.json`.
It checksums the reports and frozen sources, retains the explicit measurement
correction, and leaves `whole_plan_complete` and `production_quality_established`
false. Passing software tests, prepared task views, decoded buffers and finished
optimizer runs must not turn either flag true without the remaining generated
behavior and quality evidence.


### Strict current-contract work

The local experiments listed above are historical diagnostics, not evidence that
older artifacts can be loaded by current code. New acceptance runs must start with
new preparation and current weights. Original recordings and unchanged donor assets
remain reusable only after their own exact identity checks. The repository does not
support backward compatibility, migration switches, or guessed missing artifact
metadata. Tests explicitly reject old model configurations, prepared episodes,
data cursors, action encodings, and native selections without member bindings.

Request construction no longer adds a training-only position number or inferred
inventory of inputs. Track purposes and causal context are already part of the
model state; only explicitly available instructions, settings and evidence requests
become request controls. The ordinary serving and collection interfaces construct
those same bytes. Shared expert projections expose their actual parameter owners;
they do not imitate a removed dense-layer interface.


### Causal student learning from the first update

The earlier all-teacher warmup made short diagnostic runs depend entirely on
look-ahead information that serving never receives. It has been removed. Ordinary
student training reads only the causal bank and available inputs from its first
window. Tool/lifecycle training and recorded student actions never receive
look-ahead teacher values. Other supervised data can explicitly enable auxiliary
teacher windows: one in eight, independently counted as `teacher_assisted_windows`.
They must not be reported as `causal_student_windows` or as generated student quality.

Policy-only optimization also freezes the sparse memory contents, its writes and
its growth, in addition to dense encoder/renderer parameters. Exact checkpoint
restart requires the saved current memory-teaching schedule. Current model and
checkpoint version 16, grouped action version 19 and native bank identity v12
reject the superseded model-bound artifacts. Unchanged prepared-data version 6
and independently pinned donor/source assets retain their own current identities.

The counterfactual regression changes a later tool answer while keeping public
inputs fixed and requires the earlier prediction to be byte-exact. It also forbids
executing the future-reading teaching pass during student tool training. A separate
regression verifies that a populated frozen memory remains unchanged through the
sparse update boundary.


### Current integrated run and strict admission

The current code rejects earlier internal models, prepared episodes, cursors and
native actions. The older linear-transcript converter has been deleted, including
its tests and its use in mixed-data acceptance. Rebuild from original sources;
do not relabel a prepared root or fill in missing model or memory metadata.

Multiple independently prepared roots can now be bound explicitly to the same
training/restart/export/evaluation run. Their codec, preparation policy and teacher
identities must agree. The helper rejects an empty held-out evaluation stream.
For example, replacing every uppercase argument with an actual checked artifact:

```bash
python scripts/accept_streaming_translation.py \
  --data SPEECH_ROOT --data EXECUTED_ROOT --data TEXT_AND_CODE_ROOT \
  --data CONVERSATION_ROOT --data AUDIOVISUAL_ROOT \
  --packs PINNED_COMPLETE_DONORS --teacher-writer PINNED_WRITER \
  --model 32x256 --capacity 1024 --precision bf16 \
  --microbatch-episodes 4 --gradient-checkpointing --expert-backend triton \
  --root NEW_RUN_ROOT --phase B0 --steps 256 --audio-samples 192000 --visual-output
```

Teacher preparation writes a new, checked target collection for those exact roots
and window settings. A fresh process releases the writer before allocating the
student and media decoders. A later segment can reuse the exact value encoding
with `--teacher-encoding-from`; changed teacher identity or dimensions are rejected.
The original donor is an input, never an older student model being upgraded.

The cloud execution receipts are `dynamic-integrated-cloud-deployment-02.json`
under `/home/werg/megatoken-runs`. Required preparations remain explicit dependencies:
web text/code, real speech, executed tool tasks, naturally overlapping meetings,
recorded visual prefixes and paired translation. A failed or absent required
preparation stops the sequence. The larger meeting command requests sixteen
nonoverlapping annotated intervals from each of four original meetings, with
disjoint speaker identities across training and evaluation. This is a requested
supply size until its preparation report confirms admission.

The current architecture/checkpoint version is 16; grouped actions use version 19
and native memory uses identity v12. Prepared-data version 6 and unchanged donor
assets retain their independent current identities. Quality, autonomous execution,
all-cost neural timing benefits and human review remain required acceptance gates;
launching this sequence does not establish them.


### Cloud interruption and retained evidence

The original cloud machine was stopped at 2026-10-23 19:54:16 UTC. Its service
reports a requested stop, and it has no retained filesystem snapshot or volume.
Starting the same identifier does not recover its stopped filesystem. The new
speech, text/code, visual and conversation preparations on that machine were not
collected before the stop; do not count them as retained current training supply.
Pinned originals and unchanged donor assets can be reacquired, with exact checks.

Replacement requests, including a smaller CPU allocation with the GPU and required
memory preserved, were rejected with a quota error. The connected Hugging Face
account also rejected a GPU job with HTTP 402. No cloud training is currently
claimed active on the strength of the earlier launch receipts. Local bounded GPU
experiments remain separate and retain their source and result directories.

The reproducible combined driver is `scripts/run_dynamic_training.py`. It prepares
all declared domains, runs foundation, media-alignment and joint optimization in
fresh processes, checks restart/export, and attempts autonomous tools, actual
student outcomes, paired translation, independent decoded checks and a recorded
interruption scenario. Every failed independent check remains in the report;
finishing that computation explicitly does not assert production or human quality.
Its writer is supplied with an exact declared SHA256, and absence or mismatch
stops admission to training. This is the current execution path for the wider
experiment once cloud quota is restored.

The previous tiny cost-choice experiment still chose the serial procedure in all
twelve held-out attempts: eight correct choices, equal to the constant serial
baseline. It therefore fails its declared learning threshold. It began recurrent
training with an untrained, fixed text reader. Fresh cost-choice runs now train
that reader and the prediction/rendering paths together using bootstrap training;
only their completed generated evaluations can establish an improvement.


### Preserve progress before the cloud machine ends

Run `scripts/collect_dynamic_progress.py` from a separately authenticated local
process while the experiment is still running. It captures committed current
prepared roots, exact teacher targets, exported models, the newest complete
checkpoint, reports and bounded log tails. It never marks outputs collected or
stops the machine. For example:

```bash
python scripts/collect_dynamic_progress.py --sandbox SANDBOX_ID \
  --remote-root /work/out/dynamic-integrated-v16 \
  --output /home/werg/megatoken-runs/dynamic-progress-backup
```

The collector checks the checkpoint members against their committed checksums and
pins them before the trainer prunes older checkpoints. Local content-addressed
copies avoid downloading unchanged bytes repeatedly. Each capture retains an
immutable path-to-checksum receipt; captures with errors do not replace the latest
successful receipt. A local disk reserve stops collection before it fills the
machine. Collected bytes retain their original contracts: this supplies no schema
conversion or missing-field reconstruction.

A direct CPU-only quota probe requesting one CPU core and one GiB RAM was rejected
with the same account/plan message as the GPU probes, while the account listed no
active machines. This points toward account credits or a spending/plan limit;
the service did not expose a more specific quota identifier or remaining balance.
