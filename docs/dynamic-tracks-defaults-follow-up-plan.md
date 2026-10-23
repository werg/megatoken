# Follow up plan for reliable dynamic tracks and training data

Proposed implementation and training work, 23 October 2026.

The system should make useful progress from an ordinary user request: create tracks
when they help, coordinate tools and sources, keep listening during speech, recover
from failure, and finish without leaving resources occupied. Training must teach
the choices that produce this behavior, including choosing to keep work serial.

This plan extends the [original track design](dynamic-tracks-and-concordant-training-plan.md)
and [preparation and training runbook](dynamic-training-runbook.md). Its scope covers
the runtime defaults, data changes, and downstream training and evaluation needed
for those defaults. Implementation changes described below are proposed work.

## Current support and remaining work

The current working tree includes changes beyond the last track implementation
commit. Preserve those additions and reconcile them before implementing this plan.
Use a fixed source revision for each preparation and experiment.

| Area | Existing support | Follow up |
| --- | --- | --- |
| Dynamic tracks | Stable addresses, semantic purposes, lifecycle actions, per-event modality choice, shared attention | Teach useful creation and completion; integrate the host execution loop |
| Scheduling | Bounded ready queue, prerequisites, priorities, periodic service of older work | Separate waiting from execution capacity; explain and recover from blocked work |
| Tools and sources | Concurrent execution, snapshot reads, source publication, evidence addresses | Exercise real adapters through the automatic loop and recover across disconnection |
| Speech and media | Asynchronous rendering, delivery acknowledgements, cancellation, bounded output memory | Provide a usable voice profile and demonstrate sustained natural conversation |
| Training supply | Timed source preparation, checked execution import, source inventory and measured sampling | Supply representative families and strengthen admission beyond behavior presence |
| Streaming translation | A newer preparation path authors wait, continue, and finish decisions over source arrivals | Teach and evaluate the model's own sustained timing choices |
| Outcome learning | Recorded context, verified outcomes, failures, and a restricted mode that keeps output decoding fixed | Collect useful successes and failures with trustworthy costs and independent verification |
| Session memory | Scoped conversation memory and source references | Preserve scope and evidence through long sessions, cleanup, and recovery |

The current lifecycle limit counts active and waiting tracks together. Several
recipe requirements accept a single original family per behavior. These are
starting safeguards, not sufficient production defaults. The authored execution
generator currently centers on reading numbers from two documents and writing a
verified result; expanding its values alone will not provide broad task diversity.

Implementation entry points are the [runtime](../src/megatoken/multitrack_runtime.py),
[track lifecycle](../src/megatoken/lifecycle.py),
[service](../src/megatoken/multitrack_service.py),
[coverage checks](../src/megatoken/data/coverage.py), and
[training recipes](../src/megatoken/data/stages.json).

## Automatic session execution

Provide one standard host loop used by interactive serving, recorded student
sessions, and generated evaluation. It should accept input, integrate completed
tools and published sources, execute committed lifecycle requests, select ready
work, collect rendering outcomes, deliver available output, acknowledge delivery,
and release retained outputs. Existing individual operations remain available for
controlled experiments and integrations.

The host owns resource accounting and external execution. The model chooses task
decomposition, requests, modality, waiting, and completion through recorded actions.
Do not fill in those decisions with scripted answers during learned evaluation.

The loop must:

- Wake on incoming input or completed work without requiring the client to poll
  every individual operation. Sleep when nothing can progress; avoid repeated
  model calls on unchanged state.
- Admit user interruptions promptly, stop future delivery as requested, and retain
  facts about selected content, delivered speech, and executed effects.
- Run compatible independent calls together and publish each ready result. Respect
  dependencies and conflicting writes. A slow or failed peer must not hold back
  unrelated completed work.
- Distinguish successful completion, waiting for an external event, resource
  pressure, recoverable failure, and inability to finish within the task budget.
- Close resources on session end and host shutdown. Reconnection must not repeat
  acknowledged delivery or an already executed effect.
- Retain an inspectable record of the model decisions and host actions used by
  collection, training replay, and evaluation.

Acceptance requires an ordinary request to progress through generated track
creation, actual adapter calls, source use, and final response without a test
driver supplying intermediate task decisions. Exercise both serial and parallel
requests, interruptions, decoder failures, and disconnected clients.

## Capacity and blocked work

Separate the number of known track identities, queued requests, waiting tasks,
running model work, tool calls, and retained output bytes. Waiting releases its
execution allocation while retaining its purpose, prerequisites, and address.
Bound waiting-task metadata separately so unlimited waiting cannot consume memory.
Do not reintroduce a lifetime track vocabulary or learned address table.

Add admission and resumption rules based on available resources. Keep capacity for
urgent input and for the work that unblocks other tasks. Preserve fair service
across tracks and sessions. Waiting parents must not occupy every allocation while
their children are unable to run. Exhausted tasks require an explicit budget
extension or a recorded unsuccessful completion; avoid silently resetting budgets.

Validate prerequisites when queued work is admitted and whenever they change.
Detect cycles, unknown dependencies, cancelled producers, unavailable capabilities,
stale task revisions, and exhausted budgets. Allow explicitly declared future
external arrivals with a bounded wait, rather than treating every unknown future
event as valid indefinitely.

Report what prevents progress and what could unblock it. Depending on the cause,
the model can revise the task, choose a fallback modality, request missing input,
cancel dependent work, or report an incomplete answer. Tool deadlines and retry
rules must account for whether an external effect already happened. An uncertain
write outcome requires reconciliation before retrying.

Acceptance includes all parents waiting, dependency cycles, a producer that fails,
late results after cancellation, saturated rendering, and continuous urgent input.
Every case must either make eventual progress or produce a bounded, explained
terminal outcome, with no duplicate external effect.

## Operating profiles and long sessions

Add named text, voice, and corresponding-media profiles. Each declares installed
input and output capabilities, public output bounds, chunk sizes, interruption
handling, work budgets, tool contracts, source settings, and delivery behavior.
Validate the profile before opening a session. Advertise the actual capabilities
to the model and client. Missing speech support should produce an explicit startup
failure or an explicitly selected text fallback.

Choose initial profile values from measured hardware and decoder behavior. Publish
their input-to-response and interruption-to-stop targets, measurement boundaries,
and memory budgets. Keep media output geometry independent of hidden target data.
Changing modality within a track remains supported by every applicable profile.

Bound session bookkeeping as well as attention and output tensors. Account for
copied text, completed event addresses, task transitions, tool deduplication,
scheduling records, source receipts, and any recorded policy context. Distinguish
the compact live state from durable audit and replay records.

Retain enough information to attach late results correctly, prevent repeated
effects, and preserve unresolved references. Retire completed details only after
all consumers and late-result rules permit it. Archive old transcript content and
allow explicit retrieval when its attention entries have been evicted. Do not
silently retarget an old evidence address to a summary or a different passage.

Specify process recovery: persist selected actions and effect receipts in the
correct order, reconcile in-flight work, restore task revisions and source
visibility, and require fresh model context reconstruction where necessary. Keep
private conversation memory attached to its original scope.

Acceptance requires sustained sessions at fixed concurrency to reach a stable
live-memory range. Measure the live memory excluding intentionally growing durable
archives. Restart during tool execution and partial speech delivery; verify no
effect duplication, wrong-track delivery, lost prerequisites, or cross-scope memory.

## Data collection and preparation

### Teach when delegation helps

Create related tasks in which independent work helps, overhead makes serial work
preferable, and dependencies require ordered execution. Examples should cover
independent evidence extraction, repository investigation, candidate comparison,
document comparison, search, translation, and synthesis of actual completed work.
Include easy requests for which a direct answer is sufficient.

For each task family, run comparable serial and parallel approaches against the
same immutable initial inputs and execution rules. Preserve separate histories and
verify their results independently. Record selected work, actual calls, model
computation, and elapsed time. Do not expose future results or the better approach
as a public task hint. Keep related approaches in the same data split.

Provide complete demonstrated trajectories for initial supervised learning, then
collect the student's own decomposition choices. Cost-sensitive outcome training
should reward verified useful work while accounting for measured delay and wasted
execution. Completion speed must not compensate for an incorrect result. Distinguish
host queue delay from decisions the model could improve, and never assign the
student credit for a teacher's later repair.

Acceptance requires improvement on independently authored families: comparable
task quality with less unnecessary work on simple tasks, and useful concurrency on
tasks with independent work. Separately test tasks that require serial execution.

### Collect complete conversations with concurrent work

Record user and assistant channels independently, including source clocks,
availability, selected outputs, actual rendering, host-reported delivery, tool
execution, and searchable-source publication. Preserve interruptions, silence,
backchannels, corrections, unfinished utterances, cancelled output, and resumed
answers. Natural meetings supply useful audio overlap; assistant tool coordination
requires actual assistant execution records.

Collect combined sessions where speech, independent tools, incoming documents, and
user corrections overlap. Vary arrival order and duration, including tools finishing
after interruption, sources becoming available after a search, and partial success.
Train a resumed answer to use the user's latest request and only available evidence.

Use the standard session loop for student collection. Keep scripted schedules
labeled as authored and natural channels labeled by their real origin. Preserve
unsuccessful attempts and actual decoding failures with their complete selected
groups. Audit a sample of recordings and annotations before expanding supply.

### Expand task and failure diversity

Add distinct task generators and real adapters across repositories, structured
tools, documents, evidence search, translation, and media. Vary task structure,
number of dependencies, resource conflicts, source quality, and result size.
Generating new numeric values from one template does not establish a new task
family for evaluating generalization.

Include rendering errors, empty retrieval, conflicting sources, delayed and failed
tools, truncated results, stale versions, lost connections, exhausted budgets, and
unavailable modalities. Label injected faults separately from naturally observed
failures. Teach recovery, clarification, a bounded partial answer, and appropriate
termination. Retain successful peers when another peer fails.

Maintain both useful successes and informative failures. Sample by original family
and behavior so many near-identical failures cannot overwhelm other learning. Use
verified outcomes where possible and human review for conversational usefulness.
Do not treat lack of a success label as independently verified failure.

### Strengthen corresponding media and translation

Retain corresponding source audio, video, captions, speakers, and translation spans
with exact clocks and availability. Prepare features from the available source
interval. Keep the target speech duration independent of source duration. Separate
speech transcription, visible-event captions, and audible-event descriptions.

Create matching and deliberately incorrect pairs: shifted audio, wrong captions,
speaker swaps, semantically wrong translations, and missing modalities. Keep source
and modified versions in one family. Record the transformation and apply it only
to the intended relation; corrupting media must not change an unrelated target.
Add time-local alignment supervision and explicit correspondence discrimination
with a defined loss and denominator. Test both independent content quality and
whether peers agree in content and time.

Extend the existing authored wait/continue translation path with diverse source
prefixes, long dependencies, repairs, and different language orders. Train when to
wait, emit a supported partial translation, continue, and finish. Ensure the public
history contains neither future source nor target-readiness labels. Compare joint
text-and-speech output with text followed by speech using equal source availability
and public budgets. Generated evaluation must let the model choose timing.

### Preserve meaning while varying addresses

Vary opaque track and event addresses, track creation order, group size, and result
arrival order while preserving the actual dependencies and semantic purposes.
Teach speech, text, and tool work on the same continuing track, including multiple
corresponding peers. Include descriptions whose meaning changes through an explicit
revision, so a stale worker result cannot silently answer a revised request.

Renaming requires a consistent rewrite of references, prerequisites, tape addresses,
and execution bindings. Reuse raw selected content only where replay remains valid;
otherwise recollect under the transformed request. Verify address invariance with
controlled random draws and semantic sensitivity with changed task descriptions.

### Separate families and prove exposure

Split original tasks, conversations, recordings, and all derived variants together
before augmentation. Group related repository revisions and task templates. Where
the evaluation claims generalization to new speakers, repositories, or templates,
hold out those entities explicitly as well. Document which split tests which claim.

Expand admission checks beyond behavior presence to cover original-family diversity,
task domains, successful and failed outcomes, source origin, language, dependency
patterns, and simultaneous behaviors. A corpus containing separate audio and tool
examples cannot satisfy a combined conversation-and-tool requirement.

Set separate pilot and production requirements. A small software fixture may use
minimal supply and must identify itself as a pilot. Production admission requires
frozen training, validation, and test inventories with enough independent families
for the planned quality comparison. Determine those counts from pilot variability
and a declared minimum useful improvement before the final evaluation is unsealed.
Do not silently lower requirements to fit currently accessible sources.

Extend the existing inventory and training counters to show prepared, eligible,
sampled, and actually supervised examples for every required behavior and outcome
stratum. Deduplicate original families across roots. Counts must survive restart
and distributed reduction. Detect a supplied stream receiving no effective loss or
being squeezed out by the mixture. Preserve source terms, checksums, preparation
versions, and original task ancestry in the exported evidence.

## Downstream training and evaluation

Carry new annotations through episode validation, asset copying, deduplication,
prepared export and restore, causal window formation, batch preparation, shared
loss accounting, held-out reporting, checkpointing, and served export. Version any
changed preparation or replay meaning; reject incompatible resumes explicitly.

Keep complete selected groups in history and policy scoring even when individual
outputs fail. Reconstruction and quality losses apply only to valid corresponding
targets. Failed and structural decisions still need correct progress accounting.
Keep negative correspondence labels distinct from output targets and task rewards.

Use supervised demonstrations to establish useful decomposition, waiting, recovery,
modality selection, and corresponding output. Follow with student collection and
independently verified outcome updates. Preserve the existing restricted training
mode that fixes output decoding when optimizing the selected-content policy.
Measure replay consistency and unchanged decoding under that restriction.

Calibrate mixture shares using actual decision counts, media duration, useful-work
coverage, and held-out learning curves. Text-token volume alone must not determine
exposure to speech, failures, or lifecycle choices. Keep replay of ordinary language,
serial tasks, and existing retrieval tasks to test and limit regressions. Check
whether an outcome example is selected, scored, and weighted meaningfully rather
than merely listed in the stage recipe.

Evaluate through the exported standard session loop, with the model making its own
track, tool, wait, and modality decisions. Keep requested-output diagnostics and
scripted protocol checks separately labeled. Use fixed input schedules and paired
serial comparisons; record hardware, decoder identities, concurrent load, random
seeds, and all denominators.

| Measure | Definition and desired result |
| --- | --- |
| Task success | Independently verified correct completions divided by attempted tasks; preserve or improve the serial baseline |
| Useful delegation | Task quality together with work and elapsed time for serial, parallel, and dependency-constrained families; gain where independence exists and avoid waste on simple tasks |
| Work cost | Actual selected outputs, executed tool calls, and measured model/decoder work per attempted task, reported separately; avoid counting queued requests as completed work |
| Response latency | Time from an available user input boundary to the first useful delivered response; meet the frozen profile target under declared load |
| Interruption latency | Time from accepted interruption to the last subsequently delivered output chunk; meet the profile target and distinguish host acknowledgement from physical playback |
| Evidence validity | Claims linked to source observations available at the relevant decision, with original provenance retained; reject use of future or wrong-source evidence |
| Translation timing | Delay relative to source availability alongside independent translation quality, omissions, and revisions; improve the declared quality and delay trade-off |
| Media correspondence | Independent matching judgments and time-local checks on genuine and deliberately incorrect pairs; detect shifts and mismatches without rejecting valid pairs indiscriminately |
| Recovery | Correct or appropriately explained terminal outcomes after each fault type, divided by affected attempts; preserve successful effects and prevent duplicates |
| Session stability | Live memory, queued work, and user latency during sustained use and recovery; remain within declared budgets without starvation or lost references |

Freeze quality margins and latency/memory targets after the development pilot and
before final held-out testing. Report uncertainty over original families and review
representative generated sessions. Tune on validation only. A finite decoded tensor,
an authored successful trace, or a passing scheduling test cannot satisfy a learned
behavior or media-quality requirement.

## Implementation order and delivery gates

| Delivery | Concrete result | Required gate |
| --- | --- | --- |
| Reconciled baseline | Fixed source snapshot, current inventory, behavior matrix, serial baseline, and profile measurements | Existing preparation, restart, export, and protocol acceptance remain reproducible |
| Automatic runtime | Standard session loop, separate execution/waiting limits, blocked-work outcomes, profiles, and cleanup | Real adapter integration and failure/restart tests pass without scripted intermediate model decisions |
| Diverse pilot data | Broad task families, paired serial/parallel approaches, combined conversations, translation waits, and correspondence negatives | Source review, causal validation, family splitting, manual annotation audit, and export/restore checks pass |
| Training integration | New coverage requirements, losses, sampling reports, and recipe settings | Actual optimizer steps reach intended parameters; denominators and progress match single-process and distributed references; restart preserves exposure |
| Learned behavior pilot | Supervised and student-outcome training with generated validation reports | Useful successes, bounded failure recovery, address invariance, and absence of future-evidence shortcuts are demonstrated |
| Production defaults | Calibrated profiles and mixture, frozen evaluation report, and deployment guidance | Held-out quality, serial comparison, correspondence, sustained operation, and recovery meet predeclared targets |

Build the task inventory and initial source audit alongside the runtime loop. Collect
final student trajectories only after the loop's execution and delivery semantics
are stable. Waiting/capacity changes precede stalled-work collection. Preparation
and annotation changes precede rebuilding data and changing production mixtures.
Run representative real-decoder pilots before selecting voice chunk defaults.

For each delivery, update the runbook, serving instructions, examples, CI checks,
and implementation status. Archive source and training revisions, admitted data,
coverage and exposure reports, checkpoints, generated sessions, and evaluation
settings. Keep rollback to the last validated profile and served model possible.

Completion requires representative prepared supply, measured exposure through the
ordinary training engine, and generated behavior through the exported automatic
session loop. A feature remains a pilot until its own data, quality, and operational
gates pass.
