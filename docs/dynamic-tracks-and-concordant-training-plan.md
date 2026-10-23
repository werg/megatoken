# Dynamic tracks and concordant training

Review and implementation plan, 23 October 2026.

Implementation follow-up: this document preserves the original review and its
acceptance requirements. The descriptions under “How tracks currently work” are
the reviewed starting point, not the current implementation. See the
[current implementation status](multi-track-implementation.md) and
[preparation and training runbook](dynamic-training-runbook.md) for delivered
changes, runnable commands, and remaining data/quality requirements.

Tracks should identify continuing work or conversation. The model should be able to create them during a session, describe what they are for, and choose the modality of each event independently. Workers, tools, retrieved sources, speech, video and captions should use the same shared model and causal history.

The current implementation supplies much of the shared computation and grouped training. It does not yet supply a complete lifecycle for model-created tracks, sustained asynchronous execution, or training that consistently teaches those behaviors. Extend that foundation rather than introducing separate reasoning models for each track.

This proposal incorporates the requested changes to track addressing, semantic conditioning and modality selection. It specifies remaining work; it does not claim that the proposed changes or training runs have happened. The review covers the current working tree, including existing uncommitted changes. The older [implementation status](multi-track-implementation.md) describes an earlier architecture and should be reconciled with the code as this work lands. The current [media design](media.md) and [concordant task roadmap](pretrained-omnimodal-bottleneck-plan.md) remain the starting points for media components and tasks.

Completion means that the selected sources are prepared, sampled by the real training engine, used to train all new decisions and content paths, and demonstrated in generated execution through the exported serving system. A new reader, training option, helper, or passing synthetic example alone does not complete a capability. Data access, decoder support and compute shortages leave the affected work incomplete; they must not silently remove it from the curriculum.

## How tracks currently work

A track is a named stream of events within one session. All tracks use the same model and retained attention history. A worker track does not have a private copy of the model or a private context. Moving work to another track separates its progression and purpose; it does not by itself remove its content from shared attention or reduce the total computation.

The model processes events in atomic groups, called meridians in the implementation. Already available observations can be integrated together. Requested outputs can be predicted together from the earlier history, with interaction between their prediction queries. Once selected, their representations are integrated together for future steps. An output cannot respond to a newly arriving observation until that observation has been integrated in an earlier group.

| Capability | Current behavior | Remaining limitation |
| --- | --- | --- |
| New track names | First use of a name registers it automatically | The caller chooses names and requested outputs; there is no complete model-driven create, finish and resume protocol |
| Track addressing | Names receive session-local numbered slots | Slots accumulate for the session lifetime and index learned routing tables |
| Semantic track information | Names are encoded for prediction with multiple outputs; event instructions and track descriptions can supply encoded task meaning | This information does not replace the numbered-slot lookups in MoE routing; the name path is conditional on group width |
| Shared attention | Events can read retained history across tracks; attention also distinguishes same-track events | There is no private worker context, and separating tracks alone does not limit attention cost |
| Mixed modalities | Each event has its own modality; successive events on a track may differ | A jointly processed group currently allows at most one event per track, and the caller specifies output modality |
| Worker execution | Ordinary named tracks can represent workers | Task assignment, lifecycle, progress, dependencies and completion need an integrated runtime and training protocol |
| Parallel tools | A bounded executor runs externally supplied functions concurrently and returns ready results | Serving does not automatically bind model-generated calls to this executor or real adapters |
| Retrieval | Serving retrieves source evidence; training includes query alignment, annotated-source ranking and continuation over retrieved observations | Concurrent ingestion, worker coordination and interruption during retrieval need additional tasks and execution support |
| Concordant media | An aligned-source compiler creates joint translation and audiovisual tasks; training has semantic relation losses | Detailed timing, streaming chunk alignment and decoded synchronization remain unfinished |
| Live output | Group generation renders outputs before accepting the resulting state | Listening, rendering, playback and source ingestion do not yet progress through one complete asynchronous lifecycle |

The default limit is **eight ordinary track identities over the entire session**, with internal system tracks treated separately. This is an implementation limit, not a recommended operating target or an inherent attention requirement. Increasing it enlarges learned tables and does not solve lifetime accumulation.

Current-code evidence:

- Dynamic registration and shared grouped state: [grouped model](../src/megatoken/model/multitrack.py), particularly track registration, prediction and generation.
- Slot-dependent expert selection: [expert routing](../src/megatoken/model/expert_pools.py) and [learned adjustments to experts](../src/megatoken/model/adapters.py). Both contain learned tables indexed by track slot.
- Semantic information: [prediction](../src/megatoken/model/multitrack.py) encodes track names for multiple outputs; [instruction resolution](../src/megatoken/controls.py) uses event instructions or track descriptions.
- Identity comparisons in attention: [attention](../src/megatoken/model/attention.py). The same-track comparison itself does not need a bounded vocabulary of learned track identities.
- Serving and tools: [session runtime](../src/megatoken/multitrack_runtime.py) and [service](../src/megatoken/multitrack_service.py). A session has one pending speculative candidate; this is not a scheduler for many independently progressing workers.
- Grouped data and training: [episode representation](../src/megatoken/data/schema.py), [task compiler](../src/megatoken/data/concordance.py) and [grouped learner](../src/megatoken/training/meridians.py).
- Existing concordance objectives: [semantic relation loss](../src/megatoken/model/relations.py). Its pooled semantic comparisons do not establish acoustic or visual synchronization, even for a relation labeled as synchronized.

## Rework track addressing and semantic conditioning

Separate three things currently coupled through the track slot: identity, meaning and resource allocation.

**Identity:** Give each track a stable session-scoped address allocated by the runtime. Events, tasks and tool calls receive their own stable addresses. A readable name is a label. Renaming it must not redirect an outstanding reply. Never recycle an address while historical events or late results can refer to it. Compact integer handles may still be used for tensor equality comparisons, without indexing a learned table or imposing a small model-defined lifetime vocabulary.

**Meaning:** Reuse the existing instruction encoder to represent the track's purpose, task and output expectations. Supply that semantic representation consistently to prediction, MoE routing and the learned adjustments to experts. Remove dependence on the arbitrary numbered track slot in both routing paths. Keep task role, modality and content as distinct useful inputs. Cache a track description's encoding by its revision and encoder version; changing the description creates an explicit causal revision.

The current encoded-name path is a partial foundation, not a complete semantic track representation. Make descriptions explicit, apply them for singleton as well as grouped execution, and remove opaque addresses from text used as semantic instructions. System functions should have explicit meanings rather than occupy slots offset from a user-track capacity. Update memory ingestion and its training paths as well as ordinary generation: they also construct track indices today.

**Resource allocation:** Limit active work, events processed together, queued observations, retained history, tool requests and rendering memory independently. A session should be able to create and finish many temporary tracks without retaining all their active state. Archive completed descriptions and event records as needed; keep a compact record sufficient to recognize late replies. Allocation failure should explain the exhausted resource and offer queueing or rejection before selection.

Do not solve this by hashing addresses into a small learned table: collisions would make unrelated tracks share learned identity information. Do not merely recycle old slots while their history remains visible.

Acceptance requires creation and retirement beyond the old lifetime limit, correct late-result addressing, identical routing under arbitrary address reassignment when semantics and content are preserved, and unchanged attention relationships after compact-handle remapping. Changing a task description should be able to change behavior; changing an opaque address should not.

## Choose modality per event

Add the requested classifier for output modality. Its input is the available shared history, semantic track description, event request and supported output capabilities. Explicit user or application constraints restrict its choices. Input events retain the actual modality declared by their transport or media container; the output classifier must not reinterpret known input media arbitrarily.

Make modality selection precede content selection so the joint predictor knows which kinds of outputs it is coordinating. The modality choice selects the appropriate output decoder. MoE expert routing remains a separate decision about how to compute the representation, using semantic context rather than a track address.

For simultaneous speech and captions on one track, support separately addressed peer events belonging to that same track. Each carries its modality, speaker or direction where relevant, and its relationship to other events. Retain an ordering distinction for successive chunks of the same stream. Two chunks with an actual dependency cannot become simultaneous peers merely because they share a track.

This requires replacing the current one-event-per-track restriction with unique event addressing throughout validation, grouped prediction, noise assignment, rendering, training and replay. Joint attention still uses track equality where useful. Deterministic random draws must be assigned by event identity, since track identity will no longer uniquely identify a group member. Existing media recipes may keep separate tracks where they represent different continuing work; modality alone should not require a new track.

Train the modality choice from declared target modalities. Add supervision for choosing multiple peer events and for waiting, continuing or finishing. Record these choices and their probabilities in replay and outcome learning alongside content choices. Do not count only the continuous content probability after introducing discrete decisions. Unsupported decoders must be excluded before selection. Requested duration and output geometry must come from available instructions or explicit predictions, never hidden target measurements.

## Make creation and delegation part of normal operation

Add a structured action through which the model requests a new track with a purpose, parent task, completion condition, allowed capabilities and budget. The runtime validates the request, allocates its address and commits the description before scheduling outputs on it. The model refers to returned addresses; it does not need to invent globally unique strings.

Support assigning work, publishing progress, waiting for named results, completing, cancelling and explicitly resuming work. Closing a track stops new selections while outstanding replies can still drain to their original addresses. Distinguish cancelling computation from stopping delivery and from cancelling an external tool effect. Already executed effects and already heard speech remain real.

Use one owner for session commits. Let independent workers make progress in bounded chunks and publish completed events between steps. Group compatible ready events; do not wait for every worker to finish. A dependent critique or synthesis follows the work it consumes. Long-running work records the history and task revision it used, allowing explicit acceptance, cancellation or recomputation if new input makes it stale.

Split selected representation, rendering, delivery and execution into separately recorded states. Commit the selected event before starting external effects; render from an immutable snapshot of its selected content and conditioning. Rendering failure becomes a later outcome rather than silently erasing a selected action. Extend the existing speculative-candidate checks without treating speculative state as completed worker communication.

Priority should protect user input, interruptions and requested responses while allowing useful delegated work to progress. Add queue limits, fair scheduling, timeouts, cancellation and memory pressure handling. Verify whether a completed event requires another retrieval before continuing; today's automatic lookups must not become an unnecessary blocking barrier on every independent activity.

## Audit and convert training approaches

Produce a source-by-source inventory across broad text, code, tools, executed environments, student-generated continuations, long-context tasks, retrieval and every media family. The inventory must distinguish listed, accessible, downloaded, prepared and actually sampled data. A catalog entry or a training mixture percentage does not establish training exposure.

For each source, record the original ordering, known dependencies, timing availability, tool execution identities, supervision quality and justified conversion. Keep linear traces linear when their independence is unknown. The common importer currently gives each record its own causal group on a role-based track; useful grouped behavior needs explicit conversion or newly collected execution.

| Training approach | Useful parallel work | Information that must remain ordered |
| --- | --- | --- |
| Ordinary text and documents | Independent analyses, evidence extraction or requested outputs from the same available source | Continuations and arguments that depend on earlier generated choices |
| Code and repositories | Search different areas, propose independent candidates, run tests on separate immutable candidates | Editing shared state, testing the exact edited version, review of a completed patch |
| Tools and environments | Independent calls with compatible resource contracts | Calls using another result or changing the same resource incompatibly |
| Worker trajectories | Separate bounded tasks sharing the available context | Critique, reconciliation and synthesis of actual completed work |
| Long-context tasks | Reading and extracting from independent available sections | Answers needing combined evidence that has not yet arrived |
| Retrieval | Search different sources, prepare incoming items, extract independently | Use of sources before ingestion or arrival, dependent search queries |
| Media | Corresponding audio, video, captions and translations; multiple views of the same scene | Unavailable source prefixes, later edits and future media |
| Student and outcome training | Record real scheduling and concurrent execution, including failures | Results must match the selected calls and exact candidate versions |

Train the decision to delegate as well as the delegated output. Include tasks where delegation helps, tasks where it wastes time, and tasks where shared dependencies require serial execution. Vary addresses, creation order, group widths, result arrival order and descriptions. Preserve each task's meaning while changing opaque identifiers so the model cannot rely on a favored slot.

Keep existing grouped training and the single optimization path. Extend the existing record dependencies to runtime readiness and lifecycle prerequisites, checking that required producers completed in earlier groups. Dependencies govern scheduling; they do not require replacing shared causal attention with an arbitrary graph of attention permissions. Preserve whole groups across training-window boundaries, exact executed-action continuations, and distributed loss accounting.

## Refactor media around corresponding timed sources

Concordant training means that outputs agree about content and, when required, timing. Speech should match its transcript; translation should preserve meaning; a soundtrack should match visible events. Independent plausible outputs are insufficient.

Retain one versioned collection of corresponding source artifacts and compile several tasks from it. Extend the existing compiler with complete media intervals, clocks, actual availability times, speaker identity, text spans, chunk correspondence and transformation provenance. Preserve exact sample or frame coordinates where needed rather than repeatedly rounding time. Split by original recording, conversation and relevant speaker identities before creating variants or synthetic derivatives.

Keep content time distinct from availability time and model processing order. A video segment and its transcript can describe the same interval while becoming available at different times. Feature extraction must use only the available prefix: cropping a whole-recording embedding after extraction does not prove causal safety.

Add these task sequences:

- Observe corresponding video and audio chunks; predict visible-event captions, spoken transcripts and audible-event descriptions with distinct meanings.
- Generate corresponding video, audio and captions from a supplied request. For underspecified creative tasks, optionally generate and commit a shared content choice before producing the peers. Do not reveal a target-derived content choice as an external input.
- Continue media across chunks while preserving speakers, scene state and event timing. Use missing-modality examples so the model can handle audio-only or silent video without inventing unavailable evidence.
- Translate source prefixes into corresponding target text and speech. Train when to wait, how much new source to consume, and how to continue while the next source chunk arrives. Target speech has its own duration; translation does not inherit source timestamps word for word.
- Preserve overlap in full-duplex conversations, meaning simultaneous listening and speaking. Include backchannels, interruption, repairs, turn changes, resumed speech and pending tool results.
- Retain edit, reference and endpoint tasks already supported by the compiler, with their actual causal order and preservation requirements.

Keep direct joint translation and add text-then-speech as a separate task. Compare them on quality and latency. Existing pooled semantic relation losses can encourage agreement but cannot establish word accuracy, lip synchronization or event timing. Add time-local supervision and decoded-output evaluations; test deliberately shifted audio and mismatched captions so a check must detect the disagreement. Use independent verifiers and a human-reviewed subset.

The present media design gives each event's decoder its own selected representation. Joint prediction can coordinate content, but independent sampling and separate decoders do not guarantee tightly synchronized output. Measure this explicitly. A shared committed content choice is the first mechanism for underspecified generation. If measured timing failures require coupled audiovisual decoding, specify its conditioning and retrain it explicitly rather than silently giving one decoder hidden access to another target.

## Execute parallel tools and ingest sources during conversation

Connect typed generated calls to real adapters after selection and validation. Each call and result retains its task, track, action and execution addresses. Admit ready results without waiting for unrelated calls. Record failures, timeouts, retries and cancellation; a successful peer remains a real effect when another fails.

The existing environment adapter changes its state version even after reads. Co-generated calls cannot simply share the old version and then be dispatched sequentially. Add an adapter that supports shared immutable snapshots or independent resources, and reject or schedule conflicting operations according to its declared contract. Test out-of-order completion, repeated delivery, conflicting writes and retries without duplicated effects.

Keep retrieval training integrated with ordinary continuation. Existing query alignment and annotated-source ranking should remain, with additional supervision for source usefulness, attribution, conflicting evidence, empty searches and multi-step search. Returned source content remains evidence, not an instruction override. Learned internal memory and external source search continue to have distinct roles.

Add source preparation as asynchronous work: discover or receive an item, prepare its available content, publish searchable entries, and announce readiness. Index publication must be atomic or versioned; a search must not read a half-written entry. Let speech and other work continue while preparation runs. Train using exact arrival and search schedules so an answer cannot cite a source that was not available yet. Preserve original evidence addresses through worker summaries and final answers.

Build a recorded full-duplex scenario in which a user speaks, the assistant starts an appropriate response, independent calls run, a document becomes searchable, and the user interrupts before all work finishes. The resumed answer must use only completed relevant results and available sources. This combined scenario is a required implementation and training case.

## Dataset shortlist and acquisition work

These are candidate sources and task roles, not acquired inventories. Public descriptions were checked on 23 October 2026. Pin revisions and inspect actual accessible samples before implementing each reader. Access restrictions below affect whether a source can supply the planned training; no external access requests were sent.

| Source | Proposed use | Important limitation or acquisition work |
| --- | --- | --- |
| [AMI meetings](https://groups.inf.ed.ac.uk/ami/corpus/) | Corresponding speaker audio, video, transcripts, slides and meeting evidence; useful initial integration source | Signals and annotations are distributed separately. Preserve channels and timing; not all auxiliary signals are synchronized |
| [AMI release terms and downloads](https://groups.inf.ed.ac.uk/ami/download/) | Obtain the relevant signals and annotations under the published CC BY 4.0 release | Check each requested signal and annotation release |
| [AMI synchronization notes](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml) | Select and validate material for timing-sensitive tests | Published notes describe repaired dropouts and weaker synchronization in some recordings; do not assume frame-accurate truth throughout |
| [otoSpeech task-oriented conversations](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-task-oriented-20h) | Real separated speaker channels with interaction events, task stimuli and artifacts on a shared timeline | Gated files; card declares CC BY 4.0. Task interaction is not automatically a verified parallel function-call trace; some referenced images are not redistributed |
| [otoSpeech task-oriented follow-up release](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-task-oriented-v2) | Additional recorded task interactions, including assistant-related activity | Gated files; inspect participant-specific visibility. Do not expose one participant's private interaction to another role |
| [otoSpeech conversation releases](https://www.oto.earth/) | Natural overlap, turn-taking and interruption training | Releases have distinct terms and access conditions. Local data notes record inaccessible, unwired entries; implement readers after sample access |
| [Basis Conversations 1500](https://huggingface.co/datasets/basis-ai/basis-conversations-1500) | Separated multilingual speaker tracks with overlap and conversational annotations | Gated under a custom data license; useful conversation data, not a tool-execution corpus |
| [CVSS](https://github.com/google-research-datasets/cvss) | Source speech, translated text and synthesized translated speech for concordant translation | Sentence-level pairs need chunk availability construction; target speech is synthetic. Match the required original source release |
| [Granary](https://huggingface.co/datasets/nvidia/Granary/blob/dba7072a5132f6b821e727ed5a0b8dbcf1e891b5/README.md) | Expand transcription and speech-to-text translation with source lineage | Manifests are not the underlying audio; component corpora have their own terms. It does not supply full-duplex tool trajectories |
| [MuST-C](https://aclanthology.org/N19-1202/) | Study segmented talk translation and longer source context | Published corpus terms are noncommercial and prohibit derivatives; resolve suitability before any planned conversion or production use |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool/blob/main/README.md) | Spoken multi-intent commands, corrections and tool-argument evaluation | Noncommercial terms; not evidence of concurrent execution, continuous source ingestion or full-duplex assistant output |
| [xLAM function calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/blob/main/README.md) | Seed tool planning and independently executable call sets where the actual calls justify it | Text supervision; execute and validate converted cases. Add audio and timing as explicitly synthetic derivatives only where permitted |
| [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) | Candidate corresponding visible and audible events | Official page says download links are no longer available and original video copyright remains with owners. Do not budget it as immediately accessible training material |
| [Ego4D and Ego-Exo4D](https://github.com/facebookresearch/Ego4d) | Grounded audiovisual evidence and synchronized views | Dataset access uses a separate [agreement](https://ego4d-data.org/); the software license is not the dataset license. Check modality availability per recording |
| [Tau full-duplex and knowledge framework](https://github.com/sierra-research/tau2-bench) | Evaluation and a possible basis for separately authored training environments combining voice, tools and retrieval | A framework, not a ready-made corpus proving all requested concurrency. Keep official evaluation tasks out of training and pin a release |

The local catalog already includes speech pairs and video captions, and the compiler already has joint audiovisual tasks. A captioned video row alone does not provide a soundtrack, timestamped transcript or synchronized generation supervision. Build an actual coverage report before setting training proportions.

No single verified corpus in this review supplies the whole combination of natural full-duplex interaction, model-created workers, parallel executed calls and live source ingestion. Combine complementary sources and collect new execution traces in controlled environments. Keep natural recordings, synthesized speech, simulated timing and verified execution distinguishable. Do not present synthesized overlap as naturally recorded interaction.

## Put every capability through the actual training pipeline

### Select sources against the behaviors being trained

Use the existing source catalogs as the starting inventory, including ordinary documents, code, agentic chat, coding sessions, later-stage tasks and media. Keep broad text and code competence in the mixture. Select new sources by the missing behavior they can actually supervise, rather than by headline size or the presence of a media column.

The first integrated preparation must include each row below. A blocked preferred source can be replaced only by a source or newly collected execution with the same required supervision. A caption-only source cannot replace timed audiovisual data, and independent spoken commands cannot replace interrupted tool execution.

| Required training supply | Initial selection | Preparation and admission requirements |
| --- | --- | --- |
| Track meaning and lifecycle | Existing approved tool and agent trajectories, supplemented by newly authored executable tasks | Recover actual tasks and calls; collect creation, assignment, progress, completion, cancellation and resumption through the new runtime. Imported serial logs retain their actual ordering |
| Deciding whether to delegate | Paired serial and delegated executions of the same training tasks | Use separate candidate environments, the same task requirements and comparable compute limits. Verify success and record work and delay; include cases where delegation is harmful |
| Parallel tool execution | Existing tool-schema examples plus controlled tools over immutable snapshots and independent resources | Derive independent call sets from arguments and resource contracts, execute them, and retain actual calls/results, errors and readiness times |
| Speech and exact transcripts | Approved paired-speech sources already in the catalog, including the available LibriTTS-R route | Check transcript normalization, names, numbers, language and speaker provenance. Keep transcription and speaking supplied text as distinct task directions |
| Translation text and speech | Admitted matching CVSS source and target releases; approved existing CoVoST/FLEURS routes as complementary coverage | Join by released example identity, verify language and meaning, label synthesized speech, and check the installed speech decoder supports the selected language/profile. Sentence pairs alone do not supervise live waiting |
| Chunkwise translation | Corresponding longer recordings and transcript/translation material; deliberately collected prefix-based translation tasks where long material is missing | Produce source-prefix availability, target chunk correspondence and explicit wait/emit examples. Require a reviewed alignment subset; do not invent true simultaneous timing from sentence order |
| Timed audiovisual understanding | AMI signals and annotations as the initial integration candidate; admitted corresponding audiovisual sources already in the catalog | Join original channels, frames and timed text. Exclude or flag timing defects according to the task's needs. Captions describe visible events; transcripts describe spoken words |
| Concordant audiovisual generation and continuation | Corresponding audio/video/caption material with a valid supplied request, plus explicitly generated training requests where needed | Construct causal generation tasks, including a generated shared content choice for underspecified cases. Preserve soundtrack continuity, clip boundaries and the distinction between genuine requests and target-derived supervision |
| Natural full-duplex behavior | Accessible otoSpeech or Basis material, supplemented by suitable separated AMI speaker channels | Preserve separate channels, silence, overlap and interruptions. Admit each release under its actual terms; missing access remains an acquisition task |
| Retrieval and source ingestion | Existing collection-reading producer, approved question/evidence tasks, and newly authored source-arrival environments | Keep support items out of the prompt; build split-specific collections with distractors, actual publication times and exact evidence references |
| Combined voice, workers, tools and source ingestion | Newly collected training environments using the implemented scheduler | Real executed tools, recorded user/assistant channels, interruption schedules and source publication. Simulated audio or delays are labeled as such. Reserve independently authored tasks for evaluation |
| Student corrections and successful replay | Rollouts from the new trained model through the same runtime used in serving | Retain complete multi-event decisions and actual failures; verify corrections against the recorded student state and exact resources |

Every admitted source needs a machine-readable record of its immutable revision, files and checksums, access status, terms decision, reader settings, original family identity, available modalities, language/speaker coverage, timing quality, annotation origin, decoder compatibility and allowed task directions. Reuse the existing recorded terms checks; do not mark access or approval complete from a public dataset card alone. Training use and redistribution remain separate decisions.

Choose training and validation samples before preparing derivatives. Union related recordings, excerpts, speakers where speaker separation is required, translations, edits, teacher-generated variants and alternate mirrors into their original families. Deduplicate across all preparation shards and catalogs before publishing the split assignment. Quarantine evaluation families and benchmark transcripts, including material used to author synthetic tasks. Keep a final untouched test selection outside recipe tuning.

For each selected source, run a bounded preparation pilot and report accepted original families, generated task views, valid training decisions, media duration, rejection causes, preparation time and bytes. These are supply measurements, not quality targets. Use them to calculate the production acquisition and compute budget. Freeze that inventory and the chosen recipe before the full run; an inaccessible source must not remain counted as usable supply.

### Prepare media without destroying timing or supervision

Extend the existing [media preparation path](../src/megatoken/data/media_prepare.py), [source readers](../src/megatoken/data/open_media.py) and [task compiler](../src/megatoken/data/concordance.py). Do not introduce another feature extraction or training representation for the new tasks.

1. Read original sessions and join their media and annotation records before dividing work into shards. Assign each original family to one preparation partition or provide an explicit cross-partition join. A per-archive reader must not lose the other speaker, a corresponding transcript or a later edit located in another archive.
2. Preserve original speaker channels before audio conversion. The current audio decoder requests mono; running it on a full-duplex mixture would destroy channel identity. Extract each speaker's channel separately, retain its original start time and silent intervals, and then resample each event for the installed decoder. Keep the original mixture as a distinct observation only for tasks that request it.
3. Preserve displayed-frame timestamps and the mapping through crop, resize, dropped-frame repair, resampling and trimming. The current video helper uses an average frame rate; extend it to preserve actual presentation times when the source is not uniform. Audio and video trims must describe the same intended interval.
4. Choose semantic chunks under explicit duration, frame, byte, token and memory limits. Preserve word/sentence and speaker boundaries when possible. Long media must become several causally ordered events rather than be silently truncated. Playback packets inside an event remain separate from model decisions.
5. Extract source features from the exact available chunk or prefix. Record the teacher, processor, crop and source hashes. Produce output targets with the corresponding installed decoder's own codec. Record valid coordinates separately from padding, and verify finite values and compatible geometry.
6. Align transcripts to source audio and translations by meaning, using verified labels or explicitly marked estimated alignment. Forced alignment may use a complete recording to produce supervision, but future text, derived timings and target durations must stay out of the predictor's public inputs. Confidence filtering must not discard every difficult overlap example; retain a separately characterized difficult subset.
7. Compile understanding, reconstruction, speaking supplied text, translation, generation, continuation and edit views only where their inputs and targets exist. Store relations between generated peer events and between current outputs and earlier evidence. Today's same-group relation selection must be extended for the latter; do not score a source/output relation as if both were new generated peers.
8. Prepare independent evaluation annotations for exact words, meaning and timing. Add deliberately shifted sound, swapped captions, wrong speaker assignment and missing-source cases as validation probes or appropriately labeled training contrasts. A mismatch is not a positive target.

Synthetic requests, shared content choices, translations, speech and execution schedules need versioned generation instructions, model identity and verification records. Target-derived content choices are generated targets, not supplied observations. Do not duplicate identical source features for every task direction; reference checked shared assets. Re-extract when the available prefix differs, since whole-clip features can contain future information.

### Make the prepared examples express the complete decision sequence

Extend the existing episode format rather than encoding lifecycle only in unparsed prose. Each example must carry stable task/track/event addresses, track descriptions and their revisions, speaker/direction information, observed or generated status, prerequisites, media clock intervals, readiness times, action/result links, source publication revisions and outcome verification.

Represent creation and cancellation as actual decisions even when they produce no user-facing text. Give structural decisions an explicit chronological representation; the present recorded-action handling cannot safely infer their positions from text-record counts. Each prepared example also declares which fields are public inputs, which are supervised choices, and which are loss-only annotations. Validate those distinctions before exporting data.

Train a structured decision sequence: choose the operation, choose existing tracks through an address-independent selection over eligible tracks or request new ones, choose output membership and modality, then predict content conditioned on the committed request. Use an explicit stopping decision for a variable number of requested outputs. The runtime reserves addresses without feeding arbitrary address spellings into semantic encoding. New descriptions and output specifications must be present before the content prediction that uses them.

Record masks for legal actions and modalities using only currently available capabilities and state. Supervised choices must be legal under the recorded mask. Multiple modalities may be valid for an unconstrained request; use explicit constraints and valid-choice sets where known rather than teaching every alternate choice as wrong. Include examples where the model chooses its modality, rather than always announcing the answer modality in the prompt.

The compiler must produce reconstructible state before every decision and preserve the exact order of observations, selection, rendering completion, delivery, tool execution and source publication. Observations and generated responses remain separate causal groups. Extend dependency validation to structural and nontext events. Failed and cancelled tasks retain their identity and chronology; success labels require the existing independent verification standard.

Add executable conversion specifications for every reader: source fields, joins, split identity, timing conversion, supported task views, public-input rules, loss masks and rejection reasons. Snapshot a few actual admitted examples for inspection. A missing field must produce an explicit rejection or a simpler justified task, never fabricated parallel supervision.

### Publish and load complete prepared data

Extend [prepared storage](../src/megatoken/data/storage.py), [family deduplication](../src/megatoken/data/dedup.py), [preparation sharding](../src/megatoken/data/media_sharding.py) and the cloud preparation jobs together. New readers must be usable in the existing pilot and full-volume launch paths, with restartable work and checked asset publication.

Prepared releases must include the frozen family split, original-source inventory, compiler and schema identities, task selection settings, channel/timeline transforms, structured-event summaries, decoder requirements, source collections and all referenced tensors. Validate checksums, missing assets, collection references and event prerequisites across roots before declaring a release complete. Publishing a table without its media or index is a failed release.

Extend supply summaries beyond text tokens: count examples by behavior, modality combinations, group width, lifecycle decision, language, speaker, real/synthetic provenance and readiness pattern. Report original-family counts separately from task-view counts so many derivatives do not masquerade as broad coverage. Update the existing volume/census tooling and published/restored caches to retain this information and enforce source-specific redistribution rules.

Training startup must check that every declared recipe supply exists for training and that every required behavior has held-out coverage. Existing startup rejects an empty named training stream, but held-out evaluation only requires some validation data overall. Strengthen this for the required new behaviors. Missing data or decoders must fail before allocating a long training run; do not silently substitute another stream or disable an objective.

### Extend the learner and all loss accounting

Keep the existing [training engine](../src/megatoken/training/engine.py), [grouped learner](../src/megatoken/training/meridians.py) and shared optimizer. Install new trainable parts before parameter enumeration and checkpoint restore. Required changes include:

- Train creation, assignment, wait/continue, completion and cancellation choices; eligible-track selection; event membership; modality classification; and any predicted duration or output geometry. Supervise description content through the normal text path. Distinguish model choices from scheduler actions and external arrival times, which are recorded conditions rather than fabricated policy targets.
- Replace numbered-slot inputs in MoE and learned-adjustment routing with the semantic description plus appropriate content, role and modality information. Update global expert-balancing replay, which currently also groups work using bounded track slots. Preserve one owner for shared weights and make gradient flow to the new semantic projections testable.
- Encode descriptions under current model weights. A cache may be reused during frozen inference, but a detached cached vector must not silently remove a training gradient. During joint training recompute trainable encodings within each optimization transaction, or explicitly freeze the encoder and train the documented projection. Rebuild after a weight update or description revision.
- Extend generated-media and relation losses for corresponding chunks, observed evidence and earlier generated events. Use exact validity masks. Semantic similarity, literal speech agreement, translation adequacy and timing are distinct requirements; no single pooled similarity term substitutes for them.
- Train waiting and interruptions using source availability and playback state. Score stopping speech and revising an answer only where the model could know the interruption. Track what has already been delivered so training cannot reward retracting heard content.
- Keep reconstruction and teacher encodings in their proper auxiliary or target branches. Never expose a current gold peer representation to prospective generation. Preserve exact recorded calls when later environment observations depend on those calls.
- Specify a sum and denominator for every objective in the [shared loss vocabulary](../src/megatoken/contracts.py) and [loss ledger](../src/megatoken/training/losses.py). Count modality decisions per event, lifecycle decisions per choice, declared relation pairs per pair, and rendering losses per event over valid coordinates. Do not let long speech tensors or many worker outputs silently dominate other tasks.
- Extend planning counts, accumulation, distributed reduction, logging and held-out scoring together. Include zero-local-count ranks, padded slots and structural-only windows. Every planned scored term must match the actually accumulated term, and each new trainable part must receive a finite gradient on a representative admitted example.

Use the existing modality bridge's restricted training to establish input/output adaptation, then train the new control decisions and semantic routing during joint task training. Update parameter-freezing rules and distributed gradient partitions explicitly; adding a classifier to the model does not ensure that the selected training stage updates it. Verify this with a short real training run and weight changes, followed by generated behavior.

### Change the curriculum and sampling policy

The [single packaged training recipe](../src/megatoken/data/stages.json) remains authoritative. Update its consumers, training examples and census in the same delivery. Some compiler task families are not named in the current recipe, and the long-context mixture currently has no explicit media task allocation. Existing recipe names alone therefore do not guarantee the requested coverage or its retention.

| Training part | Required exposure and changes |
| --- | --- |
| Representation bootstrap | Retain text and codec foundations. Initialize new semantic projections and decision heads through the normal model construction; do not claim them trained by a text-only objective |
| Modality bridge | Keep paired media reconstruction and verified input/output adaptation. Preserve channel-separated preparation and compatible decoder targets. Check exactly which parameters are allowed to update |
| Broad joint training | Add explicit tracked supply for lifecycle/assignment, modality choice, same-track peer events, parallel tools, audiovisual understanding and generation, chunkwise translation, full duplex, and live retrieval/ingestion alongside existing text/code tasks |
| Code and verified-execution specialization | Train actual delegated repository work and compatible parallel tools. Retain speech, concordance and retrieval examples so specialization does not erase them |
| Long-context training | Include long-lived workers, completed-track history, delayed replies, long recordings, translation backlog and late evidence. Extend dependency windows for nontext and structural decisions; retain basic mixed-modality performance |
| Verified instruction and success training | Require exact task constraints, correct tool effects, grounded answers, appropriate modality and useful scheduling. Verify temporal behavior as well as final text |
| Student correction and replay | Collect the student's own dynamic-track, media and asynchronous executions. Correct the actual student state; replay successful complete groups with exact structural decisions and rendered effects |

For selection, preserve the existing stream mixture mechanism and add named supplies for missing behaviors where source-level weights need independent control. Within large supplies, stratify by the behavior and provenance fields above. Avoid a second, untracked mixture inside a reader. Freeze stage weights, within-supply weights, per-family repetition caps and corpus identities as one versioned recipe before launching production.

Choose numerical weights from admitted supply, preparation cost and a bounded pilot comparison, then commit them to that recipe. The required pilot compares the current serial training baseline, new semantic routing, learned modality choices, and the complete dynamic/concordant mixture. Choose using held-out task success, concordance, useful response delay and training stability. A pilot recipe may be small; a production recipe is incomplete until its actual weights, event and decision budgets, supported language pairs and decoder identities are filled in and its supply check passes. There is no automatic redistribution of an unavailable required behavior's share.

Track text exposure, media events and structural policy decisions separately. Existing late-stage budgets measure free lexical positions; structural-only choices and waits can contribute no such positions. Add explicit decision coverage and bounded training-progress accounting so these examples neither vanish from sampling nor create endless zero-progress training. Do not relabel decisions as text tokens to inflate existing budgets. State which counter advances the schedule and which counters enforce minimum behavior exposure; checkpoint all of them.

Set group and window limits from actual prepared examples and measured memory, including content features, media targets, descriptions, rendering conditions and retrieval. If a group exceeds a limit, reject it during preparation or recompile it into a causally valid smaller task. Do not split an atomic group inside the trainer. Define behavior when dependency spans exceed the gradient window: reconstruct the causal prefix, score only the current decisions, and separately schedule examples that train the needed long-range relationship.

### Collect and replay actual model behavior

Extend the [student adapter](../src/megatoken/data/student.py), [collection protocol and durable ledger](../src/megatoken/data/collection.py), [recorded training history](../src/megatoken/training/history.py) and grouped action tapes together. This is a required downstream change: the current student requests one text output on one track, and the recorded-history validator explicitly accepts only single-event text tapes. The ability to replay a group in the model does not make the present collection-to-training route support grouped media execution.

Replace the text-cursor-only assumption with a mapping from every selected group to its structural choices, member events and optional physical payloads. Include valid empty output, invalid decoding, lifecycle-only decisions, failed rendering, provisional work, cancellation and partial delivery without dropping their chronology or policy probability. Do not invent dummy text outputs to fit the old route.

Store the behavior model and policy version; legal-choice masks and categorical probabilities; continuous content probabilities; event membership; descriptions and revisions; raw selected representations; rendering seeds and decoder identities; committed history revision; actual dispatch/results; and source publication and playback records. Reconstruct the same state and supported capabilities before rescoring. Store external times and scheduler decisions for replay, without treating them as choices sampled by the model.

Corrections must be evaluated in the student's actual environment snapshot. A replacement call cannot inherit the original call's result. Verify new effects or use the original committed action for continuation. Keep unsuccessful attempts as labeled recovery training, while masking unsafe imitation targets; do not promote every failed action to a positive example.

For outcome learning, include the full probability of structural choices and content, with explicit credit for task success, wasted work, timing and duplicated effects. Independent readiness schedules are conditions, not policy probability factors. The current repository has no complete integrated multitrack outcome-learning route; implementing and verifying that route remains required if claiming outcome learning for these behaviors. Supervised student correction alone does not establish it.

### Update distributed training and restart behavior

Update ordinary multi-process training and DiLoCo, including model broadcasts, optimizer ownership, gradient partitions, global expert balancing, synchronization/export, source collections and new decision counters. Every worker must construct the same architecture and declared trainable parts. Idle ranks must contribute zero loss counts without deadlock, and variable group width must not change effective objective weights.

Check restart in the middle of an epoch and at stage handoff. Save model/optimizer state, exact source and sampling cursors, behavior exposure, random generators, frozen recipe identities, decoder identities and memory-training state. No uncommitted external call may be silently rerun from a saved training collection state. Extend the collection ledger's execution checks separately from ordinary optimizer restart.

The new addressing, grouped media history and structural policy change artifact meaning. Bump the affected model, prepared-data, tape, checkpoint and served formats together. Follow the project's strict compatibility approach: regenerate prepared episodes and collected trajectories, rebuild model-bound knowledge bases, and retrain for the changed architecture. Do not claim an exact resume from old slot-based routing or old single-event tapes. Preserve reusable original media and compatible content-addressed feature assets, with explicit identity checks, rather than downloading everything again.

### Carry the changes into serving and evaluation

The exported bundle must include the semantic routing projections, modality and structure heads, supported capabilities, lifecycle protocol version, preparation/chunk policy and required decoder identities. Update loading, one-shot generation, JSONL serving, collection adapters and evaluation together. Provide constrained output requests and model-selected modality requests, including several peer events on one track. Expose track purpose, lifecycle and per-event modality without requiring the application to reconstruct private model state.

Separate host receipt of raw audio/video, causal feature extraction, model selection, rendering and playback. Bounded asynchronous work must permit input ingestion while output renders. Verify whether the installed teacher and decoders can meet the selected chunk and latency requirements on real hardware; use shorter valid chunks and incremental transport where supported. A non-streaming decoder cannot be advertised as full duplex merely because audio events exist in the training format.

Run both held-out objective scoring and generated multi-step execution through the served runtime. Add the new objectives to the existing evaluator; do not rely on its current retrieval-only diagnostic aggregation to report lifecycle, modality or timing performance. Generated evaluation must use the model's own track creation, output choices and shared content choices. Teacher-forced decisions are a separate diagnostic.

Require regression coverage for ordinary text/code, constrained tools, media reconstruction, memory ingestion and source retrieval. Required adversarial cases include new addresses beyond the old limit, identical descriptions on distinct tracks, renamed addresses, same-track multimodal peers, future-prefix leakage, late and duplicate tool results, source publication during speech, cancellation during rendering, and resumable interruption. Apply the existing grouped attention/replay, data preparation, student, retrieval, media, distributed and checkpoint suites, extending them for each changed contract.

An end-to-end training acceptance run must perform source preparation, published-data restore, sampler selection, a short optimization run, checkpoint/restart, distributed reduction, export and generated serving for each required behavior. A later real-weight training run must establish quality and accelerator performance on the held-out evaluation. Archive the exact recipe, admitted inventory, preparation reports, training counts, checkpoint identities and generated execution records with the results.

### Concrete integration changes

The locations below identify the current paths to extend. Each row is required where affected; it is not satisfied by a parallel unused implementation.

| Pipeline responsibility | Current integration points | Required resulting artifact or behavior |
| --- | --- | --- |
| Source selection and acquisition | [data catalogs](../configs/data/README.md), [pinned source readers](../src/megatoken/data/sources.py) | Selected revisions, reviewed use, actual files, coverage and immutable family identities |
| Timed media reading and extraction | [open media readers](../src/megatoken/data/open_media.py), [media preparation](../src/megatoken/data/media_prepare.py), [media assets](../src/megatoken/data/media_assets.py) | Speaker-preserving source chunks, checked features, matching targets and timeline transforms |
| Episode compilation and validation | [schema](../src/megatoken/data/schema.py), [concordance compiler](../src/megatoken/data/concordance.py), [adapters](../src/megatoken/data/adapters.py), [review](../src/megatoken/data/review.py) | Valid lifecycle and mixed-modality events, complete prerequisite/public-input checks and explicit supervision |
| Family splits and quarantine | [deduplication](../src/megatoken/data/dedup.py), [lineage joins](../src/megatoken/data/lineage.py) | One split across all related derivatives, mirrors and shards |
| Preparation launches and assets | [media preparation entry point](../scripts/prepare_open_media.py), [sharding](../src/megatoken/data/media_sharding.py), [cloud preparation](../src/megatoken/cloud_prepare.py) | Restartable pilot and production jobs for every selected source family |
| Storage and restoration | [prepared storage](../src/megatoken/data/storage.py), [publication](../src/megatoken/publish.py), [cache restoration](../src/megatoken/cache.py) | Complete, versioned, checksummed roots and source collections |
| Retrieval task preparation | [collection-reading producer](../scripts/prepare_collection_reading.py), [source collection](../src/megatoken/retrieval/collection.py) | Split-specific evidence, publication schedules, positives/distractors and grounded targets |
| Canonical recipe and progress | [stage recipe](../src/megatoken/data/stages.json), [recipe loading](../src/megatoken/data/recipes.py), [training schedule](../src/megatoken/training/schedule.py), [counters](../src/megatoken/training/metrics.py) | Explicit behavior supply and weights, correctly defined budgets and no invisible structural decisions |
| Windowing and deterministic sampling | [packing](../src/megatoken/data/packing.py), [training stream](../src/megatoken/training/stream.py) | Atomic groups, preserved dependencies, balanced behavior exposure and exact restart |
| Semantic track conditioning | [grouped model](../src/megatoken/model/multitrack.py), [MoE routing](../src/megatoken/model/expert_pools.py), [learned expert adjustments](../src/megatoken/model/adapters.py), [memory ingestion](../src/megatoken/model/ingest.py) | Description-conditioned routing with stable addresses used for equality and attachment |
| Instructions and modality/structure selection | [controls](../src/megatoken/controls.py), [grouped model](../src/megatoken/model/multitrack.py), [model settings](../src/megatoken/model/config.py) | Trainable legal structural decisions and event-level modality choices, independent of track number |
| Losses and optimization | [shared objectives](../src/megatoken/contracts.py), [grouped learner](../src/megatoken/training/meridians.py), [engine](../src/megatoken/training/engine.py), [optimizer](../src/megatoken/training/optimizer.py), [ledger](../src/megatoken/training/losses.py) | All new parameters updated, correctly reduced sums/counts and checked gradient paths |
| Startup and decoder readiness | [training settings](../src/megatoken/training/config.py), [decoder preflight](../src/megatoken/training/media_dependencies.py), [pack verification](../scripts/verify_packs.py) | Required data and real decoder capabilities verified before training |
| Recorded execution and student correction | [collector](../src/megatoken/data/collection.py), [student adapter](../src/megatoken/data/student.py), [recorded history](../src/megatoken/training/history.py) | Full structured/grouped/media collection and replay of actual execution |
| Runtime and external effects | [session runtime](../src/megatoken/multitrack_runtime.py), [memory and source serving](../src/megatoken/transducer_service.py), [environment adapters](../src/megatoken/environments.py) | Working lifecycle, compatible concurrent tools, asynchronous ingestion and interruption |
| Distributed learning and export | [distributed training](../src/megatoken/training/distributed.py), [DiLoCo client](../src/megatoken/training/diloco_client.py), [DiLoCo export](../src/megatoken/training/diloco_export.py) | Consistent parameter sets, counters and synchronization under variable work |
| Checkpoint and served compatibility | [checkpoints](../src/megatoken/training/checkpoint.py), [loading](../src/megatoken/loading.py), [served export](../src/megatoken/served.py), [service](../src/megatoken/multitrack_service.py) | Explicit new-format rejection, reproducible restart, complete serving behavior |
| Objective and generated evaluation | [held-out scoring](../src/megatoken/training/heldout.py), [served evaluation](../src/megatoken/training/evaluation.py), [concordance evaluation](../src/megatoken/training/concordance_eval.py) | Per-behavior scoring and actual generated multimodal/parallel execution with source and timing correctness |

### Release requirements

Before a production launch, require a versioned source-selection record, complete frozen prepared roots, reviewed example samples, a filled numerical recipe, validated memory/decoder needs and a reproducible launch description. Before claiming training integration, require nonzero sampled and scored coverage for every mandatory behavior, verified updates to the new parameters, and checkpoint/restart evidence. Before claiming the system works, require generated held-out execution after export, including the combined full-duplex, tool and source-ingestion task.

Update the README launch instructions, data preparation runbook, architecture and track descriptions, media/training documentation, cloud examples and implementation status to the delivered behavior. Remove outdated claims that one track implies one modality, one group member, a numbered semantic slot or single-event student replay. Keep unfinished acquisition, quality and runtime requirements visible until their acceptance evidence exists.

## Additional parallelization experiments

| Experiment | Potential benefit | Required comparison |
| --- | --- | --- |
| Shared source preparation and retrieval across workers | Avoid repeatedly encoding or searching identical content | Same task success and provenance with less total preparation and search work |
| Deadline-aware grouping | Batch compatible ready work while protecting speech and interruption latency | Serial scheduling versus batching on identical arrival schedules; measure both throughput and response delay |
| Incremental worker publication | Let another task use a completed chunk before the entire worker finishes | Correctness under partial results and clear revision tracking |
| Independent candidate generation with later checking | Explore alternatives for code, reasoning or plans | Equal-compute comparison with a serial approach; evaluation must check the actual candidates |
| Speculative retrieval or computation | Hide predictable delays | Preserve cancellation and stale-result handling; no speculative external writes |
| Streaming source preparation | Make completed sections searchable while later sections arrive | No exposure of future content or half-published entries |
| Pipelined media preparation, generation and playback | Reduce idle time while preserving meaningful chunk sizes | Decoded continuity, interruption behavior, device memory and end-to-end delay |
| Grouping work across independent sessions on the accelerator | Improve hardware use when a single session has little ready work | Keep sessions' attention and state isolated; compare latency and throughput |

These are hypotheses to measure. Joint attention can grow more expensive as the number of simultaneous events increases; more tracks do not automatically mean faster execution. Maximize useful overlap on the task's critical path, not the number of tracks created.

## Implementation order and completion evidence

| Delivery | Required changes | Evidence required before declaring it complete |
| --- | --- | --- |
| Track identity and meaning | Remove slot-based routing tables, reuse encoded descriptions, separate stable addressing from resource limits, update memory ingestion | Create and retire tracks beyond the old limit; preserve results across renaming, handle remapping and late arrival; show semantic routing works without opaque-ID dependence |
| Event modality and addressing | Add learned modality and output-set decisions, allow peer events on one track, update probability accounting and replay | One continuing track emits different modalities and simultaneous corresponding peers; constrained choices stay valid and replay exactly |
| Worker lifecycle and scheduling | Model-requested creation, task dependencies, progress, cancellation, asynchronous rendering and fair readiness queues | Main work continues while another track waits; interruption and stale work behave correctly; deterministic schedule replay |
| Training inventory and conversion | Audit every source family, preserve dependency evidence, add true delegated and concurrent execution traces | Prepared and sampled coverage report; validation rejects causal leaks and call/result mismatches |
| Timed media compilation | Corresponding source chunks, translation prefixes, speaker channels and versioned clock transforms | Held-out examples demonstrate available-prefix safety, correct chunk links and preserved overlap |
| Tools and retrieval integration | Real concurrent adapters, atomic source publication, source readiness events and grounded continuations | Independent calls finish out of order without corruption; newly ingested evidence is used only after publication |
| Integrated training and evaluation | Train structure decisions and content together; evaluate actual generated continuations and decoded media | Measured benefit over serial execution at comparable quality and compute, with full-duplex voice/tool/source scenarios |

The source inventory, access checks and preparation of existing aligned sources can proceed alongside track addressing work. Runtime scheduling follows the addressing and event changes. Final trajectory collection uses that runtime, so training and serving share the same meaning of creation, readiness and completion. Reconcile the existing architecture, training and media status documents as each delivery lands.

Evaluate held-out tasks under fixed source splits and replayable arrival schedules. Set acceptance thresholds before looking at final evaluation results, using application requirements and a measured serial baseline. Report these quantities with their dataset, hardware, denominators and quality comparison:

- Task success: the fraction of held-out tasks completed according to executable or independently checked criteria; higher is better.
- End-to-end completion time and time until the first useful response: elapsed time for the same tasks and hardware; lower is better at preserved success.
- Interruption delay: time from a valid interruption to stopped or revised playback; lower is better, with already delivered speech accounted for.
- Compute and memory: accelerator time, peak memory and external work per successfully completed task; lower is better at matched quality.
- Useful parallel exposure: the share of sampled training events participating in justified multi-event work, separated by task family; interpret alongside success rather than maximizing it alone.
- Concordance: decoded speech versus transcript agreement, translation adequacy, audiovisual event offsets, speaker consistency and human judgments on held-out corresponding content; report each measure separately.
- Retrieval: required-source recall, attribution correctness, answer success and delay until newly arrived evidence becomes usable; report sources missed or unavailable at answer time.
- Causal and replay correctness: no future-source access, incorrect result attachment, duplicate execution or unrecorded structural choice in the validation scenarios; any violation leaves that capability incomplete.

The current environment's Python does not have the model training dependency installed, so this review does not supply a new executed test result. Existing tests document intended grouped attention, replay, concurrent helper behavior, concordance and retrieval checks; implementation work must run them in the supported environment and add the lifecycle, modality and timed-scheduling checks above. Media quality and accelerator performance require real-weight runs, not test doubles.
