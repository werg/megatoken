# Concordant multi-track tasks, data and curriculum

**Status:** Roadmap for the concordant task catalogue, data acquisition and curriculum. Proposed mixtures, acquisition targets and objectives below are not yet measured model capabilities; none of the data has been acquired and no training result is implied.

**Scope:** Make prompted, concordant generation and sequential editing central to training, on top of the single media path described in [media.md](media.md): pinned EmbeddingGemma2 hidden states in, frozen pretrained realizers (Pocket, Sana, Sana-Video) out, everything crossing the 32-facet interface. This page does not define the media components, packs or preparation; media.md is authoritative for them. The grouped learner is extended, not replaced by another semantic model or training system.

The contract used throughout is **R decodes selected raw Z directly; C supplies context for subsequent P predictions**. Earlier contextual-realization examples (`B = C_final`) in other planning documents must not be copied into new implementation work.

## 1. Decisions and non-goals

| Responsibility | Decision | Boundary |
| --- | --- | --- |
| Media input | Pinned EmbeddingGemma2 512-wide hidden states -> `MediaEncoder` -> shared E reductions | The only route; no raw PCM/pixel E, no per-modality stems or alternative feature sources. See [media.md](media.md) |
| Text input | Always bytes through byte E | Teacher text features serve only as alignment anchors |
| Speech realization | Frozen Pocket (flow LM + Mimi, 24 kHz latents) with a trainable `FacetCondition` | Loss on the donor's own codec latents |
| Still-image realization | Frozen Sana + DC-AE with a trainable `FacetCondition` | Preserve the pretrained generative prior |
| Video realization | Frozen Sana-Video + its VAE with a trainable `FacetCondition` | Do not swap in a newer codec under unchanged generator weights |
| Semantic representation | **32 x 512 raw facets per event**; retain compact width | No global facet increase; use additional ordinary events selectively |
| Realizer content interface | **One selected raw 32-facet block per event** | No independent transcript, prompt, source-latent, reference-media, C-feature, or historical-KV bypass |
| Controls | One prompt-compatible system with explicit per-track/per-event requests | Resolve before P selects the output; preserve exact request revisions in replay |
| Training unit | **One aligned source bundle, compiled into several causal task views** | Understanding, verbalization, translation, concordant generation, and edits share lineage and splits |
| Optimization | Existing grouped learner and loss ledger, driven by a task recipe | Reconstruction is an auxiliary; integrated causal tasks become the majority |

No new per-modality reasoning transformer, learned low-level scheduler, correlated-latent subsystem, universal acoustic/visual engram table, or mandatory exact-media side channel is needed for this plan. Audio frames and video patches remain internal E/R work, not smaller central megatokens.

## 2. Current implementation and the remaining work

Implemented in code: joint meridians with shared full-resolution KV, grouped text/media training through the engine, checksummed tensor assets with codec-latent targets, pinned-teacher media E, resident Pocket/Sana/Sana-Video realizers, separate media-loss denominators, a `G` event budget, the aligned-bundle compiler (`data/concordance.py`, `scripts/compile_aligned_bundles.py`), and one data-mix mechanism (the stage plan, `src/megatoken/data/stages.json`). The architecture, packs, preparation, phases and serving contract are in [media.md](media.md); [architecture.md](architecture.md) and the [grouped model](../src/megatoken/model/multitrack.py) describe the core.

Not established: any media-quality, perceptual or GPU-scale training result; rights-cleared media inventories; measured gates per phase. The remaining work this document tracks is data and evaluation:

- Acquire and admit the datasets of Sections 14-15 and prepare them with the media preparer.
- Resolve real voice/style references and time-local per-event prompts upstream of selection.
- Compile aligned bundles into grouped tasks and versioned edit trajectories at scale.
- Add relation losses and evaluation of sampled concordant outputs.
- Measure the evaluations of Section 19 on real weights.

This document changes no runtime version, model code, stage budget, downloaded weights, or dataset inventory.

## 3. Precise semantics: E/P select Z; R reads Z; C serves the future

Distinguish three representations:

| Symbol | Meaning |
| --- | --- |
| `F` | Ordered pinned EmbeddingGemma2 hidden states `[T, 512]` of a media event (media.md) |
| `Z` | Authoritative raw `[32,D]` facets produced by E or selected from P's Gaussian |
| `U` | Sequence/grid of codec latents in a specific realizer's coordinate system (Mimi, DC-AE, Sana-Video VAE) |

The paths are:

```text
Observed media -> pinned teacher features F -> MediaEncoder + E -> raw Z
                                           |
                                           +-> joint C -> final KV/history -> later P

Past shared history + effective track prompts + references
    -> joint P queries -> Gaussian selection of Z[tracks,32,D]
                              |
                              +-> R_modality(Z_event, structural_spec, noise)
                              |      -> codec latents U -> matching decoder -> media
                              |
                              +-> joint C -> final KV/history -> later P
```

`C_final` is not the current realizer's input. Each selected group receives one joint C integration and one final-KV publication, while realization depends directly on its raw blocks. The two branches need not duplicate computation, but concurrent hardware execution is an optimization to measure.

The ordinary media R interface is:

```python
realizer.realize(
    raw,        # [B,32,D], sole learned content condition
    spec,       # MediaOutputSpec: geometry, rates, requested/predicted limits
    generator,  # explicit policy-independent renderer randomness
)
```

The `FacetCondition` maps all 32 facets to the donor's conditioning width; there is no second narrower bottleneck. The original text/schema-copy contract can remain for typed/text outputs; it does not authorize a speech renderer to re-read the transcript.

P may attend across its live output queries. C may attend across already selected peers. Neither prospective P nor the raw-to-R branch may read encodings of withheld target peers. Current outputs do not become concordant merely because C will jointly process them afterward.

## 4. Concordance: shared meaning, not identical representations

Concordant outputs express the same intended content in appropriate forms: transcript and speech, translation text and speech, caption and image, video and soundtrack, or video and endpoint frames. They need not have equal lengths, codec coordinates, or semantic-facet tensors.

### Constrained concordance

For literal transcription, verbalization, grounded description, and translation, available evidence already constrains the content. Predict the requested outputs directly as one group when appropriate:

```text
source speech -> joint {source transcript, target-language text, target-language speech}
```

The target-language text is a peer target, not an unacknowledged input to its speech peer. Keep direct joint training; an ordered text-then-speech task is an additional useful view, not a replacement for it.

### Creative concordance

For underspecified creation, use an optional preceding **content-plan track**:

```text
request -> selected content plan
        -> joint {caption, video, audio, first frame, last frame}
```

A normal text or structured plan chooses scene entities, actions, wording, speaker, and event timing from available instructions. All subsequent outputs condition on this one committed choice through P. This uses existing tracks and recurrence, not a new latent coordinator or a planner required for every transcription.

Joint P attention coordinates predicted means, but the current diagonal Gaussian uses conditionally independent residual noise. With separate raw-only realizers, it does not guarantee correlated creative choices. A committed plan, consistent supervision, and relation objectives are the initial solution. Shared random seeds alone are not an alignment algorithm.

A plan derived from training targets must be supervised as an earlier generated output, never relabeled as an externally supplied prompt. Report teacher-forced-plan adaptation separately from generation with the model's own plan. At evaluation, generate it unless the task genuinely supplies it.

### Keep the interface honest

The default still dispatches each selected event to its own R. Separate audio/video codecs can run in parallel. Fine synchronization must be learned and measured; independently plausible media are not necessarily aligned.

A donor joint AV model may supply teacher targets. A future jointly coupled renderer must explicitly declare all raw blocks it consumes and its larger conditioning budget; it must not quietly reintroduce peer access into a supposedly single-event 32-facet test. Such a change is not a prerequisite for the initial curriculum.

## 5. Speech

Input is the common media path: speech audio becomes pinned EmbeddingGemma2 hidden states, then a `MediaEncoder` projection and the shared E reduction produce **one** `[32,D]` raw audio event ([media.md](media.md)). There is no separate speech-semantic stem, acoustic input view or per-modality feature selection to configure. A long recording is split into several events before extraction; no event may include audio unavailable at its decision boundary.

Output is the frozen Pocket realizer (Mimi latents at 24 kHz, 12.5 latent frames/second) with a trainable `FacetCondition` from the raw `Z[32,D]`; the donor's own flow objective is applied to real recordings' Mimi latents cached at preparation time. The 32 acoustic latent channels are unrelated to Megatoken's 32 facet slots.

Task-level consequences that remain this roadmap's concern:

- Keep real recordings prominent in targets; use Pocket-generated speech for controlled rereadings and counterfactuals, and verify transcripts, names, numbers and requested delivery. Requested style is not automatically a realized style label.
- Evaluate free rollouts from selected facets, not only teacher-forced next-frame losses. Framewise regression between unrelated stochastic readings with different timings is not meaningful.
- Only language/profile combinations verified for the pinned Pocket pack enter the initial generation curriculum; English/German is the initial priority, not a claim that one English checkpoint covers every language.
- Any other teacher's speech must be re-encoded through the Pocket codec before it becomes a target; never compare unrelated codec coordinates.

## 6. Visual components and optional capacity

Images and video enter exactly like audio: pinned EmbeddingGemma2 hidden states (frame sampling follows the teacher processor's own policy), then `MediaEncoder` and E. Output is the frozen Sana + DC-AE realizer for stills and the frozen Sana-Video + VAE realizer for video, each with a trainable `FacetCondition` ([media.md](media.md)). A newer VAE is not a compatible substitute merely because its tensors have similar dimensions: a realizer's identity is its pack and checkpoint digest. Anything the teacher features discard cannot be restored by R; there is no source-latent skip connection around Z.

### Do not increase the default facet budget

First improve features, reduction, conditioning, and supervision. If a dense document region or complex scene still requires more detail, encode a small number of additional ordinary 32-facet events. Available global/detail crops may share a meridian on fixed reusable tracks; long temporal segments follow the normal event sequence.

Report total events/facets per task. Five outputs contain 160 facets in aggregate, not one 32-facet representation. At current production KV geometry, a retained event uses 32 KiB, so a five-event group adds 160 KiB before other state/workspaces. More tracks also increase live attention and rendering work; shared parameters do not make them free.

Prefer 32 versus two semantically justified 32-facet events on a small difficult set before any global 64-facet experiment. Larger donors can offload generative complexity but cannot recover source-specific information discarded by the teacher features and E. Plausible reconstruction is not evidence of faithful information retention.

## 7. Aligned bundles: the unit of data acquisition

An aligned bundle is an immutable collection of related source assets and annotations. A task compiler selects which members are inputs, which are outputs, and which relations to supervise. Do not curate a disconnected corpus for every task direction.

A speech bundle may contain:

```text
original recording + source sample spans
written text / normalized spoken form / verbatim transcript
word or phrase timing and language
speaker identity and eligible separate reference segments
checked translation text and derived target speech
controlled rereadings and subsequent edits
source family, artifact versions, and derivation records
```

An audiovisual bundle may contain:

```text
original synchronized video and audio + clip interval
visual caption / audio-event description / spoken transcript
sampled/cropped video view and its actual endpoint frames
optional translated narration, alternate styles, and edited versions
clock, fps/sample rate, crop/sample mapping, source identities
```

### Different annotation meanings stay different

- A transcript states what was said; it is not evidence that the described object is visible.
- A visual caption describes the scene; it need not match the narration.
- An audio-event description is not a speech transcript.
- A written token sequence and its normalized spoken form can differ legitimately.
- Source and translated speech correspond semantically but need not have equal duration or word order.

Retain `written_text`, `spoken_form`, `verbatim_transcript`, and `display_caption` separately. Requests such as "transcribe verbatim", "read exactly", and "normalize for display" select different contracts, avoiding contradictory labels for numbers, currency, abbreviations, and punctuation.

The bundle is storage and supervision, not an automatic model prompt. No target annotation enters P unless that specific task view declares it as an available input or an earlier selected event.

## 8. Task compiler and relation specification

A small deterministic compiler (`data/concordance.py`, driven by `scripts/compile_aligned_bundles.py`) sits beside the existing source adapters. It emits ordinary `Episode`, `TrackSpec`, `Meridian`, and `TrackEvent` records plus versioned supervision metadata. It is not a general workflow engine.

Each compiled task needs:

| Field | Meaning |
| --- | --- |
| Bundle/source family | Original recording/book/speaker/video/scene and derived ancestry |
| Task family and version | Transcription, verbalization, translation, captioning, joint generation, editing, etc. |
| Available artifacts | Exact observations and reference versions visible before each prediction |
| Track requests | Explicit effective prompts and revisions for all participating tracks |
| Output plans | Modalities, codec identities, masks, and requested/predicted structural limits |
| Alignment relations | Content correspondence, media clocks/spans, endpoint mapping, synchronization intent |
| Edit relations | Base version, requested changes, preservation constraints, and related outputs |
| Verification | Oracle/human/teacher provenance and quality checks, kept outside model inputs |
| Split and fingerprints | Family split, preprocessing, chunking, teacher, codec, and compiler identities |

Use a bounded relation vocabulary initially: `same_spoken_content`, `translation_of`, `describes`, `synchronized_with`, `first_frame_of`, `last_frame_of`, `edit_of`, and `preserves`. These describe loss/evaluation relationships; they do not grant attention access to future outputs.

Missing modalities are absent, not zero-length fabricated evidence. Compile only views supported by the actual bundle. Do not infer hidden voices, translations, masks, or edit order from a filename and call them ground truth.

## 9. Prompt every track, at the correct causal time

Every input track has an explicit interpretation/task description; every generated event has an explicit instruction. Defaults can be concise and reusable. A name such as `track_2` is an identity, not adequate task semantics.

Resolve one control source in this order:

```text
session defaults -> track defaults -> current event request
```

The result preserves ordinary instruction bytes, canonical typed settings where supplied, causal reference identities, and a request revision. No independent voice/video control tree is needed.

Illustrative schema, not current executable configuration:

```yaml
track_id: speech_en
modality: audio
instruction: >
  Translate the available German speech into English.
  Preserve names and numbers. Use the approved narrator voice, speaking calmly.
reference_event_ids: [narrator_reference]
request_revision: 3
effective_from_meridian: 12
alignment_group: segment_12
```

For a joint audiovisual request, use distinct prompts such as:

```text
caption: Describe the selected scene and event order, without inventing dialogue.
video: Realize the selected scene, retaining object identities and motion.
audio: Produce sound matching the selected events; do not add speech.
first_frame: Render the opening frame of this video view.
last_frame: Render the final frame of this video view.
```

Compile shared constraints once and bind them into each effective request; do not copy a gold caption into all prompts and call it joint caption generation. Style prompts apply to the appropriate tracks and shared content constraints apply across the group.

### Implementation rules

Resolve one payload per event (`controls.resolve_event_control`; one episode payload is never broadcast to all tracks), used identically in preparation, training, collection, serving, and tape replay. Store the exact effective payload/revision, not only a template ID. Input interpretation prompts live in available context; E remains the local media encoder.

A later edit instruction becomes available only at its edit turn. Do not place all future requests into the initial episode control payload. Reference media must be resolved and encoded upstream of P; a SHA256 string alone is not an acoustic/style embedding.

`OutputSpec` carries structural settings, not hidden content. Duration/geometry provenance must distinguish `requested`, `predicted`, and `teacher_only`. Target-derived duration or endpoint information may guide a labeled training branch but not silently become public free-generation conditioning.

Track IDs remain categorical. Renaming physical array positions must not change task meaning when descriptors, identities, and randomness mappings are preserved. Keep reusable track names across edits rather than allocating an unbounded new track for each artifact version.

## 10. Required task catalogue

| Family | Available inputs | Predicted outputs and core check |
| --- | --- | --- |
| Direct transcription | Speech + verbatim transcription prompt | Transcript; optionally separately prompted normalized transcript. Check WER/CER, names, numbers. |
| Direct verbalization | Written text + voice/language/delivery request | Pocket speech; optionally spoken-form text as a joint peer. Check exact content and delivery. |
| Speech translation | Source speech + target-language request | **Joint target text and target speech**, optionally source transcript. Neither target conditions the other's same-step prediction. |
| Image description | Image + description request | Grounded description and optionally requested attributes/counts. |
| Concordant image creation | Creation request and optional selected content plan | **Image and description together**, both scored as outputs. |
| Video/audio understanding | Available clip/audio + task-specific prompts | Visual caption, transcript, and/or audio-event description, without conflating them. |
| Concordant audiovisual creation | Request and optional selected content plan | **Caption, video, and separate audio track**. Caption agreement alone is not a synchronization score. |
| Endpoint-concordant creation | Same creative inputs | **Video plus independently generated first/last images**, optionally caption/audio. |
| Endpoint-conditioned video | Observed first/last frame or earlier clip + request | Interpolated/continued video and corresponding audio/caption. This is distinct from joint endpoint prediction. |
| Voice/style counterfactual | Same content with a changed causal reference/request | Different voice/style while preserving semantic content. |
| Prompted image edit | Earlier image version + change/preserve request | Edited image plus a change description; optional updated caption. |
| Prompted speech edit | Earlier recording/text + change/preserve request | Corrected words, translated speech, or changed delivery under the same speaker/content constraints. |
| Prompted video/AV edit | Earlier clip/audio versions + edit request | Updated media and relevant captions/endpoint frames; preserve unaffected content. |
| Cross-track evidence/tool task | Media/evidence plus actual independent tool results | Grounded answer/action requiring multiple inputs, with correct call/result attribution. |

Generate reciprocal understanding and generation views where legitimate, but keep their causal partitions explicit. An image may be an observed input in one view and a withheld target in another. One recording supports ASR, TTS, translation, and editing views without becoming four independent source recordings.

## 11. Concrete meridian schedules

### Literal speech and direct concordant verbalization

```text
g0 observed: text to speak + separately available voice reference/request
g1 generated together:
  spoken_form: render the normalized words requested
  speech: verbalize the original text using the requested voice and delivery
```

The speech R reads only its selected Z. The spoken-form peer is supervision and another output, not a decoder side input. An ordered text-then-speech view remains useful for testing exact content transfer through the shared history.

### Direct voice-in, voice-out translation

```text
g0 observed: available source-audio segment A + track requests
g1 generated together:
  source_transcript_A
  translated_text_A
  translated_speech_A

source-audio B may arrive while A is rendered or played
next observed meridian: source-audio B
next generated meridian: translated_text_B + translated_speech_B
```

This is chunk-scale simultaneous operation, not a claim of word-level latency. Keep direct joint translation in the curriculum rather than only running external ASR -> translation -> TTS as a pipeline. Such a pipeline is a baseline or teacher, not the native model path.

### Joint caption/video/audio/endpoints

```text
g0 observed: creation request + any genuinely supplied references
g1 generated: optional content plan, selecting underspecified scene choices
g2 generated together:
  caption
  video
  audio
  first_frame
  last_frame
```

Five outputs contribute five ordinary megatokens. Verify configured track and event capacities; do not silently drop one target. A supplied endpoint instead belongs in the observed prefix for the separately labeled endpoint-conditioned task.

### Sequential editing

```text
g0 observed: original media version v0
g1 observed: edit request #1
g2 generated: edited media v1 + change description + affected related outputs
g3 observed: edit request #2, referring to v1
g4 generated: edited media v2 + change description + affected related outputs
```

Future requests and target artifacts stay outside preceding prompts. Retain original source and intermediate versions in the dataset lineage even when only a bounded prefix is loaded into the model.

## 12. Endpoint frames and audiovisual timing

Create first/last-frame targets **after** the video target's final temporal sampling and spatial crop/resize. Record the exact displayed frame indices, crop mapping, and valid interval. Codec temporal padding is not the last visible frame.

Image/video decoders can have different resolutions and latent spaces. Compare decoded frames after a declared common transformation, using pixel/perceptual or semantic features as appropriate. Do not compare arbitrary image/video codec coordinates directly.

Keep three evaluations separate:

1. Independently generated endpoint images agree with the sampled video's decoded endpoints.
2. Given endpoints actually constrain generated video.
3. Video itself contains correct motion and temporal order, rather than simply repeating an easy endpoint.

Audio and video share a content clock where the task requires synchronization. Track natural speech duration separately from fixed clip length. Distinguish simultaneous interpretation, where translated speech can lag and differ in length, from dubbing, where fitting an explicit interval is part of the request. Never time-stretch translated speech or copy source timestamps without recording the transformation and validating its quality.

Fine sound-event and lip timing are explicit relation targets. A common caption, a common seed, or one shared KV history is not enough to establish physical synchronization.

## 13. Editing as a versioned trajectory

Each edit records `base_artifact_version`, `change`, `preserve`, and `related_outputs`. Start with two-to-four-turn sequences. Include no-op requests, reversions, minimal content changes, voice-only/style-only edits, and cross-modal changes requiring synchronized updates.

Examples:

- Change a vehicle's color while preserving motion, camera, scene identity, and soundtrack.
- Translate a spoken line while preserving visual content and the requested speaker identity.
- Correct one number without changing the rest of the utterance.
- Make delivery warmer while preserving the exact spoken words.
- Change scene lighting and update its caption while leaving the described event unchanged.

An independent edit pair cannot be chained after an unrelated previous edit just because both share a source ID. Sequence order must correspond to actual artifact versions. Generate and verify additional intermediate states when constructing a new chain.

### Recorded versus model-generated continuation

For recorded chains, later targets refer to the recorded earlier media. Continue from that actual earlier artifact's native group tape when supported, otherwise a clearly labeled detached E approximation. Do not continue from an unrelated sampled artifact while supervising an edit intended for the gold one.

For student rollouts, ask the teacher/verifier to edit or evaluate the artifact the student actually produced. Keep the same principle already used for executed tool results. Validate the full group when recorded peers jointly determine later state; do not fabricate independent native tapes for an originally grouped decision.

Under the strict bottleneck, preservation initially means semantic identity and perceptual fidelity, not byte-exact copying of arbitrary unedited pixels. Masks can guide losses and can be declared inputs encoded upstream; they do not grant the media renderer direct access to the source image/video. Report native bottleneck edits separately from any external source-conditioned baseline.

## 14. Dataset acquisition shortlist

Acquire original paired media and derive many task views, rather than maximizing unrelated caption rows. Counts below are publisher-reported inventory, not assets already acquired by this project. The first six sources are the initial shortlist, with optional research expansions kept separate.

| Source | Useful inventory | Initial task views and acquisition notes |
| --- | --- | --- |
| **HiFiTTS-2** [D1] | Metadata for about 36.7K hours / 5K speakers; a 31.7K-hour higher-bandwidth subset | Real speech/transcript pairs, codec targets, reference segments, controlled rereadings. Audio is obtained separately from LibriVox; retain chapter offsets, bandwidth, text source, and published splits. Card: CC BY 4.0. |
| **Multilingual LibriSpeech** [D2] | Audiobook speech in eight languages, including English and German | Multilingual ASR/verbalization and long-chunk construction from genuine adjacent material. Not a parallel translation corpus. Publisher: CC BY 4.0. |
| **CVSS** [D3] | 21 source languages into English, with canonical-voice and voice-transferred synthetic target speech | Direct speech translation with matching normalized target text. Official source pairing requires Common Voice v4. Preserve original joins and train/dev/test splits; check all source terms. Published CVSS license: CC BY 4.0. |
| **DOCCI** [D4] | 9,647 training image/description pairs with detailed human descriptions | Grounded description, image/description joint generation, contrasting attributes. Protect the 5,000-image test split and qualitative evaluation sets. Card: CC BY 4.0. |
| **FineVideo** [D5] | 43,751 videos, about 3,425 hours, with time-coded transcripts and structured metadata | Original A/V clip bundles, captions, event grounding, endpoint extraction, continuation. Preserve source attribution and required removal/version updates. Original-media rights and generated-annotation quality need separate checks. |
| **MagicBrush** [D6] | 8,807 training edit turns in 4,512 sessions | Sequential image editing, change descriptions, and preservation targets. Retain session order and COCO/source ancestry. Use training only; protect dev/test. Card: CC BY 4.0, with source-image terms retained. |

HiFiTTS-2, MLS, LibriTTS-derived material, and teacher outputs can overlap. Do not add headline hours without cross-source deduplication. FineVideo's metadata is not wholly human ground truth; recheck annotations used for fine timing, spoken content, or edit supervision. [D1, D5]

### Optional expansion, not a dependency of the first run

| Candidate | Why relevant | Admission policy |
| --- | --- | --- |
| CVSS-X [D7] | English-to-28-language synthetic translation; complements CVSS direction | Current synthetic translation/speech pairs are CC BY-NC 4.0. Use only in an approved research manifest; source-version alignment and family overlaps require verification. Do not assume its planned permissive successor already exists. |
| CoinVE-200K [D8] | Compositional video edits and corresponding key-frame edits | Review exact dataset terms and upstream videos separately from MIT code. Quarantine its benchmark and derived siblings. Useful for video/still edit concordance. |
| JAVEdit-100k [D9] | Joint audio/visual editing triplets, including speech and scene changes | Card combines MIT labeling with academic/educational restrictions and upstream-source terms. Keep out of the release-compatible pool until clarified. Preserve automated-generation provenance. |

Maintain one release-compatible approved manifest and a separate research-only expansion manifest. This is ordinary source metadata, not a new runtime policy subsystem. A published license label or model distillation does not erase source-media or teacher-weight obligations.

### Fill specific gaps with controlled data

Prioritize approved English/German source speech with checked bidirectional translations and Pocket-rendered target speech in a verified supported profile. Record translation-annotation rights and quality independently of the speech renderer. Keep the target codec fixed.

Add procedurally rendered scenes with object IDs, event order, physical sound timing, and exact frames; controlled Pocket rereadings; and verified media-edit chains. These are high-value sources of matched counterfactuals. Do not substitute tones/bars for evidence of natural speech/video quality, or replace all natural recordings with teacher samples.

## 15. Inventory targets, quality tiers, and source-family splits

### First frozen manifest targets

| Inventory | Initial acquisition target |
| --- | ---: |
| Quality-filtered real English/German speech | 2,000 unique source hours |
| Checked speech-translation pairs, both directions where available | 50,000 source pairs |
| Detailed image/description examples | All eligible DOCCI training pairs |
| Sequential image-edit sessions | All eligible MagicBrush training sessions |
| Aligned audiovisual clips from approved original recordings | 50,000 clips |
| Controlled speech/style and audiovisual edit sequences | 20,000 verified sequences |

These are pilot targets, not ceilings, budget authorizations, or claims of downloaded inventory. Expand successful source pools while preserving quality and diversity. Count original source hours/scenes, derived task views, and accepted teacher outputs separately. A clip reused ten ways supplies ten supervision views but not ten independent world observations.

### Admission pipeline

1. Pin source revision/terms, original asset identity, source family, and published split.
2. Preserve original audio/video masters; record deterministic sample/crop/resampling transforms.
3. Deduplicate by ancestry and cross-media evidence: recording/video IDs, transcript similarity, speaker/source identity, and perceptual fingerprints.
4. Validate language, timing, speaker attribution, captions, continuity, and usable bandwidth/resolution.
5. Freeze family-level splits before crops, translations, reference snippets, alternate styles, or counterfactual siblings are emitted.
6. Compile only task views supported by the bundle, with explicit available-input and target partitions.
7. Cache frozen F and U under exact identities; never cache mutable learned E facets as immutable source features.
8. Record quality/verification tier, teacher revision, requested controls, realized checks, and rejection reasons.

Use human/oracle-verified examples for high-confidence concordance, quality-checked original pairs for scale, and bounded weak/synthetic data where the objective tolerates it. Teacher confidence is not proof of correctness. Keep benchmarks and all derivatives quarantined across source pools.

Family joins include original recording/book/video/scene and derived artifacts. Speaker-held-out splits are separate explicit tests of voice generalization. Where source families connect conflicting published train/test memberships, quarantine the conflict rather than silently relaxing the split. Repeated crops must not leak across partitions.

Prompt templates and track names should not disclose class labels. Test held-out wording, entity combinations, speaker/reference pairs, and edit compositions as well as held-out raw assets.

## 16. Curriculum and sampling recipe

Use the existing engine, prepared cursor, optimizer, DiLoCo coordinator and distributed loss ledger. Data mixes are **one mechanism**: the stage plan (`src/megatoken/data/stages.json`) names prepared streams and the share of *events* each stage draws from them; [media.md](media.md) shows where media sits in it. A stage transition is a new DiLoCo coordinator initialized from the previous stage's export.

| Stage | Trainable | Media streams |
| --- | --- | --- |
| MB media bridge (after the codec bootstrap) | `media_bridge` (media projection, modality embedding, realizer conditions) | text 0.40; audio, image and video identity 0.20 each |
| P1 base pretraining | all | 15% identities and concordant task families, 3% long-document reading |
| M1, M2, LC, S | all | small media replay and long-document reading shares |

The shipped shares are the starting point, not tuned values. Keep every stream a prepared dataset really contains; a stage naming an absent stream is an error rather than a silent relabeling.

### Bridge: establish the shared modality interface

Identity episodes ("reproduce this image") and text anchors teach the bridge to land where byte E puts text, while only the bridge and the realizer conditions move. It runs right after the codec bootstrap, when E is trained but C/P are young, and only briefly: do not freeze random C/P indefinitely and expect adapters to create a reasoning model.

### Base pretraining: media and concordance alongside text

Text dominates; identity streams keep every modality's E -> C -> R path exercised while all owners adapt, and the concordant task families that `data/concordance.py` compiles into their own streams (transcription, verbalization, speech translation, image description, image creation, audiovisual creation, and image/speech/audiovisual editing) make concordance an objective from the start. Aim, over a sufficiently large sampling window, for these properties of the media share:

- **Every** participating track has a resolved prompt/interpretation descriptor.
- **At least one-third** of tasks have two or more genuinely joint generated outputs.
- **About one-quarter** require information from multiple input tracks.
- **About one-quarter** include matched counterfactual or preservation variants.

These properties overlap; they are not extra mixture rows to sum. Sampling cannot claim a quota when the corresponding modality/targets are unavailable. Early modality-limited runs record an explicit recipe variant and its coverage rather than silently relabeling serial text as AV generation.

Shares are events, not equal GPU-time or token shares. `max_media_events_per_window` caps media work per window and existing accumulation applies. Log the realized selected events, task counts, seconds/frames, and compute per family; a video event is not computationally equivalent to text.

Reconstruction and distillation remain auxiliary losses (`auxiliary_weight`: 1.0, 0.5, 0.35 across the three shipped recipes). Reduce their relative weight as the interface stabilizes, while retaining enough to prevent E/R drift.

### Later work: coherent trajectories and outcome learning

After phase 3 works, increase edit-chain length, source-prefix variation, delayed evidence, student-generated content plans, actual tool continuations, and translation backlog, keeping direct literal tasks as anchors for pronunciation, exact strings and descriptions. Start edits at two-to-four turns and extend only after recorded/model-generated continuation is correctly represented. Policy-side rewards on executed/verified tasks come only after grouped replay is available; renderer preference adaptation is a distinct owner/phase, not a hidden change to the latent-policy action map. No outcome-RL phase is required to validate the first concordant supervised recipe.

## 17. Loss definitions and KL units

### Three separately reported computations

```text
codec baseline:       x -> codec encoder -> U -> decoder -> x_hat
bottleneck auxiliary: x -> F (teacher) -> E -> noisy Z -> R -> U_hat -> decoder
causal generation:    available history/prompts -> P -> sampled Z -> R -> output
```

C processes selected events for subsequent predictions. It is not part of the current reconstruction or current R path. Alternate reconstruction branches do not write hypothetical history into the live trajectory.

The auxiliary is VAE-like through noisy encoded facets and prior correspondence. With detached targets, native flow objectives, and separate optimization owners, it is not automatically a jointly optimized likelihood-calibrated ELBO.

### Native media objectives

Each realizer uses its donor's own objective on the donor's own codec latents: flow matching for Sana and Sana-Video, Pocket's flow/LSD loss for speech (media.md). Noisy target latents and teacher-forced acoustic prefixes are local loss inputs, not information available to E/P or semantic KV. Evaluate complete rollouts from base noise, since target-conditioned denoising can otherwise conceal ignored facet conditioning.

Keep modality-specific losses normalized over valid coordinates (codec padding masked by `valid_count`) and separately metered (`audio_*`, `visual_*`). Text uses lexical/structure losses, not pixel MSE. Latent regression is appropriate for aligned targets, not for averaging unrelated valid speech/video trajectories.

### KL units

`DiagonalGaussian.kl()` in [generation/latents.py](../src/megatoken/generation/latents.py) returns a **per-example sum over latent coordinates**, not an elementwise mean. The numerator/denominator convention shared with the production ledger is a summed detached KL divided by the active coordinate count (padding excluded), so `facet_kl` stays a per-coordinate quantity. Test any new reduction against an analytic Gaussian example and duplicated active events: summed KL and count scale together, leaving per-coordinate KL unchanged. A test that repeats the implementation's reduction is not independent verification.

Bootstrap uses this correspondence; ordinary recurrent supervision retains the existing no-unrelated-E-to-prior-regression policy. Normalize distributed losses using globally aggregated numerators/counts, not a mean of unequal rank-local means.

### Concordance relations

Add a bounded relation-loss family with explicit valid-pair counts:

| Relation | Appropriate supervision/evaluation |
| --- | --- |
| Speech/transcript | Spoken-content agreement under the requested normalization policy |
| Translation/source | Meaning, names/numbers, omissions/additions, language, and coverage |
| Image/description | Objects, attributes, counts, spatial relationships |
| Video/caption | Events and temporal ordering |
| Audio/video | Sound-source timing and applicable speaker/lip correspondence |
| Endpoint/video | Decoded endpoint perceptual/semantic agreement after declared transforms |
| Edit/base | Requested change and preservation of unaffected content/identity |
| Voice/style/request | Requested acoustic/visual attributes while content remains stable |

Do not force the full Z of speech to equal its transcript Z or a video Z to equal one frame. Use suitable semantic projections, differentiable output features, or supervised counterpart tasks. Nondifferentiable ASR/human/verifier scores are evaluation/filtering or explicit later reward signals, not gradients obtained by labeling them a loss.

Train and measure agreement on actual sampled peers, not predicted output versus a ground-truth peer secretly injected into P. Expensive differentiable decoding/perceptual comparisons may run on a sampled subset; most media work can use cached latent targets. Normalize relation counts independently, so a five-output group does not accidentally receive ten times the intended weight through all possible pairs.

## 18. Gradient ownership and recurrence

Keep one owner per shared weight. Share across modalities inside E, C, P, R, and control families before tying across families with incompatible training semantics. The existing pair-expert P bank and row-expert U bank are not interchangeable dimensions. No preset allocation of modality-specific expert counts is required now.

Bootstrap teacher-forces clean target encodings only for the subsequent live history, after the current prior was predicted without them. In ordinary recurrent continuation, keep the sampled raw group; do not re-encode its decoded output into replacement state.

For the E-only recurrent auxiliary, freeze R and its conditioning adapters while allowing input gradients back to E. C need not be executed merely to obtain that reconstruction. Detach reference/control branches from this auxiliary when their ownership requires it. Preserve the independent clipping domain and prevent frozen donor updates.

Recorded tool and edit continuations require the actual earlier action/artifact, as specified above. New supervised target branches cannot inherit outcomes from unexecuted alternatives.

For latent-only RL, initially keep E/C/R/codecs fixed for consistent state/replay semantics; train P, its upstream control injection, and Gaussian parameters. Controls can remain trainable when they affect only selection, not R for a fixed raw action. Renderer noise is a fixed realization distribution, not an additional unrecorded learned policy. A change in renderer weights/sampler requires a new materialization identity.

## 19. Counterfactuals and held-out evaluation

A multi-track task should demonstrably depend on its claimed inputs or coordinate multiple meaningful outputs. Merely splitting one prompt into tagged tracks does not establish that.

Construct matched variants that change only one source or request:

- Swap the audio instruction while keeping the visual scene fixed, and vice versa.
- Change voice/style while preserving the requested words/objects/actions.
- Reverse a relevant action order while preserving scene appearance.
- Change one source number while preserving all other evidence.
- Change an edit instruction while keeping its base artifact fixed.
- Deliver independent tool results in different orders without changing their facts.

Keep counterfactual siblings in one source family. Score the changed component and preservation of the unchanged component. A model that always produces a safe generic caption should not pass a content-specific concordance test.

Maintain four comparable pack evaluations: native donor, codec round trip, 32-facet E/R auxiliary, and causal P/R output. These locate codec loss, E compression loss, R adaptation failure, and P prediction failure separately.

Required sampled-output measurements include WER/CER and exact entity fidelity; speaker/style adherence; translation adequacy plus actual latency; image counts/spatial/OCR tasks; video event order; AV timing; endpoint agreement; and edit preservation across successive versions. Include a modest human-scored subset and evaluators not identical to the teacher.

Protect source-level tests, speaker-held-out tests, prompt/template generalization, and unseen edit compositions. Do not evaluate free-generation with gold future duration, gold content plan, complete future encoder features, target endpoints, or a teacher acoustic prefix unless that input is explicitly part of the task.

## 20. Chunking, clocks, and translation availability

Retain the reviewed text policy of 512 target / 768 soft / 1,024 hard reference tokens. Use roughly 256 transcript-equivalent tokens as the initial full speech target, permitting larger coherent units where reliable. Natural short utterances remain underfilled. Do not splice unrelated audiobook sentences to manufacture continuous conversation.

Teacher windows, Pocket acoustic frames, codec tiles, and network packets are not semantic commits. For video understanding, start with coherent event/shot segments; generation initially follows the chosen donor's tested clip geometry. Longer allowed tensors do not imply learned long-video competence.

For simultaneous translation, record the source prefix available at each decision, source-to-target semantic alignment, target speech's own timing, untranslated backlog, and any explicit wait. The teacher encoder, which is bidirectional inside an available chunk, must never see a later source chunk for an earlier decision.

Physical media time remains separate from logical meridian time. Core RoPE continues to index causal meridians; local media positions describe the content. Rate conversion and trimming must be versioned and deterministic. Waiting for a large source chunk creates real acquisition delay, even if the core and renderer are fast.

## 21. Assets, identities and loading

Pack manifest, locking, preparation and assets are specified in [media.md](media.md); this section lists what the dataset roadmap must preserve. Preserve original raw media separately from prepared assets. Image/video masks must distinguish valid data from codec padding, and Pocket audio is 24 kHz throughout.

Each feature/codec/realizer identity includes:

```text
upstream revision and weight digest
preprocessing policy; sample/fps clock; channels/layout
latent convention; scale/shift or mean/std
crop/temporal mapping, masks, and valid spans
realizer sampler/seed convention
source family, artifact version, reference provenance, and terms
```

Cache frozen teacher features and codec targets, not learned final facets. Prepared datasets stay immutable; changed extraction/segmentation/teacher policies create new derived identities. For sources with removal obligations, retire affected prepared versions and rebuild approved manifests rather than ignoring updates in the name of reproducibility.

Frozen donors still consume memory and compute. Report core, trainable conditions, resident frozen donors, and optimizer/workspace memory separately. Distill a donor only after it performs the facet-conditioned task satisfactorily.

## 22. Replay, speculation, and multi-turn delivery

A group tape records all raw selected blocks, effective prompt revisions, event/track identities, realizer identity/spec/codec versions, and renderer randomness. Rendering replays directly from raw Z. Joint C is replayed as necessary to rebuild future shared history, not to supply current R inputs.

Derive noise streams from stable event identities so realizer installation order cannot reassign randomness. Exact raw replay, supported-backend latent replay, and cross-backend perceptual equivalence are separate guarantees.

Speculation keeps a copy-on-write candidate group until acceptance. New relevant input or a changed request invalidates undelivered work. Already played speech, displayed frames, or executed calls cannot be erased by dropping the branch. Keep delivery cursors and interruption records without turning transport packets into additional semantic megatokens.

Selected group publication/acceptance remains atomic. Even when current R is per-event, future shared C history depends on the complete selected group. Do not keep a different subset while claiming identical recurrence.

## 23. Implementation sequence and acceptance criteria

| Change | Repository area | Deliverable |
| --- | --- | --- |
| 1. Real-weight verification | `scripts/verify_packs.py`, media.md packs | Pocket/Sana/Sana-Video round trip, objective and rollout on an accelerator |
| 2. Prepared inventories | `scripts/prepare_media.py`, `prepare_media_identity.py` | Identity and text-anchor streams from rights-cleared media, immutable identities |
| 3. Bundle/task compiler at scale | `data/concordance.py`, `scripts/compile_aligned_bundles.py` | ASR/TTS, direct joint translation, image/description, AV/endpoints, and recorded edits with explicit input/target partitions |
| 4. Stage runs | `configs/train/media-bridge.example.yaml`, the stage plan | Media bridge, then base pretraining with media shares, chained by DiLoCo coordinator exports |
| 5. Relation objectives and evaluation | `training/concordance_eval.py`, `scripts/eval_omni_concordance.py`, loss ledger | Sampled peer agreement, endpoint and edit-preservation checks |
| 6. Interactive/long trajectories | Grouped recurrence, collectors, runtime | Prefix-limited translation, real student edit continuations, actual parallel tools, safe speculation |

Do not defer all joint-task work until expensive video generation is ready. Text/speech and procedural visual pairs can establish prompt/causality/lineage semantics first; the same compiler then handles pretrained media latents.

Required regression tests:

- Distinct same-meridian prompts change the intended output conditioning; later prompt revisions cannot leak backward.
- R receives raw Z only, with no original transcript/reference/source-latent path.
- Withheld target peers do not affect the current prior or unrelated branch state.
- Every generated group publishes complete KV once and replays its future history correctly.
- Joint prediction remains invariant to storage permutation with identities and RNG assignments preserved.
- Endpoint target indices match the actual prepared video's displayed interval.
- Translation encoders see only available source prefixes; target timing is not copied blindly.
- Recorded edit continuation uses the actual base version; no-op/revert chains retain ancestry.
- Counterfactual siblings and all bundle views stay in the same split family.
- KL, modality, and relation numerator/count reductions agree across batch sizes and ranks.
- Frozen-owner gradients and realizer reinstall/normalization parity are checked.

Acceptance is task-based and not yet measured: intelligible facets-only Pocket speech; correct direct translation; meaningful joint descriptions and media; measured endpoint/AV agreement; and preservation through edits. No benchmark score is invented as a claim of current performance. Only reopen component choices or facet size after identifying a concrete failure of the chosen path.

## 24. References and evidence boundaries

The dataset shortlist was checked against the primary pages below for this revision. Links are not immutable production lockfiles. Pin actual revisions, terms, and decoder contracts before acquisition or training. No model benchmarks, weight downloads, paid generation, or dataset acquisition were performed by this documentation change.

### Realizer components (pinned through the pack lock, see media.md)

- **[P1]** [Pocket configuration](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/english.yaml): native sample/frame rates and latent geometry; verify each selected language/profile.
- **[P2]** [Pocket Mimi implementation](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/mimi.py): continuous `encode_to_latent` / `decode_from_latent` and padding/state conventions.
- **[P3]** [Pocket flow model](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/flow_lm.py): conditioning and latent normalization.
- **[P4]** [Pocket training recipe](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) and [weight card](https://huggingface.co/kyutai/pocket-tts): training/distillation and artifact conditions.
- **[I1]** [Sana 600M image checkpoint](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers).
- **[G1]** [SANA-Video implementation and checkpoint contracts](https://github.com/NVlabs/Sana/blob/main/docs/sana_video.md).

### Data sources

- **[D1]** [NVIDIA HiFiTTS-2](https://huggingface.co/datasets/nvidia/hifitts-2): metadata, original-download procedure, bandwidth, splits, and terms.
- **[D2]** [Multilingual LibriSpeech publisher page](https://www.openslr.org/94/): languages, downloads, and license.
- **[D3]** [CVSS official repository](https://github.com/google-research-datasets/cvss): source-version joins, target variants, normalized text, splits, and license.
- **[D4]** [DOCCI](https://huggingface.co/datasets/google/docci): detailed descriptions, official training/evaluation splits, and license.
- **[D5]** [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo): original media, generated annotations, source attribution, and removal/update terms.
- **[D6]** [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush): ordered edit sessions, masks, splits, and terms.
- **[D7]** [CVSS-X official repository](https://github.com/Ermisai/CVSS-X): reverse-direction synthetic speech, generation provenance, current terms, and planned changes clearly distinguished.
- **[D8]** [CoinVE-200K](https://github.com/coinve200k/CoinVE-200K): compositional video edits, key-frame additions, source ancestry, and separate benchmark.
- **[D9]** [JAVEdit-100k](https://huggingface.co/datasets/Coraxor/JAVEdit-100k): paired audiovisual edits, automated-generation provenance, and source/use restrictions.

### Repository anchors

[Architecture](architecture.md), [media](media.md), [core](../src/megatoken/model/core.py), [grouped model](../src/megatoken/model/multitrack.py), [Gaussian definitions](../src/megatoken/generation/latents.py), [media model](../src/megatoken/model/media.py), [controls](../src/megatoken/controls.py), [asset schema](../src/megatoken/data/media_assets.py), [meridian learner](../src/megatoken/training/meridians.py), [stage plan](../src/megatoken/data/stages.json), [concordance compiler](../src/megatoken/data/concordance.py), and [broader multi-track plan](multi-track-plan.md).

## 25. Final direction

**Pinned EmbeddingGemma2 features supply perception; frozen Pocket, Sana and Sana-Video supply rendering priors. Megatoken supplies the shared selected representations, causal decisions, prompts, and cross-track coordination.**

The primary learning unit is a prompted aligned bundle, not a disconnected modality sample. Train direct transcription and verbalization, voice-to-voice translation, image/description pairs, caption/audio/video/endpoints, and successive media edits through the same raw 32-facet interface, in the bridge, joint and task phases. Keep reconstruction and distillation as bridges into that integrated capability, not substitutes for it.
