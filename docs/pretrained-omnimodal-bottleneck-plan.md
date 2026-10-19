# Pretrained omnimodal components and concordant multi-track training

**Status:** Agreed development direction and implementation plan. Proposed schemas, mixtures, acquisition targets, and objectives below are not yet executable production configurations or measured model capabilities.

**Revision:** 2, 2026-10-19.

**Reviewed repository baseline:** `main` at `011e3603132337ea6104f6a5cdaee8b68cb53ac3`, including media-training integration and PR #15's direct raw-facet realization in architecture v8.

**Scope:** Keep Pocket TTS as the speech realization family; preserve the selected pretrained perception/visual components; make prompted, concordant generation and sequential editing central to training. All content crosses the existing 32-facet interface. Extend the current grouped learner rather than introducing another semantic model or training system.

This revision supersedes this document's earlier `B = C_final` realization rule and its earlier training allocation. It also narrows the alternatives in [omnimodal-architecture-and-training.md](omnimodal-architecture-and-training.md). The authoritative contract here is **v8: R decodes selected raw Z directly; C supplies context for subsequent P predictions**. Earlier contextual-realization examples in other planning documents must not be copied into new implementation work.

## 1. Decisions and non-goals

| Responsibility | Selected direction | Boundary |
| --- | --- | --- |
| Speech understanding | Whistle encoder embeddings | Frozen speech-semantic stem; existing Wav2Vec2 path is a fallback, not another required resident model |
| Acoustic input and reference features | The exact continuous Mimi variant packaged with Pocket TTS | Optional acoustic view fused before the same 32-facet E readout |
| Speech realization | **Pocket-derived generator plus its matching Mimi decoder** | Initialize from trained components, adapt to facets-only conditioning, then distill |
| Ming TTS | **No required integration or acquisition dependency now** | Do not replace Pocket or add a second production speech codec; a future approved offline teacher would not change the native interface |
| Image/video understanding | MiniCPM-V 4.7 visual tower and trained mergers | Experiment B is preferred; 16x normally and the supported 4x detail profile when explicitly requested |
| Visual-ingestion fallback | Existing SigLIP2 bridge | Switch only for a concrete extraction, memory, or required-quality problem |
| Still-image realization | Sana 0.6B with its matching DC-AE | Preserve the pretrained generative prior |
| Initial video realization | SANA-Video 2B 720p with the exact LTX-2 VAE required by that checkpoint | Do not swap in a newer codec under unchanged generator weights |
| General audiovisual extension | A matched LTX audiovisual component family | Optional later work for soundtrack/effects and paired AV; not a replacement for Pocket speech |
| Semantic representation | **32 x 512 raw facets per event**; retain compact width | No global facet increase; use additional ordinary events selectively |
| Realizer content interface | **One selected raw 32-facet block per event** | No independent transcript, prompt, source-latent, reference-media, C-feature, or historical-KV bypass |
| Controls | One prompt-compatible system with explicit per-track/per-event requests | Resolve before P selects the output; preserve exact request revisions in replay |
| Training unit | **One aligned source bundle, compiled into several causal task views** | Understanding, verbalization, translation, concordant generation, and edits share lineage and splits |
| Optimization | Existing grouped bootstrap/recurrent learner and loss ledger | Reconstruction is an auxiliary; integrated causal tasks become the majority |

No new per-modality reasoning transformer, learned low-level scheduler, correlated-latent subsystem, universal acoustic/visual engram table, or mandatory exact-media side channel is needed for this plan. Audio frames and video patches remain internal E/R work, not smaller central megatokens.

## 2. Current implementation and the remaining work

The reviewed repository contains joint meridians, shared full-resolution KV, grouped text/media supervised training, checksummed tensor assets, E-owned external-feature projections, separate media-loss denominators, a `G` event budget, and procedural counterfactual examples. The merged v8 path decodes raw facets rather than raw/early/final contextual features. See [architecture.md](architecture.md), [media pipeline](omni-media-pipeline.md), and [grouped model](../src/megatoken/model/multitrack.py).

The next work is not another core redesign. It is:

- Replace the native waveform/pixel test heads with compatible pretrained latent realizers.
- Extend raw/feature assets to codec-latent targets, multiple input views, native sample rates, and temporal grids.
- Resolve real voice/style references and time-local per-event prompts upstream of selection.
- Compile aligned bundles into actual grouped tasks and versioned edit trajectories.
- Add relation losses and evaluation of sampled concordant outputs.
- Correct the standalone bootstrap KL normalization described in Section 17.

One concrete gap is [training/meridians.py](../src/megatoken/training/meridians.py): it currently constructs one episode payload and broadcasts `(payload,) * len(step.tracks)`. Distinct track names are not a substitute for distinct task prompts. Another is that current prepared media targets and debug heads still use native PCM/pixels rather than the selected pretrained codec spaces.

This document changes no runtime version, model code, live stage budget, downloaded weights, or dataset inventory.

## 3. Precise v8 semantics: E/P select Z; R reads Z; C serves the future

Distinguish three representations:

| Symbol | Meaning |
| --- | --- |
| `F` | Ordered perceptual features produced by a frozen media encoder |
| `Z` | Authoritative raw `[32,D]` facets produced by E or selected from P's Gaussian |
| `U` | Sequence/grid of acoustic or visual latents in a specific decoder's coordinate system |

The paths are:

```text
Observed media -> frozen features F -> E -> raw Z
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
realize(
    raw_facets,  # [B,32,D], sole learned content condition
    output_spec, # codec/renderer identity, geometry, rates, requested/predicted limits
    noise,       # explicit policy-independent renderer randomness, if needed
) -> latent_payload
```

Map all 32 facets to the donor's conditioning width initially. Do not silently introduce a second eight-slot bottleneck. The original text/schema-copy contract can remain for typed/text outputs; it does not authorize a speech renderer to re-read the transcript.

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

## 5. Speech components: one native codec family

### Ingestion

Use Whistle's frozen embeddings for speech meaning and optionally the exact Pocket-Mimi acoustic features for timbre/prosody or reconstruction. The previously selected Whistle profile uses 16 kHz input; the inspected Pocket configuration uses 24 kHz mono, 12.5 latent frames/second, and a 32-channel continuous acoustic representation. Verify these conventions per pinned checkpoint. [W1, P1, P2]

```text
original audio master
  -> declared 16 kHz view -> Whistle semantic features
  -> declared Pocket-rate view -> Pocket acoustic features when enabled
  -> timestamp-aware projections, view identifiers, and shared E reduction
  -> ONE [32,D] raw audio event
```

Equal nominal frame rates do not imply equal frame origins or receptive fields. Preserve padding, offsets, masks, window boundaries, and the deterministic overlap/trimming policy. Long inputs may require several encoder windows but still form one semantic event. During prefix-limited tasks, no window may include audio unavailable at the decision boundary.

Use an explicit missing-view mask and view-dropout training. Choose the profile from the task and available inputs, not from hidden future targets. A speech-recognition stem need not preserve arbitrary music or environmental sounds; general-audio tasks require an appropriate acoustic input/decoder profile rather than mislabeled speech support.

### Realization and distillation

Pocket remains the selected native speech path. Its released code exposes continuous codec encoding/decoding, generator conditioning, normalization, and training/distillation machinery. The acoustic channel count of 32 is unrelated to Megatoken's 32 facet slots. [P1-P4]

1. Pin the Pocket generator, codec, normalization buffers, and supported language/profile.
2. Initialize the student latent generator/flow head from trained weights; keep the matching decoder frozen.
3. Project raw `Z[32,D]` to the original prefix-conditioning width, preserving structural BOS/EOS conventions.
4. Train the projection, then limited conditioning/backbone LoRA as necessary.
5. Remove the student's original text and voice-prefix content path from native execution. Such inputs may remain on the teacher side for target generation.
6. Distill depth or acoustic sampling steps only after full facets-only rollouts work.

Cache codec latents from real recordings and exact pre-decoder latent trajectories from Pocket-generated speech. Keep real recordings prominent. Verify teacher transcripts, names, numbers, and requested delivery; requested style is not automatically a realized style label.

Use the native acoustic flow objective and matched teacher trajectories. Framewise regression between unrelated stochastic readings with different timings is not meaningful. Evaluate free acoustic rollouts from selected facets, not only teacher-forced next-frame losses. Only language/profile combinations verified for the chosen components enter the initial generation curriculum; English/German is the initial priority, not a claim that one English checkpoint already covers every language.

If another teacher is approved later, re-encode its waveform through the native Pocket codec. Never compare unrelated teacher codec coordinates directly. **Ming is neither a runtime dependency nor a required first-round data-generation step.**

## 6. Visual components and optional capacity

Retain MiniCPM-V 4.7 visual extraction as experiment B: export its visual tower/mergers, normally use the selected 16x profile, and use its detail profile for explicit OCR/small-object tasks. Do not load the donor language model merely to extract patches. Verify an extracted tower against its parent at a pinned revision, and count its actual parameters/resident memory rather than using the whole VLM's name. [V1, V2]

Preserve spatial/crop coordinates, frame order, local time, and masks until E performs reduction. Where reconstruction or editing needs appearance details that semantic features discard, a matching codec-latent input view may enter E as well. It must still cross Z; no source-latent skip connection to R.

Keep pretrained generators paired with their exact codecs: Sana/DC-AE for stills and the selected SANA-Video/LTX-2 combination for initial video. A newer VAE is not a compatible substitute merely because its tensors have similar dimensions. Adapt donor conditioning to all 32 raw facets before further speed distillation. [I1, G1]

The later LTX audiovisual family is optional. Its separate codec components are useful for general sound/video work, but importing VAEs alone does not import a generative prior. Keep teachers and latent generators explicitly distinct from codec decoders. [G2]

### Do not increase the default facet budget

First improve features, reduction, conditioning, and supervision. If a dense document region or complex scene still requires more detail, encode a small number of additional ordinary 32-facet events. Available global/detail crops may share a meridian on fixed reusable tracks; long temporal segments follow the normal event sequence.

Report total events/facets per task. Five outputs contain 160 facets in aggregate, not one 32-facet representation. At current production KV geometry, a retained event uses 32 KiB, so a five-event group adds 160 KiB before other state/workspaces. More tracks also increase live attention and rendering work; shared parameters do not make them free.

Prefer 32 versus two semantically justified 32-facet events on a small difficult set before any global 64-facet experiment. Bigger edge models can offload perceptual/generative complexity but cannot recover source-specific information discarded by E. Plausible reconstruction is not evidence of faithful information retention.

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

Add a small deterministic compiler beside existing source adapters. It emits ordinary `Episode`, `TrackSpec`, `Meridian`, and `TrackEvent` records plus versioned supervision metadata. It is not a general workflow engine.

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

Replace the current episode-payload broadcast with one resolved payload per event, used identically in preparation, training, collection, serving, and tape replay. Store the exact effective payload/revision, not only a template ID. Input interpretation prompts live in available context; E remains the local media encoder.

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

Use the existing stage engine, prepared cursor, optimizer, and distributed loss ledger. Add a versioned concordant recipe once its readers and objectives exist. Do not paste these tables into the live text-token `stages.json` before implementation.

### Phase A: establish the shared modality interface

Pin codecs and measure round trips; train feature projections and facets-only Pocket/Sana/video conditioning. Keep codecs frozen. Begin with reconstruction and aligned teacher targets, while introducing direct ASR/TTS and simple conditional tasks from the start.

If the core already has useful skills, protect them with rehearsal and initially limited updates. If it is untrained, do not freeze random C/P indefinitely and expect adapters to create a reasoning model. Train the causal predictor as soon as its targets and realizers are usable.

### Phase B: concordance becomes a primary objective

Initial task-family selection weights after interface warm-up:

| Family | Share |
| --- | ---: |
| General text, code, tools, and existing skill replay | 20% |
| Direct transcription and verbalization | 20% |
| Speech-to-speech translation with text/speech agreement | 15% |
| Image understanding and concordant image/description generation | 15% |
| Joint caption/audio/video/endpoint generation | 15% |
| Sequential prompted editing across modalities | 15% |
| **Total** | **100%** |

These are probabilities for choosing task families at the sampler, not equal GPU-time or token shares. Cap media work per microbatch and use existing accumulation. Log the realized selected events, task counts, latent coordinates, seconds/frames, and compute per family; a video event is not computationally equivalent to text.

Within this mixture, target over a sufficiently large sampling window:

- **Every** participating track has a resolved prompt/interpretation descriptor.
- **At least one-third** of tasks have two or more genuinely joint generated outputs.
- **About one-quarter** require information from multiple input tracks.
- **About one-quarter** include matched counterfactual or preservation variants.

These properties overlap; they are not extra mixture rows to sum. Sampling cannot claim a quota when the corresponding modality/targets are unavailable. In early modality-limited runs, record an explicit recipe variant and its coverage rather than silently relabeling serial text as AV generation.

Reconstruction and distillation remain auxiliary losses inside appropriate batches; do not maintain the earlier separate 20% reconstruction family as an additional unaccounted allocation. Reduce their relative weight after the interface is stable, while retaining enough to prevent E/R drift.

### Phase C: coherent trajectories

Increase edit-chain length, source-prefix variation, delayed evidence, student-generated content plans, actual tool continuations, and translation backlog. Maintain direct literal tasks as anchors for pronunciation, exact strings, and descriptions.

Start edits at two-to-four turns and extend only after recorded/model-generated continuation is correctly represented. Use scene changes, voice/style swaps, and reordered independent tool arrivals as targeted perturbations, not arbitrary data noise.

### Phase D: selective outcome learning

Train policy-side rewards on executed/verified tasks only after grouped replay is available. Renderer preference adaptation is a distinct owner/phase, not a hidden change to the latent-policy action map. No outcome-RL phase is required to validate the first concordant supervised recipe.

## 17. Loss definitions and the KL correction

### Three separately reported computations

```text
codec baseline:       x -> codec encoder -> U -> decoder -> x_hat
bottleneck auxiliary: x -> F -> E -> noisy Z -> R -> U_hat -> decoder
causal generation:    available history/prompts -> P -> sampled Z -> R -> output
```

C processes selected events for subsequent predictions. It is not part of v8's current reconstruction or current R path. Alternate reconstruction branches do not write hypothetical history into the live trajectory.

The auxiliary is VAE-like through noisy encoded facets and prior correspondence. With detached targets, native flow objectives, and separate optimization owners, it is not automatically a jointly optimized likelihood-calibrated ELBO.

### Native media objectives

Use the donor's actual acoustic/visual prediction parameterization. For a newly defined continuous flow target, one explicit convention is:

```text
s ~ Uniform(0,1); epsilon ~ Normal(0,I)
U_s = (1-s) * epsilon + s * U
v_target = U - epsilon
L_flow = mean_valid_coordinates(||v_theta(U_s, s, Z, spec) - v_target||^2)
```

A donor with a different convention must retain or explicitly convert it. Noisy target latents and teacher-forced acoustic prefixes are local loss inputs, not information available to E/P or semantic KV. Evaluate complete rollouts from base noise, since target-conditioned denoising can otherwise conceal ignored facet conditioning.

Keep modality-specific losses normalized over valid coordinates and separately metered. Text uses lexical/structure losses, not pixel MSE. Latent regression is appropriate for aligned targets, not for averaging unrelated valid speech/video trajectories.

### Correct the standalone helper's units

The reviewed `DiagonalGaussian.kl()` in [generation/latents.py](../src/megatoken/generation/latents.py) returns a **per-example sum over latent coordinates**, not an elementwise mean. The earlier review's claim of double normalization was wrong.

Use a numerator/denominator convention shared with the production ledger:

```python
kl_sum = posterior.detach().kl(prior).sum()
kl_coordinate_count = target_raw.numel()  # active coordinates only; exclude padding
kl_per_coordinate = kl_sum / kl_coordinate_count
```

The standalone `supervised_group` helper currently omits this denominator following the earlier change; document it as a pending code correction, not a fixed issue. Update its comment and regression test. Test against an analytic Gaussian example and duplicated active events: summed KL/count should scale together, leaving per-coordinate KL unchanged. A test that repeats the implementation's reduction is not independent verification.

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

For the E-only recurrent auxiliary, freeze R and its conditioning adapters while allowing input gradients back to E. In v8, C need not be executed merely to obtain that reconstruction. Detach reference/control branches from this auxiliary when their ownership requires it. Preserve the independent clipping domain and prevent frozen donor updates.

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

Whistle windows, Pocket acoustic frames, codec tiles, and network packets are not semantic commits. For video understanding, start with coherent event/shot segments; generation initially follows the chosen donor's tested clip geometry. Longer allowed tensors do not imply learned long-video competence.

For simultaneous translation, record the source prefix available at each decision, source-to-target semantic alignment, target speech's own timing, untranslated backlog, and any explicit wait. An ASR/vision encoder used bidirectionally inside an available chunk must never see a later source chunk for an earlier decision.

Physical media time remains separate from logical meridian time. Core RoPE continues to index causal meridians; local media positions describe the content. Rate conversion and trimming must be versioned and deterministic. Waiting for a large source chunk creates real acquisition delay, even if the core and renderer are fast.

## 21. Assets, packs, and loading

Extend the existing media store with a distinct codec-latent target kind and, where needed, multiple declared input feature views. Preserve original raw media separately. Current 16 kHz raw assumptions must not silently reinterpret Pocket-rate audio, and image/video masks must distinguish valid data from codec padding.

Each feature/codec/renderer identity includes:

```text
upstream revision and weight digest; exported submodules
preprocessing and resampling; sample/fps clock; channels/layout
feature layer or latent convention; scale/shift or mean/std
coordinates, crop/temporal mapping, masks, and valid spans
renderer parameterization, scheduler, seed/state convention
source family, artifact version, reference provenance, and terms
```

Cache frozen features and codec targets, not learned final facets. Prepared datasets stay immutable; changed extraction/segmentation/teacher policies create new derived identities. For sources with removal obligations, retire affected prepared versions and rebuild approved manifests rather than ignoring updates in the name of reproducibility.

Use a small explicit pack registry, not a general neural-plugin framework. Components can be loaded/unloaded independently when no live computation depends on them. Cached-feature training need not load perception towers; speech understanding need not load a video generator. Trainable adapters retain stable optimizer owners across residency changes.

Frozen weights still consume memory and compute; optional loading does not reduce their intrinsic parameter count. Report core, trainable adapters, resident frozen packs, and optimizer/workspace memory separately. Distill a large pack only after it performs the bottleneck-conditioned task satisfactorily.

## 22. Replay, speculation, and multi-turn delivery

A group tape records all raw selected blocks, effective prompt revisions, event/track identities, pack/spec/codec versions, and renderer randomness. v8 rendering replays directly from raw Z. Joint C is replayed as necessary to rebuild future shared history, not to supply current R inputs.

Derive noise streams from stable event identities so pack loading order cannot reassign randomness. Exact raw replay, supported-backend latent replay, and cross-backend perceptual equivalence are separate guarantees.

Speculation keeps a copy-on-write candidate group until acceptance. New relevant input or a changed request invalidates undelivered work. Already played speech, displayed frames, or executed calls cannot be erased by dropping the branch. Keep delivery cursors and interruption records without turning transport packets into additional semantic megatokens.

Selected group publication/acceptance remains atomic. Even when current R is per-event, future shared C history depends on the complete selected group. Do not keep a different subset while claiming identical recurrence.

## 23. Implementation sequence and acceptance criteria

| Change | Repository area | Deliverable |
| --- | --- | --- |
| 1. Validate v8 and fix KL units | `model/multitrack.py`, `generation/latents.py`, focused tests | Raw-only R tests; analytic KL/count parity between helper and production ledger |
| 2. Effective per-event requests | `controls.py`, `data/schema.py`, `training/meridians.py`, serving/tapes | Different prompts on joint peers; time-local edits; training/serving/replay equivalence |
| 3. Bundle/task compiler | Existing data adapters/preparation/storage | ASR/TTS, direct joint translation, image/description, AV/endpoints, and recorded edits with explicit input/target partitions |
| 4. Codec-aware target assets | `data/media_assets.py`, `model/media.py`, target plans and export | F/Z/U distinction, native rates, masks, normalization, and immutable identities |
| 5. Pocket facets-only student | Media conditioning adapters and extraction scripts | Verified Pocket/Mimi round trip, real/teacher latent targets, free speech rollout from Z alone |
| 6. Visual adapters and realizers | MiniCPM extraction; Sana and matching SANA-Video packs | Grounded visual understanding and content-faithful generation through unchanged facets |
| 7. Relation objectives and recipe | `contracts.py`, target denominators, existing loss ledger/stage engine | Explicit task mixture, sampled peer agreement, endpoint and edit-preservation checks |
| 8. Interactive/long trajectories | Grouped recurrence, collectors, runtime | Prefix-limited translation, real student edit continuations, actual parallel tools, safe speculation |

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
- Frozen-owner gradients and pack reload/normalization parity are checked.

Acceptance is task-based: intelligible facets-only Pocket speech; correct direct translation; meaningful joint descriptions and media; measured endpoint/AV agreement; and preservation through edits. No benchmark score is invented as a claim of current performance. Only reopen component choices or facet size after identifying a concrete failure of the chosen path.

## 24. References and evidence boundaries

Model-family choices are retained from the prior component review; the new dataset shortlist was checked against the primary pages below for this revision. Links are not immutable production lockfiles. Pin actual revisions, terms, and decoder contracts before acquisition or training. No model benchmarks, weight downloads, paid generation, or dataset acquisition were performed by this documentation change.

### Selected components

- **[W1]** [Whistle model card](https://huggingface.co/Cactus-Compute/whistle) and [publisher description](https://www.cactuscompute.com/blog/whistle): selected speech-semantic feature source.
- **[P1]** [Pocket configuration](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/english.yaml): native sample/frame rates and latent geometry; verify each selected language/profile.
- **[P2]** [Pocket Mimi implementation](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/mimi.py): continuous `encode_to_latent` / `decode_from_latent` and padding/state conventions.
- **[P3]** [Pocket flow model](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/flow_lm.py): conditioning and latent normalization.
- **[P4]** [Pocket training recipe](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) and [weight card](https://huggingface.co/kyutai/pocket-tts): training/distillation and artifact conditions.
- **[V1]** [MiniCPM-V 4.7 feature documentation](https://huggingface.co/docs/transformers/model_doc/minicpmv4_7).
- **[V2]** [MiniCPM-V 4.7 publisher documentation](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_v4dot7_en.md).
- **[I1]** [Sana 600M image checkpoint](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers).
- **[G1]** [SANA-Video implementation and checkpoint contracts](https://github.com/NVlabs/Sana/blob/main/docs/sana_video.md).
- **[G2]** [LTX audiovisual component family](https://huggingface.co/Lightricks/LTX-2.5): optional paired-generation direction, not a silent codec upgrade.

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

[Architecture v8](architecture.md), [media pipeline](omni-media-pipeline.md), [core](../src/megatoken/model/core.py), [grouped model](../src/megatoken/model/multitrack.py), [Gaussian definitions](../src/megatoken/generation/latents.py), [media model](../src/megatoken/model/media.py), [controls](../src/megatoken/controls.py), [asset schema](../src/megatoken/data/media_assets.py), [meridian learner](../src/megatoken/training/meridians.py), and [broader multi-track plan](multi-track-plan.md).

## 25. Final direction

**Pocket supplies the initial speech competence. Pretrained visual components supply perception and rendering priors. Megatoken supplies the shared selected representations, causal decisions, prompts, and cross-track coordination.**

The primary learning unit is a prompted aligned bundle, not a disconnected modality sample. Train direct transcription and verbalization, voice-to-voice translation, image/description pairs, caption/audio/video/endpoints, and successive media edits through the same raw 32-facet interface. Keep reconstruction and distillation as bridges into that integrated capability, not substitutes for it.
