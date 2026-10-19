# Megatoken v7: omnimodal architecture, component reuse, and training recipe

**Status:** Proposed revised architecture/training plan, reviewed 2026-10-19. This is a design and acquisition plan, not a production-ready training configuration. It is anchored to the merged v7 multitrack implementation at `a441316f19ea6409ff39f4502bc56068f0d06ed2`. See [the multi-track implementation status](multi-track-implementation.md) and [the complete original plan](multi-track-plan.md).

## 1. Decision and actual state

**Keep one model of information-dense megatokens.** Every content-bearing event maps to **32 x D raw facets**, shares the same **C4/P12**, participates in a **joint same-meridian C pass**, and writes **32 full KV slots** into one session memory. Image patches, acoustic codec frames, and video tubelets stay **inside E/R**, not independent semantic timeline tracks.

The goal is not the highest acoustic frame rate. It is a coherent cross-modal model in which text, image, audio, video, evidence, workers, and tools interact under the same causal rules.

**Already implemented:** joint selected meridians, global final-KV, grouped latent Gaussian/tape/replay, modality-aware prediction, native STFT/3D-convolution E, raw waveform/pixel R, typed control payloads, speculation, JSONL group serving, and an independent grouped **bootstrap** training API.

**Implementation update (mixed-media training branch):** grouped B0 and recurrent meridians containing text, audio, image and video now use `training/engine.py` with checked native asset references, pinned pretrained feature projections and modality-specific loss units. Event-budget pilot training is opt-in through `budget_meter: G`, leaving existing token budgets unchanged. See [the runnable media pipeline](omni-media-pipeline.md). **Still pending:** production-scale rights-cleared media data, real high-fidelity pretrained output generators, reference-media embeddings for voice/style, grouped-native RL tapes, joint audiovisual alignment training and measured perceptual quality. **Do not describe native waveform/pixel MSE as state-of-the-art generation.**

**Immediate code corrections from this review:** the independent grouped-bootstrap facet KL was normalized twice; normalize it **once per active latent coordinate**. The media stem now exposes an explicit `encode_ordered_features(core, features, mask)` boundary for ordered, already projected `source_dim` features. Group action-tape JSON now uses the same strict duplicate-key and nonstandard-constant rejection as the existing serialized contracts. None of these changes downloads or validates a pretrained donor.

## 2. Minimal architectural changes to make pretrained components easy

Keep the E / C / P / R / control boundaries and the same parameter ownership. Use a **one-way feature bridge** into E and a **conditional latent-output bridge** out of R, rather than importing another general transformer history.

### Encoder path

```text
original waveform/image/video
   -> optional frozen pretrained encoder or VAE
   -> ordered local feature tokens (+ local coordinates/mask)
   -> small trainable projection to source_dim
   -> existing shared E reduction/readout/refinement
   -> 32 x D raw facets -> shared C and global KV
```

The new `MediaStems.encode_ordered_features` provides the final common reduction path. Next engineering addition is a small **registered projection adapter** per selected pretrained feature family, plus a manifest containing:
- `modality`, feature model name and immutable revision/hash, preprocessing and normalization fingerprints;
- the exact projected feature width, ordering, masks, stride/fps/sample rate, and timestamp convention;
- the frozen/unfrozen owner, the projector's optimizer/checkpoint owner, and whether cached features are allowed;
- source asset and lineage identifiers, with validation/test ancestry joins and media rights.

**Never mix feature tensors from two different encoder revisions or resampling policies under one silent identity.** Do not represent decoded pixels or PCM with the same fingerprint as pretrained latent features. Frozen encoders should run offline and cache ordered *features*, not the final 32 facets: E must remain trainable and able to learn the common space. Cache at an explicitly chosen intermediate layer that still preserves sequence/spatial locality.

Use the native STFT/Conv3D path as smoke/debug control, not the preferred quality path. For audio start with a pretrained **Whisper/ASR-family encoder or Mimi**; for images start with **SigLIP2 / MiniCPM-V visual stem**; for video use a **pretrained video/image frame feature extractor** and ordered spatiotemporal reduction. Run one encoder family per modality first, not six interchangeable backbones.

Avoid copying complete donor multimodal transformer contexts or K/V caches into Megatoken. Encode local media once; normalize/project; let Megatoken's own facets and KV handle semantic interaction. If high-quality media references are required later, pass them as explicit reference features through the unified control pathway and pin their source identities.

### Realizer path

```text
selected raw facets + explicit control + OutputSpec
   -> small Megatoken conditioning adapter (possibly cross-attention slots)
   -> frozen or LoRA-tuned pretrained acoustic / visual latent generator
   -> pinned decoder/vocoder/VAE
   -> waveform, image, or multi-frame clip
```

Start by **conditioning a pretrained renderer**, rather than retraining a large flow/diffusion model from scratch or fitting an MSE loss to raw pixels/PCM. Keep the current R baselines as tiny differentiable integration tests. Text R remains unchanged. Share the visual image/video family when practical; audio and visual output distributions need separate final heads.

Begin with a learned `32 x D -> K x D_renderer` context projection (e.g. K=8 or 16 reference slots) and LoRA/cross-attention at the donor's supported conditioning interface. If a donor only accepts text, use it initially for **teacher media and captions**, not pretend a prompt bridge trains direct facet-conditioned high-fidelity generation. Integration should ultimately allow conditioning on raw facet tensors without requiring a private autoregressive language model.

A whole megatoken may decode into many acoustic frames or video latents. This does **not** introduce a finer-grained shared semantic transformer. The realizer may take multiple flow/denoising steps; those stay local and do not write core KV. For synchronized audiovisual output, use one *paired* renderer or explicitly trained paired latent conditioning inside R. Two independent heads plus a common caption are not sufficient for tight lip synchronization.

**Randomness:** retain the grouped Gaussian for semantic selection. Renderer base noise/solver seeds are explicit versioned realization inputs. Do not claim exact cross-hardware pixel reproducibility without tested deterministic kernels. Store content hashes and media delivery cursors.

### One control system

Keep `ControlBundle` as the single prompt-compatible control source. Its current `ControlReference` entries are **asset identities and hashes**, not actual speaker or visual-style embeddings. A working reference path must **resolve real media**, run the corresponding frozen feature encoder, and provide bounded control/reference slots to R (and a pooled causal summary to P), with lineage/authorization recorded.

Use ordinary context instructions for most selection. Typed settings or user-supplied reference assets are precision controls, not parallel control subsystems. In latent-only RL, freeze any control feature encoder used by R, as well as R/C/E; train P-side injection and the learned Gaussian. Preserve separate supervised ownership.

### Timing and chunk density

The current production text policy is **512 target / 768 soft / 1,024 hard reference tokens**, not the earlier 256-token plan. Therefore start speech at **256–512 transcript-equivalent tokens per megatoken**, with underfilled natural utterance boundaries and an explicit media duration/resource cap. This is an experimental band, not a mandate to hold a short user utterance until a long buffer fills.

Video uses **coherent shot/event clips**, not one central token per frame. Start with duration- and shot-aware clips and compare 8/16/32 seconds; dynamically shorten complex scenes and extend stable scenes only when content tests support it. An internal video VAE can still process every required frame. Train temporal-order distinctions so long segments do not collapse to averaged keyframes.

Keep **media time** (clock, span, fps) separate from **causal meridian index**. Current C RoPE is indexed by logical meridian; spatial/time coordinates belong to local media feature extraction. Track identity is categorical, not an ordinal RoPE axis.

## 3. Recent public systems and what to reuse (verified 2026-10-19)

These are **component and recipe precedents, not drop-in Megatoken checkpoints**. Distinguish model-weight rights from source-code licenses, pretrained codec licenses, and source-media rights.

| System and date | Relevant result | Useful part for Megatoken | Reuse recommendation |
| --- | --- | --- | --- |
| [MiniCPM-V 4.7](https://github.com/OpenBMB/MiniCPM-V), **Oct 12, 2026** | Released compact on-device MoE vision family; [V4.6](https://github.com/OpenBMB/MiniCPM-V) earlier used mixed 4x/16x visual token compression | Compressed vision feature path and quality teacher; image/video feature tradeoff | **High** for visual encoding/distillation; pin exact family/revision |
| [vLLM-Omni 0.30](https://github.com/vllm-project/vllm-omni), **Sep 2026** | Sessions, cross-stage media/KV transfer, concurrent audio-video serving | Serving patterns for R output transfer and speculative branches | **Reference only**; do not adopt a second core runtime |
| [VeRL-Omni v1](https://github.com/verl-project/verl-omni), **Sep 2026** | Multimodal rollout/RL with transfer queues, replay buffers, async rewards | Reward pipeline and media-component LoRA training recipes | **Later**, for external renderer adaptation/rollout; core policy RL remains Megatoken's |
| [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3), **Aug 2026** | Joint text/image/video/audio context to native synchronized video + stereo audio; released H3-Base FL2VA/Ref2VA | Separate visual/audio VAEs, reference conditioning, generation teacher | **Promising** for VAE/teacher tests, but community-weight restrictions and model size need review |
| [NAVA](https://github.com/ernie-research/NAVA), **2026 NeurIPS** | Public alignment-then-fuse audiovisual diffusion training with timbre-in-context references | Audio/video alignment loss and paired reference conditioning | **Strong design reference** for coupled R, not a second semantic processor |
| [LTX-2.3](https://github.com/Lightricks/LTX-2), **Feb 2026**, [trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer) | Updated image/video VAE and joint audio/video generation and finetuning | Frozen VAE + LoRA-conditioned joint media output | **High** for first paired A/V R experiment; check community license |
| [Qwen3-ASR / ForcedAligner](https://qwen.ai/blog?id=qwen3asr), **2026** | Released 0.6B ASR and 0.6B multilingual forced aligner with Apache 2.0 framework/weights | Timestamped speech/transcript joins, long-utterance segmentation, teacher transcriptions | **Very high** for dataset preparation and speech alignment |
| [CosyVoice3](https://github.com/QwenAudio/CosyVoice), **2026** | Open multi-speaker/multilingual TTS inference and training tooling | Speech targets, voices, optional codec/latent interface | **High** as teacher and potential R backend, pinned rights |
| [InfoTok](https://proceedings.iclr.cc/paper_files/paper/2026/hash/432f048a844654ba981953491e6dc80e-Abstract-Conference.html), **ICLR 2026** | Adaptive video tokenization by content information density | Learn where temporal compression loses semantics | **Design inspiration**, do not adopt an extra tokenizer prematurely |
| [Dynin-Omni](https://github.com/AIDASLab/Dynin-Omni), **2026** | Open staged adaptation: video->text, ASR, TTS, then all-modality SFT/rehearsal | Multi-stage curriculum and replay to prevent text forgetting | **Strong recipe reference**; incompatible diffusion/AR cores shouldn't be transplanted |
| [InternVideo3](https://github.com/OpenGVLab/InternVideo), **Jun 2026** | Long-video model, SFT set and video-agent recipe | Video teacher/feature backbone and temporal QA data | **Candidate** if image-frame features lose long temporal ordering |

The key lesson of recent work is **frozen pretrained modality edges, progressive joint-task training, and explicit paired generation alignment**. No evidence justifies replacing shared C/P with a large third-party omni LM. Use the smallest donor that retains needed local information; measure quality per megatoken before unfreezing.

## 4. Data: maximize *usable* high-quality inventory

Data procurement is a continuous operation. Optimize for **verified, licensed, deduplicated, varied, causally correctly aligned** examples. Huge noisy video collections are not automatically better than smaller accurately timed, rights-cleared media.

Maintain a new source inventory **alongside**, not silently inside, the text-only `sources.v13.json`. The inventory should record identity, access terms, source metadata, parent clips/speakers, modality, exact crop spans, quality, rights, approval, checksum, preparation/teacher revisions and splits. Verify train/validation/test **by source family, video, speaker, and derived artifact**, not just by cropped file hash.

### Sources to acquire/evaluate

| Priority | Sources | Task value | Rights and curation considerations |
| --- | --- | --- | --- |
| A | [LibriTTS-R](https://www.openslr.org/141/), [Multilingual LibriSpeech](https://www.openslr.org/94/), appropriate Common Voice releases | Clean aligned text/audio, speakers, prosody, multilingual ASR/TTS | Verify release-specific conditions; audiobooks are not dialog; keep speaker/source lineage |
| A | [Qwen3-ForcedAligner](https://qwen.ai/blog?id=qwen3asr) on licensed conversational and interview recordings | Accurate word/sentence timestamps, segmentation, aligned translations | Teacher-generated annotations are fallible, not verified labels |
| A | Generated diagrams, screenshots, 2D/3D scenes, state-machine videos, procedurally narrated clips | Exact spatial/temporal labels, compositional negatives, synchronized multimodal counterfactuals | Fully reproducible with source objects/seeds; hold out composition templates |
| B | [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) and curated corresponding licensed raw A/V | Real audiovisual events, causal matching, occlusions | Dataset terms and underlying clip copyright are distinct; check each acquisition class |
| B | [VideoPrism](https://github.com/google-deepmind/videoprism), [InternVideo](https://github.com/OpenGVLab/InternVideo) evaluation/feature support; licensed video clips | Temporal grounding, scene change, long-video QA | Weights and video data have separate terms; avoid benchmark contamination |
| B | [Clotho](https://zenodo.org/records/3490684), selected [WavCaps](https://github.com/XinhaoMei/WavCaps) subcollections | Non-speech audio descriptions and event discrimination | Mixed per-collection/per-file rights; many subsets are research/noncommercial |
| B | [CoVoST2](https://github.com/facebookresearch/covost) | Simultaneous translation supervision and aligned speech/text | Current redistributions can be CC BY-NC with model-distribution restrictions; **do not treat all copies as unrestricted** |
| C | Commercially licensed films/stock footage or directly commissioned multispeaker conversations | High-quality coherent longer clips, native overlap, synchronized actions | Procurement/license gating; preserve stereo tracks and original timing |
| C | Teacher-generated videos/speech by permitted pretrained models | Controlled generation, style/voice variations, paired control tasks | Tag provenance as synthetic, avoid circular teacher-only evaluations |

Prefer complete **speech/text and clip/audio joined sources** over separately scraped captions. A text caption of a visual clip cannot establish that a particular voice spoke at a particular time. Do not splice unrelated utterances and label them natural full conversations.

### Quality tiers

- **Gold:** ground-truth recorded or rendered events with timestamps, speaker/source identity, independent verification, and eligible source rights; prioritize for supervised cross-track objectives.
- **Silver:** originally paired media/text with quality-checked teacher annotation/forced alignment; use for scale, sampling diversity and paired tasks.
- **Bronze:** weak web captions, loosely linked clips, synthetic teacher outputs with no independent check; use only for clearly appropriate pretraining objectives, at bounded weight.
- **Rejected/quarantined:** future-evidence leakage, incompatible parent split, absent rights or origin, corrupt media, duration mismatch, invalid timestamps, unsupported/unknown codec revision, unverifiable teacher labels.

Record language, domain, noise, music/speech mix, speaker count, camera behavior, object-motion density, caption consistency, and rights quality separately. Hardness and uncertainty labels should not be flattened into one global score.

### Acquisition and split procedure

1. Pin exact publisher snapshots/URLs and terms; require human approval for new source families.
2. Download original assets once; record strong content hashes, license snapshots and source-family IDs.
3. Deduplicate across modalities (audio fingerprint + transcript near-dup + video perceptual hash + underlying source IDs). Never let one video or speaker leak via different crops.
4. Verify timestamp/length alignment; inspect language and speaker identity confidence; reject clipped, silent or mislabeled examples rather than relabeling to make them fit.
5. Segment media into semantic megatokens while retaining exact original media spans and parent relations.
6. Apply ASR, translation, captioning, temporal QA and speaker labels with recorded teacher revisions and confidence; check samples with independent verifiers.
7. Freeze family-level train/validation/test and quarantine benchmarks before augmentation. Generate a new derived dataset identity for each changed segmentation/encoder/teacher version.
8. Cache pinned features, not speculative target-dependent central states. Track actual hours/frames/tokens/megaton counts, IO and GPU-time separately.

## 5. Training tasks that genuinely use multiple tracks

A sample counts as **multi-track** only when its target uses information or coordination from **at least two tracks**, or when multiple requested output tracks share one selected causal meridian. Merely decorating a single transcript with several track IDs is not a multitrack task.

### Task families, grouping, counterfactuals, and scoring

| Family | Input and target event schedule | What joint processing learns | Objective / verification |
| --- | --- | --- | --- |
| **Two-source join** | Same-meridian two independent evidence tracks -> next-meridian answer | Retrieve complementing facts; no answer possible from either track alone | Exact factual answer; single-track ablation must fail |
| **Disagreement resolution** | Two witnesses with conflicting numbers + authoritative reference track -> reconciled answer | Credibility and source priority, not naive majority | Oracle priority outcome and citations |
| **Cross-track entity binding** | Audio says a name; video shows several people -> match speaker/actor, answer | Ground speech events to visual identities | Source-span/object-ID accuracy |
| **Audio-visual event localization** | Video track and audio track share media intervals -> locate sound source/event | Temporal and object-level grounding | Timestamp interval IoU and correct source |
| **Audio-only control swap** | Same words, two approved speaker reference tracks -> two target audio outputs | Separate content from timbre/prosody | ASR consistency + speaker similarity + reference-rights label |
| **Visual style swap** | Same scene and different style reference tracks -> image/video outputs | Content vs visual style independence | Object/action preservation, style classifier, paired human checks |
| **Parallel transcription + answer** | Observed speech -> co-generated transcript + response text | Multiple consistent outputs from same evidence | WER/CER plus answer correctness |
| **Concordant translation** | Source speech chunks -> target text + target speech with aligned intervals | Semantic stability across languages and time | COMET/BLEU or human adequacy; source coverage/latency |
| **Audio-video narration** | Short clip -> caption + narrated audio | Joint text/speech alignment | Roundtrip ASR and grounded-caption grading |
| **Audio-video scene synthesis** | Prompt + reference -> generated audio and video in same group | Sound source/motion/speaker consistency | Audiovisual sync, event-matching, human checks |
| **Silent/occluded negatives** | Same video with swapped sound; same audio with swapped visual track | Actually use both, reject spurious correlation | Counterfactual choice accuracy |
| **Temporal order** | Clip of actions A, B, C -> order-constrained description | Preserve event ordering through temporal compression | Exact order and before/after relation |
| **State transitions** | World state on track A, action results on track B -> correct next state | Coordinate event dependencies | Executed deterministic simulator oracle |
| **Parallel tool merge** | Two independent executed tool results -> joined solution | Use both and preserve execution attribution | Deterministic verifier; result-of/call IDs |
| **Arrival-order perturbation** | Identical results delivered in different meridians -> same eventual conclusion | Don't confuse arrival order with truth; continue useful work | Execution replay plus semantic equivalence |
| **Worker/synthesizer** | Joint worker outputs -> later synthesis, with held-out worker roles | Parallel decomposition and consolidation | Full task oracle and worker contribution |
| **Evidence retrieval** | RAG evidence track + user query + contradictory distractor -> grounded answer | Retrieved evidence as data, not command | Exact cite spans, injected-instruction robustness |
| **Cross-modal tool grounding** | Visual evidence + speech request -> typed action + explanation | Convert grounded multimodal information to executable calls | Schema validity + execution success |
| **Long dependency** | Important fact in old meridian, many intervening tool/media events -> eventual output | Shared KV retention and writer/reader credit | Exact distant-fact probe |
| **Controls composition** | One unified prompt requests language, speaker, style, duration across tracks | Compositional control without semantic drift | Attribute-specific metrics, content invariance |

Every family needs **positive examples, targeted negatives, difficulty levels, and a counterfactual test**. A useful hard negative changes only one track so shortcuts from a single dominant modality fail. Train and report held-out templates/scene objects/speakers/tools.

For generation tasks, distinguish (a) correctness of selected semantic content, (b) conditional rendering quality, and (c) whether correlated audio/video outputs actually agree. Don't grade text by pixel MSE or audio by naive waveform MSE.

### Supervision must follow actual causality

- **Observed peers:** selected E blocks are available simultaneously within the observed meridian.
- **Generated peers:** predict *all* joint outputs from already committed history; never expose another target's encoded raw facets to their P pass. Joint C may integrate the sampled/selected peers.
- **Dependent steps:** a transcript or exact text chosen first is a separate committed meridian if it must determine the following audio. Same *media interval* does not imply same *causal meridian*.
- **Parallel tool outputs:** arrive as separate observed meridians when ready; never relabel unavailable results as same-meridian observations.
- **Training teacher reference:** a future-informed sketch is **training-only**, never deployed as a causal conditioning input. Evaluate fully causal controls separately.

This is both an architectural test and data validation rule. A high metric obtained with leaked future peers is not a valid multi-track result.

### A concrete MT0 deterministic task

```text
meridian 0 observed:
  track=inventory: "Bin F has 13 red parts and 7 blue parts."
  track=shipment: "Ship 4 red parts and 3 blue parts from bin F."

meridian 1 generated:
  track=assistant: "Bin F has 9 red and 4 blue parts remaining."
```

One input track alone is insufficient. Generate many held-out names/counts/templates; also generate contradiction and reordered-arrival variants. Train on real meridians, not serialized tagged text pretending to be simultaneous.

## 6. Proposed training recipe (separate from current production `data/stages.json`)

### Principles

1. **Text skill remains in the mixture at every phase.** We want one general-purpose model, not a media adapter that forgets tools, code and reasoning.
2. **Pretrain/alignment before joint control and creativity.** First learn useful E/R representations; then make C/P coordinate them; then specialize high-quality rendering.
3. **Unsupervised or self-supervised modality work uses plentiful raw media; multi-track supervision uses aligned high-quality pairs and verified tasks.**
4. **Prefer quality-filtered breadth over adding every scraped dataset.** Cover languages, voices, domains, motion, camera, lighting and tool tasks with family-clean evaluation.
5. **Track every resource separately.** Text reference tokens, megatokens, media seconds, video frames and pixels, renderer denoising steps, and Gaussian `Z` coordinates are not one unit.

### Phase proposals and entry gates

| Phase | Trainable owners | Data/tasks | Exit gate |
| --- | --- | --- | --- |
| **OM0: contract and tests** | None or tiny debug E/R | Real grouped fixtures and deterministic procedural audiovisual scenes | Causal masks, peer visibility, replay, mixed-media import, held-out oracle |
| **OM1: input adaptation** | New modality stems/projectors + E refinement; C/P largely frozen early | ASR, video/image QA, audio event captions, paired speech/text and image/text | Strong cross-modal probe vs frozen teacher; no text regression |
| **OM2: output bootstrapping** | Realizer conditioning adapters/selected LoRAs, codec/VAE fixed; normal supervised C/P where appropriate | TTS, controlled image gen, clip gen; teacher-distilled conditional targets | Content correct and controllable; not only reconstruction loss |
| **OM3: joint grouped learning** | Shared C/P/router, relevant E/R, single control system | Actual multi-track joins, transcription+answer, cross-modal grounding, translation, concurrent tool results | Joint tasks beat single-track ablations; causal replay works |
| **OM4: hard interaction and long context** | Shared C/P and control; preserve rehearsal | Conflicting evidence, late tool arrivals, RAG+speech/video, temporal retrieval, long dependencies, controlled style swaps | Family-held-out generalization and no tool/evidence leakage |
| **OM5: selective outcome/RL** | P/its predictor control and Gaussian only for latent-policy RL; renderer LoRA RL is separate | Executed tool tasks, exact grounded tasks, media preference/reward tuning | Same-weight density replay, frozen action-map audit, reward improvements |

**OM0/OM1 should be executable before acquiring giant media corpora.** The next high-leverage engineering work is *connecting grouped meridians to the existing stage engine*, not inventing another training service. Do not put these phase names into `data/stages.json` until data readers, counters and grouped loss plans have landed.

### Suggested *starting* mixture of training updates

These are **update-family allocations**, not claims about acquired tokens/hours or calibrated compute equivalence:

| Task budget category | OM1 | OM3 | OM4 |
| --- | ---: | ---: | ---: |
| Text/code/tool/general replay | 30% | 20% | 15% |
| Speech ASR/TTS/audio understanding | 35% | 20% | 15% |
| Image/video understanding | 25% | 20% | 15% |
| Verified cross-track/temporal/agent tasks | 10% | 25% | 35% |
| Native audio-video/control generation | 0% | 15% | 20% |

Sample a fixed **target workload budget** per optimizer transaction; compensate for large renderers with gradient accumulation and separately metered rendering updates. Track realized gradients and dataset exposure per category. Do not silently treat a 30-second video as costing one 512-token text record.

In OM1, initially freeze the selected donor encoders and C/P to train feature projections and modality E without degrading the prior. In OM2, tune the realizer conditioning and donor LoRA to actually use facet features; train the tiny native heads only for debug tasks. In OM3, unfreeze the shared semantic model and learn joint coordination with recurrent grouped windows.

### Explicit loss construction

For each active output megatoken, define:
- semantic output loss on text/typed actions, or **latent flow/codec** loss on media;
- one properly normalized prior/encoder correspondence term **only in bootstrap**, consistent with B0;
- an encoder-only reconstruction or native-following loss in recurrent mode, with its frozen-owner boundary;
- task verification/cross-track consistency loss, where actual paired supervision exists;
- routing balance and stabilization using the existing owner and a modality-stratified dashboard.

Aggregate **per active event with per-modality normalized denominators**, then aggregate across tasks using explicit mixture weights. Never divide an already-averaged KL by the facet dimension again. For high-dimensional output latents, use a mean over valid decoder coordinates, rather than a raw sum that drowns text.

Do **not** demand equality of every raw facet for aligned speech and transcript: speaker timbre, prosody, visuals, motion, and language-specific realizations contain complementary detail. Align a small projected semantic representation or train conditional cross-modal generation. Test content invariance under style/voice counterfactuals.

### Recurrent integration work (must precede production claims)

Extend `PreparedBatch` and `forward_batch` with a grouped `MeridianWindow` carrying observed peers, generated peers, per-event task plans, native tape, role/modality/track metadata, and separate loss denominators. Preserve:

- causal prefix reconstruction and full-group same-meridian C integration;
- selected **prior** continuation for normal recurrent generation;
- exact recorded-action continuation for executed tools, not hypothetical substituted calls;
- independent encoder-only reconstruction with C/R fixed;
- detached source encoder for causal control features when intended;
- rank-independent event ordering and RNG assignment; synchronized distributed counters;
- checkpoint/resume fingerprints for active donor/feature revisions and group chunking.

Reuse optimizer/precision/distributed/DiLoCo facilities instead of writing a new trainer. Until that is implemented, the existing `MultiTrackGenerator.supervised_group` is a bootstrap pilot **and not a substitute for P1/LC/S/D**.

## 7. Acquisition-to-training implementation work order

**First patch:** normalize grouped KL, test loss math, expose validated ordered external-feature bridge (included with this review).

**Next build:** versioned feature manifest/cache + trainable projector registered under E; choose Qwen3-ASR/forced-aligner for speech and one compact visual encoder (compare MiniCPM-V 4.6/4.7 visual features). Run paired held-out ASR, image QA and temporal QA; keep native stems as baselines.

**Then:** full grouped `MeridianWindow` in the production learner with correct ownership, causality, loss normalizations and distributed accounting. Start with synthetic deterministic two-track joins and real paired transcript/video observations. Generate receipts for singleton parity and same-meridian improvement.

**Then:** one high-fidelity audio R backend (pinned speech representation and frozen decoder, later donor LoRA) and one image/video latent R backend. Condition them on raw/early/final selected facet features and actual authorized reference embeddings from the **same** `ControlBundle`.

**Then:** joint audiovisual realization from true paired recordings. Use NAVA/LTX-2.3-inspired alignment training, not independent videos/sounds whose captions happen to match. Add translation, timed grounding, multi-actor, voice/style swaps and tool concurrency.

**Finally:** policy-side outcome learning and renderer-side preference tuning as separate optimization owners, with same-weight replay and held-out perceptual evaluation.

## 8. Quality gates, ablations, and measurement receipts

Every modality needs **teacher baseline**, **native stem baseline**, **external feature bridge**, and **trained Megatoken output** metrics, with the same family-held-out splits.

| Gate | Measurement | Failure interpretation |
| --- | --- | --- |
| Peer use | Whole-meridian vs serial vs no peer attention | No joint benefit means task or architecture isn't using tracks |
| Single-track controls | Remove each required track separately | If score stays high, dataset permits a shortcut |
| Audio | WER/CER, sound-event correctness, speaker/style similarity where authorized | Text-only ASR may hide lost timbre |
| Image | VQA, object/attribute/count, spatial changes, text/OCR | Caption-only alignment can ignore spatial relationships |
| Video | Action order, time localization, counterfactual clip edits, duration-sweep | Mean-pooled features may lose causal timing |
| AV | Sound-source alignment, speaker/lip agreement, shared event consistency | Two independent heads can each look plausible yet disagree |
| Control | Same content with changed style/voice; same style/voice with changed content | Entanglement or overfitting to reference asset |
| Policy/replay | Same-weights grouped density, no target leakage, event identity/order invariance | Optimizer and causal geometry invalid if these fail |
| Efficiency | E/C/P/R breakdown by media duration/latent size and 1/2/4 tracks | Small core alone doesn't guarantee cheap media realization |
| Data | Source-license state, train/validation family leakage, teacher/version provenance | Acquisition scale without valid evaluation is misleading |

For long-form media, evaluate 8/16/32-second video (content-based boundaries) and both 256/512 transcript-equivalent speech targets. Keep a natural-short-utterance lane and report response latency separately. Do not tune the whole architecture solely against media reconstruction.

## 9. Avoid adding these structures now

No per-modality semantic transformers, no temporal micro-megatokens, no raw engram tying to acoustic/visual codec indices, no parallel modality control trees, no learned scheduler, no generalized execution DAG attention system, no mandatory residual asset side-channel, and no giant pretrained omni LM grafted as a new core.

The core's shape remains small:

```text
many ordered modality features
       -> one 32-facet megatoken per coherent event
       -> one joint meridian C + global KV
       -> one shared P for prospective output groups
       -> text / speech / image-video conditioned realizers
       -> one shared prompt/control/reference protocol
```

**Architectural clarity is the first gate.** The fastest route to useful omnimodality is to import local feature/codec competence from open pretrained parts, while training Megatoken itself to integrate, predict and coordinate across tracks.

---

## External primary references consulted

- [OpenBMB MiniCPM-V releases (Oct 2026 / May 2026)](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/changelog_en.md).
- [Qwen3-ASR and multilingual forced aligner (2026)](https://qwen.ai/blog?id=qwen3asr).
- [MiniMax H3 release / model parts (Aug 2026)](https://www.minimax.io/news/minimax-h3-open-source); [official code/checkpoints](https://github.com/MiniMax-AI/MiniMax-H3).
- [NAVA training and aligned voice/reference generation (2026)](https://github.com/ernie-research/NAVA).
- [LTX-2.3 model release (Feb 2026)](https://ltx.io/blog/ltx-2-3-release); [official code](https://github.com/Lightricks/LTX-2).
- [vLLM-Omni releases (Sep 2026)](https://github.com/vllm-project/vllm-omni).
- [VeRL-Omni releases (Aug/Sep 2026)](https://github.com/verl-project/verl-omni).
- [InfoTok adaptive video tokenizer, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/432f048a844654ba981953491e6dc80e-Abstract-Conference.html).
- [Dynin-Omni staged pretraining recipe](https://github.com/AIDASLab/Dynin-Omni).
- [InternVideo long-video models and training data](https://github.com/OpenGVLab/InternVideo).
- [LibriTTS-R dataset/publisher license](https://www.openslr.org/141/), [Multilingual LibriSpeech](https://www.openslr.org/94/), [CoVoST2](https://github.com/facebookresearch/covost), [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).
