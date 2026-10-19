# Pretrained omnimodal components through a shared 32-facet bottleneck

**Status:** Implementation plan and architectural decisions; not a claim of implemented pretrained integration or measured model quality.

**Date:** 2026-10-19.

**Reviewed repository baseline:** `main` at `8e71d8fbab494873bb7d89164b43c72c9b193dd0`, after the media-training integration in PR #13.

**Scope:** Choose a small number of trained components, define exactly how they connect to Megatoken, and extend the existing grouped learner. Preserve information-dense megatokens, one shared C/P/KV system, one control system, and optional loading of modality components.

This document narrows the component choices in [omnimodal-architecture-and-training.md](omnimodal-architecture-and-training.md) and the implementation direction in [omni-media-pipeline.md](omni-media-pipeline.md). It supersedes earlier suggestions to give media realizers independent raw/early/final inputs, bypass speech facets with a transcript, or standardize a new video latent space before choosing a compatible pretrained generator. Existing implementation/status documents remain descriptions of their respective revisions.

## 1. Decisions to implement

| Responsibility | Initial choice | Policy |
| --- | --- | --- |
| Speech understanding | **Whistle encoder embeddings** | Frozen; replace Wav2Vec2 as the preferred speech-semantic input path |
| Speech acoustics and reference encoding | **The exact continuous Mimi variant shipped with Pocket TTS** | Frozen; fuse with speech-semantic features when acoustic information matters |
| Speech realization | **Pocket-derived latent generator, initialized from trained weights, plus its matching Mimi decoder** | Adapt conditioning, then distill; do not learn speech acoustics from scratch |
| Image/video understanding | **MiniCPM-V 4.7 visual tower and trained mergers** | Experiment B is the default; 16x compression normally, 4x for declared detail-sensitive inputs |
| Visual understanding fallback | Existing SigLIP2 bridge | Switch only for a concrete extraction, memory, or quality problem |
| Still-image realization | **Sana 0.6B and its matching DC-AE** | Preserve the trained generative prior; avoid introducing a video-sized dependency for still images |
| Initial video realization | **SANA-Video 2B 720p and the exact LTX-2 VAE used by that checkpoint** | Keep the trained generator/codec pair together |
| Paired audiovisual extension | **A matched LTX-2.5 audiovisual pack**, including separate audio/video codecs | Optional, later stage for genuinely coupled audiovisual output; not a dependency of speech or still images |
| Semantic representation | **32 x 512**, with the existing compact width retained | No global facet increase |
| Additional capacity | Additional ordinary **32-facet events** for dense regions or long event sequences | Sparse and explicitly accounted, not one core event per frame |
| Realizer content interface | **One 32-facet block per selected event** | No independent text, source-latent, reference-media, or historical-KV input to an ordinary media realizer |
| Controls | Existing unified control source, expanded upstream of selection | Voice/style/language instructions are represented in the selected facets, not separate downstream control systems |

### The most important revision

**Reuse a trained generative prior, not only a trained VAE.** A decoder can reconstruct valid codec latents; it does not supply the model that predicts those latents from 32 facets. Consequently, adapting SANA-Video with its existing codec is the initial video path. Training a small generator from random initialization over newer LTX-2.5 latents is a later compression experiment, not the critical path.

The LTX-2.5 paired pack remains useful when synchronized general audio/video becomes the task. Its newer codec is not silently substituted under the SANA checkpoint. Separate codec versions can coexist behind the same Megatoken interface without sharing latent coordinates.

The choices above are development defaults, not a tournament of parallel experiments. Only the stated fallback or a demonstrated bottleneck should reopen them.

## 2. What the repository already provides

At the reviewed commit, the repository contains grouped text/media training, checksummed tensor assets, E-owned external-feature projections, a procedural audiovisual corpus, counterfactual siblings, separate media-loss denominators, and the `G` event budget. These are useful foundations; they need extending rather than replacing. See [media pipeline](omni-media-pipeline.md), [media model](../src/megatoken/model/media.py), [asset schema](../src/megatoken/data/media_assets.py), and [grouped training](../src/megatoken/training/meridians.py).

The current production media path still has several deliberate baseline limits:

- Audio is standardized to 16 kHz, while the selected speech codec has a different native rate.
- Cached inputs are features, but realization targets are raw waveform/pixel arrays.
- The native media realizer fuses raw, early, and final core tensors and receives a separate control vector.
- Voice/style references are serialized identities; that alone does not encode the referenced voice or image.
- Small direct waveform/pixel regressors demonstrate gradients and geometry, not pretrained speech or visual synthesis.

This plan replaces those boundaries incrementally. It does not require changing C4, P12, the shared expert topology, or the meridian causality model.

## 3. One bottleneck, with precise semantics

### Distinguish three representations

Use different names and schema identities for:

1. **Perceptual features `F`:** ordered outputs of a frozen encoder, used as inputs to E.
2. **Semantic facets `Z`:** the authoritative selected or encoded `[32, D]` block, processed by C and recorded in the action tape.
3. **Codec latents `U`:** the sequence/grid consumed by a particular frozen media decoder.

These are not interchangeable because they share a tensor shape or originate from the same media. Pocket acoustic channels, MiniCPM visual tokens, and LTX latent channels have different learned coordinate systems.

### The ordinary path

```text
Available input media
    -> frozen perceptual features F
    -> small E adapter + shared reduction
    -> raw Z [32,D]
    -> joint C with other selected events and prior shared KV
    -> one final contextual block B [32,D] per event

Prior history + unified controls + output request
    -> shared P predicts a group of raw Z blocks
    -> select the joint raw Gaussian action
    -> joint C, publishing final KV once
    -> one B [32,D] per selected output
    -> modality realizer predicts codec latents U
    -> frozen matching codec decoder
    -> physical media
```

**Initial dispatch rule: `B = C_final(event)`.** Use the already computed final central block. Do not add another cross-modal transformer or pass three independent raw/early/final memories into each realizer. Raw `Z` remains authoritative for recurrence, Gaussian density, and replay; `B` is its deterministic contextual materialization interface, not a second recurrent state.

Observed inputs and direct injected codec-training examples enter through the same `advance_group` path. Thus reconstruction and causal generation exercise the same C-to-R interface. A standalone raw-only autoencoder may be a diagnostic, but its results do not substitute for this served path.

A small width projection inside each realizer maps all 32 slots to its pretrained conditioning width. **Do not immediately pool those 32 slots into one vector or eight slots.** That would introduce another bottleneck before establishing whether the main one works.

### What R can receive

The ordinary media contract is:

```python
realize(
    facets,       # [batch, 32, D]: the sole learned content condition
    output_spec,  # codec/renderer identity, geometry, rates, allowed length
    noise,        # explicit base randomness under a pinned protocol
) -> latent_payload
```

R may expand positions, run its own local acoustic autoregression or denoising iterations, and use a frozen decoder. It cannot independently read an earlier transcript, arbitrary prompt embeddings, the original image/audio, source codec latents, or the global history. Output specifications must not conceal captions, speaker-reference embeddings, or other content-bearing side inputs.

The existing exact text-copy and schema-formatting mechanisms need not be removed. They are explicit text/action contracts, not permission to give a speech or video realizer an undeclared content shortcut.

### Same-meridian cooperation

All selected peer blocks already communicate through joint C. For ordinary separate outputs, dispatch each resulting B to its own R. When a paired audiovisual pack is enabled, `realize_group` may consume the finite set of B blocks for its declared audio/video outputs. Two such events supply 64 facets in total, and must be accounted that way. The paired pack may exchange local denoising features internally; it still cannot access uncompressed input media or historical KV.

This preserves the group semantics without claiming that a coupled two-event renderer is a single-event 32-facet capacity test.

## 4. Speech ingestion: Whistle for meaning, Pocket-Mimi for acoustics

### Whistle is the selected semantic stem

The official Whistle card documents a 16.9 MB quantized model, 16 kHz mono input, encoder embeddings every 80 ms, and 30-second native passes with longer-input windowing. It exposes an embedding API without requiring transcription. The file size is not an unquantized parameter count. [W1]

Use those embeddings as frozen input features. Train only their projection and the Megatoken reduction/refinement path initially. Package the native runtime and model revision explicitly; do not make differentiating through a quantized C++ runtime part of the training requirement.

Whistle is an ASR-oriented stem, not an audio autoencoder. Its documented speech-focused frontend and objective are reasons not to assume faithful timbre, music, or environmental-sound representation. [W2] This is a scope distinction, not a reason to reject it for spoken instructions, transcription, and translation.

### Add acoustic features before the same bottleneck

For speech reconstruction, voice references, and expressive delivery, supply Pocket's matching continuous acoustic features as an additional E input view:

```text
original audio master
    -> deterministic 16 kHz view -> Whistle -> semantic features
    -> deterministic 24 kHz view -> Pocket-Mimi -> acoustic features
    -> timestamp-aware small fusion + shared E reduction
    -> ONE [32,D] speech event
```

Both views describe the same event; they are not separate semantic transformers or mandatory extra tracks. Each gets a small projection and view identifier. Their information must cross the same 32-facet interface.

The nominal 80 ms rate is convenient but does not establish exact frame alignment. Preserve origins, receptive-field offsets, padding, and valid spans; align by timestamps, not only by array index. Long Whistle windows can have boundary effects because the encoder sees a completed local window. Use a fixed window/overlap/trimming policy and record it in the extractor fingerprint.

**One trained audio adapter supports missing views.** Train view dropout with an explicit presence mask, so speech-only deployments can omit the acoustic encoder. Select the ingestion profile before reading supervision; never enable an acoustic view because a hidden future target happens to require it.

Wav2Vec2 remains a working fallback and offline teacher. It is not another mandatory resident model. Environmental audio outside the selected speech codec's demonstrated scope belongs to the later general-audio pack.

## 5. Speech realization: distill Pocket latents without a transcript bypass

### Pin the correct codec contract

Pocket's inspected English configuration specifies 24 kHz mono audio, 12.5 latent frames/second, a 32-dimensional inner latent, and a six-layer 1024-wide generator backbone. Its `encode_to_latent` and `decode_from_latent` operate on continuous latents; the implementation uses a `DummyQuantizer`, not Moshi's interchangeable-looking discrete RVQ interface. [P1, P2]

Record the exact Pocket codec weights, resampling, normalization, axis order, padding, and decoder-state convention. The flow model stores latent mean/standard-deviation buffers; caching normalized generator coordinates and caching codec coordinates must have distinct, explicit conventions. [P3]

The number 32 in Pocket's **channels per acoustic frame** is unrelated to Megatoken's **facets per semantic event**. Do not tie those axes or their embeddings.

### Start from a trained generator

The upstream training recipe includes teacher training and depth distillation to a smaller student. Reuse that engineering and trained weights rather than beginning with an untrained linear latent expander. [P4]

Initial implementation:

1. Freeze a selected Pocket teacher and its matching codec.
2. Initialize our speech R from the trained compact generator and flow head.
3. Add a learned projection from `B[32,D]` to 32 vectors at the generator's conditioning width. Feed these through its prefix-conditioning interface, preserving required structural BOS/EOS conventions.
4. Replace the student's original text/voice-prefix content with these vectors. Start with adapter training, then enable limited conditioning/backbone LoRA if necessary.
5. Keep acoustic output projection, flow/sampler conventions, and Mimi decoding compatible with the chosen checkpoint.
6. Only after useful direct facet-conditioned speech works, distill depth or sampling steps further.

The original text-conditioned Pocket system is the quality baseline and teacher, not an alternate deployed semantic path. We may temporarily use both conditioning forms for a documented adaptation warm-up, but the completion criterion is **facets-only student conditioning**. Cache or generate the teacher's text/voice conditions only on the teacher side.

### Exact TTS remains an integrated task

An exact transcript can be a preceding observed text event, or a text event selected in a preceding meridian. P predicts the speech Z from that context and the voice control; C produces B; R receives B only. It must not re-read the transcript directly.

This is stricter than the earlier conversational suggestion. It makes word fidelity a real test of the bottleneck. Train names, numbers, punctuation-sensitive delivery, code-like strings, and corrections deliberately. For utterances exceeding a demonstrated reliable content budget, split into coherent speech events instead of hiding extra tokens in the renderer interface.

### Distillation targets and losses

Use two complementary target sources:

- **Real paired speech:** encode licensed recordings through the exact Pocket codec. These targets preserve natural prosody, noise conditions, and speaker variation beyond a single teacher's habits.
- **Teacher-generated speech:** supply known text and authorized reference voices to Pocket; capture the exact latent sequence immediately before its decoder, with its seed and normalization state. Verify that the decoded speech actually follows the transcript before keeping the pair.

A matched latent target supports reconstruction/teacher-forced acoustic training. Arbitrary samples of the same utterance need not have matching durations or phase; do not average unrelated teacher trajectories using naive framewise MSE. Use the native conditional flow objective and, where supported, teacher flow/velocity distillation evaluated at the same noisy point and acoustic prefix.

Retain teacher forcing only within local acoustic training. Evaluate full student rollouts conditioned on B, not on a target acoustic prefix. A useful flow loss under teacher forcing does not establish stable long-utterance synthesis.

Speech realization state is local to the selected event. Initially reset decoder state at semantic event boundaries and use the codec's valid trimming/fade conventions. If continuity later requires carrying a codec tail, make that bounded, reconstructible physical-decoder state explicit; it must not become a private semantic history.

## 6. Visual ingestion: MiniCPM-V 4.7, without its language model

Make experiment B the standard path: the trained visual tower, its intermediate compression, and its final merger/projection. The official feature interface exposes the visual pipeline and supports 16x or 4x downsampling. [V1, V2]

**Initial mode: 16x.** Use 4x for a declared detailed-reading profile such as OCR, dense interfaces, or small-object inspection. This is one pretrained tower with a supported compression setting, not two model families to benchmark indefinitely.

Extract visual modules directly, or load a donor once offline and export the exact required submodules. Verify extracted features against the parent model at a pinned revision. A convenient `get_image_features` method does not itself guarantee that a naive loader avoided allocating the entire language model.

Do not advertise the retained visual tower as a 1B model or infer its size from the full VLM's name. Count the exported visual/merger parameters and measured resident memory. Frozen towers still occupy memory and perform work; their training advantage is the absence of optimizer state and usually cached inference.

Preserve crop coordinates, patch grids, frame order, timestamp offsets, and sampling policy until E's reduction. Do not globally average visual features before the model has a chance to select salient information. An image and a short video use the same visual feature family; the temporal reduction operates within a megatoken.

For reconstruction/editing tasks, MiniCPM features alone may discard exact appearance. Permit an additional **matching visual-codec latent view into E**, analogous to the acoustic view for speech. It too passes through the same 32 facets. Do not pass that view directly into R or falsely describe semantic-only reconstruction as a lossless autoencoder.

Use SigLIP2 only if the preferred extraction cannot be isolated reliably, exceeds the agreed resident budget, or fails a specific required feature task. Preserve this as a fallback, not a second default configuration.

## 7. Image/video generation: keep pretrained generators and codecs paired

### Still images

Use the ordinary Sana 0.6B generator with its matching DC-AE, not a randomly initialized image decoder. The official checkpoint describes the compact generator and its deep-compression latent path. [I1]

Train a 32-slot conditioning adapter at the original conditioning interface, then selected LoRA parameters. The original text encoder can supply teacher conditioning offline; it is not a required deployed content path after adaptation. Avoid adopting a heavily few-step-distilled variant before the new conditioning works; speed distillation is a later step.

### Initial video pack

Use the **SANA-Video 2B 720p checkpoint and its specified LTX-2 video VAE**. The official implementation identifies that pairing and distinguishes it from the 480p/Wan-VAE configuration. [G1]

This is preferable to selecting LTX-2.5 latents and then having to train the entire video prior ourselves. Reuse the pretrained denoiser and train its facet-conditioning projection/LoRA. Start with short, moderate-resolution, codec-valid clips; extend duration after semantics and local motion survive the bottleneck.

A pretrained checkpoint is not automatically resolution- or duration-invariant. Retain its latent scaling, channel order, timestep parameterization, scheduler, padding, and geometry rules. Enlarging E's clip budget and enlarging R's generation horizon are separate operations.

### Later paired audiovisual pack

LTX-2.5 explicitly packages separate video and audio codecs and both trained and distilled generation components. A lighter convolutional video decoding option is documented alongside the more expensive diffusion decoder. [G2]

When joint general audio/video is required, adapt a **matched trained LTX audiovisual generator**, using one B block for audio and one for video. Keep its audio/video interaction mechanism inside group realization. Decode the two latent outputs with their respective codecs, potentially in parallel when resources permit.

Start with the convolutional video decoder in this pack. Do not add diffusion fidelity rendering, multiple upscalers, or a prompt-enhancement model to the initial integration. The larger trainable generator can be optional/offloaded; compression into a smaller student comes after direct conditioning works.

Do not replace the initial SANA pack's VAE with this newer VAE merely because both are called LTX. Matching nominal channel counts is insufficient. Compatibility requires the actual encoder/decoder transforms, scaling, packing, and trained generator distribution.

### Separate speech and soundtrack tasks

Pocket-Mimi remains the speech codec. The audiovisual pack's audio codec handles its paired soundtrack. These are distinct codec identities behind one audio event type, not independent reasoning/control systems. They need not be resident simultaneously.

Independent audio/video decoders are natural for separate tracks. **Independent generation is not a synchronization method.** Early separate-track outputs may be only coarsely aligned. Fine lip/event synchronization belongs to paired realization training and the coupled pack, with explicit media-clock and duration conditions.

The main measured result should be that the facets determine the requested scene, events, and sounds. Frozen codecs provide physical reconstruction competence, not guaranteed semantic consistency.

## 8. Large encoders/realizers versus the core's intelligence budget

Larger pretrained components can reduce the amount of phonetic recognition, texture synthesis, local motion modeling, and acoustic realism that C/P must learn. They do not eliminate cross-track reasoning, causal interpretation, grounding, or the need to encode requested details in B.

A stronger encoder can pack *more relevant information* into the same coordinates. A stronger generator can fill unspecified detail using its learned prior. It cannot recover arbitrary source-specific information that never crossed the bottleneck. Distinguish **faithful retention** from **plausible regeneration** in both objectives and reports.

A missing name in a transcript, a changed chart value, or the wrong actor cannot be excused by realistic output. Conversely, reproducing exact pixel noise is not the goal of a semantic scene generator.

Keep four parameter/memory ledgers:

| Ledger | Includes | Why |
| --- | --- | --- |
| Always-resident core | C/P, shared experts, common state/control modules | Semantic architecture budget |
| Trainable modality parameters | E projections, R conditioning, selected LoRAs | Optimizer/gradient cost |
| Optional frozen components | Perceptual encoders, pretrained generators, codecs | Loaded memory and compute, even without gradients |
| Offline teachers | Full donor VLMs/TTS/video generators used for labels or distillation | Preparation cost, not a required inference dependency |

Share *interfaces and compatible blocks*, not every parameter. Keep byte/ngram embeddings for text, transcripts, names, and prompts. Acoustic/visual latent coordinates should not be tied to arbitrary byte-ngram rows. Preserve shared P experts across modalities; keep E, P, and R parameter ownership distinguishable. Do not distort a pretrained renderer merely to force its incompatible hidden width into the existing U bank.

## 9. Facet capacity: hold the default, add sparse semantic events

### Why 32 remains the default

At width 512, an event has 16,384 coordinates. That is 32 KiB in BF16 or 64 KiB as an FP32 raw action tape, before history and workspace allocations. These are storage counts, not measured information capacity.

Pocket's configured latent rate gives a useful comparison: `12.5 frames/s * 32 channels = 400 scalars/s`. A 30-second speech segment has about 12,000 latent coordinates; a 60-second segment has about 24,000. These are calculated from [P1], not quality predictions. The shared facet block is not intrinsically tiny relative to this acoustic space, but robust word/prosody retention still depends on E/R and the noise objective.

Video is more demanding. With the 720p SANA contract's 128 channels, 32x spatial compression, and 8x causal temporal compression, an illustrative 81-frame, 704x1280 clip has about `11 * 22 * 40 * 128 = 1,239,040` codec coordinates. That is approximately 76 times the facet coordinate count. This arithmetic explains why reconstruction requires selective information and a strong prior; it does not establish a required facet count. [G1]

### Preferred capacity escalation

1. Improve pretrained features and preserve coordinates/order through reduction.
2. Use MiniCPM's existing detail mode when needed.
3. Split genuinely dense content into a small number of ordinary 32-facet events.
4. Only then consider a global geometry change.

For a detailed document image, a global overview and one important region may be two observed events on fixed reusable `visual_global` and `visual_detail` tracks, in the same meridian when both are available. This is supported by the one-event-per-track rule without serializing simultaneous evidence unnecessarily. Keep source/crop relations explicit; do not invent one permanent track ID per crop.

Long videos use coherent temporal segments on successive meridians. Neither case introduces per-frame semantic tokens. Generated dependent segments occur sequentially; jointly requested outputs can use a selected group. More events increase total conditioning and KV, so report the **total facets per task**, not only facets per event.

### Trade-offs

| Option | Capacity and cost | Decision |
| --- | --- | --- |
| 32 x 512 | Existing representation and schedules | Default |
| Two 32-facet events where needed | Twice the semantic slots for that content, explicit source structure | First escape hatch |
| 64 x 512 everywhere | Twice raw/KV slots per event; more attention and changed expert/remix geometry | Deferred global change |
| Wider 32-facet vectors | More channel capacity but larger projections/expert math | Deferred |
| Unbounded donor-feature access by R | Avoids the intended bottleneck | Not the architecture |

At fixed per-event history count, doubling facets doubles stored slot count; both query/key dimensions of live attention can double, making that pairwise work four times larger. Overall latency does not necessarily quadruple. A two-event representation also costs additional slots; its advantage is applying capacity selectively and keeping the existing block geometry.

Do not launch a broad 16/32/64 sweep. First compare 32 against two semantically justified 32-facet events on a small dense-content evaluation set. A 64-facet experiment is warranted only if retained information, rather than a weak encoder/renderer or bad supervision, is the demonstrated limit.

Noise robustness matters more than pretending every floating-point bit is usable capacity. Track latent norm, signal/noise ratio, held-out fidelity, and sensitivity to modest perturbations. Do not let an autoencoder defeat its bottleneck simply by increasing latent amplitude without bound.

## 10. Chunking and timing

The reviewed text policy targets 512 reference tokens with a 768 soft and 1,024 hard bound. Use **roughly 256 transcript-equivalent tokens as the initial full speech target**, with 512 as an allowed larger semantic unit. Natural short utterances close underfilled. These are content targets, not a fixed speaking-rate assumption.

A single speech megatoken may require several Whistle windows or many local Pocket frames. Keep encoder windowing, realizer steps, and transport packets separate from semantic commits. The current 60-second raw-audio bound must not be mistaken for the final architectural duration limit.

For video understanding, start around a coherent 16-second event/shot segment, shortening at natural boundaries or high event density. Extend stable content only after temporal-order checks. Begin video **generation** with the selected pretrained pack's tested short-clip geometry, not an immediate mandate to generate 16 or 32 seconds from every B.

Use rational sample/frame rates and a common media-clock origin. Codec frame rounding and output trimming must be deterministic. Predicted duration is a structural output learned from available context; a supplied duration is a public task condition. Ground-truth target length can drive a labeled teacher-forced loss branch, but must not silently become input to free-generation evaluation.

Media timestamps do not change causal availability. Same-interval transcripts, translations, and audio can occupy different meridians. Keep core RoPE at logical meridian time; local temporal/spatial coordinates belong in the modality adapters.

## 11. Unified controls, entirely before the media bottleneck

Keep one `ControlBundle` for instructions, typed settings, reference identities, and per-track overrides. Resolve reference media and encode its features through the appropriate frozen stem plus a bounded learned reference adapter. A content hash alone is not a voice embedding.

Those control/reference features condition P **before** the raw output group is selected. Ordinary context prompts remain equally valid. For the initial implementation, reference features may be a small transient set read during predictor initialization, then folded into its 32-facet queries; do not create an independent persistent control KV history.

The selected Z, followed by C, must contain the intended voice, style, language, and content. R consumes B plus structural output parameters and randomness. Changing a reference after selection cannot retroactively change an already selected event.

Train same-content/different-voice, same-voice/different-content, same-scene/different-style, and composed language/style instructions. Use reference clips distinct from target clips and hold out reference speakers/styles and content compositions. Preserve consent and provenance in the existing control/asset records; no additional authorization architecture is needed for this plan.

For exact TTS, keep the transcript in causal context and score rendered word fidelity. Do not turn it into an R side input to make the auxiliary easier. A separately labeled external text-to-TTS baseline remains useful but is not evidence that the native bottleneck works.

## 12. Training: reconstruction is a bridge, not the destination

### Three complementary computations

**Codec round-trip baseline:** `x -> codec encoder -> U -> codec decoder -> x_hat`. Establish what the frozen codec itself retains before blaming the shared bottleneck.

**Same-path bottleneck reconstruction:** `x -> features -> E -> noisy Z -> C -> B -> R -> U_hat -> codec decoder`. Start with an empty or detached causal history so unrelated context cannot conceal missing information. This trains the modalities to communicate through the same interface used in deployment.

**Integrated causal prediction:** `past tracks + controls -> P -> selected Z -> joint C -> B -> R`. Output targets are not encoded until after P has constructed the causal prior. This is the main capability objective.

A learned prior/encoder correspondence and injected Gaussian noise make the auxiliary VAE-like. A flow loss, detached encoder target, and the repository's separate ownership rules do not automatically constitute a conventional jointly optimized ELBO. Name and report the actual objectives.

### Basic latent loss

Use each pretrained renderer's native objective initially. For a continuous target `U`, a new flow head can use the explicit convention:

```text
s ~ Uniform(0,1), epsilon ~ Normal(0,I)
U_s = (1-s) * epsilon + s * U
v_target = U - epsilon
L_flow = mean_valid_coordinates(||v_theta(U_s, s, B, spec) - v_target||^2)
```

The noisy target `U_s` is legitimate local denoising supervision. It is not a source for E/P or shared KV. For discrete targets, use the corresponding masked code likelihood instead of pretending continuous MSE is equivalent.

Keep teacher latent regression only for aligned target sequences. Add a small number of semantically meaningful checks: speech intelligibility, speaker similarity, image content, temporal order, and audiovisual agreement. Expensive differentiable decoder/perceptual losses can run on a sampled subset; most training should use cached latents.

### Preserve recurrent ownership

Extend the existing grouped bootstrap/recurrent paths and `LossLedger`:

- Bootstrap correspondence compares the detached noisy E distribution with the causal P distribution; normalize once over active facet coordinates.
- In recurrent training, ordinary continuation retains selected P facets; do not re-encode generated output as replacement state.
- E-only reconstruction retains gradients to E through frozen C/R, with its existing independent clipping/ownership.
- Recorded tool results continue only from their recorded actions or explicit supported approximations.
- Frozen donor weights are not updated by accident when C/P unfreezes.

The student cannot train on a ground-truth peer block while another peer is claimed to be prospectively predicted. All generated peers use sampled/selected blocks from the same causal prediction; observed conditions belong to an earlier committed meridian.

### Latent-policy RL becomes simpler under the strict interface

Freeze E, C, the C-to-B dispatch rule, R, codecs, and any decoder state transition when optimizing the latent policy. The control head can remain trainable **if it affects only P before selection**. This removes the current complication in which a trainable control encoder directly changes R's materialization for a fixed selected action.

If renderer noise is sampled from a fixed policy-independent distribution, it is environment/realization randomness, not another learned policy factor. Record it for replay. Continue scoring the joint semantic Gaussian, not an invented waveform/pixel likelihood. Changing renderer weights or sampler behavior requires a new materialization version.

## 13. A focused curriculum and data plan

Use the existing stage engine. Add an explicitly versioned component-adaptation recipe rather than rewriting all broad pretraining budgets.

| Stage | Main work | Trainable owners | Deliverable |
| --- | --- | --- | --- |
| A: interfaces and codec baselines | Extract pinned features/latents; verify independent decode | Small adapters where needed | One checked round-trip fixture per selected pack |
| B: speech bottleneck | Whistle/Pocket acoustic input, facets-only Pocket student, real and teacher speech | E bridge and R conditioning; selected R LoRA | Intelligible controlled speech from selected facets |
| C: visual bottleneck | MiniCPM input, Sana still images, matched SANA video latents | Visual E/R bridges and selected LoRAs | Correct content through 32 facets, not just plausible images |
| D: integrated meridians | Cross-modal conditional tasks and counterfactual joins | Shared C/P, controls, relevant E/R | Output changes correctly when a required track changes |
| E: paired AV and execution | Coupled AV pack, translation, retrieval/tools, long dependencies | Appropriate shared/model and pack owners | Measured cross-track agreement and executed-task performance |

Begin real speech with aligned, clean, licensed material and broaden to conversational delivery, accents, and German/English coverage. Pocket's released recipe and precomputed-latent preparation are useful implementation references, not guarantees that Megatoken can reproduce its results with identical compute. [P4] Add teacher speech for systematic control variation; do not replace all natural speech with teacher outputs.

Use image/video sources with original media, captions/task labels, and clip timing. Add procedural scenes for exact counting, object identities, motion order, and audiovisual edits. Teacher video should supplement rather than define the entire evaluation distribution.

Split at the original recording/book/speaker/video/scene family before crops, translations, teacher outputs, reference snippets, or counterfactual variants are produced. Family siblings remain together. Track dataset rights, teacher/checkpoint identity, and real/synthetic origin in existing lineage records.

### Initial update allocation after adapter warm-up

As a starting scheduler allocation, not a compute-equivalence claim:

- **25%** text/code/tool rehearsal.
- **25%** conditional cross-modal tasks, such as TTS, ASR, captions, and grounded answers.
- **30%** tasks requiring multiple tracks or coordinated output groups.
- **20%** same-path bottleneck reconstruction and teacher distillation.

The adapter warm-up can emphasize reconstruction; once it works, integrated objectives become the majority. Meter actual media seconds, latent elements, selected events, renderer work, and text tokens separately. The current `G` meter is useful for event progress but cannot make a large video event computationally equivalent to text.

## 14. Tasks that force the shared model to do useful work

Prioritize a small set of reusable task families rather than dozens of unrelated datasets.

| Family | Required inputs/outputs | Non-shortcut test |
| --- | --- | --- |
| Speech meaning plus scene grounding | Spoken instruction + image/video -> action/answer | Change only the named object or only the scene |
| Audio/visual evidence join | Each track contains a different indispensable fact -> exact result | Remove or swap each track independently |
| Exact controlled TTS | Earlier text + separate authorized voice reference -> speech | Words stay fixed under voice swap; voice stays fixed under content swap |
| Visual content/style control | Scene request + style reference -> image/clip | Keep entities/actions under style swap |
| Temporal order | Ordered clip + question -> before/after/event answer | Reverse only the relevant actions |
| Translation | Source speech -> target text and then speech | Preserve names/numbers and source coverage across modalities |
| Paired AV | Shared event request -> audio/video group | Match sound source and timing, not just caption similarity |
| Tool/evidence coordination | Grounded media + actual parallel tool results -> synthesis/action | Change arrival order without changing the eventual facts |

Use the existing counterfactual pilot machinery for implementation checks, but do not count tones and bars as evidence of natural speech/video quality.

Two example schedules:

```text
Exact speech:
  g0 observed: requested text + voice reference
  g1 generated: speech Z -> B -> Pocket-derived R
  R never reads the original text/reference directly.

Grounded multimodal action:
  g0 observed: spoken request + scene video
  g1 generated: typed action + explanatory text
  g2 observed: actual action result
  g3 generated: spoken confirmation + visual summary
```

For a later paired AV output, audio and video may be co-generated in one meridian. Dependent exact text-to-speech can use consecutive meridians while playback and new input acquisition overlap. Neither schedule requires a second fine-grained semantic clock.

## 15. Optional component packs and reproducible assets

### A small registry, not a new plugin framework

A modality pack names its perceptual modules, trainable bridges, generator, codec encoder/decoder, and supported output specifications. Load only the necessary subset: transcription need not load a speech decoder; text-only sessions load neither vision nor audio; cached-feature training need not load the original perceptual tower.

Implement `load`, `ensure_loaded`, and `unload` around explicit module ownership. No mid-forward or live-autograd unloading. Inference may offload a pack after work and derivative caches have been released. During training, group work by modality or use gradient accumulation to reduce load/unload thrashing; unloading a trainable pack does not make its optimizer state cease to exist.

Keep adapters and their canonical parameter names stable across residency changes. A loaded module's weights must match the required identity; reloading a different revision into an active session invalidates affected caches and replay.

### Asset and latent contracts

Extend the existing tensor store with a distinct **codec-latent** kind, not a feature tensor mislabeled as a raw target. An input event may reference several declared feature views, while a target names the decoder-compatible U representation.

Each pack/cache identity must include:

```text
upstream repository and immutable revision / weight digest
exact exported submodules and parameter inventory
preprocessing version, sample/frame rates, channels and normalization
feature or latent layer, coordinate ordering and layout
codec scale/shift or mean/std, quantized vs continuous convention
padding, valid spans, crop/frame coordinates, clock origin
renderer objective/parameterization, scheduler and seed protocol
parent media hashes, lineage, reference provenance and permitted use
```

Pocket's 24 kHz waveform must not be silently interpreted as the current 16 kHz asset type. Generalize sample rates and rational video fps in `MediaOutputSpec`, resolved training targets, serving, and serialization together. Keep raw audio, encoded acoustic latents, and semantic features separate.

Video grids need codec-valid spatial/temporal shapes and masks. Tensor limits should reflect declared resource budgets instead of the native test renderer's 32-frame ceiling. A larger allowed shape is not evidence that a pretrained generator learned that horizon.

Cache frozen features and codec targets, not final learned E facets. Final facets become stale as E changes. Reuse immutable original assets; do not rewrite prepared data in place after changing segmentation or codec normalization.

The document's source links identify candidate artifacts, not a completed immutable lockfile. Implementation must resolve and record exact revisions before extraction. Do not invent commit hashes or download gated weights as a side effect of normal model import.

### Licenses as pack metadata

Record code, weight, codec, dataset, and reference rights independently. Pocket's public model card includes weight/access conditions; LTX uses a community license rather than a blanket Apache grant. [P5, G2] Distilling a model or using only its VAE does not by itself establish different redistribution rights. Resolve this once per selected pack through the existing provenance process, rather than introducing a new runtime subsystem.

## 16. Replay, speculation, and decoder-local state

The selected raw Z group remains the policy record. Reconstruct its shared C context to obtain B, then render with the pinned pack/spec/noise. A group tape must include pack identities, codec/scaling identity, all selected raw blocks, output geometry, sampler protocol, and valid output spans.

Decoder RNG streams should be keyed by stable event identity rather than depend on which optional pack happened to run first. A latent seed is not a guarantee of cross-device bitwise equality; distinguish exact raw-action replay, supported-backend latent replay, and perceptual cross-backend equivalence.

Speculation forks the semantic state and any decoder-local state. A new observation or control revision may invalidate undelivered work. Already played audio, displayed frames, or executed tools cannot be erased by dropping a branch. Preserve delivery cursors and interruption records without turning every transport packet into a megatoken.

Grouped speculative acceptance remains atomic when selected peers influenced each other. Optional unloading must not leave a branch pointing to a different version of its renderer or control interpretation.

## 17. Implementation sequence in this repository

### Change 1: enforce the realization boundary

Update `model/state.py`, `model/multitrack.py`, `model/media.py`, and the realizer dispatch to expose one `B[32,D]` per event. Use final C features as the initial shared rule. Remove direct media-R control vectors and raw/early/final concatenation from the new path. Preserve raw tapes and joint C publication.

Add tests that R receives no transcript/source/reference tensors, and that changing the public context or controls after selection cannot alter rendering for fixed B/spec/noise.

### Change 2: extend assets and pack ownership

Extend `data/media_assets.py`, `MediaOutputSpec`, `MediaTrainTarget`, `model/config.py`, export manifests, and `multitrack_service.py` for codec latents, rates, layouts, immutable identities, and optional pack loading. Keep one optimizer owner per trainable weight; do not copy donor modules into each track.

Retain native E/R as lightweight test packs, clearly named as baselines. Build one codec round-trip fixture before integrating a large generator.

### Change 3: speech first

Extend `scripts/cache_omni_features.py` with Whistle and the exact Pocket acoustic view. Add pinned latent extraction and the Pocket-prefix conditioning adapter. Train a direct facets-only student, first on matched speech targets and then on causal text/voice-to-speech meridians.

Finish with intelligibility, voice transfer, sample-rate/normalization parity, missing-view behavior, and full-rollout tests. This is the first useful high-quality modality milestone.

### Change 4: MiniCPM and still images

Export the MiniCPM visual tower/mergers without its language model; implement coordinate-preserving feature caching and detail mode. Add Sana's trained image generator and matching codec through the same B interface. Check content retention before increasing visual budgets.

### Change 5: matched video, then paired AV

Adapt SANA-Video 2B with its original codec, without upgrading latent conventions mid-experiment. Reuse the image-path registry, conditioning interface, loss accounting, and data layout machinery.

Only when paired audio/video tasks are scheduled, add the matched LTX audiovisual pack, separate decoders, and paired realization objective. Do not make this pack necessary to serve earlier milestones.

### Change 6: integrated recipe and evaluation

Extend `training/meridians.py`, `generation/targets.py`, `training/engine.py`, `contracts.py`, and the existing loss ledger with codec-aware targets/objectives. Retain grouped recurrent continuation and separate encoder ownership. Replace raw-media MSE only for pretrained packs; keep the baseline losses for baseline fixtures.

Add actual representation/pack identities to checkpoint and dataset fingerprints. A changed realizer interface or latent interpretation requires explicit versioning, not a shape-compatible silent migration. Update runtime version numbers in the implementation change, not in this documentation commit.

## 18. Small evaluation set, decisive outcomes

Avoid exhaustive alternative-model experiments. For each selected pack, maintain four comparable evaluations:

1. Original pretrained donor under its native conditioning.
2. Codec round-trip without the semantic bottleneck.
3. Same-path E/C/R reconstruction through 32 facets.
4. Causal P/C/R generation from held-out context and controls.

These distinguish poor codec reconstruction, E compression loss, R adaptation failure, and weak P prediction. A single aggregate reconstruction loss cannot locate the problem.

Required checks include speech WER/names/numbers and speaker similarity; image entities/OCR/spatial relations; video event ordering and identity; joint AV event/speaker timing; and counterfactual cross-track sensitivity. Use independent evaluation models where practical and inspect a modest human-scored sample, rather than judging a teacher exclusively by its own scorer.

For architecture tests, retain no-target-leakage, same-meridian influence, track-order permutation behavior, complete group replay, frozen-owner gradients, and correct loss denominators. Add reload parity, feature-rate alignment, masked padding, and normalized-latent decode fixtures.

The initial success criteria are relative and task-based, not invented benchmark scores: a clear improvement over native media baselines, content fidelity approaching the selected donor on the adaptation domain, control swaps that preserve content, and measurable degradation when indispensable input tracks are removed.

Only investigate more facets when the 32-facet compression gap remains after competent encoders and pretrained realizers are working. Only distill large packs further when the current bottleneck-conditioned system already performs the required task.

## 19. Final architecture

```text
Whistle / Pocket acoustic features / MiniCPM / optional codec input views
                                |
                    small modality E bridges
                                |
                     canonical raw [32,D]
                                |
             shared meridian C4 / P12 / experts / KV
                                |
                 ONE contextual [32,D] per output
                                |
          optional pretrained, adapted realization packs
               /                 |                  \
        Pocket + Mimi      Sana + DC-AE       SANA-Video + its VAE
                                                   |
                                  later: paired LTX audio/video pack
```

The central idea is **pretrained competence at the physical edges, a strict and measurable semantic interface, and integrated multi-track learning**. Bigger edge models are allowed and optional. More semantic events are allowed when justified. Neither implies a bigger default facet budget or another cognitive system.

The first deliverable is speech through this exact interface. The next is MiniCPM-grounded image/video generation using compatible pretrained priors. Reconstruction, distillation, and integrated causal tasks are three parts of one learner, not separate architectures.

## References and verification notes

Primary sources below were consulted on 2026-10-19. Component behavior is sourced; interface choices, capacity policy, curricula, and integration steps are proposals. No donor weights were downloaded or models benchmarked for this document.

- **[W1] Whistle model card:** [Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle). Embedding API, quantized artifact size, sample rate, windowing, and license metadata.
- **[W2] Whistle technical description:** [Cactus Compute: Whistle](https://www.cactuscompute.com/blog/whistle). Speech-oriented architecture and frontend context.
- **[P1] Pocket codec/generator configuration:** [english.yaml](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/english.yaml). Inspected file blob `3cfe0b69f765755f1f4e7eab1196a559c2e94824`; configuration includes immutable upstream weight references. Do not assume other language checkpoints share every setting.
- **[P2] Pocket continuous codec interface:** [models/mimi.py](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/mimi.py). Inspected file blob `2656cd57f5516ad54cbe56abdd93a42482ed2388`; `encode_to_latent`, `decode_from_latent`, padding, and `DummyQuantizer`.
- **[P3] Pocket conditioning and normalization:** [models/flow_lm.py](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/flow_lm.py). Prefix embeddings, latent statistics, flow/sampler and EOS conventions.
- **[P4] Pocket training:** [training/README.md](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md). Latent preparation, adaptation, and teacher/student recipes.
- **[P5] Pocket weights/access:** [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts). Published family and reference-voice conditions.
- **[V1] MiniCPM-V 4.7 feature interface:** [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/minicpmv4_7). Visual feature extraction and 16x/4x merger behavior.
- **[V2] MiniCPM-V 4.7 release description:** [OpenBMB documentation](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_v4dot7_en.md). Visual compression and model-family context. Count exported visual parameters directly rather than using a full-model marketing size.
- **[I1] Sana image checkpoint:** [Sana_600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers). Trained compact image generator and matching compression components.
- **[G1] SANA-Video official implementation:** [sana_video.md](https://github.com/NVlabs/Sana/blob/main/docs/sana_video.md). The 720p checkpoint's LTX-2 VAE contract and distinction from the 480p variant.
- **[G2] LTX-2.5 official model card:** [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5). Separate component packaging, convolutional/diffusion video decoding choices, audiovisual model and license conditions.

Repository anchors: [core](../src/megatoken/model/core.py), [grouped model](../src/megatoken/model/multitrack.py), [media model](../src/megatoken/model/media.py), [controls](../src/megatoken/controls.py), [media assets](../src/megatoken/data/media_assets.py), [meridian learner](../src/megatoken/training/meridians.py), [current media pipeline](omni-media-pipeline.md), and [broader multi-track plan](multi-track-plan.md).
