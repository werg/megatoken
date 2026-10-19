# Multi-track Megatoken: architecture and implementation plan

**Status:** Proposed architecture and staged implementation plan; not an assertion that these capabilities are implemented.

**Date:** 2026-10-19.

**Reviewed baseline:** `main` at `a26c5bb02e68563e6d579b2be8d8fbc3b1788bd1`, architecture v6. The next architectural version is referred to provisionally as v7. This document does not change runtime versions, training configurations, or existing architecture decisions.

## 1. The decision in one page

Build **one block-causal transformer over multiple tracks of megatokens**, not a collection of independent transformers connected by bridges.

Every modality enters the central model as the same `32 x 512` raw facet block. A **meridian** is a set of selected megatokens processed together. Its members can attend fully to one another and to earlier meridians. All tracks use the same central processor, predictor, expert banks, and projected historical KV space. Modality-specific encoders and realizers handle the high-resolution work at the edges.

The intended system supports text, speech, images, video, translation, cooperating sub-agents, retrieval, and parallel tools through the same event model. It prioritizes coherent, information-dense chunks over frame-level semantic updates.

| Area | Initial decision |
| --- | --- |
| Canonical representation | Keep 32 facets per megatoken and the existing 512/256 widths |
| Causal unit | An atomic meridian containing one or more megatokens |
| Same-meridian interaction | Joint attention over selected facets, before one final KV publication |
| Prediction | One shared P12 operating on all requested output blocks together |
| Historical memory | One session-level KV store; all 32 slots retained per stored megatoken |
| Temporal structure | Causal meridian index plus separate media timing/alignment metadata |
| Positions | Logical-time RoPE in the core; categorical track relations; spatial/time positions inside modality adapters |
| Chunk policy | Text-scale semantic chunks for every modality; codec frames remain internal |
| Parameter sharing | Share core and prediction experts; share encoder/realizer structure and compatible weights within ownership boundaries |
| Realization | Existing lexical realizer plus an audio realizer and a shared image/video family |
| Controls | One versioned control source with predictor and realizer consumers; ordinary prompts remain first-class |
| Runtime | A ready-event scheduler and bounded speculative branches, not a general distributed graph engine |
| Initial implementation | Multi-track text first, then speech, then images/video, then joint media realization |

The central architectural change is small to state:

```text
v6: selected [32, D]       -> C -> one block of final KV -> R and next P
v7: selected [T, 32, D]    -> C -> T blocks of final KV  -> per-output R and next joint P
```

`T` is the number of active events in this meridian, not the number of allocated tracks. An inactive track creates no placeholder megatoken, no Gaussian draw, and no KV entry.

### What improves on the earlier proposal

The plan makes several choices explicit rather than accumulating mechanisms:

- **Remove the central preview/reprocessing pass.** Observations are committed before predictions that depend on them. Each selected event still receives one central pass.
- **Separate a causal meridian from a media time interval.** Two events can depict the same interval without becoming circularly dependent predictions.
- **Use ordinary block-causal execution instead of a general event-DAG attention engine.** Dependency metadata determines scheduling; the model consumes a simple ordered sequence of meridians.
- **Do not make track IDs a rotary spatial axis.** Track identity is categorical; physical time and spatial coordinates have different meanings.
- **Do not require a hidden high-resolution residual memory.** The semantic path uses megatokens. Detailed assets are explicit references for tasks that request them.
- **Do not partition expert counts prematurely.** Begin with shared banks and modality-aware conditioning. Add specialization only where it produces a demonstrated benefit.

These are design simplifications, not restrictions on the eventual capabilities.

## 2. Baseline and invariants to preserve

The current architecture already has the right main boundaries: ingestion E, central processor C, prediction P, and realization R. E and P produce the same raw facet geometry. C writes final-stage KV and exposes early/final activations. P is read-only with respect to committed history. R reuses C's computation. The implementation and rationale are in [architecture.md](architecture.md), [core.py](../src/megatoken/model/core.py), and [ADR 0004](decisions/0004-contextual-realization.md).

Preserve these invariants:

1. **One recurrent representation.** Every selected event owns its exact raw `32 x D` facets. Generated media is not re-encoded to replace its selected facets.
2. **One central computation per selection.** A meridian receives one joint C pass. Each event contributes exactly one complete final KV block.
3. **One shared historical representation.** No modality-specific historical KV towers and no age-dependent shrinking of stored megatokens.
4. **Pure prediction.** P may construct and mix current output queries but cannot publish them as observations or committed memory.
5. **Explicit realization.** Each output uses selected raw facets, cached central features, declared controls, and declared public references. No silent source-payload bypass.
6. **Explicit randomness and replay.** Raw learned policy draws are authoritative. Any renderer randomness has an explicit, versioned protocol.
7. **Large semantic units.** Audio frames, image patches, video tubelets, and flow iterations are local encoder/realizer work, not additional central megatokens.
8. **One control protocol.** Prompts, typed settings, and reference media join one causal conditioning system, not independent modality controllers.

The v6 full-resolution history decision remains the starting point. See [ADR 0006](decisions/0006-full-resolution-kv.md). The new grouping rule changes how records are processed together, not the resolution of a retained record.

## 3. Terms and the minimal data model

### Session

A session owns one shared context, event log, KV history, control state, and scheduler. Cooperating tracks within the session share context by design. Independent contexts remain separate sessions; the initial implementation does not promise private sub-agent compartments inside an otherwise fully shared context.

### Track

A track is a named source or destination stream: user speech, assistant text, assistant speech, video, a worker, a tool result stream, or retrieved evidence.

Track, role, and modality are different properties. Two audio tracks can represent different speakers. A worker can own several output tracks. The existing source roles continue to distinguish instructions, observations, assistant actions, tools, and documents.

Use stable track IDs for event routing and equality tests. Their numerical spelling carries no learned distance. Descriptive roles and task instructions supply their semantic meaning.

### Event / megatoken

An event is one coherent payload represented by one raw facet block. It has an identity, track, modality, causal group, payload or asset reference, and optional media interval.

Do not put two distinct recurrent blocks inside one event merely to keep its external name. A larger document or video can produce several explicit megatokens under the shared chunking policy.

### Meridian

A meridian is one atomic **selection and integration group**. All participating raw blocks exist before joint C begins. Every member can attend to every other member in that group.

For the first implementation, a meridian contains at most one event per track. Events needing additional causal steps use subsequent meridians. Several tracks may be absent.

### Alignment

An alignment associates content across modalities: source speech and its transcript; translated speech and its caption; video and its audio. It may carry a media clock and start/end offsets.

Alignment does **not** grant causal visibility. It can span multiple meridians. Media time describes the content; the meridian index describes when the model was entitled to use the content.

### Suggested records

The following are schema sketches, not new parallel implementations of existing contracts:

```python
@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    role: str
    description: str
    default_modality: str

@dataclass(frozen=True)
class MediaSpan:
    clock_id: str
    start_ms: int
    end_ms: int

@dataclass(frozen=True)
class EventMeta:
    event_id: str
    track_id: str
    role: str
    modality: str
    meridian_id: int
    alignment_id: str | None = None
    media_span: MediaSpan | None = None
    result_of: str | None = None

@dataclass(frozen=True)
class MeridianSpec:
    meridian_id: int
    events: tuple[EventMeta, ...]
    control_revision: str
```

Extend [data/schema.py](../src/megatoken/data/schema.py) rather than creating a second unrelated episode format. Preserve lineage, source spans, logical-action identities, and supervision separation. Media sample/frame spans need their own representation; they must not be forced into the existing byte-span length invariant.

## 4. Causality: full attention where it is well-defined

### The ordinary execution rule

At causal index `g`, let `H_<g` be committed history. A meridian is either:

- **Observed:** all its raw blocks come from available input payloads or previously recorded native events.
- **Generated:** all its raw blocks are selected jointly from a prediction conditioned on `H_<g` and the current public request.

Then:

```text
observed meridian:
    X_g -> E_modality -> Z_g -> joint C(Z_g, H_<g) -> append final KV

generated meridian:
    P(H_<g, request, controls) -> sample Z_g
    Z_g -> joint C(Z_g, H_<g) -> append final KV -> realize outputs
```

When outgoing content must respond to new observations, first commit an observed meridian, then generate the next meridian. This removes the need to preview C and then process the same observations again.

These two meridians can share an alignment ID and media interval. They need not represent different moments in the depicted scene.

### Same-meridian attention is joint computation, not circular execution

During C, every raw block has already been selected. A layer reads the previous layer's activations for all participating blocks. Ordinary bidirectional attention within this finite block therefore creates no execution cycle.

During P, the live output-query activations also may attend across tracks: they are deterministic functions of available history and requests. They are not ground-truth targets or already sampled future output. Mixing them is ordinary joint prediction.

The distinction is simple:

```text
Allowed in P:    prior history, public requests, controls, live prediction queries
Not allowed:    encodings of this meridian's ground-truth outputs
Allowed in C:    every already-selected raw block in this meridian
```

### Mask semantics

For selected-event processing, the conceptual mask is:

```text
same session AND (key.meridian < query.meridian OR same current selected meridian)
```

For prospective prediction, historical keys must have `key.meridian < g`; the additional live-query attention is confined to the current P computation.

Do not use a lower-triangular token mask within the `T * 32` selected facets. That would impose an arbitrary track order and defeat same-meridian attention.

The training loop should initially execute meridians explicitly. A flattened, one-pass training representation must reproduce this computation, including P's restriction on target-derived states; merely assigning equal RoPE positions is not sufficient.

### No global barrier

A meridian includes the events ready for joint processing, not every track in the session. If one tool is pending, other tracks continue. If two results are ready, the scheduler can group them. A late result enters a later meridian with its original source timing intact.

If a critic must inspect a worker's actual answer, they run in consecutive meridians. If two workers should explore independently from common context, they can be co-generated. The implementation does not need a special agent communication mechanism for either case.

## 5. Shared C and P: exact tensor and computation changes

### Shapes

Use a padded reference representation first:

```text
selected raw:       [B, T, 32, D]
active-event mask:   [B, T]
central live slots: [B, T * 32, D]
historical KV:      [B, kv_heads, retained_events * 32, head_dim]
```

Ragged packing is an optimization, not a prerequisite. Padding must contribute neither KV, loss, routing exposure, nor policy density.

### Central processor

Keep C4, early features after C2, the three temporary writers, and the single final historical writer. Change the current-slot dimension from 32 to `T * 32` for attention and temporary KV. Repeated facet addresses identify the 32 within-event slots; role/modality descriptors and track relations distinguish events.

All current events participate in the same C layers. Reshape early and final outputs back to `[B, T, 32, D]` for the realizer and next prediction. Publish the entire group's final KV once.

This is an extension of [DenseFacetBlock](../src/megatoken/model/blocks.py), [AttentionRead](../src/megatoken/model/attention.py), and `CentralProcessor`, not a new cross-modal fusion tower.

### Predictor initialization

Maintain one **frontier**: the last committed meridian's raw/early/final blocks, just as v6 stores the last selected block. Do not introduce private learned recurrent states or private KV caches for every track.

For the first version, form a transient per-facet mean over the frontier's active final blocks, apply the existing remix, and broadcast it to the requested output blocks. Add target modality, role/task descriptors, control conditioning, and effort. Each output query is then processed by the same P12.

```text
base[f]       = mean_active_events(frontier.final[:, f, :])
query[t]      = remix(base) + target_descriptor[t] + control[t] + effort
mean[1:T_out] = joint_P12(query[1:T_out], H_<g)
```

This mean is only an inexpensive predictor initializer. It does not replace any stored megatoken, average historical keys, or become an additional recurrent memory. P still reads all retained full-resolution KV. With a singleton frontier it reduces to the existing initialization.

A track's older context is available through shared history and same-track addressing, rather than requiring another persistent state object. Add a more elaborate seed read only if a concrete retrieval/continuation failure warrants it.

### Attention and experts in P

Keep the existing P3/P6/P9/P12 historical-read schedule initially. Make live-query attention span all current requested blocks, including in the stages without historical reads.

Keep pair-expert topology **within each 32-facet block**. Flatten event rows into the dispatch batch for the existing packed expert kernels; do not invent all cross-track facet-pair combinations. Cross-track communication is attention's job.

This preserves the existing pair schedules, effort levels, expert widths, and most kernel code. The 64-expert P bank remains one shared parameter owner.

The router compiles once per requested output block from its causal initializer, modality/task descriptors, and fixed row/stage descriptors. It must not depend on unrelated batch members, target payloads, or physical track array order.

### Joint raw distribution

Extend the current Gaussian to `[B, T_out, 32, D]`. Start with conditionally factorized coordinates, retaining the current tractable density:

```text
Z_g = mean_g + sigma_g * epsilon_g
log p(Z_g | H_<g, controls, request)
    = sum over active events, facets, coordinates of Gaussian log density
```

Select the complete group before C. Independent raw innovations do not force independent rendered outputs: joint C and joint realization can couple the resulting media. The initial raw prior nevertheless is factorized; the plan does not claim that arbitrary latent correlations have been modeled.

Use stable event identities to derive reproducible RNG substreams. Reordering the physical event axis, while preserving identities and metadata, should not change which random draw belongs to which event.

The active output set is specified before this draw. If the model later chooses to spawn tracks or alter topology, express that as an ordinary preceding typed action, so its selection is included in the recorded policy trajectory. Do not hide a separately sampled scheduler decision outside the policy accounting.

## 6. One KV space and simple position semantics

### Historical storage

Keep the shared final-stage K/V projection. The KV space contains **Megatoken-processed facets**, not a concatenation of caches copied from unrelated pretrained speech and vision transformers. External encoders must project into the facet interface before entering C.

Store event metadata alongside KV blocks: event/track IDs, role, modality, meridian ID, and optional media timing. Most metadata is per megatoken, not duplicated as a heavyweight object per facet.

Keep a global capacity measured in retained megatokens. For simplicity, evict the oldest complete meridians until a new group fits. This preserves all 32 slots of every retained event and avoids partially retaining an atomic group. A group larger than the configured capacity is invalid.

No track-specific retention policies or compressed archive are needed initially. Add those only after observing a real context-budget problem.

### RoPE and identities

Use **meridian index as the core RoPE coordinate**. All facets of all events in a meridian have the same logical position. Keep the existing base and numerical construction initially; architecture changes do not require simultaneously retuning frequency schedules.

Track identity is categorical. Use source/task descriptors and a learned same-track attention bias, initialized neutrally, rather than rotary arithmetic on integer track IDs. This works for new runtime track IDs and avoids implying that track 6 is geometrically nearer track 7 than track 2.

Maintain learned facet addresses. Facet indices are latent identities, not automatically image grid coordinates.

Physical media time remains explicit, separate metadata. Encode known interval start, duration, and within-clip timing in the relevant adapters and conditional features. Compare times only within a common clock. Do not use future ground-truth duration, timestamp, speaker identity, or alignment information as a prospective input unless the task explicitly supplies it.

Inside visual adapters, use spatial or spatiotemporal positional encoding over actual patch/latent coordinates. Audio uses its actual local time axis. The core does not need a universal two-dimensional layout imposed on every modality.

Qwen3-VL's interleaved MRoPE and timestamp treatment illustrate that modern multimodal position schemes distinguish spatial and temporal information; they do not establish ordinal track IDs as a useful rotary axis. This plan uses the simpler core scheme because its tokens are already semantic chunks. [R1]

### Implementation detail

Generalize `temporal_rope` and `AttentionFeatures` so positions, key validity, and track relations can be broadcast over multiple current blocks. A validity vector alone is no longer the whole attention contract once query-dependent relations are added.

Keep an uncomplicated dense/reference mask builder for correctness. Optimize block masks and kernels later. A logical shared KV space does not require one continually reallocated physical tensor; a ring buffer or paged allocation can preserve the same semantics when justified.

## 7. Chunking: comparable semantic density across modalities

### Common policy

Use the current text chunking policy as the reference, not audio-codec or video-frame counts:

- Text remains centered on 256 reference-codec tokens, with the existing 512 soft and 1,024 hard limits.
- Conversational audio initially targets roughly 128 transcript-equivalent tokens per megatoken, allowing about two audio chunks for one typical 256-token text chunk.
- A complete short utterance, image, tool result, or scene may naturally be underfilled.
- Media resource limits constrain encoder/realizer work independently of the lexical 1,024-position limit.

These are initial operating targets, not measured semantic entropy. Do not add a learned information-density estimator before ordinary segmentation has been tested.

For aligned speech/text training, use transcript boundaries and timestamps to produce corresponding spans. Preserve the ability to pack two adjacent speech-aligned spans into one larger text megatoken; alignment is therefore not restricted to a one-to-one diagonal grid.

For inference without a transcript, use utterance boundaries and a duration budget calibrated on the training corpus. No mandatory external ASR system is needed to decide every input boundary. An available transcript can improve segmentation, but is not secretly required by speech understanding.

Short requests close at natural utterance boundaries rather than waiting to fill 128 tokens. Long speech chunks do accumulate real listening time; speculative computation can reduce subsequent processing delay, not make future speech arrive sooner.

### Video policy

Represent a coherent clip with one megatoken rather than promoting every frame to the core. Begin with shot/event boundaries around a configurable coarse duration budget. An initial sweep of 8, 16, and 32 seconds is sufficient to find a useful setting; these durations are experimental choices, not codec limits.

Prefer extending a visually stable clip and splitting a densely changing sequence. Initially implement this with duration bounds, shot boundaries, and simple frame-change statistics. Do not introduce a separate learned scene-planning network.

Keep codec/feature sampling rate separate from chunk duration. A 16-second megatoken may contain many internal frames or tubelets. Lowering internal resolution reduces encoder cost without changing how many causal events the central model receives.

Preserve event ordering and timestamps during temporal downsampling. A longer chunk should still distinguish opening a door before entering from entering before opening it.

### Fingerprints and provenance

Extend the existing chunking fingerprint with modality, codec/feature-extractor identity, media sampling policy, and segmentation policy. Prepared native tapes remain immutable; never silently resegment generated events to fit a changed chunk target.

Track exposure in meaningful units: text tokens, audio seconds, image count/pixels, video seconds/frames, and selected megatokens. A codec frame is not an interchangeable training-budget unit with a reference text token.

## 8. Ingestion: specialized stems, common reduction and facets

The encoder contract remains:

```text
E_modality(payload, declared format) -> raw [32, D]
```

E does not read future targets, other tracks' history, or prospective output controls. Cross-track interpretation happens in C. Modality-specific preprocessing may be bidirectional within a completed input chunk; only available samples may be used for a provisional prefix.

### Shared shape of the encoders

```text
modality stem
    -> ordered local features with coordinates/masks
    -> hierarchical reduction
    -> bounded 32-query readout
    -> shared facet refinement
    -> modality-specific final calibration/projection where needed
    -> canonical raw facets
```

Reuse the existing reduction and `DenseFacetBlock` building blocks where their assumptions match. Share implementation before forcing every low-level weight to be identical. The final facet-resolution refinement is the first place to try weight sharing across modalities.

### Text

Keep the ordered byte/ngram stem and current hierarchy. All transcripts, captions, code, tool JSON, retrieved text, and textual controls reuse this same text machinery.

### Audio

Use a fixed or pretrained audio feature/codec stem for the first audio experiment. Preserve phonetic content, timing, speaker information, and non-speech acoustic cues rather than converting audio only to its transcript.

Project temporal features to the shared source width, then apply ordered temporal reduction and the common facet readout. Tens of seconds of codec frames can be internal input to one encoder call.

Mimi is a concrete reusable codec baseline; its 12.5 Hz representation is an internal rate, not a proposed Megatoken clock. [R2] The abstraction should also admit continuous speech features. No pretrained audio transformer's historical cache enters the core.

### Images and video

Use one visual family. An image is the one-frame case at the interface; codec-specific padding or minimum-frame requirements remain local implementation details.

A video stem can use a pretrained spatiotemporal VAE or visual feature encoder. Apply local spatial processing and ordered temporal reduction before the 32-query readout. Preserve spatial and temporal coordinates until the readout; do not merely average frame embeddings.

Open-Sora-Plan provides a concrete image/video VAE precedent with joint spatial-temporal compression. Its released v1.0 report describes 4 x 8 x 8 compression and joint image/video handling. These are reusable mechanisms, not evidence that our additional 32-facet compression is already validated. [R3]

Start with one visual encoder path. Separate appearance encoders, optical-flow networks, object trackers, and learned event segmenters are not initial architectural requirements.

### What the bottleneck is supposed to preserve

The core should preserve meaning, entities, relationships, temporal order, and enough generative specification to realize the requested content. It is not a lossless container for arbitrary original waveforms or pixels.

Do not force the entire audio or video latent to equal its transcript latent. That would discard voice, motion, and appearance information. Use cross-modal task supervision and alignment of semantic projections while retaining modality-specific information in the full facet block.

For exact editing or reference copying, keep the original media as a declared asset, addressable through the normal reference mechanism. The default semantic path must work without a mandatory residual side channel. Evaluation should distinguish generation from facets alone from reference-conditioned editing.

## 9. Realization: different outputs, one architectural family

Yes: audio, images, and video can be generated through different realizers while sharing the same central model.

The stable boundary is:

```text
R_modality(
    selected raw facets,
    selected early/final C features,
    unified control context,
    output specification,
    explicit public references,
    renderer randomness
) -> payload
```

Realizers may use many internal tokens and several decoding steps. They do not write semantic KV, create hidden megatokens, or run another general-purpose agent policy.

### Text realizer

Keep the current deterministic lexical realizer, structured decoding, exact copies, and schema constraints. Retain its existing output limits and losses.

### Audio realizer

Use chunk-level conditional generation of an acoustic representation followed by a fixed codec decoder or vocoder. For the first substantial speech implementation, conditional flow matching over a vocoder-compatible continuous representation is a strong default: it supports whole-chunk generation without making every frame a central autoregressive action.

F5-TTS provides a concrete flow-matching speech recipe and training code. Its conditioning architecture is a reference, not a requirement to transplant a second language model. Its code and pretrained weights have different licenses, so choose assets deliberately. [R4]

A codec-token output head is also compatible with the interface, but is not an additional mandatory backend. Choose one acoustic target representation for the first training run and keep it pinned.

For exact TTS, the earlier selected text and any declared exact text reference are available as causal content constraints. Otherwise the audio realizer must learn to render the content selected in its facets, not independently invent another response.

### Shared image/video realizer

Use a visual latent generator conditioned on raw/early/final facets and unified controls, followed by a fixed visual decoder. Treat the one-frame case and multi-frame case as one family with shared spatial processing and temporal support for video.

Flow matching gives numeric media realizers a common training and inference structure. Different latent grids and output projections do not require different central reasoning models. Transfusion is a useful precedent for keeping discrete text and continuous image generation under a common model while retaining modality-specific edges. [R5]

### How much of R should be shared

Start with a common conditioning interface, facet-conditioning blocks, normalization, attention/FFN implementations, and training harness. Share image/video weights from the outset where the codec permits it. Keep audio-specific input/output projections and local time processing.

Where feature widths match, try the existing U bank as shared facet-conditioned refinement. Do not require every high-resolution denoising layer to use the text U weights. Parameter sharing that changes optimization ownership is not free.

A renderer can be bootstrapped from pretrained media components and later distilled into a smaller shared family. That is a deployment/training choice at R, not a reason to alter the core event model.

### Synchronized audio and video

Generate audio and video as distinct megatokens in one generated meridian, under a shared alignment and duration specification. Joint C provides semantic agreement. The realizers use the same media clock and duration budget.

Fine acoustic/visual alignment is a realization task as well as a core task. For the joint-media milestone, permit the audio and video latent grids to exchange information inside a shared media-realization block or coupled flow solver. This remains bounded work for one selected meridian; it does not become a second historical KV system.

LTX-2 is a concrete reference for joint audiovisual generation and released fine-tuning infrastructure. Its existence supports reusing media-generation techniques, not assuming that two independently trained heads will automatically lip-sync. [R6]

Begin with coarse synchronized clips and consistent content; add fine lip synchronization through paired realization training. The core architecture does not change between those quality levels.

### Output shape and duration

Keep duration, dimensions, sampling rate, and format in an `OutputSpec`. Initially use requested or dataset-task-supplied output budgets for media experiments. For unrequested variable duration, add a supervised deterministic duration/length head using the existing structure-head pattern.

Ground-truth output length may be used in an explicitly teacher-forced realization branch, but must not silently become public conditioning for an evaluated free-generation task. Do not rotate a generated event's historical keys using a future duration that was unknown when it was selected.

## 10. Parameter sharing and engram reuse

### Sharing boundaries

| Parameter family | Initial policy |
| --- | --- |
| Byte/ngram table | Shared by all textual material; retain its byte semantics |
| Audio/visual input embeddings and codec stems | Separate |
| Reduction block implementation | Shared; tie compatible weights after modality projection where useful |
| Facet-resolution encoder refinement | Shared within E, with small modality-specific calibration |
| Central C and final KV projection | Fully shared across tracks and modalities |
| Predictor P and P expert bank | Fully shared, modality/task-conditioned routing |
| Realizer conditioning and block implementation | Shared |
| Image/video realizer | Shared family |
| Acoustic/visual output heads and codecs | Modality-specific |
| U expert bank | Shared candidate for compatible refinement; no fixed modality quota initially |

The current P bank acts on facet pairs of width `2D`; U acts on rows of width `D`. They are not interchangeable tables to tie directly. Reuse their structure and share each bank across modalities, while keeping distinct P and R ownership. See [experts.py](../src/megatoken/model/experts.py) and [routing.py](../src/megatoken/model/routing.py).

Do not create a full P12 for every modality or agent. Do not preallocate an arbitrary number of experts to each modality before seeing routing and gradient statistics.

Add modality/task descriptors to routing. Report utilization and task gradients by modality as well as in aggregate, since large visual workloads can otherwise dominate a superficially balanced bank. Loss normalization should prevent that before introducing more routing machinery.

### Engrams

The existing table contains exact/hashed byte n-grams. Directly identifying an audio-codec index or visual code index with a byte n-gram row has no semantic justification.

Useful reuse is available without that identification:

- All transcripts, captions, prompt controls, names, code, and tool descriptions use the existing table.
- Modality-specific embeddings project into shared feature and facet spaces.
- Shared experts learn relationships among those spaces.
- Cross-modal objectives align semantic features rather than arbitrary raw token IDs.

A future multimodal lookup-memory experiment could share lookup machinery with namespaced keys, but merely putting separate namespaces into one allocation would not demonstrate useful parameter transfer. It is not part of the first plan.

### Ownership is part of the architecture

Continue to distinguish E, C, P, R, and control owners in the optimizer/export audit. Share across modalities **within** those families first. Do not tie E and P weights, or P and R experts, in a way that makes the existing encoder-only auxiliary or frozen-realizer RL contract ambiguous.

## 11. One central control system

### One source, several consumers

Expand the existing [controls.py](../src/megatoken/controls.py) protocol, rather than inventing separate voice, visual-style, agent, and retrieval controllers.

A control bundle contains:

```text
original instruction bytes
canonical typed settings
explicit reference assets or earlier event references
optional per-track overrides
provenance and revision identity
```

Examples of typed settings include language, output modality, authorized speaker reference, speaking style, visual style, duration budget, worker role, and tool schema. Track overrides are views of the same control object, not independent control systems.

Keep precedence deterministic: session defaults, then request settings, then explicit track overrides. Control updates affect future selections, never retroactively alter a selected event's rendering contract.

### Representation

Retain a pooled control summary for the existing predictor injection. Add a small bounded set of control/reference features for realizers to attend to. A concrete starting point is eight control slots, with the width set in configuration, plus the existing pooled summary.

Text uses the shared byte frontend. Audio and visual references use their modality stems and a bounded reference readout. These features join one control encoder and versioned `ControlContext`.

The reference path is explicit conditioning, analogous to the existing public schema/copy mechanism. It is not an undeclared observation history or a hidden store of the target's original media.

### Prompt support

Ordinary instructions in context remain valid. The model should understand a request such as:

> Translate this speech into English, read it in the approved narrator voice, and produce a watercolor visual summary.

Do not require a fragile natural-language-to-JSON parser before the model can act on such prompts. P reads ordinary context; R receives the resulting central features and unified controls. Structured controls provide precision, reproducibility, and an API surface when the caller has them.

When the model itself chooses a new control setting or creates tracks, use a normal preceding typed action. This reuses the action machinery instead of adding a hidden control policy.

### Training

Use paired prompt/structured requests, paraphrases, composed controls, and reference examples. Train the same content under different voices/styles and different content under the same voice/style. Use separate reference and target segments so the model learns reference transfer, not target copying.

Train control dropout and absent-control cases, preserving exact neutral behavior. Keep speaker identity, prosody, visual identity, and style distinguishable within the unified representation rather than forcing them into one undifferentiated scalar label.

Carry reference authorization and source provenance in the existing asset/lineage machinery. This is a normal part of the control contract, not a separate architectural subsystem.

## 12. Randomness, grouped tapes, and latent-policy RL

### Policy randomness versus renderer randomness

Keep one learned Gaussian policy draw for the complete generated meridian. A media realizer may additionally consume base noise under a fixed rendering protocol. That noise is not another learned agent policy.

A deterministic solver with an explicit seed is deterministic conditional on its complete environment and versions, not necessarily bitwise identical across arbitrary hardware. Record the codec/renderer version, solver settings, PRNG protocol, and seed or base noise as needed for the supported replay guarantee.

A frozen stochastic renderer is compatible with latent-action RL when its conditional output distribution is policy-independent. The Gaussian log density remains the learned action density. Do not pretend to have computed an intractable likelihood for the rendered waveform or video.

### Meridian tapes

A generated meridian needs a grouped tape containing:

```text
pre-state/history identity and meridian identity
complete active output specifications and stable event order
all exact FP32 raw facet blocks
joint behavior log probability and per-event density contributions
control source bytes, typed fields, reference hashes, and revisions
codec/renderer identifiers and realization randomness
payload hashes, validity, terminal state, and typed-action metadata
```

Replay joint C with the complete selected group. An event's contextual features depend on its peers, so replaying only its own raw block is not generally equivalent.

Native encoder-following labels must identify the group and rendering context. There can be multiple valid latent explanations of the same payload, particularly with reference-conditioned or stochastic rendering; native following remains an auxiliary, not proof of an invertible codec.

### Important control-ownership change

In v6, the trainable control head feeds P, while the RL action-materialization map is frozen. If the expanded control encoder also feeds R directly, freezing R's weights alone no longer freezes that map.

The simple rule for latent-only RL is:

**Freeze E, C/KV, R/codecs, and the shared control/reference encoder that R consumes. Train P, its predictor-side control injection, routing, remix, and Gaussian parameters.**

Supervised training can train the unified control representation and realizers. Alternating supervised and RL phases must rebuild caches and version assets as today. No duplicate trainable-versus-frozen control system is required.

## 13. Training curriculum and losses

### Extend the current trainer, not replace it

The current bootstrap/recurrent distinction and explicit gradient ownership in [training.md](training.md) remain useful. Extend the prepared target representation to grouped multimodal events, keeping main prediction, encoder reconstruction, and recorded-action continuation as separate branches.

For a generated training meridian:

1. Construct its causal history and public output request.
2. Run joint P and select all raw prior blocks without reading target payloads.
3. Run joint C on those selected blocks and apply the appropriate output losses.
4. Separately encode targets for reconstruction/bootstrap objectives.
5. Advance the live trajectory according to the configured bootstrap or recurrent mode.

Never decode one predicted track while giving its central context another jointly predicted track's ground-truth encoding. If that other modality is an intended observed condition, put it in a preceding observed meridian.

A flow realizer legitimately receives a noisy target during its denoising training objective. That is local realization supervision; the noisy target and its features must not be promoted into P's context or persistent semantic KV.

### Loss families

Use only the losses needed by the active task:

| Objective | Purpose |
| --- | --- |
| Existing lexical/structure/copy losses | Text and exact typed actions |
| Codec reconstruction or continuous-media flow loss | Learn the modality's realization and ingestion correspondence |
| Existing bootstrap Gaussian correspondence | Initialize the common raw latent interface |
| Cross-modal supervised generation | Transcription, TTS, captioning, grounded answers, translation |
| Small semantic-alignment objective | Align meaning without collapsing all modality-specific information |
| Paired audiovisual timing/consistency loss | Train synchronized realization |
| Existing routing auxiliaries | Stable shared expert training |

Normalize within each modality before combining task weights. Do not let a million visual elements overwhelm one text response merely because their raw loss sum is larger. Keep separate counters and denominators for modalities, selected events, and stochastic facet coordinates.

Retain the current encoder-only recurrent auxiliary with C/R owners fixed for that branch. Shared E refinement is still E-owned. Preserve the recorded-tool continuation rule: an offline result belongs to its actual recorded call, not to an unrelated sampled substitute.

### Causal controls first

The existing trainer has an explicitly noncausal, successful-trajectory reference mixture. Do not silently inherit it into the first multi-track curriculum. Use causal prompt/reference conditioning in the new baseline; keep any privileged-reference experiment separately configured and separately reported.

This makes evaluation interpretable without removing the legacy training capability.

### Staged curriculum

**MT0: multi-track text.** Use existing text/code/tool examples and synthetic tasks with two to four tracks. Train copying across tracks, joining independent results, worker/critic ordering, and evidence-conditioned answers. Include singleton meridians throughout.

**MT1: text and speech.** Bootstrap audio E/R with paired speech/transcript data, then train transcription, exact TTS, audio-conditioned answers, and chunk-aligned translation. Keep text capability replay in the mixture.

**MT2: images and video understanding.** Train image captions, questions, temporal ordering, event descriptions, and cross-track audiovisual grounding. Start with the same visual stem for images and clips.

**MT3: visual and joint media realization.** Train image generation, short clips, reference/style controls, then paired audio/video output under common duration/alignment conditions.

**MT4: interactive execution.** Train on actual concurrent tools, retrieval results, bounded sub-agent workflows, and speculative branch acceptance/cancellation. Use recorded runtime causality, not an idealized transcript reordered after execution.

No stage requires a corpus containing every modality simultaneously. Paired tasks share the core and control interface; joint-media data is used where synchronized generation actually needs it.

## 14. Data and reusable recipes

There is enough accessible material to build and test this architecture without waiting for a universal omnimodal dataset. The first deliverables do not depend on large natural full-duplex conversation corpora.

| Need | Initial source/recipe direction | Integration work |
| --- | --- | --- |
| Multi-track reasoning and tools | Existing repository preparation/collection plus controlled text fixtures | Track assignment, actual dependencies, grouping, and joins |
| Speech/transcripts and voice references | LibriTTS-R as a concrete first corpus | Preserve speaker/session lineage; join only genuine adjacent material into longer spans |
| Chunk speech generation | F5-TTS-style flow training and a pinned vocoder/representation | Replace/adapt conditioning to facets and unified controls |
| Audio features/codecs | Mimi or a pinned continuous audio stem | Map local features to the shared encoder interface |
| Image/video compression | Open-Sora-Plan-style visual VAE/feature components | Ordered long-clip reduction and 32-facet readout |
| Visual and audiovisual realization | Transfusion's objective separation; LTX training infrastructure | Facet conditioning, paired timing, and shared renderer interfaces |
| First vision correctness tasks | Procedurally rendered shapes, scenes, motion, captions, and event questions | Exact temporal labels and controllable information density |

LibriTTS-R is published under CC BY 4.0 and is directly downloadable. [R7] It provides a tractable initial speech dataset, not a complete conversational or general-sound curriculum. Do not concatenate unrelated utterances and present the result as natural continuous dialogue.

Use procedural visual/audio-text tasks to test the representation and masks before acquiring a broad media corpus. Expand with selected paired image/video and environmental-audio datasets once the adapters work. This is an acquisition task in the existing data system, not a prerequisite for implementing shared tracks.

Pin codec/model revisions, preprocessing, data revisions, and source rights in the manifest. Code licenses, weight licenses, and dataset terms are distinct. F5-TTS, for example, explicitly distinguishes MIT code from noncommercial pretrained weights. [R4]

The cited projects supply techniques, components, or fine-tuning recipes. None is a drop-in training recipe for this specific multi-track 32-facet architecture.

## 15. Runtime: concurrency without a second architecture

### Initial scheduler

Use a small ready queue and a fixed upper bound on active output tracks. A practical starting configuration is eight allocated tracks and up to four events per meridian, with tests at widths one, two, and four. These are deployment defaults to measure, not architectural constants.

The host determines which observations are ready and which output tracks are requested. A controller can later request track creation through the existing typed-action path. There is no need for a learned low-level scheduling network in the first release.

A minimal API is:

```text
open_session(track_specs)
observe_group(completed_events)
generate_group(output_specs, controls)
status()
close_track(track_id)
close_session()
```

Keep the current single-observe and single-generate API as wrappers around singleton groups. Track lifetime is metadata; opening an empty track does not append a phantom event.

### Parallel tools

A generated meridian can contain several complete tool calls. After validation and acceptance, the executor launches them independently. Each result keeps `result_of` and its execution identity and is observed when ready.

A dependent follow-up waits for its input. Independent work does not. The runtime must genuinely dispatch concurrent external requests to claim parallel execution; the shared architecture only enables it.

### Sub-agents

A sub-agent is a track with a task/role description and requested output contract, using the same weights and context. Co-generate workers where useful; let critique, synthesis, and dependent reasoning consume their committed results later.

Do not add private transformer copies, bespoke agent messages, or a separate communications bus. Different trust/context boundaries can use different sessions when needed.

### Retrieval

Retrieval returns ordinary evidence events. Encode them through the same E/C path, retain provenance and exact source assets, and let all cooperating tracks attend to their KV.

A retrieval projection can be trained from facet-derived semantic features. Do not assume the existing reconstruction representation is already an optimal nearest-neighbor index. The retrieval service and index remain external storage, not another transformer history.

Retrieved text remains source evidence, not a control override. Existing roles and public-control rules should make that distinction explicit.

### Progressive media output

A realizer can emit transport packets or render internal segments from one selected megatoken. These packets do not each create a core event. This preserves large semantic chunks while allowing progressive playback or display.

## 16. Speculative generation and delivery

Implement speculation after the deterministic grouped path works.

Start with one bounded speculative generated meridian from a snapshot. Its raw facets, central features, renderer work, and KV append live on a copy-on-write branch. The canonical session is unchanged until acceptance.

```text
snapshot -> predict/sample group -> joint C on branch -> render candidate

accept only if the input/control/history revisions still match
otherwise discard the branch and generate from the updated state
```

This is simple stale-work rejection, not distribution-preserving speculative decoding with an acceptance-ratio theorem. Report it as such.

Any new relevant input or control revision invalidates the branch by default. No learned branch-validity classifier is required. Receiving new media fragments can mark work stale even before their larger observation chunk is closed; authoritative ingestion still follows the normal chunk policy.

Once any output has been externally delivered or a tool has executed, do not erase the corresponding selected group. Record interruption, partial playback, or execution outcome as part of the subsequent observation/runtime ledger. A model may remember its complete planned utterance while also being told that only a prefix was heard.

Delivery cursors are runtime metadata, not per-frame megatokens. Surface a relevant delivery change with the next semantic observation. This avoids both a fine-grained alternate recurrence and pretending that undelivered output was heard.

Because peers share joint C, speculative acceptance is group-atomic. Discarding one peer and keeping the others as though it never participated would generally change their conditional computation.

Prefix re-encoding for provisional input can be added behind this same branch interface later. It is not required by the core design and must not mutate committed KV in place.

## 17. Worked schedules

### Speech, transcript, and a spoken answer

```text
meridian 0: user_audio_0                         [observed]
meridian 1: transcript_0 + answer_text_0          [jointly generated]
meridian 2: answer_audio_0                       [generated from selected answer]
```

The transcript and answer can share a selected group because both are predictions from observed audio. The speech output follows selected answer text when exact content agreement is wanted. Playback can overlap computation for later chunks.

For a task that does not require exact text/audio matching, text and speech may instead be co-generated in one meridian and trained jointly. Use the ordered schedule as the initial exact-TTS default.

### Chunk-aligned simultaneous translation

```text
meridian 0: source_audio_A                       [observed]
meridian 1: source_transcript_A + target_text_A   [generated]
meridian 2: target_audio_A                       [generated]
meridian 3: source_audio_B                       [observed when ready]
...
```

Receiving source B can overlap rendering/playback of target A. The scheduler orders completed events by actual availability rather than blocking source input until speech playback ends. Alignment IDs connect the translations to the original intervals even when target speech has a different duration.

This is simultaneous operation at chunk scale, not a promise of word-level delay. More source lookahead is a task/scheduling choice, not a new model clock.

### Video and audio generation

```text
meridian 0: scene/request text                   [observed]
meridian 1: video_A + audio_A                    [jointly generated]
meridian 2: video_B + audio_B                    [jointly generated continuation]
```

Both media blocks on a meridian attend during C. The media realizers share clock/duration conditions and, when enabled, joint realization blocks. Continuation conditions on prior selected blocks and declared earlier media references.

### Workers, tools, and retrieval

```text
meridian 0: task + retrieved_evidence            [observed, both available]
meridian 1: worker_A_call + worker_B_call         [jointly generated]
             -> execute both independently
meridian 2: result_B                             [observed first]
meridian 3: worker_B_followup                    [generated]
meridian 4: result_A                             [observed later]
meridian 5: synthesis                            [generated]
```

No empty record stands in for result A while it is pending. A later synthesis can use the complete shared history.

## 18. Implementation plan mapped to this repository

Implement this as a sequence of reviewable changes. Each step should produce a working capability rather than merely another abstraction.

### Milestone A: multi-track text reference path

**Scope:** grouped event schema, shared state, C/P grouping, positions, masks, and basic generation. No media dependencies.

| Location | Change |
| --- | --- |
| `data/schema.py` | Add tracks, meridians, alignment metadata, and grouped episode validation |
| `model/state.py` | Represent a grouped frontier and shared history metadata; retain one session state |
| `model/attention.py` | Support grouped positions, same-track relations, and block-causal masks |
| `model/blocks.py` | Apply existing current/temporary attention to all current selected slots |
| `model/core.py` | Add `advance_group` and `predict_group`; keep per-event pair topology and shared banks |
| `generation/policy.py` | Joint Gaussian selection and grouped materialization |
| `generation/structure.py` | Versioned meridian tape and output specifications |
| `inference.py` | Group operations with singleton compatibility wrappers |

**Deliverable:** a two-to-four-track text model can combine evidence, generate parallel worker outputs, and replay its grouped decisions. Include a small training run, not only tensor-shape tests.

### Milestone B: unified controls and data/training integration

**Scope:** bounded multimodal-ready controls, grouped target plans, ownership auditing, and loss accounting; still runnable on text fixtures.

| Location | Change |
| --- | --- |
| `controls.py` | Canonical bundle, reference identities, overrides, pooled/slot context |
| `model/frontend.py` | Preserve byte/ngram reuse; extend bounded control readout |
| `generation/targets.py` | Grouped targets and explicit per-modality denominators |
| `training/engine.py`, `training/stream.py` | Execute grouped branches and causal windows |
| `training/history.py`, `training/rl.py` | Group replay, policy density, and revised freeze ownership |
| `data/storage.py`, `data/prepare.py` | Persist group/chunk/reference identities without discarding lineage |
| `training/optimizer.py`, `model/audit.py` | One parameter owner and optimizer state per shared weight |

**Deliverable:** equivalent prompt/typed-control tasks and exact grouped density replay work through preparation, training, export, and serving.

### Milestone C: speech megatokens

**Scope:** one pinned audio input representation, one audio realization target, long-chunk segmentation, transcription, TTS, and speech-conditioned answers.

Add a small `modalities` package with codec/feature adapters and media specs. Keep E/C/P/R boundaries in the existing model/generation modules; do not introduce a second trainer.

**Deliverable:** one checkpoint performs text-only tasks, audio-to-text, text-to-audio with a reference voice, and chunk-aligned translation. Report quality by chunk size and preserve text replay performance.

### Milestone D: images and video understanding/generation

**Scope:** shared visual stem, temporal reduction, image/video target grids, and a visual realizer.

**Deliverable:** the same core ingests images and coarse clips, answers temporal questions, and produces controlled images and short videos. Demonstrate that increasing clip duration does not force frame-level central events.

### Milestone E: joint media and concurrent execution

**Scope:** paired audio/video realization, actual parallel tools, retrieval tracks, worker workflows, and one-meridian speculation.

**Deliverable:** a single session can consume media, retrieve evidence, run concurrent tools, and generate aligned audio/video through the same track and control protocol. Speculative branches can be accepted/discarded without history corruption.

### What not to implement in these milestones

Do not add dynamic historical compression, a general graph database, per-track private transformer stacks, a learned scheduler, a universal raw embedding table, several redundant codec backends, or a new distributed training protocol. Keep the current distributed/DiLoCo machinery and adapt the event/gradient payloads it already handles.

## 19. Tests, measurements, and first experiments

### Architectural correctness tests

These are ordinary unit/integration tests supporting the design, not a separate risk program:

- **Singleton parity:** the grouped path with one track reproduces the original mathematical path under compatible initialization and metadata defaults.
- **Peer visibility:** changing selected peer facets can affect another member's C features and realized output.
- **No future-target leakage:** changing withheld target payloads cannot affect that meridian's P output or prior-sample C context.
- **Permutation behavior:** permuting a meridian's storage order with identities, metadata, and RNG mapping preserved permutes its outputs within numerical tolerance, rather than changing the task.
- **Atomic history:** one append per group; unchanged retained historical KV; no partial event slots or padding records.
- **Replay:** all-peer raw replay reproduces grouped context and conditional materialization on the supported numerical backend.
- **Density:** joint density equals the sum over active raw coordinates, with no padded contributions or unrecorded topology decisions.
- **Ownership:** encoder-only losses cannot update C/P/R; latent-only RL cannot change the conditional materialization map through a shared control encoder.
- **Variable tracks:** newly opened, inactive, and closed tracks do not create phantom state transitions.
- **Timing:** alignment metadata does not override causal availability; unknown future durations are not model-visible inputs.
- **Execution:** parallel results preserve call identity; draft rejection leaves canonical state unchanged; delivered actions are not rolled back.

### Quality measurements

Keep a small dashboard of text task quality, cross-track retrieval/join accuracy, speech intelligibility, speaker/style adherence, temporal video understanding, and paired-media alignment. Compare controls that change style while preserving requested content.

For the facet bottleneck, test the same examples at increasing speech/clip duration. Measure retained content and temporal order, not just reconstruction loss. Keep reference-free generation and asset-conditioned editing as separate evaluation modes.

### Three initial experiments

1. **Shared-KV text cooperation:** singleton versus two/four-track grouping, with and without same-meridian attention, at matched useful work. This isolates the core idea.
2. **Speech at text-scale density:** speech chunks near 128 versus 256 transcript-equivalent tokens, keeping the core geometry fixed. Test transcription, TTS, and content/control consistency.
3. **Temporal visual reduction:** 8/16/32-second clips through one shared visual encoder, evaluating event order and detail retention before adding elaborate segmentation.

Test expert sharing only at a small number of boundaries: fully shared compatible blocks versus a small modality-specific adapter or expert subset. Avoid a combinatorial architecture search before the baseline works.

### Performance arithmetic

At the current `32x512` geometry, one retained megatoken uses:

```text
32 facets * 2 KV heads * 128 coordinates * 2 (K and V) * 2 BF16 bytes
= 32,768 bytes = 32 KiB
```

A global budget of 1,024 megatokens is still 32 MiB of projected KV, excluding activations, metadata, raw tapes, and renderer work. Four continuously active events per meridian mean about 256 fully populated meridians in that budget, not 1,024 per track.

With `T` active events, live attention has `32T` queries/keys and quadratic current-block work; historical-read work grows with `32T` queries against the retained shared slots. Pair-expert work grows approximately with active event count while its weights remain shared.

The small core may make semantic decisions quickly, but audio/video encoding and realization can dominate end-to-end latency. Measure E, P, C, R, KV writes, and external scheduling separately. Do not treat parameter count as a throughput result.

Use the existing benchmark/Blackwell workflow, first at widths one, two, and four and a fixed global history budget. Kernel/ring-buffer optimization follows those measurements, not a parallel architectural redesign.

## 20. Versioning, compatibility, and release criteria

The plan adds a new execution geometry and grouped replay semantics. Introduce an explicit architecture version and grouped-tape version when implementation begins; do not reinterpret v6 checkpoints or v4 tapes silently.

Single-track data can migrate to singleton meridians with original record IDs, byte boundaries, roles, and controls preserved. Media timing remains absent rather than fabricated. Preserve a legacy replay/import path for old native tapes.

A checkpoint migration should copy compatible E/C/P/R weights, initialize new modality/task projections and track biases neutrally, and verify singleton numerical behavior. New parameters get their own optimizer state; existing parameter owners retain theirs. Do not promise that an unmodified old cache can be loaded into the new state geometry.

Codec, renderer, control, chunking, model, and tape identities travel together through export and resume. The current version-rejection and exact-resume discipline should extend to them.

The first architecture release is complete when:

- text-only multi-track training and serving work end to end;
- same-meridian attention and strict prospective causality are both tested;
- grouped native tapes and controls replay correctly;
- singleton tasks remain supported through the shared implementation;
- one concrete cross-track task improves or becomes simpler than its serial wrapper equivalent.

Media milestones can then land incrementally without reopening the causal model. Full audiovisual quality is not a prerequisite for accepting the multi-track foundation.

## 21. Recommended implementation starting point

Start with **Milestone A**, using text-only tracks and the existing tiny model, followed immediately by a small learned cooperation task at the production geometry or its compact variant.

The first implementation should contain only the essential extensions:

```text
track/event metadata
    + grouped frontier
    + joint current-block attention
    + shared-history reads
    + grouped Gaussian selection
    + atomic final KV publication
    + grouped replay
```

Everything else builds on that foundation. Audio and visual codecs remain adapters; media decoding remains realization; sub-agents remain tracks; retrieval and tools remain observations/actions; controls remain one conditioning system.

The target is not a frame-rate version of every specialist model. It is a coherent model that processes, predicts, and coordinates information-dense megatokens across modalities and concurrent work.

## References and evidence boundaries

### Repository sources reviewed

The implementation review was anchored to commit `a26c5bb02e68563e6d579b2be8d8fbc3b1788bd1`. Relative links below remain convenient while the repository evolves:

- [Architecture](architecture.md), [training](training.md), and [generation](generation.md).
- [Contextual realization decision](decisions/0004-contextual-realization.md) and [full-resolution KV decision](decisions/0006-full-resolution-kv.md).
- [Core](../src/megatoken/model/core.py), [attention](../src/megatoken/model/attention.py), [state](../src/megatoken/model/state.py), and [blocks](../src/megatoken/model/blocks.py).
- [Ingestion](../src/megatoken/model/ingestion.py), [byte/control frontend](../src/megatoken/model/frontend.py), [expert banks](../src/megatoken/model/experts.py), and [routing](../src/megatoken/model/routing.py).
- [Control protocol](../src/megatoken/controls.py), [data schema](../src/megatoken/data/schema.py), [chunking](../src/megatoken/data/chunking.py), and [inference](../src/megatoken/inference.py).
- [Policy](../src/megatoken/generation/policy.py), [realizer](../src/megatoken/generation/realizer.py), [tapes](../src/megatoken/generation/structure.py), and [training engine](../src/megatoken/training/engine.py).

### External primary sources checked on 2026-10-19

**[R1] Qwen3-VL Technical Report.** [Paper](https://arxiv.org/abs/2511.21631), [official repository](https://github.com/QwenLM/Qwen3-VL). Relevant for interleaved multimodal RoPE and explicit timestamp alignment, not evidence for arbitrary track-number geometry.

**[R2] Moshi / Mimi.** [Official repository](https://github.com/kyutai-labs/moshi), [paper](https://arxiv.org/html/2410.00037v2). Relevant for reusable speech codec/features. This plan deliberately does not adopt its audio-frame semantic clock.

**[R3] Open-Sora-Plan.** [Official repository](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [v1.0 technical report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md). Relevant for image/video codec sharing and spatiotemporal compression.

**[R4] F5-TTS.** [Paper](https://aclanthology.org/2025.acl-long.313/), [official code and license notes](https://github.com/SWivid/F5-TTS). Relevant for non-autoregressive, flow-matching speech realization and a concrete training implementation.

**[R5] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.** [Paper](https://arxiv.org/abs/2408.11039). Relevant for combining discrete and continuous generation objectives with modality-specific edges.

**[R6] LTX-2.** [Paper](https://arxiv.org/html/2601.03233v1), [official repository](https://github.com/Lightricks/LTX-2), [trainer](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/README.md). Relevant for paired audiovisual realization, reference conditioning, and available fine-tuning infrastructure. Pin the specific version chosen for an experiment rather than depending on a moving default branch.

**[R7] LibriTTS-R.** [Publisher download and license](https://www.openslr.org/141/), [dataset project page](https://google.github.io/df-conformer/librittsr/). Concrete initial paired speech/text data, not a universal multimodal dataset.

The multi-track execution contract, chunk operating points, sharing boundaries, and implementation sequence in this document are proposed engineering decisions. The references support component techniques; they do not establish measured quality or speed for the proposed Megatoken system.
