# EmbeddingGemma2 internal-state unified ingestion and parallel warmup

Status: **implementation experiment**, not a trained/checkpoint-validated capability. The
change is intentionally additive to architecture v9 and does not silently load older
non-EmbeddingGemma teacher caches into the new route. Authoritative existing
interfaces: `model/core.py`, `model/media.py`, `model/multitrack.py`,
`training/meridians.py`, `training/engine.py`, `training/diloco_*.py`,
`omni/donors.py`, and `model/native_latents.py`.

## Decision: hidden states, not EmbeddingGemma2 embeddings

```text
audio ────────────── audio preprocessing / modality adapter ┐
video ────────────── sampled frames / vision adapter        │
image ────────────── vision adapter                         ├── frozen EmbeddingGemma2
document ─────────── source parser -> text / scanned pages  │     multimodal backbone
long text [optional] ─── text tokenizer                     ┘             │
                                                            ordered [T,512] PRE-projection
                                                                       │
                                                     learned 512→source_dim projection
                                                                       │
                                                 shared E pair reductions / facet readout
                                                                       │
                                                       Z = [batch, 32, facet_dim]
                                                           typically [B,32,512]
                                                  /             |             \
                                    native text R     audio/image/video R      C→P
```

The model's 768-wide **embedding output** is *not* used as Megatoken's
raw facets and is *not* the source of the bottleneck. The teacher's final
`language_model.embedding_projection` maps its internal 512-wide backbone
state to the 768-wide embedding coordinate space; a pre-forward hook captures
its **input** (ordered `[B,T,512]`), and the adapter maps it to
`source_dim` before **existing** E reduction/readout produces all 32 raw
facets. Never call `SentenceTransformer.encode()`, mean-pool tokens, or
renormalize Z. EmbeddingGemma2's semantic retrieval embedding may continue to
be used separately for its pinned 768-wide retrieval supervision; do not
confuse retrieval embeddings with 32-facet media inputs.

This unifies feature *ingestion*, not the generative decoder. Audio, image
and video realizers remain separate frozen pretrained models with trainable
facet-conditioning projections during warmup. The downstream C/P recurrence
stays unchanged, as does direct R from the selected raw Z.

**No hidden teacher model is included in a Megatoken checkpoint.** Teacher
inference is performed offline and cached as a checksummed FP32 ordered
feature asset. The student is responsible for projecting it into raw facets.
Free serving must have the same pinned teacher/extractor at its boundary.

## Source / cache contracts

- Required teacher snapshot: `google/embeddinggemma-2` at an immutable
  40-character revision. Checkpoint/processor weights and all other files are
  locally hashed and locked in `teacher-lock.json`. Never trust a mutable
  `main` tag or automatically download from inference/training.
- Tap: `language_model.embedding_projection:input`. Width **512**.
  Source identity:
  `embeddinggemma2:<commit>:language_model.embedding_projection:input:chat-template-v1`.
  Source masks must come from the processor, not guessed from zero feature
  values. Pad/future audio or video must never become actual evidence.
- Maximum teacher context is **8192 tokens**. Preserve event boundaries,
  ordered patches and model-visible timestamps. For long audio/video:
  externally split into auditable time ranges, process each available piece
  causally, preserve the actual timebase, and do not treat per-event 32
  facets as pixel- or sample-exact. Video caches pin rational FPS.
- Documents: extract original text with source/page provenance, pass through
  EmbeddingGemma2 text input; PDF/DOCX text parsing is provided as a local
  preparation aid. Scanned pages and diagrams must be fed through the
  **image** adapter as individual visual media events, not falsely claimed to
  have been understood via empty PDF text. Text extraction loses layout;
  keep page/crop lineage when this matters.
- Long text has two distinct routes. The canonical native byte exact E
  remains used for supervised text generation. The **optional compressed**
  document/text-summary input uses EmbeddingGemma2 hidden states and the
  same 32-facet projector; it trades exact copy/line-level fidelity for
  semantic and contextual coverage. Preserve the original chunk if the task
  needs quotations or span-exact edits; test access to details explicitly.
- Output codecs are a **separate** pinned artifact identity. For production
  R, freeze original generator/codec weights during warmup, expose only
  facet-conditioner/projector weights. Support reconstruction for the
  declared native output geometry, timebase, target masks, and codec latent
  masks. Keep media errors normalized **per event**, not per sample/patch.

### Offline preparation

```bash
# Start from a locally downloaded immutable revision, with all files present.
python scripts/lock_embeddinggemma2_snapshot.py \
  --snapshot /models/embeddinggemma2 --revision <40-hex-snapshot-revision>

# Original target asset must ALREADY exist in the matching prepared media root.
# Features must be from the exact SAME original source as the target.
python scripts/cache_embeddinggemma2_hidden.py \
  --snapshot /models/embeddinggemma2 --revision <same-40-hex> \
  --modality image --input /sources/photograph.png \
  --asset-root /datasets/unified --identity-target-sha256 <raw-target-asset-sha256> \
  --output /datasets/unified/teacher/image-0001.json

# Repeat for audio and video. For video also pass --fps-num and --fps-den.
# For text and document, OMIT identity-target-sha256. Cache one exact original
# text chunk per text record; concatenate the resulting JSON descriptors as
# JSONL for --embeddinggemma2_text_sidecar.

# JSONL pair row (target_asset is asdict(TensorAsset) from TensorAssetStore.put):
# {"modality":"image","teacher_descriptor":"teacher/image-0001.json",
#  "target_asset":{...}, "family_id":"source-family-0001"}
# Run only after the teacher and raw/codec target assets are staged under root.
python scripts/prepare_embeddinggemma2_identity.py \
  --pairs-jsonl /datasets/unified/pairs.jsonl \
  --text-episodes-jsonl /datasets/unified/text-episodes.schema-v3.jsonl \
  --output /datasets/unified --codec /models/tokenizer
```

The pack must contain all four required prepared stream names:
`embeddinggemma2_text_pretraining`,
`embeddinggemma2_audio_identity`,
`embeddinggemma2_image_identity`, and
`embeddinggemma2_video_identity`.
A prepared feature's `source_family` **must equal** the SHA256 of
its original target media `TensorAsset`; this prevents later edits or
different sources from being mislabeled as identity autoencoding.
Also retain any native train/validation split, source-family deduplication,
hash checks, exact target geometry, and physical media clock.

A text anchor sidecar is a JSONL of text/document cache **descriptors**
(`format: megatoken.embeddinggemma2.hidden.v1`) for the exact original
byte payloads. The student aligns its own byte-E raw facets to the detached
teacher-hidden→E raw facets. No teacher vectors are allowed as policy tokens
or prediction outputs. Source SHA256 mismatch skips the anchor; mismatched
file hashes fail closed.

## Training objective and parameter ownership

### Stage A — simultaneous text and media warmup

**Text workers**: train the normal Megatoken byte-E, text R, C, P,
prediction loss, retrieval query/document supervision, and native
reconstruction. Add same-text anchor
`L_text-align = mean((stopgrad(E_teacher_hidden(text)) - E_native_bytes(text))²)`.
The teacher feature projection is **frozen on text workers**; anchoring
teaches native E to share the space already induced by the media bridge,
without making long text compression mandatory for ordinary text.
Retain short, copy-intensive examples and byte exact generation.

**Media workers**: for each modality `m∈{audio,image,video}` train only

```
Z_m = E_shared(frozen_teacher_hidden(m) @ W_in)
L_m = L_R_m(R_m(Z_m), original_m)
```

where **both** `W_in` (single unified input projection) and the
selected R conditioning projections are trainable; the shared E hierarchy,
text E, text R, C, P, memory, donor perception, renderer backbones and
codec weights are **frozen** on media workers. Gradient-through-frozen E
is intentional, since W_in must still see the R loss. The initial output
heads for native debug targets are learned, while Pocket/Sana/Sana-Video
generation/codec backbones remain frozen. This is direct R from encoded
Z, **not** from sampled P outputs or text captions.

No text decoder loss is scored on media workers. No gold future content is
given to current P, C, or a seeded control. Teacher features are only
allowed to describe the actual visible current media, with native
preprocessing clocks/masks. Identity target/source must match by SHA256.

If a modality cannot be reconstructed meaningfully from one 32-facet
event, increase **number of events / temporal granularity** or
introduce a residual high-frequency codec that does not contaminate Z;
do not silently add a teacher sample/prompt into R.

### DiLoCo coordination

Use `configs/train/embeddinggemma2-parallel-warmup.example.yaml` on all
workers, keeping **the same train config file**, model architecture, tokenizer,
retrieval/teacher identities, pack lock, initial weights, optimizer-owner set,
and parameter fragments. The config specifies *worker ID membership* in
the media lane (for example 2 and 3 of 4); the remaining workers train text.

Each independent worker owns an otherwise identical local model and
FP32 optimizer masters. It only accumulates gradients for its lane.
Frozen owners are retained in its optimizer state to satisfy the coordinator's
parameter and fragment identity. Existing DiLoCo synchronization/outer
updates combine the text and media parameter deltas; changes to the shared
frozen E on text workers are subsequently rebased on media workers.
Sync periodically with bounded local steps, avoid stale sidecars/model
geometry, and use the existing checkpoint, coordinator restart, straggler
and quorum semantics. GPU/compute shares are not inferred from data
probabilities: measure wall-clock and throughput by lane.

**Potential hazard:** heterogeneous worker deltas are not mathematically
equivalent to an alternating joint mini-batch, particularly at large
DiLoCo local step counts. Begin with smaller local steps, track gradient
conflict and representation drift after each sync, then scale if gates pass.
Use the same train_identity (sha of both recipe manifests/sidecars) for all.

### Stage B — anchored joint pretraining

Once per-modality direct recon and native text gates pass, start the
recurrent mixed training recipe with unfrozen shared E/C/P and trainable
input/output projections. Continue **same-text representation alignment**
and native text copy/prediction anchors (do not let compressed long-text
examples replace byte exact short text). Media R reconstruction remains
an auxiliary objective alongside per-event P prediction on
**causal** history. Use frozen teacher input, verify downstream
audio/video consistency and retrieval memory. The example joint config is
`configs/train/embeddinggemma2-joint-edit.example.yaml`; substitute actual
datasets and pack identities, and use an explicit phase transition/checkpoint
validation. The joint config is a scaffold and does not claim prebuilt data.

### Stage C — editing tasks

After warmup: enable the existing concordant family mixture
(`configs/omni/concordant-v1.yaml`) with text-conditioned media
creation/editing, source-preserving changes, speech translation,
cross-modal description, temporal continuation, joint audiovisual
outputs, and recorded edit chains. Causal role/track clocks, same-step
relations and native media output geometry must remain explicit. Continue
interleaving anchored text copy and identity reconstruction as replay
regularizers. Do not train an editing sample against a mismatched source
identity or attach a future target as an observed input. Only now
consider LoRA/full donor tuning if the frozen-backbone reconstruction
ceiling is established by held-out measurements.

## Gates and metrics

The warmup **does not** establish a good autoencoder merely because the
projection loss decreases. Establish hard phase-transition gates:

1. For each media family, evaluate E→R on held-out source families:
   event loss and reference-decoder metrics; audio intelligibility/WER,
   perceptual features and sample-rate fidelity; image semantic/structural
   similarity and spatial detail; video motion, frame consistency,
   A/V alignment and temporal localization. Compare against
   (a) frozen unconditional R, (b) its native pretrained original
   conditioning, and (c) older non-unified E input baselines.
2. Text: exact copy/character accuracy, native lexical/P loss,
   byte-E ↔ hidden-state-E facet MSE, retrieval query/document scores,
   and performance on text longer than a single teacher window.
   Explicitly measure how much detail is lost when a long text
   chunk is compressed to one event versus byte-exact ingestion.
3. Check active gradient owner names by lane, finite grads, no
   gradients in frozen donor/C/P/E modules on media workers,
   original-only teacher footprints and masks, no future spans,
   identical FP32 master/fragments across DiLoCo workers,
   resumption at the correct corpus cursor, and exact pack SHA.
4. After DiLoCo sync monitor modality reconstruction regression,
   text loss regression, facet distribution mean/variance,
   source-role separability, representation drift and catastrophic
   forgetting. Block editing if text or a modality degrades
   beyond the chosen audited tolerances.

## Known limits

- This does not mean a 32×512 bottleneck is mathematically invertible
  for long/high-resolution media. Discrete lossless original data
  cannot generally be recovered from a small semantic hidden state.
  Learnable projection-only adaptation cannot fix information that
  the frozen teacher omitted.
- With current teacher preprocessing, **long video** is sparsely
  sampled; prepare short real-clock events or tune a verified encoder
  preprocessing policy. PDF/DOCX extraction does not solve scanned
  pages; use the visual adapter and keep page lineage separately.
- This change has no trained weights, GPU quality numbers, or
  measured phase-transition thresholds. The new tests are offline
  contract/gradient tests and do not replace real multimodal trials.
- The checked pretrained student loaders and native codec packs are
  still expected to be installed *before* enumerating optimizer
  masters. All workers must load the exact same pack manifest and
  new hidden-state projector geometry at coordinator bootstrap.
