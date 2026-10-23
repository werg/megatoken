# Media: one input path, one realization path

Megatoken observes and produces audio, images and video through the same
32-facet interface as text. There is exactly one way in and one way out per
modality; this document is the contract for both and for the training phases
that teach them. The task catalogue and data roadmap for concordant
cross-modal training live in
[pretrained-omnimodal-bottleneck-plan.md](pretrained-omnimodal-bottleneck-plan.md).

```text
media file ──EmbeddingGemma2 (frozen, offline)──► [T ≤ 8192, 512] hidden states
          └─realizer codec (frozen, offline)────► codec latents (training target)

hidden states ─► MediaEncoder (Linear 512→E width, RMSNorm, 32-query attention read) ─► [32, D] raw facets
text bytes    ─► byte E ─────────────────────────────────────────────────────────────► [32, D] raw facets
raw facets ─► C (context, + modality embedding) ─► P (prediction) ─► sampled facets
sampled facets ─► FacetCondition ─► frozen donor: Pocket (speech) | Sana (image) | Sana-Video (video)
```

## Input: pinned EmbeddingGemma2 hidden states

Every observed media event is the **512-wide final normalized hidden states**
of `google/embeddinggemma-2`, captured at the input of
`language_model.embedding_projection` (one vector per teacher token, after its
audio/vision towers and bidirectional backbone). It is never the pooled
768-d embedding and never raw PCM or pixels.

- **One teacher, every role.** `google/embeddinggemma-2` is the only teacher.
  One forward pass (`omni/embeddinggemma2.py`, class `EmbeddingGemma2`) yields
  both these hidden states and the teacher's own sentence embedding (the mean
  of the projection's 768-wide outputs over all tokens, normalized), which is
  what the embedding-head targets are made of. Weights load only from the locked
  pack manifest; nothing is truncated, and inputs over
  8192 teacher tokens are refused (split them into several events). Video is
  sampled by the processor's own frame policy, so features carry no clock.
- **Identity.** `megatoken.teacher.PINNED_TEACHER` names the hub revision, a
  digest of every snapshot model file (hub `.cache/` bookkeeping excluded), the
  tap, and the preprocessing policy (`chat-bf16-v1`: the processor's chat
  template, bf16 weights). `ModelConfig.media_extractor` must equal it; features,
  checkpoints and tapes from any other teacher are rejected. Changing the teacher is an architecture change.
- **Model side.** `FacetedCore.encode_media(features, mask, extractor=...)` is
  the single entry point: a trainable `MediaEncoder` projects the teacher
  states, 32 learned queries attend over all of them at once, and E's shared
  facet refinement finishes the facets, so media and text land in the same
  facet geometry. C adds a learned modality embedding; no separate media stem,
  transformer or KV exists.
- **Why one attention read.** Measured against reusing text's pairwise
  halving on the teacher states (same initialization, same items, 1,500
  encode-and-reconstruct steps on 400 items per modality from LibriSpeech,
  Flickr, UCF101 and HotpotQA; 22 October 2026). The attention read matched
  the teacher's own embedding far better once the shared average direction is
  removed (centered cosine on held-out items, 0 = unrelated, 1 = identical:
  0.25 to 0.36 per modality against about 0 for halving), reconstructed held-out
  media at least as well (realizer loss, lower is better: speech 0.435 vs
  0.513, image 1.05 vs 1.12, video 2.34 vs 2.68), trained more stably, and cost
  the same time per step (0.80 vs 0.85 s) for 2.4M instead of 0.13M
  parameters.
- **Text is bytes.** Text always enters through byte E. Teacher features of
  text are used only as *text anchors* (below), never as input.

## Output: resident pretrained realizers

R decodes sampled raw facets into media through one frozen pretrained donor
per modality (`omni/realizers.py`), registered in `model.media_realizers`:

| Modality | Realizer | Target latents | Physical clock |
|---|---|---|---|
| audio (speech) | Pocket TTS flow LM + Mimi | Mimi latents `[frames, C]` | 24 kHz |
| image | Sana + DC-AE | `[C, h, w]` | — |
| video | Sana-Video + VAE | `[frames, C, h, w]` | declared fps |

Each realizer owns a trainable `FacetCondition` (per-slot projection of the
32 facets into the donor's conditioning width); donor weights stay frozen and
outside the module tree, so checkpoints hold only the conditions. Training
uses the donor's own objective on its own codec latents (flow matching for
Sana/Sana-Video, Pocket's flow/LSD loss), with codec padding masked by
`valid_count`. Generation calls `realize(raw, spec, generator)` with a public
`MediaOutputSpec` (requested or predicted geometry; teacher-only geometry is
refused). A realizer's identity is `<pack>:<checkpoint digest>`; generation
tapes record it per event and replay requires the same identity.

## Packs and the lock

All pretrained weights come from one locked pack manifest, `<root>/lock.json`,
written by `python scripts/acquire_packs.py --root <root>` from the pinned
[inventory](../configs/packs/inventory.json): `embeddinggemma2`, `pocket`,
`sana`, `sana_video`, each at a 40-hex revision with a SHA256 for every file.
Acquisition fetches only what the models load: Pocket's English model and
tokenizer with a fully local `pocket.yaml` (the `kyutai/pocket-tts` repo is
gated: accept its terms and set `HF_TOKEN`; its ungated sibling zeroes the Mimi
encoder), and the Sana/Sana-Video Diffusers snapshots without their Gemma-2 text
encoders, which facets replace. `omni.loader.read_locked_manifest` verifies
every file, rejects symlinks, unhashed files and path escapes, and
`install_realizers` builds the realizers before optimizer enumeration or
checkpoint restore. Checkpoints store the manifest's `packs_fingerprint`;
`load_trained` refuses a different manifest. `scripts/verify_packs.py` probes
real weights (codec round trip, objective, rollout) on an accelerator.
Install with `pip install -e '.[train,media]'`.

## Preparation

Preparation runs once, offline, on a GPU, with the teacher and realizers
resident (`data/media_prepare.MediaPreparer`):

```bash
# items.jsonl: {"modality": "image", "path": "x.png", "target": true}
#              {"modality": "text", "text": "exact chunk"}   (a text anchor)
python scripts/prepare_media.py --manifest packs/lock.json --root prepared \
  --items items.jsonl --records records.jsonl --anchors anchors.jsonl
python scripts/prepare_media_identity.py --records records.jsonl --root prepared \
  --codec <codec> --text-episodes <native text episodes>
```

`scripts/prepare_open_media.py` builds the identity streams and the concordant task streams
(transcription, verbalization, speech translation, image description/creation/editing, speech
and audiovisual creation/editing) from pinned public datasets in one pass; the sources, licences
and gaps are in [data.md](data.md#scaling-the-media-streams) (the pilot sizes are in
[data.md](data.md#open-licence-pilot-mix)).

One decode of each original file yields the assets of a `MediaRecord`: the
teacher features (`source`) and the teacher's pooled embedding of the same
input (`embedding`, from the same pass) and, when `target` is set, the
realizer's codec latents plus the physical `OutputGeometry`. Both carry `source_family` = the
SHA256 of the original file, so a record's input and target describe the same
media by construction. Assets are immutable little-endian FP32 files under
`<root>/media/<sha256>.f32le`, verified by size and digest on every load.
Observed inputs keep their own latents (they are the input itself, never a
future answer); training reads only an observed event's features, and uses the
latents for encode-and-reconstruct (below).

`prepare_media_identity.py` turns each record into an identity episode
(“reproduce this image”) in the `<modality>_identity` stream and imports
native text episodes into the `text` stream. Concordant task families
(transcription, captioning, editing, …) are compiled by `data/concordance.py`
into their own streams. `MediaPreparer.video_endpoints` derives opening and
closing frames of a clip as image events with an audited index map.

## Text anchors

The media bridge must land where byte E puts text. For a text chunk with
prepared teacher features, the anchor loss is the mean squared difference
between E(bytes) and E(teacher features of the same bytes)
(`training/text_anchors.py`, `text_anchors` + `text_anchor_weight` in the
train config). Whichever side is trainable moves: only the bridge in phase 1,
both afterwards.

## Media in the training plan

Media is part of the one stage plan (`src/megatoken/data/stages.json`), not a
separate track of phases. Every stage runs on DiLoCo
([diloco-coreweave.md](diloco-coreweave.md)); a stage transition is a **new
coordinator** initialized from the previous stage's coordinator export:

```bash
megatoken diloco export --config <previous config> --root <previous root> --output exports
megatoken diloco init --config <next config> --root <new root> --resume exports/latest.json
```

Exports carry the global FP32 masters. Owners new to a stage (the realizer
conditions when packs are first installed) start from initialization; owners
frozen in the previous stage keep their exported weights and start fresh
inner-optimizer moments. Nothing else may appear or disappear.

| Stage | Media in its data mix | Trainable |
|---|---|---|
| B0 codec bootstrap | none | all |
| **MB media bridge** ([example config](../configs/train/media-bridge.example.yaml)) | text 0.40; audio, image and video identity 0.20 each | `media_bridge`: media input projection, modality embedding, realizer conditions |
| P1 base pretraining | 15% media (identities and the concordant task families: transcription, verbalization, speech translation, image description/creation/editing, speech and audio-visual creation/editing), 3% long-document reading | all |
| M1, M2 repository and agent work | 5% media replay, 2% long-document reading | all |
| LC long context | 5% long-document reading | all |
| S supervised dialogue | 5% media tasks with text answers | all |

The bridge comes right after the codec bootstrap, so the media projection
learns to land where the byte encoder already puts text (with text anchors),
before the core trains on media. `trainable: media_bridge` freezes every other
owner identically on all workers, so their outer deltas are exactly zero and
the DiLoCo merge is not diluted; the stage plan requires it for MB and forbids
it elsewhere. A run uses the stage's mix with `use_stage_mixture: true`; shares
are shares of *events* (a text chunk, an image, a clip, a compressed document
each count one), and every named stream must exist in one of the prepared roots
listed in `datasets` (each preparation script writes its own root, with its media
tensors; see [training.md](training.md)).
`max_media_events_per_window` caps media events per window. Long-document
reading streams are episodes whose reference documents are read compressed
([compressed-reading.md](compressed-reading.md)); in agent stages, large tool
results are also compressed at import.

Each media event contributes one per-event mean to `audio_*` or `visual_*`
loss sums with their own global denominators in the `LossLedger`, so tensor
sizes never become token weights.

### Two ways the media projections are shaped

- **Through the core, in episodes.** For every generated media event, the
  realizer reproduces the media both from P's prediction (`*_prediction`) and,
  as a shortcut, from E's own noisy encoding of the target
  (`*_reconstruction`).
- **Without the core, every step: encode-and-reconstruct**
  (`training/autoencode.py`). Each step also draws prepared feature items from
  the training split: media inputs and outputs alike (an input keeps its own
  latents for this) and compressed-text pieces, in equal turns per kind. Their
  teacher states go through E only; the item's realizer reproduces its codec
  latents from a noisy sample of the facets, and the embedding readout is
  distilled to the teacher's own pooled embedding of the item (centered
  alignment distance). This trains the media projection, E, the realizer
  conditions and the embedding readout at encoder cost. The stage plan sets
  the items per step (`constraints.autoencode`: MB 64, P1 16, M1/M2/S 4, LC 2)
  and it runs whenever a run uses its stage's mix. Prepared datasets index
  every distinct feature item per split at export for this.

## Serving and replay

`multitrack_service.py` accepts media as pinned teacher features
(`features`, `extractor`) and returns realized bytes plus a `tape_payload`: a
`media-sha256-v1` digest binding the output bytes to the realizer identity
and seed. `GroupActionTape` v10 records renderer seeds and pack identities for
every event; replay refuses a different realizer pack or drifted output.

## Retrieving the original items

A model that can observe a picture, a recording or a passage can also be handed the
original again: a **source collection** (`megatoken.retrieval`) holds items of any
medium, each with one search key, and a lookup can pull the best ones into the
session. The design is in
[multimodal-knowledge-plan.md](multimodal-knowledge-plan.md) section 3; this is the
contract.

- **One key space.** The key of an item is the teacher's own pooled 768-wide embedding
  of it (the `embedding` asset every prepared item already carries), taken raw: EmbeddingGemma2
  embeds text, images, audio and video into one space, so a photo, a recording and a
  paragraph about the same thing lie near each other. Search is by cosine on these unit
  vectors. The embedding head is trained toward the same raw vectors (its loss is centered, the
  head's output is not), so the student's query and the stored keys are directly
  comparable and nothing is centered at search time.
- **Items.** A short passage (at most 4,096 bytes) is kept as its exact bytes. A longer
  text is cut into the teacher's own pieces, each kept as its teacher features and
  observed as compressed text. An image, a clip or a recording is kept as its teacher
  features. Every item also keeps its original file reference (a path, or the hash of the
  original for items taken from a prepared root).
- **Index.** A directory with one SQLite file: the catalog, the float keys and an
  inverted-file index (spherical k-means cells, int8 copies of the keys in a covering
  cell index). Only the cell centres are in memory. A search ranks the probed cells by the
  int8 codes and reads float keys only until the exact top-k is certain by the
  quantization error bound, so its result equals exact rescoring of the probed cells; at
  build the probe count is calibrated to the smallest that matches exact search at 97%
  (on noisy copies of the keys, standing in for a student's imperfect queries).
- **Building.** `megatoken collection build --name X --output DIR --items ...` takes
  prepared dataset roots (their feature items as they are, `--split train` by default) and
  raw files or folders. Raw files go through the preparation path above (`MediaPreparer`
  and the teacher, so `--packs` is required): media become teacher features and an
  embedding, text gets the document prompt, long text is cut into pieces. The command then
  fits the cells and calibrates the probe count. `collection add --path DIR --items ...`
  extends a built collection (new items are posted to their nearest cell) and
  `collection info` reports it. A serving process with a preparer adds files to a mounted
  collection while it runs (`MultiTrackRuntime.add_files(paths, collection=...)`, the `add_files` operation of
  `serve --collection NAME=DIR --packs ...`); sessions find them from their next
  lookup.
- **At a lookup.** The query block P samples on the query track also goes through the
  embedding head's query role, giving a 768-wide query; no teacher runs. All mounted
  collections are searched; up to `max_sources` (2) unobserved items scoring at least the
  threshold (0.5 on the raw cosine unless the collection or the session sets another) are
  observed right after the knowledge event, one meridian each, as events of their own
  medium on the knowledge track with the role of a document: text as bytes, media and long
  text as their stored features. A session never observes an item twice. Cancelling a
  lookup, even midway through the items, leaves the session untouched. Training and serving
  run the same code (`model/sources.py`), and a served session lists what it retrieved with
  each item's original file reference.
- **The question needs no special field.** A finished user message is a lookup boundary
  (the same rule in training and serving), so the lookup right after the question has read it
  and searches for what it asks. The first lookup, right after the metadata, comes before
  anything was asked; `open({"request": question})` can still give it one (`request` is an
  ordinary optional metadata field).

## Verified, and not established

Verified on real weights (2026-10-21, `configs/cloud/verify-packs.json`, one RTX
PRO 6000 Blackwell): packs acquired and locked from the pinned inventory, with
the teacher identity reproducing `PINNED_TEACHER` from a fresh download; Sana
and Sana-Video encode, training loss with gradients to their conditions, and a
free rollout (image `[1, 1, 3, 256, 256]`, 9-frame video); and `MediaPreparer`
end to end on synthetic text, image, speech and video (teacher features for all,
realizer latents and geometry for image and video). Pocket was verified the same
way on 2026-10-22 (gated repo acquired with `HF_TOKEN` delivered by
`megatoken.cloud launch --secret hf-token=<file>`): Mimi latents, a training
loss with gradients to its condition, a one-second 24 kHz rollout, and speech
preparation (teacher features plus Mimi latents and geometry).

No media-quality, perceptual or GPU-scale training result is implied.
Rights-cleared media inventories and measured gates per phase are prerequisites
for such claims.


## Dynamic events and timed preparation

An output track can carry different modalities and separately addressed simultaneous
peers, such as speech and its caption. The output-kind classifier runs before content
prediction and only considers supported specifications supplied by the host. Explicit
audio duration and visual geometry remain necessary; target-only extents are refused
in serving. Selection tapes retain both categorical and content probabilities.

Prepared events can retain exact original sample/frame ticks, a clock rate, speaker,
direction and external availability time. Content time and actual availability are
independent. The aligned-source compiler preserves these fields and transformation
provenance. This is a representation contract, not evidence that the current mono
conversion and average-frame-rate helpers preserve separated conversations or variable
frame timing. Audit or replace those preparation steps before admitting timing-sensitive
sources. Current decoder integration does not establish incremental full-duplex output.
