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

hidden states ─► MediaEncoder (Linear 512→E width, RMSNorm) ─► shared E reductions ─► [32, D] raw facets
text bytes    ─► byte E ───────────────────────────────────────────────────────────► [32, D] raw facets
raw facets ─► C (context, + modality embedding) ─► P (prediction) ─► sampled facets
sampled facets ─► FacetCondition ─► frozen donor: Pocket (speech) | Sana (image) | Sana-Video (video)
```

## Input: pinned EmbeddingGemma2 hidden states

Every observed media event is the **512-wide final normalized hidden states**
of `google/embeddinggemma-2`, captured at the input of
`language_model.embedding_projection` (one vector per teacher token, after its
audio/vision towers and bidirectional backbone). It is never the pooled
768-d embedding and never raw PCM or pixels.

- **Identity.** `megatoken.media_teacher.PINNED_EXTRACTOR` names the hub
  revision, a digest of every snapshot model file (hub `.cache/` bookkeeping
  excluded), the tap, and the preprocessing policy (`chat-bf16-v1`: the
  processor's chat template, bf16 weights). `ModelConfig.media_extractor`
  must equal it; features, checkpoints and tapes from any other teacher are
  rejected. Changing the teacher is an architecture change.
- **Extraction.** `omni/embeddinggemma2_hidden.py` loads the snapshot only
  from a verified pack manifest and returns FP32 `[T, 512]`; sequences longer
  than 8192 teacher tokens must be split into several events first. Video is
  sampled by the processor's own frame policy, so features carry no clock.
- **Model side.** `FacetedCore.encode_media(features, mask, extractor=...)` is
  the single entry point: a trainable `MediaEncoder` projection feeds the same
  E reduction hierarchy as text, so media and text share facet geometry. C
  adds a learned modality embedding; no separate media stem, transformer or KV
  exists.
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

One decode of each original file yields both assets of a `MediaRecord`: the
teacher features (`source`) and, when `target` is set, the realizer's codec
latents plus the physical `OutputGeometry`. Both carry `source_family` = the
SHA256 of the original file, so a record's input and target describe the same
media by construction. Assets are immutable little-endian FP32 files under
`<root>/media/<sha256>.f32le`, verified by size and digest on every load. A
causal reader sees `record.observed()` (source only); targets appear only on
the generated side of a meridian.

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

## Training phases

Media training follows native text pretraining and runs on DiLoCo like every
phase ([diloco-coreweave.md](diloco-coreweave.md)). All workers of a phase run
the same config; data is sharded per worker inside each stream. A phase
transition is a **new coordinator** initialized from the previous phase's
coordinator export:

```bash
megatoken diloco export --config <previous config> --root <previous root> --output exports
megatoken diloco init --config <next config> --root <new root> --resume exports/latest.json
```

Exports carry the global FP32 masters. Owners new to a phase (the realizer
conditions when packs are first installed) start from initialization; owners
frozen in the previous phase keep their exported weights and start fresh
inner-optimizer moments. Nothing else may appear or disappear.

| Phase | Train config | Task recipe | Trainable |
|---|---|---|---|
| 1 bridge | `configs/train/media-bridge.example.yaml` | `configs/tasks/media-bridge.yaml`: text 0.40, identities 0.20 each | `media_bridge`: media projection, modality embedding, realizer conditions |
| 2 joint | `configs/train/media-joint.example.yaml` | `configs/tasks/media-joint.yaml`: text 0.55, identities 0.15 each | all |
| 3 tasks | `configs/train/media-tasks.example.yaml` | `configs/tasks/media-tasks.yaml`: text, identity replay and concordant task families | all |

`trainable: media_bridge` freezes every other owner identically on all
workers, so their outer deltas are exactly zero and the DiLoCo merge is not
diluted. A task recipe (`training/task_recipe.py`) names prepared streams and
the share of *events* drawn from each — not a share of accelerator time; a
video event costs far more than a text window — plus the weight of the
reconstruction/alignment auxiliaries. Media phases use the `G` (event) budget
meter, and `max_media_events_per_window` caps media events per window.

Each media event contributes one per-event mean to `audio_*` or `visual_*`
loss sums with their own global denominators in the `LossLedger`, so tensor
sizes never become token weights.

## Serving and replay

`generation/multitrack_service.py` accepts media as pinned teacher features
(`features`, `extractor`) and returns realized bytes plus a `tape_payload`: a
`media-sha256-v1` digest binding the output bytes to the realizer identity
and seed. `GroupActionTape` v10 records renderer seeds and pack identities for
every event; replay refuses a different realizer pack or drifted output.

## Verified, and not established

Verified on real weights (2026-10-21, `configs/cloud/verify-packs.json`, one RTX
PRO 6000 Blackwell): packs acquired and locked from the pinned inventory, with
the teacher identity reproducing `PINNED_EXTRACTOR` from a fresh download; Sana
and Sana-Video encode, training loss with gradients to their conditions, and a
free rollout (image `[1, 1, 3, 256, 256]`, 9-frame video); and `MediaPreparer`
end to end on synthetic text, image, speech and video (teacher features for all,
realizer latents and geometry for image and video). Pocket is not yet verified:
`kyutai/pocket-tts` is gated, and its terms must be accepted for the account
whose `HF_TOKEN` acquires it.

No media-quality, perceptual or GPU-scale training result is implied.
Rights-cleared media inventories and measured gates per phase are prerequisites
for such claims.
