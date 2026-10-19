# Omnimodal media training: runnable ingestion and grouped optimization

**Status (2026-10-19):** Implementation for review. The feature bridge,
checksummed media references, media target losses, event-budget scheduling,
procedural pilot, and an offline donor extractor are implemented. This does
**not** mean an omnimodal model has been pretrained or that the small native
audio/image/video output heads have high perceptual quality.

## Architecture unchanged

```text
original media / pinned frozen donor features -> E -> [32,D] raw facets
simultaneous observed meridian -> joint C4 -> one full-resolution shared KV
prior P12 predicts multiple target tracks -> one joint sampled meridian
C4 jointly contextualizes samples -> modality-specific R outputs
```

Audio frames, visual patches, and diffusion/vocoder steps never become
smaller semantic megatokens. The same `ControlBundle` path is used by P
and R. Natural short utterances remain underfilled megatokens when appropriate.

## What is actually supported

The v7 `PhysicalRecord` retains **canonical JSON asset-reference bytes**
with `media_type=application/vnd.megatoken.media-reference+json`. The
corresponding `TrackEvent.modality` must be audio/image/video and its
optional `asset_sha256` must match the declared source. Media descriptor
bytes have **zero reference-text positions**. This preserves correct
`D`, `I`, and `P` accounting; no audio samples or video pixels masquerade
as tokens.

`TensorAssetStore` stores immutable little-endian FP32 arrays addressed by
their SHA256 in `<prepared_root>/media/<sha>.f32le`. A `MediaRecord` names
one source asset and an optional native raw target. The loader checks
canonical metadata, dtype/rank/shape, finite coordinates, byte length, and
content digest. Stored media has no arbitrary filesystem paths or network
dereferences. Keep the immutable assets and prepared manifest together
for reproducible distributed runs.

Two input forms are supported:

- **Native tensors:** 16 kHz mono PCM `[samples]`, RGB images `[3,H,W]`,
  videos `[frames,3,H,W]`; native E uses the existing STFT/Conv3D baseline.
- **Frozen external features:** ordered `[tokens, feature_dim]` tensors, pinned
  by `extractor`. Configure `audio_feature_dim/audio_feature_id` or
  `vision_feature_dim/vision_feature_id` using the exact producer identity.
  A *trainable E-owned projection* feeds the shared reduction and facet
  readout. No independent pretrained KV or source transformer is added.

Generated media targets must be native tensors for the current direct
output-head baseline. Images/videos share one visual realizer. Speech and
visual tasks have independent `audio_prediction`, `audio_reconstruction`,
`visual_prediction`, `visual_reconstruction` loss sums and event counts.
Every tensor's local loss is a mean over valid coordinates. The pretrained
feature projection and media realization losses integrate with the ordinary
`LossLedger`, optimizer gradient ownership, and B0/recurrent transitions.
In recurrent mode the E-only reconstruction branch freezes C/R owners.

**Media quality boundary:** current R predicts PCM/pixels with a tiny
direct regressor trained against per-event MSE. That is an API and gradient
reference, not a replacement for pretrained flow/diffusion speech/video
generators. High-quality native output still needs a pretrained conditional
renderer, compatible latent targets, and paired fine-tuning.

## CPU end-to-end pilot

```bash
python scripts/prepare_omni_pilot.py \
  --output artifacts/omni-pilot --count 64 --seed 91

megatoken train --config configs/train/omni-pilot.yaml
python -m pytest -q tests/test_omni_media_training.py
```

This corpus has two independent text evidence tracks, an observed image,
an observed sine-tone track, and a joint predicted text/image/tone meridian.
The text answer is an exact arithmetic oracle; synthetic images encode
remaining parts and tones encode a count. It tests real E/C/P/R mixing,
shared KV, and coordinated outputs **without pretending tones are speech**.
Generated tasks carry source-family IDs to prevent derived train/test leakage.

To train **paired cross-track counterfactuals**, use a distinct immutable output:

```bash
python scripts/prepare_omni_pilot.py \\
  --output artifacts/omni-counterfactual-pilot \\
  --count 64 --seed 91 --counterfactual-variants
```

Each parent produces three examples: a baseline, an **audio-only evidence swap**
(changes the hidden blue count), and a **visual-only evidence swap** (changes the
hidden red count). Instructions and shipment quantities are identical across
the three examples. The visual source is identical between baseline/audio-swap;
the audio source is identical between baseline/visual-swap; answer and paired
media targets change accordingly. **All three siblings retain one lineage
family ID**, so the family-based split must keep them entirely within the same
split. The generator records the option in the prepared source receipt and
pipeline fingerprint. This is a causal multitask *data unit test*, not a
measurement that an untrained model actually uses both tracks. Evaluate a
trained checkpoint by holding one track constant and swapping only the other;
the corresponding answer component should change while the other remains
unchanged. Run `pytest -q tests/test_omni_counterfactual.py` for source-pair
invariants.

The new `budget_meter: G` counts **physical scored-window megatokens**
across modalities; it requires an explicit `budget` and an event
accumulation limit. `G` is not `D`, `P`, media frames, or Gaussian
`Z`. Existing stage budgets and text-token definitions are unchanged.
Additional counters report media events, known audio samples, video frames,
and image events. Feature-only audio does not claim an original sample count
unless the raw source is explicitly available.

For media pilots set `eval_every: 0` until media evaluation is installed.
The pilot's ordinary text benchmark evaluator cannot score raw PCM/video.
The `DataCursor` and `TrainConfig` fingerprints preserve the selected
event-budget policy on exact resume.

## Pin and cache pretrained features

The optional extractor script supports two reviewed, conventional donor
APIs rather than an opaque pluggable LLM:

- [Wav2Vec2 base 960h](https://huggingface.co/facebook/wav2vec2-base-960h):
  16 kHz speech features, Apache-2.0 model repository.
- [SigLIP2 base patch16 224](https://huggingface.co/google/siglip2-base-patch16-224):
  ordered image patches, reusable frame-by-frame for video, Apache-2.0 model repository.

Install only for offline preparation:
`pip install transformers pillow`. Review repository/model terms, do not
enable remote code, and supply a **full immutable 40-digit revision**.

```bash
python scripts/cache_omni_features.py \
  --root artifacts/omni-pilot \
  --record artifacts/source-media-record.json \
  --output artifacts/cached-media-record.json \
  --model facebook/wav2vec2-base-960h \
  --revision <FULL_VERIFIED_GIT_SHA>
```

Donor weights are **local-only by default**. Pass `--allow-download`
explicitly if appropriate. The command emits `extractor`,
`feature_dim`, token count and asset SHA. Update the prepared Episode
media descriptor and `TrackEvent.asset_sha256`, then **reprepare immutable
episode shards**. Do not modify an already exported dataset in place.
Use matching `model_overrides.audio_feature_dim` and
`model_overrides.audio_feature_id` in the training configuration.
For the visual donor, replace `--model` with the supported SigLIP2 ID.

The donor cache is immutable and deliberately produced **offline**.
Do not fetch multi-gigabyte donor weights from the training loop or put the
donor transformer on every serving GPU. For production, a separate
feature-preparation/serving service must reproduce the same pinned encoder,
preprocessing, sample rate, feature ordering and masks.

## High-quality data and tasks

The procedural pilot tests implementation correctness; it is not a source
of natural audio/video generation quality. Acquire licensed original media
with provenance, speaker/source family ancestry and clip/time alignment.
Prefer verified aligned speech/transcripts, image/video event descriptions,
natural conversational audio, and genuinely synchronous audiovisual samples.
Keep benchmark splits frozen at parent audio/video and speaker-family level.

A worthwhile next task set has **counterfactual track ablations**:
change only the audio while holding video constant, swap video frames while
preserving narration, join independent evidence tracks, or swap an authorized
voice/style reference while holding the semantic answer constant. Require
the task to fail if one indispensable track is removed. Use the
[omnimodal training plan](omnimodal-architecture-and-training.md) for the
full family inventory and OM0–OM5 progression.

## Version and ownership notes

The new four media losses and the external donor projectors alter training
configuration/model shapes. Exact resume must use the new checkpoint owners,
loss vocabulary, source manifest, feature ID, and fixed media data. Older
checkpoints are not silently granted missing optimizer owners or relabeled
loss weights. Existing single-track text code paths continue to operate.

**Known next-quality milestones:** choose and pin a high-quality speech
codec/vocoder, integrate a facet-conditioned speech flow generator, integrate
an image/video latent VAE/flow renderer, train aligned media controls through
the one shared control system, and add held-out perception and audiovisual
synchronization evaluation. These require pretrained weights and paired
training data not shipped in this repository.
