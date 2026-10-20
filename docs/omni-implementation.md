# Pretrained omnimodal implementation and acceptance

This is the implementation companion to
[pretrained-omnimodal-bottleneck-plan.md](pretrained-omnimodal-bottleneck-plan.md).
The authoritative contract remains **architecture v8**: E encodes observations
to raw Z; P selects a raw [32, 512] block for each generated event; R consumes
only that selected block and a structural spec/noise; C publishes its complete
joint group to shared history for *future* predictions.

## Components and boundaries

| Responsibility | Resident/cached implementation | Safety boundary |
|---|---|---|
| Speech meaning | Whistle 16 kHz frozen embeddings (`omni/perception.py`) | Never uses future source chunks |
| Speech appearance | Matching Pocket-Mimi 24 kHz continuous latents | P sees approved references only after E commits them |
| Speech generation | Pocket FlowLM + frozen Mimi, raw-facet projection | No text tokens/voice prefix enter the native student R |
| Image/video perception | MiniCPM-V 4.7 exported vision tower + trained merger, 16x/4x | LLM excluded from export |
| Image generation | Sana 0.6B transformer + matching DC-AE | No image-text donor encoder at native R runtime |
| Video generation | SANA-Video 2B transformer + checkpoint-matched LTX-2 VAE | No silent VAE substitution |
| AV/endpoint relations | Matched sampled peer events, validity/timing masks, endpoint maps | Gold caption/endpoint never conditions same-step P |
| Generic acoustic/sound later extension | Separate declared codec/generator family | Not a replacement for Pocket speech |

The model uses the existing grouped E/C/P/R, Gaussian and loss-ledger engine.
No second semantic scheduler or modality-specific reasoning transformer is
introduced. Native debug PCM/pixel heads still exist to support the tiny CPU
reference pilot; selecting `renderer=native` is **not** pretrained output.
The `FacetFlow` generic adapter is an untrained primitive, not a Pocket/Sana
checkpoint.

### Native pretrained checkpoints and terms

The first-party pack manifest is a **full file-digest lock**, not a model-name
alias. Every checked-out file, including the generator, codec decoder, configs
and tokenizers, is hashed; a Pocket config with any unresolved remote
`hf://`/HTTP resource is rejected. The manifest records exact upstream
40-hex commit revisions and is checked both before loading and at checkpoint
resume. The selected renderer+codec identity is also included in new v8 group
action tapes, independent of renderer residency order.

The `configs/omni/pack-lock.example.json` placeholders cannot pass real
validation. Gated models require their upstream access agreement. Use the
actual languages/rights supported by the locally pinned Pocket variant;
do not assume that an English checkpoint also supports German generation.

## Installation and validation without any training

Install the repository's existing optional training/testing environment, then
install the tested, pinned versions of Pocket, Diffusers, Transformers, PEFT
(if using LoRA), and the Whistle/Needle runtime as appropriate for each pack.
A production lock must record those dependency and upstream revisions.

```bash
megatoken doctor
python -m pytest -q tests/test_omni_infrastructure.py \
    tests/test_omni_media_training.py \
    tests/test_omni_sana_contracts.py \
    tests/test_omni_pocket_codec_state.py \
    tests/test_omni_session.py \
    tests/test_omni_codec_geometry.py
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

GitHub Actions must run and pass independently; a runner-provisioning failure
is neither proof of successful tests nor a model regression.

### Checkpoint acquisition/locking (offline until explicitly invoked)

1. Supply a JSON inventory of the approved repo IDs, **exact** 40-hex commit
   SHAs and local destination names. Acquisition is a separate user-triggered
   operation, not part of model construction.
2. `python scripts/acquire_pretrained_omni.py --root artifacts/omni-packs
   --inventory inventories/pinned-models.json --output artifacts/acquired.json`
3. For Pocket, materialize a local YAML whose flow, Mimi and tokenizer resource
   paths are local and whose referenced weights are present. An unmodified
   upstream config may still refer to a remote Hub URL and will **fail** lock.
4. `python scripts/lock_pretrained_omni.py --root artifacts/omni-packs
   --inventory inventories/local-pack-paths.json
   --output artifacts/omni-packs/lock.json`
5. `python scripts/export_minicpm_tower.py --manifest
   artifacts/omni-packs/lock.json --output artifacts/omni-packs/minicpm-export`;
   include the exported tower, trained merger and export receipt in a separate
   `minicpm_export` lock identity before using the feature extraction script.
6. `python scripts/verify_pretrained_omni.py --manifest
   artifacts/omni-packs/lock.json --device cuda --roundtrip --rollout
   --output artifacts/omni-donor-preflight.json`

The real donor preflight checks projection gradients, frozen codec parameters,
native training-loss finiteness, matched encode/decode with synthetic input,
and a true free rollout through each donor. It reports measured physical
contracts and synthetic round-trip error, **not** intelligibility, content
fidelity or human-approved performance. Full-rollout video requires sufficient
GPU memory; cached-feature training need not keep its video decoder resident.

## Inputs, task compilation and training

Raw source media is not assumed to use one universal sample rate. Whistle
ingests 16 kHz and Pocket-Mimi requires 24 kHz; other profiles explicitly
declare clocks, valid spans and time offsets. The `MediaRecord` v2 format
supports an exact displayed `OutputGeometry` (including requested vs
teacher-only provenance), multiple E feature views and a matching pretrained
`codec_sha256`. Native codec targets are immutable checksummed FP32 arrays
with valid coordinate masks. A large video can use a much smaller cached
MiniCPM feature source and a native codec-latent target without also storing
an enormous raw FP32 pixel array in the prepared root.

The offline scripts
`cache_pretrained_omni.py`, `cache_omni_codecs.py`,
`cache_pocket_teacher.py`, and `make_omni_endpoint_targets.py`
exist for later, explicit source preparation. A teacher-side Pocket trajectory
stores the exact **pre-Mimi decoder** continuous U coordinates plus its
corresponding waveform and EOS flags. It does not become an external prompt.

`compile_aligned_bundles.py` converts approved, source-family-aligned
bundles into ordinary native `Episode` JSONL:
- Distinct observed meridians for real text/media/voice/tool evidence.
- Explicitly generated content-plan meridian for underspecified creation;
  the plan is *not* an external prompt or a future label visible to P.
- Joint output meridian for direct ASR/TTS, speech translation,
  image/description, video/audio/endpoint generation and counterfactuals.
- Ordered recorded edit requests and actual artifact-version ancestry;
  later gold edits never treat a different student artifact as their base.
- Typed supervision relationships kept out of model-visible input and splits
  frozen by original source family, not by generated view.

The resulting canonical JSONL is consumed by the existing native data source
adapter with `preserve_native_stream=true` and the ordinary prepared-dataset
pipeline. Dataset assembly and acquisition are intentionally not performed by
this implementation.

`configs/train/omni-concordant.yaml` connects the existing grouped learner
to the full versioned sampling recipe. `configs/omni/concordant-v1.yaml`
chooses family probabilities **20/20/15/15/15/15%**: replay, literal speech,
direct translation, image tasks, AV/endpoints and sequential editing.
This is a family-choice distribution, not a GPU-time quota. Realized event
counts, valid codec coordinates, available modalities, joint source coverage
and counterfactual coverage are measured, not inferred from a stage label.

```bash
# Only once approved data and checkpoint assets exist; shown for operability,
# not executed by this change:
megatoken train --config configs/train/omni-concordant.yaml
```

During bootstrap, the VAE-style raw Z correspondence KL is summed over active
coordinates and normalized through the existing distributed numerator/count
loss ledger. Recurrent encoder-only reconstruction freezes R/C where required.
Codec/flow objectives use per-modality valid coordinates. Every shared weight
has one optimizer owner; the frozen donor VAE/Mimi decoder receives no updates.
Renderers consume no source U skip connections, gold speech transcript, gold
image description or hidden peer block.

## Serving, tapes, interrupts and exact replay

`OmniSession` wraps the existing `MultiTrackGenerator` and accepts checked
`ObservedEvent` features, including simultaneous frozen Whistle+Pocket or
MiniCPM detail views. `resolve_reference` converts approved voice/style asset
references into actual E observations before selection; an asset hash string
in control text is only provenance. A proposal is a copy-on-write whole group.
New evidence invalidates unaccepted work. Acceptance atomically commits all
selected raw blocks into one shared C/KV update.

A `GroupActionTape` contains per-event control bytes/revisions, raw FP32
facets, group roles, exact output specs, policy densities, deterministic
renderer seeds and pinned generator/codec pack identities. A
`RenderedOutputStore` optionally moves large physical audio/video payloads
out of tape JSON into immutable content-addressed files; compact tapes retain
their lengths/digests. Delivered byte cursors and interruption records cannot
be retroactively erased by rejecting a later speculation. Rebuild uses
observations and the same complete group tape, not a new teacher target.

**Replay guarantees differ:** same-backend raw-facet/action identity; exact
byte-level codec/renderer replay only when the pinned backend and deterministic
sampling implementation support it; cross-backend perceptual equivalence is a
separate evaluation, not presumed bitwise identity.

## Evaluation and acceptance: independently measured only

`scripts/eval_omni_concordance.py` scores decoded sampled peers, not gold
future outputs. It includes independently recognized speech/text WER/CER,
endpoint RMSE after the final displayed crop/time sampling, sound-event timing,
edit-region preservation and optional external semantic verifiers with explicit
provenance. A missing verifier produces an **unmeasured** result rather than
a fabricated perfect score. Native codec, codec roundtrip, bottleneck E/R and
causal P/R remain distinct comparisons. Include source/speaker/prompt/edit
held-outs, a human-scored subset, latency, counts, spatial relations and OCR.
Do not use teacher-only duration or a gold content plan in free evaluation.

Acceptance requires both the CPU contracts and pinned-checkpoint hardware
smokes to pass, followed by genuine free-rollout task results. No dataset
preparation, training, checkpoint acquisition, benchmark scoring or pretrained
model performance claims have been made by merely committing this code.
