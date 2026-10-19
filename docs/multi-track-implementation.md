# Multi-track v7 implementation status

**Date:** 2026-10-19. **Branch:** `feat/multitrack-foundation-20261019`.
**PR:** [#10](https://github.com/werg/megatoken/pull/10).

This page describes the code that has been written, not model quality. The complete
research and engineering design is in [multi-track-plan.md](multi-track-plan.md).

## Architecture status

| Capability | Current state |
| --- | --- |
| Breaking architecture v7 + strict format identities | Implemented in code |
| Same-meridian full attention using one C and shared final KV | Implemented in code |
| Joint P12 live queries and existing per-event expert topology | Implemented in code |
| Stable session track indices and learned track/modality features | Implemented in code |
| Atomic whole-meridian eviction | Implemented in code |
| Joint Gaussian action, typed group tape, density and deterministic replay | Implemented in code |
| Heterogeneous text, audio, image and video observations | Trainable baseline implemented |
| Shared E reduction and 32-facet readout across modalities | Implemented in code |
| Direct waveform and joint image/video realizers | Trainable native baseline, **untrained** |
| Canonical typed control payloads and public reference identities | Implemented in code |
| Grouped differentiable multimodal bootstrap objective | Implemented as an independent training API |
| Group-native inference and binary media JSONL protocol | Implemented in code |
| Speculative atomic branches and ready tool result ingestion | Implemented in code |
| Parallel external IO tool execution | Basic executor implemented |
| RAG evidence and worker tracks | Supported structurally as ordinary observations/tracks |
| Data schema meridian IDs, track identities and media alignment intervals | Implemented in code |
| Native distributed/recurrent *multi-track* training integration | **Not implemented** |
| Paired speech/video production corpus acquisition | **Not implemented** |
| Pretrained speech/vocoder and modern image/video flow backbones | **Not implemented** |
| High-quality generated voices/images/videos | **Not established** |
| Joint audiovisual lip-synchronization and perceptual evaluations | **Not implemented** |
| Full multitrack latent-only outcome RL and executed-trajectory integration | **Not implemented** |
| GPU memory/latency benchmarking of grouped rendering | **Not measured** |

The neural media implementations deliberately contain no external downloads. They
produce valid, bounded tensor outputs and expose gradients but cannot be claimed
to synthesize useful perceptual content from random initial weights. Production
audio/image/video quality requires data and additional modality modeling.

## Source organization

- `src/megatoken/model/core.py`: common C, additional track/modality features.
- `src/megatoken/model/multitrack.py`: immutable shared state, grouped E/C/P,
  Gaussian policy, tapes, realization and grouped bootstrap.
- `src/megatoken/model/media.py`: trainable audio and visual stems with
  shared text E reduction/readout, lightweight native media decoders.
- `src/megatoken/controls.py`: byte-exact seed/reference/tool protocol with
  typed settings, media asset references and track-specific overrides.
- `src/megatoken/data/schema.py`: `TrackSpec`, `TrackEvent`, `MediaSpan`,
  `Meridian`, and validated episode metadata.
- `src/megatoken/multitrack_runtime.py`: per-session scheduler, speculative
  commit/rollback and asynchronous IO-only tool dispatch.
- `src/megatoken/multitrack_service.py`: group-native JSONL transport, including
  explicitly shaped little-endian FP32 media buffers.
- `tests/test_multitrack*.py`: group causality, replay, media, controls, training
  gradients, concurrency and serving contract tests.

Existing `FacetGenerator` shares its trained core, text decoder and Gaussian
with the new generator and owns the new media realization weights. There is no
second recurrent model and no independent modality KV history.

## Local smoke and training entry points

The API can run entirely on CPU with the tiny model and the existing byte codec:

```python
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime

core = FacetedCore(ModelConfig.tiny())
model = FacetedGenerator(core, ByteCodec.byte_fallback())
runtime = MultiTrackRuntime(MultiTrackGenerator(model))
session = runtime.open()

runtime.observe(session, (
    ObservedEvent("user", "text", b"Please describe this scene."),
    ObservedEvent("camera", "image", torch.rand(1, 3, 16, 16)),
))
result = runtime.generate(
    session,
    tracks=("voice", "picture"),
    specs=(
        MediaOutputSpec("audio", sample_count=512),
        MediaOutputSpec("image", height=16, width=16),
    ),
    effort=1,
)
print(result.tape.modalities)
print([None if x is None else tuple(x.shape) for x in result.media_outputs])
```

For grouped bootstrap learning:

```python
from megatoken.generation.structure import ActionSpec

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
state = runtime.generator.tracks.initial_state(1)
out = runtime.generator.supervised_group(
    state,
    tracks=("caption", "image"),
    targets=(b"A red object", torch.rand(1, 1, 3, 16, 16)),
    specs=(ActionSpec(max_tokens=64),
           MediaOutputSpec("image", height=16, width=16)),
    effort=1,
)
out.loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
# Rebuild an episode state and advance core.model_version after weight updates.
```

This bootstrap API is a reference gradient path and has not been wired into the
existing P1/M1/M2/LC/S/D/R distributed stage engine. It should not be treated
as a replacement training recipe yet.

## Group-native JSONL protocol

Export an actually trained v7 checkpoint with the existing export command,
then run:

```bash
megatoken serve-multitrack --model artifacts/served-v7
```

The server accepts `open`, `observe`, `generate`, `speculate`, `accept`,
`cancel`, `status` and `close`. For an example request (supply a live session
ID returned by `open`):

```json
{"op":"generate","session":"<session-id>","effort":1,"outputs":[
  {"track":"caption","modality":"text","max_tokens":64},
  {"track":"voice","modality":"audio","sample_count":16000},
  {"track":"illustration","modality":"image","height":64,"width":64}
]}
```

Media input uses a `payload_f32_b64` field and an explicit `shape`.
For audio the tensor is `[1, samples]`; image is `[1, 3, H, W]`; video is
`[1, frames, 3, H, W]`. Raw media output buffers use little-endian FP32
with their shape alongside the base64 payload. Text stays byte-exact.
Control payloads use `control_b64` per output track, produced by the common
control builder. Speculative output is not a canonical commit until `accept`.

The server does **not** execute model-produced tool commands, download remote
reference assets or supply trained speech/video weights automatically.

## Strict formats and no migrations

- Architecture version: **7**.
- Legacy scalar latent tape protocol changed to v7; multi-event tape is
  `fmt-multitrack-facets-v7`.
- Checkpoint and served-bundle version: **7**.
- Episode and prepared dataset schema: **3**.
- Prior checkpoints, episode archives, latent action tapes and served bundles
  are not importable. Retrain and regenerate v7 artifacts; no migration path
  or numerical preservation requirement exists.
- A singleton meridian is a normal one-event case of the new architecture.

## Validation required

The branch includes tests intended to cover joint same-meridian influence,
absence of unintended state writes on prediction, atomic history retention,
density replay, multimodal tensor shapes, media realization, gradient paths,
control serialization, speculative branch invalidation and concurrent IO.

GitHub Actions has reported **early failed runs before exposing executable
test steps or logs** during this branch's development. Treat the test files
as written but do not treat CPU/GPU verification as complete. Execute locally:

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
pytest -q tests/test_multitrack.py tests/test_multitrack_media.py tests/test_multitrack_runtime.py
pytest -q
```

The previous architecture's receipt files are historical and are not
validation evidence for v7. Once the code runs, capture new results and
profile actual audio/video E/R behavior before making quality claims.

## Next required engineering work

The path from this baseline to a truly trained omnimodal system is
straightforward but substantial: integrate grouped meridians with all
training stages and optimizer ownership; add a recurrent grouped objective
with real recorded-action continuations; select and train/acquire high-quality
codec and flow-based media heads; assemble licensed aligned multimodal data;
train the complete shared model; and evaluate cross-modal understanding,
generation quality, synchronized audio/video, and perceptual control transfer.

The core multi-track dataflow is designed so these efforts extend E/R and
the existing learner rather than creating separate agent transformers.
