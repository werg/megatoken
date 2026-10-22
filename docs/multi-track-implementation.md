> **Scope:** This page describes the multi-track code (architecture v10), not model quality. Training is launched only through the phase-gated recipes in [README.md](../README.md) and [retrieval-memory-implementation-status.md](retrieval-memory-implementation-status.md); media phases are in [media.md](media.md). Older stage commands and optional-memory instructions are not supported.

# Multi-track implementation status (architecture v10)

This page describes the code that has been written, not model quality. The complete
research and engineering design is in [multi-track-plan.md](multi-track-plan.md).

## Architecture status

| Capability | Current state |
| --- | --- |
| Breaking architecture v10 + strict format identities | Implemented in code |
| Same-meridian full attention using one C and shared final KV | Implemented in code |
| Joint P12 live queries and existing per-event expert topology | Implemented in code |
| Stable session track equality bias and learned modality features | Implemented in code |
| Atomic whole-meridian eviction after the incoming C context read | Implemented in code |
| Joint Gaussian action, per-event density and effort, canonical track-order draws, grouped replay | Implemented in code |
| Heterogeneous text, audio, image and video observations | Implemented in code (media as teacher features) |
| Shared E reduction and 32-facet readout across modalities | Implemented in code |
| Media input as pinned EmbeddingGemma2 hidden states through `MediaEncoder` and shared E | Implemented in code ([media.md](media.md)) |
| Resident frozen Pocket / Sana / Sana-Video realizers with trainable `FacetCondition` | Implemented in code; **real-weight quality not verified** |
| Canonical typed control payloads and public reference identities | Implemented in code |
| Grouped differentiable multimodal bootstrap objective | Implemented; driven by the training engine with a task recipe |
| Group-native inference and binary media JSONL protocol | Implemented in code |
| Speculative atomic branches and ready tool result ingestion | Implemented in code |
| Parallel external IO tool execution | Basic executor implemented |
| RAG evidence and worker tracks | Supported structurally as ordinary observations/tracks |
| Data schema meridian IDs, track identities and media alignment intervals | Implemented in code; runtime propagation incomplete |
| Distributed (DiLoCo) multi-track media training phases | Configured ([media.md](media.md)); **no measured GPU-scale result** |
| Paired speech/video production corpus acquisition | **Not implemented** ([roadmap](pretrained-omnimodal-bottleneck-plan.md)) |
| High-quality generated voices/images/videos | **Not established** |
| Joint audiovisual lip-synchronization and perceptual evaluations | **Not implemented** |
| Full multitrack latent-only outcome RL and executed-trajectory integration | **Not implemented** |
| GPU memory/latency benchmarking of grouped rendering | **Not measured** |

Media realization uses frozen pretrained donors (Pocket, Sana, Sana-Video)
installed from a locked pack manifest; this code never downloads weights. A
model without installed realizers can observe and generate text but refuses
media output. Perceptual quality of the trained conditions is not established.

## Source organization

- `src/megatoken/model/core.py`: common C, additional track/modality features.
- `src/megatoken/model/multitrack.py`: immutable shared state, grouped E/C/P,
  Gaussian policy, tapes, realization and grouped bootstrap.
- `src/megatoken/model/media.py`: `MediaEncoder` (pinned 512-wide teacher
  features into the shared E reduction), `MediaOutputSpec`, and the
  `MediaRealizers` container for the resident realizers
  (`src/megatoken/omni/realizers.py`).
- `src/megatoken/teacher.py`: `PINNED_TEACHER`, the identity every
  media event's features must carry.
- `src/megatoken/controls.py`: byte-exact seed/reference/tool protocol with
  typed settings, media asset references and track-specific overrides.
- `src/megatoken/data/schema.py`: `TrackSpec`, `TrackEvent`, `MediaSpan`,
  `Meridian`, and validated episode metadata.
- `src/megatoken/multitrack_runtime.py`: per-session scheduler, speculative
  commit/rollback and asynchronous IO-only tool dispatch.
- `src/megatoken/multitrack_service.py`: group-native JSONL transport, including
  explicitly shaped little-endian FP32 media buffers.
- `tests/media_support.py`: teacher-feature helpers and `FakeRealizer`, used by
  `tests/test_multitrack*.py` (group causality, replay, media, controls,
  concurrency and serving contract).

The `FacetedGenerator` shares its trained core, text decoder and Gaussian
with the grouped generator and owns the media realizers. There is no
second recurrent model and no independent modality KV history.

## Local smoke example and training entry points

The API runs on CPU with the tiny model and the byte codec. Media events are
**pinned EmbeddingGemma2 hidden states** `[1, T, 512]` tagged with
`extractor=PINNED_TEACHER`; raw samples or pixels are rejected. Media output
needs resident realizers. In production they come from a locked pack manifest,
`build_model(config, codec, packs="lock.json")`
([media.md](media.md)); this smoke example instead installs the tiny test
doubles from `tests/media_support.py` (a real `Realizer` whose frozen donor is
a small linear map), so no weights are downloaded and the output is
meaningless noise. Run it from the repository root with
`PYTHONPATH=src:. python smoke.py`:

```python
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.teacher import PINNED_TEACHER
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime
from tests.media_support import features, install_fake_realizers

core = FacetedCore(ModelConfig.tiny())
model = FacetedGenerator(core, ByteCodec.byte_fallback())
install_fake_realizers(model)  # production: build_model(..., packs="lock.json")
runtime = MultiTrackRuntime(MultiTrackGenerator(model))
session = runtime.open()

runtime.observe(session, (
    ObservedEvent("user", "text", b"Please describe this scene."),
    # features(7): 7 random teacher tokens [1, 7, 512]; real ones come from
    # the offline EmbeddingGemma2 extraction of the camera frame.
    ObservedEvent("camera", "image", features(7), extractor=PINNED_TEACHER),
))
result = runtime.generate(
    session,
    tracks=("voice", "picture"),
    specs=(
        MediaOutputSpec("audio", sample_count=512, sample_rate=24000),
        MediaOutputSpec("image", height=16, width=16),
    ),
    effort=1,
)
print(result.tape.modalities)
print([None if x is None else tuple(x.shape) for x in result.media_outputs])
```

This prints `('audio', 'image')` and `[(1, 512), (1, 1, 3, 16, 16)]`. Speech is
24 kHz mono (`sample_rate=24000` is the only accepted audio rate); video specs
declare `frames` and `fps_num`.

Training does not use an ad hoc loop. The grouped losses (`supervised_meridian`
and the media targets built from prepared `MediaRecord`s) run inside the
training engine, which draws prepared streams according to a task recipe
(`training/task_recipe.py`, `configs/tasks/media-{bridge,joint,tasks}.yaml`)
and runs on DiLoCo like every phase. See [media.md](media.md) for the three
phases and their `configs/train/media-*.example.yaml`, and
[diloco-coreweave.md](diloco-coreweave.md) for the coordinator workflow.

## Group-native JSONL protocol

Export a trained checkpoint and serve it with the locked realizer pack
manifest it was trained with (the bundle records its fingerprint and refuses
any other manifest; text-only bundles take no `--packs`):

```bash
megatoken export --checkpoint runs/media-tasks/checkpoints/latest.json \
  --output artifacts/served --codec <codec> --packs packs/lock.json
megatoken serve-multitrack --model artifacts/served --packs packs/lock.json
```

The server accepts `open`, `observe`, `generate`, `speculate`, `accept`,
`cancel`, `status` and `close`. For an example request (supply a live session
ID returned by `open`):

```json
{"op":"generate","session":"<session-id>","effort":1,"outputs":[
  {"track":"caption","modality":"text","max_tokens":64},
  {"track":"voice","modality":"audio","sample_count":24000},
  {"track":"illustration","modality":"image","height":64,"width":64}
]}
```

Media input uses a `payload_f32_b64` field, an explicit `shape`
`[1, tokens, 512]` and the pinned `extractor` string: the teacher's hidden
states, never raw PCM or pixels (a wrong extractor, missing extractor or raw
tensor is rejected and commits nothing). Realized media output buffers use
little-endian FP32 with their shape (audio `[samples]` at 24 kHz, image
`[1, 3, H, W]`) alongside the base64 payload, plus a `tape_payload`: a
`media-sha256-v1` digest binding the bytes to the realizer identity. Text stays
byte-exact.
Control payloads use `control_b64` per output track, produced by the common
control builder. Speculative output is not a canonical commit until `accept`.

The server does **not** execute model-produced tool commands, download remote
reference assets or fetch pack weights.

## Strict formats and no migrations

- Architecture version: **10** (`ARCHITECTURE_VERSION`); the grouped tape is
  `GroupActionTape` v10 (`fmt-multitrack-facets-v10`), recording renderer seeds
  and realizer pack identities per event. The text-only single-event tape
  keeps its own `fmt-canonical-facets-v8` identifier.
- Checkpoints carry the locked pack manifest's `packs_fingerprint`.
- Episode and prepared dataset schema: **3**.
- Prior checkpoints, episode archives, latent action tapes and served bundles
  are not importable. Retrain and regenerate artifacts; no migration path
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
validation evidence for v10. Once the code runs, capture new results and
profile actual media E/R behavior before making quality claims.

## Next required engineering work

The remaining path to a trained omnimodal system: verify each pack on real
weights and an accelerator (`scripts/verify_packs.py`); assemble
licensed aligned multimodal data ([plan](pretrained-omnimodal-bottleneck-plan.md));
run the bridge, joint and task phases ([media.md](media.md)); add a recurrent
grouped objective with real recorded-action continuations; and evaluate
cross-modal understanding, generation quality, synchronized audio/video, and
perceptual control transfer. A multi-worker DiLoCo launcher on
`megatoken.cloud` does not exist yet ([diloco-coreweave.md](diloco-coreweave.md)).

The core multi-track dataflow is designed so these efforts extend E/R and
the existing learner rather than creating separate agent transformers.

## Cross-track RoPE and media alignment

Core RoPE uses **one logical meridian index per committed group**. Every
selected facet in that group has the same time coordinate, irrespective of
track ordering or modality. The same-track relation is a zero-initialized
learned per-query-head attention bias for matching categorical track identities;
it applies to current, temporary, and retained-history attention, without an
ordinal track axis or separate track KV. There is no additional cross-track
history window: all retained earlier meridians remain visible.

`MediaSpan` and `alignment_id` describe content-time relationships, not
causal permission or rotary positions. They are currently validated in the
dataset schema, **not yet wired to a synchronized media realizer or clock-aware
serving scheduler**. An observed audio clip and a later transcript can share
media time while remaining causally sequential. Predictors must not receive
future target-derived span or duration metadata.
