> **Scope:** This page describes the multi-track code (architecture v15), not model quality. Training is launched only through the phase-gated recipes in [README.md](../README.md); the memory mechanism is in [continuous-state-transducer-plan.md](continuous-state-transducer-plan.md) and media phases are in [media.md](media.md). Older stage commands are not supported.

# Multi-track implementation status (architecture v15)

This page describes the code that has been written, not model quality. The complete
research and engineering design is in [multi-track-plan.md](multi-track-plan.md).

## Dynamic addressing and scheduling

Track addresses now serve attachment and equality only. Neither the shared expert
router nor the retrieval-conditioned expert adjustments contain a learned table
indexed by track number. Public descriptions and event requests supply meaning for
singleton and grouped predictions. The instruction encoder is recomputed under the
current weights; no detached description cache survives an optimizer update.
Changing a purpose requires a new revision. Changing a label preserves the address.
The former track setting is only a default active-work budget; it no longer bounds
session lifetime or state geometry. Memory ingestion and hindsight no longer enforce
the old lifetime limit.

Each selected group has distinct event addresses, including peers on the same track.
Random draws and rendering seeds are assigned by event address. An output-modality
classifier runs before content prediction; supported output specifications determine
its legal choices. Requests can constrain a single modality or offer several options.
Absent decoders are rejected before selection, and output extents must be public
requests. Tapes retain the legal modality masks and categorical log probabilities,
alongside continuous content densities, and replay checks both. Selection records
explicitly distinguish unrendered content from a completed rendering outcome; failed
rendering can be archived without inventing an emitted text event. Served exports
declare and verify the installed output modalities.

The runtime owns track creation, assignment, progress, waiting, completion,
cancellation and resumption. Creation returns an address; completed addresses remain
recognizable for late replies. Lifecycle control messages use the existing constrained
tool decoder and training objectives. They are selected actions, rather than an
additional unrecorded policy. `lifecycle_tools` exposes their schemas;
`execute_lifecycle` accepts only already committed groups. `track_output` queues work
using host-provided output specifications (text by default). Other modalities require
explicit supported geometry and installed decoders.

Ready work has event prerequisites, task revisions, budgets and priorities. The
bounded queue batches compatible ready events without waiting for unrelated work.
Stale requests are recorded and removed. Periodic oldest-first service prevents
persistent priority traffic from starving other ready work. Selected representations,
rendering outcomes and acknowledged delivery are separate records. Rendering uses an
immutable selection snapshot; new input can commit while rendering runs. Stopping
delivery does not erase a selection or retract delivered content. Ready tool results
publish to the original task; interrupted result publication resumes at its next
chunk, and duplicate dispatch cannot repeat an effect through the same runner.

`start_generation` commits the selection and starts asynchronous rendering;
`poll_generation` reports completion or failure. `stop_delivery`, `delivered` and
`release_generation` control delivery and free retained physical outputs. Global and
session rendering admission are bounded before selection. The synchronous
`generate` and speculative `speculate`/`accept` interfaces remain available. Call
`MultiTrackRuntime.close()` when the host stops serving.

The prepared episode and aligned-source compiler retain optional exact media ticks,
clock rates, speakers, direction, transformation provenance and external availability
and selection times. Validation rejects events selected before availability,
same-group prerequisites, dependent stream chunks grouped as peers, and call results
without a completed earlier call. These metadata checks do **not** establish causal
feature extraction for an acquisition pipeline that has not been audited.

### Delivery boundary

This delivery changes the model/runtime contracts and adds modality supervision to
the existing grouped learner and shared loss accounting. It does not establish a
trained dynamic-worker policy. The collection adapter and native-history pairing
support grouped selections and media receipts, including empty and failed peers.
Verified outcome supervision and whole-decision progress accounting are integrated
with the existing training path. Timed source preparation and checked execution-trace
import use ordinary deduplication, export, training, and held-out coverage gates.
The [training runbook](dynamic-training-runbook.md) gives executable preparation
steps and distinguishes real local producers from acquired supply. No acquired
production timed corpus, natural full-duplex training run,
decoded synchronization result, or accelerator speedup is claimed. Timing metadata
alone cannot make a non-streaming decoder full duplex. Source selection, audited
channel/prefix preparation, split-wide lineage checks, production supply coverage and
generated held-out acceptance remain required before a production launch.

The focused dynamic-track tests exercise address remapping, retirement beyond the
old limit, categorical replay, same-track speech/caption peers, finite gradients,
input during failed rendering, cancelled delivery, late results after rename,
prerequisite scheduling, stale tasks, memory release, and exactly-once lifecycle
execution. Test media decoders isolate protocol behavior; they do not measure quality.
Existing training/export/restart, distributed, retrieval, student and media suites
remain regression gates.

The completion review used the local CPU environment with PyTorch 2.14.1. Its broad
run passed 2,476 tests, skipped five optional ICU checks, and found one outdated
assertion requiring the old retrieval-response fields. That assertion now includes
the exact evidence address. The subsequent affected-boundary run passed all 33
tests, including collection, retrieval, native history, and a newly added collected
speech/caption export-and-backward test. These sets overlap and must not be added
together as independent coverage. Ruff lint and whitespace checks passed. A fresh
authored-text acceptance run completed preparation, optimization, checkpoint restart,
export, and generation through the public service. No real-weight media or
production-quality result is implied. See the runbook for the remaining acquisition
and evaluation gates.

## Architecture status

| Capability | Current state |
| --- | --- |
| Breaking architecture v15 + strict format identities | Implemented in code |
| Same-meridian full attention using one C and shared final KV | Implemented in code |
| Joint P12 live queries and existing per-event expert topology | Implemented in code |
| Stable session track equality bias and learned modality features | Implemented in code |
| Atomic whole-meridian eviction after the incoming C context read | Implemented in code |
| Joint Gaussian action, per-event density and effort, canonical event-order draws, grouped replay | Implemented in code |
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
| Data group IDs, track/event identities and exact media intervals | Implemented; audited source preparation remains incomplete |
| Distributed (DiLoCo) multi-track media training phases | Configured ([media.md](media.md)); **no measured GPU-scale result** |
| Paired speech/video production corpus acquisition | **Not implemented** ([roadmap](pretrained-omnimodal-bottleneck-plan.md)) |
| High-quality generated voices/images/videos | **Not established** |
| Joint audiovisual lip-synchronization and perceptual evaluations | **Not implemented** |
| Executed-trajectory integration and verified whole-selection outcome supervision | Implemented in the ordinary grouped trainer; quality unmeasured |
| Frozen-decoder latent-only outcome improvement | Implemented as an explicit restricted mode of the ordinary student-stage optimizer; held-out quality unmeasured |
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
training engine, which draws prepared streams by the stage plan's data mix
(`src/megatoken/data/stages.json`) and runs on DiLoCo like every stage. See
[media.md](media.md) for where media sits in that plan, and
[diloco-coreweave.md](diloco-coreweave.md) for the coordinator workflow.

## Group-native JSONL protocol

Export a trained checkpoint and serve it with the locked realizer pack
manifest it was trained with (the bundle records its fingerprint and refuses
any other manifest; text-only bundles take no `--packs`):

```bash
megatoken export --checkpoint runs/base-pretraining/checkpoints/latest.json \
  --output artifacts/served --codec <codec> --packs packs/lock.json
megatoken serve --model artifacts/served --packs packs/lock.json
```

This is the one serving program: `megatoken generate` is a one-shot session of the same
runtime (a prompt, then one answer on the `assistant` track), and the frozen student that
collects rollouts for training runs it too, so all three behave the same (lookups, the
adapter mixture, mounted collections and knowledge bases).

The server accepts `open`, `observe`, `observe_text`, `generate`, `speculate`, `accept`,
`cancel`, `status` and `close`, plus the lifecycle, queue and asynchronous-rendering
operations described above. `observe_text` (`track`, `role`, `text` or `payload_b64`)
cuts a message into chunks by the bundle's chunk policy, one meridian each, and only the
last ends the message; `observe` takes events that are already one physical chunk.
For an example request (supply a live session
ID returned by `open`):

```json
{"op":"generate","session":"<session-id>","effort":1,"outputs":[
  {"track":"caption","modality":"text","max_tokens":64},
  {"track":"voice","modality":"audio","sample_count":24000},
  {"track":"illustration","modality":"image","height":64,"width":64}
]}
```

To request corresponding peers on one continuing track, give distinct event addresses:

```json
{"op":"start_generation","session":"<session-id>","effort":1,"outputs":[
  {"track":"reply","event_id":"speech-1","modality":"audio","sample_count":24000},
  {"track":"reply","event_id":"caption-1","modality":"text","max_tokens":64}
]}
```

The response supplies `selection_revision`. Send it to `poll_generation`; acknowledge
actual playback with `delivered` and its `event_ids`, then `release_generation` after
archiving the tape. Input can arrive through `observe` or `observe_text` while rendering
runs. `stop_delivery` suppresses later delivery and preserves the selected action.
Already played audio remains delivered. Renderer failure reports an outcome rather
than undoing shared selected history. Poll responses include the original `selection`
record even when no output was rendered.

To let the classifier choose, supply one output with an `options` list, for example
`[{"modality":"text","max_tokens":64},{"modality":"audio","sample_count":24000}]`.
Every offered decoder must be installed before selection; no target duration is inferred.
Repeated calls may omit event addresses when their tracks are distinct within the group;
the runtime then allocates fresh event addresses from the session chronology.

Media input uses a `payload_f32_b64` field, an explicit `shape`
`[1, tokens, 512]` and the pinned `extractor` string: the teacher's hidden
states, never raw PCM or pixels (a wrong extractor, missing extractor or raw
tensor is rejected and commits nothing). Realized media output buffers use
little-endian FP32 with their shape (audio `[samples]` at 24 kHz, image
`[1, 3, H, W]`) alongside the base64 payload, plus a `tape_payload`: a
`media-sha256-v1` digest binding the bytes to the realizer identity. Text stays
byte-exact.
A text output also takes `kind`, `min_tokens`, `terminal`, public `tools` (or
`tools_json`, with `wire_formats`; a tool call may copy from the text the session holds)
and its causal control: `control_seed`, `control_reference` with `reference_origin`, or the
raw bytes `control_b64`, produced by the common control builder. Text answers carry
`text` when valid UTF-8. Speculative output is not a canonical commit until `accept`.

Knowledge bases ([multimodal-knowledge-plan.md](multimodal-knowledge-plan.md))
are served when the server is started with `--kb-root DIR`, a directory every
path in the requests must lie inside (add `--packs` so files can be prepared):

```bash
megatoken kb build --name manuals --items manuals/*.txt --model artifacts/served \
  --packs packs/lock.json --output kb/manuals.kb      # offline, once
megatoken serve --model artifacts/served --packs packs/lock.json --kb-root kb
```

`kb_open` (`name`, `path`) mounts a base file built for this model, `kb_create`
(`name`) starts an empty base for a user's files, `kb_save` (`name`, `path`), `kb_close`
and `kb_list`. `open` takes `bases` (base
name to blend weight; 0 leaves a base out) and `trained_weight`, and the source
collections the session searches (`collections`, `max_sources`, `threshold`); a session
that mounts a base or a collection finds what is added to it from its next lookup.
`add_files` (`paths`, a `base`, a `collection` or both, optional `metadata` filing the
base's entries) prepares each file once and writes it into a writable base with no
retraining and/or adds it to a mounted collection. A base built for
other model weights is refused.

The server does **not** execute model-produced tool commands, download remote
reference assets or fetch pack weights.

## Strict formats and no migrations

- Architecture version: **10** (`ARCHITECTURE_VERSION`); the one tape serving, collection
  and training exchange is `GroupActionTape` (`fmt-multitrack-facets-v18`), recording
  renderer seeds and realizer pack identities per event. A single-track draw (a student's
  rollout step) is a group of one event; the decoder-level single-event tape
  `fmt-canonical-facets-v8` is only the output of the single-track diagnostic sampler.
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
ordinal track axis or separate track KV. Attention now has separate bounded windows for earlier events on the current
track and recent events on other tracks. Both use the same shared model and
causal event archive. Older evidence must be explicitly read again before it is
used; moving work between tracks does not grant access to future events.

`MediaSpan` and `alignment_id` describe content-time relationships, not
causal permission or rotary positions. They are currently validated in the
dataset schema, **not yet wired to a synchronized media realizer or clock-aware
serving scheduler**. An observed audio clip and a later transcript can share
media time while remaining causally sequential. Predictors must not receive
future target-derived span or duration metadata.
