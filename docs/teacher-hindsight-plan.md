# Plan: hindsight targets from a strong model

**Status:** joint training path implemented; bounded correctness checks only. **Date:** 23 October 2026.
**Current scope:** bring stored teacher targets and known-link supervision into the upcoming
training run through existing mechanisms. Only cheap correctness checks precede training.
Implement the combined teacher and known-link training path for the available compute window;
comparative studies and staged rollouts are not prerequisites. Dense searchable teacher memory
is deferred. See the companion
[structure-supervision plan](structure-seeded-graph-plan.md) for the combined implementation.

## 1. The idea in plain words

The memory bank is filled during training from "hindsight values": for each lookup point in a
training sequence, a training-only component (the hindsight realizer, a small head on the
central processor) produces the knowledge that *would have been ideal* at that point, judged by
what the sequence does next. Early in training that component is itself untrained, so the first
16,384 steps of warm-up produce poor targets and the bank starts out filled with poor values.

The proposal: compute those targets **offline with a strong pretrained model that reads the
actual future text**, store them with the prepared data, and use them during warm-up (and
beyond, as an extra training signal). The aim is to give the reader and memory writer more informative targets early on. Whether
those targets remain useful after pooling, projection and packing must be measured; matching
the existing vector shape does not establish that.

## 2. What the other project (`~/hindsight-kb`) actually offers

I read it and its disk contents. It is a **design reference, not a source of weights or data.**

- Its writer is the pplx-embed-v2-late-0.6b text encoder (a bidirectional Qwen3.5 backbone) with a
  linear value head. For block `j` it reads only the *suffix* text `blocks[j:]`, rendered as a
  separate view (not a mask, because the encoder mixes information along paths a mask cannot
  block), and keeps only block `j`'s token positions (`integrations/pplx.py`).
- It outputs one record per token position: a 128-d key and a 256-d value, not one value per
  lookup. Values are trained only indirectly (through the frozen decoders' next-token loss),
  except in one late stage that regresses onto consolidated records.
- Its **suffix rule** is "a query in block `i` may only retrieve writer records from blocks
  `≥ i+2`". Megatoken's curriculum already contains the same idea (reading `G_{t+1}`, one group
  past the group served).
- **Update (same day): the trained writer has been retrieved.** The local `~/hindsight-kb` folder
  holds no weights, but one of its CoreWeave storage services is still alive and holds the newest
  checkpoint (stage 5a, step 12; the archive is 148 GB because it also contains a 133 GB
  knowledge database that we do not need). Using range requests we fetched only the writer:
  `/home/werg/megatoken-runs/hindsight-kb-writer/` (1.0 GB, every file checked against the
  checkpoint's own checksum list). The older 27 GB stage-3 archives are no longer available (their
  storage services are gone), so step 12 of stage 5a is the newest writer we can get.
- What the writer is (read from the weight file): a 12-layer hybrid backbone (hidden width 1,024,
  494M parameters, Qwen3.5 tokenizer with 248,320 tokens, non-causal), plus a key head
  (1,024 → 128) and a value head (1,024 → 256), run once per token over the suffix view.
- Evidence that it works: the project's own held-out evaluation at stage 3, step 30,000
  (`runs/coreweave/stage3-30000-evaluation.json`) measured next-token loss 1.07 without the
  knowledge base and 0.62 with it (relative gain 0.42; 0.48 for one decoder and 0.32 for the other),
  on 64 held-out identities paired with *related* sources (same repository, toolset or task
  family). Those are the project's own numbers and the pairing makes them favourable, so treat them as
  "the writer carries real, usable information", not as a measure of how it will do for us. We have
  not re-run them.

What to copy from it: the suffix-view idea, the use of character offsets to align the teacher's
tokens to our text, its tests for "no leakage across the suffix boundary", and now the trained
writer itself as a candidate teacher (section 4.1b).

## 3. What megatoken needs (verified in the code)

- A hindsight value `h_t` is a 32,768-wide vector per lookup, in 4-bit packed form (16,384 bytes
  of values, 2,048 bytes of per-16 scales, 4 bytes of row scale = 18,436 bytes).
  `HindsightRealizer` (`model/transducer.py:147-217`) emits it as FP32 and packs then decodes it
  with a straight-through gradient.
- The reader reshapes a value into **64 rows of 512**, adds learned row identities, runs a small
  adapter, then the shared U expert bank, then 32 learned facet queries gather a 32×512 block
  (`model/transducer.py:91-144`). Nothing in it ties the value to a particular teacher: it learns
  what the values mean.
- `h_t` is used in three places (`memory/transducer.py:1043-1049` and
  `training/transducer.py`): the lookup value becomes `(1-α)·bank + α·h`; the new location blends
  in `normalize(Π·h)`; and the weighted-mean write into the edges used (`add_write`) and write-out
  of new edges (`candidate`).
- The only direct loss on `h_t` today is the ingest-head distillation (weight 0.1); the realizer
  itself is trained only through the future-text loss, through the overlay (when α > 0).
- Which group a lookup reads is picked by `HindsightSchedule.reads`; lookups whose read group is
  past the end of the sequence are flagged *sink-only* and are never written.
- Lookup placement is **random per training window** (a fresh seed draws the 4 to 12 meridian
  interval and the metadata dropout, `training/meridians.py:638`). Offline values need a
  *fixed* plan of lookup points stored with the episode.

## 4. How to encode a strong model's hindsight into that format

Three options were considered.

| Option | What it is | Verdict |
| --- | --- | --- |
| A. Fixed offline encoding; the reader adapts | Split the future text's teacher states into 64 spans, project each to 512, pack to 4 bits, store. The reader learns to decode it during warm-up. | **Recommended.** Same 64×512 structure the reader already expects; no new trainable modules. |
| B. Teacher-input realizer | A copy of the realizer whose keys and values read teacher states through a learned `d→512` layer. | More flexible, but needs token-level teacher states stored or recomputed online: much larger. |
| C. Distill the existing realizer to teacher states | Add a regression loss between the realizer's output and the stored value. | Use as an **extra**, not instead: it nudges the realizer toward the teacher's language. |

### 4.1 The recommended encoding (option A), step by step

For each episode and each planned lookup `t`:

1. **Fixed boundary plan.** Compute the lookup points with the boundary planner with no random
   jitter (`plan_boundaries(..., generator=None)`, interval 8), real boundaries included. Store
   the plan in the episode annotations (`annotations_json["hindsight"]["plan"]`).
2. **Choose the future text.** For lookup `t` take the text of groups `G_{t+1}` onward, truncated
   to a horizon (start at 8,192 teacher tokens, a pilot cap rather than an asserted model limit).
   A lookup whose group `G_{t+1}` does not exist is **sink-only**: record the flag and no
   teacher asset. At runtime use ordinary bank retrieval with teacher overlay zero, rather than
   a learned-hindsight fallback or an arbitrary empty-future vector.
3. **Run the teacher** on that future text as its own view (the hindsight-kb suffix-view rule), in
   the teacher's own tokenisation. Keep the final-layer token states (`[tokens, d]`).
4. **Align and pool.** Use character offsets to map teacher tokens back to byte positions, then
   cut the horizon into **64 equal spans by bytes**. Mean-pool each span's token states. This
   gives `[64, d]`.
5. **Condition before projecting.** Subtract a running mean and divide by a running standard
   deviation per channel (whitening estimated on a few thousand lookups), and clip outliers
   (strong models have a few massive-activation channels that would dominate the 4-bit row scale).
   Skipping this can also collapse locations, because `normalize(Π·h)` uses a random projection and a
   common offset would send every lookup to the same place.
6. **Project to 512.** If the teacher is the 512-wide EmbeddingGemma2 that the project already
   pins, no projection is needed. For a wider teacher, use a PCA basis fitted on the same few
   thousand lookups (fixed, not learned, so stored values stay valid). Result: `[64, 512]`,
   flattened to 32,768.
7. **Pack and store.** Pack with `pack_fp4` and store as a content-addressed asset keyed by
   episode + lookup. This needs a new asset kind in `TensorAssetStore` (`data/media_assets.py`):
   today it writes FP32 only and caps arrays at 16M elements, so a packed 4-bit kind is new work.
   The episode annotation records the checksum, the teacher identity, the horizon, the whitening
   and projection identities.

### 4.1b Using the trained hindsight-kb writer as the teacher

The writer retrieved from `~/hindsight-kb` is a promising candidate teacher: it was trained
so that its outputs help other models predict text. That is relevant evidence, but does not
establish that its hidden states will transfer better than a generic embedding model after
our pooling and projection. It produces, per token, a 1,024-wide hidden state, a 256-wide
value and a 128-wide key. The encoding steps in 4.1 stay the same; only steps 3 to 6 change:

- **Run:** the writer over the suffix view, in its own rendering (`"[D] " + text`), on the GB10
  (about 0.5B parameters; measure throughput on this hardware before scaling).
- **Start with the 1,024-wide hidden state.** This is a hypothesis, not an established advantage:
  the 256-wide value is the output directly trained for the original readers. Keep the
  hidden-state choice for the first run; comparing alternatives must not delay training.
- **Pool** into 64 equal byte spans, giving `[64, 1024]`, and condition (whiten and clip) as in 4.1.
- **Project 1,024 → 512** with a fixed PCA basis fitted on a few thousand lookups (measure fitting cost on the pilot),
  pack to 4 bits, store. The reader's existing per-row adapter then learns whatever else is needed.
- **Keys and locations:** do not map the writer's 128-wide key into our 1,024-wide key space. Our
  keys are always trained by gradient and are not tied to any teacher; the location comes from the
  value through `normalize(Π·h)` as before.

Before training, check finite values, non-collapsed locations and reasonable packing error on
one small batch. A linear-readout transfer study is optional later work, not a launch gate.
The teacher's usefulness will be assessed through the real run's ordinary validation.

### 4.1c The interface: different tokenizer, different chunk size

The model and writer use different tokenizers. Pass rendered text to the writer, not the model's
token IDs, and align its output to source bytes. The writer configuration inspected for this
proposal allows a 12,288-token inverse view; use an 8,192-token pilot cap, counting rendering and
special tokens in that cap. Recheck the actual loader's limit before preparation.

1. **Text-only first.** Use valid UTF-8 text records and the writer's own rendering conventions.
   Exclude unsupported mixed-media windows from this pilot rather than invent placeholder
   semantics. Record every source span and inserted rendering span.
2. **Map offsets.** Convert tokenizer character offsets to UTF-8 byte offsets, distinguishing
   source text from role markers and special tokens. Drop non-content tokens. Test Unicode,
   empty offsets and tokens crossing source-record boundaries. Every retained token belongs to
   exactly one pooling span; every source byte belongs to exactly one span, although token spans
   need not partition bytes themselves.
3. **One pooling rule.** Cut the rendered content horizon into 64 equal byte spans, without
   snapping to chunk boundaries; assign each retained token by its starting byte and mean-pool
   states in that span. Condition and project nonempty rows; an empty span becomes a zero row
   after conditioning. Record the rule so short futures and empty rows are reproducible.
4. **One horizon.** Truncate within the verified token cap using a recorded deterministic rule,
   keeping the exact source-byte extent actually encoded. Do not add distant windows or reserve
   special rows for far-future content in the pilot. A shorter horizon is an approximation to
   hindsight, and its adequacy is measured by held-out answers.
5. **Re-chunking.** Source-byte pooling avoids tying row meaning to the writer's vocabulary or
   our chunk boundaries. It does not make the per-lookup targets invariant to a changed lookup
   plan: re-chunking changes which future each lookup is assigned, requiring fresh targets.

### 4.1d Later experiment: writing more entries than there are lookups

The owners have allowed more written entries than queried lookups. Keep that option, but separate
it from the initial stored-target experiment. It is a distinct use of the teacher: building a
searchable corpus rather than supplying one privileged target at a training boundary. It does
not remove boundary alignment requirements for the initial per-lookup targets.

Do not implement a global teacher bank as a prerequisite. A later, bounded test can mount a
read-only base made from an explicitly allowed set of reference documents. Requirements:

1. **Track everything the value read.** Provenance must include the complete suffix/view span,
   source revision and duplicate/family lineage, not just the first chunk or the entry's starting
   byte. An entry beginning before a question can still contain its future answer.
2. **Enforce visibility before candidate selection.** A different episode ID is not by itself a
   permission to retrieve: it may be an overlapping window, a duplicate, or another view of the
   same held-out example. Searchable entries must come from the task's allowed reference corpus;
   exclude disallowed future spans and evaluation targets. Current trained-bank lookup has no
   per-entry source-span visibility filter. Adding annotation fields alone would not enforce one.
   For a small test, mount only prefiltered safe bases instead of building a global filter.
3. **Keep keys and contents aligned.** A key made from the first chunk may describe that chunk,
   while the teacher value summarizes a much longer suffix. Measure whether real causal queries
   can retrieve useful entries; do not infer this from the writer's original key performance.
4. **Keep the pilot bounded.** Use a fixed sampled corpus and read-only entries. Defer clustering,
   special pruning protection and permanent-bank selection until this test demonstrates benefit.
5. **Keep addresses consistent.** A graph source and its outgoing edges must agree on the same
   item-only location. A teacher suffix location is a different object. Random projection does
   not guarantee that adjacent chunks or related suffixes have nearby locations.

Use fresh held-out tasks and deliberately contaminated references as a negative test of exclusion.
The criterion is that forbidden references are never eligible, not an arbitrary limit on how much
accuracy may improve: a legitimate reference corpus can yield a large gain.

### 4.2 Using it in the upcoming training run

- **One fixed boundary plan first.** Store the deterministic lookup plan for a small text-only
  pilot and make training consume that exact plan. Identify boundaries by stable record/meridian
  positions and align them to the actual training window, including any prefix and metadata
  dropout. Assert agreement; do not attach values by lookup ordinal alone. Re-chunking or changing
  the plan invalidates the targets. Keep random lookup placement for the ordinary baseline;
  use this fixed plan for the initial teacher-backed training segment. No scheduling comparison
  is required before launch.
- **One pinned encoding.** Fit conditioning/projection only on training data; freeze and identify
  them along with writer weights, rendering, horizon, byte-span rules and packing. Use one
  supported teacher-token horizon and deterministic byte spans for the first run. Do not add
  chunk-snapping, far-future rows or multiple jitter plans until a measured failure calls for them.
  Specify empty-span and special-token behavior in the preparation output.
- **Reuse the hindsight path.** Build `HindsightValues` from stored targets and missing-read-group
  flags for the selected training windows. Keep the existing overlay, bank-writing and write-out
  mechanisms rather than add a second writer. This is an insertion point, not a claim that
  replacing `self.values` alone suffices: current ingest-item matching is invoked by the learned
  hindsight pass and must still run in the stored-target path.
- **Make plain-text matching real.** Current matching selects `FEATURE_MODALITIES`, which includes
  compressed text and media but excludes ordinary `text`. Extend it to eligible physical text
  records using their item-only encoding through the existing ingest heads. Retain stable record
  identity and the rule that the selected future group contains exactly one eligible item; do
  not count a whole multi-record group as one item. Declared references outside that group do not
  become matches merely because they are mounted. Use the same matching in stored and learned
  hindsight paths. High-level raw-text reference ingestion alone does not provide this loss.
- **Represent absent targets explicitly.** Dense runtime storage may contain zero placeholder
  rows for sink-only lookups, but a separate validity mask must prevent treating those rows as
  teacher values. Preserve the sink-only flag in lookup records while passing no hindsight value
  and overlay zero for that lookup; ordinary bank retrieval and its usual gradients remain active.
  Exclude absent targets from realizer and ingest distillation, hindsight writes and new-entry
  creation. A missing future group is expected coverage, not an unsupported window or a reason to
  mix in learned hindsight. Handle windows where every lookup lacks a target as well.
- **Do not call low confidence sink-only.** Missing future groups are sink-only. Missing or mostly
  non-text teacher input is unsupported for this text-only teacher segment: leave those windows
  out of its coverage and report that limitation. They can use the existing learned training
  path outside this segment. A poor teacher summary does not mean no useful future exists.
- **Use one curriculum.** Stored suffix targets start one group ahead; use this gap-2 view for
  teacher-backed windows from the start. This changes the original warm-up curriculum. Accept
  that as an implementation choice; do not run a curriculum comparison before training. Apply
  this consistently to both future-group selection and eligibility for new-entry creation, which
  currently also depends on the schedule stage; changing only the teacher's read offset is insufficient.
- **Keep lasting writes active alongside reference graphs.** Current lookup recording disables
  hindsight writes whenever extra bases are mounted (`writes=self.mounted is None`). Replace that
  proxy with explicit per-window training-memory eligibility. Ordinary training-source windows
  may write their own teacher targets into the trained base even when allowed reference graphs
  are mounted. Reading-from-a-base question/answer episodes retain their existing no-write policy;
  held-out windows never write. Record the policy with the training segment and carry it through
  lookup recording, weighted writes and new-entry creation. Weighted writes address only selected
  edges in the writable trained base; new entries are created there through the existing writer.
  Mounted reference rows remain read-only. Mounting a reference never makes its contents or its
  question/answer episode eligible for lasting memorization. Include eligible natural training
  windows in the combined run so this policy actually fills the lasting bank.
- **Avoid an unplanned value-space switch.** Use teacher targets for all supported windows in
  the initial teacher-backed segment; retain existing overlay annealing and bank writes. Train
  the learned realizer toward matching valid stored targets with a small fixed relative-error loss
  (start at 0.1) so it has a path into the same value language. Do not mix unrelated learned and
  teacher targets within a window or automatically switch an old bank to unaligned values.
  The older learned-only training route has been deleted. Prepare teacher coverage before
  adopting a text tranche; missing text targets are errors. Explicitly unsupported non-text
  windows use causal retrieval without substitute targets or lasting hindsight writes.
  Fully covered means every lookup with an existing future group has a valid target; sink-only
  lookups require none. Teacher-backed segments should select only fully covered text windows,
  rather than silently fall back per lookup. No handover experiment is required before the first segment.
- **Evaluate without the oracle.** Turn off teacher overlay, exclude held-out-derived entries and
  use fresh held-out worlds/reference bases. Existing hindsight writes intentionally put training
  knowledge into the trained bank; do not treat that bank as uncontaminated when evaluating a
  previously seen episode. Direct use of the answer-bearing teacher vector is a teaching condition,
  not evidence that ordinary memory search improved.

### 4.3 Interface with structural supervision

The teacher supplies **what would help**, while known links teach **where to look**. Share the
existing value format and reader, but keep their preparation annotations and objectives separate.
An item-only ingest value and a suffix-derived teacher value are different representations even
when both have 32,768 coordinates.

For the first graph experiment, read-only temporary graph entries always use item-only ingest
values and locations. Teacher targets enter through the privileged hindsight path and may persist
through its existing writer in eligible training windows; they are never inserted directly into
the read-only reference graph. Extend existing ingest matching to ordinary text as specified above.
Ingest distillation can train an item to predict a teacher target where the future-group matching rule
finds exactly one eligible item. That is lossy when the teacher summarized multiple future items;
do not assume a raw chunk can reconstruct all of its suffix. Measure this mismatch before adding
more alignment losses or forcing a shared frozen address projection.

Implement the teacher path and bounded structural supervision together, run the cheap
correctness checks, then start training. Do not require separate teacher-only, structure-only or
multi-seed comparisons first. Both use existing components; neither needs permanent pre-seeding,
new relation-conditioned keys or a dense teacher bank. Keep a switch for disabling structural
supervision if it is unstable, while preserving the teacher-backed training path.

## 5. Cost

Lookups per 1 billion training tokens: about 0.24M for plain documents (a lookup roughly every
4,000 tokens), perhaps 1 to 3M for short-turn chat or tool traces (a guess from the per-turn
boundary rule).

| Form | Per value | 0.24M lookups | 1M lookups |
| --- | --- | --- | --- |
| 4-bit packed | 18.4 KB | 4.5 GB | 18.4 GB |
| FP32 (the current asset store) | 131 KB | 31 GB | 131 GB |

Teacher compute at an 8,192-token horizon is about 2B teacher tokens per 1B training tokens at
0.24M lookups and about 8B at 1M lookups. GPU-days must be measured on the actual writer, horizon and hardware before scheduling
corpus-scale preparation; the token-volume estimates alone do not establish throughput. A fixed horizon is a truncation of "everything after"; this is the main departure
from the central-processor pass.

## 6. Only cheap checks before training

Use a bounded small batch and one combined smoke run; no multi-arm training study, horizon sweep,
linear-readout benchmark or several-seed comparison blocks launch.

1. **Stored-value integrity.** Load the pinned writer, encode a few representative text windows,
   verify finite values, empty-span handling, packing round-trip error and distinct content
   locations. Fit and pin conditioning/projection on training examples only; use a bounded sample
   and fixed seeds. Do not make projection fitting into a corpus-wide job.
2. **Boundary and visibility correctness.** Check actual window/prefix matching, missing read
   groups, UTF-8 offsets and rendering tokens. Assert agreement with stored plans. Verify a
   teacher target never becomes a searchable held-out answer reference.
3. **One combined smoke.** Run forward, backward and checkpoint/resume with stored targets,
   realizer distillation, ingest matching and the structural auxiliary loss. Confirm finite
   losses, intended gradient paths and the ability to run validation with the overlay disabled.
   Include an ordinary-text item match and verify its ingest heads receive gradients. Exercise a
   missing future group and an all-sink-only window: ordinary bank retrieval remains active, no
   placeholder becomes a teacher vector, and no absent-target loss or hindsight write occurs.
   Include an eligible training window with a read-only reference graph mounted: verify an actual
   new teacher-backed entry survives checkpoint/resume in the trained base, reference rows stay
   unchanged, and the reading-from-a-base no-write policy still holds. Exercise new-entry creation
   from the start of teacher-backed training rather than waiting for the old curriculum stage.
   Cap this smoke at ten minutes; it has no accuracy threshold. These are assertions within the
   same combined smoke, not separate training stages or comparison runs.
4. **Proceed.** Start training on the bounded prepared tranche after these pass. Monitor existing
   validation plus target coverage and auxiliary-loss health. Optional studies can follow the
   training start. If writer loading or target correctness is broken, use the existing learned
   training path rather than delay all training; never silently use misaligned teacher targets.

The teacher's original results motivate this choice; useful transfer to this reader is still an
assumption. We accept that uncertainty to prioritize starting the real run.

## 7. Work items, in order

1. Use the retrieved writer's hidden states, one byte-span pooling rule and an 8,192-token cap.
   Define deterministic per-window boundary matching and pin the writer/encoding identities.
2. Add packed-value assets and resume-safe `megatoken prepare-hindsight`; prepare a
   bounded initial tranche with train-fitted conditioning/projection and source-view provenance.
3. Wire stored targets and a small learned-realizer distillation loss into existing overlay,
   writing and ingest-matching paths. Extend matching to ordinary text, implement absent-target
   masking with ordinary retrieval at sink-only lookups, and enable new-entry creation from the
   start. Replace mount-based write suppression with explicit per-window eligibility so eligible
   teacher-backed windows accumulate lasting entries alongside read-only graphs. Use teacher-backed
   windows consistently in this segment.
4. Add the companion plan's raw-text reference items and bounded structural auxiliary loss.
5. Run the cheap checks and combined smoke, then start training. Prepare further immutable
   tranches during the run, recording the selected data/teacher identity in each segment. Do not
   wait for whole-corpus preparation or silently change an active segment's target coverage.

## 8. Deferred rather than launch-blocking

Teacher alternatives, horizon tuning, independent contribution comparisons, lookup jitter,
learned-versus-fixed projections, dense searchable teacher banks and permanent preloading are
follow-up options. Value-space compatibility matters before a later handover, not as a reason to
run a handover research program before starting the first teacher-backed segment.

## 9. Implemented preparation and launch

The production launch templates require the joint memory recipe. Start with bounded
prepared roots covering the required mixture and validation supply, and prepare subsequent
tranches during training. Whole-corpus teacher preparation is not a launch prerequisite.
The production helper indexes every window of the selected tranche. Follow
[default memory recipe and launch readiness](training.md#default-memory-recipe-and-launch-readiness).
Use `megatoken prepare-hindsight` to prepare an indexed target sidecar for the exact existing roots and
window settings. It preserves the original source splits, media supply, stage mixture, decoder
packs and output path. The earlier sampled text-tranche helper and its artifact reader have
been deleted; there is one preparation path and one indexed artifact format.

Plain-text windows use stored teacher targets. Non-text windows are explicitly indexed as
unsupported and keep ordinary retrieval without an oracle or a learned substitute. The main
media objectives remain active. Frozen stages keep retrieval but disable privileged hindsight and
all updates to the bank, including coordinator pruning. This supersedes the earlier suggestion
to isolate all media into a separate training segment: the normal mixture is now retained.

Preparation runs offline in one job, reusing the prior phase's frozen encoding with
`--encoding-from`. A cloud handoff delivers the same target manifest and index to the learners;
the coordinator needs only the manifest. The writer backbone stays on the preparation side.
Startup checks actual target/link coverage and corpus identity, and fails if the default memory
objectives are omitted. A ready/not-ready report is available without allocating model weights.

Verification on 23 October 2026: 121 targeted tests passed. A tiny CPU smoke using the retrieved
writer completed preparation, one update, checkpoint/resume, a second update and held-out
validation in about 13 seconds. With read-only references mounted, the trained bank grew from
empty to one lasting entry after the first update and two after resume. This checks the joint
execution path, not answer quality or production throughput. Main training has not been launched.
