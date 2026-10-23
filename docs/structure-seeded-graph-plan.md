# Plan: use known links and teacher hindsight in the upcoming training run

**Status:** joint training path implemented; bounded correctness checks only. **Date:** 23 October 2026.
**Priority:** start training soon. Only cheap correctness checks precede training; comparative
experiments, extraction expansion and bulk memory preloading are not launch requirements.

## 1. One practical path

Use the [teacher-hindsight plan](teacher-hindsight-plan.md) to improve what memory contains,
and use links already present in the data to teach search where to go. Implement them together
through the existing reader, ingest heads, hindsight overlay and training writer. We are not
building an experimental framework or requiring independent demonstrations of each contribution.

The teacher reads future text and produces a useful training target. The existing writer puts
those targets into the long-lived trained bank as training proceeds. Known document-part and
tool-call/result links provide additional guidance for the model's causal search queries.

Two kinds of value share a reader, but must not be confused:

- **Item memory:** produced by reading one reference item alone. Its location describes that item.
- **Hindsight target:** produced by reading future text. It is a privileged training signal, not
  automatically a reference item that ordinary search may read.

Use item-only values and locations for reference graph edges; use teacher suffix values through
the training-only hindsight path and its lasting writer for eligible training-source windows.
Extend existing ingest matching to ordinary text: it currently accepts compressed text and media
only. Ingest distillation connects the two where a lookup's future group contains exactly one
eligible item. Do not force a chunk to reproduce an entire multi-item future, or assume the two
values have the same location because their shapes match.

## 2. Use what is already there

| Existing mechanism | What to do now |
| --- | --- |
| Parts share `logical_id` and ordered `part_index`; ingestion already chains document parts | Reuse the chain rather than invent another document graph |
| Tool records carry `result_of` | Extract exact call/result pairs; preserve existing packing behavior |
| Episode annotations already hold `relations` | Reuse `kind`, `left`, `right` for local navigation links, with explicit kinds |
| `core.encode_chunk` and low-level ingest accept text | Extend high-level declared knowledge items, which currently expect teacher features |
| Training mounts small read-only reference bases and refreshes them periodically | Reuse this route for allowed reference items and graph links |
| Retrieval supervision inserts known correct items into its candidates | Follow this rule in the auxiliary navigation loss |

No new SQLite relation table, cross-episode join infrastructure, import parser, learned relation
vector, frozen address projection or production stream is needed before training. Do not change
production mixture weights merely to introduce another named stream. Use the existing reading
stream and natural training examples where applicable.

## 3. Concrete implementation

### Local links and raw-text items

Support raw text alongside existing feature descriptors in declared knowledge items. Preserve
record IDs and document part order. Also extend future-group ingest matching to ordinary text
using item-only encoding and stable record identity; declared references outside the selected
group are not matches. This is separate from adding high-level raw-text declared items.
Extract local part-chain and call/result relations and include
the extractor version in preparation identity. Reject dangling endpoints. If a call or result
spans multiple physical records, use the end of the call as the teaching boundary and the first
result part as the first target, with the remaining result parts chained normally.

Do not silently reinterpret media/translation semantic relations as navigation links; leave their
existing losses alone. Start with one successor per teaching example and cap examples per window.

### Reference graph

Encode each endpoint once with the current ingest heads. An edge A→B departs from A's item-only
content location and carries B's item-only key, value and destination, with zero bias. Reuse
existing ingestion for document chains. Additional predecessors may duplicate edges within the
small base's limit; no shared-value storage is required.

Give reference graphs entry edges from the normal metadata start, including graphs with no root.
Exclude the generated answer from reference items. A future tool result must not enter ordinary
search when predicting that live result. It can provide privileged auxiliary teaching, or be a
legal reference in a task about an already completed trace. This availability distinction is a
correctness requirement, not an extra experiment.

Reuse the existing refresh interval (currently 1,024 optimizer steps) initially. It limits but
does not eliminate drift as ingest heads change. Do not block training on finding an optimal
refresh period. Keep these bases read-only so hindsight updates cannot overwrite their references.

Preserve lasting teacher writes alongside the graph. Current training suppresses those writes
whenever additional bases are mounted; read-only reference rows alone do not solve this. Use the
teacher plan's explicit per-window training-memory eligibility: ordinary training-source windows
may write their own suffix targets to the trained base even with a graph mounted. Reading-from-a-base
question/answer episodes keep their existing no-write policy, and held-out windows never write.
Weighted writes touch only selected trained-base edges; new teacher-backed entries also belong
only to that writable base. Include eligible natural training windows so the combined run actually
accumulates lasting knowledge. Mounted reference contents are not automatically eligible for it.

### Auxiliary navigation loss

For a bounded sample of exact A→B links, use the model's causal query after A and, for this
auxiliary calculation only, A's item-only location as the starting point. If ordinary scheduling
does not provide a query there, obtain one from the same causal prefix without committing an
extra serving lookup. Never substitute B's key as the query.

Score bounded negatives plus known positive edges with the existing location/key rule and sink.
Explicitly insert the positives if normal shortlisting omitted them, as existing retrieval ranking
training does. Minimize minus log total positive probability. Train the query and existing
trainable scoring components; reference rows remain fixed. Match relation endpoints to edge IDs
by both source and destination so duplicate incoming edges do not create incorrect positives.

This auxiliary calculation teaches a transition. Ordinary answer generation retains normal
starting locations and candidate selection: no forced positive and no supplied intermediate
location. Cap its weight and number of examples, and expose an off switch so an unstable loss
can be disabled without restarting the implementation effort.

## 4. Only cheap checks before training

Use existing small fixtures/probes; no new benchmark system and no full comparison runs.

1. Verify raw-text ingestion, valid link endpoints and the expected A→B edge fields.
2. Verify a missing positive is inserted, loss is finite and a backward pass reaches the causal
   query. Check that the query and A encoding read no future target text.
3. Verify normal search cannot retrieve an unavailable future result or a held-out-derived teacher
   target. Eligible training-derived teacher writes may persist in the trained base; evaluate on
   fresh held-out examples whose answers they do not contain. A fresh held-out reference base
   must not include held-out training writes.
4. Run one tiny combined forward/backward/checkpoint-resume smoke with stored teacher targets,
   raw-text reference items and the navigation loss. Include at least one valid ingest match and
   one missing read group. Verify an ordinary-text match supplies gradients to the ingest heads;
   absent targets use ordinary retrieval with no teacher blending, distillation or hindsight writes,
   including an all-sink-only window. In an eligible training window with references mounted,
   create an actual lasting teacher-backed entry and verify it survives resume while reference
   rows remain unchanged; check that reading-from-a-base episodes still do not write. New-entry
   creation must be enabled from the first teacher-backed step. Cap this smoke at ten minutes;
   these checks belong to the same smoke, and no accuracy target or multi-seed comparison blocks
   the real training start.

Then start training. Follow ordinary training/validation loss, teacher-target coverage, auxiliary
loss and ordinary positive shortlist recall using lightweight existing logging. These are checks
for breakage, not a research acceptance program. If a component fails its correctness checks,
disable that component and start the working training path; do not leave the whole run waiting
for an optional mechanism. Never bypass source-visibility or target-alignment failures.

The existing graph probe uses oracle target keys in its seeded-path test. It is useful as a
mechanics smoke, not evidence of learned navigation. Multi-hop comparisons, shuffled controls,
several random seeds and refresh-period studies may be useful later; none precedes training.

## 5. Why bulk preloading is deferred

**Permanent pre-seeding** means filling the long-lived trained bank offline before training begins.
The bank then checkpoints those entries and learns over them. That is different from both the
small refreshed reference bases and the existing writer accumulating lasting memories during
training. We keep the latter two mechanisms.

A bulk preload requires choosing entries, preserving learned rows when seeds are refreshed,
handling their changing addresses and defining how they age or are overwritten. It is not needed
to use teacher targets or known links in the upcoming run. Leave the recorded empty-bank startup
policy unchanged for now; teacher-assisted training will populate it through the existing writer.

Also defer dense searchable teacher suffix memories, code import extraction, cross-episode joins,
clustering, special pruning protection and any new relation-conditioned architecture. They are
possible follow-ups, not planned launch dependencies. Edge budgets should come from current
geometry and actual memory use; the earlier large relation counts were unmeasured estimates.

## 6. Work order

1. Implement the teacher plan's pinned stored-target route, extend ingest matching to ordinary
   text, and define absent-target retrieval. Keep lasting writes active for eligible training
   windows with graphs mounted while retaining the reading-from-a-base no-write policy. Enable
   new-entry creation from the first teacher-backed step. Prepare a bounded initial tranche
   rather than wait for whole-corpus teacher output.
2. Add raw-text declared items, exact local links and bounded temporary reference edges.
3. Add the optional auxiliary loss through existing scoring, with positive insertion and causal
   queries. Keep normal serving search unchanged.
4. Run the cheap correctness checks and combined smoke, then start training with the completed
   components. Continue preparing targets during training only by publishing immutable tranches
   that each training segment records; do not mutate data beneath an active segment.
5. Use observations from the real run to choose further work. No comparative research campaign,
   new extractor or permanent-bank preload is required to start.

## 7. Implemented entry points

Preparation and launch commands are in [the teacher plan](teacher-hindsight-plan.md#9-implemented-preparation-and-launch).
The emitted training configuration uses the existing lasting writer and reference-base reader,
with bounded link teaching enabled. Setting `navigation_weight` to zero disables auxiliary link
teaching; raw-text references still work. Local link extraction is identified as
`local-navigation-v1` in new preparation identities. Existing prepared episodes can also supply
exact links at training time without reparsing or bulk-preloading their contents.

The cheap checks cover a missing positive, causal-query gradients, raw-text matching, absent
teacher targets, checkpoint/resume, allowed lasting writes alongside unchanged references, and
base-reading questions that do not write. Bulk seeding, import extraction, multi-hop benchmark
studies and mixture changes remain deferred.

Verification on 23 October 2026: 121 targeted tests passed. A tiny CPU smoke using the retrieved
writer completed preparation, one update, checkpoint/resume, a second update and held-out
validation in about 13 seconds. With read-only references mounted, the trained bank grew from
empty to one lasting entry after the first update and two after resume. This checks the joint
execution path, not answer quality or production throughput. Main training has not been launched.

The default production recipe now requires teacher targets and structural supervision, with
corpus-preserving preparation and verified cloud delivery. See
[default memory recipe and launch readiness](training.md#default-memory-recipe-and-launch-readiness).
Media windows keep their original training share and normal retrieval without a teacher substitute.
Frozen training scopes also freeze sparse rows and coordinator pruning. The sampled text-tranche
preparation route has been deleted. Production requirements live in the shared settings loader,
not in standalone scripts or opt-in flags.
