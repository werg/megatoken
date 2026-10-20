# Retrieval and semantic memory: architecture plan

**Status:** Proposed extension to the v8 prototype; implementation and learned quality remain to be established.
**Date:** 2026-10-20.
**Reviewed megatoken baseline:** [`82be7ef563e6cf55a2e2b0cb3d11bf55793071e1`](https://github.com/werg/megatoken/commit/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1).
**Audited donor:** [`hindsight-kb@450164784589b2576256b8b0ba3d2ccc89af992d`](https://github.com/werg/hindsight-kb/commit/450164784589b2576256b8b0ba3d2ccc89af992d).

Sections 1–9 specify the native architecture and curriculum. Sections 10–19 are the implementation specification: donor components, causal/data contracts, storage, optimization, source lifecycle, writer methodology, recipient modules and acceptance gates. Donor code and test references are pinned to the audited revision; proposed megatoken paths are explicitly identified.

## 1. Decision

Use **source RAG for literal evidence**, and a **progressively grown semantic memory KB for complete attention K/V and LoRA control**. Both use the same learned semantic retrieval interface and latency-boundary scheduler. The two content tracks remain distinct:

| Path | Canonical stored value | How it enters megatoken |
| --- | --- | --- |
| Source RAG | Original source chunks, ordered spans and provenance | Ordinary observations through E/C on the source RAG track; incremental MaxSim admits only newly relevant, previously uninjected content. |
| Semantic memory | Source-supported, freely optimized semantic values, strength and lifetime | Similarity-weighted point/pair semantic mixtures produce full native-facet K/V positions on a dedicated KB track and drive sparse LoRA control across tracks. |

The semantic KB **replaces the KV-delta/overlay proposal**. Its entries do not modify or attach to existing native cache entries. Their complete K/V participates directly in the shared attention context. Literal source sequences still use RAG and ordinary ingestion; learned KB slots need not pretend to be source tokens.

First establish generator coherence and distill useful per-megatoken/per-chunk semantic embeddings. Then continually add free memory records, beginning in the query-densest parts of keyspace, including supported pair correlations as their endpoints become available. Train queries, selected stored keys, semantic values, directional correlation weights, the full-KV projector, controllers and model together. Train source/experience **memory-value writers only after this pretraining**.

The [multi-track architecture](multi-track-plan.md) already defines shared history, categorical tracks and atomic meridians. Grouped recurrent training exists in [training/meridians.py][m-meridians]. Direct KV admission requires a small new context-state path: C and P read the combined native history and KB segment, while the native raw/final frontier retains its existing meaning. This is one attention context with two evidence paths, not an additional transformer.

LoRA scope is all tracks; expert choices may specialize by track, stage and causal context. Balance expert demand over the largest practical synchronous training window, rather than requiring every sequence or track to use every expert. Section 12.8 specifies an optimizer-step-global, router-only objective without extra optimizer moments.

## 2. Embedding realizer, one query interface, two payload readers

Use the [v8 shared representation](architecture.md): E emits raw facets, P predicts the same geometry, and R realizes it directly. Add an **embedding realizer** alongside the text and media realizers. Its latent input is the selected raw block `Z: [batch, facets, width]` (normally 32 facets), not C activations or a teacher-produced vector.

At a latency boundary, request one retrieval-purpose event on a reusable query track:

~~~text
Z_query ~ P(committed_context, main_request, retrieval_spec, admitted_control)
q = R_embedding(Z_query, space_id, role="query")
hits = ANN(q, eligible_collections)
~~~

Condition P on the main request, retrieval purpose and semantic-space specification **before** predicting. Do not assume `state.final` is the main track: it contains the latest native meridian, which may contain only tool or evidence events. Use explicit causal request/control conditioning and the existing historical attention path. Merely choosing an embedding realizer after ordinary answer prediction does not establish search intent.

Start with one learned output slot reading all raw facets with their identities, followed by projection and normalization into the chosen key space. Use the normal selected Gaussian raw-facet event. R is deterministic given the raw tensor and output spec; the query policy still includes P's sampling. C processes that same selected block for future predictions. Keep embedding R canonical initially, even when content realizers receive explicit LoRA control snapshots under section 12.5.

The typed output spec identifies semantic space/version, query/document role, dimension, normalization and output count. Preserve the teacher's query/document asymmetry. The teacher vector is a training target; serving needs neither an online teacher call nor a vector ingester.

A reusable facet readout can later support classification, scoring or typed extraction inside ordinary realizers. Implement embeddings first. Facet identities are not source-token positions; exact span extraction needs source alignment. Additional complementary query slots are optional; start with one vector per retrieval event, not one per facet. Multiple outputs from one raw block do not create additional events.

Use one ANN service with typed **source** and **memory** collections in a compatible learned address space, with independent result budgets. The ANN address \(k_r\), retrieved semantic value \(m_r\), and materialized attention key \(K_r\) have different roles. KV and LoRA share the learned semantic **value** space; equality of dimensions would not make an address vector a native attention key.

Source RAG uses **incremental MaxSim** over newly committed query vectors. Qualify source chunks whose incoming similarities are high, fetch only source intervals not already injected or reserved, and keep deferred relevance separate from durable injected coverage. Section 12.6 defines overlap, retries and score epochs. This content-novelty rule does not suppress memory refresh, learned decay or training-time memory allocation.

Source hits return exact bytes, source identity/version, spans and document order. Ingest them with document/tool roles on a reusable RAG track. Ordered chunks arrive through successive meridians because one track contributes at most one event per meridian. A small fixed set of RAG tracks can admit independent documents concurrently; source IDs remain metadata.

RAG uses normal committed-meridian RoPE and native history eviction. The separate semantic KB track uses the admission-anchor convention in section 12.4. Both are readable through shared attention; a track is a categorical context identity, not an isolation boundary.

## 3. Initialize semantic addressing, then learn both sides

Use a **teacher-initialized learned key space**. During generator-coherence bootstrap, precompute semantic targets for megatoken's actual chunks and distill per-chunk/document and causal-query readouts into the shared raw-facet architecture. Keep stored keys fixed only for bounded addressing warm-up. Then require local joint learning of the native query side and selected stored keys in **both** source and memory collections. Stable source bytes and existing record assignments remain fixed while their addresses adapt and new records are allocated. The teacher initializes and optionally anchors the coordinates; it does not permanently fix them.

Warm the embedding realizer on E-encoded explicit queries with teacher query targets. Then train causal P retrieval events through the same realizer, using teacher query targets, similarity distributions and labeled useful evidence. The E route is auxiliary supervision, not a requirement to generate query text at serving. P and R must learn together on the raw blocks actually used for retrieval; the teacher vector is a target, not P's input.

After pretraining, adapt source-key generation to the **optimized learned space** as part of corpus reproduction. Reuse the canonical document/key realizer over permitted native source representations, such as `R_embedding(E(source), role="document")`, alongside the new memory writer's key/value outputs. Its initial semantic distillation belongs in bootstrap; reproducing freely adapted final keys is a later target. An isolated E(chunk) cannot reproduce a target requiring unavailable neighbors, so preserve complete source/context footprints. Teacher-only keys remain declared initialization or an evaluated fallback.

**Selected teacher space (2026-10-20):** [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) at revision `914f7f89`, 768 dimensions, chunks encoded **in isolation** (no contextual chunk encoding), pinned in [configs/retrieval-memory-pilot.json](../configs/retrieval-memory-pilot.json). Documents use `title: {title} | text: {text}` and queries `task: question answering | query: {q}`, mean-pooled and L2-normalized. The contextual Perplexity `pplx-embed-v2-context-9b-preview` was the earlier preference; at 33.6 GB of FP32 weights it was rejected for the pilot. A contextual or late-interaction teacher can still be evaluated later as a separate space; never mix spaces.

Normalize vectors consistently and pin the model, dimension, query/document formatting and chunk context policy. Pin weight revisions for local teachers; for hosted teachers, retain returned vectors with model identifiers, request settings and input hashes. Do not mix unrelated embedding spaces. The newer [Perplexity late-interaction family](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-9b) produces multiple token vectors; it can be evaluated separately as a ranking teacher without making the first index multi-vector.

The query teacher sees only the available request/prefix. The evidence teacher may see the permitted source context. Seed memory records intended for local hindsight with keys computed from bounded source windows that a later suffix writer can see; reserve broader-context records for corpus reproduction. In continuation and hindsight tasks, every chunk that influenced a contextual embedding is part of its visibility footprint; filtering only the returned chunk is insufficient.

Train for **useful evidence** with multiple positives, hard distractors and no-answer examples; unannotated results are not automatically negatives. Select ANN/top-k IDs discretely, then fetch live selected-key leaves and recompute similarities with the live query. Similarity-weighted full-KV attention and LoRA control let continuation loss train query keys, stored keys, the projector/controllers and selected shared semantic values. For source RAG, the local value objective in section 12.7 trains both address sides from detached features of the retrieved source content and useful-support targets. Literal byte fetch remains discrete. Historical MaxSim maxima, candidate identities and queue decisions are detached scheduling state, not substitutes for this local gradient path.

## 4. Progressive free semantic memory during pretraining

Introduce memory **after generator coherence and per-megatoken/per-chunk embedding distillation**. Start with a small source-supported bank and grow it throughout retrieval warm-up and main pretraining. There is no learned memory-value writer during this phase.

Use the source RAG catalog and its coarse semantic regions as the allocation catalog. Accumulate bounded causal query-demand statistics, including memory misses, before relation bonuses or source coverage suppression. First allocate in the densest regions; subsequently rank demand relative to already allocated capacity. Reserve a small miss/exploration quota so unrepresented regions can enter. A catalog miss records unmet demand rather than creating an unsupported source-less record.

At declared growth barriers, add a bounded batch of stable records. Each record has:

- A key initialized from its current compatible source-catalog address, with retained teacher/source provenance, then independently trainable.
- Complete source/context support, stable source-to-record assignments and version/scope metadata.
- A small independent random semantic value \(m_r\), plus trainable strength and lifetime fields.
- Source-derived pair associations; add supported pair values when both endpoints exist.

Keep existing learned rows and optimizer moments intact. Give eligible newly allocated records limited candidate exposure while keeping their application strength small. Repeated growth does not restart all key warm-up or reset trained memory. Section 14.1 specifies the allocator, correlation updates and atomic publication.

“Free” means the semantic value is an independently optimized variable, not a writer output. Source IDs are bookkeeping, not query-model features. Do not cluster near-zero random values by numerical proximity. Reuse records across related tasks; train missing memory, irrelevant records, conflicts and source-excluded retrieval alongside useful memory. Preserve ordinary no-memory examples and independent KV/LoRA dropout.

### Complete facet KV from a shared semantic mixture

Each admitted KB position mirrors a complete native facet block: \(F=\mathrm{config.n\_facets}\), currently 32, with the native facet identities and KV head geometry. Combine compatible point and independently supported pair values in semantic space, then use the same aggregate for KV and LoRA:

\[
\bar m_s=\frac{\sum_{r\in s}w_r m_r}{\sum_{r\in s}w_r},
\qquad
(K_s^0,V_s)=G_{\rm KV}(\bar m_s;\text{native facet identities, KB type}),
\qquad
u_s=G_{\rm LoRA}(\bar m_s).
\]

Both \(K_s^0,V_s\) have shape \(H_{\rm KV}\times F\times d_h\). The weights depend on live query/key similarity, eligible contextual support, learned directional correlation dependence and record lifetime. Section 12.2 supplies the normalized mixture and separate null-aware strength gate; section 12.3 defines the correlation adjustment.

Supported pair values participate directly in this mixture with an explicit bounded pair gain. They are more than a neighbor-ranking bonus. Preserve conflicting or incompatible evidence in separate bounded groups; the mixture happens **before** projection, not by assuming that averaged projected K/V is equivalent. Source/record identities remain traceable through each group's constituent list.

Admit these full facet blocks to the separate KB context segment that C and P can read. Matching native geometry does not make a projected block a raw-facet policy event. Keep the native raw/final frontier and event clock intact. Source RAG retains ordinary ingestion and literal source order.

Canonical per-record semantic values remain the trainable storage authority. Each aggregate and its full KV are derived from the current contributor weights. Keep per-record lifetime: compute the decayed aggregate and readouts once per native transaction, then hold that snapshot fixed through its P/C/R work. This bounded local calculation uses no new retrieval and retains the original admission anchor. Unequal constituent decay can change the semantic mixture, so scaling an old nonlinear projection alone is insufficient.

Anchor all F keys at the group's admission position and rotate each materialization once; never rotate values. Gate the block's attention participation with its current mixture strength, using an exact mask for zero evidence. Zero values alone are not null, because their keys would still alter attention normalization.

### Shared LoRA control with specialized choices

Train a shared bank of independently initialized low-rank experts:

\[
y_g=W_gx+\sum_{e\in S_g}c_{g,e}\,B_{g,e}A_{g,e}x.
\]

Retrieved semantic values supply the controller's memory input. The current track, stage and causal local state can choose a different sparse mixture from the same bank. Use residual computation without mutating shared base weights. All tracks may use memory; they need not use the same experts.

Cover future P and C attention/feed-forward work across generated and observed tracks, and explicitly selected content-realizer matrices with a recorded adapter snapshot. Keep canonical source feature/index E, embedding readouts, the new KV projector and the shared native final-KV projector outside session-dependent control initially. These are coordinate-stability defaults, not a main-track scope restriction. Section 12.5 defines owners, grouping and extensions.

Use small nonzero independent factors, bounded soft warm-up followed by sparse choice, and an exact null intervention. Distinguish expert-selection probabilities from signed residual coefficients, total strength and lifetime. Bound active experts across retrieved records, and avoid a Cartesian memory-expert × base-expert × track expansion.

For both uses, retain activation and learned decay per admitted record/version:
\(\omega_r(t)=b_r\exp[-\operatorname{softplus}(\lambda_r)\operatorname{age}_r(t)]\), where \(b_r\) is that point/pair's unnormalized admission coefficient. Form the current aggregate and its null-aware group strength from these decayed weights; do not apply an additional independent group-decay multiplier.
Use committed **native meridians across the session** as the initial all-track clock. Predictor effort and KB-only admission/refresh do not advance it. Refresh one record without resetting unrelated records. Decay and eviction end future direct use; effects already incorporated into native history remain part of the trajectory.

The main uncertainty remains source reproducibility: useful free memory can encode information a later writer cannot infer. Grounding, limited capacity and reuse reduce that risk; the regenerated-bank and new-source tests remain decisive.

## 5. Sparse relation-reinforced retrieval

Initially **second order means pairwise corpus relations/correlations**. Compute bounded source-window co-occurrences and explicit associations over stable source/catalog identities, independently of active memory allocation. Deduplicate windows, retain complete support footprints, cap degree and normalize common hubs. A statistical association is not itself a necessary condition or a factual assertion.

When both endpoints have allocated memory records, admit their supported pair value and **two learnable directional dependence weights**, one for each endpoint. Preserve corpus statistics as a separate fixed prior. These learned weights express how much the usefulness of one point depends on context supporting the other endpoint; the directions can differ.

Retrieve semantic seeds and expand eligible neighbors once. Rescore against the original query, use capped ranking bonuses, and never recursively reinforce boosted results. Check complete source/scope/time eligibility before truncation and before using a relation to increase or decrease another record's weight.

Use two effects in the semantic-memory output:

- A query-relevant endpoint plus independent causal support for its partner activates the learned pair value, with explicit prominence in the point/pair semantic mixture.
- A prominent learned dependence whose partner was checked but lacks contextual support attenuates the lower-order point value. The attenuation is bounded; unknown, unsearched or excluded support produces no penalty.

Section 12.3 defines the single-pass formula. Both effects use original unboosted similarities and an explicit causal support ledger. Repeated queries, duplicate spans and synthetic KB content cannot manufacture the other leg. A first retrieval can use earlier independent native context as support; one isolated query/support unit still cannot activate a pair.

Counts and topology change only at logged growth/maintenance barriers. Endpoint weights, keys and semantic values learn locally during free-KB pretraining, with topology fixed within each window. The final pair targets include both directional weights for later corpus reproduction. Keep incompatible values and singletons. Source replacement invalidates affected support and dependent weights/materializations.

Source RAG retains its own MaxSim/content-novelty policy. The new attenuation adjusts learned semantic-memory output; it does not silently discard literal RAG evidence whose direct source similarity has qualified it.

## 6. Retrieval tasks and data format

Grow a broad retrieval-task inventory through the existing [source manifests and Episode preparation](data.md), while mixing by capability so one large dataset does not dominate. Map every example to: **available prefix/request, eligible corpus snapshot, positive source spans or evidence sets, target outputs, and boundary/arrival schedule**. Answer labels and future traces remain supervision. Ordinary continuation examples exclude their unavailable suffix and duplicate ancestry from both candidate paths and relation expansion; the later local hindsight phase uses its separately declared mask.

| Capability | Initial primary sources and use |
| --- | --- |
| Passage lookup and absent answers | [Natural Questions](https://aclanthology.org/Q19-1026/) training data |
| Multiple supporting sources and evidence sufficiency | [MuSiQue](https://github.com/StonyBrookNLP/musique) training splits |
| Cross-file code retrieval | [RepoBench](https://github.com/Leolty/repobench) formats and its archived retrieval training split; preserve repository/commit boundaries |
| Tool/API selection | [ToolRet](https://github.com/mangopy/tool-retrieval-benchmark) training collection; verify execution separately |
| Time-qualified facts | [TimeQA](https://github.com/wenhuchen/Time-Sensitive-QA) training data |
| Support, refutation and insufficient evidence | [FEVER](https://fever.ai/dataset/fever.html) training data |
| End-to-end RAG and robustness | [CRAG](https://github.com/facebookresearch/CRAG) and [RGB](https://github.com/chen700564/RGB) as held-out evaluation candidates; build distinct training analogues |

These are acquisition candidates, not acquired or approved inventory; follow the existing source-use and split records, including RGB's stated noncommercial terms. Preserve source ancestry across datasets: apply MuSiQue's `data/dev_test_singlehop_questions_v1.0.json` exclusions when mixing its single-hop source datasets into training or teacher fitting. Refuting evidence is useful evidence, not a retrieval negative.

Add corpus-derived lookup/join tasks and actual attempt/feedback/correction trajectories. Train failures, stale evidence, competing versions and no useful match alongside successful retrieval. Keep evaluation questions and their synthetic derivatives out of pretraining, KB optimization and writer fitting; evaluation corpora contain only the evidence allowed by each benchmark.

## 7. Real and artificial latency boundaries

Use one scheduler with separate source and memory result budgets:

1. At a completed turn or declared artificial boundary, snapshot causal state, select a retrieval-purpose P event and realize its vector. Commit that raw event through the ordinary grouped transition before dispatching the lookup.
2. Score new source-query embeddings, qualify novel content and reserve uncovered source intervals before fetching. Retrieve semantic memory records under their separate refresh policy. During pretraining, also record raw query demand for later allocation.
3. At a valid later boundary, revalidate source results and admit them through E/C as observed native meridians. Commit only successfully ingested intervals to source coverage.
4. Independently validate available point/pair values and contextual support, form the correlation-adjusted semantic groups, project full-facet KV, then atomically install those blocks and the corresponding LoRA snapshot. This changes the session revision but preserves the native raw/final frontier and meridian counter.
5. Subsequent P/C/R work uses the admitted snapshot. Results cannot affect the query that fetched them or already selected peer outputs. On a miss, admit nothing new and let admitted constituents decay; deterministically rematerialize their mixtures before each native transaction without a new lookup.

A direct KB admission is a deterministic context transaction, not a Gaussian raw action. It allocates F facet slots per aggregate position in the independent KB segment without calling `advance_group` or C.write or creating a native event. A real query or source observation still commits a native event and advances its containing meridian once.

For long runs, insert artificial opportunities after configurable committed main-track progress, with a small range of training intervals. Scheduling opportunities create no empty events. Memory age uses the separately specified all-session native clock. Evidence admission alone must not recursively trigger retrieval.

Use bounded waiting and the same arrival/deadline policy in training and serving. Late results wait for a valid later boundary. Any admitted evidence or changed KB/control snapshot increments session revision and invalidates speculation based on the old state. Resolve simultaneous source and memory admissions in a deterministic logged order; do not change control midway through a grouped meridian or between its R/C operations.

Record selected query raw facets/densities, vector specs, scoring/query/key/index epochs, bounded causal support vectors with encoding/role/span provenance, source fetch/reservation/coverage receipts, complete KV admission identities and anchors, constituent records/weights and directional-edge revisions, checked-support masks, adapter snapshots, activation/expiry and actual arrival order. Replay uses historical values and controls. Bound queries, source bytes/events, full-KV slots, graph expansion and adapter staging, and measure the whole boundary latency.

Training-time KB growth is an owned optimizer/publication operation. Newly allocated records become visible to the next pinned retrieval window; growth does not mutate a serving session or an in-flight gradient graph.

## 8. Post-pretraining writer and corpus reproduction

| Phase | Work and completion evidence |
| --- | --- |
| Coherence and embeddings | Establish useful native generation and distill causal-query/per-chunk semantic embeddings. No memory-value writer or dependence on populated memory. |
| Progressive free-KB warm-up | Allocate source-backed entries in query-dense regions, including correlations; warm KV/LoRA use and bounded key addressing. |
| Main free-KB pretraining | Continue bounded growth and jointly train native queries, selected source/memory keys, semantic values, full-KV projector, controllers and expert bank. |
| Local hindsight writer | Freeze a coherent final model, allocation/assignment manifest, learned address space, projector/controllers/bank and optimized targets. Train the permitted suffix to reproduce supported keys and semantic values. |
| Corpus reproduction | Reproduce optimized source keys and admitted point/pair keys/values from retained sources, rebuild indexes, and regenerate full K/V and LoRA controls through the frozen materializers. |
| New-source use | Insert sources held out from pretraining and writer fitting with inference weights fixed; measure useful transfer without optimizing against their test answers. |

Initial embedding distillation is ordinary representation/readout training. It is distinct from the post-pretraining writer that must reconstruct the learned memory values and optimized addresses.

For local hindsight, the prefix queries while the writer sees only its permitted suffix. Preserve [hindsight-kb's rule][h-retrieval]: at **original source block** \(i\), same-source writer records begin at \(j\geq i+2\). These are meaningful source blocks, not runtime meridians, facets or inserted query events. Validate every source range influencing a key or value. This future-suffix access is a training-only memory-writer exception; source RAG and deployment keep causal visibility.

Local warm-up can retain the donor's indirect contamination through earlier admitted memory. Label and bound that phase; require subsequent cross-source transfer. A record or directional pair weight needing broader context belongs in corpus reproduction, not an underinformed local target.

Fit keys, shared semantic values and controls, then evaluate behavior through the frozen full-KV and LoRA materializers. Rebuild a **fresh** KB from writer outputs and compare it with the optimized bank. Reproduce the admitted target subset and source assignments; the writer is not asked to infer the historical query-density allocator from source text. Reconstruction error alone is insufficient. Diagnose source sufficiency, assignment coverage and free-value capacity before adding writer complexity.

## 9. Implementation and acceptance

Implement in this order:

1. Add typed embedding output/training targets, registered raw-facet realization, causal retrieval-purpose conditioning, grouped losses and query tapes. Train generator coherence and per-megatoken/per-chunk embeddings. Add source RAG incremental MaxSim, eligibility, reservations and coverage.
2. Add source/catalog demand accounting, progressive point/pair allocation and sparse storage. Implement the common semantic value, full-KV projector and separate KB context admission/read path.
3. Add independent LoRA banks with all-track snapshots, explicit parameter owners and sparse specialization from the same point/pair semantic aggregates. Replace local expert equalization with optimizer-step-global router-only balancing.
4. Integrate bilateral query/stored-key learning, shared-value gradients, sparse key/index updates, density/growth cursors, compatible checkpoints and exact historical admission/control replay.
5. After completed free-KB pretraining, add local hindsight, cross-source transfer, full corpus reproduction and new-source evaluation.

Register new realizers, projectors, routers and banks on the underlying native `FacetedCore`/`FacetedGenerator` owners. Plain multi-track wrappers alone do not provide optimizer ownership. Extend MultiTrackState and attention features for direct KV admission while preserving native event/frontier contracts. The dispatcher, typed vector targets, attention log-strength support, all-track adapter materialization and retrieval-aware recurrent/distributed integration remain new work.

Compare no-query, matched null-query, RAG only, full KV only, LoRA only, KV+LoRA and all three, with separate relation/growth/routing ablations. Report novel evidence recall, native/task quality, no-memory regression, specialization, global load and dropped work, memory lifetime, regenerated-bank quality, replay fidelity and complete serving cost.

For later policy learning, extend the native action-map contract explicitly. Broad memory affecting C/E/R or a query realizer cannot simply inherit a P-only temporary-record reward objective. Section 17.2 specifies a conservative initial frozen-map path.

The detailed donor-to-recipient migration and acceptance specification follows.

## 10. What transfers from hindsight-kb

### 10.1 Scope and evidence

The donor supplies executable objectives, storage backends, source authoring, sparse row optimization, replay and experience-writing machinery. Its [implementation boundaries][h-status] explicitly leave useful completed-pretraining KB behavior and large-scale performance unestablished. This plan is based on inspection of code, documentation and tests; ported components must pass megatoken's own acceptance checks.

**Copy** means transplant a small mechanism and its relevant tests, retaining donor provenance and adapting imports. **Adapt** means preserve its invariant but revise data shapes, ownership or lifecycle. **Implement** identifies machinery that neither the donor nor the current recipient already supplies. Keep one megatoken data pipeline, model, trainer, runtime and checkpoint format.

| Donor component and exact entry points | Transfer | Megatoken specification |
| --- | --- | --- |
| [base.py][h-base]: Record, Relation, Contribution, Candidates, empty_candidates, chunked_topk | Adapt types; copy numerical reference | Stable IDs, bounded exact search, finite-vector validation and explicit empty results; add typed payloads and complete footprints. |
| [tensor.py][h-tensor]: TensorKB | Copy/adapt | Exact RAM/VRAM reference and bounded experiments behind the common collection interface. |
| [sqlite.py][h-sqlite]: SQLiteKB, transaction, read_snapshot, backup | Adapt | Durable authoring authority, portable tensors, scoped transactions and immutable snapshots; extend schema and generations. |
| [ivf.py][h-ivf]: DiskIVF | Adapt | Disk-backed coarse retrieval after exact search is correct; adapt update(records) for transactional changed-key postings, causal eligibility and learned-space identity. |
| [authoring_index.py][h-authoring-index]: _AuthoringIndex, _RecordViews | Adapt when authoring scales | Bounded packed arrays, mmap spill and optional GPU key cache; SQLite remains authoritative. |
| [online.py][h-online]: DurableConsolidator, source_batch, ingest, delete_source | Adapt | Source lifecycle, contributions and explicit replacement; preserve admitted assignments while progressively allocating additional source-backed rows. Batch ingestion without whole-scope graph rebuilding; implement affected-pair admission. |
| [consolidation.py][h-consolidation]: OnlineClusterer, CooccurrenceBuilder | Copy as reference | Small correctness oracle for clustering/counts; durable runs use the transactional authoring path. |
| [provenance.py][h-provenance]: merge_metadata, fold_batch, excluded_records, excluded_relations | Adapt | Monotonic lineage and pre-selection exclusion, extended beyond the donor's task/kind filters. |
| [relation_build.py][h-relation-build]: rebuild_relations, _durable_rebuild | Adapt | Stable pair IDs, support/degree limits and resumable publication; keep catalog correlation statistics independent of allocation and add bounded affected-pair growth. |
| [retrieval.py][h-retrieval]: Retriever, Evidence, structured_candidates, slot_assignments, relation_effects | Adapt | Selected-key weighting, null mass, conflict groups and distinct pair support; add learned directional dependence and missing-partner attenuation before a shared point/pair aggregate feeds full-facet KV and LoRA. Keep RAG's feature auxiliary and literal payloads. |
| [trainable.py][h-trainable]: SparseTrainableKB | Adapt substantially | Selected source/memory keys, semantic values, strength/lifetime and two directional fields per pair; dynamic rows with fresh moments, full eligibility and owned key/posting/update transactions. |
| [free.py][h-free]: FreeRecordView, consulted | Adapt later | Differentiable trial records and consulted lineage for writer/experience evaluation without publishing trial leaves. |
| [record_generation.py][h-record-generation] and [density.py][h-density] | Adapt | Separate key, payload and policy generations; retain duplicate-density maintenance and add distinct versioned query-demand counters for allocation. |
| [data.py][h-data]: block_mask, lexical_recurrence, recurrence_keep, match_recurrence_mask | Copy methodology; adapt coordinates | Source-block visibility and family-coordinated anti-copy dropout, independent of facet and meridian numbering. |
| [training.py][h-training]: FreeLatents, FreeKB, encoded_loss, kb_loss, distillation_loss, RelationWriter | Adapt objectives, not stage runner | Free-KB pretraining first; writer training afterward; source-only full-target reconstruction and behavioral verification. |
| [models/encoders.py][h-encoders]: BlockEncoder; [integrations/pplx.py][h-pplx]: PPLXBlockEncoder | Reference for visibility | Native P plus embedding realizer for queries; an offline suffix/full-context writer for sources. |
| [models/readers.py][h-readers] and [models/control.py][h-control] | Reference controls; implement full-KV path | Reuse evidence gates, lifetime and valid-position invariants. Replace delta binding with a shared-semantic projector, exact attention masks and direct KB-context admission. |
| [models/adapters.py][h-adapters]: AdapterLinear | Reference, new parameterization | Replace shared B-C-A factors with independently parameterized LoRA experts and per-event mixtures. |
| [recipe/knowledge.py][h-knowledge] and [recipe/source_bank.py][h-source-bank] | Adapt | Resolve original linked support documents, stream sources, preserve assignment targets and source coverage. |
| [recipe/state.py][h-state] and [artifacts.py][h-artifacts] | Adapt reliability mechanisms | Add KB snapshots, sparse optimizer state and writer targets to megatoken's existing checkpoint/publication path. |
| [serving/replay.py][h-replay] and [serving/evaluation.py][h-serving-eval] | Adapt state-machine tests | One serving/training schedule using native raw-event tapes, actual retrieval receipts, full-KV admissions and historical adapter snapshots. |
| [experience/trainer.py][h-experience-trainer], [experience/objectives.py][h-experience-objectives], [kb/experience.py][h-experience-journal] | Adapt after source writers | Temporary record fitting, writer distillation, task exclusions, accepted regeneration and ordinary KB insertion. |
| [policy.py][h-policy]: collect, calibrate, fit | Later adaptation | Benefit-trained extra latency boundaries and optional direct steering using megatoken's action/state contracts. |

### 10.2 Boundaries that change the port

The donor's six-stage runner learns writers before corpus free-KB optimization; megatoken deliberately uses the opposite order around pretraining. The donor's default record has one dense value vector, its adapters use shared outer factors, and adapter decay is configured rather than learned. These are reference mechanisms, not the desired target interfaces.

Several source-code details matter more than older prose:

- The donor's retriever loads relations whose members already occur in selected candidates; one-hop candidate expansion is new.
- Its distinct-span relation gate cannot activate a pair from one query/support unit.
- Its corpus split validator hashes the complete serialized block list, rather than independently excluding every shared block; megatoken must use chunk/family ancestry.
- Its current native PPLX query path supports block-causal caching; older prefix-view prose is not a reason to add a second query encoder here.
- Its experience writer steps currently disable direct RL; reward fits free records and evaluates regenerated writes.
- Its key-generation counter changes on several non-key record updates; split key, posting, payload and policy state for a bank whose keys and values both learn.
- Its default source ingestion can rebuild a whole scope's relations. Reuse bounded source statistics, but implement incremental active-pair admission for progressive growth.
- Its delta readers do not already supply complete-KV projection, native KB-context admission, or globally balanced all-track independent adapters; those are recipient work.

Keep native LFM/Qwen executors, their tokenizer alignment, paged token caches and shared-basis adapter kernels as references. Megatoken already owns those architectural responsibilities. Snapshot transport scripts are operational examples, not a distributed KB implementation.

## 11. Data identity, source coordinates and causal eligibility

### 11.1 Reuse the recipient data model

Extend [SourceSpan, Lineage, Episode, PhysicalRecord, TrackEvent, Meridian and Supervision][m-schema], the existing [deduplication/quarantine machinery][m-dedup] and [checksummed prepared storage][m-data-storage]. Add a retrieval sidecar keyed by stable episode/event IDs. Do not replace these with the donor's smaller Trajectory format.

Each retrieval training example must identify:

| Field group | Required content |
| --- | --- |
| Query context | Main request/control, completed-prefix cursor, retrieval purpose, semantic-space spec, originating event and original source-block cursor when applicable. |
| Eligible knowledge | Source/KB snapshot, scope, availability cut, valid-time conditions, excluded tasks/source ancestors and permitted collections. |
| Supervision | Teacher query target, positive source spans/supporting sets and fixed support weights, annotation-completeness mask, local feature-value target/cache recipe, desired assistant/media output, explicit unsupported-query label when known. |
| Provenance | Exact source hashes and versions, source-family/duplicate lineage, chunk ordinals/ranges, chunker/tokenizer/teacher preprocessing identities. |
| Visibility footprints | Separate complete sets for query-teacher inputs, document-key context, key-optimization supervision, payload support and relation-support windows. |
| Source retrieval state | Scoring epoch and query/key/index identities, qualified/reserved source intervals, committed injected coverage and ordered admission receipts; keep runtime state distinct from training labels. |
| Schedule | Query selection, search budget/deadline, actual result arrival, native source admission, direct KB admission and any artificial-boundary RNG/cursor. |
| Growth | Catalog/partition epoch, deduplicated query-demand window, source-to-memory assignment, birth/growth step, initialization seed and point/pair admission manifest. |
| Memory context/control | Constituent record/key/value/edge revisions, semantic grouping/weights, native facet layout and projector identities, full-block anchor, support-window/check masks, per-record lifetime and immutable grouped adapter snapshot. |

A footprint is a collection of versioned source ranges, not one integer or one SourceSpan. Byte offsets remain source coordinates; Unicode character offsets or tokenizer positions require declared conversion. Reuse original-source routing from [referenced_documents/distillation_batch][h-knowledge]; QA answers and teacher solutions are supervision, not substitutes for the support documents the writer must encode.

### 11.2 Enforce visibility before selection

For ordinary pretraining and serving, compute eligibility from the whole record footprint and the actual causal snapshot. Exclude unavailable suffixes, prohibited source ancestry, task conflicts, invalid versions and unsupported payload layouts before candidate truncation and mixing. A relation is eligible only when its support footprint is eligible as well as its members.

Exact and IVF adapters should filter during scoped scans before local top-k. For an ANN backend that cannot enforce a filter internally, use bounded overfetch followed by eligibility filtering and exact rescoring; report candidate shortfall and fall back to no evidence when necessary. Do not call a globally truncated, mostly forbidden candidate set a successful eligible top-k.

Hindsight writer training has the separately declared exception: query block i sees completed source blocks through i; writer block j sees its own suffix; j must be at least i+2. Preserve original semantic block IDs through rechunking. If one megatoken or writer output spans several blocks, validate its entire footprint. Require meaningful source blocks; inserted query events and empty scheduling gaps do not satisfy the exclusion.

Scope, source-content availability and optimization lineage are distinct. Preserve direct training/tuning and consulted-memory ancestry for cross-task tests and release exclusion, even after sources are deleted. The local hindsight exception does not authorize an evaluation record optimized against the evaluation answer.

### 11.3 Splits and supervision

Use source/project/task-family and, where useful, time holdouts; keep train, development/rebuild-tuning and untouched release partitions separate. Add per-chunk content hashes and descendant checks to the donor's whole-trajectory identity tests. Near-duplicate and family leakage still need the recipient's origin metadata.

Agentic behavior losses score intended assistant actions and outputs. Tool results are observed evidence; reconstruct them only under an explicitly separate auxiliary objective. Failed actions remain useful inputs when paired with corrections, without becoming desired behavior. Reject or explicitly reprepare examples whose length limits would remove outcomes, evidence or source alignment.

Local writer examples require at least three meaningful source blocks and eligible supervised continuation. Corpus writer examples require original source assignments; missing assignments become a reported coverage gap, never invented targets.

## 12. Retrieval, query training and semantic-memory application

### 12.1 A native vector target is not a new vector codec

Implement an EmbeddingTrainTarget alongside the embedding output spec. It holds the permitted teacher query vector or ranking targets, eligible candidate keys/IDs, positive evidence sets, unknown-label mask and planned loss counts. A native grouped tape reference is optional only when such a tape actually exists.

The current [supervised_meridian/_GroupedTeacherReadout][m-multitrack] path encodes every text/media target through E to form a codec posterior. Add a target-specific branch:

1. Use E(explicit query text) for supervised realizer warm-up.
2. For a retrieval event, let causal P select its normal Gaussian raw block; apply embedding/ranking losses to the realized vector.
3. Commit that selected raw event for recurrent continuation.
4. Use the actual eligible retrieval, source admission and direct full-KV/control schedule for the later memory path.

A teacher vector is neither an E input nor a fake text payload. Do not manufacture a vector posterior simply to satisfy the old target loop. Extend generated-output descriptors, serialization and counters without implying that arbitrary observed vectors are now a supported ingestion modality.

### 12.2 Reuse local evidence weights for full KV and LoRA

Copy the selected-candidate numerical reference from [Retriever.forward and Evidence][h-retrieval]. With live cosine similarity, eligibility/dropout mask \(\mu_i\), nonnegative strength \(u_i\), clipped duplicate density \(\rho_i\) and threshold \(\theta\), the original point evidence is:

\[
a_i=\mu_i u_i f_\epsilon(\cos(q,k_i)-\theta)/\rho_i.
\]

Use the donor's smooth positive-part substitute during warm-up and the declared hard threshold afterward. Candidate IDs and eligibility are discrete; recompute similarities with the live query and selected stored-key leaves. Graph ranking bonuses never enter these original evidence scores.

**Combine compatible semantic values before projection.** Section 12.3 produces attenuated point weights \(\tilde a_i\) and independently supported pair weights \(b_{ij}\). Apply each constituent's own learned decay to its weight, then combine point and pair values inside a bounded compatible group \(s\):

\[
A_s=\sum_{i\in s}\tilde a_i+\sum_{(i,j)\in s}b_{ij},\qquad
\bar m_s=
\frac{\sum_{i\in s}\tilde a_i m_i+\sum_{(i,j)\in s}b_{ij}m_{ij}}{A_s},
\qquad
g_s=\frac{A_s}{\lambda+A_s}.
\]

Here the displayed weights are their current decayed weights. Define \(\bar m_s=g_s=0\) at \(A_s=0\). Use a bounded deterministic grouping policy with explicit compatibility/scope/version metadata; key proximity alone does not certify compatibility. Group membership is fixed within a retrieval window; singleton groups remain valid, and incompatible versions or conflicts stay separate.

The same \(\bar m_s,g_s\) condition the shared KV and LoRA materializers. One compatible mixture produces one complete native-facet KV position:

\[
(K_s^0,V_s)=G_{\rm KV}(\bar m_s),\qquad
K_s^0,V_s\in\mathbb R^{H_{\rm KV}\times F\times d_h},
\quad F=\texttt{config.n\_facets}=32\ \text{initially}.
\]

This deliberately replaces separate per-record KV blocks with preprojection semantic mixing. It is not equivalent to averaging independently projected keys or to attention over separate record keys. Train this mixture/projector computation from memory introduction. The group attention gate is \(g_s\), with an exact slot mask at zero; multiply final LoRA residuals by the same evidence gate even when their controller has biases.

At an admission boundary, pin contributors, group membership, original query/support receipts and operand revisions. Between retrieval boundaries, retain those selections and each constituent's activation age/lifetime. Recompute decayed weights, the aggregate and its full KV projection once per native transaction as bounded local work, under the same admission anchor; all grouped peers use that transaction's immutable snapshot. This requires no new ANN lookup or native event. Do not freeze an obsolete mixture while claiming its constituents decay independently, and do not reset unrelated constituent ages when refreshing one contributor. Section 12.4 defines admission, anchors and invalidation.

**Literal source RAG is unchanged.** It still returns exact source IDs/ranges and ordered bytes. Its training-only local feature objective may reuse original similarity weights, but correlation-conditioned singleton suppression and semantic mixing do not suppress or average literal source evidence.

After bounded key warm-up, native losses train both query and stored keys through mixture weights, and both materializers train the shared values. Detached ANN vectors, support summaries or serving materializations must not replace the live local computation described in sections 12.7 and 13.4.

### 12.3 Relation expansion, pair admission and joint activation

Port bounded source-window counts from [CooccurrenceBuilder][h-consolidation] and stable support/degree/signature rules from [rebuild_relations][h-relation-build], with arity two initially. Count each source window once over catalog identities. At growth barriers, inspect adjacency touching newly allocated endpoints and admit supported pairs when both exist. Preserve learned pair values, weights and moments. Topology and bounded hub-normalized statistical prior \(c_{ij}\in[0,1]\) stay fixed within a gradient window; they are separate from learned directional prominence.

Store two ordered logits per admitted pair:

\[
\gamma_{ij\to i}=\sigma(\eta_{ij\to i}),\qquad
\gamma_{ij\to j}=\sigma(\eta_{ij\to j}).
\]

They express how strongly the other endpoint qualifies each singleton's applicability. Initialize conservatively from the statistical prior and learn them through native utility; co-occurrence is not a logical prerequisite or a factual assertion.

**Check independent causal context support.** Maintain a bounded ledger of original support units with source/span identities, complete lineage, scope and availability. It can include canonical embedding readouts of already committed raw context: this is a readout of existing events, not an extra event, position advance or ANN query. Align the context-to-key role through embedding/paired-support supervision and local training; arbitrary embedding cosine is not automatically calibrated support.

For each checked partner, let \(h_{j\mid i}\in[0,1]\) be the strongest unboosted match from an eligible context unit independent of the unit supporting \(i\). Repeated queries over the same span, duplicate ancestry, graph-derived bonuses and memory-generated restatements cannot manufacture independent support. Retrieved memory cannot certify its own other leg.

Use a detached check flag \(o_{j\mid i}=1\) only after checking the complete declared bounded eligible context window. An unavailable/excluded endpoint, exhausted check budget or unsearched partner has \(o=0\); an ANN omission is not evidence of absence. Incomplete checks may retain observed positive support but supply no absence penalty. An empty maximum is zero. Only eligible, non-dropped relations enter \(N(i)\); excluded relation support cannot suppress an eligible singleton. Existing-event support readouts do not create additional query-demand events.

**Attenuate a singleton without turning absence into contradiction.**

\[
d_i=\max_{j\in N(i)}
\gamma_{ij\to i}o_{j\mid i}(1-h_{j\mid i}),
\qquad
\tilde a_i=a_i(1-\beta d_i),\quad 0<\beta<1.
\]

A prominent checked-but-unsupported partner lowers the singleton weight. Unknown partners leave it unchanged. The floor \((1-\beta)a_i\) and maximum prevent hard rejection and multiplicative degree punishment. Max is differentiable almost everywhere; other examples and supported pair use train the remaining edge weights.

**Make verified pair information prominent.**

\[
b_{ij}=\lambda_2\,\mu_{ij}u_{ij}c_{ij}
\max\!\left(
\gamma_{ij\to i}a_i h_{j\mid i},
\gamma_{ij\to j}a_j h_{i\mid j}
\right).
\]

Set unsupported/distinct-unit-invalid terms to zero exactly. Use original unboosted \(a\) and context support, not attenuated points or graph-amplified values. Start with one bounded global pair-channel gain, for example \(\lambda_2=2\), and ablate it; this favors useful independently supported correlations without forcing pair activation or equal utilization. Retain degree/support normalization and total candidate/pair budgets.

Search semantic seeds, fetch one bounded eligible neighbor set, deduplicate and rescore; a capped graph bonus can propose/rank candidates but cannot produce support or another reinforcement pass. Complete source/scope/time masks precede expansion and truncation.

Learn selected endpoint keys and both prominence logits locally. Recompute endpoint similarities against detached historical vectors; keep the current query live. Cached \(h\) values have no gradient and require invalidation/recomputation after operand changes. The two logits are sparse relation fields with stable endpoint ordering, revisions, touched-row moments and deletion/reset semantics. After pretraining, freeze them as reproduction targets alongside pair values; the pair writer learns both directions from permitted source/member context and statistics. No new key space or writer backbone is required.

### 12.4 Full learned KV on the KB context track

Each admitted KB position contains one **complete native facet block** generated from a compatible semantic evidence group. Use \(F=\texttt{config.n\_facets}\), currently 32, rather than an independent small slot count. Authoritative KB records retain their semantic values \(m_j\), keys, strength, lifetime and provenance; a query-dependent group and its projected tensors are derived state. Keep contradictory or incompatible memories in separate bounded groups.

**Combine semantic evidence before projection.** Let \(b_j\geq0\) denote an admitted constituent's unnormalized lookup coefficient: section 12.3's attenuated point coefficient or supported pair coefficient. Pin coefficients, constituent versions and group membership until a recorded boundary refresh. Retain each constituent's lifetime and admission origin. At native transaction \(t\), calculate:

\[
a_j(t)=b_j\exp[-\operatorname{softplus}(\ell_j)\,\operatorname{age}_j(t)],
\quad A_s(t)=\sum_{j\in s}a_j(t),
\]
\[
\bar m_s(t)=\frac{\sum_{j\in s}a_j(t)m_j}{A_s(t)},
\qquad g_s(t)=\frac{A_s(t)}{\lambda+A_s(t)},\quad\lambda>0.
\]

Apply recorded validity/expiry masks to the constituent weights. Define both outputs as zero without support. Eligible singleton and independently activated pair values use this same aggregation. Age counts globally committed native meridians since each constituent's admission: one tick per committed group, regardless of its track count. Predictor effort, rejected speculation and KB-only updates do not tick it. There is no second independent group-decay multiplier.

One registered MemoryKVProjector expands the aggregate into all native facet positions:

\[
(K_s^0(t),V_s(t))=G_{\rm KV}(\bar m_s(t),\{c_f\}_{f=0}^{F-1},c_{\rm KB}),
\qquad K_s^0,V_s\in\mathbb R^{H_{\rm KV}\times F\times d_h}.
\]

Reuse the native facet-address scheme and ordered identities from [central.slot_codes][m-core]; add a categorical KB descriptor. Facets identify learned addresses, not source-token sequence positions. A small facet-expansion readout and native-shaped KV projection suffice. Tying its weights to central.memory is optional; matching geometry does not require sharing weights. C and P read one shared historical-KV geometry, so no per-layer payload stack is needed. The same aggregate and evidence strength drive the LoRA controller in section 12.5.

**Rematerialize once per native transaction.** Independent constituent decay changes the semantic mixture. A nonlinear projector of that mixture cannot be replaced by scaling its old K/V. Prepare the bounded active groups from retained constituents once at the transaction's committed age, then share that immutable memory/control snapshot across P, R and C. Cache within the transaction; no ANN call, new event, admission or anchor reset occurs. A rejected speculative candidate consumes no lifetime tick. Group membership and lookup coefficients remain fixed; only declared lifetime/validity and their resulting aggregate change. Training must retain the permitted value, coefficient and projector gradient paths.

**Admit a distinct KB context segment.** Extend MultiTrackState with a bounded KBContext, separate from native history and the source RAG track. Its slots have a fixed categorical KB track identity. C's historical reads and P's existing history-read stages combine native history with these full-facet blocks. All tracks can attend; native C publication writes only native events and does not copy the external segment.

At a valid admission boundary, validate source/scope/time eligibility and versions; record constituent identities, coefficients, groups and lifetimes; anchor each new position at the current next-native-meridian position \(p_s\); and atomically install context plus matching adapter state. Increment session revision and invalidate stale speculation. Preserve the native raw/final frontier and meridian/event counters. Matching \(F\) does not make this a raw event: do not call advance_group or C.write, or add a shadow event to alter P's seed.

All facets share their position's retained anchor. Apply \(K_s(t)=\operatorname{RoPE}(p_s,K_s^0(t))\) once to each fresh unrotated materialization; never rotate values. The anchor may share the next native event's phase. Availability and revision enforce causality.

**Gate attention participation.** Add \(\log g_s(t)\) to each facet's attention logit, alongside existing validity masks and track relations. At zero strength, mask the entire position exactly. Extra zero-valued KV changes softmax normalization and is not null. Use the same strength to gate the final LoRA residual. Keep the full \(F\)-facet layout fixed and budget active positions times \(F\).

Evict complete positions. Explicit boundary refresh replaces a recorded admission; idempotent retries reset nothing. Preserve unaffected constituent lifetime origins. Source invalidation removes affected contributions under the lifecycle policy. Removal stops future direct effects, not influence already propagated into native history.

Materialization/replay identities include constituents and revisions, query/key/edge versions, coefficients, grouping, model/projector/facet-layout versions, anchor and transaction age. A per-record KV cache alone cannot represent a projected mixture. Reconstruct historical snapshots in causal order; null checks hold native history, query schedule and prior memory state fixed.

### 12.5 Context-wide independent LoRA experts

Use the same correlation-adjusted aggregate states \(\bar m_s(t)\) and null-aware group strengths as the KV projector to drive the LoRA controllers. **Each module family (C, P, R) owns its own controller** (semantic norm, router, track/stage/module preferences) over the shared semantic input, so the fixed C/R map can be pinned for latent RL while P keeps learning how it selects memory experts. Preserve source/point/pair provenance, and condition on declared causal context and module descriptors. Its output is a sparse mixture over independently parameterized adapter experts:

\[
y_g=W_gx+\sum_{e\in A_g}c_{g,e}\,B_{g,e}A_{g,e}x.
\]

Each expert owns its low-rank factors for explicitly configured parameter groups. The global factors belong to registered model owners and their optimizer/checkpoint; records store semantic states, not copies of every adapter. Signed coefficients, evidence strength, expert choice and lifetime have separate meanings. Apply an exact null mask to the controller's final residual contribution so biases cannot activate adapters without memory.

**Scope covers future computation on all tracks.** Main, tool, source-evidence and multimodal tracks can use the same admitted memory, with different expert choices where their operations differ. Preserve event/track identity through flattened rows and grouped dispatch. Shared context does not require identical coefficients on every track, and context-wide control does not rewrite past activations or weights.

The initial parameter coverage is:

| Owner | Initial contract |
| --- | --- |
| P | Adapt attention query/output and feed-forward/pair projections for all generated tracks. Add packed base-expert projections only under an explicit coverage/compute budget. |
| C | Adapt attention query/output and feed-forward computation for every newly processed native event, including observed source RAG events. Publish its resulting native KV once under the historical control snapshot. |
| R | Adapt ordinary text and multimodal realization. Make the interface explicitly \(R(Z,\mathrm{spec},\mathrm{adapter\_snapshot})\): it remains a direct raw-facet realizer, with no dependence on C activations, but its conditional adapter state is now part of the output contract. |
| Canonical paths | Initially keep source E, query/document embedding readouts, MemoryKVProjector and the shared central.memory projector free of session-dependent memory adapters. Their ordinary model weights still train under the pretraining schedule. |

Canonical paths keep semantic addressing and KV projection coordinates identifiable without blocking all-track P/C/R control. Optional adaptation of observation E must distinguish it from canonical source/index encoding. Record its control identity or actual encoded raw facets, rebuild versioned feature caches appropriately, and never silently use a session-adapted encoding as a canonical document key. Broader projector coverage likewise requires an explicit compatibility and historical-cache contract.

**Use one immutable control snapshot per native transaction.** Compile P, R and C programs from the same admitted-memory/lifetime snapshot, with their permitted inputs and fixed module/stage/row descriptors. R may use its selected raw block within that snapshot. Realizing before or after C must not change the effective adapter state. Memory arriving during speculative work invalidates the candidate; it cannot change realization or central processing halfway through the group. A genuine memory refresh affects only subsequent transactions.

Store selected expert identities, coefficients or their reproducible deterministic control inputs, controller/bank versions and the snapshot identity with native history/replay receipts. Observations processed by adapted C retain their original context too. Rebuilding old history under today's adapters is not equivalent to replaying it.

The existing [base ExpertBank][m-experts] and the memory-LoRA bank have different expert axes. Do not allocate their full Cartesian product across every stage. Start with named compatible matrix families, fixed low ranks and a global active-expert/transfer budget; group residual computations by selected experts rather than mutating shared matrices. Use small nonzero initial outputs and a soft routing warm-up before sparse execution.

The base P routing program in [predict_group][m-multitrack] is compiled before its first historical read. If immediate KB-aware specialization of those base experts is required, provide the same admitted semantic summary to that compilation, alongside its native causal seed. Direct KV admission alone cannot change an already compiled route. The LoRA controller already sees the semantic states and can specialize its own expert selection immediately.

Preserve specialization within a task while measuring participation globally across the optimizer step. Section 12.8 specifies the efficient router-only replay and fairness objective. Do not force uniform expert use independently on each track or make serving routes depend on unrelated batch occupants.

This broader conditional map changes the outcome-RL boundary. Changing a shared semantic state can now alter C or R through its controller, even when their factor weights are frozen. Section 17.2 must pin the C/R materialization controls for the initial native-likelihood objective; P-only learning remains possible under that fixed map. Supervised pretraining and writer fitting can train the broader differentiable system under their declared ownership.

### 12.6 Incremental MaxSim and source-content coverage

Keep one vector per committed retrieval-purpose event initially. Successive query events provide the incoming embedding stream. Let \(\Delta Q_t\) contain the new query vectors processed at a boundary and \(K_c\) the key set for canonical source chunk \(c\), initially one key. Define:

\[
s_t(c)=\max_{q\in\Delta Q_t,\;k\in K_c}\cos(q,k),
\qquad
M_t(c)=\max\{M_{t-1}(c),s_t(c)\}.
\]

Here MaxSim is a maximum over arriving query vectors and chunk keys, not a sum over every past query. Within one pinned request/scoring epoch, sequential updates equal the corresponding batched maximum over the same evaluated query/key pairs. Search ANN only for \(\Delta Q_t\), exactly rescore the returned eligible candidates, and merge with a bounded deferred queue. ANN candidate coverage remains approximate; a missing candidate is unknown, not an exact low score. Store maxima for touched candidates only.

**Newly high** means the incoming query batch gives an eligible, still-uninjected chunk \(s_t(c)\geq\tau_{\rm RAG}\). Enqueue that source obligation once and use its maximum, age and deterministic tie rule to rank deferred work. An old maximum alone cannot discover new work. A strict new record-high score is not required to retry or admit an already qualified candidate: budget exhaustion, fetch failure and delayed arrival must not permanently suppress it. Expire stale deferred relevance at a declared deadline or request/scope change; a fresh qualifying query may enqueue it again. Graph bonuses can rank already qualified source candidates, but cannot bypass the unboosted similarity threshold. Do not automatically fetch adjacent chunks that have not separately qualified.

Maintain three distinct states in the native session:

| State | Contract |
| --- | --- |
| Deferred/qualified | Novel source work with its qualifying query, score, source and semantic snapshot, order, expiry and remaining byte/event budget. |
| Pending/reserved | Exact source intervals claimed by an outstanding fetch or ready result; concurrent searches cannot claim them twice. |
| Injected | Exact source intervals whose observed evidence events have successfully committed through E/C; a search, download or queue entry is insufficient. |

Key coverage by canonical immutable source/content identity and versioned half-open source ranges, using existing duplicate-lineage mappings. Key revisions, ANN row IDs and memory-value revisions are not content identity. Before source payload fetch, subtract injected and reserved coverage from each eligible candidate. Start with canonical ingestion chunks; normalize overlaps to those units or trim only at valid source/encoding boundaries. Revalidate the original key/context footprint before subtraction. For partial sequence admission, mark only committed chunks/ranges, retain the remainder's order and reservation, and never treat an admitted prefix as the entire source.

Push exclusions into candidate search or use bounded overfetch/refill so already injected hits cannot consume the entire candidate budget. Return an explicit shortfall when the budget is exhausted. Reserve before fetch; release or idempotently retry failed/cancelled reservations. At admission, recheck source version, scope, deadline and remaining coverage, then commit evidence and its coverage receipt together. Simultaneous or repeated high matches inject each interval once. Keep injected coverage after its source leaves the model's KV history, and reconstruct it from committed receipts on replay; explicit rereading would be a separate future policy.

Bound scoring epochs, candidate/deferred state and retained query support. A scope change can clear stale relevance while preserving previously injected content. Query/key changes require a fresh score epoch or recomputation with compatible operands (section 13.4); unchanged content stays injected. Scoring, queue management and ANN lookup do not create native events or advance RoPE. Source novelty has no authority over the separate KV/LoRA refresh and decay rules.

### 12.7 Local learning of query keys and stored keys

Adapt the differentiable selected-row path in [SparseTrainableKB][h-trainable] and [Retriever.forward][h-retrieval]. The local training unit is a query and its bounded eligible retrieved keys/values. Candidate IDs, eligibility decisions, graph topology and historical MaxSim state are detached. Fetch one shared FP32 optimizer leaf per selected collection/record/key field, normalize it, and recompute scores with the live normalized query. Reusing detached ANN result vectors would silently disable stored-key learning.

In supervised pretraining, preserve the query gradient through embedding R into P: use the normal differentiable/reparameterized Gaussian raw sample \(Z_q=\mu_P+\sigma_P\odot\epsilon\), with the exact selected sample committed as the native event. Detach the copy used by ANN, not the sample feeding the vector loss. Later outcome-policy scoring follows its separate recorded-action and frozen-map contract.

**Semantic memory uses its actual downstream effect.** Native continuation loss flows through full-facet KV attention and LoRA residuals, the point/pair semantic mixtures and their similarity/correlation weights, into both the query and selected stored keys and directional relation logits. It also trains the declared projector, controller and adapter-bank parameters. Compatible values combine before projection; conflicting groups remain separate. Both the aggregate and attention log-strength have live query/key gradients for active groups. Use smooth positive weights during introduction so weak selected candidates can learn. The exact zero mask remains the no-evidence path; detached serving materializations must not replace the live training projection.

**Source RAG uses a small local feature-value objective.** Its deployed values remain exact source bytes and provenance. For training only, reuse detached, deterministically encoded native source features:
\(v_j=\operatorname{stopgrad}(\operatorname{normalize}(\operatorname{flatten}(E(x_j))))\), using canonical no-adapter E, the fixed raw-facet layout and deterministic posterior mean where E exposes a distribution. These features are neither semantic memory values nor soft source observations. Cache them only under complete source/context, E revision, representation layout and preprocessing identities.

At each boundary, form the supervised local pool with the same detached eligibility and already injected/reserved coverage masks as source retrieval, before adding this query's own reservations. Let \(P_t\) contain the annotated useful support still uncovered and unreserved, including all required remaining multi-hop evidence. Use
\(t=\operatorname{stopgrad}(\sum_{p\in P_t}\beta_pv_p)\), with fixed annotation weights renormalized over \(P_t\): \(\beta_p\geq0\), \(\sum_{p\in P_t}\beta_p=1\). Never derive the target from current learned retrieval weights. Already injected or reserved relevant content is excluded from this objective, not relabeled as irrelevant.

Compare eligible known positives with known/constructed negatives; unknown relevance is not a negative label. If no useful new support remains and annotations are complete, use only the null objective. If useful remaining support exists but ANN misses it, record a coverage miss and use declared eligible-positive assistance or omit the supported local term; do not relabel a retrieval miss as unsupported. Record assisted candidates separately from measured ANN recall.

Use the same positive similarity weighting and null mass as the evidence-weight numerical reference, with source strength initially one:

\[
a_j=m_j\,f_\epsilon(\cos(q,k_j)-\theta)/\rho_j,\quad
D=\lambda+\sum_j a_j,\quad
w_j=a_j/D,\quad w_\varnothing=\lambda/D,\quad
z=\sum_jw_jv_j.
\]

Here \(f_\epsilon\) is the smooth positive-part square during warm-up, later the declared hard form. For supported examples use
\(L_{\rm source}=\tfrac12\lVert z-t\rVert^2-\eta\log\sum_{j\in P_t}w_j\).
For explicitly unsupported or completely annotated no-new-support examples use \(-\eta_0\log w_\varnothing\); an unannotated query does not acquire a fictitious zero target. Stabilize logs numerically. Detached, query-specific targets and source values, supported-mass supervision and known negatives prevent trivial target/value collapse or an all-null solution on supported examples. E still learns through its ordinary native objectives.

For the feature term, \(s_i=q^\top k_i\) and local residual \(g=z-t\) give:

\[
c_i=\frac{\partial a_i}{\partial s_i}\frac{g^\top(v_i-z)}{D},
\qquad
\frac{\partial L_{\rm value}}{\partial q}=\sum_i c_i k_i,
\qquad
\frac{\partial L_{\rm value}}{\partial k_i}=c_i q,
\]

in normalized coordinates, with the normalization Jacobians applied by autograd. Thus the utility of each retrieved value changes **both addresses** through its similarity weight. This needs neither a corpus-wide differentiable search nor another encoder. The value loss is an auxiliary retrieval signal; actual source fetching stays hard and its benefit must be checked by support recall and native RAG outcomes. Detached next-target-block features are an optional task-specific auxiliary, not a replacement for useful-support targets.

Use teacher/query ranking supervision for coverage and a small, explicit teacher-anchor penalty when needed to limit drift; reduce it if it prevents useful local adaptation. Preserve unit normalization, bounded norms/updates and null handling. Do not train addresses merely to increase their current mutual similarity. Test bilateral gradients with teacher/anchor losses disabled, and do not backpropagate through cached maxima or an unbounded history of past retrievals.

### 12.8 Global MoE balance with local specialization

Use causal sparse routing with a small balance objective over the **complete synchronized optimizer update**, pooling eligible training dispatches across ranks, accumulation microbatches and tracks. Individual events can specialize strongly. Do not impose uniform routing within each sequence, track or position, or maximize per-position routing entropy. The Switch objective supplies the basic hard-load/soft-probability construction; DeepSeek-V3's batch-versus-sequence ablation supports using the broader population to preserve specialization. [Switch Transformers][switch-transformers]; [DeepSeek-V3, section 4.5.3][deepseek-v3].

**Preserve the existing two-level router's declared proxy.** [EpochRouter][m-routing] first selects an expert palette, then assigns each executed row to one expert inside that palette. For this router, the balance probability is the full-bank `palette_probability`, computed before hard palette selection. It is a preference proxy, not the marginal probability of the final row assignment. Use actual row assignments for hard load. The balance objective trains the normalization/projection/palette parameters; `row_codes` retain native utility/confidence and existing logit-stability gradients. Do not add a second row-head fairness term in the first implementation, or claim that this proxy directly balances row-head probabilities.

Define one pool per shared expert bank and compatible fixed expert universe. Pool across stages or router heads only when their expert IDs refer to that same bank. Dynamic selected palettes are recorded routing decisions within a pool, not separate fairness populations. Unrelated banks or incompatible fixed eligibility sets require separate pools. All participating ranks must share the parameter version and synchronized update; asynchronous workers with divergent weights are outside this exact objective.

For a pool of E experts, let decision t have detached captured router input x_t, soft probability vector p_t, and c_t selected assignment slots for the used native-program prefix **before capacity dropping**, excluding unused compiled suffixes and other ineligible work. For `EpochRouter`, p_t is its full-bank palette probability and c_t counts those selected row assignments. For a direct top-k adapter router, use its full-bank probability and count every selected expert assignment under the same pre-drop convention, weighting p_t by that assignment count. Record the selected expert IDs. Use this identical population for c_t, A and C_e; log post-drop execution separately. Then:

```text
A   = total eligible selected assignment slots before drops across the update
C_e = number of those selected slots assigned to expert e
f_e = stop_gradient(C_e / A)
P_e = sum_t c_t * p_t,e / A
L_balance = lambda * E * sum_e f_e * P_e
```

Skip a pool with A = 0. The assignment-slot convention keeps hard and soft totals consistent when dispatch sizes differ. Per-pool coefficients have a declared total budget; adding a bank must not silently multiply the overall regularization strength.

**Compute exact current-update statistics through router-only replay.** During each native forward, capture detached compact router-entry inputs, pool/router identity, stage descriptors, actual selected IDs, conditional palette IDs, executed-row masks/counts and any required RNG identity. Native backward proceeds immediately, retaining the usual task gradients into P, retrieval keys, semantic values, projections, expert factors and routers. Keep parameters unchanged until the optimizer barrier.

After the final native microbatch backward, pack all pools' detached C and A statistics into one fixed-order SUM all-reduce. Every rank participates, including empty ranks. Replay only the relevant soft probability heads from captured inputs in bounded chunks; no backbone or expert computation is replayed. For rank r, backward:

```text
S_r,e = sum of c_t * p_t,e over that rank's captured decisions
L_r   = lambda * E * sum_e f_e * S_r,e / A
```

Global f and A are constants for this backward; S_r remains differentiable. Summing rank gradients gives the exact gradient of the specified global objective **with captured inputs and assignments held fixed**. Fairness gradients intentionally stop at router inputs; native utility gradients continue to train upstream representations. The current [DistributedContext.synchronize_gradients][m-distributed] uses SUM and [LossLedger][m-losses] uses true global denominators: apply **no world-size multiplier** and no second normalization of L_r. Accumulate native and balance gradients before the existing synchronization, clipping and optimizer step. Do not detach the replayed probabilities or try to obtain global balancing by averaging finished local balance losses.

Capture only eligible learning dispatches; count selected demand before any capacity drop, while logging execution/drop/gradient exposure separately. Balance expert banks, not KB-record retrieval frequencies or source-catalog allocation.

Capture masks for those dispatches. Exclude padding, unused compiled suffixes, no-grad prefix reconstruction, discarded speculative work and checkpoint/replay duplicates. `RowRoute.prefix` does not currently recompute its auxiliary fields; collect the actual executed prefix and masks explicitly. Honor recorded path-dropout and no-evidence masks. Null remains a separate intervention gate, receives no uniform expert quota and gets no balance gradient. Do not mask the full-bank palette probability down to its selected palette: that would change the chosen proxy and remove its direct pressure on unselected palette alternatives. Router replay must not increment dispatch or learning-exposure counters.

Replace the existing local `router_balance` contribution with this global term; preserve the separately normalized logit-stability term. Bound the temporary input/ID cache by the training window, copy compact slices without retaining backbone storage, and release it after the update. Balancing adds counters and a temporary replay buffer, **no extra Adam state**; model-owned adapter factors still have their ordinary optimizer cost.

Forward selection remains a function of causal event context and recorded exploration, independent of future batch members. Global statistics influence the subsequent parameter update, not reassignment of earlier tokens in the current forward. Track global load, padding, null rate, native task-gradient exposure and expert-use distributions by track/domain. Verify uneven-rank/microbatch gradient equivalence, executed-prefix accounting and absence of duplicate replay exposure. Measure native utility alongside specialization; equal counts alone do not establish useful experts.

## 13. KB storage schema and backend contract

### 13.1 Two typed collections behind one service

Use one search interface over canonical **source** and **memory** collections in a named learned semantic space. Start with **one authoritative SQLite database**, collection-qualified IDs and fixed geometry for each typed collection. One transaction can therefore update selected source/memory keys, semantic values, pair endpoint-dependence fields, sparse moments and changed-key postings. Adapt donor table patterns with explicit collection ownership; separate files require a later joint publication/recovery protocol.

| Entity | Required stored contract |
| --- | --- |
| Semantic-space manifest | Teacher/anchor recipe, learned-space family/version, query/document roles, dimensions, normalization, permitted source context, preprocessing hashes and compatible query-model/key snapshots. Equal dimensions do not establish compatibility. |
| Record header | Database identity, collection-qualified stable ID, scope, origin, key/value revisions, layout and policy generations, support/density, source assignment, availability and valid-time policy. IDs are never reused within a database identity. |
| Source record | Trainable semantic key plus immutable content/version/hash, ordered chunk/range and source-bank or checksummed asset reference. Source bytes are fixed. Detached native feature caches identify their E/layout revision. |
| Memory record | Trainable semantic key and fixed-width semantic value `m`, with declared gate/lifetime coordinates, birth step and reproducible initialization identity. Compatible point/pair values mix before the shared `G_KV` and `G_LoRA` readouts; per-record `m` remains canonical. |
| Materialization contract | `G_KV` produces full F-facet unrotated K/V from each compatible semantic aggregate; `G_LoRA` reads that same aggregate to control the shared expert bank. Model/projector, native KV geometry, slot/RoPE convention and adapter-bank/layout identities determine compatibility. |
| Contribution | Immutable source/context footprint, source/output-slot assignment, assigned record, original key and initial value or immutable reproducible initialization reference, assignment distance and optimization lineage. Multiple distinct assignments must not overwrite each other. |
| Relation | Source-catalog pair signature and support windows/counts; for an admitted memory pair, stable relation ID, collection-qualified member IDs, fixed statistical prior, learned semantic `m`, strength/lifetime and an ordered two-element `endpoint_dependence_logits` field, with complete joint-support lineage. |
| Optimizer row | `(collection, entity_kind, entity_id, field, layout_generation)`; distinguish record from relation. Store FP32 moments, touched-row update count, read revision and committed step under this same identity in storage and checkpoints. |
| Growth/snapshot state | Growth policy/cursor, demand window and partition identity, allocation/init seed, admitted point/pair identities, and coherent model/catalog/key/index/graph/value/optimizer manifests with content hashes. |

Full K/V and LoRA coefficients are **derived materializations**, not independently optimized delta tensors or stored mixture codes. Cache them only under the complete constituent IDs/key/value/edge revisions, grouping, query/support identities, current decayed mixture weights, projector/controller/bank/facet layout and any track/stage conditioning. Cache canonical unrotated K/V before position binding; a positioned session cache additionally identifies its admission anchor and receipt. During training, compute them from live values, query/selected keys and relation fields so both uses train the full weighted aggregate; a detached materialization cannot replace that path. An aggregate is derived session state, not another free record or optimizer row.

Retain canonical source bytes outside the vector tables in immutable recipient storage. Preserve complete source assignments and original contributions for later writer targets and replacement/deletion; stable assignment IDs join supervision offline and are not writer input features. Source keys and corresponding memory keys are separate learned leaves, even when one initializes the other. Store the two relation logits under `(memory, relation, relation_id, endpoint_dependence_logits, layout_generation)` with fixed canonical endpoint order. Each field maps explicitly to the member whose applicability it modulates. Version the relation layout; migrate older snapshots explicitly or reject them. This adds two scalars and their ordinary touched-row moments per pair, without another optimizer or dense edge matrix.

### 13.2 Copy the reliable SQLite core

Port portable little-endian FP32 vector encoding/length validation, finite checks, normalized keys, foreign keys, scope immutability, bounded batch fetches, nested savepoint rollback and read snapshots from [SQLiteKB][h-sqlite]. Retain durable journal/synchronous settings and its supported-runtime checks; begin with the tested single-writer transaction model.

Use SQLite's backup API for coherent snapshots, including committed journal contents. Finish owned write transactions first and publish a self-contained backup without unlisted journal sidecars, following the donor's backup path. An immutable reader must open a truly immutable snapshot. One connection has one owner; writes inside a read snapshot are rejected.

Port the existing meta/records, relations/members, sources/contributions/source_windows/occurrence/relation_registry/sequences and record_policy/record_tasks/relation_tasks tables. Keep bounded relation-build staging (relation_build_state/proposals/degree/selected) and density_progress state checkpoint-portable. Extend them with indexed source-range and support-footprint references; serialized metadata alone must not be the only exclusion path.

Preserve foreign_keys=ON, synchronous=FULL and a bounded busy timeout. Select WAL only when the actually loaded SQLite runtime passes the donor's supported-version/backport check; otherwise use DELETE journaling. Reject unsupported schema versions, dimensions, payload layouts and corrupt vector lengths before exposing records.

Adapt the generation counters:

- **Key generation:** key changes, insertions/deletions or space/membership changes that affect indexed vectors.
- **Index generations:** postings must cover the committed key generation; centroid-training identity and candidate-search policy are tracked separately.
- **Payload generation:** changed semantic values, strength and lifetime; invalidate dependent K/V and controller materializations.
- **Policy/provenance generation:** exclusions, strength/eligibility, support and lineage changes.
- **Graph generation:** topology, statistics and relation-field revisions. Endpoint-weight changes invalidate relation-dependent scores, attenuation and aggregate materializations, but do not alter semantic keys or their postings.
- **Published snapshot identity:** a coherent tuple of the above plus the native model/projector/controller/bank and growth manifest.

Payload-only updates do not require rebuilding the key index. Changed keys update their postings and resident mirrors as specified below, while payload and eligibility caches follow their own generations. All results must come from one declared read snapshot. Session source-coverage receipts are keyed by content identity independently of these score/index generations.

### 13.3 Backend progression

Use [TensorKB][h-tensor] and chunked exact search as the small reference. SQLite exact/HNSW search caches all scope keys in RAM; it is not a disk-scalable default for arbitrary corpus size. [DiskIVF][h-ivf] retains coarse centroids and reads selected cell contents from SQLite; benchmark cell imbalance, filtered recall and bytes read before choosing nlist/nprobe.

For large authoring passes, port the packed working arrays, mmap spill, bounded metadata cache and optional GPU key mirror from [authoring_index.py][h-authoring-index]. Those are disposable accelerators over canonical records. Bound candidate fetches, relation neighborhoods, hot payload cache and tensor transfers; batch and deduplicate IDs per boundary.

Keep original source bytes in the recipient's immutable source/prepared storage. The donor's source-bank iterators and progress cursors are reusable, but an all-source JSON/Python materialization is not required.

The donor does not implement DiskANN, distributed KB shards or a columnar object-store database. Add another backend only when measured recall/IOPS or authoring throughput justifies it, while preserving exact-reference tests and the same typed interface.

### 13.4 Sparse key updates and compatible MaxSim epochs

Pin query-model parameters, source/key/index state and semantic-value revisions for each bounded retrieval/training window. Collect gradients locally through live selected rows, then update at the owned optimizer boundary. Source RAG learns keys while source bytes stay immutable; memory rows learn keys and their declared semantic/strength/lifetime fields; selected pair rows also learn directional dependence logits. Keep the two collections' key parameters distinct unless shared ownership is explicitly configured.

For an exact reference bank, patch the changed key rows by stable ID. Adapt [DiskIVF.update][h-ivf] and [SQLiteKB.upsert][h-sqlite] so changed normalized keys, reassigned cell postings, sparse moments and revisions commit in the same transaction. Keep coarse centroids fixed between scheduled or measured-drift rebuilds. Track centroid training separately from current postings: centroid quality can degrade, but stale postings must not be published as current. Patch disposable mirrors after commit and acknowledge the new bank epoch before another training window reads it. Avoid a full all-key exact/HNSW cache rebuild on every sparse step; immutable HNSW serving is an optional separately tested path.

Incremental maxima are valid only for unchanged scoring operands. After query/key updates, do not compute max(old maximum, new score) across incompatible vectors: an old maximum can decrease, and an older query can become the new winner. Clear score/deferred-ranking and relation-match state, or recompute it over the explicitly retained bounded compatible query/candidate window. Record a new scoring epoch; a version transition alone is not newly high evidence. With cleared state, the next compatible incoming query can qualify novel content again.

Preserve committed injected-content coverage across these transitions. Revalidate or cancel reservations/receipts from an incompatible epoch; unchanged bytes do not become unseen. A rebuild with identical vectors can preserve mathematical scores but must refresh candidate-search coverage state. A memory-only value, endpoint-weight or allocation update does not invalidate raw source MaxSim when its source keys and query-scoring semantics are unchanged. Invalidate derived relation-dependent rankings, support/attenuation snapshots and aggregate materializations independently.

Retain teacher anchors or their reproducible immutable references, plus the task/source lineage of key optimization. Immutable source bytes do not imply an untrained address. Serving uses accepted compatible snapshots; writer and outcome-policy phases keep their declared query/key interfaces fixed. Source-key generation after pretraining must reproduce learned coordinates, with actual new-source retrieval evaluated before replacing the teacher-seeded fallback.

## 14. Free-bank growth, optimization, ownership and checkpoints

### 14.1 Progressive allocation from query demand

Grow the free memory bank progressively **after both generation coherence and chunk/query embedding distillation pass their gates**. Distill native chunk embeddings and causal P retrieval queries into the chosen teacher-initialized space first. Then allocate a small source-backed bank and continue adding entries throughout main pretraining. This readiness gate does not restart global warm-up whenever records are born: native queries, selected source/memory keys, semantic values, `G_KV`, `G_LoRA` and the expert bank continue learning together. No source-to-free-value reconstruction or experience writer runs during this phase; ordinary embedding/readout distillation remains part of native pretraining.

**Measure demand independently of successful memory retrieval.** Reuse the source catalog's coarse semantic partition and eligible source-key search. Count incoming causal queries before graph reinforcement, memory-hit filtering or source-RAG seen-content suppression, so empty-bank misses and previously read sources still inform memory allocation. Keep decayed or windowed per-region counts, cap repeated contributions from a trajectory/task family, and retain only a bounded candidate shortlist. Do not derive demand from future answers or post-retrieval success alone.

A simple priority is `D_c / (1 + A_c)`: recent query demand in region `c` divided by one plus its allocated eligible point-record count. Spend most of a fixed growth budget on high-priority regions, while reserving a small quota for supported misses, unrepresented regions and exploration. Filter already assigned or ineligible catalog entries before truncation, with bounded refill. A query with no eligible source support records an unmet need; it cannot create a supposedly source-backed memory from nothing. Partition changes reset or rebin the small demand state under an explicit generation. These counters are allocation demand, distinct from duplicate-density correction in retrieval.

**Initialize from the current learned catalog.** Choose an eligible unallocated source chunk or explicit reviewed source group. Copy its current compatible live catalog key into a new independent memory-key leaf, retain the teacher anchor, complete context footprint and catalog-key optimization lineage, and initialize a small independent semantic value. Record the source-key revision and deterministic initializer identity. Never cluster near-zero random values by numeric distance. Existing records keep their IDs, assignments, learned values and optimizer moments; growth adds capacity without reseeding the bank.

New rows need opportunities to train. Reserve a small training candidate quota for eligible nearby newborn or underused records, using birth step and existing touched-row counts. Fetch live keys and recompute ordinary scores; use the existing smooth warm-up weights where needed rather than forcing a large intervention. This quota cannot bypass causal/source exclusions, pair-support gates or the independent source-RAG novelty rules. Keep it bounded so growth does not displace the established retrieval distribution.

**Grow correlations with the points.** Precompute bounded source-window pair counts keyed by stable catalog source identities, independently of memory allocation. Count each source window once; repeated queries and allocating another memory never increment corpus support. Each growth batch examines only bounded stored adjacency touching its newly assigned sources. When both endpoints have available distinct memory assignments, admit supported pairs subject to support, strength and degree caps, initialize their semantic values and two conservative directional dependence logits with fresh moments, and preserve original support footprints. Missing endpoints wait for later growth.

Use one-hop relations to promote candidates under the existing limits. A learned joint pair value still requires independently eligible causal query/span support for its members; one query cannot manufacture that evidence. Preserve already trained pair IDs, values and moments. Begin with deterministic admission into available degree capacity and a bounded pending list, rather than replacing learned neighbors after every small batch. The resulting graph follows its recorded incremental admission policy; it need not equal a whole-corpus greedy rebuild.

Reuse donor source windows, occurrence counts, relation registry and bounded rebuild machinery, but implement the affected-pair admission path explicitly: donor default ingestion can rebuild the whole scope. Batch source changes without per-source relation rebuilds. Changes to actual corpus support follow section 15's invalidation rules; scheduled maintenance handles any broader graph refresh.

**Publish growth at the owned optimizer barrier.** Finish the current retrieval/training window first. One bounded SQLite transaction commits new point/pair IDs, contributions, normalized keys, semantic values, source assignments, support, postings/generations, initialization identities and the growth cursor. New optimizer slots start at zero or are created lazily; allocation never advances untouched-row moments. Derive initialization reproducibly from a run seed and stable source/pair identity, and commit allocation sequences with the cursor so retry cannot duplicate records.

Patch disposable mirrors after commit; every rank acknowledges the new bank epoch before its next window. A failure rolls back the entire batch and any affected authoring caches. Memory-only growth changes memory key/posting and graph generations without resetting source MaxSim when its source/query scoring operands are unchanged. Preserve source-content coverage receipts in all cases. Deduplicate retried/replayed demand events by stable query-event identity; record the demand-window and partition generation with the growth cursor.

The minimum additional durable state is the growth policy/readiness/cursor, bounded demand counters and partition identity, init seed and birth/assignment metadata. Existing source support, pair registry, revisions and optimizer counts supply the rest. Stop at declared row, pair, byte and throughput budgets. After pretraining, freeze the final admitted assignment/graph manifest as the writer's reproduction target; reproducing that bank does not require a writer to infer historical query-density allocation decisions.

Keep donor mechanisms distinct: FreeLatents remains an optional per-source capacity diagnostic, SparseTrainableKB supplies persistent free optimization, and FreeRecordView belongs to later writer/experience trials.

### 14.2 Integrate sparse rows with the existing training transaction

Adapt [SparseTrainableKB][h-trainable] into a stable-ID optimizer bridge. Keep global neural parameters, including independent LoRA factors, under megatoken's registered [MasterAdamW owners][m-optimizer]. The ANN service locates IDs; it is not the owner of trainable tensors.

For each optimizer step:

1. Pin query-model/source/key/index/graph and payload revisions for the retrieval window; retrieve eligible candidates and deduplicate collection-qualified record IDs across uses and microbatches.
2. Materialize one FP32 trainable leaf per selected (collection, entity_kind, entity_id, field, layout_generation), reusing it for repeated hits. Source RAG has trainable keys and detached source-feature values; memory rows have trainable keys and declared semantic values/controls. Gather declared relation payloads the same way.
3. Recompute selected scores with live queries and materialize full K/V and LoRA controls from live semantic values. Backpropagate native continuation, local source-value, addressing and regularity terms through LossLedger. Capture compact detached router inputs/actual choices for section 12.8's globally pooled balance replay; complete that replay before gradient synchronization, clipping or any optimizer update.
4. Sum duplicate/rank contributions without dividing by those denominators again. Unscale gradients if loss scaling is enabled, then apply finite checks and declared model/KB clipping rules before either update.
5. Update neural parameters and selected rows under one coordinated step identity. Commit all touched source/memory keys, memory and relation fields, changed-key index postings, optimizer moments, counts and revisions for that step in one SQLite transaction; publish updated row revisions only after that commit succeeds.
6. Patch or invalidate affected key/materialization caches, start required score epochs and release bounded leaves/router inputs. If a growth batch is due, admit it at this owned barrier under section 14.1's transaction before opening the next pinned window. Preserve injected source coverage. Publish a checkpoint only after the coherent step and any scheduled growth complete.

The donor currently optimizes keys, values and relation fields. Add typed collection ownership, the bounded frozen-key warm-up schedule followed by required key learning, typed semantic-value/strength/decay and relation endpoint-dependence fields, full eligibility arguments and optimizer-state reset rules. Untouched rows use explicitly documented touched-row Adam semantics: their moments, counters and weight decay do not advance as if they had dense zero gradients. Record this distinction from dense Adam.

The donor's embedding_optimizer table has no foreign-key cascade to record/relation deletion. Explicitly remove or reset its slots when a row is deleted, regenerated, reaggregated or changes field/layout generation; never attach retained moments to a new logical record. Do not reuse record/relation IDs within a database identity.

The model and database do not magically form a cross-system atomic transaction. If an update fails after one component changes, restore the last complete joint checkpoint; never resume a model checkpoint against whatever newer mutable KB happens to remain.

### 14.3 Start with synchronous ownership

Begin with one authoritative authoring/optimizer owner and synchronous training. Every rank must read the same declared bank state for the step. Preserve row ownership and ordering on resume; never let ranks race independent Adam updates through SQLite.

Each rank sends sorted (collection, entity_kind, entity_id, field, layout_generation, read_revision, gradient) contributions for the step; record and relation IDs have separate namespaces. Those gradients are already globally normalized by LossLedger; the owner rejects stale or incompatible revisions and sums repeated contributions in fixed rank/ID order without dividing again, then performs one touched-row Adam update. A nonfinite result on any participating rank skips both dense and sparse updates; all ranks proceed only after the committed revisions are acknowledged.

A small tensor bank is a correctness oracle for record gradients and the specified touched-row updates, not a second production trainer. Sharded row owners are later work if the single owner becomes a measured bottleneck.

The existing [DiLoCo client][m-diloco] synchronizes versioned named model parameters, not an arbitrary dynamic record database. Enable decoupled free-KB training only after defining stable row ownership, staleness, duplicate update handling and compatible model/reader/value publication.

### 14.4 Extend the existing checkpoint format

Extend [megatoken checkpoint.py][m-checkpoint] through the existing engine. Port coherent SQLite backup, inventory verification, run ownership and immutable publication patterns from [recipe/state.py][h-state] and [artifacts.py][h-artifacts]. The model and SQLite are **coordinated, not atomically committed across systems**. Save only after a complete owned dense/sparse step and any associated growth transaction; suspend mutation while capturing the same declared step and bank state. If one component updates and another fails, restore the last complete joint checkpoint.

Include these state groups:

- Model parameters and layouts for E/C/P, all realizers, `G_KV`, `G_LoRA`, the global adapter experts, and dense optimizer/scaler state.
- A coherent backup of both canonical collections, trainable point/pair semantic values, source/memory keys, typed sparse moments/counts, assignments/contributions, source-window/pair support, relation registry, density maintenance, postings and generation manifests.
- Growth readiness/policy, budgets, cursor and ID sequences; demand counts/window/partition identity; pending bounded pair admission, initializer state and RNG; plus source/authoring and index/graph-maintenance cursors. Saving rows without this state can allocate duplicates or change resumed growth.
- Per-rank data/RNG cursors, normalized optimizer-step identity, artificial-boundary schedule, source-content inventory and semantic/output-spec/tape identities. Writer phases also include writer parameters and the immutable optimized target-bank/assignment snapshot.

For session restart or training-trajectory replay, retain actual selected query raw facets, output specs/vectors and policy densities; query/key/index identities and scoring epochs; bounded MaxSim support/deferred state; fetch reservations/results; and committed source-content coverage receipts. Record pending work as idempotent requests with declared retry/cancel policy, never live futures or connections. Restore durable source coverage independently of incompatible score caches.

Persist every KB context admission's constituent record/key/value/edge revisions, original coefficients and grouping, support-window/check identities, full F-facet layout, projector identity, track, anchor and per-constituent lifetime origins. Recompute the decayed aggregate and KV/LoRA readouts at each recorded native transaction age; cached tensors must identify that exact mixture snapshot. Replay admissions at the same causal frontier and native meridian positions. Preserve committed native clocks and each affected track's progress; catalog growth, ANN search and projected-KV materialization do not fabricate events or backdate evidence. Results cannot change already selected peer outputs. Save adapter selections, coefficients and activation-age state for **all participating tracks**, including their null/default snapshots, so grouped replay cannot apply a refreshed mixture to an earlier peer prediction.

Derived full K/V can be recomputed from retained constituents, exact mixture weights and pinned projectors, or restored from a validated aggregate-materialization cache. Its identity must include all conditioning, contributor/edge revisions, decay age, grouping and position inputs. Never restore a compatible-looking tensor solely by shape, and never restore an old materialization as an independent trainable payload.

Finish writes, take a self-contained SQLite backup, stage model/state files, verify all hashes and matching identities, then flush and atomically publish the completed snapshot and pointer. This publication makes a coherent artifact visible; it does not make the preceding model/database update cross-system atomic. Exclude half-committed meridians and invalidate stale speculation.

Pin session model, projectors, query space, adapter bank and native/payload layouts. Admit a KB-only refresh at a latency boundary only if compatible; revalidate admissions, revisions and affected caches. Source key/query changes follow section 13.4's scoring-epoch transition while retaining injected-content coverage; memory-only value changes refresh only dependent materializations and controls. Model/projector/adapter-weight changes require a new session or complete supported native replay/cache reconstruction, not partial reuse of historical track state.

## 15. Source lifecycle, consolidation and regeneration

### 15.1 Authoring and insertion

Adapt [DurableConsolidator][h-online] and [AuthoredSources/RetrievalDocuments][h-source-bank] into this lifecycle:

1. Resolve immutable original content and its permitted context. Fingerprint source version, bytes/assets, ranges, chunking, semantic-space recipe and writer version.
2. Detect exact repeated ingestion without modifying state. Changed content/alignment requires an explicit replacement operation against the expected source version.
3. In one bounded transaction, register source support, create/update compatible records, retain original contributions and assignments, update source-window counts and advance the authoring cursor.
4. Finalize derived density, graph and index state with resumable fingerprints/cursors.
5. Validate the candidate and publish a compatible snapshot for admission.

The donor's ingest digest does not include the separately passed positions array; add all alignment/context fields to megatoken's identity hash. Cursor progress and record publication must roll back together. Clear authoring mirrors that observed rolled-back writes.

During pretraining, register sources and correlation support independently of active memory allocation. Section 14.1 grows source-backed free rows with compatible seed keys, small random semantic values and fresh sparse moments; local optimization adapts both addresses and values. After writer training, this lifecycle generates accepted learned source keys and memory keys/values. Experience writing uses the same canonical publication contract.

### 15.2 Consolidation is a measured later operation

Reuse the donor's create-new-cluster option, singleton retention, original contribution storage and separate key/value compatibility checks. Its normalized numeric value-distance test is a useful candidate filter, not a semantic contradiction detector.

Require matching scope, time/version policy, origin kind, semantic space and payload/adapter layout before considering a merge. Compare typed payload fields on appropriate scales and check their behavior on representative uses. Keep incompatible near-neighbors separate. Preserve record assignments and measure record count, total storage, reconstruction error and transfer before accepting consolidation.

Do not average LoRA factor matrices; they are global model parameters. Shared semantic values can be compared or combined only under the frozen projector/controller contract, with behavioral validation of complete-KV attention and adapter effects. Independently averaging derived K/V or controller outputs need not reproduce projecting a merged value. For larger or split records, rebuild source-to-output assignments explicitly rather than silently reusing a now-ambiguous slot.

Useful superposition means shared records preserve distinctions and improve transfer or storage efficiency. Fewer rows alone does not establish it. Compare unclustered, clustered and clustered-plus-relations banks at stated payload and compute budgets.

### 15.3 Replacement and deletion

Port source-window decrementing, original-contribution retention and whole-relation removal from the donor. Replacing/deleting a source invalidates affected payloads, relations, dependent experience records, optimizer rows and derived caches.

For freely optimized keys, directional dependence and semantic values, averaging surviving teacher seeds or original random payloads does not recover the useful adapted record. Mark affected shared records unavailable, then regenerate from surviving sources with an accepted writer, or reset/reoptimize them in a declared training run. Clear stale moments and publish only the validated replacement. If no surviving support exists, delete the complete record.

Recompute affected relation statistics and writer-produced relation payloads from remaining support; do not silently reduce a relation's arity or keep unsupported free drift. Preserve training/tuning provenance for holdout checks. Removing KB support does not remove its past influence on neural parameters. Cancel invalid source reservations and score entries, but retain receipts for bytes already committed. Re-embedding unchanged source content does not create new injection eligibility; genuinely changed content follows its new version/range identity.

### 15.4 Fresh corpus reproduction

Freeze a coherent optimized target bank **and its final admitted point/pair subset, source assignments and growth manifest**. Re-encode original sources into a fresh database; reproduce adapted source-index keys and memory keys/semantic values/strength/lifetime and ordered pair-dependence targets. Recompute source statistics, restore declared target correspondences, generate learned pair values and build fresh learned-space indexes.

Generate full-KV blocks and LoRA controls through the frozen materializers. Do not retain optimized target leaves, detached learned K/V copies or old per-record controller outputs behind a nominally regenerated bank. The writer need not predict the historical query-density allocation policy.

Compare optimized, regenerated, deployed and no-new-memory behavior on the same held-out tasks. Report source/assignment coverage, skipped incompatible targets, point/pair value error, materialized behavior and task outcomes. An accepted rebuild creates a new compatible snapshot; rejection keeps the previous one. For genuinely new sources, use the declared insertion policy rather than pretending that old allocation IDs are a writer feature.

## 16. Detailed training methodology

### 16.1 Trainability and phase handoffs

| Phase | Train or allocate | Keep fixed or absent | Completion evidence |
| --- | --- | --- | --- |
| Coherence and embedding bootstrap | Native E/C/P/R objectives; per-megatoken/per-chunk document and causal-query embedding distillation | No memory-value writer or required populated memory | Useful generation, aligned held-out query/chunk embeddings and native replay. |
| Progressive memory warm-up | Allocate dense-demand point/pair entries; train full-KV projector, shared-value LoRA controller/bank and native use, with small positive activation | Existing seed keys fixed only for bounded warm-up; source assignments stable; writer absent | Empty-bank bootstrap, query-density/tail coverage, exact null and useful reader/value gradients. |
| Main free-KB pretraining | Continue bounded growth; train native queries, selected source/memory keys, semantic values, strength/lifetime, directional pair weights, projector, all-track adapters and model; step-global router balance | Source bytes and existing IDs/assignments stable; graph/index fixed within each window; no source/experience memory writer | Bilateral keys and both materializers improve utility; growth preserves learned rows; sparse publication/replay/global gradients agree. |
| Local hindsight writer after pretraining | New source writer and key/value readouts | Final model/query readouts, projector/controllers/bank, learned keys/values and allocation/assignment targets frozen | Permitted-suffix reconstruction and behavioral improvement, followed by transfer. |
| Cross-source writer transfer | Source writer on related/distractor groups | Same accepted interface snapshot | New examples benefit from other-source writes and depend less on same-source access. |
| Corpus reproduction | Document-key readout adaptation, full-context semantic writer and pair-value writer | Optimized source keys and admitted point/pair targets; native query/materializers fixed | Assignment coverage, optimized-coordinate reproduction and fresh-index/bank behavior. |
| Acceptance/new-source use | Source insertion through accepted writers | Inference weights fixed; no answer-based free optimization | Newly inserted permitted evidence and memory improve held-out tasks. |
| Later experience/joint adaptation | Explicitly declared trial records/writer or P policy | Conditional map pinned or explicitly extended for the objective | Regenerated-write benefit or correctly scored policy improvement in a new version. |

Growth continues through warm-up and main pretraining until the declared capacity/coverage stopping rule. A new allocation batch neither restarts the global curriculum nor resets established keys or values. Freeze the final admitted subset and target snapshot only at the writer handoff.

This adapts the donor's free-latent, local-hindsight, group-transfer and reconstruction objectives to the user's order. Initial semantic embedding distillation is allowed and required; source-to-learned-memory reconstruction remains post-pretraining.

### 16.2 Free-KB pretraining losses and data mix

Keep the existing native loss as the main objective:

\[
L_{\rm pre}=L_{\rm native}
+\lambda_eL_{\rm embedding}
+\lambda_rL_{\rm ranking}
+\lambda_vL_{\rm source}
+\lambda_aL_{\rm anchor}
+\lambda_nL_{\rm null}
+\lambda_mL_{\rm memory\ regularity}
+L_{\rm global\ balance}.
\]

Embedding/ranking losses establish useful address coverage; section 12.7's local source-value objective learns both sides of RAG retrieval. A small optional teacher anchor can limit drift and decay when it constrains useful learning. Native continuation gradients already include the full-KV attention and LoRA effects and must not be counted twice.

Train shared point/pair semantic aggregates through **both** materializers, similarity/support-derived activation through query and stored keys and directional pair logits, and explicitly owned projector/controller/bank parameters through actual native behavior. Keep live projected K/V in the bounded training graph. Canonical source features and targets for the RAG auxiliary remain detached; ANN identities, growth choices and historical scheduling state are discrete.

Memory regularity bounds semantic/projected norms, attention strength, adapter residuals and lifetimes over valid uses. Null supervision handles truly unsupported/no-new-evidence examples. Routing balance uses section 12.8's current-update global statistics with detached upstream inputs; its gradients train existing router parameters only. Native utility still trains upstream representations and memory. Keep logit stabilization separate and replace the old local balancing term.

Start new rows at small positive active strength on eligible supervised uses so value/projector gradients can exist; retain the exact zero branch for null cases. Give new or undertrained nearby rows a small candidate-inclusion quota, then rescore them normally. This is exposure rather than forced strong activation. Do not assign experts or memory to unsupported examples merely to satisfy utilization statistics.

Mix ordinary continuation, supported RAG, other-source transfer, permitted reconstruction and verified assistant behavior. Include duplicates, changing candidate counts, hard distractors, conflicts, stale versions, unsupported queries and singletons. Add matched-query examples with the correlated partner present, absent from a fully checked context, and unknown because checking was incomplete. Include both directions and cases where the point remains useful alone, so co-occurrence is not automatically learned as a prerequisite. Retain no-memory examples and independently disable full KV or LoRA on some batches. Any positive assistance obeys the same source eligibility and is reported separately from ANN recall.

Adapt valid-position regularity from [steering_regularity][h-control]; native padding and masked KB slots do not dilute it. The donor's variational FreeLatents/KL budget remains an optional capacity diagnostic. A deterministic semantic value has no KL term without a declared stochastic distribution; norm/noise penalties are not measured storage-bit bounds.

### 16.3 Anti-copy curriculum

Adapt [lexical_recurrence, recurrence_keep and match_recurrence_mask][h-data] to original source spans. Begin with cross-block n-gram families such as the donor's 3/5/8 lengths, group overlapping detections, suppress boilerplate and coordinate dropout across a repeated family.

Increase match dropout only when independent recurrence evidence supports it; high query/key similarity alone is also what useful retrieval requires. Use detached lexical/frozen-feature signals and cap dropout below one. Do not rescale survivors by inverse keep probability: missing evidence is part of the task.

Apply the same dropped evidence to KV and adapter inputs. Compare recurrence-weighted dropout with uniform dropout at the same average rate. Reduce the same-source penalty as cross-source training dominates while retaining general memory dropout. A frozen semantic recurrence detector is an optional extension; the donor's implemented lexical machinery is not already a complete paraphrase detector.

The future-aware recurrence calculation is training-only. It is not an inference dependency or a proof of leakage prevention.

### 16.4 Local source writer

Implement an offline writer contract that accepts ordered source blocks, a permitted footprint and a suffix/full-context mode, and emits source-associated normalized keys plus shared semantic values and strength/lifetime. The pair writer additionally emits both ordered directional dependence logits. Initialize compatible ingestion/features from the trained native system; writer parameters are separately registered so training them does not change the frozen deployed query/model path.

Use a bounded source-context encoder/readout over actual E source representations. Its output slots have explicit source-chunk or source-window assignments. The 32 latent facets do not supply those assignments automatically. The writer must actually receive every source span needed by its target; an isolated chunk cannot stand in for an unseen contextual suffix.

Borrow directional-view and source-block mask tests from [BlockEncoder][h-encoders] and [PPLXBlockEncoder][h-pplx]. Every mixing/cache path obeys the permitted view. Separate physical suffix views are an acceptable initial offline implementation; changing one attention mask while carrying forbidden recurrent state is not.

First fit source-supported key/payload targets in the frozen coordinates, then include continuation loss through the frozen readers/model. **Freeze parameters while retaining autograd through their computation**; wrapping the steered forward in no_grad would prevent learning the writer. Use the declared local j>=i+2 mask, shared evidence restrictions and later-only memory visibility.

Train all eligible continuation blocks of a trajectory under the same historical control schedule; do not construct a separate clean prediction stream for each target. Isolated-prefix examples are a leakage diagnostic, not a replacement for the local objective.

Local warm-up intentionally tolerates the donor's indirect contamination. Keep it labeled and bounded, then move to related/distractor groups with same-source matching suppressed. Keep singleton, conflict and missing-memory cases. Only targets inferable from the local source view belong here.

### 16.5 Full-target corpus distillation

Port [HindsightSystem.distillation_loss and RelationWriter][h-training]. For original assignment c with stored distance d_c and frozen target key/payload:

\[
w_c=\exp(-d_c/T),\qquad
L_{\rm point}=\operatorname{mean}_c w_c\left[
1-\cos(\hat k_c,k_c^*)+
\sum_f\alpha_f\,\ell_f(\hat v_{c,f},v_{c,f}^*)
\right].
\]

Fields f include the shared semantic value, strength and lifetime, plus both directional dependence fields for pair targets under the pinned layout. Derive complete K/V and LoRA controls through the frozen materializers; they are behavioral/readout targets, not separately owned free parameter copies. Normalize field losses by their declared dimensions/scales; measure realized lifetime curves and intervention behavior as well as raw parameter error. Targets are detached. Use the original assignment distance; the donor implements distance weighting, not a separately calibrated confidence probability.

Each contributing source output learns the **complete useful record**, not a fraction of a mixture or an averaged target that erases conflicts. Retain incompatible assignments explicitly. Fix slot correspondence before pointwise loss. Use the donor's [matched_loss][h-experience-memory] only where regenerated slots have established correspondence. When layouts differ, adapt evidence_loss to compare retrieved evidence at identical permitted queries and verify downstream behavior; an explicit set-assignment solver would be additional implementation work.

Resolve linked support documents through [referenced_documents][h-knowledge]. Full-context writer training is an actual objective, not an untrained switch from suffix mode at deployment.

For pair payloads, construct relation-writer inputs from current source-writer member outputs and permitted source-window summaries/statistics. Do not feed optimized target vectors back as source inputs. Reproduce every admitted pair's shared semantic value, strength, lifetime and both directional dependence fields under fixed endpoint ordering. Measure the resulting missing-partner attenuation and supported pair contribution, not just scalar reconstruction error. Recompute source correlation statistics, then use the final assignment/admission manifest to identify the learned pair subset. Do not require the writer to infer historical density allocation. A point-only rebuild that loses useful pair memories is incomplete.

Add behavioral continuation loss and run the fresh rebuild in section 15. If reconstruction fails, inspect source sufficiency, incompatible consolidation, target coverage and free-payload capacity before adding writer complexity or allowing hidden free-leaf retention.

### 16.6 Loss normalization and efficiency

Extend [LossLedger][m-losses] with preplanned globally reduced counts for query targets, local source-value examples, query-normalized key/teacher anchors, ranking examples, unsupported/no-new-evidence examples, writer assignments, relation targets and memory regularity. The global router term uses its own observed assignment-slot counts from section 12.8, not a mean of per-microbatch penalties. Normalize per-query candidate losses internally so changing top-k does not change the unit of supervision. Count writer assignments and valid target fields consistently; report missing coverage separately.

Prepare supervised loss counts from scheduled events/assignments, not from a stochastic number of successful hits. Empty candidate sets still have well-defined teacher/null losses. Mixed modalities retain the recipient's own loss denominators; no fictitious text tokens are introduced for vector events.

Batch teacher encoding of identical permitted views; cache only under complete content/context/model fingerprints. Reuse differentiable tensors within a loss graph where correct, then discard the graph. Keep query realization, selected-row weighting, materialization and the credited continuation in one bounded differentiable window, or use an explicit differentiable replay against its pinned receipts and parameter snapshot. An asynchronous receipt containing detached weights alone cannot carry query/key credit into later continuation; retain local query/key objectives when truncation ends that path. Stop interval batching at retrieval, admission, tool/action or control boundaries. Preserve actual historical states and selected controls; do not replay an old prefix under new adapters to make batching convenient.

## 17. Experience learning and difficult-position retrieval

### 17.1 Later experience writer on the same KB

After source writing and the two readers are useful, adapt [ExperienceWriter][h-experience-model], [FreeRecordView][h-free], [MemoryGradients][h-experience-objectives] and [ExperienceJournal][h-experience-journal]. This is a later objective in the same trainer/runtime/storage system.

The cycle is:

1. Collect a real student attempt and its available outcome/feedback. Hidden verifier state, privileged hints and teacher solutions remain teaching signals, not student-writer inputs.
2. Optimize a bounded temporary record set through frozen native readers/model using verified teacher behavior, teacher-memory/evidence targets and the separately scoped reward extension in section 17.2.
3. Distill the optimized records into the experience writer with matched-record/evidence and verified behavior objectives.
4. Regenerate records and compare no-write, optimized-write and regenerated-write retries plus later task transfer.
5. Publish only accepted regenerated records; retain lineage of direct optimization targets and consulted memory. Rebuild into a shadow candidate at phase boundaries.

The donor currently applies direct RL in free-record fitting and passes with_rl=False in writer updates. Retain that division as a reference, while obeying section 17.2's stricter native materialization contract; begin broad shared-value fitting with verified supervision. Different teacher/student lengths are compared through permitted evidence and actions, not forced token-position alignment.

Progress from deliberate same-task retries to related tasks excluding the current task, then persistent family streams with bounded retention. Include successful attempts and near-null writes to teach restraint. New-source/online uptake evaluations keep inference weights fixed and do not fit free records to test answers.

### 17.2 Use megatoken's actual policy likelihood

Port experience methodology onto native raw-event tapes and [outcome_policy_loss][m-losses], not text-token probabilities. Reward credit uses summed Gaussian log probabilities for selected generated/query events and independent same-task leave-one-out baselines. Observed evidence and deterministic KB admissions are not sampled policy actions. An additional stochastic selector or writer requires its own likelihood accounting.

Implement a differentiable grouped scorer over detached recorded raw events and pinned causal retrieval/admission receipts. The current [replay_group][m-multitrack] is a no-gradient exact-replay verifier; retain it and add a scorer. Reconstruct original context, active KV blocks and historical adapter snapshots, then compute each selected Gaussian density. Verify behavior-density agreement at the same model/KB snapshot before an on-policy update.

**The existing native RL path freezes the conditional materialization map.** [training/rl.py][m-rl] treats deterministic E/C/R owners as fixed. Add embedding readouts, full-KV projector/layout and memory/control receipts to that explicit contract. Freezing factor weights alone does not freeze the map when an optimized semantic value changes C or R through an adapter.

Use the following initial scope:

- For outcome RL, freeze memory, projectors/controllers/bank and materialization controls, and train the declared P/query policy.
- For broad shared-memory fitting, begin with verified behavior and evidence/value supervision through the frozen model; retain autograd into trial semantic values. Use rollout reward for evaluation before assuming a valid broad record-update score.
- A P-only temporary-control experiment may use Gaussian reward fitting if C/E/R and their admission/materialization effects are explicitly pinned. Its trial effects must remain P-only during collection, scoring and evaluation.

A later differentiable scorer can support additional C/context dependence if it reconstructs every conditional transition under the evaluated parameters. That is an explicit extension to test. Changing R's raw-action-to-observable-output map is a stronger change: raw Gaussian score replay alone does not account for that moving decoder. Do not publish a broadly acting shared value on the assumption that a restricted P-only reward update certified all its effects.

For any accepted reward-based temporary-record objective, collect independent same-task rollouts against one fixed trial snapshot. Detach raw actions/rewards but preserve the permitted trial-value graph through the scored policy. Update trial records only after scoring the group, then collect fresh rollouts. Do not detach trial values at retrieval, wrap scoring in no_grad, or reuse on-policy rollouts after changing trial memory.

The donor's current free-record RL and writer updates with with_rl=False are methodology references, not automatic compatibility with this broader native control map. Preserve verified teacher supervision and regenerated-write evaluation. Constant reward supplies no reward gradient.

### 17.3 Optional extra boundaries and direct steering

Adapt [policy.py][h-policy] only after fixed/artificial boundary retrieval is established. Generate counterfactual examples from the same causal state with/without an extra retrieval and train predicted benefit minus a measured or declared retrieval cost. A high-loss position alone does not prove retrieval would help.

The decision precedes the next selected event. Invalidate an uncommitted candidate and regenerate after evidence admission; already emitted output cannot be silently repaired. Use held-out calibration, a per-rollout budget and an explicit never-retrieve option.

The donor's direct residual reader is a later control extension. First test whether the shared semantic records, complete-KV context and all-track LoRA bank already solve the problem. Add a P activation reader in the same key space only if a specific measured gap remains; a separate learned semantic space is not a prerequisite.

## 18. Concrete recipient work packages

All new paths below are **proposed**, relative to `src/megatoken/`. Port small donor modules/tests with commit attribution; retain one native trainer, runtime and storage authority.

| Package | Recipient owner or proposed path | Deliverable and dependency |
| --- | --- | --- |
| A: data/contracts | New memory/contracts.py and data/retrieval.py; extend data/schema.py and prepared manifests | Learned-space/layout and source-footprint identities; query/result/epoch, growth, complete-KV admission and adapter-snapshot contracts. |
| B: vector realization | New generation/embedding.py; extend FacetedGenerator/model/multitrack.py | Registered embedding R, output/train targets, causal P conditioning, bootstrap chunk/query distillation, dispatch, loss and serialization. Depends on A. |
| C: storage/source retrieval | New memory/base.py, tensor.py, sqlite.py, index.py, retrieval.py | Exact/SQLite/IVF references, typed source/memory collections, incremental MaxSim, eligibility/refill, coverage and transactional learned-key postings. Depends on A. |
| D: growth and correlations | New memory/authoring.py, growth.py and relations.py | Versioned source-catalog demand; bounded dense-first/exploration allocation, stable assignments, fresh sparse state, source-level counts, affected-pair admission, learned directional dependence, checked causal support and one-pass lower-order attenuation. Depends on C. |
| E: full-KV context | New memory/semantic.py and memory/context.py; extend native C/P read sites, AttentionFeatures and MultiTrackState | Shared value layout, registered MemoryKVProjector, bounded separate F-facet KB positions from compatible aggregates, anchor/log-strength/mask convention, exact null, direct admission without raw-event/frontier/clock mutation. Depends on A/B/C. |
| F: all-track adapters | New model/adapters.py; extend C/P owners and explicit R dispatch context | Independent factor banks, shared-value controller, sparse stage/track specialization, canonical coordinate paths, immutable transaction snapshots and historical control. Depends on A/E. |
| G: training/optimization | New training/memory.py, training/router_balance.py and memory/optimizer.py; extend engine/meridians/losses/optimizer/distributed/checkpoint | Progressive free-bank pretraining, local bilateral keys, shared-value gradients, exact step-global router-only balance, coherent sparse updates/growth and joint checkpoints. Depends on B–F. |
| H: runtime/replay | Extend multitrack_runtime.py, multitrack_service.py, training/history.py and grouped tapes | Query/source chronology, MaxSim/coverage, direct KB admission, all-track controls, revision invalidation and compatible training replay; separate differentiable policy scorer. Develop with E–G. |
| I: post-pretraining writers | New training/hindsight.py, training/reproduction.py and registered source writers | Local suffix/cross-source training, optimized source-key and admitted point/pair value/dependence target reproduction, fresh indexes/materializations and acceptance. Depends on completed G/H snapshot. |
| J: later experience | New training/experience.py and source facade in memory/authoring.py; extend environments.py/rl.py | Supervised trial fitting, accepted regenerated writes, optional benefit boundaries and explicitly scoped native-likelihood reward fitting. Depends on I. |

Immediate owners include [prepare_meridian_batch/forward_meridian_batch][m-meridians], [MultiTrackGenerator][m-multitrack], [FacetedCore][m-core], [EpochRouter][m-routing], [LossLedger][m-losses], [MasterAdamW][m-optimizer], [DistributedContext][m-distributed], [checkpoint.py][m-checkpoint] and [MultiTrackRuntime][m-runtime].

Serving supports GroupActionTape, but [NativeHistory][m-history] currently consumes singleton tapes and grouped preparation rejects singleton tapes used to certify a multi-event meridian. Package H must close that gap and add deterministic memory/control receipts. An E encoding is not a substitute for the selected raw action; a serving serializer alone does not supply training replay.

Reference PyTorch operations come first. Reuse donor matching, gating, storage and source-methodology oracles. Complete-KV admission and independent adapter computation need new native references; the donor's delta and shared-basis kernels do not implement them. Make the global balancing accumulator replace the existing local term rather than adding a second optimizer or dense-memory training system.

## 19. Acceptance tests, experiments and resource budgets

### 19.1 Tests to port and extend

These are gates for future implementation, not claims that this documentation change ran or passed the donor suites.

| Donor reference | Preserve | Add for megatoken |
| --- | --- | --- |
| [test_storage.py][h-test-storage] | Finite/dimension checks, scope integrity, empty results, rollback, reopen, coherent snapshots and exact parity | Space/layout mismatch rejection, typed collections, offset-sensitive idempotence, key/posting/moment atomicity and split generation invalidation. |
| [test_ivf_batch_search.py][h-test-ivf] | Exact rescoring within probed cells, bounded decoding, sparse IDs | Full eligibility/coverage exclusion before truncation, bounded refill/shortfall, and moved-key posting parity against exact scores in probed cells. |
| [test_authoring_cache.py][h-test-authoring-cache], [test_authoring_scale.py][h-test-authoring-scale], [test_authoring_source_batch.py][h-test-authoring-batch] | Sequential assignment, mmap/cache bounds and cursor/record rollback | Stable admitted assignments during progressive growth, complete context fingerprints and typed source replacement. |
| [test_relation_build.py][h-test-relations] | Stable IDs, support/degree caps, deterministic ties, resumable atomic publication | Pair-only baseline, support-footprint exclusion, allocation-independent counts, affected-pair admission, preserved old pair moments and source-change invalidation. |
| [test_retrieval.py][h-test-retrieval] | Masked current candidates contribute no direct lookup gradient, exact null, conflict slots, density compensation, distinct-span relation gates | One isolated query/support unit cannot activate a pair; earlier independent checked context may supply the other leg. Repeated same-support queries do not manufacture evidence. |
| [test_training.py][h-test-training] | Reader/memory gradients with frozen owners, cross-source/scope masks, full-target point/relation distillation | Source i+2 invariant under query/evidence insertion, raw-facet writer mappings and shared semantic-value/strength/lifetime targets and materialized full-KV/LoRA behavior. |
| [test_sparse_candidates.py][h-test-sparse] | Repeated IDs share a differentiable parameter and accumulate gradients | Bilateral key/value gradients, collection-qualified touched-row parity, denominator/rank equivalence, finite failure handling and key/payload moment resume/reset. |
| [test_serving_replay.py][h-test-replay] | Target-free replay, boundary scheduling and no-match behavior | Native grouped tapes, source coverage and deferred/reserved receipts, delayed arrivals, key/score epochs, direct KB admission with unchanged native frontier/clock, full-block expiry and selected-raw identity. |
| [test_learning.py][h-test-experience], [test_provenance_regressions.py][h-test-provenance] | Trial/writer gradient boundaries, matched targets, rejected-promotion retention, persistent lineage and resume | Native Gaussian event likelihood, source-footprint/task exclusions and fixed-weight new-source uptake. |

Add recipient-specific gradient and ownership checks: embedding/projector/controllers/banks are registered exactly once; vector training needs no vector ingester; teacher targets cannot enter P as inputs; full-KV admission preserves native cache ownership; all-track adapter scopes survive flattened/permuted grouped dispatch and R/C order.

**Exact-null parity uses the same selected raw-event history, query schedule and prior KB context/control/decay state.** Compare absent new candidates with candidates whose new activation is forced to null, while existing memory evolves identically. Full-KV null must exclude slots from attention normalization, not just zero their values. A native query event itself changes C history and consumes compute even when lookup returns nothing. Separately compare task quality with the ordinary no-query baseline and a compute-matched null-query control.

Add focused MaxSim and learned-key gates:

1. **Incremental score parity:** one-query/one-key equals cosine; sequential maxima equal the batched reference for identical evaluated pairs and epochs. Distinguish approximate ANN discovery from exact rescoring.
2. **Novel content only:** repeated high matches, duplicate aliases and concurrent overlapping requests commit each source interval once. Sequence admission marks only its committed prefix; key-only changes and KV eviction do not erase coverage.
3. **Deferred work:** a qualifying source survives budget deferral and idempotent fetch retry without a new score record; stale/cancelled work releases reservations. Low new scores or graph bonuses cannot independently fetch unqualified content.
4. **Bilateral local gradients:** with teacher/anchor losses disabled, controlled value-dependent examples give finite nonzero gradients to P/embedding R and selected stored keys in both collections. Source-feature values/targets and masked current candidates receive no direct auxiliary/lookup gradients; repeated IDs update once. Vary useful values/targets to verify that key updates follow usefulness, not similarity alone. Covered/reserved useful support is masked without repulsion; only remaining support forms the target, with null-only loss for completely annotated empty remainder. Records and relations sharing a numeric ID/field retain distinct leaves and moments.
5. **Key/index and score epochs:** force a key across an IVF cell boundary; verify atomic key/posting/moment updates and rollback. Key changes that lower a cached maximum or change its winning old query require recomputation/reset. Identical-vector index refresh preserves content coverage; memory-only growth/value updates preserve source scores when scoring operands are unchanged.
6. **Learning and regeneration:** local learning improves held-out useful-support retrieval and actual RAG/full-KV/LoRA behavior; regenerated/new-source keys work in the frozen learned space without per-test-answer optimization. Replay reconstructs the same admitted sources and epoch transitions.

Add full-KV, growth and all-track routing gates:

1. **Readiness and demand:** no free-value allocation before both coherence/embedding gates; an empty memory bank can bootstrap from source-supported query demand. Dense regions win first, while the declared miss/exploration quota reaches uncovered support. Replayed queries, graph bonuses and repeated source families cannot multiply counts. Catalog misses never fabricate support.
2. **Incremental allocation/pairs:** each source window contributes once regardless of allocation order. New pairs appear when both endpoints become available, with original distinct-support gates intact. Growth preserves existing point/pair values and moments. Failure injection across row/posting/pair/cursor writes rolls back and deterministic retry reproduces IDs and initialization.
3. **Full-KV admission:** each compatible aggregate projects to exactly F native facet slots in order, with the native head geometry; C and every P track can read them. Admission increments session revision and invalidates old speculation while preserving raw/final frontier, native counters and existing native K/V. Anchors rotate K once, never V; KB-only updates do not advance lifetime age. Whole-block eviction/refresh is idempotent.
4. **Null and materialization:** masked memory leaves attention output unchanged under the same prior context. Verify mixture-aware cache identities and that unequal constituent decay rematerializes the nonlinear projection under the unchanged anchor and immutable transaction snapshot; equal relative weights may reuse compatible projection results. Demonstrate that zero values alone fail this test. Invalid value/projector/layout caches are rejected; live training gradients reach shared semantic values through full KV and LoRA independently and together. Derived materializations have no separate optimizer rows.
5. **All-track ownership and history:** P/C/R use one immutable admitted snapshot per grouped transaction, with recorded per-module choices. Native raw-facet R remains independent of C activations; canonical E/index/query/projector paths remain distinguishable. Old events replay with historical controls, not refreshed values. A changed semantic summary reaches any explicitly memory-conditioned base-MoE compile before its first history read.
6. **Global routing equivalence:** partition captured decisions unevenly across ranks/microbatches and compare router-only gradients with the concatenated reference. Include no-route ranks, shortened compiled prefixes, fixed eligibility, null/path dropout and actual pre-capacity-drop choices. Check no world-size error, old-local-term duplication or replay/counter duplication. Fairness gradients stop at captured inputs; native utility gradients still train upstream memory and representations.
7. **Specialization and serving independence:** at fixed parameters, causal context and RNG, regrouping unrelated sessions does not change their expert assignments. Measure per-track/domain expertise and useful task-gradient exposure under globally balanced utilization. Do not mistake uniform counts, padding or dropped assignments for useful specialization.


Add correlation-specific gates: checked unsupported partners attenuate only the intended endpoint; supported partners avoid that penalty; unavailable, excluded or unchecked partners create no absence penalty. Verify the `(1-beta)` floor, degree-independent maximum, distinct-support pair gating and absence of graph self-reinforcement. Confirm finite local gradients into both endpoint logits, current query and selected point/partner keys, with detached historical features. At matched budgets compare pair gain, attenuation disabled, learned directions and fixed statistical relations; reproduce both directional fields and their behavior through the pair writer. A changed edge field must reset only its dependent relation/materialization state and leave semantic postings and durable RAG coverage intact.

### 19.2 Evaluation matrix

Run no-query, null-query control, source-RAG only, full-KV only, LoRA only, KV+LoRA and all-three conditions. Ablate graph candidate expansion separately from joint pair payloads; report their actual activation rates. Compare uniform/recurrence dropout, fixed/sparse routing and independent reader dropout at matched budgets. Compare teacher-frozen, query-only and joint query/stored-key learning; ablate the local source-value objective separately from ranking/anchors. Compare ordinary repeated RAG with incremental MaxSim/coverage at matched query, byte and latency budgets.

Compare a fixed one-time bank, uniform progressive growth and query-density growth with tail coverage at matched final bytes and training compute. Report demand coverage, newborn training exposure, useful pair activation and incremental allocation cost. Compare initial P-only versus all-track P/C/R coverage at a declared parameter budget; compare existing local balance with global router-only balance using both quality and track/domain specialization. Measure router replay memory/time and actual load/drop/gradient coverage.

For writers, keep distinct reports for local hindsight initialization, cross-source transfer, optimized-bank behavior, regenerated-bank behavior and new-source insertion. Include source/assignment coverage and record/pair reconstruction errors. Low local reconstruction loss is not the same result as useful deployed memory.

For consolidation, report quality against both record count and **total retained bytes**, including original contributions. Test singletons, near-key contradictions, duplicates, changed rules, version conflicts and deletion. Demonstrate novel combinations of separate knowledge where that is an intended capability.

Measure teacher/query embedding recall, eligible novel-support recall, duplicate bytes avoided, deferred/fetch failure behavior, ANN recall against current learned keys, local source-value error, downstream native loss, actual task success, no-memory regression, memory lifetime/reset behavior, writer/key-regeneration throughput, key/posting update cost, cold/warm and tail boundary latency, peak working set and end-to-end task time. The donor's ANN benchmark is a component reference; include query realization, fetch, ingestion, complete-KV projection/admission and adapter staging in the new serving measurements.

Development data used to choose rebuild or trigger thresholds is tuning data. Release and new-source tests remain untouched by base pretraining, free-leaf optimization, writer fitting and policy calibration under their declared exclusion rules.

### 19.3 Configuration and capacity accounting

Version the configuration for semantic spaces/native-facet layouts, support-vector roles/windows/check budgets, directional-logit transforms and initialization, attenuation cap beta, pair gain lambda_2, compatible grouping and aggregate rematerialization; MaxSim thresholds/epochs and source overlap/retry/coverage budgets; query and artificial-boundary scheduling; growth readiness, regional demand windows, dense/miss/exploration quotas, allocation and pair limits; local key/value losses and touched-row rates; index refresh; full-KV slots, anchor/strength/expiry; all-track parameter coverage, expert pools/top-k/ranks and router-replay budgets; writer views and rebuild acceptance.

Use one query vector per retrieval event, pair relations and one complete native F-facet block per compatible semantic aggregation group; F is currently 32. The donor's candidate count near 32, up to four incompatible aggregation slots and rank-eight adapters are reference values, not measured native optima. Canonical record count and admitted aggregate-position count are separate budgets.

For \(N_s\) source keys of dimension \(d_k\) and \(N_m\) point memories with the same address dimension, semantic width \(d_m\) and \(d_a\) strength/lifetime coordinates, canonical FP32 storage is approximately:

\[
4\,[N_s d_k+N_m(d_k+d_m+d_a)]\ \text{bytes}.
\]

Two FP32 Adam moments add twice that storage for rows/fields that have actually been touched. Add admitted pair semantic values/controls, two directional logits per pair and their touched-row moments separately; graph-addressed pairs need no independent ANN key by default. Teacher anchors add storage unless deduplicated through immutable existing vectors. Original contributions, source bytes/features, catalogs, adjacency, query-demand state, indexes, coverage receipts and snapshots remain separate costs.

For \(B\) active aggregation positions with \(F=\mathrm{config.n\_facets}\) facet slots each, \(H_{\rm KV}\) heads, head dimension \(d_h\), and \(b\) bytes per cache scalar, projected full-KV working storage is:

\[
2\,B\,F\,H_{\rm KV}\,d_h\,b.
\]

This is one shared context-KV geometry initially, not a per-layer copy. Count temporary aggregation/projection/attention buffers, bounded causal support vectors, per-transaction rematerialization and compatible unrotated caches separately. Derived K/V and controller codes have **no independent optimizer moments**. A serving cache cannot become a detached shortcut around training the canonical semantic values.

Independent LoRA factors require \(\sum_g E_g r_g(d_{{\rm in},g}+d_{{\rm out},g})\) parameters, plus only explicitly chosen packed-base-expert multiplicity. Their normal model optimizer state is real. Growing a KB row does not create another bank or optimizer. Keep a modest resident/shared or explicitly sharded bank initially; if selected factors need external staging, complete it at the admission boundary. No disk/KB fetch should appear unexpectedly inside an event's MoE execution.

Global balancing adds \(O(E)\) temporary counters per pool and a bounded cache of compact router inputs/IDs, with one packed count reduction per synchronized update. It adds **no persistent balancing parameters or Adam moments**. Measure that replay buffer and latency, and define pool eligibility for genuinely available experts rather than forcing unavailable factor loads. Log chosen demand, actual execution, dropped/padded work and task-gradient exposure separately.

Query count, native facet count, active KB slots, allocated memory rows and expert parameters are distinct budgets. Fewer megatokens reduce native query opportunities but do not cap corpus size or force every fact into one existing native facet. Preserve durable source coverage after native cache eviction, and do not confuse query-demand allocation with duplicate-density compensation or expert fairness.

### 19.4 Milestone decisions

1. **Coherence and semantic bootstrap:** useful native generation, causal per-megatoken queries, per-chunk embeddings, typed storage and complete-footprint eligibility.
2. **Source retrieval:** incremental MaxSim, correct native query clocks, novel-only ordered source admission, learned bilateral addresses and exact replay.
3. **Progressive semantic-memory pilot:** density-first plus tail allocation, incremental pairs, full-KV and all-track LoRA benefit, exact null and canonical shared-value gradients.
4. **Synchronous scale:** growth/sparse/index checkpoints recover; globally pooled router gradients match the reference; specialization and task utility survive balanced utilization under explicit memory/latency budgets.
5. **Post-pretraining writing:** freeze final allocation/assignment/materializer targets; achieve local-to-cross-source transfer and useful fresh-bank reproduction.
6. **New-source/experience expansion:** accepted source-written memories improve tasks at fixed inference weights; later experience and adaptive-boundary objectives satisfy their own materialization/replay gates.

Choose numerical promotion tolerances in the experiment recipe before inspecting the corresponding holdout. A failed writer rebuild calls for source/assignment/capacity diagnosis, not an automatic increase in machinery or retention of hidden optimized leaves.

**The decisive result is that source-written semantic memories improve new tasks after pretraining, while ordinary source evidence remains useful on its own.** The free KB establishes what can be learned in the shared semantic value space; the regenerated KB establishes what can be added and maintained from sources.

[h-status]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/docs/implementation_status.md
[h-base]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/base.py
[h-tensor]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/tensor.py
[h-sqlite]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/sqlite.py
[h-ivf]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/ivf.py
[h-authoring-index]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/authoring_index.py
[h-online]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/online.py
[h-consolidation]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/consolidation.py
[h-provenance]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/provenance.py
[h-relation-build]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/relation_build.py
[h-retrieval]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/retrieval.py
[h-trainable]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/trainable.py
[h-free]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/free.py
[h-record-generation]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/record_generation.py
[h-density]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/density.py
[h-data]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/data.py
[h-training]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/training.py
[h-encoders]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/models/encoders.py
[h-pplx]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/integrations/pplx.py
[h-readers]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/models/readers.py
[h-control]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/models/control.py
[h-adapters]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/models/adapters.py
[h-knowledge]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/recipe/knowledge.py
[h-source-bank]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/recipe/source_bank.py
[h-state]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/recipe/state.py
[h-artifacts]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/artifacts.py
[h-replay]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/serving/replay.py
[h-serving-eval]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/serving/evaluation.py
[h-experience-trainer]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/experience/trainer.py
[h-experience-objectives]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/experience/objectives.py
[h-experience-journal]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/kb/experience.py
[h-experience-model]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/models/experience.py
[h-policy]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/policy.py
[h-test-storage]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_storage.py
[h-test-ivf]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_ivf_batch_search.py
[h-test-authoring-cache]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_authoring_cache.py
[h-test-authoring-scale]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_authoring_scale.py
[h-test-authoring-batch]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_authoring_source_batch.py
[h-test-relations]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_relation_build.py
[h-test-retrieval]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_retrieval.py
[h-test-training]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/test_training.py
[h-test-sparse]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/production/test_sparse_candidates.py
[h-test-replay]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/production/test_serving_replay.py
[h-test-experience]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/experience/test_learning.py
[h-test-provenance]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/tests/experience/test_provenance_regressions.py
[m-schema]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/data/schema.py
[m-dedup]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/data/dedup.py
[m-data-storage]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/data/storage.py
[m-meridians]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/meridians.py
[m-multitrack]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/model/multitrack.py
[m-attention]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/model/attention.py
[m-experts]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/model/experts.py
[m-optimizer]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/optimizer.py
[m-diloco]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/diloco_client.py
[m-checkpoint]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/checkpoint.py
[m-losses]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/losses.py
[m-rl]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/rl.py
[m-runtime]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/multitrack_runtime.py
[m-history]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/history.py
[h-experience-memory]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/experience/memory.py
[m-core]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/model/core.py
[m-routing]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/model/routing.py
[m-distributed]: https://github.com/werg/megatoken/blob/82be7ef563e6cf55a2e2b0cb3d11bf55793071e1/src/megatoken/training/distributed.py
[switch-transformers]: https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf
[deepseek-v3]: https://arxiv.org/html/2412.19437v2
