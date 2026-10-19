# Retrieval and correction memory: architecture plan

**Status:** Proposed extension to the v8 prototype; implementation and learned quality remain to be established.
**Date:** 2026-10-20.
**Reviewed megatoken baseline:** [`b89b10726e395364baeecffa5ead1703b1300553`](https://github.com/werg/megatoken/commit/b89b10726e395364baeecffa5ead1703b1300553).
**Audited donor:** [`hindsight-kb@450164784589b2576256b8b0ba3d2ccc89af992d`](https://github.com/werg/hindsight-kb/commit/450164784589b2576256b8b0ba3d2ccc89af992d).

Sections 1–9 specify the native architecture and curriculum. Sections 10–19 are the implementation specification: donor components, causal/data contracts, storage, optimization, source lifecycle, writer methodology, recipient modules and acceptance gates. Donor code and test references are pinned to the audited revision; proposed megatoken paths are explicitly identified.

## 1. Decision

Use **true RAG on a retrieval track for all retrieved source content**. Separately, train a **free correction KB containing KV deltas and LoRA mixture controls during pretraining**. Only after this pretraining concludes, train the machinery that writes and reconstructs the correction KB from sources.

Both paths use one boundary scheduler and a compatible semantic retrieval interface. They have distinct payloads:

| Path | Stored payload | How it enters megatoken |
| --- | --- | --- |
| Source RAG | Original source chunks, ordered spans and provenance | Ordinary observations on a retrieval track, through E and C |
| Correction KB | Learned KV corrections, LoRA mixture codes and lifetimes | Bounded interventions in subsequent computation |

This replaces the earlier proposal to import complete source KV sequences into the live cache. Ordered retrieval now uses source chunks and normal ingestion. KV **corrections** remain part of the design.

The [multi-track architecture](multi-track-plan.md) already defines shared history, categorical tracks and atomic meridians. Grouped recurrent training also exists in [training/meridians.py][m-meridians] and uses the existing training engine. The vector realizer, retrieval service and **retrieval-aware** recurrent/distributed integration below remain work to implement and train; older status prose should not be read as saying all grouped training is absent.

## 2. Embedding realizer, one query interface, two payload readers

Use the [v8 shared representation](architecture.md): E emits raw facets, P predicts the same geometry, and R realizes it directly. Add an **embedding realizer** alongside the text and media realizers. Its latent input is the selected raw block `Z: [batch, facets, width]` (normally 32 facets), not C activations or a teacher-produced vector.

At a latency boundary, request one retrieval-purpose event on a reusable retrieval track:

```text
Z_query ~ P(committed_history, main_request, retrieval_spec)
q = R_embedding(Z_query, space_id, role="query")
hits = ANN(q, eligible_collections)
```

Condition P on the main request, retrieval purpose and semantic-space specification **before** predicting. Do not assume `state.final` is the main track: it contains the latest meridian, which may contain only tool or evidence events. Use explicit causal request/control conditioning and the existing historical attention path. Merely choosing an embedding realizer after ordinary answer prediction does not establish search intent.

Start with one learned output slot reading all raw facets with their identities, followed by a projection and normalization into the chosen key space. This is a small direct realizer; it does not need lexical expansion or vocabulary decoding. Use the normal selected Gaussian raw-facet event. R is deterministic given that tensor and its output spec; the complete query policy still includes P's sampling. C separately processes the same selected raw block for future predictions; R can run before C. Section 7 defines event commitment and lookup dispatch.

The typed embedding output spec identifies the semantic space/version, query or document role, dimension, normalization and output count. Shared raw geometry does not align unrelated external embedding spaces; preserve each teacher's query/document asymmetry through the appropriate output head or conditioning. Emission needs neither an online teacher call nor a vector ingester.

A reusable facet readout can later support classification logits, scores or typed extraction fields inside ordinary realizers. Implement the embedding output first and factor out that internal component when another task needs it. Per-event readout simply maps over megatokens. Facet identities are not source-token positions; exact token/span extraction needs explicit source alignment. Additional query slots can later express complementary searches, but start with one vector per retrieval event, without imposing one query per facet. Multiple outputs from one raw block do not create additional events.

Use a single ANN service with typed source and correction collections in the chosen embedding space. A query may search either or both with independent result budgets. Semantic keys locate useful records; native attention keys govern their later use.

The source RAG track uses **incremental MaxSim** over incoming query embeddings. Search only the newly committed query vectors, qualify source chunks whose new similarities are high, and fetch/admit only source content not already injected or reserved in the current session. Maintain a bounded relevance queue separately from the durable record of committed source coverage; section 12.6 specifies scoring, overlap and retry semantics. This source-content rule does not suppress correction refresh or learned decay.

Source hits return exact bytes plus source identity, version, span and document order. Ingest them with document/tool roles on a reusable retrieval track. A single track contributes at most one event per meridian, so ordered chunks arrive through successive meridians. A small fixed set of retrieval tracks can admit independent documents concurrently. Source IDs belong in evidence metadata, not in newly allocated track names.

All evidence uses normal committed-meridian RoPE and the shared history budget. Source order is retained through ingestion order and span metadata. Track identity is an attention relation, not an isolation boundary. There is no additional source KV cache or special positional scheme.

## 3. Initialize semantic addressing, then learn both sides

Use a **teacher-initialized learned key space**. Precompute semantic keys for megatoken's actual chunks and keep them fixed only during a bounded addressing warm-up. Then require local joint learning of the native query side and selected stored keys in **both** source RAG and correction collections. Source bytes, stable record identities and initial source assignments stay fixed while their addresses adapt. The external teacher supplies initialization and optional alignment regularization; it does not permanently define the optimized coordinates.

Warm the embedding realizer on E-encoded explicit queries with teacher query targets. Then train causal P retrieval events through the same realizer, using teacher query targets, similarity distributions and labeled useful evidence. The E route is auxiliary supervision, not a requirement to generate query text at serving. P and R must learn together on the raw blocks actually used for retrieval; the teacher vector is a target, not P's input.

After pretraining, train source-key generation into the **optimized learned space** as part of corpus reproduction. Use an ordinary document/key realizer over the permitted native source representation, such as `R_embedding(E(source), role="document")`, alongside the correction writer's key/payload outputs. This is required for new-source insertion once stored keys adapt; it is not a writer trained during pretraining. An isolated E(chunk) cannot reproduce a target depending on unavailable neighboring chunks, so preserve each target's complete source/context footprint. Teacher-only keys remain a declared initialization or evaluated fallback.

Select one teacher space before bulk indexing, after a small comparison:

| Candidate, checked 2026-10-19 | Role and interface |
| --- | --- |
| [Perplexity `pplx-embed-v2-context-9b-preview`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview) | Preferred contextual-chunk pilot: one vector per chunk; distinct `encode_queries` and document `encode`; 2048 dimensions or the trained 1024-dimensional truncation. Pin weights and preprocessing because preview revisions may break compatibility. |
| [Perplexity contextual v1](https://docs.perplexity.ai/docs/embeddings/contextualized-embeddings) | Established hosted alternative: ordered document chunks with `pplx-embed-context-v1-4b`; its documented query recipe uses the same contextual model with `[[query]]`. |
| [EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) | Smaller local comparison and a route to multimodal retrieval; use its distinct query-task and document formats. |

Normalize vectors consistently and pin the model, dimension, query/document formatting and chunk context policy. Pin weight revisions for local teachers; for hosted teachers, retain returned vectors with model identifiers, request settings and input hashes. Do not mix unrelated embedding spaces. The newer [Perplexity late-interaction family](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-9b) produces multiple token vectors; it can be evaluated separately as a ranking teacher without making the first index multi-vector.

The query teacher sees only the available request/prefix. The evidence teacher may see the permitted source context. Seed correction records intended for local hindsight with keys computed from bounded source windows that a later suffix writer can see; reserve broader-context records for corpus reproduction. In continuation and hindsight tasks, every chunk that influenced a contextual embedding is part of its visibility footprint; filtering only the returned chunk is insufficient.

Train for **useful evidence** with multiple positives, hard distractors and no-answer examples; unannotated results are not automatically negatives. Select ANN/top-k IDs discretely, then fetch live selected-key leaves and recompute similarities with the live query. Similarity-weighted correction use lets continuation loss train query keys, stored keys, readers and selected values. For source RAG, the local value objective in section 12.7 trains both address sides from detached features of the retrieved source content and useful-support targets. Literal byte fetch remains discrete. Historical MaxSim maxima, candidate identities and queue decisions are detached scheduling state, not substitutes for this local gradient path.

## 4. Free correction KB during pretraining

After the existing ingestion/bootstrap warm-up, introduce correction memory within the main pretraining curriculum. The base model, correction readers, adapter bank, query keys, selected stored keys and KB values learn together after addressing warm-up. **There is no learned KB writer in this phase.**

Seed records from source chunks, compatible source groups and retained corpus relations. Every record has:

- A teacher-seeded semantic key that becomes trainable after warm-up, and a stable identity.
- Contributing source spans, scope, version and compatible-group assignments.
- A small, randomly initialized trainable payload: canonical KV delta slots, LoRA mixture code, strength and lifetime.
- Sparse relation links with their support and provenance.

“Free” means the payload is an independent optimized variable rather than a writer output. Source IDs are bookkeeping, not a shortcut feature for the query model. Keep initial source-to-record assignments fixed. Do not cluster small random payloads by value distance: proximity near zero is not evidence of compatible meaning.

Reuse records across related examples and train with source-excluded retrieval, irrelevant records, conflicts and missing memory. Keep payload capacity, intervention magnitude and the active working set bounded; retain ordinary no-memory examples and occasionally disable one correction path. These measures encourage reusable corrections without introducing writer training early.

The important uncertainty is **source reproducibility**: a free bank can become useful while encoding information its later writer cannot infer. Source grounding and reuse reduce that risk but do not prove it away. Acceptance therefore depends on a regenerated bank and new-source tests in section 8.

### KV corrections

Store a small fixed number of canonical, unrotated delta slots per record. A reader binds retrieved slots to selected retained event/facet addresses, using the current query and destination state.

For a destination at meridian position `p`:

```text
K_read(p) = K_base_rotated(p) + a * RoPE(p, delta_K)
V_read(p) = V_base(p)         + a * delta_V
```

Begin with temporary overlays on the main track's P history reads. Use a query-scoped view or equivalent grouped attention so peer-track queries continue reading base K/V. The shared base KV remains immutable; C still publishes its normal final KV exactly once. This scopes direct correction application; normal joint P attention can still propagate effects between tracks. Values are never rotated. Bind by stable event/facet identity rather than a shifting cache-array offset, and expire an overlay when its destination is evicted.

Refresh replaces or explicitly blends an existing overlay. Repeated retrieval never silently accumulates the same delta in place. Activation cannot change earlier outputs or cached historical computation.

### Mixtures of LoRA experts

Train a large shared bank of independently initialized low-rank experts. Each expert owns its factors for configured parameter groups; KB records store mixture controls, not a fresh complete adapter per source:

```text
W_effective(t) = W + sum_e c_e(t) * B_e * A_e
```

Start with P-owned stage attention query/output and always-on pair/feed-forward projections. Add packed P expert projections only after specifying their coverage and budget. Exclude the C-owned shared `central.memory` KV projector. Apply mixtures as per-event residual computations, so a main-track correction does not mutate weights shared by every track. Allow lightweight stage/facet gates inside the interval. Extend coverage to C/R only if measured gains justify their additional cache and materialization contracts.

Use small nonzero initial adapter outputs, soft balanced routing for warm-up, then sparse selection. Bound the total active experts across records. Keep expert identity, signed mixture coefficient, intervention strength and lifetime distinct, including an exact null intervention.

For both correction paths, use a learned gate and decay, for example `a = gate * exp(-softplus(rate) * age)`. Measure age in committed progress of the affected track, not predictor effort or the number of admitted evidence chunks. Refresh/reset handles superseding evidence. Decay stops future direct effects; it cannot undo effects already propagated into the trajectory.

## 5. Sparse relation-reinforced retrieval

For the initial design, **second order means pairwise corpus relations/correlations**. Precompute sparse associations from bounded source windows, explicit references and deduplicated co-occurrences. Keep support counts, cap degree and normalize for common hubs. Preserve the distinction between an explicit relation and a statistical association; correlation is not a factual or causal assertion.

Retrieve semantic seeds, expand their eligible neighbors once, and add a bounded relation bonus from the **original** seed scores. Then select a capped set of records. Apply source/scope/time masks to complete key, relation-support and payload-source footprints before expansion and scoring; boosted results do not recursively reinforce themselves. This avoids a dense all-pairs KB and an iterative graph reasoner.

Each retained relation may have its own small free correction payload, backed by its contributing sources. One seed can propose a neighbor, but pair payloads require independent causal query/span support for both members. Retain a bounded window of original support from completed query events; repeated queries over the same support do not create independent evidence. With only one support unit, use the relation for candidate promotion and leave its joint payload inactive. This adapts hindsight-kb's distinct-span gate; its existing retriever does not itself implement graph-neighbor expansion. Keep graph topology and statistical weights deterministic in the first version so corpus reproduction can recompute them. Two-hop propagation or learned edge strengths are later comparisons, not prerequisites.

Keep incompatible values and singleton records. Consolidation requires compatible meaning, scope and version, not just nearby embeddings. Removing or replacing a source rebuilds its contributions and affected relations.

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

Use one scheduler for both retrieval paths:

1. At a completed user/tool turn or other declared boundary, snapshot the causal state, select a retrieval-purpose P event and realize its vector directly from the raw facets. Accept the event through normal grouped commitment, then dispatch the lookup.
2. Incrementally score the new source-query embeddings, enqueue qualifying novel chunks and reserve uncovered intervals before fetching their bytes; retrieve correction candidates under their separate refresh policy. Apply independent budgets.
3. Revalidate available source results and admit them as later observed meridians; atomically record only successfully committed source intervals as injected. Activate selected corrections for the next main prediction. Results cannot affect the query that fetched them or any already-selected peer output.
4. Continue until the next boundary. On no match, activate nothing new and let existing corrections decay.

For long uninterrupted runs, insert artificial boundaries after a configurable amount of committed main-track progress, with a small range of training intervals. A boundary itself is only a scheduling opportunity between complete meridians. Issuing a native query there commits a real event, using the normal C/KV transition and history budget; no empty event is inserted for the boundary. Evidence admission alone must not trigger a recursive retrieval loop.

Use bounded waiting and the same arrival/deadline policy in training and serving. Late results wait for the next valid admission boundary; never grant them to a prediction made before their availability. If speculation is used, evidence or correction admission changes the session revision and invalidates candidates from the old state. Realization and ANN lookup themselves do not mutate causal state or advance RoPE. The selected query event is committed once, and its containing meridian advances the clock once. Speculative query candidates must be accepted against the current revision before external dispatch.

Record selected query raw facets and policy density, output specs/vectors, scoring epochs and query/key/index identities, source qualification/fetch/admission receipts, committed source coverage, bounded deferred/reserved work, activation times, correction selections, gates and decay state for replay. Simulate misses and realistic delayed arrivals during training. Bound query frequency, retrieved bytes/events, graph expansion, active overlays and adapter transfers, and measure actual end-to-end latency.

## 8. Post-pretraining writer and corpus reproduction

| Phase | Work and completion evidence |
| --- | --- |
| Bootstrap | Establish useful native ingestion and grouped training; no correction writer |
| Pretraining with free KB | After teacher warm-up, jointly train the base, retrieval-purpose P/embedding realizer, selected source and correction keys, both readers, expert bank and free correction payloads, with local RAG value and native continuation losses |
| Local hindsight writer | Freeze a coherent final model/embedding-realizer/index/readers/adapter-bank/KB snapshot; train a suffix encoder to reproduce source-supported keys and payloads in those fixed coordinates |
| Corpus reproduction | Reproduce optimized source-index keys and correction keys/payloads, including compatible groups and relations; build fresh learned indexes and records from retained sources |
| New-source use | Insert memories from sources held out of both pretraining and writer fitting, and from completed feedback episodes, with inference weights fixed; evaluate useful transfer without free optimization on their test answers |

For local hindsight, the prefix queries while the writer sees only its permitted suffix. Preserve the [hindsight-kb eligibility rule][h-retrieval]: at **original source block** `i`, same-source writer records begin at `j >= i + 2`. These are source/semantic blocks, not runtime meridians, facet indices or inserted query events. Apply the rule to all contributing source context; a scheduling event cannot manufacture the exclusion gap. This future-suffix access is a training-only exception for correction-writer supervision; source RAG and deployment retain causal visibility. As in the donor, local warm-up may still admit indirect contamination through earlier corrections; its acceptance depends on subsequent cross-source transfer, not same-source reconstruction alone. Supervise only records whose content is inferable from the writer's allowed view; relations needing broader evidence belong in corpus reproduction.

Fit keys/payloads and also measure behavior through the frozen readers. Build a fresh KB from writer outputs and compare it with the optimized bank. Latent reconstruction loss alone is insufficient. If reconstruction fails, inspect source context, consolidation and free-value capacity before increasing writer complexity. Any later joint adaptation starts from an accepted regenerated bank and creates a new versioned snapshot.

## 9. Implementation and acceptance

Implement in this order:

1. Add the embedding realizer, typed vector output spec and retrieval-purpose/space conditioning before P selection. Extend dispatch, grouped losses, output serialization and action-tape replay; begin with teacher query warm-up and one teacher-backed document index adapter. Extend Episode/runtime contracts with selected retrieval events, incremental MaxSim, source coverage/reservations, evidence provenance, eligibility and artificial boundaries.
2. Add free record storage, sparse relation expansion, P-read KV overlays and per-event LoRA mixtures; train and compare both paths.
3. Extend the existing recurrent trainer, loss ledger and checkpoint path with local query/stored-key/value learning, sparse key/index optimizer ownership, synchronous distributed updates and exact query/source-coverage/correction replay.
4. Add the post-pretraining local writer, corpus reproduction and new-source insertion evaluation.

The current dispatcher accepts text/media specifications; vector outputs and their prediction descriptors are new implementation work. Register the embedding realizer/readouts and adapters on the underlying `FacetedCore`/`FacetedGenerator` owners, not only the plain multi-track wrappers. Version KB values and optimizer state with the model, embedding realizer, index, readers and adapter banks. Rebuild derived caches after weight changes. When extending outcome RL, include the embedding realizer and its output contract in the frozen conditional action map, score the native query's Gaussian selection, and explicitly score any additional stochastic correction actions.

Run controlled comparisons for no retrieval, RAG only, KV only, LoRA only, KV+LoRA, and all three; ablate relation reinforcement separately. Report evidence recall, downstream task quality, no-memory regression, correction lifetime behavior, new-source uptake, regenerated-bank quality, replay fidelity and serving cost.

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
| [online.py][h-online]: DurableConsolidator, source_batch, ingest, delete_source | Adapt | Source lifecycle, contributions, bounded authoring and explicit replacement; fixed assignments during initial free pretraining. |
| [consolidation.py][h-consolidation]: OnlineClusterer, CooccurrenceBuilder | Copy as reference | Small correctness oracle for clustering/counts; durable runs use the transactional authoring path. |
| [provenance.py][h-provenance]: merge_metadata, fold_batch, excluded_records, excluded_relations | Adapt | Monotonic lineage and pre-selection exclusion, extended beyond the donor's task/kind filters. |
| [relation_build.py][h-relation-build]: rebuild_relations, _durable_rebuild | Adapt | Stable pair IDs, support/degree limits, resumable staging and atomic graph publication. |
| [retrieval.py][h-retrieval]: Retriever, Evidence, structured_candidates, slot_assignments, relation_effects | Adapt | Correction soft aggregation, exact null, conflict slots and bounded relation effects; reuse local weighting for the RAG training auxiliary while runtime source RAG returns exact source references. |
| [trainable.py][h-trainable]: SparseTrainableKB | Adapt substantially | Selected-row optimization of source/correction keys and typed correction payloads after bounded teacher warm-up, with full eligibility, FP32 state and engine-owned key/posting/update transactions. |
| [free.py][h-free]: FreeRecordView, consulted | Adapt later | Differentiable trial records and consulted lineage for writer/experience evaluation without publishing trial leaves. |
| [record_generation.py][h-record-generation] and [density.py][h-density] | Adapt | Separate key, payload and policy generations; resumable duplicate-density maintenance. |
| [data.py][h-data]: block_mask, lexical_recurrence, recurrence_keep, match_recurrence_mask | Copy methodology; adapt coordinates | Source-block visibility and family-coordinated anti-copy dropout, independent of facet and meridian numbering. |
| [training.py][h-training]: FreeLatents, FreeKB, encoded_loss, kb_loss, distillation_loss, RelationWriter | Adapt objectives, not stage runner | Free-KB pretraining first; writer training afterward; source-only full-target reconstruction and behavioral verification. |
| [models/encoders.py][h-encoders]: BlockEncoder; [integrations/pplx.py][h-pplx]: PPLXBlockEncoder | Reference for visibility | Native P plus embedding realizer for queries; an offline suffix/full-context writer for sources. |
| [models/readers.py][h-readers] and [models/control.py][h-control] | Adapt contracts | Query/destination-conditioned KV use, explicit gates, compact interval controls, valid-position penalties and lifetime state. |
| [models/adapters.py][h-adapters]: AdapterLinear | Reference, new parameterization | Replace shared B-C-A factors with independently parameterized LoRA experts and per-event mixtures. |
| [recipe/knowledge.py][h-knowledge] and [recipe/source_bank.py][h-source-bank] | Adapt | Resolve original linked support documents, stream sources, preserve assignment targets and source coverage. |
| [recipe/state.py][h-state] and [artifacts.py][h-artifacts] | Adapt reliability mechanisms | Add KB snapshots, sparse optimizer state and writer targets to megatoken's existing checkpoint/publication path. |
| [serving/replay.py][h-replay] and [serving/evaluation.py][h-serving-eval] | Adapt state-machine tests | One serving/training schedule using native raw-event tapes, actual retrieval receipts and historical corrections. |
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
- Its key-generation counter changes on several non-key record updates; split key, posting, payload and policy state for a bank whose keys and payloads both learn.

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
| Schedule | Query selection, search budget/deadline, actual result arrival, admission boundary and any artificial-boundary RNG/cursor. |

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

## 12. Retrieval, query training and correction application

### 12.1 A native vector target is not a new vector codec

Implement an EmbeddingTrainTarget alongside the embedding output spec. It holds the permitted teacher query vector or ranking targets, eligible candidate keys/IDs, positive evidence sets, unknown-label mask and planned loss counts. A native grouped tape reference is optional only when such a tape actually exists.

The current [supervised_meridian/_GroupedTeacherReadout][m-multitrack] path encodes every text/media target through E to form a codec posterior. Add a target-specific branch:

1. Use E(explicit query text) for supervised realizer warm-up.
2. For a retrieval event, let causal P select its normal Gaussian raw block; apply embedding/ranking losses to the realized vector.
3. Commit that selected raw event for recurrent continuation.
4. Use the actual eligible retrieval and arrival schedule for the later evidence/correction path.

A teacher vector is neither an E input nor a fake text payload. Do not manufacture a vector posterior simply to satisfy the old target loop. Extend generated-output descriptors, serialization and counters without implying that arbitrary observed vectors are now a supported ingestion modality.

### 12.2 Reuse the donor's correction evidence algebra

Copy the selected-candidate numerical reference from [Retriever.forward and Evidence][h-retrieval]. With cosine score s, eligibility/dropout mask m, stored strength u, clipped redundancy rho and threshold theta:

\[
a_j=m_j\,u_j\,\frac{[s(q,k_j)-\theta]_+^2}{\rho_j}.
\]

Use the donor's smooth softplus substitute during warm-up, then its hard threshold. For each compatible payload slot, retain a conditional payload mean and a separate null-aware gate:

\[
\bar v_s=\frac{\sum_{j\in s}a_jv'_j}{\sum_{j\in s}a_j},
\qquad
g_s=\frac{\sum_{j\in s}a_j}{\lambda+\sum_{j\in s}a_j}.
\]

Define both outputs as zero when the slot has no match. Multiply every reader's final output by its evidence gate; a biased reader must not produce a new correction from an empty slot. Distinct nearby but incompatible payloads occupy separate slots. A gate is activation strength, not a calibrated probability of factual truth.

Reuse this algebra for correction payloads only. RAG returns scored source IDs/ranges and exact content. It does not average source text, apply latent slot reduction to bytes or treat a source embedding as a correction value.

Candidate identity selection is detached. After bounded teacher warm-up, recompute selected scores from the live query and live selected-key optimizer leaves; query/readout, selected-key and correction-payload gradients must remain connected. Joint key learning is required during main pretraining. Section 12.7 specifies local losses; section 13.4 specifies sparse key/index publication. The fixed final interface used for later writer distillation and policy scoring remains a separate phase contract.

### 12.3 Relation expansion and joint correction are separate operations

Port bounded, source-supported pair construction from [CooccurrenceBuilder][h-consolidation] and [rebuild_relations][h-relation-build], using maximum arity two initially. Retain deduplication within windows, support counts, hub normalization, degree caps, stable signatures and atomic graph publication.

Implement one-hop expansion: retrieve semantic seeds, fetch a bounded eligible neighbor set, deduplicate, rescore every candidate against the original query, add a capped statistical relation bonus, and truncate to the result budget. Expanded or boosted results cannot seed a second pass.

Port the donor's distinct-support activation invariant for pair payloads. Keep a bounded ledger of unboosted matches from completed query events, carrying causal support-span identities, scope and source availability. Collapse repeated queries over the same support; eviction or scope reset removes obsolete support. Do not treat arbitrary facet IDs or duplicate query rows as independent evidence. Record query/key scoring revisions with every support; invalidate or recompute match coefficients after their operands change.

A pair correction is inactive until both members have distinct eligible support. Thus a first one-vector boundary can use relation-assisted ranking while leaving joint payloads off. Relation training examples must contain genuine multiple support opportunities; report how often pair payloads actually activate. Learned gate alternatives or additional vector slots are later experiments with their own supervision.

### 12.4 Bind canonical deltas to stable native history

Adapt [KVReader][h-readers] and [BlockControls][h-control] to the single shared history geometry. KV templates are stored unrotated, with an explicit slot/layout identity; the reader binds them to retained event/facet destinations using the current query and available destination state.

Extend MultiTrackState's retained metadata with stable event IDs and original meridian positions aligned with whole-meridian eviction. Its current track IDs and meridian counts are insufficient to identify a destination after insertions and eviction. Bindings contain record/payload revision, destination event/facet, RoPE position, target track, activation boundary and lifetime.

Apply temporary views at the P history-read sites in [MultiTrackCore._stage][m-multitrack] and [AttentionRead][m-attention]. Preserve base tensors and use the existing destination RoPE convention for delta keys; values remain unrotated. Refresh by identity, expire on eviction and reject stale model/layout bindings.

Per-track scoping limits direct application. Joint P attention still allows ordinary indirect cross-track influence. Tests should check read views and parameter ownership, not incorrectly demand identical peer outputs after an intervened track participates in joint computation.

### 12.5 Implement independent adapter experts

Retain the donor's compact block controls, evidence gating, valid-position penalties and historical-state discipline. Implement the chosen sum of independent B_e A_e residuals, with the bank registered on the native model.

Initial eligible parameter groups are P stage attention query/output matrices and always-on feed-forward projections. Add packed P expert projections only after specifying their coverage and budget. The existing base-MoE expert axis in [ExpertBank][m-experts] is distinct from the new memory-adapter expert axis; do not accidentally allocate every memory expert against every base expert and stage.

Thread event/track ownership through flattened pair rows and grouped dispatch. Store compact record mixture codes and current sparse selections; global adapter factors belong in the model checkpoint. Keep an exact null route, a cap on total active experts, soft balanced warm-up, small nonzero initialization, and learned main-progress decay. Retain activation age and lifetime per active record/version before combining its current contribution, so refreshing one record does not reset unrelated memories. Dense coefficient blending is valid in the fixed bank coordinates; averaging independently trained factor matrices is not the same operation.

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

Key coverage by canonical immutable source/content identity and versioned half-open source ranges, using existing duplicate-lineage mappings. Key revisions, ANN row IDs and correction payload revisions are not content identity. Before source payload fetch, subtract injected and reserved coverage from each eligible candidate. Start with canonical ingestion chunks; normalize overlaps to those units or trim only at valid source/encoding boundaries. Revalidate the original key/context footprint before subtraction. For partial sequence admission, mark only committed chunks/ranges, retain the remainder's order and reservation, and never treat an admitted prefix as the entire source.

Push exclusions into candidate search or use bounded overfetch/refill so already injected hits cannot consume the entire candidate budget. Return an explicit shortfall when the budget is exhausted. Reserve before fetch; release or idempotently retry failed/cancelled reservations. At admission, recheck source version, scope, deadline and remaining coverage, then commit evidence and its coverage receipt together. Simultaneous or repeated high matches inject each interval once. Keep injected coverage after its source leaves the model's KV history, and reconstruct it from committed receipts on replay; explicit rereading would be a separate future policy.

Bound scoring epochs, candidate/deferred state and retained query support. A scope change can clear stale relevance while preserving previously injected content. Query/key changes require a fresh score epoch or recomputation with compatible operands (section 13.4); unchanged content stays injected. Scoring, queue management and ANN lookup do not create native events or advance RoPE. Source novelty has no authority over the separate KV/LoRA refresh and decay rules.

### 12.7 Local learning of query keys and stored keys

Adapt the differentiable selected-row path in [SparseTrainableKB][h-trainable] and [Retriever.forward][h-retrieval]. The local training unit is a query and its bounded eligible retrieved keys/values. Candidate IDs, eligibility decisions, graph topology and historical MaxSim state are detached. Fetch one shared FP32 optimizer leaf per selected collection/record/key field, normalize it, and recompute scores with the live normalized query. Reusing detached ANN result vectors would silently disable stored-key learning.

In supervised pretraining, preserve the query gradient through embedding R into P: use the normal differentiable/reparameterized Gaussian raw sample \(Z_q=\mu_P+\sigma_P\odot\epsilon\), with the exact selected sample committed as the native event. Detach the copy used by ANN, not the sample feeding the vector loss. Later outcome-policy scoring follows its separate recorded-action and frozen-map contract.

**Corrections use their actual downstream value effect.** Native continuation loss flows through applied KV/LoRA corrections, the weighted payload and null gate, then the similarity weights into both the query and selected stored keys. It also updates declared payload, reader and adapter-bank parameters. Keep incompatible value slots separate and use smooth positive weights during introduction so initially weak selected candidates can receive gradients.

**Source RAG uses a small local feature-value objective.** Its deployed values remain exact source bytes and provenance. For training only, reuse detached, deterministically encoded native source features:
\(v_j=\operatorname{stopgrad}(\operatorname{normalize}(\operatorname{flatten}(E(x_j))))\), using the fixed raw-facet layout and deterministic posterior mean where E exposes a distribution. These features are neither correction payloads nor soft source observations. Cache them only under complete source/context, E revision, representation layout and preprocessing identities.

At each boundary, form the supervised local pool with the same detached eligibility and already injected/reserved coverage masks as source retrieval, before adding this query's own reservations. Let \(P_t\) contain the annotated useful support still uncovered and unreserved, including all required remaining multi-hop evidence. Use
\(t=\operatorname{stopgrad}(\sum_{p\in P_t}\beta_pv_p)\), with fixed annotation weights renormalized over \(P_t\): \(\beta_p\geq0\), \(\sum_{p\in P_t}\beta_p=1\). Never derive the target from current learned retrieval weights. Already injected or reserved relevant content is excluded from this objective, not relabeled as irrelevant.

Compare eligible known positives with known/constructed negatives; unknown relevance is not a negative label. If no useful new support remains and annotations are complete, use only the null objective. If useful remaining support exists but ANN misses it, record a coverage miss and use declared eligible-positive assistance or omit the supported local term; do not relabel a retrieval miss as unsupported. Record assisted candidates separately from measured ANN recall.

Use the same positive similarity weighting and null mass as the correction numerical reference, with source strength initially one:

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

## 13. KB storage schema and backend contract

### 13.1 Two typed collections behind one service

Use one search interface with source and correction collections in a named learned semantic space. Start with **one authoritative SQLite database and two explicitly typed collections**, using collection-qualified IDs and fixed key/payload geometry per collection. This permits one transaction to update selected keys in both collections, correction payloads, sparse moments and index postings. Adapt donor table patterns with explicit collection ownership; no universal variable-shape tensor store is required. Separate database files are a later backend option requiring a joint publication/recovery barrier, not a cross-file atomicity claim.

| Entity | Required stored contract |
| --- | --- |
| Semantic-space manifest | Teacher initialization/anchor recipe, learned-space family/version, compatible query-model/key snapshots, query/document recipes, normalization, dimensions, context policy and preprocessing hashes. Equal dimensions alone do not imply compatible spaces. |
| Record header | Database/record identity, scope, collection, factual/experience origin, key version, payload version, policy generation, density/support, compatibility group and available/valid-time metadata. |
| Source payload | Immutable original source/content hash, ordered chunk ordinal/range, source version and source-bank reference; its separate stored key can learn. Media can reference the recipient's checksummed assets. Training-only native feature caches carry E/layout revisions. |
| Correction payload | Canonical delta slots, LoRA mixture code, activation strength and learned lifetime fields under a fixed payload-layout ID. |
| Correction compatibility | Native model family/geometry, target-module and reader layout, KV/RoPE convention, adapter-bank identity and schema version. |
| Contribution | Source/content version, original range/chunk/output-slot mapping, assigned record ID, original key/payload, original assignment distance and complete visibility footprints. |
| Relation | Stable ID/signature, ordered member IDs, support windows/counts, statistical weight, separately typed learned joint/member adjustments and their lineage. |
| Optimizer row | (collection, entity_kind, entity_id, field, layout_generation), with entity_kind distinguishing record from relation; FP32 moments, update count, read revision and last committed step. Use the same identity in optimizer primary keys and checkpoints. |
| Snapshot manifest | Model, source bank, semantic collections, graph, payloads, optimizer and writer-target identities with content hashes. |

A correction payload layout can flatten canonical delta tensors and a compact adapter code into one fixed float tensor plus typed metadata. Store strength and decay in explicitly declared parameter coordinates. The reader and writer must agree on that layout; an opaque dense donor value is not a sufficient compatibility contract.

Original contributions are mandatory for reconstruction. Extend the donor's single position integer and one-assignment primary key to explicit source/output-slot assignments; support multiple distinct assignments where needed without overwriting them. Preserve incompatible targets as distinct assignments/slots. Their stable IDs join supervision offline and are not writer input features.

### 13.2 Copy the reliable SQLite core

Port portable little-endian FP32 vector encoding/length validation, finite checks, normalized keys, foreign keys, scope immutability, bounded batch fetches, nested savepoint rollback and read snapshots from [SQLiteKB][h-sqlite]. Retain durable journal/synchronous settings and its supported-runtime checks; begin with the tested single-writer transaction model.

Use SQLite's backup API for coherent snapshots, including committed journal contents. Finish owned write transactions first and publish a self-contained backup without unlisted journal sidecars, following the donor's backup path. An immutable reader must open a truly immutable snapshot. One connection has one owner; writes inside a read snapshot are rejected.

Port the existing meta/records, relations/members, sources/contributions/source_windows/occurrence/relation_registry/sequences and record_policy/record_tasks/relation_tasks tables. Keep bounded relation-build staging (relation_build_state/proposals/degree/selected) and density_progress state checkpoint-portable. Extend them with indexed source-range and support-footprint references; serialized metadata alone must not be the only exclusion path.

Preserve foreign_keys=ON, synchronous=FULL and a bounded busy timeout. Select WAL only when the actually loaded SQLite runtime passes the donor's supported-version/backport check; otherwise use DELETE journaling. Reject unsupported schema versions, dimensions, payload layouts and corrupt vector lengths before exposing records.

Adapt the generation counters:

- **Key generation:** key changes, insertions/deletions or space/membership changes that affect indexed vectors.
- **Index generations:** postings must cover the committed key generation; centroid-training identity and candidate-search policy are tracked separately.
- **Payload generation:** changed correction values, codes and lifetimes.
- **Policy/provenance generation:** exclusions, strength/eligibility, support and lineage changes.
- **Graph generation:** topology, statistics and relation payload versions.
- **Published snapshot identity:** a coherent tuple of the above plus the native model/reader/bank.

Payload-only updates do not require rebuilding the key index. Changed keys update their postings and resident mirrors as specified below, while payload and eligibility caches follow their own generations. All results must come from one declared read snapshot. Session source-coverage receipts are keyed by content identity independently of these score/index generations.

### 13.3 Backend progression

Use [TensorKB][h-tensor] and chunked exact search as the small reference. SQLite exact/HNSW search caches all scope keys in RAM; it is not a disk-scalable default for arbitrary corpus size. [DiskIVF][h-ivf] retains coarse centroids and reads selected cell contents from SQLite; benchmark cell imbalance, filtered recall and bytes read before choosing nlist/nprobe.

For large authoring passes, port the packed working arrays, mmap spill, bounded metadata cache and optional GPU key mirror from [authoring_index.py][h-authoring-index]. Those are disposable accelerators over canonical records. Bound candidate fetches, relation neighborhoods, hot payload cache and tensor transfers; batch and deduplicate IDs per boundary.

Keep original source bytes in the recipient's immutable source/prepared storage. The donor's source-bank iterators and progress cursors are reusable, but an all-source JSON/Python materialization is not required.

The donor does not implement DiskANN, distributed KB shards or a columnar object-store database. Add another backend only when measured recall/IOPS or authoring throughput justifies it, while preserving exact-reference tests and the same typed interface.

### 13.4 Sparse key updates and compatible MaxSim epochs

Pin query-model parameters, source/key/index state and correction payload revisions for each bounded retrieval/training window. Collect gradients locally through live selected rows, then update at the owned optimizer boundary. Source RAG learns keys while source bytes stay immutable; correction rows learn keys and their declared payload fields. Keep the two collections' key parameters distinct unless shared ownership is explicitly configured.

For an exact reference bank, patch the changed key rows by stable ID. Adapt [DiskIVF.update][h-ivf] and [SQLiteKB.upsert][h-sqlite] so changed normalized keys, reassigned cell postings, sparse moments and revisions commit in the same transaction. Keep coarse centroids fixed between scheduled or measured-drift rebuilds. Track centroid training separately from current postings: centroid quality can degrade, but stale postings must not be published as current. Patch disposable mirrors after commit and acknowledge the new bank epoch before another training window reads it. Avoid a full all-key exact/HNSW cache rebuild on every sparse step; immutable HNSW serving is an optional separately tested path.

Incremental maxima are valid only for unchanged scoring operands. After query/key updates, do not compute max(old maximum, new score) across incompatible vectors: an old maximum can decrease, and an older query can become the new winner. Clear score/deferred-ranking and relation-match state, or recompute it over the explicitly retained bounded compatible query/candidate window. Record a new scoring epoch; a version transition alone is not newly high evidence. With cleared state, the next compatible incoming query can qualify novel content again.

Preserve committed injected-content coverage across these transitions. Revalidate or cancel reservations/receipts from an incompatible epoch; unchanged bytes do not become unseen. A rebuild with identical vectors can preserve mathematical scores but must refresh candidate-search coverage state. A payload-only correction update does not invalidate source MaxSim when its source keys and query-scoring semantics are unchanged.

Retain teacher anchors or their reproducible immutable references, plus the task/source lineage of key optimization. Immutable source bytes do not imply an untrained address. Serving uses accepted compatible snapshots; writer and outcome-policy phases keep their declared query/key interfaces fixed. Source-key generation after pretraining must reproduce learned coordinates, with actual new-source retrieval evaluated before replacing the teacher-seeded fallback.

## 14. Free-bank optimization, distributed ownership and checkpoints

### 14.1 Seed before any learned writer

Create stable records from permitted source chunks and explicit source-supported groups; precompute teacher keys and bounded pair statistics. Initialize small independent correction payloads. Keep initial IDs, assignments and graph topology fixed; freeze keys only during bounded addressing warm-up. Main pretraining then jointly learns native queries, selected source/correction keys, readers, adapter factors and correction payloads.

Do not run writer-era numeric key/value consolidation over near-zero random payloads. Early sharing may use reviewed source-only duplicate/group assignments; learned consolidation follows meaningful payload training and a measured compatibility test.

Distinguish three donor mechanisms: FreeLatents is a per-source oracle/capacity diagnostic; SparseTrainableKB is the persistent freely optimized bank; FreeRecordView is temporary trial memory for later writer/experience work. The first is optional and does not replace source-backed sparse pretraining.

### 14.2 Integrate sparse rows with the existing training transaction

Adapt [SparseTrainableKB][h-trainable] into a stable-ID optimizer bridge. Keep global neural parameters, including independent LoRA factors, under megatoken's registered [MasterAdamW owners][m-optimizer]. The ANN service locates IDs; it is not the owner of trainable tensors.

For each optimizer step:

1. Pin query-model/source/key/index/graph and payload revisions for the retrieval window; retrieve eligible candidates and deduplicate collection-qualified record IDs across uses and microbatches.
2. Materialize one FP32 trainable leaf per selected (collection, entity_kind, entity_id, field, layout_generation), reusing it for repeated hits. Source RAG has trainable keys and detached source-feature values; correction rows have trainable keys and declared payloads. Gather declared relation payloads the same way.
3. Recompute selected scores with live queries. Backpropagate continuation, local source-value, addressing and regularization terms through LossLedger so neural parameters and selected record leaves receive the same globally normalized losses.
4. Sum duplicate/rank contributions without dividing by those denominators again. Unscale gradients if loss scaling is enabled, then apply finite checks and declared model/KB clipping rules before either update.
5. Update neural parameters and selected rows under one coordinated step identity. Commit all touched source/correction keys, correction and relation fields, changed-key index postings, optimizer moments, counts and revisions for that step in one SQLite transaction; publish updated row revisions only after that commit succeeds.
6. Patch or invalidate affected key/payload caches, start the required scoring epoch and release the bounded leaf working set. Preserve injected-content coverage. Publish a checkpoint only after the coherent step completes.

The donor currently optimizes keys, values and relation fields. Add typed collection ownership, the bounded frozen-key warm-up schedule followed by required key learning, typed KV/code/strength/decay fields, full eligibility arguments and optimizer-state reset rules. Untouched rows use explicitly documented touched-row Adam semantics: their moments, counters and weight decay do not advance as if they had dense zero gradients. Record this distinction from dense Adam.

The donor's embedding_optimizer table has no foreign-key cascade to record/relation deletion. Explicitly remove or reset its slots when a row is deleted, regenerated, reaggregated or changes field/layout generation; never attach retained moments to a new logical record. Do not reuse record/relation IDs within a database identity.

The model and database do not magically form a cross-system atomic transaction. If an update fails after one component changes, restore the last complete joint checkpoint; never resume a model checkpoint against whatever newer mutable KB happens to remain.

### 14.3 Start with synchronous ownership

Begin with one authoritative authoring/optimizer owner and synchronous training. Every rank must read the same declared bank state for the step. Preserve row ownership and ordering on resume; never let ranks race independent Adam updates through SQLite.

Each rank sends sorted (collection, entity_kind, entity_id, field, layout_generation, read_revision, gradient) contributions for the step; record and relation IDs have separate namespaces. Those gradients are already globally normalized by LossLedger; the owner rejects stale or incompatible revisions and sums repeated contributions in fixed rank/ID order without dividing again, then performs one touched-row Adam update. A nonfinite result on any participating rank skips both dense and sparse updates; all ranks proceed only after the committed revisions are acknowledged.

A small tensor bank is a correctness oracle for record gradients and the specified touched-row updates, not a second production trainer. Sharded row owners are later work if the single owner becomes a measured bottleneck.

The existing [DiLoCo client][m-diloco] synchronizes versioned named model parameters, not an arbitrary dynamic record database. Enable decoupled free-KB training only after defining stable row ownership, staleness, duplicate update handling and compatible model/reader/value publication.

### 14.4 Extend the existing checkpoint format

Use [megatoken checkpoint.py][m-checkpoint] and the existing engine. Port the donor's coherent SQLite backup, content inventory, run ownership and immutable publication patterns from [recipe/state.py][h-state] and [artifacts.py][h-artifacts], rather than introducing its legacy checkpoint helper or a second serializer.

Checkpoint model/realizer/readers/bank, dense optimizer state, free rows and sparse moments, source/key/graph manifests, row/assignment mapping, per-rank data and RNG cursors, boundary schedule and schema/spec/tape identities. After writer training, also record writer parameters and the exact optimized target-bank snapshot.

For runtime restart/replay, serialize actual query tapes, result IDs/revisions, source hashes or bytes, scores/support, scoring epoch and retained support window, deferred/reserved work, committed source-coverage receipts, activation times, binding identities and decay state. Restore coverage separately from version-sensitive MaxSim/score state and replay declared key/index transitions. Record completed retrieval receipts; outstanding work is reissued idempotently or cancelled by policy, not serialized as live futures or connections.

Finish the owned update transaction, create a consistent SQLite backup, flush the staged snapshot and validate its inventory before atomic publication. Pin each session's model, readers, adapter bank and payload layout. A KB-only refresh may be admitted at a latency boundary only when it matches those pinned identities; admission changes the revision and invalidates stale speculation and affected caches. A compatible key update follows section 13.4's score-epoch transition while preserving injected-content coverage. Changing model/reader/bank weights requires a new session or a complete native replay/cache rebuild supported by the new version's contracts.

## 15. Source lifecycle, consolidation and regeneration

### 15.1 Authoring and insertion

Adapt [DurableConsolidator][h-online] and [AuthoredSources/RetrievalDocuments][h-source-bank] into this lifecycle:

1. Resolve immutable original content and its permitted context. Fingerprint source version, bytes/assets, ranges, chunking, semantic-space recipe and writer version.
2. Detect exact repeated ingestion without modifying state. Changed content/alignment requires an explicit replacement operation against the expected source version.
3. In one bounded transaction, register source support, create/update compatible records, retain original contributions and assignments, update source-window counts and advance the authoring cursor.
4. Finalize derived density, graph and index state with resumable fingerprints/cursors.
5. Validate the candidate and publish a compatible snapshot for admission.

The donor's ingest digest does not include the separately passed positions array; add all alignment/context fields to megatoken's identity hash. Cursor progress and record publication must roll back together. Clear authoring mirrors that observed rolled-back writes.

During pretraining, this lifecycle seeds teacher keys and maintains source-backed free records without a learned writer; local optimization adapts their addresses. After writer training, it encodes new sources into accepted learned source-index keys and correction keys/payloads. Experience writing uses the same canonical records and publication contract.

### 15.2 Consolidation is a measured later operation

Reuse the donor's create-new-cluster option, singleton retention, original contribution storage and separate key/value compatibility checks. Its normalized numeric value-distance test is a useful candidate filter, not a semantic contradiction detector.

Require matching scope, time/version policy, origin kind, semantic space and payload/adapter layout before considering a merge. Compare typed payload fields on appropriate scales and check their behavior on representative uses. Keep incompatible near-neighbors separate. Preserve record assignments and measure record count, total storage, reconstruction error and transfer before accepting consolidation.

Do not average LoRA factor matrices; they are global model parameters. Codes in a fixed bank can be combined, but activation/lifetime fields and query-conditioned KV effects still need behavioral validation. For larger or split records, rebuild source-to-output assignments explicitly rather than silently reusing a now-ambiguous slot.

Useful superposition means shared records preserve distinctions and improve transfer or storage efficiency. Fewer rows alone does not establish it. Compare unclustered, clustered and clustered-plus-relations banks at stated payload and compute budgets.

### 15.3 Replacement and deletion

Port source-window decrementing, original-contribution retention and whole-relation removal from the donor. Replacing/deleting a source invalidates affected payloads, relations, dependent experience records, optimizer rows and derived caches.

For freely optimized keys and typed corrections, averaging surviving teacher seeds or original random payloads does not recover the useful adapted record. Mark affected shared records unavailable, then regenerate from surviving sources with an accepted writer, or reset/reoptimize them in a declared training run. Clear stale moments and publish only the validated replacement. If no surviving support exists, delete the complete record.

Recompute affected relation statistics and writer-produced relation payloads from remaining support; do not silently reduce a relation's arity or keep unsupported free drift. Preserve training/tuning provenance for holdout checks. Removing KB support does not remove its past influence on neural parameters. Cancel invalid source reservations and score entries, but retain receipts for bytes already committed. Re-embedding unchanged source content does not create new injection eligibility; genuinely changed content follows its new version/range identity.

### 15.4 Fresh corpus reproduction

Freeze a coherent optimized target bank. Re-encode original sources into a fresh database, reproduce adapted source-index keys and correction keys/payloads, recreate assignments/clusters and deterministic relation statistics, generate learned relation payloads, and build new learned-space indexes. The candidate must not retain hidden links to optimized leaves.

Compare optimized, regenerated, current deployed and no-new-correction behavior on the same held-out tasks. Record source coverage, skipped/incompatible assignments, point/pair payload error and task outcome. Rejection keeps the prior compatible snapshot; accepted regeneration creates a new version and becomes the basis for new-source insertion.

## 16. Detailed training methodology

### 16.1 Trainability and phase handoffs

| Phase | Train | Keep fixed or absent | Completion evidence |
| --- | --- | --- | --- |
| Native bootstrap | Existing E/C/P/R objectives under the native schedule | Correction writer absent; no memory dependency | Useful native representations and normal training/replay behavior. |
| Retrieval introduction | Embedding realizer/P query behavior, correction readers, global LoRA bank and free correction payloads; native base follows pretraining schedule | Teacher keys fixed only for a bounded warm-up; assignments/graph fixed; writer absent | Addressing coverage, smooth local gradients, exact null and bounded effects. |
| Main free-KB pretraining | Native queries, selected source/correction keys, readers, adapter bank and correction payloads; local RAG value and actual correction continuation objectives | No source/experience writer; stable source bytes, IDs and initial topology; coherent index per training window | Bilateral address gradients with teacher losses disabled, useful novel-source retrieval and correction benefit, reproducible sparse key/index updates. |
| Local hindsight writer after pretraining | New source writer and its key/payload readouts | Final native model including query realizer, readers, expert bank, index and optimized targets frozen | Permitted-suffix reconstruction and improvement over an untrained writer, followed by transfer checks. |
| Cross-source writer transfer | Source writer on related/distractor groups | Same fixed interface snapshot | New examples benefit from other-source writer records; same-source dependence falls. |
| Corpus reproduction | Source/document key realizer, full-context correction writer and relation-payload writer | Optimized source-index and point/pair targets, native query/interfaces fixed | Learned-key reproduction, source-assignment coverage and fresh full-index/bank behavior. |
| Acceptance/new-source use | No answer-based optimization | Inference weights and accepted writers fixed | Newly inserted permitted sources improve held-out tasks. |
| Later experience/joint adaptation | Explicitly configured temporary records/writer or P policy | Other interfaces pinned according to the objective | Regenerated-write benefit or correctly scored policy improvement in a new version. |

This maps the donor's free-latent, local-hindsight, group-transfer and stage-5 reconstruction objectives into the user's chosen order. Do not silently run the donor's writer-first stages during pretraining.

### 16.2 Free-KB pretraining losses and data mix

Use the existing native loss as the main objective and add explicit terms:

\[
L_{\rm pre}=L_{\rm native}
+\lambda_eL_{\rm embedding}
+\lambda_rL_{\rm ranking}
+\lambda_vL_{\rm source}
+\lambda_aL_{\rm anchor}
+\lambda_nL_{\rm null}
+\lambda_cL_{\rm correction}
+\lambda_bL_{\rm routing}.
\]

Embedding/ranking losses bootstrap addressing and identify useful eligible evidence; L_source is section 12.7's local similarity-weighted source-feature objective. An explicitly weighted, optional L_anchor stabilizes query/selected-key drift from teacher initialization and can decay after warm-up. Actual correction continuation gradients already belong to L_native and must reach both address sides without adding the same loss twice. Null supervision trains unsupported/no-new-evidence queries and exact no-new-correction behavior. Correction regularity bounds applied effects on valid destinations; routing penalties and reader dropout support useful participation without forcing every record or expert to activate.

Introduce these terms gradually after native bootstrap. Early supervised candidate sets may ensure coverage of eligible positives; retain the distinction between this training assistance and measured ANN recall. Never inject a positive from an unavailable source merely to provide a gradient.

Mix ordinary continuation, supported RAG, other-source transfer, original-document reconstruction where allowed, and verified assistant behavior. Include duplicates, changing candidate counts, hard distractors, conflicts, stale versions, unsupported queries and occasional singleton evidence. Retain ordinary no-memory examples and independently disable KV or LoRA use on a fraction of batches.

Port valid-position penalties from [steering_regularity][h-control]; padding must not dilute them. The donor's variational FreeLatents/KL budget can diagnose reader capacity or excessive private coding. It is optional: a deterministic free payload has no KL term unless an explicit stochastic distribution is introduced. Norm/noise penalties are not a measured bitstream storage bound.

### 16.3 Anti-copy curriculum

Adapt [lexical_recurrence, recurrence_keep and match_recurrence_mask][h-data] to original source spans. Begin with cross-block n-gram families such as the donor's 3/5/8 lengths, group overlapping detections, suppress boilerplate and coordinate dropout across a repeated family.

Increase match dropout only when independent recurrence evidence supports it; high query/key similarity alone is also what useful retrieval requires. Use detached lexical/frozen-feature signals and cap dropout below one. Do not rescale survivors by inverse keep probability: missing evidence is part of the task.

Apply the same dropped evidence to KV and adapter inputs. Compare recurrence-weighted dropout with uniform dropout at the same average rate. Reduce the same-source penalty as cross-source training dominates while retaining general memory dropout. A frozen semantic recurrence detector is an optional extension; the donor's implemented lexical machinery is not already a complete paraphrase detector.

The future-aware recurrence calculation is training-only. It is not an inference dependency or a proof of leakage prevention.

### 16.4 Local source writer

Implement an offline writer contract that accepts ordered source blocks, a permitted footprint and a suffix/full-context mode, and emits source-associated normalized keys plus typed correction payloads. Initialize compatible ingestion/features from the trained native system; writer parameters are separately registered so training them does not change the frozen deployed query/model path.

Use a bounded source-context encoder/readout over actual E source representations. Its output slots have explicit source-chunk or source-window assignments. The 32 latent facets do not supply those assignments automatically. The writer must actually receive every source span needed by its target; an isolated chunk cannot stand in for an unseen contextual suffix.

Borrow directional-view and source-block mask tests from [BlockEncoder][h-encoders] and [PPLXBlockEncoder][h-pplx]. Every mixing/cache path obeys the permitted view. Separate physical suffix views are an acceptable initial offline implementation; changing one attention mask while carrying forbidden recurrent state is not.

First fit source-supported key/payload targets in the frozen coordinates, then include continuation loss through the frozen readers/model. **Freeze parameters while retaining autograd through their computation**; wrapping the steered forward in no_grad would prevent learning the writer. Use the declared local j>=i+2 mask, shared evidence restrictions and later-only correction visibility.

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

Fields f include KV slots, LoRA code, strength and lifetime under the pinned layout. Normalize field losses by their declared dimensions/scales; measure realized lifetime curves and intervention behavior as well as raw parameter error. Targets are detached. Use the original assignment distance; the donor implements distance weighting, not a separately calibrated confidence probability.

Each contributing source output learns the **complete useful record**, not a fraction of a mixture or an averaged target that erases conflicts. Retain incompatible assignments explicitly. Fix slot correspondence before pointwise loss. Use the donor's [matched_loss][h-experience-memory] only where regenerated slots have established correspondence. When layouts differ, adapt evidence_loss to compare retrieved evidence at identical permitted queries and verify downstream behavior; an explicit set-assignment solver would be additional implementation work.

Resolve linked support documents through [referenced_documents][h-knowledge]. Full-context writer training is an actual objective, not an untrained switch from suffix mode at deployment.

For pair payloads, construct relation-writer inputs from current source-writer member outputs and permitted source-window summaries/statistics. Do not feed optimized target vectors back as source inputs. Reproduce every learned pair/member adjustment and lifetime field; deterministic co-occurrence weights can be recomputed. A point-only rebuild that loses useful relation corrections is incomplete.

Add behavioral continuation loss and run the fresh rebuild in section 15. If reconstruction fails, inspect source sufficiency, incompatible consolidation, target coverage and free-payload capacity before adding writer complexity or allowing hidden free-leaf retention.

### 16.6 Loss normalization and efficiency

Extend [LossLedger][m-losses] with preplanned globally reduced counts for query targets, local source-value examples, query-normalized key/teacher anchors, ranking examples, unsupported/no-new-evidence examples, writer assignments, relation targets and correction regularity. Normalize per-query candidate losses internally so changing top-k does not change the unit of supervision. Count writer assignments and valid target fields consistently; report missing coverage separately.

Prepare counts from scheduled supervised events/assignments, not from a stochastic number of successful hits. Empty candidate sets still have well-defined teacher/null losses. Mixed modalities retain the recipient's own loss denominators; no fictitious text tokens are introduced for vector events.

Batch teacher encoding of identical permitted views; cache only under complete content/context/model fingerprints. Reuse differentiable tensors within a loss graph where correct, then discard the graph. Stop interval batching at retrieval, admission, tool/action or control boundaries. Preserve actual historical states and selected controls; do not replay an old prefix under new adapters to make batching convenient.

## 17. Experience learning and difficult-position retrieval

### 17.1 Later experience writer on the same KB

After source writing and the two readers are useful, adapt [ExperienceWriter][h-experience-model], [FreeRecordView][h-free], [MemoryGradients][h-experience-objectives] and [ExperienceJournal][h-experience-journal]. This is a later objective in the same trainer/runtime/storage system.

The cycle is:

1. Collect a real student attempt and its available outcome/feedback. Hidden verifier state, privileged hints and teacher solutions remain teaching signals, not student-writer inputs.
2. Optimize a bounded temporary record set through frozen native readers/model using verified teacher behavior, teacher-memory/evidence targets and optional future rollout reward.
3. Distill the optimized records into the experience writer with matched-record/evidence and verified behavior objectives.
4. Regenerate records and compare no-write, optimized-write and regenerated-write retries plus later task transfer.
5. Publish only accepted regenerated records; retain lineage of direct optimization targets and consulted memory. Rebuild into a shadow candidate at phase boundaries.

The donor currently applies direct RL in free-record fitting and passes with_rl=False in writer updates. Preserve that simple arrangement first; direct writer RL is additional work. Different teacher/student lengths are compared through permitted evidence and actions, not forced token-position alignment.

Progress from deliberate same-task retries to related tasks excluding the current task, then persistent family streams with bounded retention. Include successful attempts and near-null writes to teach restraint. New-source/online uptake evaluations keep inference weights fixed and do not fit free records to test answers.

### 17.2 Use megatoken's actual policy likelihood

Port the experience methodology onto native raw-event tapes and [outcome_policy_loss][m-losses], not the donor's text-token policy probabilities. Reward credit uses the sum of Gaussian log probabilities for selected generated/query events, with independent same-task leave-one-out baselines. Observed tool/evidence events are not sampled policy actions.

Implement a differentiable grouped scoring path over detached recorded raw events and pinned retrieval receipts. The current [replay_group][m-multitrack] is a no-gradient exact-replay verifier that checks stored behavior densities; retain it for verification and add a scorer for optimization. Reconstruct causal history, compute each selected event's Gaussian density and preserve gradients through P and explicitly trainable correction leaves. Verify behavior-density agreement against the same model/KB snapshot before an on-policy update.

For reward-based temporary-record fitting, collect independent same-task rollouts against one fixed trial-record snapshot. Freeze model/readout parameters while retaining autograd from scored Gaussian log probabilities through the KV/LoRA effects on P into temporary record tensors. Recorded raw actions and rewards are detached; trial evidence payloads remain differentiable. Update records only after scoring the rollout group, then collect fresh rollouts. Do not wrap scoring in no_grad, detach trial payloads at retrieval or silently reuse on-policy rollouts after changing trial memory.

Freeze the conditional materialization map, including embedding R, as required by [training/rl.py][m-rl]. Pin retrieval snapshots and record actual deterministic lookup outcomes. A deterministic lookup is not an additional sampled action; an added stochastic selector or writer requires explicit likelihood accounting.

A first outcome-RL experiment can freeze memory/readers/bank and train P/query policy. Training a P-only correction module as part of that policy needs explicit ownership and correct replay; modules that alter frozen C/R behavior cannot silently change under the existing likelihood accounting. Constant reward provides no reward gradient; retain verified teacher supervision rather than normalize it into apparent improvement.

### 17.3 Optional extra boundaries and direct steering

Adapt [policy.py][h-policy] only after fixed/artificial boundary retrieval is established. Generate counterfactual examples from the same causal state with/without an extra retrieval and train predicted benefit minus a measured or declared retrieval cost. A high-loss position alone does not prove retrieval would help.

The decision precedes the next selected event. Invalidate an uncommitted candidate and regenerate after evidence admission; already emitted output cannot be silently repaired. Use held-out calibration, a per-rollout budget and an explicit never-retrieve option.

The donor's direct residual reader is a later control extension. First test whether the shared correction records, KV deltas and LoRA bank already solve the problem. Add a P activation reader in the same key space only if a specific measured gap remains; a separate learned semantic space is not a prerequisite.

## 18. Concrete recipient work packages

All new paths in this table are **proposed**. Port small modules/tests with source commit attribution; avoid a runtime dependency on the donor's native decoder stack.

| Package | Recipient owner or proposed path | Deliverable and dependency |
| --- | --- | --- |
| A: typed data/contracts | New memory/contracts.py and data/retrieval.py; extend data/schema.py and prepared manifests | Learned-space/layout manifests, key-optimization footprints, source-block and coverage identities, query/result/epoch contracts and explicit data conversion. |
| B: vector realization | New generation/embedding.py; extend FacetedGenerator and model/multitrack.py | Registered embedding R, output/train target specs, causal P conditioning, dispatch, losses and canonical vector serialization. Depends on A. |
| C: storage/reference retrieval | New memory/base.py, tensor.py, sqlite.py, index.py, retrieval.py | Adapt exact/SQLite/IVF paths, typed collections, incremental MaxSim, exclusion/refill, transactional key/posting updates and null/conflict handling. Depends on A; usable before learned retrieval. |
| D: source/graph authoring | New memory/authoring.py and relations.py | Stable source-backed seeding, contributions, pair counts, bounded one-hop expansion, independent-support activation, progress and replacement. Depends on C. |
| E: correction computation | New memory/corrections.py and model/adapters.py; extend P attention/read sites and retained-state metadata | Canonical KV binding, per-event independent LoRA mixtures, exact null, learned lifetime and eviction/refresh contracts. Depends on A/B/C. |
| F: training integration | New training/memory.py and memory/optimizer.py; extend engine.py, meridians.py, losses.py, optimizer/distributed/checkpoint paths | Joint native/free-bank pretraining, local source-value and correction-driven bilateral key learning, typed vector objectives and coherent sparse key/payload/index updates. Depends on B–E. |
| G: replay and runtime | Extend multitrack_runtime.py, multitrack_service.py, training/history.py and grouped tapes | Native query/evidence chronology, MaxSim epochs and deferred/reserved/injected coverage receipts, correction state, grouped training history, separate differentiable scoring and stale-branch invalidation. Develop with F. |
| H: post-pretraining writers | New training/hindsight.py, training/reproduction.py and registered source writer modules | Local suffix training, cross-source transfer, learned source-key and full-target point/pair reproduction, and fresh-index/bank acceptance. Depends on a completed F/G snapshot. |
| I: later experience/policy | New training/experience.py and experience source facade in memory/authoring.py; extend environments.py/rl.py | Trial-record fitting, accepted experience writes, optional benefit boundaries and native-likelihood reward training. Depends on H and task evaluation. |

The immediate integration targets already exist: [prepare_meridian_batch/forward_meridian_batch][m-meridians], [MultiTrackGenerator][m-multitrack], [LossLedger][m-losses], [MasterAdamW][m-optimizer], [checkpoint.py][m-checkpoint] and [MultiTrackRuntime][m-runtime].

Serving supports GroupActionTape, but [NativeHistory][m-history] currently consumes singleton tapes and grouped preparation rejects singleton tapes used to certify a multi-event meridian. Package G must close that gap; do not reinterpret an E encoding as the selected raw event or assume a serving serializer alone supplies training replay.

Reference PyTorch operations come first. Reuse donor numerical oracles for matching, gates and overlays; adapt kernels only after the native row layouts and gradients are stable. HKB's shared-basis adapter kernel does not implement the independent LoRA-bank calculation.

## 19. Acceptance tests, experiments and resource budgets

### 19.1 Tests to port and extend

These are gates for future implementation, not claims that this documentation change ran or passed the donor suites.

| Donor reference | Preserve | Add for megatoken |
| --- | --- | --- |
| [test_storage.py][h-test-storage] | Finite/dimension checks, scope integrity, empty results, rollback, reopen, coherent snapshots and exact parity | Space/layout mismatch rejection, typed collections, offset-sensitive idempotence, key/posting/moment atomicity and split generation invalidation. |
| [test_ivf_batch_search.py][h-test-ivf] | Exact rescoring within probed cells, bounded decoding, sparse IDs | Full eligibility/coverage exclusion before truncation, bounded refill/shortfall, and moved-key posting parity against exact scores in probed cells. |
| [test_authoring_cache.py][h-test-authoring-cache], [test_authoring_scale.py][h-test-authoring-scale], [test_authoring_source_batch.py][h-test-authoring-batch] | Sequential assignment, mmap/cache bounds and cursor/record rollback | Fixed random-bank assignments, complete context fingerprints and typed source replacement. |
| [test_relation_build.py][h-test-relations] | Stable IDs, support/degree caps, deterministic ties, resumable atomic publication | Pair-only baseline, support-footprint exclusion, one-hop/no-recursion and optimizer reset after rebuilding. |
| [test_retrieval.py][h-test-retrieval] | Masked current candidates contribute no direct lookup gradient, exact null, conflict slots, density compensation, distinct-span relation gates | One query cannot activate a pair; repeated same-support queries do not manufacture evidence. |
| [test_training.py][h-test-training] | Reader/memory gradients with frozen owners, cross-source/scope masks, full-target point/relation distillation | Source i+2 invariant under query/evidence insertion, raw-facet writer mappings and typed lifetime/code targets. |
| [test_sparse_candidates.py][h-test-sparse] | Repeated IDs share a differentiable parameter and accumulate gradients | Bilateral key/value gradients, collection-qualified touched-row parity, denominator/rank equivalence, finite failure handling and key/payload moment resume/reset. |
| [test_serving_replay.py][h-test-replay] | Target-free replay, boundary scheduling and no-match behavior | Native grouped tapes, source coverage and deferred/reserved receipts, delayed arrivals, key/score epochs, stable destination eviction and selected-raw identity. |
| [test_learning.py][h-test-experience], [test_provenance_regressions.py][h-test-provenance] | Trial/writer gradient boundaries, matched targets, rejected-promotion retention, persistent lineage and resume | Native Gaussian event likelihood, source-footprint/task exclusions and fixed-weight new-source uptake. |

Add recipient-specific gradient and ownership checks: registered embedding/readers/bank parameters occur exactly once; vector training does not demand a vector ingester; target gradients cannot enter P through teacher inputs; correction views leave base KV immutable; LoRA event scopes survive flattened/permuted grouped dispatch.

**Exact-null parity uses the same selected raw-event history, query schedule and prior correction/decay state.** Compare absent new candidates with candidates whose new activation is forced to null, while existing corrections evolve identically. A native query event itself changes C history and consumes compute even when lookup returns nothing. Separately compare task quality with the ordinary no-query baseline and a compute-matched null-query control.

Add focused MaxSim and learned-key gates:

1. **Incremental score parity:** one-query/one-key equals cosine; sequential maxima equal the batched reference for identical evaluated pairs and epochs. Distinguish approximate ANN discovery from exact rescoring.
2. **Novel content only:** repeated high matches, duplicate aliases and concurrent overlapping requests commit each source interval once. Sequence admission marks only its committed prefix; key-only changes and KV eviction do not erase coverage.
3. **Deferred work:** a qualifying source survives budget deferral and idempotent fetch retry without a new score record; stale/cancelled work releases reservations. Low new scores or graph bonuses cannot independently fetch unqualified content.
4. **Bilateral local gradients:** with teacher/anchor losses disabled, controlled value-dependent examples give finite nonzero gradients to P/embedding R and selected stored keys in both collections. Source-feature values/targets and masked current candidates receive no direct auxiliary/lookup gradients; repeated IDs update once. Vary useful values/targets to verify that key updates follow usefulness, not similarity alone. Covered/reserved useful support is masked without repulsion; only remaining support forms the target, with null-only loss for completely annotated empty remainder. Records and relations sharing a numeric ID/field retain distinct leaves and moments.
5. **Key/index and score epochs:** force a key across an IVF cell boundary; verify atomic key/posting/moment updates and rollback. Key changes that lower a cached maximum or change its winning old query require recomputation/reset. Identical-vector index refresh preserves content coverage; payload-only correction updates preserve source scores.
6. **Learning and regeneration:** local learning improves held-out useful-support retrieval and actual RAG/correction behavior; regenerated/new-source keys work in the frozen learned space without per-test-answer optimization. Replay reconstructs the same admitted sources and epoch transitions.

### 19.2 Evaluation matrix

Run no-query, null-query control, source-RAG only, KV only, LoRA only, KV+LoRA and all-three conditions. Ablate graph candidate expansion separately from joint pair payloads; report their actual activation rates. Compare uniform/recurrence dropout, fixed/sparse routing and independent reader dropout at matched budgets. Compare teacher-frozen, query-only and joint query/stored-key learning; ablate the local source-value objective separately from ranking/anchors. Compare ordinary repeated RAG with incremental MaxSim/coverage at matched query, byte and latency budgets.

For writers, keep distinct reports for local hindsight initialization, cross-source transfer, optimized-bank behavior, regenerated-bank behavior and new-source insertion. Include source/assignment coverage and record/pair reconstruction errors. Low local reconstruction loss is not the same result as useful deployed memory.

For consolidation, report quality against both record count and **total retained bytes**, including original contributions. Test singletons, near-key contradictions, duplicates, changed rules, version conflicts and deletion. Demonstrate novel combinations of separate knowledge where that is an intended capability.

Measure teacher/query embedding recall, eligible novel-support recall, duplicate bytes avoided, deferred/fetch failure behavior, ANN recall against current learned keys, local source-value error, downstream native loss, actual task success, no-memory regression, correction lifetime/reset behavior, writer/key-regeneration throughput, key/posting update cost, cold/warm and tail boundary latency, peak working set and end-to-end task time. The donor's ANN benchmark is a component reference; include query realization, fetch, ingestion and correction admission in the new serving measurements.

Development data used to choose rebuild or trigger thresholds is tuning data. Release and new-source tests remain untouched by base pretraining, free-leaf optimization, writer fitting and policy calibration under their declared exclusion rules.

### 19.3 Configuration and capacity accounting

Add a versioned retrieval/memory configuration covering learned semantic spaces, typed layouts, MaxSim threshold/scoring epochs, candidate/deferred/reservation/coverage budgets, overlap and retry policy, relation windows/degree/support, query frequency/deadlines, key warm-up and per-field optimizer rates, local feature/anchor losses, index-posting/centroid refresh, KV slot/binding limits, LoRA budgets, dropout, writer views and rebuild acceptance. Calibrate novelty thresholds in the accepted learned coordinates. Bound training feature caches by E revision and keep lifetime source coverage durable even when hot state is compacted.

Use one vector per event and pairs only initially. The donor's candidate count around 32, up to four incompatible slots and low-rank settings such as rank eight are starting reference values, not measured megatoken optima. Choose KV template count, code dimension and bank coverage against an explicit storage/compute budget.

For N_s source records with K_s key coordinates and N_c correction records with K_c key plus P FP32 payload coordinates, base vectors require approximately 4[N_s K_s + N_c(K_c+P)] bytes. Once rows have been touched, FP32 Adam moments can add 8[N_s K_s + N_c(K_c+P)] bytes. Retained teacher anchors add up to 4[N_s K_s + N_c K_c] bytes unless deduplicated through an existing immutable vector store. Count source features/bytes, original contributions, relation optimizer state, indexes, coverage/queue metadata, snapshots and resident caches separately. Sparse GPU access does not eliminate disk optimizer-state cost.

For independent LoRA factors, count parameters as the sum over target groups of E_g r_g(d_in,g+d_out,g), plus any intended packed-base-expert multiplicity. Include their dense optimizer state separately. Do not measure the memory system only by the size of its final consolidated record table.

Query count, latent facet count, destination count and KB record count are separate budgets. Fewer megatokens reduce the number of query/destination opportunities; they do not require shrinking the external corpus or assigning only one fact to each native facet.

### 19.4 Milestone decisions

1. **Contracts/reference storage:** deterministic typed round trips, complete-footprint masks, exact/IVF reference comparisons and source lifecycle.
2. **Native retrieval event:** teacher-initialized query realization, incremental MaxSim, correct event/clock ownership, novel-only ordered source admission and exact replay.
3. **Free correction pretraining pilot:** local losses train native queries and selected source/correction keys; both readers receive useful gradients, null/bounds hold, sparse key/index updates resume coherently, and actual task benefit survives the null-query control.
4. **Corpus-scale pretraining:** useful retrieval under latency/storage budgets, duplicate/conflict robustness and agreed synchronous multi-rank correctness.
5. **Post-pretraining writing:** local-to-cross-source transfer followed by a fresh bank whose behavior approaches the optimized target.
6. **New-source/experience expansion:** accepted source-written memories improve tasks with fixed inference weights; later experience and adaptive-boundary work passes its own task and replay gates.

Choose numerical promotion tolerances in the experiment recipe before evaluating the corresponding holdout. A failed writer rebuild calls for source/assignment/capacity diagnosis, not an automatic increase in machinery or silent retention of optimized leaves.

**The decisive result is that source-written corrections improve new tasks after pretraining, while ordinary retrieved evidence remains useful on its own.** The free KB establishes what memory can learn; the regenerated KB establishes what the system can add and maintain.

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
[m-schema]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/data/schema.py
[m-dedup]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/data/dedup.py
[m-data-storage]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/data/storage.py
[m-meridians]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/training/meridians.py
[m-multitrack]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/model/multitrack.py
[m-attention]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/model/attention.py
[m-experts]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/model/experts.py
[m-optimizer]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/training/optimizer.py
[m-diloco]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/training/diloco_client.py
[m-checkpoint]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/training/checkpoint.py
[m-losses]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/training/losses.py
[m-rl]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/training/rl.py
[m-runtime]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/multitrack_runtime.py
[m-history]: https://github.com/werg/megatoken/blob/b89b10726e395364baeecffa5ead1703b1300553/src/megatoken/training/history.py
[h-experience-memory]: https://github.com/werg/hindsight-kb/blob/450164784589b2576256b8b0ba3d2ccc89af992d/src/hindsight_kb/experience/memory.py
