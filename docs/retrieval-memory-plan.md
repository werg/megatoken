# Retrieval and correction memory: architecture plan

**Status:** Proposed extension to the v7 prototype; implementation and learned quality remain to be established.
**Date:** 2026-10-19.
**Reviewed baseline:** [`a441316f19ea6409ff39f4502bc56068f0d06ed2`](https://github.com/werg/megatoken/commit/a441316f19ea6409ff39f4502bc56068f0d06ed2).

## 1. Decision

Use **true RAG on a retrieval track for all retrieved source content**. Separately, train a **free correction KB containing KV deltas and LoRA mixture controls during pretraining**. Only after this pretraining concludes, train the machinery that writes and reconstructs the correction KB from sources.

Both paths use one boundary scheduler and a compatible semantic retrieval interface. They have distinct payloads:

| Path | Stored payload | How it enters megatoken |
| --- | --- | --- |
| Source RAG | Original source chunks, ordered spans and provenance | Ordinary observations on a retrieval track, through E and C |
| Correction KB | Learned KV corrections, LoRA mixture codes and lifetimes | Bounded interventions in subsequent computation |

This replaces the earlier proposal to import complete source KV sequences into the live cache. Ordered retrieval now uses source chunks and normal ingestion. KV **corrections** remain part of the design.

The [multi-track architecture](multi-track-plan.md) already defines shared history, categorical tracks and atomic meridians. Its [implementation status](multi-track-implementation.md) identifies the trained retrieval projection, retrieval service and recurrent/distributed multi-track training integration as remaining work.

## 2. One query interface, two payload readers

At a latency boundary, a small retrieval head emits a normalized vector from a **causal read of the main request and committed shared history**. Attach it to a main-request-conditioned, read-only predictor block; the retrieval track owns lookup requests and returned evidence. Start with one deterministic query per boundary, without a generated textual search query or an extra recurrent model.

Do not assume `state.final` is the main track: it contains the latest meridian, which may contain only tool or evidence events. Condition the query on the main request explicitly and use the existing historical attention path.

Use a single ANN service with typed source and correction collections in the chosen embedding space. A query may search either or both with independent result budgets. Semantic keys locate useful records; native attention keys govern their later use.

Source hits return exact bytes plus source identity, version, span and document order. Ingest them with document/tool roles on a reusable retrieval track. A single track contributes at most one event per meridian, so ordered chunks arrive through successive meridians. A small fixed set of retrieval tracks can admit independent documents concurrently. Source IDs belong in evidence metadata, not in newly allocated track names.

All evidence uses normal committed-meridian RoPE and the shared history budget. Source order is retained through ingestion order and span metadata. Track identity is an attention relation, not an isolation boundary. There is no additional source KV cache or special positional scheme.

## 3. Supervise semantic addressing first

Precompute keys for megatoken's actual chunks with a frozen embedding teacher. Warm up the retrieval query head using both teacher similarity distributions and labeled useful evidence. Keep source keys fixed during the first pretraining run; train correction values freely. This avoids simultaneously moving the index, reader and writer targets.

Select one teacher space before bulk indexing, after a small comparison:

| Candidate, checked 2026-10-19 | Role and interface |
| --- | --- |
| [Perplexity `pplx-embed-v2-context-9b-preview`](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview) | Preferred contextual-chunk pilot: one vector per chunk; distinct `encode_queries` and document `encode`; 2048 dimensions or the trained 1024-dimensional truncation. Pin weights and preprocessing because preview revisions may break compatibility. |
| [Perplexity contextual v1](https://docs.perplexity.ai/docs/embeddings/contextualized-embeddings) | Established hosted alternative: ordered document chunks with `pplx-embed-context-v1-4b`; its documented query recipe uses the same contextual model with `[[query]]`. |
| [EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) | Smaller local comparison and a route to multimodal retrieval; use its distinct query-task and document formats. |

Normalize vectors consistently and pin the model, dimension, query/document formatting and chunk context policy. Pin weight revisions for local teachers; for hosted teachers, retain returned vectors with model identifiers, request settings and input hashes. Do not mix unrelated embedding spaces. The newer [Perplexity late-interaction family](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-9b) produces multiple token vectors; it can be evaluated separately as a ranking teacher without making the first index multi-vector.

The query teacher sees only the available request/prefix. The evidence teacher may see the permitted source context. Seed correction records intended for local hindsight with keys computed from bounded source windows that a later suffix writer can see; reserve broader-context records for corpus reproduction. In continuation and hindsight tasks, every chunk that influenced a contextual embedding is part of its visibility footprint; filtering only the returned chunk is insufficient.

Train for **useful evidence**, not merely resemblance to the prefix. Use multiple positives, hard distractors and no-answer examples. Do not automatically label every sibling chunk or unannotated result negative. ANN/top-k selection is discrete: ranking supervision trains the query head, while downstream generation losses train the selected readers and values.

## 4. Free correction KB during pretraining

After the existing ingestion/bootstrap warm-up, introduce correction memory within the main pretraining curriculum. The base model, correction readers, adapter bank and KB values learn together. **There is no learned KB writer in this phase.**

Seed records from source chunks, compatible source groups and retained corpus relations. Every record has:

- A supervised semantic key and stable identity.
- Contributing source spans, scope, version and compatible-group assignments.
- A small, randomly initialized trainable payload: canonical KV delta slots, LoRA mixture code, strength and lifetime.
- Sparse relation links with their support and provenance.

“Free” means the payload is an independent optimized variable rather than a writer output. Source IDs are bookkeeping, not a shortcut feature for the query model.

Reuse records across related examples and train with source-excluded retrieval, irrelevant records, conflicts and missing memory. Keep payload capacity, intervention magnitude and the active working set bounded; retain ordinary no-memory examples and occasionally disable one correction path. These measures encourage reusable corrections without introducing writer training early.

The important uncertainty is **source reproducibility**: a free bank can become useful while encoding information its later writer cannot infer. Source grounding and reuse reduce that risk but do not prove it away. Acceptance therefore depends on a regenerated bank and new-source tests in section 8.

### KV corrections

Store a small fixed number of canonical, unrotated delta slots per record. A reader binds retrieved slots to selected retained event/facet addresses, using the current query and destination state.

For a destination at meridian position `p`:

```text
K_read(p) = K_base_rotated(p) + a * RoPE(p, delta_K)
V_read(p) = V_base(p)         + a * delta_V
```

Begin with temporary overlays on the main track's P history reads. Use a query-scoped view or equivalent grouped attention so peer-track queries continue reading base K/V. The shared base KV remains immutable; C still publishes its normal final KV exactly once. Values are never rotated. Bind by stable event/facet identity rather than a shifting cache-array offset, and expire an overlay when its destination is evicted.

Refresh replaces or explicitly blends an existing overlay. Repeated retrieval never silently accumulates the same delta in place. Activation cannot change earlier outputs or cached historical computation.

### Mixtures of LoRA experts

Train a large shared bank of independently initialized low-rank experts. Each expert owns its factors for configured parameter groups; KB records store mixture controls, not a fresh complete adapter per source:

```text
W_effective(t) = W + sum_e c_e(t) * B_e * A_e
```

Start with P-owned query/output and pair/feed-forward/expert projections. Exclude the C-owned shared `central.memory` KV projector. Apply mixtures as per-event residual computations, so a main-track correction does not mutate weights shared by every track. Allow lightweight stage/facet gates inside the interval. Extend coverage to C/R only if measured gains justify their additional cache and materialization contracts.

Use small nonzero initial adapter outputs, soft balanced routing for warm-up, then sparse selection. Bound the total active experts across records. Keep expert identity, signed mixture coefficient, intervention strength and lifetime distinct, including an exact null intervention.

For both correction paths, use a learned gate and decay, for example `a = gate * exp(-softplus(rate) * age)`. Measure age in committed progress of the affected track, not predictor effort or the number of admitted evidence chunks. Refresh/reset handles superseding evidence. Decay stops future direct effects; it cannot undo effects already propagated into the trajectory.

## 5. Sparse relation-reinforced retrieval

For the initial design, **second order means pairwise corpus relations/correlations**. Precompute sparse associations from bounded source windows, explicit references and deduplicated co-occurrences. Keep support counts, cap degree and normalize for common hubs. Preserve the distinction between an explicit relation and a statistical association; correlation is not a factual or causal assertion.

Retrieve semantic seeds, expand their eligible neighbors once, and add a bounded relation bonus from the **original** seed scores. Then select a capped set of records. Apply source/scope/time masks to complete key, relation-support and payload-source footprints before expansion and scoring; boosted results do not recursively reinforce themselves. This avoids a dense all-pairs KB and an iterative graph reasoner.

Each retained relation may have its own small free correction payload, backed by its contributing sources. One seed can propose a neighbor, but a pair-specific correction requires joint applicability or a learned compatibility gate; one matching endpoint does not automatically activate it. Keep graph topology and statistical weights deterministic in the first version so corpus reproduction can recompute them. Two-hop propagation or learned edge strengths are later comparisons, not prerequisites.

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

1. At a completed user/tool turn or other declared boundary, snapshot the causal state and issue the vector query.
2. Fetch source evidence and correction candidates within explicit budgets.
3. Admit available evidence as observed meridians; activate selected corrections for the next main prediction.
4. Continue until the next boundary. On no match, activate nothing new and let existing corrections decay.

For long uninterrupted runs, insert artificial boundaries after a configurable amount of committed main-track progress, with a small range of training intervals. These are scheduling opportunities between complete meridians, not empty events or extra RoPE steps. Evidence admission alone must not trigger a recursive retrieval loop.

Use bounded waiting and the same arrival/deadline policy in training and serving. Late results wait for the next valid admission boundary; never grant them to a prediction made before their availability. If speculation is used, evidence or correction admission changes the session revision and invalidates candidates from the old state. Query execution itself is read-only and creates no history event.

Record query/model/index identities, source results, activation times, correction selections, gates and decay state for replay. Simulate misses and realistic delayed arrivals during training. Bound query frequency, retrieved bytes/events, graph expansion, active overlays and adapter transfers, and measure actual end-to-end latency.

## 8. Post-pretraining writer and corpus reproduction

| Phase | Work and completion evidence |
| --- | --- |
| Bootstrap | Establish useful native ingestion and grouped training; no correction writer |
| Pretraining with free KB | Train the base, query head, both correction readers, expert bank and free payloads, together with RAG tasks |
| Local hindsight writer | Freeze a coherent final model/query/readers/adapter-bank/KB snapshot; train a suffix encoder to reproduce source-supported keys and payloads in those fixed coordinates |
| Corpus reproduction | Train source/corpus writing against the optimized KB, including compatible groups and relation payloads; regenerate keys, records and deterministic relations from retained sources |
| New-source use | Insert memories from sources held out of both pretraining and writer fitting, and from completed feedback episodes, with inference weights fixed; evaluate useful transfer without free optimization on their test answers |

For local hindsight, the prefix queries while the writer sees only its permitted suffix. Preserve the [hindsight-kb eligibility rule](https://github.com/werg/hindsight-kb/blob/6dfc779a51759549f96743318a4f9e72a8975210/docs/research_plan.md): at block `i`, same-source writer records begin at `j >= i + 2`. Apply it to all contributing source context. This future-suffix access is a training-only exception for correction-writer supervision; source RAG and deployment retain causal visibility. Supervise only records whose content is inferable from the writer's allowed view; relations needing broader evidence belong in corpus reproduction.

Fit keys/payloads and also measure behavior through the frozen readers. Build a fresh KB from writer outputs and compare it with the optimized bank. Latent reconstruction loss alone is insufficient. If reconstruction fails, inspect source context, consolidation and free-value capacity before increasing writer complexity. Any later joint adaptation starts from an accepted regenerated bank and creates a new versioned snapshot.

## 9. Implementation and acceptance

Implement in this order:

1. Extend existing Episode/grouped-training and runtime contracts with retrieval requests, evidence provenance, eligibility and artificial boundaries; add the shared semantic head and one real index adapter.
2. Add free record storage, sparse relation expansion, P-read KV overlays and per-event LoRA mixtures; train and compare both paths.
3. Integrate them with recurrent/distributed pretraining and exact checkpoint/cache reconstruction.
4. Add the post-pretraining local writer, corpus reproduction and new-source insertion evaluation.

Register trainable heads and adapters on the underlying `FacetedCore`/`FacetedGenerator` owners, not only the plain multi-track wrappers. Version KB values and optimizer state with model/readers/adapter banks. Rebuild derived caches after weight changes. When extending outcome RL, preserve the complete frozen conditional action map or explicitly score any additional stochastic correction actions.

Run controlled comparisons for no retrieval, RAG only, KV only, LoRA only, KV+LoRA, and all three; ablate relation reinforcement separately. Report evidence recall, downstream task quality, no-memory regression, correction lifetime behavior, new-source uptake, regenerated-bank quality, replay fidelity and serving cost.

**The decisive result is that source-written corrections improve new tasks after pretraining, while ordinary retrieved evidence remains useful on its own.** The free KB establishes what memory can learn; the regenerated KB establishes what the system can add and maintain.
