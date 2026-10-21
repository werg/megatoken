# Continuous State Transducer: a search-guided knowledge base

**Status:** Architecture proposal, not implemented.  
**Date:** 21 October 2026.  
**Based on:** [Megatoken v9 at 4d24d05](https://github.com/werg/megatoken/commit/4d24d05dbf4e55d5289e1c2d4727f854d66fddf1). See the [current retrieval design](retrieval-memory-plan.md) and [implementation status](retrieval-memory-implementation-status.md).

## 1. The change in one page

Replace the current knowledge base of learned points and pair correlations with a **continuous state transducer**: a searchable graph of transitions between positions in a learned state space.

An edge says: **when searching for this kind of information from this part of the search space, return this knowledge and move the search state in this direction.**

Each edge stores four vectors:

| Vector | Purpose |
| --- | --- |
| **Departure state** | Where this transition is useful |
| **Key** | What question or search intent activates it |
| **Value** | What knowledge the transition returns |
| **State update** | How the search position changes after using it |

The model maintains a **search agenda state** alongside its usual transformer history. It starts at a position informed by an initial metadata prefix. At every eligible step, the model asks the KB a new question. The KB blends nearby edges according to both their **key-to-query match** and **departure-state-to-current-state match**, then returns a blended value and a blended state update using **the same weights**.

The wide blended value is reduced to **32 native facets** using the existing mixture-of-experts (MoE) machinery, then fed back as a new megatoken on a dedicated **knowledge retrieval track**. The normal central transformer processes it against earlier queries, retrieved knowledge and generated output. This allows the model to adapt what it learns from a retrieval rather than treating the KB answer as fixed attention cache content.

**The objective is search, not correlation:** generation becomes a sequence of questions and guided moves through a fuzzy, learned graph. Source facts still need provenance, but source co-occurrence is only evidence for *proposing* an edge, not a special correlation mechanism used at query time.

### What stays and what goes

Keep ordinary E/C/P/R processing, the shared multitrack history, native KV caching for committed megatokens, raw-facet query generation, source provenance, exact-source RAG, indexed lookup, sparse KB optimization, and the current MoE pools.

Remove the **semantic KB's** point/pair correlation scoring, learned directional pair-dependence, correlation-based attenuation, direct stored-value-to-KV projection, and KB-specific semantic LoRA control as the retrieval answer path. **Ordinary transformer KV caching stays**: the *new knowledge megatoken* writes KV through normal central processing. Existing v9 training remains the supported implementation until a new, explicitly versioned architecture passes acceptance tests.

## 2. How a retrieval works

Let the current agenda state be \(s_t\), the model's search query be \(q_t\), and edge \(i\) store departure \(d_i\), key \(k_i\), wide value \(v_i\), and state residual \(r_i\).

For a small, causally eligible set of nearby edges, calculate:

\[
a_i=\exp\left(
\frac{\cos(q_t,k_i)}{T_q}+
\frac{\cos(s_t,d_i)}{T_s}+b_i
\right),\qquad
w_i=\frac{a_i}{a_{\varnothing}+\sum_j a_j}
\]

Here \(b_i\) is a bounded learned strength/prior; \(T_q,T_s>0\) control how sharply matching works; \(a_{\varnothing}>0\) is an explicit **no useful transition** option. Scope, source version, time and full provenance are checked **before** scoring. Normalize key and departure vectors independently. In the real reader, calibrate the null score and optionally reject candidates below minimum similarity thresholds: a small ANN shortlist must not make weak matches look certain.

Return:

\[
\bar v_t=\sum_i w_i v_i,\qquad
\overline{\Delta s}_t=\sum_i w_i r_i,\qquad
s_{t+1}=\operatorname{retract}(s_t+\eta\overline{\Delta s}_t).
\]

**The weights must be identical** for values and residuals. The retraction (for example, bounded norm followed by normalization) and a capped step size prevent runaway state drift. When null wins, do not inject a fake knowledge megatoken or move the state. Keep the query event and its cost visible in the record. A low-confidence mixture can instead trigger a narrower question or a new search branch.

This simple form is the **reference algorithm**, not a proven optimum. Compare additive versus multiplicative match gates, temperatures, null calibration and per-edge uncertainty in experiments. Do not let a strong query-key match override a clearly wrong departure state, or vice versa.

### Find candidates without searching the whole graph

A single index on keys is not enough: the best joint match might be absent from its key-only top-k. Use a **two-index shortlist** (query-key neighbors **union** state-departure neighbors), plus a small quota for nearby directed graph edges. Check eligibility first and score that union exactly with both similarities. Measure recall against an exhaustive reference; later replace with a joint index if useful.

Group only mutually compatible source versions and meanings in one mixture. Do not average contradictory claims into an invented third claim. When directions disagree strongly, produce separate candidate branches or ask a disambiguating query rather than letting residuals cancel invisibly. Keep the originating edge IDs and source references for attribution.

## 3. Put the answer back through the transformer

At each completed **ordinary** model position:

1. Produce and commit a normal selected retrieval-query event through P, its embedding realizer, and C.
2. Search the transducer using that query **and the current agenda state**.
3. Mix values and state residuals. On a useful result, project the value to a raw \([32,D]\) block and commit it through **C on the knowledge retrieval track**.
4. Atomically publish the corresponding agenda-state update and retrieval receipt.
5. Continue generation. Its next query can attend to **all previous retrieval and generation history**.

At production geometry, \(D=512\), so the knowledge input is **32 × 512**. It is an **observed retrieval result**, not a sampled assistant answer and not an external KV slot. A native history entry is written only if C successfully commits the result.

Do **not** automatically query again just because the internal query event or its knowledge-result event was committed; that would recurse forever. Query at every eligible **ordinary generation/observation position**, with explicit caps on work in flight, repeated queries, branch depth and latency. The first retrieval is a required break; later searches may complete asynchronously and become visible only at a valid later boundary. Late results never rewrite already generated text or prior history. Parallel tracks share the same state only under an explicit deterministic commit order; independent search branches receive their own agenda states until merged deliberately.

A query, its selected raw facets, candidate IDs, weights, old/new state, source eligibility, model/index versions, and knowledge result must be recorded for exact training and replay. A failed lookup must not partially advance the agenda.

### Wide values, narrow output

The proposal deliberately makes values **wider than the final facet width**. Mixing information in a broad space gives different aspects room to coexist before being compressed into 32 facets. Start by testing **8,192, 16,384 and 32,768** value dimensions; use 32,768 as the wide pilot, not as an unexamined permanent default.

**Mix the wide values first.** Then split or adapt the mixture into 512-wide working blocks, route those blocks through the **existing U MoE expert bank**, and use 32 learned facet queries to gather the result into the native 32 × 512 block. The current U experts accept 512-wide rows; they cannot directly process a 32,768-wide row. A small input/output adapter and a bounded routing plan are therefore required. Test sharing versus a separate projection expert pool to watch for interference with text realization. The learned projection may use the query and current state as *routing context*, but must not receive future target information.

This is a hypothesis about interference at the **mixture** stage, not proof that bigger values preserve more knowledge. Compare with the same parameter/storage budget at smaller widths and with a plain linear projection.

## 4. Metadata first: put the first pause in the right place

Prepare each training episode with a **compact, causally available metadata prefix**. Include as much useful information as is honestly known at that point: task and retrieval purpose, title/topic, source type, language, modality, time and version, relevant document/repository context, permitted scope, and the user's request. Mark missing fields explicitly. Keep source family, hashes and permissions for data integrity, but do not expose hidden benchmark answers or arbitrary internal IDs as if they were meaningful facts.

Use that prefix to initialize the search state:
\[
s_0=\operatorname{retract}(s_{\mathrm{base}}+G_{\mathrm{meta}}(\text{visible metadata, request})).
\]

Encode the **first metadata megatoken** through the ordinary input/central path, then schedule the **first query and retrieval/latency break immediately after it**, before ordinary answer generation. Longer metadata can arrive through later causal events; avoid waiting to consume an entire long document before asking anything. A missing-metadata case uses a learned neutral initial state and still gets its first search opportunity.

Training preparation must separate **model-visible metadata** from labels used only for supervision, and reproduce the same first-boundary schedule at inference. Do not move answer-bearing future annotations into the prefix to improve apparent retrieval quality.

## 5. Build an initial graph efficiently

Avoid starting with millions of random state positions. Prewarm the graph in stages:

1. **Cluster source concepts and questions.** Reuse the existing source chunking and semantic embeddings to group related facts, search intents and evidence types. Keep distinct source versions and conflicting claims separate; preserve singleton/tail clusters.
2. **Generate plausible search paths.** Use real multi-step question/evidence trajectories where available. Add controlled synthetic search tasks to cover missing joins, verification and backtracking. Record which paths are synthetic. Consecutive source chunks alone do **not** prove a useful search transition.
3. **Learn state anchors and directed edges.** Give common agenda situations coarse anchor positions. For each demonstrated query step, form a departure state, a question key, a source-backed wide value and a residual aimed toward its next useful agenda state. Distill these from the path and test whether the weighted reader can actually follow it.
4. **Count transition bigrams.** Identify frequently useful **adjacent pairs of transitions**, not merely word bigrams or pairs of neighboring documents. Common two-step sequences become candidates for better placement and training exposure.
5. **Find patterns of patterns.** Cluster recurring longer paths and small directed subgraphs, then find recurring combinations of those motifs. Examples include *identify entity → retrieve relation → check evidence* and *propose answer → find contradiction → correct answer*. Use these higher-level patterns as navigation hints and optional shortcut-edge **proposals**, not as automatic hardwired truths.
6. **Place and publish a small supported seed bank.** Allocate edges around the strongest anchors and high-demand routes, reserve capacity for rare paths, seed values from available source representations, and validate on held-out paths before scaling.

This gives three levels of useful structure: **local edges, transition bigrams, and reusable motifs of motifs**. Graph hierarchy helps propose where to search; the live query/state scoring rule remains the authority for what is retrieved. Stop adding hierarchy when it increases complexity without improving held-out path completion or latency.

**Important:** state-space coordinates are learned *search* coordinates, not generic source-embedding coordinates. Source clustering is a scaffold. Anchor positions and residuals must be refined against actual model-generated queries and search trajectories. A changed state encoder requires coordinated edge reindexing; old coordinates are not silently reusable.

## 6. Grow and improve edges during pretraining

Collect a small, durable set of statistics for each state region and edge: eligible query demand, null results, successful transitions, downstream improvement, repeated visits/loops, competing candidates, source support, value use, age, latency and how often a rarely used edge actually gets gradient updates. Deduplicate by query event and source family so repeated data cannot manufacture demand.

At **bounded optimizer barriers**, the single KB owner should:

- **Propose** edges in regions with many supported searches but poor hit or path-completion rates, using new source evidence, useful query→next-query transitions and higher-level motifs.
- **Admit** only source-supported, causally eligible proposals with a versioned anchor, key, value, residual and initialization seed. Give new edges a modest exposure quota; do not initially let them dominate answers.
- **Improve** selected keys, departures, values and residuals from later task outcomes; preserve existing learned edges and their optimizer moments.
- **Merge or retire** redundant edges only after checking that source attribution, alternative paths and behavior survive. Deleting a source invalidates all dependent edges and cached results.
- **Reindex and publish** changed addresses, graph adjacency, sparse moments and growth cursor together, then expose a coherent snapshot to learners.

The existing SQLite owner, IVF infrastructure and sparse-row optimizer are a foundation, but they need new **edge tables, state-address postings, state revisions and field layouts**. During DiLoCo training, the coordinator must allocate IDs and publish coordinate/edge changes; learners cannot independently invent incompatible graph states. A state-space re-anchoring is a versioned maintenance operation, not a quiet local optimizer update.

Source/value writers remain important after useful free-edge pretraining: freeze the learned space and edge assignments, then train a writer to reconstruct edge keys, departures, wide values and residuals **from permitted source material**. Rebuild a fresh graph and check whether it behaves like the optimized one. Otherwise the free bank may learn knowledge that cannot be loaded from new documents.

## 7. Training objectives and state-space fairness

Keep native generation/continuation quality as the main objective. Add small, separately measured terms for:

- **Useful retrieval:** put supported transitions ahead of known distractors; treat unannotated or unsearched edges as *unknown*, not negatives.
- **Useful movement:** move toward states whose next retrieval improves the answer; train no-op on genuine no-evidence cases and penalize aimless loops, not justified revisits.
- **Faithful content:** compare returned knowledge against permitted source evidence and train useful subsequent generations; discourage unsupported blends.
- **Stable geometry:** cap state steps and value norms, keep distinguishable search purposes apart, and penalize collapse of all departures into one hub.
- **Coverage fairness:** avoid a graph that works only in a few heavily trained state regions.

**Fairness here means coverage of the search state space, not forcing every state to receive equal traffic.** Divide the learned space into coarse regions. For each region, estimate supported query demand and useful-hit rate. Give a modest penalty when a region with real, repeated demand has poor coverage, with caps for rare/noisy regions. Keep a reserved exploration budget for underrepresented but source-supported regions. Do not use unverified absent matches as evidence that a region is bad. Also report route length and success by task/source family to detect hidden domination by popular datasets.

One starting objective is:
\[
L=L_{\mathrm{native}}+\lambda_rL_{\mathrm{retrieval}}+
\lambda_sL_{\mathrm{state}}+\lambda_fL_{\mathrm{coverage}}+
\lambda_gL_{\mathrm{geometry}}.
\]
Set the coefficients using development data, then freeze them for release comparisons. Global MoE expert-load balancing stays **separate** from state-space coverage: making experts equally busy does not prove the graph is searchable or fair.

Train with metadata-only starts, real and synthetic multi-hop paths, hard lookalikes, unsupported questions, version conflicts, contradictory evidence, loops, source deletions, delayed retrieval and truly new sources. Credit is local through selected edges and the wide-value MoE projection; discrete ANN candidate IDs and external fetching are not differentiable. Use short, replayable unrolls to train multi-step state movement without keeping unbounded training graphs.

## 8. Costs, risks and safeguards

| Consequence | Required response |
| --- | --- |
| **More latency and KV work.** Each useful result runs a real C pass and adds a full native megatoken. | Measure first-result delay, queries per answer, search/transfer/projection/C costs, and total task time. At 32 × 512 with two 128-wide KV heads, a result adds **32 KiB** of BF16 native KV. Enforce budgets and asynchronous admission rules. |
| **Large edge values are expensive.** | One million 32,768-wide BF16 values need about **65.5 GB (61 GiB)** just for value storage; trainable FP32 masters and two Adam moments may add about **393 GB**, before keys, residuals, index and backups. Test smaller/factored values, cold-value quantization and memory-tiering without silently losing trainability. |
| **Blending can wash out or fabricate knowledge.** | Retain source IDs and uncertainty, separate incompatible versions/claims, test mixtures against individual-edge outputs and permit branch splits. |
| **The state can drift, loop or collapse.** | Bound steps, train meaningful progress and null, reserve backtracking, log revisits and check performance on unseen routes. |
| **Two-address retrieval can miss the best joint edge.** | Compare dual-index union recall with exact joint scoring, monitor shortlist bias and null calibration. |
| **Growth can destabilize distributed learning.** | Use one canonical publisher, versioned state anchors, frozen search windows and atomic neural+KB checkpoint publication/recovery rules. |
| **Metadata can leak labels.** | Audit what was knowable before the first step and preserve strict source/task-family held-out splits. |
| **Source deletion can leave unsupported learned knowledge.** | Track complete support footprints, revoke affected edges and rematerialize from accepted writers or explicitly retrain. |

The wide-value argument, graph motif hierarchy and fairness losses are **hypotheses to test**, not established benefits.

## 9. Implementation order and acceptance

| Stage | Main work and code owners | Exit test |
| --- | --- | --- |
| **A. Exact reference** | New transition reader and edge contract in \`memory/\`; tiny in-memory brute-force index | Dual similarity, identical mixture weights, exact null, bounded state updates and gradients match hand-computed cases. |
| **B. Data and initial placement** | \`data/schema.py\`, \`data/retrieval.py\`, preparation scripts and new graph builder | Causally safe metadata starts; source-backed clustering, transition bigrams and larger motifs reproducibly produce a usable pilot graph. |
| **C. Native integration** | \`model/core.py\`, \`model/multitrack.py\`, \`memory/runtime.py\`, MoE projector, runtime replay | First metadata break; at each eligible step query → retrieve → native 32-facet knowledge event → next query. No direct KB-KV admission or query recursion. |
| **D. Training and growth** | \`training/engine.py\`, \`training/bank.py\`, sparse optimizer, index, DiLoCo and checkpoint code | Useful gradients through selected edges and projector; synchronized growth, coherent state revisions, resumed replay and fairness metrics. |
| **E. Writer and scale** | Source-edge writer, index maintenance, serving benchmark | A fresh graph built from permitted new sources improves held-out tasks with inference weights fixed and meets a declared memory/latency budget. |

**Breaking migration:** bump architecture/KB/sidecar/tape identities (provisionally v10), add explicit new config fields, and reject old KB rows and replay receipts. Reuse compatible v9 E/C/P/R weights only through a tested **weight-only warm start**; do not import v9 point/pair values or their optimizer moments as edges. Training must not silently select an old v9 recipe. Retire the old direct-KV path only once the new one has implementation and acceptance receipts; leave the current README's supported training instructions intact until then.

### What would establish success?

Run matched-budget comparisons against **no KB, source RAG only, current v9 memory, query-only retrieval without state, and full state-plus-query retrieval**. Ablate metadata-first retrieval, wide-value width, MoE projection, seeded transition bigrams, higher-order motifs, continuous growth and coverage loss independently.

Report **multi-hop task success, useful-source recall, new-source uptake, trajectory length, unnecessary revisits, null accuracy, state-region coverage, first-result latency, end-to-end time, peak RAM/VRAM, storage, and replay fidelity**. Train/evaluate across different source families and unseen paths; a small synthetic demo is a correctness check, not proof of quality.

**Promotion gate:** the full transducer must beat the strongest budget-matched simpler baseline on held-out search/answer quality, without unacceptable ordinary-generation regressions or cost, and must work after reconstruction from new source material. Until then, this is a documented proposed redesign, not the supported architecture.
