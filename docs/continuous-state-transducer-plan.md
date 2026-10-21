# Continuous State Transducer: an adaptive knowledge base

**Status:** Proposed successor to Megatoken v9; not implemented.  
**Revised:** 21 October 2026.  
**Baseline:** [Megatoken v9](https://github.com/werg/megatoken/commit/4d24d05dbf4e55d5289e1c2d4727f854d66fddf1), the [current retrieval specification](retrieval-memory-plan.md), and its [implementation ledger](retrieval-memory-implementation-status.md).

## 1. The design in brief

Replace the current semantic memory of points and pair correlations with a **continuous-state transducer**: a learned collection of local transitions. The model searches it with both a query and its current search state. Each transition supplies knowledge **and** a change in search state. Generation becomes a sequence of retrievals and useful movements through a learned space.

The transducer is a trainable function, not a registry of verified facts. **Its knowledge grows when additional capacity improves future predictions enough to justify the memory and computation it costs.** There is no separate system for approving candidate edges, assigning semantic state anchors, or requiring a source citation for each latent parameter.

Keep the following core architecture:

- Each edge has a **departure state**, **query key**, **wide knowledge value**, and **state residual**.
- Query similarity and state similarity jointly determine the edge weights. The same weights blend both values and residuals.
- The blended wide knowledge is projected through a mixture of experts (MoE) to one native **32-facet megatoken** and committed through the **knowledge retrieval track**, not written directly as semantic KV.
- An initial metadata megatoken gives the model its first search context. The first retrieval happens immediately after that event; later queries use all earlier events.
- The graph starts from clustered sequences, **transition bigrams**, and recursively repeated patterns (**motifs of motifs**), then learns its own positions and connections.
- Train, add, split, merge, and remove edges under **one prediction-quality-versus-cost objective**.

This document specifies the target behavior, a small reference algorithm, an implementable growth rule, and the tests needed to tell whether the idea works. It does **not** assert that adaptive transducer growth has already been validated for Megatoken.

### The distinction that matters

**Latent memory** learns reusable representations from training data. Its edges can combine evidence across many examples and need not map to documents.  
**Literal source retrieval** returns particular bytes, passages, and citations. The existing source RAG path retains source identity, access rules, and document versions where those are needed.

Do not turn document-store bookkeeping into a constraint on how latent transitions can learn.

## 2. The transducer

Let the current search state be \(s_t\), the model's query be \(q_t\), and edge \(i\) hold:

| Vector | Role |
| --- | --- |
| \(d_i\) | Departure address: where the edge is useful |
| \(k_i\) | Query key: what it can help retrieve |
| \(v_i\) | Wide knowledge value: what it returns |
| \(r_i\) | State residual: where it moves the search |

These are trainable vectors. Departure state and query key can have different widths. The state does **not** need a human-assigned label or a separately versioned semantic anchor.

Normalize query and departure-address vectors in their own spaces. Start with this score:

\[
z_i=\beta_q\,\hat q_t^\top\hat k_i+
    \beta_s\,\hat s_t^\top\hat d_i+b_i,
\qquad \beta_q,\beta_s>0.
\]

Use one learned null logit \(z_\varnothing\), meaning that none of the retrieved transitions is useful:

\[
w_i=\frac{\exp(z_i)}
{\exp(z_\varnothing)+\sum_j\exp(z_j)}.
\]

The null option has zero knowledge and zero state residual. The two outputs are:

\[
\bar v_t=\sum_i w_i v_i,\qquad
\overline{\Delta s}_t=\sum_i w_i r_i,\qquad
s_{t+1}=\operatorname{retract}(s_t+\eta\overline{\Delta s}_t).
\]

A retraction (such as norm clipping followed by normalization) and bounded \(\eta\) keep state movement finite. **Never use different mixture weights for knowledge and movement.** If null is selected under an inference threshold, commit neither knowledge nor a state change. Train and calibrate that threshold against unsupported queries rather than declaring every nearest neighbor useful.

This reference rule is deliberately simple. Compare it with joint nonlinear scoring and a query-only ablation before adding complexity. Learned biases, null behavior, and state norms need calibration; the equation alone does not guarantee useful search.

### One joint search index

For the additive score above, two separate ANN indexes are unnecessary. With fixed \(\beta_q,\beta_s\), form:

\[
x_t=[\sqrt{\beta_s}\,\hat s_t;\sqrt{\beta_q}\,\hat q_t;1],
\qquad
y_i=[\sqrt{\beta_s}\,\hat d_i;\sqrt{\beta_q}\,\hat k_i;b_i].
\]

Then **\(x_t^\top y_i=z_i\)**. Search one maximum-inner-product index over \(y_i\), rerank its candidates with the exact score, and compare shortlist recall against exhaustive joint scoring. The augmented bias coordinate requires an inner-product-capable index; ordinary normalized-cosine indexing of the whole concatenated vector is not equivalent.

The index is an acceleration structure, not the learned graph itself. Rebuild or update it when trainable addresses move. Approximate retrieval is discrete: ordinary backpropagation reaches selected edges, not unseen neighbors. Reserve a small number of exploratory candidate slots during training so an ignored region can still improve and grow.

An edge's departure state and residual define a local directed move. An explicit adjacency graph or a second hierarchy is **optional**, and should only be introduced if it measurably improves navigation or speed.

## 3. Integration with native Megatoken

Keep ordinary E/C/P/R processing, native KV for committed events, the retrieval-query track, the shared multitrack history, and the existing 512-wide U MoE expert bank. Remove the semantic KB's point/pair correlation path, directional pair scores, correlation attenuation, direct stored-value-to-KV projection, and semantic-KB-specific all-track LoRA control from the **new** retrieval architecture. The source RAG route remains separate and available.

### Event sequence

At each eligible **ordinary** generation or observation position:

1. P generates a normal selected raw-facet retrieval query, the query realizer turns it into \(q_t\), and C commits the query event.
2. Search the transducer with \(q_t\) and the current \(s_t\).
3. Mix selected values and residuals with identical weights. If retrieval is useful, project \(\bar v_t\) to raw native **[32, 512]** facets and commit that megatoken through **C on the knowledge retrieval track**.
4. Apply the paired state update when the result is committed. Continue ordinary generation, so the next query can attend to all prior retrievals and output.

A retrieved megatoken is an **observation**, not a generated answer. Its native KV is created by the ordinary central path, never inserted directly from stored semantic vectors. The model can therefore interpret the new knowledge in the context of what it has already done.

Do not trigger another search merely because a query event or a retrieval-result event was committed; this would recurse indefinitely. Cap total queries, in-flight work, branch depth, repeated queries, and latency. A failed or cancelled lookup leaves the state unchanged. Asynchronous results can appear only at a later legal event boundary; they cannot rewrite prior events. Parallel tracks either follow a deterministic state-update order or have separate search states until an explicit merge.

During training, retain enough selected-event information to recompute gradients: sampled raw query, realized query, selected edge IDs and weights, prior state, retrieved event, and the subsequent continuation. For a stochastic query generated by P, use the existing replayable sampled-event policy machinery where necessary; do **not** claim that gradients pass through the discrete sample or ANN candidate choice automatically. Keep replay spans bounded.

### Wide value to native knowledge megatoken

**Blend wide values first, then reduce to the native 32 facets.** Test value widths **8,192**, **16,384**, and **32,768**; treat 32,768 as a research pilot, not a fixed production requirement.

The existing U experts operate on 512-wide rows. They cannot consume a whole 32,768-wide vector directly. Add a small adapter that maps or splits the blended value into 512-wide working blocks. Route those blocks through the U MoE pool, then use 32 learned facet queries or an equivalent bounded readout to assemble **[32, 512]**. The current query and state can influence routing. Compare sharing U with a separate projection pool, because shared experts may interfere with ordinary text realization.

Do not assume wide values are better merely because they are wider. Compare at equal total parameter count, storage, and compute with narrower or factored values and a simple linear projector. A mixture of learned vectors is **not** necessarily a literal averaging of factual claims; test whether the downstream model preserves necessary distinctions.

### Metadata and the first retrieval

Construct a short prefix from information visible at the start: the request, retrieval purpose, topic, source type, language, modality, task context, and any available date or version. Unknown fields stay unknown. Initialize:

\[
s_0=\operatorname{retract}(s_{\mathrm{base}}+
G_{\mathrm{meta}}(\text{available metadata and request})).
\]

After C processes the **first metadata megatoken**, schedule the first query and retrieval pause, before ordinary answer generation. Do not wait to finish a long document merely to begin searching. If there is no metadata, use a learned neutral initial state.

**Training targets are allowed to teach the memory after a prediction; they must not be smuggled into the input for that prediction.** This is the relevant causal boundary. Apply the same rule at inference and preserve clean held-out test data. Do not impose an artificial requirement that every trained edge can be traced to a single document.

## 4. The single learning objective

The core objective is **better future prediction at a reasonable cost**:

\[
J(\theta,\mathcal K)
=
\mathbb E_{\tau\sim\mathcal D}
\big[L_{\mathrm{continuation}}(\tau;\theta,\mathcal K)\big]
+\lambda\,C_{\mathrm{memory}}(\mathcal K)
+\mu\,\mathbb E_\tau[C_{\mathrm{retrieval}}(\tau)].
\]

Here \(\theta\) denotes the normal model and projection parameters; \(\mathcal K\) denotes the edges; and the expectation is over training trajectories. Start with edge count or stored bytes for memory cost, and candidate scans plus retrieval/C passes for runtime cost. Establish the task distribution and loss units before tuning \(\lambda,\mu\). The cost terms should be evaluated consistently when comparing architectures.

This is **complexity-penalized predictive learning**, motivated by minimum-description-length model selection. It is **not** an exact MDL code-length calculation unless actual codes and their lengths are specified. Structural changes optimize this objective at growth intervals; gradients optimize existing parameters between those intervals.

The main loss is ordinary future-generation or task-continuation loss **after** the relevant retrieval. Truncated multi-step rollouts allow gradients through the state residual, later queries, and subsequent knowledge megatokens. A useful edge may improve a prediction several retrievals later even if its immediate output looks weak.

Keep auxiliary terms small and only when a specific failure is observed:

- **Null calibration** on unsupported queries: do not retrieve knowledge when none helps.
- **State stability** for genuine loops, runaway norms, or repeated useless moves.
- **Optional future-prediction probe** if the state fails to carry useful history.

Do not start with separate objectives for source attribution, manually designated state regions, uniform state occupancy, or a catalogue of edge-admission conditions.

### What a good state represents

The state is a compact record of **which future retrievals would help**, not a position in a hand-named ontology. Histories that need the same future search behavior may share a state; histories needing different continuations should become distinguishable through training.

A basic test is to hold the query fixed while changing the preceding search history. Does the state select the different transitions that improve the respective continuations? Compare against a query-only reader. If the state adds no predictive value, spreading states across a larger geometric area is not a fix.

This is motivated by predictive-state representations and predictive compression, but the proposed neural state is not claimed to be a formally minimal sufficient statistic.

### Coverage without artificial uniformity

Do **not** make every region of state space equally busy. State coordinates are learned and carry no intrinsic uniform-density requirement. Set a meaningful training/task mixture, including rare but important situations, and measure loss by task group. Allocate memory where there is repeated **correctable error** within that mixture.

A small exploration quota for poorly served and unseen query/state combinations prevents popular paths from monopolizing learning. That is sufficient for the initial fairness experiment; add a dedicated coverage loss only if controlled comparisons show a real gap that the base objective cannot address.

## 5. Initialize from sequences, bigrams, and motifs of motifs

Avoid millions of randomly placed edges. Use the training data to discover reusable sequential structure **without treating it as fixed truth**.

1. **Get coarse symbols.** Encode and cluster chunks, search events, query intents, or other ordered observations. Retain sequence boundaries and useful coarse context. Clusters only provide an initial vocabulary.
2. **Count transition bigrams.** Find repeated adjacent pairs of **transitions or events**, not just adjacent words. Repeated patterns suggest where the model can reuse search structure.
3. **Compress recurrent patterns recursively.** Replace repeated sequences with reusable grammar rules, then find recurring combinations of those rules. This is the practical meaning of **motifs of motifs**. SEQUITUR is a concrete model for this kind of recursive pattern discovery.
4. **Initialize coarse states and edges.** Embed contexts from those patterns to obtain starting departure positions. Initialize query keys from the corresponding questions or next-event representations, wide values from available content representations, and residuals toward the next initial state: \(r\approx s_{\mathrm{next}}-s_{\mathrm{departure}}\), accounting for the chosen state update scale.
5. **Train through actual predictions.** Let gradient descent relocate, specialize, and reshape the vectors. The seed hierarchy can be discarded after initialization unless keeping it earns a measurable quality or compute benefit.

Natural ordered sequences provide a starting point even without annotated successful search paths. Real or synthetic multistep tasks can improve initialization, but **do not require synthetic trajectories to justify each edge**.

A frequent pattern does not prove that skipping its intermediate steps is safe. The second query may depend on what the first retrieval returns. Initially use larger motifs for sharing or initialization while preserving the steps; learn a shortcut only if it improves downstream prediction when tested.

The hierarchy is therefore a **compression-based starting guess**, not a second scoring system and not a permanent graph of prescribed meanings.

## 6. Grow the KB from error

**One rule:** use capacity where it lowers expected future loss more than it costs. Every edge is a local component of the function \(F_{\mathcal K}(s,q)=(\bar v,\overline{\Delta s})\).

### Find places that need capacity

For a bounded sample of training steps, retain:

\[
(s_t,\ q_t,\ F_t,\ g_t),\qquad
g_t=\frac{\partial L_{\mathrm{future}}}{\partial F_t},
\qquad F_t=F_{\mathcal K}(s_t,q_t).
\]

The gradient \(g_t\) measures which change to **both knowledge and state movement** would have helped later continuation. Get it by backpropagating through short subsequent retrieval/generation windows, not by attaching only an immediate similarity label.

Group nearby query/state inputs with persistently aligned corrections, and use them as locations for candidate new edges. A handful of samples from a replay buffer is enough to pilot this; a permanent region-approval database is not required.

Interpret the evidence carefully:

- **Many visits, little correctable error:** no growth needed.
- **Recurring aligned error:** likely missing local capacity.
- **Large, inconsistent error:** may be noise, ambiguity, or missing observations; do not automatically allocate an edge.

Include fresh examples and exploratory candidates, because a region omitted by ANN receives no useful selected-row gradient. Do not confuse high traffic with high need.

### Why the gradient points to useful new edges

Write the combined output as \(F_t=(\bar v_t,\overline{\Delta s}_t)\), including the null component. Suppose an extra component with output \(u\) gets a small local mixture weight \(\epsilon h_t\):

\[
F'_t=(1-\epsilon h_t)F_t+\epsilon h_t u.
\]

To first order,

\[
\left.\frac{dL}{d\epsilon}\right|_{\epsilon=0}
=\sum_t h_t\,g_t^\top(u-F_t).
\]

A **negative derivative** means a small contribution from that component would improve the measured continuation loss. Choose its initial location, key, value and residual to make that improvement repeat across similar steps; regularize their scale and compare with the new edge's cost.

This is a **local differentiable argument** with the candidate set and surrounding trajectory held fixed. It is not a guarantee that a large change, discrete ANN insertion, or long-run recurrence will improve. Confirm candidate growth with a small held-out training/selection sample and actual reruns. Keep the final test set untouched.

### Four structural operations, one decision rule

| Operation | When and how |
| --- | --- |
| **Add** | A coherent residual-gradient pattern is poorly served. Initialize a new local edge near those inputs and make its contribution small at first. |
| **Split** | One edge repeatedly receives incompatible corrections. Replace it with two initially equivalent children, perturb their keys/departures slightly, then train them to specialize. |
| **Merge** | Two edges serve the same contexts and can be replaced by a fitted shared edge without meaningful future-loss increase. |
| **Remove** | Ablating an edge increases loss less than the storage and retrieval work saved, after allowing a small local adjustment. |

For the softmax reader, splitting can preserve its exact initial function: duplicate an edge's key, departure, value and residual, and halve its positive pre-softmax strength on each child (subtract \(\log 2\) from each child's bias). Their combined numerator and denominator match the parent. **Top-k candidate selection must temporarily treat the children as one group**, or the equivalence fails because the two children consume extra shortlist slots. After the split settles, they can compete separately.

Do **not** merge just because two vectors are nearby or delete just because an edge is rarely visited. Use a cheap candidate screen followed by sampled continuation-loss trials. A rare edge may be essential for a rare task.

Evaluate structural changes as approximate changes in \(J\). Apply a limited batch of the best beneficial changes, train briefly, and measure again. Hard memory limits may force replacement when no free capacity remains; choose the least harmful change under the same objective. No independent "proposal/admission" bureaucracy is needed.

### A minimal operating loop

~~~text
For each training window:
    run query -> retrieval -> native knowledge event -> continuation
    update model and selected edges from downstream loss
    sample (state, query, output, future-loss gradient) into a small buffer

At occasional safe optimizer boundaries:
    find repeated, aligned missing corrections
    try add/split candidates; try cheap merge/removal candidates
    estimate continuation gain minus added memory and retrieval cost
    apply a bounded set of beneficial changes
    refresh the search index; resume training
~~~

Start with fixed intervals and a strict limit on changed edges. Tune the interval only after showing that growth beats a fixed-size bank. If all changes fail to improve held-out continuation, **stop growing**.

### Continual learning and new information

New training observations follow the same loop: make predictions, measure error, update existing edges, and grow if needed. Replay a modest mix of older tasks to detect and reduce forgetting. There is no mandatory source-to-edge writer or required reconstruction of the graph from individual documents.

An **optional amortized initializer** may learn to create useful edge seeds directly from new data, after the free adaptive bank works. Judge it by adaptation speed, predictive quality, and memory cost. Distilling a bank into a compact writer or rebuilding it can be useful for deployment, but neither is a prerequisite for training latent knowledge.

For applications with citations, permissions, time-limited documents, or deletion requirements, retain the literal-source subsystem and enforce those policies at the source-retrieval boundary. **A trained latent parameter cannot be guaranteed to forget a source simply by deleting its metadata.** True machine unlearning or isolation of restricted material is a separate requirement that must be implemented and evaluated explicitly.

## 7. Practical implementation and migration

This is a **breaking successor (proposed v10)**, not a change to supported v9. Do not reinterpret v9 point/pair rows as transducer edges.

### Components to change

| Area | Required work |
| --- | --- |
| **Config / model** | Add search-state size, value width, null scoring, retrieval-event limits, and a dedicated knowledge-result track. Keep native 32 × 512 geometry and ordinary E/C/P/R history. |
| **Memory reader** | Replace point/pair correlations and group KV admission with state-plus-query joint scoring, null behavior, blended value/residual, and one joint ANN index. |
| **Projection / central** | Add wide-to-512 adapters, MoE routing, and native 32-facet knowledge-event admission through C. Remove the semantic KB's direct KV and KB-specific LoRA answer path. |
| **Training** | Unroll short multi-step retrieval trajectories; carry gradients through selected edges, knowledge projection, subsequent state updates, and later continuation. Handle stochastic query replay explicitly. |
| **Growth / optimizer** | Maintain a small residual-gradient sample, score add/split/merge/remove trials, sparse parameters and optimizer state. Use fixed safe growth intervals. |
| **Storage / distributed** | Store trainable edge fields, stable IDs, sparse optimizer moments and joint-index postings. Publish model, KB and compatible index snapshots together; one coordinator assigns structural changes during DiLoCo. |
| **Serving** | Keep metadata-first retrieval, deterministic event order, null handling, query/latency caps, and consistent native KV cache updates. Keep literal source RAG separate. |

Existing SQLite storage, IVF infrastructure, sparse-row optimizers, query replay, source RAG, and DiLoCo coordinator are useful foundations. New layouts and tests are required; the existing point/pair, direct-KV, and writer-specific contracts are **not** the new architecture. The [implementation ledger](retrieval-memory-implementation-status.md) continues to describe the current code, not this proposal.

### Cost and stability

At one million edges, **32,768 BF16 value dimensions alone need about 65.5 GB (61 GiB)**. A separate FP32 master plus two FP32 Adam moments add roughly **393 GB**, before keys, residuals, indexes, and backups. This makes wide-value factorization, cold-value compression, smaller pilot banks, and sparse updates important comparisons—not assumed solutions.

At native 32 × 512 and two 128-wide KV heads, one new knowledge-result event adds about **32 KiB of BF16 KV**. Every such result also runs a real C pass. Measure end-to-end latency rather than only ANN lookup time.

Potential failures require tests, not a growing list of admission rules:

- **State collapse or loops:** inspect future-query usefulness, repeated visits, norm drift, and justified backtracking.
- **False retrieval:** calibrate the null case and evaluate unsupported questions.
- **Knowledge lost in projection:** compare wide mixtures, individual edge outputs, narrower values, and an ordinary projector.
- **Bad growth:** test no-change, split equivalence, gradient agreement, selection bias, and retention after later training.
- **Approximate index blind spots:** measure top-k joint recall and use small exploratory slots while learning.
- **Forgetting:** evaluate old and new task mixes after growth.
- **Distributed inconsistency:** apply structural changes at an optimizer boundary; refresh row IDs, optimizer moments and index together. Save a compatible model/KB snapshot.

The data's provenance and evaluation partition remain relevant for **dataset integrity** and for literal source citations. They are not required attributes of each latent transition.

## 8. Experiments and acceptance

Build a small exact-search reference before optimizing indexes or distributed growth. Use known synthetic multistep problems for correctness, then real held-out tasks and fresh data for quality. Avoid evaluating candidates on the final test data.

### Required correctness tests

1. **Joint-score equivalence:** concatenated inner product equals separate query/state scoring, including bias; joint ANN recall is measured against exhaustive retrieval.
2. **Mixture and null:** exactly the same weights blend knowledge and residual; null does not create a knowledge event or advance the state; normalization remains stable.
3. **Native event path:** projected results have shape [32, 512], use C on the knowledge track, write ordinary KV, and never enter as direct semantic KV.
4. **Causality:** metadata-only first retrieval; no future-answer input leakage; no query recursion; delayed results do not modify committed history.
5. **Gradients:** finite-difference checks for selected-edge value/residual and state/query paths; downstream gradients reflect later retrievals. Discrete ANN choice is not reported as differentiable.
6. **Growth:** initialization is small, a split preserves the initial softmax function under grouped top-k, beneficial changes lower sampled loss, and losing candidates do not get admitted.
7. **Lifecycle:** sparse moments survive add/split/merge/remove, the index reflects moved addresses, distributed learners resume from one coherent snapshot, and old tasks are retested for forgetting.
8. **State necessity:** with the query fixed, different histories can produce different useful retrievals. If a query-only model performs equally well, do not claim an advantage from stateful search.

### Budget-matched comparisons

Compare on equal stored bytes, retrieval operations, and training effort where possible:

- **Source RAG alone.**
- **Current v9 semantic memory.**
- **Fixed-size transducer without growth.**
- **Uniform/random edge growth at the same budget.**
- **Error-driven growth without state** (query-only).
- **Full error-driven state transducer.**
- **With versus without recursive bigram/motif prewarming.**
- **Different wide-value sizes and projection methods.**

Report continuation/task loss, multi-hop answer success, retrieval precision and null accuracy, benefit from the search state, held-out and rare-task performance, new-data adaptation, forgetting, graph size and growth rate, useful growth per added edge, ANN recall, queries per task, end-to-end time, RAM/VRAM, disk use, and replay fidelity. Include variance across seeds and task distributions, not just a favorable demonstration.

**Promotion criterion:** the full adaptive transducer must beat the strongest budget-matched simpler alternative on held-out continuation or task quality without unacceptable generation regressions, instability, or cost. If the state is unused, the initializer does not help, or growth is not worth its cost, remove those features rather than preserving them for architectural symmetry.

### Order of work

1. Implement exact reference scoring, state update, null handling, and native knowledge-event projection with a **fixed small bank**.
2. Train short multistep continuations and prove that learning the state improves over query-only retrieval.
3. Add the single concatenated-address index; verify recall, gradient behavior and replay.
4. Add the minimal residual-gradient buffer and **add/split**; establish loss improvement at equal capacity.
5. Add **merge/remove**, budget penalties, and forgetting checks.
6. Compare random versus clustered/bigram/recursive-motif initialization.
7. Add scalable storage, sparse training, DiLoCo coordination, and serving only after simpler experiments justify them.

This order keeps the theoretical question visible: **can a learned search-state transition function improve future prediction, and can it allocate its own memory more efficiently than simpler alternatives?**

## 9. Research basis and limits

The design draws on several established ideas, **without claiming that their original guarantees automatically apply** to an ANN-retrieved recurrent Megatoken model:

- **Local error-driven growth:** J. Platt, [A Resource-Allocating Network for Function Interpolation](https://doi.org/10.1162/neco.1991.3.2.213) (1991). Add local units when existing function approximation is inadequate.
- **Function-based network growth:** Q. Liu, L. Wu and D. Wang, [Splitting Steepest Descent for Growing Neural Architectures](https://arxiv.org/abs/1910.02366) (2019). Optimize representation capacity by splitting units.
- **Complexity versus predictive fit:** J. Rissanen, [Minimum Description Length](https://www.scholarpedia.org/article/Minimum_description_length). Our explicit memory/runtime penalty is an engineering proxy, not a derived exact code length.
- **States defined by useful future predictions:** M. Littman, R. Sutton and S. Singh, [Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) (2001).
- **Hierarchical structure from repeated sequences:** C. Nevill-Manning and I. Witten, [Identifying Hierarchical Structure in Sequences: A Linear-Time Algorithm](https://doi.org/10.1613/jair.374) (1997). Recursive digram compression motivates the seed motifs.

What remains unproven is whether residual-gradient placement works reliably with sampled queries, discrete approximate retrieval, high-dimensional wide values, and long recurrent search. The implementation and comparison plan above exists to answer precisely that question.
