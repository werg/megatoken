# Continuous State Transducer: the selected v10 design

**Decision:** build this architecture and run it as one integrated training system.  
**Status:** implementation plan, not implemented or trained. **Updated:** 21 October 2026.  
**Reviewed repository:** [7621549](https://github.com/werg/megatoken/commit/762154991782df2d9f789db8e32359015c8c49dd). The [v9 implementation ledger](retrieval-memory-implementation-status.md) remains the account of running code. The [quantization and optimizer research](kb-quantization-optimizer-research.md) records the evidence and alternatives considered; **this document makes the decisions and takes precedence over its experimental roadmap**.

## 1. One architecture, one run

Replace point/pair semantic memory with a **continuous-state transducer**. Each learned edge answers a query from a particular search state, returns a wide knowledge vector, and transforms that state. Search only ever reaches edges near the current location, so retrieval is a walk through a spatially embedded graph, not a flat key-value lookup: what a query can retrieve depends jointly on the query, the current location, and the order of the retrievals that led there. Generation and retrieval become one recurrent search process.

Commit to the following design. These are implementation choices, not a menu for a preliminary architecture search.

| Part | Selected design |
| --- | --- |
| Edge | Departure vector, query key, knowledge value, low-rank transition operator with offset, and scalar bias |
| Geometry | 256-dimensional search state; 1,024-dimensional query/key; 32,768-dimensional value; rank-8 transition factors |
| Retrieval | Density-adaptive location neighbourhood only, with no global search; location-modulated query scoring; 32 selected edges plus a null alternative |
| State movement | Mixture of state-dependent affine operators, so paths are order-sensitive |
| Returned knowledge | Mix wide values first; shared U mixture-of-experts (MoE); one **32 x 512** knowledge megatoken |
| Persistent precision | **NVFP4-style values; FP8 keys, departures, offsets and transition factors** |
| Learning | Momentum-free, discounted block-scale updates with a local scale prior |
| Optimizer storage | One FP32 statistic per 256 coordinates, plus a count and timestamp per edge field |
| Small-update retention | An 8,192-edge FP32 working cache; stochastic rounding on publication; no full-bank master weights |
| Initialization | Clustered sequences, transition bigrams, and recursively reused patterns |
| Evolution | A sparse location inserts an edge during training; error-driven add/split/merge/remove under a byte budget |
| Distributed ownership | DiLoCo learners update locally; one coordinator merges and changes structure; **no outer KB momentum** |

There is no separate correlation mechanism, manually named state ontology, per-edge source approval, mandatory source writer, second expert pool, or graph-management service. Keep literal-document retrieval for exact passages and citations; latent edges are learned parameters, not document records.

The ambition is a large, plastic, low-bit memory that organizes itself around useful future predictions. The combination is our design, not a claim of an already established state-of-the-art result.

## 2. Retrieval and state movement

An edge is $(d_i,k_i,v_i,U_i,V_i,r_i,b_i)$: departure, query key, wide value, rank-8 transition factors $U_i,V_i\in\mathbb R^{256\times8}$, offset, and bias. Normalize the decoded addresses before use.

### The location neighbourhood is the only search

For state $s_t$, the candidate set is the edges departing near it, within a radius set by local density:

$$
c(s_t)=\operatorname{clip}\big(\hat s_t^\top\hat d_{(16)},\,0.3,\,0.5\big),
\qquad
\mathcal N(s_t)=\{\,i:\hat s_t^\top\hat d_i\ge c(s_t)\,\},
$$

where $\hat d_{(16)}$ is the 16th-nearest departure. The set is truncated to the 128 nearest departures. Where the bank is dense, the radius stays at cosine 0.5. Where it is sparse, the radius widens until it holds 16 departures, but never past cosine 0.3. In 256 dimensions, unrelated directions have cosine standard deviation of about 0.06, so 0.3 is still a local bound. Edges reached only through the widened radius pay through the departure term of the score (below), so a distant edge wins only when its key fits the query well; otherwise null wins. **A query never searches outside $\mathcal N(s_t)$.** There is no index over query keys, no global candidate pass, and no global fallback. An edge's successors are the edges whose departures lie near where its transition lands, so the graph is this geometry rather than a separately maintained adjacency list. An empty neighbourhood, with nothing within cosine 0.3, is a null result: no knowledge event and no state change. During training, a **sparse** location, with nothing within cosine 0.5, also inserts an edge there (§6), even when the widened radius found candidates. Retrieval keeps working from the wider radius while growth fills the gap. Serving never inserts, so a serving session whose location moves beyond 0.3 of every edge stops retrieving. Training's jump edges (§6) make this rare, and the rate is logged, not hidden behind another search path.

The departure search is an exact scan of the packed FP8 departure field on device: 256 bytes per edge, or about 0.26 GB and 0.26 GFLOP per lookup at one million edges. Exact search is both the production path and the test oracle, so there is no approximate-recall gap to manage. Coarse departure cells (4,096 k-means centroids, small FP32 derived data) exist only for the optimizer's spatial prior (§5).

### Location-modulated scoring

Inside the neighbourhood, the query is read through the current location:

$$
\tilde q_t=\operatorname{normalize}\!\big(\hat q_t\odot(1+\tanh(M\hat s_t))\big),
\qquad
z_i=\beta_s\,\hat s_t^\top\hat d_i+\beta_q\,\tilde q_t^\top\hat k_i+b_i,
\qquad
w_i=\frac{e^{z_i}}{e^{z_0}+\sum_j e^{z_j}}.
$$

$M\in\mathbb R^{1024\times256}$ is one shared dense model parameter, initialized to zero. Before normalization, $\tilde q^\top k_i=q^\top\operatorname{diag}(k_i)\,(1+\tanh(Ms))$: each edge's query/location interaction is a bilinear form built from its own key and one shared factor. Which query features matter depends on where the search is. The same query therefore selects different edges at different locations, and an edge can answer a query only in combination with the path that led there. The departure term keeps closer departures preferred within the neighbourhood. $z_0$ is the learned null logit; the positive scales $\beta_s,\beta_q$ are model parameters. Serving keeps the 32 highest-scoring neighbourhood edges.

### Transitions are operators

Null contributes zero value and the identity transition. For the selected real edges:

$$
\bar v_t=\sum_i w_i v_i,\qquad
\bar m_t=\sum_i w_i\big(U_i(V_i^\top\hat s_t)+r_i\big),\qquad
s_{t+1}=\operatorname{normalize}(s_t+\bar m_t).
$$

Each edge applies the affine map $(I+U_iV_i^\top)s+r_i$ before renormalization. Because the move depends on the state it is applied to, transitions do not commute. Retrieving A then B lands somewhere different from B then A, and so reaches a different neighbourhood and different knowledge. This is the weighted-automaton form of a transducer, with continuous states. The location is a superposition: when several edges share weight, the next state mixes their transforms, and the next neighbourhood is whatever lies near that mixture. Edges near one another share neighbourhoods, so structure is shared rather than enumerated per path.

**The weights are identical for knowledge and movement.** Keep the live state on the unit sphere. Bound $\|U_i\|_F,\|V_i\|_F\le0.5$ and $\|r_i\|\le2$. Two is the sphere's diameter, so one edge can reach any location, which a jump edge (§6) needs. Ordinary offsets start small and grow only through capped updates. Use FP32 norms, scores, softmax, weighted sums, operator products and state accumulation. Nonfinite inputs are errors, not values to hide with clamping.

Let $a=\sum_i w_i=1-w_0$, with $a=0$ for an empty neighbourhood. A null-dominated result with $a<0.1$ commits neither a knowledge event nor a state change. Serving applies exactly this rule. Start the null bias so ordinary initialized candidates have nonzero mass.

The hard threshold would otherwise freeze learning below it: an edge whose mass stays under 0.1 is never committed, so it receives no gradient that could raise it. Growth would also never see the loss at places where null currently wins. Training therefore makes one change. **One in eight sub-threshold lookups in a non-empty neighbourhood is admitted anyway**, chosen by a seeded draw. It is committed exactly like an admitted result, with its readout scaled by its true $a$ and its full state update, and counted in the retrieval budget. These exploration admissions give the mixture, bias and null logit a live gradient in both directions, and their $(s,q,F,g)$ samples enter the growth buffer. That buffer is the only route by which Add finds missing capacity where null wins.

The no-answer labels mean "unanswerable from any knowledge" by construction, never "absent from the current bank". On labelled no-answer positions, the loss trains $a$ down. Unlabelled positions are not negatives. Growth and null training therefore never pull against each other: growth fills what the bank lacks, and null learns only what no bank could supply.

### Exploration stays local

During training, use the 30 best neighbourhood edges plus two distinct edges sampled uniformly from the boundary ring $c(s_t)-0.1\le\hat s_t^\top\hat d_i<c(s_t)$. Rescore them normally (their departure term is lower), and count this exploration in the budget. These slots let edges just outside the current location receive gradient and move their departures in. They are not forced high-weight answers, and nothing is sampled from the bank at large.

## 3. Put retrieval through the transformer

One ordinary position means one committed group of externally observed or generated megatokens, called a meridian in the existing code. There is one search agenda per session, shared across its tracks.

After every ordinary position: select and commit the native retrieval-query event; score the edges in the current location's neighbourhood; commit any useful knowledge result through C, the central transformer; apply its paired state update; then continue. Query and knowledge-result events are internal events and **do not trigger another query**. Use one outstanding state-changing lookup per session. Parallel tracks share its serialized commit order; there are no private agenda branches or branch-merge machinery in this design.

The first ordinary event is a compact metadata prefix containing the available request, task context, topic, language and modality, plus available date/version information. Initialize $s_0$ from that prefix and a learned base vector. **The first query and retrieval pause occur immediately after this first metadata event**, before an answer. Missing metadata uses the same path with explicit missing fields, not a separate startup mode.

A query cannot use the answer it is about to predict. Training targets may update parameters after the forward pass, just as they update transformer weights. Do not confuse this with requiring a document justification for every latent edge.

### A fixed wide-value projection

Decode and blend the 32,768-wide values before projection. Divide the resulting vector into **64 rows of width 512**, add learned row identities, apply a small shared adapter, route the rows through the **existing U expert bank**, and use **32 learned facet queries** to gather a native $[32,512]$ block. Query and search-state summaries condition the routing/readout. No new expert bank and no giant dense 32,768-to-16,384 matrix are needed.

Pass non-null mass separately: form the content readout from $\bar v/a$ and multiply its final raw block by $a$. This avoids losing evidence strength inside normalization or creating full-strength knowledge from projector biases. State movement remains $\bar m$, with no second application of $a$.

Commit this block as an observed event on a reserved knowledge track. It attends to previous retrieval and generation history through C and writes **ordinary native KV**. There is no direct semantic-KV insertion. Register new adapters/readouts once; reuse the U bank's existing parameter owner and optimizer. Shared U remains part of the frozen realization map for any later latent-policy RL phase.

The query sample is reparameterized during supervised training, so gradients reach P and its embedding realizer. Neighbourhood membership, top-32 selection and hard admission remain discrete. Unroll **eight ordinary positions** for future-loss credit, preserving complete event groups; gradients pass through the transition operators, so an edge is credited for the neighbourhoods its move makes reachable. Recorded-action policy scoring retains exact raw actions and remains distinct from this supervised path.

Retrieval is a blocking boundary. IO can overlap independent work, but a result may never be applied to a different agenda state or halfway through a committed group. On cancellation, publish neither its state change nor its knowledge event. Internal events advance native history, not the external source-availability clock.

## 4. Low-bit memory that can still learn

Use one storage layout, not format sweeps:

| Field | Representation |
| --- | --- |
| Knowledge values | E2M1 FP4 elements, one E4M3 FP8 scale per 16 elements, one FP32 scale per row |
| Keys, departures, offsets, transition factors | E4M3 FP8 elements with FP32 scales per 128 elements |
| Biases, null/scoring controls, optimizer summaries | FP32 |
| Live query/state and accumulations | FP32; cast the projected knowledge block to the model's BF16 input format |
| Active parameter masters | FP32, at most 8,192 complete edges per learner |
| Native transformer KV / shared experts | Keep the existing BF16 path; do not add a simultaneous backbone/cache-quantization project |

Pack actual bits in storage and on the wire. Merely rounding values inside an FP32 array is not compression. FP4 is a storage choice here, not a promise that sparse lookup runs on FP4 tensor cores. Train with packed-and-decoded values in the forward pass; use a straight-through gradient to the bounded working masters. This is the selected quantization-aware surrogate, not a derivative through rounding.

### One rule for fractional updates

A training read loads an edge's packed value into its FP32 working master unless the master is already resident. Accumulate all gradients for that row, update the master once, and publish a newly packed row with **stochastic rounding**. Keep the fractional difference in that resident master across local updates. Quantizer scales cover the updated range; reads never change scales or consume rounding randomness.

Round once per changed row per update, with a seed derived from its ID, field, update number and worker. Reuse the packed row during forward/recomputation. Do not randomly requantize on every read. Evict least-recently-updated masters outside the active backward window; preserve the already published packed value. At a DiLoCo merge, flush the changed masters, send their packed result relative to the baseline, and restart the cache from the installed bank. Checkpoint resident masters and rounding state for an exact interrupted-run resume.

There is **no permanent compensation vector, per-row precision-promotion subsystem, or full-bank master table**. Sub-grid changes can be lost after eviction; stochastic rounding preserves their expectation under the quantizer's unsaturated conditions, not every individual update. We accept this trade-off and measure actual movement on cold returns. It is not a reason to conceal an additional high-precision bank.

## 5. The KB optimizer

Use **discounted block-scale descent with a spatial prior**. It has no first-moment vector, no Adam second-moment vector, and no weight decay on idle knowledge. Each field has FP32 squared-gradient sums $S_{ib}$ for 256-coordinate blocks, an effective observation count $n_i$, and its last-update timestamp. Scalars use one block. Quantizer blocks and optimizer blocks are different things.

For the current accumulated, correctly normalized gradient $g_{ib}$ and elapsed local optimizer steps $\Delta t$:

$$
\rho=2^{-\Delta t/4096},\qquad
e_{ib}=\operatorname{mean}_j g_{ibj}^2,\qquad
S_{ib}\leftarrow\rho S_{ib}+e_{ib},\qquad
n_i\leftarrow\rho n_i+1.
$$

Combine the remaining evidence with four pseudo-observations from a current prior $P_{cb}$ for the edge's coarse departure cell $c$:

$$
V_{ib}=\frac{S_{ib}+4P_{cb}}{n_i+4},\qquad
\delta_{ib}=-\eta_f\frac{g_{ib}}{\sqrt{\max(V_{ib},10^{-12})}+10^{-8}}.
$$

Use base rates $\eta_v=10^{-3}$ for values and $\eta_f=10^{-4}$ for keys, departures, offsets, transition factors and biases, following the same pinned training schedule. Clip value-block update RMS to 1% of $\max(\operatorname{RMS}(v_{ib}),10^{-3})$. For unit addresses, take the tangent gradient, precondition it, project the update back onto the tangent plane and cap angular movement at 0.01 radians before normalization. Cap offset and per-factor update norms at 0.01, offset norm at 2, and each factor's Frobenius norm at 0.5. These are launch settings, not claimed universal optima.

### Locality shares scale, not direction

Maintain a recent mean squared-gradient estimate per field/block in each **coarse departure cell**, and one global estimate. Update these once per optimizer window from distinct touched rows, with a 256-step half-life. Count a row once regardless of how many times it appeared in that window. Pool cell estimates toward the global estimate with weight $N_c/(N_c+32)$, where $N_c$ is the discounted observed-row count. Start the global scale at $10^{-6}$; use the previous window's priors for the current update.

This gives an under-observed row a useful step scale without copying a neighbor's knowledge or gradient direction. A moved row changes its prior cell but retains its own discounted history. The cell statistics are small numeric arrays, not a second learned graph.

Discount numerator and count together. After a long absence, confidence returns toward the current prior; an old momentum direction cannot reappear because none is stored. A computed zero gradient is an observation. A row that was not trained is not a stream of zero-gradient observations. Preserve global task-loss normalization; do not divide by visit frequency or retrieval weight and accidentally amplify weak evidence.

A positive preconditioner supplies a local descent direction for a fresh gradient before clipping, quantization and discrete-selection effects. The particular discounting and spatial shrinkage are selected engineering choices, not a convergence theorem for the complete model.

### Drift and stale updates have different remedies

An edge visited after a long absence gets a **fresh gradient** and discounted history. Changes to the shared reader are addressed by the two boundary-ring exploration slots and a fixed **10% replay mixture of earlier training episodes**, recomputed under current weights. Do not penalize knowledge merely for being old.

A worker delta computed from an obsolete shared snapshot is different. Handle that at the merge boundary, not by pretending it is a fresh gradient or by inventing another moment accumulator.

## 6. Initialize once, evolve continuously

Seed **65,536 edges**, then grow within a configured packed-byte budget with a one-million-edge ceiling for the first large run. All components, including low-bit storage, stateful retrieval, the chosen optimizer and growth, are present from the beginning of this run.

Cluster existing chunk/event representations into coarse symbols while preserving sequence boundaries. Count **transition bigrams**, replace recurrent sequences with reusable rules, and recursively find repeated combinations of those rules: **patterns of patterns**. Use this compressed sequence structure to initialize departure positions, query keys, knowledge and short outgoing transitions.

Lift existing 32-facet content representations into the wide value space using a fixed signed duplication/permutation with norm preservation, not a giant learned initializer. Then make every value coordinate independently trainable. Align initial state/key projections through causal sequence training, including real search trajectories where available. Fit small offsets toward the next initial state. Start each transition with $V_i=0$ and small random $U_i$ (Frobenius norm 0.01): the first moves are pure offsets, gradients still reach both factors, and order dependence is learned rather than imposed. Do not skip an intermediate query merely because a two-step pattern is common.

After initialization, train the vectors freely. The grammar is an initialization artifact, not an online hierarchy or a requirement that states retain human-readable meanings.

### A sparse location grows the graph

During training, a lookup with no departure within cosine 0.5 of $\hat s_t$ proceeds normally over its widened neighbourhood, or returns null if that is empty. After that step's backward pass, it inserts one edge at that location:

- $d=\hat s_t$ and $k=\tilde q_t$, so the edge answers this query at this place;
- $v$ is the fixed wide lift (above) of the mean 32-facet raw block of the next committed ordinary position, scaled to the bank's median value RMS. This uses the training target only to set parameters after the prediction, as §3 permits;
- $V=0$ and small random $U$ (Frobenius norm 0.01);
- if the widened neighbourhood found candidates, $r=0$, and the bias gives $a\approx0.25$ when the same query revisits that location. The edge clears admission and trains from its next visit onward;
- if the neighbourhood was **empty**, the edge is a **jump edge**. Sample one existing edge $j$ uniformly from the bank and set $r=\hat d_j-\hat s_t$, with the bias giving $a\approx0.95$ on revisit. The admitted move lands at about $\hat d_j$, a location that has departures, so the next lookup has candidates. Sampling edges uniformly makes destinations proportional to density.

The inserted edge lies within 0.5 of that location, so one learner never inserts twice for the same place. Each learner makes at most 512 insertions per synchronization round. Jump edges count toward this cap. Beyond it, sparse and empty visits are only counted. Insertions are provisional rows in the learner's working cache, with learner-local IDs. They are trained like any resident edge and submitted at the round boundary (§7). Their bytes count in $J$, and Remove prunes those that do not earn them.

A jump starts random, but its offset trains like any other through the unrolled future loss. Over training it moves toward destinations whose retrievals help, and Remove retires jumps that never do. Training thereby places an exit at every empty location it reaches, and serving dead ends are confined to locations training never visited.

Sparseness is the one structural trigger that needs no gradient evidence, because nothing departs from that location to correct. A dense neighbourhood where null wins is not, by itself, evidence of missing knowledge; that case belongs to Add below.

### One objective controls capacity

$$
J=\mathbb E[L_{\mathrm{future}}]
+\lambda\,\mathrm{stored\ bytes}
+\mu\,\mathbb E[\mathrm{retrieval\ work}].
$$

Count parameters, quantizer scales, optimizer summaries, the device departure table, working cache and transfer/decode/C work. Use the intended task mixture, including rare tasks, rather than a uniform-state-occupancy loss. Retain native MoE load balancing separately. This is complexity-penalized prediction, not a claim to an exact minimum-description-length code.

Each learner keeps a bounded 1,024-example training buffer of $(s,q,F,g)$ from admitted and exploration-admitted lookups, where $F=(\bar v,\bar m)$ and $g=\partial L_{\mathrm{future}}/\partial F$. Store detached BF16 snapshots, not live graphs. Sample from across the task mixture so frequent visits do not fill the entire buffer. Clear the buffer after every structural submission, so each one reflects only the bank it was computed against.

A new local output $u$ with small mass $\epsilon h_t$ changes the function to $(1-\epsilon h_t)F_t+\epsilon h_tu$. Its first-order loss change is:

$$
\left.\frac{dL}{d\epsilon}\right|_0
=\sum_t h_t g_t^\top(u-F_t).
$$

Each edge's output $u=(v,\,U(V^\top\hat s_t)+r)$ is linear in its mixture weight, so this holds for operator transitions as written. $h_t$ is nonzero only where the new edge's departure puts it inside $\mathcal N(s_t)$; candidate placement must respect that. Use recurring aligned corrections inside dense neighbourhoods to place new edges. Fit their initial bias for approximately 1% mass on the supporting training examples. A large error alone does not establish missing capacity: check that the selected rows actually moved after packing, and that a suitable boundary-ring edge does not just need to move its departure in.

Every **256 local optimizer steps** (every eighth synchronization round), structure changes in three steps:

1. **Propose.** Each learner submits its growth buffer with that round. The coordinator pools the buffers and scores add/split/merge/remove candidates by the first-order formula above, together with their byte cost. This uses only dot products over the stored samples and candidate fields, with no model forward pass. It publishes the best **256 candidates** with the next round as a trial set.
2. **Trial.** During that round, each learner takes a disjoint share of the trial set. On one refreshed batch of its own episodes, it reruns forward-only, with and without each change, only those eight-position windows whose lookups touch a neighbourhood containing an affected edge. It submits each change's loss difference and example count.
3. **Commit.** At the following round boundary, the coordinator applies every trial whose example-weighted loss change plus byte cost lowers $J$. It refreshes the device departure table and republishes. Untested or losing trials are discarded.

The changes:

| Change | Rule |
| --- | --- |
| Add | Place a small new component where existing components repeatedly miss an aligned correction. |
| Split | Separate an edge serving incompatible corrections. Duplicate its packed fields, transition factors included, and subtract $\log 2$ from both biases before perturbation. |
| Merge | Fit one replacement for two behaviorally redundant components when the saved bytes outweigh the loss. |
| Remove | Remove a component whose measured contribution does not justify its cost. Rare use alone is insufficient. |

An unperturbed split preserves both the value mixture and the operator mixture; treat the children as one candidate slot for their first update so shortlist truncation does not break that equivalence. Copy quantized codes, not independently rounded approximations. New edges start with the cell prior; split/merged edges use parent scale estimates as initialization and reset effective counts. Retiring an edge removes its statistics and departure-table entry.

This is an automatic optimizer operation, not a review or approval process. If no trial lowers $J$, parameter training continues without structural change. At the byte ceiling, growth must pay for its capacity through a beneficial replacement. The final evaluation set never determines individual edges.

## 7. DiLoCo and storage ownership

Reuse the current storage, sparse update and DiLoCo infrastructure; replace its IVF index, full-vector KB optimizer state, demand forwarding and point/pair semantics. Learners keep only their own compact statistics and bounded active cache. The coordinator owns IDs, canonical packed rows and structure.

Use **32 local steps per KB synchronization round**. Each round starts from a compatible model/KB snapshot. Merge only submissions for that round; late or duplicate submissions do not get applied to a newer graph. Slow workers install the next snapshot rather than carrying unbounded stale progress forward. This deliberately bounds distributed staleness while allowing an individual edge to remain dormant for an arbitrarily long time.

Workers send $\delta_{wi}=\theta^{\mathrm{local}}_{wi}-\theta^{\mathrm{base}}_i$ for changed rows. Merge by processed-example weight across **all participating workers**, with zero for an untouched row, then repack:

$$
\theta_i^{\mathrm{new}}
=\theta_i^{\mathrm{base}}+
\frac{\sum_w W_w\delta_{wi}}{\sum_w W_w}.
$$

Do not average only over workers that touched the row, which would silently change its effective learning rate. **There is no outer KB momentum or Adam state.** These deltas are not gradients and are not inputs to the block squared-gradient estimator. Dense model fragments keep their existing optimizer/merge machinery but publish under the same compatible round identity.

Provisional edges from sparse-location insertion have no baseline, so they bypass the delta formula. Each learner submits their decoded fields, block statistics and processed-example weight. The coordinator combines provisional rows from different learners whose departures and keys both have cosine at least 0.9, using an example-weighted mean. It then packs them, assigns canonical IDs and publishes them with the round. At the byte ceiling, a provisional row enters only by replacing a lower-value edge under $J$. Otherwise it is dropped, and its location is sparse again on the next visit. Learners create provisional rows; only the coordinator makes them canonical.

Hold a coherent row snapshot through each forward/backward window. Update packed rows, optimizer statistics and the device departure table together at its boundary. Departures are therefore fixed within a window, and the table never lags the rows it indexes; never perform fused row updates halfway through a recurrent backward pass. Save model, KB, learner summaries, active fractional progress, structural cursor and random state as one consistent checkpoint. Publication can be atomic; GPU parameter updates and a database transaction are not magically one transaction. On a partially failed step, resume the last complete checkpoint.

These are ordinary training ownership rules. No semantic anchor versions, detailed per-edge citation ledger or additional approval workflow is introduced. Dataset splits and literal-source access controls remain intact.

## 8. Build the selected system

Target architecture **v10**, new packed-edge layout, and new retrieval-event/checkpoint identities. Reject old point/pair rows and optimizer moments. Reuse compatible v9 model weights by an explicit named weight import; construct the new bank and optimizer fresh. Preserve the recently integrated media input/realizer path.

| Work | Repository owners and deliverable |
| --- | --- |
| Transducer and neighbourhood search | `memory/`: packed rows with transition factors, exact density-adaptive departure-neighbourhood scan, location-modulated score, null, paired value/operator mixtures and boundary-ring exploration |
| Native model integration | `model/config.py`, `core.py`, `multitrack.py`: search state, reserved knowledge track, shared-U readout, one normal C commit |
| Training data and recurrence | `data/`, `training/engine.py`, `meridians.py`: metadata prefix, first retrieval boundary, eight-position credit, causal labels and replay |
| Sparse optimizer and growth | `memory/optimizer.py`, `growth.py`, `training/bank.py`: selected block algorithm, bounded master cache, sparse-location insertion and gradient-driven structural trials |
| Distributed execution | `training/diloco_kb.py`, checkpoint/runtime owners: round-bound deltas without outer KB momentum, provisional-row reconciliation (replacing learner demand forwarding), packed publication and exact resume |

Delete the **v10** direct semantic-KV projector, semantic KB LoRA controllers, point/pair dependence and writer-reproduction launch requirements. Ordinary transformer KV, source RAG, native parameter ownership and exact action tapes stay. Do not modify v9 runtime versions or supported launch commands in this documentation commit; update them together when implementation lands.

Build a small executable reference of this **same** design, close its correctness failures, then use the identical path at scale. There is no requirement to run an optimizer tournament, prove every feature against every ablation, or complete successive research promotion stages before the integrated run. The reference is for checking mathematics and code, not a permanent conservative fallback.

## 9. Resource envelope and run discipline

For one million edges with the selected geometry, calculated storage is approximately:

| Allocation | Decimal GB |
| --- | ---: |
| FP4 values, 16-element scales and one row scale | 18.436 |
| FP8 keys/departures/offsets/transition factors and 128-element FP32 scales | 5.808 |
| FP32 block statistics for all six vector fields | 0.600 |
| 8,192-edge FP32 working cache | 1.258 |

That is **about 26.10 GB for these components**, before biases, counters, centroids, growth samples, gradients, activations, replicas and checkpoints. Per-worker summaries/caches and baseline/delta buffers count against the machine budget. The device-resident FP8 departure table that the neighbourhood scan reads is about 0.26 GB with scales; there is no query-key index. The 1,024-example growth buffer is approximately 0.14 GB with BF16 outputs and gradients. These are design arithmetic, not measured memory usage.

At the existing native geometry, each knowledge event adds 32 KiB of BF16 KV and a C pass. Use the existing whole-meridian history bound. Cap a request at 256 retrieval opportunities; after the cap or an explicit retrieval timeout, continue without changing the agenda. The cap is a runtime budget, not a learned alternative architecture.

**Check correctness before burning the run, and measure quality during it.** Required automated checks are neighbourhood-only candidates (no code path reaches an edge outside $\mathcal N(s_t)$ or the training boundary ring); location-modulated scores against an unfused reference; order sensitivity of composed transitions; matched value/operator weights; density-adaptive radius bounds (0.5 dense, 0.3 floor); empty-neighbourhood null at serving; single sparse-location insertion in training; jump-edge landing on the sampled departure; exploration-admission rate and seeding; provisional-row reconciliation; three-step structural trial/commit; null/no-recursion; metadata causality; query and selected-row gradients; packed-forward/replay identity; small-update and cache-eviction behavior; dormant-row return updates; split behavior; sparse merge normalization; and checkpoint recovery. Finite-difference tests apply to the decoded differentiable reference; test the quantization surrogate separately.

During training log continuation loss, multi-hop completion, rare-task retention, first-visit recovery after dormancy, null rate, adaptive-radius distribution, sparse-location and empty-neighbourhood rates in training and serving, insertions and jump edges and their survival through Remove, exploration-admission gain, realized versus intended updates, edge/byte growth, latency and all memory tiers. Use a fixed validation stream for observation, not per-edge admissions or repeated redesign selection. Keep test targets outside training and growth. Numerical corruption, broken causality or failed resume is a defect to repair; an uncertain quality curve is a learning result to investigate, not a reason to pretend the run succeeded.

**The design is decided. Implementation tests protect the run; they do not reopen the architecture.** Train the FP4 transducer, local-scale optimizer and adaptive graph together, document the actual outcome, and make any later architecture change explicitly rather than accumulating hidden fallback paths.
