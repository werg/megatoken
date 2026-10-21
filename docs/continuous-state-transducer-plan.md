# Continuous State Transducer: the selected v10 design

**Decision:** build this architecture and run it as one integrated training system.  
**Status:** implementation plan, not implemented or trained. **Updated:** 21 October 2026.  
**Reviewed repository:** [2c6d263](https://github.com/werg/megatoken/commit/2c6d263349c732015b9f50750f8ce0632732e925). The [v9 implementation ledger](retrieval-memory-implementation-status.md) remains the account of running code. The [quantization and optimizer research](kb-quantization-optimizer-research.md) records the evidence and alternatives considered; **this document makes the decisions and takes precedence over its experimental roadmap**.

## 1. One architecture, one run

Replace point/pair semantic memory with a **continuous-state transducer**. Each learned edge answers a query from a particular search state, returns a wide knowledge vector, and moves that state. Generation and retrieval become one recurrent search process.

Commit to the following design. These are implementation choices, not a menu for a preliminary architecture search.

| Part | Selected design |
| --- | --- |
| Edge | Departure vector, query key, knowledge value, state residual, and scalar bias |
| Geometry | 256-dimensional search state; 1,024-dimensional query/key; 32,768-dimensional value |
| Retrieval | One joint inner-product index; 32 selected edges plus a null alternative |
| Returned knowledge | Mix wide values first; shared U mixture-of-experts (MoE); one **32 x 512** knowledge megatoken |
| Persistent precision | **NVFP4-style values; FP8 keys, departures and state residuals** |
| Learning | Momentum-free, discounted block-scale updates with a local scale prior |
| Optimizer storage | One FP32 statistic per 256 coordinates, plus a count and timestamp per edge field |
| Small-update retention | An 8,192-edge FP32 working cache; stochastic rounding on publication; no full-bank master weights |
| Initialization | Clustered sequences, transition bigrams, and recursively reused patterns |
| Evolution | Error-driven add/split/merge/remove under a byte budget |
| Distributed ownership | DiLoCo learners update locally; one coordinator merges and changes structure; **no outer KB momentum** |

There is no separate correlation mechanism, manually named state ontology, per-edge source approval, mandatory source writer, second expert pool, or graph-management service. Keep literal-document retrieval for exact passages and citations; latent edges are learned parameters, not document records.

The ambition is a large, plastic, low-bit memory that organizes itself around useful future predictions. The combination is our design, not a claim of an already established state-of-the-art result.

## 2. Retrieval and state movement

For state $s_t$, query $q_t$, and edge $(d_i,k_i,v_i,r_i,b_i)$, normalize the decoded addresses and compute:

$$
z_i=\beta_s\,\hat s_t^\top\hat d_i+\beta_q\,\hat q_t^\top\hat k_i+b_i,
\qquad
w_i=\frac{e^{z_i}}{e^{z_0}+\sum_j e^{z_j}}.
$$

$z_0$ is the learned null logit; the positive similarity scales $\beta_s,\beta_q$ are model parameters. Null contributes zero value and zero movement. For the selected real edges:

$$
\bar v_t=\sum_i w_i v_i,\qquad
\bar r_t=\sum_i w_i r_i,\qquad
s_{t+1}=\operatorname{normalize}(s_t+\bar r_t).
$$

**The weights are identical for knowledge and movement.** Keep the live state on the unit sphere and bound each stored residual to norm 0.25. Use FP32 norms, scores, softmax, weighted sums and state accumulation. Nonfinite inputs are errors, not values to hide with clamping.

Let $a=\sum_i w_i=1-w_0$. A null-dominated result with $a<0.1$ commits neither a knowledge event nor a state change. Use the same rule in training and serving. Start the null bias so ordinary initialized candidates have nonzero mass; use existing supported/no-answer labels to train the gate where labels exist. Unknown relevance is not a negative label. The hard admission decision is not differentiable; gradients through the mixture are live on admitted steps.

### One index, no separate adjacency graph

Use query $x=[\beta_s\hat s;\beta_q\hat q;1]$ and indexed edge address $y_i=[\hat d_i;\hat k_i;b_i]$. Their inner product is exactly $z_i$ for the decoded addresses. Applying the similarity scales on the query side avoids rebuilding the index when those scalars change.

Use the existing disk-backed inverted-file approach with 4,096 coarse cells, probing 32 cells per query and rescoring at most 128 candidates to obtain 32 edges. Store packed FP8 addresses and their normalization factors; do not retain a permanent FP32 address-table duplicate. Coarse centroids are small FP32 derived data. Do not normalize the entire concatenated vector: that changes a score containing a varying bias.

During training, use 30 best candidates and two uniformly sampled distinct edges, rescore all of them normally, and count this exploration in the budget. Serving uses 32 best candidates. The exploration slots supply learning opportunities outside the current neighborhood; they are not forced high-weight answers. Exact search is the numerical test oracle, not a second production architecture.

Update postings when an address or bias changes. The edge's departure and residual already describe the directed move; a separately maintained adjacency graph is unnecessary.

## 3. Put retrieval through the transformer

One ordinary position means one committed group of externally observed or generated megatokens, called a meridian in the existing code. There is one search agenda per session, shared across its tracks.

After every ordinary position: select and commit the native retrieval-query event; look up the edges from the current state; commit any useful knowledge result through C, the central transformer; apply its paired state update; then continue. Query and knowledge-result events are internal events and **do not trigger another query**. Use one outstanding state-changing lookup per session. Parallel tracks share its serialized commit order; there are no private agenda branches or branch-merge machinery in this design.

The first ordinary event is a compact metadata prefix containing the available request, task context, topic, language and modality, plus available date/version information. Initialize $s_0$ from that prefix and a learned base vector. **The first query and retrieval pause occur immediately after this first metadata event**, before an answer. Missing metadata uses the same path with explicit missing fields, not a separate startup mode.

A query cannot use the answer it is about to predict. Training targets may update parameters after the forward pass, just as they update transformer weights. Do not confuse this with requiring a document justification for every latent edge.

### A fixed wide-value projection

Decode and blend the 32,768-wide values before projection. Divide the resulting vector into **64 rows of width 512**, add learned row identities, apply a small shared adapter, route the rows through the **existing U expert bank**, and use **32 learned facet queries** to gather a native $[32,512]$ block. Query and search-state summaries condition the routing/readout. No new expert bank and no giant dense 32,768-to-16,384 matrix are needed.

Pass non-null mass separately: form the content readout from $\bar v/a$ and multiply its final raw block by $a$. This avoids losing evidence strength inside normalization or creating full-strength knowledge from projector biases. State movement remains $\bar r$, with no second application of $a$.

Commit this block as an observed event on a reserved knowledge track. It attends to previous retrieval and generation history through C and writes **ordinary native KV**. There is no direct semantic-KV insertion. Register new adapters/readouts once; reuse the U bank's existing parameter owner and optimizer. Shared U remains part of the frozen realization map for any later latent-policy RL phase.

The query sample is reparameterized during supervised training, so gradients reach P and its embedding realizer. ANN identities and hard admission remain discrete. Unroll **eight ordinary positions** for future-loss credit, preserving complete event groups. Recorded-action policy scoring retains exact raw actions and remains distinct from this supervised path.

Start with a blocking retrieval boundary. IO can overlap independent work, but a result may never be applied to a different agenda state or halfway through a committed group. On cancellation, publish neither its state change nor its knowledge event. Internal events advance native history, not the external source-availability clock.

## 4. Low-bit memory that can still learn

Use one storage layout, not format sweeps:

| Field | Representation |
| --- | --- |
| Knowledge values | E2M1 FP4 elements, one E4M3 FP8 scale per 16 elements, one FP32 scale per row |
| Keys, departures, residuals | E4M3 FP8 elements with FP32 scales per 128 elements |
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

Combine the remaining evidence with four pseudo-observations from a current prior $P_{cb}$ for the edge's coarse index cell $c$:

$$
V_{ib}=\frac{S_{ib}+4P_{cb}}{n_i+4},\qquad
\delta_{ib}=-\eta_f\frac{g_{ib}}{\sqrt{\max(V_{ib},10^{-12})}+10^{-8}}.
$$

Use base rates $\eta_v=10^{-3}$ for values and $\eta_f=10^{-4}$ for keys, departures, residuals and biases, following the same pinned training schedule. Clip value-block update RMS to 1% of $\max(\operatorname{RMS}(v_{ib}),10^{-3})$. For unit addresses, take the tangent gradient, precondition it, project the update back onto the tangent plane and cap angular movement at 0.01 radians before normalization. Cap residual-update norm at 0.01 and residual norm at 0.25. These are launch settings, not claimed universal optima.

### Locality shares scale, not direction

Maintain a recent mean squared-gradient estimate per field/block in each **existing coarse ANN cell**, and one global estimate. Update these once per optimizer window from distinct touched rows, with a 256-step half-life. Count a row once regardless of how many times it appeared in that window. Pool cell estimates toward the global estimate with weight $N_c/(N_c+32)$, where $N_c$ is the discounted observed-row count. Start the global scale at $10^{-6}$; use the previous window's priors for the current update.

This gives an under-observed row a useful step scale without copying a neighbor's knowledge or gradient direction. A moved row changes its prior cell but retains its own discounted history. The cell statistics are small numeric arrays, not a second learned graph.

Discount numerator and count together. After a long absence, confidence returns toward the current prior; an old momentum direction cannot reappear because none is stored. A computed zero gradient is an observation. A row that was not trained is not a stream of zero-gradient observations. Preserve global task-loss normalization; do not divide by visit frequency or retrieval weight and accidentally amplify weak evidence.

A positive preconditioner supplies a local descent direction for a fresh gradient before clipping, quantization and discrete-selection effects. The particular discounting and spatial shrinkage are selected engineering choices, not a convergence theorem for the complete model.

### Drift and stale updates have different remedies

An edge visited after a long absence gets a **fresh gradient** and discounted history. Changes to the shared reader are addressed by the two exploration slots and a fixed **10% replay mixture of earlier training episodes**, recomputed under current weights. Do not penalize knowledge merely for being old.

A worker delta computed from an obsolete shared snapshot is different. Handle that at the merge boundary, not by pretending it is a fresh gradient or by inventing another moment accumulator.

## 6. Initialize once, evolve continuously

Seed **65,536 edges**, then grow within a configured packed-byte budget with a one-million-edge ceiling for the first large run. All components, including low-bit storage, stateful retrieval, the chosen optimizer and growth, are present from the beginning of this run.

Cluster existing chunk/event representations into coarse symbols while preserving sequence boundaries. Count **transition bigrams**, replace recurrent sequences with reusable rules, and recursively find repeated combinations of those rules: **patterns of patterns**. Use this compressed sequence structure to initialize departure positions, query keys, knowledge and short outgoing moves.

Lift existing 32-facet content representations into the wide value space using a fixed signed duplication/permutation with norm preservation, not a giant learned initializer. Then make every value coordinate independently trainable. Align initial state/key projections through causal sequence training, including real search trajectories where available. Fit small residuals toward the next initial state. Do not skip an intermediate query merely because a two-step pattern is common.

After initialization, train the vectors freely. The grammar is an initialization artifact, not an online hierarchy or a requirement that states retain human-readable meanings.

### One objective controls capacity

$$
J=\mathbb E[L_{\mathrm{future}}]
+\lambda\,\mathrm{stored\ bytes}
+\mu\,\mathbb E[\mathrm{retrieval\ work}].
$$

Count parameters, quantizer scales, optimizer summaries, index copies, working cache and transfer/decode/C work. Use the intended task mixture, including rare tasks, rather than a uniform-state-occupancy loss. Retain native MoE load balancing separately. This is complexity-penalized prediction, not a claim to an exact minimum-description-length code.

Keep a bounded 1,024-example training buffer of $(s,q,F,g)$, where $F=(\bar v,\bar r)$ and $g=\partial L_{\mathrm{future}}/\partial F$. Store detached BF16 snapshots for growth scoring, not live graphs; refresh stale samples from their training episodes before committing structural changes. Sample from across the task mixture so frequent visits do not fill the entire buffer.

A new local output $u$ with small mass $\epsilon h_t$ changes the function to $(1-\epsilon h_t)F_t+\epsilon h_tu$. Its first-order loss change is:

$$
\left.\frac{dL}{d\epsilon}\right|_0
=\sum_t h_t g_t^\top(u-F_t).
$$

Use recurring aligned corrections to place new edges. Fit their initial bias for approximately 1% mass on the supporting training examples. A large error alone does not establish missing capacity: check that the selected rows actually moved after packing and that the index did not simply miss them.

Every **256 local optimizer steps**, at a coordinator publication boundary, make at most **256 structural changes**, using one small refreshed training batch to score expected loss change plus byte cost:

| Change | Rule |
| --- | --- |
| Add | Place a small new component where existing components repeatedly miss an aligned correction. |
| Split | Separate an edge serving incompatible corrections. Duplicate its packed fields and subtract $\log 2$ from both biases before perturbation. |
| Merge | Fit one replacement for two behaviorally redundant components when the saved bytes outweigh the loss. |
| Remove | Remove a component whose measured contribution does not justify its cost. Rare use alone is insufficient. |

An unperturbed split preserves the softmax mixture; treat the children as one candidate slot for their first update so shortlist truncation does not break that equivalence. Copy quantized codes, not independently rounded approximations. New edges start with the cell prior; split/merged edges use parent scale estimates as initialization and reset effective counts. Retiring an edge removes its statistics and index entry.

This is an automatic optimizer operation, not a review or approval process. If no change improves the batch objective, continue parameter training without structural change. At the byte ceiling, growth must pay for its capacity through a beneficial replacement. The final evaluation set never determines individual edges.

## 7. DiLoCo and storage ownership

Reuse the current storage, sparse update, indexing and DiLoCo infrastructure; replace its full-vector KB optimizer state and point/pair semantics. Learners keep only their own compact statistics and bounded active cache. The coordinator owns IDs, canonical packed rows and structure.

Use **32 local steps per KB synchronization round**. Each round starts from a compatible model/KB snapshot. Merge only submissions for that round; late or duplicate submissions do not get applied to a newer graph. Slow workers install the next snapshot rather than carrying unbounded stale progress forward. This deliberately bounds distributed staleness while allowing an individual edge to remain dormant for an arbitrarily long time.

Workers send $\delta_{wi}=\theta^{\mathrm{local}}_{wi}-\theta^{\mathrm{base}}_i$ for changed rows. Merge by processed-example weight across **all participating workers**, with zero for an untouched row, then repack:

$$
\theta_i^{\mathrm{new}}
=\theta_i^{\mathrm{base}}+
\frac{\sum_w W_w\delta_{wi}}{\sum_w W_w}.
$$

Do not average only over workers that touched the row, which would silently change its effective learning rate. **There is no outer KB momentum or Adam state.** These deltas are not gradients and are not inputs to the block squared-gradient estimator. Dense model fragments keep their existing optimizer/merge machinery but publish under the same compatible round identity.

Hold a coherent row snapshot through each forward/backward window. Update packed rows, moments and postings together at its boundary; never perform fused row updates halfway through a recurrent backward pass. Save model, KB, learner summaries, active fractional progress, structural cursor and random state as one consistent checkpoint. Publication can be atomic; GPU parameter updates and a database transaction are not magically one transaction. On a partially failed step, resume the last complete checkpoint.

These are ordinary training ownership rules. No semantic anchor versions, detailed per-edge citation ledger or additional approval workflow is introduced. Dataset splits and literal-source access controls remain intact.

## 8. Build the selected system

Target architecture **v10**, new packed-edge layout, and new retrieval-event/checkpoint identities. Reject old point/pair rows and optimizer moments. Reuse compatible v9 model weights by an explicit named weight import; construct the new bank and optimizer fresh. Preserve the recently integrated media input/realizer path.

| Work | Repository owners and deliverable |
| --- | --- |
| Transducer and index | `memory/`: four-vector packed rows, joint score, null, paired mixtures, FP8 postings and exploration |
| Native model integration | `model/config.py`, `core.py`, `multitrack.py`: search state, reserved knowledge track, shared-U readout, one normal C commit |
| Training data and recurrence | `data/`, `training/engine.py`, `meridians.py`: metadata prefix, first retrieval boundary, eight-position credit, causal labels and replay |
| Sparse optimizer and growth | `memory/optimizer.py`, `growth.py`, `training/bank.py`: selected block algorithm, bounded master cache and residual-driven structural updates |
| Distributed execution | `training/diloco_kb.py`, checkpoint/runtime owners: round-bound deltas without outer KB momentum, packed publication and exact resume |

Delete the **v10** direct semantic-KV projector, semantic KB LoRA controllers, point/pair dependence and writer-reproduction launch requirements. Ordinary transformer KV, source RAG, native parameter ownership and exact action tapes stay. Do not modify v9 runtime versions or supported launch commands in this documentation commit; update them together when implementation lands.

Build a small executable reference of this **same** design, close its correctness failures, then use the identical path at scale. There is no requirement to run an optimizer tournament, prove every feature against every ablation, or complete successive research promotion stages before the integrated run. The reference is for checking mathematics and code, not a permanent conservative fallback.

## 9. Resource envelope and run discipline

For one million edges with the selected geometry, calculated storage is approximately:

| Allocation | Decimal GB |
| --- | ---: |
| FP4 values, 16-element scales and one row scale | 18.436 |
| FP8 keys/departures/residuals and 128-element FP32 scales | 1.584 |
| FP32 block statistics for all four vector fields | 0.536 |
| 8,192-edge FP32 working cache | 1.124 |

That is **about 21.68 GB for these components**, before biases, counters, centroids, index duplication, growth samples, gradients, activations, replicas and checkpoints. Per-worker summaries/caches and any baseline/delta buffers must be included in the machine budget. A fully duplicated FP8 joint-address index adds roughly 1.32 GB plus biases/postings. The 1,024-example growth buffer is approximately 0.14 GB with BF16 outputs and gradients. These are design arithmetic, not measured memory usage.

At the existing native geometry, each knowledge event adds 32 KiB of BF16 KV and a C pass. Use the existing whole-meridian history bound. Cap a request at 256 retrieval opportunities; after the cap or an explicit retrieval timeout, continue without changing the agenda. The cap is a runtime budget, not a learned alternative architecture.

**Check correctness before burning the run, and measure quality during it.** Required automated checks are joint-score equality; matched value/residual weights; null/no-recursion; metadata causality; query and selected-row gradients; packed-forward/replay identity; small-update and cache-eviction behavior; dormant-row return updates; split behavior; sparse merge normalization; and checkpoint recovery. Finite-difference tests apply to the decoded differentiable reference; test the quantization surrogate separately.

During training log continuation loss, multi-hop completion, rare-task retention, first-visit recovery after dormancy, null rate, index recall, realized versus intended updates, edge/byte growth, latency and all memory tiers. Use a fixed validation stream for observation, not per-edge admissions or repeated redesign selection. Keep test targets outside training and growth. Numerical corruption, broken causality or failed resume is a defect to repair; an uncertain quality curve is a learning result to investigate, not a reason to pretend the run succeeded.

**The design is decided. Implementation tests protect the run; they do not reopen the architecture.** Train the FP4 transducer, local-scale optimizer and adaptive graph together, document the actual outcome, and make any later architecture change explicitly rather than accumulating hidden fallback paths.
