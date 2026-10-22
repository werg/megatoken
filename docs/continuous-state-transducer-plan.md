# Continuous State Transducer: the selected v10 design

**Decision:** build this architecture and run it as one integrated training system.  
**Status:** implementation plan, not implemented or trained. **Updated:** 21 October 2026.  
**Reviewed repository:** [7621549](https://github.com/werg/megatoken/commit/762154991782df2d9f789db8e32359015c8c49dd). The [v9 implementation ledger](retrieval-memory-implementation-status.md) remains the account of running code. The [quantization and optimizer research](kb-quantization-optimizer-research.md) records the evidence and alternatives considered; **this document makes the decisions and takes precedence over its experimental roadmap**.

## 1. One architecture, one run

Replace point/pair semantic memory with a **continuous-state transducer**. Each learned edge answers a query from a particular search state, returns a wide knowledge vector, and transforms that state. Search only ever reaches edges near locations the session has already reached, its agenda, so retrieval is a walk through a spatially embedded graph, not a flat key-value lookup: what a query can retrieve depends jointly on the query, the agenda, and the order of the retrievals that built it. Generation and retrieval become one recurrent search process.


| Part | Selected design |
| --- | --- |
| Edge | Departure vector, query key, knowledge value, low-rank transition operator with offset, and scalar bias |
| Geometry | 256-dimensional search state; 1,024-dimensional query/key; 32,768-dimensional value; rank-8 transition factors |
| Retrieval | Every query searches the density-adaptive neighbourhoods of all agenda locations, with no global search; location-modulated query scoring; one softmax over 32 selected location–edge pairs plus a null alternative |
| State movement | Each pair applies its edge's affine operator to its own location; the blended landing joins the agenda; paths are order-sensitive |
| Returned knowledge | Mix wide values first; shared U mixture-of-experts (MoE); one **32 x 512** knowledge megatoken |
| Persistent precision | **NVFP4-style values; FP8 keys, departures, offsets and transition factors** |
| Learning | Values are written from hindsight as decayed weighted means; addresses and transitions use momentum-free, discounted block-scale updates |
| Optimizer storage | One FP32 statistic per 256 coordinates, plus a count and timestamp per edge field |
| Small-update retention | An 8,192-edge FP32 working cache; stochastic rounding on publication; no full-bank master weights |
| Hindsight | A training-only hindsight track (full, then group-inverse masking read at gap 1, then gap 2) whose realizer on C emits bank-format values. It is trained first, overlaid on retrieval while the bank trains, initializes values and writes them |
| Initialization | Clustered sequences, transition bigrams, and recursively reused patterns; values from hindsight |
| Evolution | A sparse location inserts an edge during training; per-edge utility and gradient-disagreement statistics, accumulated during ordinary training, drive split and remove under a byte budget |
| Distributed ownership | DiLoCo learners update locally; one coordinator merges and changes structure; late submissions are age-weighted, not rejected; **no outer KB momentum** |

Literal-document retrieval (source RAG) stays for exact passages and citations.

## 2. Retrieval and state movement

An edge is $(d_i,k_i,v_i,U_i,V_i,r_i,b_i)$: departure, query key, wide value, rank-8 transition factors $U_i,V_i\in\mathbb R^{256\times8}$, offset, and bias. Normalize the decoded addresses before use.

### The agenda

Each session keeps an **agenda**: every location its search has reached, $\mathcal A_t=\{s_0,\dots,s_t\}$. Each committed lookup appends its landing location. The request's 256-lookup cap bounds the agenda. **Every query searches from all agenda locations at once.** Each location offers its own neighbourhood, and the query's similarity to the keys there decides how much each location contributes. Reach therefore grows with the session, and a late query can still use structure near where the session started or passed through.

### Location neighbourhoods are the only search

Each agenda location $s_j$ has a neighbourhood of the edges departing near it, within a radius set by local density:

$$
c(s_j)=\operatorname{clip}\big(\hat s_j^\top\hat d_{(16)},\,0.3,\,0.5\big),
\qquad
\mathcal N(s_j)=\{\,i:\hat s_j^\top\hat d_i\ge c(s_j)\,\},
$$

where $\hat d_{(16)}$ is the 16th-nearest departure. The set is truncated to the 128 nearest departures. Where the bank is dense, the radius stays at cosine 0.5. Where it is sparse, the radius widens until it holds 16 departures, but never past cosine 0.3. In 256 dimensions, unrelated directions have cosine standard deviation of about 0.06, so 0.3 is still a local bound.

The candidates for a query are the pairs $(j,i)$ with $i\in\mathcal N(s_j)$ for every agenda location. An edge near several agenda locations appears once per location, because reaching it from a different place is a different context. **A query never searches outside the agenda's neighbourhoods.** There is no index over query keys, no global candidate pass, and no global fallback. An edge's successors are the edges whose departures lie near where its transition lands, so the graph is this geometry rather than a separately maintained adjacency list.

A neighbourhood is computed once, when its location joins the agenda, and cached as up to 128 edge IDs. Each lookup therefore scans the departure table only for its newest location. Departures change only at window boundaries (§7), where a session's cached neighbourhoods are recomputed. The scan is exact, over the packed FP8 departure field on device: 256 bytes per edge, or about 0.26 GB and 0.26 GFLOP per scan at one million edges. Exact search is both the production path and the test oracle.

If every agenda neighbourhood is empty, the lookup is a null result: no knowledge event and no new location. This can only happen while the agenda's locations all lie beyond cosine 0.3 of every edge. During training, a **sparse** newest location, with nothing within cosine 0.5, also inserts an edge there (§6). Serving never inserts.

### Location-modulated scoring

Each pair reads the query through its own location:

$$
\tilde q_{t,j}=\operatorname{normalize}\!\big(\hat q_t\odot(1+\tanh(M\hat s_j))\big),
\qquad
z_{ji}=\beta_s\,\hat s_j^\top\hat d_i+\beta_q\,\tilde q_{t,j}^\top\hat k_i+b_i,
\qquad
w_{ji}=\frac{e^{z_{ji}}}{e^{z_0}+\sum_{(j',i')}e^{z_{j'i'}}}.
$$

$M\in\mathbb R^{1024\times256}$ is one shared dense model parameter, initialized to zero; $M\hat s_j$ is cached with each agenda location. Before normalization, $\tilde q^\top k_i=q^\top\operatorname{diag}(k_i)\,(1+\tanh(Ms))$: each edge's query/location interaction is a bilinear form built from its own key and one shared factor. The same query therefore selects different edges at different locations. The departure term keeps each pair's edge close to its own location. The softmax runs over all pairs from all locations together, so query–key similarity decides which locations matter for this question. $z_0$ is the learned null logit; the positive scales $\beta_s,\beta_q$ are model parameters. Serving keeps the 32 highest-scoring pairs across the whole agenda.

### Transitions are operators

Null contributes zero value and keeps the newest location. Each selected pair moves from its own location:

$$
\bar v_t=\sum_{(j,i)} w_{ji}\,v_i,\qquad
\ell_{ji}=(I+U_iV_i^\top)\hat s_j+r_i,\qquad
s_{t+1}=\operatorname{normalize}\Big(w_0\,\hat s_t+\sum_{(j,i)} w_{ji}\,\ell_{ji}\Big).
$$

With a single agenda location this is $\operatorname{normalize}(s+\sum_i w_i(U_iV_i^\top s+r_i))$. Because each move depends on the location it is applied to, transitions do not commute. Retrieving A then B lands somewhere different from B then A, and so reaches different neighbourhoods and different knowledge. This is the weighted-automaton form of a transducer, with continuous states. The new location is a superposition: it blends the landings from every agenda location in proportion to how well each answered the query. Edges near one another share neighbourhoods, so structure is shared rather than enumerated per path.

**The weights are identical for knowledge and movement.** Keep every location on the unit sphere. Bound $\|U_i\|_F,\|V_i\|_F\le0.5$. Offsets are unbounded: renormalization keeps the state finite, and an edge's reach is whatever training makes useful. Use FP32 norms, scores, softmax, weighted sums, operator products and state accumulation. Nonfinite inputs are errors, not values to hide with clamping. Agenda locations older than the unroll window (§3) are detached.

Let $a=\sum_{(j,i)}w_{ji}=1-w_0$, and let an edge's weight $\omega_i=\sum_j w_{ji}$ be its total over its pairs. **Every lookup with a non-empty candidate set commits** its knowledge event, with the readout scaled by $a$ (§3), and appends its new location. Null mass weakens the result continuously; there is no admission threshold. Gradients therefore always reach the mixture, biases and null logit, and every committed lookup updates its edges' statistics (§6). Start the null bias so ordinary initialized candidates have nonzero mass.

The no-answer labels mean "unanswerable from any knowledge" by construction, never "absent from the current bank". On labelled no-answer positions, the loss trains $a$ down. Unlabelled positions are not negatives.

### Exploration stays local

During training, use the 30 best pairs plus two edges sampled uniformly from the boundary rings $c(s_j)-0.1\le\hat s_j^\top\hat d_i<c(s_j)$ of uniformly chosen agenda locations, each paired with its location. Rescore them normally (their departure term is lower), and count this exploration in the budget. These slots let edges just outside the agenda receive gradient and move their departures in.

## 3. Put retrieval through the transformer

An ordinary meridian is one committed set of externally observed or generated megatokens, as in the existing code. There is one search agenda per session, shared across its tracks.

Lookups happen at **latency boundaries**. Real boundaries are completed turns and arrivals of source results. Artificial boundaries are inserted after 8 committed ordinary meridians without one; training samples that interval from 4 to 12. Lookup $t$'s **retrieval group** $G_t$ is the ordinary meridians from its boundary to the next.

At each boundary: select and commit the native retrieval-query event; score the pairs from every agenda location's neighbourhood; commit the knowledge result through C, the central transformer; append its landing to the agenda; then continue. Query and knowledge-result events are internal events and **do not trigger another query**. Use one outstanding state-changing lookup per session. Parallel tracks share its serialized commit order; there are no private agenda branches or branch-merge machinery in this design.

The first ordinary event is a compact metadata prefix containing the available request, task context, topic, language and modality, plus available date/version information. Initialize $s_0$ from that prefix and a learned base vector. **The first query and retrieval pause occur immediately after this first metadata event**, before an answer. Missing metadata uses the same path with explicit missing fields, not a separate startup mode.

A query cannot use the answer it is about to predict. Training targets may update parameters after the forward pass, just as they update transformer weights.

### A fixed wide-value projection

Decode and blend the 32,768-wide values before projection. Divide the resulting vector into **64 rows of width 512**, add learned row identities, apply a small shared adapter, route the rows through the **existing U expert bank**, and use **32 learned facet queries** to gather a native $[32,512]$ block. Query and search-state summaries condition the routing/readout.

Pass non-null mass separately: form the content readout from $\bar v/a$ and multiply its final raw block by $a$. This avoids losing evidence strength inside normalization or creating full-strength knowledge from projector biases. The new location uses the raw pair weights, with no second application of $a$.

Commit this block as an observed event on a reserved knowledge track. It attends to previous retrieval and generation history through C and writes **ordinary native KV**. Register new adapters/readouts once; reuse the U bank's existing parameter owner and optimizer. Shared U remains part of the frozen realization map for any later latent-policy RL phase.

The query sample is reparameterized during supervised training, so gradients reach P and its embedding realizer. Neighbourhood membership and top-32 selection remain discrete. Unroll **four retrieval groups** for future-loss credit, preserving complete event groups; gradients pass through the transition operators, so an edge is credited for the neighbourhoods its move makes reachable. Recorded-action policy scoring retains exact raw actions and remains distinct from this supervised path.

Retrieval is a blocking boundary. IO can overlap independent work, but a result may never be applied to a different agenda state or halfway through a committed group. On cancellation, publish neither its state change nor its knowledge event. Internal events advance native history, not the external source-availability clock.

### The hindsight track

The hindsight track is a reserved, **training-only** track. For each lookup it produces $h_t$, the knowledge value that would have been ideal, judged by what the training sequence actually does next. This follows the hindsight writer in `~/hindsight-kb`, carried out natively on tracks.

**The hindsight pass** runs C once over the sequence's ordinary events with a **group-inverse mask**. Ordinary events are those observed or generated, never query or knowledge events. Under the mask, an event attends bidirectionally to every event of its own retrieval group, and to every event of later groups. Each group's final C states therefore summarize that group and everything after it. The pass never touches retrieval, so it runs once per sequence, before the causal pass, for all lookups in parallel. Its activations never enter `KVHistory`, and no causal track or serving path can read them. The cost is one C pass over the ordinary events, in training only.

**The hindsight realizer** is a head on C, a sibling of the retrieval-query realizer. Its 64 learned row queries attend over the final states of one group's events and emit $h_t$ in the bank's own format: a 32,768-wide value, packed to FP4 and decoded exactly like a stored edge value. $h_t$ is the only thing the track emits. It is not conditioned on the query: it is what the future needs at lookup $t$, and queries, keys and locations learn to find it.

Tracks gain a masking mode, implemented as an explicit mask built from per-query and per-key group, meridian and track metadata. This replaces today's purely structural causality. Causal stays the default. Which group the realizer reads decides what $h_t$ may see, and that follows a curriculum during warm-up:

1. **Full** (first 4,096 steps): the pass is bidirectional over the whole sequence; read $G_t$.
2. **Gap 1** (next 4,096 steps): group-inverse pass; read $G_t$, the group the knowledge serves, and everything after it.
3. **Gap 2** (from then on): group-inverse pass; read $G_{t+1}$, from one group past the one served, following `~/hindsight-kb`'s suffix rule.

Stages 2 and 3 are the same pass read one group apart.

**Training order.**

- **Hindsight warm-up (first 16,384 steps, spanning the mask curriculum).** No bank exists yet. Every knowledge event's value is $h_t$, read through the ordinary readout and committed through C. The future loss trains the hindsight track and realizer, the readout, P and C together. Locations move only by edges, so the agenda stays at $s_0$.
- **Seeding.** Edge values are initialized from $h$ (§6).
- **Overlay.** Every committed lookup's value becomes $(1-\alpha)\,\bar v_t/a+\alpha\,h_t$ before readout, and the final block is scaled by $(1-\alpha)a+\alpha$ instead of $a$. $\alpha$ anneals linearly from 1 to 0 over 65,536 steps. While $\alpha>0$, every lookup commits; an empty candidate set contributes $\bar v_t/a=0$. The model keeps receiving good knowledge while the bank learns to supply it.
- **Hindsight write, for the whole run.** Every committed training lookup writes $h_t$ into the edges it used, weighted and blended (§4). This is how values learn.
- **After the overlay ends,** one in eight training sequences runs hindsight-only (every value is $h_t$). This keeps the hindsight track training against the current reader. The other seven use the bank alone, with the hindsight write.

Because the hindsight track is co-trained with the reader, $h_t$ is always in the reader's current language, and every visit writes that language into the bank.

## 4. Low-bit memory that can still learn

Storage layout:

| Field | Representation |
| --- | --- |
| Knowledge values | E2M1 FP4 elements, one E4M3 FP8 scale per 16 elements, one FP32 scale per row |
| Keys, departures, offsets, transition factors | E4M3 FP8 elements with FP32 scales per 128 elements |
| Biases, write masses, null/scoring controls, optimizer summaries | FP32 |
| Live query/locations and accumulations | FP32; cast the projected knowledge block to the model's BF16 input format |
| Active address masters | FP32 address fields of at most 8,192 edges per learner |
| Write accumulators | FP32 weighted sums for at most 8,192 written edges per learner |
| Native transformer KV / shared experts | Existing BF16 path |

Pack actual bits in storage and on the wire. The forward pass always reads packed-and-decoded rows.

### Writing hindsight into the bank

**Values are written, not gradient-trained.** Each edge keeps a write mass $A_i$. Within an optimizer window, every committed training lookup adds $\omega_i$ to the edge's accumulator $W_i$ and $\omega_i h_t$ to its accumulator $H_i$. At the window boundary, with $\rho_w=2^{-\Delta t/16384}$:

$$
A_i'=\rho_wA_i+W_i,\qquad
v_i'=\frac{\rho_wA_iv_i+H_i}{A_i'}.
$$

A value is thus the decayed mean of the hindsight values of the lookups it served, weighted by how much each lookup relied on it, and blended with what it held before. The decay lets values follow the reader as it changes. Each written row is packed once per window with stochastic rounding, seeded from its ID, window and worker. The future-loss gradient still flows through $\bar v_t$ into the mixture weights, addresses, transitions and the hindsight track; it is never applied to values. An edge whose lookups want different values shows up in its gradient sketch and is split (§6). Values need no optimizer state and no working masters.

### One rule for fractional address updates

A training read loads an edge's packed address fields (key, departure, offset, transition factors) into its FP32 working master unless the master is already resident. Accumulate all gradients for that row, update the master once, and publish newly packed fields with **stochastic rounding**. Keep the fractional difference in that resident master across local updates. Quantizer scales cover the updated range; reads never change scales or consume rounding randomness. Gradients reach the masters straight through the packed forward pass.

Round once per changed row per update, with a seed derived from its ID, field, update number and worker. Reuse the packed row during forward/recomputation. Evict least-recently-updated masters outside the active backward window; preserve the already published packed fields. At a DiLoCo merge, flush the changed masters and write accumulators, send their results (§7), and restart the cache from the installed bank. Checkpoint resident masters, write accumulators and rounding state for an exact interrupted-run resume.

There is no full-bank master table. Sub-grid changes can be lost after eviction; stochastic rounding preserves them in expectation.

## 5. The KB optimizer

Use **discounted block-scale descent with a global prior**. It has no first-moment vector, no Adam second-moment vector, and no weight decay on idle knowledge. It updates address fields and biases; values are written (§4). Each address field has FP32 squared-gradient sums $S_{ib}$ for 256-coordinate blocks, an effective observation count $n_i$, and its last-update timestamp. Scalars use one block. Quantizer blocks and optimizer blocks are different things.

For the current accumulated, correctly normalized gradient $g_{ib}$ and elapsed local optimizer steps $\Delta t$:

$$
\rho=2^{-\Delta t/4096},\qquad
e_{ib}=\operatorname{mean}_j g_{ibj}^2,\qquad
S_{ib}\leftarrow\rho S_{ib}+e_{ib},\qquad
n_i\leftarrow\rho n_i+1.
$$

Combine the remaining evidence with four pseudo-observations from the current global prior $P_b$ for that field and block:

$$
V_{ib}=\frac{S_{ib}+4P_b}{n_i+4},\qquad
\delta_{ib}=-\eta_f\frac{g_{ib}}{\sqrt{\max(V_{ib},10^{-12})}+10^{-8}}.
$$

Use base rate $\eta_f=10^{-4}$ for keys, departures, offsets, transition factors and biases, following the same pinned training schedule. For unit addresses, take the tangent gradient, precondition it, project the update back onto the tangent plane and cap angular movement at 0.01 radians before normalization. Cap offset and per-factor update norms at 0.01 and each factor's Frobenius norm at 0.5.

The global prior $P_b$ is a recent mean squared-gradient estimate per field/block, updated once per optimizer window from distinct touched rows with a 256-step half-life, starting at $10^{-6}$. Each update uses the previous window's prior. It gives a new or rarely seen row a sensible step scale until its own history accumulates.

Discount numerator and count together. After a long absence, confidence returns toward the current prior; an old momentum direction cannot reappear because none is stored. A computed zero gradient is an observation. A row that was not trained is not a stream of zero-gradient observations. Preserve global task-loss normalization; do not divide by visit frequency or retrieval weight and accidentally amplify weak evidence.

### Staleness

There is no replay. Old rows and late updates stay usable as follows:

- **Values** are rewritten from the current hindsight value on every visit (§4), and older writes decay, so a row returning from dormancy moves to the reader's current language at once.
- **Keys** are never the gate to retrieval. Candidates come from the agenda's locations, so an edge with an outdated key still enters the mixture whenever its location is visited, and receives the gradient that realigns it.
- **Statistics** are discounted sums of observations: optimizer block sums here, structural statistics in §6. A late contribution is still valid evidence, and its age is applied as discount.
- **Distributed updates** arriving late are age-weighted rather than rejected (§7).

## 6. Initialize once, evolve continuously

After the hindsight warm-up (§3), seed **65,536 edges**, then grow within a configured packed-byte budget with a one-million-edge ceiling for the first large run. From seeding onward, low-bit storage, stateful retrieval, the chosen optimizer, growth and the hindsight overlay all run together.

Cluster existing chunk/event representations into coarse symbols while preserving sequence boundaries. Count **transition bigrams**, replace recurrent sequences with reusable rules, and recursively find repeated combinations of those rules: **patterns of patterns**. Use this compressed sequence structure to initialize departure positions, query keys, knowledge and short outgoing transitions.

Run the warmed hindsight track over the seeding corpus. Each seed edge's value is the mean $h$ over its pattern's occurrences, with write mass equal to their count. Align initial state/key projections through causal sequence training, including real search trajectories where available. Fit small offsets toward the next initial state. Start each transition with $V_i=0$ and small random $U_i$ (Frobenius norm 0.01): the first moves are pure offsets, gradients still reach both factors, and order dependence is learned rather than imposed. Do not skip an intermediate query merely because a two-step pattern is common.

After initialization, train the vectors freely. The grammar is used only for initialization.

### A sparse location grows the graph

During training, a lookup whose newest location $\hat s_t$ has no departure within cosine 0.5 proceeds normally over the agenda, or returns null if every agenda neighbourhood is empty. After that step's backward pass, it inserts one edge at $\hat s_t$:

- $d=\hat s_t$ and $k=\tilde q_{t,t}$, so the edge answers this query at this place;
- $v=h_t$ with write mass 1. The hindsight value is used only to set parameters after the prediction, as §3 permits;
- $V=0$ and small random $U$ (Frobenius norm 0.01);
- if the agenda found candidates, $r=0$, and the bias gives $a\approx0.25$ when the same query revisits that location;
- if every agenda neighbourhood was **empty**, the edge is a **jump edge**. Sample one existing edge $j$ uniformly from the bank and set $r=\hat d_j-\hat s_t$, with the bias giving $a\approx0.95$ on revisit. Its move lands at about $\hat d_j$, a location that has departures, so the agenda has candidates from then on. With the agenda, this happens only when a session's search has never reached a populated region. Sampling edges uniformly makes destinations proportional to density.

The inserted edge lies within 0.5 of that location, so one learner never inserts twice for the same place. Each learner makes at most 512 insertions per synchronization round. Jump edges count toward this cap. Beyond it, sparse and empty visits are only counted. Insertions are provisional rows in the learner's working cache, with learner-local IDs. They are trained like any resident edge and submitted at the round boundary (§7). Their bytes count in $J$, and Remove prunes those that do not earn them.

A jump starts random, but its offset trains like any other through the unrolled future loss. Over training it moves toward destinations whose retrievals help, and Remove retires jumps that never do. Training thereby gives every empty start an exit; serving dead ends are confined to starts training never saw.

Sparseness is the one structural trigger that needs no gradient evidence, because nothing departs from that location to correct. A dense neighbourhood with high null mass is not, by itself, evidence of missing knowledge; that case belongs to Split below.

### One objective controls capacity

$$
J=\mathbb E[L_{\mathrm{future}}]
+\lambda\,\mathrm{stored\ bytes}
+\mu\,\mathbb E[\mathrm{retrieval\ work}].
$$

Count parameters, quantizer scales, optimizer summaries, the device departure table, working cache and transfer/decode/C work. Use the intended task mixture, including rare tasks. Native MoE load balancing stays separate.

### Structure from statistics training already computes

Structural decisions use no buffer, no candidate search and no extra forward passes. On every committed lookup, the backward pass already has the future-loss gradient $g=\partial L_{\mathrm{future}}/\partial F$ for the mixture output $F=(\bar v_t,\ w_0\hat s_t+\sum w_{ji}\ell_{ji})$, each selected pair's output $(v_i,\ell_{ji})$, and its weight. Let $\bar u_i$ be edge $i$'s $\omega_i$-weighted mean pair output; null's output is $(0,\hat s_t)$. Removing edge $i$ would renormalize the mixture to $(F-\omega_i\bar u_i)/(1-\omega_i)$, so to first order it would change the loss by

$$
\Delta_i=\frac{\omega_i}{1-\omega_i}\,g^\top(F-\bar u_i).
$$

That is one dot product per selected edge. Each edge accumulates, with a 16,384-local-step half-life:

- **utility** $\mathcal U_i$: the discounted sum of $\Delta_i$, the loss its presence has been saving;
- **gradient sketch** $G_i\in\mathbb R^{64}$: the discounted sum of $R\gamma_{i,t}$, where $\gamma_{i,t}$ is the loss gradient with respect to all its fields at that visit, including its value even though values are written rather than stepped, and $R$ is one fixed seeded random projection;
- **gradient energy** $Q_i$: the discounted sum of $\|R\gamma_{i,t}\|^2$, and the discounted visit count $n_i$.

If an edge's gradients agree, $\|G_i\|^2\approx n_iQ_i$; if they cancel, $\|G_i\|^2\ll n_iQ_i$. Their ratio $\kappa_i=\|G_i\|^2/(n_iQ_i)$ is the edge's gradient coherence.

At every round boundary, the coordinator applies two rules from these merged statistics:

| Change | Rule |
| --- | --- |
| Split | $n_i\ge64$, $\kappa_i<0.1$, and mean energy $Q_i/n_i$ in the bank's top decile: the edge is pushed hard in directions that cancel, because it serves contexts that want different things. Duplicate its packed codes, transition factors included, halve the write mass into each child, subtract $\log2$ from both biases, and perturb the two children's departures and keys in opposite random directions by 0.01 rad. Training separates them. |
| Remove | The edge is at least 4,096 local steps old, and its utility rate (utility divided by the decay window) is below $\lambda$ times its bytes. A rarely used but important edge keeps its utility through the long half-life. |

New capacity comes from sparse-location insertion and from splits. Redundant edges need no merge rule: they divide their utility, and Remove retires whichever no longer pays. Children start from their parent's optimizer and structural statistics, halved; new edges start with the global prior and zero statistics. Retiring an edge removes its statistics and departure-table entry. At the byte ceiling, the lowest-utility-rate eligible edges make room. The final evaluation set never determines individual edges.

## 7. DiLoCo and storage ownership

Reuse the current storage, sparse update and DiLoCo infrastructure; replace its IVF index, full-vector KB optimizer state, demand forwarding and point/pair semantics. Learners keep only their own compact statistics and bounded active cache. The coordinator owns IDs, canonical packed rows and structure.

Use **32 local steps per KB synchronization round**. A learner installs the newest published snapshot at its next window boundary and submits its deltas relative to the base it trained from. A submission $\alpha_w$ rounds behind the current canonical version merges with weight $2^{-\alpha_w/8}$. Late work is discounted, never refused. A committed per-worker sequence ledger rejects duplicates, and deltas for edges retired since their base are skipped.

Workers send $\delta_{wi}=\theta^{\mathrm{local}}_{wi}-\theta^{\mathrm{base}}_i$ for changed rows. Merge with $W_w$ = processed examples $\times\,2^{-\alpha_w/8}$ across **all workers merged this round**, with zero for an untouched row, then repack:

$$
\theta_i^{\mathrm{new}}
=\theta_i^{\mathrm{base}}+
\frac{\sum_w W_w\delta_{wi}}{\sum_w W_w}.
$$

Do not average only over workers that touched the row, which would silently change its effective learning rate. **There is no outer KB momentum or Adam state.** These deltas are not gradients and are not inputs to the block squared-gradient estimator. Dense model fragments keep their existing optimizer/merge machinery but publish under the same compatible round identity.

Values merge as weighted means, not deltas. For each written row, a learner sends its window-summed write mass $W_w$ and the mean value it wrote, $H_w/W_w$, packed to FP4. The coordinator ages $W_w$ by $2^{-\alpha_w/8}$ into $W_w'$ and blends:

$$
A_i\leftarrow A_i+\sum_wW_w',\qquad
v_i\leftarrow\frac{A_i^{\mathrm{old}}v_i+\sum_wW_w'\,(H_w/W_w)}{A_i}.
$$

This is the write rule applied across learners, so concurrent and late writes blend in proportion to the evidence behind them.

Learners also send their increments to each touched edge's optimizer and structural statistics. These are discounted sums, so the coordinator ages each increment by its submission's age and adds it.

Provisional edges from sparse-location insertion have no baseline, so they bypass the delta formula. Each learner submits their decoded fields, write mass, statistics and processed-example weight. The coordinator combines provisional rows from different learners whose departures and keys both have cosine at least 0.9: values by write-mass-weighted mean, other fields by example-weighted mean. It then packs them, assigns canonical IDs and publishes them with the round. At the byte ceiling, a provisional row enters only by replacing a lower-value edge under $J$. Otherwise it is dropped, and its location is sparse again on the next visit. Learners create provisional rows; only the coordinator makes them canonical.

Hold a coherent row snapshot through each forward/backward window. Update packed rows, written values, optimizer statistics, the device departure table and sessions' cached neighbourhoods together at its boundary. Departures are therefore fixed within a window, and the table never lags the rows it indexes; never perform fused row updates halfway through a recurrent backward pass. Save model, KB, learner summaries, active fractional progress, structural cursor and random state as one consistent checkpoint. On a partially failed step, resume the last complete checkpoint.

## 8. Build the selected system

Target architecture **v10**, new packed-edge layout, and new retrieval-event/checkpoint identities. Reject old point/pair rows and optimizer moments. Reuse compatible v9 model weights by an explicit named weight import; construct the new bank and optimizer fresh. Preserve the recently integrated media input/realizer path.

| Work | Repository owners and deliverable |
| --- | --- |
| Transducer and agenda search | `memory/`: packed rows with transition factors and write masses, the session agenda with cached density-adaptive neighbourhoods, exact departure scan, per-location query modulation, pair softmax with null, value/landing mixtures and boundary-ring exploration |
| Native model integration | `model/config.py`, `core.py`, `multitrack.py`: search state, reserved knowledge track, shared-U readout, one normal C commit; per-track masking modes with an explicit meridian/track mask; the reserved training-only hindsight track (full and group-inverse masks) and its hindsight realizer on C emitting FP4 bank-format values |
| Training data and recurrence | `data/`, `training/engine.py`, `meridians.py`: metadata prefix, first retrieval boundary, latency-boundary schedule with artificial boundaries, four-group credit and causal labels; hindsight pass, mask curriculum, warm-up, overlay schedule and hindsight-only sequences |
| Sparse optimizer and growth | `memory/optimizer.py`, `growth.py`, `training/bank.py`: selected block algorithm for address fields, bounded master cache, hindsight write with write masses, sparse-location insertion, per-edge utility/sketch statistics, split and remove |
| Distributed execution | `training/diloco_kb.py`, checkpoint/runtime owners: age-weighted deltas, write-mass value blending and statistics without outer KB momentum, provisional-row reconciliation (replacing learner demand forwarding), packed publication and exact resume |

Delete the **v10** direct semantic-KV projector, semantic KB LoRA controllers, point/pair dependence and writer-reproduction launch requirements. The v9 source and relation writers (`training/hindsight.py`, `hindsight_native.py`) are superseded by the hindsight track. Ordinary transformer KV, source RAG, native parameter ownership and exact action tapes stay. Do not modify v9 runtime versions or supported launch commands in this documentation commit; update them together when implementation lands.

Build a small executable reference of this same design, fix its correctness failures, then run the identical path at scale.

## 9. Resource envelope and run discipline

For one million edges with the selected geometry, calculated storage is approximately:

| Allocation | Decimal GB |
| --- | ---: |
| FP4 values, 16-element scales and one row scale | 18.436 |
| FP8 keys/departures/offsets/transition factors and 128-element FP32 scales | 5.808 |
| FP32 block statistics for the five address fields | 0.088 |
| FP32 write masses | 0.004 |
| 8,192-edge FP32 address masters and write accumulators | 1.259 |
| Structural statistics: utility, 64-d gradient sketch, energy, count, creation step | 0.272 |

That is **about 25.87 GB for these components**, before biases, counters, gradients, activations, replicas and checkpoints. Per-worker summaries/caches and baseline/delta buffers count against the machine budget. The device-resident FP8 departure table that the neighbourhood scan reads is about 0.26 GB with scales; there is no query-key index. A full agenda holds 257 locations with cached neighbourhoods and modulations, about 1.5 MB per session.

At the existing native geometry, each knowledge event adds 32 KiB of BF16 KV and a C pass. Use the existing whole-meridian history bound. Cap a request at 256 retrieval opportunities; after the cap or an explicit retrieval timeout, continue without changing the agenda.

**Check correctness before burning the run, and measure quality during it.** Required automated checks are agenda-only candidates (no code path reaches an edge outside the agenda's neighbourhoods or the training boundary rings); cached neighbourhoods equal fresh scans after each window boundary; each pair's landing applied from its own location; location-modulated scores against an unfused reference; order sensitivity of composed transitions; matched value/operator weights; density-adaptive radius bounds (0.5 dense, 0.3 floor); empty-agenda null at serving; single sparse-location insertion in training; jump-edge landing on the sampled departure; provisional-row reconciliation; utility and coherence statistics against brute-force removal and gradient sets on a small bank; split and remove rules; hindsight masks (full and group-inverse, the realizer reading $G_t$ or $G_{t+1}$ by stage, and no hindsight activation reachable from any causal track or the serving path); artificial-boundary scheduling; FP4 round-trip of $h_t$; overlay schedule; the write rule equal to the decayed weighted mean, with no optimizer step on values; cross-learner value blending equal to a single learner writing the same lookups; age-weighted late merges of deltas and statistics; null/no-recursion; metadata causality; query and selected-row gradients; packed-forward/recomputation identity; small-update and cache-eviction behavior; dormant-row return updates; split behavior; sparse merge normalization; and checkpoint recovery. Finite-difference tests apply to the decoded differentiable reference; test the quantization surrogate separately.

During training log continuation loss, multi-hop completion, rare-task retention, first-visit recovery after dormancy, null rate, adaptive-radius distribution, sparse-location and empty-agenda rates in training and serving, insertions and jump edges and their survival through Remove, agenda size and the share of weight by location age, hindsight-assisted versus bank-only loss gap, write residual $\|h_t-\bar v_t/a\|$ by edge age, split and removal rates, utility distribution, merge age distribution, realized versus intended updates, edge/byte growth, latency and all memory tiers. Observe on a fixed validation stream; keep test targets outside training and growth.
