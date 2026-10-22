# Continuous State Transducer: the selected v10 design

**Decision:** build this architecture and run it as one integrated training system.  
**Status:** implemented and always on (see §10), not yet trained. **Updated:** 22 October 2026.  
**Reviewed repository:** [88df397](https://github.com/werg/megatoken/commit/88df3978c68dcfd3ef00de12b6e513c57c92be46). The [quantization and optimizer research](kb-quantization-optimizer-research.md) records the evidence and alternatives considered; **this document makes the decisions and takes precedence over its experimental roadmap**.

## 1. One architecture, one run

Replace point/pair semantic memory with a **continuous-state transducer**. Each learned edge answers a query from a particular search state, returns a wide knowledge vector, and moves the search to its destination location. Search only ever reaches edges near locations the session has already reached, its agenda, so retrieval is a walk through a spatially embedded graph, not a flat key-value lookup: what a query can retrieve depends jointly on the query, the agenda, and the order of the retrievals that built it. Retrieval blends like ordinary softmax attention: it always takes what is nearby and concentrates on what fits best. Generation and retrieval become one recurrent search process.

| Part | Selected design |
| --- | --- |
| Edge | Departure location, query key, knowledge value, destination location, and scalar bias |
| Geometry | 256-dimensional locations; 1,024-dimensional query/key; 32,768-dimensional value |
| Retrieval | Every query scores the 1,024 nearest departures of every agenda location, with no global search; location-modulated query scoring; one softmax over the 32 best location–edge pairs and a learned attention sink |
| State movement | The softmax blend of the selected edges' destinations joins the agenda; paths are order-sensitive because which edges can be taken depends on where the search is |
| Returned knowledge | Mix wide values first; shared U mixture-of-experts (MoE); one **32 x 512** knowledge megatoken |
| Persistent precision | **NVFP4-style values; FP8 keys, departures and destinations** |
| Hindsight | A training-only hindsight track (full, then group-inverse masking read at gap 1, then gap 2) whose realizer on C emits bank-format values. It trains first, steers the search and supplies values while the bank warms up, writes out the bank's entries for the whole run, and writes values until the switch |
| Learning | Two phases. **Written:** values are decayed weighted means of hindsight values. **Learned:** after the switch, values train by gradient like every other field. Keys, departures and destinations always use momentum-free, discounted block-scale updates |
| Optimizer storage | One FP32 statistic per 256 coordinates, plus a count and timestamp per trained edge field |
| Small-update retention | An 8,192-edge FP32 working cache; stochastic rounding on publication; no full-bank master weights |
| Evolution | Successful hindsight entries are written out as new edges for the whole run; splits (write dispersion, then gradient coherence) add capacity; removal by utility and the byte budget |
| Distributed ownership | DiLoCo learners update locally; one coordinator merges and changes structure; late submissions are age-weighted, not rejected; **no outer KB momentum** |

The transducer is the only retrieval. The earlier plan kept literal-document retrieval (source RAG) for exact passages and citations; it was removed together with the rest of the v9 stack (§10).

## 2. Retrieval and state movement

An edge is $(d_i,k_i,v_i,e_i,b_i)$: departure location, query key, wide value, destination location, and bias. Departures and destinations are unit vectors in the same 256-dimensional location space. Normalize the decoded addresses before use.

### The agenda

Each session keeps an **agenda**: every distinct location its search has reached. Each lookup appends its landing location only if its cosine to every agenda location is below 0.99. Otherwise the agenda is unchanged, and the near-identical location already there stands in for it. This keeps a sink-dominated lookup, which barely moves, from adding a duplicate scan and duplicate candidates. The request's 256-lookup cap bounds the agenda at 257 locations. **Every query searches from all agenda locations at once.** Each location offers its own neighbourhood, and the query's similarity to the keys there decides how much each location contributes. Reach therefore grows with the session, and a late query can still use structure near where the session started or passed through.

### Location neighbourhoods are the only search

Each agenda location $s_j$ has a neighbourhood $\mathcal N(s_j)$: the 1,024 edges whose departures are nearest to it. There is no radius and no empty neighbourhood. Where the bank is dense the neighbourhood is tight; where it is sparse it reaches further, and the departure term of the score (below) makes distant edges pay for that distance. The neighbourhood is a departure shortlist, not the selection: every pair in it is scored with the query, and only then are the best 32 kept. Many edges can therefore share one departure and still each be retrieved by their keys. A shortlist of the 128 nearest would make all but 128 of them unreachable however well they match.

The candidates for a query are the pairs $(j,i)$ with $i\in\mathcal N(s_j)$ for every agenda location. An edge near several agenda locations appears once per location, because reaching it from a different place is a different context. **A query never searches outside the agenda's neighbourhoods.** There is no index over query keys, no global candidate pass, and no global fallback. An edge's successors are the edges whose departures lie near where its transition lands, so the graph is this geometry rather than a separately maintained adjacency list.

A neighbourhood is computed once, when its location joins the agenda, and cached as 1,024 edge IDs. Each lookup therefore scans the departure table only for a newly appended location. Scoring a full agenda's 257 × 1,024 pairs against a 1,024-dimensional query costs about 0.5 GFLOP per lookup. Departures change only at window boundaries (§7), where a session's cached neighbourhoods are recomputed. The scan is exact, over the packed FP8 departure field on device: 256 bytes per edge, or about 0.26 GB and 0.26 GFLOP per scan at one million edges. Exact search is both the production path and the test oracle.

### Location-modulated scoring

Each pair reads the query through its own location:

$$
\tilde q_{t,j}=\operatorname{normalize}\!\big(\hat q_t\odot(1+\tanh(M\hat s_j))\big),
\qquad
z_{ji}=\beta_s\,\hat s_j^\top\hat d_i+\beta_q\,\tilde q_{t,j}^\top\hat k_i+b_i,
\qquad
w_{ji}=\frac{e^{z_{ji}}}{e^{z_\varnothing}+\sum_{(j',i')}e^{z_{j'i'}}},
\qquad
z_\varnothing=b_\varnothing+u_\varnothing^\top\hat q_t.
$$

$M\in\mathbb R^{1024\times256}$ is one shared dense model parameter, initialized to zero; $M\hat s_j$ is cached with each agenda location. Before normalization, $\tilde q^\top k_i=q^\top\operatorname{diag}(k_i)\,(1+\tanh(Ms))$: each edge's query/location interaction is a bilinear form built from its own key and one shared factor. The same query therefore selects different edges at different locations. The departure term keeps each pair's edge close to its own location. The softmax runs over the 32 highest-scoring pairs from all locations' shortlists together, so query–key similarity decides which locations matter for this question. The positive scales $\beta_s,\beta_q$ are model parameters.

### The retrieval sink

The softmax always includes one learned **attention sink**, as in ordinary transformer attention. It is not an empty state and not a gate: every lookup commits its knowledge event and its new location, whatever the sink's share. The sink only gives the softmax somewhere to put mass when nothing nearby fits, so the best pair is not forced to full weight. Its logit $z_\varnothing$ has a learned bias $b_\varnothing$ and a learned query direction $u_\varnothing$, so a query can ask for less. Its value $v_\varnothing$ is a learned dense 32,768-wide parameter, the bank's default knowledge. Its destination is the newest location, so mass on the sink holds the search still. $b_\varnothing$ starts at the mean initial pair score. The sink is a model parameter trained by gradient, never written by hindsight. Before the first edge exists, it is the only entry.

### Transitions are destinations

Each edge stores where it leads. With the bank alone:

$$
\bar v_t=w_\varnothing v_\varnothing+\sum_{(j,i)} w_{ji}\,v_i,\qquad
s_{t+1}=\operatorname{normalize}\Big(w_\varnothing\hat s_t+\sum_{(j,i)} w_{ji}\,\hat e_i\Big).
$$

An edge is a directed graph edge from the region around $d_i$ to $e_i$, answering query $k_i$ on the way. Paths are order-sensitive because the edges available depend on where the search is: after A then B the agenda holds A's and B's destinations and offers their neighbourhoods, which differs from B then A. Each query is also read through each location (above). This is the weighted-automaton form of a transducer, with continuous states. The new location is the softmax blend of the selected destinations, dominated by the pairs that answered the query best. Edges near one another share neighbourhoods, so structure is shared rather than enumerated per path.

While hindsight supplies values (§3), it also steers. Hindsight acts as one virtual pair whose destination is $\operatorname{normalize}(\Pi h_t)$, the same content-derived location a written-out entry would lead to (§6). With $\lambda_t=\sum_{(j,i)}w_{ji}\hat e_i+w_\varnothing\hat s_t$ the bank's blended landing, the new location is

$$
s_{t+1}=\operatorname{normalize}\big((1-\alpha)\,\lambda_t+\alpha\,\operatorname{normalize}(\Pi h_t)\big),
$$

with $\alpha$ the overlay weight: 1 in warm-up and in hindsight-only sequences, annealing to 0 in the overlay, and 0 in bank-only sequences. The search therefore moves through content-derived locations from the very first sequence, and write-outs depart from where earlier hindsight led, so chains of edges form from the start.

**The weights are identical for knowledge and movement.** Keep every location on the unit sphere. Use FP32 norms, scores, softmax, weighted sums and state accumulation. Nonfinite inputs are errors, not values to hide with clamping. Agenda locations older than the unroll window (§3) are detached. An edge's weight $\omega_i=\sum_j w_{ji}$ is its total over its pairs; $w_\varnothing$ is the sink's weight.

### Exploration stays local

During training, use the 30 best pairs plus two edges sampled uniformly from the 1,025th to 2,048th nearest departures of uniformly chosen agenda locations, each paired with its location. Rescore them normally (their departure term is lower), and count this exploration in the budget. These slots let edges just outside the agenda receive gradient and move their departures in.

## 3. Put retrieval through the transformer

An ordinary meridian is one committed set of externally observed or generated megatokens, as in the existing code. There is one search agenda per session, shared across its tracks.

Lookups happen at **latency boundaries**. Real boundaries are completed turns and arrivals of source results. Artificial boundaries are inserted after 8 committed ordinary meridians without one; training samples that interval from 4 to 12. Lookup $t$'s **retrieval group** $G_t$ is the ordinary meridians from its boundary to the next.

At each boundary: select and commit the native retrieval-query event; score the pairs from every agenda location's neighbourhood; commit the knowledge result through C, the central transformer; append its landing to the agenda; then continue. Query and knowledge-result events are internal events and **do not trigger another query**. Use one outstanding state-changing lookup per session. Parallel tracks share its serialized commit order; there are no private agenda branches or branch-merge machinery in this design.

The first ordinary event is a compact metadata prefix containing the available request, task context, topic, language and modality, plus available date/version information. Initialize $s_0$ from that prefix and a learned base vector. **The first query and retrieval pause occur immediately after this first metadata event**, before an answer. Missing metadata uses the same path with explicit missing fields, not a separate startup mode.

A query cannot use the answer it is about to predict. Training targets may update parameters after the forward pass, just as they update transformer weights.

### A fixed wide-value projection

Decode and blend the 32,768-wide values before projection. Divide the resulting vector into **64 rows of width 512**, add learned row identities, apply a small shared adapter, route the rows through the **existing U expert bank**, and use **32 learned facet queries** to gather a native $[32,512]$ block. The facet queries also see one learned sink row, so a facet can take little from the value. Query and search-state summaries condition the routing/readout.

Commit this block as an observed event on a reserved knowledge track. It attends to previous retrieval and generation history through C and writes **ordinary native KV**. Register new adapters/readouts once; reuse the U bank's existing parameter owner and optimizer. Shared U remains part of the frozen realization map for any later latent-policy RL phase.

The query sample is reparameterized during supervised training, so gradients reach P and its embedding realizer. Neighbourhood membership and top-32 selection remain discrete. Unroll **four retrieval groups** for future-loss credit, preserving complete event groups; gradients pass through destinations into later scores, so an edge is credited for the neighbourhoods its destination makes reachable. Recorded-action policy scoring retains exact raw actions and remains distinct from this supervised path.

Retrieval is a blocking boundary. IO can overlap independent work, but a result may never be applied to a different agenda state or halfway through a committed group. On cancellation, publish neither its state change nor its knowledge event. Internal events advance native history, not the external source-availability clock.

### The hindsight track

The hindsight track is a reserved, **training-only** track. For each lookup it produces $h_t$, the knowledge value that would have been ideal, judged by what the training sequence actually does next. This follows the hindsight writer in `~/hindsight-kb`, carried out natively on tracks.

**The hindsight pass** runs C once over the sequence's ordinary events with a **group-inverse mask**. Ordinary events are those observed or generated, never query or knowledge events. Under the mask, an event attends bidirectionally to every event of its own retrieval group, and to every event of later groups. Each group's final C states therefore summarize that group and everything after it. The pass never touches retrieval, so it runs once per sequence, before the causal pass, for all lookups in parallel. Its activations never enter `KVHistory`, and no causal track or serving path can read them. The cost is one C pass over the ordinary events, in training only.

**The hindsight realizer** is a head on C, a sibling of the retrieval-query realizer. Its 64 learned row queries attend over the final states of one group's events, plus one learned sink key and value, and emit $h_t$ in the bank's own format: a 32,768-wide value, packed to FP4 and decoded exactly like a stored edge value. The sink lets a row query take little from a group, and it is the whole input when the group to read lies past the end of the sequence: the last lookups' $h_t$ then comes from the sink alone. Such a **sink-only** $h_t$ still serves as that lookup's value and steering, but it is never written into the bank or written out (§4, §6): it describes no actual future. $h_t$ is the only thing the track emits. It is not conditioned on the query: it is what the future needs at lookup $t$, and queries, keys and locations learn to find it.

Tracks gain a masking mode, implemented as an explicit mask built from per-query and per-key group, meridian and track metadata. This replaces today's purely structural causality. Causal stays the default. Every attention layer of C and P gains one learned, position-free sink key/value slot per KV head, visible to every query under every mask. It is added at attention time and never stored in `KVHistory`. Heads get a place to park attention instead of overloading the metadata prefix or the newest events, and no query row can ever be fully masked. Which group the realizer reads decides what $h_t$ may see, and that follows a curriculum during warm-up:

1. **Full** (first 4,096 steps): the pass is bidirectional over the whole sequence; read $G_t$.
2. **Gap 1** (next 4,096 steps): group-inverse pass; read $G_t$, the group the knowledge serves, and everything after it.
3. **Gap 2** (from then on): group-inverse pass; read $G_{t+1}$, from one group past the one served, following `~/hindsight-kb`'s suffix rule.

Stages 2 and 3 are the same pass read one group apart.

**Training order.**

- **Hindsight warm-up (first 16,384 steps, spanning the mask curriculum).** Every knowledge event's value is $h_t$, read through the ordinary readout and committed through C, and hindsight steers the location (§2, $\alpha=1$). The future loss trains the hindsight track and realizer, the readout, P and C together. From the gap-2 stage on, successful hindsight entries are written out as edges (§6), so the bank exists when warm-up ends.
- **Overlay.** Every lookup's value becomes $(1-\alpha)\,\bar v_t+\alpha\,h_t$ before readout, and its location blends with the hindsight destination by the same $\alpha$ (§2). $\alpha$ anneals linearly from 1 to 0 over 65,536 steps. While no edge exists yet, the bank side is the sink alone. The model keeps receiving good knowledge while the bank learns to supply it.
- **Hindsight write, until the switch.** Every training lookup with a non-sink-only $h_t$ writes it into the edges it used, weighted and blended (§4).
- **After the overlay ends,** one in eight training sequences runs hindsight-only: every value is $h_t$ and hindsight steers ($\alpha=1$). The bank is still scored in these sequences, unused, for the write-out check (§6). This keeps the hindsight track training against the current reader and keeps writing out entries. The other seven use the bank alone, with the hindsight write.
- **Switch to a learned bank (§6).** When the bank has warmed up, values train by gradient and the hindsight write stops. Hindsight-only sequences and write-out continue for the whole run, so new territory keeps gaining edges.

Because the hindsight track is co-trained with the reader, $h_t$ is always in the reader's current language, and every visit writes that language into the bank.

## 4. Low-bit memory that can still learn

Storage layout:

| Field | Representation |
| --- | --- |
| Knowledge values | E2M1 FP4 elements, one E4M3 FP8 scale per 16 elements, one FP32 scale per row |
| Keys, departures, destinations | E4M3 FP8 elements with FP32 scales per 128 elements |
| Biases, write masses, scoring controls, optimizer summaries | FP32 |
| Live query/locations and accumulations | FP32; cast the projected knowledge block to the model's BF16 input format |
| Working cache | FP32 address masters, and before the switch write accumulators or after it value masters, for at most 8,192 edges per learner |
| Native transformer KV / shared experts | Existing BF16 path |

Pack actual bits in storage and on the wire. The forward pass always reads packed-and-decoded rows.

### Writing hindsight into the bank

**Until the switch, values are written, not gradient-trained.** Each edge keeps a write mass $A_i$. Within an optimizer window, every training lookup whose $h_t$ is not sink-only adds $\omega_i$ to the edge's accumulator $W_i$ and $\omega_i h_t$ to its accumulator $H_i$. At the window boundary, with $\rho_w=2^{-\Delta t/16384}$:

$$
A_i'=\rho_wA_i+W_i,\qquad
v_i'=\frac{\rho_wA_iv_i+H_i}{A_i'}.
$$

A value is thus the decayed mean of the hindsight values of the lookups it served, weighted by how much each lookup relied on it, and blended with what it held before. The decay lets values follow the reader as it changes. Each written row is packed once per window with stochastic rounding, seeded from its ID, window and worker. The future-loss gradient still flows through $\bar v_t$ into the mixture weights, keys, departures, destinations and the hindsight track; in this phase it is never applied to values. Written values need no optimizer state and no working masters.

### One rule for fractional updates

A training read loads an edge's trained packed fields into its FP32 working master unless the master is already resident: the address fields (key, departure, destination) always, and the value after the switch. Accumulate all gradients for that row, update the master once, and publish newly packed fields with **stochastic rounding**. Keep the fractional difference in that resident master across local updates. Quantizer scales cover the updated range; reads never change scales or consume rounding randomness. Gradients reach the masters straight through the packed forward pass.

Round once per changed row per update, with a seed derived from its ID, field, update number and worker. Reuse the packed row during forward/recomputation. Evict least-recently-updated masters outside the active backward window; preserve the already published packed fields. At a DiLoCo merge, flush the changed masters and write accumulators, send their results (§7), and restart the cache from the installed bank. Checkpoint resident masters, write accumulators and rounding state for an exact interrupted-run resume.

There is no full-bank master table. Sub-grid changes can be lost after eviction; stochastic rounding preserves them in expectation.

## 5. The KB optimizer

Use **discounted block-scale descent with a global prior**. It has no first-moment vector, no Adam second-moment vector, and no weight decay on idle knowledge. It updates address fields and biases throughout, and values after the switch. Each trained field has FP32 squared-gradient sums $S_{ib}$ for 256-coordinate blocks, an effective observation count $n_i$, and its last-update timestamp. Scalars use one block. Quantizer blocks and optimizer blocks are different things.

For the current accumulated, correctly normalized gradient $g_{ib}$ and elapsed local optimizer steps $\Delta t$:

$$
\rho=2^{-\Delta t/4096},\qquad
m_{ib}=\operatorname{mean}_j g_{ibj}^2,\qquad
S_{ib}\leftarrow\rho S_{ib}+m_{ib},\qquad
n_i\leftarrow\rho n_i+1.
$$

Combine the remaining evidence with four pseudo-observations from the current global prior $P_b$ for that field and block:

$$
V_{ib}=\frac{S_{ib}+4P_b}{n_i+4},\qquad
\delta_{ib}=-\eta\frac{g_{ib}}{\sqrt{\max(V_{ib},10^{-12})}+10^{-8}}.
$$

Use base rates $\eta_f=10^{-4}$ for keys, departures, destinations and biases, and, after the switch, $\eta_v=10^{-3}$ for values, following the same pinned training schedule. Clip value-block update RMS to 1% of $\max(\operatorname{RMS}(v_{ib}),10^{-3})$. For unit addresses (keys, departures, destinations), take the tangent gradient, precondition it, project the update back onto the tangent plane and cap angular movement at 0.01 radians before normalization.

The global prior $P_b$ is a recent mean squared-gradient estimate per field/block, updated once per optimizer window from distinct touched rows with a 256-step half-life, starting at $10^{-6}$. Each update uses the previous window's prior. It gives a new or rarely seen row a sensible step scale until its own history accumulates. At the switch, value statistics start from it.

Discount numerator and count together. After a long absence, confidence returns toward the current prior; an old momentum direction cannot reappear because none is stored. A computed zero gradient is an observation. A row that was not trained is not a stream of zero-gradient observations. Preserve global task-loss normalization; do not divide by visit frequency or retrieval weight and accidentally amplify weak evidence.

### Staleness

There is no replay. Old rows and late updates stay usable as follows:

- **Values,** until the switch, are rewritten from the current hindsight value on every visit (§4), and older writes decay, so a row returning from dormancy moves to the reader's current language at once. After the switch, every visit's gradient realigns them.
- **Keys** are never the gate to retrieval. Candidates come from the agenda's locations, so an edge with an outdated key still enters the mixture whenever its location is visited, and receives the gradient that realigns it.
- **Statistics** are discounted sums of observations: optimizer block sums here, structural statistics in §6. A late contribution is still valid evidence, and its age is applied as discount.
- **Distributed updates** arriving late are age-weighted rather than rejected (§7).

## 6. Grow from hindsight, evolve continuously

The bank starts empty and grows within a configured packed-byte budget, with a one-million-edge ceiling for the first large run. There is no separate seeding procedure: the bank is built by writing out successful hindsight entries.

### Writing out hindsight entries

From the gap-2 stage of warm-up on, every training lookup whose forward pass includes $h_t$ is a write-out candidate, unless its $h_t$ is sink-only. These are lookups in warm-up, in the overlay, and in hindsight-only sequences, before and after the switch. With $g_t$ the future-loss gradient with respect to the value entering the readout, the first-order benefit of the hindsight value is

$$
\beta_t=-g_t^\top h_t,
$$

one dot product the backward pass already supports. A lookup is written out when $\beta_t>0$, so hindsight helped, and the bank does not already supply it: there are no candidates yet, or $\|h_t-\bar v_t\|^2>0.25\,\|h_t\|^2$, where $\bar v_t$ is the bank's own blend whether or not it entered the forward pass. After that step's backward pass, it becomes a new edge:

- $d=\hat s_t$ and $k=\tilde q_{t,t}$: the edge answers this query from the location the search actually reached, which hindsight steering has already moved along content-derived locations;
- $v=h_t$, with write mass 1 before the switch;
- $e=\operatorname{normalize}(\Pi h_t)$, where $\Pi\in\mathbb R^{256\times32768}$ is one fixed seeded Gaussian projection;
- the bias makes its score equal to the best score among this lookup's pairs, or zero when there were none.

The destination is a location derived from the knowledge just retrieved, the same place hindsight steering sent the search, so the new edge reproduces the move hindsight made. Sessions that retrieve similar knowledge therefore land near each other and share the edges written there. This is how the transducer's initial geometry arises: positions come from content, and connections come from the order in which hindsight was useful. Destinations, keys and departures then train like any others.

The hindsight value is used only to set parameters after the prediction, as §3 permits. Each learner writes out at most 512 entries per synchronization round. Write-outs are provisional rows in the learner's working cache, with learner-local IDs. They are trained like any resident edge and submitted at the round boundary (§7).

### Structure from statistics training already computes

Structural decisions use no buffer, no candidate search and no extra forward passes. On every lookup, the backward pass already has the future-loss gradient $g=\partial L_{\mathrm{future}}/\partial F$ for the mixture output $F=(\bar v_t,\ w_\varnothing\hat s_t+\sum w_{ji}\hat e_i)$, which includes the sink's output $(v_\varnothing,\hat s_t)$, each selected edge's output $u_i=(v_i,\hat e_i)$, and its weight $\omega_i$. Removing edge $i$ would renormalize the mixture to $(F-\omega_iu_i)/(1-\omega_i)$, so to first order it would change the loss by

$$
\Delta_i=\frac{\omega_i}{1-\omega_i}\,g^\top(F-u_i).
$$

That is one dot product per selected edge. Each edge accumulates, with a 16,384-local-step half-life, its **utility** $\mathcal U_i$, the discounted sum of $\Delta_i$, and its discounted visit count $n_i$. Before the switch it also accumulates its **write dispersion** $D_i$, the discounted sum of $\omega_i\|h_t-v_i\|^2$. After the switch, $D_i$ is replaced by a **gradient sketch** $G_i\in\mathbb R^{64}$, the discounted sum of $R\gamma_{i,t}$ where $\gamma_{i,t}$ is the loss gradient with respect to all its fields and $R$ is one fixed seeded random projection, and a **gradient energy** $Q_i$, the discounted sum of $\|R\gamma_{i,t}\|^2$. The coherence $\kappa_i=\|G_i\|^2/(n_iQ_i)$ is near 1 when an edge's gradients agree and near 0 when they cancel.

At every round boundary, the coordinator applies these rules from the merged statistics:

| Change | Rule |
| --- | --- |
| Split, before the switch | $n_i\ge64$ and $D_i/A_i>0.5\,\|v_i\|^2$: the hindsight values the edge serves spread widely around it, so it serves contexts that want different knowledge. |
| Split, after the switch | $n_i\ge64$, $\kappa_i<0.1$, and mean energy $Q_i/n_i$ in the bank's top decile: the edge is pushed hard in directions that cancel. |
| Remove | The edge is at least 4,096 local steps old and its utility is not positive. |
| Budget | Above 95% of the byte budget, remove the eligible edges with the lowest utility per byte until the bank is at 90%. |

A split duplicates the packed codes, halves the write mass into each child, subtracts $\log2$ from both biases, and perturbs the two children's departures and keys in opposite random directions by 0.01 rad. Training separates them. Redundant edges need no merge rule: they divide their utility, and Remove or the budget retires whichever no longer pays. Children start from their parent's optimizer and structural statistics, halved; new edges start with the global prior and zero statistics. Retiring an edge removes its statistics and departure-table entry. The final evaluation set never determines individual edges.

### The switch to a learned bank

The bank switches from written to learned values once, when both hold:

- it holds at least 262,144 edges;
- bank-only continuation loss on the fixed validation stream, evaluated every 1,024 steps, has improved by less than 0.5% across the last 16 evaluations.

At the switch, the hindsight write stops (the overlay has already ended). Values become gradient-trained fields with value statistics starting from the global prior. Write masses and write dispersion are dropped. The gradient sketch and energy start from zero, and the post-switch split rule applies. Hindsight-only sequences continue, one in eight, with the hindsight pass running only for them. They keep the hindsight track current and keep writing out entries, whose values then train by gradient like the rest. New capacity therefore comes from both write-out and splits for the whole run, and data seen after the switch can still claim new territory.

## 7. DiLoCo and storage ownership

Reuse the current storage, sparse update and DiLoCo infrastructure; replace its IVF index, full-vector KB optimizer state, demand forwarding and point/pair semantics. Learners keep only their own compact statistics and bounded active cache. The coordinator owns IDs, canonical packed rows and structure.

Use **32 local steps per KB synchronization round**. A learner installs the newest published snapshot at its next window boundary and submits its deltas relative to the base it trained from. A submission $\alpha_w$ rounds behind the current canonical version merges with weight $2^{-\alpha_w/8}$. Late work is discounted, never refused. A committed per-worker sequence ledger rejects duplicates, and deltas for edges retired since their base are skipped.

Workers send $\delta_{wi}=\theta^{\mathrm{local}}_{wi}-\theta^{\mathrm{base}}_i$ for changed rows of trained fields. Merge with $W_w$ = processed examples $\times\,2^{-\alpha_w/8}$ across **all workers merged this round**, with zero for an untouched row, then repack:

$$
\theta_i^{\mathrm{new}}
=\theta_i^{\mathrm{base}}+
\frac{\sum_w W_w\delta_{wi}}{\sum_w W_w}.
$$

Do not average only over workers that touched the row, which would silently change its effective learning rate. **There is no outer KB momentum or Adam state.** These deltas are not gradients and are not inputs to the block squared-gradient estimator. Dense model fragments keep their existing optimizer/merge machinery but publish under the same compatible round identity.

Before the switch, written values merge as weighted means, not deltas. For each written row, a learner sends its window-summed write mass $W_w$ and the mean value it wrote, $H_w/W_w$, packed to FP4. The coordinator ages $W_w$ by $2^{-\alpha_w/8}$ into $W_w'$, decays the canonical mass by the steps elapsed since its last write, and blends:

$$
A_i\leftarrow\rho_wA_i+\sum_wW_w',\qquad
v_i\leftarrow\frac{\rho_wA_i^{\mathrm{old}}v_i+\sum_wW_w'\,(H_w/W_w)}{A_i}.
$$

This is the write rule applied across learners, so concurrent and late writes blend in proportion to the evidence behind them.

Learners also send their increments to each touched edge's optimizer and structural statistics. These are discounted sums, so the coordinator ages each increment by its submission's age and adds it.

Written-out edges have no baseline, so they bypass the delta formula. Each learner submits their decoded fields, write mass, statistics and processed-example weight. The coordinator combines written-out rows from different learners whose departures and keys both have cosine at least 0.9: values by write-mass-weighted mean, other fields by example-weighted mean. It then packs them, assigns canonical IDs and publishes them with the round, applying the budget rule if needed. Learners create provisional rows; only the coordinator makes them canonical. The coordinator evaluates the switch conditions, decides the switch and publishes it with a round.

Hold a coherent row snapshot through each forward/backward window. Update packed rows, written values, optimizer statistics, the device departure table and sessions' cached neighbourhoods together at its boundary. Departures are therefore fixed within a window, and the table never lags the rows it indexes; never perform fused row updates halfway through a recurrent backward pass. Save model, KB, learner summaries, active fractional progress, structural cursor and random state as one consistent checkpoint. On a partially failed step, resume the last complete checkpoint.

## 8. Build the selected system

Target architecture **v10**, new packed-edge layout, and new retrieval-event/checkpoint identities. Reject old point/pair rows and optimizer moments. Reuse compatible v9 model weights by an explicit named weight import; construct the new bank and optimizer fresh. Preserve the recently integrated media input/realizer path.

| Work | Repository owners and deliverable |
| --- | --- |
| Transducer and agenda search | `memory/`: packed rows with destinations and write masses, the session agenda with distinct-location appends and cached 1,024-nearest departure shortlists, exact departure scan, per-location query modulation, pair softmax, value/destination mixtures and rank-band exploration |
| Native model integration | `model/config.py`, `core.py`, `multitrack.py`: search state, reserved knowledge track, shared-U readout, one normal C commit; per-track masking modes with an explicit group/meridian/track mask; the reserved training-only hindsight track (full and group-inverse masks) and its hindsight realizer on C emitting FP4 bank-format values |
| Training data and recurrence | `data/`, `training/engine.py`, `meridians.py`: metadata prefix, first retrieval boundary, latency-boundary schedule with artificial boundaries, four-group credit and causal labels; hindsight pass, mask curriculum, warm-up, overlay schedule with hindsight steering, hindsight-only sequences, sink-only exclusion, validation cadence and the switch |
| Sparse optimizer and growth | `memory/transducer_optimizer.py`, `transducer_growth.py`, `training/transducer.py`: block algorithm, bounded master cache, hindsight write and write-out, utility, dispersion and sketch statistics, split, remove and the budget rule |
| Distributed execution | `training/transducer_diloco.py`, `transducer_sync.py`, checkpoint/runtime owners: age-weighted deltas, write-mass value blending and statistics without outer KB momentum, written-out row reconciliation (replacing learner demand forwarding), the switch, packed publication and exact resume |

The direct semantic-KV projector, semantic KB LoRA controllers, point/pair dependence and writer-reproduction launch requirements are deleted. The hindsight track supersedes the v9 source and relation writers, which are deleted too. Ordinary transformer KV, native parameter ownership and exact action tapes stay.

Build a small executable reference of this same design, fix its correctness failures, then run the identical path at scale.

## 9. Resource envelope and run discipline

For one million edges with the selected geometry, calculated storage is approximately:

| Allocation | Written phase, GB | Learned phase, GB |
| --- | ---: | ---: |
| FP4 values, 16-element scales and one row scale | 18.436 | 18.436 |
| FP8 keys/departures/destinations and 128-element FP32 scales | 1.584 | 1.584 |
| FP32 block statistics (address fields; all fields after the switch) | 0.024 | 0.536 |
| 8,192-edge FP32 working cache | 1.124 | 1.124 |
| Per-edge structural statistics | 0.024 | 0.276 |
| **Total** | **21.19** | **21.96** |

These totals exclude biases, counters, gradients, activations, replicas and checkpoints. Per-worker summaries/caches and baseline/delta buffers count against the machine budget. The device-resident FP8 departure table that the neighbourhood scan reads is about 0.26 GB with scales; there is no query-key index. A full agenda holds 257 locations with cached 1,024-edge shortlists and modulations, about 2.4 MB per session.

At the existing native geometry, each knowledge event adds 32 KiB of BF16 KV and a C pass. Use the existing whole-meridian history bound. Cap a request at 256 retrieval opportunities; after the cap or an explicit retrieval timeout, continue without changing the agenda.

**Check correctness before burning the run, and measure quality during it.** Required automated checks are agenda-only candidates (no code path reaches an edge outside the agenda's neighbourhoods or the training exploration band); cached neighbourhoods equal fresh scans after each window boundary; every shortlisted pair scored before top-32 selection; agenda appends only below cosine 0.99; hindsight steering by $\alpha$ and its absence in bank-only sequences; sink-only $h_t$ excluded from writes and write-outs; write-out continuing after the switch; location-modulated scores against an unfused reference; order sensitivity of paths; matched value/destination weights; the retrieval sink in every softmax and its hold-still destination; C/P sink slots under every mask, absent from `KVHistory`; no fully masked attention row; the realizer's sink-only output past the sequence end; write-out conditions and fields, including the content-derived destination; written-out row reconciliation; utility, dispersion and coherence statistics against brute-force removal and gradient sets on a small bank; split, remove and budget rules; hindsight masks (full and group-inverse, the realizer reading $G_t$ or $G_{t+1}$ by stage, and no hindsight activation reachable from any causal track or the serving path); artificial-boundary scheduling; FP4 round-trip of $h_t$; overlay schedule; the write rule equal to the decayed weighted mean, with no optimizer step on values before the switch; cross-learner value blending equal to a single learner writing the same lookups; the switch conditions on the 1,024-step validation cadence and state hand-over; age-weighted late merges of deltas and statistics; no-recursion; metadata causality; query and selected-row gradients; packed-forward/recomputation identity; small-update and cache-eviction behavior; dormant-row return updates; split behavior; sparse merge normalization; and checkpoint recovery. Finite-difference tests apply to the decoded differentiable reference; test the quantization surrogate separately.

During training log continuation loss, multi-hop completion, rare-task retention, first-visit recovery after dormancy, nearest-departure distance per location, write-out rate and the survival of written-out edges, agenda size and the share of weight by location age, hindsight-assisted versus bank-only loss gap, write residual $\|h_t-\bar v_t\|$ by edge age, sink weight in retrieval, readout and realizer, and C/P attention mass on sink slots, split and removal rates, utility distribution, merge age distribution, realized versus intended updates, edge/byte growth, latency and all memory tiers. Observe on a fixed validation stream; keep test targets outside training and growth.

## 10. Implementation notes

Where the code had to decide something this plan leaves open, it decided as follows.

- **Always on; v9 deleted.** Every model builds `model.transducer` from its `transducer_*` geometry fields; there is no switch. The whole v9 stack is gone: the free KB and its sparse sync, source RAG with its catalog and search index, the semantic K/V projector, memory LoRA controllers, the source/relation writers and the embedding-alignment sidecars. The encoder's embedding head stays, trained by the encode-and-reconstruct stage.
- **Synchronous ranks.** torchrun ranks are learners of one canonical bank kept on rank 0: every 32 steps their rounds are gathered there, merged exactly as DiLoCo rounds are, and the change set is broadcast back. A multi-rank checkpoint holds the canonical bank; on resume each rank's working copy restarts from it, so up to 31 steps of unmerged sparse progress are not carried over.
- **Scoring caches.** Beside the packed columns the table keeps each edge's unit departure (FP32) and unit key (BF16; an E4M3 code is exact in BF16, only its scale rounds), rewritten whenever the field is. The exact neighbourhood scan and shortlist scoring are then plain matrix products. They add 3 KiB per edge at the default geometry (about 15% of a 20,024-byte packed row), outside the packed-byte budget and never saved. Only selection reads the BF16 keys; the selected pairs' logits are recomputed exactly from the packed rows and FP32 masters.
- **Sparse bookkeeping.** Optimizer statistics, pending writes, round origins, structural evidence and FP32 masters are dense per-edge row stores, not per-edge objects; a training lookup gathers each field's masters into one leaf, and the step sums the window's leaf gradients per edge.
- **Removal clock.** During warm-up the bank's output never reaches the loss ($lpha=1$), so utility cannot be observed and every early write-out would look useless. The Remove and Budget rules therefore count an edge's age from the later of its birth and the overlay's start.
- **Where lookups run.** A lookup planned after meridian $m$ executes just before meridian $m+1$ is committed, under that meridian's gradient mode, so the lookup preceding the first scored meridian of a window is trained. A boundary after a window's last meridian serves nothing in that window and is not planned.
- **Hindsight pass positions.** The pass gives each event its own temporal position. With one time per slot, the C/P attention sink's logit is added exactly as a mask column, so it stays position-free.
- **Coordinator change sets.** The coordinator publishes changed rows per version and keeps the last 64 versions; an older learner downloads the whole bank. Its full state is persisted every 16 merges; after a coordinator restart learners re-base from a snapshot.
- **Learner installs.** A learner installs a newer canonical version between optimizer steps. Rows the coordinator republished become that learner's new round base; its provisional rows are dropped (the coordinator adopted or rejected them).
- **Several mounted bases (see the [multimodal knowledge plan](multimodal-knowledge-plan.md)).** The bank described above is the *trained base*, the only one the coordinator, DiLoCo sync, checkpoints, growth and the sparse optimizer know. A session may mount further, read-only *ingested bases* (`memory/knowledge.py`) with blend weights; a lookup scans every mounted base's departures, adds each base's log weight to its pairs' scores, and keeps the best 32 pairs of all of them in one softmax with the one sink. The exact scan, the 1,024-neighbour shortlists, the 30 + 2 exploration rule (exploration draws from the trained base only), the blend and the landing are otherwise as written in §2, and a session that mounts the trained base alone behaves as before.
- **What a lookup reports.** Its selected pairs are `(location, base, edge)`; `bank_value` is the blend over every mounted base (the sink included), the single tensor the readout and the adapter router read; `edge_ids` and `edge_weights` list the trained base's selected edges only, since only they are written, trained or evolved. A training window that mounts another base writes nothing into the trained base.
- **Agenda caching.** An agenda caches each location's shortlist per mounted base and the version it scanned at; a base that changed is rescanned and the others are not. A lookup computed before any mounted base changed cannot commit after.
- **Ingest heads.** The ingest realizer (the hindsight realizer's architecture, its own weights) and the key realizer read C's states of one item under the full mask and are part of the served model, unlike the hindsight head. An entry departs from where the item's metadata starts a session, arrives at the content location of its value, and has bias zero. Their two training losses (value distillation to $h_t$, key contrast against the lookup's query) run in the hindsight pass, for lookups whose hindsight group is or contains exactly one item.
