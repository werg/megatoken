# Low-bit storage and optimization for the transducer KB

**Research checked:** 21 October 2026. **Status:** design and experiment specification, not an implemented or benchmarked optimizer. Companion to the [continuous-state transducer plan](continuous-state-transducer-plan.md).

## 1. Decision

Make **four-bit storage of the large, cold knowledge values** the target. Do not require four-bit arithmetic or four-bit optimizer statistics everywhere. Separate persistent storage, temporary working precision, and compute formats.

The most promising starting point is **sparse SGD versus row/block-wise adaptive learning without a full momentum vector**, with explicit forgetting of obsolete scale estimates. Test a locality-based prior on gradient scales after establishing the simpler baseline. Keep small optimizer statistics in FP32: a few reliable numbers are cheaper than two quantized vectors for every edge.

There is no established winner for the exact combination of an evolving recurrent KB, highly uneven visits, changing query/value interfaces, asynchronous updates, and four-bit persistent parameters. Recent work supplies strong components, not a validated complete solution. The drift-aware optimizer below is our proposed synthesis, not a published state-of-the-art claim.

## 2. What the current code actually does

At the reviewed [repository snapshot](https://github.com/werg/megatoken/tree/331777a0fb569104bfc8b958d8fc734381e8fd65), [SparseRowAdam](https://github.com/werg/megatoken/blob/331777a0fb569104bfc8b958d8fc734381e8fd65/src/megatoken/memory/optimizer.py) creates FP32 leaves only for selected row fields and stores full FP32 first and second moments for fields that have been updated. It is already sparse and lazy; it does not eagerly allocate moments for every untouched row.

Its bias-correction counter advances on **visits**, while the saved global step is checked for update ordering. Long periods without visits do not age the stored moments. The optimizer rejects rows changed during a training window, which is useful, but is not a solution to changes in the shared model during a long absence. As more rows are eventually visited, the two full moment arrays become a major storage cost.

The [DiLoCo KB protocol](https://github.com/werg/megatoken/blob/331777a0fb569104bfc8b958d8fc734381e8fd65/src/megatoken/training/diloco_kb.py) exchanges changes from a baseline, not individual fresh gradients. Local optimization and coordinator merging therefore need separate semantics. This document changes the proposed successor, not the supported v9 runtime.

## 3. What is genuinely relevant in the literature

| Evidence | What it supports here | Important limit |
| --- | --- | --- |
| **FBGEMM table-batched embedding optimizers** [1] | Row-wise AdaGrad, row-counter variants, and partial row-wise Adam are implemented approaches for large sparse tables. | Row-wise Adam still retains a full first-moment vector. Fused immediate updates and supported weight formats are not automatically compatible with our recurrent training windows or FP4. |
| **Continual Learning via Sparse Memory Finetuning** (2025) [2] | In its sparse-memory experiments, SGD reduced forgetting relative to AdamW. SGD is a serious zero-state baseline, not a placeholder. | Finetuning selected memory values is not joint pretraining of moving addresses, residuals, and a changing reader. |
| **Engram** (2026) [3] | A recent large conditional-memory system uses Adam without weight decay for its memory embeddings, while using Muon for the backbone. | Contemporary practice does not establish one universal sparse-memory optimizer. |
| **Matrix AdaGrad**, revision 25 September 2026 [4] | Online mirror-descent analysis gives a basis for row/column adaptation and advantages under specified structured gradient conditions. | Its guarantees do not automatically transfer to discounted statistics, discrete retrieval, or a nonstationary recurrent model. |
| **SM3 and Adafactor** [5,6] | Compress the structure of adaptive statistics rather than just their numerical precision. Both deserve comparison. | Sparse observations and changing distributions require care; global column statistics can be dominated by frequently visited rows. |
| **DeepRec AdaGradDecay** [7] | Discounting old AdaGrad accumulation using elapsed global training time has a practical sparse-learning precedent. | It does not solve shared representation drift or delayed worker gradients by itself. |
| **Optimizer-state staleness study**, revision 27 September 2026 [8] | Quantized moment updates can round away, making the effective memory of the optimizer too long. Stochastic rounding and resets are studied remedies. | Numerical stalling is different from a parameter being unvisited. |
| **ZIP-SR / ZE-EDEN**, 8 October 2026 [9] | Four-bit AdamW state quantization can be designed around the adaptive update, not just moment reconstruction error. | Tested dense-model settings are not our sparse KB. The reported setup retains FP32 master parameters. |
| **NVFP4 pretraining and Full-Stack FP4** [10,11] | Four-bit computation and specialized low-bit state formats are increasingly credible research tools. | The NVFP4 recipe retains FP32 main weights/states; Full-Stack FP4 retains BF16 master weights. Neither establishes a persistent, master-free FP4 KB. |
| **Adaptive Low-Precision Training of Embeddings** [12] | Low-bit tables can be decoded, updated, and requantized only where accessed, without a full high-precision embedding-table copy. | Its results do not establish lossless four-bit learning under extreme dormancy. |

Muon, SOAP/Shampoo-style matrix methods, and newer dense optimizers should not be selected just because they win backbone benchmarks. AF-Muon explicitly studies embedding matrices [13], so embeddings are not categorically excluded. However, its tied-embedding setting is not a mostly dormant, dynamically growing external table. Full matrix coupling and persistent state need a separate cost/quality justification here.

Adam-mini is also not a direct answer: its published recipe treats embedding/output blocks specially rather than indiscriminately replacing their second moments with one scalar [14]. Conversely, plain row-wise statistics are not automatically too crude: the KB already has a strong row-local access structure.

## 4. Precision should follow function

### Target representation

| Component | Proposed initial format | Reason |
| --- | --- | --- |
| Cold, wide values `v` | Block-scaled FP4; compare NF4 and INT4 codecs | Largest persistent allocation; quality must be tested through the actual projection and continuation. |
| Stored keys `k` and departures `d` | FP8, with decoded normalization; four-bit alternatives as an ablation | Small relative to values, but errors change which edge is selected. |
| Stored state residuals `r` | FP8 initially; promote sensitive rows to BF16 | Errors propagate over successive search steps. |
| Live query and agenda state | BF16 or FP32; FP32 norm and update accumulation | Small working set with recurrent numerical sensitivity. |
| Similarities, softmax/null, weighted sums | FP32 reference accumulation | Do not make the recurrence four-bit just because its storage is four-bit. |
| Optimizer scale statistics, counts and timestamps | FP32 statistics and integer counters | Tiny compared with values; preserve reliability. |
| Frequently updated rows | Strictly bounded BF16/FP32 working cache | Preserve small updates during bursts without retaining a master copy of the entire bank. |
| Shared MoE and projection matrices | Supported FP8/NVFP4 compute recipes, with explicitly counted higher-precision state | Dense compute optimization is separate from compressing the KB. |
| Native transformer KV | FP8 first; four-bit cache experiment separately | Changes attention/history accuracy; does not restore the removed direct semantic-KV path. |

Scalars such as null logits and temperatures need not be quantized to four bits. Original source bytes stay lossless. Exact selected-action replay tensors cannot be lossily compressed and still be described as the same Gaussian action; retain their exact representation or introduce a separately evaluated policy/tape change.

### FP4 does not mean four bits of total storage

NVFP4 uses four-bit E2M1 elements with an FP8 scale per 16 elements and an additional higher-level scale. MXFP4 uses a scale per 32 elements. Their effective element costs are approximately **4.5** and **4.25** bits before the remaining metadata, not four [10].

Use independently scaled rows or small pages, not a single scale spanning the entire evolving bank. A changed row must not require rewriting unrelated rows. Record the codec and scale layout in the model/checkpoint format, not in a separate semantic approval system.

NF4 is a four-bit codebook, not the same format as hardware FP4. INT4, NF4, and FP4 can all be reasonable storage choices for a gather-and-weighted-sum kernel. Native FP4 matrix throughput does not automatically accelerate sparse retrieval. Memory Layers at Scale identifies the weighted memory lookup as a bandwidth-sensitive operation [15]. Measure packed bytes fetched, decoding cost, and complete retrieval latency.

Decode selected values and accumulate their weighted sum in FP32; project the resulting wide mixture afterward. Preserve exactly the same retrieval weights for values and state residuals. FP32 rescoring of decoded FP8 addresses is exact for those decoded addresses, not a recovery of an original FP32 table.

A shared fixed orthogonal transform can spread value outliers before quantization. Because it is shared, linear mixtures remain compatible; absorb its inverse into the projection or decode it once after summation. Do not assign unrelated bases to edges and then mix their codes as though their coordinates agree. Test the transform rather than assuming it helps.

### Use search-aware compression for the index

Compressing an ANN index is a separate choice from compressing trainable parameters. Compare score-aware quantization such as anisotropic vector quantization [16] and multi-bit RaBitQ/GPU implementations [17] against an FP8 joint-address reference. Count both canonical addresses and any additional index codes.

Preserve the joint inner-product score, including the edge bias. Normalizing a concatenated vector with a varying bias changes the ranking unless the transformation is designed to preserve that score. The two scalar similarity weights can be applied on the query side, avoiding index rewrites solely because those weights change. Changing stored addresses or biases still requires consistent index updates.

## 5. Four-bit learning must preserve small updates

A compact serving copy with a permanent FP32 master for every value is **not** the target storage design. Use a packed canonical cold bank, selected-row working tensors, and a bounded hot cache. During training, the forward path must see the same quantization effect intended for deployment; fake quantization/straight-through gradients are approximations and must be reported as such.

At a completed optimizer boundary: sum contributions for each selected row, compute the update in FP32, update its working value, then publish its packed representation. Read-only retrieval must not randomly requantize or change the bank. Seed stochastic rounding by update identity, not by incidental read order. Sparse embedding low-precision training provides a precedent for this read/update/repack pattern [12].

The central problem is an update smaller than a representable interval. Under stochastic rounding with fixed grid spacing `Delta`, a desired move of size `delta << Delta` changes a stored scalar with probability roughly `|delta|/Delta`. The expected wait is therefore about `Delta/|delta|` **visits**. For an illustrative spacing of 0.1 and desired update of 0.0001, that is about 1,000 visits. Unbiasedness in expectation does not guarantee timely adaptation of a rare edge. Saturation, clipping, and changing scales complicate it further.

Compare three remedies: retain fractional progress in the bounded working cache; store an explicitly budgeted compressed compensation residual for selected rows; or promote persistently stalled rows to FP8/BF16. A compensation vector is real storage and must be counted. Promotion cannot reconstruct information already discarded; it allows subsequent learning to retain more detail.

Measure quantized-update survival, actual movement versus intended movement, and loss after a cold row returns. Quantization of moment recurrences has its own stalling problem [8]. ZIP-SR/ZE-EDEN address how second-moment errors affect the denominator [9]; they are useful experimental Adam comparators, not a reason to quantize a handful of cheap block statistics.

## 6. Separate four kinds of staleness

| Problem | What has become old | Appropriate response |
| --- | --- | --- |
| Long absence | A row's optimizer history | Discount confidence in old statistics; avoid replaying an ancient momentum direction. |
| Representation drift | The shared query/state encoder or value projector | Rehearsal, sampled cold-row refresh, exploratory retrieval, or slower interface updates. |
| Delayed worker update | A gradient or parameter delta computed using older parameters | Limit delay, compare compatible baselines, and recompute or reject seriously outdated updates. |
| Quantization stalling | Moment/parameter changes smaller than storage resolution | Better rounding/scales, retained fractional progress, or precision promotion. |

A newly computed gradient for a long-unvisited row is **fresh**, even though the row itself is old. Conversely, reducing an old gradient's magnitude does not make its direction current.

No optimizer can repair a row that is never selected again because the query space has moved away from its address. Independent exploratory candidates and small fresh replays are therefore necessary controls. Similarly, a changed projector can reinterpret an untouched value. This is a shared-interface issue, not evidence that every edge needs a named or versioned semantic anchor.

Inactivity alone is not proof that knowledge has become wrong. Discount optimizer confidence, not the value itself. Keep automatic global weight decay off for the initial latent-bank experiments. Touch-only decay and elapsed-time decay implement different objectives; neither should be introduced accidentally.

## 7. Proposed optimizer: small, current scale estimates

### Baseline and candidate

First compare momentum-free SGD and row/block-wise AdaGrad against the existing touched-row Adam reference. Then test discounted block-wise adaptation. Use **256 value coordinates per block** as an initial engineering choice, sweep 128/256/512, and handle keys, departures, residuals, and scalar fields separately. Quantizer block size and optimizer block size need not match.

The proposed method maintains a squared-gradient summary `S_ib` for each edge/block and an effective observation count `n_i` for each row field. It has **no full first-moment vector**. This removes both most optimizer storage and an old directional estimate that could be misleading on a return visit.

For a current, globally normalized, accumulated gradient `g_ib`, calculate:

\[
e_{ib}=\operatorname{mean}_j g_{ibj}^{2},\qquad
\rho_i=\exp[-\Delta t_i/T_f-\kappa_f D_i],
\]
\[
S_{ib}\leftarrow\rho_i S_{ib}+e_{ib},\qquad
n_i\leftarrow\rho_i n_i+1.
\]

`Delta t_i` is elapsed committed training progress since the last update, not time spent waiting in a network queue. `D_i` is an optional estimate of shared-interface change. Start with time-only discounting, and compare it with no discount and a drift-based version. A small fixed set of training probes can produce a cumulative change measure; store only each row's last scalar cursor. Do not build a permanent neighborhood-management service.

After a long absence, the effective count falls as well as the numerator. Otherwise forgetting can make the denominator arbitrarily tiny and cause an excessive first update. This formulation estimates a conditional gradient scale from observed visits; it is **not** ordinary cumulative AdaGrad with a new name.

With a prior scale `V_prior,fb`, form:

\[
V_{ib}=\frac{S_{ib}+\nu V_{\mathrm{prior},fb}}{n_i+\nu},\qquad
u_{ib}=-\eta_f\frac{g_{ib}}{\sqrt{\max(V_{ib},v_{\min,f})}+\epsilon_f}.
\]

The prior has `nu` pseudo-observations. Use a recent global per-field/block prior first. Locality, discussed next, is optional. Clip the RMS of the proposed step to a field-specific trust bound before applying it. For normalized addresses, project the proposed step onto the tangent plane and renormalize, with an angular-step bound. Bound residual changes separately because they alter a recurrent path.

One observation means one valid aggregated gradient for a row field at the declared update boundary. A true computed zero gradient is an observation; a row not read or trained is not thousands of observations of zero. Do not divide the task gradient by visit frequency or retrieval weight unless the training objective explicitly requires that reweighting. In particular, small softmax responsibilities can produce small noisy gradients: adaptive normalization must not turn every weak edge into a large intervention.

The formula changes optimizer history, not the target loss or its global normalization. If the bank uses sampled or assisted candidates, retain the existing distinction between the sampled objective and full-corpus retrieval; a new optimizer does not correct candidate-selection bias automatically.

### Why this is principled, and what is not proved

For a positive block-diagonal scale matrix `P`, the unquantized unconstrained update minimizes a local quadratic model:

\[
u^*=\arg\min_u\left[g^Tu+\frac{1}{2\eta}u^TPu\right]
=-\eta P^{-1}g.
\]

Thus `g^T u* <= 0` for a fresh gradient: it is a descent direction in the local first-order approximation. This is the adaptive-metric/mirror-descent motivation behind row-wise methods [4]. It is not a guarantee for a finite step through a nonconvex recurrent model, hard candidate selection, or stochastic requantization.

Discounting follows the practical need for plasticity under drift [7]. The prior is statistical pooling for poorly observed rows. The combination, discount schedule, clipping, quantized update, and dynamic region assignment are **new design choices to test**. They do not inherit all regret guarantees of AdaGrad, SM3, or multitask mirror descent.

A cheap method without vector momentum may need more updates than Adam. That is an empirical trade-off, especially during initial pretraining. Keep an eight-bit Adam and a four-bit-state Adam comparator rather than ruling them out in advance.

## 8. Use spatial locality to share confidence, not answers

The user's locality idea is promising because nearby state/query addresses may encounter similar downstream computations. However, proximity is not evidence that their desired updates point in the same direction. Two nearby edges may distinguish different answers or outgoing state changes.

Start by sharing **nonnegative gradient-scale estimates**, not averaging values, keys, residuals, or gradients. Reuse an existing coarse ANN cell as a source of a recent per-field/block prior. Shrink a rarely visited row toward that estimate; let its own data dominate as observations accumulate. Limit each row's contribution to regional statistics so one frequently read edge does not determine the whole region. Fall back to the global prior when a region has little evidence, and treat a moved row's regional prior as a prior, not as inherited personal history.

Measure whether locality predicts gradient scale better than a global estimate. Include nearby edges with conflicting update directions. If sharing provides no benefit, remove it. Spatial borrowing cannot be justified solely by prettier state clusters.

SM3 offers a separate, more compressed route: statistics on overlapping parameter groups rather than one vector per parameter [5]. A row/group cover based on spatial locality is an experiment, not the unchanged published SM3 algorithm. Multitask mirror descent gives theoretical support for benefiting from related tasks under explicit similarity assumptions [18]; it also underlines why arbitrary neighborhood smoothing is not automatically beneficial.

A second useful locality experiment is a small set of refresh probes around a returning edge. Use them to estimate how much the local reader changed, not to update every neighbor with the same gradient. The cheap field/global prior should remain a competitive baseline.

## 9. Distributed training and changing graph structure

Keep one update per selected field after its full declared accumulation window. A fused embedding kernel that mutates rows during backward cannot simply be dropped into a recurrent graph that still references those rows [1]. Preserve repeated-ID gradient summation and the existing global loss denominators.

For actual delayed updates, track the parameter baseline and compatible model interface. Small lag may justify a bounded merge; large changes require a fresh forward/backward or rejection. Timestamp-only scaling is insufficient when an old gradient points in the wrong direction. Measure delay in row changes and shared-interface changes as well as global steps.

For DiLoCo, local optimizers process local gradients. The coordinator processes baseline-to-current deltas. Do not treat those deltas as fresh gradients for the proposed squared-gradient estimator, or average independently biased Adam moments as though they were one synchronized optimizer. Retain a separately specified outer merge and count any outer momentum vectors in the memory budget; a low-memory outer SGD/row-scaled merge deserves comparison with the existing Nesterov merge.

Keep model, packed bank, optimizer summaries, index, and hot-cache fractional state coherent at checkpoint publication. A failed multi-component update requires rollback to the last coherent snapshot; this does not imply an ACID transaction spanning neural computation and a database. Bound and checkpoint the active precision cache so restart does not erase all small-update progress.

On edge birth, initialize scale from the chosen prior. On split or merge, use explicit fresh-history initialization or a tested statistic transformation; do not blindly clone Adam counters. Children initially receive different responsibility-scaled gradients from the parent. On deletion, remove obsolete state. A split preserving the floating-point mixture must also be checked after packing, null normalization, and grouped top-k selection.

## 10. Account for the whole bank

Illustration: **1,000,000 edges, 32,768 value coordinates each**. Decimal GB; calculations exclude addresses, residuals, indexes, row headers, temporary activations, replicas, backups and source data unless stated.

| Value-related storage | GB |
| --- | ---: |
| FP32 values | 131.072 |
| BF16 values | 65.536 |
| Raw four-bit values, no scales | 16.384 |
| MXFP4-style four-bit values + one byte scale per 32 elements | 17.408 |
| NVFP4-style four-bit values + one byte scale per 16 elements | 18.432 |
| Additional FP32 higher-level scale per row | 0.004 |
| Separate FP32 master + two FP32 moment arrays | 393.216 |
| Two four-bit moment arrays + FP32 scales per 128 elements per array | 34.816 |
| One FP32 statistic per 256 value coordinates | 0.512 |
| FP32 working masters for 1% of value rows | 1.311 |

The proposed FP4 value storage plus block statistics is about **18.95 GB**, before the clearly listed other costs, versus about **458.75 GB** for a BF16 read copy plus FP32 master and two moments. This is storage arithmetic, not a measured performance or quality result. A deployed representation can avoid some duplicate copies; comparisons must name what actually exists.

Even an eight-bit full first-moment vector alone is about 32.77 GB here. Reducing the **number of optimizer statistics** can matter more than reducing their precision.

The adaptive-capacity objective must count:

\[
C_{\mathrm{memory}}=C_{\mathrm{packed\ parameters}}+C_{\mathrm{scales}}+
C_{\mathrm{optimizer}}+C_{\mathrm{index}}+C_{\mathrm{working\ cache}}.
\]

Report storage tiers and worker/coordinator replication separately. Offloading moves bytes; it does not remove them. Grow edges only after checking whether the apparent residual error is better addressed by precision promotion, refreshed scale estimates, or a fresh retrieval index. A quantization problem should not automatically create more graph structure.

## 11. Experiments that answer the actual question

Separate numerical correctness, stationary learning, and genuine drift. Use equal task data and report both equal-byte and equal-compute comparisons.

| Experiment | Required comparison |
| --- | --- |
| Precision | BF16/FP32 reference, FP8, block FP4, NF4/INT4; parameter precision separate from optimizer-state precision. |
| Sparse optimizer | Existing touched-row Adam; SGD without momentum; row/block AdaGrad; discounted AdaGrad; proposed discounted block scale; eight-bit and ZIP-SR/ZE-EDEN-style four-bit Adam. |
| Compressed statistics | SM3 and Adafactor references where their sparse implementation is well-defined; count full first-moment state when enabled. |
| Dormancy | Long-tailed visits; return after 100, 10,000, and 1,000,000 global updates under both stationary and changing objectives. |
| Representation drift | Fixed reader, gradually moving query/projector, abrupt controlled basis/interface change; with and without fresh cold-row probes. |
| Actual delay | Fresh gradients after dormancy versus delayed gradients and DiLoCo deltas; keep these results separate. |
| Locality | Personal statistics only; global prior; spatial prior; adversarial neighboring edges with different useful updates. |
| Small updates | Intended update below a quantization bin; measure lost movement, waiting in visits, rounding bias, saturation, and cache-eviction effects. |
| Recurrence | Matched one-step retrieval and long search trajectories; address recall, null calibration, residual drift, loops, and downstream task success. |
| Growth | Fixed bank versus adaptive edges versus adaptive precision at equal total bytes; test split/merge behavior after quantization. |

Before large runs, check touched-row gradient equivalence, single-update repeated-ID behavior, masked rows, finite and zero gradients, index movement, quantizer scales, deterministic decode, checkpoint resume, and stochastic-rounding reproducibility. Do not test training by serving dequantized FP32 masters that will not exist at deployment.

Measure task loss and success, rare-task retention, adaptation after the first 1/5/20 return visits, old-task forgetting, edge discovery rate, ANN recall, quantized-update survival, recurrent state drift, bytes across every tier, bytes transferred per query/update, p50/p95 latency, and end-to-end training cost. Search discovery, optimizer quality, and low-bit storage must not be conflated.

Choose the optimizer by a quality/adaptation/bytes trade-off, not the newest name. Begin with a small controlled bank; verify that any spatial prior adds value before scaling it. No experiment described here has yet been run for this repository.

## References

[1] [PyTorch FBGEMM: table-batched embedding training and optimizer state](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/tbe_ops_training.html).

[2] [Continual Learning via Sparse Memory Finetuning](https://arxiv.org/html/2510.15103v1), 2025, section 5, optimizer discussion.

[3] [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram)](https://arxiv.org/html/2601.07372v1), 2026, experimental setup and appendix A.

[4] [Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods](https://arxiv.org/html/2609.21815v2), revision 25 September 2026.

[5] [Memory-Efficient Adaptive Optimization (SM3)](https://arxiv.org/html/1901.11150v2), 2019.

[6] [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235), 2018.

[7] [DeepRec: AdagradDecay optimizer](https://deeprec.readthedocs.io/en/latest/AdagradDecay-Optimizer.html).

[8] [Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets](https://arxiv.org/html/2603.16731v2), revision 27 September 2026.

[9] [Rounding in Preconditioner Space: Redesigning 4-bit AdamW Optimizer-State Quantization](https://arxiv.org/html/2610.12444v1), 8 October 2026.

[10] [Pretraining Large Language Models with NVFP4](https://arxiv.org/html/2509.25149v2), revision 4 March 2026.

[11] [Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention](https://arxiv.org/html/2607.04422v2), revision 7 August 2026.

[12] [ALPT: Adaptive Low-Precision Training for Embeddings in Click-Through Rate Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/25564), AAAI 2023.

[13] [AF-Muon](https://arxiv.org/html/2610.01395), October 2026.

[14] [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/html/2406.16793v4).

[15] [Memory Layers at Scale](https://arxiv.org/html/2412.09764v1), 2024.

[16] [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://proceedings.mlr.press/v119/guo20h.html), ICML 2020.

[17] [Extended RaBitQ](https://arxiv.org/html/2409.09913v1) and [GPU-native IVF-RaBitQ](https://arxiv.org/abs/2602.23999). Evaluate the actual selected backend's dynamic update and inner-product support.

[18] [Multitask Online Mirror Descent](https://arxiv.org/html/2106.02393v3). Relatedness assumptions motivate testing statistical sharing; they do not prove arbitrary address-space smoothing is beneficial.
