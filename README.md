# Megatoken — required retrieval-memory training

**Breaking architecture migration.** Native retrieval + semantic memory is the **only supported training architecture**. There is no disabled, optional, ungrouped or pre-retrieval training route. Model creation always registers the semantic embedding realizer, the full-native-facet memory K/V projector and independent all-track LoRA controllers, even while the canonical semantic-memory bank is initially empty.

This project is a software prototype, not a pretrained model. Training quality, performance and release fitness have **not** been established. See the [implementation and validation ledger](docs/retrieval-memory-implementation-status.md).

## One architecture

The same selected raw-facet representation powers ordinary E/C/P/R generation and learned retrieval:

1. **Causal query event:** Condition P on the main request, retrieval purpose and pinned semantic-space identity. Sample its normal raw Gaussian event, realize one typed normalized embedding with asymmetric query/document R, then commit the selected raw event through C.
2. **Literal evidence:** Retrieve eligible source chunks with incremental MaxSim, deduplicate the original source byte intervals, and admit only successfully fetched/committed bytes through native E/C on reusable source tracks. Source content and original lineage remain immutable.
3. **Semantic memory:** In a source-supported, progressively grown free memory bank, jointly learn queries, point/source keys, pair dependence, semantic values, strength and lifetime. Mix compatible point/pair semantic values **before** projecting each active group into the complete native F-facet K/V geometry. Admit them directly to a separate bounded KB context segment, not to the native raw frontier.
4. **C/P/R control:** Use the same admitted semantic mixture for independently parameterized sparse LoRA experts across every track. C, P and R share one transaction snapshot; grouped native meridian clocks advance only when real events commit. Null KB strength must remove the position from attention normalization.
5. **Training:** Learn useful retrieval jointly with continuation, detached original-source feature-value supervision, teacher-aligned query/document R, query-demand-driven allocation and optimizer-step-global router balance. Preserve exact complete-footprint causal eligibility, sparse row moments and coherent neural/KB snapshot identities.
6. **Writer handoff:** Only *after* useful free-bank pretraining, freeze the accepted learned coordinates and fit permitted source/pair writers. Fully reconstruct a fresh source+point+pair KB from the immutable original sources, never by quietly continuing with optimized free values. Experience writing comes afterward.

[Architecture and acceptance specification](docs/retrieval-memory-plan.md). [Donor methodology: hindsight-kb@4501647](https://github.com/werg/hindsight-kb/tree/450164784589b2576256b8b0ba3d2ccc89af992d).

## Prepare once, train only through the new recipe

Python 3.11/3.12 and the tested native PyTorch runtime are expected. Model defaults use a 1,024-dimensional learned address geometry (the teacher's trained truncation), a separate 256-dimensional semantic value space, and four bounded complete-KV groups; the actual chosen teacher, output dimension, native model checkpoint geometry and reproducible preprocessing must agree in the phase recipe. A deliberate zero-evidence *ablation* is allowed inside this architecture, but does not reinstate a legacy trainer.

```bash
python -m pip install -e '.[train,data,dev]'
megatoken doctor
python -m pytest -q tests/test_memory_*.py

# Create prepared native-meridian episodes and hash-verified retrieval sidecars.
# Pin the real teacher weights, original source hashes and learned-space roles.
megatoken-memory configs/retrieval-memory-pilot.json
```

The checked-in recipe and stage configurations deliberately use `PIN_*` placeholders: they **cannot** start training until preparation outputs exist and match. This is safer than silently running on demo data or ungrounded vectors.

The sole supervised training command is:

```bash
NPROC_PER_NODE=8 scripts/launch.sh configs/train/retrieval-coherence.example.yaml
```

Provide a filled-in, immutable copy of that config. It must name the checksummed causal query sidecar, phase-matched retrieval recipe, prepared episode dataset, codec, original-source snapshot and model geometry. The trainer rejects ungrouped episodes, missing teacher main-request bytes, incompatible learned spaces and old checkpoints.

Continue in order with **phase receipts**:

```bash
NPROC_PER_NODE=8 scripts/launch.sh configs/train/retrieval-warmup.example.yaml \
  --resume runs/retrieval-coherence/checkpoints --phase-transition
NPROC_PER_NODE=8 scripts/launch.sh configs/train/retrieval-free-kb.example.yaml \
  --resume runs/retrieval-warmup/checkpoints --phase-transition
```

These two phases additionally require the authoritative prepared SQLite KB, source assets, pinned IVF centroids, signed-off readiness/holdout gate receipts and a synchronized neural+KB checkpoint. Query-density growth cannot begin before generator-coherence and query/chunk embedding gates pass. Development/release holdouts, source-family ancestry, sparse updates and index freshness must be verified.

There is **no** `configs/train/b0.yaml`, implicit launcher default, `megatoken smoke` or asynchronous DiLoCo training path. The old stand-alone `megatoken rl` training command is also retired until the native retrieval-memory experience/writer acceptance gates are implemented.

## Storage and writer utilities

After preparing immutable source chunks and their support identities, use:

```bash
megatoken-memory-catalog --help
megatoken-memory-centroids --help
megatoken-memory-index --help
megatoken-memory-writer --help
megatoken-memory-reproduce --help
```

These are infrastructure commands, not substitutes for the native curriculum. Writer checkpoints have pinned model/free-bank/space identities, and fresh source+point+pair shadow banks are accepted only when all original assignment and causal support checks pass.

See [the preparation/order/status ledger](docs/retrieval-memory-implementation-status.md) for exact artifact dependencies, release blockers, empirical evaluation requirements, known gaps and reproducible validation commands. For hardware-specific setup, see [Blackwell](docs/blackwell.md); prior stand-alone B0/P1/M1/M2 experiments in historical documents are no longer supported launch instructions.

## Validation contract

A merge/production launch requires: complete CPU and Python-version tests, GPU/native attention gradient and RoPE parity, true multi-rank sparse/global-router equivalence, checkpoint corruption/recovery, full delayed retrieval replay, locality/visibility/negative support checks, teacher source-document and causal-query alignment, valid fresh source/pair writer reproduction, no-memory regressions and new-source task uptake. Passing structural tests does not assert these learned-quality results.

The development PR stays **draft until verified**, but the architecture is not optional.
