# Retrieval / semantic-memory implementation status

**Branch:** `feat/retrieval-semantic-memory` ([draft PR #18](https://github.com/werg/megatoken/pull/18)).  
**Specification:** [retrieval-memory-plan.md](retrieval-memory-plan.md).  
**Donor reference:** `werg/hindsight-kb@450164784589b2576256b8b0ba3d2ccc89af992d`.  
**Status:** **NOT production-training-ready**. This is an honest engineering ledger, not acceptance evidence.

## Implemented foundation

- Semantic-space identity, complete source ranges/visibility and checksummed event sidecars.
- SQLite source/memory bank, transactional source-version invalidation, typed revisions, exact/IVF selection, incremental MaxSim, deferred coverage, point/pair growth and optimizer-row storage.
- Embedding realizer directly over selected native raw facets. The training query now comes from a query-purpose P draw, with an exact hashed request that must occur in the previously committed episode prefix; it no longer mislabels an answer megatoken as a query.
- Full-facet semantic KV projection, native KB segment, per-transaction snapshots, independently controlled C/P/R LoRA modules.
- Global routing balance replay, replayed selected-row source/memory loss helpers and sparse-row Adam with atomic pre-step freshness validation.
- Density-first repeated allocation with separate source-supported tail exploration, idempotent source-version tombstones and dependent relation/index invalidation.
- Frozen-coordinate post-pretraining SourceWriter/RelationWriter, suffix/full physical attention views and per-assignment distillation; frozen downstream readers preserve autograd to writer outputs.
- Immutable checksum/policy/assignment-pinned safetensors writer shards and a phase-gated distillation command:
  `megatoken-memory-writer --recipe <recipe.json> --gate-receipt <gate.json> --dataset <writer-dataset.json> --output <new-output-directory> --epochs 1`.
  The command **does not** claim to execute the behavioral continuation or cross-source training callbacks.

## Non-negotiable remaining code gates

1. **Native bank-aware production training owner.** `training/engine.py` still deliberately rejects memory-recipe phases after `coherence_embedding`. Wire selected source/memory key/value leaves, learned pair fields, KB admission and independent KV/LoRA dropout into forward grouped-meridian training. Gradient-normalize across true global denominators; synchronize sparse touched rows across ranks, and combine model/bank publication, growth and checkpoints at a coherent step barrier. No source-free memory allocation.
2. **Native chronological training/replay.** Causally scheduled query requests, delayed source responses, committed ordered RAG bytes, durable coverage/deferred queue revisions and admitted KB position/LoRA snapshots must survive grouped training tapes, replay, checkpoint resume and serving. The bootstrap query is currently a *noncommitting shadow action*, as intended for embedding distillation; this is not full native retrieval-policy replay.
3. **Local bilateral retrieval and corpus-key objectives.** Drive live selected source/memory key leaves from value/usefulness targets and actual native continuation, including missing-answer labels. Document-key distillation, exact frozen source feature targets, optimized final keys and full contextual writer targets still require production connection and audited data.
4. **Writer behavioral phases.** Use `training.hindsight.fit_source_writer_epoch` and `fit_relation_writer_epoch` with explicit frozen-reader behavior callbacks. Implement the j>=i+2 continuation objective and later cross-source transfer over actual source trajectories, then full-target point/pair reconstruction in the accepted final learned space. The CLI only executes supervised frozen-target distillation and marks `behavioral_continuation_verified=false`.
5. **Data production / teacher pilot.** The checked-in pilot recipe has intentional `PIN_...` placeholders for teacher weights and source partition. Compare/pin a teacher, prepare actual causal query/document targets, source byte spans/lineage, independent relation windows, negative examples and immutable writer shards. Do not fabricate gate receipts.
6. **Release validation.** Run the full Python 3.11/3.12 suites, Ruff, distributed uneven-rank router and sparse-optimizer parity, replay and snapshot recovery, GPU native KV/LoRA numerical tests, and held-out retrieval/continuation uplift experiments. GitHub Actions currently reports failed CPU/i18n jobs without step logs, so the test suite cannot be claimed green from that signal.

## Training and artifact order

1. Run `python -m megatoken.memory.recipe configs/retrieval-memory-pilot.json` to inspect the prototype's declared phase and pinned geometry. Fill genuine teacher revision/data partition fields only when known.
2. Complete coherence and both causal-query/chunk embedding bootstrap quality gates. Produce independently auditable gate evidence; a manifest string by itself is not a measured pass.
3. After implementing the bank-aware owner, run progressive free-bank warm-up and full pretraining, preserving optimized point/pair rows and moments across growth barriers.
4. Freeze and snapshot the accepted trained P/R/query model, semantic materializers and bank. Only then switch to the writer phases with the *same* optimized semantic space and pinned original assignments.
5. Prepare safetensors writer shards from actual frozen native E source blocks; every shard needs a SHA256, complete source-footprint identity and original assignment digest. A JSON writer dataset declares:
   `schema=megatoken-writer-dataset-v1`, `kind=source|pair`, `recipe_fingerprint`, `semantic_space_fingerprint`, `model_checkpoint_id`, `free_bank_sha256`, `source_context_policy`, `mode=suffix|full`, explicit `layout` and `shards`. Source tensors include `encoded,valid_blocks,slots,key,value,strength,lifetime_logit,distance,valid`; pair shards contain both ordered dependence targets and source-writer member features.
6. Fit permitted suffix, cross-source and full-context/pair writers; run frozen-reader continuation objectives and shadow bank reproduction. Assert output correspondence, no answer leakage, exact key/index/graph identity, no optimized leaf needed at serving, and quality improvements on unseen sources.

## Focused gates added on this branch

```bash
python -m pytest -q \
  tests/test_memory_foundations.py \
  tests/test_memory_causal_sidecar.py \
  tests/test_memory_writer_causality.py \
  tests/test_memory_support_supervision.py \
  tests/test_memory_sparse_atomicity.py \
  tests/test_memory_router_global.py \
  tests/test_memory_query_policy.py \
  tests/test_memory_hindsight_training.py \
  tests/test_memory_writer_artifacts.py
ruff check src/megatoken/memory src/megatoken/training src/megatoken/data/retrieval.py \
  tests/test_memory_*.py
ruff format --check src/megatoken/memory src/megatoken/training \
  tests/test_memory_*.py
```

The commands are **validation requirements**, not a claim they passed. No corpus was generated, no teacher weights were downloaded, no GPU training ran, and no model quality result was established.
