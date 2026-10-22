# Megatoken — causal faceted generation with a continuous-state transducer

This project is a software prototype, not a pretrained model. Training quality, performance and release fitness have **not** been established.

## One architecture

One selected raw-facet representation powers ordinary E/C/P/R generation, and one learned memory sits beside it: the [Continuous State Transducer](docs/continuous-state-transducer-plan.md) (v10, implemented, not yet trained). At chosen boundaries the model emits a query, searches a sparse bank of location-to-location edges with an agenda-wide neighbourhood search, and admits what it finds as extra attention slots ("knowledge megatokens"). Edges carry FP8 routing vectors and 32,768-wide FP4 values; a momentum-free local-scale optimizer trains them, hindsight writes fill them, and error-driven growth adds and splits them. The transducer is always on: there is no switch and no older retrieval stack (the v9 source catalog, search index and embedding-alignment sidecars were removed). The plan takes precedence over the accompanying research survey.

The bank travels with every checkpoint (`transducer.pt` beside the weights). Under DiLoCo the coordinator owns the canonical bank and merges each learner's 32-step round of sparse changes; synchronous multi-GPU ranks share it the same way, merged on rank 0.

## Prepare, then train

Python 3.11/3.12 and the tested native PyTorch runtime are expected.

```bash
python -m pip install -e '.[train,data,dev]'
megatoken doctor
python -m pytest -q tests
```

The checked-in stage configurations deliberately use `PIN_*` placeholders: they **cannot** start training until preparation outputs (prepared episodes, codec, pack manifest) exist and match. Single node:

```bash
NPROC_PER_NODE=8 scripts/launch.sh configs/train/coherence.example.yaml
```

There is **no** implicit launcher default or `megatoken smoke`.

**DiLoCo is the target training methodology.** `megatoken diloco init|serve|export` run decoupled learners ([configs/train/diloco-coreweave.example.yaml](configs/train/diloco-coreweave.example.yaml), [docs/diloco-coreweave.md](docs/diloco-coreweave.md)). Every phase runs decoupled: neural parameters sync as fragments and the coordinator merges the transducer bank's rounds. Each phase is a new coordinator started from the previous phase's export. For hardware-specific setup, see [Blackwell](docs/blackwell.md).

## Validation contract

A production launch requires: complete CPU and Python-version tests, GPU/native attention gradient and RoPE parity, true multi-rank equivalence (dense gradients and the merged bank), checkpoint corruption/recovery, and the learned-quality measurements in the transducer plan. Passing structural tests does not assert learned-quality results.

The development PR stays **draft until verified**.

## Media: one input path, one realization path

Media has exactly one path in and one path out ([docs/media.md](docs/media.md)). Every image, audio or video event enters as the **512-wide EmbeddingGemma2 hidden states** (not its pooled 768-dimensional embedding), extracted offline by the pinned frozen teacher and projected by a trainable `MediaEncoder` into the same E reductions as text; text enters as bytes, except long reference documents and large tool results, which can be read compressed through the same teacher ([docs/compressed-reading.md](docs/compressed-reading.md)); its pooled 768-dimensional embedding is the target of the encoder's embedding head. Media leaves through frozen pretrained realizers (Pocket for 24 kHz speech, Sana for images, Sana-Video for video) with trainable `FacetCondition`s, trained on the donors' codec latents. All pretrained weights come from one locked pack manifest written by [scripts/acquire_packs.py](scripts/acquire_packs.py) from the pinned [inventory](configs/packs/inventory.json); install with `pip install -e '.[train,media]'`. Media is part of the stage plan (`src/megatoken/data/stages.json`): a short [media-bridge stage](configs/train/media-bridge.example.yaml) after the codec bootstrap, explicit media and long-document shares in base pretraining, and replay in later stages, each stage a new DiLoCo coordinator started from the previous export. No media-quality or GPU-training results are implied by these contracts.
