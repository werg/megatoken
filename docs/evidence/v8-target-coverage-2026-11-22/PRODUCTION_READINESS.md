# Remaining production training inputs

The 64-source support campaign qualifies admitted source supply and a bounded
ordinary optimizer objective. Production launch additionally depends on concrete
model, teacher, donor, checkpoint, coordinator and hardware inputs. This read-only
audit records the requirements implemented by the inspected native source and the
inputs available in this conversation workspace. It does not start a learner or
invoke the production launch qualifier.

`production-readiness-input-audit.json` retains the source-file hashes, exact
inspected revision, unresolved template fields, runtime observation and asset
inventory scope. The production template still contains nine `PIN_*` bindings.

| Input or gate | Available evidence and remaining requirement |
| --- | --- |
| Source supply | The frozen 64-source corpus has 48 training, 12 validation and four test sources. Its actual strict SE supply gate passes both train and validation. This resolves the earlier missing child, repeated-invocation and source-reread supply prerequisite. |
| Full support-target optimizer pass | The separate 348-update diagnostic pass is the active target-coverage campaign. Complete exposure is credited only by its final checkpoint comparison, with exact windows, occurrences and positive-weight denominators. |
| Bound production configuration | `configs/train/semantic-v8-production.example.yaml` remains a template. Its dataset, codec, donor paths, run ID, coordinator URL and policy revision/registry fields require actual reviewed bindings. The existing hindsight path also requires its real immutable asset. |
| Code identity | The launch qualifier requires separate clean `tracing-js-self`, `megatoken` and `natlang` checkouts. The actual imported consumer must come from the supplied native checkout; producer/consumer revisions must match the V8 policy. The standard Natlang checkout is absent from this workspace. |
| Tokenizer | The exact approved Ling tokenizer is available and its bytes are verified. This resolves one input; it does not supply donor descriptors, hindsight targets or model weights. |
| Donors | Both SC and SE require the pinned `deepseek-v4.1-flash-1017` and `qwen3.8-flash-next` donor profiles in curriculum order. Their actual `engram.json` metadata and checksum-matching `descriptors.npz` assets were not present in the inspected workspace. The qualifier loads each asset against the model geometry. |
| Hindsight teacher targets | Production requires the corpus-bound target tranche, `targets.sqlite`, each referenced target tensor, `encoding.pt`, and the inherited teacher encoding. Every actual training window must match the target loader and manifest coverage. These assets were not present in the inspected workspace. |
| Production checkpoint | SC and SE both require a preceding S-stage checkpoint or an exact same-stage worker resume. The qualifier verifies actual tensor shapes, finite values, optimizer/master layout and lineage. The tiny diagnostic model is not a production-geometry predecessor checkpoint. |
| Coordinator | Production uses independent DiLoCo learners. The actual initialized coordinator must bind the objective, corpus, model/master layout, V8 lineage and bootstrap provenance. A new coordinator needs its real initialization receipt. No authenticated coordinator probe or mutation was performed by this audit. |
| Hardware | The example selects `32x512`, bf16, Blackwell and eight independent workers. The inspected runtime is CPU-only PyTorch 2.10.0 and has no available CUDA device. These are template choices and an observed local limitation, not a claim that every production deployment must use exactly eight workers. |

The canonical production recipe also requires stage-mixture sampling, positive
teacher/navigation objectives, held-out evaluation and periodic checkpoints. The
bounded coverage fit uses the ordinary trainer with a diagnostic tiny fp32 model,
flat support stream and one optimizer window per update. Its complete target
inventory binds that precise objective and window geometry. It cannot substitute
for target use under a different production objective or window configuration.

Once real production inputs exist, run the existing
`scripts/qualify_semantic_launch.py` with the bound configuration, all three code
checkouts, the selected resume or bootstrap checkpoint, the actual bootstrap
receipt when applicable, and `--probe-coordinator`. That command performs
authenticated reads and records every independent blocked gate. A successful
preflight still has zero optimizer steps. Production memory fit, multi-worker
failure recovery, exported held-out execution quality and measured cost then need
their own actual results.
