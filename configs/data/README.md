# Data configuration

The canonical, wheel-packaged phase budgets and mixtures are in
[`src/megatoken/data/stages.json`](../../src/megatoken/data/stages.json).
Load them through `megatoken.data.stage_recipe("P1")`; the training scheduler and
the data census use the same D/I/P/Z distinctions. We deliberately maintain one
copy of those recipe values.

See [`docs/data.md`](../../docs/data.md) for the complete offline preparation,
source acquisition, family split, masking, census, and collection workflow.
Semantic chunking is shared with inference: target 512, soft maximum 768 and
hard maximum 1,024 reference BPE tokens in every production stage, including B0.
Explicitly smaller hard limits clamp the target and soft maximum.
Preparation stores the versioned chunker,
codec and parser-runtime identity plus an accepted-record size histogram.
Use [`chunking.production.json`](chunking.production.json) with
`--chunking-config`, or `--chunk-target-tokens`, `--chunk-soft-max-tokens` and
`--max-record-tokens` to make an explicit new preparation policy. The policy
includes recursive parser budgets, isolation, and tree movement costs.
[`parser-review.json`](parser-review.json) configures the default durable
review queue; pass it through `--parser-review-config`. Parser faults and
resource exhaustion are excluded from shards until an amended parser passes
verified replay, followed by a fresh build from the pinned sources.
See the [parser and review runbook](../../docs/chunking.md). Prepared/native
schema version 2 rejects legacy records and tapes; existing generated actions
are never silently resegmented.

`task.example.jsonl` and `collection.smoke.json` run a deliberately tiny executed
fixture pilot. The production pilot defaults come from the same canonical
`stages.json`; the smoke override is not a production supply recipe.

`sources.v13.json` records immutable publisher revisions and adapter settings.
It is an acquisition manifest, not a claim that any source has been downloaded
or approved for your use. Before importing a source, complete its terms review,
choose specific subsets/files and supply any parent-family joins indicated by
the adapter. Use bounded pilot row counts and inspect the rejection report.

`benchmark_quarantine.example.json` documents the explicit benchmark/family
quarantine interface. Replace it with the IDs and transcript hashes from your
versioned evaluation set before production preparation.
