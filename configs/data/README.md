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

`sources.open.json` and `sources.open-mb.json` are the permissively licensed pilot sources
(licences read from each dataset card, recorded in `terms`, approval left to the maintainer);
the cloud job `configs/cloud/prepare-open-mix.json` prepares them together with the open
media datasets. See "Open-licence pilot mix" in [`docs/data.md`](../../docs/data.md).

`sources.agentic.json` holds the agent-trajectory, tool-use, chat and reasoning sources (any
licence, each recorded); `sources.agentic-gated.json` holds the ones that need the
maintainer to accept terms on the Hub or restore withheld prompts first. The cloud job
`configs/cloud/prepare-agentic-mix.json` prepares the first in eight shards, scaled with
`AGENTIC_ROW_SCALE`. See "Agentic and chat data" in [`docs/data.md`](../../docs/data.md).

`files.agentic.json` and `files.open.json` list every file of every Hub source in the
manifest of the same name, with its stored size, at the pinned revisions. They fix how a
full-volume build is split into shards, so every machine plans the same shards from the same
committed copy; rebuild one with `scripts/shard_sources.py MANIFEST OUT --full --listing FILE`
after changing a manifest's sources or revisions (it only reads metadata, but a gated
source needs a token that has accepted its terms). The runs that use them are
`configs/cloud/prepare-full-agentic.json` and `configs/cloud/prepare-full-open.json`
(`python -m megatoken.cloud_prepare plan|launch`); see "Preparing at full volume" in
[`docs/data.md`](../../docs/data.md).

`sources.agentic-more.json` (417 more chat, reasoning, tool-use and agent sources) and
`sources.agentic-more-gated.json` (23 that need a click on the Hub) come from a second sweep
of public sets. The cloud job `configs/cloud/prepare-agentic-more.json` prepares the first,
scaled with `AGENTIC_MORE_ROW_SCALE`. See "More chat, reasoning and agent data" in
[`docs/data.md`](../../docs/data.md).

`sources.later-stages.json` holds the public sources for the streams the later stages read
(issue and pull-request data, change histories, logs, long documents, grounded artifacts,
tested code, clarification and recovery, code rewrites, environment episodes). The cloud job
`configs/cloud/prepare-later-stages.json` prepares it in eight shards, scaled with
`LATER_ROW_SCALE`. See "Later-stage streams" at the end of [`docs/data.md`](../../docs/data.md).
