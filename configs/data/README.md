# Data configuration

The canonical, wheel-packaged phase budgets and mixtures are in
[`src/megatoken/data/stages.json`](../../src/megatoken/data/stages.json).
Load them through `megatoken.data.stage_recipe("P1")`; the training scheduler and
the data census use the same D/I/P distinctions. We deliberately maintain one
copy of those recipe values.

See [`docs/data.md`](../../docs/data.md) for the complete offline preparation,
source acquisition, family split, masking, census, and collection workflow.
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
