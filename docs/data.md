# Data preparation and executed collection

The data package provides a complete local path from source rows to indexed
training windows, and a separate budgeted path from original task seeds to newly
executed teacher/student trajectories. Both produce the same immutable `Episode`
contract. Preparation requires Python and a byte codec; PyTorch is loaded only
when a local model is actually used as the student.

The checked-in publisher manifest contains frozen revisions and adapter settings.
It does **not** represent acquired inventory. No bulk public corpus or paid teacher
generation was performed while implementing this pipeline. The local demo runs
real in-memory operations and checks their answers; it is an integration corpus.

## Start with a complete offline round trip

From an installed checkout:

```bash
megatoken demo --output work/demo.jsonl --count 32 --max-record-bytes 128
megatoken prepare --demo work/demo.jsonl --output work/prepared-demo \
  --codec byte --max-record-tokens 256
```

`demo` also writes `work/demo.jsonl.sources.json`, including its exact file hash
and the local fixture provenance. `prepare` writes `manifest.json`, independent
checksummed JSONL shards, `index.sqlite`, `preparation.sqlite`, and a rejection
report. An identical invocation resumes an interrupted build or returns the
already committed dataset. Changing a source, codec, adapter configuration,
split policy, quarantine snapshot, or chunking policy requires a different output
directory.

Use the real pinned codec asset directory for training data. `--codec byte` is a
256-byte smoke codec, with a distinct fingerprint. The model, data manifest,
collection ledger, and served bundle must agree on the codec fingerprint. See
[generation.md](generation.md) for raw Ling codec import and byte preservation.

```python
from megatoken.data import PreparedDataset, WindowConfig, iter_windows
from megatoken.loading import load_codec

codec = load_codec("work/codec")
with PreparedDataset("work/prepared", split="train", stream="natural_documents") as data:
    for episode in data.iter_episodes(rank=0, world_size=1):
        for window in iter_windows(
            episode,
            codec,
            WindowConfig(max_window_tokens=8192, max_decisions=4, max_records=32),
        ):
            # Rebuild window.prefix under current weights without loss.
            # Train the ordered window.records in one differentiable segment.
            print(window.episode_id, window.D, window.I, window.P)
```

The training runtime owns deterministic mixture scheduling, distributed rank
assignment, compatible batching, loss evaluation, and checkpoint cursors. The
reader exposes indexed source/stream/token counts without materializing every
episode. `PreparedDataset` defaults to the training split; `prepare_sources`
returns a reader over all non-quarantined splits for census inspection.

## What an episode guarantees

`Episode`, `PhysicalRecord`, `Lineage`, `LogicalAction`, `SourceSpan`, `Dependency`,
and `Supervision` are frozen dataclasses. They use immutable bytes and tuples.
The wire schema stores payloads as base64, preserving non-UTF-8 files and exact
whitespace. JSON framing and chat-template delimiters are not inserted into the
student transcript. Prepared datasets and native episodes use schema version 2;
legacy versions fail explicitly. Native action metadata must contain complete
version-3 facet tapes (`fmt-canonical-facets-v3`), which retain the exact public
control payload used by the prior. Older tapes must be regenerated.

| Field group | Meaning |
| --- | --- |
| `records` | Actual physical record order, role, logical/physical identities, part index/count, byte payload and source span |
| `lineage` | Source revision, original family, all known ancestors, repository/task identities, original row hash and optional inherited split |
| `actions` | Complete tool action descriptor, immutable serialized-byte hash, exact physical members, call ID and execution ID |
| `result_of` | An observation's exact prior completed action; a changed action cannot inherit another action's result |
| `policy_mask`, `free_spans`, `emitted_free_mask` | Role/turn eligibility and lexical bytes or emitted positions eligible for supervised decoder cross-entropy |
| `chunk_boundary`, `chunking_fingerprint`, `forced_chunk_split` | Auditable semantic boundary, exact chunking policy identity, and unavoidable token-cut marker |
| `supervision` | Outcomes, verification, rewards, benchmark tags and diagnostic metadata; never rendered into model records |
| `tools_json`, `control_seed` | Public tools and task policy available before generation |
| `control_reference`, `reference_origin` | An explicitly available reference; future-target references are rejected |
| `dependencies` | Real earlier writer and later scored reader record IDs for long-context credit assignment |

Construction rejects duplicate physical IDs, reordered turns, incomplete logical
parts, actions with the wrong serialized hash, observations preceding complete
actions, reused results, and mismatched action/result execution IDs. Complete
logical actions stay contiguous even when their bytes span several physical
records. Images and unrecognized content blocks fail explicitly rather than
silently disappearing during text import.

`SourceSpan.start/end` are half-open offsets in the **decoded original source
field**, not offsets in JSON's escaped transport representation. Native content
uses `transform="identity"`. Source adapters label necessary tool-envelope
serialization separately, retain the delivered argument string, and never claim
that generated framing was original source prose. Published scrubbed/redacted
text is preserved as delivered; the importer cannot reconstruct unavailable
pre-redaction bytes.

Observation roles (`user`, `tool`, `environment`, `system`) cannot acquire an
assistant policy mask. Document language modeling is an explicit document-role
objective. Failed traces retain their actual history but only explicitly marked
recovery turns receive imitation loss. Unknown and incomplete executions are
separate from verified successes. A simulator judge is recorded as simulation
provenance and cannot satisfy executed-success quotas.

## Semantic megatokens

New codec-backed imports use one shared, deterministic chunker with a **256-token
target, 512-token soft limit, and 1,024-token hard limit**. These are reference
BPE positions, not bytes or words. Complete paragraphs, fenced code blocks,
functions and list items stay intact when they fit the soft limit. Small adjacent
units from the same logical event pack toward the target. Roles, turns, source
events and episodes are never merged to fill a chunk.

Oversized units use statement, sentence, line and whitespace boundaries in that
order of structural suitability. Python functions and statements use source
coordinates from the standard-library AST. Other code uses a conservative brace,
string and comment scanner, with line boundaries as a fallback. Large indivisible
code or list units may exceed the soft limit while respecting the hard limit;
complete JSON tool actions can use the entire hard capacity. If a source has no
suitable boundary, the chunker cuts at reference-token boundaries and records
`forced_chunk_split=true`. Invalid UTF-8 follows this lossless binary path.

The parser borrows block distinctions from the
[CommonMark specification](https://spec.commonmark.org/0.31.2/) and uses
[Python AST source locations](https://docs.python.org/3/library/ast.html#ast.AST)
for Python code. It only locates boundaries: it never renders Markdown,
reformats code, replaces NUL, changes newlines, trims whitespace or normalizes
Unicode. Sentence and unknown-language boundaries are deterministic heuristics;
the configured token targets are engineering defaults to evaluate on real data.

The complete logical source field is encoded once. Every output chunk retains
contiguous byte offsets and a contiguous slice of those original reference IDs.
A boundary falling inside a BPE token is used only when it can move within a
whitespace separator; otherwise the candidate is skipped. No alternate decoder
IDs are invented to force an exact character boundary. This keeps accepted U
independent of the physical chunk size. UTF-8 scalar and CRLF boundaries are
preserved when the configured physical caps allow them; extremely small test
caps can force a byte-exact partial scalar or newline, which remains marked as a
forced cut.

```python
from megatoken.data.chunking import ChunkingConfig, segment_bytes

policy = ChunkingConfig()  # target=256, soft=512, hard=1024
for chunk in segment_bytes(payload, codec, policy, content_kind="code", language="python"):
    assert chunk.payload == payload[chunk.byte_start : chunk.byte_end]
    assert codec.decode(chunk.token_ids) == chunk.payload
```

`--max-record-tokens 256` explicitly clamps both target and soft limit to 256.
All production stages, including B0, use the 1,024-token hard capacity.
`--chunk-target-tokens` and `--chunk-soft-max-tokens` expose the two semantic
settings; `--max-record-bytes` remains an independent physical byte ceiling.
Source `options.content_kind` and `options.language` supply explicit parser hints;
the code adapter also reads a declared source language field. A preparation's
manifest stores `chunking` with the algorithm version, complete config, codec
identity and Python AST runtime identity. `ChunkingConfig.from_identity(identity,
codec)` rejects missing, altered or incompatible identities. Checkpoints and
served bundles retain that same policy.

`manifest["chunking_stats"]` reports record counts, byte and reference-token
totals, actual decoder positions, a fixed size histogram, mean/max positions,
soft-limit exceedances, boundary kinds, forced cuts and unversioned native
records. These are accepted-inventory statistics, not training exposures.
Native generated output keeps its original IDs, tape and physical boundary;
an oversized native record is rejected instead of silently changing its state
trajectory. Existing prepared artifacts must be regenerated for the new schema
and chunking identity.

Inference's logical `observe` method applies this same chunker to a complete
event. Already prepared records use `observe_physical` to preserve their supplied
boundaries and token IDs. Every selected action advances recurrent state with
its sampled raw facets, including invalid decodes; validation gates environment
execution only. Its deterministic output bytes are never encoded again to
approximate that latent action. Training supervises each chunk's exact output
length and marks only the last physical part of a logical output as terminal.

## Keep accounting units separate

| Meter | Definition | When it increments |
| --- | --- | --- |
| U | Reference-codec tokens in accepted deduplicated transcript inventory | Preparation census |
| D | Newly scheduled transcript exposure | Actual training window consumption |
| I | Actual ingestion, including D and repeated no-loss prefix reconstruction | Actual current-weight reconstruction and training |
| P | Actually evaluated supervised decoder cross-entropy positions | Loss evaluation |
| T | Newly generated teacher output, including rejected attempts and exposed reasoning | Collection request settlement |
| Z | Gaussian facet coordinates in sampled RL actions | Actual latent action generation |

There is no combined total that adds these unlike quantities. Prepared manifests
report D/I/P/T/Z as zero and give `free_position_candidates` separately from actual
training P. Per-source/stream/split census includes eligible assistant tokens,
observation tokens, executed observations, accepted families, duplicate and
quarantine counts. The export also reports executed/simulated transcript U,
verified-success U, incomplete U, and repeated initial-prefix bytes and tokens.
These categories can overlap and should not be summed into another inventory.

`EncodedRecord.token_ids` preserves an actual model's emitted IDs. A noncanonical
two-ID sample may decode to bytes that the reference codec encodes as one token:
its D contribution is one, while each applicable decoder objective has two
candidate cross-entropy positions.
`EncodedRecord.positions` is the actual ID count used for capacity checks;
`EncodedRecord.tokens` aliases `transcript_tokens` for reference exposure.
`EncodedRecord.P` and `TrainingWindow.P` are static masked target counts. Training
records actual P from evaluated target plans. B0 bootstrap evaluates both
canonical target reconstruction and prediction from prior facets, so P is twice
the eligible target positions. Later recurrent phases evaluate only prior-decoder
cross-entropy, so P is once the eligible positions. Their target encoder receives
only alignment loss toward detached completed prior facets. RL lexical decoding
is deterministic and contributes zero P; its stochastic work uses the separate Z
coordinate budget.

For imported text, preparation tokenizes the complete logical source field once
and partitions those reference IDs at semantic/token boundaries. Its
`cached_ids` are declared reference chunks, keeping U independent of the chosen
preparation chunk limit. Native imports validate such caches against the complete
logical field. Original native generated boundaries and emitted IDs are retained;
uncached native records use reference tokenization of their exact physical
payload for D/I. Native IDs remain intact; the Gaussian policy likelihood is
evaluated on the tape's raw facet coordinates.

When public tool schemas are supplied, data masks exclude fixed outer keys,
delimiters, constant fields and bound repetitions from the auxiliary raw
reconstruction objective. A free JSON field retains its whole value, including
nested punctuation represented by the decoder. Non-JSON strings exclude
their quote/escape framing. Masks are derived on the complete logical action and
then projected onto its original physical boundaries; any actual token crossing
a free/fixed boundary is conservatively unscored. Unsupported legacy envelopes
retain their explicit raw-text objective. Auxiliary decoder reconstruction and
the native tape's Gaussian likelihood are separate objectives.

## Pin and import a real source

The `sources` command resolves source identity without accepting source terms.
For example, pin a bounded local document source:

```bash
megatoken sources pin-local /data/documents.jsonl \
  --source-id our-documents --adapter document --stream natural_documents \
  --max-rows 1000 --output work/sources.json
```

A source with completed internal review can be recorded using explicit
`--approve-terms --reviewed-by NAME --reviewed-at YYYY-MM-DD --license LICENSE`
arguments, or by completing the equivalent manifest fields. Merely resolving a
revision never sets those fields. For the Hub, `pin-hf` resolves a branch/tag to
an immutable 40-character dataset commit and retrieves metadata only:

```bash
megatoken sources pin-hf HuggingFaceTB/smoltalk2 \
  --source-id smoltalk-mid --adapter smoltalk2 --stream synthetic_reasoning \
  --subset Mid --split Llama_Nemotron_Post_Training_Dataset_reasoning_r1 \
  --max-rows 1000 \
  --output work/smoltalk.sources.json
```

Local JSONL, gzip JSONL and Parquet use explicit SHA-256 file receipts. Parquet
reads one row group at a time. Hub ingestion uses `datasets.load_dataset` with
`streaming=True`, a pinned revision, declared subset and split. It never downloads
an entire dataset into Python memory. The `data` extra installs the optional
Hub/Parquet readers. The default CLI row cap is 1,000; `--all-rows` is an explicit
acquisition choice.

After review, inspect a bounded preparation:

```bash
megatoken prepare --sources work/sources.json --output work/prepared \
  --codec work/codec --max-record-tokens 1024 --max-record-bytes 16384 \
  --benchmark-quarantine work/benchmark-quarantine.json
```

Malformed rows stop the default build with their source and row number.
`--reject-invalid` records rejection receipts and continues. The full rejected
row status remains in the preparation journal; `rejections.json` contains an
explicitly labeled bounded sample and the full count. A source that rejects
nearly every row is not silently counted as usable supply.

### Shared adapters and explicit field maps

One parser handles roles, original turn masks, complete tool calls/results,
outcomes, source spans and physical chunking. Source profiles supply field paths,
provenance and lineage requirements. OpenAI tool calls, legacy `function_call`,
Anthropic ordered text/tool blocks, Terminus-2 command batches and mini-SWE-agent
single-bash actions reuse that contract. Importing a recorded command never
executes a local shell.

The generic adapters are `document`, `chat`, `artifact`, `repository`, `fim` and
`native`. Repository repair places issue and pre-change files before the patch
target; review places the supplied patch before the review target. FIM explicitly
observes prefix/suffix before scoring the missing middle. Arbitrary corpus
variants use declared `field_map` dotted paths, not per-dataset parser forks.

An optional mapping file passed to `sources pin-* --mapping FILE` has the shape:

```json
{
  "field_map": {"text": "content", "family": "metadata.original_id"},
  "options": {
    "quality": {"min_tokens": 32, "max_tokens": 262144},
    "row_filters": [{"path": "metadata.quality", "op": "gte", "value": 0.8}]
  }
}
```

Quality policies can require scored positions, a verified success, an executed
observation, particular outcomes, and a maximum repeated-line fraction. Row
filters support a small explicit comparison vocabulary, without evaluating code.
Only use a field path and threshold that match the actual pinned source schema.

### Published source boundaries

The exact commit pins, subsets, row limits and outstanding joins are recorded in
[`sources.v13.json`](../configs/data/sources.v13.json). Primary publisher cards
provide the following schema/provenance information; adapters still require a
bounded import against the selected actual files before scale.

| Publisher source | Adapter behavior and material boundary |
| --- | --- |
| [Nemotron-CC v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) | Natural document text and synthetic variants are separate streams. Synthetic row UUIDs alone do not establish original parent lineage. |
| [Nemotron Pretraining Code v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | Reads delivered `content` with metadata. Synthetic corpus labels or unique row IDs are not substituted for original repository/file ancestry. |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Preserves ordered conversations with assistant-only targets. The selected subset is explicit in the manifest. |
| [Instruction Following Chat v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4) | Requires the original full-message `train_turns_mask`; excluded assistant turns remain context. Reasoning inclusion is explicit. |
| [Agentic v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4) | Keeps tool-call structure and marks synthetic observations as simulated. Missing original retrieval/backend documents are not invented. |
| [Open-SWE Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | Uses task/repository identities, messages/tools and publisher resolved status; reference patches remain metadata. |
| [Orchard](https://huggingface.co/datasets/microsoft/Orchard) | Handles JSON-string messages/tools/metadata and delivered anonymized SWE traces; uses the source's verification status. |
| [AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) | Imports original Terminus-2 histories and publisher outcomes. Heterogeneous scalar rewards are not assumed to prove test verification. |
| [Nemotron Terminal Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | Preserves executed terminal provenance. Selected adapters still need their exact schema and parent-task joins inspected locally. |
| [SWE-ZERO trajectories](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories) | Parses the mini-SWE action/result harness. `Submitted` does not mean tests passed; incomplete and unverified traces cannot fill verified-success quotas. |
| [Conversational Tool Use Seeds](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1) | Imports task seeds for an injected simulator. Customer scenario is private environment state; an actual communicated user message must be supplied separately. |

Several large procurement streams in the design remain unacquired: natural code
recovery, connected repository/issue/PR sequences, change histories and build
logs, sufficiently long verified dependencies, grounded artifacts, fresh student
corrections and fresh RL rollouts. They have usable adapter/collection interfaces
and declared mixture slots. A recipe target is not evidence of enough independent
high-quality examples to fill it.

## Freeze ancestry, deduplicate, then split

Preparation spools normalized episodes and source cursors to SQLite. It performs
exact role-aware transcript deduplication and bounded near deduplication using
word shingles, deterministic bottom-k sketches and indexed anchors. Union-find
joins explicit family/ancestor/repository/task relationships and duplicate
clusters. The component graph is completely resolved before any output shard is
released or any descendant's split is inherited.

Near matching is an approximate candidate search, not an exhaustive semantic
proof. `DedupConfig` makes thresholds, minimum text length, sketch/anchor sizes and
candidate limits explicit. Candidate saturation quarantines an item instead of
silently accepting an unchecked match. At large scale, validate the chosen recall
and rejection distribution on held-out duplicate labels. The per-original-task
variant limit defaults to five; a whole repository's independent files are not
mistakenly limited to five records.

For sources missing parent ancestry, supply a pinned JSONL join:

```json
{"row_id":"synthetic-row-7","family_id":"original-document-42","ancestor_ids":["original-document-42"]}
```

Configure `options.lineage_join` with `path`, `sha256` and `row_key` (for example,
`"uuid"`). The join is loaded into its own external-memory SQLite index. Missing
or conflicting parent rows reject the variant; no content fields can be changed
through this join. Relative join paths resolve against the source manifest.

The quarantine snapshot accepts `family_keys` and exact `content_sha256` values;
the example file documents the interface. Matching episodes, all connected
descendants and benchmark-tagged families stay out of exported training/eval
shards. Optional date holdout normalizes ISO-8601 offsets to UTC; date-only and
timezone-naive values are interpreted as UTC.
Inherited frozen evaluation splits never move into training; conflicting
inherited splits quarantine the whole connected family. Add newly discovered
cross-family relationships in a new complete preparation build.

An interrupted source row is either fully committed with its cursor or not
committed. An interrupted export can be rebuilt from the frozen spool; the final
manifest is the commit marker. Local JSONL resume scans old lines without parsing
them; local Parquet skips completed row groups. Hub resume counts logical rows
sequentially and reads preceding rows again, avoiding format-dependent Arrow
`skip` behavior. This implementation has
not been benchmarked on a multi-hundred-billion-token acquisition run.

## Causal windows and long-context credit

`iter_windows` never shuffles, crosses episodes, truncates actual earlier history,
or splits a complete logical action. Each record appears exactly once as new D
in one pass. Later windows carry the complete actual prefix as reconstruction
I, with no policy loss on the prefix. Ordinary prose and observation parts are
valid window boundaries, so a long document cannot defeat `max_records` merely
by retaining one logical source ID. Limits remain soft for complete tool actions
and explicitly extended dependencies; any token overshoot is reported.

An annotated writer receives gradient from a later reader only if both are in
the window's differentiable `records`. With `respect_dependencies=True`, packing
extends the segment to include linked readers and applies the explicit
`max_dependency_tokens` hard graph limit. Prefix replay alone does not qualify
as a writer-reader training window. `WindowAudit` counts actual writer-reader
scored windows, long-history D with a late dependency, and long-context I, and
can enforce the requested fractions at a stage gate.

The canonical stage recipes live once in packaged
[`stages.json`](../src/megatoken/data/stages.json). Read them with
`stage_recipe(phase)`; the training sampler consumes `.mixture`, `.meter` and
`.limit`, and reports actual consumed exposure rather than row-count fractions.

| Phase | Budget | Main role |
| --- | ---: | --- |
| B0 | 250M I | Local skills/bootstrap with prior paths active |
| P1 | 400B D | Broad natural/synthetic document, code, reasoning and trace foundation |
| M1 | 60B D | Connected repository, issue/PR/change/build context with replay |
| M2 | 40B D | Verified trajectories, grounded artifacts and corrections |
| LC | 20B D | Long dependencies, code replay and mixed-quality context |
| S | 2B P | Verified assistant behavior, recovery, tested artifacts and exact instructions |
| D | 1B P | Frozen-student-state corrections and successful replay |
| R | 32B Z | Fresh executed rollouts with one Gaussian facet action per prediction |

The JSON preserves the stated effort and history schedules, exposure mixtures,
LC requirements (at least 50% long-dependency D and 30% differentiable
writer-reader scored windows), S context/control constraints, and correction/RL
gates. B0's single `bootstrap` source stream is explicitly an implementation
choice. R rollout collection and PPO replay belong to the training runtime;
teacher T does not substitute for RL Z. R also retains its independent episode
ceiling; emitted byte/BPE counts are reported separately from Gaussian work.

## Run a budgeted collection pilot

`TaskSeed` separates a public prompt/policy/tool schema from private environment
configuration and source ancestry. CLI task rows use `seed_id`, `family_id`,
`category`, `prompt` (or `prompt_base64`), `environment`, `tools`, `revision`,
optional ancestors and `mode`. Tools default only for the local fixture. See
[`task.example.jsonl`](../configs/data/task.example.jsonl).

The fixture executes isolated in-memory file operations (`list`, `read`, `write`,
`finish`) with a hidden expected-files verifier. It never runs host shell code.
For a remote executor, use the shared `HTTPEnvironment` reset/step/verify protocol;
the remote service owns its actual repository/container sandbox and test runner.
Every step includes the expected current state and an immutable idempotency key,
and returns a fresh state and execution receipt. Verifier labels are stored in
supervision, never appended to observations.

Validate the explicit fixture task and a small collection configuration without
contacting a teacher:

```bash
megatoken collect --tasks configs/data/task.example.jsonl \
  --output work/collection-smoke --config configs/data/collection.smoke.json \
  --teacher-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --teacher-model YOUR_LOCAL_TEACHER --environment fixture --codec byte --dry-run
```

Run the same command without `--dry-run` against your configured teacher to
perform the collection. Credentials are read only from the named `--token-env`
and `--environment-token-env` environment variables; they are not stored in
manifests. Remote services use HTTPS; localhost HTTP is supported for local
servers. The client makes no speculative retry after a transport failure.

Without a smoke override, collection reads the canonical 10,000-task pilot,
two-attempt limit and 200M T budget from `stages.json`. Its seed quotas are 35%
repository, 20% terminal, 20% structured tools, 15% long artifacts and 10%
evidence, with deterministic largest-remainder quotas for small test budgets.
Stable original task/family IDs must represent real ancestry. Rephrasing a prompt
does not create a new executable task within a family. An accepted task cannot
be repeatedly counted as fresh accepted supply.

`CollectionLedger` reserves each request before submission, counts failed and
interrupted attempts against the per-task cap, and charges rejected output.
Teacher input, exposed reasoning, provider output/input counts and cache usage
are recorded separately. Raw input/T use the pinned reference codec; provider
and cache counters retain provider-native units. An unresolved request retains
its remaining reservation, reports `T_complete=false`, and cannot be retried
under the same request ID. Known hidden reasoning without complete raw bytes
stops collection with that reservation held instead of claiming complete T.

The provider must honor its output limit. A generic Chat Completions server's
native `max_tokens` need not equal the pinned raw codec's count. If a response
exceeds its raw reservation, the ledger records the **full actual usage** and
stops; it never clips the bill to make a cap appear satisfied. Production
collection should verify this token convention with the actual configured
teacher during the bounded pilot.

Re-running the command continues the same task/config/source-bound ledger.
Use `--max-new-attempts` for a bounded slice. For a confirmed stopped worker,
`--recover-attempt ID` marks its attempt interrupted while retaining unknown
usage. `--reconcile-usage receipts.jsonl` accepts known `request_id`,
`raw_output` and optional provider/cache counts; it settles usage without
resubmitting the request. Keep upstream response/usage receipts to support that
reconciliation. Changing the declared budget requires a separately budgeted
ledger.

Each collection run without `--dry-run` exports accepted and failed canonical
episodes, a census, and a pinned source manifest in
`work/collection-smoke/export`. Import that manifest through
the regular `prepare` path. Inherited splits and actual stream labels survive
the export; failed branches remain context with their explicit recovery masks.

### Frozen student correction and long artifacts

Add `--student-bundle work/frozen-served-student --student-prefix-steps 2` to
collection with a new output directory: an existing ledger is bound to its
original student configuration. `ModelTeacher` loads the shared checksum-verified
inference engine, uses prior public records, and forwards the public task policy
and tool schema through the same control builder as training and serving.
Private environment configuration and verifier labels never enter these controls.
It retains its exact native recurrent session between student actions. Emitted
IDs, serialized action bytes and complete version-3 raw-facet tapes are retained,
including each draw's exact encoded-source control bytes. Every selected facet
draw advances state once, including an invalid decode, and its output bytes are
not ingested again. Each tape's metadata carries a `record_cursor` giving the
number of physical records that preceded the draw, so invalid attempts retain
their exact chronological position. Valid empty draws also own no transcript
record and retain their position through this cursor.
The collector executes only valid actions against the current environment;
invalid actions produce no invented environment observation. New actual
observations are ingested before giving the resulting full history to the teacher.
Student outputs are context, teacher corrections are targets, and student draws
do not inflate teacher T.

An already-terminated student task is not labeled a correction; collect it through
successful replay. D's default correction/replay proportions and its gated
correction ramp are in the stage recipe. Use distinct source stream mappings to
make both populations available to the scheduler.

The Python `Collector` also supports `mode="artifact"` with an explicitly injected
artifact verifier. Truncated artifact output continues from the exact accumulated
assistant text through another causally ordered request; it does not restart from
an empty prefix. A missing verifier produces an unverified outcome rather than
inventing success. Simulated tool/customer environments use the same protocol
but must return `provenance="simulated"`; private customer intent remains in that
environment. These extension points share the collection journal, budgets,
masking and episode serializer.

## Validation and scope

The data tests exercise binary/UTF-8 round trips; exact spans and masks; actual
emitted-ID versus reference-meter separation; tool/result identity and causality;
failed-trace recovery; native publisher schemas; exact/near dedup and family
quarantine; inherited splits; checksummed lineage joins; tampered files; interrupted
preparation; rank partitioning; raw budget overshoot and unresolved hidden output;
fresh executed student correction; frozen native tape retention; CLI pin/prepare
round trips and no-network dry runs. Real local PyArrow/datasets integration
covers multiple row groups, resume offsets and null rejection. Semantic chunk
tests cover paragraph/list/code preservation, token-crossing boundaries, Unicode,
CRLF, binary input, forced cuts, stage caps, immutable native output, versioned
identity, size statistics and bounded document windows.

These tests establish implementation invariants on small CPU fixtures. They do
not establish public-source yield, large-scale near-dedup recall, remote sandbox
correctness, teacher quality, long-context task coverage, or Blackwell throughput.
The bounded source/teacher pilots and the resulting actual census provide the
evidence needed to choose production acquisition and training scale.
