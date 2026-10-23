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
checksummed JSONL shards, `index.sqlite`, `preparation.sqlite`, a rejection
report, and `parser-review.json`. An identical invocation resumes an interrupted build or returns the
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
version-4 facet tapes (`fmt-canonical-facets-v4`), which retain the exact public
control payload used by the prior. Older tapes must be regenerated.

| Field group | Meaning |
| --- | --- |
| `records` | Actual physical record order, role, logical/physical identities, part index/count, byte payload and source span |
| `lineage` | Source revision, original family, all known ancestors, repository/task identities, original row hash and optional inherited split |
| `actions` | Complete tool action descriptor, immutable serialized-byte hash, exact physical members, call ID and execution ID |
| `result_of` | An observation's exact prior completed action; a changed action cannot inherit another action's result |
| `policy_mask`, `free_spans`, `emitted_free_mask` | Role/turn eligibility and lexical bytes or emitted positions eligible for supervised decoder cross-entropy |
| `chunk_boundary`, `chunking_fingerprint`, `forced_chunk_split` | Auditable semantic boundary, exact chunking policy identity, and unavoidable cut marker |
| `chunk_context`, `chunk_languages`, `chunk_diagnostics` | Original parser context, language labels, and bounded diagnostic codes for each physical record |
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

New codec-backed imports use the shared **512-token target, 768-token soft limit,
and 1,024-token hard limit**. Tokens here mean positions in the configured raw
reference codec, not words or characters. The independent physical byte limit is
16 KiB. Explicitly smaller stage limits clamp both semantic sizes to the available
capacity. The 32-facet architecture and 1,024-position realizer allocation are
unchanged by the new data policy.

The parser establishes outer conversation boundaries before interpreting content.
Structured user messages, assistant emissions, tool calls, and tool results are
separate logical fields. Every user message and every tool result has a boundary
on both sides; tiny events never borrow adjacent events to fill spare capacity.
Known serialized chat formats are framed before their header/body/footer fields
receive canonical encodings. A quoted chat example inside an already structured
message is ordinary content. The `serialized_chat` adapter and automatic transcript
recognition for documents preserve exact source ranges and role provenance.

Inside an event, parsing proceeds through a shared hierarchy: Markdown and other
documents, source code, markup, strings, comments, nested code, and prose. The
installed offline grammar baseline has 173 bindings. Generic lexical analysis
handles unknown languages and incomplete snippets explicitly; a grammar name alone
is not a claim of perfect parsing. Multilingual prose starts with paragraphs,
Unicode separators, and script-appropriate punctuation. Sentence/word and optional
locale refinement run only for spans that need subdivision. Extended grapheme
clusters, CRLF, raw escapes, and exact source offsets are retained.

The planner selects a path through legal canonical token boundaries using bounded
dynamic programming. It considers the remaining source, penalizes tiny tails and
soft-limit overflow, and adds a language-sensitive cost for excessive movement
through the parse tree. Headings, decorators, and function signatures attach to
their own following content. Whole small containers can pack together; large
following sections and functions introduce structural barriers. Complete JSON
values and protected introductions may use available hard capacity. Unavoidable
cuts remain explicit in `forced_chunk_split` and diagnostic metadata.

```python
from megatoken.data import ChunkingConfig, segment_with_report

policy = ChunkingConfig()  # target=512, soft=768, hard=1024
report = segment_with_report(
    payload, codec, policy, content_kind="code", language="python", chat_template="none"
)
assert b"".join(part.payload for part in report.chunks) == payload
for part in report.chunks:
    assert part.payload == payload[part.byte_start : part.byte_end]
    assert codec.decode(part.token_ids) == part.payload
```

Each logical input field is encoded once. Physical records contain contiguous
slices of its canonical token IDs. Serialized framing establishes independent
canonical fields first, so no BPE token can cross a latency boundary. Their
concatenated IDs are not claimed to equal a separate whole-transcript encoding.
A content boundary inside a BPE token is skipped unless it can move within a
whitespace separator. No alternate IDs or rewritten text are used to force an AST
boundary. `U` therefore remains independent of physical packing within the same
logical-field/framing policy.

Use `--chunking-config configs/data/chunking.production.json` for the full parser
and planning policy. `--chunk-target-tokens`, `--chunk-soft-max-tokens`,
`--max-record-tokens`, and `--max-record-bytes` provide explicit overrides; the two
record limits set the hard physical limits, whether that lowers or raises the policy's. Parsing hints include `content_kind`, `language`, `filename`, `mime_type`,
`locale`, and outer `chat_template`. Source options, row metadata, messages, and
content blocks share one allowlisted hint path; a more local hint takes precedence.
`locale` is the preferred natural-language hint. Templates are interpreted only
at the outer framing layer.

The manifest records the algorithm version, complete policy, exact codec identity,
parser package/provider versions, Unicode/optional ICU identity, and an
implementation hash covering parsing and logical-field construction.
`ChunkingConfig.from_identity(identity, codec)` rejects an incompatible runtime.
Prepared data, exact training resume, and served observations use the same policy.
`recursive-parser-v2` replaces `semantic-v1`; regenerate old prepared artifacts
under the new identity. Native generated actions retain their physical records,
IDs, and facet tapes. An oversized native action is rejected; it cannot be silently
rechunked into a different recurrent trajectory.

`manifest["chunking_stats"]` reports accepted record counts, byte/reference-token
totals, actual decoder positions, size histograms, mean/maximum positions,
soft-limit exceedances, boundary kinds, forced cuts, languages, and diagnostic
counts. These are inventory statistics rather than training exposures.

### Agent review during preparation

Parser faults, exhausted analysis budgets, severe syntax failure, invalid nested
ranges, and ambiguous conversation framing are held out of accepted shards by
default. Preparation captures the exact decoded source row, pinned source receipt,
lineage, parser identity, sizes, quality policy, diagnostics, and lossless episode
where available. The capture and source cursor commit in one SQLite transaction.
Known ancestry is registered even for held-out rows so parser failures cannot
hide benchmark family relationships from split quarantine.

```bash
megatoken chunks review list --queue work/prepared/preparation.sqlite
megatoken chunks review claim --queue work/prepared/preparation.sqlite \
  --owner parser-agent-01
megatoken chunks review export --queue work/prepared/preparation.sqlite \
  --output work/parser-tasks
```

The exported tasks contain fixed `AGENT_TASK.md` instructions plus separate exact
sample files. Agents amend the shared parser or registry and add a regression case;
they never edit the captured training example. Claims have renewable leases, so
multiple agents can work without taking the same task. After the fix, the claiming
agent runs `chunks review replay` with the original codec and a commit/PR reference.
Only a changed parser identity that passes the normal adapter, source, quality,
capacity, and review gates can mark the task verified. Failed replay writes its
receipt and exits with status 1. Verified repair requires a new preparation output
from the same pinned sources; existing accepted shards remain frozen.

`parser-review.json` is the preparation-time summary, and `manifest["parser_review"]`
records the same snapshot. Use the queue CLI for current claims and verified tasks.
`--parser-review-config configs/data/parser-review.json` selects the review policy.
Ordinary unknown grammar and localized syntax recovery remain accepted under the
lenient default; the explicit policy can hold them out as well. Native records and
tapes are exempt from resegmentation and historical parser diagnostic review.

See [the parser and review runbook](chunking.md) for supported template families,
nested content behavior, resource bounds, extension interfaces, replay commands,
identity migration, and operational limitations.

The serving runtime's `observe_text` applies the same content chunker to a complete
authoritative message. Already prepared records are observed whole (`observe`), which
preserves their supplied boundaries; the frozen student reads collected records this way. Every selected action advances recurrent state
with its sampled raw facets, including invalid decodes; validation gates environment
execution only. Deterministic output bytes are never re-encoded to approximate that
latent action. Training supervises each chunk's exact output length and marks only
the last physical part of a logical output as terminal.

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
records actual P from evaluated target plans. Both bootstrap and recurrent
training evaluate a causal prediction pass and a reconstruction pass, so P is
twice the eligible target positions. Bootstrap reconstructs from a noisy
encoder draw directly. Recurrent reconstruction trains only the encoder through
the fixed central/realizer map and detached pre-action context. Optional facet
following uses only a native tape paired with the exact bytes its action produced;
ordinary offline text does not supply a latent alignment target. RL lexical decoding
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

Parser failures enter the durable review queue described above. Ordinary malformed
rows stop the default build with their source and row number.
`--reject-invalid` records rejection receipts and continues. The full rejected
row status remains in the preparation journal; `rejections.json` contains an
explicitly labeled bounded sample and the full count. A source that rejects
nearly every row is not silently counted as usable supply. Rows excluded by a
source's own declared `row_filters` are selection, not errors: they are counted
as `filtered_rows` and never stop the build. An error that is not about a row's data
(a bug in an adapter) always stops it, even with `--reject-invalid`. Source
`options` are checked against the names the selected adapter reads, so a typo fails
before any row is read. Rows are committed to the preparation journal in batches
together with their cursor, so a crash repeats at most one batch.

### Shared adapters and explicit field maps

One parser handles roles, original turn masks, complete tool calls/results,
outcomes, source spans and physical chunking. Source profiles supply field paths,
provenance and lineage requirements. OpenAI tool calls, legacy `function_call`,
Anthropic ordered text/tool blocks, Terminus-2 command batches and mini-SWE-agent
single-bash actions reuse that contract. Importing a recorded command never
executes a local shell.

The generic adapters are `document`, `chat`, `serialized_chat`, `artifact`,
`repository`, `fim` and `native`. Serialized tool trajectories require the original
structured call/result identities; the importer never invents causal dependencies
from ambiguous markup. Repository repair places issue and pre-change files before the patch
target. FIM explicitly
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
| [Instruction Following Chat v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3) (instruction-following split; v4 was withdrawn) | Requires the original full-message `train_turns_mask`; excluded assistant turns remain context. Reasoning inclusion is explicit. |
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
high-quality examples to fill it. The later-stage streams among them now have pinned public
sources; see "Later-stage streams" at the end of this page for what each holds and what
is still short.

## Open-licence pilot mix

Every declared stream needs prepared episodes before its stage can train. This section
records where each stream's pilot data comes from, the licence found on the publisher's
dataset card (read 2026-10-22; a card is not legal advice), and what is still missing. The
pilot is small (a few thousand episodes per stream) so the full pipeline can be exercised;
the stage plan's targets are orders of magnitude larger.

`configs/cloud/prepare-open-mix.json` builds five prepared roots in one sandbox (nothing
launches by itself; it needs the approvals below, a clean commit, and
`--secret hf-token=FILE`):

| Root | Built by | Streams |
| --- | --- | --- |
| `text-p1` | `megatoken prepare --sources configs/data/sources.open.json` | natural_documents, natural_code, synthetic_documents, synthetic_reasoning, chat, synthetic_code_dialogue, executed_trajectories |
| `text-mb` | `megatoken prepare --sources configs/data/sources.open-mb.json` | text (replay for the media bridge) |
| `media` | `scripts/prepare_open_media.py` (readers in `data/open_media.py`) | audio_identity, image_identity, video_identity, transcription, verbalization, speech_translation, image_description, image_creation, image_edit, speech_edit, av_creation, av_edit |
| `long-reading` | `scripts/prepare_long_document_reading.py` | long_document_reading |
| `base-reading` | `scripts/prepare_base_reading.py` | base_reading (questions whose reference items are in a knowledge base, not the context; same HotpotQA, DOCCI and LibriSpeech data as above) |
| `collection-reading` | `scripts/prepare_collection_reading.py` | collection_reading (with its `collections/` of supporting items) |

MB trains on `text-mb` + `media` (identity streams only); the reduced P1 trains on all five.
Splits freeze per original family (speaker, FLoRes sentence, photo, COCO image, recording,
repository, web page), and every media-bundle task is compiled by `data/concordance.py`,
which emits the stage plan's own stream names.

### Text sources (`sources.open.json`, `sources.open-mb.json`)

| Stream | Source (pinned commit in the manifest) | Licence on the card | Notes |
| --- | --- | --- | --- |
| natural_documents, text | `HuggingFaceFW/fineweb-edu` `sample-10BT` | ODC-By 1.0 | Also subject to Common Crawl terms of use. |
| natural_code | `common-pile/stackv2_edu_filtered` (5 shards) | per file, Blue Oak certified permissive list | Licence labels can be wrong (see the card); family = repository. |
| synthetic_documents | `HuggingFaceTB/cosmopedia` (`web_samples_v2`, `stanford`, `stories`) | Apache-2.0 | Generated by Mixtral-8x7B-Instruct. |
| synthetic_reasoning | `nvidia/OpenMathInstruct-2` (`train_1M`); `open-thoughts/OpenThoughts-114k` | CC-BY-4.0; Apache-2.0 | Llama-3.1-405B and DeepSeek-R1 generated; Llama attribution terms may follow the outputs. |
| chat | `OpenAssistant/oasst_top1_2023-08-25`; `HuggingFaceTB/smoltalk` (4 new subsets) | Apache-2.0 | OASST is human written. SmolTalk's other subsets are third-party and are filtered out by a row filter. |
| synthetic_code_dialogue | `nvidia/OpenCodeInstruct` | CC-BY-4.0 | LLM generated; its test results are not imported as outcomes. |
| executed_trajectories | `nvidia/Open-SWE-Traces`, `open-thoughts/AgentTrove`, `nvidia/Nemotron-Terminal-Corpus` | CC-BY-4.0, Apache-2.0, CC-BY-4.0 | Copied from `sources.v13.json` at the same revisions. |
| long_document_reading | `hotpotqa/hotpot_qa` (existing loader) | **CC-BY-SA-4.0** | ShareAlike; the maintainer must decide whether that is acceptable. MuSiQue's card names no licence. |
| base_reading | the sources above: HotpotQA paragraphs, DOCCI photos, LibriSpeech clips | CC-BY-SA-4.0, CC-BY-4.0, CC-BY-4.0 | The same items as `long_document_reading`, `image_description` and `transcription`, read through a knowledge base instead of the context. |
| collection_reading | the same HotpotQA loader (supporting paragraphs are the annotated items) | **CC-BY-SA-4.0** | As above; the paragraphs are copied into the prepared root's collections. |

Every source keeps `terms.approved: false`: recording approval is the maintainer's
decision, so the job's text steps stop with "Source terms are unapproved" until it is
done. To approve, set `approved: true` with `reviewed_by` and `reviewed_at` for each
`source_id` in `sources.open.json` (14: `fineweb-edu-10bt`, `stackv2-edu-0000/0020/0040/0060/0080`,
`cosmopedia-web_samples_v2/stanford/stories`, `openmathinstruct2-1m`, `openthoughts-114k`,
`oasst-top1`, `smoltalk-new`, `opencodeinstruct`)
and for `fineweb-edu-text` in `sources.open-mb.json`.

### Media sources (`data/open_media.py`, `SOURCES`)

| Streams | Source | Licence on the card | Notes |
| --- | --- | --- | --- |
| audio_identity, transcription, speech_edit | `openslr/librispeech_asr` `train.clean.100` | CC-BY-4.0 | Utterances up to 20 s; family = speaker. |
| verbalization | `mythicinfinity/libritts_r` | CC-BY-4.0 | Original text, normalized spoken form (when it differs) and a same-speaker voice sample. |
| speech_translation, transcription (de, fr, es, it) | `google/fleurs` train (de, fr, es_419, it to en_us) | CC-BY-4.0 | The same FLoRes sentence read in two languages; translated text is the English transcript. Family = sentence. |
| image_identity, image_description, image_creation | `google/docci` (images and descriptions from the publisher's bucket, pinned by sha256) | CC-BY-4.0 | The creation request is the description's first sentence. |
| image_edit | `osunlp/MagicBrush` train | CC-BY-4.0 | **Caveat:** source images are MS-COCO (mixed per-image Creative Commons licences, some NonCommercial/NoDerivatives); the card does not address it. Pilot scale only until cleared. |
| video_identity, av_creation, av_edit | `RekaAI/RekaDaily-10k-processed` `video_capture` | Apache-2.0 | First-person clips with sound and a dense caption; the creation request is the clip's activity label. May contain incidental speech. |
| speech_edit, av_edit | procedural | n/a | Deterministic ffmpeg edits (louder, faster, backwards; mirror, grayscale, double speed, reverse) of the clips above, recorded with `executed` provenance and verification tier `procedural`. No open dataset supplies human speech or audiovisual edits. |

Media records carry no `TermsReview`; each episode's annotations record `source_license`.
The FLEURS tsv files and the DOCCI archive are the only inputs not hashed by the Hub: their
sha256 was computed from the downloads.

### Still missing

- **synthetic_code_rewrite** and the **other phases' streams** (M1/M2/LC/S: repository,
  issue/PR, build-log and verified streams) were not sourced in this pilot. They are now in
  `configs/data/sources.later-stages.json`; see "Later-stage streams" at the end of this
  page. `sources.v13.json` pins the Nemotron code subsets, which stay behind the NVIDIA
  Data Access Agreement and need `options.lineage_join` parent files.
- **Higher-volume captioned video with speech**: `HuggingFaceFV/finevideo` (CC-BY YouTube videos with
  sound and rich annotations) is gated and this account has not accepted its terms (HTTP 403);
  accepting them would also add removal-request obligations. Common Voice, CoVoST 2
  (`facebook/covost2` is CC-BY-NC-4.0 on its card, the `fixie-ai` mirror has no licence) and
  UCF101 (the Hub subset says Apache-2.0, but UCF101's own distribution is research-only) were
  not used for licence uncertainty. OpenVid-1M (CC-BY-4.0 card) has no audio, unclear
  upstream rights and 50 GB archives.
- **Scale**: counts in the job are pilot sizes; DOCCI (about 9.6k training photos) and
  MagicBrush (about 8.8k turns) cap their streams well below the plan.

## Agentic and chat data

The open-licence pilot above fills every stream with a few thousand episodes. This section
records the much larger agent-trajectory, tool-use, chat and reasoning sources added after
it, how each is turned into episodes, how the stage plan now leans on them, and what is
still short. Licences are the ones printed on each publisher's dataset card, read on
2026-10-22 (a card is not legal advice); every source is recorded in
`configs/data/sources.agentic.json` with `approved: true`, `reviewed_by: Gabriel Pickard`,
`reviewed_at: 2026-10-22`. No source was left out for its licence. Non-commercial,
share-alike, research-only and model-licence terms are named in the tables below and in each
entry's `terms.notes`, so a release decision can filter on them.

Sources are pinned to a 40-character dataset commit. Row counts and token counts come from
the cards and file sizes (token counts are our estimate at about four bytes per token,
rounded, taken after the filter described for each source). Each source was read through
its real adapter on its first 12 to 150 rows from the pinned revision; the pass rate, and
what the rejects were, is in "What was checked".

### Why the stage plan moved

Raw web text and generic synthetic documents are cheap but teach no tool use, no
multi-step recovery and no use of an executed result. Agent trajectories are the opposite:
each assistant action is scored and each environment reply is context that is not scored.
The plan therefore moves share from raw text towards four agent streams and chat:

| Stream | What it holds |
| --- | --- |
| `executed_trajectories` | Long agent runs in a real sandbox (shell, editor, tests). The outcome is either recorded by the publisher or, for curated releases, assumed successful (see "Outcome labels"). |
| `verified_trajectories` | Runs whose hidden tests passed. The training engine refuses this stream unless every episode carries a recorded passing verdict, so only sources filtered to passing rows go here. |
| `tool_use` | Function-calling conversations: tool definitions, calls, results, a final answer. Results are mostly simulated; executed ones are labelled as such. |
| `environment_trajectories` | Short text-game or web-shop style tasks (AgentInstruct, AgentBank, AgentTraj-L). Tiny but diverse. |
| `chat` | General assistant conversations. |

Share changes (each phase sums to 1; shares not listed are unchanged, including every media
share and `long_document_reading`):

**P1** (400B tokens)

| Stream | Before | After |
| --- | --- | --- |
| executed_trajectories | 4.1% | 17% |
| verified_trajectories | - | 5% |
| tool_use | - | 6% |
| environment_trajectories | - | 0.1% |
| chat | 2.05% | 8% |
| synthetic_reasoning | 6.15% | 10% |
| natural_documents | 16.4% | 6% |
| synthetic_documents | 20.5% | 8% |
| natural_code | 12.3% | 8% |
| synthetic_code_rewrite | 12.3% | 3.9% |
| synthetic_code_dialogue | 8.2% | 8% |
| base_reading | - | 1% |
| collection_reading | - | 1% |

Raw and generic web text (`natural_documents` plus `synthetic_documents`) falls from 36.9% to
14% (two points went to reading from a knowledge base and from a source collection); the
three agent streams plus environment tasks rise from 4.1% to 28.1%.
`synthetic_code_rewrite` has a source only since the later-stage round (see "Later-stage
streams"); the smaller share is kept.

**M1** (60B): `executed_trajectories` 9.3% to 21%, new `verified_trajectories` 8%,
`tool_use` 4%, `chat` 2%; paid for by `issue_pr_sequences` 37.9% to 28.4%,
`connected_repositories` 18.6% to 13% and `base_replay` 18.6% to 8%.

**M2** (40B): `verified_trajectories` 46.5% to 49%, new `executed_trajectories` 8%,
`tool_use` 7%, `chat` 2%; paid for by `grounded_artifacts` 18.6% to 9% and `base_replay`
14% to 4.05%.

**LC** (20B): new `executed_trajectories` 10%, `verified_trajectories` 10%, `tool_use` 5%;
paid for by `long_verified_dependencies` 47.5% to 40%, `long_natural_code_replay` 23.75% to
10% and `mixed_quality_episodes` 23.75% to 20%.

**S** (2B scored positions): `verified_successful_multiturn` 57% to 46%, new `tool_use`
6% and `chat` 5%, `typed_exact_instructions` 9.5% to 14%; `natural_document_replay`
4.75% to 2%, `tested_artifacts` 9.5% to 8%.

### Unique tokens against planned exposure

The plan only helps if the data exists. "Unique tokens" is what the pinned sources hold
after their filters (a lower bound on what can be prepared; the first-run caps below
prepare far less). "Planned" is the share times the phase budget. The last column says how
many times each unique token would be seen if the plan ran as written; above about 4 it
is time to find more data or lower the share.

| Stream | Unique tokens (B) | P1 | M1 | M2 | LC | Planned total (B) | Times each token is seen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| executed_trajectories | 47.2 | 68.0 | 12.6 | 3.2 | 2.0 | 85.8 | 1.8x |
| verified_trajectories | 20.9 | 20.0 | 4.8 | 19.6 | 2.0 | 46.4 | 2.2x |
| tool_use | 13.0 | 24.0 | 2.4 | 2.8 | 1.0 | 30.2 | 2.3x |
| chat | 10.2 | 32.0 | 1.2 | 0.8 | - | 34.0 | 3.3x |
| synthetic_reasoning | 23.1 | 40.0 | - | - | - | 40.0 | 1.7x |
| mixed_quality_episodes | 12.0 | - | - | - | 4.0 | 4.0 | 0.3x |
| environment_trajectories | 0.1 | 0.4 | - | - | - | 0.4 | 4.0x |

S (scored positions, not tokens): `recovery_clarification` plans 0.28B positions from a
source of about 7M tokens (When2Call), roughly 40 passes over the same 15,000 rows. That is
the worst mismatch in the plan; the stream needs generated or collected recovery
episodes. `typed_exact_instructions` has about 0.28B tokens against 0.28B positions
planned. `verified_successful_multiturn` re-reads the same pinned verified sets as
`verified_trajectories` through second manifest entries (`s-verified-*`), so it adds no
unique tokens.

Gated sources that need the maintainer's click (next section) would add about 12B tokens
of chat. Exposure of the chat stream drops to about 1.5x if they are accepted and prepared.

### How rows become episodes

One adapter profile per data shape (`PROFILES` in `data/adapters.py`); converters for the
tool-call spellings live in `data/dialects.py` and all return the same Anthropic-style
blocks (`tool_use` and `tool_result`), so one pairing routine decides which call a result
answers. Calls are assistant records; every tool result is its own `tool` record that is
never trained on. Policy masks fall only on assistant turns the source marks as trained
(or all assistant turns when it marks none), never on a result, a system prompt or a user.

| Profile | Shape it reads | Used for |
| --- | --- | --- |
| `open_swe` | OpenAI chat messages with `tool_calls` and `tool` replies | Open-SWE, Nemotron SWE sets, SWE-Hero, ReBench, CoderForge, SWE-Gym |
| `orchard_swe` | the same, with Orchard's metadata | Orchard |
| `openhands_xml` | calls written as `<function=...>` blocks in assistant text | R2E-Gym, SWE-Dev, SWE-smith |
| `swe_agent_fenced`, `mini_swe_agent` | an action in a fenced code block, result in the next user turn | Nebius SWE-agent, SWE-smith variants |
| `codeact_agent` | `<execute>` blocks | CodeActInstruct |
| `agent_trove`, `nemotron_terminal` | JSON batches of terminal keystrokes | AgentTrove, Nemotron-Terminal |
| `nemotron_agentic_v4` | OpenAI tool-calling messages with a tool list | Nemotron agentic sets, ToolMind |
| `hermes_tools` | `<tool_call>` / `<tool_response>` tags in text | Hermes function calling |
| `python_call_tools` | `[f(a=1)]` call text, or Dolci's separate `function_calls` field | ToolACE, Dolci |
| `glaive_tools` | Glaive's `SYSTEM:/USER:/ASSISTANT:/FUNCTION RESPONSE:` text dump | Glaive function calling |
| `xlam_calls` | a query, a tool list and the expected calls (no result) | xLAM |
| `simulated_tool_roles`, `mcp_tools` | `function_call` / `observation` or `tool_call` / `tool_response` roles | APIGen-MT, Toucan |
| `react_agent` | Thought/Action replies and environment replies as turns | AgentInstruct, AgentBank, AgentTraj-L, Agent-FLAN |
| `turn_pairs` | input/output pairs per turn | Capybara |
| `chat`, `artifact` | plain messages, or a prompt and a response | the rest |

Rules that apply everywhere:

- **Executed or simulated.** Each profile states whether tool results came from running
  something (`executed`), or from a model playing the tool (`simulated`), or are
  unknown. APIGen-MT, Glaive, Hermes, ToolACE, xLAM, Dolci and Nemotron agentic tool sets
  are simulated; Toucan calls were executed against real MCP servers; SWE and terminal
  traces are executed. Agent-FLAN states nothing, so it is `unknown`. The training
  engine's executed-observation requirement is switched on for the SWE and terminal
  sources only.
- **Structured results.** A tool result that arrives as an object or list is stored as
  canonical JSON, with the transformation recorded as `canonical-json-v1`.
- **Strict pairing.** A result must answer exactly one earlier call. Call IDs are used where
  the source has them; sources without IDs set `positional_call_ids`, which pairs results
  to calls in order. A reused ID, an orphan result or an answer to no call rejects the row.
- **Withheld prompts.** A row whose first user turn is null or empty is rejected with
  "a source that withholds prompts must be restored first" instead of training on a
  conversation with its question missing.
- **Tool schemas.** Null fields and Python type names (`str`, `List[float]`, `"int, optional"`)
  in tool definitions are cleaned to JSON Schema; unusable ones reject only that row.

#### Outcome labels

| Label | When it is used | Verification recorded | Streams it may enter |
| --- | --- | --- | --- |
| success, `publisher_tests` | the source records a passing hidden test and we kept only passing rows (row filter) | the publisher's tests | `verified_trajectories`, `verified_successful_multiturn` |
| success, `publisher_curated` | the whole release is the publisher's kept, successful runs but carries no per-row verdict (`assume_outcome: "success"`) | curated, not verified | `executed_trajectories` only |
| failure or success, `publisher_tests` | the source records a mix (Open-SWE-Traces) | the publisher's tests | `executed_trajectories`; failed turns need explicit recovery masks |
| unknown | the source says nothing | none | allowed in `executed_trajectories` only when `admit_unknown_outcome` is set (AgentTrove, Nemotron-Terminal); chat and tool use do not need an outcome |

A curated release is never relabelled as verified, so the engine's refusal of unverified
episodes in the verified streams still protects them.

### Sources by stream

The agent-trajectory sources that execute code in a sandbox (software-engineering and
terminal rollouts, code-action traces) moved to their own manifest,
`configs/data/sources.agentic-coding.json`, with no row caps; see "Agentic coding sources"
below. What stays here are the chat, reasoning, tool-use and environment (web, game, database)
sources, whose first-run caps count rows read from the start of the pinned data.

#### tool_use

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `nemotron-agentic-v4-tools` | `nvidia/Nemotron-SFT-Agentic-v4` (default) | openmdw-1.1 | 3,000 | 1.5B |
| `nemotron-agentic-v4-knowledge` | `nvidia/Nemotron-SFT-Agentic-v4` (default) | openmdw-1.1 | 1,500 | 1B |
| `nemotron-agentic-v1-tools` | `nvidia/Nemotron-Agentic-v1` | cc-by-4.0 | 3,000 | 1.4B |
| `nemotron-agentic-v1-interactive` | `nvidia/Nemotron-Agentic-v1` | cc-by-4.0 | 2,000 | 110M |
| `nemotron-agentic-v2-tools` | `nvidia/Nemotron-SFT-Agentic-v2` | cc-by-4.0 | 3,000 | 3.7B |
| `nemotron-agentic-v2-interactive` | `nvidia/Nemotron-SFT-Agentic-v2` | cc-by-4.0 | 2,000 | 1.5B |
| `nemotron-agentic-v2-search` | `nvidia/Nemotron-SFT-Agentic-v2` | cc-by-4.0 | 1,500 | 150M |
| `nemotron-pt-v1-tool-calling` | `nvidia/Nemotron-Post-Training-Dataset-v1` | cc-by-4.0 | 3,000 | 1.6B |
| `toucan-sft` | `Agent-Ark/Toucan-1.5M` (SFT) | apache-2.0 | 3,000 | 340M |
| `hermes-reasoning-tool-use` | `interstellarninja/hermes_reasoning_tool_use` | apache-2.0 | 5,000 | 100M |
| `toolmind-graphsyn` | `Nanbeige/ToolMind` | apache-2.0 | 5,000 | 600M |
| `hermes-func-calling` | `NousResearch/hermes-function-calling-v1` | apache-2.0 | 3,000 | 7M |
| `hermes-func-calling-single` | `NousResearch/hermes-function-calling-v1` | apache-2.0 | 3,000 | 5M |
| `hermes-glaive-5k` | `NousResearch/hermes-function-calling-v1` | apache-2.0 | 3,000 | 5M |
| `glaive-fc-v2` | `glaiveai/glaive-function-calling-v2` | apache-2.0 | 8,000 | 70M |
| `toolace` | `Team-ACE/ToolACE` | apache-2.0 | 11,000 | 9M |
| `xlam-60k` | `Salesforce/xlam-function-calling-60k` | cc-by-4.0 | 8,000 | 25M |
| `apigen-mt-5k` | `Salesforce/APIGen-MT-5k` | cc-by-nc-4.0 | 5,000 | 35M |
| `dolci-tool-use` | `allenai/Dolci-Instruct-SFT-Tool-Use` | odc-by | 4,000 | 650M |
| `dolci-tool-use-sa` | `allenai/Dolci-Instruct-SFT-Tool-Use-SA` | cc-by-sa-4.0 | 2,000 | 100M |
| `agent-flan-toolbench` | `internlm/Agent-FLAN` | apache-2.0 | 5,000 | 30M |
| `orca-agentinstruct-tool-use` | `microsoft/orca-agentinstruct-1M-v1` | cdla-permissive-2.0 | 5,000 | 40M |

#### recovery_clarification

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `when2call-sft` | `nvidia/When2Call` | cc-by-4.0 | 15,000 | 7M |

#### environment_trajectories

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `agentinstruct-os` | `zai-org/AgentInstruct` | none on the card | 300 | 1M |
| `agentinstruct-db` | `zai-org/AgentInstruct` | none on the card | 600 | 1M |
| `agentinstruct-alfworld` | `zai-org/AgentInstruct` | none on the card | 400 | 1M |
| `agentinstruct-webshop` | `zai-org/AgentInstruct` | none on the card | 400 | 1M |
| `agentinstruct-kg` | `zai-org/AgentInstruct` | none on the card | 400 | 2M |
| `agentinstruct-mind2web` | `zai-org/AgentInstruct` | none on the card | 200 | 1M |
| `agenttraj-l` | `AgentGym/AgentTraj-L` | none on the card | 20,000 | 30M |
| `agentbank-alfworld` | `Solaris99/AgentBank` (alfworld) | apache-2.0 | 4,000 | 3M |
| `agentbank-webshop` | `Solaris99/AgentBank` (webshop) | apache-2.0 | 6,000 | 6M |
| `agentbank-webarena` | `Solaris99/AgentBank` (webarena) | apache-2.0 | 700 | 2M |
| `agentbank-mind2web` | `Solaris99/AgentBank` (mind2web) | apache-2.0 | 5,000 | 3M |
| `agentbank-intercode-bash` | `Solaris99/AgentBank` (intercode_bash) | apache-2.0 | 500 | 0.2M |
| `agentbank-intercode-sql` | `Solaris99/AgentBank` (intercode_sql) | apache-2.0 | 3,000 | 2M |
| `agentbank-apps` | `Solaris99/AgentBank` (apps) | apache-2.0 | 5,000 | 3M |
| `agentbank-hotpotqa` | `Solaris99/AgentBank` (hotpotqa) | apache-2.0 | 5,000 | 2M |
| `agent-flan-agentinstruct` | `internlm/Agent-FLAN` | apache-2.0 | 5,000 | 2M |
| `orca-agentinstruct-webagent` | `microsoft/orca-agentinstruct-1M-v1` | cdla-permissive-2.0 | 20,000 | 40M |

#### chat

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `tulu3-sft-mixture` | `allenai/tulu-3-sft-mixture` | odc-by (mixture; some subsets are non-commercial) | 8,000 | 700M |
| `openhermes-2.5` | `HuggingFaceTB/OpenHermes-2.5-H4` | none on the card (mixture of GPT-4 generated sets) | 8,000 | 450M |
| `ultrachat-200k` | `HuggingFaceH4/ultrachat_200k` | mit | 8,000 | 350M |
| `wildchat-4.8m` | `allenai/WildChat-4.8M` | odc-by | 8,000 | 2.5B |
| `magpie-llama31-pro-1m` | `Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1` | llama3.1 | 8,000 | 1B |
| `magpie-qwen25-pro-1m` | `Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1` | none on the card (Qwen2.5-72B-Instruct outputs) | 8,000 | 1B |
| `slimorca-dedup` | `Open-Orca/SlimOrca-Dedup` | mit | 8,000 | 250M |
| `capybara` | `LDJnr/Capybara` | apache-2.0 | 8,000 | 20M |
| `no-robots` | `HuggingFaceH4/no_robots` | cc-by-nc-4.0 | 8,000 | 4M |
| `dolly-15k` | `databricks/databricks-dolly-15k` | cc-by-sa-3.0 | 8,000 | 4M |
| `openorca-gpt4` | `Open-Orca/OpenOrca` | mit | 8,000 | 450M |
| `orca-agentinstruct-chat` | `microsoft/orca-agentinstruct-1M-v1` | cdla-permissive-2.0 | 8,000 | 500M |
| `dolci-instruct-sft` | `allenai/Dolci-Instruct-SFT` | odc-by (mixture) | 8,000 | 1.8B |
| `smol-smoltalk` | `HuggingFaceTB/smol-smoltalk` | apache-2.0 | 8,000 | 450M |
| `chat-v3-instruction-following` | `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` (default) | odc-by / cc-by-4.0 / apache-2.0 (per card) | 8,000 | 800M |

#### synthetic_reasoning

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `openthoughts3` | `open-thoughts/OpenThoughts3-1.2M` | apache-2.0 | 3,000 | 15B |
| `miromind-m1-sft` | `miromind-ai/MiroMind-M1-SFT-719K` | apache-2.0 | 3,000 | 3B |
| `bespoke-stratos-17k` | `bespokelabs/Bespoke-Stratos-17k` | apache-2.0 | 3,000 | 250M |
| `numinamath-cot` | `AI-MO/NuminaMath-CoT` | apache-2.0 | 3,000 | 450M |
| `am-thinking-other` | `a-m-team/AM-Thinking-v1-Distilled` | research use only (custom terms on the card) | 2,000 | 3.5B |
| `am-thinking-multiturn` | `a-m-team/AM-Thinking-v1-Distilled` | research use only (custom terms on the card) | 2,000 | 900M |

#### typed_exact_instructions

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `am-thinking-if` | `a-m-team/AM-Thinking-v1-Distilled` | research use only (custom terms on the card) | 2,000 | 280M |
| `s-typed-xlam` | `Salesforce/xlam-function-calling-60k` | cc-by-4.0 | 2,000 | - |
| `s-typed-toolace` | `Team-ACE/ToolACE` | apache-2.0 | 2,000 | - |
| `s-typed-hermes` | `NousResearch/hermes-function-calling-v1` | apache-2.0 | 2,000 | - |
| `s-typed-agentic-v4` | `nvidia/Nemotron-SFT-Agentic-v4` (default) | openmdw-1.1 | 1,500 | - |

#### verified_successful_multiturn

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `s-verified-coderforge` | `togethercomputer/CoderForge-Preview` (trajectories) | per row (Apache-2.0 in the sampled rows); none on the card | 500 | - |
| `s-verified-rebench` | `nebius/SWE-rebench-openhands-trajectories` | cc-by-4.0 | 1,000 | - |
| `s-verified-swe-smith` | `SWE-bench/SWE-smith-trajectories` (tool) | mit | 1,000 | - |
| `s-verified-orchard` | `microsoft/Orchard` (swe) | mit | 700 | - |

### Sources that need the maintainer, and ones with withheld prompts

`configs/data/sources.agentic-gated.json` holds six chat sources kept out of the main
manifest because the first run cannot read them. They are fully pinned and recorded, so
once the blocker is cleared they are added to a preparation run unchanged.

| Source id | Dataset | Blocker |
| --- | --- | --- |
| `lmsys-chat-1m` | `lmsys/lmsys-chat-1m` | Gated: the token has not accepted the LMSYS-Chat-1M agreement (HTTP 403). Accept it on the dataset page. |
| `infinity-instruct-7m`, `infinity-instruct-gen` | `BAAI/Infinity-Instruct` (configurations `7M` and `Gen`) | Gated: accept the terms on the dataset page (CC-BY-SA-4.0 once accepted). |
| `nemotron-pt-v2-chat` | `nvidia/Nemotron-Post-Training-Dataset-v2` (split `chat`) | Gated: accept the terms on the dataset page (CC-BY-4.0 once accepted). |
| `chat-v3-chat` | `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` (`data/chat.jsonl`, 17 GB) | The publisher withholds the first user turn of rows seeded from LMSYS-Chat-1M or WildChat-1M. `configs/cloud/prepare-chat-gated.json` restores them with `scripts/restore_chat_v3_prompts.py`, which matches every user turn of LMSYS-Chat-1M and WildChat-1M (the publisher's script matches only first turns and aborts on any miss; 5,110 of 42,835 LMSYS prompts were not first turns or are absent) and drops the rows it cannot restore, then prepares the result with the other gated chat sets. |
| `nemotron-pt-v1-chat` | `nvidia/Nemotron-Post-Training-Dataset-v1` (chat) | Every first user turn is empty (checked on the first file: 93,328 of 93,328 rows); the card says to restore them from LMSYS-Chat-1M and supplies no matching key. |

Also behind terms the token has not accepted, and not used: `lmsys/chatbot_arena_conversations`,
`cx-cmu/agent_trajectories` (content unknown) and `nvidia/Nemotron-Pretraining-SFT-v1`
(manual approval, custom licence). `Salesforce/xlam-function-calling-60k` is auto-gated and
the token already has access. `miromind-ai/MiroVerse-v0.1` is also accessible; it is
non-commercial and was not needed.

### Considered and not added

- **Mind2Web and WebLINX raw releases**: page-element tasks, not conversations, and
  non-commercial; the AgentInstruct and AgentBank conversions of Mind2Web are in.
- **ToolBench / ToolLLaMA original**: one 2 GB JSON array that cannot be streamed, and the
  mirrors found are either queries only or column-oriented. Agent-FLAN carries the
  ToolBench conversations.
- **OpenHermes-2.5 original**: a 1.9 GB JSON array; the Hugging Face parquet copy
  (`HuggingFaceTB/OpenHermes-2.5-H4`) is used instead.
- **Toucan, Kimi-K2, Qwen3 and OSS configurations**: each uses its generator's own call
  syntax (Python-repr strings, a `function` role) rather than the standard one; only the
  clean SFT configuration is in.
- **Nemotron terminal `dataset_adapters` files** (code, math, swe): each file is one row
  group of 2 to 8 GB uncompressed. Listed Hub Parquet files are now read by range requests
  in batches of about 64 MiB, so they are in as `nemotron-terminal-adapters-*` (226,313 traces).
- **Orchard `gui`**, AndroidControl and other screenshot sets: need images, not text.
- **CC-Bench** (370 rows, Claude Code format), **Llama-Nemotron SFT chat** (input/output
  shape) and **OpenThoughts-Agent** (a subset of AgentTrove): little or nothing new.
- **Nemotron-Cascade-SFT-SWE and Nemotron-SFT-SWE-v2 agentless splits**: single-turn
  patches, better placed in `synthetic_code_dialogue`; left for that stream.

### Preparing the sources

Preparation is single-threaded per process. It reads about 300 to 500 KB of source text per
CPU second on chat, documents and long SWE trajectories, and 200 KB on tool-use rows and
code (measured numbers are under "Preparing at full volume"), so a useful amount needs many
processes. `configs/cloud/prepare-agentic-mix.json` (one 16-CPU, 32 GiB sandbox, no GPU,
nothing launches by itself) is the pilot: it splits `sources.agentic.json` into
`AGENTIC_SHARDS` (8) pieces with `scripts/shard_sources.py`, which assigns the costliest
sources first to the lightest shard and scales each source's row cap by `AGENTIC_ROW_SCALE`,
and runs one `megatoken prepare --reject-invalid` per shard into `out/prepared/agentic-<i>`.
Those shards are separate roots, each deduplicated on its own. For every row a source offers,
use the full-volume build below, which also deduplicates across the shards.

Parser review decides whether a trajectory is usable, so it was checked end to end on real
rows (a 15-source run of 20 to 40 rows each). Every agent trajectory was first held out in
the review queue, for two reasons that are now fixed:

- A row-level `language` column (Open-SWE-Traces has `language: python`) was handed to the
  parser for every message, so system prompts, questions and answers were read as Python and
  failed with `low_syntax_coverage`. In conversation profiles only a message's own hints and
  the source options now set the language, file name or MIME type.
- A terminal action written as `<think>...</think>` followed by a JSON call was parsed as one
  JSON document. Only a payload that starts with `{` is treated as JSON now, and the terminal
  adapter reads the call after the thinking block.

Even then a long trajectory has hundreds of chunks and one prose-and-code chunk that the
grammar reads badly would hold out the whole episode. `configs/data/parser-review.agentic.json`
is the default review policy without `low_syntax_coverage`; the cloud job passes it with
`--parser-review-config`. Every other fault (parser crashes, exhausted budgets, ambiguous
chat framing) is still held out, and the chunk keeps its diagnostic on the record. The
default policy for other manifests is unchanged. The language fix also applies to the
pilot's `open-swe-v12` entry in `sources.open.json`, which was hit by it.

With both fixes and the agentic review policy, a real preparation run over 8 of these
sources (Open-SWE-Traces, Terminal-Corpus easy, Orchard, Nebius SWE-agent, CodeActInstruct,
ToolACE, Toucan, Tulu 3; 3 to 30 rows each) accepted 62 episodes with about 233,000 tokens of
executed transcript and 191,000 tokens in recorded passing runs. It held out one terminal row
(`depth_budget_exhausted`, a genuine parser limit) and rejected 3 rows (one with too few
scored positions, two ToolACE calls whose arguments are not Python literals).

The manifest's caps are a first real run: about 364,000 rows read, about 1.5B tokens of
prepared text (estimated at 4 bytes per token), roughly 8 hours of wall clock on 8
processes. Caps count rows read before filters, so the filtered SWE sources carry large
caps for few kept rows. To prepare more, set `AGENTIC_ROW_SCALE` (it multiplies every
cap and keeps the leading rows identical, so a run at 0.25 is a quarter-length smoke test
and a run at 8 reads eight times as many rows from the same pinned revisions). The sandbox's
24-hour lifetime and ~0.5 MB/s uplink mean prepared roots should be collected from the
sandbox, not rebuilt locally.

Rows the adapters reject are counted and skipped (`--reject-invalid`). Expect rejects
for: a missing or withheld first prompt, a tool call whose result is missing or reused,
tool schemas that are not valid JSON Schema, and traces too short to score (fewer than
16 scored positions for agent streams, 4 for the rest). About one row in twelve of
CoderForge and APIGen-MT is rejected by the segmenter ("Parser diagnostic is outside the
original source"); it is raised by the chunking step after the adapter has accepted the
row, and has not been investigated.

The earlier pilot entries for `open-swe-v12`, `agent-trove` and `terminal-corpus` that were
in `sources.open.json` are removed: they lost most rows because they set neither
`allow_pending_final_action` (traces that end on a call with no result) nor positional call
IDs, and the coding manifest reads the same data completely.

### What was checked

Each source was streamed from its pinned revision through its own adapter on its first
12 rows (more for filtered sources; Nebius SWE-agent: 17 of 150 pass its filter, SWE-Gym:
11 of 149 pass). All sources produced episodes for every row that passed its filter,
with these exceptions: CoderForge 11 of 12 and APIGen-MT 11 of 12 (the segmenter message
above), SWE-Gym 1 of 12 passing rows rejected (a reused tool call ID in the source), and
the withheld-prompt sources moved to the gated manifest. Open-SWE-Traces produced 8
passing and 4 failed trajectories out of 12; the failed ones keep their failed outcome and
carry no training mask until a recovery mask is given. Offline, `tests/test_agentic_adapters.py`
checks every dialect on tiny fixtures, and `tests/test_agentic_sources.py` builds a fixture
row in each source's shape, runs the adapter named in the manifest, and checks the stream,
masks, result provenance and outcome label.

### Still missing for these streams

- **recovery_clarification** and **environment_trajectories** were thin here (7M and 0.1B
  tokens); the later-stage round adds sources to both (see "Later-stage streams").
- **Executed tool results from real APIs.** Most tool-use results are simulated; only
  Toucan's MCP calls were executed.
- **Failure and recovery in SWE traces.** Only Open-SWE-Traces records failed runs;
  everywhere else the failed attempts were already filtered out by the publisher.

## Freeze ancestry, deduplicate, then split

Preparation spools normalized episodes and source cursors to SQLite. It performs
exact role-aware transcript deduplication and bounded near deduplication using
word shingles, deterministic bottom-k sketches and indexed anchors. Union-find
joins explicit family/ancestor/repository/task relationships and duplicate
clusters. The component graph is completely resolved before any output shard is
released or any descendant's split is inherited.

Near matching is an approximate candidate search, not an exhaustive semantic
proof. `DedupConfig` makes thresholds, minimum text length, sketch/anchor sizes and
candidate limits explicit. A sketch value that more than `max_candidates` earlier
documents share (a licence header, a system prompt) cannot tell near duplicates
apart, so it is skipped and the next-smallest value takes its place; no document is
dropped for it. At large scale, validate the chosen recall
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

## Agentic coding sources

`configs/data/sources.agentic-coding.json` holds every public agentic coding set found so far
that the readers can convert: 354 sources over 301 Hub repositories, each read in
full (no row cap). A set counts when a coding agent (or a developer working with one) acted in a
real or sandboxed environment and the tool output came back from that environment: bug-fix
rollouts in the OpenHands, SWE-agent, mini-SWE-agent, Claude Code, Codex, OpenCode and pi
harnesses, terminal-agent traces, data-science notebook agents, and raw session logs. Nothing
here was generated for this project. Every licence is accepted and recorded exactly as the
publisher states it on the dataset card; 206 sources state none, and each is approved with
`reviewed_by: Gabriel Pickard`, `reviewed_at: 2026-10-22` like the rest.

The sources moved here from `sources.agentic.json` and `sources.agentic-more.json` (30 and 18
entries), so no source is prepared by two manifests. Those two manifests keep their smoke-run row
caps; this one has none.

### Volume

Rows are what the Hub reports for the configuration and split each source reads, before the
source's own row filter; for the raw session logs a row is one session file. Text is the stored
size of the files read times the usual expansion (3 for Parquet, 4 for compressed JSON, 1 for
plain JSON), so it is an upper estimate. "Before" is the cap each source had in the older
manifests (rows read, and text at the manifest's per-stream average row size); "new" means the
source did not exist there.

| | Sources | Rows | Stored on the Hub (GB) | Text (GB, estimated) |
| --- | ---: | ---: | ---: | ---: |
| Before (coding sources in the older manifests, with their caps) | 46 | 84,155 | n/a | 8.9 |
| After (this manifest, all rows) | 354 | 25,972,598 | 767 | 1,654 |

By stream: 304 `executed_trajectories`, 36 `verified_trajectories`, 8 `mixed_quality_episodes`, 4 `verified_successful_multiturn`, 1 `environment_trajectories`, 1 `synthetic_code_dialogue`. Of the 354:
about 142 are terminal-agent trace sets (AgentTrove and the many DCAgent sets, Nemotron
Terminal, TerminalTraj, LiteCoder), 45 are raw session logs of Claude Code, Codex, pi
and OpenCode (SWE-chat, which holds 16,010 real developer sessions, is read as four sources, one per
harness), and the rest are software-engineering rollouts. 117 of them are DCAgent trace sets.
Many sets hold failed runs as well as passing ones. The failures are kept as readable context
with no training target (zero scored positions, `min_scored_positions: 0`) where the publisher
labels outcomes per row, and a passing-only selection goes to `verified_trajectories` through a
row filter on the recorded verdict, so the same files are listed twice (an `-unresolved` or
`-fail` source beside the verified one); the second pass over a file costs only reading.

### Outcome labels

| Stream | What its sources record | Count |
| --- | --- | ---: |
| `verified_trajectories` | the publisher's own passing tests (a row filter keeps only passing rows) | 36 |
| `executed_trajectories` | real tool output, outcome unknown, curated, or a mix; runs with a recorded failure are kept unmasked | 304 |
| `mixed_quality_episodes` | executed but unvetted (execution-free SWE-ZERO rollouts, CodeScout localisation) | 8 |
| `verified_successful_multiturn` | a second reading of four verified sets, for the supervision stream that wants successful multi-turn runs | 4 |
| others | AgentBank environment trajectories, one code-dialogue set | 2 |

A curated release (no per-row verdict, `assume_outcome: success`) is never marked as verified.
The four `s-verified-*` sources re-read the data of a verified source in another stream; the
deduplication step compares transcripts, not streams, so when both are prepared together the
second copy is counted as a duplicate of the first.

### Readers added

The adapters that were missing for these layouts are in `data/adapters.py` and
`data/sessions.py`: Claude Code, Codex, pi and OpenCode session logs (one file is one row,
read with the `whole_files` option), Responses-API item lists, step records, normalised event
lists, Harbor rollouts with tagged observations, calls held in a `parts` list, code-action
traces with `<code>` or fenced blocks, and several text conventions (`<function=...>` replies
with one or several calls). Tests are in `tests/test_agentic_coding_rows.py`. Linking changes:
a call ID reused after its result is a new call; one result may close a whole batch of calls
(a terminal screen, or one reply for parallel calls); a result naming no known call answers the
only open one; a trajectory, chat or tool-use dialogue may end on a call; a blank follow-up user
turn is skipped, while a blank task still rejects the row. A row filter `nonempty` drops rows
whose conversation is empty. Source reading changed in two ways: a listed Hub file is read by
the builder of its own format (so a dataset card that declares other columns cannot break the
cast), and `explode` turns a row that holds several runs into one row per run.

### Left out, on purpose

- **Evaluation-set tasks.** Terminal-Bench runs, the SWE-bench Verified trajectory sets
  (CoderForge's 32B evaluation run, the two tarsur385 sets), SWE-bench Multilingual and
  SWE-Bench Pro runs (JetBrains SWE-TaskFlow, Intelligent-Internet), the JetBrains
  test-minus-verified set, HumanEval and MBPP rows of AgentBank, and the DABench set. Training on
  their solutions would contaminate those evaluations.
- **Programs, not agents.** `arpandeepk/swe-cleanroom-trajs`, `swe-zero-grounded-full` and
  `swe-zero-free-v2` were written by a program that fills templates (the observations are real, the
  reasoning is not).
- **Copies.** `ricdomolm/nanoswe-trajs-v0` holds the very files of `nanoswe/nanoswe-trajs-260812`;
  a re-export of SWE-chat (`neulab/adp-v2` swe-chat); one claude-code-glm53 set listed twice;
  DCAgent re-uploads and a mix of other DCAgent sets; sources the later-stage manifest already reads
  for its own stream (the OpenCode question-tool splits, Scale-SWE-Distilled, SWE-Lego real data,
  Code-Feedback), which stay there.
- **Not coding.** UltraData's General, Search and Tool-Use agents, Toolathlon, and the zake7749 and
  nex tool-calling sets belong to the chat and tool-use manifests.

### Needs a click on the Hub

Nothing is left to click. `GAIR/OpenSWE` (about 13,000 trajectories in one 4.9 GB file) was
accepted on 2026-10-23 and is `openswe-trajectories` in the main manifest; its first 15 rows
prepared with none rejected. Also gated but accepted: `GAIR/daVinci-Dev` is a gated set whose terms the project account has accepted.

### Layouts still without a reader

Multi-SWE-bench and Moatless zip archives,
CooperBench's OpenHands event lists, EnvBench LangChain logs, openhands-feedback event lists,
sailplane `.traj` files, AlexCuadron's O1 runs, the Nemotron-Cascade agentless split, 75 million
relational rows of `risenlab/agentlogs`, token-id layouts (`open-athena` `*-opencode-traces`),
about 2 percent of AgentTrove replies that call a `bash_command` tool, 2.5 percent of
Nemotron-SFT-SWE-v3 rows in a mixed harness, Cursor raw sessions and a trace index
(`melissapan`). None of them can be read today.

### Real-row check

Every source was read through the real reader (Hub streaming, its pinned revision and file
list) for its first 40 rows (150 to 400 for sources with a row filter, so that some pass it), and
those rows were prepared with the real codec and the agentic review policy. The 117 DCAgent sets
were instead sampled at spread positions of each file (12 to 42 rows), and a handful of them were
also read through the real reader. In all, 20,374 rows from all 354 sources: 13,593 became
episodes, 5,923 were dropped by the source's own row filter, 431 were duplicates or held for
parser review, and 427 were rejected, which is 3.0 percent of the rows that passed the filter.
Per-source results are in the last column of the table below ("ok" = became an episode;
"rejected" = the adapter or the quality gate refused the row; "filtered" = the source's own
filter dropped it). 277 sources lose under 5 percent; the other 77 are mostly raw logs of
people's sessions and chat-only dialogues (no tool call, too few scored turns, or longer than the
262,144-token episode limit, which are skipped and counted, not errors), the OpenCode splits
(plain answers without a tool call), and DCAgent sets whose trials stop before running a command.
Several filtered sources showed no passing row in their first 150 rows (their files are ordered by
outcome); their filters were checked on spread samples instead.

### Per-source table

| Source id | Dataset (config) | Stream | Licence on the card | Rows | Text (GB) | Before (rows cap; GB) | Real-sample result |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `aaaaliou-pi-mono` | `aaaaliou/pi-mono` | executed_trajectories | other | 145* | 0.1 | new | 36 ok, 4 rejected |
| `termgrade-trajectories-fail` | `ai-and/termgrade-trajectories` (trajectories_messages) | executed_trajectories | apache-2.0 | 36.1k | 2.1 | new | 13 ok, 1 rejected, 26 filtered |
| `termgrade-trajectories-pass` | `ai-and/termgrade-trajectories` (trajectories_messages) | verified_trajectories | apache-2.0 | 36.1k | 2.1 | new | 19 ok, 1 rejected, 20 filtered |
| `akseljoonas-codeagent-traces` | `akseljoonas/codeagent-traces` | executed_trajectories | not stated on the dataset card | 7.0k | 0.1 | new | 40 ok |
| `glm-5-2-pi-agent` | `AletheiaResearch/GLM-5.2-Agent` | executed_trajectories | not stated on the dataset card | 319* | 0.1 | new | 38 ok, 2 rejected |
| `codex-gpt-5-3-spark` | `AletheiaResearch/GPT-5.3-Codex-Spark-Codex` | executed_trajectories | not stated on the dataset card | 200* | 0.1 | new | 38 ok, 2 rejected |
| `codex-gpt-5-5` | `AletheiaResearch/GPT-5.5-Codex` | executed_trajectories | not stated on the dataset card | 317* | 0.7 | new | 38 ok, 2 rejected |
| `multi-swe-smith-rust-glm46` | `AlienKevin/Multi-SWE-smith-Rust-GLM-4.6-trajectories` (default) | verified_trajectories | not stated on the dataset card | 142 | 0.0 | new | 142 ok |
| `swe-smith-rs-gemini3flash` | `AlienKevin/SWE-smith-rs-gemini-3-flash-trajectories` | verified_trajectories | mit | 1.4k | 0.1 | new | 119 ok, 31 filtered |
| `swe-smith-rs-glm46` | `AlienKevin/SWE-smith-rs-glm-4.6-trajectories` | executed_trajectories | not stated on the dataset card | 369 | 0.0 | new | 40 ok |
| `swe-smith-rs-gpt5mini` | `AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories` | verified_trajectories | mit | 4.0k | 0.2 | new | 127 ok, 23 filtered |
| `swe-smith-rs-minimax-m25` | `AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories` | verified_trajectories | mit | 5.3k | 0.2 | new | 0 ok, 150 filtered |
| `swe-zero-12m-trajectories` | `AlienKevin/SWE-ZERO-12M-trajectories` (default) | executed_trajectories | apache-2.0 | 12.29M | 107.9 | new | 0 ok, 400 filtered |
| `swe-zero-1k-trajectories` | `AlienKevin/SWE-ZERO-1k-trajectories` (default) | executed_trajectories | apache-2.0 | 1.0k | 0.0 | new | 29 ok, 121 filtered |
| `swe-zero-1k-32k-trajectories` | `AlienKevin/SWE-ZERO-1k-trajectories-32k` (default) | executed_trajectories | apache-2.0 | 1.0k | 0.0 | new | 79 ok, 71 filtered |
| `claude-fable-5-cc-alinciocan` | `AlinCiocan/fable-5-claude-code-traces` | executed_trajectories | cc-by-4.0 | 18* | 0.1 | new | 17 ok, 1 rejected |
| `sera-4.5a-django-t1` | `allenai/Sera-4.5A-Django-T1` | executed_trajectories | odc-by | 17.1k | 7.5 | new | 40 ok |
| `sera-4.5a-django-t2` | `allenai/Sera-4.5A-Django-T2` | executed_trajectories | odc-by | 21.6k | 5.4 | new | 39 ok, 1 held/duplicate |
| `sera-4.5a-full-t1` | `allenai/Sera-4.5A-Full-T1` | executed_trajectories | odc-by | 16.6k | 23.2 | new | 40 ok |
| `sera-4.5a-full-t2` | `allenai/Sera-4.5A-Full-T2` | executed_trajectories | odc-by | 21.3k | 16.6 | new | 40 ok |
| `sera-4.5a-lite-t1` | `allenai/Sera-4.5A-Lite-T1` | executed_trajectories | odc-by | 16.6k | 11.8 | new | 40 ok |
| `sera-4.5a-lite-t2` | `allenai/Sera-4.5A-Lite-T2` | executed_trajectories | odc-by | 21.8k | 9.0 | new | 39 ok, 1 held/duplicate |
| `sera-4.5a-sphinx-t1` | `allenai/Sera-4.5A-Sphinx-T1` | executed_trajectories | odc-by | 17.7k | 7.4 | new | 40 ok |
| `sera-4.5a-sphinx-t2` | `allenai/Sera-4.5A-Sphinx-T2` | executed_trajectories | odc-by | 12.7k | 3.0 | new | 40 ok |
| `sera-4.5a-sympy-t1` | `allenai/Sera-4.5A-Sympy-T1` | executed_trajectories | odc-by | 20.0k | 7.3 | new | 40 ok |
| `sera-4.5a-sympy-t2` | `allenai/Sera-4.5A-Sympy-T2` | executed_trajectories | odc-by | 25.4k | 5.2 | new | 40 ok |
| `sera-4.6-lite-t1` | `allenai/Sera-4.6-Lite-T1` | executed_trajectories | odc-by | 23.4k | 8.4 | new | 39 ok, 1 held/duplicate |
| `sera-lite-t2-glm46` | `allenai/Sera-4.6-Lite-T2` | executed_trajectories | odc-by | 21.3k | 9.8 | 1.5k; 0.18 | 39 ok, 1 rejected |
| `tmax-sft-qwen36-success` | `allenai/tmax-sft` (skill_tax_20260505_2.2k_combined_balanced_thinking_only_success) | executed_trajectories | odc-by | 5.8k | 0.1 | 3.0k; 0.37 | 40 ok |
| `tmax-sft-big-ntc-adapters` | `allenai/tmax-sft-big` | executed_trajectories | odc-by | 327.3k | 4.7 | 3.0k; 0.37 | 149 ok, 1 held/duplicate |
| `claude-fable-5-cc-armand0e` | `armand0e/claude-fable-5-claude-code` | executed_trajectories | mit | 55* | 0.1 | new | 36 ok, 2 rejected, 2 held/duplicate |
| `kimi-k26-codex-agent` | `armand0e/kimi-k2.6-agent` | executed_trajectories | not stated on the dataset card | 14* | 0.0 | new | 12 ok, 2 rejected |
| `kimi-k26-cc-traces` | `armand0e/kimi-k2.6-claude-code-traces` | executed_trajectories | not stated on the dataset card | 36* | 0.0 | new | 34 ok, 2 rejected |
| `minimax-m27-pi-agent` | `armand0e/minimax-m2.7-agent` | executed_trajectories | not stated on the dataset card | 21* | 0.0 | new | 19 ok, 2 rejected |
| `minimax-m3-cc-traces` | `armand0e/minimax-m3-claude-code-traces` | executed_trajectories | not stated on the dataset card | 31* | 0.0 | new | 30 ok, 1 rejected |
| `qwen37-max-pi-traces` | `armand0e/qwen3.7-max-pi-traces` | executed_trajectories | not stated on the dataset card | 47* | 0.0 | new | 39 ok, 1 rejected |
| `denovoswe-filtered` | `AweAI-Team/DeNovoSWE-Trajectory-Filtered` | verified_trajectories | cc-by-4.0 | 11.7k | 4.6 | 4.5k; 0.55 | 67 ok, 83 filtered |
| `denovoswe-trajectory-raw` | `AweAI-Team/DeNovoSWE-Trajectory-Raw` | executed_trajectories | cc-by-4.0 | 34.0k | 24.7 | new | 40 ok |
| `scale-swe-v4pro-41k` | `AweAI-Team/Scale-SWE-Distilled-DeepSeek-v4-Pro-High-41k` | verified_trajectories | cc-by-4.0 | 41.1k | 6.4 | 2.0k; 0.25 | 150 ok |
| `pi-mono-sessions-badlogic` | `badlogicgames/pi-mono` | executed_trajectories | other | 626* | 0.2 | new | 33 ok, 6 rejected, 1 held/duplicate |
| `codex-assistant-rollouts-basedlsg` | `basedlsg/codex-assistant-rollouts` | executed_trajectories | not stated on the dataset card | 27* | 0.8 | new | 14 ok, 13 rejected |
| `codex-traces-becks723` | `Becks723/codex-traces` | executed_trajectories | not stated on the dataset card | 131* | 0.2 | new | 31 ok, 9 rejected |
| `kaggle-notebooks-intent` | `bigcomputer/kaggle-notebooks-intent-conversations` | executed_trajectories | not stated on the dataset card | 503.4k | 81.7 | new | 38 ok, 2 held/duplicate |
| `ptbr-text2sql-agent` | `Boakpe/pt-br-agentic-text-to-sql-distilled-trajectories` | executed_trajectories | apache-2.0 | 7.4k | 0.3 | new | 40 ok |
| `championswimmer-pi-coding-sessions` | `championswimmer/pi-coding-sessions` | executed_trajectories | mit | 27* | 0.0 | new | 26 ok, 1 rejected |
| `mimo-cc-traces-1k` | `choucsan/mimo-claude-code-traces-1k` | executed_trajectories | mit | 1.0k* | 0.0 | new | 39 ok, 1 rejected |
| `cooperbench-qwen9b-coop-cc` | `CooperBench/qwen9b-coop-claude-code` | executed_trajectories | mit | 736* | 0.4 | new | 40 ok |
| `cooperbench-qwen9b-solo-cc` | `CooperBench/qwen9b-solo-claude-code` | executed_trajectories | mit | 368* | 0.2 | new | 40 ok |
| `wisp-cc-sessions` | `crispwisp/wisp-claude-code-sessions` | executed_trajectories | mit | 61* | 0.0 | new | 40 ok |
| `dacorvo-hf-hub-session-opencode` | `dacorvo/hf-hub-session-opencode-traces` | executed_trajectories | apache-2.0 | 196* | 0.0 | new | 39 ok, 1 rejected |
| `dacorvo-hf-hub-session-pi` | `dacorvo/hf-hub-session-pi-traces` | executed_trajectories | apache-2.0 | 210* | 0.0 | new | 32 ok, 8 rejected |
| `opencode-transformers-sessions` | `dacorvo/transformers-coding-session-opencode-traces` | executed_trajectories | apache-2.0 | 209* | 0.0 | new | 36 ok, 4 rejected |
| `dacorvo-transformers-session-pi` | `dacorvo/transformers-coding-session-pi-traces` | executed_trajectories | apache-2.0 | 177* | 0.0 | new | 27 ok, 13 rejected |
| `dcagent-all-puzzles-5k-sandboxes-10k` | `DCAgent/All_Puzzles_5k-sandboxes_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.5k | 0.1 | new | 147 ok, 3 rejected |
| `dcagent-bash-textbook-tasks` | `DCAgent/bash_textbook_tasks_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.4k | 1.2 | new | 27 ok, 15 filtered |
| `dcagent-t2-dcagent-code-contests-sandboxes-traces-terminus-2` | `DCAgent/code-contests-sandboxes-traces-terminus-2` | executed_trajectories | not stated on the dataset card | 10.0k | 0.2 | new | 12 ok |
| `dcagent-code-contests-sandboxes-with-tests-10k` | `DCAgent/code-contests-sandboxes-with-tests_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.0k | 1.1 | new | 39 ok, 3 filtered |
| `dcagent-codeforces-sandboxes-1-10k` | `DCAgent/codeforces-sandboxes-1_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 6.7k | 0.8 | new | 33 ok, 6 rejected, 3 filtered |
| `dcagent-t2-dcagent-exp-gfi-staqc-random-filtered-10k-glm-4-7-traces-locetash` | `DCAgent/exp-gfi-staqc-random-filtered-10K_glm_4.7_traces_locetash` | executed_trajectories | not stated on the dataset card | 9.8k | 0.2 | new | 12 ok |
| `dcagent-exp-psu-swesmith-31k` | `DCAgent/exp-psu-swesmith-31K_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 31.6k | 2.4 | new | 39 ok, 3 filtered |
| `dcagent-t2-dcagent-exp-swd-swesmith-wo-docker-glm-4-7-traces-locetash` | `DCAgent/exp-swd-swesmith-wo-docker_glm_4.7_traces_locetash` | executed_trajectories | not stated on the dataset card | 10.0k | 0.6 | new | 9 ok, 3 filtered |
| `dcagent-t2-dcagent-exp-syh-r2egym-swesmith-mixed-glm-4-7-traces-locetash` | `DCAgent/exp-syh-r2egym-swesmith-mixed_glm_4.7_traces_locetash` | executed_trajectories | not stated on the dataset card | 8.8k | 0.6 | new | 6 ok, 6 filtered |
| `dcagent-t2-dcagent-exp-uns-r2egym-2-1x-glm-4-7-traces-locetash` | `DCAgent/exp-uns-r2egym-2_1x_glm_4.7_traces_locetash` | executed_trajectories | not stated on the dataset card | 8.4k | 0.7 | new | 3 ok, 9 filtered |
| `dcagent-t2-dcagent-exp-uns-r2egym-33-6x-glm-4-7-traces-locetash` | `DCAgent/exp-uns-r2egym-33_6x_glm_4.7_traces_locetash` | executed_trajectories | not stated on the dataset card | 5.5k | 0.5 | new | 8 ok, 1 rejected, 3 filtered |
| `dcagent-exp-flat25-baseline-10k` | `DCAgent/exp_flat25_baseline_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.5 | new | 34 ok, 3 rejected, 3 filtered |
| `dcagent-exp-flat25-pseudocode-10k` | `DCAgent/exp_flat25_pseudocode_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 1.2 | new | 32 ok, 4 rejected, 6 filtered |
| `dcagent-exp-flat25-speed-bonus-10k` | `DCAgent/exp_flat25_speed_bonus_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 1.4 | new | 36 ok, 2 rejected, 4 filtered |
| `dcagent-exp-flat25-stackoverflow-10k` | `DCAgent/exp_flat25_stackoverflow_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 1.4 | new | 32 ok, 7 rejected, 3 filtered |
| `dcagent-exp-rle-2skill-10k` | `DCAgent/exp_rle_2skill_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 1.4 | new | 30 ok, 3 rejected, 9 filtered |
| `dcagent-exp-rle-adversarial-10k` | `DCAgent/exp_rle_adversarial_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.4 | new | 40 ok, 2 rejected |
| `dcagent-exp-rle-curated-10k` | `DCAgent/exp_rle_curated_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.4 | new | 34 ok, 2 rejected, 6 filtered |
| `dcagent-exp-rle-detailed-10k` | `DCAgent/exp_rle_detailed_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.8k | 0.6 | new | 141 ok, 9 rejected |
| `dcagent-exp-rle-error-report-10k` | `DCAgent/exp_rle_error_report_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.4 | new | 36 ok, 6 filtered |
| `dcagent-exp-rle-expert-10k` | `DCAgent/exp_rle_expert_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 0.4 | new | 41 ok, 1 rejected |
| `dcagent-exp-rle-github-issue-10k` | `DCAgent/exp_rle_github_issue_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.4 | new | 35 ok, 1 rejected, 6 filtered |
| `dcagent-exp-rle-heavy-padding-10k` | `DCAgent/exp_rle_heavy_padding_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.5 | new | 31 ok, 2 rejected, 9 filtered |
| `dcagent-exp-rle-minimal-instructions-10k` | `DCAgent/exp_rle_minimal_instructions_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.2 | new | 35 ok, 1 rejected, 6 filtered |
| `dcagent-exp-rle-moderate-padding-10k` | `DCAgent/exp_rle_moderate_padding_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.5 | new | 32 ok, 4 rejected, 6 filtered |
| `dcagent-exp-rle-partial-ambiguity-10k` | `DCAgent/exp_rle_partial_ambiguity_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 1.3 | new | 37 ok, 2 rejected, 3 filtered |
| `dcagent-exp-rle-proportional-10k` | `DCAgent/exp_rle_proportional_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.4 | new | 40 ok, 2 rejected |
| `dcagent-exp-rpt-bigcodebench-10k` | `DCAgent/exp_rpt_bigcodebench_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 6.2k | 0.4 | new | 12 ok |
| `dcagent-exp-rpt-bugsinpy-mf-large-10k` | `DCAgent/exp_rpt_bugsinpy-mf-large_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.2k | 0.5 | new | 38 ok, 4 rejected |
| `dcagent-exp-rpt-codeelo-v2-10k` | `DCAgent/exp_rpt_codeelo-v2_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.4k | 0.5 | new | 31 ok, 2 rejected, 9 filtered |
| `dcagent-exp-rpt-defects4j-v3-10k` | `DCAgent/exp_rpt_defects4j-v3_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 1.6 | new | 36 ok, 6 filtered |
| `dcagent-exp-rpt-e2egit-10k` | `DCAgent/exp_rpt_e2egit_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 0.4 | new | 12 ok |
| `dcagent-exp-rpt-ghactions` | `DCAgent/exp_rpt_ghactions_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 7.0k | 0.8 | new | 17 ok, 1 rejected, 24 filtered |
| `dcagent-t2-dcagent-exp-rpt-issue-10k-glm-4-6-traces-jupiter` | `DCAgent/exp_rpt_issue_10k_glm_4.6_traces_jupiter` | executed_trajectories | not stated on the dataset card | 2.5k | 0.8 | new | 0 ok, 9 filtered, 3 held/duplicate |
| `dcagent-exp-rpt-methods2test` | `DCAgent/exp_rpt_methods2test_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.7k | 0.9 | new | 36 ok, 6 rejected |
| `dcagent-exp-rpt-nemotron-bash-withtests-gpt5mini` | `DCAgent/exp_rpt_nemotron-bash-withtests-gpt5mini_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.2k | 1.4 | new | 12 ok |
| `dcagent-exp-rpt-nemotron-bash-withtests` | `DCAgent/exp_rpt_nemotron-bash-withtests_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.0k | 1.1 | new | 149 ok, 1 rejected |
| `dcagent-exp-rpt-nemotron-csharp-10k` | `DCAgent/exp_rpt_nemotron-csharp_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.3k | 1.6 | new | 9 ok, 3 filtered |
| `dcagent-exp-rpt-nemotron-junit-10k` | `DCAgent/exp_rpt_nemotron-junit_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 0.4 | new | 12 ok |
| `dcagent-exp-rpt-nemotron-ruby-10k` | `DCAgent/exp_rpt_nemotron-ruby_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.1 | new | 12 ok |
| `dcagent-exp-rpt-nemotron-rust-10k` | `DCAgent/exp_rpt_nemotron-rust_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.2 | new | 34 ok, 8 filtered |
| `dcagent-exp-rpt-pymethods2test-v3-10k` | `DCAgent/exp_rpt_pymethods2test-v3_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 7.2k | 0.7 | new | 9 ok, 3 filtered |
| `dcagent-exp-rpt-quixbugs-python-10k` | `DCAgent/exp_rpt_quixbugs-python_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.9 | new | 35 ok, 7 filtered |
| `dcagent-exp-rpt-softwareheritage-10k` | `DCAgent/exp_rpt_softwareheritage_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.1k | 1.5 | new | 9 ok, 3 filtered |
| `dcagent-exp-rpt-stack-bash-withtests-gpt5mini` | `DCAgent/exp_rpt_stack-bash-withtests-gpt5mini_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.7k | 2.0 | new | 9 ok, 3 filtered |
| `dcagent-exp-rpt-stack-dockerfile` | `DCAgent/exp_rpt_stack-dockerfile_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.4k | 0.8 | new | 12 ok |
| `dcagent-exp-rpt-stack-go-v3-test-10k` | `DCAgent/exp_rpt_stack-go-v3-test_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.7k | 2.0 | new | 23 ok, 1 rejected, 16 filtered |
| `dcagent-exp-rpt-stack-php-large-10k` | `DCAgent/exp_rpt_stack-php-large_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 2.0 | new | 12 ok |
| `dcagent-exp-rpt-stack-pytest-gpt5mini` | `DCAgent/exp_rpt_stack-pytest-gpt5mini_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 6.8k | 1.3 | new | 27 ok, 3 rejected, 12 filtered |
| `dcagent-exp-rpt-stack-pytest-synthetic-gpt5nano` | `DCAgent/exp_rpt_stack-pytest-synthetic-gpt5nano_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 1.4 | new | 38 ok, 1 rejected, 3 filtered |
| `dcagent-exp-rpt-stack-pytest-withtests` | `DCAgent/exp_rpt_stack-pytest-withtests_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 3.6 | new | 26 ok, 1 rejected, 15 filtered |
| `dcagent-exp-rpt-stack-selfdoc-gpt5mini` | `DCAgent/exp_rpt_stack-selfdoc-gpt5mini_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 7.4k | 3.5 | new | 19 ok, 2 rejected, 21 filtered |
| `dcagent-exp-rpt-stack-selfdoc-10k` | `DCAgent/exp_rpt_stack-selfdoc_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.8k | 2.6 | new | 21 ok, 3 rejected, 18 filtered |
| `dcagent-exp-rpt-unitsyn-python` | `DCAgent/exp_rpt_unitsyn-python_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 5.8k | 0.6 | new | 12 ok |
| `dcagent-freelancer-projects-sandboxes` | `DCAgent/freelancer-projects-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 0.5 | new | 148 ok, 2 rejected |
| `dcagent-glaive-code-assistant-sandboxes` | `DCAgent/glaive-code-assistant-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.6 | new | 35 ok, 1 rejected, 6 filtered |
| `dcagent-t2-dcagent-glm46-bash-textbook-traces` | `DCAgent/glm46-bash-textbook-traces` | executed_trajectories | not stated on the dataset card | 6.6k | 0.2 | new | 12 ok |
| `dcagent-t2-dcagent-kimi-2-5-exp-gfi-staqc-embedding-mean-filtered-10k-maxeps-32k` | `DCAgent/Kimi-2.5-exp-gfi-staqc-embedding-mean-filtered-10K-maxeps-32k` | executed_trajectories | not stated on the dataset card | 8.4k | 0.6 | new | 9 ok, 3 filtered |
| `dcagent-t2-dcagent-kimi-2-5-exp-gfi-swesmith-random-filtered-10k-maxeps-32k` | `DCAgent/Kimi-2.5-exp-gfi-swesmith-random-filtered-10K-maxeps-32k` | executed_trajectories | not stated on the dataset card | 3.3k | 0.6 | new | 6 ok, 6 filtered |
| `dcagent-kimi25-swe-rebench` | `DCAgent/Kimi-2.5-swe_rebench_patched-maxeps-32k` | executed_trajectories | not stated on the dataset card | 12.4k | 0.7 | new | 44 ok, 1 rejected, 105 filtered |
| `dcagent-t2-dcagent-kimi-2-6-exp-gfi-swesmith-random-filtered-10k-maxeps-32k` | `DCAgent/Kimi-2.6-exp-gfi-swesmith-random-filtered-10K-maxeps-32k` | executed_trajectories | not stated on the dataset card | 2.0k | 0.4 | new | 6 ok, 6 filtered |
| `dcagent-magicoder-evol-instruct-110k-sandboxes-1-10k` | `DCAgent/Magicoder-Evol-Instruct-110K-sandboxes-1_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.5k | 0.3 | new | 27 ok, 15 filtered |
| `dcagent-nemo-prism-math-sandboxes` | `DCAgent/nemo-prism-math-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 8.8k | 0.4 | new | 39 ok, 3 rejected |
| `dcagent-neulab-agenttuning-alfworld-sandboxes` | `DCAgent/neulab-agenttuning-alfworld-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.9 | new | 35 ok, 1 rejected, 6 filtered |
| `dcagent-neulab-agenttuning-db-sandboxes` | `DCAgent/neulab-agenttuning-db-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.5 | new | 12 ok |
| `dcagent-neulab-agenttuning-kg-sandboxes` | `DCAgent/neulab-agenttuning-kg-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.7 | new | 33 ok, 9 filtered |
| `dcagent-neulab-agenttuning-os-sandboxes` | `DCAgent/neulab-agenttuning-os-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.3 | new | 12 ok |
| `dcagent-neulab-agenttuning-webshop-sandboxes` | `DCAgent/neulab-agenttuning-webshop-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.1k | 2.2 | new | 31 ok, 9 filtered |
| `dcagent-neulab-code-feedback-sandboxes` | `DCAgent/neulab-code-feedback-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 0.7 | new | 35 ok, 5 rejected |
| `dcagent-neulab-codeactinstruct-sandboxes` | `DCAgent/neulab-codeactinstruct-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.8 | new | 9 ok, 3 filtered |
| `dcagent-neulab-go-browse-wa-sandboxes` | `DCAgent/neulab-go-browse-wa-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.5 | new | 33 ok, 9 filtered |
| `dcagent-neulab-mind2web-sandboxes` | `DCAgent/neulab-mind2web-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.6 | new | 38 ok, 4 filtered |
| `dcagent-nebius-swe-agent-gpt5nano-terminus2` | `DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes-traces-terminus-2` | mixed_quality_episodes | not stated on the dataset card | 12.0k | 0.3 | new | 40 ok |
| `dcagent-neulab-nebius-swe-agent-trajectories-sandboxes` | `DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 12.4k | 2.3 | new | 35 ok, 7 rejected |
| `dcagent-t2-dcagent-neulab-nnetnav-wa-sandboxes-traces-terminus-2` | `DCAgent/neulab-nnetnav-wa-sandboxes-traces-terminus-2` | executed_trajectories | not stated on the dataset card | 6.0k | 0.2 | new | 12 ok |
| `dcagent-t2-dcagent-neulab-openhands-sandboxes-traces-terminus-2` | `DCAgent/neulab-openhands-sandboxes-traces-terminus-2` | executed_trajectories | not stated on the dataset card | 7.3k | 0.2 | new | 12 ok |
| `dcagent-neulab-orca-agentinstruct-sandboxes` | `DCAgent/neulab-orca-agentinstruct-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.3 | new | 35 ok, 5 filtered |
| `dcagent-neulab-swe-gym-openhands-sampled-trajectories-sandboxes` | `DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 2.1 | new | 3 ok, 9 filtered |
| `dcagent-nl2bash-10k` | `DCAgent/nl2bash_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 7.6k | 0.1 | new | 12 ok |
| `dcagent-qasper-sandboxes` | `DCAgent/qasper-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.6k | 2.1 | new | 5 ok, 1 rejected, 6 filtered |
| `dcagent-r2egym-sandboxes-10k` | `DCAgent/r2egym_sandboxes_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 0.7 | new | 12 ok |
| `dcagent-r2egym-sandboxes` | `DCAgent/r2egym_sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 1.2k | 1.7 | new | 3 ok, 9 filtered |
| `dcagent-selfinstruct-naive-sandboxes-2-10k` | `DCAgent/selfinstruct-naive-sandboxes-2_10k_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.9k | 0.7 | new | 6 ok, 6 filtered |
| `dcagent-stackexchange-overflow-sandboxes-skywork-response` | `DCAgent/stackexchange-overflow-sandboxes-skywork-response_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.5k | 0.6 | new | 12 ok |
| `dcagent-stackexchange-superuser-sandboxes` | `DCAgent/stackexchange-superuser-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.7 | new | 6 ok, 6 filtered |
| `dcagent-t2-dcagent-stackexchange-tezos-sandboxes-glm-4-6-traces-jupiter` | `DCAgent/stackexchange-tezos-sandboxes_glm_4.6_traces_jupiter` | executed_trajectories | not stated on the dataset card | 2.5k | 0.8 | new | 11 ok, 1 rejected |
| `dcagent-stackexchange-tor-sandboxes` | `DCAgent/stackexchange-tor-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 1.0 | new | 3 ok, 9 filtered |
| `dcagent-swesmith-sandboxes-with-tests` | `DCAgent/swesmith-sandboxes-with_tests_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.1k | 0.7 | new | 12 ok |
| `dcagent-tulu3-sft-personas-math-sandboxes` | `DCAgent/tulu3-sft-personas-math-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 10.0k | 0.2 | new | 12 ok |
| `dcagent-wizardlm-orca-sandboxes` | `DCAgent/wizardlm-orca-sandboxes_glm_4.7_traces_jupiter` | executed_trajectories | not stated on the dataset card | 9.4k | 0.5 | new | 11 ok, 1 rejected |
| `freelancer-projects-100k-traces` | `DCAgent2/freelancer-projects-100k-traces` (default) | executed_trajectories | not stated on the dataset card | 97.0k | 3.7 | new | 40 ok |
| `dcagent-t2-dcagent2-glm-4-6-combined-traces` | `DCAgent2/GLM-4.6-combined-traces` | executed_trajectories | not stated on the dataset card | 14.7k | 0.3 | new | 40 ok |
| `dcagent-t2-dcagent2-glm-4-7-inferredbugs-sandboxes-maxeps-131k` | `DCAgent2/GLM-4.7-inferredbugs-sandboxes-maxeps-131k` | executed_trajectories | not stated on the dataset card | 10.0k | 0.6 | new | 12 ok |
| `dcagent-t2-dcagent2-glm-4-7-ling-coder-sft-sandboxes-1-maxeps-131k` | `DCAgent2/GLM-4.7-ling-coder-sft-sandboxes-1-maxeps-131k` | executed_trajectories | not stated on the dataset card | 9.8k | 0.2 | new | 12 ok |
| `dcagent-t2-dcagent2-glm-4-7-neulab-bash-textbook-tasks-maxeps-131k` | `DCAgent2/GLM-4.7-neulab-bash_textbook_tasks-maxeps-131k` | executed_trajectories | not stated on the dataset card | 8.7k | 0.3 | new | 11 ok, 1 rejected |
| `dcagent-t2-dcagent2-glm-4-7-stackexchange-overflow-sandboxes-maxeps-131k` | `DCAgent2/GLM-4.7-stackexchange-overflow-sandboxes-maxeps-131k` | executed_trajectories | not stated on the dataset card | 10.0k | 0.2 | new | 10 ok, 2 rejected |
| `dcagent-t2-dcagent2-glm-4-7-stackexchange-tezos-sandboxes-maxeps-131k` | `DCAgent2/GLM-4.7-stackexchange-tezos-sandboxes-maxeps-131k` | executed_trajectories | not stated on the dataset card | 10.0k | 0.3 | new | 12 ok |
| `dcagent2-glm47-swe-rebench` | `DCAgent2/GLM-4.7-swe_rebench-sandboxes-maxeps-131k-jup_chunk0` | executed_trajectories | not stated on the dataset card | 1.2k | 0.1 | new | 0 ok, 150 filtered |
| `dcagent-t2-dcagent2-glm-4-7-swesmith-sandboxes-with-tests-oracle-verified-120s-maxeps-131k` | `DCAgent2/GLM-4.7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k` | executed_trajectories | not stated on the dataset card | 9.4k | 0.7 | new | 12 ok |
| `dcagent-t2-dcagent2-glm46-ling-coder-sft-sandboxes-1-maxeps-131k` | `DCAgent2/glm46-ling-coder-sft-sandboxes-1-maxeps-131k` | executed_trajectories | not stated on the dataset card | 10.0k | 0.3 | new | 12 ok |
| `dcagent-t2-dcagent2-glm46-neulab-mind2web-sandboxes-maxeps-131k` | `DCAgent2/glm46-neulab-mind2web-sandboxes-maxeps-131k` | executed_trajectories | not stated on the dataset card | 9.8k | 0.4 | new | 12 ok |
| `dcagent-t2-dcagent2-gpt-oss-120b-code-contests-maxeps-131k` | `DCAgent2/GPT-OSS-120B-Code-Contests-maxeps-131k` | executed_trajectories | not stated on the dataset card | 9.0k | 0.3 | new | 11 ok, 1 rejected |
| `dcagent-t2-dcagent2-kimi-2-5-inferredbugs-sandboxes-maxeps-32k` | `DCAgent2/Kimi-2.5-inferredbugs-sandboxes-maxeps-32k` | executed_trajectories | not stated on the dataset card | 18.8k | 1.0 | new | 10 ok, 2 rejected |
| `cc-traces-pt-br` | `DedeProGames/claude-code-traces-pt-br` | executed_trajectories | not stated on the dataset card | 12* | 0.2 | new | 8 ok, 4 rejected |
| `codex-traces-pt-br` | `DedeProGames/codex-traces-pt-br` | executed_trajectories | not stated on the dataset card | 359* | 0.5 | new | 27 ok, 13 rejected |
| `menvdata-swe-trajectory` | `ernie-research/MEnvData-SWE-Trajectory` | executed_trajectories | apache-2.0 | 3.9k | 1.0 | new | 40 ok |
| `fineenvs-data-agent-sft` | `FineEnvs/data-agent-sft` | executed_trajectories | mit | 4.7k | 0.1 | new | 40 ok |
| `harbor-rollouts-claude-code` | `FineEnvs/qwen38-27b-harbor-rollouts` (claude-code) | verified_trajectories | not stated on the dataset card | 781 | 0.3 | new | 150 ok |
| `harbor-rollouts-codex` | `FineEnvs/qwen38-27b-harbor-rollouts` (codex) | verified_trajectories | not stated on the dataset card | 797 | 0.2 | new | 150 ok |
| `harbor-rollouts-mini-swe-agent` | `FineEnvs/qwen38-27b-harbor-rollouts` (mini-swe-agent) | verified_trajectories | not stated on the dataset card | 810 | 0.1 | new | 150 ok |
| `harbor-rollouts-opencode` | `FineEnvs/qwen38-27b-harbor-rollouts` (opencode) | verified_trajectories | not stated on the dataset card | 801 | 0.2 | new | 143 ok, 7 held/duplicate |
| `focustiki-sft-coding-agent-traces-test` | `focustiki/sft-coding-agent-traces` (test) | executed_trajectories | mit | 663 | 0.1 | new | 31 ok, 9 rejected |
| `focustiki-sft-coding-agent-traces-train` | `focustiki/sft-coding-agent-traces` | executed_trajectories | mit | 6.0k | 0.6 | new | 36 ok, 4 rejected |
| `davinci-dev-env-native` | `GAIR/daVinci-Dev` | verified_trajectories | other | 72.7k | 15.6 | new | 23 ok, 127 filtered |
| `drkernel-coldstart-8k` | `hkust-nlp/drkernel-coldstart-8k` | environment_trajectories | mit | 8.9k | 0.5 | new | 40 ok |
| `swe-rebench-july-2026-trajectories` | `ibragim-bad/swe_rebench_07_2026_trajectories` | executed_trajectories | other | 9.4k | 1.9 | new | 40 ok |
| `council-of-chuckles-codex` | `jdsouza/council-of-chuckles-codex-traces` | executed_trajectories | not stated on the dataset card | 10* | 0.0 | new | 6 ok, 4 rejected |
| `jetbrains-swesmith-random-subset` | `JetBrains-Research/agent-trajectories-swesmith-random-subset` | verified_trajectories | mit | 1.5k | 0.1 | new | 68 ok, 82 filtered |
| `jetbrains-swe-traj-complete` | `JetBrains-Research/swe-traj-complete` (default) | executed_trajectories | not stated on the dataset card | 269 | 0.0 | new | 40 ok |
| `swerebench-openhands-minimax-m25-corrected` | `JierunChen/swerebench-filtered-openhands-minimax-m2_5-corrected-completions` | executed_trajectories | not stated on the dataset card | 2.2k* | 0.7 | new | 40 ok |
| `julien-c-pi-sessions` | `julien-c/pi-sessions` | executed_trajectories | cc-by-4.0 | 17* | 0.0 | new | 11 ok, 6 held/duplicate |
| `synthtraces-pi` | `julien-c/synthtraces` | executed_trajectories | mit | 2.9k* | 0.3 | new | 29 ok, 11 rejected |
| `jupyter-agent-thinking` | `jupyter-agent/jupyter-agent-dataset` (thinking) | executed_trajectories | apache-2.0 | 5.5k | 107.6 | 2.6k; 0.32 | 84 ok, 66 filtered |
| `jwei-swe-smith-go-qwen35` | `JWei05/swe_smith_go_qwen3.5_35b_trajs_1629_no_p2p` | executed_trajectories | not stated on the dataset card | 1.6k | 1.0 | new | 29 ok, 11 filtered |
| `jwei-swe-smith-java-qwen35` | `JWei05/swe_smith_java_qwen3.5_35b_trajs_4369` | executed_trajectories | not stated on the dataset card | 5.0k | 4.7 | new | 40 ok |
| `jwei-swe-smith-js-qwen35` | `JWei05/swe_smith_js_qwen3.5_35b_trajs_4358` | executed_trajectories | not stated on the dataset card | 5.0k | 3.3 | new | 35 ok, 5 filtered |
| `jwei-swe-smith-py-qwen35` | `JWei05/swe_smith_py_qwen3.5_35b_trajs_3934_no_p2p` | executed_trajectories | not stated on the dataset card | 3.9k | 3.7 | new | 40 ok |
| `jwei-swe-smith-rs-qwen35` | `JWei05/swe_smith_rs_qwen3.5_35b_trajs_2477` | executed_trajectories | not stated on the dataset card | 5.0k | 1.3 | new | 40 ok |
| `swe-smith-mini` | `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` | executed_trajectories | mit | 66.0k | 4.7 | 500; 0.06 | 40 ok |
| `legoflow-swe-glm52-opencode` | `Lego-X/LegoFlow-SWE` | verified_trajectories | not stated on the dataset card | n/a | 2.0 | new | 0 ok, 150 filtered |
| `legoflow-swe-glm52-openhands-sdk` | `Lego-X/LegoFlow-SWE` | verified_trajectories | not stated on the dataset card | n/a | 2.5 | new | 0 ok, 150 filtered |
| `swe-lego-synthetic-resolved` | `Lego-X/SWE-Lego-Synthetic-Data` (default) | verified_trajectories | apache-2.0 | 9.1k | 1.4 | new | 150 ok |
| `swe-review-traj` | `Lego-X/SWE-Review-Traj` (default) | executed_trajectories | apache-2.0 | 8.9k | 1.8 | new | 40 ok |
| `terminal-lego-traj-8k-4teachers` | `Lego-X/Terminal-Lego-Traj-8k` | executed_trajectories | none stated | 33.2k | 0.8 | 3.0k; 0.37 | 40 ok |
| `lego-x-terminal-lego-traj-deepseek-v3-2-15k` | `Lego-X/Terminal-Lego-Traj-Deepseek-V3-2-15k` | executed_trajectories | not stated on the dataset reposit~ | 14.8k | 0.5 | new | 40 ok |
| `ml-intern-sessions` | `lewtun/ml-intern-sessions` | executed_trajectories | other | 575* | 0.0 | new | 10 ok, 30 rejected |
| `claude-code-glm53-swesmith-compact-full-train` | `liangzhidanta/claude-code-glm53-swesmith-compact-trajectories` | executed_trajectories | other | n/a | 0.0 | new | 150 ok |
| `claude-code-glm53-swesmith-compact-full-validation` | `liangzhidanta/claude-code-glm53-swesmith-compact-trajectories` | executed_trajectories | other | n/a | 0.0 | new | 16 ok |
| `claude-code-glm53-swesmith-trajectories` | `liangzhidanta/claude-code-glm53-swesmith-trajectories` | verified_trajectories | other | 1.0k | 0.1 | new | 150 ok |
| `litecoder-sft-terminal-preview` | `Lite-Coder/LiteCoder-SFT-Terminal-preview` | executed_trajectories | mit | 940 | 0.1 | new | 40 ok |
| `litecoder-terminal-sft` | `Lite-Coder/LiteCoder-Terminal-SFT` | executed_trajectories | mit | 11.3k | 1.2 | new | 40 ok |
| `litecoder-terminal-world-model-sft` | `Lite-Coder/LiteCoder-Terminal-World-Model-SFT` | executed_trajectories | not stated on the dataset reposit~ | 7.1k | 0.5 | new | 40 ok |
| `swe-star` | `LogicStar/SWE-Star` | verified_trajectories | not stated on the dataset card | 244.0k | 101.5 | new | 61 ok, 89 filtered |
| `terminaltraj-20k` | `m-a-p/TerminalTraj` | executed_trajectories | not stated on the card | 20.0k | 0.5 | 2.0k; 0.25 | 39 ok, 1 rejected |
| `real-pi-sessions-maxdevv` | `MaxDevv/real-pi-coding-agent-traces-sessions` | executed_trajectories | other | 249* | 0.2 | new | 36 ok, 3 rejected, 1 held/duplicate |
| `orchard-swe-unresolved` | `microsoft/Orchard` (swe) | executed_trajectories | mit | 107.2k | 29.1 | new | 0 ok, 150 filtered |
| `orchard-swe-verified` | `microsoft/Orchard` (swe) | verified_trajectories | mit | 107.2k | 29.1 | 1.4k; 0.17 | 150 ok |
| `s-verified-orchard` | `microsoft/Orchard` (swe) | verified_successful_multiturn | mit | 107.2k | 29.1 | 700; 0.09 | 0 ok, 150 held/duplicate |
| `dcagent-t2-mlfoundations-dev-magicoder-evol-instruct-110k-sandboxes-traces-terminus-2` | `mlfoundations-dev/magicoder-evol-instruct-110k-sandboxes-traces-terminus-2` | executed_trajectories | not stated on the dataset card | 10.0k | 0.2 | new | 12 ok |
| `dcagent-t2-mlfoundations-dev-stackexchange-superuser-sandboxes-traces-terminus-2` | `mlfoundations-dev/stackexchange-superuser-sandboxes-traces-terminus-2` | executed_trajectories | not stated on the dataset card | 10.0k | 0.2 | new | 12 ok |
| `dcagent-t2-mlfoundations-dev-stackexchange-unix-sandboxes-traces-terminus-2` | `mlfoundations-dev/stackexchange-unix-sandboxes-traces-terminus-2` | executed_trajectories | not stated on the dataset card | 10.0k | 0.3 | new | 12 ok |
| `mlf-swesmith-plain-docker-gpt5nano-terminus2` | `mlfoundations-dev/swesmith_with_plain_docker-sandboxes-traces-terminus-2` | mixed_quality_episodes | not stated on the dataset card | 10.0k | 0.5 | new | 40 ok |
| `dcagent-t2-mlfoundations-dev-wizardlm-orca-evol-instruct-110k-sandboxes-traces-terminus-2` | `mlfoundations-dev/wizardlm_orca-evol-instruct-110k-sandboxes-traces-terminus-2` | executed_trajectories | not stated on the dataset card | 10.0k | 0.1 | new | 12 ok |
| `nanoswe-260812-swe-smith-extra-short` | `nanoswe/nanoswe-trajs-260812` | verified_trajectories | mit | 1.58M | 51.7 | new | 0 ok, 150 filtered |
| `nanoswe-260812-unverified` | `nanoswe/nanoswe-trajs-260812` | mixed_quality_episodes | mit | 1.58M | 51.7 | new | 0 ok, 150 filtered |
| `nanoswe-261002-verified` | `nanoswe/nanoswe-trajs-261002` | verified_trajectories | mit | 1.38M | 46.6 | new | 65 ok, 85 filtered |
| `nanoswe-swesmith-qwen36-35b` | `nanoswe/swesmith-qwen3.6-35b-a3b` | verified_trajectories | not stated on the dataset card | 101.9k | 4.4 | new | 105 ok, 45 filtered |
| `swe-agent-nebius` | `nebius/SWE-agent-trajectories` | verified_trajectories | cc-by-4.0 | 80.0k | 3.3 | 6.0k; 0.74 | 17 ok, 133 filtered |
| `swe-agent-nebius-unresolved` | `nebius/SWE-agent-trajectories` | executed_trajectories | cc-by-4.0 | 80.0k | 3.3 | new | 133 ok, 17 filtered |
| `rebench-openhands` | `nebius/SWE-rebench-openhands-trajectories` | verified_trajectories | cc-by-4.0 | 67.1k | 6.2 | 2.0k; 0.25 | 77 ok, 73 filtered |
| `rebench-openhands-unresolved` | `nebius/SWE-rebench-openhands-trajectories` | executed_trajectories | cc-by-4.0 | 67.1k | 6.2 | new | 73 ok, 77 filtered |
| `s-verified-rebench` | `nebius/SWE-rebench-openhands-trajectories` | verified_successful_multiturn | cc-by-4.0 | 67.1k | 6.2 | 1.0k; 0.12 | 0 ok, 73 filtered, 77 held/duplicate |
| `adp-mini-coder` | `neulab/agent-data-collection` | executed_trajectories | not stated on the dataset card | 87.7k | 5.3 | 1.0k; 0.12 | 40 ok |
| `adp-openhands-feedback` | `neulab/agent-data-collection` | executed_trajectories | mit | 226 | 0.0 | 200; 0.03 | 33 ok, 4 rejected, 3 held/duplicate |
| `adp-swe-play` | `neulab/agent-data-collection` | executed_trajectories | not stated on the dataset card | 630 | 0.2 | 500; 0.06 | 40 ok |
| `nex-agent-sft-agentic-code` | `nex-agi/agent-sft` | executed_trajectories | odc-by | 24.8k | 4.0 | new | 40 ok |
| `claude-code-merged-traces` | `nlile/misc-merged-claude-code-traces-v1` | executed_trajectories | not stated on the dataset card | 32.1k | 3.5 | new | 13 ok, 6 rejected, 16 filtered, 5 held/duplicate |
| `share-codex-sessions` | `nmuendler/share-codex` | executed_trajectories | cc-by-4.0 | 4.0k | 0.6 | new | 100 ok, 1 rejected, 49 filtered |
| `nemotron-cascade2-swe` | `nvidia/Nemotron-Cascade-2-SFT-Data` (swe) | executed_trajectories | other | 22.0k | 36.3 | new | 40 ok |
| `nemotron-opencode-agent-skills` | `nvidia/Nemotron-SFT-OpenCode-v1` | executed_trajectories | cc-by-4.0 | 67.0k | 6.3 | new | 33 ok, 7 rejected |
| `nemotron-opencode-bash-only-tool` | `nvidia/Nemotron-SFT-OpenCode-v1` | executed_trajectories | cc-by-4.0 | 97.0k | 6.9 | new | 37 ok, 3 rejected |
| `nemotron-opencode-bash-only-tool-skills` | `nvidia/Nemotron-SFT-OpenCode-v1` | executed_trajectories | cc-by-4.0 | 96.0k | 5.6 | new | 27 ok, 13 rejected |
| `nemotron-opencode-general` | `nvidia/Nemotron-SFT-OpenCode-v1` | executed_trajectories | cc-by-4.0 | 90.0k | 5.5 | new | 28 ok, 12 rejected |
| `nemotron-swe-v2-openhands` | `nvidia/Nemotron-SFT-SWE-v2` | executed_trajectories | cc-by-4.0 | 46.3k | 11.4 | 500; 0.06 | 40 ok |
| `nemotron-swe-v3` | `nvidia/Nemotron-SFT-SWE-v3` | executed_trajectories | cc-by-4.0 | 238.0k | 35.0 | 800; 0.10 | 40 ok |
| `nemotron-swe-v35-opencode` | `nvidia/Nemotron-SFT-SWE-v3.5` | executed_trajectories | cc-by-4.0 | 5.1k | 1.1 | 500; 0.06 | 40 ok |
| `nemotron-swe-v1-r2e` | `nvidia/Nemotron-SWE-v1` | executed_trajectories | cc-by-4.0 | 51.0k | 11.1 | 500; 0.06 | 40 ok |
| `nemotron-terminal-easy` | `nvidia/Nemotron-Terminal-Corpus` (skill_based_easy) | executed_trajectories | cc-by-4.0 | 44.8k | 2.4 | 1.0k; 0.12 | 40 ok |
| `nemotron-terminal-medium` | `nvidia/Nemotron-Terminal-Corpus` (skill_based_medium) | executed_trajectories | cc-by-4.0 | 89.3k | 6.8 | 1.5k; 0.18 | 40 ok |
| `nemotron-terminal-mixed` | `nvidia/Nemotron-Terminal-Corpus` (skill_based_mixed) | executed_trajectories | cc-by-4.0 | 5.7k | 0.3 | 500; 0.06 | 39 ok, 1 rejected |
| `open-swe-v10-openhands` | `nvidia/Open-SWE-Traces` (v1.0) | executed_trajectories | cc-by-4.0 | 84.1k | 22.6 | new | 40 ok |
| `open-swe-v10-sweagent` | `nvidia/Open-SWE-Traces` (v1.0) | executed_trajectories | cc-by-4.0 | 67.2k | 16.5 | new | 39 ok, 1 held/duplicate |
| `open-swe-v11-minisweagent` | `nvidia/Open-SWE-Traces` (v1.1) | executed_trajectories | cc-by-4.0 | 95.3k | 12.0 | new | 40 ok |
| `open-swe-v11-openhands` | `nvidia/Open-SWE-Traces` (v1.1) | executed_trajectories | cc-by-4.0 | 90.7k | 21.3 | new | 40 ok |
| `open-swe-v11-sweagent` | `nvidia/Open-SWE-Traces` (v1.1) | executed_trajectories | cc-by-4.0 | 77.3k | 18.9 | new | 40 ok |
| `open-swe-v12-minisweagent` | `nvidia/Open-SWE-Traces` (v1.2) | executed_trajectories | cc-by-4.0 | n/a | 32.1 | new | 40 ok |
| `open-swe-v12-openhands` | `nvidia/Open-SWE-Traces` (v1.2) | executed_trajectories | cc-by-4.0 | n/a | 10.6 | new | 40 ok |
| `open-swe-v12-sweagent` | `nvidia/Open-SWE-Traces` (v1.2) | executed_trajectories | cc-by-4.0 | n/a | 19.4 | new | 40 ok |
| `open-swe-v20-minisweagent` | `nvidia/Open-SWE-Traces` (v2.0) | executed_trajectories | cc-by-4.0 | 36.6k | 11.0 | new | 40 ok |
| `open-swe-v20-openhands` | `nvidia/Open-SWE-Traces` (v2.0) | executed_trajectories | cc-by-4.0 | 26.0k | 9.9 | new | 40 ok |
| `open-swe-v20-sweagent` | `nvidia/Open-SWE-Traces` (v2.0) | executed_trajectories | cc-by-4.0 | 34.7k | 11.5 | new | 40 ok |
| `swe-hero` | `nvidia/SWE-Hero-openhands-trajectories` | executed_trajectories | cc-by-4.0 | 34.3k | 7.2 | 500; 0.06 | 40 ok |
| `swe-zero-openhands` | `nvidia/SWE-Zero-openhands-trajectories` | mixed_quality_episodes | cc-by-4.0 | 318.1k | 36.6 | 500; 0.06 | 40 ok |
| `oa-code-contests-noblock-opencode` | `open-athena/code-contests-noblock-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 4.6k | 0.4 | new | 38 ok, 2 rejected |
| `oa-rle-adversarial-opencode` | `open-athena/exp_rle_adversarial-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 4.1k | 0.4 | new | 40 ok |
| `oa-crosscodeeval-csharp-v4-opencode` | `open-athena/exp_rpt_crosscodeeval-csharp-v4-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 1.8k | 0.1 | new | 40 ok |
| `oa-crosscodeeval-java-opencode` | `open-athena/exp_rpt_crosscodeeval-java-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 1.9k | 0.1 | new | 40 ok |
| `oa-curriculum-easy-opencode` | `open-athena/exp_rpt_curriculum-easy-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 466 | 0.0 | new | 40 ok |
| `oa-curriculum-hard-opencode` | `open-athena/exp_rpt_curriculum-hard-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 22 | 0.0 | new | 22 ok |
| `oa-curriculum-medium-opencode` | `open-athena/exp_rpt_curriculum-medium-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 451 | 0.0 | new | 39 ok, 1 rejected |
| `oa-e2egit-large-opencode` | `open-athena/exp_rpt_e2egit-large-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 4.1k | 0.2 | new | 40 ok |
| `oa-e2egit-v2-opencode` | `open-athena/exp_rpt_e2egit-v2-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 475 | 0.0 | new | 40 ok |
| `oa-ghactions-v3-opencode` | `open-athena/exp_rpt_ghactions-v3-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 7.4k | 0.4 | new | 39 ok, 1 rejected |
| `oa-issue-opencode` | `open-athena/exp_rpt_issue-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 279 | 0.0 | new | 35 ok, 5 rejected |
| `oa-methods2test-large-v3-opencode` | `open-athena/exp_rpt_methods2test-large-v3-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 108 | 0.0 | new | 40 ok |
| `oa-multifile-opencode` | `open-athena/exp_rpt_multifile-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 99 | 0.0 | new | 40 ok |
| `oa-nemotron-junit-opencode` | `open-athena/exp_rpt_nemotron-junit-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 339 | 0.0 | new | 40 ok |
| `oa-pr-opencode` | `open-athena/exp_rpt_pr-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 3.8k | 0.4 | new | 33 ok, 7 rejected |
| `oa-pymethods2test-large-opencode` | `open-athena/exp_rpt_pymethods2test-large-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 4.8k | 0.2 | new | 40 ok |
| `oa-pymethods2test-v3-opencode` | `open-athena/exp_rpt_pymethods2test-v3-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 482 | 0.0 | new | 40 ok |
| `oa-stack-junit-v6-opencode` | `open-athena/exp_rpt_stack-junit-v6-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 660 | 0.1 | new | 38 ok, 2 rejected |
| `oa-stack-pytest-large-opencode` | `open-athena/exp_rpt_stack-pytest-large-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 1.3k | 0.1 | new | 40 ok |
| `oa-stack-pytest-v2-opencode` | `open-athena/exp_rpt_stack-pytest-v2-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 472 | 0.0 | new | 39 ok, 1 rejected |
| `oa-unitsyn-python-large-opencode` | `open-athena/exp_rpt_unitsyn-python-large-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 2.2k | 0.1 | new | 40 ok |
| `oa-unitsyn-python-v3-opencode` | `open-athena/exp_rpt_unitsyn-python-v3-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 130 | 0.0 | new | 40 ok |
| `dcagent-t2-open-athena-glm-4-6-freelancer-32eps-131k` | `open-athena/GLM-4.6-freelancer-32eps-131k` | executed_trajectories | not stated on the dataset card | 7.6k | 0.4 | new | 12 ok |
| `dcagent-t2-open-athena-glm-4-6-inferredbugs-32ep-65k-reasoning` | `open-athena/GLM-4.6-inferredbugs-32ep-65k-reasoning` | executed_trajectories | not stated on the dataset card | 8.2k | 0.3 | new | 12 ok |
| `dcagent-t2-open-athena-glm-4-6-stackexchange-overflow-sandboxes-32eps-65k-reasoning` | `open-athena/GLM-4.6-stackexchange-overflow-sandboxes-32eps-65k-reasoning` | executed_trajectories | not stated on the dataset card | 9.4k | 0.3 | new | 11 ok, 1 rejected |
| `dcagent-t2-open-athena-glm-4-6-swesmith-32ep-131k-nosumm` | `open-athena/GLM-4.6-swesmith-32ep-131k-nosumm` | executed_trajectories | not stated on the dataset card | 6.0k | 0.4 | new | 40 ok |
| `glm53-swesmith-attempts` | `open-athena/GLM-5.3-tmax-swesmith-k8-traces-2026.10` (attempts) | verified_trajectories | other | 35.0k | 4.4 | new | 53 ok, 97 filtered |
| `glm53-tmax-attempts` | `open-athena/GLM-5.3-tmax-swesmith-k8-traces-2026.10` (attempts) | executed_trajectories | other | 35.0k | 4.4 | new | 0 ok, 150 filtered |
| `glm53-swesmith-relay` | `open-athena/GLM-5.3-tmax-swesmith-k8-traces-2026.10` (relay) | verified_trajectories | other | 11.0k | 1.5 | new | 46 ok, 104 filtered |
| `glm53-tmax-relay` | `open-athena/GLM-5.3-tmax-swesmith-k8-traces-2026.10` (relay) | executed_trajectories | other | 11.0k | 1.5 | new | 0 ok, 150 filtered |
| `dcagent-t2-open-athena-glm46-code-feedback-maxeps-131k` | `open-athena/glm46-code-feedback-maxeps-131k` | executed_trajectories | not stated on the dataset card | 9.8k | 0.4 | new | 12 ok |
| `oa-glm52-swe-rebench-oracle` | `open-athena/glm52-datagen-r11-52-swe-rebench-oracle-traces` | executed_trajectories | not stated on the dataset card | 1.9k | 0.4 | new | 47 ok, 103 filtered |
| `oa-glm52-openswe-oracle` | `open-athena/glm52-datagen-r11-61-openswe-oracle-traces` | executed_trajectories | not stated on the dataset card | 592 | 0.1 | new | 53 ok, 97 filtered |
| `oa-inferredbugs-sandboxes-verifier-opencode` | `open-athena/inferredbugs-sandboxes-verifier-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 5.6k | 0.7 | new | 38 ok, 2 rejected |
| `dcagent-t2-open-athena-kimi-2-5-swesmith-sandboxes-with-tests-oracle-verified-120s-maxeps-32k` | `open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k` | executed_trajectories | not stated on the dataset card | 9.4k | 0.5 | new | 12 ok |
| `dcagent-t2-open-athena-kimi-k2t-bash-textbook-tasks-maxeps-32k` | `open-athena/Kimi-K2T-bash_textbook_tasks-maxeps-32k` | executed_trajectories | not stated on the dataset card | 7.2k | 0.2 | new | 12 ok |
| `dcagent-t2-open-athena-kimi-k2t-neulab-agenttuning-kg-sandboxes-maxeps-32k-neulab-agenttuning-kg-sandboxes` | `open-athena/Kimi-K2T-neulab-agenttuning-kg-sandboxes-maxeps-32k_neulab-agenttuning-kg-sandboxes` | executed_trajectories | not stated on the dataset card | 8.0k | 0.2 | new | 12 ok |
| `dcagent-t2-open-athena-kimi-k2t-neulab-agenttuning-mind2web-sandboxes-maxeps-32k-neulab-agenttuning-db-sandbox` | `open-athena/Kimi-K2T-neulab-agenttuning-mind2web-sandboxes-maxeps-32k_neulab-agenttuning-db-sandboxes` | executed_trajectories | not stated on the dataset card | 15.8k | 0.2 | new | 12 ok |
| `dcagent-t2-open-athena-kimi-k2t-neulab-agenttuning-webshop-sandboxes-maxeps-32k` | `open-athena/Kimi-K2T-neulab-agenttuning-webshop-sandboxes-maxeps-32k` | executed_trajectories | not stated on the dataset card | 5.7k | 0.2 | new | 12 ok |
| `oa-mix-h4-binary-easy-opencode` | `open-athena/mix_h4_binary_easy-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 2.0k | 0.1 | new | 40 ok |
| `oa-nemotron-code-oracle-filtered-opencode` | `open-athena/nemotron-code-oracle-filtered-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 5.4k | 0.4 | new | 39 ok, 1 rejected |
| `oa-nl2bash-tasks-cleaned-oracle-opencode` | `open-athena/nl2bash-tasks-cleaned-oracle-qwen3.5-122b-131k-opencode-sft-serveparity` | executed_trajectories | not stated on the dataset card | 633 | 0.0 | new | 40 ok |
| `oa-swegym-minimax-m27` | `open-athena/swegym-tasks-patched-validated-v5-minimax-m27-131k-traces` | executed_trajectories | not stated on the dataset card | 2.2k | 0.6 | new | 24 ok, 126 filtered |
| `oa-swesmith-oracle-minimax-m27` | `open-athena/swesmith-oracle-filtered-minimax-m27-131k-traces` | executed_trajectories | not stated on the dataset card | 12.9k | 3.4 | new | 0 ok, 150 filtered |
| `oa-swesmith-oracle-qwen35-32k` | `open-athena/swesmith-oracle-filtered-qwen3.5-122b-32k-traces` | executed_trajectories | not stated on the dataset card | 864 | 0.2 | new | 0 ok, 150 filtered |
| `agent-trove-terminus2` | `open-thoughts/AgentTrove` | executed_trajectories | apache-2.0 | 1.70M | 58.7 | 3.0k; 0.37 | 150 ok |
| `otagent-sft-100k` | `open-thoughts/OpenThoughts-Agent-SFT-100K` | executed_trajectories | apache-2.0 | 94.3k | 5.2 | 4.0k; 0.49 | 93 ok, 57 filtered |
| `otagent-sft-coldstart-9k` | `open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K` | executed_trajectories | apache-2.0 | 9.4k | 0.7 | 4.0k; 0.49 | 137 ok, 13 filtered |
| `openthoughts-agent-v1-sft` | `open-thoughts/OpenThoughts-Agent-v1-SFT` | executed_trajectories | apache-2.0 | 15.2k | 0.3 | new | 40 ok |
| `ultradata-agent-code` | `openbmb/UltraData-SFT-Agent-2609` | executed_trajectories | apache-2.0 | 22.8k | 14.9 | 1.0k; 0.12 | 40 ok |
| `codescout-training-rollouts-14b` | `OpenHands/CodeScout_Training_Rollouts` (CodeScout_14B) | mixed_quality_episodes | not stated on the dataset card | 39.0k | 1.2 | new | 40 ok |
| `codescout-training-rollouts-4b` | `OpenHands/CodeScout_Training_Rollouts` (CodeScout_4B) | mixed_quality_episodes | not stated on the dataset card | 15.8k | 0.7 | new | 40 ok |
| `dcagent-t2-penfever-stackexchange-tezos-sandboxes-kimi-2-5-smaxeps-32k` | `penfever/stackexchange-tezos-sandboxes__Kimi-2.5-smaxeps-32k` | executed_trajectories | not stated on the dataset card | 8.6k | 0.3 | new | 12 ok |
| `afm-codeagent-sft` | `PersonalAILab/AFM-CodeAgent-SFT-Dataset` | executed_trajectories | apache-2.0 | 59.9k | 0.6 | new | 40 ok |
| `mlevolve-cc-kaggle` | `Pran-Ker/mlevolve-cc-kaggle` | executed_trajectories | mit | 25* | 0.0 | new | 18 ok, 5 rejected, 2 held/duplicate |
| `r2e-sonnet-github-issue-traj` | `r2e-edits/sonnet_github_issue_traj` | executed_trajectories | not stated on the dataset card | 404 | 0.0 | new | 40 ok |
| `r2egym-sft` | `R2E-Gym/R2EGym-SFT-Trajectories` | executed_trajectories | not stated on the dataset card | 3.2k | 0.2 | 800; 0.10 | 40 ok |
| `combust-labs-pi-mono-docker` | `rgruchalski/combust-labs_pi-mono-docker` | executed_trajectories | apache-2.0 | 26* | 0.0 | new | 25 ok, 1 rejected |
| `combust-labs-pi-mono-http-proxy` | `rgruchalski/combust-labs_pi-mono-http-proxy` | executed_trajectories | apache-2.0 | 24* | 0.0 | new | 24 ok |
| `mini-coder-trajs-400k` | `ricdomolm/mini-coder-trajs-400k` (default) | verified_trajectories | mit | 396.5k | 28.2 | new | 51 ok, 349 filtered |
| `code-datascience-instruct-analysis` | `RUC-DataLab/DataScience-Instruct-500K` | synthetic_code_dialogue | mit | 26.2k | 0.4 | 3.0k; 0.03 | 40 ok |
| `swe-chat-claude` | `SALT-NLP/SWE-chat` | executed_trajectories | odc-by | 12.1k* | 25.1 | new | 39 ok, 1 rejected |
| `swe-chat-codex` | `SALT-NLP/SWE-chat` | executed_trajectories | odc-by | 2.1k* | 4.5 | new | 31 ok, 9 rejected |
| `swe-chat-opencode` | `SALT-NLP/SWE-chat` | executed_trajectories | odc-by | 1.5k* | 1.3 | new | 40 ok |
| `swe-chat-pi` | `SALT-NLP/SWE-chat` | executed_trajectories | odc-by | 377* | 1.6 | new | 30 ok, 10 rejected |
| `si2ca-discovered-strategy` | `Self-Improving-Coding-Agents/SI2CA-Training-Trajectories` (discovered_strategy) | verified_trajectories | cc-by-4.0 | 10.8k | 1.7 | new | 110 ok, 40 filtered |
| `si2ca-full-self-judgement` | `Self-Improving-Coding-Agents/SI2CA-Training-Trajectories` (full_self_judgement) | verified_trajectories | cc-by-4.0 | 10.8k | 2.7 | new | 109 ok, 41 filtered |
| `si2ca-standard` | `Self-Improving-Coding-Agents/SI2CA-Training-Trajectories` (standard) | verified_trajectories | cc-by-4.0 | 10.8k | 0.7 | new | 105 ok, 45 filtered |
| `smolagents-codeagent-original` | `smolagents/codeagent-traces` (original) | executed_trajectories | not stated on the dataset card | 48.7k | 1.0 | new | 40 ok |
| `agentbank-apps` | `Solaris99/AgentBank` (apps) | mixed_quality_episodes | apache-2.0 | 4.4k | 0.0 | 5.0k; 0.04 | 40 ok |
| `agentbank-intercode-bash` | `Solaris99/AgentBank` (intercode_bash) | executed_trajectories | apache-2.0 | 200 | 0.0 | 500; 0.00 | 40 ok |
| `agentbank-intercode-sql` | `Solaris99/AgentBank` (intercode_sql) | executed_trajectories | apache-2.0 | 4.5k | 0.0 | 3.0k; 0.03 | 40 ok |
| `fable5-claude-code-sessions` | `suayptalha/fable-5-claude-code` | executed_trajectories | not stated on the dataset card | 63 | 0.0 | new | 36 ok, 3 rejected, 1 held/duplicate |
| `swe-smith-ticks` | `SWE-bench/SWE-smith-trajectories` (default) | verified_trajectories | mit | 25.8k | 3.1 | 1.2k; 0.15 | 9 ok, 141 filtered |
| `swe-smith-ticks-unresolved` | `SWE-bench/SWE-smith-trajectories` (default) | executed_trajectories | mit | 25.8k | 3.1 | new | 141 ok, 9 filtered |
| `s-verified-swe-smith` | `SWE-bench/SWE-smith-trajectories` (default) | verified_successful_multiturn | mit | 24.1k | 3.2 | 1.0k; 0.12 | 107 ok, 43 filtered |
| `swe-smith-tool` | `SWE-bench/SWE-smith-trajectories` (default) | verified_trajectories | mit | 24.1k | 3.2 | 1.2k; 0.15 | 0 ok, 43 filtered, 107 held/duplicate |
| `swe-smith-tool-unresolved` | `SWE-bench/SWE-smith-trajectories` (default) | executed_trajectories | mit | 24.1k | 3.2 | new | 39 ok, 107 filtered, 4 held/duplicate |
| `swe-smith-xml` | `SWE-bench/SWE-smith-trajectories` (default) | verified_trajectories | mit | 26.1k | 3.2 | 1.2k; 0.15 | 75 ok, 75 filtered |
| `swe-smith-xml-unresolved` | `SWE-bench/SWE-smith-trajectories` (default) | executed_trajectories | mit | 26.1k | 3.2 | new | 75 ok, 75 filtered |
| `deepswe-kimi-k2-trajectories` | `SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K` | executed_trajectories | mit | 2.8k | 0.2 | new | 40 ok |
| `deepswe-kimi-k2-rejection-sampling` | `SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-Rejection-Sampling` | executed_trajectories | mit | 729 | 0.1 | new | 27 ok, 13 held/duplicate |
| `swegym-sampled` | `SWE-Gym/OpenHands-Sampled-Trajectories` (train.raw) | verified_trajectories | not stated on the dataset card | 6.1k | 0.9 | 6.1k; 0.74 | 12 ok, 138 filtered |
| `swegym-sampled-unresolved` | `SWE-Gym/OpenHands-Sampled-Trajectories` (train.raw) | executed_trajectories | not stated on the dataset card | 6.1k | 0.9 | new | 138 ok, 12 filtered |
| `swegym-sft-trajectories` | `SWE-Gym/OpenHands-SFT-Trajectories` (default) | executed_trajectories | mit | 491 | 0.0 | new | 40 ok |
| `deepseek-v4-pro-pi-agent` | `TeichAI/DeepSeek-v4-Pro-Agent` | executed_trajectories | not stated on the dataset card | 4.0k* | 0.3 | new | 35 ok, 5 rejected |
| `fable5-cursor-sessions` | `TeichAI/Fable-5-Cursor-Traces` | executed_trajectories | apache-2.0 | 244 | 0.1 | new | 40 ok |
| `teichai-hunter-alpha-coding-agent` | `TeichAI/Hunter-Alpha-Coding-Agent-SFT` | executed_trajectories | not stated on the dataset card | 1.2k | 0.2 | new | 40 ok |
| `sera-kimik3-django-sweagent-raw-t1` | `thientrangngv/SERA-KimiK3-Django-SWEAgent-Raw-T1` | mixed_quality_episodes | odc-by | 300 | 0.1 | new | 40 ok |
| `pi-extensions-sessions` | `thomasmustier/pi-extensions-sessions` | executed_trajectories | mit | 132* | 0.1 | new | 27 ok, 12 rejected, 1 held/duplicate |
| `pi-for-excel-sessions` | `thomasmustier/pi-for-excel-sessions` | executed_trajectories | mit | 215* | 0.4 | new | 27 ok, 13 rejected |
| `pi-mono-sessions-thomasmustier` | `thomasmustier/pi-mono-sessions` | executed_trajectories | mit | 150* | 0.1 | new | 16 ok, 23 rejected, 1 held/duplicate |
| `pi-nes-sessions` | `thomasmustier/pi-nes-sessions` | executed_trajectories | mit | 12* | 0.0 | new | 10 ok, 2 rejected |
| `pi-session-hud-sessions` | `thomasmustier/pi-session-hud-sessions` | executed_trajectories | mit | 10* | 0.0 | new | 10 ok |
| `swe-next-sft` | `TIGER-Lab/SWE-Next-SFT-Trajectories` | executed_trajectories | mit | 3.7k | 0.2 | 2.0k; 0.25 | 40 ok |
| `swe-qa-pro-sft` | `TIGER-Lab/SWE-QA-Pro-SFT-Trajectories` | executed_trajectories | mit | 1.0k | 0.1 | 1.0k; 0.01 | 40 ok |
| `coderforge-r2e-gym-failed` | `togethercomputer/CoderForge-Preview` (trajectories) | executed_trajectories | not stated on the dataset card | 33.0k | 5.5 | new | 31 ok, 119 filtered |
| `coderforge-swe-rebench-failed` | `togethercomputer/CoderForge-Preview` (trajectories) | executed_trajectories | not stated on the dataset card | 77.2k | 14.4 | new | 62 ok, 88 filtered |
| `coderforge-swe-smith-failed` | `togethercomputer/CoderForge-Preview` (trajectories) | executed_trajectories | not stated on the dataset card | 148.0k | 24.0 | new | 25 ok, 125 filtered |
| `coderforge-reward1` | `togethercomputer/CoderForge-Preview` (trajectories) | verified_trajectories | not stated on the dataset card | 155.1k | 24.5 | 1.0k; 0.12 | 40 ok |
| `s-verified-coderforge` | `togethercomputer/CoderForge-Preview` (trajectories) | verified_successful_multiturn | per row | 155.1k | 24.5 | 500; 0.06 | 0 ok, 40 held/duplicate |
| `trace-commons-claude-code` | `trace-commons/agent-traces` | executed_trajectories | cc-by-4.0 | 28* | 0.1 | new | 25 ok, 3 rejected |
| `ukisai-qwen38-terminus2` | `ukisai/Qwen3.8-27B-multi-turn-agent-sft` (default) | executed_trajectories | apache-2.0 | 15.2k | 0.4 | new | 40 ok |
| `glm51-pi-swe-rebench-v2-successful` | `whitecircle/swe-rebench-v2-glm-5.1-pi-agent-successful-traces` (default) | executed_trajectories | not stated on the dataset card | 7.8k | 0.9 | new | 40 ok |
| `codeact-instruct` | `xingyaoww/code-act` | executed_trajectories | apache-2.0 | 7.1k | 0.0 | 3.0k; 0.37 | 38 ok, 2 rejected |
| `cc-bench-trajectories` | `zai-org/CC-Bench-trajectories` | executed_trajectories | mit | 260 | 0.1 | new | 40 ok |
| `swe-dev-rft` | `zai-org/SWE-Dev-train` (rft) | verified_trajectories | not stated on the dataset card | 2.3k | 0.1 | new | 150 ok |
| `swe-dev` | `zai-org/SWE-Dev-train` (sft) | executed_trajectories | not stated on the dataset card | 17.9k | 0.9 | 500; 0.06 | 40 ok |
| **Total** | 354 sources | | | 25.97M | 1654 | | |

### Running it

`configs/cloud/prepare-full-agentic-coding.json` (20 sandboxes x 14 shards = 280
shards; it launches nothing by itself) reads the manifest with `files.agentic-coding.json`, the
file list at the pinned revisions. The plan reads 767 GB once, plus the re-reads that the
split of single-file sources by row stripes needs: about 2.5 TB in total, because one huge JSONL
file (a 24 GB file in a stripe plan of 20 shards) is read by every shard that keeps a stripe of
it. Larger shards would shrink that but raise the work per sandbox; this choice finishes the
estimated work in about four hours of ingest per sandbox. Shard cost uses the pass rates in
`data/sharding.py` for the big filtered sources.

For a smoke run, `configs/cloud/prepare-agentic-coding-mix.json` (one sandbox, nothing launches by
itself) plans the same manifest with `scripts/shard_sources.py --cap` (`CODING_ROW_CAP`, 300 by
default) into `CODING_SHARDS` (16) pieces, so the sources need no caps of their own.

## Preparing at full volume

Training also updates the per-row memory table (the knowledge-base parameters), and that
table improves with every distinct row it sees, so a build should take every row each
source offers, not a few hundred from each. That is about 82 million rows and 1.1 TB of
text across the current configs (estimate below), too much for one machine and one
process. Full-volume preparation therefore cuts a manifest into shards, prepares the shards
on separate machines, and then decides duplicates, families and splits once over all of
them.

**Cutting the manifest.** `scripts/shard_sources.py MANIFEST OUT --full --listing FILE
--shard i/n` writes shard `i` of `n`: the same sources with no row cap, each reading a
share of its files (a source with one huge file, such as a 15 GB JSON, is cut by row
stripes instead: shard `i` keeps every `n`-th row). `FILE` holds each source's file names
and sizes at the pinned revision (`configs/data/files.agentic.json`,
`configs/data/files.open.json`; rebuilt from Hub metadata when missing). Every machine
computes the same plan from the same two files and keeps its own shard, so no coordinator
is needed. Shards are balanced by expected work to within about 3 to 13 percent.
In the cloud runs a shard also holds the sources of one licence group only (see "Keeping the
prepared data" below), so the plan first shares the shards out between the groups by work.

**Three steps, on any machines:**

```bash
# 1. Each shard, anywhere: read, parse and deduplicate within the shard, then stop.
megatoken prepare --sources shard-7.json --output shards/00007 --codec codec \
    --reject-invalid --ingest-only        # leaves shards/00007/digest.sqlite
# 2. One machine, from the digests only (about 2.3 KB per episode, no text):
megatoken reconcile --digests shards/*/digest.sqlite --output verdicts
# 3. Each shard again, with its verdicts: writes the final root.
megatoken prepare --sources shard-7.json --output shards/00007 --codec codec \
    --reject-invalid --decisions verdicts/decisions-0007.sqlite
```

Step 3 does not read the sources again: step 1 already stored every parsed episode in the
shard's own database, and the verdicts only decide which of them stay and in which split.
`reconcile` replays each shard's surviving episodes, in the order given, through the same
exact and near-duplicate comparison a single store would have made, joins families that
meet across shards, quarantines benchmark families everywhere, applies the per-parent
variant cap and freezes the splits once. Fed the same rows in the same order it returns
what one single run over all of them returns (`tests/test_prepare_reconcile.py` checks
this, and checks a cross-shard exact copy, a near copy, a family split over three shards,
a benchmark hit and a variant cap). Shard roots are ordinary roots: `PreparedCorpus`
reads any number of them together.

**In the cloud.** `python -m megatoken.cloud_prepare` runs the three steps as ordinary
`megatoken.cloud` jobs on CPU sandboxes (16 CPU, 32 GiB, no GPU), reading a run file such
as `configs/cloud/prepare-full-agentic.json`:

```bash
python -m megatoken.cloud_prepare plan   configs/cloud/prepare-full-agentic.json
python -m megatoken.cloud_prepare launch configs/cloud/prepare-full-agentic.json
python -m megatoken.cloud_prepare status prepare-full-agentic
python -m megatoken.cloud_prepare stop   prepare-full-agentic   # after training took the roots
```

`plan` writes and validates every job offline (it costs nothing). `launch` runs three
waves of sandboxes, each started only when the one before has published its files:

1. **ingest**: `sandboxes` machines, each running `processes` shard preparations at a time
   over its `shards_per_sandbox` shards (more shards than processes keeps the machines
   evenly loaded), each writing `digest.sqlite` and `preparation.sqlite` per shard;
2. **reconcile**: one machine fetches all digests (sha256-pinned, over the sandboxes'
   share-token HTTPS, at data-centre speed) and writes one verdict file per shard;
3. **finish**: one fresh machine per group of shards fetches that group's
   `preparation.sqlite` files from its ingest machine and its verdicts, and writes
   `out/prepared/<shard>`. These machines serve `out/prepared` to training like any job
   with `serve_outputs`. With a `cache` block in the run file they also seal the roots and
   upload them to the Hub, after which every sandbox is stopped (see "Keeping the prepared
   data" below).

`launch` is resumable. If a shard fails twice, its machine's log is kept under
`~/.local/state/megatoken/cloud/prepare/<id>.logs`, that one machine is stopped, and running
`launch` again redoes only it; the earlier waves stay up until every root is finished
(they are billed while idle, which is why the reconcile and finish hours should be tight).
The sources are read with `HF_TOKEN` from `~/.cache/huggingface/token`, delivered as a
secret file, never in the job. Gated sources need a token that has accepted their terms.
Nothing here was run against the live service; the launcher is tested against a fake
fleet, and the whole ingest, reconcile and finish flow is tested on one machine.

**Speed.** One preparation process used to read 30 to 160 KB of source text per CPU second.
Unicode text segmentation in pure Python (finding the legal cut points, and sentence and
word edges for prose) took half to two thirds of that time on documents and long traces, a
JSON-schema check of every tool definition took up to half on tool-use rows, and per-token
loops and the pipe to the parser worker took most of the rest. The segmentation is now
checked in bulk and only the rare joined characters go to the slow path
(`data/parsing/segmentation.py`, tested against the library on random text), the cut search
runs on arrays, token widths are cached per codec, tool schemas built from common keywords
skip the metaschema walk (anything unusual still takes it), and the pipe carries plain
tuples. Every output was compared byte for byte with the old code (all eight samples below
produce identical prepared files, with the chunker's version stamp held fixed, since
editing the parser changes that stamp by design).

The rate of one process (the process plus its parser worker, so a 16-CPU sandbox does about
16 times this) on a real sample of each kind of source, before and after, measured on a
busy shared machine (repeat runs differ by up to about 15 percent):

| source | kind | sample (source KB) | before (KB per CPU s) | after (KB per CPU s) | faster | output identical |
|---|---|---:|---:|---:|---:|---|
| smol-smoltalk | chat | 6,129 | 145 | 452 | 3.1x | yes |
| tulu3-sft-mixture | chat | 2,501 | 128 | 347 | 2.7x | yes |
| open-swe-traces | long SWE trajectories | 16,107 | 100 | 558 | 5.6x | yes |
| nemotron-agentic-v4-tools | tool-use trajectories | 10,552 | 168 | 391 | 2.3x | yes |
| xlam-60k | short tool calls | 3,171 | 155 | 376 | 2.4x | yes |
| fineweb-edu-10bt | web documents | 5,238 | 32 | 393 | 12.1x | yes |
| stackv2-edu-0000 | source code files | 3,076 | 131 | 240 | 1.8x | yes |
| cosmopedia-stanford | synthetic textbooks | 4,946 | 136 | 592 | 4.3x | yes |

Web documents gain the most (about 12 times); long SWE trajectories and synthetic
textbooks gain 4 to 6 times, chat about 3 times, and tool-use rows and code files about 2
times, because their time is spread over many small steps. Weighted by how much text each
kind has in the configs, a full build is about 6 times cheaper than before.
What is left is spread thin: the Rust tokenizer, tree-sitter, the chunk planner's cut
search, near-duplicate sketches, and the dataclass objects of the episode format, each under
15 percent of a profile.

**How long everything takes.** Text is counted as Hub-reported in-memory size, which is
close to the source rows' JSON size, and each kind is priced at the rate measured above.
A sandbox of 16 CPUs running 14 processes keeps about 85 percent of its CPUs busy, which
gives per sandbox: trajectories 7.6 MB/s, documents 6.4, chat 5.7, tool use 5.3, code 3.3
(MB of source text per second; 1 MB/s is 3.6 GB per hour). Every row each source offers:

| config | rows (millions) | text (GB) | CPU-hours | sandbox-hours |
|---|---:|---:|---:|---:|
| agentic (as first measured, before its coding sources moved out) | 20.0 | 400 | 225 | 17 |
| agentic-coding (see "Agentic coding sources"; text is the upper estimate) | 26.0 | 1,654 | 920 | 70 |
| open | 36.1 | 190 | 128 | 9 |
| v13 | 15.9 | 471 | 240 | 18 |
| agentic-gated | 10.5 | 24 | 16 | 1 |
| all (the rows above except agentic-coding, as first measured) | 82.5 | 1,084 | 610 | 45 |

So the current configs need about 45 sandbox-hours of ingest work in total (the "all" row
counts a source once even when two configs use it), which 16 sandboxes do in under three
hours. The estimate is an upper bound on rows (rows a source's own filter drops are cheaper
than prepared ones) and is only as good as the vCPU assumption: if the provider's vCPUs are
hyperthreads, expect up to 1.5 times as long. The real cost is not ingest but the idle time
of the waiting waves: reconcile replays about 4,000 episodes per second on one core (measured
on 800,000 synthetic digests), a digest is 2.3 KB per episode, so 20 million episodes are 46 GB
of digests, about 1.5 hours plus transfers, with a working store of roughly 70 GB of disk, and
every ingest sandbox is billed while it waits. For `prepare-full-agentic` (16 sandboxes,
about 20 million rows) that is roughly 16 x 2 = 32 idle sandbox-hours on top of 17 of work.
Run a second manifest's ingest while the first reconciles to use that time.

Run files exist for `sources.agentic.json` (16 sandboxes, 448 shards), `sources.open.json`
(8 sandboxes, 224 shards) and `sources.agentic-coding.json` (20 sandboxes, 280 shards; the
agentic row above still counts the coding sources that moved out of it). The v13 and gated manifests list sources that need an accepted
click-through on the Hub: build their listings with a token that has access
(`scripts/shard_sources.py ... --listing`), then copy a run file.

Before the first full run, launch a one-sandbox run (set `sandboxes` to 1 in a copy of the
run file, with a pilot manifest) to check two numbers this document could only estimate: the
local disk of a sandbox (an ingest sandbox keeps about as much SQLite as source text, 25 GB
or more for the agentic run) and the real source read rate, since streaming a large parquet
file from the Hub shares one connection among the 14 processes.

Public media (images, speech, sound, video and edits) is prepared the same way with `"kind": "media"`
run files on GPU sandboxes (`configs/cloud/prepare-full-media-*.json`); see "Cloud jobs and
full-volume runs" under "Scaling the media streams" below.

### Keeping the prepared data: one Hub repository, public subsets and an encrypted folder

Sandboxes disappear within hours and there is no durable cloud disk or bucket, so a finished
build would be lost with them. A run file with a `cache` block (`"cache": {"repo":
"gabrielpickard/megatoken-training-data"}`; the four text run files `prepare-full-agentic`,
`-open`, `-agentic-chat` and `-agentic-coding` have one) makes every finishing sandbox publish its
finished roots to that one public Hugging Face dataset repository. The repository name is the only
thing to change to publish elsewhere. Code: `src/megatoken/publish.py` (layout, Parquet, pin, card)
and `src/megatoken/cache.py` (the encryption); tests: `tests/test_publish.py`, `tests/test_cache.py`
(real files and real encryption against a fake Hub, no network).

**Which sources are public.** Every text source records a decision in its `terms`:
`redistribute` (`permissive`, `share_alike`, `non_commercial` or `no`) and `redistribute_basis`
(a sentence saying why). The rule, applied conservatively to each source's recorded licence text
and notes:

| group | what goes in | published as |
|---|---|---|
| `permissive` | MIT, Apache-2.0, BSD, CC0, CC-BY, ODC-By, ODbL, PDDL, Unlicense (and the clearly equivalent CDLA-Permissive-2.0 and OpenMDW-1.1) stated by the publisher, with nothing in the notes that limits re-sharing | readable Parquet, each row under its source's licence |
| `share_alike` | CC-BY-SA, or a permissive set whose notes record CC-BY-SA rows | readable Parquet, CC-BY-SA-4.0 |
| `non_commercial` | CC-BY-NC or CC-BY-NC-SA, no ND | readable Parquet, the NC licence (NC-SA if any source is NC-SA) |
| `no` | everything else: gated sets and anything behind a click-through; no licence stated, or only inferred; "no blanket grant"; ND, research-only and educational-only; custom or vendor licences (Llama, Gemma, NVIDIA, Microsoft, RAIL, AI2 ImpACT); mixtures whose parts keep their own terms; rows that are outputs of closed commercial models (OpenAI, Anthropic, Google), whose terms limit use of outputs; sessions recorded by real developers; copyleft code; per-file or per-row licences; anything the notes leave unsure | encrypted |

A source with no recorded decision counts as `no`. `tests/test_redistribution.py` checks that
every text source has the fields, that gated, unlicensed, ND and custom-licence sources are `no`,
and that a published source states a licence of its group.

**Review of uncertain sources (2026-10-23).** The source entries retain the reason for
withholding each source. The ambiguous groups remain encrypted: missing or inferred licences,
per-row and per-file grants, mixed-source grants, custom terms, gated access, and provider-output
conditions. In particular, UltraData's Apache heading does not override its upstream terms or
its explicit prohibition on unauthorized mirroring ([publisher card](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609/blob/main/README.md#-license-and-data-sources)).
OpenSWE's recorded conflicting licence statements remain unresolved, so it stays encrypted.
Gating alone is not treated as proof of a legal redistribution restriction: it is a reason to
encrypt under the user's chosen storage policy. CC-BY-SA-3.0 sources keep that original licence
in the attribution table; the group's CC-BY-SA-4.0 label does not replace upstream notices
([CC-BY-SA-3.0 terms](https://creativecommons.org/licenses/by-sa/3.0/legalcode), section 4).
The pilot uses three existing public datasets, one each from the permissive, non-commercial
and encrypted groups, before the full chat build. It tests all three preparation steps and
publishing without first allocating the full build.

**Keeping groups apart.** Full-volume preparation plans shards per group (`plan_group_shards`):
no shard mixes groups, and shards are numbered permissive first. `reconcile` replays shards in
that order and keeps the first copy of a duplicate, so a text that a restricted source shares with
a public one stays in the public group and is dropped from the restricted one (an exact copy, or a
near copy above the usual similarity threshold). Everything else (families, splits, benchmark
quarantine) is decided once over all groups. `tests/test_cloud_prepare.py` runs this on one machine
with two sources and checks it.

**The repository.**

```
README.md                                  the dataset card
permissive/<run>/part-0003/<shard>/data-00000.parquet ...   readable rows, files of about 1 GiB
permissive/<run>/part-0003/<shard>/root/   the root's manifest, index and reports
permissive/<run>/part-0003/sources.json    which sources that part holds, with attribution
share_alike/...   non_commercial/...         the same for the other public groups
encrypted/README.md                        says plainly that the folder is unusable ciphertext
encrypted/<opaque>/<opaque>.bin            everything in group `no`, sealed (see below)
```

The card (`license: other`) opens with a prominent notice that `encrypted/` is an ENCRYPTED cache of
preprocessed training data for the megatoken project that cannot be used without the project's
private key, then gives one licence per subset, a contact and takedown note, the columns, and a
per-source attribution table grouped by subset (upstream repository URL, revision, licence, licence
URL, episodes). Only the public subsets are declared as loadable configs; `encrypted/` is not. The
card is rebuilt from what the repository holds, once, at the end of each run (and by `python -m
megatoken.publish card --repo REPO` at any time), so it always covers every run. Set `cache.contact`
in the run file to say how to reach you (default: the repository's Community tab).

A Parquet row has the index columns (`source_id`, `stream`, `split`, `family_id`, `tokens`, ...), a
readable `messages` transcript, the `tools`, and `episode_json`, the stored episode verbatim, from
which a root is rebuilt byte for byte (each rebuilt shard file is checked against the root's own
manifest, the index against its recorded hash). The Parquet therefore holds the text twice (and the
token ids of the stored episode); its size at full volume is not measured.

**Hub limits.** Under 100,000 files per repository (a full set is a few thousand: files of about 1
GiB), under 10,000 entries per folder (every folder is a part, a root or a run: dozens), no file
near 200 GB, and uploads in many commits (`upload_large_folder`). Several sandboxes upload at once:
each writes only its own part's folders, so no path is ever written twice, and a commit the Hub
refuses as a race (409 or 412) is repeated (`megatoken/hub.py`, with the 429 and 503 waits).
Not checked against the live service: the account's storage quota (the four runs below prepare over 1 TB before Parquet compression;
free public storage is best-effort and large storage requires a paid plan or an agreed grant;
there is no spill-over to a second repository), its commit-rate limits, and the upload speed
from the sandboxes. The first real run is a one-sandbox pilot (`configs/cloud/prepare-pilot-storage.json`).
Current guidance: [Hub storage limits](https://huggingface.co/docs/hub/storage-limits), checked
2026-10-23; it no longer gives a 300 GB per-repository recommendation.

**The encrypted folder.** A finishing sandbox seals its group-`no` roots (just the files, never
the working databases `preparation.sqlite` and `digest.sqlite`). The files are laid end to end and
cut into pieces of exactly 1 GiB of plaintext (the last is padded with zeros), so the pieces are all
the same size. Each piece is encrypted with AES-256-GCM, 4 MiB at a time, so a multi-GB piece is
never held in memory. A piece's segments carry their position and the last one is marked, so
reordering, cutting off or swapping a piece for another file's is detected. The manifest (file
paths, offsets, the sha256 of every plaintext file and of every encrypted piece) is encrypted the
same way and padded to a multiple of 1 MiB. File and folder names are `HMAC(key, logical name)`, so
nothing in a name says what it holds. There is no source list, no licence and no per-source size
there. Identical content sealed twice gives identical bytes (the salt is a hash of the content,
which is what lets a rebuilt sandbox repeat an upload the Hub already holds).

**The key.** One random 256-bit key, in a local file only:

```bash
megatoken cache keygen          # writes ~/.config/megatoken/cache.key (mode 600, 64 hex characters)
```

`keygen` refuses to overwrite an existing key: replacing it would make everything stored under the
old one unreadable. The file is refused if other users can read it. Override the location with
`$MEGATOKEN_CACHE_KEY_FILE` or `--cache-key FILE` on the launchers. The key reaches a sandbox only
as a secret file (`$MEGATOKEN_CW_SECRETS/cache-key`, like `hf-token`), and only in the sandboxes
that need it: the finishing sandboxes of a build and the learners of a training run. It is never in
a job file, on a command line, in a log, in a commit or in the repository. Back it up somewhere other
than this machine: the encrypted part is only useful with it. The other credential is the Hub token
already used for preparation, `~/.cache/huggingface/token`, which must be a **write** token for
publishing (reading the public repository needs none, but sandboxes send it anyway).

**Publishing.** `launch` runs as before, and then each finishing sandbox runs `python -m
megatoken.cloud_prepare publish` on its roots: public roots become Parquet under the part's own
folders, group-`no` roots are sealed under `encrypted/<opaque>/`; both are uploaded with
`upload_large_folder` and then checked by listing the repository (every file there at its exact size
and sha256; the Hub records the hash of large files and a small file is downloaded and hashed). The
sandbox's receipt lists its public files and hashes. When every receipt is in, the launcher rebuilds
the card in one commit and writes the **pin** (`~/.local/state/megatoken/cloud/prepare/<run
id>.cache-pin.json`): the repository, that one commit (which holds every part and the card), and the
sha256 of every public file. It checks the pin before stopping every sandbox, the finishing ones too.
The pin holds no secret and may be committed. Resuming is cheap: converted and sealed files are kept
in the sandbox until published, the upload skips what the Hub holds, and a redone sandbox produces
the same sealed bytes. The finishing sandbox needs free disk for one more copy of its roots while
converting and sealing, and `hours.finish` of these run files is 4 to cover it.

**How a training run gets its data (the decision).** `python -m megatoken.cloud_prepare ensure
RUN.json` (also run by `cloud_diloco launch` when the data is `{"cache": {"run": "RUN.json"}}`):

| situation | answer |
|---|---|
| a pin exists, its commit still holds every public file at its recorded size and hash, and every encrypted manifest decrypts and every piece is at its recorded hash (checked without downloading the data) | **stored**: learners fetch it |
| no pin has been recorded | **rebuild** |
| a file, piece, commit or the repository is missing or altered, or the key no longer opens the manifests | **rebuild** |
| the Hub cannot be reached | error, no guess (a rebuild needs the Hub too) |

A rebuild is `launch` from the pinned sources (the old run record is set aside), which publishes and
pins the result again. Learners then run `python -m megatoken.publish fetch --pin PIN --dest
/work/data/prepared`: public files are downloaded from the pinned commit and checked against the
pin, rebuilt into prepared roots and checked against each root's manifest; each encrypted piece's
ciphertext hash is checked before it is opened and each file's plaintext hash before it gets its
name; a rerun skips encrypted files already there and verified. Any mismatch stops the learner,
since training on altered data must not start; run `ensure` again to rebuild. A training config can
list `/work/data/prepared/*` among its `datasets`; a pattern stands for every directory it matches
and matching nothing is an error. By hand: `python -m megatoken.publish verify --pin FILE`,
`python -m megatoken.publish fetch --pin FILE --dest DIR [--part N]`.

**If the key is lost.** The encrypted folder is unreadable; delete it and run `ensure`: with no way
to verify it, nothing is fetched and the data is rebuilt (the sources are pinned, so the rebuild is
the same data). Make a new key first (the old file must be moved away by hand). **If the key
leaks**, every encrypted copy ever stored under it is readable by whoever has it, and a public
repository cannot be un-published once copied, so treat what was stored as exposed from then on:
make a new key, delete `encrypted/`, and rebuild. A new key gives new names.

**Sizes.** From the file listings (`configs/data/files.*.json`) at the pinned revisions: stored
bytes are the source files as the Hub holds them; text is that expanded to row text (parquet x3,
gzip and zstd x4, the sharding plan's factors, minus rows a source's filter drops), an upper bound
on what is prepared since deduplication and rejects only remove. Sources per group (and GiB stored
and of text) for the four text run files:

| run file | permissive | share_alike | non_commercial | no |
|---|---|---|---|---|
| `prepare-full-agentic` | 33 (92 / 163) | 2 (under 0.1) | 2 (0.1 / 0.1) | 26 (51 / 111) |
| `prepare-full-open` | 5 (46 / 137) | 0 | 0 | 9 (40 / 123) |
| `prepare-full-agentic-chat` | 72 (41 / 64) | 0 | 4 (0.4 / 1.2) | 93 (54 / 86) |
| `prepare-full-agentic-coding` | 80 (341 / 612) | 0 | 0 | 278 (382 / 568) |
| all four | 190 (519 / 976) | 2 | 6 (0.5 / 1.3) | 406 (526 / 887) |

The other text manifests (`sources.agentic-more*.json`, `later-stages`, `v13`, the gated ones) have
no file listing yet: `agentic-more` has 127 permissive, 28 share_alike, 7 non_commercial and 237
`no`; `later-stages` 18, 0, 1 and 28; `v13` 6, 0, 0 and 10; the three gated manifests 34 `no`. The
encrypted size is the text plus under 0.5 percent and up to 1 GiB of padding per sandbox; the Parquet size is not
measured (see above). Pulling a run into a training
sandbox moves its bytes once; at 100 MB/s that is about 3 minutes per 17 GiB part, so a learner that
fetches everything needs several hours at that rate and the local disk for the rebuilt roots
(the speed of the Hub from the sandboxes is not measured). `cloud_diloco` restores every part into
each learner; pass `--part` to `python -m megatoken.publish fetch` yourself to restore a share.

## Causal windows and long-context credit

`iter_windows` never shuffles, crosses episodes, truncates actual earlier history,
or splits a complete logical action. Each record appears exactly once as new D
in one pass. Later windows carry the complete actual prefix as reconstruction
I, with no policy loss on the prefix. Ordinary prose and observation parts are
valid window boundaries, so a long document cannot defeat `max_records` merely
by retaining one logical source ID. Limits remain soft for complete tool actions
and explicitly extended dependencies.

An annotated writer receives gradient from a later reader only if both are in
the window's differentiable `records`. With `respect_dependencies=True`, packing
extends the segment to include linked readers and applies the explicit
`max_dependency_tokens` hard graph limit. Prefix replay alone does not qualify
as a writer-reader training window.

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

B0's single `bootstrap` source stream is explicitly an implementation
choice. R rollout collection and PPO replay belong to the training runtime;
teacher T does not substitute for RL Z. R also retains its independent episode
ceiling; emitted byte/BPE counts are reported separately from Gaussian work.


### Stage targets the runtime does not read

The packaged `stages.json` carries only what code consumes (budget meter and limit, mixture, effort mix, `constraints.autoencode`, `constraints.trainable`, and the collection pilot). The remaining plan targets below are not enforced by any code; they are the intended gates for each stage and are recorded here so they are not lost.

- Plan version `contextual32-v5`. Procurement targets: `unique_transcript_U` = [200000000000, 350000000000], `unique_executed_U_subtarget` = [15000000000, 25000000000], `accepted_fresh_episodes` = [500000, 1500000], `subtargets_overlap` = true, `inventory_verified` = false.
- **B0**: history lengths (context lengths drawn from) until 250,000,000 consumed: [4096]. Implementation pilot stratum; mix concrete reconstruction, next-facet, structure, and copy skills. gates `representation_mode` = "bootstrap", `maximum_generated_capacity` = 1024, `workspace_follows_target_length` = true, `prior_from_first_batch` = true, `noisy_codec_reconstruction` = true, `encoder_distribution_kl` = true, `max_decisions_initial` = 2, `max_decisions_final` = 4.
- **MB**: history lengths (context lengths drawn from) until 4,000,000 consumed: [4096]. Media bridge after the codec bootstrap: only the media input projection, modality embedding and realizer conditions train (trainable: media_bridge) while text replay and text anchors keep the facet space fixed. gates `representation_mode` = "recurrent".
- **P1**: history lengths (context lengths drawn from) until 20,000,000,000 consumed: [8192]; until 200,000,000,000 consumed: [32768]; until 400,000,000,000 consumed: [32768, 65536]. unique acquisition targets `natural_documents` 50,000,000,000-70,000,000,000, `natural_code` 30,000,000,000-50,000,000,000, `synthetic_documents` 60,000,000,000-85,000,000,000, `synthetic_code_rewrite` 30,000,000,000-45,000,000,000, `synthetic_code_dialogue` 20,000,000,000-30,000,000,000, `synthetic_reasoning` 15,000,000,000-25,000,000,000, `chat` 3,000,000,000-7,000,000,000, `executed_trajectories` 8,000,000,000-15,000,000,000. gates `representation_mode` = "recurrent", `encoder_contextual_reconstruction` = true, `encoder_follows_native_pairs_only` = true, `short_examples_min_fraction` = 0.2, `max_decisions_initial` = 4, `max_decisions_final` = 8, `selected_max_decisions` = 16, `large_capacity_initial_max_fraction` = 0.1, `large_capacities_require_prior_gate` = true, `bake_engram_near_end` = true.
- **M1**: history lengths (context lengths drawn from) until 60,000,000,000 consumed: [65536, 262144]. gates `representation_mode` = "recurrent", `encoder_contextual_reconstruction` = true, `encoder_follows_native_pairs_only` = true, `preserve_short_replay` = true, `trace_loss` = "assistant_only", `pre_change_state_before_patch` = true, `diagnostic_sft_at_boundary` = true.
- **M2**: history lengths (context lengths drawn from) until 40,000,000,000 consumed: [32768, 65536, 131072, 262144]. trajectory mix `repository` 0.5, `terminal` 0.2, `structured_tools` 0.2, `evidence` 0.1. gates `representation_mode` = "recurrent", `encoder_contextual_reconstruction` = true, `encoder_follows_native_pairs_only` = true, `fim_max_fraction` = 0.02, `student_prefix_fraction_initial` = 0.1, `student_prefix_fraction_final` = 0.3, `changed_actions_require_new_observations` = true, `diagnostic_sft_at_boundary` = true.
- **LC**: history lengths (context lengths drawn from) until 20,000,000,000 consumed: [32768, 65536, 131072, 262144]. gates `representation_mode` = "recurrent", `encoder_contextual_reconstruction` = true, `encoder_follows_native_pairs_only` = true, `long_episode_min_tokens` = 131072, `long_episode_max_tokens` = 262144, `long_dependency_D_min_fraction` = 0.5, `writer_reader_scored_windows_min_fraction` = 0.3, `preserve_short_medium` = true.
- **S**: history lengths (context lengths drawn from) until 2,000,000,000 consumed: [4096, 32768, 65536, 131072, 262144]. gates `representation_mode` = "recurrent", `encoder_contextual_reconstruction` = true, `encoder_follows_native_pairs_only` = true, `long_context_exposure_min_fraction` = 0.2, `seed_only_control_fraction` = 0.7, `valid_reference_control_fraction` = 0.3, `future_control_fraction` = 0.0, `initial_serving_effort` = 6.
- **D**: history lengths (context lengths drawn from) until 1,000,000,000 consumed: [4096, 32768, 65536, 131072, 262144]. gates `representation_mode` = "recurrent", `encoder_contextual_reconstruction` = true, `encoder_follows_native_pairs_only` = true, `frozen_student_collection` = true, `corrected_windows_initial_fraction` = 0.5, `corrected_windows_max_fraction` = 0.8, `correction_ramp_requires_coverage_gate` = true, `direct_teacher_token_kl` = false, `entry_requires_legal_prior_tasks` = true.
- **R**: history lengths (context lengths drawn from) until 32,000,000,000 consumed: [4096, 32768, 65536, 131072, 262144]. gates `initial_task_seeds` = 10000, `rollouts_per_seed` = 4, `initial_episodes` = 40000, `maximum_task_seeds` = 50000, `maximum_total_episodes` = 200000, `likelihood` = "canonical_facet_gaussian_density", `freeze_encoder_central_and_decoder` = true, `latent_draws_per_chunk` = 1, `deterministic_decoding` = true, `same_weight_replay_gate` = true, `initial_full_differentiable_replay` = true, `candidate_state_rebuild` = true.
- **Collection pilot**: `count_rejected_outputs` = true, `meter_teacher_inputs_and_cache` = true, `bulk_or_paid_generation_performed` = false.

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
IDs, serialized action bytes and complete version-4 raw-facet tapes are retained,
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

## Scaling the media streams

The open-licence pilot above draws a few thousand clips per media stream from six datasets. This
section records the scale-up: 174 pinned public media datasets (130 of them added
in a second pass that put image and video editing first), a reader for each kind of file layout,
pilot jobs that build a slice of every source, and full-volume runs that build all of it across
many sandboxes. Licence, size and layout facts were read from the Hub on 2026-10-22 and 23 (a card
is not legal advice). Every source carries the maintainer's recorded approval in its `terms`.
Any licence is accepted; each source's `license` and `caveat` say exactly what the card states and
what it leaves open.

### How a source is described and read

Four manifests list the sources: `configs/data/sources.media.json` (the first forty-four, some
corrected), `sources.media-more.json` (115 more: 100 open ones and 15 that needed a click on the Hub,
which the account has now made), `sources.media-more-gated.json` (the 2 that still need a
click; the gated sources that were already known stay in the first file) and `sources.media-agentic.json` (13 that teach judging and planning
edits, three of them published on a web site rather than the Hub: see "Agentic editing").
`open_media.load_sources()` reads all four. Each entry holds

- the Hub repository and the immutable revision it is pinned to, the licence found on its card,
  the approval record, notes on what the licence does not settle, and whether the repository is
  gated (the Hub account must accept its terms first);
- `include`, patterns naming the files to read at that revision;
- a **reader** (how the files are laid out) and a **task** (what the labels mean);
- `columns`, which field of a row or sample plays which role (`image`, `caption`, `audio`,
  `text`, `translation`, `original`, `edited`, `instruction`, ...), `options` (length limits,
  which caption to pick, which rows to skip) and `inventory` (items, hours, repository size and
  the basis of each number).

Readers (`data/open_media.py`):

- `parquet`: one record per row.
- `webdataset`: tar files whose members sharing a name form one sample. A member name may hold
  dots (`1.0.jpg`), and `key_split: "last"` keeps dots inside the sample name.
- `archive`: zip or tar members joined to a table (csv, JSON array or JSON lines; one table or a
  list of them; rows of one key merged; or one table per archive), optionally with a same-named
  member of a second zip, as in source/edited video pairs.
- `linked` (new): a table whose rows name their images or videos, joined to the files in tar
  (or tar.gz) archives in one pass over the archives. A row is built when its last file has
  arrived; a row whose file never comes is dropped. The archive may be one stream cut into parts
  (`multipart`, as `cat part-*` would join it), the table may be parquet with nested turns, a
  leading folder of the row's paths may be stripped, and a shard may read only some of the
  archives. This is the reader behind the multi-turn edit sessions.
- `table` (new): rows of a table file (parquet, csv, JSON array or JSON lines); for datasets that
  name their pictures by URL, with `options.fetch` (below). The files may be on a web site: a
  source whose `repository` is a URL (not a Hub dataset) lists its files in `include`, downloads
  each whole with byte-range resume, and checks it against `options.pins` (SHA-256).
- `folders` (new): one record per folder of loose files; `include` names an anchor file of each
  folder and `options.members` the files to read (field name -> path inside the folder).
- two special readers, `fleurs` (the same sentence read in two languages) and `docci` (files
  that are not on the Hub).

Tasks turn a record into a clip and then into episodes of the stage plan's own streams:

| Task | Streams |
| --- | --- |
| `transcription` | transcription, audio_identity (speech_edit with procedural edits; verbalization when `options.verbalize` is set) |
| `verbalization` | verbalization, audio_identity (the previous clip of the same speaker is the voice sample) |
| `translation_text` | speech_translation (text target), transcription, audio_identity |
| `translation_pair` | speech_translation (speech target), transcription |
| `captioned_image` | image_description, image_creation, image_identity |
| `picture` | image_identity |
| `image_questions` (new) | image_description: a multi-turn conversation about one image (question, answer, follow-up, ...), plus image_identity |
| `image_edit` | image_edit |
| `image_edit_chain` (new) | image_edit: a session of edits, each applied to the previous result |
| `edit_judging` (new) | edit_judging: the original, the request and candidate edits are shown, the verdict is written |
| `image_edit_plan` (new) | edit_plan: the original and request are shown, a plan or tool call and then the edited picture are written |
| `captioned_audio` (new) | audio_description, audio_identity |
| `audio_questions` (new) | audio_description: a conversation about one recording |
| `sound` | audio_identity |
| `captioned_video` (new) | video_description, video_identity |
| `video_questions` (new) | video_description: a conversation about one video |
| `av_caption`, `av_scenes` | av_creation, video_identity (av_edit with procedural edits) |
| `video` | video_identity (and av_edit with procedural edits) |
| `video_edit` | av_edit (instructed edits that the dataset supplies) |
| `video_edit_chain` (new) | av_edit: a session of video edits, or a pair read through `linked` |

Two stream families are new: `audio_description` (describe a recording) and `video_description`
(describe or answer questions about a silent video), each with its own share in the stage plan
and a standard concordant view (`data/concordance.py`: `grounded-audio-description`,
`grounded-video-description`). Edit sessions and question conversations compile to multi-turn
episodes (`edit_chain_view`, `question_chain_view`); see the example under "Image and video
editing" below.

Options added in this pass: `session` (rows that are the turns of one session are grouped by a
column, turns ordered by another), `choose` (take the better rated of several candidate edits),
`verbalize` (a transcription source also gives text to speech), `judge` (how a verdict is built
from scores, a vote, a winner or a column), `transform` (reshape a publisher's own row layout),
`fetch`, `originals`, `pins` and `base` (sources published as URLs), `members` (folders),
`clean: "image_tags"` (drop `<image>` markers from a prompt), `linked` (archives, `multipart`,
`strip`, the `sequence` of images and texts of a row), `whole_only` and `fps` (video questions
about whole videos at a given frame rate), `table.per_archive`, `table.merge`, `format`
(`pairs`, `messages`, `parallel` or `choices` for a conversation) and `keep` filters.
Three more were added for the gated repositories opened on 2026-10-23: a table may be a `.zip`
holding one file or a header-less `.tsv` (`table.names` names its columns), a JSON array is
decoded one row at a time instead of being loaded whole, `table.prefix` keeps only the rows
whose field starts with a text (one table of several kinds of row, read one kind at a time),
and `spool` (WebDataset) writes an archive to scratch disk and pairs its files by name, for
tars that list all their audio before the transcripts.

A translation whose target is only text was not expressible before: the concordance compiler
emits the same `speech_translation` view without the translated speech when a source has none.

Nothing is mirrored. Files are streamed from the pinned revision over HTTP range requests (the
revision is content addressed, so it is the integrity pin) or, when `<inputs>/<source_id>` exists,
read from that local directory. A shard of a full-volume run opens each file with the size from
the committed file listing, so reading it needs no call to the Hub's API. Every clip's files are
deleted as soon as its episodes are built, so a run needs scratch space for a few clips only.
Splits freeze per original family (speaker, recording, photo, video, edit session), as before.

Labels that a model wrote are recorded as such: the text artifacts carry provenance `teacher` and
the bundle's verification tier is `model-labeled` (machine transcripts or captions) or
`model-generated` (machine-made images, edits and videos); human-labelled sources keep
`publisher-annotated`.

Clip limits: speech and sound clips are 0.5-20 s (a longer recording is dropped, or for `crop`
sources cut to its central 20 s; some sources raise the limit), images are scaled to at most 1024
pixels (edit pairs losslessly; an original of another size follows the edited image when the
shapes agree within 3%), video clips are scaled into 640x640 at 16 frames/s and at most 8 s (6 s
for edit pairs).

`scripts/prepare_open_media.py` builds one prepared root: `--count SOURCE=N` draws N clips of a
source, `--procedural SOURCE=N` adds a procedural edit for its first N clips, `--option
SOURCE.KEY=JSON` overrides an option, `--scale F` multiplies every count, `--shard-slice I/N`
reads every N-th file starting at the I-th, and `--shard-spec FILE` reads one shard of a full-volume
plan (below). Without `--count` it builds the pilot sizes, so `prepare-open-mix.json` is unchanged
except that it no longer lists the pilot's media inputs; only the two DOCCI files, which are not
on the Hub, remain job inputs.

### Image and video editing

Editing was put first. The table lists every editing source with its size: edit requests (one
request is one pair of images or clips: an original, an instruction and the edited result),
sessions (a run of edits where each applies to the previous result), images or clips, and bytes.

| Source | Kind | Edit requests | Sessions | Mean edits per session | Images or clips | Repository GB | Prepared GB (est.) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `magicbrush` | image chains | 8,807 | 4,500 | 2.0 | 13,307 | 27 | 10 |
| `ultraedit` | image single edits | 3,028,083 | 0 | 1.0 | 6,056,166 | 1,555 | 4,384 |
| `hq_edit` | image single edits | 98,675 | 0 | 1.0 | 197,350 | 319 | 143 |
| `instructpix2pix` | image single edits | 313,010 | 0 | 1.0 | 626,020 | 130 | 453 |
| `omniedit` | image single edits | 1,200,000 | 0 | 1.0 | 2,400,000 | 2,977 | 1,738 |
| `anyedit` | image single edits | 2,169,605 | 0 | 1.0 | 4,339,210 | 207 | 3,141 |
| `insvie` | video single edits | 1,000,000 | 0 | 1.0 | 2,000,000 | 394 | 5,280 |
| `sharegpt4o_image_edit` | image single edits | 46,539 | 0 | 1.0 | 93,078 | 170 | 67 |
| `ultraedit_region` | image single edits | 108,179 | 0 | 1.0 | 216,358 | 46 | 157 |
| `pipe` | image single edits | 888,982 | 0 | 1.0 | 1,777,964 | 80 | 1,287 |
| `bytemorph_demo` | image single edits | 780,308 | 0 | 1.0 | 1,560,616 | 45 | 1,130 |
| `nhr_edit` | image single edits | 358,463 | 0 | 1.0 | 716,926 | 1,591 | 519 |
| `nhr_edit_part2` | image single edits | 361,625 | 0 | 1.0 | 723,250 | 801 | 524 |
| `gpt_edit_simpler` | image single edits | 1,553,575 | 0 | 1.0 | 3,107,150 | 4,936 | 2,249 |
| `editreward_data` | image single edits | 170,772 | 0 | 1.0 | 341,544 | 87 | 247 |
| `flux_kontext_ipa` | image single edits | 10,802 | 0 | 1.0 | 21,604 | 15 | 16 |
| `maskedit_10k` | image single edits | 9,613 | 0 | 1.0 | 19,226 | 44 | 14 |
| `escher_aurora_kubric` | image single edits | 93,000 | 0 | 1.0 | 186,000 | 53 | 135 |
| `escher_aurora_ag` | image single edits | 8,306 | 0 | 1.0 | 16,612 | 2 | 12 |
| `escher_human_edit` | image single edits | 4,887 | 0 | 1.0 | 9,774 | 1 | 7 |
| `escher_vismin` | image single edits | 13,668 | 0 | 1.0 | 27,336 | 10 | 20 |
| `x2edit` | image single edits | 4,800,000 | 0 | 1.0 | 9,600,000 | 4,880 | 6,950 |
| `seed_data_edit_multi_turn` | image chains | 95,000 | 21,382 | 4.4 | 116,382 | 47 | 77 |
| `seed_data_edit_real` | image single edits | 51,610 | 0 | 1.0 | 103,220 | 2 | 75 |
| `imgedit_content_memory` | image chains | 61,722 | 30,861 | 2.0 | 92,583 | 270 | 67 |
| `imgedit_content_understanding` | image chains | 84,278 | 42,139 | 2.0 | 126,417 | 560 | 92 |
| `imgedit_version_backtracking` | image chains | 84,046 | 42,023 | 2.0 | 126,069 | 560 | 91 |
| `ditto_local` | video single edits | 330,463 | 0 | 1.0 | 660,926 | 935 | 1,745 |
| `ditto_global_style1` | video single edits | 60,977 | 0 | 1.0 | 121,954 | 160 | 322 |
| `ditto_global_style2` | video single edits | 46,872 | 0 | 1.0 | 93,744 | 119 | 247 |
| `ditto_global_freeform1` | video single edits | 119,136 | 0 | 1.0 | 238,272 | 322 | 629 |
| `ditto_global_freeform2` | video single edits | 138,632 | 0 | 1.0 | 277,264 | 342 | 732 |
| `ditto_global_freeform3` | video single edits | 49,357 | 0 | 1.0 | 98,714 | 123 | 261 |
| `reco_add` | video single edits | 104,428 | 0 | 1.0 | 208,856 | 243 | 551 |
| `reco_remove` | video single edits | 85,031 | 0 | 1.0 | 170,062 | 204 | 449 |
| `reco_replace` | video single edits | 125,656 | 0 | 1.0 | 251,312 | 253 | 663 |
| `reco_style` | video single edits | 120,985 | 0 | 1.0 | 241,970 | 408 | 639 |
| `openve3m_background_change` | video single edits | 352,718 | 0 | 1.0 | 705,436 | 323 | 1,862 |
| `senorita_local_style` | video single edits | 348,986 | 0 | 1.0 | 697,972 | 135 | 1,842 |
| `senorita_obj_removal` | video single edits | 414,501 | 0 | 1.0 | 829,002 | 158 | 2,188 |
| `senorita_obj_addition` | video single edits | 532,502 | 0 | 1.0 | 1,065,004 | 40 | 2,811 |
| `senorita_obj_swap` | video single edits | 177,336 | 0 | 1.0 | 354,672 | 76 | 936 |
| `openve_sub` | video single edits | 1,600,000 | 0 | 1.0 | 3,200,000 | 2,850 | 8,447 |
| `taesiri_image_editing_request` | image single edits | 3,061 | 0 | 1.0 | 6,122 | 0 | 4 |
| `editworld_reasoning` | image single edits | 7,736 | 0 | 1.0 | 15,472 | 1 | 11 |
| `multiedit` | image single edits | 106,534 | 0 | 1.0 | 213,068 | 45 | 154 |
| `pico_banana_multi_turn` | image chains | 306,513 | 77,229 | 4.0 | 383,742 | 540 | 112 |
| `pico_banana_single` | image single edits | 257,728 | 0 | 1.0 | 515,456 | 379 | 373 |
| **total** | | 22,692,707 | 218,134 | 2.9 | 44,963,182 | 27,467 | 53,865 |

What the numbers say, in plain words:

- **Sessions of several edits in a row** are the scarcest kind of editing data. Real,
  hosted, strictly chained sessions: MagicBrush (4,500 sessions of 1 to 3 edits, human-made
  with an image-editing model), SEED-Data-Edit part 3 (21,382 sessions of 1 to 6 edits) and
  ImgEdit's multi-turn sets (115,023 sessions, two chained edits each after the cut explained
  below) and Pico-Banana-400K (77,229 sessions of 2 to 6 edits, fetched from its
  publisher's site: see "Agentic editing" below). Together 218,134 sessions and 640,366 edit requests, an average of
  2.9 edits per session. A single edit set is far larger: 16,444,761 image
  edits and 5,607,580 video edits.
- **Image edits** now cover instruction edits by human retouchers (SEED-Data-Edit part 2:
  51,610 requests answered on photo-editing forums; the small `taesiri` set of 3,061), by
  machine pipelines of every kind (inpainting, instruction-following editors, generated
  before/after frames of video), and best-of-two choices rated by people (EditReward-Data).
- **Video edits** are all machine-made (Ditto, ReCo, OpenVE, Senorita, InsViE); no open set of
  human video edits exists. Speech and audio edits stay procedural (a deterministic program on a
  real recording, recorded as such) because no open instruction-edit data exists for them.

**A three-step edit session as one episode.** Sessions are one multi-turn episode, so the model
sees its own earlier results when it makes the next edit:

```
record 0  user       original image
record 0  user       "make the sky red"                  (the same turn: image and request)
record 1  assistant  edit 1 (image)                      supervised
record 2  user       "add a boat"
record 3  assistant  edit 2 (image)                      supervised
record 4  user       "turn it into a night scene"
record 5  assistant  edit 3 (image)                      supervised
```

Each edit is generated by looking at the original, every earlier edit and every request so far:
the meridian (one step of the episode) of edit k lists the original and edit k-1 as references,
and three dependencies of kind `edit` chain each result to the next. The episode's stream is
`image_edit`; its split family is the original photo, so no session straddles train and
validation. Video sessions have the same shape with clips trimmed alike.

**What the chain readers could not use.** ImgEdit's third edit in two of its three multi-turn sets
starts again from the first result (the user withdraws the previous round), so only the first
two edits of each session are chained and the third is not read. MagicBrush sessions that span
two parquet files are read as two shorter sessions when the files go to different shards.

**Preference and feedback** are now their own task (edit judging, below). EditReward-Data is
read both ways: as best-of-two edits above (the `choose` option) and as judging.

**Not wired, with the reason:**

- Pico-Banana-400K (Apple): the best chained set (77,250 sessions of 2 to 6 edits, 56,473
  preference rows with rejected edits, 257,728 single edits). It is not on the Hub: the edited
  images sit on Apple's CDN, and the original photos are Flickr links that already fail
  (HTTP 429) or must come from the Open Images tars (86 GB). Licence CC BY-NC-ND 4.0.
- GPT-Image-Edit-1.5M original archives (4.5 TB in three tar.gz cut in 20 GiB parts whose
  input, output and annotations are in separate directories): its parquet copy
  (`UCSC-VLAA/gpt-edit-simpler`) is read instead.
- ImgEdit single-turn (1.18M pairs, 4.38 TB), Inter-Edit-Train (1.1M, two archive families),
  OmniGen2/X2I2 video-frame edits and SEED-Data-Edit part 1 Open Images (source and target in
  different archives): each needs a join between archives that a one-pass reader cannot do.
- OpenGPT-4o-Image (40k edits): its archive is shuffled and its table names `.png` files that
  are `.jpg` in the archive.
- SEED-Data-Edit part 1, Unsplash (2.0M edits of small photos, 200 GB in one archive cut in
  parts): the archive is in random order (a sample of 494 consecutive members had 494 different
  photos), so the original and the edited photo of a request are far apart and a one-pass reader
  would keep almost none of the rows. OpenGPT-4o has the same problem.
- Ditto-1M, ReCo-Data and OpenVE-3M raw archives (tars whose two clips of a pair are far apart
  or in separate tars): read through third-party parquet repacks and OpenVE's WebDataset shards
  instead.
- Agentic editing traces that are not read: ThinkEdit (about 142,000 critique-and-rewrite
  conversations, 310 GB in zips with no licence on the card), UniREdit-Data (about 100,000
  reasoning paragraphs, but the pictures sit in a 149 GB multi-part zip), the
  RationalRewards critiques (about 75,000 pairwise and 100,000 single-edit judgements written by
  a teacher model; the pictures are in two tar.gz files of 79 and 93 GB that cannot be read in
  random order), a 120-row collection of plan, critique and intermediate-picture
  trajectories (`NilanE/ComplexMultistepImageEditing`), and five small computer-use sets of an
  editor's screen (`contralabs`: Premiere, Descript, Photoshop, Firefly, Gemini: 1,734 steps with
  screenshots, thoughts and tool calls but no source or exported media).
- Video edit plans (an edit-decision list, timeline or ffmpeg command written before the video):
  no open dataset with the videos exists. Three command-only collections were found (about 1,000
  natural-language-to-ffmpeg pairs, 30 with reasoning) but they name no video, so the executed
  plans below are made instead.

### Agentic editing: judging, plans and fetched pictures

Two new task kinds teach the model to act like an editor rather than only to carry out one edit,
and a new fetch step makes a dataset usable whose pictures are published as links. Each has its
own stream in the stage plan (`data/stages.json`): `edit_judging` and `edit_plan`.

**Edit judging (`edit_judging`).** The model is shown an original picture, an edit request and
one or more candidate edits, and writes the verdict. Nothing but the verdict is supervised.

```
record 0  user       original picture
record 1  user       "Add sweeping speed lines behind the runner and a radial blur around their feet."
record 2  user       candidate A (picture)
record 3  user       candidate B (picture)
record 4  assistant  Candidate A: instruction following 2 of 4, visual quality 3 of 4, overall 5 of 8.
                     Candidate B: instruction following 1 of 4, visual quality 4 of 4, overall 5 of 8.
                     Verdict: the candidates are equally good.                   supervised
```

(a real EditReward-Data row; the scores are the people's, the candidates are edits by two editing
models). The candidates are shown in a fixed pseudo-random order, so the best one is not always
first. A source supplies the verdict as scores per aspect (`options.judge`), as a vote, as "the
publisher lists the preferred edit first" (Pico-Banana) or as a text the publisher wrote
(EditScore's judge replies). Sources: `editreward_judging`, `editscore_judging`, `pico_banana_preference`.

Volume: 271,245 judgements over 769,735 pictures (334 GB in the
repositories), so an `edit_judging` share of 0.15% of the first stage is about
2.5 passes over the data. The EditScore count is an estimate: about 45% of its rows
hold both a source and an edited picture (the rest judge one picture alone and are skipped). The
EditScore scores and reasons come from a vision-language model, not from people.

**Edit with a plan (`edit_plan`).** The model is shown the original and the request, then writes
a plan or a tool call, and then renders the edited picture; both are supervised, so the plan is
something the model must be able to carry out.

```
record 0  user       original picture
record 1  user       "Remove factory smoke."
record 2  assistant  1. The source image shows a lake scene at sunset with factory smoke rising
                     from the background. 2. The object to be edited is the factory smoke ...
                     3. The area to be edited is within the bounding box [...] ...   supervised
record 3  assistant  edited picture                                                  supervised
```

(a real Escher row). Plans are plain text or code:

- *Reasoning plans*: the six Escher repacks, EditWorld reasoning (`escher_human_edit_plan`, `escher_vismin_plan`, `escher_aurora_kubric_plan`, `escher_aurora_ag_plan`, `escher_ss2_plan`, `escher_magicbrush_plan`, `editworld_reasoning_plan`).
  The plans are machine-written and sometimes contradict the request.
- *Tool calls*: JarvisArt gives 4,055 portraits retouched by experts with the retoucher's
  step-by-step reasoning and then the exact Lightroom develop settings, written as
  `Tool call: {"tool": "lightroom.develop", "settings": {...}}` after the reasoning (only the
  non-default settings are listed). It is stored as loose files, so it has a folder reader
  (`folders`) that reads about six Hub files per portrait; the `members` option names the files.
- *Executed plans*: every procedural edit is also a plan whose result is verified by running it.
  A speech or video edit's plan is the ffmpeg command that was run
  (`ffmpeg -i input.mp4 -vf hflip output.mp4`, from 6 speech and 8 video edit types); a
  picture edit's plan is the short PIL program that was run (12 edit types, such as
  `edited = ImageOps.grayscale(image).convert("RGB")`), and for pictures only the plan is made,
  because real picture edits are plentiful. A source makes these only for the clips named by
  `--procedural` or `procedural_every`: the full-volume runs give one picture clip in 400 and
  one video clip in 40 a plan (about 345,000 and 245,000), and one speech edit in eight (the
  speech runs edit one clip in 40, so about 680,000), so the plan stream is not mostly audio.
  Their tier is `procedural`.

Volume: 173,052 real plans over 346,104 pictures (80 GB). The 0.15% share
of the first stage asks for about 862,069 plan episodes, so the real plans would be used
about 5.0 times each; the executed plans fill the rest and can be made in any
number (the stream's row in the gap table counts all of them).

**Fetching pictures from links (Pico-Banana-400K).** Apple publishes three row files and the
edited pictures on its own web site, and the original photos as Flickr links
(`options.fetch`; the reader is `table`, with `options.pins` fixing each file's SHA-256 and
`options.transform` reshaping its rows). The fetching runs in the sandbox that prepares the
shard, only for the rows that shard keeps, and never on a workstation:

- every body is stored once under the SHA-256 of its content, and every link's outcome
  (content, or dead) is remembered, so a retried shard asks the server nothing again;
- a dead link (404, 403, 410, ...) is skipped and counted. A session is cut at its first missing
  picture, or skipped when its original is missing; a judging row needs every candidate;
- a busy server (429, 503, timeouts) is retried with growing waits, and while one host asks for
  peace every request to it waits (at most 2 at a time to Flickr, 4 to other hosts);
- the counts (`fetched`, `cached`, `dead`, `failed`, `http_404`, `http_429`, ..., rows read and
  rows with a dead link) are printed after each source and written to `fetch-report.json` in the
  prepared root;
- the row files are downloaded by byte ranges that resume where a cut-off answer stopped (the
  CDN ended a long download early without an error twice), and refused when the SHA-256 differs.

Pico-Banana gives three sources: `pico_banana_preference`, `pico_banana_multi_turn`, `pico_banana_single`. The original photos are published as Flickr
links, and the Flickr route was tried first: about one photo in eight was gone (HTTP 404 or 410,
7 of 60 links) and Flickr answered a few parallel downloads with "too many requests" (429), so
on a first real try more than half of the session rows lost a picture. Open Images keeps its own
copy of every one of these photos (its table lists the original Flickr file name of each; all
241,133 distinct originals of the three row files are in it, checked against the 638 MB table), so
`options.originals` points the originals at those copies (`open-images-dataset.s3.amazonaws.com`),
with the table pinned by SHA-256 and downloaded once per machine. Only a photo missing from the
table would still be fetched from Flickr. The edited pictures are PNG files of about 1.5 MB, so
the whole set is about 1 TB to download; the shard plan treats a picture as 0.2 s of work
(`clip_seconds`) instead of the 0.05 s of a picture read from the Hub. The licence is CC BY-NC-ND
4.0 (the README's) and the source photos are Open Images (CC BY 2.0).

**Proposals still open** (not implemented):

1. *Screen trace*: a sequence of screenshots each followed by a thought and an action, for the
   `contralabs` sets. Only 1,734 steps exist openly and they have no media to edit, so it is
   cheap but small.
2. *Video plans*: a video edit written as an edit-decision list or timeline needs paired videos;
   the natural-language ffmpeg command sets could be made into executed plans by running each
   command on a public clip and keeping the ones that succeed.
3. *Critique-and-retry*: ThinkEdit's and RationalRewards' reasoned critiques followed by a
   rewritten request (about 240,000 rows) once their zips and tar.gz files can be read in order.

### The sources

Sizes are items (clips, images, pairs or sessions), hours of audio or video where the card or a
sample gives them, and the repository's size for the files read. "Gated" means the Hub account must
first accept the publisher's terms.

#### Captioned and plain images (image_identity, image_description, image_creation)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `docci` (`google/docci`) | Dense human-written descriptions of 15k photos (the training split is read). | CC-BY-4.0 | 9.65k items | Images and descriptions are fetched from the publisher's storage bucket (the Hub repository holds only a loading script); the job input pins their sha256. |
| `llava_recap_cc3m` (`lmms-lab/LLaVA-ReCap-CC3M`) | Detailed captions written by LLaVA-NeXT-34B for Conceptual Captions 3M images; images keep their web owners' rights. | unspecified on the card | 2.86M items; 138 GB |  |
| `llava_recap_558k` (`lmms-lab/LLaVA-ReCap-558K`) | LLaVA-NeXT-34B captions for the LLaVA-LCS-558K images (LAION, Conceptual Captions, SBU). | unspecified on the card | 558k items; 13 GB |  |
| `llava_recap_118k` (`lmms-lab/LLaVA-ReCap-118K`) | LLaVA-NeXT-34B captions for the 118k COCO images. | unspecified on the card | 118k items; 6.1 GB | Images are MS-COCO Flickr photos with mixed per-image Creative Commons licences. |
| `blip3o_long_caption` (`BLIP3o/BLIP3o-Pretrain-Long-Caption`) | Long captions by Qwen2.5-VL-7B for SA-1B, web (CC12M-style) and JourneyDB images. | apache-2.0 | 29.4M items; 1,374 GB | Images come from SA-1B (research licence), web crawls and JourneyDB (Midjourney outputs); the card states only Apache-2.0. |
| `text_to_image_2m` (`jackyhate/text-to-image-2M`) | Synthetic images (Flux and Midjourney style) with the prompt that produced them, as caption. | mit | 2M items; 418 GB | Images are model outputs; FLUX.1-dev outputs fall under the FLUX.1-dev non-commercial licence. |
| `commoncatalog_cc_by` (`common-canvas/commoncatalog-cc-by`) | Flickr photos under Creative Commons Attribution licences, with BLIP-2 captions. | cc-by-4.0 | 14.6M items; 21,465 GB | Each photo carries its own Creative Commons Attribution licence (licensename/licenseurl columns); BLIP-2 captions are short and generic. |
| `coco_captions` (`jxie/coco_captions`) | Human captions for MS-COCO (five per image; the longest is used). | unspecified on the card | 113k items; 19 GB | COCO annotations are CC-BY-4.0; the Flickr photos carry mixed per-image Creative Commons licences (some NonCommercial/NoDerivatives). |
| `recap_coco_30k` (`UCSC-VLAA/Recap-COCO-30K`) | LLaMA-3-powered recaptions of 30k COCO images. | cc-by-4.0 | 30.5k items; 1.6 GB | COCO photos carry mixed per-image Creative Commons licences. |
| `textcaps` (`lmms-lab-encoder/TextCaps`) | Human captions that mention the text in Open Images photos (five per image; the longest is used). | unspecified on the card | 22k items; 6.2 GB | TextCaps captions are CC-BY-4.0 upstream; Open Images photos are CC-BY-2.0 per image. |
| `flickr30k` (`lmms-lab-encoder/flickr30k`) | Human captions for Flickr30k (five per image; the longest is used). | unspecified on the card | 31.8k items; 4.4 GB | Flickr30k is distributed for non-commercial research; the Flickr photos have their own owners. |
| `densefusion_1m` (`BAAI/DenseFusion-1M`) | Dense captions fused from several vision experts for 1M web images. | cc-by-4.0 | 1M items; 798 GB | Images are web crawls (LAION-derived) with their owners' rights. |
| `scienceqa_images` (`derek-thomas/ScienceQA`) | Diagrams, maps and photos from school science questions (the questions are not used); rows without an image are skipped. | cc-by-sa-4.0 | 6.22k items; 0.4 GB | ShareAlike; upstream images come from many web sources. |
| `videoufo_frames` (`WenhaoWang/VideoUFO`) | The middle frame of each VideoUFO clip with the clip's detailed caption. | cc-by-4.0 (card) | 1.09M items; 46 GB |  |
| `pixelprose` (`pixelprose/pixelprose-shards`) | PixelProse: 16.9M web images (CC12M, DataComp CommonPool, RedCaps) with dense Gemini captions; duplicate shards are left out. | cc-by-4.0 (card; captions by Gemini-1.0-Pro-Vision; the images are web crawls with their owners' rights) | 16M items; 2,521 GB | Captions shorter than 50 characters are advised against by the authors; the reader keeps those of 80 or more. Images keep their owners' rights. |
| `pd12m` (`Spawning/pd12m-full`) | PD12M: 12M public-domain and CC0 images (Wikimedia Commons, Smithsonian, museums) at original resolution with model-written captions; shards 00000-00154 are missing from this mirror. | cdla-permissive-2.0 (card; every image is public domain or CC0) | 11.6M items; 29,248 GB | 29 TB of original-resolution JPEG (about 2.3 MB each); the images are cut to 1024 px. |
| `recap_datacomp_12m` (`cornuHGF/recap-datacomp-12m-wds`) | A hosted 13.6M-image part of Recap-DataComp-1B: 256 px DataComp web images with LLaVA-1.5-LLaMA3-8B recaptions. | none on the card (Recap-DataComp-1B: cc-by-4.0) | 13.6M items; 350 GB | Low-resolution (256 px) images; DataComp web images keep their owners' rights. |
| `cc12m_recaptioned` (`ooutlierr/cc12m-recaptioned`) | Conceptual 12M images with a synthetic VLM caption of 100-600 characters next to the original alt-text. | none on the card (Conceptual 12M terms: free for any purpose) | 11M items; 1,182 GB | Images keep their owners' rights; the VLM that wrote the captions is not named on the card. |
| `wit_base` (`wikimedia/wit_base`) | Wikipedia image captions (the caption shown in the article, in over 100 languages) for 300 px Commons thumbnails; the longest caption is used. | cc-by-sa-4.0 (card; images are Wikimedia Commons files with their own licences) | 6.48M items; 308 GB | The longest caption can be in any language (no language filter) and is often only a title of two to five words. |
| `met_openaccess` (`metmuseum/openaccess`) | The Metropolitan Museum's open-access object photographs with the object title as caption. | cc0-1.0 (card; public-domain objects only) | 260k items; 394 GB | Titles are short; the catalogue fields (artist, date, medium, culture) are not assembled into a description. |
| `roco_radiology` (`eltorio/ROCO-radiology`) | ROCO: radiology figures with their captions. | mit (card; figures from PubMed Central open access with per-image Creative Commons licences) | 65.4k items; 13 GB |  |
| `scimmir` (`m-a-p/SciMMIR`) | SciMMIR: 498k arXiv figure and table images with their captions. | none on the card (arXiv figures and tables; per-paper licences) | 498k items; 60 GB |  |
| `wikiart_recaption` (`AterMors/wikiart_recaption`) | 81k WikiArt paintings with synthetic recaptions (the longest of the recaptions is used). | apache-2.0 (card; the artworks' copyright varies, many are public domain) | 81.4k items; 52 GB |  |
| `pmc_oa` (`axiong/pmc_oa`) | PMC-OA: biomedical figures with sub-captions (images in one zip, captions in a jsonl). | none on the card (PubMed Central open-access subset: per-article licences, mostly CC-BY and CC0) | 1.65M items; 23 GB |  |
| `sharegpt4o_image_t2i` (`FreedomIntelligence/ShareGPT-4o-Image`) | ShareGPT-4o-Image text-to-image part: 45.7k 1024 px GPT-4o images with the prompt that made each. | apache-2.0 (card; the images are GPT-4o outputs: OpenAI's terms apply) | 45.7k items; 90 GB |  |
| `pixmo_cap_images` (`anthracite-org/pixmo-cap-images`) | PixMo-Cap with its images: 707k photos with dense 100-300 word descriptions written from annotators' speech (the official repository only lists image URLs). | none on the card (PixMo-Cap is ODC-By-1.0; this repack of its images carries no licence text) | 708k items; 372 GB | A third-party repack; the captions were cleaned up by a language model from human speech. |
| `art_museums_pd_440k` (`Mitsua/art-museums-pd-440k`) | 440k museum object photographs (Art Institute of Chicago, Cleveland, Rijksmuseum and others) with their catalogue keywords as caption. | CC0 and public domain per object (the records' License field) | 440k items; 90 GB | Captions are keyword lists built from catalogue fields, not natural sentences. |
| `swiss_ai_smithsonian` (`swiss-ai/smithsonian`) | Smithsonian collection objects at high resolution with catalogue-grounded descriptions. | CC0 / public domain (agency collections; the card) | 41.8k items; 414 GB | Full-resolution images are cut to 1024 px. |
| `swiss_ai_nasa` (`swiss-ai/nasa`) | NASA images with descriptions. | CC0 / public domain (agency collections; the card) | 37k items; 230 GB | Full-resolution images are cut to 1024 px. |
| `megalith_10m` (`drawthingsai/megalith-10m`) | Megalith-10M: 9.1M openly licensed Flickr photographs (no captions are read). | CC0 (the photographers' dedication; the photos are Flickr images under open licences) | 9.1M items; 1,863 GB | Images only: image identity episodes, no captions (the long captions live in a separate repository). |
| `medtrinity_demo` (`UCSC-VLAA/MedTrinity-25M`) (gated) | MedTrinity-25M demo subset (the card's `25M_demo` config): 161,630 medical images (ten imaging kinds) with multi-granular captions (lesion type, modality, regions of interest, relations). | none on the card (the images come from about 30 public medical datasets, most CC BY, CC0 or CC BY-NC; the card lists each; the captions are machine-written) | 162k items; 8.3 GB | The 18M-image `25M_accessible` shards (1.6 TB of `.tar.zst`, captions in a metadata file inside each archive) are not read: they need a zstd tar reader and a join that holds the images until the metadata arrives. |

#### Questions about images (image_description)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `finevision` (`HuggingFaceM4/FineVision`) | FineVision: 24M visual conversations over 17.3M images from about 200 public sources (documents, charts, tables, OCR, diagrams, general VQA, captions, GUI, maths), de-duplicated and rated by a judge model; rows with a relevance or image-correspondence rating below 3 are dropped, rows with several images and the text-only subsets are skipped. | mixed (each of the 185 sub-datasets keeps its own licence; the prompts are CC-BY-4.0; some subsets hold GPT-4V, Gemini or Claude outputs and non-commercial sets) | 14M items; 4,637 GB | Subsumes the Cauldron, LLaVA-OneVision-Data, Cambrian, CoSyn, Docmatix, ALLaVA, LVIS-Instruct4V and ShareGPT4V sets, which are therefore not listed separately. Several large subsets hold multi-page documents and are mostly skipped. |
| `llava_next_data` (`lmms-lab/LLaVA-NeXT-Data`) | LLaVA-NeXT instruction data: 779k visual conversations (captions, VQA, OCR, charts). | mixed (LLaVA-NeXT release: ShareGPT4V, VQA and OCR sets with their own licences; GPT-4V outputs inside) | 779k items; 92 GB | Overlaps FineVision, which holds most of these sets. |
| `sharegpt4o_image_questions` (`OpenGVLab/ShareGPT-4o`) (gated) | ShareGPT-4o image conversations: 57k images, each with a question and a long GPT-4o answer. The repository's other parts (about 4.8 TB of audio-pretraining zips of GigaSpeech, LibriSpeech, WenetSpeech and others, and the audio benchmark folder) are not read. | mit (card; the answers are GPT-4o outputs: OpenAI's terms apply; image and video copyrights stay with their creators) | 57.4k items; 6.5 GB | Read from `images.zip` joined to `image_conversations/gpt-4o.jsonl` (a zip the archive reader opens by random access). |
| `infinity_mm_stage1` (`BAAI/Infinity-MM`) (gated) | Infinity-MM stage 1: 10M image-caption pairs, each a request to describe the image and a long caption from the publishers' pipeline. | cc-by-sa-4.0 (card; the images come from many upstream datasets whose own licences apply) | 10M items; 6,500 GB | WebDataset tars of `N.jpg` and `N.json`. The card says 10M samples. |
| `infinity_mm_synthetic` (`BAAI/Infinity-MM`) (gated) | Infinity-MM stage 4 synthetic data (two parts, 640k and 142k conversations): visual instruction conversations the publishers generated with open models. | cc-by-sa-4.0 (card) | 782k items; 147 GB | |
| `infinity_mm_zh` (`BAAI/Infinity-MM`) (gated) | Infinity-MM stage 4 Chinese data: visual instruction conversations translated into Chinese and filtered (many ask for bounding boxes). | cc-by-sa-4.0 (card) | about 400k items; 73 GB | The item count is an estimate from the size; the card gives none. |
| `svit_detail` (`BAAI/SVIT`) (gated) | SVIT detailed descriptions: one long GPT-4 description of each of 106k Visual Genome images. | cc-by-4.0 (card; also OpenAI's terms; Visual Genome keeps its licence) | 106k items; 15 GB | The four SVIT sources read the same 3.0M-row LLaVA-format table (`format/llava-v1.5/SVIT.zip`, a zipped JSON array decoded one row at a time) and the same two image zips, each keeping the rows of one kind (`options.table.prefix`); an image's rows of that kind merge into one conversation of at most 8 turns. The same picture therefore appears once in each of the four. |
| `svit_conversation` (`BAAI/SVIT`) (gated) | SVIT conversations: multi-turn GPT-4 question-and-answer chats about Visual Genome images, built from the annotations. | cc-by-4.0 (card; as above) | 318k conversations over up to 108k images | |
| `svit_complex` (`BAAI/SVIT`) (gated) | SVIT complex reasoning: GPT-4 questions that need reasoning about a Visual Genome image, with answers. | cc-by-4.0 (card; as above) | 1.56M conversations over up to 108k images | |
| `svit_referring` (`BAAI/SVIT`) (gated) | SVIT referring questions: GPT-4 questions about named regions of a Visual Genome image; answers give bounding boxes. | cc-by-4.0 (card; as above) | 1.01M conversations over up to 108k images | |

#### Instructed image edits and edit sessions (image_edit)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `magicbrush` (`osunlp/MagicBrush`) | Human-made multi-turn edit sessions (DALL-E 2 inpainting guided by crowd workers) of COCO images; each turn edits the previous turn's result, so a session is read as one multi-turn episode. Train and dev are read; the test split is the benchmark and is left out. | CC-BY-4.0 | 4.5k items; 27 GB | Source images come from MS-COCO, whose Flickr photos carry mixed per-image Creative Commons licences (some NonCommercial/NoDerivatives); the card does not address this. |
| `ultraedit` (`BleachNick/UltraEdit`) | 3M free-form edits generated with Stable Diffusion 3 inpainting from LLM-written instructions. | cc-by-4.0 | 3.03M items; 1,555 GB | Source images come from COCO, Flickr and Open Images with their own licences; edited images are Stable Diffusion 3 outputs. |
| `hq_edit` (`UCSC-VLAA/HQ-Edit`) | About 100k edits whose images were generated with DALL-E 3 and instructions written with GPT-4V. | cc-by-nc-4.0 | 98.7k items; 319 GB | Images and instructions are OpenAI model outputs; the card licence is NonCommercial. |
| `instructpix2pix` (`timbrooks/instructpix2pix-clip-filtered`) | The CLIP-filtered InstructPix2Pix set: GPT-3 instructions with Prompt-to-Prompt edits by Stable Diffusion. | unspecified on the card | 313k items; 130 GB | The InstructPix2Pix release is CC-BY-4.0 for the text; images are Stable Diffusion outputs (CreativeML Open RAIL-M). |
| `omniedit` (`TIGER-Lab/OmniEdit-Filtered-1.2M`) | 1.2M edits from seven specialist editing models, filtered by GPT-4o quality scores. | mit | 1.2M items; 2,977 GB | Source images come from several web datasets; 5 GB shards of full-resolution PNG. |
| `anyedit` (`Bin1117/anyedit-split`) | Instruction edits across 25 types; rows that need a depth/segment guide image are skipped. | cc-by-4.0 | 2.17M items; 207 GB | A third-party re-split of AnyEdit by edit type (row totals match); the weather, season and time-of-day edits that the old shard list dropped are now included. |
| `sharegpt4o_image_edit` (`FreedomIntelligence/ShareGPT-4o-Image`) | ShareGPT-4o-Image instruction edits: 46.5k pairs of an input image, an instruction and the GPT-4o edit (1024 px). | apache-2.0 (card; the images are GPT-4o outputs: OpenAI's terms apply) | 46.5k items; 170 GB | Both images of a pair are in the same tar stream; the reader holds a pair's first image until the second arrives. |
| `ultraedit_region` (`BleachNick/UltraEdit_Region_Based_100k`) | UltraEdit region-based edits: 108k object-level edits ("replace the cows with flamingos") with the edited region's mask. | cc-by-4.0 (card; source photos from COCO, Flickr and Open Images keep their own licences, edits are Stable Diffusion 3 outputs) | 108k items; 46 GB | The edit mask is not read. |
| `pipe` (`paint-by-inpaint/PIPE`) | Paint-by-Inpaint: 888k object additions; the target is the real photo and the source is the same photo with the object inpainted out. | cc-by-4.0 (card; COCO and Open Images photos keep their own licences) | 889k items; 80 GB | Only additions of an object; the instruction text is model-written. |
| `bytemorph_demo` (`Boese0601/ByteMorph-6M-Demo`) | ByteMorph: 780k edits of motion, camera and pose taken from two frames of one generated video. | cc0-1.0 (card; frames of videos made by the Seaweed video model) | 780k items; 45 GB | Neighbouring rows are frames of the same clip, so the clip is the split family. |
| `nhr_edit` (`iitolstykh/NHR-Edit`) | NHR-Edit part 1: 358k synthetic edits (forward, inverse and composite) of generated photos, split by the seed of the source image. | apache-2.0 (card; fully synthetic text-to-image plus editing-model images) | 358k items; 1,591 GB | About 2.2 MB per pair; the images are cut to 1024 px. |
| `nhr_edit_part2` (`iitolstykh/NHR-Edit-part2`) | NHR-Edit part 2: 362k more synthetic edits, same layout as part 1 (seeds are shared across both parts). | apache-2.0 (card; fully synthetic) | 362k items; 801 GB |  |
| `gpt_edit_simpler` (`UCSC-VLAA/gpt-edit-simpler`) | GPT-Image-Edit-1.5M in parquet: 1.55M HQ-Edit, OmniEdit and UltraEdit inputs re-edited by GPT-Image-1, with the original, rewritten and complex instruction variants. | not stated on this repository (the original GPT-Image-Edit-1.5M is cc-by-4.0; the outputs are GPT-Image-1 images, so OpenAI's terms apply) | 1.55M items; 4,936 GB | The inputs are the same source images as HQ-Edit, OmniEdit and UltraEdit, with different edited outputs; about 3 MB per row. |
| `editreward_data` (`TIGER-Lab/EditReward-Data`) | EditReward-Data: 171k source images, each with two edits from different editing models and human scores; the higher rated edit is kept (best of two). | cc-by-nc-4.0 (card) | 171k items; 87 GB | The rejected edit and the scores are not used as a preference signal yet (proposal in the docs). |
| `flux_kontext_ipa` (`obvious-research/flux-kontext-ipa-dataset`) | 10.8k instruction edits made with FLUX.1 Kontext. | not stated on the card (FLUX.1 Kontext dev outputs: its non-commercial terms apply) | 10.8k items; 15 GB |  |
| `maskedit_10k` (`ReyChiaro/MaskEdit-10k`) | MaskEdit-10k: 11k mask-guided edits of scenes and of Chinese and English infographics. | other (the repository's LICENSE and LICENSE_NOTICE.md) | 9.61k items; 44 GB | The mask is not read; the test splits are left out. |
| `escher_aurora_kubric` (`Image-editing/escher-aurora-kubric`) | Escher repack of AURORA Kubric: rendered 3D scenes with one object, camera or action change; each row also carries a written reasoning trace that is not read. | mit (card) | 93k items; 53 GB | A third-party repack of research datasets; the original licences of the underlying images apply. |
| `escher_aurora_ag` (`Image-editing/escher-aurora-ag`) | Escher repack of AURORA Action Genome: action edits between frames of real videos; each row also carries a written reasoning trace that is not read. | mit (card) | 8.31k items; 2.2 GB | A third-party repack of research datasets; the original licences of the underlying images apply. |
| `escher_human_edit` (`Image-editing/escher-human-edit`) | Escher repack of human-made edits; each row also carries a written reasoning trace that is not read. | mit (card) | 4.89k items; 0.9 GB | A third-party repack of research datasets; the original licences of the underlying images apply. |
| `escher_vismin` (`Image-editing/escher-vismin`) | Escher repack of VisMin: minimal object, attribute, count and position edits; each row also carries a written reasoning trace that is not read. | mit (card) | 13.7k items; 10 GB | A third-party repack of research datasets; the original licences of the underlying images apply. |
| `x2edit` (`OPPOer/X2Edit-Dataset`) | X2Edit: about 5 million instruction edits at 1024 px made by Bagel, GPT-4o, FLUX Kontext, Step1X-Edit, LaMa (object removal), OmniConsistency and TextFlux; the two portrait sets of faces are left out. | apache-2.0 (card; each editing model that made a subset has its own output terms) | 4.8M items; 4,880 GB | Sample numbers restart at zero in every archive, so the archive path is part of each sample's ID. Instructions are English. |
| `seed_data_edit_multi_turn` (`AILab-CVC/SEED-Data-Edit-Part2-3`) | SEED-Data-Edit part 3: 21k human-annotated editing sessions of up to five edits in a row, each edit applied to the previous result. | cc-by-nc-4.0 (card, non-commercial research; the photos come from Unsplash, SAM and JourneyDB and stay with their owners) | 21.4k items; 47 GB | The photos' paths in the table lack the archive's data/ prefix and their case differs; the reader matches by path tail without case. |
| `seed_data_edit_real` (`AILab-CVC/SEED-Data-Edit-Part2-3`) | SEED-Data-Edit part 2: 51.6k real retouching requests answered by people (Photoshop forums), with the request in English and Chinese. | cc-by-nc-4.0 (card; images collected from photo-editing request forums, copyright stays with their owners) | 51.6k items; 2.5 GB |  |
| `imgedit_content_memory` (`sysuyy/ImgEdit`) | ImgEdit multi-turn, content memory: 30,861 sessions of two chained edits (the user first states a rule for all later edits (that rule is not read), then asks for two edits that refer to the objects before). | apache-2.0 (card; the source photos are web images whose licences the card does not address) | 30.9k items; 270 GB | Each session has a third edit that starts again from the first result, not the second; it is left out so every edit applies to the one before. The tar is one 10 GiB-piece archive per subset; edits are machine-written from detections. |
| `imgedit_content_understanding` (`sysuyy/ImgEdit`) | ImgEdit multi-turn, content understanding: 42,139 sessions of two chained edits (three edits that refer back with pronouns (it, them, her)). | apache-2.0 (card; the source photos are web images whose licences the card does not address) | 42.1k items; 560 GB | Each session has a third edit that starts again from the first result, not the second; it is left out so every edit applies to the one before. The tar is one 10 GiB-piece archive per subset; edits are machine-written from detections. |
| `imgedit_version_backtracking` (`sysuyy/ImgEdit`) | ImgEdit multi-turn, version backtracking: 42,023 sessions of two chained edits (edits where the third request withdraws the second and redoes the first). | apache-2.0 (card; the source photos are web images whose licences the card does not address) | 42k items; 560 GB | Each session has a third edit that starts again from the first result, not the second; it is left out so every edit applies to the one before. The tar is one 10 GiB-piece archive per subset; edits are machine-written from detections. |
| `taesiri_image_editing_request` (`taesiri/ImageEditingRequestV1`) | 3.9k real photo-editing requests, each with the original photo, a person's edited version and the request as written (typos kept). | none on the card (requests and images from an online photo-editing request forum: the posters keep their rights) | 3.06k items; 0.3 GB |  |
| `editworld_reasoning` (`tyfeld/editworld-reasoning`) | EditWorld reasoning: 7.7k edits whose request is a question about the world ("what appears when the clock strikes midnight?"), with the edited picture. | none on the card | 7.74k items; 1.3 GB | The long reasoning text of each row (output_text) is not read. |
| `multiedit` (`inclusionAI/MultiEdit`) (gated) | MultiEdit: 106.5k hard instruction edits in six groups (objects, people, text in posters, interface screenshots, new viewpoints, 38 artistic styles). | apache-2.0 (card; the edits are GPT-4o and GPT-Image-1 outputs: OpenAI's terms apply) | 107k items; 45 GB | Read on 2026-10-23 from the first archive: each row of `multiedit/train.jsonl` names an original and an edited picture inside `multiedit/train.tar.gz` (one 44 GB stream, so every shard reads it and keeps its share). The 1.1k-sample benchmark (`multiedit/test.*`) is not read. |
| `pico_banana_multi_turn` (`https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb`) | Pico-Banana-400K multi-turn sessions: 2-6 successive edits of one photo, each applied to the previous result, with the request of every turn; fetched from the publisher's web site. | cc-by-nc-nd-4.0 (the dataset's README); the source photos are Open Images (CC BY 2.0) | 77.2k items; 540 GB | Read from the publisher's CDN (originals from Open Images' copies). A session loses its later turns after a missing picture. |
| `pico_banana_single` (`https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb`) | Pico-Banana-400K single edits: 35 kinds of edit of a photo with a request written by a model; fetched from the publisher's web site. | cc-by-nc-nd-4.0 (the dataset's README); the source photos are Open Images (CC BY 2.0) | 258k items; 379 GB | Read from the publisher's CDN (originals from Open Images' copies). |

#### Judging edits (edit_judging)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `editreward_judging` (`TIGER-Lab/EditReward-Data`) | EditReward-Data read as judging: the source image, the request and two edits by different editing models are shown; the verdict lists each edit's human scores (instruction following and visual quality, each 1-4, and their sum) and which edit people preferred. | cc-by-nc-4.0 (card) | 171k items; 87 GB | Rows that share a key are several pairs of edits of one source image and request. |
| `editscore_judging` (`chibifire/editscore-reward-train`) | EditScore reward data (parquet mirror with the pictures inside): a judge model is shown a source image, an edit and a rubric, and answers with a short reasoning and two 0-25 scores (how well the edit succeeded, how little it over-edited). Rows that judge one picture alone are skipped. | apache-2.0 (card) | 44k items; 117 GB | The scores and reasoning are a vision-language model's, not people's. The row count of rows with an edit is an estimate. |
| `pico_banana_preference` (`https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb`) | Pico-Banana-400K preference rows: one accepted edit and one to three rejected edits of a photo, with the (model-written) request; the pictures are fetched from the publisher's web site and the original photos from Open Images' copies of them. | cc-by-nc-nd-4.0 (the dataset's README); the source photos are Open Images (CC BY 2.0) | 56.5k items; 130 GB | The rows are read from the publisher's CDN, not the Hub, and the original photos from Open Images' copies. The accepted edit passed a model judge; there is no numeric score. |

#### Edits written as a plan or tool call (edit_plan)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `escher_human_edit_plan` (`Image-editing/escher-human-edit`) | Escher repack of human-made edits, read for its written reasoning: a numbered plan (what is in the picture, which object, where, what the edit changes) comes before the edited picture. | mit (card) | 4.89k items; 0.9 GB | The plans are machine-written and sometimes contradict the request. The repack's cards say MIT but the underlying datasets keep their licences. |
| `escher_vismin_plan` (`Image-editing/escher-vismin`) | Escher repack of VisMin minimal object, attribute, count and position edits, read for its written reasoning: a numbered plan (what is in the picture, which object, where, what the edit changes) comes before the edited picture. | mit (card) | 13.7k items; 10 GB | The plans are machine-written and sometimes contradict the request. The repack's cards say MIT but the underlying datasets keep their licences. |
| `escher_aurora_kubric_plan` (`Image-editing/escher-aurora-kubric`) | Escher repack of AURORA Kubric rendered scenes, read for its written reasoning: a numbered plan (what is in the picture, which object, where, what the edit changes) comes before the edited picture. | mit (card) | 93k items; 53 GB | The plans are machine-written and sometimes contradict the request. The repack's cards say MIT but the underlying datasets keep their licences. |
| `escher_aurora_ag_plan` (`Image-editing/escher-aurora-ag`) | Escher repack of AURORA Action Genome action edits, read for its written reasoning: a numbered plan (what is in the picture, which object, where, what the edit changes) comes before the edited picture. | mit (card) | 8.31k items; 2.2 GB | The plans are machine-written and sometimes contradict the request. The repack's cards say MIT but the underlying datasets keep their licences. |
| `escher_ss2_plan` (`Image-editing/escher-ss2`) | Escher repack of SS2 (Something-Something) motion edits, read for its written reasoning: a numbered plan (what is in the picture, which object, where, what the edit changes) comes before the edited picture. | mit (card) | 32.6k items; 6.9 GB | The plans are machine-written and sometimes contradict the request. The repack's cards say MIT but the underlying datasets keep their licences. |
| `escher_magicbrush_plan` (`Image-editing/escher-magicbrush`) | Escher repack of MagicBrush human-request edits, read for its written reasoning: a numbered plan (what is in the picture, which object, where, what the edit changes) comes before the edited picture. | mit (card) | 8.8k items; 1.9 GB | The plans are machine-written and sometimes contradict the request. The repack's cards say MIT but the underlying datasets keep their licences. |
| `editworld_reasoning_plan` (`tyfeld/editworld-reasoning`) | EditWorld reasoning: the request is a concept ("growing tall and strong"), the plan is a paragraph that works out what the picture should show, then the edited picture. | none on the card | 7.74k items; 1.3 GB |  |
| `jarvisart_plan` (`JarvisArt/MMArt-PPR10k`) | JarvisArt: 4,055 portrait photos retouched by experts; the request, a step-by-step retoucher's reasoning (revised version) and the exact Lightroom develop settings, written as a tool call, come before the retouched photo. | apache-2.0 (card) | 4.05k items; 3.3 GB | Loose files: about 6 Hub requests per folder. Each folder has three request lengths; the middle one is read. The reasoning is model-written from the experts' settings. |

#### Speech (transcription, verbalization, speech_translation)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `librispeech_asr` (`openslr/librispeech_asr`) | 960 hours of read English audiobooks (LibriVox) with transcripts in upper case. | CC-BY-4.0 | 281k items; 960 h; 60 GB |  |
| `libritts_r` (`mythicinfinity/libritts_r`) | Restored LibriTTS: read English with the written text and its normalized spoken form. | CC-BY-4.0 | 355k items; 585 h; 95 GB |  |
| `mls_english` (`parler-tts/mls_eng`) | Multilingual LibriSpeech English: 44.5k hours of read audiobooks (LibriVox), 10-20 s segments. | cc-by-4.0 | 10.8M items; 44.5k h; 705 GB | Transcripts are lower case without punctuation. |
| `mls_multilingual` (`facebook/multilingual_librispeech`) | Multilingual LibriSpeech in seven further languages (read audiobooks). | cc-by-4.0 | 1.45M items; 6.03k h; 94 GB | Transcripts are lower case without punctuation. |
| `peoples_speech` (`MLCommons/peoples_speech`) | English speech from the internet (the clean, well-aligned subset); family is the source recording. | mixed Creative Commons (cc-by-2.0 to cc-by-4.0, cc-by-sa-3.0, cc-by-sa-4.0) | 1.5M items; 5.97k h; 392 GB | Share-alike and attribution terms apply per recording; transcripts are force-aligned and sometimes noisy. |
| `voxpopuli` (`facebook/voxpopuli`) | European Parliament speech in 14 languages with transcripts. | cc0-1.0 (European Parliament recordings; reuse terms of the Parliament also apply) | 587k items; 1.8k h; 314 GB |  |
| `common_voice_17` (`fixie-ai/common_voice_17_0`) | Volunteers reading sentences, many accents and languages; family is the speaker. | CC0-1.0 (Mozilla Common Voice; the card states none) | 3.61M items; 5k h; 132 GB | Several clips per speaker and sentence; the mirror's card has no licence. |
| `gigaspeech` (`speechcolab/gigaspeech`) (gated) | 10,000 hours of audiobooks, podcasts and YouTube with transcripts. | apache-2.0 (card; audio carries its sources' own terms) | 8.3M items; 10k h; 1,152 GB | Gated: the Hub account must accept the terms. Podcast and YouTube sources have their own terms. Transcripts hold punctuation tags that the reader converts and silence/noise tags that make it skip the segment. |
| `emilia` (`amphion/Emilia-Dataset`) (gated) | Emilia, the original half of the Emilia-Large corpus (English, German, French, Chinese, Japanese, Korean); its YODAS half is read from an ungated mirror (emilia_yodas). | cc-by-4.0 on the card (Emilia-YODAS); the original Emilia part is CC-BY-NC-4.0 under the gate's terms | 36.6M items; 102k h; 2,610 GB | Gated: the Hub account must accept the terms. Transcripts are Whisper-pipeline output, not human. |
| `yodas_granary_en` (`espnet/yodas-granary`) | English YouTube speech (Creative Commons videos) with punctuated Whisper-large-v3 transcripts, filtered for quality. | cc-by-3.0 | 40.8M items; 104k h; 11,264 GB | Pseudo-labelled by Whisper and punctuation-restored by Qwen2.5-7B, not human transcripts; per-video Creative Commons terms. |
| `yodas_granary_ast` (`espnet/yodas-granary`) | Non-English YouTube speech with its transcript and an English translation by EuroLLM-9B, quality filtered. | cc-by-3.0 | 23.2M items; 58.6k h; 9,727 GB | Transcripts and translations are model output, not human; per-video Creative Commons terms. |
| `covost2` (`fixie-ai/covost2`) | Common Voice speech in 21 languages with human English translations of the sentence. | unspecified on this mirror's card (CoVoST 2 release: CC0-1.0; facebook/covost2's card says CC-BY-NC-4.0) | 653k items; 1.5k h; 25 GB |  |
| `fleurs` (`google/fleurs`) | The same FLoRes sentence read in about 100 languages: each language's reading paired with the English reading. | CC-BY-4.0 | 140k items; 440 h |  |
| `hifi_tts` (`MikhailT/hifi-tts`) | Ten studio-quality speakers reading audiobooks (44.1 kHz), 292 hours. | cc-by-4.0 | 322k items; 292 h; 42 GB |  |
| `expresso` (`ylacombe/expresso`) | Four speakers reading in several styles; only the neutral styles are used (the style is not part of the request). | cc-by-nc-4.0 | 6k items; 20 h; 5.8 GB |  |
| `lj_speech` (`MikhailT/lj-speech`) | One speaker reading seven non-fiction books, 24 hours. | cc-by-4.0 on this mirror's card (LJ Speech is public domain upstream) | 13.1k items; 24 h; 3.8 GB |  |
| `yodas_granary_asr_nonen` (`espnet/yodas-granary`) | YODAS-Granary speech in 22 languages other than English (Russian, Spanish, Portuguese, French, German, Italian, Dutch, Polish, Ukrainian, Hungarian and more) with Whisper transcripts restored with punctuation. | cc-by-3.0 (card; YouTube videos under Creative Commons, per-video terms) | 4.75M items; 12k h; 478 GB | Pseudo-labelled by Whisper and punctuation-restored by an LLM, not human transcripts. Disjoint from the translation split already read. |
| `yodas_granary_ast_more` (`espnet/yodas-granary`) | The eight languages of YODAS-Granary's translation split that the first entry left out (Bulgarian, Danish, Greek, Estonian, Croatian, Lithuanian, Latvian, Slovak). | cc-by-3.0 (card; YouTube videos under Creative Commons, per-video terms) | 38.3k items; 97 h; 9.0 GB |  |
| `peoples_speech_dirty` (`MLCommons/peoples_speech`) | People's Speech, the noisier configuration: 5.5M more English clips whose force-aligned transcripts are sometimes off. | cc-by-2.0, 2.5, 3.0 and 4.0 per recording (the mixed attribution licences of the People's Speech 'dirty' configuration) | 5.48M items; 22k h; 1,490 GB | A transcript can be wrong for part of its 15 seconds; the first clips of this and the clean configuration share ids, which are skipped when met twice in a shard. |
| `peoples_speech_sharealike` (`MLCommons/peoples_speech`) | People's Speech recordings under share-alike licences (clean and dirty halves). | cc-by-sa-3.0 and cc-by-sa-4.0 per recording (share-alike) | 805k items; 3.2k h; 209 GB | Share-alike terms apply to what is made from these recordings. |
| `libriheavy` (`mythicinfinity/libriheavy`) | Libriheavy: 50,800 hours of English audiobook reading with the book's own cased and punctuated text (6,700 speakers). | apache-2.0 (card; the LibriVox recordings are public domain) | 11.2M items; 50.8k h; 1,346 GB | Partly the same audiobooks as the Multilingual LibriSpeech English entry. The medium and small configurations are subsets of large and are not read. |
| `wenetspeech` (`zxsddcs/wenetspeech`) | WenetSpeech (L): 10,000 hours of Mandarin from YouTube and podcasts with automatically derived transcripts. | mirror's card is empty; WenetSpeech is CC-BY-4.0 (project README) | 13.9M items; 10k h; 1,151 GB | Transcripts come from OCR and speech recognition with confidence of 0.95 or more, not from people. |
| `fleurs_asr` (`google/fleurs`) | FLEURS read speech in 102 languages with original-case transcripts (the first FLEURS entry only pairs each language with its English reading). | cc-by-4.0 (card) | 272k items; 1k h; 224 GB | The same sentences as the first FLEURS entry, in every language, with the English reading kept apart. |
| `ami_ihm` (`edinburghcstr/ami`) | AMI meetings, close-talk headset microphones: 80 hours of English meeting speech with disfluencies. | cc-by-4.0 (card) | 109k items; 80 h; 12 GB |  |
| `earnings22` (`distil-whisper/earnings22`) | Earnings-22: 119 hours of company earnings calls in accented English, with verbatim punctuated text. | CC-BY-SA-4.0 (the Earnings-22 corpus; the mirror's card states none) | 57.4k items; 119 h; 24 GB |  |
| `tedlium3` (`sanchit-gandhi/tedlium-data`) | TED-LIUM 3: 452 hours of English TED talks with lower-case transcripts. | CC-BY-NC-ND-3.0 for the TED talks (TED-LIUM 3 licence; the mirror's card is empty) | 268k items; 452 h; 52 GB |  |
| `parlaspeech_hr` (`classla/ParlaSpeech-HR`) | ParlaSpeech: 867,581 Croatian parliamentary speech segments with word-aligned transcripts. | CC-BY-4.0 (CLARIN.SI master record; the Hub card states none) | 868k items; 179 GB | Hours are not on the card. |
| `parlaspeech_rs` (`classla/ParlaSpeech-RS`) | ParlaSpeech: 277,764 Serbian parliamentary speech segments with word-aligned transcripts. | CC-BY-4.0 (CLARIN.SI master record; the Hub card states it) | 278k items; 58 GB | Hours are not on the card. |
| `parlaspeech_cz` (`classla/ParlaSpeech-CZ`) | ParlaSpeech: 711,109 Czech parliamentary speech segments with word-aligned transcripts. | CC-BY-4.0 (CLARIN.SI master record; the Hub card states none) | 711k items; 156 GB | Hours are not on the card. |
| `parlaspeech_pl` (`classla/ParlaSpeech-PL`) | ParlaSpeech: 530,773 Polish parliamentary speech segments with word-aligned transcripts. | CC-BY-4.0 (CLARIN.SI master record; the Hub card states none) | 531k items; 61 GB | Hours are not on the card. |
| `mediaspeech` (`ymoslem/MediaSpeech`) | MediaSpeech: 10 hours each of broadcast speech in Arabic, Spanish, French and Turkish. | cc-by-4.0 (card) | 10k items; 40 h; 2.5 GB |  |
| `mtedx` (`deepdml/mtedx`) | Multilingual TEDx: 765 hours of talks in Spanish, French, Portuguese, Italian, Russian, Greek, Arabic and German with transcripts. | cc-by-nc-nd-4.0 (the TEDx talks' licence) | 412k items; 765 h; 97 GB | The size covers the audio configurations only; the translation-pair configurations repeat the same talks. |
| `cml_tts` (`ylacombe/cml-tts`) | CML-TTS: 3,176 hours of audiobook reading in Dutch, French, German, Italian, Polish, Portuguese and Spanish, with the book's cased and punctuated text. | cc-by-4.0 (card) | 1.2M items; 3.18k h; 561 GB | The text is the book's, aligned to the audio; the rows' agreement scores (how much the text matches a recogniser) are not used. |
| `globe_v3` (`MushanW/GLOBE_V3`) | GLOBE v3: 700k clips of Common Voice speech cleaned and re-cut, with accent, age and gender labels and the original sentence. | cc0-1.0 (card) | 705k items; 161 GB | Re-cut Common Voice audio: some sentences also occur in the Common Voice entry. |
| `magpie_speech_orpheus` (`Aratako/Magpie-Speech-Orpheus-125k`) | Magpie-Speech-Orpheus: 125k sentences read by a speech synthesiser (307 hours), with the recogniser's check of each. | llama3.2 (the Llama 3.2 community licence; speech made by a text-to-speech model) | 125k items; 307 h; 26 GB | Synthetic speech. |
| `primewords_zh` (`urarik/primewords_chinese_corpus_set_1`) | Primewords Chinese Corpus Set 1: 100 hours of Mandarin read speech. | none on the card (OpenSLR SLR47: free for research, Apache-2.0 on the index) | 45.3k items; 100 h; 10 GB |  |
| `covost2_en_to_x` (`fixie-ai/covost2`) | CoVoST 2 the other way: English Common Voice speech translated into 15 languages (Arabic, Catalan, Welsh, German, Estonian, Persian, Indonesian, Japanese, Latvian, Mongolian, Slovenian, Swedish, Tamil, Turkish, Chinese). | unspecified on this card (CoVoST 2 release: CC0-1.0) | 4.34M items; 6.45k h; 185 GB | Every clip appears 15 times with a different translation; the clip, not the row, is the unit that is split. |
| `emilia_yodas` (`TTS-AGI/emilia-yodas`) | The YODAS half of Emilia (English, German, French, Japanese, Korean, Chinese): 114,000 hours of in-the-wild speech, read from an ungated mirror of the gated Emilia repository. | cc-by-4.0 (card of Emilia-YODAS; YouTube videos under Creative Commons) | 41M items; 114k h; 2,136 GB | Transcripts are Whisper-pipeline output, not human. A third-party mirror: file names and sizes match the gated original. |
| `spgispeech` (`kensho/spgispeech`) (gated) | SPGISpeech: 5,000 hours of company earnings calls with fully formatted (cased, punctuated, numerals) transcripts. | Kensho non-commercial research agreement (the card's 'other' licence) | 1.93M items; 5k h; 557 GB | Gated: accept the terms at the repository. Columns come from the card, not from a read of the files. |
| `ksponspeech` (`jp1924/KsponSpeech`) (gated) | KsponSpeech: 969 hours of Korean spontaneous conversation. | none on the card (AI Hub terms: research use, Korean residents) | 620k items; 969 h; 48 GB | Gated: accept the terms at the repository. Columns come from the card, not from a read of the files. |
| `gigaspeech2_th` (`speechcolab/gigaspeech2`) (gated) | GigaSpeech 2 Thai: 10,000 hours of web audio (podcasts, videos) with machine transcripts refined by a second pass (the card's `refine` list, 9.9M segments). | apache-2.0 (card; the access terms limit use to non-commercial research and education, and the audio keeps its owners' copyright) | 9.90M items; 10,000 h; 1,273 GB | Only the training archives (`data/th/train/*.tar.gz`, wavs) and `train_refined.tsv` are read; dev and test are the benchmark and are left out. The table is read whole (about 1.5 GB), so a sandbox needs a few GB of memory for it. |
| `gigaspeech2_id` (`speechcolab/gigaspeech2`) (gated) | GigaSpeech 2 Indonesian: 6,000 hours, refined list of 6.1M segments. | as above | 6.13M items; 6,000 h; 979 GB | As above. |
| `gigaspeech2_vi` (`speechcolab/gigaspeech2`) (gated) | GigaSpeech 2 Vietnamese: 6,000 hours, refined list of 5.0M segments. | as above | 5.00M items; 6,000 h; 824 GB | As above. |
| `wenetspeech4tts` (`Wenetspeech4TTS/WenetSpeech4TTS`) (gated) | WenetSpeech4TTS: Mandarin speech cut from WenetSpeech with refined segment boundaries and transcripts, graded by audio quality (DNSMOS P.808). The Premium, Standard and Basic parts (score 3.6 or more; 7,226 hours) are read; the Rest part (score below 3.6; 5,574 hours of poorer audio) is not. | cc-by-4.0 (card; non-commercial research and education terms; the audio keeps its owners' copyright) | 3.93M items; 7,226 h; 619 GB | Each archive lists its wavs first and its transcripts (`txts/<id>.txt`, an id, a tab and the text) afterwards, so the reader writes an archive (about 8.5 GB) to scratch disk and pairs the files by name (`options.spool`). Overlaps `wenetspeech`, its source, with better boundaries. |

#### Sounds and music (audio_identity, audio_description)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `audioset` (`agkphysics/AudioSet`) | 10-second YouTube clips of 527 sound classes (the class labels are not used). | cc-by-4.0 | 1.77M items; 4.93k h; 2,444 GB | The card licenses the labels; the audio belongs to the YouTube uploaders. |
| `audiocaps` (`OpenSound/AudioCaps`) | 10-second AudioSet clips with human captions (one caption per row; the five captions of a clip are five rows). The validation and test splits are the standard benchmark and are left out. | cc-by-nc-4.0 | 45.2k items; 126 h; 39 GB |  |
| `clotho` (`CLAPv2/Clotho`) | 15-30 second Freesound recordings with five human captions (the captions are not used); a central 20 s excerpt is kept. | unspecified on this mirror's card | 3.84k items; 24 h; 8.1 GB | Upstream Clotho terms were not verified here. |
| `song_describer` (`renumics/song-describer-dataset`) | Full music tracks with human captions (the captions are not used); a central 20 s excerpt is kept. | unspecified on this mirror's card | 746 items; 41 h; 2.6 GB | Audio is Jamendo music under per-track Creative Commons licences (upstream; not verified here). |
| `wavcaps_10s_16k` (`txya900619/wavcaps-10s-16k`) | WavCaps clips of at most 10 s at 16 kHz (BBC sound effects, SoundBible, Freesound, AudioSet without the AudioCaps clips) with ChatGPT-polished captions. | none on the card (WavCaps: CC-BY-4.0 captions, academic use only; clips keep their sources' terms) | 187k items; 480 h; 41 GB | Academic use only per the WavCaps card; BBC sound effects are non-commercial/research; the captions were rewritten by ChatGPT from the original descriptions. |
| `musicbench` (`amaai-lab/MusicBench`) | Ten-second music clips (MusicCaps/AudioSet music with pitch, tempo and volume augmentations) with a ChatGPT-rephrased caption and music-theory prompts; the training json lists them. | cc-by-sa-3.0 (card) | 52.8k items; 147 h; 17 GB | Many clips are augmented copies of about 5k MusicCaps clips; the family is the YouTube id so copies stay in one split. The test json files are left out. |
| `freesound_laion_640k` (`benjamin-paine/freesound-laion-640k`) | Freesound recordings with the uploader's title, description and tags (LAION-Audio-630K); a central 20 s excerpt is kept and the uploader's description is the caption. | cc (per clip: CC0-1.0, CC-BY-3.0/4.0, CC-BY-NC-3.0/4.0, CC-Sampling+; flags in each row) | 506k items; 679 GB | Descriptions are written by the uploaders: often long or boilerplate. The clip licences differ (the row's licence flags say which allow commercial use). |
| `jamendomaxcaps` (`amaai-lab/JamendoMaxCaps`) | 362k full-length Jamendo tracks, instrumentals only; a central 20 s excerpt is kept (the captions live in a side table that is not joined). | cc-by-sa-3.0 (card); tracks keep their per-track Creative Commons licences | 362k items; 1,061 GB | 1.06 TB read for 20 s of each track. Many Jamendo tracks are CC BY-NC-ND. |
| `fma_large_clips` (`benjamin-paine/free-music-archive-large`) | Free Music Archive: 30 s clips of 105k tracks (mp3); a central 20 s excerpt is kept; the family is the artist. | cc (per track; FMA metadata CC-BY-4.0) | 105k items; 869 h; 100 GB | Per-track Creative Commons licences including NonCommercial and NoDerivatives variants (columns allow_commercial_use, allow_derivatives, require_attribution, require_share_alike). |
| `mtg_jamendo` (`rkstgr/mtg-jamendo`) | MTG-Jamendo: about 55k full tracks (opus); a central 20 s excerpt is kept. Mood, genre and instrument tags are not used. | apache-2.0 (card, for the packaging); tracks keep their Jamendo Creative Commons licences | 55k items; 3.7k h; 118 GB | MTG-Jamendo includes NonCommercial and NoDerivatives tracks. The repository's train/val split is random by track, not by artist; this reader freezes its own split per track. |
| `fsd50k` (`philgzl/fsd50k`) | FSD50K sound events (dev and eval), re-encoded to opus; the AudioSet-ontology labels are not used. | cc-by-4.0 (card; per-clip Freesound licences CC0, CC-BY, CC-BY-NC, Sampling+) | 51.2k items; 108 h; 3.0 GB |  |
| `esc50` (`ashraq/esc50`) | 2,000 five-second environmental recordings from Freesound (the 50 class labels are not used). | none on the card (ESC-50 upstream: CC BY-NC 3.0) | 2k items; 2 h; 0.8 GB |  |
| `urbansound8k` (`danavery/urbansound8K`) | 8,732 short urban sound slices from Freesound recordings (the class labels are not used); the family is the source recording. | cc-by-nc-4.0 (card) | 8.73k items; 8 h; 7.0 GB |  |
| `nsynth` (`jg583/NSynth`) | NSynth: single instrument notes of 4 s (acoustic, electronic and synthetic); the family is the instrument. | cc-by-4.0 (card) | 293k items; 326 h; 28 GB | Isolated notes: useful for timbre, not for general sound. |
| `birdset_xeno_canto` (`samuelstevens/BirdSet`) | BirdSet training recordings from Xeno-canto (focal and mixed): bird song and field recordings; a central 20 s excerpt is kept, the family is the recordist. | cc-by-nc-4.0 (BirdSet card; recordings are Xeno-canto uploads under CC BY-NC-SA, BY-NC-ND and BY) | 528k items; 637 GB | Row counts are not verified; recordings are non-commercial. |

#### Video (video_identity, video_description, av_creation, av_edit)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `rekadaily_processed` (`RekaAI/RekaDaily-10k-processed`) | First-person household and workplace clips (nine subsets) with sound and a model-written dense caption. | Apache-2.0 | 4.5M items; 5.4k h; 74,218 GB | First-person recordings of paid collectors; clips contain ambient audio and possibly incidental speech. |
| `finevideo` (`HuggingFaceFV/finevideo`) (gated) | 43,751 YouTube videos with sound and rich scene/activity annotations; clips are cut at the annotated activities. | cc (per-video Creative Commons; see the card) | 131k items; 3.42k h; 673 GB | Gated: the Hub account must accept the terms (removal requests must be honoured). Annotations are Gemini and GPT output, not human. Layout is taken from the card; not sampled. |
| `sharegpt4o_video_captions` (`OpenGVLab/ShareGPT-4o`) (gated) | ShareGPT-4o video captions: 2,184 videos (up to 1080p, 4-28 MB each) with a long GPT-4o description. | mit (card; GPT-4o outputs: OpenAI's terms apply; video copyrights belong to the original creators or platforms, academic research use only) | 2.18k items; 28 GB | Read from `videos.zip` joined to `video_conversations/gpt4o.jsonl`; the clips are cut to 8 s. |
| `vggsound` (`Loie/VGGSound`) | 200k ten-second YouTube clips with sound, labelled by the sound class (for example "dog barking"). | cc-by-4.0 | 199k items; 550 h; 338 GB | Captions are the class label only. The copyright remains with the original uploaders. |
| `openvid_1m` (`Dev-Jahn/OpenVid-1M-wds`) | All 1,019,957 OpenVid-1M clips (silent, 2-30 s; Pexels, Panda-70M, CelebV-HQ) with their LLaVA-written captions, repackaged as tar shards. | cc-by-4.0 (card; the videos keep the terms of Panda-70M, ChronoMagic, CelebV-HQ and Pexels) | 1.02M items; 2k h; 7,535 GB | Silent clips: video description and video identity only. Replaces the earlier read of nkp37/OpenVid-1M's zip parts, which skipped the split parts and the HD folder. |
| `insvie` (`wyh6666/InsViE`) | One million instructed video edits (Pexels clips edited by generative models). | cc-by-4.0 | 1M items; 394 GB | Edits are model-generated; clips have no sound. |
| `videoufo_clips` (`WenhaoWang/VideoUFO`) | 1.09M clips of Creative Commons YouTube videos (3-5 s) with a detailed caption each, grouped by topic. | cc-by-4.0 (card; the source videos are Creative Commons YouTube videos) | 1.09M items; 1.1k h; 865 GB | Captions are model-written. The csv (1.2 GB) is held in memory by each reader. |
| `vidgen_1m` (`Fudan-FUXI/VIDGEN-1M`) | About 1M short YouTube scene clips with sound and a VLM-and-LLM-written caption. | apache-2.0 (card; the videos are YouTube clips under their uploaders' rights) | 1M items; 2.8k h; 2,237 GB | Captions are model output. The caption file is one 490 MB JSON array held in memory by each reader. |
| `vript_short` (`Mutonix/Vript`) | Vript: YouTube Shorts and TikTok scene clips with structured captions (shot type, camera movement, content, scene title). | academic use only (card forbids commercial use and redistribution) | 20k items; 12 GB | Captions are GPT-4V output. Some zip folders are empty (card warning). |
| `vript_long` (`Mutonix/Vript`) | Vript: scene clips of long YouTube videos with structured captions (shot type, camera movement, content, scene title). | academic use only (card forbids commercial use and redistribution) | 400k items; 882 GB | Captions are GPT-4V output. The long-clip zips were not range-sampled; their layout is taken from the card. |
| `openhumanvid_hq` (`Owen777/HQ-OpenHumanVid`) | Quality-filtered clips of people from films, television and online video with a caption each (one csv per tar.gz part). | none (no card in the repository; the OpenHumanVid project releases for research) | 1.2M items; 1.3k h; 837 GB | No licence text in the repository. Each tar.gz is one 70 GB stream. |
| `llava_video_0_30_s_academic` (`lmms-lab/LLaVA-Video-178K`) | LLaVA-Video-178K, 0 30 s academic videos: each whole video with a detailed description, open-ended questions and multiple-choice questions about it, as one conversation. | apache-2.0 on the card, for academic research and education only (GPT-4o-written text; videos keep their own licences) | 45k items; 38 GB | Questions and answers are GPT-4o output. Videos longer than the cut limit are skipped, not cut (a question is about the whole video). |
| `llava_video_0_30_s_youtube` (`lmms-lab/LLaVA-Video-178K`) | LLaVA-Video-178K, 0 30 s youtube videos: each whole video with a detailed description, open-ended questions and multiple-choice questions about it, as one conversation. | apache-2.0 on the card, for academic research and education only (GPT-4o-written text; videos keep their own licences) | 100k items; 100 GB | Questions and answers are GPT-4o output. Videos longer than the cut limit are skipped, not cut (a question is about the whole video). |
| `llava_video_30_60_s_academic` (`lmms-lab/LLaVA-Video-178K`) | LLaVA-Video-178K, 30 60 s academic videos: each whole video with a detailed description, open-ended questions and multiple-choice questions about it, as one conversation. | apache-2.0 on the card, for academic research and education only (GPT-4o-written text; videos keep their own licences) | 20k items; 48 GB | Questions and answers are GPT-4o output. Videos longer than the cut limit are skipped, not cut (a question is about the whole video). |
| `llava_video_30_60_s_youtube` (`lmms-lab/LLaVA-Video-178K`) | LLaVA-Video-178K, 30 60 s youtube videos: each whole video with a detailed description, open-ended questions and multiple-choice questions about it, as one conversation. | apache-2.0 on the card, for academic research and education only (GPT-4o-written text; videos keep their own licences) | 30k items; 67 GB | Questions and answers are GPT-4o output. Videos longer than the cut limit are skipped, not cut (a question is about the whole video). |
| `panda70m_test` (`Qinghao/panda70m_wds`) | The 5,892 hosted test clips of Panda-70M, with sound and a caption each. | none on the card (Panda-70M: research, non-commercial; YouTube videos) | 5.89k items; 13 h; 12 GB | Only the test clips are hosted; the other 3.78M Panda-70M rows are links. |
| `ditto_local` (`oyyggbond/ditto_local`) | Ditto local edits (add, remove or change an object or its look) on 5-second 720p clips: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of Ditto-1M (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 330k items; 935 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `ditto_global_style1` (`oyyggbond/global_style1`) | Ditto whole-clip style changes, set 1: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of Ditto-1M (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 61k items; 160 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `ditto_global_style2` (`oyyggbond/global_style2`) | Ditto whole-clip style changes, set 2: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of Ditto-1M (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 46.9k items; 119 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `ditto_global_freeform1` (`oyyggbond/global_freeform1`) | Ditto free-form whole-clip edits, set 1: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of Ditto-1M (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 119k items; 322 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `ditto_global_freeform2` (`oyyggbond/global_freeform2`) | Ditto free-form whole-clip edits, set 2: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of Ditto-1M (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 139k items; 342 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `ditto_global_freeform3` (`oyyggbond/ditto_global_freeform3`) | Ditto free-form whole-clip edits, the harder set 3: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of Ditto-1M (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 49.4k items; 123 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `reco_add` (`oyyggbond/reco_add`) | ReCo object additions on 5-second 480p clips: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of ReCo-Data (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 104k items; 243 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `reco_remove` (`oyyggbond/reco_remove`) | ReCo object removals: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of ReCo-Data (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 85k items; 204 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `reco_replace` (`oyyggbond/reco_replace`) | ReCo object replacements: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of ReCo-Data (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 126k items; 253 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `reco_style` (`oyyggbond/reco_style`) | ReCo style transfers: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of ReCo-Data (cc-by-nc-sa-4.0), whose terms apply; the edited clips are generated) | 121k items; 408 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `openve3m_background_change` (`oyyggbond/openve3m_background_change`) | OpenVE-3M background changes: each row holds the original clip, the edited clip and the request, as mp4 bytes. | none on this repository's card (a third-party repack of OpenVE-3M (cc-by-nc-4.0), whose terms apply; the edited clips are generated) | 353k items; 323 GB | Silent clips. A repack by an unrelated account: its row counts follow its own card, not the upstream release. |
| `senorita_local_style` (`SENORITADATASET/Senorita`) | Senorita-2M: 348,986 local style transfers (restyle one part of the clip), 4-second 592x336 silent clips, original and edited clip in the same archive. | cc-by-nc-2.0 (the card's header; its README has no other text) | 349k items; 795 h; 135 GB | Edited clips of some subsets have another frame rate than the original; both are cut to their common length. |
| `senorita_obj_removal` (`SENORITADATASET/Senorita`) | Senorita-2M: 414,501 object removals, 4-second 592x336 silent clips, original and edited clip in the same archive. | cc-by-nc-2.0 (the card's header; its README has no other text) | 415k items; 944 h; 158 GB | Edited clips of some subsets have another frame rate than the original; both are cut to their common length. |
| `senorita_obj_addition` (`SENORITADATASET/Senorita`) | Senorita-2M: 532,502 object additions (the removal clips run the other way: the clip without the object is the original), 4-second 592x336 silent clips, original and edited clip in the same archive. | cc-by-nc-2.0 (the card's header; its README has no other text) | 533k items; 1.21k h; 40 GB | Edited clips of some subsets have another frame rate than the original; both are cut to their common length. |
| `senorita_obj_swap` (`SENORITADATASET/Senorita`) | Senorita-2M: 177,336 object replacements (the table names the two clips the other way round), 4-second 592x336 silent clips, original and edited clip in the same archive. | cc-by-nc-2.0 (the card's header; its README has no other text) | 177k items; 404 h; 76 GB | Edited clips of some subsets have another frame rate than the original; both are cut to their common length. |
| `openve_sub` (`SanRiiiii/openve_sub`) | OpenVE-3M in tar shards: whole-clip style changes and local object additions, changes and removals (720p, 2-3 seconds), each with the original clip, the edited clip and the request. | none on this repository's card (a third-party repack of OpenVE-3M, cc-by-nc-4.0; the edited clips are generated) | 1.6M items; 2,850 GB | The count is unverified; the shards are about 21 GB each. Background changes are read from their parquet repack instead. |


### Cloud jobs and full-volume runs

**Pilot jobs.** Six jobs (`configs/cloud/prepare-media-*.json`) each build a few thousand clips of
every source of one media family: one sandbox, the pilot's shape (8 CPU, 32 GiB, one GPU,
12 hours). Nothing launches by itself; each needs a clean commit and `--secret hf-token=FILE`.
A count is capped by what the source holds, and a clip that a filter drops (too long, no
sound, no caption) is replaced by the next one. The test suite checks that every source is in
exactly one pilot job and that the gated sources are in the gated job only.

| Job (`configs/cloud/<name>.json`) | Prepared root | Sources and clip counts |
| --- | --- | --- |
| `prepare-media-edits` | `media-edits` | magicbrush 4,500, ultraedit 10,000, hq_edit 4,000, instructpix2pix 10,000, omniedit 4,000, anyedit 6,000, sharegpt4o_image_edit 1,500, ultraedit_region 2,000, pipe 2,000, bytemorph_demo 2,000, nhr_edit 2,000, nhr_edit_part2 2,000, gpt_edit_simpler 2,000, editreward_data 2,000, flux_kontext_ipa 2,000, maskedit_10k 2,000, escher_aurora_kubric 2,000, escher_aurora_ag 2,000, escher_human_edit 2,000, escher_vismin 2,000, x2edit 2,000, seed_data_edit_multi_turn 1,500, seed_data_edit_real 1,500, taesiri_image_editing_request 2,000, editworld_reasoning 2,000, imgedit_content_memory 1,500, imgedit_content_understanding 1,500, imgedit_version_backtracking 1,500, editreward_judging 1,500, editscore_judging 1,500, pico_banana_preference 1,500, pico_banana_multi_turn 1,500, pico_banana_single 2,000, escher_human_edit_plan 1,500, escher_vismin_plan 1,500, escher_aurora_kubric_plan 1,500, escher_aurora_ag_plan 1,500, escher_ss2_plan 1,500, escher_magicbrush_plan 1,500, editworld_reasoning_plan 1,500, jarvisart_plan 1,500 |
| `prepare-media-gated` | `media-gated` | gigaspeech 10,000, emilia 10,000, finevideo 6,000, multiedit 1,500, spgispeech 2,000, ksponspeech 2,000, sharegpt4o_image_questions 2,000, sharegpt4o_video_captions 600, infinity_mm_stage1 4,000, infinity_mm_synthetic 2,000, infinity_mm_zh 2,000, svit_detail 1,500, svit_conversation 1,500, svit_complex 1,500, svit_referring 1,500, medtrinity_demo 2,000, gigaspeech2_th 2,000, gigaspeech2_id 2,000, gigaspeech2_vi 2,000, wenetspeech4tts 2,000 |
| `prepare-media-images` | `media-images` | docci 2,000, scienceqa_images 5,000, llava_recap_cc3m 15,000, llava_recap_558k 8,000, llava_recap_118k 8,000, blip3o_long_caption 15,000, text_to_image_2m 15,000, commoncatalog_cc_by 8,000, coco_captions 8,000, recap_coco_30k 8,000, textcaps 8,000, flickr30k 8,000, densefusion_1m 8,000, videoufo_frames 2,000, pixelprose 2,000, pd12m 2,000, recap_datacomp_12m 2,000, cc12m_recaptioned 2,000, wit_base 2,000, met_openaccess 2,000, roco_radiology 2,000, scimmir 2,000, wikiart_recaption 2,000, pmc_oa 2,000, sharegpt4o_image_t2i 2,000, finevision 3,000, pixmo_cap_images 2,000, art_museums_pd_440k 2,000, swiss_ai_smithsonian 2,000, swiss_ai_nasa 2,000, megalith_10m 2,000, llava_next_data 3,000 |
| `prepare-media-sound` | `media-sound` | audioset 20,000, audiocaps 10,000, clotho 3,839, song_describer 746, wavcaps_10s_16k 1,500, musicbench 1,500, freesound_laion_640k 1,500, jamendomaxcaps 1,500, fma_large_clips 1,500, mtg_jamendo 1,500, fsd50k 1,500, esc50 1,500, urbansound8k 1,500, nsynth 1,500, birdset_xeno_canto 1,500 |
| `prepare-media-speech` | `media-speech` | librispeech_asr 15,000, libritts_r 8,000, mls_english 15,000, mls_multilingual 10,000, peoples_speech 8,000, voxpopuli 8,000, common_voice_17 8,000, yodas_granary_en 8,000, yodas_granary_ast 15,000, covost2 8,000, fleurs 4,000, hifi_tts 4,000, expresso 2,000, lj_speech 3,000, yodas_granary_asr_nonen 2,000, yodas_granary_ast_more 2,000, peoples_speech_dirty 2,000, peoples_speech_sharealike 2,000, libriheavy 2,000, wenetspeech 2,000, fleurs_asr 2,000, ami_ihm 2,000, earnings22 2,000, tedlium3 2,000, parlaspeech_hr 2,000, parlaspeech_rs 2,000, parlaspeech_cz 2,000, parlaspeech_pl 2,000, mediaspeech 2,000, mtedx 2,000, cml_tts 2,000, globe_v3 2,000, magpie_speech_orpheus 2,000, primewords_zh 2,000, covost2_en_to_x 2,000, emilia_yodas 2,000 |
| `prepare-media-video` | `media-video` | rekadaily_processed 3,000, vggsound 10,000, openvid_1m 10,000, insvie 5,000, videoufo_clips 600, vidgen_1m 600, vript_short 600, vript_long 600, openhumanvid_hq 600, llava_video_0_30_s_academic 600, llava_video_0_30_s_youtube 600, llava_video_30_60_s_academic 600, llava_video_30_60_s_youtube 600, panda70m_test 600, ditto_local 600, ditto_global_style1 600, ditto_global_style2 600, ditto_global_freeform1 600, ditto_global_freeform2 600, ditto_global_freeform3 600, reco_add 600, reco_remove 600, reco_replace 600, reco_style 600, openve3m_background_change 600, senorita_local_style 400, senorita_obj_removal 400, senorita_obj_addition 400, senorita_obj_swap 400, openve_sub 600 |


**Full-volume runs.** `configs/cloud/prepare-full-media-*.json` build every source completely,
one run for each media family (edits, images, speech, sound, video, gated). A run is a
`megatoken.cloud_prepare` run with `"kind": "media"`; the text runs' machinery is reused:

1. `scripts/shard_media.py --listing configs/data/files.media.json` lists the files of every
   file-based source at its pinned revision with their stored sizes (Hub metadata only). The file is
   committed, so every sandbox plans the same shards without a call to the Hub, and each shard opens
   its files with the listed size (no further API call). Sources added later are listed on the fly
   and appended.
2. `data/media_sharding.py` splits the sources into balanced shards by estimated GPU seconds:
   a source with many files is divided by file (a linked source by archive), a source with one
   enormous file or one archive cut in parts is divided by stripe (every shard reads it and keeps
   its share of the records), FLEURS by language. The per-clip costs are assumptions
   (`CLIP_SECONDS`: an image 0.05 s of GPU, a speech clip 0.025 s, a video clip 0.5 s, and so
   on) until one shard has been timed; a wrong constant makes shards unequal, never wrong.
3. Each GPU sandbox runs `scripts/prepare_open_media.py --shard-spec` for its shards (the teacher
   and the codec encode every clip) and stops at a digest of what it built.
4. One CPU sandbox collects the digests and decides which episodes are duplicates or fall in the
   validation split (`reconcile`).
5. The finishing step runs *inside* each ingest sandbox, with its shards' verdicts: a shard's
   prepared media is about a megabyte per image (float32 features), far more than its episodes, so
   moving it to a new sandbox would cost more than the GPU idling while step 4 runs. Each
   sandbox then serves its finished roots to training.

Speech sources carry `procedural_every` (one clip in forty also gets a procedural edit), which
supplies the `speech_edit` stream; no video is edited procedurally in these runs because real
video edits now exceed what the stage plan asks for.

| Run (`configs/cloud/<name>.json`) | Sources | GPU sandboxes | Shards | Estimated GPU hours per shard | Ingest hours per sandbox | Sandbox-hours billed | Repository TB | Prepared TB (est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prepare-full-media-edits` | 41 | 48 | 96 | 5.5-5.5 | 16 | 1,209 | 20.7 | 24.9 |
| `prepare-full-media-gated` | 20 | 48 | 96 | 7.2-7.2 | 20 | 1,410 | 15.6 | 24.0 + 10.5 (est.) |
| `prepare-full-media-images` | 32 | 128 | 256 | 7.5-7.5 | 22 | 3,977 | 66.2 | 100.0 |
| `prepare-full-media-sound` | 15 | 4 | 8 | 4.1-4.1 | 12 | 93 | 5.2 | 2.9 |
| `prepare-full-media-speech` | 36 | 96 | 192 | 6.2-6.2 | 18 | 2,601 | 31.5 | 94.5 |
| `prepare-full-media-video` | 30 | 160 | 320 | 9.1-9.1 | 26 | 5,609 | 94.3 | 57.1 |


Size and cost of building everything, from the manifests' inventories (`scripts/media_volume.py
summary`): clips or pairs before and after, the media they contain, the repository's size, the
size of the prepared form (teacher features at 2 KB per vector, plus codec latents; the prepared
form is typically larger than the source for speech and images because float32 features are
bigger than compressed media), and estimated GPU hours.

| Kind | Sources before | Sources after | Clips or pairs before | Clips or pairs after | Media after | Repository TB after | Prepared TB after (est.) | GPU hours after (est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Speech | 16 | 40 | 169,623,619 | 218,231,332 | 597,366 hours | 35.9 | 117.8 | 1,516 |
| Sound and music | 4 | 15 | 1,807,103 | 3,975,221 | 14,796 hours | 5.2 | 2.9 | 33 |
| Images with text | 13 | 32 | 50,729,133 | 138,195,802 | 138,195,802 images | 66.2 | 100.0 | 1,919 |
| Image edits | 6 | 31 | 6,946,900 | 16,662,895 | 33,509,459 images | 20.4 | 24.3 | 506 |
| Video | 4 | 14 | 5,845,210 | 9,757,814 | 26,347 hours | 87.9 | 28.0 | 1,382 |
| Video edits | 1 | 17 | 1,000,000 | 5,607,580 | 18,692 hours | 7.1 | 29.6 | 1,558 |
| Edit judging | 0 | 3 | 0 | 271,245 | 769,735 images | 0.3 | 0.6 | 19 |
| Edit plans | 0 | 8 | 0 | 173,052 | 346,104 images | 0.1 | 0.3 | 5 |

"Before" is the manifest before this work (44 sources; MagicBrush counted as single edits). Hours
for speech are the sources' own; images and video are counted by image or clip.

The table predates the 14 sources opened on 2026-10-23 (the repositories the account gained
access to). They add 25.0M speech clips (Thai, Indonesian and Vietnamese GigaSpeech 2 and
Mandarin WenetSpeech4TTS: 29,226 hours, 3.7 TB), 11.7M visual conversations over images
(Infinity-MM, SVIT, ShareGPT-4o: 6.8 TB), 162k captioned medical images and 2.2k captioned
videos; about 340 estimated GPU hours, which is why the `gated` run now has 48 sandboxes
(`scripts/media_volume.py summary` gives the full recount).

Things to know before launching:

- **The estimates are not measured.** GPU seconds per clip are assumptions (above). A first run
  of one shard of each family should replace them; the plan and the hours in the run files
  follow from the constants in `data/media_sharding.py` and `hours` in the run files.
- **Storage.** The prepared form is about 1.4 times the repository bytes in total
  (303 TB against 223 TB). It stays in the ingest sandboxes until training pulls
  it, so each of the 460 GPU sandboxes holds a few hundred gigabytes. The P1 media shares
  need about 120 TB of it (60B D tokens at 2 KB of features per token), so building
  everything is about 2.5 times what P1 reads once; build the families in turn, or lower
  `sandboxes` in a run file, to build less.
- **Hub limits.** One account's calls are shared by every sandbox (the Hub's response headers
  gave the account 500 API calls every five minutes on 2026-10-22). The committed listing removes
  the listing and size calls. Every Hub request that remains goes through
  `megatoken.hub.install()` (wait out 429 and 503), as all scripts here do. File reads are range
  requests of 32 MiB; reading 223 TB makes millions of them, so the account's download
  limit, not the GPUs, may set the pace.
- **Gated.** A source the account cannot read (the two marked 403 below) stops the sandbox
  that holds it at its first file. Accept their terms first, or take them out of the `gated` run's
  `media_sources`; relaunching after a failure redoes only the failed sandboxes.

### Gated repositories that need a click on the Hub

The sources below ask for the publisher's terms to be accepted on the Hub. With the token in
`~/.cache/huggingface/token` (account `gabrielpickard`), the first bytes of a data file of each
were requested on 2026-10-23: the last column says whether the account can read it. A request
for a repository's README, licence or file list is answered even when the terms are not
accepted, so it proves nothing; only a data file does. Where the answer is HTTP 403, click
"Agree" on the dataset page before the `gated` run starts.

| Source | Hub page | Account access on 2026-10-23 | What it is |
| --- | --- | --- | --- |
| `gigaspeech` | https://huggingface.co/datasets/speechcolab/gigaspeech | can read it | 10,000 hours of audiobooks, podcasts and YouTube with transcripts. |
| `emilia` | https://huggingface.co/datasets/amphion/Emilia-Dataset | can read it | Emilia, the original half of the Emilia-Large corpus (English, German, French, Chinese, Japanese, Korean); its YODAS half is read from an ungated mirror (emilia_yodas). |
| `finevideo` | https://huggingface.co/datasets/HuggingFaceFV/finevideo | can read it | 43,751 YouTube videos with sound and rich scene/activity annotations; clips are cut at the annotated activities. |
| `multiedit` | https://huggingface.co/datasets/inclusionAI/MultiEdit | can read it (read and checked on 2026-10-23) | MultiEdit: 106.5k hard instruction edits in six groups (objects, people, text in posters, interface screenshots, new viewpoints, 38 artistic styles). |
| `sharegpt4o_image_questions`, `sharegpt4o_video_captions` | https://huggingface.co/datasets/OpenGVLab/ShareGPT-4o | can read it (read and checked) | 57k image conversations and 2.2k video descriptions written by GPT-4o. |
| `infinity_mm_stage1`, `infinity_mm_synthetic`, `infinity_mm_zh` | https://huggingface.co/datasets/BAAI/Infinity-MM | can read it (read and checked) | Three parts of Infinity-MM, the 10.5 TB multimodal instruction set. |
| `svit_detail`, `svit_conversation`, `svit_complex`, `svit_referring` | https://huggingface.co/datasets/BAAI/SVIT | can read it (read and checked) | Four kinds of GPT-4 conversation about 108k Visual Genome images. |
| `medtrinity_demo` | https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M | can read it (read and checked) | 162k medical images with multi-granular captions (the demo subset). |
| `gigaspeech2_th`, `gigaspeech2_id`, `gigaspeech2_vi` | https://huggingface.co/datasets/speechcolab/gigaspeech2 | can read it (read and checked) | GigaSpeech 2: Thai, Indonesian and Vietnamese web speech (22,000 refined hours). |
| `wenetspeech4tts` | https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS | can read it (read and checked) | Mandarin speech graded by audio quality (7,226 hours read). |
| `spgispeech` | https://huggingface.co/datasets/kensho/spgispeech | 403: click Agree on the page first | SPGISpeech: 5,000 hours of company earnings calls with fully formatted (cased, punctuated, numerals) transcripts. |
| `ksponspeech` | https://huggingface.co/datasets/jp1924/KsponSpeech | 403: click Agree on the page first | KsponSpeech: 969 hours of Korean spontaneous conversation. |

Readable but not wired:

- `https://huggingface.co/datasets/peter-sushko/realedit_images` (141 GB, the pictures of the
  RealEdit photo-editing requests; the table of requests is open at `peter-sushko/RealEdit`).
  The ten archives are sorted by picture name over the whole set, and an original photo (named
  by its Reddit post, `3v1cvp.jpg`) and its edits (named by comment, `cxjhczp_1.jpg`) are in
  different archives, so a pair can only be formed by reading all 141 GB in one pass. The
  sharded reader reads each archive alone, so no source is wired.
- `https://huggingface.co/datasets/BAAI/Infinity-MM` stages 2 and 3 and most of stage 4: the
  stage 2 text part has no pictures; the rest (Cambrian, LLaVA-OneVision, Docmatix, ALLaVA,
  LVIS-Instruct4V, ShareGPT4V, DenseFusion, MathV360K, DocStruct4M and others) is part of
  `HuggingFaceM4/FineVision`, which is already read; `Infinity_Preference` is preference pairs
  that the question reader cannot use.
- `https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M` `25M_accessible` (see above).

Gated and still refused (HTTP 403 on a data file on 2026-10-23, so not read; their entries are
unchanged and carry the page to click): `kensho/SPGISpeech2.0` (388 GB, long earnings-call
snippets with speaker tags), `ai4bharat/NPTEL` (121 GB, Indian lecture speech with
translations), `otoearth/otoSpeech-full-duplex-280h` and `otoSpeech-full-duplex-turn-104h`
(two-speaker conversations) and `sarulab-speech/J-CHAT` (Japanese podcasts; the repository
holds only URL lists). Not wired for lack of a layout to read.

### What the stage plan implies, and the gap

A media event enters the data meter (D) by the number of teacher feature vectors the model
reads for it (`data/schema.py: encode_record`): audio at 40 ms per vector (25 a second), an image at
most 280 vectors, a video one frame a second (at most 32) of at most 140 vectors each (the
teacher's own `processor_config.json`). The earlier version of this table counted a media event
as zero and so found that the shares implied billions of episodes; counted by real input size,
an image episode costs about 300 D and a ten-second clip about 260, and the shares imply tens
of millions of episodes. The table below is computed from the manifests' inventories (items are
unique clips before the length and caption filters) by `scripts/media_volume.py gap`, with
per-episode text tokens measured with the reference codec on real label text from the first
forty-four sources and estimated for the others (marked in the script).

Streams, now (160 sources):

| Stream | Sources | Episodes available | D tokens per episode | P1 share (tokens) | Episodes the share implies | Passes over the data |
| --- | --- | --- | --- | --- | --- | --- |
| audio_description | 4 | 790,603 | 443 | 0.50% (2.0B) | 4,509,718 | 5.7 |
| audio_identity | 54 | 222,066,553 | 260 | 1.00% (4.0B) | 15,362,784 | 0.1 |
| av_creation | 8 | 7,451,145 | 975 | 1.00% (4.0B) | 4,102,540 | 0.6 |
| av_edit | 31 | 15,365,394 | 1,725 | 0.75% (3.0B) | 1,739,612 | 0.1 |
| edit_judging | 3 | 271,245 | 887 | 0.15% (0.6B) | 676,222 | 2.5 |
| edit_plan | 91 | 351,438,711 | 406 | 0.15% (0.6B) | 1,477,520 | 0.0 |
| image_creation | 28 | 114,310,295 | 446 | 1.00% (4.0B) | 8,958,567 | 0.1 |
| image_description | 30 | 129,089,584 | 421 | 1.50% (6.0B) | 14,258,927 | 0.1 |
| image_edit | 31 | 16,662,895 | 590 | 0.70% (2.8B) | 4,742,527 | 0.3 |
| image_identity | 32 | 138,195,802 | 293 | 1.50% (6.0B) | 20,477,816 | 0.1 |
| speech_edit | 39 | 218,091,332 | 518 | 0.75% (3.0B) | 5,796,599 | 0.0 |
| speech_translation | 5 | 28,365,589 | 294 | 1.50% (6.0B) | 20,377,443 | 0.7 |
| transcription | 36 | 217,534,974 | 296 | 1.50% (6.0B) | 20,256,447 | 0.1 |
| verbalization | 16 | 151,400,485 | 274 | 1.00% (4.0B) | 14,621,713 | 0.1 |
| video_description | 6 | 2,306,669 | 1,176 | 0.50% (2.0B) | 1,701,153 | 0.7 |
| video_identity | 14 | 9,757,814 | 871 | 1.50% (6.0B) | 6,888,724 | 0.7 |
| **all media streams** | | | | 15.00% (60.0B) | 145,948,311 | |

Streams before this work (the same arithmetic on the first forty-four sources; the new
`audio_description` and `video_description` streams had no source):

| Stream | Sources | Episodes available | D tokens per episode | P1 share (tokens) | Episodes the share implies | Passes over the data |
| --- | --- | --- | --- | --- | --- | --- |
| audio_description | 0 | 0 | 0 | 0.50% (2.0B) | 0 | inf |
| audio_identity | 19 | 171,290,722 | 255 | 1.00% (4.0B) | 15,697,538 | 0.1 |
| av_creation | 3 | 4,825,253 | 914 | 1.00% (4.0B) | 4,378,022 | 0.9 |
| av_edit | 5 | 6,845,210 | 1,628 | 0.75% (3.0B) | 1,843,099 | 0.3 |
| edit_judging | 0 | 0 | 0 | 0.15% (0.6B) | 0 | inf |
| edit_plan | 32 | 226,057,962 | 396 | 0.15% (0.6B) | 1,516,394 | 0.0 |
| image_creation | 12 | 50,722,915 | 446 | 1.00% (4.0B) | 8,958,567 | 0.2 |
| image_description | 12 | 50,722,915 | 430 | 1.50% (6.0B) | 13,947,001 | 0.3 |
| image_edit | 6 | 6,946,900 | 587 | 0.70% (2.8B) | 4,769,205 | 0.7 |
| image_identity | 13 | 50,729,133 | 293 | 1.50% (6.0B) | 20,477,816 | 0.4 |
| speech_edit | 15 | 169,483,619 | 509 | 0.75% (3.0B) | 5,888,737 | 0.0 |
| speech_translation | 3 | 23,985,811 | 309 | 1.50% (6.0B) | 19,432,531 | 0.8 |
| transcription | 12 | 168,927,261 | 292 | 1.50% (6.0B) | 20,529,428 | 0.1 |
| verbalization | 4 | 696,358 | 274 | 1.00% (4.0B) | 14,610,954 | 21.0 |
| video_description | 0 | 0 | 0 | 0.50% (2.0B) | 0 | inf |
| video_identity | 4 | 5,845,210 | 810 | 1.50% (6.0B) | 7,407,247 | 1.3 |
| **all media streams** | | | | 15.00% (60.0B) | 139,456,538 | |

Columns: "Episodes available" counts each source's clips once per stream it feeds; "D tokens per
episode" is the mean cost in D including the media vectors, weighted by items; "Episodes the share
implies" is the share of the 400B-token P1 budget divided by that cost; "Passes over the data" is
implied over available (below 1: the data would not be used up).

Reading it:

- Counted by real input size **every media stream now needs less than one pass over what
  the manifests can supply**, except `audio_description` (5.7 passes: the open audio caption sets
  hold 0.8M clips). Before, `verbalization` needed 21 passes over 0.7M clips; now transcription
  sources also read their text aloud (`verbalize`), 16 sources feed it, and it needs 0.1 passes.
  The two new editing streams are covered by real data more thinly: `edit_judging` needs
  2.5 passes over its 271,245 judgements, and the real plans of `edit_plan` are
  173,052 (about 5.0 passes; the executed plans, made from any clip, fill the rest).
- Video is the closest to its limit: `video_identity` 0.7, `video_description` 0.7 and
  `av_creation` 0.6 passes, and `av_edit` has the most real data (5,607,580 pairs against
  1.7M asked).
- The edit streams are covered with real data: image edits need 6.8M episodes and there are
  17,085,127 pairs (31 sources), of which 218,134 sessions hold several edits
  (each edit and result of a session counts); video edits need 1.7M and there are 5,607,580.
- The sixteen media streams hold 15% of P1 (60B D). At the tokens per episode above that is
  about 146 million episodes (the last row of the table), each a training window,
  against 222M unique audio clips, 138M images, 9.8M video clips and 23M edits in the
  manifests. Media dominate P1's data volume: 60B D of media is about 120 TB of teacher features
  alone (2 KB a vector).

Share changes made (`data/stages.json`; the P1 media total is unchanged at 15%, M1 and M2 keep
theirs): `audio_description` takes 0.5% of P1 (2.0B D) and `video_description` 0.5% (2.0B D), taken
from `audio_identity` (1.5% to 1.0%) and `video_identity` (2.0% to 1.5%); M1 and M2 move 0.25%
of their budgets the same way. Prepared roots built before this change lack the two new streams.

The editing streams then gave a part of their share to the two new ones: in the first stage
`image_edit` goes from 1.0% to 0.7% and `edit_judging` and `edit_plan` take 0.15% each; in M1 and
M2 `image_edit` goes from 0.5% to 0.4% and each new stream takes 0.05%. The media total stays at
15% (first stage) and the other stages' media totals are unchanged.

### Not added, and why

- **Links only** (an image or video fetch step would be needed and links rot): PixelProse's
  URL release, Recap-DataComp-1B, Spawning/PD12M's metadata, COYO-700M, DataComp-DR, BLIP3-KALE,
  SAM-LLaVA-Captions, InternVid, WebVid, HD-VILA, MiraData, MusicCaps (YouTube ids), Panda-70M
  (only a 5,892-clip test subset on the Hub has clips), the official PixMo sets (the images are
  URLs; the `pixmo-cap-images` repack is used), HowTo100M and VoxPopuli's 400,000 unlabelled
  hours.
- **Covered by FineVision**: the Cauldron, LLaVA-OneVision-Data, Cambrian-10M, CoSyn-400K,
  Docmatix, ALLaVA, LVIS-Instruct4V, ShareGPT4V and about 180 other visual-instruction sets are
  part of `HuggingFaceM4/FineVision` (24M conversations over 17.3M images), which is read as one
  source with the text-only and multi-image subsets and the low-rated rows left out. Several
  larger repackings (LLaVA-OneVision-1.5, Honey-Data, MAmmoTH-VL) overlap it and need a table
  and archive join (MAmmoTH-VL's table is one 22 GB JSON array).
- **Needs a reader that does not exist**: PDF-page sets (pdfa-eng-wds, IDL, olmOCR-mix); LAION
  captions stored apart from images (laion-coco-13m-molmo, megalith-10m-sharecap); a join across
  two repositories (pixmo-cap-qa and pixmo-points over `UWGZQ/pixmo_images`); sets with several
  images per conversation (Mantis, M4-Instruct, Molmo2 multi-image QA); Cambrian and
  Nemotron-VLM image archives; SpeechMatrix; CVSS (the Google files; the X-to-English CoVoST
  pairs are read instead); Common Voice releases 22 and later (tab-separated table per language);
  the Kazakh, Icelandic and AISHELL-1 tar sets (tables the archive reader cannot read).
- **Loose files, not archives** (one Hub request per file would be needed): SongEval, MAESTRO,
  MSR-VTT, JarvisArt, AISHELL-3/4, EARS.
- **Sound sets with doubtful rights**: Epidemic Sound's catalogue and several scraped
  stock-footage sets were left out because the licence is unclear.
- **Not instruction edits**: VIVID-10M (video, mask, caption), ROSE, GIER, EditScore's RL data
  (inputs only), the Emu Edit test set and the benchmark sets (KRIS, RISE, EditReward-Bench and
  others) are evaluation or conditioning data and are left alone.
- **Speech-to-speech and dialogue with spoken answers** (VoiceAssistant-400K, InstructS2S,
  Magpie-Speech-Orpheus' chat form, DailyTalk, SpokenWOZ, SLUE, SD-QA, Spoken-SQuAD,
  VoiceBench): the question is audio and the answer is text or audio, which no existing task
  expresses; a spoken-dialogue task would need a family and a stage share.
- **HiFi-TTS-2** is a manifest of audio that must be fetched from LibriVox; its 36.7k hours are
  not read.

### Sample checks

Every source was read with the real reader over a tiny sample taken from the Hub (a few rows or the
first tens of megabytes of one archive), turned into episodes with a stand-in for the GPU teacher,
and counted per stream. A parquet source's sample is the smallest row group of its first files; an
archive or linked source's sample is the head of its first archive with the table rows that match
the files in it. The results, one line per source, are in the table below; a source marked
"NOT READ" could not be read at all (gated, access refused) and was written from its card; two
more (`recap_coco_30k`, `fleurs`) were not sampled for the reasons given in their lines.

| Source | Reader | Sample result |
| --- | --- | --- |
| `docci` | docci | read: 3 clips from the first 31 train images of the real archive and descriptions file (a special reader for files that are not on the Hub); episodes image_description 3, image_creation 3, image_identity 3 |
| `llava_recap_cc3m` | parquet | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `llava_recap_558k` | parquet | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `llava_recap_118k` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `blip3o_long_caption` | webdataset | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `text_to_image_2m` | webdataset | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `commoncatalog_cc_by` | parquet | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `coco_captions` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `recap_coco_30k` | parquet | rows NOT read: the file is one 1.6 GB row group (30,504 rows), too slow to fetch here; its footer was read and the columns the reader needs exist (`image` raw bytes, `image_id` integer, `recaption` text), the same shapes other sources read |
| `textcaps` | parquet | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `flickr30k` | parquet | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `densefusion_1m` | archive | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `scienceqa_images` | parquet | read: 2 clips from a sample, episodes image_identity 2 |
| `magicbrush` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `ultraedit` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `hq_edit` | parquet | read: 1 clip from a sample, episodes image_edit 1 |
| `instructpix2pix` | parquet | read: 3 clips from a sample, episodes image_edit 3 |
| `omniedit` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `anyedit` | parquet | read: 3 clips from a sample, episodes image_edit 3 |
| `librispeech_asr` | parquet | read: 3 clips from a sample, episodes audio_identity 3, transcription 3 |
| `libritts_r` | parquet | read: 3 clips from a sample, episodes audio_identity 3, verbalization 3 |
| `mls_english` | parquet | read: 3 clips from a sample, episodes audio_identity 3, transcription 3 |
| `mls_multilingual` | parquet | read: 3 clips from a sample, episodes audio_identity 3, transcription 3 |
| `peoples_speech` | parquet | read: 3 clips from a sample, episodes audio_identity 3, transcription 3 |
| `voxpopuli` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `common_voice_17` | parquet | read: 3 clips from a sample, episodes audio_identity 3, transcription 3 |
| `gigaspeech` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2, verbalization 2 |
| `emilia` | webdataset | read: 3 clips from a sample, episodes audio_identity 3, transcription 3, verbalization 3 |
| `yodas_granary_en` | parquet | read: 3 clips from a sample, episodes audio_identity 3, transcription 3 |
| `yodas_granary_ast` | parquet | read: 3 clips from a sample, episodes audio_identity 3, speech_translation 3, transcription 3 |
| `covost2` | parquet | read: 3 clips from a sample, episodes audio_identity 3, speech_translation 3, transcription 3 |
| `fleurs` | fleurs | not sampled from the Hub: the reader pairs a language's reading of a sentence with the English reading of the same sentence, so a head-of-file sample of two archives has no pairs; the pairing and the whole task are exercised by the unit tests (`test_fleurs_pairs_the_same_sentence_in_two_languages`) |
| `hifi_tts` | parquet | read: 3 clips from a sample, episodes audio_identity 3, verbalization 3 |
| `expresso` | parquet | read: 3 clips from a sample, episodes audio_identity 3, verbalization 3 |
| `lj_speech` | parquet | read: 2 clips from a sample, episodes audio_identity 2, verbalization 2 |
| `audioset` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `audiocaps` | parquet | read: 3 clips from a sample, episodes audio_identity 3 |
| `clotho` | parquet | read: 3 clips from a sample, episodes audio_identity 3 |
| `song_describer` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `rekadaily_processed` | webdataset | read: 3 clips from a sample, episodes av_creation 3, video_identity 3 |
| `finevideo` | parquet | read: 3 clips from a sample, episodes av_creation 3, video_identity 3 |
| `vggsound` | archive | read: 1 clip from a sample, episodes av_creation 1, video_identity 1 |
| `openvid_1m` | webdataset | read: 3 clips from a sample, episodes video_description 3, video_identity 3 |
| `insvie` | archive | read: 3 clips from a sample, episodes av_edit 3 |
| `wavcaps_10s_16k` | parquet | read: 2 clips from a sample, episodes audio_description 2, audio_identity 2 |
| `musicbench` | archive | read: 3 clips from a sample, episodes audio_description 3, audio_identity 3 |
| `freesound_laion_640k` | parquet | read: 2 clips from a sample, episodes audio_description 2, audio_identity 2 |
| `jamendomaxcaps` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `fma_large_clips` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `mtg_jamendo` | webdataset | read: 3 clips from a sample, episodes audio_identity 3 |
| `fsd50k` | parquet | read: 3 clips from a sample, episodes audio_identity 3 |
| `esc50` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `urbansound8k` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `nsynth` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `birdset_xeno_canto` | parquet | read: 2 clips from a sample, episodes audio_identity 2 |
| `videoufo_clips` | archive | read: 3 clips from a sample, episodes video_description 3, video_identity 3 |
| `videoufo_frames` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `vidgen_1m` | archive | read: 2 clips from a sample, episodes av_creation 2, video_identity 2 |
| `vript_short` | archive | read: 3 clips from a sample, episodes av_creation 3, video_identity 3 |
| `vript_long` | archive | read: 3 clips from a sample, episodes av_creation 3, video_identity 3 |
| `openhumanvid_hq` | archive | read: 3 clips from a sample, episodes av_creation 3, video_identity 3 |
| `llava_video_0_30_s_academic` | archive | read: 3 clips from a sample, episodes video_description 3, video_identity 3 |
| `llava_video_0_30_s_youtube` | archive | read: 3 clips from a sample, episodes video_description 3, video_identity 3 |
| `llava_video_30_60_s_academic` | archive | read: 3 clips from a sample, episodes video_description 3, video_identity 3 |
| `llava_video_30_60_s_youtube` | archive | read: 3 clips from a sample, episodes video_description 3, video_identity 3 |
| `panda70m_test` | webdataset | read: 3 clips from a sample, episodes av_creation 3, video_identity 3 |
| `pixelprose` | webdataset | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `pd12m` | webdataset | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `recap_datacomp_12m` | webdataset | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `cc12m_recaptioned` | webdataset | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `wit_base` | parquet | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `met_openaccess` | parquet | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `roco_radiology` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `scimmir` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `wikiart_recaption` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `pmc_oa` | archive | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `sharegpt4o_image_t2i` | archive | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `sharegpt4o_image_edit` | linked | read: 3 clips from a sample, episodes image_edit 3 |
| `finevision` | parquet | read: 1 clip from a sample, episodes image_description 1, image_identity 1 |
| `ultraedit_region` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `pipe` | parquet | read: 1 clip from a sample, episodes image_edit 1 |
| `bytemorph_demo` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `nhr_edit` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `nhr_edit_part2` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `gpt_edit_simpler` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `editreward_data` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `flux_kontext_ipa` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `maskedit_10k` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `escher_aurora_kubric` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `escher_aurora_ag` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `escher_human_edit` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `escher_vismin` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `x2edit` | webdataset | read: 3 clips from a sample, episodes image_edit 3 |
| `seed_data_edit_multi_turn` | linked | read: 3 clips from a sample, episodes image_edit 3 |
| `seed_data_edit_real` | linked | read: 1 clip from a sample, episodes image_edit 1 |
| `imgedit_content_memory` | linked | read: 2 clips from a sample, episodes image_edit 2 |
| `imgedit_content_understanding` | linked | read: 2 clips from a sample, episodes image_edit 2 |
| `imgedit_version_backtracking` | linked | read: 1 clip from a sample, episodes image_edit 1 |
| `ditto_local` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `ditto_global_style1` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `ditto_global_style2` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `ditto_global_freeform1` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `ditto_global_freeform2` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `ditto_global_freeform3` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `reco_add` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `reco_remove` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `reco_replace` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `reco_style` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `openve3m_background_change` | parquet | read: 2 clips from a sample, episodes av_edit 2 |
| `senorita_local_style` | linked | read: 3 clips from a sample, episodes av_edit 3 |
| `senorita_obj_removal` | linked | read: 3 clips from a sample, episodes av_edit 3 |
| `senorita_obj_addition` | linked | read: 3 clips from a sample, episodes av_edit 3 |
| `senorita_obj_swap` | linked | read: 3 clips from a sample, episodes av_edit 3 |
| `openve_sub` | webdataset | read: 3 clips from a sample, episodes av_edit 3 |
| `taesiri_image_editing_request` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `editworld_reasoning` | parquet | read: 2 clips from a sample, episodes image_edit 2 |
| `yodas_granary_asr_nonen` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2, verbalization 2 |
| `yodas_granary_ast_more` | parquet | read: 2 clips from a sample, episodes audio_identity 2, speech_translation 2, transcription 2 |
| `peoples_speech_dirty` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `peoples_speech_sharealike` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `libriheavy` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2, verbalization 2 |
| `wenetspeech` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `fleurs_asr` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `ami_ihm` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `earnings22` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2, verbalization 2 |
| `tedlium3` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `parlaspeech_hr` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `parlaspeech_rs` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `parlaspeech_cz` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `parlaspeech_pl` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `mediaspeech` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `mtedx` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `cml_tts` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2, verbalization 2 |
| `globe_v3` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2, verbalization 2 |
| `magpie_speech_orpheus` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `primewords_zh` | parquet | read: 2 clips from a sample, episodes audio_identity 2, transcription 2 |
| `covost2_en_to_x` | parquet | read: 2 clips from a sample, episodes audio_identity 2, speech_translation 2, transcription 2 |
| `emilia_yodas` | webdataset | read: 3 clips from a sample, episodes audio_identity 3, transcription 3, verbalization 3 |
| `pixmo_cap_images` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `art_museums_pd_440k` | webdataset | read: 3 clips from a sample, episodes image_creation 3, image_description 3, image_identity 3 |
| `swiss_ai_smithsonian` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `swiss_ai_nasa` | parquet | read: 2 clips from a sample, episodes image_creation 2, image_description 2, image_identity 2 |
| `megalith_10m` | webdataset | read: 3 clips from a sample, episodes image_identity 3 |
| `llava_next_data` | parquet | read: 2 clips from a sample, episodes image_description 2, image_identity 2 |
| `sharegpt4o_image_questions` | archive | read 2026-10-23: 3 clips from `images.zip` joined to the jsonl, pictures decoded, one question and a long answer each; episodes image_description and image_identity |
| `sharegpt4o_video_captions` | archive | read 2026-10-23: 2 clips from `videos.zip`, cut to 8 s, with their long captions; episodes video_description and video_identity |
| `infinity_mm_stage1` | webdataset | read 2026-10-23: 3 clips from the first tar of the first folder, pictures decoded; episodes image_description and image_identity |
| `infinity_mm_synthetic` | webdataset | read 2026-10-23: the first tar of each part gave 3 clips (up to 9 turns before the 8-turn cut) |
| `infinity_mm_zh` | webdataset | read 2026-10-23: 3 clips with up to 17 turns before the cut, Chinese text |
| `svit_detail` | archive | read 2026-10-23: the 3-million-row table decoded in 3.4 minutes, then 3 clips from `raw/images2.zip`, pictures decoded; the other three kinds use the same table with another prefix |
| `medtrinity_demo` | parquet | read 2026-10-23: 3 clips from the first file (brain CT), 512 px, captions of several hundred characters; episodes image_description, image_creation and image_identity |
| `gigaspeech2_vi` | archive | read 2026-10-23: 3 clips from `data/vi/train/0.tar.gz` with their refined transcripts (16 kHz wav, 2-7 s) |
| `wenetspeech4tts` | webdataset | read 2026-10-23: one whole archive (`Premium_9`, 8.5 GB) was written to scratch and paired, 3 clips with their Mandarin transcripts (3.5-9.9 s); the spooling took 38 minutes on the shared link |
| `gigaspeech2_th`, `gigaspeech2_id` | archive | read 2026-10-23: 3 clips each from `data/th/train/1.tar.gz` and `data/id/train/1.tar.gz` |
| `multiedit` | linked | read: 3 clips from the head of the archive (2026-10-23), pictures decoded (1024 and 682 px), episodes image_edit 3 |
| `spgispeech` | parquet | NOT READ: gated (HTTP 403 for this account on 2026-10-23, re-checked); the layout was written from the dataset card. Needs Agree on the Hub, then a sample check. |
| `ksponspeech` | parquet | NOT READ: gated (HTTP 403 for this account on 2026-10-23, re-checked); the layout was written from the dataset card. Needs Agree on the Hub, then a sample check. |
| `editreward_judging` | parquet | read: 3 clips from a sample, episodes edit_judging 3 |
| `editscore_judging` | parquet | read: 3 clips from a sample, episodes edit_judging 3 |
| `pico_banana_preference` | table | read: 8 clips from a sample, episodes edit_judging 8; fetched 87 pictures (99 MB) over the network, 0 of 25 rows with a dead link |
| `pico_banana_multi_turn` | table | read: 8 clips from a sample, episodes image_edit 8; fetched 104 pictures (151 MB) over the network, 0 of 23 rows with a dead link |
| `pico_banana_single` | table | read: 8 clips from a sample, episodes image_edit 8; fetched 50 pictures (44 MB) over the network, 0 of 25 rows with a dead link |
| `escher_human_edit_plan` | parquet | read: 2 clips from a sample, episodes edit_plan 2 |
| `escher_vismin_plan` | parquet | read: 2 clips from a sample, episodes edit_plan 2 |
| `escher_aurora_kubric_plan` | parquet | read: 2 clips from a sample, episodes edit_plan 2 |
| `escher_aurora_ag_plan` | parquet | read: 2 clips from a sample, episodes edit_plan 2 |
| `escher_ss2_plan` | parquet | read: 2 clips from a sample, episodes edit_plan 2 |
| `escher_magicbrush_plan` | parquet | read: 2 clips from a sample, episodes edit_plan 2 |
| `editworld_reasoning_plan` | parquet | read: 2 clips from a sample, episodes edit_plan 2 |
| `jarvisart_plan` | folders | read: 3 clips from a sample, episodes edit_plan 3 |

### Corrections to the pilot section above

The media sources table and the "Still missing" list in the pilot section predate this work:
RekaDaily is now read from all nine subsets (10.2k hours, not only `video_capture`) and its
captions are recorded as model-written; MagicBrush is now read as multi-turn sessions and its edited
images are recorded as machine-made; the FLEURS files and the pilot's other media files are
streamed from the Hub rather than being hashed job inputs; FineVideo, Common Voice and CoVoST 2
are now sources; OpenVid-1M is read from its WebDataset repack (all 1.02M clips with captions, not
only the zip parts, so it now also feeds `video_description`); AudioCaps' captions now feed
`audio_description` (its validation and test splits stay unread as benchmarks); AnyEdit is read
from a per-type re-split because the original shard list dropped its weather, season and
time-of-day edits; UltraEdit now has an id column; the GigaSpeech glob that also matched its
evaluation files now names the training files only; and the Emilia entry now reads only its
original half, its YODAS half coming from an ungated mirror.

## More chat, reasoning and agent data

The first agentic manifest (`sources.agentic.json`, above) leaves the chat, reasoning and
tool-use streams far below what the stage plan asks for. This section adds a second sweep:
417 further open sources and 23 sources that need a click on the Hub. Coverage of coding
agents, terminal agents and code-execution traces is a separate piece of work: 18 coding sources
of this sweep moved to `sources.agentic-coding.json` ("Agentic coding sources" below), which
leaves 399 here. The ones that stay are listed under "Coding sources kept".

### How the sweep was done

1. About 13,800 candidate dataset repositories were collected from the Hub by running about 95
   search terms (chat, instruction, reasoning, distill, tool, function calling, agent,
   trajectory, multi-turn, per-language names and so on) sorted by downloads and by likes, plus
   the full dataset lists of about 110 organisations that publish training data.
2. About 540 repositories were read by hand or by a helper agent: the card, the file list, the
   column names, and a few real rows. 215 of them were written down as skipped with a reason.
3. For each kept source the maintainer's policy was applied: public data only, nothing
   generated here, any licence accepted if it is recorded accurately, so every entry carries the
   licence text from the card, `approved: true`, `reviewed_by` Gabriel Pickard and
   `reviewed_at` 2026-10-22.
4. Every source is pinned to a 40-character revision of its repository. For each, a data file
   was requested from the Hub with the project token to prove read access; the 23 that return
   HTTP 403 are in the gated manifest.
5. Each source has a size estimate (rows, bytes divided by four as tokens) written in its
   `notes`, and a first-run row cap in `max_rows`.

The manifests are `configs/data/sources.agentic-more.json` (399 sources over 238 repositories after the coding sources moved out)
and `configs/data/sources.agentic-more-gated.json` (23 sources over 9 repositories). Source ids
do not clash with any earlier manifest. The cloud job that prepares the open manifest is
`configs/cloud/prepare-agentic-more.json`.

### New readers

The existing readers cover most shapes. These were added, each with a tiny-fixture test in
`tests/test_agentic_adapters.py`:

- Role names are case-folded, so `User` and `Assistant` are read like `user` and `assistant`.
- **Reasoning stored beside the answer.** A message-level `reasoning_content` string is put
  in front of the assistant text as `<think>` ... `</think>`, exactly how existing chat
  sources that already inline their thinking are read. Reasoning is therefore trained on or
  masked together with the answer, by the same rule as before. A reasoning field that is a
  separate column of the row is used only by `prompt_and_reply` (below), and only together with
  the `answer` column.
- `human_assistant_text`: the Anthropic helpfulness and harmlessness format, one string with
  `Human:` and `Assistant:` turns separated by blank lines.
- `alternating_turns`: a list of bare strings, the first spoken by the user, then alternating
  (UltraChat, SODA).
- `prompt_and_reply`: one prompt column and one reply column. The reply is read from the first
  present of `generated_message`, `teacher_response`, `answer` or `output`.
- `turn_pairs` also accepts `input`/`output` and `human`/`assistant` pairs (MOSS).
- `prompt_completion`: a list of prompt messages plus a list of completion messages.
- Function-calling readers: OpenAI-wrapped tool lists are accepted by the xLAM reader, a row
  whose `answers` text is not JSON becomes a plain assistant message (an answer that declines
  to call a tool), and Glaive's first release (one string holding the system prompt and chat)
  is split at the first `USER:` line.

Tool results stay separate records. Sources whose tools ran for real (Toucan's MCP servers,
the ADP web agents, Jupyter kernels) are marked `executed`; sources whose results were written
by a model (Glaive, xLAM, APIGen, ToolACE-style sets) are marked `simulated`. Training masks
fall only on assistant turns in every new source.

### What was added

Counts are rows of the open manifest by stream. Tokens are the sum of the per-source "unique
tokens" below (bytes divided by four, taken from the dataset sizes on the Hub, so it counts
the same prompt again whenever two sources share it). Read them as ceilings, not as the
deduplicated size.

| Stream | Sources | Tokens before deduplication |
| --- | --- | --- |
| chat | 156 | 121.9B |
| synthetic_reasoning | 167 | 799.7B |
| synthetic_code_dialogue | 28 | 22.6B |
| tool_use | 23 | 7.2B |
| typed_exact_instructions | 10 | 1.5B |
| executed_trajectories | 20 | 27.6B |
| verified_trajectories | 3 | 2.2B |
| environment_trajectories | 9 | 0.7B |
| recovery_clarification | 1 | 1.5M |

Notable sources:

- **Reasoning**: the NVIDIA Nemotron families (Cascade SFT stages 1 and 2 and Cascade-2,
  Math v2, SFT-Math v3 and v4, Post-Training v1 math and science, v2 code, math, science and
  multilingual, OpenMathReasoning, OpenScience, OpenCodeReasoning), the AM-Team distills of
  DeepSeek-R1, Qwen3 and thinking models, OpenThoughts2-1M, OpenR1-Math, Mixture-of-Thoughts,
  open-r1 codeforces traces, Glaive reasoning, Ring-lite, rStar-Coder, MegaScience, Dolci
  (OLMo 3) think SFT, and several community distills of GLM, Kimi and DeepSeek.
- **Chat**: Magpie variants, Hermes-3, ShareGPT-GPT4, WizardLM V2, the four Anthropic HH-RLHF
  preference sets (chosen replies only), UltraChat full, LMSYS-Chat-1M with DeepSeek-V3
  answers (non-gated copy), LMSYS GPT-5 responses, Tulu v2, WildChat-1M, Aya, Nemotron
  multilingual, Daring-Anteater, Infinity-Instruct variants.
- **Tool use**: Toucan with Kimi-K2, gpt-oss and Qwen3 teachers (calls executed against
  real MCP servers), Synth-APIGen, Glaive function calling v1, the SmolTalk2 tool splits.
- **Agents (non-coding)**: the Agent Data Protocol web collection, ETO, UltraData-SFT-Agent,
  jupyter-agent rows that were run on a real kernel.
- **Restricted licences** are recorded, not hidden: AM-DeepSeek-R1 (research use), Updesh
  (Microsoft Research licence), KodCode, LaMini, Canarim and AceMath (non-commercial).

Per-stream source tables, generated from the manifest:


#### chat (open, 156 sources, 121.9B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `sharegpt-v3-cleaned` | `shibing624/sharegpt_gpt4` | cc-by-4.0 | 8,000 | 82M |
| `sharegpt-gpt4-6k` | `shibing624/sharegpt_gpt4` | cc-by-4.0 | 8,000 | 22M |
| `wizardlm-evol-instruct-v2` | `WizardLMTeam/WizardLM_evol_instruct_V2_196k` | mit | 8,000 | 87M |
| `samantha-1-1` | `QuixiAI/samantha-data` | apache-2.0 | 8,000 | 5.0M |
| `dolphin-flan1m-gpt4` | `QuixiAI/dolphin` | apache-2.0 | 8,000 | 360M |
| `dolphin-flan5m-gpt35` | `QuixiAI/dolphin` | apache-2.0 | 8,000 | 1.1B |
| `open-instruct-uncensored-tulu` | `QuixiAI/open-instruct-uncensored` | apache-2.0 | 8,000 | 250M |
| `hermes-3-dataset` | `NousResearch/Hermes-3-Dataset` | apache-2.0 | 8,000 | 400M |
| `alpaca-cleaned` | `yahma/alpaca-cleaned` | cc-by-4.0 | 8,000 | 9.0M |
| `smoltalk-systemchats-30k` | `HuggingFaceTB/smoltalk` (systemchats-30k) | not stated for this subset on the  | 8,000 | 18M |
| `smoltalk-longalign` | `HuggingFaceTB/smoltalk` (longalign) | apache-2.0 | 8,000 | 28M |
| `smoltalk-explore-instruct-rewriting` | `HuggingFaceTB/smoltalk` (explore-instruct-rewriting) | not stated on the card | 8,000 | 8.0M |
| `smoltalk-metamathqa-50k` | `HuggingFaceTB/smoltalk` (metamathqa-50k) | mit | 8,000 | 14M |
| `everyday-conversations-llama31-2k` | `HuggingFaceTB/everyday-conversations-llama3.1-2k` | apache-2.0 | 8,000 | 1.4M |
| `mosaic-instruct-v3` | `mosaicml/instruct-v3` | cc-by-sa-3.0 | 8,000 | 55M |
| `vmware-open-instruct` | `VMware/open-instruct` | cc-by-3.0 | 8,000 | 31M |
| `swype-instruct` | `swype/instruct` | mit | 8,000 | 240M |
| `finetome-100k` | `mlabonne/FineTome-100k` | not stated on the card | 8,000 | 60M |
| `magpie-pro-300k-mt` | `HuggingFaceTB/MagPie-Pro-300k-MT` | llama3 | 8,000 | 375M |
| `lamini-instruction` | `MBZUAI/LaMini-instruction` | cc-by-nc-4.0 | 8,000 | 290M |
| `lmsys-chat-1m-clean-deepseek` | `OpenLeecher/lmsys_chat_1m_clean` | not stated on the card | 8,000 | 190M |
| `orpo-dpo-mix-40k-chosen` | `mlabonne/orpo-dpo-mix-40k` | apache-2.0 | 16,000 | 29M |
| `open-perfectblend` | `mlabonne/open-perfectblend` | apache-2.0 | 8,000 | 740M |
| `ultrafeedback-argilla-chosen` | `argilla/ultrafeedback-binarized-preferences-cleaned` | mit | 8,000 | 65M |
| `wizard-vicuna-70k-unfiltered` | `QuixiAI/wizard_vicuna_70k_unfiltered` | apache-2.0 | 8,000 | 35M |
| `self-instruct-82k` | `hakurei/open-instruct-v1` | apache-2.0 | 8,000 | 6.0M |
| `magpie-llama33-pro-1m` | `Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1` | llama3.3 | 8,000 | 638M |
| `magpie-llama3-pro-1m` | `Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1` | llama3 | 8,000 | 586M |
| `magpie-llama3-air-3m` | `Magpie-Align/Llama-3-Magpie-Air-3M-v0.1` | llama3 | 8,000 | 1.6B |
| `magpie-qwen2-pro-1m` | `Magpie-Align/Magpie-Qwen2-Pro-1M-v0.1` | not stated on the card | 8,000 | 390M |
| `magpie-qwen2-air-3m` | `Magpie-Align/Magpie-Qwen2-Air-3M-v0.1` | not stated on the card | 8,000 | 1.2B |
| `magpie-pro-mt-300k` | `Magpie-Align/Magpie-Pro-MT-300K-v0.1` | llama3 | 8,000 | 390M |
| `magpie-llama31-pro-mt-300k-filtered` | `Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered` | llama3.1 | 8,000 | 367M |
| `magpie-gemma2-pro-200k-filtered` | `Magpie-Align/Magpie-Gemma2-Pro-200K-Filtered` | not stated on the card | 8,000 | 101M |
| `magpie-phi3-pro-300k-filtered` | `Magpie-Align/Magpie-Phi3-Pro-300K-Filtered` | not stated on the card | 8,000 | 216M |
| `magpielm-sft-550k` | `Magpie-Align/MagpieLM-SFT-Data-v0.1` | llama3.1 | 8,000 | 287M |
| `magpie-qwen2-pro-200k-chinese` | `Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese` | not stated on the card | 8,000 | 87M |
| `magpie-qwen25-coder-pro-300k` | `Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1` | other | 8,000 | 180M |
| `magpie-qwen25-math-pro-300k` | `Magpie-Align/Magpie-Qwen2.5-Math-Pro-300K-v0.1` | not stated on the card | 8,000 | 175M |
| `magpie-zoo-gemma2-27b` | `Magpie-Align/Magpie-100K-Generator-Zoo` | llama3.1 | 8,000 | 67M |
| `magpie-ultra-v1-filtered` | `argilla/magpie-ultra-v1.0` (filtered) | not stated on the card | 8,000 | 1.0B |
| `magpie-ultra-v01` | `argilla/magpie-ultra-v0.1` | llama3.1 | 8,000 | 28M |
| `kalo-opus-instruct-22k` | `anthracite-org/kalo-opus-instruct-22k-no-refusal` | apache-2.0 | 8,000 | 20M |
| `claude-multiround-chat-30k` | `Norquinal/claude_multiround_chat_30k` | not stated on the card | 8,000 | 30M |
| `gpt4o-writing-prompts` | `Gryphe/ChatGPT-4o-Writing-Prompts` | unknown | 8,000 | 7.0M |
| `longwriter-6k` | `zai-org/LongWriter-6k` | apache-2.0 | 8,000 | 37M |
| `longalign-10k` | `zai-org/LongAlign-10k` | not stated on the card | 3,000 | 160M |
| `longcite-45k` | `zai-org/LongCite-45k` | apache-2.0 | 3,000 | 1.4B |
| `wildchat-creative-writing-10k` | `sam-paech/wildchat_creative_writing_annotated_10k` | not stated on the card | 8,000 | 14M |
| `creative-writing-sft-50k` | `stindardlogic/creative-writing-sft-50k` | apache-2.0 | 8,000 | 15M |
| `bitext-customer-support` | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | cdla-sharing-1.0 | 8,000 | 5.0M |
| `counsel-chat` | `nbertagnolli/counsel-chat` | not stated on the card | 8,000 | 1.0M |
| `korean-safe-conversation` | `jojo0217/korean_safe_conversation` | apache-2.0 | 8,000 | 4.0M |
| `math-orca-math-200k` | `microsoft/orca-math-word-problems-200k` | mit | 8,000 | 39M |
| `math-metamathqa` | `meta-math/MetaMathQA` | mit | 8,000 | 55M |
| `math-mathinstruct` | `TIGER-Lab/MathInstruct` | mit | 8,000 | 47M |
| `math-math-plus` | `TIGER-Lab/MATH-plus` | mit | 8,000 | 180M |
| `math-webinstruct-sub` | `TIGER-Lab/WebInstructSub` | apache-2.0 | 8,000 | 520M |
| `math-gsm-instructions` | `qwedsacf/grade-school-math-instructions` | not stated on the card | 8,000 | 1.2M |
| `math-numinamath-tir` | `AI-MO/NuminaMath-TIR` | apache-2.0 | 8,000 | 40M |
| `math-numinamath-1.5` | `AI-MO/NuminaMath-1.5` | apache-2.0 | 8,000 | 330M |
| `math-aops-instruct` | `DeepStudentLlama/AoPS-Instruct` (default) | mit | 8,000 | 317M |
| `code-stack-exchange-instruction` | `ArmelR/stack-exchange-instruction` | not stated on the card | 8,000 | 2.0B |
| `math-metamath-qwen2-verified` | `yingyingzhang/metamath-qwen2-math` | apache-2.0 | 8,000 | 130M |
| `math-numina-qwen2-verified` | `yingyingzhang/metamath-qwen2-math` | apache-2.0 | 8,000 | 235M |
| `sci-trendyol-cybersecurity` | `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` | apache-2.0 | 8,000 | 35M |
| `daring-anteater` | `nvidia/Daring-Anteater` | cc-by-4.0 | 8,000 | 240M |
| `nemotron-ifchat-v1-chat-if` | `nvidia/Nemotron-Instruction-Following-Chat-v1` | odc-by-1.0 | 8,000 | 1.5B |
| `nemotron-ifchat-v2-reasoning-off` | `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` | odc-by | 8,000 | 1.4B |
| `nemotron-ifchat-v2-reasoning-on` | `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` | odc-by | 8,000 | 1.4B |
| `nemotron-sft-safety-v2` | `nvidia/Nemotron-SFT-Safety-v2` | cc-by-4.0 | 8,000 | 70M |
| `nemotron-sft-bbq-v1` | `nvidia/Nemotron-SFT-BBQ-v1` | openmdw-1.1 | 8,000 | 1.2M |
| `nemotron-sft-long-context-transduction-v1` | `nvidia/Nemotron-SFT-Long-Context-Transduction-v1` | openmdw-1.1 | 8,000 | 300M |
| `nemotron-sft-long-context-translation-pg19-v1` | `nvidia/Nemotron-SFT-Long-Context-Translation-PG19-v1` | openmdw-1.1 | 4,335 | 555M |
| `nemotron-multilingual-v1-de` | `nvidia/Nemotron-SFT-Multilingual-v1` | cc-by-4.0 | 8,000 | 660M |
| `nemotron-multilingual-v1-es` | `nvidia/Nemotron-SFT-Multilingual-v1` | cc-by-4.0 | 8,000 | 484M |
| `nemotron-multilingual-v1-fr` | `nvidia/Nemotron-SFT-Multilingual-v1` | cc-by-4.0 | 8,000 | 494M |
| `nemotron-multilingual-v1-it` | `nvidia/Nemotron-SFT-Multilingual-v1` | cc-by-4.0 | 8,000 | 611M |
| `nemotron-multilingual-v1-ja` | `nvidia/Nemotron-SFT-Multilingual-v1` | cc-by-4.0 | 8,000 | 526M |
| `nemotron-multilingual-v1-zh` | `nvidia/Nemotron-SFT-Multilingual-v1` | cc-by-4.0 | 8,000 | 428M |
| `nemotron-multilingual-v2-hi` | `nvidia/Nemotron-SFT-Multilingual-v2` | cc-by-4.0 | 8,000 | 67M |
| `nemotron-multilingual-v2-ja` | `nvidia/Nemotron-SFT-Multilingual-v2` | cc-by-4.0 | 8,000 | 67M |
| `nemotron-multilingual-v2-ko` | `nvidia/Nemotron-SFT-Multilingual-v2` | cc-by-4.0 | 8,000 | 64M |
| `nemotron-multilingual-v2-pt` | `nvidia/Nemotron-SFT-Multilingual-v2` | cc-by-4.0 | 8,000 | 76M |
| `helpsteer2-rated-responses` | `nvidia/HelpSteer2` | cc-by-4.0 | 11,500 | 8.0M |
| `aegis2-safe-responses` | `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | cc-by-4.0 | 40,000 | 2.0M |
| `coconot-original-train` | `allenai/coconot` (original) | AI2 ImpACT low-risk artifact licen | 8,000 | 1.5M |
| `prosocial-dialog` | `allenai/prosocial-dialog` | cc-by-4.0 | 8,000 | 12M |
| `sciriff-4096` | `allenai/SciRIFF` (4096) | odc-by | 8,000 | 97M |
| `tulu-v2-sft-mixture` | `allenai/tulu-v2-sft-mixture` | odc-by | 8,000 | 310M |
| `wildchat-1m` | `allenai/WildChat-1M` | odc-by | 9,000 | 600M |
| `nemotron-pt-v2-code` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 245M |
| `nemotron-pt-v2-math` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 127M |
| `nemotron-pt-v2-stem` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 202M |
| `cascade2-chat` | `nvidia/Nemotron-Cascade-2-SFT-Data` (chat) | nvidia-open-model-license | 8,000 | 53.2B |
| `cascade2-safety` | `nvidia/Nemotron-Cascade-2-SFT-Data` (safety) | nvidia-open-model-license | 3,570 | 4.0M |
| `smoltalk2-smoltalk-multilingual-8languages-lang-5-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 167M |
| `smoltalk2-table-gpt-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | mit | 3,000 | 11M |
| `smoltalk2-longalign-64k-context-lang-annotated-lang-6-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 96M |
| `dolphin-r1-nonreasoning` | `QuixiAI/dolphin-r1` | apache-2.0 | 3,000 | 346M |
| `finance-agentic-sft-v1` | `nvidia/Nemotron-SFT-Agentic-Finance-v1` | openmdw-1.1 | 8,000 | 5.0M |
| `glaive-rag-v1` | `glaiveai/RAG-v1` | apache-2.0 | 8,000 | 68M |
| `aya-dataset-human` | `CohereLabs/aya_dataset` | apache-2.0 | 8,000 | 47M |
| `aya-collection-translated-dolly` | `CohereLabs/aya_collection` | apache-2.0 | 8,000 | 470M |
| `aya-collection-japanese-instruct` | `CohereLabs/aya_collection` | apache-2.0 | 8,000 | 170M |
| `updesh-g-generative` | `microsoft/Updesh_beta` | microsoft-research-license | 8,000 | 7.5B |
| `updesh-r-translated` | `microsoft/Updesh_beta` | microsoft-research-license | 8,000 | 13.0B |
| `dolci-instruct-sft-translated-12lang` | `openeurollm/Dolci-Instruct-SFT-translated` | apache-2.0 | 8,000 | 6.0B |
| `smoltalk-chinese` | `opencsg/smoltalk-chinese` | apache-2.0 | 8,000 | 370M |
| `belle-school-math-0.25m` | `BelleGroup/school_math_0.25M` | gpl-3.0 | 8,000 | 30M |
| `coig-cqia` | `m-a-p/COIG-CQIA` | not stated on the card | 8,000 | 35M |
| `coig-kun` | `m-a-p/COIG-Kun` | not stated on the card | 8,000 | 132M |
| `coig-translated-flan` | `BAAI/COIG` | apache-2.0 | 8,000 | 9.0M |
| `mxode-zh-magpie` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 329M |
| `mxode-zh-firefly` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 554M |
| `mxode-zh-chinese-medical` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 407M |
| `mxode-zh-industryinstruction` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 79M |
| `mxode-zh-stem` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 102M |
| `mxode-zh-agri-qa` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 307M |
| `mxode-zh-infinity-zh` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 130M |
| `mxode-zh-neo-sft-phase2` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 22M |
| `mxode-zh-stackoverflow-c` | `Mxode/Chinese-Instruct` | cc-by-sa-4.0 | 8,000 | 108M |
| `alpaca-gpt4-zh` | `shibing624/alpaca-zh` | cc-by-4.0 | 8,000 | 8.0M |
| `evol-instruct-chinese-gpt4` | `FreedomIntelligence/Evol-Instruct-Chinese-GPT4` | not stated on the card | 8,000 | 29M |
| `evol-instruct-hindi` | `FreedomIntelligence/evol-instruct-hindi` | not stated on the card | 8,000 | 60M |
| `alpaca-gpt4-arabic` | `FreedomIntelligence/alpaca-gpt4-arabic` | apache-2.0 | 8,000 | 14M |
| `huatuogpt2-sft-gpt4-140k` | `FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K` | apache-2.0 | 8,000 | 55M |
| `huatuogpt-sft-v1` | `FreedomIntelligence/HuatuoGPT-sft-data-v1` | apache-2.0 | 8,000 | 82M |
| `ru-turbo-alpaca` | `IlyaGusev/ru_turbo_alpaca` | cc-by-4.0 | 8,000 | 10M |
| `russian-dialogues-den4ik` | `Den4ikAI/russian_dialogues` | mit | 8,000 | 65M |
| `alpaca-spanish` | `bertin-project/alpaca-spanish` | cc-by-4.0 | 8,000 | 5.0M |
| `turkce-atlas-instruct` | `AlicanKiraz0/Turkce-Atlas-Instruct` | mit | 8,000 | 118M |
| `instructurca` | `turkish-nlp-suite/InstrucTurca` | cc-by-sa-4.0 | 8,000 | 1.0B |
| `hh-rlhf-49k-ja` | `kunishou/hh-rlhf-49k-ja` | mit | 8,000 | 15M |
| `canarim-instruct-ptbr` | `dominguesm/Canarim-Instruct-PTBR-Dataset` | cc-by-nc-4.0 | 8,000 | 23M |
| `chatdoctor-healthcaremagic-100k` | `lavita/ChatDoctor-HealthCareMagic-100k` | not stated on the card | 8,000 | 31M |
| `ai-medical-chatbot` | `ruslanmv/ai-medical-chatbot` | not stated on the card | 8,000 | 70M |
| `alpacare-medinstruct-52k` | `lavita/AlpaCare-MedInstruct-52k` | not stated on the card | 8,000 | 16M |
| `disc-med-sft` | `Flmc/DISC-Med-SFT` | apache-2.0 | 8,000 | 185M |
| `disc-law-sft-pair` | `ShengbinYue/DISC-Law-SFT` | apache-2.0 | 8,000 | 78M |
| `finance-alpaca` | `gbharti/finance-alpaca` | mit | 8,000 | 10M |
| `anthropic-hh-harmless-base-chosen` | `Anthropic/hh-rlhf` | mit | 8,000 | 6.7M |
| `anthropic-hh-helpful-base-chosen` | `Anthropic/hh-rlhf` | mit | 8,000 | 8.6M |
| `anthropic-hh-helpful-online-chosen` | `Anthropic/hh-rlhf` | mit | 8,000 | 10M |
| `anthropic-hh-helpful-rejection-sampled-chosen` | `Anthropic/hh-rlhf` | mit | 8,000 | 13M |
| `lmsys-gpt5-chat-response` | `ytz20/LMSYS-Chat-GPT-5-Chat-Response` | cc-by-4.0 | 8,000 | 95M |
| `ultrachat-full-1p4m` | `stingning/ultrachat` | mit | 8,000 | 1.9B |
| `nebius-deepseek-v3-infinity-instruct` | `nebius/DeepSeek-V3-Infinity-Instruct-0625` | cc-by-4.0 | 8,000 | 672M |
| `nebius-gpt-oss-120b-infinity-instruct` | `nebius/gpt-oss-120b-Infinity-Instruct-0625` | cc-by-4.0 | 8,000 | 1.7B |
| `soda-dialogues` | `allenai/soda` | cc-by-4.0 | 8,000 | 190M |
| `llama-nemotron-sft-chat` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT) | cc-by-4.0 | 8,000 | 64M |
| `llama-nemotron-sft-safety` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT) | cc-by-4.0 | 8,000 | 14M |
| `acemath-general-sft-stage2` | `nvidia/AceMath-Instruct-Training-Data` | cc-by-nc-4.0 | 3,000 | 1.2B |
| `sft-datablend-v1` | `nvidia/sft_datablend_v1` | cc-by-4.0 | 8,000 | 67M |
| `moss-003-sft` | `YeungNLP/moss-003-sft-data` | not stated on the card | 8,000 | 2.1B |
| `sharegpt-zh-en-90k` | `shareAI/ShareGPT-Chinese-English-90k` | apache-2.0 | 8,000 | 230M |

#### synthetic_reasoning (open, 167 sources, 799.7B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `code-kodcode-sft-r1` | `KodCode/KodCode-V1-SFT-R1` (default) | cc-by-nc-4.0 | 3,000 | 1.1B |
| `sci-datascience-instruct-science` | `RUC-DataLab/DataScience-Instruct-500K` | mit | 3,000 | 50M |
| `nemotron-pt-v1-math` | `nvidia/Nemotron-Post-Training-Dataset-v1` | cc-by-4.0 | 3,000 | 19.8B |
| `nemotron-pt-v1-stem` | `nvidia/Nemotron-Post-Training-Dataset-v1` | cc-by-4.0 | 3,000 | 82.4B |
| `nemotron-pt-v2-multilingual-ja` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 4.5B |
| `nemotron-pt-v2-multilingual-de` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 4.7B |
| `nemotron-pt-v2-multilingual-it` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 4.7B |
| `nemotron-pt-v2-multilingual-es` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 4.1B |
| `nemotron-pt-v2-multilingual-fr` | `nvidia/Nemotron-Post-Training-Dataset-v2` | cc-by-4.0 | 3,000 | 4.6B |
| `cascade-s1-math` | `nvidia/Nemotron-Cascade-SFT-Stage-1` (math) | cc-by-4.0 | 3,000 | 15.4B |
| `cascade-s1-code` | `nvidia/Nemotron-Cascade-SFT-Stage-1` (code) | cc-by-4.0 | 3,000 | 9.0B |
| `cascade-s1-science` | `nvidia/Nemotron-Cascade-SFT-Stage-1` (science) | cc-by-4.0 | 3,000 | 1.3B |
| `cascade-s1-general` | `nvidia/Nemotron-Cascade-SFT-Stage-1` (general) | cc-by-4.0 | 3,000 | 2.1B |
| `cascade-s2-math` | `nvidia/Nemotron-Cascade-SFT-Stage-2` (math) | cc-by-4.0 | 3,000 | 18.4B |
| `cascade-s2-code` | `nvidia/Nemotron-Cascade-SFT-Stage-2` (code) | cc-by-4.0 | 3,000 | 16.0B |
| `cascade-s2-science` | `nvidia/Nemotron-Cascade-SFT-Stage-2` (science) | cc-by-4.0 | 3,000 | 2.3B |
| `cascade-s2-general` | `nvidia/Nemotron-Cascade-SFT-Stage-2` (general) | cc-by-4.0 | 3,000 | 6.0B |
| `cascade2-math-notool` | `nvidia/Nemotron-Cascade-2-SFT-Data` (math) | nvidia-open-model-license | 3,000 | 29.9B |
| `cascade2-math-proof` | `nvidia/Nemotron-Cascade-2-SFT-Data` (math) | nvidia-open-model-license | 3,000 | 14.0B |
| `cascade2-science` | `nvidia/Nemotron-Cascade-2-SFT-Data` (science) | nvidia-open-model-license | 3,000 | 11.6B |
| `nemotron-math-v2-high` | `nvidia/Nemotron-Math-v2` | cc-by-4.0 | 3,000 | 37.8B |
| `nemotron-math-v2-medium` | `nvidia/Nemotron-Math-v2` | cc-by-4.0 | 3,000 | 9.9B |
| `nemotron-math-v2-low` | `nvidia/Nemotron-Math-v2` | cc-by-4.0 | 3,000 | 3.0B |
| `nemotron-sft-math-v3` | `nvidia/Nemotron-SFT-Math-v3` | cc-by-4.0 | 3,000 | 38.5B |
| `nemotron-sft-math-v4` | `nvidia/Nemotron-SFT-Math-v4` | cc-by-4.0 | 3,000 | 7.3B |
| `nemotron-math-proofs-v1-lean` | `nvidia/Nemotron-Math-Proofs-v1` | cc-by-sa-4.0 | 4,500 | 7.4B |
| `nemotron-math-proofs-v2` | `nvidia/Nemotron-Math-Proofs-v2` | cc-by-4.0 | 3,000 | 4.3B |
| `nemotron-math-proofs-v3-sft` | `nvidia/Nemotron-Math-Proofs-v3-SFT` | cc-by-4.0 | 3,000 | 32.7B |
| `nemotron-science-v1-mcq` | `nvidia/Nemotron-Science-v1` | cc-by-4.0 | 3,000 | 316M |
| `nemotron-science-v1-rqa` | `nvidia/Nemotron-Science-v1` | cc-by-4.0 | 3,000 | 204M |
| `nemotron-sft-science-v2-syn-mcq` | `nvidia/Nemotron-SFT-Science-v2` (syn_mcq) | cc-by-sa-4.0 | 3,000 | 142M |
| `nemotron-sft-science-v2-rqa` | `nvidia/Nemotron-SFT-Science-v2` (rqa) | cc-by-sa-4.0 | 3,000 | 2.4B |
| `nemotron-sft-science-v2-so` | `nvidia/Nemotron-SFT-Science-v2` (so) | cc-by-sa-4.0 | 3,000 | 4.0B |
| `nemotron-sft-science-v2-vendor` | `nvidia/Nemotron-SFT-Science-v2` (vendor) | cc-by-sa-4.0 | 3,000 | 6.7B |
| `openscience-q25-32b-10` | `nvidia/OpenScience` (OS-Q2.5-32B-10) | cc-by-4.0 | 3,000 | 10.6B |
| `openscience-q25-72b-10` | `nvidia/OpenScience` (OS-Q2.5-72B-10) | cc-by-4.0 | 3,000 | 8.8B |
| `openscience-q25-32b-4` | `nvidia/OpenScience` (OS-Q2.5-32B-4) | cc-by-4.0 | 3,000 | 241M |
| `openscience-q3-235b-4` | `nvidia/OpenScience` (OS-Q3-235B-4) | cc-by-4.0 | 3,000 | 665M |
| `openscience-reasoning-2` | `nvidia/OpenScienceReasoning-2` | cc-by-4.0 | 3,000 | 7.4B |
| `openmathreasoning-cot` | `nvidia/OpenMathReasoning` | cc-by-4.0 | 3,000 | 17.9B |
| `openmathreasoning-tir` | `nvidia/OpenMathReasoning` | cc-by-4.0 | 3,000 | 8.9B |
| `openmathreasoning-genselect` | `nvidia/OpenMathReasoning` | cc-by-4.0 | 3,000 | 1.7B |
| `openmathinstruct-1-correct` | `nvidia/OpenMathInstruct-1` | nvidia-license | 3,000 | 332M |
| `nemotron-prismmath` | `nvidia/Nemotron-PrismMath` | cc-by-4.0 | 3,000 | 4.3B |
| `opencodereasoning-split0` | `nvidia/OpenCodeReasoning` (split_0) | cc-by-4.0 | 3,000 | 2.1B |
| `nemotron-competitive-programming-v1-cpp` | `nvidia/Nemotron-Competitive-Programming-v1` | cc-by-4.0 | 10,000 | 3.6B |
| `nemotron-competitive-programming-v1-infinibyte` | `nvidia/Nemotron-Competitive-Programming-v1` | cc-by-4.0 | 3,000 | 12.4B |
| `nemotron-sft-competitive-programming-v2-cpp` | `nvidia/Nemotron-SFT-Competitive-Programming-v2` (default) | cc-by-4.0 | 3,000 | 11.8B |
| `nemotron-sft-competitive-programming-v2-python` | `nvidia/Nemotron-SFT-Competitive-Programming-v2` (default) | cc-by-4.0 | 3,000 | 12.0B |
| `nemotron-sft-competitive-programming-v2-text-to-sql` | `nvidia/Nemotron-SFT-Competitive-Programming-v2` (default) | cc-by-4.0 | 3,000 | 293M |
| `arc-agi-large-reasoning-notools` | `nvidia/Nemotron-SFT-ARC-AGI-v1` (large_reasoning_no_tools) | cc-by-4.0 | 3,000 | 2.1B |
| `finance-document-grounded-reasoning` | `nvidia/Nemotron-SFT-Finance-Document-Grounded-Reasoning-v1` | openmdw-1.1 | 3,000 | 253M |
| `openr1-math-default` | `open-r1/OpenR1-Math-220k` (default) | apache-2.0 | 3,000 | 474M |
| `openr1-math-extended` | `open-r1/OpenR1-Math-220k` (extended) | apache-2.0 | 3,000 | 313M |
| `mot-code` | `open-r1/Mixture-of-Thoughts` (code) | not stated on the card | 3,000 | 968M |
| `mot-science` | `open-r1/Mixture-of-Thoughts` (science) | not stated on the card | 3,000 | 398M |
| `codeforces-cots-solutions` | `open-r1/codeforces-cots` (solutions) | odc-by-1.0 | 3,158 | 515M |
| `codeforces-cots-solutions-py` | `open-r1/codeforces-cots` (solutions_py) | odc-by-1.0 | 3,158 | 96M |
| `codeforces-cots-solutions-w-editorials` | `open-r1/codeforces-cots` (solutions_w_editorials) | odc-by-1.0 | 3,158 | 330M |
| `codeforces-cots-solutions-w-editorials-py` | `open-r1/codeforces-cots` (solutions_w_editorials_py) | odc-by-1.0 | 3,158 | 134M |
| `codeforces-cots-checker-interactor` | `open-r1/codeforces-cots` (checker_interactor) | odc-by-1.0 | 3,000 | 114M |
| `codeforces-cots-test-input-generator` | `open-r1/codeforces-cots` (test_input_generator) | odc-by-1.0 | 3,000 | 242M |
| `openthoughts2-1m` | `open-thoughts/OpenThoughts2-1M` | apache-2.0 | 3,000 | 4.2B |
| `reasoning-tulu-chatml-580k` | `HuggingFaceTB/reasoning_tulu_chatml_580k_all_lengths` | not stated | 3,000 | 1.2B |
| `am-r1-0528-math` | `a-m-team/AM-DeepSeek-R1-0528-Distilled` | research use only | 3,000 | 3.7B |
| `am-r1-0528-code` | `a-m-team/AM-DeepSeek-R1-0528-Distilled` | research use only | 3,000 | 2.5B |
| `am-r1-0528-science` | `a-m-team/AM-DeepSeek-R1-0528-Distilled` | research use only | 3,000 | 512M |
| `am-r1-0528-other` | `a-m-team/AM-DeepSeek-R1-0528-Distilled` | research use only | 3,000 | 2.1B |
| `am-r1-0528-multiturn` | `a-m-team/AM-DeepSeek-R1-0528-Distilled` | research use only | 3,000 | 804M |
| `am-qwen3-math` | `a-m-team/AM-Qwen3-Distilled` | research use only | 3,000 | 1.7B |
| `am-qwen3-code` | `a-m-team/AM-Qwen3-Distilled` | research use only | 3,000 | 1.8B |
| `am-qwen3-other` | `a-m-team/AM-Qwen3-Distilled` | research use only | 3,000 | 1.7B |
| `am-qwen3-multiturn` | `a-m-team/AM-Qwen3-Distilled` | research use only | 3,000 | 430M |
| `am-thinking-math` | `a-m-team/AM-Thinking-v1-Distilled` | research use only | 3,000 | 1.4B |
| `am-thinking-code` | `a-m-team/AM-Thinking-v1-Distilled` | research use only | 3,000 | 1.6B |
| `am-r1-distilled-0p5m` | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | cc-by-nc-4.0 | 3,000 | 1.8B |
| `am-r1-distilled-0p9m` | `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | cc-by-nc-4.0 | 3,000 | 2.4B |
| `am-40m-r1-math` | `a-m-team/AM-DeepSeek-Distilled-40M` | cc-by-nc-4.0 | 3,000 | 2.5B |
| `am-40m-r1-code` | `a-m-team/AM-DeepSeek-Distilled-40M` | cc-by-nc-4.0 | 3,000 | 2.8B |
| `am-40m-r1-science` | `a-m-team/AM-DeepSeek-Distilled-40M` | cc-by-nc-4.0 | 3,000 | 183M |
| `am-40m-r1-other` | `a-m-team/AM-DeepSeek-Distilled-40M` | cc-by-nc-4.0 | 3,000 | 3.1B |
| `smoltalk2-aya-dataset-qwen3-32b-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 19M |
| `smoltalk2-longalign-64k-qwen3-32b-yarn-131k-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 136M |
| `smoltalk2-smoltalk-multilingual8-qwen3-32b-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 552M |
| `smoltalk2-smoltalk-systemchats-qwen3-32b-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 30M |
| `smoltalk2-table-gpt-qwen3-32b-think` | `HuggingFaceTB/smoltalk2` (SFT) | mit | 3,000 | 26M |
| `smoltalk2-smoltalk-everyday-convs-reasoning-qwen3-32b-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 2,057 | 3.0M |
| `magpie-reasoning-v1-150k` | `Magpie-Align/Magpie-Reasoning-V1-150K` | llama3 | 3,000 | 76M |
| `magpie-reasoning-v1-150k-cot-r1-llama70b` | `Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B` | research only: Tongyi Qianwen lice | 3,000 | 306M |
| `magpie-reasoning-v2-250k-cot-r1-llama70b` | `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` | llama3.3 | 3,000 | 452M |
| `magpie-reasoning-v2-250k-cot-qwq` | `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-QwQ` | llama3.1 | 3,000 | 509M |
| `dolci-think-sft-7b` | `allenai/Dolci-Think-SFT-7B` | odc-by-1.0 | 3,000 | 19.4B |
| `dolci-think-sft-32b` | `allenai/Dolci-Think-SFT-32B` | odc-by-1.0 | 3,000 | 19.4B |
| `dolci-think-python` | `allenai/Dolci-Think-SFT-Python` | odc-by-1.0 | 3,659 | 4.5B |
| `dolci-think-translated-cs` | `openeurollm/Dolci-Think-SFT-translated` (cs) | apache-2.0 | 4,989 | 328M |
| `dolci-think-translated-de` | `openeurollm/Dolci-Think-SFT-translated` (de) | apache-2.0 | 5,225 | 399M |
| `dolci-think-translated-el` | `openeurollm/Dolci-Think-SFT-translated` (el) | apache-2.0 | 5,882 | 316M |
| `dolci-think-translated-es` | `openeurollm/Dolci-Think-SFT-translated` (es) | apache-2.0 | 5,450 | 237M |
| `dolci-think-translated-fi` | `openeurollm/Dolci-Think-SFT-translated` (fi) | apache-2.0 | 4,953 | 444M |
| `dolci-think-translated-fr` | `openeurollm/Dolci-Think-SFT-translated` (fr) | apache-2.0 | 5,348 | 361M |
| `dolci-think-translated-it` | `openeurollm/Dolci-Think-SFT-translated` (it) | apache-2.0 | 3,338 | 1.1B |
| `dolci-think-translated-nl` | `openeurollm/Dolci-Think-SFT-translated` (nl) | apache-2.0 | 3,368 | 960M |
| `dolci-think-translated-pl` | `openeurollm/Dolci-Think-SFT-translated` (pl) | apache-2.0 | 3,368 | 899M |
| `dolci-think-translated-ro` | `openeurollm/Dolci-Think-SFT-translated` (ro) | apache-2.0 | 3,387 | 786M |
| `dolci-think-translated-sv` | `openeurollm/Dolci-Think-SFT-translated` (sv) | apache-2.0 | 3,370 | 822M |
| `dolci-think-translated-uk` | `openeurollm/Dolci-Think-SFT-translated` (uk) | apache-2.0 | 3,424 | 1.0B |
| `numinamath-qwq-cot-5m` | `PrimeIntellect/NuminaMath-QwQ-CoT-5M` | mit | 7,264 | 2.5B |
| `synthetic-1-sft` | `PrimeIntellect/SYNTHETIC-1-SFT-Data` | apache-2.0 | 3,000 | 2.1B |
| `synthetic-2-sft-verified` | `PrimeIntellect/SYNTHETIC-2-SFT-verified` | apache-2.0 | 3,000 | 661M |
| `big-reasoning-traces` | `allenai/big-reasoning-traces` (DeepSeek) | permissive mix | 3,000 | 2.5B |
| `r1-distill-sft-v0` | `ServiceNow-AI/R1-Distill-SFT` (v0) | cc-by-nc-sa-4.0 | 3,000 | 188M |
| `r1-distill-sft-v1` | `ServiceNow-AI/R1-Distill-SFT` (v1) | cc-by-nc-sa-4.0 | 3,000 | 2.9B |
| `glaive-reasoning-v1-20m` | `glaiveai/reasoning-v1-20m` | apache-2.0 | 3,000 | 35.8B |
| `openo1-sft` | `O1-OPEN/OpenO1-SFT` | apache-2.0 | 3,000 | 93M |
| `eurus-2-sft` | `PRIME-RL/Eurus-2-SFT-Data` | mit | 3,000 | 110M |
| `deepmath-103k-r1-solution-1` | `zwhe99/DeepMath-103K` | mit | 3,000 | 330M |
| `deepmath-103k-r1-solution-2` | `zwhe99/DeepMath-103K` | mit | 3,000 | 364M |
| `deepmath-103k-r1-solution-3` | `zwhe99/DeepMath-103K` | mit | 3,000 | 331M |
| `skunkworks-reasoning-001` | `SkunkworksAI/reasoning-0.01` | apache-2.0 | 3,000 | 9.0M |
| `ring-lite-sft` | `inclusionAI/Ring-lite-sft-data` | apache-2.0 | 3,000 | 45.0B |
| `superior-reasoning-gptoss-stage1` | `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` (stage1) | cc-by-4.0 | 3,000 | 1.1B |
| `superior-reasoning-gptoss-stage2` | `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` (stage2) | cc-by-4.0 | 3,000 | 5.0B |
| `rstar-coder-synthetic-sft` | `microsoft/rStar-Coder` (synthetic_sft) | cc-by-4.0 | 3,000 | 5.1B |
| `rstar-coder-seed-sft` | `microsoft/rStar-Coder` (seed_sft) | cc-by-4.0 | 18,000 | 1.2B |
| `megascience` | `MegaScience/MegaScience` | cc-by-nc-sa-4.0 | 3,000 | 850M |
| `webinstruct-cft-600k` | `TIGER-Lab/WebInstruct-CFT` (WebInstruct-CFT-600K) | apache-2.0 | 3,000 | 390M |
| `ultrainteract-sft` | `openbmb/UltraInteract_sft` | mit | 3,000 | 235M |
| `dart-math-hard` | `hkust-nlp/dart-math-hard` | mit | 3,000 | 233M |
| `dart-math-uniform` | `hkust-nlp/dart-math-uniform` | mit | 3,000 | 148M |
| `medical-o1-sft-en` | `FreedomIntelligence/medical-o1-reasoning-SFT` (en) | apache-2.0 | 3,000 | 6.0M |
| `medical-o1-sft-zh` | `FreedomIntelligence/medical-o1-reasoning-SFT` (zh) | apache-2.0 | 3,000 | 8.0M |
| `medical-r1-distill-en` | `FreedomIntelligence/Medical-R1-Distill-Data` (en) | apache-2.0 | 3,000 | 8.0M |
| `openmed-medical-reasoning-mega` | `OpenMed/Medical-Reasoning-SFT-Mega` | apache-2.0 | 3,000 | 2.3B |
| `ii-medical-reasoning-sft` | `Intelligent-Internet/II-Medical-Reasoning-SFT` | not stated on the card | 3,000 | 5.1B |
| `glm51-reasoning-main` | `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` (main) | apache-2.0 | 3,000 | 2.4B |
| `glm51-reasoning-phdscience` | `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` (PHD-Science) | apache-2.0 | 3,000 | 460M |
| `glm51-reasoning-multilingualstem` | `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` (Multilingual-STEM) | apache-2.0 | 3,000 | 514M |
| `glm51-reasoning-math` | `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` (Math) | apache-2.0 | 3,000 | 623M |
| `kimik25-reasoning-generaldistillation` | `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` (General-Distillation) | apache-2.0 | 3,000 | 1.8B |
| `kimik25-reasoning-phdscience` | `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` (PHD-Science) | apache-2.0 | 3,000 | 337M |
| `kimik25-reasoning-generalmath` | `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` (General-Math) | apache-2.0 | 3,000 | 925M |
| `kimik25-reasoning-multilingualstem` | `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` (MultilingualSTEM) | apache-2.0 | 3,000 | 417M |
| `dsv4pro-distilled-math` | `Jackrong/DeepSeek-V4-Pro-Distilled-200K` (Math) | cc-by-4.0 | 3,000 | 6.5B |
| `dsv4pro-distilled-stem` | `Jackrong/DeepSeek-V4-Pro-Distilled-200K` (STEM) | cc-by-4.0 | 3,000 | 800M |
| `claude-distills` | `clzoro/Claude-Distills` | mit | 3,000 | 230M |
| `congliu-chinese-r1-distill-110k` | `Congliu/Chinese-DeepSeek-R1-Distill-data-110k` | apache-2.0 | 3,000 | 70M |
| `lightblue-multilingual-r1-llama70b` | `lightblue/reasoning-multilingual-R1-Llama-70B-train` | apache-2.0 | 3,000 | 6.0M |
| `tiny-aya-multilingual-reasoning` | `CohereLabs/tiny-aya-l2-thinker-multilingual-reasoning` | cc-by-nc-4.0 | 3,000 | 110M |
| `fineproofs-sft` | `lm-provers/FineProofs-SFT` (all) | apache-2.0 | 3,000 | 160M |
| `galaxymind-stem-reasoning-complex` | `galaxyMindAiLabs/stem-reasoning-complex` | apache-2.0 | 3,000 | 1.0B |
| `goedel-prover-v2-sft` | `Goedel-LM/SFT_dataset_v2` | apache-2.0 | 3,000 | 4.0B |
| `smoltalk-multilingual8-qwen3` | `HuggingFaceTB/smoltalk-multilingual8-Qwen3-32B-main-gen` (all_formatted_lang_5_no_qwen) | not stated on the card | 3,000 | 450M |
| `nemotron-v2-decontam-multilingual-de` | `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated` | cc-by-4.0 | 3,000 | 4.7B |
| `nemotron-v2-decontam-multilingual-es` | `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated` | cc-by-4.0 | 3,000 | 4.1B |
| `nemotron-v2-decontam-multilingual-fr` | `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated` | cc-by-4.0 | 3,000 | 4.5B |
| `nemotron-v2-decontam-multilingual-it` | `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated` | cc-by-4.0 | 3,000 | 4.7B |
| `nemotron-v2-decontam-multilingual-ja` | `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated` | cc-by-4.0 | 3,000 | 4.5B |
| `chinese-deepseek-r1-110k-sft` | `Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT` | apache-2.0 | 3,000 | 160M |
| `llama-nemotron-sft-science` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT) | cc-by-4.0 | 3,000 | 1.5B |
| `llama-nemotron-sft-code-v1` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT) | cc-by-4.0 | 3,000 | 7.7B |
| `llama-nemotron-sft-code-v1.1` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT) | cc-by-4.0 | 3,000 | 4.4B |
| `llama-nemotron-sft-math-v1` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT) | cc-by-4.0 | 3,000 | 8.5B |
| `llama-nemotron-sft-math-v1.1` | `nvidia/Llama-Nemotron-Post-Training-Dataset` (SFT) | cc-by-4.0 | 3,000 | 10.3B |
| `acemath-math-sft` | `nvidia/AceMath-Instruct-Training-Data` | cc-by-nc-4.0 | 3,000 | 1.0B |
| `acemath-general-sft-stage1` | `nvidia/AceMath-Instruct-Training-Data` | cc-by-nc-4.0 | 3,000 | 1.3B |
| `dolphin-r1-reasoning-deepseek` | `QuixiAI/dolphin-r1` | apache-2.0 | 3,000 | 900M |
| `dolphin-r1-reasoning-flash` | `QuixiAI/dolphin-r1` | apache-2.0 | 3,000 | 820M |

#### typed_exact_instructions (open, 10 sources, 1.5B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `nemotron-ifchat-v1-structured-outputs` | `nvidia/Nemotron-Instruction-Following-Chat-v1` | cc-by-4.0 | 8,000 | 13M |
| `if-sft-data-verified` | `allenai/IF_sft_data_verified` | not stated on the card | 8,000 | 20M |
| `cascade-s2-instruction-following` | `nvidia/Nemotron-Cascade-SFT-Stage-2` (instruction-following) | cc-by-4.0 | 3,000 | 183M |
| `cascade2-instruction-following` | `nvidia/Nemotron-Cascade-2-SFT-Data` (instruction_following) | nvidia-open-model-license | 3,000 | 889M |
| `am-r1-0528-if` | `a-m-team/AM-DeepSeek-R1-0528-Distilled` | research use only | 3,000 | 115M |
| `am-qwen3-if` | `a-m-team/AM-Qwen3-Distilled` | research use only | 3,000 | 94M |
| `am-40m-r1-if` | `a-m-team/AM-DeepSeek-Distilled-40M` | cc-by-nc-4.0 | 3,000 | 117M |
| `smoltalk2-multi-turn-reasoning-if-think` | `HuggingFaceTB/smoltalk2` (SFT) | odc-by-1.0 | 3,000 | 98M |
| `smoltalk2-tulu-3-sft-personas-instruction-following-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | odc-by-1.0 | 3,000 | 14M |
| `hermes-json-mode` | `interstellarninja/hermes-function-calling-v1` | apache-2.0 | 8,000 | 2.0M |

#### tool_use (open, 22 sources, 7.1B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `smoltalk-apigen-80k` | `HuggingFaceTB/smoltalk` (apigen-80k) | cc-by-4.0 | 8,000 | 15M |
| `cascade-s2-tool-calling` | `nvidia/Nemotron-Cascade-SFT-Stage-2` (tool_calling) | cc-by-4.0 | 3,000 | 1.7B |
| `arc-agi-large-tools-noreasoning` | `nvidia/Nemotron-SFT-ARC-AGI-v1` (large_tools_no_reasoning) | cc-by-4.0 | 2,500 | 7.0M |
| `smoltalk2-smolagents-toolcalling-traces-think` | `HuggingFaceTB/smoltalk2` (SFT) | not stated | 3,000 | 64M |
| `smoltalk2-hermes-function-calling-v1-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 11M |
| `smoltalk2-xlam-traces-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | cc-by-4.0 | 3,000 | 29M |
| `ultradata-agent-tool-use` | `openbmb/UltraData-SFT-Agent-2609` (Tool-Use) | apache-2.0 | 3,000 | 170M |
| `synth-apigen-v0.1` | `argilla/Synth-APIGen-v0.1` | apache-2.0 | 14,000 | 8.0M |
| `synth-apigen-qwen-exec` | `plaguss/synth-apigen-qwen-exec` | not stated on the card | 15,000 | 3.0M |
| `synth-apigen-llama-exec` | `plaguss/synth-apigen-llama-exec` | not stated on the card | 9,500 | 7.0M |
| `hermes-tool-use-multiturn-reasoning` | `interstellarninja/tool-use-multiturn-reasoning` | apache-2.0 | 8,000 | 33M |
| `hermes-xlam-validated` | `interstellarninja/xlam_hermes_validated` | not stated on the card | 2,325 | 3.3M |
| `hermes-thinking-v1` | `Jofthomas/hermes-function-calling-thinking-V1` | not stated on the card | 3,570 | 3.9M |
| `dolci-think-hybrid-tool-use-sa` | `allenai/Dolci-Think-SFT-Olmo-Hybrid-Tool-Use-SA` | cc-by-sa-4.0 | 1,597 | 10M |
| `pythonic-function-calling` | `driaforall/pythonic-function-calling` | not stated on the card | 8,000 | 35M |
| `api-bank-train` | `liminghao1630/API-Bank` | mit | 8,000 | 14M |
| `glaive-fc-v1` | `glaiveai/glaive-function-calling` | apache-2.0 | 8,000 | 26M |
| `toucan-kimi-k2` | `Agent-Ark/Toucan-1.5M` (Kimi-K2) | apache-2.0 | 20,000 | 650M |
| `toucan-oss` | `Agent-Ark/Toucan-1.5M` (OSS) | apache-2.0 | 15,500 | 1.4B |
| `toucan-qwen3` | `Agent-Ark/Toucan-1.5M` (Qwen3) | apache-2.0 | 12,000 | 840M |
| `xlam-r1-cot` | `AymanTarig/function-calling-v0.2-with-r1-cot` | not stated on the card | 8,000 | 60M |
| `adp2-miroverse` | `neulab/adp-v2` | cc-by-nc-4.0 | 800 | 2.0B |

#### recovery_clarification (open, 1 sources, 1.5M)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `synth-apigen-unanswerable` | `argilla/Synth-APIGen-v0.1` | apache-2.0 | 6,000 | 1.5M |

#### environment_trajectories (open, 9 sources, 706M)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `adp-go-browse-wa` | `neulab/agent-data-collection` | mit | 1,500 | 130M |
| `adp-nnetnav-wa` | `neulab/agent-data-collection` | apache-2.0 | 500 | 120M |
| `adp-nnetnav-live` | `neulab/agent-data-collection` | apache-2.0 | 400 | 250M |
| `adp-synatra` | `neulab/agent-data-collection` | cc-by-sa-4.0 | 3,000 | 100M |
| `adp2-mind2web` | `neulab/adp-v2` | cc-by-4.0 | 1,500 | 85M |
| `adp2-webarena-successful` | `neulab/adp-v2` | other | 530 | 10M |
| `agent-eto-webshop` | `agent-eto/eto-sft-trajectory` | apache-2.0 | 1,823 | 2.5M |
| `agent-eto-scienceworld` | `agent-eto/eto-sft-trajectory` | apache-2.0 | 1,482 | 3.5M |
| `agent-eto-alfworld` | `agent-eto/eto-sft-trajectory` | apache-2.0 | 3,118 | 4.5M |

#### executed_trajectories (open, 6 sources, 18.6B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `arc-agi-large-reasoning-tools` | `nvidia/Nemotron-SFT-ARC-AGI-v1` (large_reasoning_and_tools) | cc-by-4.0 | 1,000 | 1.0B |
| `ultradata-agent-search` | `openbmb/UltraData-SFT-Agent-2609` (Search-Agent) | apache-2.0 | 2,000 | 390M |
| `ultradata-agent-general` | `openbmb/UltraData-SFT-Agent-2609` (General-Agent) | apache-2.0 | 1,000 | 8.5B |
| `scale-swe-distilled` | `AweAI-Team/Scale-SWE-Distilled` | cc-by-4.0 | 2,000 | 3.5B |
| `toolmind-web-trajectories` | `Nanbeige/ToolMind-Web-QA` | apache-2.0 | 500 | 390M |
| `adp2-openresearcher` | `neulab/adp-v2` | mit | 800 | 4.8B |

#### verified_trajectories (open, 1 sources, 231M)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `swe-lego-real-resolved` | `PrimeIntellect/SWE-Lego-Real-Data-Verified` | apache-2.0 | 500 | 231M |

#### synthetic_code_dialogue (open, 27 sources, 22.4B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `smoltalk-self-oss-instruct` | `HuggingFaceTB/smoltalk` (self-oss-instruct) | odc-by | 8,000 | 15M |
| `code-magicoder-oss-instruct` | `ise-uiuc/Magicoder-OSS-Instruct-75K` | mit | 8,000 | 35M |
| `code-magicoder-evol-instruct` | `ise-uiuc/Magicoder-Evol-Instruct-110K` | apache-2.0 | 8,000 | 61M |
| `code-nickrosh-evol-instruct-80k` | `nickrosh/Evol-Instruct-Code-80k-v1` | cc-by-nc-sa-4.0 | 8,000 | 30M |
| `code-codefuse-evol-instruction-66k` | `codefuse-ai/Evol-instruction-66k` | cc-by-nc-sa-4.0 | 8,000 | 33M |
| `code-selfoss-instruct-sc2-exec` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | odc-by | 8,000 | 25M |
| `code-tested-143k-python` | `Vezora/Tested-143k-Python-Alpaca` | apache-2.0 | 8,000 | 70M |
| `code-instructions-122k` | `TokenBender/code_instructions_122k_alpaca_style` | apache-2.0 | 8,000 | 25M |
| `code-instructcoder` | `likaixin/InstructCoder` | not stated on the card | 8,000 | 35M |
| `code-codealpaca-20k` | `HuggingFaceH4/CodeAlpaca_20K` | cc | 8,000 | 1.3M |
| `code-dolphin-coder-codegen` | `QuixiAI/dolphin-coder` | apache-2.0 | 8,000 | 55M |
| `code-dolphin-coder-translate` | `QuixiAI/dolphin-coder` | apache-2.0 | 8,000 | 18M |
| `code-search-net-python-instruct` | `Nan-Do/instructional_code-search-net-python` | apache-2.0 | 8,000 | 113M |
| `code-opc-sft1-filtered-infinity` | `OpenCoder-LLM/opc-sft-stage1` (filtered_infinity_instruct) | mit | 8,000 | 525M |
| `code-opc-sft1-realuser` | `OpenCoder-LLM/opc-sft-stage1` (realuser_instruct) | mit | 8,000 | 555M |
| `code-opc-sft1-largescale-diverse` | `OpenCoder-LLM/opc-sft-stage1` (largescale_diverse_instruct) | mit | 8,000 | 1.6B |
| `code-opc-sft2-educational` | `OpenCoder-LLM/opc-sft-stage2` (educational_instruct) | mit | 8,000 | 15M |
| `code-opc-sft2-mceval` | `OpenCoder-LLM/opc-sft-stage2` (mceval_instruct) | mit | 8,000 | 33M |
| `code-opc-sft2-package` | `OpenCoder-LLM/opc-sft-stage2` (package_instruct) | mit | 8,000 | 158M |
| `code-code-feedback` | `m-a-p/Code-Feedback` | apache-2.0 | 8,000 | 98M |
| `code-ling-coder-sft` | `inclusionAI/Ling-Coder-SFT` | apache-2.0 | 8,000 | 3.0B |
| `code-react-ui-instructions` | `cfahlgren1/react-code-instructions` | mit | 8,000 | 100M |
| `code-nextcoder-edits` | `microsoft/NextCoderDataset` | mit | 8,000 | 650M |
| `code-kodcode-sft-4o` | `KodCode/KodCode-V1-SFT-4o` (default) | cc-by-nc-4.0 | 8,000 | 180M |
| `nemotron-sft-competitive-programming-v2-exercism` | `nvidia/Nemotron-SFT-Competitive-Programming-v2` (default) | cc-by-4.0 | 3,000 | 172M |
| `opencodegeneticinstruct-mixtral` | `nvidia/OpenCodeGeneticInstruct` (mixtral-8x22b-instruct) | cc-by-4.0 | 3,000 | 8.6B |
| `opencodegeneticinstruct-qwen25` | `nvidia/OpenCodeGeneticInstruct` (qwen2.5-32b-instruct) | cc-by-4.0 | 3,000 | 6.2B |

### Coding sources kept

These sources are code data that stays in this manifest: the 28 code-instruction sources in
`synthetic_code_dialogue`, `code-code-feedback`, `scale-swe-distilled` and `swe-lego-real-resolved`
(the later-stage manifest reads both for its own streams), `math-numinamath-tir`, and UltraData's
`ultradata-agent-general` and `ultradata-agent-search`. The coding-agent, terminal-agent and
code-execution sources of this sweep, and `davinci-dev-env-native`, moved to
`sources.agentic-coding.json`; `adp2-swe-chat` was dropped there in favour of the raw SWE-chat
sessions it was made from. `openswe-trajectories` is in
`sources.agentic-coding.json`.

### Sources that need a click on the Hub

Re-checked on 2026-10-23: the first bytes of a data file of every repository below were requested again with the project token and each still answered HTTP 403, so none could be read and none was moved out of the gated manifest. (A request for a repository's README or licence file is answered even before the terms are accepted, so it is no test.)

`configs/data/sources.agentic-more-gated.json` has 23 sources over 9 repositories whose data
files return HTTP 403 for the project token. Each is pinned and recorded like the open ones.
Open each page while signed in and press the agree button, then add the manifest to a
preparation run:

| Page to open | Sources | Stream | Note |
| --- | --- | --- | --- |
| https://huggingface.co/datasets/lmsys/chatbot_arena_conversations | 2 | chat | Small (33k conversations). |
| https://huggingface.co/datasets/Amod/mental_health_counseling_conversations | 1 | chat | Manual approval, Responsible AI licence. |
| https://huggingface.co/datasets/allenai/WildChat-nontoxic | 1 | chat | AI2 ImpACT licence. |
| https://huggingface.co/datasets/allenai/WildChat-4.8M-Full | 1 | chat | Manual approval. |
| https://huggingface.co/datasets/allenai/wildguardmix | 1 | chat | Includes harmful prompts with refusals. |
| https://huggingface.co/datasets/allenai/wildjailbreak | 1 | chat | Same. |
| https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1 | 3 | synthetic_documents | Manual approval; read as plain documents, not dialogues. |
| https://huggingface.co/datasets/openbmb/UltraData-SFT-2605 | 12 | chat and synthetic_reasoning | The column names were taken from the sibling open set and have not been seen on real rows of this one. |
| https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories | 1 | verified_trajectories | Tool-use trajectories from the Toolathlon benchmark; do not train and evaluate on it. |

The gated manifest's tokens are 15.6B chat, 67.5B reasoning, 0.15B trajectories and 336.6B
documents. The Toolathlon-Verified set was not added because its card forbids use for training.

### Duplicates between the new sources

Near-duplicate removal happens during preparation, so overlap does not need to be edited out
of the manifest. The families that repeat the same prompts (so the token sums above overstate
what is unique) are:

- the AM-Team sets (R1-0528, Qwen3, thinking, 1.4M, 40M);
- Cascade stage 1 and AceReason (row for row), which also overlap OpenMathReasoning,
  OpenCodeReasoning and Post-Training v1;
- the three DeepMath solution sets, which share their questions;
- Dolci 7B, 32B, Python and synthetic-2; Dolci-translated and Updesh-R are translations of
  them;
- the Magpie series, which share prompts across teachers;
- SmolTalk subsets and SmolTalk itself; Magicoder-Evol and the OpenCoder evolution set;
  MetaMathQA inside MATH-plus;
- Nemotron multilingual, which translates the same English problems;
- WildChat-1M inside the earlier `wildchat-4.8m` source, the Aya human set inside
  `aya_collection`, and the ADP v2 set inside the first ADP collection;
- the decontaminated copies in Post-Training v2, which repeat v2 multilingual.

### Preparing these sources

`configs/cloud/prepare-agentic-more.json` is a copy of the first agentic job for this manifest:
one 16-CPU, 32 GiB sandbox, no GPU, 24 hours. Nothing launches by itself. It splits
`sources.agentic-more.json` into `AGENTIC_MORE_SHARDS` (16) shards with
`scripts/shard_sources.py`, multiplies every cap by `AGENTIC_MORE_ROW_SCALE` (0.25) and
writes `out/prepared/agentic-more-<i>`. Set `AGENTIC_MORE_MANIFEST` to prepare a different
manifest, such as the gated one after the clicks.

The caps sum to 2.26M rows and about 32.5 GB of source text. At the 25 KB per second measured for
the first manifest, the slowest of 16 shards would take about 22 hours at scale 1 and about
6 hours at 0.25, which is why 0.25 is the default; raise it only after a smoke run.

Memory is the one real risk. Sources stored as a single JSON array are loaded whole before
rows are read. Thirteen sources were dropped for being too large for that (among them
step3.5 flash SFT, Light-R1 stage 1, the Belle sets, the Glaive code assistant). A few that
remain are between 250 and 450 MB as one array (WizardLM V2, MetaMathQA, Tested-143k) and
should fit in 32 GiB, but they are the first thing to check if a shard dies.

### What was checked

- `tests/test_agentic_more_sources.py` loads the manifests and checks that every source is
  pinned to a full revision, carries a recorded licence with `approved: true` and the review
  fields, has a unique id (also unique against the earlier manifests), names a stream in
  `stages.json`, and that the cloud job validates. For every source it builds a fixture row in
  its shape, runs its adapter, and checks the stream, that training masks fall only on assistant (or document) records, and that tool results are marked executed or simulated as recorded.
- Probing: every source was read from its pinned revision through the real reader for at
  least 6 to 12 rows when the network allowed (the Hub limits to 1000 requests per five minutes
  and the uplink was slow). Sources the readers could not parse were either given a new reader
  above or dropped; the dropped ones are in the next list.
  Final real-row runs (8 rows each) passed for the Anthropic helpful-base, UltraChat full,
  LMSYS GPT-5, Nebius gpt-oss, Dolphin-R1, Llama-Nemotron chat, SFT-datablend, MOSS-003,
  Synth-APIGen (both), Glaive v1 (260 observation positions, 10 simulated results) and Toucan
  (90 executed results) sources. SODA and AceMath could not be streamed within the 15-minute
  limit on this connection (their files are 0.7 GB and larger); they are covered by the
  fixture tests only.

Size figures are approximations from the dataset size on the Hub divided by four. The Dolci 7B
and 32B figures and the r1-distill figure vary widely, and the Updesh figures are guesses.
Some community distills claim a teacher model that cannot be checked.

### Considered and not added

- **Screenshot-only GUI agents**: AITW, AndroidControl, GUI-Odyssey, OS-Atlas, Aguvis, OSWorld
  trajectories, WebSTAR, ProCUA, MolmoWeb. They need images.
- **Need another reader first** (the best follow-ups): OASST message trees (the top-ranked
  path is already present), Nemotron-SFT-OpenCode, SWE-chat and pi sessions, agentlogs,
  MEnvData parallel calls, smolagents tool-calling, the Nemotron-RL pivot sets, ToolBench ReAct
  (about three quarters of rows rejected by the current ReAct reader), AgentNet, WebLINX,
  ChatQA, HelpSteer3, CantTalkAboutThis, natural_reasoning, CodeIO, llm-jp thinking
  (channel markup), the Apertus mix, INTELLECT-3 re-mixes, Cascade-2 XML tool calls,
  CoT-Collection, roleplay in the Zephyr template, PersonaChat, RAG-Instruct, MOSS-002, and
  the LlamaFactory reasoning tool-use set (blocks keyed by `value`).
  Since then the Nemotron-RL pivot sets, ToolBench (ToolLLaMA), smolagents tool-calling,
  Cascade-2 XML tool calls and the LlamaFactory tool-use sets are read; see "Agentic chat at
  full volume".
- **Prompt-only or reward-model sets**, benchmarks and preference pairs whose rejected side is
  the only answer.
- **Withheld problem text**: Post-Training v1 code, OpenCodeReasoning (1 and 2) and
  Competitive-Programming v1 python carry only a reference to the problem text.
- **Oversized single JSON arrays and row groups**: see the memory note above.
- **Forbidden or contaminated**: Toolathlon-Verified (card forbids training), Terminal-Bench
  derived sets (the benchmark's tasks).
- **Duplicates and subsets** of sources already in an earlier manifest, and tiny single-text sets.

### Unique tokens against what the stage plan reads

For each stream, "unique tokens" are the sum of the manifests (before: the earlier open,
pilot and first agentic manifests; added: this section, before deduplication). The planned
columns are tokens the stage plan (`stages.json`) draws for that stream: the mixture share
times the phase token limit, in billions, for the phases P1, M1, M2 and LC. "Times each token
is seen" is planned divided by unique after; below 1 the stream never repeats a token, above 1
it is repeated.

| Stream | Unique before (B) | Added (B) | Unique after (B) | P1 | M1 | M2 | LC | Planned total (B) | Times each token is seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| executed_trajectories | 47.2 | 27.62 | 74.83 | 68.0 | 12.6 | 3.2 | 2.0 | 85.8 | 1.15x |
| verified_trajectories | 20.9 | 2.23 | 23.13 | 20.0 | 4.8 | 19.6 | 2.0 | 46.4 | 2.01x |
| tool_use | 13.0 | 7.16 | 20.16 | 24.0 | 2.4 | 2.8 | 1.0 | 30.2 | 1.50x |
| chat | 10.2 | 121.88 | 132.08 | 32.0 | 1.2 | 0.8 | - | 34.0 | 0.26x |
| synthetic_reasoning | 23.1 | 799.71 | 822.81 | 40.0 | - | - | - | 40.0 | 0.05x |
| synthetic_code_dialogue | - | 22.55 | 22.55 | 32.0 | - | - | - | 32.0 | 1.42x |
| environment_trajectories | 0.1 | 0.71 | 0.81 | 0.4 | - | - | - | 0.4 | 0.50x |
| mixed_quality_episodes | 12.0 | 0.00 | 12.00 | - | - | - | 4.0 | 4.0 | 0.33x |
| typed_exact_instructions | 0.28 | 1.54 | 1.82 | not planned in tokens | | | | | |
| recovery_clarification | 0.007 | 0.0015 | 0.01 | not planned in tokens | | | | | |

What this means:

- **Chat and reasoning now far exceed the plan.** The plan reads each unique chat token a
  quarter of the time and reasoning one twentieth. Deduplication will lower the real numbers (the overlap
  families above are large), but the gap to the plan is wide enough that the mixture can be
  chosen by quality rather than by what exists, and the plan's chat and reasoning shares could be
  raised without repeating data. How much remains is known only after a preparation run.
- **Tool use** fell from 2.3 to 1.5 passes. It is still repeated; most of its results are
  simulated (see below).
- **Executed trajectories** are about enough (1.15 passes); verified trajectories still repeat
  twice and need coding-side sources from the other worker.
- **recovery_clarification** (failed-then-fixed runs, asking for missing arguments) is
  still almost absent: one tiny source added. Planned positions are counted separately.

### Still missing

- Real (executed) tool results outside coding: only the Toucan MCP sets and the web-agent
  sets provide them. Every other function-calling set here is simulated.
- Recovery and clarification data.
- Non-English agent and tool-use traces beyond the Chinese and translated sets.
- The readers listed under "Need another reader first".

## Agentic chat at full volume

This section covers assistants that use tools or act in an environment without writing code
as the main task: function calling, MCP servers, web search and deep research, customer-service
agents in the style of tau-bench, text-only web and app agents, planning, and multilingual tool
dialogues. A sweep of the Hub for such sets added 169 open sources
(`configs/data/sources.agentic-chat.json`) and 5 that need a click on the Hub
(`configs/data/sources.agentic-chat-gated.json`). Every source is pinned to a full revision and
read at full volume. Any licence is accepted and recorded in `terms` (`approved: true`,
reviewed by Gabriel Pickard on 2026-10-22); where a card gives no licence the entry says so.
Licence flags worth knowing: the MiroVerse set and two Taiwanese sets (simon-tw-finance, twi)
are non-commercial on the card, the Simia sets are non-commercial in their README only, the
thirteen `dfm13-tooldialogue-*` sets say "no blanket licence grant", and 42 sets have no
licence on the card.

Coding trajectories were left to the coding sweep. The sets that contain both were split by
file, subset or a row filter.

### New row shapes

Each shape below is a profile in `src/megatoken/data/adapters.py` with its reading code in
`src/megatoken/data/dialects.py`; all have tests in `tests/test_agentic_adapters.py`.

| Profile | Shape it reads |
| --- | --- |
| `harmony`, `harmony_tools` | gpt-oss "harmony" channel turns with tool recipients (the rl-rag and Nemotron-RL families) |
| `tool_dict`, `responses`, `call_list` | calls stored as one dictionary per call, OpenAI Responses items, and lists of calls answered by lists of results (BUTTON) |
| `device_calls` | tau2 style device calls where the final answer is a separate field and thinking is folded into `<think>` |
| `environment_roles`, `react_function_agent` | ToolLLaMA style replies from a `function` role |
| `tag_trace` | search, crawl and Python calls written between tags in one assistant output, with result tags (Tool-Star, AFM) |
| `hermes_user_results` | Hermes calls whose results come back in a user turn (Simia tau) |
| `glaive_messages`, `reasoning_roles` | Glaive calls inside messages, and reasoning-role traces |
| hermes dialect | XML `<function=NAME><parameter=KEY>` bodies, several results in one tool message, packed JSON lists of results, `PythonInterpreter` code bodies, calls quoted inside thinking ignored |
| Python-call dialect | fenced call blocks, non-JSON values such as `datetime(...)` kept as text |
| call-role dialect | roles spelled `tool call` or `function-call`, a list of calls in one message, LlamaFactory `<think>..<tool_call>` text, and arguments written as a single-quoted JSON string |
| chat reader | JSON-string messages decoded, empty user turns dropped when asked, tool parameters stored as a JSON string, xLAM flat parameter maps wrapped into a JSON Schema object |

Two fixes outside the readers: a heading is never attached across the next heading in the text
parser (long OpenClaw system prompts were held out before; ClawGym went from 0 of 32 to 32 of
32 accepted), and the engine's seam diagnostics no longer reject valid rows. In the existing
manifests, `api-bank-train` now takes its family from the `input` column (5 of 400 became 400
of 400), `pythonic-function-calling` uses the Python-call reader (0 of 200 multi-turn rows
became 189), the Nemotron Post-Training tool-calling entry reads all 13 tool files (about
310k rows instead of one file) and `ultradata-agent-tool-use` filters out its copy of ToolMind.

### How the sources were checked

For every source a sample of up to 400 rows was read from the pinned revision and run through
the real `megatoken prepare` with the manifest's adapter, field map and options. Outcomes per
sample row are: accepted; rejected (the adapter could not read the row); filtered (the
manifest's own row filter dropped it on purpose); duplicate or quarantined (the preparation
step dropped a repeat of an earlier row, or capped many rows that share one first user prompt);
and held out (the parser sent the episode to the review queue instead of writing it).
Across the 169 sources, 44,374 sample rows gave 40,740 accepted episodes, 1,075 filtered rows,
830 duplicates, 1,662 capped or quarantined, 26 held out and 41 rejected. Of the 43,299 rows
not removed by a manifest filter, 41 (0.09%) were rejected, and 5 of those are rows the sample
download cut off (aman-rlvr rows are 38 KB each and the sample limit truncated half of them).
No source is above 5% except that truncated one. Before the last fixes four were: a Taiwanese
and a Portuguese set whose tools use the xLAM flat parameter spelling (8.5% each), a Korean
Glaive translation (6.8%) and a Vietnamese one (7.3%) (both with single-quoted arguments).

The ds4-agentic-tool-calls source needs a caution: its roughly 5,000 rows have only 8 distinct
task prompts, so the per-prompt cap keeps a small share (5 of 300 in the sample). The
secai cybersecurity set starts with a cluster of repeated prompts; a sample spread across the
whole file was accepted 300 of 300.

### Sources

Rows and text are estimates at full volume. Rows come from the Parquet footers, or from the
stored file size divided by the size of a sample row; text is rows times the mean serialized row
size, which is larger than the stored (compressed) size. Expected accepted is that count times
the sample's accepted share (accepted divided by the rows the manifest's filter keeps).
"Sample" columns are accepted, rejected and held-out counts out of the sample rows.
Because samples are the first rows of a file, shares for sources with repeated prompts
at the top are conservative.

| Source id | Hub dataset | Licence | Rows | Text | Sample: accepted / rejected / held out (of sample rows) | Expected accepted rows | Expected accepted text |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nex-agent-sft-tool-calling` | `nex-agi/agent-sft` | odc-by | 57k | 448.9 MB | 400 / 0 / 0 of 400 | 57k | 448.9 MB |
| `nex-agent-sft-deep-research` | `nex-agi/agent-sft` | odc-by | 14k | 2.7 GB | 21 / 0 / 0 of 21 | 14k | 2.7 GB |
| `nex-agent-sft-agent` | `nex-agi/agent-sft` | odc-by | 10k | 1.6 GB | 24 / 0 / 1 of 25 | 9k | 1.6 GB |
| `miroverse-v01-sft` | `miromind-ai/MiroVerse-v0.1` | cc-by-nc-4.0 | 150k | 8.3 GB | 73 / 0 / 0 of 73 | 150k | 8.3 GB |
| `nexus-agents-toolcalling` | `NexusProjectsAI/Nexus-Agents-ToolCalling` | apache-2.0 | 600k | 6.2 GB | 387 / 0 / 0 of 390 | 596k | 6.1 GB |
| `midtool-native-agent-traj` | `RedMod/MidTool-Mix-shuffled` | midtool-mix-license | 425k | 18.2 GB | 13 / 0 / 0 of 94 | 59k | 2.5 GB |
| `r0b0tlab-multiteacher-sft-tools` | `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation` | other | 5k | 46.7 MB | 384 / 16 / 0 of 400 | 5k | 44.8 MB |
| `ds4-agentic-tool-calls` | `r0b0tlab/deepseek-v4-pro-0813-agentic` | other | 5k | 29.7 MB | 5 / 0 / 0 of 300 | 83 | 494.2 KB |
| `ds4-agentic-dialogue-planning` | `r0b0tlab/deepseek-v4-pro-0813-agentic` | other | 3k | 17.3 MB | 241 / 0 / 0 of 400 | 3k | 17.3 MB |
| `zhangdw-tau-train` | `zhangdw/to-tool-call-datasets` | apache-2.0 | 7k | 116.6 MB | 137 / 0 / 0 of 224 | 4k | 71.3 MB |
| `zhangdw-looptool-23k` | `zhangdw/to-tool-call-datasets` | apache-2.0 | 24k | 443.3 MB | 217 / 0 / 4 of 221 | 24k | 435.3 MB |
| `zhangdw-hardgen-bfcl` | `zhangdw/to-tool-call-datasets` | apache-2.0 | 8k | 199.6 MB | 105 / 0 / 0 of 152 | 5k | 137.9 MB |
| `zhangdw-buttoninstruct` | `zhangdw/to-tool-call-datasets` | apache-2.0 | 18k | 117.6 MB | 397 / 0 / 0 of 400 | 18k | 116.7 MB |
| `deep-research-sft-qwen35` | `1stprinciple/deep-research-sft-qwen35` | none on card | 80k | 4.5 GB | 45 / 0 / 0 of 72 | 50k | 2.8 GB |
| `huggingenvs-data-agent-sft` | `HuggingEnvs/data-agent-sft` | mit | 5k | 35.7 MB | 400 / 0 / 0 of 400 | 5k | 35.7 MB |
| `clawgym-trajectory` | `RUC-AIBOX/ClawGym-Trajectory` | none on card | 24k | 2.3 GB | 43 / 0 / 0 of 43 | 24k | 2.3 GB |
| `davinci-tooldata` | `SII-Enigma/Davinci-ToolData` | other | 15k | 626.5 MB | 93 / 0 / 0 of 93 | 15k | 626.5 MB |
| `agent-reflex-sft-54k` | `dongguanting/Agent-Reflex-SFT-54K` | mit | 55k | 2.8 GB | 81 / 0 / 0 of 81 | 55k | 2.8 GB |
| `agent-world-sft-44k` | `dongguanting/Agent-World-SFT-44K` | none on card | 29k | 2.0 GB | 61 / 0 / 0 of 61 | 29k | 2.0 GB |
| `stindardlogic-tool-calling-english-100k` | `stindardlogic/tool-calling-english-100k` | apache-2.0 | 98k | 178.9 MB | 88 / 0 / 0 of 400 | 22k | 39.4 MB |
| `qwen36plus-toolscale-agent` | `zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory` | none on card | 4k | 149.5 MB | 24 / 0 / 0 of 106 | 894 | 33.9 MB |
| `deepseekv4pro-toolscale-agent` | `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory` | none on card | 3k | 84.3 MB | 23 / 0 / 0 of 120 | 481 | 16.2 MB |
| `secai-agentic-rag-cyber` | `DuyTa/secai-agentic-rag-sft-v1` | other | 122k | 504.7 MB | 300 / 0 / 0 of 300 | 122k | 504.7 MB |
| `rag-ecommerce-tool-use-sft` | `lazos/frontend-agent-sft` | other | 26k | 118.4 MB | 390 / 0 / 0 of 400 | 25k | 115.5 MB |
| `sme-docdata-tool-use` | `hotdogs/sme-sft-docdata-qwen38` | apache-2.0 | 9k | 42.2 MB | 387 / 0 / 0 of 400 | 9k | 40.8 MB |
| `agent-society-minecraft-distill` | `SebastianAldrin/agent-society-distill-v1` | mit | 1k | 15.8 MB | 358 / 0 / 0 of 359 | 1k | 15.8 MB |
| `cascade2-conversational-agent` | `nvidia/Nemotron-Cascade-2-SFT-Data` | nvidia-open-model-license | 841k | 17.5 GB | 193 / 0 / 0 of 193 | 841k | 17.5 GB |
| `mobile-actions` | `google/mobile-actions` | cc-by-4.0 | 10k | 25.8 MB | 400 / 0 / 0 of 400 | 10k | 25.8 MB |
| `smolagents-toolcalling-planning` | `smolagents/toolcalling` | none on card | 7k | 331.6 MB | 89 / 0 / 0 of 89 | 7k | 331.6 MB |
| `funcdex-mt` | `prem-research/Funcdex-MT-Function-Calling` | mit | 1k | 35.5 MB | 168 / 0 / 0 of 168 | 1k | 35.5 MB |
| `kona-ka-93k` | `tbilisi-ai-lab/kona-sft-function-calling-ka-93k` | odc-by | 76k | 280.5 MB | 400 / 0 / 0 of 400 | 76k | 280.5 MB |
| `simon-kubectl-mcp` | `Simon-Liu/kubectl-mcp-server-tool-call-reasoning-6k` | apache-2.0 | 5k | 81.2 MB | 249 / 0 / 0 of 249 | 5k | 81.2 MB |
| `simon-tw-finance` | `Simon-Liu/tw-finance-function-call-reasoning` | cc-by-nc-4.0 | 1k | 6.7 MB | 400 / 0 / 0 of 400 | 1k | 6.7 MB |
| `twinkle-tw-fc-reasoning` | `twinkle-ai/tw-function-call-reasoning-10k` | cc-by-4.0 | 10k | 52.0 MB | 399 / 0 / 1 of 400 | 10k | 51.8 MB |
| `zake-qwen36-tool-calling` | `zake7749/Qwen3.6-35B-A3B-Tool-Calling` | none on card | 36k | 0 B | 275 / 0 / 2 of 300 | 33k | 0 B |
| `toolbench-llama-g123-dfs` | `Adorg/ToolBench` | apache-2.0 | 26k | 259.4 MB | 399 / 0 / 1 of 400 | 26k | 258.7 MB |
| `bitagent-tool-calling` | `BitAgent/tool_calling` | none on card | 551k | 552.9 MB | 151 / 0 / 0 of 400 | 208k | 208.7 MB |
| `pawa-swahili-xlam` | `sartifyllc/pawa-function-calling-60k` | none on card | 60k | 121.1 MB | 400 / 0 / 0 of 400 | 60k | 121.1 MB |
| `button-mt-function-calling` | `casey-martin/BUTTON-MT-function-calling` | none-on-card | 8k | 74.0 MB | 400 / 0 / 0 of 400 | 8k | 74.0 MB |
| `tau2-bench-trajectories` | `AgentSuite/tau2-bench-trajectories` | none-on-card | 7k | 292.3 MB | 36 / 0 / 0 of 94 | 3k | 111.9 MB |
| `dfm13-tooldialogue-sk` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-sk` | not stated | 4k | 26.1 MB | 398 / 0 / 0 of 400 | 4k | 26.0 MB |
| `dfm13-tooldialogue-sr` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-sr` | not stated | 4k | 27.2 MB | 393 / 0 / 0 of 400 | 4k | 26.7 MB |
| `dfm13-tooldialogue-lb` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-lb` | not stated | 11k | 67.9 MB | 399 / 0 / 0 of 400 | 11k | 67.7 MB |
| `dfm13-tooldialogue-sl` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-sl` | not stated | 4k | 27.4 MB | 393 / 0 / 0 of 400 | 4k | 26.9 MB |
| `dfm13-tooldialogue-hu` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-hu` | not stated | 4k | 27.0 MB | 395 / 0 / 0 of 400 | 4k | 26.7 MB |
| `dfm13-tooldialogue-fa` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-fa` | not stated | 4k | 27.1 MB | 397 / 0 / 0 of 400 | 4k | 26.9 MB |
| `dfm13-tooldialogue-bs` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-bs` | not stated | 4k | 26.5 MB | 399 / 0 / 0 of 400 | 4k | 26.4 MB |
| `dfm13-tooldialogue-hr` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-hr` | not stated | 4k | 26.8 MB | 397 / 0 / 0 of 400 | 4k | 26.6 MB |
| `dfm13-tooldialogue-sq` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-sq` | not stated | 4k | 27.7 MB | 380 / 0 / 0 of 400 | 4k | 26.4 MB |
| `dfm13-tooldialogue-bg` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-bg` | not stated | 4k | 26.1 MB | 399 / 0 / 0 of 400 | 4k | 26.1 MB |
| `dfm13-tooldialogue-be` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-be` | not stated | 4k | 26.1 MB | 388 / 0 / 0 of 400 | 4k | 25.3 MB |
| `dfm13-tooldialogue-lv` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-lv` | not stated | 4k | 26.6 MB | 396 / 0 / 0 of 400 | 4k | 26.4 MB |
| `dfm13-tooldialogue-lt` | `schneiderkamplab/dfm13-multilingual-tool-dialogue-lt` | not stated | 4k | 26.5 MB | 400 / 0 / 0 of 400 | 4k | 26.5 MB |
| `dfm14-tooldialogue-zh` | `schneiderkamplab/dfm14-zh-synthetic-tool-dialogue-r1` | other | 2k | 15.6 MB | 400 / 0 / 0 of 400 | 2k | 15.6 MB |
| `dfm14-tooldialogue-id` | `schneiderkamplab/dfm14-id-synthetic-tool-dialogue-r1` | other | 2k | 15.8 MB | 397 / 0 / 0 of 400 | 2k | 15.7 MB |
| `dfm14-tooldialogue-vi` | `schneiderkamplab/dfm14-vi-synthetic-tool-dialogue-r1` | other | 2k | 15.7 MB | 400 / 0 / 0 of 400 | 2k | 15.7 MB |
| `dfm14-tooldialogue-ru` | `schneiderkamplab/dfm14-ru-synthetic-tool-dialogue-r1` | other | 2k | 15.1 MB | 400 / 0 / 0 of 400 | 2k | 15.1 MB |
| `dfm14-tooldialogue-mk` | `schneiderkamplab/dfm14-mk-synthetic-tool-dialogue-r1` | other | 4k | 31.6 MB | 400 / 0 / 0 of 400 | 4k | 31.6 MB |
| `dfm14-tooldialogue-tr` | `schneiderkamplab/dfm14-tr-synthetic-tool-dialogue-r1` | other | 2k | 15.5 MB | 400 / 0 / 0 of 400 | 2k | 15.5 MB |
| `dfm14-tooldialogue-ar` | `schneiderkamplab/dfm14-ar-synthetic-tool-dialogue-r1` | other | 2k | 15.7 MB | 399 / 0 / 0 of 400 | 2k | 15.7 MB |
| `dfm14-tooldialogue-ga` | `schneiderkamplab/dfm14-ga-synthetic-tool-dialogue-r1` | other | 4k | 30.7 MB | 396 / 0 / 0 of 400 | 4k | 30.4 MB |
| `dfm14-tooldialogue-hi` | `schneiderkamplab/dfm14-hi-synthetic-tool-dialogue-r1` | other | 2k | 15.2 MB | 400 / 0 / 0 of 400 | 2k | 15.2 MB |
| `dfm14-tooldialogue-gl` | `schneiderkamplab/dfm14-gl-synthetic-tool-dialogue-r1` | other | 4k | 31.6 MB | 400 / 0 / 0 of 400 | 4k | 31.6 MB |
| `dfm14-tooldialogue-ko` | `schneiderkamplab/dfm14-ko-synthetic-tool-dialogue-r1` | other | 2k | 15.0 MB | 397 / 0 / 0 of 400 | 2k | 14.9 MB |
| `dfm14-tooldialogue-ja` | `schneiderkamplab/dfm14-ja-synthetic-tool-dialogue-r1` | other | 2k | 15.3 MB | 400 / 0 / 0 of 400 | 2k | 15.3 MB |
| `dfm14-tooldialogue-mt` | `schneiderkamplab/dfm14-mt-synthetic-tool-dialogue-r1` | other | 4k | 27.9 MB | 400 / 0 / 0 of 400 | 4k | 27.9 MB |
| `dfm14-tooldialogue-cy` | `schneiderkamplab/dfm14-cy-synthetic-tool-dialogue-r1` | other | 4k | 30.1 MB | 400 / 0 / 0 of 400 | 4k | 30.1 MB |
| `dfm14-tooldialogue-eu` | `schneiderkamplab/dfm14-eu-synthetic-tool-dialogue-r1` | other | 4k | 30.9 MB | 400 / 0 / 0 of 400 | 4k | 30.9 MB |
| `dfm14-tooldialogue-he` | `schneiderkamplab/dfm14-he-synthetic-tool-dialogue-r1` | other | 2k | 15.2 MB | 399 / 0 / 0 of 400 | 2k | 15.2 MB |
| `danish-tooldialogues-v9` | `jensjepsen/danish-tool-dialogues-v9` | apache-2.0 | 34k | 52.9 MB | 304 / 0 / 0 of 400 | 26k | 40.2 MB |
| `danish-tooldialogues-v9-abstention` | `jensjepsen/danish-tool-dialogues-v9` | apache-2.0 | 902 | 2.4 MB | 400 / 0 / 0 of 400 | 902 | 2.4 MB |
| `fcptpt-toolace` | `jgalego/function-calling-pt-pt` | apache-2.0 | 10k | 38.4 MB | 400 / 0 / 0 of 400 | 10k | 38.4 MB |
| `fcptpt-when2call` | `jgalego/function-calling-pt-pt` | cc-by-4.0 | 15k | 30.1 MB | 400 / 0 / 0 of 400 | 15k | 30.1 MB |
| `fcptpt-xlam` | `jgalego/function-calling-pt-pt` | cc-by-4.0 | 59k | 130.0 MB | 400 / 0 / 0 of 400 | 59k | 130.0 MB |
| `glaive-v2-ru-korotkov` | `korotkov/glaive-function-calling-v2-ru-parsed` | not stated | 102k | 348.5 MB | 329 / 0 / 0 of 400 | 84k | 286.6 MB |
| `glaive-v2-ko-refined` | `tryumanshow/glaive-function-calling-v2-ko-refined` | not stated | 12k | 44.3 MB | 393 / 2 / 0 of 400 | 12k | 43.6 MB |
| `glaive-v2-ja-nappa` | `nappa0326/glaive-function-calling-v2-sharegpt-japanese` | mit | 113k | 379.3 MB | 354 / 0 / 0 of 400 | 100k | 335.7 MB |
| `toolace-ja-nntsuzu` | `nntsuzu/ToolACE-Japanese` | apache-2.0 | 5k | 26.1 MB | 398 / 2 / 0 of 400 | 5k | 26.0 MB |
| `twi-translation-toolcalling` | `ghananlpcommunity/twi-translation-tool-calling` | cc-by-nc-4.0 | 16k | 307.4 MB | 209 / 0 / 0 of 209 | 16k | 307.4 MB |
| `oddadmix-ar` | `oddadmix/tool-calling-ar-augmented` | none-on-card | 547k | 1.8 GB | 151 / 0 / 0 of 400 | 207k | 670.0 MB |
| `heegyu-glaive-ko` | `heegyu/glaive-function-calling-v2-ko` | apache-2.0 | 15k | 50.4 MB | 355 / 0 / 0 of 400 | 13k | 44.8 MB |
| `vi-locutusque` | `5CD-AI/Vietnamese-Locutusque-function-calling-chatml-gg-translated` | none-on-card | 255k | 0 B | 142 / 0 / 0 of 150 | 242k | 0 B |
| `dp-ja` | `DataPilot/Function-Calling-Japanese-Dataset` | odc-by | 7k | 14.7 MB | 336 / 1 / 0 of 400 | 6k | 12.3 MB |
| `giux78-ita` | `giux78/functioncalling-ita` | none-on-card | 113k | 256.5 MB | 348 / 0 / 0 of 400 | 98k | 223.2 MB |
| `aman-rlvr` | `AmanPriyanshu/tool-reasoning-sft-RESEARCH-rlvr-env-retrieval-source` | apache-2.0 | 112k | 4.3 GB | 5 / 0 / 0 of 5 (5 more cut off) | 112k | 4.3 GB |
| `ipf-deepresearch-traj` | `IPF/DeepResearch-traj` | mit | 99k | 27.6 GB | 8 / 0 / 0 of 15 | 53k | 14.7 GB |
| `rlrag-browsecomp-gptoss-v2` | `rl-rag/browsecomp-gpt-oss-120b-v2` | none on card | 1k | 287.4 MB | 18 / 0 / 0 of 18 | 1k | 287.4 MB |
| `rlrag-browsecomp-high-effort-full` | `rl-rag/browsecomp-high-effort-full-gpt-oss-120b` | none on card | 1k | 267.3 MB | 19 / 0 / 0 of 19 | 1k | 267.3 MB |
| `rlrag-browsecomp-oss-env-high-effort` | `rl-rag/browsecomp-oss-env-high-effort-gpt-oss-120b` | none on card | 1k | 246.4 MB | 21 / 0 / 0 of 21 | 1k | 246.4 MB |
| `rlrag-drtulu-step35-websearch` | `rl-rag/drtulu_v2_step_35_sft_web_search` | none on card | 999 | 174.3 MB | 24 / 0 / 0 of 24 | 999 | 174.3 MB |
| `rlrag-webshaper-gptoss` | `rl-rag/webshaper-gpt-oss-120b-260222` | none on card | 2k | 225.2 MB | 36 / 0 / 0 of 36 | 2k | 225.2 MB |
| `rlrag-bc-synthetic-gptoss` | `rl-rag/bc-synthetic-gpt-oss-120b-260222` | none on card | 10k | 1.0 GB | 39 / 0 / 0 of 39 | 10k | 1.0 GB |
| `rlrag-monaco-gptoss` | `rl-rag/monaco-gpt-oss-120b-260222` | none on card | 5k | 607.4 MB | 36 / 0 / 0 of 36 | 5k | 607.4 MB |
| `rlrag-personalfinance-gptoss` | `rl-rag/personalfinance-gpt-oss-120b` | none on card | 19k | 2.3 GB | 34 / 0 / 0 of 34 | 19k | 2.3 GB |
| `rlrag-open-scholar-gptoss` | `rl-rag/open-scholar-gpt-oss-120b` | none on card | 12k | 1.4 GB | 35 / 0 / 0 of 35 | 12k | 1.4 GB |
| `rlrag-nemotron-mcqa-gptoss` | `rl-rag/nemotron-mcqa-gpt-oss-120b-260222` | none on card | 8k | 631.5 MB | 50 / 0 / 0 of 50 | 8k | 631.5 MB |
| `rlrag-search-arena-gptoss` | `rl-rag/search-arena-gpt-oss-120b` | none on card | 2k | 157.7 MB | 55 / 0 / 1 of 56 | 2k | 154.9 MB |
| `rlrag-arena-expert-gptoss` | `rl-rag/arena-expert-gpt-oss-120b` | none on card | 3k | 285.1 MB | 41 / 0 / 0 of 41 | 3k | 285.1 MB |
| `rlrag-bc-synthetic-extra-hard-gptoss` | `rl-rag/bc-synthetic-extra-hard-gpt-oss-120b` | none on card | 820 | 0 B | 38 / 0 / 0 of 38 | 820 | 0 B |
| `rlrag-bc-synthetic-test-gptoss` | `rl-rag/bc-synthetic-test-gpt-oss-120b` | none on card | 844 | 0 B | 44 / 0 / 0 of 44 | 844 | 0 B |
| `openseeker-v1` | `OpenSeeker/OpenSeeker-v1-Data` | mit | 0 | 0 B | 20 / 0 / 0 of 27 | 0 | 0 B |
| `s1-deepresearch-15k` | `ScienceOne-AI/S1-DeepResearch-15k` | apache-2.0 | 0 | 0 B | 43 / 2 / 9 of 54 | 0 | 0 B |
| `browseragent-data` | `TIGER-Lab/BrowserAgent-Data` | none on card | 0 | 0 B | 400 / 0 / 0 of 400 | 0 | 0 B |
| `carr-deepdive-sft` | `THU-KEG/CaRR-DeepDive` | none on card | 0 | 0 B | 23 / 0 / 0 of 23 | 0 | 0 B |
| `zai-deepdive-trajectories` | `zai-org/DeepDive` | apache-2.0 | 858 | 0 B | 51 / 0 / 1 of 52 | 842 | 0 B |
| `acm-bcp-rollouts-epoch2` | `lixiaochuan2020/acm-browsecompplus-train-rollouts-qwen3.5-9b-epoch2` | none on card | 0 | 0 B | 6 / 0 / 0 of 8 | 0 | 0 B |
| `acm-bcp-rollouts-epoch1` | `lixiaochuan2020/acm-browsecompplus-train-rollouts-qwen3.5-9b-epoch1` | none on card | 0 | 0 B | 6 / 0 / 0 of 8 | 0 | 0 B |
| `bcp-search-agent-trajectories` | `liuqi6777/bcp_search_agent_trajectory` | apache-2.0 | 0 | 0 B | 12 / 0 / 1 of 13 | 0 | 0 B |
| `rlrag-drtulu-unified` | `rl-rag/dr-tulu-sft-unified` | odc-by | 0 | 0 B | 107 / 3 / 0 of 110 | 0 | 0 B |
| `rlrag2-oss-high-drtulu-v5` | `rl-rag-2/oss-high-drtulu-v5-0422-8rollouts` | mit | 0 | 0 B | 19 / 0 / 0 of 22 | 0 | 0 B |
| `searchr1-sft-qwen25` | `ThornZ/Search-R1-SFT` | none on card | 0 | 0 B | 400 / 0 / 0 of 400 | 0 | 0 B |
| `tau-bench-agentsuite` | `AgentSuite/tau-bench-trajectories` | none-on-card | 0 | 0 B | 98 / 2 / 0 of 165 | 0 | 0 B |
| `areal-tau2-sft` | `inclusionAI/AReaL-tau2-data` | apache-2.0 | 0 | 0 B | 103 / 0 / 2 of 147 | 0 | 0 B |
| `tau-retail-distill-teacher` | `teacher57/tau-retail-distillation` | apache-2.0 | 0 | 0 B | 107 / 0 / 0 of 107 | 0 | 0 B |
| `tau-retail-distill-rollouts` | `teacher57/tau-retail-distillation` | apache-2.0 | 0 | 0 B | 64 / 0 / 0 of 64 | 0 | 0 B |
| `tunekv-taubench-retail30` | `Zhongzhu/tunekv-taubench-data` | none-on-card | 0 | 0 B | 140 / 0 / 0 of 194 | 0 | 0 B |
| `tunekv-taubench-retail27` | `Zhongzhu/tunekv-taubench-data` | none-on-card | 0 | 0 B | 160 / 0 / 0 of 238 | 0 | 0 B |
| `tunekv-taubench-airline30` | `Zhongzhu/tunekv-taubench-data` | none-on-card | 0 | 0 B | 35 / 0 / 0 of 40 | 0 | 0 B |
| `tau-bench-synthetic-glm5` | `fuvty/tau-bench-synthetic` | apache-2.0 | 1k | 0 B | 82 / 0 / 0 of 131 | 916 | 0 B |
| `mcp-universe-finance-sft` | `Shuibai12138/mcp-universe-finance-sft` | none-on-card | 0 | 0 B | 222 / 0 / 0 of 222 | 0 | 0 B |
| `amap-mcp-tc130` | `TC130/amap_mcp` | none-on-card | 0 | 0 B | 397 / 0 / 0 of 397 | 0 | 0 B |
| `sato-mcp-messages` | `SatoHub/sato-mcp-tool-calling` | cc-by-4.0 | 0 | 0 B | 396 / 0 / 0 of 400 | 0 | 0 B |
| `retail-bank-servicing-sft` | `spkc83/retail-bank-servicing-alignment-sft` | mit | 0 | 0 B | 370 / 0 / 0 of 400 | 0 | 0 B |
| `adcampaign-agent-sft` | `SamWang0405/AdCampaignAgent-SFT` | mit | 0 | 0 B | 353 / 0 / 0 of 400 | 0 | 0 B |
| `home-assistant-acon96-v2` | `acon96/Home-Assistant-Requests-V2` | mit | 0 | 0 B | 323 / 0 / 0 of 323 | 0 | 0 B |
| `science-tool-use-shellsim` | `open-athena/science-tool-use-conversations` | other (OpenMDW) | 11k | 0 B | 212 / 0 / 0 of 212 | 11k | 0 B |
| `soc-agent-traces-100k` | `alirezaaminzadeh/soc-agent-traces-100k` | apache-2.0 | 90k | 0 B | 269 / 0 / 0 of 327 | 74k | 0 B |
| `soc-agentic-trajectories-miyad316` | `miyad316/cybersecurity-soc-agentic-trajectories` | cc-by-nc-4.0 | 0 | 0 B | 400 / 0 / 0 of 400 | 0 | 0 B |
| `phishing-email-soc-agent` | `Ellbendls/phishing-email-soc-agent` | apache-2.0 | 455 | 3.1 MB | 385 / 0 / 0 of 400 | 438 | 3.0 MB |
| `tau2-telecom-agent-sft` | `cy-330/tau2-telecom-agent-sft` | mit | 728 | 28.8 MB | 102 / 0 / 0 of 102 | 728 | 28.8 MB |
| `tau2-airline-code-agents` | `snorkelai/Tau2-Bench-Airline-With-Code-Agents` | apache-2.0 | 500 | 17.1 MB | 117 / 0 / 0 of 117 | 500 | 17.1 MB |
| `playwright-mcp-toolcalling` | `jdaddyalbs/playwright-mcp-toolcalling` | mit | 5k | 127.2 MB | 157 / 0 / 1 of 158 | 5k | 126.4 MB |
| `lumos-unified-plan-iterative` | `ai2lumos/lumos_unified_plan_iterative` | apache-2.0 | 53k | 72.6 MB | 390 / 0 / 0 of 400 | 52k | 70.8 MB |
| `lumos-unified-ground-iterative` | `ai2lumos/lumos_unified_ground_iterative` | apache-2.0 | 54k | 105.6 MB | 400 / 0 / 0 of 400 | 54k | 105.6 MB |
| `agentbench-sft-v3-plana` | `sabia0080/agentbench-sft-trajectories-v3-planA` | mit | 5k | 18.9 MB | 400 / 0 / 0 of 400 | 5k | 18.9 MB |
| `dbbench-spider-3500` | `mark-22/dbbench-spider-3500` | apache-2.0 | 2k | 8.9 MB | 400 / 0 / 0 of 400 | 2k | 8.9 MB |
| `hermes-agent-traces-kimi` | `lambda/hermes-agent-reasoning-traces` | apache-2.0 | 8k | 0 B | 8 / 0 / 0 of 37 | 2k | 0 B |
| `hermes-agent-traces-glm` | `lambda/hermes-agent-reasoning-traces` | apache-2.0 | 7k | 647.2 MB | 10 / 0 / 0 of 45 | 2k | 143.8 MB |
| `hermes-tool-calls-single-reasoning` | `interstellarninja/tool-calls-single-reasoning` | apache-2.0 | 16k | 113.9 MB | 398 / 2 / 0 of 400 | 16k | 113.3 MB |
| `hermes-tool-use-relevance-reasoning` | `interstellarninja/tool-use-relevance-reasoning` | apache-2.0 | 15k | 112.6 MB | 400 / 0 / 0 of 400 | 15k | 112.6 MB |
| `hermes-toolace-sequential-reasoning` | `interstellarninja/toolace_sequential_tool_use_reasoning` | apache-2.0 | 5k | 41.2 MB | 210 / 0 / 0 of 400 | 3k | 21.6 MB |
| `hermes-salesforce-hermes-tools` | `interstellarninja/salesforce_hermes_tools` | none-on-card | 3k | 225.6 MB | 45 / 0 / 0 of 45 | 3k | 225.6 MB |
| `simia-officebench-sft` | `Simia-Agent/Simia-OfficeBench-SFT-30k` | cc-by-nc-4.0 | 30k | 356.7 MB | 327 / 5 / 0 of 332 | 29k | 351.3 MB |
| `simia-agentbench-sft` | `Simia-Agent/Simia-AgentBench-SFT-15k` | cc-by-nc-4.0 | 15k | 59.9 MB | 190 / 1 / 0 of 400 | 7k | 28.4 MB |
| `simia-tau-sft-hermes` | `Simia-Agent/Simia-Tau-SFT-90k-Hermes` | cc-by-nc-4.0 | 92k | 3.6 GB | 103 / 0 / 0 of 103 | 92k | 3.6 GB |
| `smol-toolcalling-default` | `smolagents/toolcalling` | none-on-card | 9k | 322.0 MB | 115 / 0 / 0 of 115 | 9k | 322.0 MB |
| `agentbank-alfred` | `Solaris99/AgentBank` | apache-2.0 | 623 | 9.1 MB | 275 / 0 / 0 of 275 | 623 | 9.1 MB |
| `agentbank-iqa` | `Solaris99/AgentBank` | apache-2.0 | 2k | 12.3 MB | 400 / 0 / 0 of 400 | 2k | 12.3 MB |
| `agentbank-rearrange` | `Solaris99/AgentBank` | apache-2.0 | 299 | 3.2 MB | 290 / 0 / 0 of 299 | 290 | 3.1 MB |
| `agentbank-strategyqa` | `Solaris99/AgentBank` | apache-2.0 | 1k | 1.7 MB | 400 / 0 / 0 of 400 | 1k | 1.7 MB |
| `agentbank-triviaqa` | `Solaris99/AgentBank` | apache-2.0 | 4k | 4.0 MB | 400 / 0 / 0 of 400 | 4k | 4.0 MB |
| `agentbank-mathqa` | `Solaris99/AgentBank` | apache-2.0 | 4k | 4.2 MB | 400 / 0 / 0 of 400 | 4k | 4.2 MB |
| `agentbank-gsm8k` | `Solaris99/AgentBank` | apache-2.0 | 7k | 7.3 MB | 400 / 0 / 0 of 400 | 7k | 7.3 MB |
| `agentbank-math` | `Solaris99/AgentBank` | apache-2.0 | 2k | 2.9 MB | 400 / 0 / 0 of 400 | 2k | 2.9 MB |
| `agent-flan-agent-instruct-tflan` | `internlm/Agent-FLAN` | apache-2.0 | 2k | 8.5 MB | 275 / 0 / 0 of 400 | 1k | 5.8 MB |
| `agent-flan-toolbench-instruct-j1s1-3k` | `internlm/Agent-FLAN` | apache-2.0 | 7k | 17.6 MB | 400 / 0 / 0 of 400 | 7k | 17.6 MB |
| `agent-flan-toolbench-negative` | `internlm/Agent-FLAN` | apache-2.0 | 1k | 925.8 KB | 399 / 0 / 0 of 400 | 1k | 923.5 KB |
| `agent-flan-toolbench-react-10p` | `internlm/Agent-FLAN` | apache-2.0 | 2k | 18.3 MB | 400 / 0 / 0 of 400 | 2k | 18.3 MB |
| `agent-flan-toolbench-tflan-cot-30p` | `internlm/Agent-FLAN` | apache-2.0 | 7k | 47.6 MB | 400 / 0 / 0 of 400 | 7k | 47.6 MB |
| `openresearcher` | `OpenResearcher/OpenResearcher-Dataset` | mit | 6k | 1.8 GB | 15 / 0 / 0 of 15 | 6k | 1.8 GB |
| `redsearcher-sft` | `Zchu/REDSearcher_SFT_10K` | mit | 10k | 2.2 GB | 19 / 0 / 0 of 19 | 10k | 2.2 GB |
| `literesearcher-sft` | `simplex-ai-inc/LiteResearcher-SFT-Data` | apache-2.0 | 84k | 2.2 GB | 155 / 0 / 0 of 155 | 84k | 2.2 GB |
| `rlrag-browsecomp-gptoss-260222` | `rl-rag/browsecomp-gpt-oss-120b-260222` | none on card | 5k | 676.0 MB | 31 / 0 / 0 of 31 | 5k | 676.0 MB |
| `nemotron-rl-conv-tool-use-pivot` | `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | cc-by-4.0 | 97k | 1.8 GB | 77 / 0 / 0 of 221 | 34k | 612.2 MB |
| `nemotron-rl-function-calling-pivot` | `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | cc-by-4.0 | 10k | 281.6 MB | 67 / 0 / 2 of 148 | 5k | 127.5 MB |
| `tool-star-sft-54k` | `dongguanting/Tool-Star-SFT-54K` | mit | 44k | 292.7 MB | 400 / 0 / 0 of 400 | 44k | 292.7 MB |
| `afm-webagent` | `PersonalAILab/AFM-WebAgent-SFT-Dataset` | apache-2.0 | 9k | 222.5 MB | 170 / 0 / 0 of 170 | 9k | 222.5 MB |
| `afm-mhqa` | `PersonalAILab/AFM-MHQA-Agent-SFT-Dataset` | apache-2.0 | 6k | 99.9 MB | 223 / 0 / 0 of 223 | 6k | 99.9 MB |
| `tau2-airline-verified-code-agents` | `snorkelai/Tau2-Bench-Verified-Airline-With-Code-Agents` | apache-2.0 | 996 | 22.5 MB | 169 / 0 / 0 of 177 | 951 | 21.5 MB |

### Volume before and after

Text bytes here are serialized rows before tokenizing; "before" is the tool-use and agent
sources in the earlier manifests, read at full volume.

| | Sources | Rows | Text |
| --- | --- | --- | --- |
| Before: 40 sources of the main agentic manifest (measured; Nemotron Post-Training tool calling counted with all 13 files) | 40 | 3.44M | 56.4 GB |
| Before: 33 tool-use and agent sources of the second sweep (estimated from the notes; 4 bytes per token) | 33 | 0.18M | 31.4 GB |
| **Before, total** | 73 | 3.62M | 87.9 GB |
| New, read in full (before dropping unreadable and repeated rows) | 169 | 5.86M | 130.4 GB |
| New, expected to be accepted | 169 | 4.45M | 95.9 GB |
| By stream, expected accepted | tool_use 4.08M rows / 82.6 GB; environment_trajectories 0.24M / 13.1 GB; chat 0.11M / 0.19 GB; recovery_clarification 0.016M / 0.11 GB | | |
| **After (expected), total** | 242 | 8.07M | 183.8 GB |

Agentic chat text roughly doubles. The five biggest new sources by text are
ipf-deepresearch-traj (27.6 GB; only about 8 of 15 sample rows were kept), midtool-native-agent-traj
(18.2 GB before its row filter, 2.5 GB after), cascade2-conversational-agent (17.5 GB),
miroverse-v01-sft (8.3 GB) and nexus-agents-toolcalling (6.2 GB).

### Sources that need a click on the Hub

Re-checked on 2026-10-23: the first bytes of a data file of every repository below were requested again with the project token and each still answered HTTP 403, so none could be read and none was moved out of the gated manifest. (A request for a repository's README or licence file is answered even before the terms are accepted, so it is no test.)

`sources.agentic-chat-gated.json` holds five sources. The account could not read them, so each
reader was chosen from the publisher's description and has not been run on real rows. Open each
page while signed in and press Agree:

| Page to open | What it is |
| --- | --- |
| https://huggingface.co/datasets/minyichen/APIGen-MT-46k | 46k multi-turn APIGen-MT trajectories (tau-bench style) |
| https://huggingface.co/datasets/ReDiX/xlam-function-calling-60k-ita | Italian xLAM translation |
| https://huggingface.co/datasets/ura-hcmut/vi_glaive-function-calling | Vietnamese Glaive v1 |
| https://huggingface.co/datasets/ura-hcmut/vi_glaive-function-calling-v2 | Vietnamese Glaive v2 |
| https://huggingface.co/datasets/ENERZAiKR/agentic-think-v2-react-hybrid | ReAct and thinking hybrid traces |

### Preparing these sources

`files.agentic-chat.json` lists every file of every Hub source (about 101.6 GB stored) at the
pinned revisions, built like the other listings. `configs/cloud/prepare-full-agentic-chat.json`
prepares the manifest in 8 sandboxes with 28 shards each and 14 processes per sandbox, reviewing
parser holds with `parser-review.agentic.json`. The live storage trial reconciled 56,085 episodes
in about 9 minutes; a linear extrapolation to this run is about 12 hours, with candidate overlap
still uncertain. Its run file therefore allows 8 hours of ingest, 16 of reconciliation and 6
of finishing/upload, plus 2 for transfers (maximum ingest lifetime 32 hours, accepted by a
short-lived CPU probe on 2026-10-23). Earlier waves keep billing until their data is no longer
needed; these lifetimes do not keep successfully published jobs running. Nothing launches by itself; see "Preparing at
full volume". The largest stored file is the 17.5 GB cascade-2 file; single large files are the first
thing to check if a shard dies (see the memory note in the second sweep's section).

### Not added, and what is left

- Shapes no reader handles yet: WebLINX; WebArena-Lite-SFT (10k step rows that need the prompt
  composed from several fields); markov-ai/computer-use; AgentNet; DEBATE;
  scientific-agent-protocol-traces; cogym-real; HybridAgent-HybridSFT-v2 (a `rounds` list);
  smolagents code-action traces; NextSearch-1-Trajectories (messages and target split);
  Exgentic and agent-llm-traces-v2 (span format of benchmark runs); sammshen/taubench-sonnet-traces
  (raw HTTP records); ToolGym (ReAct inside JSON); MCPMark (one messages file per task);
  YWZBrandon/browsecomp-trajectories (rollouts cut off); the acm-browsecompplus base rollouts
  (5,440 single-object files); the Tool-Reasoning siblings of the mem_agent and
  browsecomp-plus-runs sets; vericava (the target column is the final message).
- Large sets with no tool results: jaeyong2 `*-functioncall` (about 2M rows of calls only).
- Column-name variants of xLAM (5CD, atasoglu) and translations with nonstandard markup
  (ituperceptron Turkish, iknow-lab Korean Hermes with literal `\n`, Tuguberk Turkish Hermes,
  ReDiX function-calling-ita with 41% rejects).
- Benchmarks and evaluation sets, left out to avoid training on them: tau-bench BFCL
  trajectories, MCP-Atlas, WildClaw, MCPMark tasks, Toolathlon.
- Image-based sets: Omnimodal-Agent-SFT-2K, CollovLabs/mm-toolsandbox-traces.
- Not opened or unknown shape: https://huggingface.co/datasets/dipta007/APIGen-MT-5k-with-cot-v1-deepseek_deepseek and
  https://huggingface.co/datasets/KKACHI-HUB/glaive-function-calling-ko.
- Partly read: UltraData General_Agent Claw and Office subsets (Office rows are rejected with
  "Unlinked observation" in 23% of cases).

### Changes the trajectory-linking code would need

These were found while converting and are not made here, because that code serves the coding
trajectories as well:

1. **Explicit call ids that repeat.** Vikhr's tool-plannings numbers every call in a turn
   `call_1`, `call_2`, ... again each turn; the builder reports "Tool call ID reused". When a
   source sets `positional_call_ids` the builder should let position override an explicit id.
2. **One result for several parallel calls.** Dolci tool-use and Hermes function-calling
   sometimes answer two or three calls with one tool message; the builder rejects those rows
   ("Cannot split a tool result into N results", "N tool result(s) for N pending call(s)").
   Splitting by call order, as the packed-result reader already does for Hermes, would keep them.
3. **Unlinked observations.** UltraData's Office rows and a few Glaive translations have a tool
   result with no preceding call; the builder says "Unlinked observation; call IDs may be
   inferred only for exactly one pending call". Allowing an observation to attach to the
   previous call when exactly one call was made in its turn would keep most of them.
4. **Tool turns that end the episode on a call.** Several search agents stop on an action whose
   result is missing ("Trace ends with an action whose observation is missing"). The option
   `allow_pending_final_action` already handles it per source; making it the default for chat
   streams would remove an option.

tau2 sets did not need builder changes: the device-call profile pairs their calls and results.

## Later-stage streams (M1, M2, LC, S, and two P1 streams)

The agentic section above feeds the trajectory streams. This section fills the streams that
the later stages read and that were still empty or thin: connected repository, issue and
pull-request (PR) data, change histories, logs, long documents, long dependencies, grounded
artifacts, clarification and recovery, tested code, and the two P1 streams that had no source
(`synthetic_code_rewrite`) or a thin one (`environment_trajectories`).

All of it is public data. Nothing was generated for this round: where a publisher's own
pipeline used a model to write rows, the table and the entry's `notes` say so. Licences are
those printed on each publisher's card, read on 2026-10-22 (a card is not legal advice);
every source is recorded in `configs/data/sources.later-stages.json` with `approved: true`,
`reviewed_by: Gabriel Pickard`, `reviewed_at: 2026-10-22`, and none was left out for its
licence. Non-commercial, share-alike, research-only and model-licence terms are named in the
tables and in each entry's `terms.notes`.

Every source is pinned to a 40-character dataset commit and, where a repository holds many
files, to the exact files read. Entries were run through their real adapter on the first rows
of the pinned files (see "What was checked for the later stages"). The first-run caps are row counts read from the
start of the pinned files before filters, sized to roughly 30 to 150 million tokens per source.

### New row shapes

Four things the earlier adapters could not express were added, each one code path:

| What | How it works | Used by |
| --- | --- | --- |
| Lists of files and passages | In a `repository` row the request is followed by `hints` (discussion), `before` (files as they were) and then the scored `patch`. Each of the three can be text, a list of strings, or a list of objects with a path and content (`content`, `before_content`, `body`, `comment`; patches may be `unified_patch`, `diff` or `resolving_diff`). A field map can name several row fields for one role and they are read in order. An item's path becomes the parser's filename hint; null or empty items are skipped; an object with none of the known keys is an error. `before` and `hints` are optional, `patch` is required. | issue/PR, commit, log and review sources |
| Observed context for infilling and artifacts | `fim` reads a `context` list (ranked chunks of other files) before the prefix; `artifact` reads its document as a list. | `sourcegraph-context-fim`, `imported-context-files` |
| Artifacts without a stored request, and recorded verdicts | The profiles `code_rewrite` and `human_code_rewrite` read an original as an observed document and score the rewrite; no request text is invented. `tested_artifact` and `human_tested_artifact` record `success` with verification `publisher_tests` only from the row's own verdict field (a test pass, an instruction checker, an answer match); the field map's `outcome` names it and a row filter keeps the passes. A verdict spelled `"True"`, `"1"` or `1.0` counts as a pass and `"False"` or `0` as a failure; fractions are unknown. | S-stage streams, rewrite streams |
| Evidence between tool calls | With the source option `path_dependencies`, a scored editor call (arguments `old_str`, `new_str`, `file_text` or `insert_line`) with a `path` is linked to the first earlier result that answered a call naming the same `path`. Only structured arguments are read; shell commands create no link and nothing is inferred from text. This is what lets long agent runs fill `long_verified_dependencies`, whose episodes the training engine refuses without dependencies. | `ldep-*` views |

Smaller additions: a `tool-result` content block in the AI SDK spelling (`toolCallId`,
`output.value`) answers its call like an Anthropic `tool_result`; the role `function_call`
reads as an assistant turn; and the repository normalizer in lineage keys treats `o/r`,
`github.com/o/r` and `https://github.com/o/r.git` as one repository, so a repository stays in
one split across sources that spell it differently.

### Later-stage sources by stream

Row filters keep only what a stream must contain (the passing rows of a tested source, the
`affected` traces, the human-authored commits). "Unique tokens" is the whole filtered source
(text bytes divided by four, from the cards' sizes and from rows measured with the adapter);
it is a lower bound where a quality limit later drops rows.

#### synthetic_code_rewrite

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `swallow-code-v2-rewrite` | `tokyotech-llm/swallow-code-v2` | apache-2.0 | 60,000 | 71B |
| `pie-perf-slow-to-fast` | `rootacess/pie-perf` | none on the card (derived from IBM Project CodeNet) | 40,000 | 13M |

#### tested_artifacts

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `kodcode-sft-4o` | `KodCode/KodCode-V1-SFT-4o` | cc-by-nc-4.0 | 30,000 | 130M |
| `kodcode-sft-r1` | `KodCode/KodCode-V1-SFT-R1` | cc-by-nc-4.0 | 30,000 | 100M |
| `openr1-verifiable-python-tested` | `open-r1/verifiable-coding-problems-python_decontaminated-tested` | none on the card (problems from TACO, APPS and CodeContests) | 15,068 | 10M |
| `s-tested-opencodeinstruct` | `nvidia/OpenCodeInstruct` | cc-by-4.0 | 60,000 | 1.2B |

#### verified_successful_multiturn

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `scale-swe-distilled-v4-pro` | `AweAI-Team/Scale-SWE-Distilled-DeepSeek-v4-Pro-High-41k` | cc-by-4.0 | 4,000 | 1.4B |
| `swe-lego-real-resolved` | `Lego-X/SWE-Lego-Real-Data` | apache-2.0 | 5,009 | 270M |
| `ifeval-like-filtered` | `argilla/ifeval-like-data` (filtered) | other (Qwen license) | 56,339 | 15M |
| `if-multi-constraints-sft` | `UniLu/IF_multi_constraints_upto5_SFT` | odc-by | 42,161 | 11M |
| `openmathinstruct1-correct` | `nvidia/OpenMathInstruct-1` | other (NVIDIA License) | 60,000 | 450M |

#### issue_pr_sequences

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `gharchive-threads` | `common-pile/github_archive` | per row: Blue Oak Council permissive list (metadata.license) | 200,000 | 13.7B |
| `pr-resolve-issue-to-patch` | `PotatoHD/github-pr-resolve` | other (no text on the card) | 12,000 | 0.9B |
| `pr-resolve-comment-to-fix` | `PotatoHD/github-pr-resolve` | other (no text on the card) | 8,000 | 0.4B |
| `pr-traces-10k` | `bulatSharif/gh-pr-issue-traces-10k` | apache-2.0 | 4,000 | 0.18B |

#### code_evolution

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `commit-chronicle` | `JetBrains-Research/commit-chronicle` | other (permissive repositories only; licence kept per row) | 60,000 | 9B |
| `locus-commit-pool` | `ahnaftaz/locus-commit-pool-v1` | other (each source file keeps its repository's licence) | 8,000 | 100B+ |

#### connected_repositories

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `repository-v2-long` | `placeholderlabs/pretrain-repository-v2-mix-long-context` | none on the card (code is under its repositories' own licences) | 1,500 | 130B |

#### long_natural_code_replay

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `repository-v2-long-replay` | `placeholderlabs/pretrain-repository-v2-mix-long-context` | none on the card (code is under its repositories' own licences) | 1,000 | 130B |

#### build_logs

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `code-world-traces-affected` | `marin-community/swe-rebench-v2-CodeWorldModeling` | cc-by-4.0 | 6,000 | 15B |
| `swe-gym-failure-logs` | `swesynth/SWE-Gym-logs` | none on the card (SWE-Gym itself is MIT) | 2,438 | 21M |
| `swe-synth-failure-logs` | `swesynth/SWE-Synth` | none on the card | 4,000 | 88M |

#### long_context

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `longmino-science-pdfs` | `allenai/dolma3_longmino_pool` | odc-by | 400 | 50B |
| `pg19-books` | `emozilla/pg19` | none on the card (deepmind/pg19 is Apache-2.0; texts are public domain pre-1919) | 600 | 1.7B |
| `pre-1929-books` | `common-pile/pre_1929_books` | none on the card (US books before 1929, public domain) | 300 | 10B |

#### natural_document_replay

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `s-replay-pg19` | `emozilla/pg19` | none on the card (deepmind/pg19 is Apache-2.0; texts are public domain pre-1919) | 120 | 1.7B |

#### issue_pr_diffs

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `github-patches` | `rasdani/github-patches` | none on the card | 3,000 | 1.8B |
| `swe-fixer-train-110k` | `internlm/SWE-Fixer-Train-110K` | mit | 40,000 | 0.18B |
| `swe-gym-raw` | `SWE-Gym/SWE-Gym-Raw` | none on the card (SWE-Gym is MIT) | 20,000 | 0.4B |
| `swe-rebench-v2-prs` | `nebius/SWE-rebench-V2-PRs` | cc-by-4.0 | 8,000 | 1.2B |
| `multi-swe-rl-reupload` | `PrimeIntellect/Multi-SWE-RL-Reupload` | other (card names CC0 with a ByteDance notice) | 4,703 | 25M |
| `commitpackft` | `bigcode/commitpackft` | mit | 100,000 | 0.8B |

#### grounded_artifacts

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `longcite-45k` | `zai-org/LongCite-45k` | apache-2.0 | 2,000 | 1.4B |
| `imported-context-files` | `tingtang2/the_stack_v2_2M_repos_pretraining_dataset_imported_context-dataset` | none on the card | 20,000 | 1.9B |

#### long_context_fim

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `sourcegraph-context-fim` | `sourcegraph/context-aware-fim-code-completions` | none on the card | 13,489 | 0.25B |
| `tiger-fim-midtraining` | `TIGER-Lab/FIM-Midtraining-400K` | other (multiple open-source licences) | 10,000 | 2.6B |

#### long_verified_dependencies

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `ldep-scale-swe-v4-pro` | `AweAI-Team/Scale-SWE-Distilled-DeepSeek-v4-Pro-High-41k` | cc-by-4.0 | 3,000 | 0.8B |
| `ldep-rebench-openhands` | `nebius/SWE-rebench-openhands-trajectories` | cc-by-4.0 | 4,000 | 1.3B |
| `ldep-coderforge` | `togethercomputer/CoderForge-Preview` (trajectories) | per row (Apache-2.0 in the sampled rows); none on the card | 3,000 | 2.2B |

#### mixed_quality_episodes

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `scale-swe-distilled-v32` | `AweAI-Team/Scale-SWE-Distilled` | cc-by-4.0 | 3,000 | 3.5B |

#### recovery_clarification

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `opencode-question-tool` | `nvidia/Nemotron-SFT-OpenCode-v1` | cc-by-4.0 | 2,500 | 1.1B |
| `opencode-agent-skills-question-tool` | `nvidia/Nemotron-SFT-OpenCode-v1` | cc-by-4.0 | 1,500 | 0.4B |
| `code-feedback` | `m-a-p/Code-Feedback` | apache-2.0 | 30,000 | 94M |

#### environment_trajectories

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `adp-go-browse-wa` | `neulab/agent-data-collection` | mit | 3,000 | 180M |
| `adp-nnetnav-wa` | `neulab/agent-data-collection` | apache-2.0 | 800 | 150M |
| `adp-mind2web` | `neulab/agent-data-collection` | cc-by-4.0 | 300 | 100M |
| `agentrefine-gpt4o` | `fudayuan/AgentRefine-gpt4o-64000` | apache-2.0 | 64,000 | 68M |

### Later-stage tokens against planned exposure

"Unique tokens" is what the pinned sources hold after their row filters (a lower bound where
a quality limit later drops rows); the first-run caps above prepare far less. "Planned" is
the stage's share times its budget, the last column how many times each token would be seen if
the plan ran as written. Above about 4 it is time to find more data or lower the share; far
below 1 means the source is not the limit.

| Stream | Phase and share | Unique tokens (B) | Planned (B) | Times each token is seen |
| --- | --- | --- | --- | --- |
| issue_pr_sequences | M1, 28.4% | 15.2 | 17.1 | 1.1x |
| connected_repositories | M1, 13% | 130 | 7.8 | 0.06x |
| code_evolution | M1, 2.8% | 9 or more (commit diffs); the full commit pool is far larger | 1.7 | 0.2x |
| build_logs | M1, 1.1% | 15 | 0.67 | 0.04x |
| long_context | M1, 4.7% | 60 or more | 2.8 | 0.05x |
| issue_pr_diffs | M2, 9.3% | 3.5 after removing overlaps (4.4 raw) | 3.7 | 1.1x |
| grounded_artifacts | M2, 9% | 3.3 | 3.6 | 1.1x |
| long_context_fim | M2, 4.7% | 2.85 (0.25 of it true cross-file) | 1.9 | 0.65x |
| long_verified_dependencies | LC, 40% | about 4.3 (shared with `verified_trajectories`) | 8.0 | 1.9x |
| long_natural_code_replay | LC, 8% | 130 (same pool as connected_repositories) | 1.6 | 0.01x |
| mixed_quality_episodes | LC, 20% | 15.5 (3.5 new, 12 already in `swe-zero-openhands`) | 4.0 | 0.26x |
| synthetic_code_rewrite | P1, 3.9% | 71 | 15.6 | 0.22x |
| environment_trajectories | P1, 0.1% | 0.6 (0.5 new, 0.1 before) | 0.4 | 0.7x |

The S stage is metered in scored positions (assistant tokens the loss is computed on), not
transcript tokens. Scored tokens are the share of a source's transcript that the assistant
wrote, so the transcript totals below overstate what S can learn from:

| Stream | S share | Unique transcript tokens (B) | Of which scored (B, estimate) | Planned scored positions (B) | Passes |
| --- | --- | --- | --- | --- | --- |
| verified_successful_multiturn | 46% | 2.15 new, plus the pinned verified sets that `s-verified-*` re-read | 0.45 new | 0.92 | 2x on the new rows alone; the re-read sets add more |
| recovery_clarification | 14% | 1.6 (was 0.007) | 0.5 | 0.28 | 0.6x (was 40x) |
| tested_artifacts | 8% | 1.4 | 0.6 | 0.16 | 0.3x |
| natural_document_replay | 1% | 1.7 | 1.7 | 0.02 | 0.01x |

`long_verified_dependencies` is not independent supply: its three views re-read the same
trajectories as `verified_trajectories` for another stream, as the `s-verified-*` entries do.

### What was checked for the later stages

Each of the 47 entries was streamed from its pinned revision and files through the real
streaming reader and its own adapter on its first 6 to 20 rows after the entry's row filters.
Every row produced an episode in the declared stream, with these exceptions: one CoderForge
row (the segmenter message "Parser diagnostic is outside the original source" already noted
for that source above), and `pr-resolve-comment-to-fix`, where only one of the first eight
rows of the file was a review-fix row (the file mixes both task types; the filter is
working). Findings that changed the manifest:

- The read-batch-size option is a Parquet reader setting; JSON entries that set it would not
  open, so it is now set for Parquet files only.
- `bigcode/commitpackft` ships a loading script the streaming reader refuses, so the entry
  pins the Hub's automatic Parquet conversion of that same commit.
- `argilla/ifeval-like-data` needs its `filtered` configuration; reading the file directly
  fails the card's declared columns.
- `nvidia/Nemotron-SFT-OpenCode-v1` writes tools as `{id, inputSchema.jsonSchema}`; four of
  six rows were rejected until the adapter learned that spelling.
- Numeric or label-valued `id` columns (Go-Browse, AgentRefine, the IF sets) are not used as
  lineage families; a hash of the first request is.
- The edit-to-read links come out as expected on real rows (one to six per run in Scale-SWE,
  SWE-rebench-OpenHands and CoderForge); Orchard produced none.

The fixture tests in `tests/test_later_stage_sources.py` build a row in each source's real
shape (field names and nesting from rows read that day), run the adapter named in the
manifest, and check the stream, scored records, provenance, verdict and the row filters;
`tests/test_later_stage_adapters.py` covers the new row shapes. Neither touches the network.

### Considered and not added for the later stages

- **Pairs and sets that cannot be streamed as they are.** Single-file JSON arrays
  (`xlangai/AgentTrek`, `agent-eto/eto-sft-trajectory`, `WeixiangYan/CodeTransOcean`,
  `Yhyu13/ToolBench_toolllama_G123_dfs`, `Vezora/Tested-143k-Python-Alpaca`); an archive of
  logs whose `.xz` files are really 7z (`pkgforge/build-logs`, real Actions run logs, about 5 GB
  that expand to roughly 85 GB, with no fixes); Arrow-only layouts
  (`project-themis/git-commits`, `iNeil77/CodeNet`) that need a conversion job first.
- **Shapes the adapters do not read.** `stellalisy/MediQ_AskDocs` (the one human-written set of
  clarifying questions: about 20k rows with a few dozen scored tokens each, with the question in
  a separate column from the history), `SenseLLM/ReflectionSeq-DS` (typed code and execution
  parts), `tuandunghcmut/toolbench-v1` (column-oriented conversations), the three CodeNet
  translation sets (a column per language), and the Stack v3 training split itself (nested
  file lists of a whole repository need a renderer; `placeholderlabs` already ships that
  rendering).
- **Shuffled or unsortable.** `bigcode/commitpack` (3.8 TB, rows shuffled, fork lists as
  repository), `luowenyang/issues_prs_grouped` (2.7 GB single row groups),
  `open-index/open-github-issues` (relational tables for seven repositories),
  `common-pile/stackv2` and `codeparrot/github-code` (per-file rows in random repository order).
- **Evaluation or benchmark-derived.** SWE-bench and its variants, Multi-SWE-bench,
  `Anonym01048/SWE-Bench-Pro-interactive-*`, `agentica-org/DeepCoder-Preview-Dataset` (its
  `lcbv5` and `codeforces` splits), `tianyang/repobench*`, `CSJianYang/ExecRepoBench`,
  `eyuansu71/logdx-ci` and the CI-repair benchmarks.
- **Duplicates or weaker copies.** `nebius/SWE-rebench-V2` (a subset of `-V2-PRs`),
  `nebius/SWE-bench-extra`, `SWE-Gym/SWE-Gym` (a subset of `-Raw`), the `SWE-Fixer` chain-of-thought
  variants (the same 70k rows as `rasdani/github-patches`), `bigcode/commitpackmeta`, the
  LogHub system logs (one line per row, no failures to fix) and `nvidia/Nemotron-Instruction-Following-Chat-v1`
  (no per-row verdict, so it cannot enter a verified stream).
- **No failures with recovery marks.** Unresolved runs of `SI2CA-Training-Trajectories`,
  `nebius/SWE-rebench-openhands-trajectories` and SWE-smith carry a recorded failure; an
  episode with a recorded failure gets no imitation loss unless recovery turns are marked, so
  they would add context tokens without scored positions and were not added to
  `mixed_quality_episodes`.
- **Fully model-written environments or turns.** `inclusionAI/AReaL-tau2-data` and
  `fuvty/tau-bench-synthetic` (simulated users, mostly repeated turns of the same dialogue),
  `TIGER-Lab/BrowserAgent-Data` and `HongbangYuan/sciworld_sft_warmup` (single-step rows),
  `the-acorn-ai/textarena-player-game-traces` (cumulative history that needs a custom parser).

### Gated repositories

Re-checked on 2026-10-23: the first bytes of a data file of every repository below were requested again with the project token and each still answered HTTP 403, so none could be read and none was moved out of the gated manifest. (A request for a repository's README or licence file is answered even before the terms are accepted, so it is no test.)

Nothing in `sources.later-stages.json` needs a click: every entry was read with this
account's token. Four gated repositories were checked (data file metadata, HTTP 403 on each) and
left out, because the ungated sets above cover the same ground. Clicking "Agree to access" on
these would allow them to be added:

- https://huggingface.co/datasets/bigcode/the-stack-github-issues (about 13B tokens of GH
  Archive issue and PR threads with masked usernames; the same pool as
  `common-pile/github_archive`)
- https://huggingface.co/datasets/bigcode/starcoderdata (adds `git-commits-cleaned`; its
  issues subset duplicates the one above)
- https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2 (manual approval; its
  Synthetic-Rewriting and Synthetic-Transpilation subsets are already pinned in
  `sources.v13.json` but need the parent files of Nemotron-Pretraining-Code-v1 to join their
  originals, and `swallow-code-v2` now fills the rewrite stream)
- https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1 (manual approval; the
  originals those rewrites came from)

### Preparing the later-stage sources

`configs/cloud/prepare-later-stages.json` (one 16-CPU, 32 GiB sandbox, no GPU, nothing
launches by itself; it needs `--secret hf-token=FILE` and a clean commit) splits the manifest into
`LATER_SHARDS` (8) pieces with `scripts/shard_sources.py`, which assigns the costliest
sources first to the lightest shard, and runs one `megatoken prepare --reject-invalid` per
shard into `out/prepared/later-<i>`, with the same parser-review policy as the agentic job.
`LATER_ROW_SCALE` multiplies every cap. The row-size and pass-rate tables in `megatoken.data.sharding` now
cover these streams, so shards balance by estimated text.

Things to expect on the first run:

- The long documents (`longmino-science-pdfs`, books, repository documents) are up to a
  megabyte each and sentence segmentation costs about 25 KB of text per second per process, so
  they set the wall clock; lower `LATER_ROW_SCALE` for a smoke test.
- Documents and traces over the quality limit (250k to 400k tokens) are rejected by their
  length rule rather than cut; the rejects are counted.
- `locus-commit-pool` reads nested JSON whose `meta` object differs between shards. If the
  streaming JSON reader reports a column type change on some shard, drop that file from the
  entry; the `placeholderlabs` document sets read from flat Parquet and avoid it.
- The Hub's streaming reader resolves the glob patterns used for the long shard lists on every
  start; a failed run is resumed from the committed cursor.
- Several sources are alphabetical or repository-contiguous (`pr-resolve-*`, `commit-chronicle`,
  `imported-context-files`), so a small cap covers only the first repositories.

### Still missing after this round

- **issue_pr_sequences.** No public set links an issue, its pull-request thread and the
  follow-up commits in one row at scale. The 13.7B-token thread source holds issue and PR
  threads as separate rows; the linked material (issue, files or base commit, patch) is about
  3B unique tokens, and review-then-fix pairs exist only in `pr-resolve-comment-to-fix`
  (about 27k rows) and `pr-traces-10k` (10k rows). Human issue text with the pre-change files
  exists only in `github-patches` (70k rows); the other SWE sets give a base commit alone, and
  `swe-rebench-v2-prs` writes its problem statement with a model.
- **long_verified_dependencies.** No public data has a 131k to 262k episode, annotated evidence
  and a verified result together. The three views are derived: episodes of at least 32k tokens
  (public agent runs rarely pass about 125k), with evidence taken from structured editor
  arguments only. Microsoft Orchard's runs produced no links in the 8 rows read (its calls
  name no editor `path`), so it was left out. `LongCite-45k` has real citation spans, but model-written answers and no
  verdict, so it sits in `grounded_artifacts` without links.
- **build_logs.** Real CI failures paired with their fixes exist only as small benchmarks.
  The stream is filled by executed test runs on pull-request tasks, not CI logs.
- **recovery_clarification.** Every large source is model-written, with a simulated user
  answering. The one human-written set (`MediQ_AskDocs`) is small and in a shape the adapters do
  not read. Nothing public is dedicated to pushback and self-correction after a user
  disagrees; `code-feedback` is the nearest and records no fix verdict.
- **tested_artifacts.** Volume is fine; human solutions with a recorded pass are only
  `open-r1/verifiable-coding-problems-python_decontaminated-tested` (15k rows). TACO, APPS and
  CodeContests have human solutions and tests but no per-row pass record.
- **long_context_fim.** True cross-file fill-in-the-middle is 0.25B tokens. The larger set
  has a model writing the missing body. Cutting infills from the long natural repository
  documents would add unlimited volume but is not implemented.
- **synthetic_code_rewrite.** Python only, and equivalence of the rewrite is not checked.
- **Decontamination.** `swe-fixer-train-110k` and `swe-gym-raw` include SWE-bench's own
  repositories; the KodCode sets carry a similarity score against HumanEval, MBPP, BigCodeBench
  and LiveCodeBench. Supply a quarantine snapshot (`--benchmark-quarantine`) before training.
- **Terms.** Several sources are non-commercial or carry model licences (the two KodCode
  sets are cc-by-nc-4.0, `argilla/ifeval-like-data` the Qwen licence, OpenMathInstruct-1 the
  NVIDIA licence); each is named in its entry so a release decision can
  filter on it.
