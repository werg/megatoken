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
high-quality examples to fill it.

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
`source_id` in `sources.open.json` (17: `fineweb-edu-10bt`, `stackv2-edu-0000/0020/0040/0060/0080`,
`cosmopedia-web_samples_v2/stanford/stories`, `openmathinstruct2-1m`, `openthoughts-114k`,
`oasst-top1`, `smoltalk-new`, `opencodeinstruct`, `open-swe-v12`, `agent-trove`, `terminal-corpus`)
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

- **synthetic_code_rewrite**: no open, permissively licensed set of LLM code rewrites or
  transpilations was found. `sources.v13.json` pins the Nemotron code subsets, but they are
  under the NVIDIA Data Access Agreement and need `options.lineage_join` parent files.
- **Other phases' streams** (M1/M2/LC/S: repository, issue/PR, build-log and verified
  trajectories) are untouched here; `sources.v13.json` remains their source of record.
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
`synthetic_code_rewrite` still has no source (see "Still missing" above) and keeps a smaller
share until one exists.

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

First-run caps count rows read from the start of the pinned data (before filters), so
filtered sources carry larger caps: roughly 17% of Nebius SWE-agent rows, 48% of ReBench
rows, 43% of each SWE-smith variant, 8% of SWE-Gym rows and 70% of Orchard rows pass
their filters. "Unique tokens" is for the whole filtered source, not the cap.

#### executed_trajectories

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `open-swe-traces` | `nvidia/Open-SWE-Traces` (v1.2) | cc-by-4.0 | 800 | 3B |
| `agent-trove-terminus2` | `open-thoughts/AgentTrove` | apache-2.0 | 3,000 | 20B |
| `nemotron-terminal-easy` | `nvidia/Nemotron-Terminal-Corpus` (skill_based_easy) | cc-by-4.0 | 1,000 | 350M |
| `nemotron-terminal-medium` | `nvidia/Nemotron-Terminal-Corpus` (skill_based_medium) | cc-by-4.0 | 1,500 | 1.2B |
| `nemotron-terminal-mixed` | `nvidia/Nemotron-Terminal-Corpus` (skill_based_mixed) | cc-by-4.0 | 500 | 30M |
| `nemotron-swe-v3` | `nvidia/Nemotron-SFT-SWE-v3` | cc-by-4.0 | 800 | 12B |
| `nemotron-swe-v2-openhands` | `nvidia/Nemotron-SFT-SWE-v2` | cc-by-4.0 | 500 | 3.2B |
| `nemotron-swe-v35-opencode` | `nvidia/Nemotron-SFT-SWE-v3.5` | cc-by-4.0 | 500 | 250M |
| `nemotron-swe-v1-r2e` | `nvidia/Nemotron-SWE-v1` | cc-by-4.0 | 500 | 3B |
| `swe-hero` | `nvidia/SWE-Hero-openhands-trajectories` | cc-by-4.0 | 500 | 2.5B |
| `swe-smith-mini` | `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` | mit | 500 | 1.3B |
| `r2egym-sft` | `R2E-Gym/R2EGym-SFT-Trajectories` | none on the card | 800 | 60M |
| `swe-dev` | `zai-org/SWE-Dev-train` | none on the card | 500 | 250M |
| `codeact-instruct` | `xingyaoww/code-act` | apache-2.0 | 3,000 | 10M |

#### mixed_quality_episodes

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `swe-zero-openhands` | `nvidia/SWE-Zero-openhands-trajectories` | cc-by-4.0 | 500 | 12B |

#### verified_trajectories

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `orchard-swe-verified` | `microsoft/Orchard` (swe) | mit | 1,400 | 7B |
| `coderforge-reward1` | `togethercomputer/CoderForge-Preview` (trajectories) | per row (Apache-2.0 in the sampled rows); none on the card | 1,000 | 9B |
| `rebench-openhands` | `nebius/SWE-rebench-openhands-trajectories` | cc-by-4.0 | 2,000 | 3.2B |
| `swe-agent-nebius` | `nebius/SWE-agent-trajectories` | cc-by-4.0 | 6,000 | 250M |
| `swe-smith-tool` | `SWE-bench/SWE-smith-trajectories` (tool) | mit | 1,200 | 500M |
| `swe-smith-xml` | `SWE-bench/SWE-smith-trajectories` (xml) | mit | 1,200 | 500M |
| `swe-smith-ticks` | `SWE-bench/SWE-smith-trajectories` (ticks) | mit | 1,200 | 450M |
| `swegym-sampled` | `SWE-Gym/OpenHands-Sampled-Trajectories` | none on the card | 6,055 | 20M |

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
  group over 2 GB uncompressed, which the streaming reader cannot decode; the skill-based
  synthetic task configurations are used.
- **Orchard `gui`**, AndroidControl and other screenshot sets: need images, not text.
- **CC-Bench** (370 rows, Claude Code format), **Llama-Nemotron SFT chat** (input/output
  shape) and **OpenThoughts-Agent** (a subset of AgentTrove): little or nothing new.
- **Nemotron-Cascade-SFT-SWE and Nemotron-SFT-SWE-v2 agentless splits**: single-turn
  patches, better placed in `synthetic_code_dialogue`; left for that stream.

### Preparing the sources

Preparation is single-threaded per process and dominated by sentence segmentation of long
agent traces: about 25 KB of source text per second on SWE trajectories (3 to 8 s per
150 KB trace). One process cannot prepare a useful amount, so
`configs/cloud/prepare-agentic-mix.json` (one 16-CPU, 32 GiB sandbox, no GPU, nothing
launches by itself) splits `sources.agentic.json` into `AGENTIC_SHARDS` (8) pieces with
`scripts/scale_source_caps.py`, which assigns the costliest sources first to the lightest
shard, and runs one `megatoken prepare --reject-invalid` per shard into
`out/prepared/agentic-<i>`.

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

The earlier pilot entries for `open-swe-v12`, `agent-trove` and `terminal-corpus` in
`sources.open.json` are left as they were. They lose many rows because they set neither
`allow_pending_final_action` (traces that end on a call with no result) nor positional
call IDs. The new manifest re-pins `open-swe-traces` and AgentTrove at the same revisions
with those options. The pilot's `terminal-corpus` entry reads the `dataset_adapters`
files, which the streaming reader cannot decode (see above), so the new manifest reads the
skill-based configurations instead. Do not prepare both manifests for the same stream.

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

- **recovery_clarification** has 7M unique tokens against 0.28B planned positions (about
  40 passes); it needs more recovery data (failed-then-fixed runs, asking for missing
  arguments).
- **environment_trajectories** is only 0.1B tokens (4 passes in P1). It is deliberately a
  small share.
- **Executed tool results from real APIs.** Most tool-use results are simulated; only
  Toucan's MCP calls were executed.
- **Failure and recovery in SWE traces.** Only Open-SWE-Traces records failed runs;
  everywhere else the failed attempts were already filtered out by the publisher.
- **synthetic_code_rewrite** (above) remains unsourced.

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

The open-licence pilot above draws a few thousand clips per media stream from six datasets.
This section records the scale-up: forty-four pinned public datasets, a reader for each kind of
file layout, six cloud jobs that build the streams, and how far the result is from what the
stage plan asks for. Licence, size and layout facts were read from the Hub on 2026-10-22 (a card
is not legal advice). Every source carries the maintainer's recorded approval in its `terms`.

### How a source is described and read

`configs/data/sources.media.json` lists every media source (the six pilot sources are in it too;
the old per-source constants are gone). Each entry holds

- the Hub repository and the immutable revision it is pinned to, the licence found on its card,
  the approval record, notes on what the licence does not settle, and whether the repository is
  gated (the Hub account must accept its terms first);
- `include`, patterns naming the files to read at that revision;
- a **reader** (how the files are laid out) and a **task** (what the labels mean);
- `columns`, which field of a row or sample plays which role (`image`, `caption`, `audio`,
  `text`, `translation`, `original`, `edited`, `instruction`, ...), `options` (length limits,
  which caption to pick, which rows to skip) and `inventory` (items, hours, repository size and
  the basis of each number).

Readers (`data/open_media.py`): `parquet` (one record per row), `webdataset` (tar files whose
members sharing a name form one sample), `archive` (zip or tar members joined to a csv/jsonl
table, optionally with a same-named member of a second zip, as in source/edited video pairs),
and two special readers, `fleurs` (the same sentence read in two languages) and `docci` (files
that are not on the Hub). Tasks turn a record into a clip and then into episodes of the stage
plan's own streams:

| Task | Streams |
| --- | --- |
| `transcription` | transcription, audio_identity (and speech_edit when procedural edits are asked for) |
| `verbalization` | verbalization, audio_identity (the previous clip of the same speaker is the voice sample) |
| `translation_text` | speech_translation (text target), transcription, audio_identity |
| `translation_pair` | speech_translation (speech target), transcription |
| `captioned_image` | image_description, image_creation, image_identity |
| `picture` | image_identity |
| `image_edit` | image_edit |
| `sound` | audio_identity |
| `av_caption`, `av_scenes` | av_creation, video_identity (and av_edit when procedural edits are asked for) |
| `video` | video_identity (and av_edit when procedural edits are asked for) |
| `video_edit` | av_edit (instructed edits that the dataset supplies) |

A translation whose target is only text was not expressible before: the concordance compiler now
emits the same `speech_translation` view without the translated speech when a source has none.

Nothing is mirrored. Files are streamed from the pinned revision over HTTP range requests (the
revision is content addressed, so it is the integrity pin; hashes are no longer listed per file)
or, when `<inputs>/<source_id>` exists, read from that local directory. Every clip's files are
deleted as soon as its episodes are built, so a run needs scratch space for a few clips only.
Splits freeze per original family (speaker, recording, photo, video), as before.

Labels that a model wrote are recorded as such: the text artifacts carry provenance `teacher`
and the bundle's verification tier is `model-labeled` (machine transcripts or captions) or
`model-generated` (machine-made images, edits and videos); human-labelled sources keep
`publisher-annotated`.

Clip limits: speech and sound clips are 0.5-20 s (a longer recording is dropped, or for
`crop` sources cut to its central 20 s), images are scaled to at most 1024 pixels (edit pairs
losslessly, and only when both images have the same size), video clips are scaled into 640x640
at 16 frames/s and at most 8 s (6 s for edit pairs).

`scripts/prepare_open_media.py` builds one prepared root: `--count SOURCE=N` draws N clips of a
source, `--procedural SOURCE=N` adds a procedural edit for its first N clips, `--option
SOURCE.KEY=JSON` overrides an option, `--scale F` multiplies every count, and `--shard-slice I/N`
reads every N-th file starting at the I-th so N sandboxes can build disjoint parts of one source.
Without `--count` it builds the pilot sizes, so `prepare-open-mix.json` is unchanged except that
it no longer lists the pilot's media inputs; only the two DOCCI files, which are not on the Hub,
remain job inputs.

### The sources

Sizes are items (clips, images or edits), hours of audio and the repository's size for the
files read. "Gated" means the Hub account must first accept the publisher's terms.

#### Captioned and plain images (image_identity, image_description, image_creation)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `google/docci` | Dense human-written descriptions of 15k photos (the training split is read). | CC-BY-4.0 | 9.65k items | Images and descriptions are fetched from the publisher's storage bucket (the Hub repository holds only a loading script); the job input pins their sha256. |
| `lmms-lab/LLaVA-ReCap-CC3M` | Detailed captions written by LLaVA-NeXT-34B for Conceptual Captions 3M images; images keep their web owners' rights. | unspecified on the card | 2.86M items; 138 GB |  |
| `lmms-lab/LLaVA-ReCap-558K` | LLaVA-NeXT-34B captions for the LLaVA-LCS-558K images (LAION, Conceptual Captions, SBU). | unspecified on the card | 558k items; 13 GB |  |
| `lmms-lab/LLaVA-ReCap-118K` | LLaVA-NeXT-34B captions for the 118k COCO images. | unspecified on the card | 118k items; 6.1 GB | Images are MS-COCO Flickr photos with mixed per-image Creative Commons licences. |
| `BLIP3o/BLIP3o-Pretrain-Long-Caption` | Long captions by Qwen2.5-VL-7B for SA-1B, web (CC12M-style) and JourneyDB images. | apache-2.0 | 29.4M items; 1,374 GB | Images come from SA-1B (research licence), web crawls and JourneyDB (Midjourney outputs); the card states only Apache-2.0. |
| `jackyhate/text-to-image-2M` | Synthetic images (Flux and Midjourney style) with the prompt that produced them, as caption. | mit | 2M items; 418 GB | Images are model outputs; FLUX.1-dev outputs fall under the FLUX.1-dev non-commercial licence. |
| `common-canvas/commoncatalog-cc-by` | Flickr photos under Creative Commons Attribution licences, with BLIP-2 captions. | cc-by-4.0 | 14.6M items; 21,465 GB | Each photo carries its own Creative Commons Attribution licence (licensename/licenseurl columns); BLIP-2 captions are short and generic. |
| `jxie/coco_captions` | Human captions for MS-COCO (five per image; the longest is used). | unspecified on the card | 113k items; 19 GB | COCO annotations are CC-BY-4.0; the Flickr photos carry mixed per-image Creative Commons licences (some NonCommercial/NoDerivatives). |
| `UCSC-VLAA/Recap-COCO-30K` | LLaMA-3-powered recaptions of 30k COCO images. | cc-by-4.0 | 30.5k items; 1.6 GB | COCO photos carry mixed per-image Creative Commons licences. |
| `lmms-lab-encoder/TextCaps` | Human captions that mention the text in Open Images photos (five per image; the longest is used). | unspecified on the card | 22k items; 6.2 GB | TextCaps captions are CC-BY-4.0 upstream; Open Images photos are CC-BY-2.0 per image. |
| `lmms-lab-encoder/flickr30k` | Human captions for Flickr30k (five per image; the longest is used). | unspecified on the card | 31.8k items; 4.4 GB | Flickr30k is distributed for non-commercial research; the Flickr photos have their own owners. |
| `BAAI/DenseFusion-1M` | Dense captions fused from several vision experts for 1M web images. | cc-by-4.0 | 1M items; 798 GB | Images are web crawls (LAION-derived) with their owners' rights. |
| `derek-thomas/ScienceQA` | Diagrams, maps and photos from school science questions (the questions are not used); rows without an image are skipped. | cc-by-sa-4.0 | 6.22k items; 0.4 GB | ShareAlike; upstream images come from many web sources. |

#### Instructed image edits (image_edit)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `osunlp/MagicBrush` | Human-made multi-turn edits (DALL-E 2 inpainting guided by crowd workers) of COCO images. | CC-BY-4.0 | 8.81k items; 26 GB | Source images come from MS-COCO, whose Flickr photos carry mixed per-image Creative Commons licences (some NonCommercial/NoDerivatives); the card does not address this. |
| `BleachNick/UltraEdit` | 3M free-form edits generated with Stable Diffusion 3 inpainting from LLM-written instructions. | cc-by-4.0 | 3.03M items; 1,555 GB | Source images come from COCO, Flickr and Open Images with their own licences; edited images are Stable Diffusion 3 outputs. |
| `UCSC-VLAA/HQ-Edit` | About 100k edits whose images were generated with DALL-E 3 and instructions written with GPT-4V. | cc-by-nc-4.0 | 98k items; 319 GB | Images and instructions are OpenAI model outputs; the card licence is NonCommercial. |
| `timbrooks/instructpix2pix-clip-filtered` | The CLIP-filtered InstructPix2Pix set: GPT-3 instructions with Prompt-to-Prompt edits by Stable Diffusion. | unspecified on the card | 313k items; 130 GB | The InstructPix2Pix release is CC-BY-4.0 for the text; images are Stable Diffusion outputs (CreativeML Open RAIL-M). |
| `TIGER-Lab/OmniEdit-Filtered-1.2M` | 1.2M edits from seven specialist editing models, filtered by GPT-4o quality scores. | mit | 1.2M items; 2,977 GB | Source images come from several web datasets; 5 GB shards of full-resolution PNG. |
| `Bin1117/AnyEdit` | Instruction edits across 25 types; rows that need a depth/segment guide image are skipped. | cc-by-4.0 | 2.3M items; 265 GB | Shards are grouped by edit type; the first thirty hold only types that need a guide image and are not listed. Instructions of some types name a guide image the reader does not supply; rows with a guide image are skipped. |

#### Speech

| Source | What it is | Licence on the card | Size | Streams | Notes |
| --- | --- | --- | --- | --- | --- |
| `openslr/librispeech_asr` | 960 hours of read English audiobooks (LibriVox) with transcripts in upper case. | CC-BY-4.0 | 281k items; 960 h; 60 GB | audio_identity, speech_edit, transcription |  |
| `mythicinfinity/libritts_r` | Restored LibriTTS: read English with the written text and its normalized spoken form. | CC-BY-4.0 | 355k items; 585 h; 95 GB | audio_identity, speech_edit, verbalization |  |
| `parler-tts/mls_eng` | Multilingual LibriSpeech English: 44.5k hours of read audiobooks (LibriVox), 10-20 s segments. | cc-by-4.0 | 10.8M items; 44.5k h; 705 GB | audio_identity, speech_edit, transcription | Transcripts are lower case without punctuation. |
| `facebook/multilingual_librispeech` | Multilingual LibriSpeech in seven further languages (read audiobooks). | cc-by-4.0 | 1.45M items; 6.03k h; 94 GB | audio_identity, speech_edit, transcription | Transcripts are lower case without punctuation. |
| `MLCommons/peoples_speech` | English speech from the internet (the clean, well-aligned subset); family is the source recording. | mixed Creative Commons (cc-by-2.0 to cc-by-4.0, cc-by-sa-3.0, cc-by-sa-4.0) | 1.5M items; 5.97k h; 392 GB | audio_identity, speech_edit, transcription | Share-alike and attribution terms apply per recording; transcripts are force-aligned and sometimes noisy. |
| `facebook/voxpopuli` | European Parliament speech in 14 languages with transcripts. | cc0-1.0 (European Parliament recordings; reuse terms of the Parliament also apply) | 587k items; 1.8k h; 314 GB | audio_identity, speech_edit, transcription |  |
| `fixie-ai/common_voice_17_0` | Volunteers reading sentences, many accents and languages; family is the speaker. | CC0-1.0 (Mozilla Common Voice; the card states none) | 3.61M items; 5k h; 132 GB | audio_identity, speech_edit, transcription | Several clips per speaker and sentence; the mirror's card has no licence. |
| `speechcolab/gigaspeech` (gated) | 10,000 hours of audiobooks, podcasts and YouTube with transcripts. | apache-2.0 (card; audio carries its sources' own terms) | 8.3M items; 10k h; 1,152 GB | audio_identity, speech_edit, transcription | Gated: the Hub account must accept the terms. Podcast and YouTube sources have their own terms. Transcripts hold punctuation tags that the reader converts and silence/noise tags that make it skip the segment. |
| `amphion/Emilia-Dataset` (gated) | In-the-wild speech from podcasts and videos in six languages, pipeline-filtered, with ASR transcripts. | cc-by-4.0 on the card (Emilia-YODAS); the original Emilia part is CC-BY-NC-4.0 under the gate's terms | 77.6M items; 216k h; 4,702 GB | audio_identity, speech_edit, transcription | Gated: the Hub account must accept the terms. Transcripts are Whisper-pipeline output, not human. |
| `espnet/yodas-granary` | English YouTube speech (Creative Commons videos) with punctuated Whisper-large-v3 transcripts, filtered for quality. | cc-by-3.0 | 40.8M items; 104k h; 11,264 GB | audio_identity, speech_edit, transcription | Pseudo-labelled by Whisper and punctuation-restored by Qwen2.5-7B, not human transcripts; per-video Creative Commons terms. |
| `espnet/yodas-granary` | Non-English YouTube speech with its transcript and an English translation by EuroLLM-9B, quality filtered. | cc-by-3.0 | 23.2M items; 58.6k h; 9,727 GB | audio_identity, speech_edit, speech_translation, transcription | Transcripts and translations are model output, not human; per-video Creative Commons terms. |
| `fixie-ai/covost2` | Common Voice speech in 21 languages with human English translations of the sentence. | unspecified on this mirror's card (CoVoST 2 release: CC0-1.0; facebook/covost2's card says CC-BY-NC-4.0) | 653k items; 1.5k h; 25 GB | audio_identity, speech_edit, speech_translation, transcription |  |
| `google/fleurs` | The same FLoRes sentence read in about 100 languages: each language's reading paired with the English reading. | CC-BY-4.0 | 140k items; 440 h | speech_translation, transcription |  |
| `MikhailT/hifi-tts` | Ten studio-quality speakers reading audiobooks (44.1 kHz), 292 hours. | cc-by-4.0 | 322k items; 292 h; 42 GB | audio_identity, speech_edit, verbalization |  |
| `ylacombe/expresso` | Four speakers reading in several styles; only the neutral styles are used (the style is not part of the request). | cc-by-nc-4.0 | 6k items; 20 h; 5.8 GB | audio_identity, speech_edit, verbalization |  |
| `MikhailT/lj-speech` | One speaker reading seven non-fiction books, 24 hours. | cc-by-4.0 on this mirror's card (LJ Speech is public domain upstream) | 13.1k items; 24 h; 3.8 GB | audio_identity, speech_edit, verbalization |  |

#### Sounds and music (audio_identity)

| Source | What it is | Licence on the card | Size | Notes |
| --- | --- | --- | --- | --- |
| `agkphysics/AudioSet` | 10-second YouTube clips of 527 sound classes (the class labels are not used). | cc-by-4.0 | 1.76M items; 4.88k h; 2,421 GB | The card licenses the labels; the audio belongs to the YouTube uploaders. |
| `OpenSound/AudioCaps` | 10-second AudioSet clips with human captions (the captions are not used). | cc-by-nc-4.0 | 45.2k items; 126 h; 39 GB |  |
| `CLAPv2/Clotho` | 15-30 second Freesound recordings with five human captions (the captions are not used); a central 20 s excerpt is kept. | unspecified on this mirror's card | 3.84k items; 24 h; 8.1 GB | Upstream Clotho terms were not verified here. |
| `renumics/song-describer-dataset` | Full music tracks with human captions (the captions are not used); a central 20 s excerpt is kept. | unspecified on this mirror's card | 746 items; 41 h; 2.6 GB | Audio is Jamendo music under per-track Creative Commons licences (upstream; not verified here). |

#### Video

| Source | What it is | Licence on the card | Size | Streams | Notes |
| --- | --- | --- | --- | --- | --- |
| `RekaAI/RekaDaily-10k-processed` | First-person household and workplace clips (nine subsets) with sound and a model-written dense caption. | Apache-2.0 | 4.5M items; 5.4k h; 74,218 GB | av_creation, av_edit, video_identity | First-person recordings of paid collectors; clips contain ambient audio and possibly incidental speech. |
| `HuggingFaceFV/finevideo` (gated) | 43,751 YouTube videos with sound and rich scene/activity annotations; clips are cut at the annotated activities. | cc (per-video Creative Commons; see the card) | 131k items; 3.42k h; 673 GB | av_creation, av_edit, video_identity | Gated: the Hub account must accept the terms (removal requests must be honoured). Annotations are Gemini and GPT output, not human. Layout is taken from the card; not sampled. |
| `Loie/VGGSound` | 200k ten-second YouTube clips with sound, labelled by the sound class (for example "dog barking"). | cc-by-4.0 | 199k items; 550 h; 338 GB | av_creation, av_edit, video_identity | Captions are the class label only. The copyright remains with the original uploaders. |
| `nkp37/OpenVid-1M` | One million high-aesthetic clips (Pexels, CelebV-HQ and others), 2-30 s, without sound. | cc-by-4.0 | 1.02M items; 2k h; 6,657 GB | av_edit, video_identity | Silent clips: they feed video identity only. Zip parts are 20-50 GB; split parts (partaa/partab) are not read. |
| `wyh6666/InsViE` | One million instructed video edits (Pexels clips edited by generative models). | cc-by-4.0 | 1M items; 394 GB | av_edit | Edits are model-generated; clips have no sound. |


### Cloud jobs

One sandbox per media family, all with the pilot's shape (8 CPU, 32 GiB, one GPU, 12 hours), so
they can run side by side and each writes its own prepared root. Nothing launches by itself;
each needs a clean commit and `--secret hf-token=FILE`. The counts are first-run caps of tens of
thousands of clips per stream; a count is also capped by what the source holds, and a clip that a
filter drops (too long, no sound, no caption) is replaced by the next one.

| Job (`configs/cloud/<name>.json`) | Prepared root | Sources and clip counts |
| --- | --- | --- |
| `prepare-media-edits` | `media-edits` | magicbrush 8,000, ultraedit 10,000, hq_edit 4,000, instructpix2pix 10,000, omniedit 4,000, anyedit 6,000 |
| `prepare-media-gated` | `media-gated` | gigaspeech 10,000, emilia 10,000, finevideo 6,000 |
| `prepare-media-images` | `media-images` | docci 2,000, scienceqa_images 5,000, llava_recap_cc3m 15,000, llava_recap_558k 8,000, llava_recap_118k 8,000, blip3o_long_caption 15,000, text_to_image_2m 15,000, commoncatalog_cc_by 8,000, coco_captions 8,000, recap_coco_30k 8,000, textcaps 8,000, flickr30k 8,000, densefusion_1m 8,000 |
| `prepare-media-sound` | `media-sound` | audioset 20,000, audiocaps 10,000, clotho 3,839, song_describer 746 |
| `prepare-media-speech` | `media-speech` | librispeech_asr 15,000, libritts_r 8,000, mls_english 15,000, mls_multilingual 10,000, peoples_speech 8,000, voxpopuli 8,000, common_voice_17 8,000, yodas_granary_en 8,000, yodas_granary_ast 15,000, covost2 8,000, fleurs 4,000, hifi_tts 4,000, expresso 2,000, lj_speech 3,000 |
| `prepare-media-video` | `media-video` | rekadaily_processed 3,000, vggsound 10,000, openvid_1m 10,000, insvie 5,000 |

To scale up, raise the `--scale` factor in a job's command (and `lifetime_hours`), or copy the
job and launch it several times with different `--shard-slice I/N` values and separate roots. The
`gated` job fails when its first file is read if the account has not accepted the terms, before any
data is processed.

### Gated repositories that need a click on the Hub

With the token in `~/.cache/huggingface/token` (account `gabrielpickard`), reading a data file
of each of these returned HTTP 403 on 2026-10-22, so the account has to accept their terms on
the dataset page before the `gated` job can run: `HuggingFaceFV/finevideo` (expected to be open
already; its card page loads but its data files still return 403), `speechcolab/gigaspeech` and
`amphion/Emilia-Dataset`. Their layouts were taken from the cards and have not been sampled.
Also gated and not used: `OpenGVLab/ShareGPT-4o` and `OpenGVLab/InternVid` (YouTube ids only).

### What the stage plan implies, and the gap

The D meter counts only text tokens, and a media reference costs zero. An episode therefore
adds to D only the text around its media: an identity episode ("Reproduce this image.") is 5
tokens, a one-line edit instruction about 6-25, a transcript about 40, an image caption 16-200.
The scheduler (`MixtureWindowStream`) serves one episode at a time and charges it its own D, so a
stream's share is a share of text tokens, and the number of episodes behind it is the share
divided by that small per-episode cost. At the P1 limit of 400B D the shares therefore imply
billions of media episodes: more than any source can supply once, and probably more compute than
intended. The table is computed from the manifest's inventories (items are unique clips before the
length and caption filters) and from per-episode token counts measured with the reference codec on
real label text from the sources (identity: exact; edit instructions and the captions of sources
that were not sampled: estimated from a similar source).

| Stream | Sources | Unique episodes available | Audio hours | Text tokens per episode | P1 share (tokens) | Episodes the share implies | Passes over the data | Per 0.1% of P1 | First run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audio_identity | 19 | 171M | 460k | 5.0 | 1.50% (6B) | 1.2B | 7.0 | 80M | 167k |
| transcription | 12 | 169M | 455k | 42.0 | 1.50% (6B) | 143M | 0.8 | 9.52M | 119k |
| verbalization | 4 | 696k | 921 | 19.7 | 1.00% (4B) | 203M | 290.9 | 20.3M | 17k |
| speech_translation | 3 | 24M | 60.6k | 73.5 | 1.50% (6B) | 81.6M | 3.4 | 5.44M | 27k |
| speech_edit | 1 | 170M | 456k | 10.0 | 0.75% (3B) | 300M | 1.8 | 40M | 10k |
| image_identity | 13 | 50.7M | - | 5.0 | 1.50% (6B) | 1.2B | 23.7 | 80M | 116k |
| image_description | 12 | 50.7M | - | 142.2 | 1.50% (6B) | 42.2M | 0.8 | 2.81M | 111k |
| image_creation | 12 | 50.7M | - | 158.5 | 1.00% (4B) | 25.2M | 0.5 | 2.52M | 111k |
| image_edit | 6 | 6.95M | - | 11.1 | 1.00% (4B) | 359M | 51.7 | 36M | 42k |
| video_identity | 4 | 5.85M | 11.4k | 5.0 | 2.00% (8B) | 1.6B | 273.7 | 80M | 29k |
| av_creation | 3 | 4.83M | 9.38k | 54.5 | 1.00% (4B) | 73.5M | 15.2 | 7.34M | 19k |
| av_edit | 2 | 6.85M | - | 5.1 | 0.75% (3B) | 583M | 85.2 | 78.4M | 13k |


Columns: "Unique episodes available" counts each source's clips once per stream it feeds; "Text
tokens per episode" is the mean cost in D, weighted by items; "Episodes the share implies" is the
share of 400B divided by that cost; "Passes over the data" is implied over available (below 1
means the data would not be used up); "Per 0.1% of P1" is how many episodes a tenth of a percent
of the P1 budget buys; "First run" is the clips the cloud jobs request (speech_edit counts the
procedural edits; av_edit counts the real InsViE edits plus procedural ones).

Reading it:

- **Transcription, speech_translation, image_description and image_creation are covered** (under
  four passes) once the large sources (Emilia and the YODAS English set for speech, BLIP3o and
  CommonCatalog for images) are built; the first run uses a small slice of them.
- **The identity streams, both edit streams, verbalization and av_creation cannot be filled at
  their shares from unique data**, and the shares themselves buy 0.07-1.6B episodes each: 24
  passes for image_identity, 274 for video_identity, 52 for image_edit, 85 for av_edit, 15 for
  av_creation and 291 for verbalization (four open TTS corpora, 0.7M clips, 921 hours; audio
  identity, at 7 passes, is the least short). Procedural speech and video edits can be generated
  from any clip, so those streams are bounded by the clips, not by the edits.
- The twelve media streams hold 15% of P1 (60B D) and imply about 5.8B episodes (each a training
  window), against 0.17B unique audio clips, 51M images, 4.8M AV clips and 7M edits. A text
  window holds at most 8,192 tokens, so the 60B D would be at most 7M windows if it were text: the
  media streams would dominate the compute of P1 by two to three orders of magnitude.

Recommended share changes (not applied; `stages.json` is untouched). Either of the following
works; they are not both needed:

1. **Cap each media stream by what exists, in the same unit**: `min(current share, 4 passes x
   available episodes x tokens per episode / 400B)`. That gives audio_identity 0.86%,
   verbalization 0.014%, image_identity 0.25%, image_edit 0.077%, video_identity 0.029%,
   av_creation 0.26% and av_edit 0.035%; transcription, speech_translation, image_description,
   image_creation and speech_edit stay within four passes at their current shares. The media
   total falls from 15% to 7.8% of P1 and the released 7.2% goes to text. Audio identity alone is
   still 0.7B episodes at that share, so the compute question remains.
2. **Charge media events in the meter** (a fixed number of D-equivalent tokens per media event,
   for example the realizer's latent length) so that a share means compute, not text. This
   changes training code and every phase's limit, so it is a separate decision.

The M1 and M2 phases add 1% of 60B and 1% of 40B to each identity stream, another 200M episodes
per modality, with the same effect on their edit and creation streams.

### Not added, and why

- **Links only** (an image or video fetch step would be needed and links rot): PixelProse,
  Recap-DataComp-1B, Spawning/PD12M, InternVid, WebVid, HD-VILA, MusicCaps (YouTube ids), and
  Panda-70M (only a small test subset on the Hub has clips).
- **Awkward layouts**: WavCaps (multi-part zips), Ditto-1M and GPT-Image-Edit-1.5M (split
  archives), SEED-Data-Edit and Senorita-2M (per-category archives whose instructions live in
  separate files; the controllable part of Senorita is depth and scribble conversion, not
  editing), OpenVid's `partaa`/`partab` parts, ImgEdit, and LLaVA-Video-178K (long videos with one
  caption for the whole video). ShareGPT4Video has stock-footage videos (mostly silent) with
  several timestamped captions each; it needs a cut-and-caption reader and would add only video
  identity beyond OpenVid. ShareGPT4V is captions only (its images come from other datasets).
- **Captions of little use**: the CC3M/CC12M alt-text sets with images (`pixparse/cc3m-wds`,
  `cc12m-wds`) are replaced by the recaptioned sets above; ScienceQA has no caption and is used
  for image identity only; Densely Captioned Images was not found on the Hub. The captions of
  AudioSet-style sound sets (AudioCaps, Clotho, Song Describer) are not used because no stage
  family describes a sound: those sets feed audio_identity only. A sound-description or
  sound-creation family would need a new family in the concordance compiler and a stage share.
- **Speech-to-speech translation**: SpeechMatrix is not on the Hub and CVSS (`google/cvss`) is a
  loading script that needs CoVoST 2 and Common Voice audio downloaded separately, so FLEURS pairs
  are the only speech-to-speech source. The text-target sources (YODAS-Granary AST and CoVoST 2)
  cover the translation family at scale.
- **HiFi-TTS-2** is a manifest of audio that must be fetched from LibriVox; its 36.7k hours are
  not read.

### Corrections to the pilot section above

The media sources table and the "Still missing" list in the pilot section predate this work:
RekaDaily is now read from all nine subsets (10.2k hours, not only `video_capture`) and its
captions are recorded as model-written; MagicBrush's edited images are recorded as machine-made;
the FLEURS files and the pilot's other media files are streamed from the Hub rather than being
hashed job inputs; FineVideo, Common Voice and CoVoST 2 are now sources (FineVideo gated, Common
Voice through the `fixie-ai/common_voice_17_0` mirror, CoVoST 2 through `fixie-ai/covost2`), and
OpenVid-1M is read for video identity from its zip parts.

## More chat, reasoning and agent data

The first agentic manifest (`sources.agentic.json`, above) leaves the chat, reasoning and
tool-use streams far below what the stage plan asks for. This section adds a second sweep:
417 further open sources and 23 sources that need a click on the Hub. Coverage of coding
agents, terminal agents and code-execution traces is a separate piece of work; a few
such sources that were added before that split are listed under "Coding sources kept" and are
not extended here.

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

The manifests are `configs/data/sources.agentic-more.json` (417 sources over 250 repositories)
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

#### tool_use (open, 23 sources, 7.2B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `smoltalk-apigen-80k` | `HuggingFaceTB/smoltalk` (apigen-80k) | cc-by-4.0 | 8,000 | 15M |
| `cascade-s2-tool-calling` | `nvidia/Nemotron-Cascade-SFT-Stage-2` (tool_calling) | cc-by-4.0 | 3,000 | 1.7B |
| `arc-agi-large-tools-noreasoning` | `nvidia/Nemotron-SFT-ARC-AGI-v1` (large_tools_no_reasoning) | cc-by-4.0 | 2,500 | 7.0M |
| `smoltalk2-smolagents-toolcalling-traces-think` | `HuggingFaceTB/smoltalk2` (SFT) | not stated | 3,000 | 64M |
| `smoltalk2-hermes-function-calling-v1-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | apache-2.0 | 3,000 | 11M |
| `smoltalk2-xlam-traces-no-think` | `HuggingFaceTB/smoltalk2` (SFT) | cc-by-4.0 | 3,000 | 29M |
| `ultradata-agent-tool-use` | `openbmb/UltraData-SFT-Agent-2609` (Tool-Use) | apache-2.0 | 3,000 | 170M |
| `swe-qa-pro-sft` | `TIGER-Lab/SWE-QA-Pro-SFT-Trajectories` | mit | 1,000 | 16M |
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

#### executed_trajectories (open, 20 sources, 27.6B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `arc-agi-large-reasoning-tools` | `nvidia/Nemotron-SFT-ARC-AGI-v1` (large_reasoning_and_tools) | cc-by-4.0 | 1,000 | 1.0B |
| `ultradata-agent-search` | `openbmb/UltraData-SFT-Agent-2609` (Search-Agent) | apache-2.0 | 2,000 | 390M |
| `ultradata-agent-code` | `openbmb/UltraData-SFT-Agent-2609` (Code-Agent) | apache-2.0 | 1,000 | 3.7B |
| `ultradata-agent-general` | `openbmb/UltraData-SFT-Agent-2609` (General-Agent) | apache-2.0 | 1,000 | 8.5B |
| `swe-next-sft` | `TIGER-Lab/SWE-Next-SFT-Trajectories` | mit | 2,000 | 50M |
| `scale-swe-distilled` | `AweAI-Team/Scale-SWE-Distilled` | cc-by-4.0 | 2,000 | 3.5B |
| `terminaltraj-20k` | `m-a-p/TerminalTraj` | not stated on the card | 2,000 | 135M |
| `sera-lite-t2-glm46` | `allenai/Sera-4.6-Lite-T2` | odc-by | 1,500 | 1.1B |
| `tmax-sft-qwen36-success` | `allenai/tmax-sft` (skill_tax_20260505_2.2k_combined_balanced_thinking_only_success) | odc-by | 3,000 | 19M |
| `otagent-sft-100k` | `open-thoughts/OpenThoughts-Agent-SFT-100K` | apache-2.0 | 4,000 | 910M |
| `otagent-sft-coldstart-9k` | `open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K` | apache-2.0 | 4,000 | 187M |
| `terminal-lego-traj-8k-4teachers` | `Lego-X/Terminal-Lego-Traj-8k` | none stated | 3,000 | 210M |
| `tmax-sft-big-ntc-adapters` | `allenai/tmax-sft-big` | odc-by | 3,000 | 1.0B |
| `toolmind-web-trajectories` | `Nanbeige/ToolMind-Web-QA` | apache-2.0 | 500 | 390M |
| `adp-openhands-feedback` | `neulab/agent-data-collection` | mit | 200 | 4.0M |
| `adp-swe-play` | `neulab/agent-data-collection` | see swe-play-trajectories/LICENSE  | 500 | 35M |
| `adp-mini-coder` | `neulab/agent-data-collection` | see mini-coder/LICENSE in the repo | 1,000 | 1.0B |
| `adp2-openresearcher` | `neulab/adp-v2` | mit | 800 | 4.8B |
| `adp2-swe-chat` | `neulab/adp-v2` | odc-by | 500 | 550M |
| `jupyter-agent-thinking` | `jupyter-agent/jupyter-agent-dataset` | apache-2.0 | 2,600 | 115M |

#### verified_trajectories (open, 3 sources, 2.2B)

| Source id | Dataset | Licence on the card | First-run cap (rows) | Unique tokens |
| --- | --- | --- | --- | --- |
| `scale-swe-v4pro-41k` | `AweAI-Team/Scale-SWE-Distilled-DeepSeek-v4-Pro-High-41k` | cc-by-4.0 | 2,000 | 1.5B |
| `denovoswe-filtered` | `AweAI-Team/DeNovoSWE-Trajectory-Filtered` | cc-by-4.0 | 4,500 | 500M |
| `swe-lego-real-resolved` | `PrimeIntellect/SWE-Lego-Real-Data-Verified` | apache-2.0 | 500 | 231M |

#### synthetic_code_dialogue (open, 28 sources, 22.6B)

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
| `code-datascience-instruct-analysis` | `RUC-DataLab/DataScience-Instruct-500K` | mit | 3,000 | 130M |
| `nemotron-sft-competitive-programming-v2-exercism` | `nvidia/Nemotron-SFT-Competitive-Programming-v2` (default) | cc-by-4.0 | 3,000 | 172M |
| `opencodegeneticinstruct-mixtral` | `nvidia/OpenCodeGeneticInstruct` (mixtral-8x22b-instruct) | cc-by-4.0 | 3,000 | 8.6B |
| `opencodegeneticinstruct-qwen25` | `nvidia/OpenCodeGeneticInstruct` (qwen2.5-32b-instruct) | cc-by-4.0 | 3,000 | 6.2B |

### Coding sources kept

These sources are coding-agent, terminal-agent or code-execution data, added before that
work moved to its own owner. They stay in the manifest but are not extended here:
`swe-next-sft`, `swe-qa-pro-sft`, `scale-swe-v4pro-41k`, `denovoswe-filtered`,
`scale-swe-distilled`, `sera-lite-t2-glm46`, `tmax-sft-qwen36-success`,
`tmax-sft-big-ntc-adapters`, `otagent-sft-100k`, `otagent-sft-coldstart-9k` (both partly overlap
AgentTrove), `swe-lego-real-resolved`, `terminal-lego-traj-8k-4teachers`, `terminaltraj-20k`,
`adp-swe-play`, `adp-mini-coder`, `adp-openhands-feedback`, `adp2-swe-chat`,
`jupyter-agent-thinking`, `code-code-feedback`, `code-datascience-instruct-analysis`,
`math-numinamath-tir`, `ultradata-agent-code`, `ultradata-agent-general`,
`ultradata-agent-search`, and the 28 code-instruction sources in `synthetic_code_dialogue`.
`davinci-dev-env-native` and `openswe-trajectories` were looked at and left to that work.

### Sources that need a click on the Hub

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
`scripts/scale_source_caps.py`, multiplies every cap by `AGENTIC_MORE_ROW_SCALE` (0.25) and
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
