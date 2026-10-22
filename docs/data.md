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
