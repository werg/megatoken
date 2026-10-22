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

Inference's logical `observe` method applies the same content chunker to a complete
authoritative event. Already prepared records use `observe_physical` to preserve
supplied boundaries and token IDs. Every selected action advances recurrent state
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

## Open-licence pilot mix

Every declared stream needs prepared episodes before its stage can train. This section
records where each stream's pilot data comes from, the licence found on the publisher's
dataset card (read 2026-10-22; a card is not legal advice), and what is still missing. The
pilot is small (a few thousand episodes per stream) so the full pipeline can be exercised;
the stage plan's targets are orders of magnitude larger.

`configs/cloud/prepare-open-mix.json` builds four prepared roots in one sandbox (nothing
launches by itself; it needs the approvals below, a clean commit, and
`--secret hf-token=FILE`):

| Root | Built by | Streams |
| --- | --- | --- |
| `text-p1` | `megatoken prepare --sources configs/data/sources.open.json` | natural_documents, natural_code, synthetic_documents, synthetic_reasoning, chat, synthetic_code_dialogue, executed_trajectories |
| `text-mb` | `megatoken prepare --sources configs/data/sources.open-mb.json` | text (replay for the media bridge) |
| `media` | `scripts/prepare_open_media.py` (readers in `data/open_media.py`) | audio_identity, image_identity, video_identity, transcription, verbalization, speech_translation, image_description, image_creation, image_edit, speech_edit, av_creation, av_edit |
| `long-reading` | `scripts/prepare_long_document_reading.py` | long_document_reading |

MB trains on `text-mb` + `media` (identity streams only); the reduced P1 trains on all four.
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
