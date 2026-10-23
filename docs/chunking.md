# Recursive semantic chunking

Preparation converts each logical event into exact byte slices that fit the model's
output capacity. It first understands the event's framing, then its document or
program structure, then content embedded inside that structure. A bounded planner
uses those observations to choose a coherent sequence of chunks with foresight.

The production defaults are **512 reference-codec tokens as the packing target,
768 as the soft limit, and 1,024 as the hard limit**, with an independent
**16,384-byte hard limit**. These supersede the earlier 256/512 preparation policy.
They do not change the model's 32-facet state format or increase its output capacity.
The target is an optimization preference, not a requirement that every paragraph,
function, message, or physical record have the same length.

The implementation and preparation identity are versioned as `recursive-parser-v2`.
The architectural decision is recorded in [ADR 0005](decisions/0005-recursive-parser-chunking.md).
See [data preparation](data.md) for source manifests, adapters, and training artifacts.

## Inspect before preparing a corpus

The base package includes its parser dependencies. Verify the installed grammar
bindings and inspect a representative file with the same reference codec that
will be used for preparation:

```sh
megatoken chunks coverage --output runs/grammar-coverage.json

megatoken chunks inspect \
  --input examples/document.md \
  --codec artifacts/codec \
  --content-kind markdown \
  --locale en-US \
  --output runs/document-chunks.json
```

Replace the example paths with actual inputs. `--codec byte` is available for
fixtures and debugging; its counts are byte-token counts, so it is not a substitute
for inspecting a production codec. `coverage` loads every advertised local binding
by default. `--catalog-only` returns the installed catalog without verifying that
each binding loads. Neither mode downloads grammars.

Inspection reports the source hash, parser and codec identity, canonical event
ranges, parser boundaries and regions, selected byte/token ranges, languages,
diagnostics, tree movement, and candidate/edge counts. `--include-payload` adds the
exact payload as hexadecimal and the selected token IDs. Ordinary reports contain
coordinates and metadata rather than copies of the input text.

Useful hints include `--language python`, `--filename component.vue`,
`--mime-type text/html`, `--locale ja-JP`, and `--chat-template chatml`.
The filename defaults to the inspected input's name. Choose `--chat-template none`
when the input is ordinary content that happens to discuss serialized chat formats.

### Run the real-codec validation pilot

The reusable [validation script](../scripts/validate_chunking.py) exercises six
repository Python/Markdown files plus mixed Unicode/document, raw ChatML, and
decorated-function fixtures with the actual reference codec:

```sh
megatoken codec --output artifacts/codec
python scripts/validate_chunking.py \
  --codec artifacts/codec \
  --output validation/chunking-run-1 \
  --repeats 3
```

The output directory must be fresh. The script saves input snapshots, source and
canonical-ID hashes, identities, chunk-size distributions, parser diagnostics,
planner work counts, and timing receipts. It checks exact source bytes, canonical
per-event IDs, hard capacities, causal boundaries, expected nested parsers, and
heading/signature attachments under feasible canonical tokenization. Independent
fixture expectations cover a heading followed by an oversized paragraph and a
decorated function with a large body. Correctness failures exit nonzero and retain
their reports. With one repetition, repeatability is explicitly unmeasured.

The ChatML fixture tests parsing and framing only: its tool role has no causal
call/execution IDs, so it is not a preparation-accepted training episode. The pilot
does not replace source, quality, dependency, or deduplication gates. Its local CPU
timings do not establish general corpus throughput, GPU performance, or model
quality. `--repo-root` can select another checkout; it otherwise defaults to the
checkout containing the script.

## One pipeline, shared contracts

The implementation separates five responsibilities:

| Component | Responsibility | Output |
| --- | --- | --- |
| Chat framing | Identify outer message headers, bodies, terminators, and ownership | Original byte ranges and mandatory event boundaries |
| Content parsers | Find document/program structure and embedded content | Structural regions, candidate boundaries, and injection ranges |
| Prose analysis | Find paragraphs, then refine only oversized paragraphs | Unicode sentence, word, and line candidates |
| Canonical token index | Encode each logical event once with the reference codec | Stable token IDs and their original byte spans |
| Planner | Select a coherent path through feasible cuts | Physical chunks with explanations and diagnostics |

Parsers return the same `Boundary`, `Region`, `Diagnostic`, and `ParseResult`
contracts from [parsing/types.py](../src/megatoken/data/parsing/types.py).
Every coordinate addresses the original input bytes. A region carries a kind,
language, nesting depth, optional protection, and a structural path. Parsing
contributes evidence; it does not own tokenization or materialize training records.

### Chat framing comes first

Serialized chat is framed before reference tokenization. Each resulting logical
event is then encoded once, and physical chunks partition those existing IDs.
This order matters for codecs that could otherwise merge a token across a message
boundary. Canonical IDs for framed chat are the concatenation of the separately
encoded events, which need not equal one unconstrained encoding of the entire
serialized transcript.

Built-in framing families are ChatML/Qwen, Llama 3, Gemma, Harmony, legacy Harmony,
Phi, Llama 2/Mistral instruction framing, and DeepSeek. The exact supported
delimiters and aliases live in [parsing/chat.py](../src/megatoken/data/parsing/chat.py).
Detection is conservative, with explicit diagnostics for ambiguous templates,
unknown ownership, unrecognized headers, or unframed tails. Explicit template
selection is preferable when the source format is known.

Framing runs only at the outer boundary. A chat marker quoted in code, a string,
or a message's nested document is not recursively promoted to a new role. Structured
conversation and tool adapters already know their event boundaries and disable
serialized-chat detection inside those fields. Message, observation, tool, and
other causal boundaries remain mandatory; packing never combines work from
opposite sides of those boundaries. Native generated records retain their saved
emissions and facet tapes rather than being silently rechunked.

### Recursive content and exact source views

Language selection uses an explicit language hint first, then filename, MIME type,
and finally conservative source signatures. Aliases cover common extensions,
shebangs, fence information strings, and forms such as `{.python}` or
`language-javascript`. Hints are evidence for a grammar, not assertions that a
snippet is a complete or valid program. An explicit unknown programming language
remains visible as an unavailable grammar instead of being relabeled as prose.
For explicitly selected text/prose, a BCP 47-shaped hint can supply a locale when
it does not match a programming language; separate `language` and `locale` metadata
is the clearer source contract.

The engine composes document and syntax parsers recursively. Representative paths
include:

| Host content | Nested content handled |
| --- | --- |
| Markdown | Fenced and indented code, inline syntax, HTML blocks, lists, tables, prose |
| HTML/XML | Element text, selected natural-language attributes, script/style bodies, CDATA, embedded HTML |
| JavaScript/TypeScript | Template strings, tagged SQL/GraphQL/HTML/CSS, comments, ordinary strings |
| General programming languages | Comments, docstrings, string literals, supported raw strings and heredocs |
| Vue, Svelte, Astro | Markup, script/style blocks, interpolation expressions, frontmatter |
| ERB/EJS and template documents | Host markup with embedded code or template directives |
| reStructuredText, Org, AsciiDoc | Code/literal/source blocks and surrounding document text |
| LaTeX | `minted`, `lstlisting`, and verbatim blocks |
| YAML | Block scalar content and surrounding structured data |

These paths compose: a comment can contain Markdown, whose code fence contains
HTML, whose script contains another language. Recognizers retain partial structure
for unfinished fences, incomplete statements, and malformed markup. Not every
language has every injection rule; a grammar catalog entry establishes syntax
availability, not complete knowledge of all of that language's embedding conventions.

Some nested content needs a virtual view. Consecutive line comments can be parsed
together after excluding their comment prefixes, and a docstring can be inspected
without its common indentation. Those views are ordered references to unchanged
source ranges. Nested offsets map back through the view to original byte offsets;
the output always slices the original source. The engine never decodes string
escapes or HTML entities, normalizes whitespace or Unicode, re-renders Markdown,
appends missing delimiters, evaluates source code, or follows external includes.

### Broad, explicit offline coverage

The default is the bundled `tree-sitter-language-pack==0.13.0` with
`tree-sitter==0.25.2`. The pinned environment exposes **173 loadable canonical
grammar bindings**, verified by the coverage command. The catalog includes major
systems, scripting, functional, query, markup, configuration, hardware, and
specialized languages. This is the project's tested baseline, not a claim to parse
every existing or future language. The complete current catalog is executable
evidence from `megatoken chunks coverage`, rather than a duplicated list here.

[Tree-sitter](https://tree-sitter.github.io/tree-sitter/) supplies concrete syntax
trees and recovery around syntax errors. The
[pinned language pack release](https://pypi.org/project/tree-sitter-language-pack/0.13.0/)
bundles the grammar assets. The registry deliberately refuses an incompatible
package version instead of entering a newer API's implicit download path.
Grammar provisioning belongs to environment construction; corpus parsing has no
runtime network dependency.

An unavailable grammar uses bounded lexical structure: comments, literals,
delimiters, statements, and line boundaries, with Unicode analysis for discovered
prose. `parser_backend: fallback` explicitly selects this mode. Markdown also has
a `markdown-it-py` document fallback, and document-container recognizers cover
several formats independently of a native grammar. These paths preserve source
bytes and hard capacities while reporting the reduced evidence. They do not claim
to produce a full syntax tree for an unknown language.

The `generic_lexical:v2` recovery path now distinguishes language-family
comment markers instead of treating every `#`, `--`, or `//` as a comment.
It recognizes nested Rust, Swift, Haskell, and OCaml-style block comments,
Rust/C++ raw strings, Rust lifetime labels, and CR-only line breaks. Unknown
dialects apply only conservative line-leading comment recognition; unsupported
syntax must remain visible for subsequent parser improvements, not be silently
promoted to incorrect structure. The HTML recovery path also separates
source-exact event/style/prose attribute values and uses script/style
`type` and `lang` metadata, including data-script and template types.
Neither fallback decodes entities, executes source, or claims full grammar
equivalence. See the [recovery regression matrix](../tests/test_parser_recovery_depth.py)
and the [updated validation status](validation.md#pr-6-follow-up-integration--2026-10-19).

CSV, TSV, and pipe-separated records now have an explicit
[`delimited-records-v1` scanner](../src/megatoken/data/parsing/tabular.py)
instead of relying on generic punctuation heuristics. It handles CSV quoted fields,
escaped quotes, embedded LF/CR/CRLF, empty fields, and malformed records with
bounded, source-exact row/field evidence. TSV and pipe-separated files have no
quoting convention, so a quotation mark there is ordinary text. Row protection is soft: records larger
than a physical chunk cap can still be subdivided, and canonical token IDs
remain unchanged. This is a structural lexer, not a full CSV schema validator
or dialect auto-detection engine; callers should supply `language`, a `.csv`,
`.tsv` or `.psv` filename, or a corresponding `content_kind`. Regression
tests are in [test_tabular_parser.py](../tests/test_tabular_parser.py).

Local grammar extensions use
[`register_parser_provider`](../src/megatoken/data/parsing/registry.py), with a
provider name/version, canonical language names, an offline parser factory, and
optional alias/extension/MIME mappings. Factories return independent parser
instances and must be importable module-level callables for isolated workers.
Register them in the application before preparation, using the same registration
in inspection and replay. Provider versions and mappings enter the identity;
increment the provider version when its grammar or factory behavior changes.
Existing grammar names cannot be silently overridden. The stock CLI has no
automatic plugin discovery or grammar-download command.

## Natural language: paragraphs first, refinement when needed

The default prose implementation uses `uniseg==0.10.1` and Unicode **16.0.0**.
It recognizes blank-line paragraphs, Unicode paragraph separators, common line
separator conventions, and ideographic paragraph indentation. Ordinary soft line
wrapping stays within its paragraph. This provides an inexpensive, broad first
approximation before language-specific work.

Only a paragraph that exceeds the 768-token soft limit or the byte limit requests
sentence and word refinement. Refinement uses Unicode text segmentation, with
additional conservative punctuation and abbreviation handling, and offers line
boundaries as lower-priority evidence. The default grapheme treatment preserves
combining sequences, emoji sequences, and CRLF at feasible cuts. Malformed UTF-8
is retained as opaque source bytes and diagnosed rather than replaced.

There are 20 abbreviation profiles, selected from the base of a supplied locale:

| Locale codes | Profiles |
| --- | --- |
| `en`, `de`, `fr`, `es`, `pt`, `it`, `nl` | English, German, French, Spanish, Portuguese, Italian, Dutch |
| `sv`, `da`, `nb`, `nn`, `fi` | Swedish, Danish, Norwegian Bokmål/Nynorsk, Finnish |
| `pl`, `cs`, `sk`, `hu`, `ro` | Polish, Czech, Slovak, Hungarian, Romanian |
| `ru`, `uk`, `tr` | Russian, Ukrainian, Turkish |

These are modest suppression rules for common false sentence endings. Other
languages still receive the Unicode baseline and writing-system punctuation
handling. There is no statistical language detector or universal linguistic
parser. Unicode's [text segmentation specification](https://www.unicode.org/reports/tr29/tr29-45.html)
explicitly permits tailoring; dictionary or other language-specific mechanisms
can be necessary for reliable word boundaries in some writing systems.

Optional ICU sentence/word refinement is selected explicitly:

```sh
python -m pip install -e '.[data,i18n]'
```

Set `use_icu: true` in the chunking configuration and supply a useful locale.
Merely installing ICU does not change boundaries. Selecting it without the
dependency raises a configuration error. Its Python binding, ICU library, and
Unicode versions enter the preparation identity, so pin that environment before
building a corpus. ICU's UTF-16 offsets are mapped back to original byte positions.
See the [ICU boundary analysis guide](https://unicode-org.github.io/icu/userguide/boundaryanalysis/)
for the underlying locale and dictionary behavior. Neither baseline nor ICU
refinement is a full grammatical analysis of natural language.

## Packing with foresight and structural cohesion

The planner turns parser evidence into candidates at canonical token edges.
A suggested boundary inside a BPE token or grapheme projects to a nearby feasible
edge; a selected move records `semantic_boundary_projected`. This never retokenizes
or rewrites the source. A token can span the end of one sentence and the start of
the next, so protection preserves their coherent interiors while allowing that
indivisible edge token to belong to either neighbor. Otherwise, ordinary adjacent
sentences could accidentally become one unbreakable paragraph. A sparse set of
fallback cuts lets a large literal or unknown-language region make progress.

Protected functions, blocks, paragraphs, list items, and other coherent regions
retain their feasible whole units and interiors within the soft token and hard
byte limits. A heading stays
attached to following content, and a function's decorators/signature stay attached
to its initial body. These attachments, and complete JSON units, may use the hard
capacity when needed. An oversized following section or function begins a new
chunk where the canonical token stream permits it, preventing its introduction
from being borrowed to fill the previous chunk. When an attachment's complete
body exceeds capacity, protection retains its first complete sentence or statement
that fits, or a useful word/grapheme prefix when that unit is also oversized.
Indentation and blank separators alone do not count as a body. If even the minimum
meaningful attachment exceeds hard capacity, smaller explicit cuts are unavoidable.

Two separate atomic units can occasionally demand the same BPE token while their
combined span exceeds hard capacity. The earlier unit retains that token and
`token_overlapping_protection` records the necessary ownership relaxation. This is
physical alignment evidence; `crossing_protected_regions` is reserved for actual
overlapping source constraints that cannot fit together.

A backward dynamic program chooses a complete path through the retained candidate
graph. Each candidate considers at most `planner_max_edges` successors, 64 by
default. The objective balances target size, soft-limit overflow, small tails,
boundary quality, and movement across meaningful parser ancestors. Moving between
unrelated code subtrees costs more than comparable prose movement. Leaf-token
churn and trailing whitespace are not treated as new semantic topics.

This gives the planner foresight: it can choose an earlier cut to preserve the next
unit or avoid a poor remainder. It finds the best path in its bounded graph, not
an exhaustive optimum over every conceivable source partition. A `tree_distance`
on each chunk explains structural movement, not model quality. Tree and size
weights remain explicit policy settings for corpus evaluation.

Hard token/byte limits and causal boundaries take precedence over semantic
preferences. When no preferred structural cut is feasible, the report records
`forced_grapheme_cut`; when even a grapheme cannot fit, it records
`forced_token_cut`. Neither diagnostic implies missing or rewritten bytes. A
single codec token larger than the byte capacity is a configuration/input failure
because preserving its identity and satisfying that capacity are incompatible.

## Configuration and Python integration

An explicit policy file can contain only the settings being selected; omitted
fields use the versioned defaults:

```yaml
version: recursive-parser-v2
target_tokens: 512
soft_max_tokens: 768
hard_max_tokens: 1024
max_bytes: 16384
parser_backend: tree_sitter
chat_template: auto
use_icu: false
planner_max_edges: 64
limits:
  max_depth: 12
  max_tree_depth: 128
  max_nodes: 100000
  max_parse_bytes: 1048576
  max_regions: 20000
  max_injections: 2048
  isolate_native: true
  worker_timeout_ms: 5000
```

Use `--chunking-config path/to/policy.yaml` with inspection or preparation.
Explicit `--chunk-target-tokens`, `--chunk-soft-max-tokens`, and
`--parser-backend` options override their policy fields. The stage's
`--max-record-tokens` and `--max-record-bytes` set the hard physical-record limits,
lowering or raising the policy's own (the target and soft limits only ever move down
to fit). Validation requires `1 <= target <= soft <= hard`.

Applications use the same entry point as the CLI and adapters:

```python
from pathlib import Path

from megatoken.data.chunking import ChunkingConfig, segment_with_report
from megatoken.loading import load_codec

payload = Path("examples/document.md").read_bytes()
codec = load_codec("artifacts/codec")
policy = ChunkingConfig()
report = segment_with_report(
    payload,
    codec,
    policy,
    content_kind="markdown",
    filename="document.md",
    locale="en-US",
    chat_template="none",
)
assert b"".join(chunk.payload for chunk in report.chunks) == payload
assert all(len(chunk.token_ids) <= policy.hard_max_tokens for chunk in report.chunks)
assert all(len(chunk.payload) <= policy.max_bytes for chunk in report.chunks)
details = report.to_dict()
identity = policy.identity(codec)
```

Use `segment_bytes` when only the chunks are needed, or
[`parse_content`](../src/megatoken/data/parsing/engine.py) for content evidence
without framing or packing. Source adapters propagate content kind, language,
filename, MIME type, locale, and explicit template hints. Avoid introducing a
separate segmentation path in an adapter or training stage.

## Work limits and native parser recovery

Recursive work shares byte, depth, node, region, boundary, and injection budgets.
`max_depth: 12` bounds recursive language/content injections. The separate
`max_tree_depth: 128` bounds native syntax-tree traversal and ancestor paths before
evidence is sent between processes. Deeper subtrees are pruned while later
siblings remain available, with `tree_depth_budget_exhausted` sent to review.
The byte budget counts parsing work, including repeated inspection through nested
views; it is distinct from a physical chunk's 16 KiB capacity. Exhausting a budget
retains available evidence and reports degradation. It does not truncate or drop
the source's remaining bytes.

Native grammars run in a persistent subprocess by default. One preparation process
shares one serialized worker across its threads; further preparation processes
have their own workers. The configured 5-second request watchdog covers startup
when needed and the request's pipe I/O under one deadline. A crash, timeout, invalid
reply, or provider reconstruction failure yields diagnostics and generic recovery.
The worker is restarted before later native work. Startup is amortized, provider
changes restart the worker, and forked preparation processes do not reuse their
parent's parser pipe.

This isolates native failures from preparation, but is not a sandbox for trusted
grammar-provider code. No input snippet is executed. The default native internal
timeout is disabled (`max_parse_milliseconds: 0`); the external watchdog is the
hang bound. Byte and evidence caps are deterministic, while wall-clock watchdogs
can fire differently under resource contention. Such failures enter review rather
than silently becoming accepted alternate corpus boundaries. Native parsing work
itself is bounded by the watchdog; tree traversal limits do not count native
parser instructions.

## Parser quarantine and agent repair

Preparation enables parser review by default. Ordinary unknown-language fallback,
localized syntax recovery, and incomplete snippets remain useful training data.
They are accepted unless the selected review policy explicitly requests stricter
handling. Actual parser faults, exhausted budgets, severely damaged syntax
coverage, and ambiguous chat ownership are quarantined automatically.

| Evidence | Default treatment |
| --- | --- |
| `grammar_unavailable` | Accept the diagnosed fallback |
| `recovered_syntax`, `syntax_error`, `missing_syntax` | Accept localized recovery |
| `semantic_boundary_projected`, `token_overlapping_protection` | Accept diagnosed canonical alignment/ownership decisions |
| Unclosed fence/literal/markup or unmatched delimiter | Accept incomplete content |
| `low_syntax_coverage` | Queue for repair; ERROR regions cover at least 60% of source bytes |
| Native worker failure, invalid evidence/injection, ICU failure | Queue for repair |
| Any `*_budget` or `*_budget_exhausted` diagnostic | Queue for repair (`depth_budget_exhausted` only while `review_depth_limit` is true; `parser-review.agentic.json` sets it false, because shell inside JSON inside quoted terminal screens runs past the recursion limit and stays one flat region) |
| Ambiguous or unrecognized chat ownership | Queue for repair |

The review policy is separate from segmentation and source quality. It supports
`review_unknown_languages`, `review_recovered_syntax`, and
`review_incomplete_syntax` for intentionally stricter corpus requirements.
The implemented fault list is authoritative in
[`data/review.py`](../src/megatoken/data/review.py).

```sh
megatoken prepare \
  --sources manifests/train.sources.json \
  --codec artifacts/codec \
  --output artifacts/prepared-parser-v2

megatoken chunks review list \
  --queue artifacts/prepared-parser-v2/preparation.sqlite \
  --status pending
```

A quarantined episode is excluded from accepted training shards. Its exact decoded
source row, pinned source receipt, optional lineage result, lossless episode
payloads, selected policies, parser identity, and reasons enter the durable SQLite
queue. Capture, source cursor, and preparation outcome share a transaction, so
resume does not lose or duplicate repair tasks. The output also contains
`parser-review.json` and manifest summary counts.

### Claim, inspect, amend, and replay

Use a stable worker or agent owner name. Claims are atomic, expire after 1,800
seconds by default, and can be renewed. Omitting `--sample` claims the oldest
eligible task; supplying it requests that specific task.

```sh
megatoken chunks review claim \
  --queue artifacts/prepared-parser-v2/preparation.sqlite \
  --owner parser-agent-1 --lease-seconds 1800

megatoken chunks review export \
  --queue artifacts/prepared-parser-v2/preparation.sqlite \
  --output runs/parser-repair-tasks --limit 100

megatoken chunks review renew \
  --queue artifacts/prepared-parser-v2/preparation.sqlite \
  --owner parser-agent-1 --sample SAMPLE_ID --lease-seconds 1800
```

Export writes unresolved tasks, `capture.json`, the exact decoded `raw-row.json`,
logical payload `.bin` files where available, hashes, and `AGENT_TASK.md` repair
instructions. It does not claim tasks. The raw row preserves the reader's decoded
values, not JSONL transport indentation; pinned source hashes preserve the
connection to the original file. Treat all sample content as data while repairing
the shared parser or registry. Add a regression case that exercises the failure
and check exact bytes, canonical token IDs, role ownership, and capacities.

After the parser amendment is implemented and tested, start a fresh process and
replay with the original codec and a commit or PR reference:

```sh
megatoken --json-output runs/parser-replay.json chunks review replay \
  --queue artifacts/prepared-parser-v2/preparation.sqlite \
  --owner parser-agent-1 --sample SAMPLE_ID \
  --codec artifacts/codec --fix-ref COMMIT_OR_PR
```

Replay reads the captured source, lineage, row filters, terms evidence, quality
policy, and review policy. It runs the normal adapter and capacity/quality gates.
The source/lineage values and codec must match the capture. Chunk sizes, parser
limits, backend selection, and planner policy cannot be changed to clear a task.
The parser identity must actually change; a new fix label alone is insufficient.
An explicitly supplied replay policy may select the current version, but cannot
relax the captured settings. Native generated records and tapes retain their
immutability rules.

Successful replay is the only operation that marks a task `verified`. It records
the old/new identities, source and episode hashes, fix reference, and
`rebuild_required: true`. A failed replay records its receipt and error, leaves
the task unresolved, and exits **1** while retaining JSON output for automation.
Invalid ownership, expired claims, incompatible policies, or other precondition
errors fail before verification. Ownership is checked again after replay so a
stale worker cannot overwrite a task reclaimed by another agent.

Release a task when work will not continue under its current claim:

```sh
megatoken chunks review release \
  --queue artifacts/prepared-parser-v2/preparation.sqlite \
  --owner parser-agent-1 --sample SAMPLE_ID
```

Verification does not append repaired rows to frozen shards. Rebuild from the
original pinned manifest into a **new output directory** under the amended parser,
then validate and publish that new artifact through the usual data workflow.
The queue automates durable capture and verification; it does not automatically
launch coding agents, approve parser changes, or replace prepared datasets.

## Identity, validation, and operational limits

`ChunkingConfig.identity(codec)` records the complete policy, reference codec
fingerprint, chat delimiters, Unicode/ICU runtime, grammar versions, registered
providers, and an implementation content hash. The content hash covers parsing
modules plus chunking, adapter, and schema code, so a local parser fix changes
identity without requiring a package release. It is cached within a process:
restart long-running preparation or inspection after changing parser code.

Prepared artifacts reject incompatible identity reuse. A runtime or grammar
upgrade, parser amendment, different codec, changed template policy, or new
packing policy calls for a new preparation artifact. Keep the coverage report,
representative inspection reports, policy, and dependency environment with the
run receipt. A reproducible policy identifies the exact tested implementation;
it does not claim that every grammar or every natural-language boundary is correct.

Regression coverage is grouped in
[parser](../tests/test_chunking_parsers.py),
[prose](../tests/test_chunking_prose.py),
[chat](../tests/test_chunking_chat.py),
[planner](../tests/test_chunking_planner.py),
[worker](../tests/test_parser_workers.py), and
[review](../tests/test_parser_review.py) tests, with CLI and preparation integration
tests exercising the same public path. Representative corpus inspection and queue
rates are still necessary before a large run: syntax recovery is an approximation,
locale profiles are limited, and nesting conventions evolve. This infrastructure
makes those cases visible and repairable. It does not establish training quality,
Blackwell throughput, or GPU inference performance.

Additional primary references are the
[CommonMark specification](https://spec.commonmark.org/0.31.2/),
[markdown-it-py documentation](https://markdown-it-py.readthedocs.io/), and
[uniseg documentation](https://uniseg-py.readthedocs.io/).
