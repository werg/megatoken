# ADR 0005: recursive parsing with bounded structural packing

Status: accepted for preparation version `recursive-parser-v2`.

## Context

Training sources mix prose, code, markup, structured data, and serialized chat.
A useful chunk boundary can depend on an outer message, a document section, a
function, a docstring, or another language several levels inside those containers.
Greedy punctuation/size rules lose that hierarchy and can detach a heading or
function signature from the content it introduces. Requiring fully valid syntax
would discard common training examples such as partial programs and conversations
with unfinished outputs.

The model has 32-facet megatokens and a hard output capacity of 1,024 positions.
Preparation must preserve original bytes, canonical reference IDs, role ownership,
and generated record/tape identities while using that capacity coherently. Parser
failures also need an operational repair path rather than silent corpus drift.

## Decision

Set the packing target to **512 reference-codec tokens**, the soft limit to **768**,
and the hard limit to **1,024**, with an independent **16,384-byte hard limit**.
These supersede the 256/512 preparation defaults in
[ADR 0004](0004-contextual-realization.md). The model architecture and native
generated-state semantics remain those of that decision.

Frame serialized chat before tokenization. Encode each logical event once, then
partition that fixed canonical stream. Structured adapters supply their known
causal boundaries directly. Parse message contents recursively without interpreting
nested chat-like strings as new message ownership.

Use shared byte-coordinate contracts for all parsers. The engine recursively
combines concrete syntax trees, document-container recognizers, and Unicode prose
analysis. Source views can omit comment prefixes or common indentation for
analysis while retaining an exact mapping to the original byte ranges. Output
materialization only slices original bytes; it does not normalize, decode escapes,
re-render, execute, or fetch referenced content.

Use the pinned offline Tree-sitter language pack 0.13.0 and runtime 0.25.2 as the
baseline. The installed catalog has 173 verified grammar bindings. Unknown
languages and recoverable syntax use a diagnosed, bounded lexical fallback.
Additional local grammar providers are explicit and versioned. Runtime parsing
does not provision or download grammars.

Start prose analysis with broad paragraph detection. Refine paragraphs only when
they exceed the soft token limit or byte capacity, using the Unicode 16 baseline
and conservative abbreviation profiles for 20 locales. Optional ICU refinement is
explicit, with its runtime versions included in identity. This is practical text
segmentation rather than universal linguistic parsing.

Plan a complete sequence with dynamic programming over a bounded candidate graph.
Score size fit, soft overflow, small tails, boundary quality, and movement between
meaningful structural ancestors. Code receives a greater tree-movement penalty
than prose. Protect coherent regions that fit, attach headings/signatures to
following content, and preserve mandatory causal boundaries. Hard capacities win
when no semantically complete unit fits; forced cuts are visible diagnostics.

Isolate native grammars in persistent subprocesses with an external watchdog.
Bound recursive byte, node, region, and injection work, with separate limits for
language nesting (12) and syntax-tree traversal depth (128). Preserve partial
evidence and source bytes after failure, then quarantine genuinely hard cases
instead of accepting fault-dependent fallback into the training corpus.

Capture quarantined source evidence, policies, and parser identity transactionally
with preparation progress. Agents claim leased repair tasks, amend shared parsing
code with regression coverage, and replay the unchanged captured row. Verification
requires a changed parser identity and the original codec, source/quality rules,
capacities, and planner policy. Success records a receipt; it never edits frozen
training shards. Repaired data enters a newly prepared artifact.

## Alternatives considered

- **Greedy size and punctuation rules:** cheap, but do not retain nested structure
  or look ahead to the next unit. Kept only as bounded fallback evidence.
- **Strict full-document AST parsing:** rejects too much naturally incomplete data.
  Recovering syntax trees and explicit diagnostics provide more useful evidence.
- **An unpinned, automatically downloaded grammar catalog:** broader advertised
  coverage does not justify hidden runtime provisioning or changing corpus behavior.
  New grammars can enter a tested, versioned environment or a local provider.
- **A learned semantic chunk classifier:** adds model assets and another source of
  versioned behavior before structural quality has been measured. It is unnecessary
  for the initial parser/planner contract and can be evaluated separately later.
- **Editing samples or waiving gates during parser repair:** breaks the connection
  between source evidence and training artifacts. Repair the parser and replay the
  original example under the captured policy instead.

## Consequences

Inspection can explain a selected cut using source spans, parser ancestry,
languages, diagnostics, and plan statistics. Shared contracts keep parsing,
tokenization, planning, preparation, and repair from growing separate segmentation
implementations. The bounded graph provides foresight while making its search
limits explicit.

Preparation now has CPU parsing work, worker startup, and a durable review queue.
Watchdogs are wall-clock bounds, so resource contention can affect which cases
need review; deterministic work caps and quarantine keep this visible. Broad
coverage remains incomplete, and syntax recovery and prose tailoring require
corpus-driven evaluation. No GPU throughput or model-quality result follows from
this preparation change.

The full configuration, implementation, grammars, chat definitions, Unicode/ICU
runtime, and reference codec enter artifact identity. Parser changes invalidate
identity reuse even before a package release. Existing prepared data is not
silently upgraded or mixed with new boundaries, and native generated emissions
and raw-facet tapes are never rechunked to impersonate a new generation.

Operational commands, exact limits, extension contracts, and the repair workflow
are documented in [recursive semantic chunking](../chunking.md).

## Primary references

- [Tree-sitter introduction and recovery goals](https://tree-sitter.github.io/tree-sitter/)
- [Bundled language pack 0.13.0](https://pypi.org/project/tree-sitter-language-pack/0.13.0/)
- [Unicode 16 text segmentation, UAX #29 revision 45](https://www.unicode.org/reports/tr29/tr29-45.html)
- [ICU boundary analysis](https://unicode-org.github.io/icu/userguide/boundaryanalysis/)
- [CommonMark specification](https://spec.commonmark.org/0.31.2/)
