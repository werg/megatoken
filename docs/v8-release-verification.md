# Direct V8 release verification

`scripts/verify_semantic_v8_release.py` verifies that a self-contained text
release contains the source-bound prepared data, ordinary checkpoint and served
model that its descriptor declares. A passing report performs no optimizer
updates or decoder forward passes and makes no learned-quality or default-route
claim. The producer separately verifies the complete qualification journal,
including every declared case and committed attempt; this native verifier does
not replace that selection authority.

## Interface and first supported profile

Run from the exact clean native checkout with its `src` on `PYTHONPATH`:

```sh
python scripts/verify_semantic_v8_release.py \
  --bundle-root /absolute/bundle \
  --descriptor release-inputs.json \
  --producer-root /absolute/producer \
  --consumer-revision <exact-native-commit> \
  --output /absolute/new-native-report.json
```

The optional `--source-recheck-timeout-seconds` is an integer from 1 to 900
(default 30), applied to each source correspondence check. Fresh preparation also
has a finite aggregate subprocess bound. The output must be new and outside the
input bundle. No passing report is written on failure.

The descriptor has exactly these fields:

```json
{
  "schema": "semantic-v8-release-inputs/1",
  "producerRevision": "<40 lowercase hex characters>",
  "consumerRevision": "<40 lowercase hex characters>",
  "contractDigest": "<active V8 contract digest>",
  "fixtureDigest": "<active V8 fixture digest>",
  "interoperabilityDigest": "<active V8 interoperability digest>",
  "roles": {
    "qualification": "source-qualification",
    "tokenizer": "tokenizer/tokenizer.json",
    "prepared": [{
      "path": "prepared/000",
      "sourceArtifacts": [{
        "caseId": "one-declared-case",
        "path": "source-qualification/one-declared-case.original.json"
      }]
    }],
    "checkpoint": "training/checkpoint",
    "recipe": "training/recipe.json",
    "served": "serving"
  }
}
```

Role paths must be existing canonical relative descendants of the bundle, with
no aliases, overlapping roles, symlinks or symlink ancestors. Source artifacts
must be inside the qualification role and have unique case, path and underlying
source membership. Every bundle member, including hidden journal files, is
hashed before and after verification; the verifier never creates a missing
journal lock or repairs evidence.

The checkpoint role must contain the concrete committed `state.pt`,
`metadata.json` and checked transducer attachment. Latest-pointer roots and
pending `.superseded-*` repair entries are refused before the ordinary loader
can run. This guard is necessary because ordinary resume resolution may repair
an interrupted checkpoint replacement; inspecting a release must never trigger
that mutation. Checkpoint metadata and served manifest/configuration/lineage
JSON are parsed strictly, rejecting duplicate keys and nonfinite values.

This first profile requires exactly one prepared root, a named native model
preset with inline overrides, a text-only corpus, and one ordinary training
rank. It refuses external packs, teacher features, text anchors, hindsight
targets, donor imports, distributed workers and stage/effort mixtures. The
existing independent position comparator requires a fresh-stage history;
archived stage transitions are explicitly outside this profile. None of those
unsupported inputs is silently omitted.

The raw recipe must use exactly the tokenizer role and the one prepared role as
its codec and dataset locators. Only those two locators are resolved for reading.
All other checkpoint training settings must match the recipe, apart from the
output directory. Original recipe and checkpoint bytes remain unchanged. The
current content-bound objective and checked input fingerprints are used; older
path-bound position histories are not adopted under the new identity.

## What is independently checked

The complete native V8 preflight checks every episode and the supplied registry,
policy, source/native receipts, tokenizer and manifest/index/shard seals. The
verifier then invokes the existing preparation command in a temporary directory,
using the exact clean producer and native revisions. Each supplied source gets
a fresh producer replay and family binding check plus independent native
validation, tokenization and packing. Any refused or omitted source fails the
release check.

The descriptor names the original source documents consumed by preparation;
post-admission copies and native receipts remain separate qualification evidence.
Their public records may agree while the complete document bytes, and therefore
the source-admission receipt, differ. Before regeneration, the verifier requires
exact unique source membership, equal lengths and matching splits against the
actual prepared index. It then passes those original source paths to preparation
in that index's order. Journal order is not assumed to be training order. No
stored row, index or checkpoint cursor is sorted or rewritten, and the full
reconstructed comparison below still applies.

The regenerated corpus must equal the bundled corpus in every reconstructed
episode field, byte payload, token label, loss mask, physical/logical identity,
context, terminal framing, split, index row and scope position. JSON formatting
and SQLite placement are not training semantics; complete reconstructed content
is compared, and both corpora still validate their own file hashes. No original
input or receipt is rewritten to obtain equivalence.

The actual checkpoint loader verifies the committed checkpoint and attachment
hashes. Its payload, metadata, recipe, model geometry, codec, full V8 training
lineage, chunking, corpus/stream identity and ordinary cursor must agree. A new
independent prepared-position inventory checks every recorded target against
the checkpoint's sealed actual decoder receipts and target census. A partially
trained model or repeated training pass is allowed, with the missing/repeated
counts retained; these states are never reported as a complete one-pass
qualification merely because their observed rows are consistent.

The verifier loads the real checkpoint model and the real `ServedBundle` on CPU.
It compares every exported tensor name, shape, dtype and value, the complete
served model configuration, tokenizer bytes, version, lineage and transducer
bank. The transducer loader intentionally increments a cache-invalidation
version. The raw exported bank must equal the checkpoint loader's expected
snapshot exactly; the loaded served bank must then equal that snapshot with
exactly the prescribed additional increment. The field is checked rather than
discarded. A changed model or bank with recomputed self-consistent hashes still
fails comparison with the actual checkpoint.

Configuration and metadata comparisons preserve native canonical JSON scalar
types. A resealed integer `0` cannot substitute for boolean `false`, nor can a
boolean model version substitute for an integer update count. These checks are
separate from self-consistent hashes and from the loader's structural checks.

## Evidence boundary

A passing native report is a source/data/checkpoint/export consistency result.
It does not prove accepted live history, restored service behavior, exhaustive
internal routing positions, production compute, learned correctness, default
cutover or all production routes. The report retains those boundaries explicitly.

The focused tests include strict paths and metadata, omitted or duplicated
membership, unsupported assets, complete regenerated data comparison, resealed
label/mask changes, tensor/key/type mismatches, and one actual controlled
training update followed by real export/load. That controlled execution injects
source admission and a byte-codec fixture and is labeled accordingly. A complete
production-tokenizer source-to-release qualification requires a separately
frozen composed producer/native pair and newly generated assets; historical
training runs remain evidence for their original pins only.
