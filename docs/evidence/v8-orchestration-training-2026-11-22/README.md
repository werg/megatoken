# Native source-qualified orchestration training increment

This increment is based on native `b865b5f` and producer `073870f`. It adds
independent transcript validation, two-track physical materialization, ordinary
source/native admission and policy rebuild, and a census that requires actual
ordered lifecycle records. The fixture and producer APIs are source-qualified;
local admission unit tests explicitly inject the source/tokenizer audit boundary.
Their byte-fallback codec does not qualify production token measurements.

## Executed local checks

The final integrated selection passes **44 tests, six deselected**, in 31.67
seconds. The six deselections are the existing live-source CLI checks, which
require exact clean source checkouts; active integration prevented that local
gate. They remain mandatory, with zero skips, in the hosted paired workflow.

```sh
PYTHONPATH=/workspace/scratch/517281e17c56/python-deps:src \
TMPDIR=/workspace/scratch/517281e17c56/tmp python -m pytest -q \
  tests/test_semantic_orchestration_training_v8.py \
  tests/test_semantic_orchestration_corpus_v8.py \
  tests/test_semantic_v8_behavior_census.py \
  tests/test_semantic_v8_orchestration_authority.py \
  tests/test_orchestration_admit_v8.py tests/test_admit_semantic_v8.py \
  -k 'not live' \
  --basetemp=/workspace/scratch/517281e17c56/tmp/native-orchestration-final
```

Both `ruff check src tests scripts` and `ruff format --check src tests scripts`
pass across 588 files. The separate 32-test census/parity result and 12-test
CLI/collector result are retained, but overlap the final 44 and must not be added
to it. Full stdout/stderr and hashes are retained, including the short complete
pytest summaries.

The larger predeclared ordinary source, `promise-number-proxy-memory-join`, was
freshly checked against original source and augmented with the teacher. It has
58 public rows, 56 original semantic targets and four lifecycle targets.
Independent native replay and 128-position/128-history byte-fallback framing
produce **3,177 physical / 130 logical records**, 6,721 generated and 377,292
observed byte-fallback tokens. It has coordinator/child owners `track:0` and
`track:1`, with exactly create, join, progress and complete actions. The full
transcript digest is
`7a246a95f14ae0917e906b82aecca1b56359fcbc405d800d970f9d0eccc20398`.
The exact source artifact and complete result are retained. The native receipt
correctly remains `admitted:false`: production source-receipt and tokenizer
gates were not invoked in this structural/framing check. An initial command used
an incorrect import name; that setup-error log is retained separately and is
not a native validation failure or a passing control.

## Review findings fixed here

- Initial coordinator input had included future child history; it now contains
  only the two-row source/input prefix.
- Child bootstrap must repeat the exact source/input on the child's own track
  before observing the bootstrap acknowledgement on the coordinator.
- Teacher receipts must not invent model selection IDs. A shared neutral
  observation projection now matches training and actual serving.
- Completion delivery must precede its typed child receipt.
- Coverage must validate unmasked receipt provenance and controls as well as
  targets, all public and lifecycle dependency edges, track specifications,
  and complete physical causal order. Grouping bytes without checking their
  order was insufficient.
- The digest binds owner/control/action data; augmented and child-only episodes
  have different identities within their shared source-family split.
- Logical teacher work is charged explicitly and is distinct from actual model
  fragment cost. More than 64 child targets is refused.

## Mandatory remaining evidence

The new actual shared-runtime parity test is syntax/lint checked locally only:
Torch is unavailable. It uses supplied engineering decisions and child bytes,
and is not learned success. The updated reverse paired workflow pins producer
`073870f` and requires the entire V8 selection with zero failures or skips.

The new ordinary orchestration command must freshly prepare, train, resume,
export and restore the real service. It verifies the four saved lifecycle
targets but leaves optimizer exposure per target and learned success unmeasured.
Current-code general CPU CI and exact-pair production assets remain separate
gates. Historical hosted logs are stored in adjacent evidence directories and
do not qualify this new implementation.

Every `.gz.b64` file is base64-encoded gzip. `SHA256.json` records exact decoded
and encoded hashes and byte lengths. No summary replaces the retained input or
full original output.
