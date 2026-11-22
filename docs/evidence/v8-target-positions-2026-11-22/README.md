# Executed position audit: bounded evidence

This evidence qualifies the new prepared-text position audit and its integration
with ordinary optimization, checkpoint resume and explicit stage transitions.
It does **not** qualify a new complete 64-source support run or learned semantic
execution. The previous completed 348-update runs have no position receipts;
their token/mask-position coverage remains unmeasured.

## Exact implementation

| Repository | Published executable commit | Identical local commit | Complete tree |
| --- | --- | --- | --- |
| Native | `16b6a989c79d5e4a1910a03ccee73e8817b5a95b` | `436b129d46cc3c4c4bf97bab3779e93803472656` | `598b34ea08a4c940c802ad15e285496a6c689edb` |
| Producer | `183966c94413e2f31cda12c487671020eae70a15` | `e95e90c1d79eaa8869c07c8c78400d61fc74a946` | `ca7e1b0961c18c90d26331c1440017db3f02f7b5` |

The isolated changes began at native `0066888f733c0d27d7ac7c0e92dc390231575f2e`
and producer `e8cd1475dc89648bc00e2d69afc55042bc6eece1`. Later main-branch
contract/ledger changes require their own composition and execution checks.

The native actual runs use the repository's prepared V8 fixture, with explicitly
injected source-verification and tokenizer-admission boundaries. The byte codec
is the controlled fixture vocabulary. Ordinary preparation validation, model
execution, gradients, optimizer updates, checkpoint serialization and fresh
process resume execute normally. These fixture boundaries are recorded in the
raw reports and must not be described as a new pinned-tokenizer/source campaign.

A subsequent validation-only change is published at
`8eae121e86cd0cb2735a5bc4f1e3e82026d738f9` (local `655ea580`, complete tree
`dc1a4e07b555a7889d8e153959a8fb650dcde361`). It requires a complete canonical
objective descriptor even inside an archived stage, where the active run's
settings are no longer the descriptor for that prior objective. Its negative
control rejects a resealed archive with a missing objective. The unchanged raw
actual receipts above were rechecked under this stronger validator, with two
fresh prepared-record reconstructions, four actual position states and the exact
stage/resume links verified. This recheck executes zero optimizer updates; it
does not change the actual execution pins. See `stricter-position-validation.json`
and its retained recheck script.

## Actual optimization and exact positions

The complete controlled fixture contains 64 physical input records, nine windows,
two logical policy targets and four physical policy targets. Three windows carry
scored work. The three-update CPU float32 pass took 16.112 seconds on PyTorch
2.10.0+cpu with seed 17. Every physical target appeared exactly once.

| Positive-weight objective | Independently reconstructed support | Actual loss selections |
| --- | ---: | ---: |
| Prior lexical positions | 413 | 413 |
| Reconstruction positions | 413 | 413 |
| Length and format categorical decisions across both passes | 16 | 16 |
| Terminal categorical decisions across both passes | 8 | 8 |
| Modality categorical decisions | 4 | 4 |

The comparison found zero missing or invented targets, prepared-oracle
mismatches, duplicate occurrences, or target-census window mismatches. Its full
prepared inventory digest is
`a06bbac585490a9d98a0bb5c91202564ef8ad56ad845f57272b9592863ce565b`;
the actual committed position-state digest is
`d1bf358dce849aab1bd6db710bdf51b3b26bba4444717d3873e6503e2318a87a`.
The four router-z objective events remain visible; their internal routed
positions receive no token-position credit.

The same test then explicitly changes training from SC to SE and executes one
more optimizer update. The new position census starts at global step 3, ends at
step 4, and credits only its own one update: one logical target, two physical
targets, 243 lexical positions, 243 reconstruction positions, eight structure
decisions, four terminal decisions and two modality decisions. It retains the
complete three-update earlier state as a sealed archive. Previous-objective
positions do not inflate the new stage's counts.

## Actual fresh-process resume remains partial

A separate ordinary CPU bfloat16 fixture run performs one update, saves the
checkpoint, then starts a new Python process and resumes for one further update.
The process blocks legacy-wire imports. The independent lifecycle collector
checks both checkpoints and their exact parent digest.

The first checkpoint has two physical targets and 243 lexical/reconstruction
positions. The resumed checkpoint has three physical targets and 365 positions
for each decoder pass. The independent prepared population still contains four
physical targets. The report names the missing fourth target and keeps
`fullPopulationQualified: false`; a successful resume cannot qualify complete
position coverage. Both this actual resume test and the complete-pass/transition
test passed together in 39.80 seconds.

## Independent reconstruction over the retained larger inputs

The earlier frozen oracle implementation, published as
`6b8fd1e1963ce873a27d5936dcd9717f0e3028b0` with local equivalent `0141466`,
also read the complete retained original prepared corpus without an optimizer.
Its input producer pin is `0694fa9280712fa1364ead46a477c757a0ca4bfa`; its input
native preparation pin is `dbc89dcd52cb93d1d9fdca2989fab64b85e5d427`.

It independently reconstructed 48 training sources, 696 logical targets and 720
physical targets in 2.287 seconds. The expected positive-weight counts are 26,027
lexical positions, 26,027 reconstruction positions, 2,880 structure decisions,
1,440 terminal decisions and 720 modality decisions. All 720 physical targets
fit the supported prepared-text oracle. This was **zero optimizer updates** and
does not reconstruct missing historical execution receipts. The complete oracle
is retained in `retained-old-input-position-oracle.json.gz`; the decompressed
bytes and digest are recorded in `summary.json`.

## Negative controls and remaining scope

Fifty-four position/lifecycle controls passed, including same-count mask shifts,
omitted/invented/duplicated positions, wrong labels, changed vocabulary masks,
categorical support changes, boolean-versus-integer metadata, resealed checkpoint
forgeries, missing decoder passes, observation-only windows, logical-versus-
physical partition counts, unsupported structured branches, exact resume,
archived stage corruption and changed objectives. The complete optimizer test
was selected separately for the actual run above, rather than skipped.

Ninety affected ordinary-path regressions also passed on the immediately earlier
frozen tree: grouped training, generation policy, target inventory and target
exposure. The decoder/engine instrumentation is identical between those trees.
Ruff passed. Sixteen producer controls passed, and the producer's actual full-pass
receipt validator also accepted the retained native four-target report as a
controlled input. The real 64-source driver was not launched.

The supported claim stops at prepared plain-text lexical/reconstruction positions
and public categorical labels/supports. It does not independently verify source
preparation itself. Existing source/tokenizer/contract admission gates continue
to own that boundary. Hidden routed rows, latent coordinates, media masks,
structured-action field/copy/placement masks and transducer/retrieval loss
positions remain unmeasured by this audit. Learned success is null throughout.

`controlled-runtime-reports.tar.gz` retains all 24 selected report/config/runtime/
metric/checkpoint-metadata files, including the earlier intermediate actual run.
Every archived member, both compressed artifacts and every listed report were
checked against their retained byte lengths and SHA-256 values. Checkpoint tensor
files are reproducible outputs of these bounded tests and are not Git artifacts.
