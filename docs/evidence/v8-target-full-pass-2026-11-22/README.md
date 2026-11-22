# Hosted complete optimizer-target qualification

The actual complete support campaign passed at producer
`7c468f8f3d06e32e43e3ebb482137840bdce676d` / native
`a1d7f83dc46e4829ca4226e4040519cd5835f365` in
[run 45009024141](https://github.com/werg/tracing-js-self/actions/runs/45009024141).
All 64 sources passed original-source/native admission, pinned tokenizer/packing
measurement and strict preparation into the frozen 48/12/4 split. The ordinary
two-update resume/export/service route passed, followed by a separate fresh
**348-update complete target fit**.

The final coverage checkpoint contains all **48 training sources, 696 logical
targets, 720 physical target records and 720 scored occurrences**. A fresh planner
rederivation matched every source/logical/physical identity, per-target window set,
occurrence count and positive-weight loss denominator. All missing/unexpected
target and window/occurrence/denominator mismatch lists are empty. Positive-weight
denominators were lexical 26,027, reconstruction 26,027, structure 2,880, terminal
1,440 and router_z 720. These overlapping per-loss counts are not a distinct-token
total. The full raw coverage and inventory, including every ID and row, are
retained here as ordinary gzip-compressed JSON.

The actual full fit used seed 17, CPU fp32, the tiny model, Python 3.12.14 and
torch **2.10.0+cpu**. Training took 3,997.842019281 seconds and stopped normally at
its 348-step limit. The two-update lifecycle test and the fresh 348-update fit
are independent optimizer trajectories; their reported total of 350 is not one
cumulative training history. The two-update route correctly reports partial
exposure: four logical targets and six physical records from one training source.

## Evidence identity

- Original contract:
  `0df031301d75122edad5f0efbbbe041ddbe90ebaac3cf2d6e6705540acd71a47`.
- Inventory digest:
  `08dc4b97ec973651b447de7f1393e63a8a410f660f3942bc6be007a8dde38385`.
- Coverage report digest:
  `3a422a003b5c5c9c9b54a74b8de9e2fa58b8174207b6c6a4df8d4b937cb87959`.
- Final checkpoint state SHA-256:
  `8b5a4924b064af097c5cae3cac3fa67ce1d767443d58d11201b09a6ddab189d7`.
- Original complete artifact: `14751643088`, 175,192,312 bytes, ZIP SHA-256
  `2b0d30937a05b992500a2567c0d3d98cf8a72aa96a72b03c22e1ccbab046671e`.

The original inventory, coverage, training configuration, runtime and checkpoint
metadata are copied without changing their uncompressed bytes or absolute paths.
The complete 451-file raw report archive, job log and per-file provenance for all
593 original artifact paths are retained in
[producer evidence at af3df64](https://github.com/werg/tracing-js-self/tree/af3df6454352cf127fc4cabd6e1b93aff4b0582f/docs/evidence/v8-target-full-pass-2026-11-22).
The intact original artifact retains every complete input, tokenizer, binary
checkpoint and exported model as well. `evidence-files.json` binds these copies
to their original artifact paths and hashes.

A separate supplemental audit uses complete native admission of the retained
prepared corpus, independently derives all 720 prepared training policy targets,
and requires exact equality with both planned and actual target sets and every
retained window/occurrence/denominator row. It passed with torch 2.10.0+cpu and
performed zero additional optimizer updates. The audit source helper is bound to
native `2479a9d59793569954a85d0f911085bfd4ee29f6`; it does not change the original
optimizer pin. The full audit report/log are retained here, with digest
`ad9f948a730705156f1cbc3d70b78f45a086ec1b7009f8c4c5d716e3771071b1`.
Its separately hashed script and original-config/path-relocation checks are
documented in the producer evidence.

## Limits

This full pass qualifies the exact original pair and contract above. Later
expanded contracts, executable revisions and the explicitly selected full-pass
driver need their own complete campaign. The distinct local retained-input run
used torch 2.14.1 and older input-policy pins; its evidence is separate.

Token/mask-position coverage remains unmeasured and learned behavioral success
remains null. The service continuation used unacknowledged selections and recorded
zero acknowledged semantic selections, so this run does not establish accepted-
record recovery. Production launch, production assets, GPU fit and throughput
remain outside this bounded diagnostic qualification.
