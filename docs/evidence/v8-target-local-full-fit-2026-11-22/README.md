# Complete local target exposure over retained support inputs

The actual local diagnostic fit completed **348 optimizer updates** and its final
checkpoint passed a fresh, exact comparison with the complete prepared training
target inventory. Every one of the 48 training sources contributed its required
targets: **696 of 696 logical targets, 720 of 720 physical target records, and all
720 expected scored occurrences**. Every per-target window set and positive-weight
loss denominator matched. Missing, unexpected, occurrence, window and denominator
mismatch lists are all empty in the complete retained report.

The fit took 5,607.900659478999 seconds, used seed 17, CPU, fp32 and the tiny model,
and stopped normally at the configured step limit. Its consumed P counter was
52,054, below the phase budget of 200,000,000. The checkpoint's state SHA-256 is
`ceaa39848375f601d468f22669b50261a9bf23733706ac159dbcc86ac9d8c371`.

## Scope and exact provenance

This is a completed optimizer run over **retained inputs**, with independent
checkpoint evidence. It is not a fresh producer/native qualification or evidence
for the later expanded asynchronous contract.

The complete prepared 64-source corpus comes from producer
`0694fa9280712fa1364ead46a477c757a0ca4bfa` / native admission
`dbc89dcd52cb93d1d9fdca2989fab64b85e5d427`, originally retained in artifact
`14747074516` from
[run 45005718379](https://github.com/werg/tracing-js-self/actions/runs/45005718379).
Those immutable input-policy revisions remain in the checkpoint and raw report;
they must not be read as an attestation that the later local optimizer executed
the original native revision. The training split has 48 sources; the 12 validation
and four test sources are admitted but are not optimized by this fit.

The optimizer executed the source tree
`64e4539528d02b5b19ccbb7c556c977153a2930e`, byte-identical to the training source
at pushed native `a1d7f83dc46e4829ca4226e4040519cd5835f365` (local implementation
`f4b0b938a11149b13da776195dbd8b5155e9005a`). Its actual runtime was Python
3.12.14 and **torch 2.14.1+cpu**. The separate hosted campaign uses its own pinned
runtime and input revisions and retains separate evidence.

The checkpoint collector then used the stronger inventory procedure from native
`2479a9d59793569954a85d0f911085bfd4ee29f6`. That procedure independently derives
the complete policy-target identity set directly from prepared episodes, requires
it to equal the planner's target set, rederives every ordinary window, and compares
the final optimizer census against those exact IDs and counts. No additional
optimizer update is performed by the collector. Its training source tree remained
unchanged. The source files, hashes, exact commands and runtime are recorded in
`provenance.json`.

The original contract digest is
`0df031301d75122edad5f0efbbbe041ddbe90ebaac3cf2d6e6705540acd71a47`.
The full inventory digest is
`1448910c17fa4ca735f4e9919b8acd8b28abe4fce6f7d85d44bfde09a0bf6720`;
the final coverage report digest is
`4fc9653bc2ca992163ad6f1a69c4ceb4ff7c34337c1719aee83c0bab64ebba15`.
Original config and report bytes are preserved, including their absolute paths.

## What the optimizer consumed

| Source variant | Sources | Optimizer windows | Logical targets | Physical records and scored occurrences | Lexical denominator | Reconstruction denominator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Callback and recovery | 12 | 156 | 312 | 312 | 10,830 | 10,830 |
| Child teacher | 12 | 36 | 72 | 96 | 5,063 | 5,063 |
| Pure | 12 | 12 | 24 | 24 | 966 | 966 |
| Repeated calls | 12 | 144 | 288 | 288 | 9,168 | 9,168 |
| **Total** | **48** | **348** | **696** | **720** | **26,027** | **26,027** |

Other positive-weight denominators were structure 2,880, terminal 1,440 and
router_z 720. These are overlapping per-loss normalization counts, not a count of
distinct tokens. The available router_balance denominator was 720 but its effective
weight was zero, so it is not credited as positive-weight exposure. The full report
retains every source/logical/physical identity, window index, scored occurrence,
denominator, and first/last optimizer step; the table is only a readable projection.

## Retained evidence and remaining work

The ordinary gzip files retain the full coverage and inventory JSON, complete
optimizer metrics and output, full collector output, and parameter report. The
exact supplied and resolved training configurations, runtime, tensor accounting,
checkpoint metadata and latest pointer are retained as plain JSON. The provenance
manifest lists every original and compressed byte length and SHA-256. The expensive
binary checkpoint remains separately available at the recorded workspace path;
its state and transducer hashes are included without adding model weights to Git.

A separate supplemental audit also passed complete native admission of the
retained corpus and independently derived all 720 prepared policy-target identities.
Those IDs exactly equal both the original planned targets and the final optimizer
targets; all retained per-target window, occurrence and denominator rows are equal.
Its report digest is
`cd6aad04fa8e25640a03506162bb7b6a3f455048382e9fb395e93be9798509ba`.
It performed zero additional optimizer updates. The first audit attempt reached
all comparisons successfully but failed to seal its report because torch exposes
its version as a string subclass, which strict canonical JSON rejects. Explicitly
converting that runtime field to a plain string fixed the report; the complete
rerun passed. Both logs and the full final audit report are retained.

Token/mask-position coverage remains **unmeasured**, and learned behavioral success
remains **null**. Complete target contribution establishes that the requested
training signal reached the optimizer. It does not demonstrate that a small model
learned to perform those behaviors. Production launch, GPU fit, throughput,
production donor/encoding assets and a fresh full pass on the final expanded
producer/native contract remain separate gates.

The principal provenance lesson is that admitted-input policy and actual executor
identity must be recorded separately when diagnosing a retained corpus. Rewriting
the old policy or relabeling this local CPU result as a fresh current-pair result
would discard the distinction needed to interpret this evidence correctly.
