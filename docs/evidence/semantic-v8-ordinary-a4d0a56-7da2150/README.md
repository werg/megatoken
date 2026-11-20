# Ordinary V8 training and serving qualification, 2026-11-20

This is a bounded CPU run of the ordinary data preparation, training, resume,
export and native serving paths with the actual pinned tokenizer. No source
verifier, tokenizer loader, decoder or model output was injected. The source is
the deliberately small `run(x) { return x + 1; }` case with input `3`.

The exact code pair is:

- Producer [`a4d0a56ef5b5b0ab1c1768ed16b15194630a0e89`](https://github.com/werg/tracing-js-self/commit/a4d0a56ef5b5b0ab1c1768ed16b15194630a0e89).
- Consumer [`7da2150adaf5501338d73f82a41a9058a5bd1937`](https://github.com/werg/megatoken/commit/7da2150adaf5501338d73f82a41a9058a5bd1937).

The collector is `scripts/qualify_semantic_v8_lifecycle.py` at commit `975f17c`.
Its exact bytes are bound in `summary.json`. The consumer used for preparation,
training and both serving processes was a separate clean checkout of the pinned
consumer revision. This receipt does not qualify a later contract or code pair.

## Observed results

| Check | Actual result and scope |
| --- | --- |
| Source and native preparation | One source family admitted after the normal independent producer recheck; zero refusals. |
| Actual tokenizer | Ling-3.0-tiny revision `9a98e35fe1c9ee255f78dd64771c7ae15a799481`; tokenizer file SHA-256 `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60`. |
| Prepared data | 25 physical records containing 2,649 codec positions, including 74 eligible target positions; maximum record size 126 positions under the 128-position policy. |
| Ordinary training | A normal diagnostic-purpose run performed one optimizer update. A separate normal trainer process resumed its checkpoint and performed a second update. The diagnostic purpose supplies a bounded run; it does not bypass V8 data or tokenizer checks. |
| Optimizer persistence | All 315 saved moment owners advanced from step 1 to step 2; 303 master-weight tensors changed during the resumed update. |
| Export and reload | The ordinary export command created a served-only bundle at model step 2. Fresh serving processes validated its checksums, real tokenizer and complete V8 training lineage. |
| Expected serving identity | Omitted, null and substituted training identities were refused. The correct independently supplied identity was acknowledged. |
| Actual model output | The first native prediction emitted one original token, producing three bytes. It exhausted the one-fragment budget without termination. It remains a rejected prediction, with original tokens and bytes retained. |
| Fresh-process inference recovery | One session and one consumed attempt restored. The next selection identity, original fragments, bytes, token count and model/tokenizer/training identities exactly matched an uninterrupted process. No model record had been acknowledged, so this exercises initial committed state and the consumed random stream; it does not establish accepted-record recovery. |
| Paired frozen producer experiment | One declared case, one attempted case, one rejected prediction, zero completed/source-validated executions, zero unknown token costs. The producer loaded the actual consumer service and checked the frozen model, tokenizer, training and generation settings. Both processes shut down successfully. |

`study-report.json` records zero optimizer work **during inference**. The two
earlier optimizer updates are in the checkpoint evidence. No receipt upgrades
source eligibility, data admission, loss reduction or token generation to learned
execution success. This training-family probe is not a held-out quality study.

## Durable identities

| Artifact | SHA-256 or canonical identity |
| --- | --- |
| Prepared manifest fingerprint | `56db77914a0a1032e314efc7dec3b58cad85c30f7e02316be42e96aa924561be` |
| Training lineage | `552f96aa9fb2949e5cd5a76741bc4d64a0541986110b6b94709174be3eb3a1b7` |
| Step-one checkpoint state | `ab75207a8e624eb02adce4f1c8bcc8304c96205930e51289458c3602694b387a` |
| Step-two checkpoint state | `3d1b0d1d46a9c2f4e33f7a72cdab9ba135744630c618f7dfe7267558d3b9f109` |
| Exported model weights | `fe4e067bf584aae28c13b3d49caaa1f16208ad5c410e7850d6133f813819236c` |
| Serving lineage | `a965617084f480b6abed78651d439d497394bd8929b8847902f50c6bd1cc163c` |
| Frozen producer study | `510028c914e064bf509d278f3bacf65e82dce52f7a3fda354d65667552119469` |

`summary.json` inventories every complete local checkpoint/export file with its
checksum and size. This committed evidence directory contains original JSON
receipts, request/response transcripts, source evidence and command logs. The
102.1 MiB of complete local training/export directories are not embedded in Git.
Publication of those complete binary bundles to a durable external location
remains a separate open deliverable; these hashes do not imply that publication.

## Reproduction

Use clean checkouts of the exact revisions above, the pinned tokenizer, and a
Python environment with this repository's dependencies and CPU Torch. Keep all
generated files outside both checkouts. In the examples below, `$producer`,
`$consumer` and `$assets` are absolute paths; `$python` names the Python executable.
The committed `source.json` is a source-certified input, not a supplied model
continuation. The ordinary preparation command independently rechecks it.

```sh
cd "$consumer"
PYTHONPATH="$consumer/src" "$python" scripts/prepare_semantic_v8.py \
  --input "$assets/source.json" --output "$assets/prepared" \
  --producer-root "$producer" \
  --producer-revision a4d0a56ef5b5b0ab1c1768ed16b15194630a0e89 \
  --consumer-revision 7da2150adaf5501338d73f82a41a9058a5bd1937 \
  --tokenizer-json "$assets/ling-tokenizer.json" \
  --max-positions 128 --max-history 128
```

Use `training-config.json`, updating only its tokenizer, prepared-data and output
paths for the new workspace. Retain the exact preparation policy and all model,
history and training limits. Run the ordinary commands in separate processes:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$consumer/src" \
  "$python" -m megatoken train --config "$assets/training-config.json"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$consumer/src" \
  "$python" -m megatoken train --config "$assets/training-config.json" \
  --resume "$assets/fit/checkpoints/SC-000000001-periodic"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$consumer/src" \
  "$python" -m megatoken export \
  --checkpoint "$assets/fit/checkpoints/SC-000000002-periodic" \
  --output "$assets/served"
```

Run the collector from its committed location, passing the clean pinned checkout
as `--consumer-root`. It loads both checkpoints, checks optimizer persistence and
export binding, then runs the real service in two fresh processes. Its immutable
output retains complete RPC messages, including failed model output:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python" \
  "$collector_checkout/scripts/qualify_semantic_v8_lifecycle.py" \
  --consumer-root "$consumer" --prepared "$assets/prepared" \
  --source "$assets/source.json" \
  --first-checkpoint "$assets/fit/checkpoints/SC-000000001-periodic" \
  --resumed-checkpoint "$assets/fit/checkpoints/SC-000000002-periodic" \
  --served "$assets/served" --output "$assets/lifecycle"
```

For the paired producer check, use the existing
`tools/run-semantic-v8-experiment.mjs freeze` and `run` commands. Freeze one case
pointing to `source.json`, seed `17`, settings
`{"device":"cpu","maxTokens":4,"maxFragments":1,"effort":1}`, one prediction,
four generated tokens, one outstanding prediction, zero stale retries and a
180,000 ms timeout. Supply the freshly exported manifest's `model_sha256`,
`codec_fingerprint` and `semantic_v8_lineage_digest` to `--checkpoint`,
`--tokenizer` and `--training-identity`. Run with the clean consumer checkout,
its bundle, the same Python executable, and `PYTHONPATH="$consumer/src"`.
`study.json` preserves the exact original frozen denominator and settings;
`study-report.json` preserves the actual failed outcome.

The first source-admission attempt in this environment correctly refused a
producer checkout with an untracked dependency-directory symlink. Making the
dependency directory a normally ignored directory restored a clean checkout;
the verifier was not disabled. Earlier preliminary receipts remain local and
are not substituted for this exact-pair qualification.
