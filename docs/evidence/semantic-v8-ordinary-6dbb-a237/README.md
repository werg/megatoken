# Ordinary V8 training, resume, export and serving at 6dbb8c0 / a237366

This retained run exercises the normal preparation, training and export commands
with actual pinned tokenizer bytes and real CPU model weights. It contains the
complete source, prepared corpus, both optimizer checkpoints, exported model,
tokenizer and original serving responses. It is a bounded engineering lifecycle
result, not a learned JavaScript success or production-readiness result.

| Identity | Recorded value |
| --- | --- |
| Producer | `werg/tracing-js-self@6dbb8c0baff3fd2490a7d78f31acd6e5fda63938` |
| Consumer | `werg/megatoken@a2373660a4653ca5ab485c82bf453f1755e0dc1a` |
| Source | `promise-number-proxy-memory-join` |
| Frozen inventory | All 31 interoperability sources, seed 17; the selected source belongs to train |
| Registry digest | `5c0370fe39b73dae0e76bad1fc5165cbc6ae8773090209624077177afe305b1f` |
| Tokenizer SHA-256 | `40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60` |
| Codec fingerprint | `7f52faca2fe82273a889023defd2f1a6fd6080a21096ec77e97f4798a0a370b2` |
| Model / execution | `tiny`, 4,039,640 served parameters, full 157,184-token vocabulary; CPU fp32 |
| Runtime | Node 24.22.3, Python 3.12.14, Torch 2.10.0+cpu; locked native dependency set |

## What actually passed

The source combines a Promise executor, Number conversion, a Proxy getter and
typed-memory access, followed by Promise join/await. Original source comparison
passed. The V8 source representation is detailed fallback, with 56 detailed
records and 23 protocol records; no compact-region success is claimed for it.
The ordinary preparation command independently rechecked source, family/split
binding, native replay, causal contexts and actual tokenizer/masks. It admitted
one complete source and retained zero preparation refusals.

The normal trainer ran once, saved its checkpoint, resumed it, and performed a
second update. The collector verified the complete checkpoint data:

| Measurement | Result |
| --- | ---: |
| First global optimizer step | 1 |
| Resumed global optimizer step | 2 |
| Retained optimizer state owners | 315 |
| Owners whose step advanced from one to two | 311 |
| Master-weight tensors that changed after resume | 300 |
| Acknowledged model-generated semantic records | 0 |
| Fresh-process next-selection equivalence | Exact across all seven checked fields |
| Host redispatches during restoration | 0 |

The ordinary export loaded the second checkpoint and preserved its V8 corpus,
stream and serving identities. The real JSONL service rejected missing, null and
substituted training identities. It produced an original sampled token from the
exported model; that token did not terminate a valid semantic record within the
declared fragment budget. Its original bytes and error remain in the archive.
Fresh-process restoration reproduced the next selection exactly without
acknowledging or repairing the incomplete output. Consequently, the serving
result covers restoration of initial/unacknowledged selection state and random
state. Accepted-record recovery and learned execution quality remain unqualified.

## Separate population failure at this pair

The independent full-population admission command at these same revisions
reported **31 source-qualified / 30 native-admitted / 31 declared**. Its CLI
rejected the new source profile `composed-protocol-source/1` before the live
source check. The ordinary preparation path above independently rechecked and
admitted this source, so its real lifecycle measurements are retained, but they
do not turn that full-population failure into a pass. Consumer `6997cae` adds the
missing profile to the existing strict extension-inventory checks; its fresh
qualification is a separate result and must not reuse these receipts as its own.

## Complete retained archive

`inventory.json` lists SHA-256 and byte length for every original file, all ten
archive parts, and the reassembled archive. The archive contains **43 files /
118,396,151 uncompressed bytes**, including the prepared JSONL and SQLite stores,
both complete `state.pt` / `transducer.pt` / metadata checkpoint bundles, the
complete served model and tokenizer, training logs, frozen source population,
registry, source evidence and serving request/response/checkpoint records.
Every archive member was decompressed and checked against its original file hash.

The reassembled gzip archive is 83,459,192 bytes with SHA-256
`1d665856c04441bb7370378270187c432b13dcea9ee672f9e38de5048e4b9a2c`.
Reassemble and verify it from this directory:

```sh
python - <<'PY'
import hashlib, json
from pathlib import Path
inventory = json.loads(Path('inventory.json').read_text())
archive = inventory['archive']
digest = hashlib.sha256()
with Path(archive['file']).open('wb') as output:
    for part in archive['parts']:
        data = Path(part['file']).read_bytes()
        assert len(data) == part['bytes']
        assert hashlib.sha256(data).hexdigest() == part['sha256']
        digest.update(data)
        output.write(data)
assert digest.hexdigest() == archive['sha256']
assert Path(archive['file']).stat().st_size == archive['bytes']
PY
mkdir restored
tar -xzf complete-run.tar.gz -C restored
```

The original configuration and evidence keep their actual absolute paths and
recorded revisions. They are not rewritten during archiving. `prepare-source.mjs`
inside the archive reproduces the complete source-family freeze and selected
source binding with a clean producer checkout and a new output directory. A
reproduction in another directory must use its own new paths and receipts rather
than relabeling the retained manifests or training identities.

The normal command sequence was:

1. Run `prepare-source.mjs` against the clean producer, before any optimizer work.
2. Run `scripts/prepare_semantic_v8.py` with exact producer/consumer revisions,
   the actual tokenizer, 128 positions and 128 retained history chunks.
3. Run `python -m megatoken train --config training-config.json`.
4. Resume `SC-000000001-periodic` with the same ordinary command/configuration.
5. Run `python -m megatoken export` on `SC-000000002-periodic`.
6. Run `scripts/qualify_semantic_v8_lifecycle.py` on the actual prepared source,
   both checkpoints and served bundle. The retained `summary.json` records the
   model hashes, changed optimizer/weight state and original service responses.

The source belongs to the training split. It is not a held-out evaluation, the
two updates are not a model study, and the failed generated record is not a
successful execution.
