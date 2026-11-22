# V8 training input relocation

## Original failure

The added actual-optimizer test on native base `622c37c23f93c4cc58ff226eb4805c766fee3166` reproduces a storage-location failure: one initial update and a second update from the same checkpoint at the original input paths succeed. A fresh process using byte-identical copied prepared data and tokenizer fixture refuses before its optimizer update because the codec path changed. The original data/tokenizer paths are made unavailable before that fresh process starts. No explicit stage transition is used.

The original test result is **one failure, zero errors/skips, 14.95 seconds**, retained in `initial-red.log` and `initial-red.xml`. `initial-red-source.json` binds the exact added test bytes and unchanged production base. Two optimizer updates executed in this failed experiment; no relocated update completed.

Runtime: Python 3.12.14, Torch 2.10.0+cpu, pytest 9.2.0. Pytest is outside the repository's declared `>=8.3,<9` range. The data preparation and tokenizer admission boundaries are explicitly injected fixtures; this is not a production-tokenizer, independent-source, hosted, complete-release or learned-quality qualification. The fix and its additional controls are subsequent increments.
