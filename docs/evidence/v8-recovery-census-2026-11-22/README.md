# V8 recovery census and real-runtime peer checkpoint

This increment fixes module initialization counting and requires exact causal evidence before counting recovered source effects. A reconciliation must follow its unique source request and complete dispatch history beginning at attempt 1. Fulfilled/rejected reconciliation needs the immediate matching typed result; a not-dispatched reconciliation needs the next dispatch attempt and its typed result. A bare handler reattachment or a lost intent does not establish recovered work.

The focused census suite passes **6/6 tests**. Native Ruff check and format check pass across **589 Python files**. The module fixture is a model-free public-harness control and carries no independent source or reference qualification.

The recovery fixture retains only public rows and model contexts from the separately archived source probe. The archived original callback source artifact and complete probe log show its local independent source recheck, including checkpoint restoration, source eviction/rereading, and reconciliation. Its producer revision is null; this local source evidence is not a pinned production admission receipt. Nothing in the census test promotes its stripped public fixture to source authority.

The new real Python JSONL recovery peer uses the production V8 inference adapter and shared runtime. It replaces only the final lexical decoder with bounded deterministic fragments. Syntax, Ruff and formatting checks passed locally; Torch is unavailable here, so actual runtime execution remains pending in the paired hosted gate. Its paired JavaScript test is to be committed by the producer after pinning this increment. This is engineering recovery evidence and makes no learned-quality claim.

`manifest.json` binds the complete UTF-8 logs and original local artifact. Each `.gz.b64` file is base64-encoded deterministic gzip; decode it with base64 followed by gzip. Original trace bytes and full logs are retained, not excerpts.
