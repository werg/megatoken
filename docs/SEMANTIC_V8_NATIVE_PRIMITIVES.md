# Native V8 primitive independence

Native V8 replay now obtains ECMAScript primitive and ordinary data-heap
operations from `data/semantic_primitives.py`. The module contains no trace
record reader, old semantic envelope, source interpreter, or call to the
producer. Primitive conversion, Number/BigInt operations, UTF-16 comparisons,
opaque references, dense-array bounds and graph alias rules have one independent
Python implementation. Both existing V7 and direct V8 clients consume it.

The V8 pure evaluator, detailed transitions, protocols, Number conversions,
Proxy checks and binary-memory conversions import that neutral implementation
directly. The native admission command also uses the neutral canonical-JSON
implementation. Removing a dependency does not broaden the supported language
domain or relax admission.

## Executable controls

- The existing V7/V8 algorithm and V8 primitive, detail, Number, memory and Proxy
  population passes **76 tests**, zero failures/skips.
- A fresh Python process installs an import guard before loading V8. It rejects
  imports of the V6 harness/contract and V7 algorithm, detail, context, harness
  and contract clients. With that guard active, it independently replays all
  **32 retained examples and 1,181 prediction contexts** from the base, protocol,
  callback, mixed-host and host-recovery fixtures. Each example also reconstructs
  a prediction prefix and restores its checked continuation; forged state is
  rejected. Physical framing is exercised once with the explicitly diagnostic
  byte codec. This control passes.
- The first absence-test attempt successfully reached the callback fixture but
  failed because the test reader assumed every fixture had a list wrapper. The
  reader now handles the existing single-artifact wrapper; the corrected test
  passes without changes to replay acceptance.

Reproduction from the repository root:

```sh
PYTHONPATH=.:src python -m pytest -q \
  tests/test_algorithm_v7.py tests/test_algorithm_v8.py \
  tests/test_native_primitives_v8.py tests/test_native_detail_regressions_v8.py \
  tests/test_number_v8.py tests/test_memory_v8.py tests/test_proxy_v8.py \
  tests/test_native_v8_without_legacy.py
```

These are native engineering and dependency controls. The import-guard test
does not provide private source certification, pinned-tokenizer admission,
learned execution quality, a complete installed-package cutover, or final
deployment evidence. Those remain separate gates in the main producer PR.
