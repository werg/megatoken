# Native V8 child orchestration engineering evidence

The implementation and limits are described in `../../semantic-v8-orchestration.md`.
This evidence is a scoped engineering checkpoint, not source admission, learned
success, production curriculum qualification, or default cutover.

- `local-authority.log`: 24 passing local controls in 44.33 seconds: nineteen new
  neutral binding/registry/ingress controls, existing independent V8 prediction
  boundary controls, and the native V8 lifecycle authority test. The new
  fresh-process control blocks eleven legacy modules while checking real V8
  inputs and exact terminal values. No ML runtime is used in this selection.
- `local-shared-model-unavailable.log`: the mandatory actual-runtime test module
  fails collection locally because torch is unavailable. No skip or success is
  substituted. Its eight tests are selected by the existing hosted `test_*v8*.py`
  gate and remain required before crediting shared-model reconstruction.
- `completion-retry-before.json` / `completion-retry-after.json`: independent
  reproduction of exact child receipt retries. The original implementation
  refused four historical cases; the repaired implementation accepts all five
  positive cases and does not mutate authority.
- `completion-retry-negative.json`: changed original public/context/result and
  unknown invocation are refused; an exact closed-session read preserves the
  saved receipt and does not reopen a native session.
- The two Python probes reproduce those model-free receipt controls. They use
  the actual completion method, independent V8 replay, and typed registry over
  an explicitly assembled already-delivered state. They do not execute a model
  or dispatch external work. Only repository-relative paths were changed when
  retaining the original scratch probes, and the rerun JSON matches the saved
  independent after/negative outputs exactly.
- `evidence.json`: scoped results, source content hashes, artifact hashes,
  current consumer base, mandatory hosted test command, and remaining limits.

From the consumer repository root, run the retained probes with:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python docs/evidence/v8-orchestration/completion-retry-probe.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python docs/evidence/v8-orchestration/completion-retry-negative-probe.py
```

The actual shared-model suite is deliberately required rather than skipped:

```sh
python -m pytest -q tests/test_semantic_v8_orchestration.py
```

It exercises reserved and waiting checkpoints, exact acknowledgements and lost
ack reads, native child track routing, exact reference-valued completion, terminal
cancellation, rehashed checkpoint refusals, no legacy clients, and two sessions
interleaved around a rejected candidate and acknowledgement. Successful lexical
and lifecycle selections are supplied engineering controls; the separate actual
random-model failure control retains original invalid output without fallback.
