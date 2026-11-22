# Source-bound module ledger with current recovery and training

The source-bound module ledger is composed with the existing native commit
recovery, async source semantics, immutable admission cache and target census.
The exact candidate identities and runtime are in `qualification.json`.

The first composition exposed a real integration defect: the new module
machine imported its stable hash through `algorithm_v7`. The fresh-process
training/resume control deliberately blocks legacy wire imports and refused
that dependency after an actual optimizer update. That initial run had
221 passes, one failure, three tokenizer-dependent skips and 119 passing
subtests. Its complete log and XML are retained.

The hash now imports from the neutral `semantic_json` module. With the pinned
tokenizer explicitly supplied, all **225 native tests plus 119 subtests pass,
zero failures/errors/skips**, at `5d013677`. The final `09f94f7` candidate only
sorts that import block; its 16 affected module/training controls pass again,
and Ruff is clean. These suites include actual bounded optimizer, checkpoint
and fresh-process-resume controls; they are overlapping regression evidence,
not a new complete-population training run.

Producer `ae870a1` pins that final native candidate. All **208 leaf producer
continuation controls pass, zero failures/errors/skips**, including six
original module cases, six actual native comparisons and 64 owned lifecycle
transitions. The corresponding producer evidence directory retains its raw
XML and source receipt. The canonical contract check and byte comparison
pass with digest `a9523855…`; the six earlier interoperability payloads and
fixture digest are unchanged. The full digests are in `qualification.json`.

The separate frozen ledger gate remains identified by `61c9cac/4fd7b56` and
its original hosted run. The separate module source/data bridge is not part
of this composition. No production default changes, larger admitted training
population, full optimizer fit or learned-quality qualification are claimed.
The reverse CI pin names the exact composed producer source above; subsequent
evidence and consumer-pin-only changes do not change those executable bytes.
