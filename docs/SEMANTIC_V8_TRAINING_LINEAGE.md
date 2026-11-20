# V8 training identity through export and reload

Ordinary V8 training now retains its exact preparation policy and dataset identities
inside each training checkpoint. Exported serving bundles carry that identity
alongside the weights, configuration, sparse memory bank, tokenizer and original
training checkpoint. Loading refuses missing or inconsistent evidence.

This is artifact identity validation. It does not claim learned execution quality.

## Retained evidence

The training record binds the producer and consumer revisions, complete family
registry, current V8 contract/fixture/interoperability digests, exact prepared
manifest fingerprints, combined prepared-corpus identity, training-stream identity,
chunking policy and tokenizer identity. It is created only after the ordinary
trainer's existing preparation checks. Resume checks the record against the saved
training policy, tokenizer, chunking and checkpoint metadata.

The serving record additionally binds the training checkpoint checksum, exact
served weights, model configuration, sparse bank and model step. Export and reload
use the pinned V8 tokenizer loader, including when assets are relocated. The
generic byte fallback cannot substitute for a V8 checkpoint's tokenizer.

These checks bind an artifact to the evidence it retains. The application still
needs an independently expected bundle/lineage identity to reject a completely
replaced bundle. Exposing this identity in native selected-serving initialization
and recovery is the next paired increment; it is not implemented by this change.

## Evidence and remaining work

Before an execution-environment outage, all four new identity/alteration/export
controls passed. The existing ordinary training/checkpoint/fresh-process resume
test plus the serving suite passed 27 tests. The new export metadata test explicitly
injects its tokenizer verifier; it is not a real pinned-tokenizer qualification.

The same source changes were recovered through the GitHub connector after the
local executor disconnected. Fresh hosted CI must still validate the published
revision. CI now prints the required formatting diff without weakening lint,
formatting or shell-validation gates.

Remaining: native initialization binding; actual pinned-asset preparation and
normal optimizer/export/reload qualification; training/serving continuation parity;
mixed callback native admission; learned-quality studies. No default cutover is
made here.
