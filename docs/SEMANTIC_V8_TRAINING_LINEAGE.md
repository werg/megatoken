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
replaced bundle. Native selected-serving initialization and recovery now require
that expected identity. The producer freezes it before inference and checks the
service receipt, original selected records and restored state. A normal V8 bundle
cannot silently downgrade to the explicit null used by diagnostic bundles that
have no ordinary training lineage.

## Evidence and remaining work

Before an execution-environment outage, all four new identity/alteration/export
controls passed. The existing ordinary training/checkpoint/fresh-process resume
test plus the serving suite passed 27 tests. The new export metadata test explicitly
injects its tokenizer verifier; it is not a real pinned-tokenizer qualification.

The same source changes were recovered through the GitHub connector after the
local executor disconnected. Fresh hosted CI must still validate the published
revision. CI now prints the required formatting diff without weakening lint,
formatting or shell-validation gates.

The later [ordinary V8 qualification](evidence/semantic-v8-ordinary-a4d0a56-7da2150/README.md)
uses the actual pinned tokenizer and unchanged source/native admission. On the
recorded producer/consumer pair, one source family passed preparation; a normal
trainer took one optimizer step, a fresh process resumed to step two, and ordinary
export/reload preserved the complete lineage. All 315 saved optimizer moment
owners advanced and 303 master-weight tensors changed. The real native service
refused missing/substituted lineage and restored the exact next generated tokens
in a fresh process. The paired frozen producer study also exercised the actual
service and retained its failed output. This is a bounded lifecycle qualification,
not a learned-quality result or proof for a later code pair.

Serving also retains the prepared physical chunk budget instead of silently
regrouping observations at the model's larger position capacity. A strict
prepared-corpus test compares actual training and native serving raw/cache tensors,
history identities and request controls at the first prediction boundary, with
128-position records and a 512-position model.

A [later continuation comparison](evidence/semantic-v8-window-parity-2026-11-20/README.md)
also found and fixed lost request controls when ordinary training rebuilt earlier
generated records after an optimizer-window boundary. Actual model-state, mask,
decoder and control comparisons cover split records, rejected candidates,
acknowledged-state reconstruction, host results, retained callbacks, eviction and
multiple training windows. The lexical decoder is explicitly controlled for
these engineering checks, so they do not establish learned execution success.

The complete SC launch binding is documented in
[`configs/train/semantic-v8-production.example.yaml`](../configs/train/semantic-v8-production.example.yaml).
It keeps the production teacher targets, navigation teaching, held-out evaluation,
periodic checkpoints, both curriculum donors and distributed workers together.
Its paths and V8 pins remain explicit placeholders until the matching prepared
corpus, producer/consumer revisions, registry digest and assets are selected.

Remaining: final combined-pair qualification after contract changes; complete
binary-bundle publication; broader varied control/history populations;
accepted-record recovery with trained weights; learned-quality studies. No default
cutover is made here.
