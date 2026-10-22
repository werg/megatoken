# Multimodal knowledge on the transducer: a proposal

**Status:** proposal, not built. **Date:** 22 October 2026. Builds on the
[transducer design](continuous-state-transducer-plan.md) (v10).

## What the memory can and cannot do today

The transducer bank is already modality-blind. Every event the model sees,
whether text, speech, an image, a video or a document read compressed, becomes
the same kind of 32 x 512 block, and an entry's value is written from what the
training sequence needed next, whatever its media. Knowledge *learned in
training* is therefore already multimodal.

What is missing:

1. **One bank only.** Each model owns exactly one bank, the one training grows.
   There is no way to keep several knowledge bases side by side and use them
   together.
2. **Nothing can be added without training.** Entries are created only from
   the hindsight head, which reads the future of a training sequence and so
   cannot run at serving time. A manual, a photo archive or a set of
   recordings cannot be added to what the model can look up.

Exact recall of a specific source (quoting a passage, returning a file,
citing) is out of scope: knowledge bases hold learned knowledge only, and
source tracing belongs to a separate retrieval system.

## 1. Many knowledge bases, searched as one

A **knowledge base** is one packed edge table in the shared location and query
spaces: the bank training grows is one; a corpus written by the ingest heads
(§2) is another; a user's own documents are a third. They share geometry, so
one search covers any set of them.

A session **mounts** a set of knowledge bases, each with a blend weight. A
lookup scans each mounted base's departures near the session's reached
locations, scores all their pairs together, and keeps the best 32 across all
of them in one softmax with the one sink. A base's blend weight enters as a
log-prior added to its pairs' scores ($z_{ji} + \log w_{\text{base}}$), so a
weight of 1 changes nothing, a small weight lets a base contribute only where
it fits much better, and 0 unmounts it. Each base keeps its own IDs, version
and owner; a lookup's selected pairs are (base, edge), and its committed
landing is shared, so a search can move from one base's region into another's.

Ownership and lifetime follow from the base, not from the search:

- the **trained base** keeps everything v10 already does: written and learned
  values, structure evolution, DiLoCo merges;
- an **ingested base** is fixed after it is written: no gradients, no
  splitting or removal, deleted as a whole. It can be built offline from a
  corpus or at serving time from a user's files, and shipped separately from
  the model.

Training mounts the trained base alone, plus, in the reading-from-a-base
streams (build step 3), ingested
bases built from the training corpus, so the model learns to use bases it did
not grow.

## How knowledge gets into a base

- **The trained base** fills itself during training, as today.
- **A corpus base** is built offline, before serving: an operator runs one
  command over a folder or dataset of files (text, images, audio, video),
  e.g. `megatoken kb build --name product-manuals --files ...`. Each file goes
  through the existing media preparation, then is written as entries (§2). The
  result is a self-contained base file that can be shipped and mounted next
  to any model with the same geometry.
- **A user's base** is built the same way while the model serves: the runtime
  accepts "add these files to knowledge base X", writes their entries, and from
  then on sessions that mount X can find them. Nothing is retrained.

## 2. Writing an entry by reading the item alone

An **item** is one physical record: a compressed text piece, an image, a
speech clip or a video clip, already turned into teacher features by the
existing preparation path. Writing an item into a base is legal at serving
time because it reads only the item, never a future:

- the **ingest pass** runs C over the item's own events with the full
  (bidirectional) mask, which is the first stage of the hindsight curriculum
  restricted to the item;
- an **ingest realizer** (the hindsight realizer's architecture, its own
  weights) turns the final states into a value in the bank's FP4 format;
- a **key realizer** turns them into a 1,024-wide key in the query space, so
  the queries P already asks can match it.

**Where an entry sits** uses the navigation the model already has. A session
starts its search at the location its metadata prefix maps to (request, task,
topic, language, modality, date). An item's metadata (its source, title,
topic, language, modality, date) goes through the same map, and the entry
departs from there. A session whose metadata resembles the item's starts near
it, with no global index. Its destination is its content location
($\operatorname{normalize}(\Pi v)$, as hindsight write-outs already use). The
parts of one document form a chain: part $i$ departs from part $i-1$'s
destination, so reading on is one more step.

**Training the ingest heads.** Training already contains items followed by
the use made of them. Whenever a lookup's hindsight group is or contains a
single item:

- the item's ingest value should match the hindsight value $h_t$
  (distillation: the hindsight head sees the future use, the ingest head only
  the item);
- the item's key should score highest for the query the model actually asked
  before that use, against the other items in the batch (contrastive).

## Cost

- One ingest pass per item: one C pass over the item's events, the same as
  observing it once. A 1,000-page manual is about 120 compressed pieces.
- One packed edge per item (about 20 KB at the current geometry): 100,000
  items take about 2 GB.
- A lookup over several bases scans each base's departures for a newly
  reached location; scoring is unchanged (the best 32 pairs overall).

## Build order

1. Several mounted bases in one search: per-base tables, blend weights,
   (base, edge) pairs, the coordinator and checkpoints owning only the trained
   base.
2. The ingest and key realizers, with their two losses inside the existing
   hindsight pass; metadata-placed departures and document chains.
3. Reading-from-a-base streams from the existing reading data (long
   documents, image description, transcription): items are written into a
   base instead of the context, and later turns ask about them.
4. Measurement: answer accuracy on held-out reading questions with the base
   mounted and unmounted.
