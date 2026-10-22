# Multimodal knowledge: knowledge bases, retrieval and adapters

**Status:** agreed direction, being built. **Date:** 22 October 2026. Builds on
the [transducer design](continuous-state-transducer-plan.md) (v10).

Three parts work from one lookup: **knowledge bases** (the transducer memory,
several mounted at once), **retrieval** of original items of any medium, and
an **adapter mixture** chosen by what the knowledge bases returned.

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

3. **No retrieval of originals.** Knowledge bases hold learned knowledge only.
   Quoting a passage, showing a picture again or citing a source needs the
   original item, and the earlier literal retrieval (text only) was deleted
   with the v9 stack.
4. **No adapters.** The earlier stack also chose a mixture of small low-rank
   adapters from what memory returned and applied it across the network; it
   was deleted too.

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

## 3. Retrieval of original items, any medium

A **source collection** holds original items (text passages, images, speech
and other audio clips, video clips) with one search key each. Several
collections can be mounted, like knowledge bases.

- **One key space for every medium.** The key is the teacher's own 768-wide
  pooled embedding of the item. EmbeddingGemma 2 embeds text, images, audio
  and video into the same space, and every prepared item already stores this
  embedding, so a photo, a recording and a paragraph about the same thing lie
  near each other.
- **Index.** The quantized inverted-file index of the v9 stack comes back
  unchanged: int8 copies of the keys grouped into cells, with exact rescoring
  of the probed candidates, on disk with only the cell centres in memory.
- **Query.** At each lookup boundary the model already samples one query block
  on its query track. The same block also goes through the embedding head's
  query role, giving a 768-wide query. No teacher runs at serving time.
- **Use.** The best items above a score threshold that the session has not
  already seen are observed, after the knowledge event, as ordinary events in
  their own medium on the knowledge track: text as bytes or compressed pieces,
  media as their stored teacher features. The model then reads, quotes,
  describes or regenerates them like anything it observes, and the runtime can
  return the original file. At most a few items per lookup (start with 2).
- **Adding items.** The same command and runtime call that add files to a
  knowledge base (§"How knowledge gets into a base") also add them to a source
  collection: each file is prepared once, giving both its knowledge-base
  entries and its retrievable original.
- **Training the query side.** (i) Query alignment: the embedding head's query
  output matches the teacher's embedding of the question, centered as in the
  current embedding-head training. (ii) Ranking: when the data says which item
  answers (the reading streams know their supporting documents, images and
  clips), the mass of the retrieved scores should fall on those items, with no
  penalty on unannotated items. (iii) The continuation loss itself, since
  retrieved items enter the sequence. A **reading-from-a-collection** stream
  puts the supporting items in a collection instead of the context, so the
  only way to answer is to retrieve them.

## 4. Adapter mixture chosen by the knowledge bases

A shared bank of small low-rank adapters (start with 8 experts of rank 8 per
owner) applied **throughout**: every C layer, every P stage and the realizer
conditions each get a low-rank update of their output computed from their
input.

- **Routing in the knowledge-base space.** After each lookup, a router reads the
  lookup's blended knowledge value (the same wide value the knowledge event is
  read from, across every mounted base), together with track and stage
  embeddings, and picks the top 2 experts per owner with softmax weights.
- **Strength follows retrieval.** The mixture is scaled by how much the lookup
  actually found (one minus the sink's share), so a lookup that found nothing
  changes nothing.
- **Persistence.** A lookup's mixture applies to every meridian on every track
  until the next lookup replaces it.
- **Matched to retrieval.** Because the router reads the same blend that the
  knowledge event carries, the adapters always correspond to what was
  retrieved, from whichever knowledge bases were mounted. A user's ingested
  base therefore steers the adapters without any adapter of its own.
- **Training.** Jointly with everything else through the continuation loss,
  with 10% adapter dropout so the model stays usable without them; the
  router's expert load is balanced by the existing global router-balance
  objective. The adapters are ordinary dense parameters and sync through
  DiLoCo like the rest of the model.

## Cost

- One ingest pass per item: one C pass over the item's events, the same as
  observing it once. A 1,000-page manual is about 120 compressed pieces.
- One packed edge per item (about 20 KB at the current geometry): 100,000
  items take about 2 GB.
- A lookup over several bases scans each base's departures for a newly
  reached location; scoring is unchanged (the best 32 pairs overall).

## Build order

Retrieval (§3) and the adapter mixture (§4) are independent of each other and
of multiple bases, so they are built first and in parallel.

1. Multimodal source collections: catalog, index, the query path from the
   embedding head, observing retrieved items, the query-side losses, and the
   collection CLI.
2. The adapter bank, its knowledge-base router and its application in C, P
   and the realizer conditions.
3. Several mounted bases in one search: per-base tables, blend weights,
   (base, edge) pairs, the coordinator and checkpoints owning only the trained
   base.
4. The ingest and key realizers, with their two losses inside the existing
   hindsight pass; metadata-placed departures and document chains.
5. Reading-from-a-base and reading-from-a-collection streams from the existing reading data (long
   documents, image description, transcription): items are written into a
   base instead of the context, and later turns ask about them.
6. Measurement: answer accuracy on held-out reading questions with bases and
   collections mounted and unmounted, retrieval recall against the teacher,
   and loss with adapters on and off.
