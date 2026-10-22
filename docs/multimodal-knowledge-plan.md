# Multimodal knowledge: knowledge bases, retrieval and adapters

**Status:** agreed direction. All of §1-§4 are built: several knowledge bases searched
as one (§1), writing an entry by reading an item alone (§2), source retrieval (§3) and the
adapter mixture (§4), with the `megatoken kb` and `megatoken collection` commands, serving
sessions that mount both kinds, the reading-from-a-base and reading-from-a-collection
streams and their measurements. **Date:** 22 October 2026. Builds on the
[transducer design](continuous-state-transducer-plan.md) (v10).

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
4. **No adapters** (now built, §4). The earlier stack also chose a mixture of
   small low-rank adapters from what memory returned and applied it across the
   network; it was deleted, and §4 brings it back on the transducer lookups.

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
streams (build step 5), ingested
bases built from the training corpus, so the model learns to use bases it did
not grow.

**Built.** What was decided on the way:

- A lookup's selected pairs are named `(location, base, edge)`; the same edge number
  can occur in two bases. Its blended knowledge value is still one tensor over everything
  mounted (the sink included), and its landing is shared, so a search can leave one
  base's region for another's (tested).
- Each mounted base has its own version. A session caches each location's neighbourhood
  per base and rescans only the base that changed: adding files to a user's base does not
  make the trained base rescan.
- Scoring, the exact scan and the way train and serve look things up are unchanged; a
  session that mounts only the trained base behaves exactly as before. With one ingested
  base mounted in its place, the search returns the same pairs and blend (tested).
- Random exploration reaches only the trained base: ingested bases are fixed.
- The trained base alone is checkpointed, synced by DiLoCo and evolved. An ingested base
  is a packed table with no gradient masters, optimizer or growth; entries can only be
  appended (and only to a base opened writable, which is how a user's base is filled
  while serving), never changed or removed.
- A training window that mounts another base writes nothing into the trained base (no
  hindsight write, no write-out), so what the mounted base holds is not memorized by it.
- A base file holds the packed table, where each entry came from, and the identity of
  what wrote it: the model's exact weights (a digest of every served weight), its
  architecture, the teacher and the geometry fields that fix stored values and
  locations. It mounts only on a model with the same identity, and says which
  part differs otherwise. Entries are written in one model's internal language, so a base
  built for last month's checkpoint is rebuilt, not carried over.

## How knowledge gets into a base

- **The trained base** fills itself during training, as today.
- **A corpus base** is built offline, before serving: an operator runs one
  command over prepared roots or plain files (text, images, audio, video),
  e.g. `megatoken kb build --name product-manuals --items manuals/*.txt photos/*.jpg
  --model bundle/ --packs lock.json`. Each file goes through the existing media preparation,
  then is written as entries (§2). The result is a self-contained base file
  (`megatoken kb inspect FILE` shows what it holds and which model wrote it) that is shipped
  separately from the model bundle and mounted next to the model that wrote it.
- **A user's base** is built the same way while the model serves: the runtime
  accepts "add these files to knowledge base X", writes their entries, and from
  then on sessions that mount X can find them. Nothing is retrained. In the
  multitrack JSONL service (`serve-multitrack --kb-root DIR`, every path inside `DIR`):
  `kb_create`, `kb_save`, `kb_open`, `kb_list`, `kb_close`, and `open` takes `bases`
  (name to blend weight) and `trained_weight`. Files are added by `add_files`
  (`paths`, with a `base`, a source `collection`, or both; see §3).

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

**Built.** What was decided on the way:

- The ingest and key heads are part of the served model (they run at serving time);
  only the hindsight head stays training-only.
- Placement. An item departs from where its metadata starts a session: the metadata
  goes through the same event, prefix head and starting-location map a session's prefix
  does (`metadata_start`). Files are filed under their modality and file name as topic
  unless the operator gives a task, topic, language or date. Prepared roots are filed under
  the metadata the training run itself gives that episode, so a session on the same task
  starts where its items are. Part $i$ of a document departs from where part $i-1$ leads;
  all parts of a document are written in one call. The destination is the content location
  of the written value.
- Bias. An entry's bias is zero, as a write-out into an empty bank has: it is scored by key
  and location alone, and its base's blend weight is the only prior on it. (A write-out
  takes the best score among its lookup's pairs, which an item has no lookup to give.)
- The two training losses run inside the existing hindsight pass, whenever a lookup's
  hindsight group is or contains exactly one item (an image, clip or document piece; a
  group with two items, with none, or past the end of the sequence matches nothing, so the
  sink-only lookups are excluded). *Value:* the item read alone should equal the
  hindsight value $h_t$, as a squared error relative to $\|h_t\|^2$, with the
  hindsight value as a fixed target. *Key:* among the window's items, the lookup's own
  query should pick out its item (cross-entropy over cosines at temperature 0.05, query as
  a fixed target). They enter the loss at weights 0.1 and 0.05, each a mean over the
  window's matched lookups. Negatives are only the other items in the same window, so a
  window with a single item trains the value alone; a larger pool of negatives is
  future work.
- The ingest value goes through the hindsight head's own packing, so a stored value has
  exactly the format of a written-out one.
- What the heads learn from. Streams that put items in the context (long documents,
  image description, transcription, ...) give the distillation and contrast signal.
  The reading-from-a-base stream trains the other side: how a query finds and reads an
  ingested base. Nothing flows back into the heads from there (ingested bases carry no
  gradient).

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
- **Adding items.** One runtime call (`add_files`, in the JSONL service the `add_files`
  operation) adds files to a knowledge base (§"How knowledge gets into a base"), to a
  source collection, or to both: each file is read by the teacher once, giving both its
  knowledge-base entries and its retrievable original. A serving session mounts bases
  and collections together; one lookup searches both, the base's blend feeds the adapter
  router and the collection's items are observed after the knowledge event.
- **Training the query side.** (i) Query alignment: the embedding head's query
  output matches the teacher's embedding of the question, centered as in the
  current embedding-head training. (ii) Ranking: when the data says which item
  answers (the reading streams know their supporting documents, images and
  clips), the mass of the retrieved scores should fall on those items, with no
  penalty on unannotated items. (iii) The continuation loss itself, since
  retrieved items enter the sequence. A **reading-from-a-collection** stream
  puts the supporting items in a collection instead of the context, so the
  only way to answer is to retrieve them.

**Status: built** (collections, query path, losses, producer, CLI, recall metric; the
contract is in [media.md](media.md#retrieving-the-original-items), the training in
[training.md](training.md)). Decisions the text above left open:

- **Keys are the raw teacher embeddings**, not centered ones. The embedding head's output
  is a raw unit vector (only its loss is centered), so student queries and stored keys
  are compared directly by cosine, one index serves every medium, and the stored
  embedding of an item is used as is. The centered distance is a training device; the
  collection stores the mean teacher embedding of the queries it was built for, and the
  query loss is centered on it.
- **The first lookup needs the question.** Lookups follow metadata, source results and
  assistant turns, not a user message, so the question travels as the metadata's
  `request` field and the first lookup, right after the metadata, searches with it.
  The reading episodes always show `request`; training keeps its usual random dropout of
  the other fields.
- **Ranking** is a softmax over cosine / 0.05 of the retrieved candidates (16 deep), the
  annotated items (scored even if the index missed them) and a null entry at the score
  threshold; unannotated candidates only share the normalization.
- **Threshold** 0.5 on the raw cosine and `max_sources` 2 are starting values; the
  threshold is stored per collection and overridable per session.
- **Reading-from-a-collection data**: one collection per split inside the prepared root;
  the held-out episodes search only their own split's items.

Left for later: a mounted collection for streams other than the annotated one (so a model
learns to use a collection that no annotation points at); a lookup after a completed
user message (today only the metadata, source results, assistant turns and the interval
trigger lookups); threshold and `max_sources` tuning on measured recall.

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

**Built** (`model/adapters.py`; see [architecture](architecture.md#the-knowledge-routed-adapter-mixture)).
The choices the plan left open:

- **Projection.** The router is one linear map from the layer-normalized
  32,768-wide value straight to the 8 experts (262 thousand weights per owner),
  not a 16-million-weight projection to facet width first. The map is the
  learned projection, into the only space routing needs, and the session
  carries just the resulting 3 x 8 preferences.
- **Sites.** Every C layer, every P stage and the selected raw block that all
  realizers read (the one input of text, audio, image and video R) each add
  the experts' low-rank update of their own input to their output. The experts
  are one bank per owner, shared by that owner's sites; a stage preference
  tells the sites apart. The hindsight pass has no mixture.
- **Strength.** One minus the sink's share of the value the knowledge event
  reads. A training hindsight overlay counts as knowledge, so the sink's share
  is its weight times (1 - overlay); at serving this is exactly one minus the
  sink weight, and a sink-only lookup has strength exactly zero.
- **Dropout.** A lookup's whole mixture is dropped with probability 0.1 in
  training, drawn from the window's seeded generator.
- **Balance.** Each owner's controller joins P's and U's routers in the global
  router balance (hard top-2 counts per site and track, soft preferences
  replayed from the detached lookup value).

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

1. Multimodal source collections (done): catalog, index, the query path from the
   embedding head, observing retrieved items, the query-side losses, and the
   collection CLI.
2. The adapter bank, its knowledge-base router and its application in C, P
   and the realizer conditions. **Built.**
3. **Built.** Several mounted bases in one search: per-base tables, blend weights,
   (base, edge) pairs, the coordinator and checkpoints owning only the trained
   base.
4. **Built.** The ingest and key realizers, with their two losses inside the existing
   hindsight pass; metadata-placed departures and document chains; the
   `megatoken kb` commands and mounting and adding files while serving.
5. Reading-from-a-base (**built**) and reading-from-a-collection streams from the
   existing reading data (long documents, image description, transcription): items are
   written into a base instead of the context, and later turns ask about them.
   The base stream (`base_reading`) is described below.
6. Measurement: answer accuracy on held-out reading questions with bases and
   collections mounted and unmounted, retrieval recall against the teacher,
   and loss with adapters on and off. For bases, **built**: held-out loss of the
   reading-from-a-base windows with the base mounted and unmounted, reported side by
   side.

### The reading-from-a-base stream (built)

`base_reading` episodes (`scripts/prepare_base_reading.py`, from HotpotQA / MuSiQue
paragraphs, DOCCI photos and LibriSpeech clips) have three turns: the user's question,
a short assistant turn ("Let me check the knowledge base.") and the scored answer. The
assistant turn closes a turn, so a lookup falls between the question and the answer. The
reference items are not in the episode's records: they are declared in the episode
(`knowledge_items`) and the trainer reads each one alone with the current ingest heads,
writes them into a small base of their own, and mounts it beside the trained base. So
the only way to answer is to find them. Photos and clips are asked about by quoting
the first sentence or first words of their description or transcript; video clips are not
drawn yet (the episode builder takes any prepared item).

The schedule is simple. An episode's items are ingested the first time it is drawn; the
bases of the last 8 episodes stay mounted, so the right items must be found among
others; every 1,024 optimizer steps all of them are dropped and rebuilt on demand, so no
entry is older than that by more than the heads have moved. They belong to the training
process and are neither checkpointed nor synced. Stage shares (all taken from the stage's
natural-text or replay stream so mixtures still sum to one): P1 1%, M1 0.5%, M2 0.5%, LC 1%
(the reading-from-a-collection stream takes a further 1% in every stage, from the same kinds
of streams).

Measurement: held-out evaluation scores each such window twice, with the window's own
items mounted (alone, no other episodes' items) and with nothing mounted, and reports
`reading_base`: both losses and their difference (unmounted minus mounted, positive when
the base helps). Those two entries are kept out of the pooled loss, which counts every
window once.

### What is left of the base half

- A larger pool of negatives for the key loss, and ingest-heads training from the
  reading-from-a-base stream's own windows (needs a differentiable path through a
  base, which the plan deliberately does not have).
- Video clips in the reading-from-a-base producer, and a question that scores answer
  accuracy rather than loss.
- An ingested base is rewritten if the model's weights change; a way to carry a base
  across a small fine-tune (re-reading each item from its stored features without the
  teacher) is not built.
- Every item is re-read from its stored teacher features; a base built from files
  needs the teacher once, at preparation.
