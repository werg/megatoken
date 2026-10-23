# Plan: writing to the memory bank while serving

**Status:** implemented; retrospective writes default on for scoped sessions. Trained-model calibration remains open.
**Date:** 23 October 2026.
**Decision wanted by the owners:** the model learns from conversations while it serves. This
document says how to get a first working version with the least change to the existing code,
and what to test before trusting it.

## 1. What this adds, in plain words

Today the memory bank is fixed once a model is exported. A person can add files to a separate
"knowledge base" with `add_files`, but the model never decides on its own to remember something
from a conversation.

This plan lets a session write new memories into a small, private, per-user store while it
serves, in two ways:

1. **Looking back (retrospective writes).** When a lookup was made early in a conversation, what
   the conversation did *next* tells us what knowledge would have been useful at that point. In
   training the bank is filled this way by a training-only component (the "hindsight" head).
   In a live conversation the future arrives turn by turn, so a lookup made several turns ago
   *can* be judged once enough later turns exist. We run the hindsight head over the completed
   part of the conversation and write the value it produces.
2. **Surprise (surprise-gated writes).** When the model reads a chunk (a user message, a tool
   result) that it did not expect, that chunk is probably new information. We store the chunk
   as a memory when its surprise is high and it is not already in the store.

Neither writes into the trained bank that ships inside the model. Both write into an **overlay**:
a separate, size-limited store tied to the user (or scope) that is mounted next to the trained
bank. The existing "knowledge base" mechanism already works this way, so the overlay is a
knowledge base with extra rules.

## 2. Original baseline (before this implementation)

- A knowledge base is a separate table of edges with the same layout as the trained bank
  (`memory/knowledge.py`, `memory/transducer.py`). Sessions mount the trained bank under the
  name "trained" and any number of knowledge bases beside it.
- Writing an item runs `add_items` (`kb.py`): the item's event block is encoded, a value head and
  a key head each read the item, and the edge is packed with seeded stochastic rounding.
  The edge's destination is `normalize(Π·value)`.
- A knowledge base opened `writable` has a lock; `add_files` needs a preparer. Saving is atomic
  (temporary file, fsync, rename) but only happens on an explicit `kb_save`; there is no autosave.
  A crash or `kb_close` loses unsaved writes.
- A knowledge base is bound to the exact model weights by an identity hash. After a weight
  update it is refused.
- Nothing in the code can remove or overwrite an edge in a knowledge base: there is no `retire`,
  and freed rows are not zeroed.
- The served model does not contain the hindsight head. It is excluded in three places
  (`generation/policy.py` prefix filter, `training_heads=False`, and `served.py`), and the
  hindsight pass refuses to run outside training mode.
- Surprise cannot be measured for observed chunks today: the only scorer, `score_group`, scores
  *generated* tapes.
- A tape does not record lookups, so replaying a generated turn never touches the bank. Writes do
  not need to change the tape.
- A lookup computed before a bank changes cannot be committed afterwards (a version check).
  Writes must therefore happen after a lookup commits, never between a lookup and its commit.

## 3. Design

### 3.1 The overlay store

- One overlay per scope (a user, a workspace). It is a knowledge base created with its own,
  smaller `max_edges` and byte budget. Only four geometry fields enter the identity check, so
  smaller limits are safe.
- New operations on the knowledge base class:
  - `retire(ids)`: zero the value columns, drop the source labels, bump the version.
    Deleting a memory on request is `retire` followed by an atomic save.
  - `blend(ids, values, mass, clock)`: the weighted-mean rewrite from the plan
    (`transducer_optimizer.py:123-147`): `A' = ρA + W` and `v' = (ρAv + H)/A'` with decay
    half-life `WRITE_HALF_LIFE` (16,384 on a serving clock, see section 5).
    Keep FP32 working copies for at most a few thousand overlay edges, because repacking the
    same row many times without a master adds rounding noise.
  - Provenance per edge: scope, session, turn, which rule wrote it, write time. This needs a
    format-version bump.
- `kb_save` must take the base lock (it does not today), and the service saves on session close
  and every N writes (autosave).

### 3.2 Recording what a session did

Add a bounded recorder to the session. For each ordinary event it keeps the raw event block, role,
track, modality, position and group number (the hindsight pass needs only these). For each
committed lookup it keeps: the start location, the start key, the bank's value, the best pair
logit, and which overlay edges it used with their weights. `Lookup` already carries `edge_ids` and
`edge_weights`; generalise them to one set per base.

### 3.3 Retrospective writes

1. **Export the hindsight head.** Remove the prefix filter, allow `training_heads=False` to keep
   this head, and relax the training-mode guard. Leave it out of the weights digest so existing
   knowledge bases remain mountable (open question 4). Update
   `test_the_hindsight_head_is_training_only` and the parameter report.
2. **Add `NativeTransducer.retrospective_values`.** After group t+1 plus a lag of L more groups
   (or when the session closes or flushes), run the group-inverse pass over groups t+1 and
   later and read h_t. Skip lookups whose read would fall past the end of the conversation
   (the "sink-only" case in the plan): those describe no real future.
3. **Write rule** (shaped like `candidate()` in `transducer_growth.py:252`): write only when
   the residual is large, `‖h − v̄‖² > 0.25‖h‖²`. The new edge has departure = the lookup's start
   location, key = its start key, value = h, destination = `content_location(h)`,
   bias = `best − (β_loc + β_query)`. If an overlay edge already has departure and key cosine
   ≥ 0.9, blend into it with `blend` instead of adding an edge.
4. Serving has no loss gradient, so only the residual test is available (training's other
   test, "did hindsight help", needs a gradient).

### 3.4 Surprise-gated writes

- `SurpriseMeter` computes the per-byte loss for an observed text chunk: before observing it, run
  `predict_group` for the chunk's track; then score the bytes with `decoder.teacher_raw` through
  `_GroupedTeacherReadout.losses_for_selected`. This costs about one extra prediction pass and
  one decoder pass per chunk and works for text only.
- Gate on both: loss above an exponential running threshold, **and** the chunk's ingest key is
  novel relative to existing keys (far from all keys in the trained bank and overlay).
  Two free signals to combine with it: the sink's share of the last lookup
  (`Lookup.sink_weight`, high means nothing nearby fitted), and key novelty.
- Then call `ingest_rows` on that chunk's event block with the existing ingest heads.
- Plain byte-text chunks were not part of ingest-head training (only media and compressed-text
  items), so this must be calibrated first (probe in section 6).

### 3.5 Where it runs, and safety

- Writes happen at the end of `_commit` and `generated_completion`, **after** the lookup commits,
  inside the session lock, then the base lock, in that order (as `_run` does). A rejected
  speculative branch writes nothing, because speculation touches no memory and acceptance is
  what triggers the lookup and the writes.
- Journal every write's edge ids; if a later step in the same operation fails, `retire` them.
- Budgets: a per-session write cap, a persistent per-user edge cap, and decay. When full, evict the
  lowest `mass × use count`.
- New service operations: `forget` (retire by id, by session, or by scope), `autosave` on close,
  and `promote` (opt-in): copy selected overlay rows with `table.state(ids)` and `add_edges` into
  a shared, read-only-mounted base with the same identity. Promoting into the trained bank is out
  of scope.
- Poisoning and privacy: writes come from conversation text, so a user can plant memories.
  Overlays are scoped per user for that reason; `promote` must be an explicit, reviewed step;
  `forget` must really remove the values (zeroed, not just hidden).

## 4. Known problems this must not walk into

These are issues the code review found that matter here.

- Adding files holds the writable base's lock for a whole model forward pass, so a session can
  stall past its deadline (`transducer_service.py:164`). Surprise writes must be queued and
  run between lookups, not inside a lookup. Prefer a copy-on-write snapshot over sharing one
  writable overlay across concurrent sessions, which would serialise their lookups.
- `kb.add_items` is not atomic across groups (a failure leaves earlier groups added, ids lost).
  Fix this before relying on it.
- A bank near its ceiling raises `BankFull`, and growth silently stops. The overlay needs its own
  explicit full-handling (evict, then write).
- A weight update orphans every overlay (identity mismatch). Rebuilding needs the source chunks,
  which means keeping conversation text; that has privacy consequences (open question 5).

## 5. Tests

- `tests/test_knowledge_serving.py`: writes after commit; stale-lookup handling; delete and save;
  identity after export of the hindsight head.
- `tests/test_transducer_memory.py`: `blend` and `retire` against the weighted-mean formula;
  a failed write leaves the base unchanged.
- `tests/test_transducer_serving.py`: the hindsight head is present in a served bundle; a
  write-after-lookup ordering test.
- `tests/test_mounted_together.py`: an overlay mounted with a collection and the trained bank.

## 6. Cheap checks before building (hours, on the local GPU)

1. **Surprise separates new from routine.** Run `SurpriseMeter` over a conversation with
   planted new facts and many routine messages. Healthy: the planted facts rank in the top
   10% of surprise on at least 80% of trials. (Guess; adjust once seen.)
2. **A written memory is retrievable.** Write one memory with each rule, then ask a question that
   needs it after the original text has left the retained history. Healthy: the edge is among the
   selected pairs and the answer improves versus no write. This needs a trained tiny model.
   Build these conversations through the maintained data preparation and shared trainer.
   The older teacher-free memory experiment scripts have been deleted.
3. **Retrospective lag.** With the hindsight head, vary the lag L (0, 1, 2, 4 groups) and measure
   how often the written value equals what a full-window pass would have written (cosine).
   Pick the smallest L that stays above 0.9. (Open question 1.)
4. **Forget really forgets.** After `forget`, the edge never appears in selected pairs and the
   saved file contains no trace of its value.

## 7. Open questions

1. Training's hindsight pass sees the rest of a window; serving sees only the groups so far.
   How large should the lag L be, and is a flush at close acceptable?
2. Do the ingest heads work on plain text chunks? They were trained on media and compressed text.
3. Does per-byte loss separate novel from routine chunks? No calibration exists.
4. Should the hindsight head be left out of the weights digest so existing bases survive?
5. How are overlays rebuilt after a weight update without keeping raw conversation text?
6. Shared overlay across concurrent sessions, or copy-on-write per session?
7. What unit should the decay clock use: lookups, turns or wall time?
8. There is no rewind operation, so a client undoing an accepted turn cannot undo its writes.


## 8. Implemented interface and choices

Retrospective conversation writing defaults on for sessions assigned a user or workspace
scope. Start the JSONL server with `--kb-root DIRECTORY`, then open a session with
`{"op":"open","scope":"alice"}`. The server creates or reopens that scope's private
store automatically, mounts it alongside the trained bank, and reports its name as
`overlay` and its enabled state as `memory_writes` in the response. No separate store
creation or writing-enablement request is required.

Python callers use `MultiTrackRuntime(multi, memory_root=directory)` and
`runtime.open(scope="alice")`. An existing writable private store for the scope is reused,
including a custom one created with `memory.create_overlay(...)`. Otherwise automatic
stores live at `DIRECTORY/scopes/SHA256(scope).kb`; scope text never becomes a path.
New default stores obey both the model's limits and the overlay limits. Reopening a scope
after a restart retains its saved memories and refuses an incompatible model identity.

Sessions without a scope do not automatically retain conversation memories. Set
`memory_writes=False` on `open` to disable automatic memory for a session. A scoped
session with default writing enabled requires a persistent memory directory or an
already-open writable private store; missing storage is reported as an error. For
explicit read access without writing, mount a private store using `bases` with its
matching scope and `memory_writes=False`, or name it as `overlay` with writing disabled.

Advanced callers may still create a custom store with
`memory.create_overlay(name, scope, path, max_edges=2048)` and open with
`runtime.open(overlay=name, scope=scope, write_policy=WritePolicy(...))`.
`WritePolicy` is imported from `megatoken.serving_writes`. A private store can only be
mounted in a session carrying the same scope. The calling application must authenticate
that scope; the JSONL server is a trusted local interface, not an authentication service.

Defaults for scoped sessions:

- Retrospective writes enabled, surprise writes disabled until calibrated. Enable the
  latter explicitly with `WritePolicy(surprise=True)`.
- Retrospective delay: one additional retrieval group after the future group to be read.
  `lag` is configurable; close/flush uses all available future evidence but skips
  lookups with no real future group. Evidence that leaves the bounded recorder is dropped.
- At most 128 recorded ordinary events and 64 write operations per session. Each blend
  counts as a write operation. The persistent default capacity is 2,048 edges and their
  packed byte cost; both limits are configurable independently.
- Save every eight writes and at session close/explicit flush. Unflushed writes can be
  lost on a process crash. Every save takes the store lock and atomically replaces disk data.
- The decay clock counts accepted ordinary groups across sessions sharing the store;
  evidence mass halves after 16,384 such groups between updates. Packed values retain
  FP32 working copies across saves. Blends refresh the destination from the blended value.
- Repeated evidence matches departure and key cosine at least 0.9. New surprising text
  must exceed 1.2 times its exponential loss baseline (update weight 0.05) and have key
  cosine below 0.9 against every searched trained/private base. The first observed chunk
  initializes the baseline and does not trigger a surprise write.

Stores are shared within an authenticated scope. Session locks precede store locks, and
multiple stores are locked in stable order. Expensive ingestion and retrospective model
passes are staged before publication; normal lookups are not interleaved with writes.
Publication snapshots the bounded store so failures restore existing blends and evictions
as well as removing new rows. Shared-store publication still briefly serializes sessions;
copy-on-write session merging is not implemented.

`memory.flush(session)` runs eligible delayed writes and saves; `memory.close_session(session)`
also clears retained evidence. `runtime.close()` flushes its open sessions after rendering
finishes. The `serve` command also closes the runtime when input ends or serving exits
with an error, so open sessions are flushed before process cleanup.
`memory.forget(name, ids=..., session=..., scope=...)` filters by any supplied
criteria, erases all row columns and associated provenance, and saves atomically. It also
clears retained evidence in affected open sessions so a subsequent flush cannot recreate
those memories. Sessions that contributed to a blended memory remain recorded for deletion.

`memory.promote(source, destination, ids, reviewed=True)` copies selected packed rows to
a separate shared base for the same model, assigning fresh IDs. The destination must be
writable for this operation; consumers may mount its saved copy read-only. Promotion
requires explicit review and cannot target the trained bank.

JSONL operations: `kb_overlay` (name, scope, path, optional max_edges/byte_budget), `open`
(optional scope, overlay, memory_writes boolean, write_policy object), `flush` (session), `kb_forget` (name and
ids/session/scope filters), and `kb_promote` (name, destination, ids, reviewed=true).
All named paths remain under the server's knowledge-base root. `close` saves the session's
private store, and `kb_close` saves a writable store with an assigned path.

Files use knowledge-base format v2; v1 files still open. The model identity format stays
unchanged, and the hindsight head is excluded from the weights digest even though it is
now exported and counted as served parameters. Bundles exported before this change must
be re-exported to include that head. Other model weight/architecture changes still invalidate
old stores. Raw conversation text is not persisted for rebuilding: incompatible stores
must be discarded and relearned, or rebuilt from separately authorized source material.

### Verification and calibration

Run the focused serving, packed memory, collection coexistence, bundle export, and
conversation-write tests, including `tests/test_serving_writes.py`. These exercise actual
retrospective passes and observed-byte scoring as well as injected failures.

`scripts/probes/probe_serving_writes.py --model CHECKPOINT_OR_BUNDLE --conversation INPUT.jsonl
--question "..." --answer "..." --output REPORT.json` measures observed surprise ranking,
retrospective delays of 0/1/2/4 groups versus a completed-window pass, retrieval and answer
loss with/without each writing rule after the same conversation, and deletion/reopening.
Input rows contain `text`, optional `role`, and `novel: true` for planted facts. Put enough
routine messages after the planted fact to exceed the model's retained event limit. Pass
`--device cuda` for GPU execution. `--smoke` instead of `--model` checks the full probe with
random tiny weights and a synthetic conversation; its quality numbers are not calibration.

The provisional surprise and delay targets in section 6 remain hypotheses. No trained-model
quality claim follows from passing the mechanics tests or the random-weight smoke probe.
