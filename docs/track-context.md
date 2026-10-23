# Track context windows and explicit evidence reads

New models default to longer direct attention within a track and shorter direct
attention across tracks. The local allowance is 256 committed events on that track;
the foreign allowance covers other-track events within the most recent 16 committed
events globally. These are deliberately guessed operating defaults, bounded by the
existing session cache capacity. Committed events include internal retrieval work
and ordinary messages. They apply now; calibration is not a prerequisite for use.

Busy neighbors do not age out a quiet track's local allowance. The global cache
budget can still force reclamation when several tracks collectively need too much
storage. Essential evidence can be retained under a separate bounded allowance.
Cache status distinguishes window expiration from global-budget pressure.

Track identity describes continuing work, independently of a message's role. Put
the user's messages and the assistant's replies on the same main conversation
track when they should share its long window; delegated workers get separate
tracks. Linear preparation, local student collection, one-shot generation, and the
automatic coordinator now use this convention. Tool results from explicitly
delegated calls return to their requesting worker tracks. Session opening reports
the main conversation address for clients. Custom protocol clients still choose
their own addresses and should use this address for the continuing conversation.
Reprepare old role-based corpora and use a fresh collection identity for the new
profile; roles and explicitly declared task groups retain their original meaning.

The policy covers observation processing, content and modality prediction, and
explicit evidence-reference reads. Current prediction queries still interact within
a jointly selected output group. Rendering uses the immutable selected result
processed under that policy. Earlier information can also survive in summaries,
representations, and retrieved memory. Windows control direct attention; source
permissions and session memory scopes continue to govern access independently.

## Stored history and memory

Each retained event has one set of projected keys and values. Reads gather the
union needed by their tracks and apply independent visibility where windows differ.
Groups no longer needed by local windows, recent foreign context, or retained
references are physically removed without changing their original temporal
positions. Retention conservatively keeps a whole historical joint group when any
peer remains needed.

Completion and cancellation publish a context change that stops retaining detailed
local history. Recent foreign visibility and another reader's references can still
keep some of it alive. Resumption publishes ordinary profile windows again; missing
evidence requires an explicit read.

Available input text, prepared incoming media features, and completed text outputs
also have bounded host copies. The configurable defaults allow 64 MiB of payload
and 1,024 events per session. Old copies leave this archive when it fills. This is
a source archive rather than a durable session checkpoint or saved model cache.
Generated media needs checked preparation before it can become model input again.

An explicit reread checks the session archive, then previously admitted items in
mounted source collections. It observes the material again on the consuming track
under a fresh address and reports its original source address. A former system
message becomes evidence and cannot reactivate its instructions. Missing material
produces an explained failure. Session archives remain separate between sessions.

Status reports live logical cache bytes, distinct cache-storage bytes including
retained rendering/output snapshots, host payload bytes, and event counts. It does
not measure weights, transient buffers, allocator overhead, or total accelerator
peak memory. Acknowledged outputs must be released through the delivery protocol
to reclaim their snapshots.

## Serving and worker actions

Inspect a session or track with `context_status`. Set a declared track's windows
and retained evidence with `configure_context`. Window requests cannot exceed the
session event budget; at most eight references can be retained per reader track.
References must name resident evidence. Increasing a window cannot recreate an
evicted event.

After opening a session and creating a worker, explicitly hand off available
evidence. The small windows below demonstrate the protocol rather than calibrated
production settings:

```json
{"op":"observe","session":"SESSION","events":[{"track":"main","event_id":"source-1","text":"The project key is violet."}],"effort":1}
{"op":"configure_context","session":"SESSION","track":"track:0","local_events":4,"cross_events":1,"pinned_events":["source-1"]}
{"op":"context_status","session":"SESSION","track":"track:0"}
{"op":"fetch_context","session":"SESSION","track":"track:0","source_event":"source-1","effort":1}
```

Use the fresh address returned by a reread in later reference requests. Workers
can select context-status, window-change, and fetch actions through ordinary
lifecycle tools. Selected handoffs require evidence visible to the sender. Context
actions follow committed selections and publish later observations.

Waiting tasks now have a separate bounded capacity from active execution. Waiting
parents release an execution allocation so children can start. Continuing or
resuming waiting work requires a free active allocation, reflected in legal tools.

## Preparation and training

Context changes are public system announcements. Serving, the local student, and
grouped training apply the same announcement before processing its observation.
Selected actions record the exact policy; recorded history transitions retain the
policy used for each step. Outcome scoring reconstructs the transitions under
current weights and restores the selection policy. Direct replay rejects a policy
mismatch.

Grouped tapes now use version 16. Recollect old tapes rather than relabeling them
without their visibility history. The prepared-window restart contract also
changed. Existing compatible model configurations without window settings load
with their original shared-window behavior; new configurations save asymmetric
settings. Re-export serving bundles and begin a new run when changing context
semantics. An old training cursor does not establish exact restart.

Prepare authored demonstrations through ordinary deduplication and export:

```bash
python scripts/prepare_context_tasks.py \
  --root /absolute/new/context-corpus --codec /absolute/codec \
  --families 16
```

The count denotes original instances **per task template**. The six templates
exercise document fields, invoice arithmetic, corrected source revisions, user
formatting constraints, joining two sources, and exact quotation. Each instance
has recent-evidence, local-history, pinned-evidence, archive-reread, compact-summary,
and unavailable-source variants. These are authored exercises, not natural data or
student rollouts. Archive reads execute against bounded source storage with checked
receipts. Unavailable-source clarification remains incomplete task completion.

The default curriculum uses local/foreign allowances of 16/8, 64/8, and 256/16
events; the last matches the operating defaults. All window and strategy variants
of one original task remain in the same split. Training, validation and test
contain different original task instances, with every template represented in each.
This is instance holdout, not unseen-template generalization. The preparation
report counts examples after deduplication and records the producer checksum.

The ordinary training mixture now includes a dedicated context-management stream:
0.5% of the mixture budget in pretraining and intermediate stages, and 1% in
supervised training, taken from the previous dynamic-execution allocation. Keep
the original dynamic-execution corpus too. Include the new prepared root alongside
the other required roots in the training datasets. Admission requires independent
original tasks for each context strategy, every template, the operating defaults,
and completed-history release, in both training and validation. Old one-template
demonstrations cannot satisfy this supply. Missing supply fails before allocation
of expensive model resources; start a new run after changing mixture semantics.

Answer checking recomputes facts from actual sources, verifies linked read results,
and checks visibility for answers and summaries. The tests also replay episodes
through actual serving context, including internal retrieval events, and sample
scored decisions through the normal mixture loader. Training and held-out reports
include retained events and cache bytes per window. Broader natural conversation
and media evidence, learned choices, and quality evaluation remain necessary.

Small public evidence indexes retain original source addresses on their consumers.
Fetch targets therefore name addresses the model was actually given. Evidence
retention is requested by a source owner that can still read it when the selected
request executes.

## Acceptance and remaining work

```bash
python scripts/accept_track_context.py \
  --root /absolute/new/context-acceptance --device cpu
python scripts/accept_context_training.py \
  --data /absolute/new/context-corpus \
  --root /absolute/new/context-training-acceptance --codec /absolute/codec --device cpu
```

The local tiny-model comparison with 24 background observations retained 29,696
key/value bytes under long shared context and 7,168 bytes under asymmetric windows.
It used identical weights and input observations, checked explicit rereading, and
reconstructed the selected action's score. The receipt explicitly leaves learned
quality and whole-device peak memory unmeasured. A separate bounded CUDA regression
checks masks, reconstruction, and finite gradients.

The training acceptance command runs the ordinary optimizer on prepared context
demonstrations, restarts from its checkpoint, exports, changes context through the
public service, and generates with an explicit evidence reference. A bounded local
run completed this path; it does not establish learned delegation quality. Prepare
and train against the same unchanged source revision because preparation records
the reader implementation identity.

The [follow-up plan](dynamic-tracks-defaults-follow-up-plan.md) still requires broader
natural task families, demonstrated learned handoffs and summaries, sustained voice
sessions, server-wide memory admission, durable recovery, and quality comparisons
at equal memory budgets. Archive rereading is a new causal observation; exact
offloading/restoration of a suspended track's old projected cache remains separate
work. Long-session transcript and task metadata also need the planned archival and
retention work.
