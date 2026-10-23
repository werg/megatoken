# Track context windows and explicit evidence reads

New models default to longer direct attention within a track and shorter direct
attention across tracks. The local allowance is 256 committed events on that track;
the foreign allowance covers other-track events within the most recent 16 committed
events globally. These are experimental values, bounded by the existing session
cache capacity. Committed events include internal retrieval work and ordinary
messages. Representative quality and latency comparisons must calibrate the defaults.

Busy neighbors do not age out a quiet track's local allowance. The global cache
budget can still force reclamation when several tracks collectively need too much
storage. Essential evidence can be retained under a separate bounded allowance.
Cache status distinguishes window expiration from global-budget pressure.

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
  --families 32 --local-events 4 --cross-events 1
```

The count denotes original task instances. Each has related fetch, retained-source,
long-local-history, and unavailable-source variants kept in one split. Archive
reads actually execute with checksummed receipts and linked results. The source
has one declared task template; numeric variation does not demonstrate new-template
generalization. Unavailable-source clarification is marked incomplete rather than
successful task completion.

This corpus uses the existing dynamic-execution stream in the ordinary trainer.
Preparation reports public context changes, checked reads, and unavailable-source
cases. Training and held-out reports include retained events and cache bytes per
training window. The preparation, restore, batch, and backward regression checks
that these demonstrations reach actual training losses. Broader delegation data
is still required.

## Acceptance and remaining work

```bash
python scripts/accept_track_context.py \
  --root /absolute/new/context-acceptance --device cpu
```

The local tiny-model comparison with 24 background observations retained 29,696
key/value bytes under long shared context and 7,168 bytes under asymmetric windows.
It used identical weights and input observations, checked explicit rereading, and
reconstructed the selected action's score. The receipt explicitly leaves learned
quality and whole-device peak memory unmeasured. A separate bounded CUDA regression
checks masks, reconstruction, and finite gradients.

The [follow-up plan](dynamic-tracks-defaults-follow-up-plan.md) still requires broad
task families, learned handoffs and summaries, calibrated windows, sustained voice
sessions, server-wide memory admission, durable recovery, and quality comparisons
at equal memory budgets. Archive rereading is a new causal observation; exact
offloading/restoration of a suspended track's old projected cache remains separate
work. Long-session transcript and task metadata also need the planned archival and
retention work.
