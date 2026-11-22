# Independent review of ordinary V8 input relocation

This bounded review executed native source
`b85a3301b6482bdf5a5a9ba79f7f356c5f824681`, tree
`8c92b6c9278d9885430b276278563762dc95e660`. It does not change runtime
code, historical checkpoints, position receipts or either frozen hosted fit.

The retained probe passed **29 named checks**, with zero failures or skips,
zero optimizer updates and zero decoder forward passes. The complete individual
checks, source hashes, runtime and probe hash are in `review.json`.

The checks establish that the shared identity function and position objective
perform no hidden filesystem reads; relocation and reordering of distinct
prepared fingerprints preserve identity; changed codec, prepared inputs, source
policy, seed or window settings change identity; malformed or duplicate content
identities refuse; physical configurations remain unchanged; ordinary V8
compatibility relaxes only its two input locators while non-V8 and distributed
V8 locators stay strict; old path-bound position receipts cannot become history
under the new objective; and checkpoint metadata still binds position history.

The real target-inventory collector also executes through its existing explicitly
controlled fixture. It traverses the same 48 distinct synthetic windows three
times: the existing fixture control, an original-location comparison and a
relocated comparison with the original directory absent. Thus this probe performs
144 controlled window traversals, not optimizer updates. The full collector
output is identical across relocation, and its configuration digest equals the
position objective's digest. Source admission, codec and planner boundaries are
injected in this fixture. This is not independent source/data qualification or
measured model behavior.

The separate actual optimizer continuation proof belongs to the source change's
own evidence. This review neither reruns it nor combines its update counts with
these identity controls. Production distributed input relocation, accepted live
history restoration, whole-release qualification, final default cutover and
learned quality remain outside this review.

To reproduce, use the frozen source above and Python with the project's test
dependencies, copy this evidence directory into that checkout, then run:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  docs/evidence/v8-input-relocation-independent-2026-11-22/probe-input-identity.py \
  --output /tmp/v8-input-identity-review.json
```

The probe requires the exact frozen source revision. It records the Python,
PyTorch and pytest versions rather than asserting an unverified environment
lock. `files.json` seals the retained probe, report and this explanation.
