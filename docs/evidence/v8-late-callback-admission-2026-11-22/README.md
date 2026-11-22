# One late callback: retained current refusal

This is a failing engineering execution on unchanged native runtime source
2062414631c16749815895a472ddfdb67f5b869d and producer source
cc0d61e98dc6252749f4a2a4c8b669d79b0f7efa. The added producer helper executes the
original listener source independently and supplies lexical decisions to the
production host machine. The root registers once, returns 1, and checkpoints.
Only afterward a durable journal records one callback arrival with input 5.
Restoring that journal does not register again; the callback reference returns 5.

The actual native runtime executes the four root decisions, publishes its exact
completion, checkpoints, and independently restores that runtime with zero host
dispatches. Selecting the callback then fails at the unchanged explicit dynamic
invocation refusal. The six original-input lexical records are supplied; this
is neither learned quality nor source-admission qualification.

The retained native-red run has one failure, zero skips, and 20.74 seconds of
pytest time. The earlier native-red-setup run has one fixture assertion error
(the durable event is under arrival.event), before native execution; its raw
output is also retained. The helper initially encountered missing acorn in the
assigned worktree; using the existing shared node_modules resolved that
environment issue. A subsequent helper assertion exposed and corrected the
fixture request invocation name before successful fixture generation. Neither
setup issue is counted as the native refusal.

Local runtime: Python 3.12.14, Torch 2.10.0+cpu, pytest 9.2.0. Pytest differs from
the repository's declared <9 range and hosted 8.4.2 lock. Final paired hosted
qualification remains required. No optimizer updates or workflow launches.
