# Native durable request and target inventory first probe

Exact candidate: `91e698b51aafc1283d45f7e5648dc7c3829f5fc0`.
[Run 45003800735](https://github.com/werg/megatoken/actions/runs/45003800735).

The complete unabridged log is retained in `initial-failure.log`.
The actual controls returned **51 passed / 1 failed**. All 12 journal controls,
including real subprocess crashes and live contention, passed. The inventory
duplicate-window rejection was correct but its expected error did not match the
earlier summary validation. The next candidate rejects duplicate window indices
explicitly. Ruff also found import/unused-import/exception-chain and formatting
issues. This failed run is not a passing qualification.

The next candidate measures update-producing windows and phase-meter cost with
the actual native objective planner. This remains a preparation inventory with
zero optimizer updates until a separate real ordinary training pass proves
the retained positive-weight target presence. Token/mask-position coverage and
learned quality are unmeasured. The request journal remains a storage foundation;
server post-request checkpoints and paired execution ownership are not integrated.
