# Immutable graph observation fixtures

`input-mutation-short.json`, `input-mutation-context.json`, and
`result-mutation-context.json` contain controlled trajectories produced from the
source printed in each public program record by tracing-js-self commit
`c4296c271eec32c2f2ab4cd34c30fd8b61e6ba39`.

The first two exercise mutations of an observed input with aliases and a cycle.
The short trajectory reaches final transport verification; the long trajectory
also refreshes public selected state after the mutation. The third observes a
native result with a cycle and null prototype, mutates it, and refreshes state.
All sources are small original regression fixtures.

Private interpreter and independent-source evidence is intentionally omitted.
The sidecars identify these as controlled, locally accepted component fixtures,
with unverified reference status. These tests establish native-consumer
behavior and byte preservation, not full source admission or learned success.
