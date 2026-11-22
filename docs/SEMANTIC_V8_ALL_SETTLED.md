# Independent allSettled replay and durable callback contract

The V8 native Promise join validator now derives `Promise.allSettled` reactions
from its own input Promise heap and scheduler. Each input gets a fresh ordinary
status/value-or-reason object. Rejection remains an input outcome; the aggregate
fulfills only after every reaction, preserving input order and value aliases.
Empty input fulfills immediately. Malformed entries, reserved identities and
premature settlement refuse without changing accepted state.

Restored collecting state checks pending entries against their settled inputs.
Once fulfillment exposes the result, ordinary guest mutation is permitted.
The public collecting-join bound is 4,095 inputs plus the array length property;
the producer's owned source runtime has its separately declared 64-element bound.
`Promise.any` remains outside this admitted public profile.

The producer contract and regional fixture are regenerated and mirrored byte
for byte. The private callback-arrival definition now specifies exact original
source/parent/closure binding, a shared receiver/argument graph and durable
arrival ordering. Public callback invocation records keep their existing shape;
this private metadata does not imply an actual-model callback recovery pass.

Validation before final pair qualification: 46 native join/protocol/replay tests
pass with zero failures or skips. The producer separately passes 26 join/source,
cross-feature rollback and population tests. Three new allSettled inputs expand
recurring interoperability from 37 to 40 sources. Exact source/native/tokenizer
admission, ordinary model lifecycle and hosted checks must retain their own
published pair and results; unit tests are not substituted for those gates.

See the producer's
[behavior and source evidence](https://github.com/werg/tracing-js-self/blob/codex/language-coverage-v8-cutover/docs/SEMANTIC_V8_ALL_SETTLED.md)
and [main progress ledger](https://github.com/werg/tracing-js-self/pull/47).
