# Frozen-contract direct V8 release composition

The release branch `5ecdbaa9842ab086c312a7d0be281780091df065` contained the
source-order correction together with newer Promise and recovery code. Its
contract therefore differed from producer release `4af54890`. Treating those
two heads as a pair would have been incorrect.

This composition starts from strict verifier/input relocation
`3fb3370dd3c10dbc9309fd2870760cf87751b418` and adopts only the source-order
verifier, its tests and documentation from `5ecdbaa9`. Its public contract is
byte-identical to producer `4af54890`; the Promise and recovery candidates
remain separate. The exact three changed files are listed in `proof.json`.

Published executable source is `d4afac5bb7261aa7ef85c1d65d3ae6dcaf6caeef`,
with tree `3e6d58624f5557bb5b4877a260e504a572dd90a4`. Local source commit
`5dc020216b24df94ff7fbe2463d23f8f8c367f22` has that same tree. The complete
focused suite passed **62 cases, zero failures/errors/skips, in 15.88 seconds**
using Python 3.12.14, Torch 2.10.0+cpu and pytest 8.4.2 from the exact producer
dependency lock. Original stdout and JUnit are retained unchanged.

The added seven source-order controls consist of a real two-source prepared
ordering check and six pre-dispatch membership/split refusals. Preparation in
the positive ordering check is explicitly controlled; the separate actual
checkpoint/export test executes one controlled optimizer update. The verifier
performs zero optimizer updates and zero decoder forwards. These local tests
do not qualify a production tokenizer or the complete paired source campaign.
The independently frozen hosted release gate must establish those claims and
actual package/restore with original asset paths absent.

Reproduce the focused controls from the exact source checkout with:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_semantic_v8_release_verifier.py --junitxml=native.xml
```

Use an external writable temporary directory and output path so the trusted
checkout remains clean for a subsequent release verification.
