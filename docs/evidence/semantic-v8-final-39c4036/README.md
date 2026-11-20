# Final native v8 regression at 39c4036

The complete workflow-selected suite passed **244 tests across 27 modules**,
with zero failures, errors, or skips. Both live source-authority tests ran.
Full `ruff check src tests scripts`, `ruff format --check src tests scripts`,
and `git diff --check` also passed; all 532 checked Python files were already
formatted.

The clean consumer was `39c4036564428e44cc89caf261f517d856482695`, tree
`ca55473db0ed3343ad930833a4ee09ad48bf7caf`. The clean source producer was
`012c9fe49a082319bf98ac0c571596da925741a4`, tree
`26454c685a6136ab6d6d18636e6073216ee9d195`. Both identities and empty working-tree
statuses were unchanged after the run. Later documentation and CI commits do
not change these measured identities.

[summary.json](summary.json) records every selected module, exact command,
exit code, log hash, policy, contract, tokenizer, runtime version, and clean
before/after identity. [semantic-v8-tests.xml](semantic-v8-tests.xml) and
[pytest.log](pytest.log) retain the original test evidence. The suite includes
all `tests/test_*v8.py`, `tests/test_algorithm_v7.py`, and
`tests/test_protocol_properties.py`; it is not the entire Megatoken test suite.

The environment used Python 3.12.14, PyTorch 2.10.0+cpu, tokenizers 0.23.3,
Ruff 0.17.6, and Node v24.22.3. The production Ling tokenizer's exact bytes and
identity are retained in the summary. No model-learning claim follows from
these regression checks; actual model inference is measured separately.

`artifact-index.json` binds the original validation files. Verify all packaged
files from this directory with `sha256sum -c SHA256SUMS`.
