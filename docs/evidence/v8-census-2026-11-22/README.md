# Canonical census and native baseline evidence

The local census/orchestration-authority selection on the working increment based on `88764cc0e25d866556d4d08bf1f348eab15c0f5a` passes **23/23**, zero skips: four census controls and the existing nineteen independent authority controls. The census includes same-source repeated root calls, canonical await/resume and request/result/continuation links, exact source eviction/reread, and malformed identity/order mutations. `semantic_child` remains absent from ordinary V8 training records. These are engineering controls, not learned-quality evidence.

```sh
PYTHONPATH=/workspace/python-deps:src TMPDIR=/workspace/tmp python -m pytest -q tests/test_semantic_v8_behavior_census.py tests/test_semantic_v8_orchestration_authority.py --basetemp=/workspace/tmp/census-authority
```

The actual invocation uses the session's corresponding absolute workspace directories. Decode each complete log with `base64 -d < FILE | gzip -d`; `SHA256.json` records both stored and decoded hashes.

The hosted baseline logs retain exact native `88764cc` and producer `a494f00`:

- Native [run44962499068, gallery job137024168943](https://github.com/werg/megatoken/actions/runs/44962499068/job/137024168943): **728 native checks pass with zero skips**, including actual shared-runtime orchestration controls; all eleven gallery source/native/token/packing cases pass. This does not qualify subsequent changes or learned execution.
- Native [general run44962499078, Python3.12 job137024168979](https://github.com/werg/megatoken/actions/runs/44962499078/job/137024168979): **one failure,3,668 passes,83 skips**. The failure is `test_supplied_repository_identities_cannot_stand_in_for_the_running_consumer`, whose positive setup assumes an editable checkout although this job installs a wheel. The full failure is retained. The follow-up must correct test isolation while preserving the strict running-code origin requirement. General-suite success is not claimed.

The paired workflow now retains a running qualification across incremental pushes and explicitly includes the legacy launch regression file in addition to all V8 files. Complete current-pair admission, normal production assets, persisted lifecycle behavior, and the final seventeen-package acceptance ledger remain open.
