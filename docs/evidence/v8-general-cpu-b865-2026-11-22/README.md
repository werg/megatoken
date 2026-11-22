# Complete general CPU CI at b865b5f

Both Python 3.11 and Python 3.12 lanes of run 44966454098 completed successfully
at exact native head `b865b5f25f07b437075cc66e38c5bbfd14c5ae1c`. Each reports
**3,673 passed, 83 declared skips and 20 warnings**. Both repository-wide Ruff
gates pass over 584 files. These results verify the earlier installed-wheel
fixture and formatting corrections. The earlier failed runs remain retained.

The full job logs preserve every skip reason. Asset-dependent cross-repository
and production-tokenizer controls belong to the separately provisioned paired
semantic gates; accelerator coverage is skipped here. A complete general CPU
pass does not mean those cases ran in this workflow or establish model quality.
The native head is a workflow-only descendant of executable `10847cb` and pins
producer `14e3c07` for the reverse paired workflow. These exact historical results
do not qualify later orchestration transcript/materialization changes.

`SHA256.json` records exact job IDs and full encoded/decoded log hashes. Logs are
base64-encoded gzip of complete UTF-8 job output.
