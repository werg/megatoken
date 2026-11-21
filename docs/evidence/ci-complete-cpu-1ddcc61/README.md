# Complete CPU gate budget and retained original outcomes

The CI run [44936714352](https://github.com/werg/megatoken/actions/runs/44936714352)
tested consumer `1ddcc6110d70703f6b08244d53ec461047677bec` on actual pull-request
merge `a19c6afa4875ad4274d346338b4b970bc5ec8349`.

| Job | Original outcome |
| --- | --- |
| Python 3.11, `136944918289` | Complete CPU suite: 3,567 passed, 82 skipped, 20 warnings in 1,033.62 seconds. The installed-package and CLI steps passed. |
| Python 3.12, `136944918187` | Installed-package check passed. The complete CPU suite reached 88% and was cancelled; it did not produce a complete result. |

The Python 3.12 runner began at 18:39:50 UTC and reported cancellation at
19:05:02 UTC on 21 November 2026, with the CPU job configured for 25 minutes.
That timing is consistent with the configured job limit. The original log does
not identify the cancellation initiator; this report does not claim a later
push cancelled it or count the partial output as a pass.

The CPU job now has the same bounded 45-minute allowance as the semantic CI
workflows. Its matrix, complete test selection, package checks, artifact retention
and failure handling are unchanged. The new budget needs a fresh hosted run;
neither original result is relabeled as a result of this change.

The adjacent inventory records compressed and decoded SHA-256 hashes of the
original decoded job logs and the observed job-status response. Decompress a
`.gz` member with `gzip -dc` to inspect it. The 82 skips remain skips and this
general CPU run is distinct from the separately retained 625-test V8 selection.
