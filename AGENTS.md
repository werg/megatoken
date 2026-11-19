# Agent guidelines

## Pre-release development

This repository and its paired producer, `werg/tracing-js-self`, are pre-release.
Nobody depends on compatibility with the current implementation or saved formats.

- Make the smallest direct change that solves the problem. Change APIs, formats,
  schemas, and implementations in place when that is the quickest sensible path.
- Do not require version bumps, migrations, compatibility shims, deprecation
  periods, or a format-release process. Existing version labels may stay in place
  while their definitions evolve. Do not make versioning a blocker for a PR.
- Update affected producers, consumers, tests, fixtures, and examples together.
  Regenerate affected artifacts when necessary; do not preserve an obsolete
  format merely to keep old generated data working.
- Keep correctness checks and honest evidence. Historical results describe their
  recorded code and inputs; they are not proof that a later change works.
- Save authorized work in small commits and push completed increments promptly.

These instructions supersede older repository guidance that requires version
bumps or compatibility work during this pre-release period.

## Execution resources

- The local GPU is authorized for appropriate bounded work, including small CUDA
  regressions and inference checks. Check its current load and available memory;
  preserve other users' processes. Large downloads and preparation remain cloud work.
- On cloud placement exhaustion, try explicitly sized smaller CPU/RAM requests
  that remain sufficient for the workload. Preserve required GPUs and memory floors.
  Account concurrency quotas need a free job slot; smaller requests do not bypass them.

## Subagents

- When delegating to subagents, only use Sonnet or Luna models (e.g. `model: "sonnet"`).

## Talking to the user

- Never use internal implementation jargon (class, constant, file, config-key,
  metric or phase-code names such as `PINNED_EXTRACTOR`, "locked pack manifest",
  "B0/P1", "raw facets") unless the user has used that term first and clearly
  knows what it refers to.
- Write for a bright but forgetful senior undergraduate: describe what a thing
  does and why it matters in plain words. If a name is needed so the user can
  find something, give it once, after the plain description, in parentheses.
- Define any number you report: what was measured, on what, and what good
  looks like.
- Never assume the user has read a plan, design document, proposal, transcript
  or subagent report. Do not use terms coined there (design names such as
  "agenda", "library", "open action", "write-out", step or section labels) or
  in your own working notes as if the user knows them. Either say in plain
  words what the thing is, or introduce the term with its meaning the first
  time it appears in a message, every time it is needed again later.
- Do not use subagent jargon: no agent names or IDs, "forks", "background
  tasks", "the reviewer found" without saying what was reviewed, or references
  to work the user did not see. Report what was done and found, not who did it.
- When asking the user to decide something, explain each option so that it can
  be understood without the surrounding conversation: what it is, what it
  changes, and the trade-off.
- Before relying on a fact about the project in a question or recommendation
  (for example, "we still have X"), check the current code, and say plainly
  when something the user may expect has been removed or changed.
