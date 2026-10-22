# Agent guidelines

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
