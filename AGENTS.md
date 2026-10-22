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
