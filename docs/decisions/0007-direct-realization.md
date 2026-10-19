# ADR 0007: realize directly from the shared raw-facet representation

Status: accepted for architecture v8; supersedes [ADR 0004](0004-contextual-realization.md).

## Decision

E ingests and P predicts exactly the same canonical 32-facet representation.
Every realizer (text, audio, image, video) reads that block directly. For the
next causal prediction, C separately processes selected facets against
role/track/modality and KV history, publishes full-resolution history, and
caches final C features for P. No realizer consumes early/final C activations,
and no history is a hidden decoder input.

```text
observation → E ─┐
                 ├→ canonical raw facets → R → output
history → C → P ┘                 └→ C → KV/history → future P
```

The current output's R can be evaluated before or independently of C.
Text output uses public schema and copy inputs. Media heads use a bounded
OutputSpec and optional explicit media control, never a private C/KV read.

## Training

- Main prediction readout: `R(sample(P(history)))`, with no decoder C pass.
  Recurrent continuation still commits that selected sample through C once.
- Bootstrap reconstruction: noisy `E(target) → R` trains E and R. Only the
  clean target needs C to continue the teacher-forced trajectory.
- Recurrent E reconstruction: noisy `E(target) → R` with all R/U weights
  detached; gradients reach E alone. This branch does not execute C or
  read the previous history.
- Grouped text/media readouts use the identical direct interface. No
  hypothetical meridian is integrated just to score R.
- Latent-action RL freezes E, transition C/KV and R/U, preserving a fixed
  conditional action map while training P.

History affects outputs indirectly through P's sampled facets, not through
a separate post-prediction C→R dependency. In particular, equal raw facets
and equal explicit output controls must render identically under differing
KV history or final C features.

## Compatibility and validation

Remove the no-longer-used early C cache, `central_early_layer`, contextual
text decoder projections/attentions and three-way media fusion. The
`RealizationContext` now contains only facets. v8 checkpoints and v8 one-
and multi-event tapes are incompatible with v7; regenerate them. Dataset
schema is unchanged. Media decoders remain untrained reference baselines.

Regression gates: direct-R history independence; encoder-only gradient
isolation; one C commit for recurrent selected inputs; text and media
supervision without an extra C pass; grouped replay; strict version checks.
CPU validation does not measure media fidelity or Blackwell throughput.
