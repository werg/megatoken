# Omnimodal implementation integration

The implementation adds causal per-event request revisions to TrackEvent and
uses them in the existing grouped meridian trainer. A later event request does
not enter an earlier prediction. The standalone grouped KL is normalized by
the number of active raw coordinates.

The `megatoken.omni` package contains immutable aligned bundle/task/relation
contracts; strict pack revision/codec pairing; and a pure raw-facets-conditioned
flow-matching latent head with masked numerator/count accounting. The generic
flow adapter is a fallback training primitive, **not** initialized Pocket,
Sana, or SANA-Video weights. The chosen donor generators must be imported and
their exact latents/normalization and schedules verified before capabilities
can be claimed. Existing waveform/pixel debug realizers remain unchanged.
No data, model weights, or benchmark results are supplied in this change.

The concordant recipe is versioned separately from the existing production
stage recipes. It is not enabled automatically. External assets must be pinned
by SHA256 and codecs matched via PackRegistry, with input/output partitions
verified before feeding the native episode stream.
