"""One codec, sampler, scorer and parallel realizer for training and serving."""

from .codec import ByteCodec, ByteFallbackCodec

__all__ = [
    "ByteCodec",
    "ByteFallbackCodec",
    "FacetedGenerator",
    "ActionTape",
    "ActionSpec",
    "FieldSpec",
    "SchemaSpec",
    "CopySource",
    "PreparedTargets",
    "PreparedAction",
    "Observation",
]


def __getattr__(name):
    # Data ingestion must remain usable without importing PyTorch.
    if name in {"FacetedGenerator", "PreparedTargets", "PreparedAction"}:
        from .policy import FacetedGenerator, PreparedAction, PreparedTargets

        return {
            "FacetedGenerator": FacetedGenerator,
            "PreparedTargets": PreparedTargets,
            "PreparedAction": PreparedAction,
        }[name]
    if name in {"ActionTape", "ActionSpec", "FieldSpec", "SchemaSpec", "CopySource", "Observation"}:
        from . import structure

        return getattr(structure, name)
    raise AttributeError(name)
