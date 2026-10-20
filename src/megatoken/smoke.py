"""Legacy smoke/training pipeline was removed with required retrieval-memory training."""

def run_smoke(*args, **kwargs):
    raise RuntimeError(
        "The old B0/P1/.../R offline smoke is no longer a supported training "
        "route. Run mandatory causal retrieval-memory training with a pinned "
        "recipe, source sidecar and native-meridian validation suite."
    )
