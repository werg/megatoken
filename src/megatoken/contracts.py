"""Stable non-neural protocol identifiers shared by data, models, and serving."""

from types import MappingProxyType

ROLE_TO_ID = MappingProxyType(
    {"system": 0, "user": 1, "assistant": 2, "tool": 3, "document": 4, "environment": 5}
)
# One loss vocabulary and default weighting, shared by preparation and training.
DEFAULT_LOSS_WEIGHTS = {
    "lexical": 1.0,
    "reconstruction": 1.0,
    "structure": 0.25,
    "binding": 0.5,
    "placement": 0.25,
    "terminal": 0.1,
    "facet_prediction": 0.1,
    "encoder_alignment": 1.0,
    "router_balance": 0.005,
    "router_z": 1e-4,
}
LOSS_NAMES = tuple(DEFAULT_LOSS_WEIGHTS)
