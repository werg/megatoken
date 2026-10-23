"""Stable non-neural protocol identifiers shared by data, models, and serving."""

from types import MappingProxyType

ROLE_TO_ID = MappingProxyType(
    {"system": 0, "user": 1, "assistant": 2, "tool": 3, "document": 4, "environment": 5}
)
# One loss vocabulary and default weighting, shared by preparation and training.
DEFAULT_LOSS_WEIGHTS = {
    "lexical": 1.0,
    "reconstruction": 1.0,
    # Each media loss is a mean over valid renderer coordinates PER EVENT.
    # Separate denominators prevent speech and video tensor sizes from
    # silently becoming reference text tokens or dominating shared training.
    "audio_prediction": 1.0,
    "audio_reconstruction": 1.0,
    "visual_prediction": 1.0,
    "visual_reconstruction": 1.0,
    # One mean per declared generated peer pair, not per arbitrary all-pairs clique.
    "relation_semantic": 0.25,
    "structure": 0.25,
    "modality": 0.25,
    "policy_outcome": 0.1,
    "binding": 0.5,
    "placement": 0.25,
    "terminal": 0.1,
    "facet_prediction": 0.1,
    "relation": 0.025,
    "encoder_alignment": 1.0,
    # Per annotated lookup (``megatoken.training.retrieval``): the 768-wide query
    # matches the teacher's embedding of the question, and the annotated items
    # take the softmax mass of the retrieved scores.
    "query_alignment": 0.5,
    "retrieval_ranking": 0.25,
    "router_balance": 0.005,
    "router_z": 1e-4,
}
LOSS_NAMES = tuple(DEFAULT_LOSS_WEIGHTS)
# Losses produced by the deterministic action decoder (a leaf constant, so the
# decoder and model packages need not import each other).
DECODER_LOSSES = ("lexical", "structure", "binding", "placement", "terminal")
