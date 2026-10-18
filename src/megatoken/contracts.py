"""Stable non-neural protocol identifiers shared by data, models, and serving."""

from types import MappingProxyType

ROLE_TO_ID = MappingProxyType(
    {"system": 0, "user": 1, "assistant": 2, "tool": 3, "document": 4, "environment": 5}
)
RECORD_VERSION = 1
