"""Validate authorized semantic request payloads without legacy corpus dependencies.

This is an external-effect safety boundary, not an alternate trace format.
"""
from typing import Any

from .host_operations import materialize_request


def _host_request(fields: dict[str, Any], *, heap=None, require_promise=False):
    try:
        return materialize_request(fields, heap=heap, require_promise=require_promise)
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(str(error)) from error
