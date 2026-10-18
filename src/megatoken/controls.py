"""One causal control protocol shared by training, serving, and tape replay.

Seed/reference bytes retain their exact spelling and framing. Only public tool
JSON is canonicalized, so transport whitespace cannot change model conditioning.
Payload construction needs no ML runtime; tensor packing is imported on demand.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from megatoken.io import strict_json_loads

if TYPE_CHECKING:
    from torch import Tensor

    from megatoken.model import FacetedCore


def build_control_payload(
    *,
    seed: bytes = b"",
    reference: bytes = b"",
    reference_origin: str | None = None,
    tools_json: str = "[]",
) -> bytes:
    """Frame only controls whose reference has explicit causal provenance.

    References may come from external evidence or earlier history. Current or
    future targets and verifier outcomes are not accepted reference origins.
    This validates declared provenance; callers must supply the actual earlier
    evidence rather than relabeling target bytes as an external source.
    """
    if not isinstance(seed, bytes) or not isinstance(reference, bytes):
        raise TypeError("control seed and reference must be bytes")
    origins = {"external_evidence", "earlier_history"}
    if reference_origin is not None and (
        not isinstance(reference_origin, str) or reference_origin not in origins
    ):
        raise ValueError("control reference requires causal external_evidence or earlier_history")
    if reference and reference_origin is None:
        raise ValueError("control reference has no causal provenance")
    if not isinstance(tools_json, str):
        raise TypeError("tools_json must be a JSON string containing a tool list")
    tools = strict_json_loads(tools_json)
    if not isinstance(tools, list):
        raise ValueError("tools_json must contain a list")
    payload = seed
    if reference:
        payload += b"\n<reference>\n" + reference
    if tools:
        canonical = json.dumps(
            tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload += b"\n<tools>\n" + canonical
    return payload


def encode_controls(
    core: FacetedCore,
    payloads: Sequence[bytes],
    *,
    encoder_grad: bool = True,
) -> Tensor | None:
    """Encode independent controls; absent controls are exactly the neutral zero.

    An all-empty batch skips the encoder. Empty rows in a mixed batch have zero
    output and zero encoder gradient, matching the core's ``control=None`` path
    even when its learned control encoder has nonzero biases.
    """
    if not payloads:
        raise ValueError("cannot encode an empty control batch")
    if any(not isinstance(payload, bytes) for payload in payloads):
        raise TypeError("control payloads must be bytes")
    if not any(payloads):
        return None
    from megatoken.tensors import byte_tensor

    values, lengths = byte_tensor(payloads, core.initial_facets.device)
    encoded = core.encode_control(values, lengths=lengths, encoder_grad=encoder_grad)
    return encoded.masked_fill(lengths.eq(0)[:, None], 0)
