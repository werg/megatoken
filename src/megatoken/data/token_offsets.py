"""Where each codec token starts in the bytes it spells, checked once and cheaply.

The chunk planner and the record encoder both need every token's byte span. Looking a
token up costs a dictionary read when its length is remembered per codec, so the
check that the tokens spell exactly the payload (one ``decode``) is the only per-token
work that touches the codec.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np

if TYPE_CHECKING:
    from .schema import Codec

_LENGTHS: WeakKeyDictionary[Codec, dict[int, int]] = WeakKeyDictionary()


def token_edges(payload: bytes, ids: Sequence[int], codec: Codec) -> np.ndarray:
    """Byte offsets of every token edge: ``edges[i]`` starts token ``i``, ``edges[-1]`` is the end.

    Raises unless the codec's tokens are nonempty and spell ``payload`` byte for byte,
    which also proves each token sits exactly at its edge.
    """
    lengths = _LENGTHS.setdefault(codec, {})
    try:
        sizes = [lengths[token] for token in ids]
    except KeyError:
        for token in set(ids).difference(lengths):
            piece = codec.decode([token])
            if not isinstance(piece, bytes) or not piece:
                raise ValueError("The raw codec must have nonempty byte-exact tokens") from None
            lengths[token] = len(piece)
        sizes = [lengths[token] for token in ids]
    edges = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=edges[1:])
    if edges[-1] != len(payload) or codec.decode(ids) != payload:
        raise ValueError("The raw codec must preserve every source byte compositionally")
    return edges
