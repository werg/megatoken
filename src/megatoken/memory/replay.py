"""Byte-exact selected-query replay; native P policy density is never inferred from text.

A query is its own real raw-facet meridian. Teacher vectors and source bytes are
not a substitute for the selected latent sample; immutable tape bytes are used
to reconstruct the native C/KV transition and evaluate P's action likelihood.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
import torch
from torch import Tensor

from megatoken.generation.embedding import EmbeddingSpec
from megatoken.model.core import ROLE_IDS
from megatoken.model.multitrack import (
    MultiTrackGenerator,
    MultiTrackState,
    retrieval_control_payload,
)


@dataclass(frozen=True)
class QueryActionReceipt:
    schema: str
    event_id: str
    meridian: int
    model_version: int
    request_b64: str
    request_sha256: str
    purpose: str
    space_fingerprint: str
    raw_shape: tuple[int, int, int]
    raw_f32_b64: str
    vector_f32_b64: str
    effort: int
    snapshot_id: str

    def __post_init__(self):
        if self.schema != "megatoken-query-action-v1":
            raise ValueError("Query tape schema is incompatible")
        if not self.event_id or not self.purpose or not self.snapshot_id or (
            not self.space_fingerprint or self.meridian < 0 or self.model_version < 0
        ):
            raise ValueError("Query tape needs complete causal/model identity")
        if self.raw_shape[0] != 1 or min(self.raw_shape) < 1:
            raise ValueError("Query tape records exactly one selected complete facet event")
        request = self.request
        if not request or sha256(request).hexdigest() != self.request_sha256:
            raise ValueError("Replay request bytes differ from source identity")
        if len(self._decode(self.raw_f32_b64)) != 4 * math.prod(self.raw_shape):
            raise ValueError("Raw query tape is not complete FP32 facet geometry")
        if len(self._decode(self.vector_f32_b64)) % 4:
            raise ValueError("Query tape embedding must be FP32")
        self._raw_array()
        self._vector_array()

    @staticmethod
    def _decode(value: str) -> bytes:
        return base64.b64decode(value, validate=True)

    @property
    def request(self) -> bytes:
        return self._decode(self.request_b64)

    def _raw_array(self) -> np.ndarray:
        raw = np.frombuffer(self._decode(self.raw_f32_b64), dtype="<f4").copy()
        if not np.isfinite(raw).all():
            raise ValueError("Query action contains nonfinite raw facets")
        return raw.reshape(self.raw_shape)

    def _vector_array(self) -> np.ndarray:
        vector = np.frombuffer(self._decode(self.vector_f32_b64), dtype="<f4").copy()
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError("Query vector is empty or nonfinite")
        return vector

    @classmethod
    def capture(cls, *, event_id: str, meridian: int, model_version: int,
                request: bytes, purpose: str, space_fingerprint: str,
                selected: Tensor, vector: Tensor, effort: int,
                snapshot_id: str) -> QueryActionReceipt:
        if selected.ndim != 4 or selected.shape[0:2] != (1, 1) or (
            vector.ndim != 2 or vector.shape[0] != 1
        ):
            raise ValueError("Replay tape requires one batched selected query event")
        raw = selected[0].detach().float().contiguous().cpu().numpy().astype("<f4")
        value = vector[0].detach().float().contiguous().cpu().numpy().astype("<f4")
        return cls(
            "megatoken-query-action-v1", event_id, meridian, model_version,
            base64.b64encode(request).decode("ascii"), sha256(request).hexdigest(),
            purpose, space_fingerprint, tuple(raw.shape),
            base64.b64encode(raw.tobytes()).decode("ascii"),
            base64.b64encode(value.tobytes()).decode("ascii"),
            effort, snapshot_id,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, item: dict) -> QueryActionReceipt:
        return cls(**{**item, "raw_shape": tuple(item["raw_shape"])})

    @classmethod
    def from_json(cls, serialized: str) -> QueryActionReceipt:
        return cls.from_dict(json.loads(serialized))


def replay_query_action(
    generator: MultiTrackGenerator, state: MultiTrackState,
    tape: QueryActionReceipt, *, atol: float = 1e-5, rtol: float = 1e-5,
) -> tuple[MultiTrackState, Tensor, Tensor]:
    """Return (native successor, selected semantic query, log P(raw|causal state)).

    Training can differentiate the native P log density with respect to the
    current model; replay never re-samples or substitutes E(document) facets.
    Exact raw bytes and query vector are verified against the pinned readout.
    """
    if state.batch_size != 1 or int(state.meridians_seen[0]) != tape.meridian:
        raise ValueError("Query replay does not match native meridian chronology")
    if generator.generator.core.model_version != tape.model_version:
        raise ValueError("Replay requires the recorded query model version")
    core = generator.generator.core
    if tape.raw_shape != (1, core.config.n_facets, core.config.facet_dim):
        raise ValueError("Query replay uses incompatible facet geometry")
    realizer = generator.generator.embedding_realizer
    if realizer is None or realizer.dimension != tape._vector_array().size:
        raise ValueError("Missing or incompatible embedding realizer")
    device = state.device
    selected = torch.from_numpy(tape._raw_array()).to(device)[None]
    control = generator._controls((
        retrieval_control_payload(
            main_request=tape.request, purpose=tape.purpose,
            space_fingerprint=tape.space_fingerprint,
        ),
    ), device)
    snapshot = generator.tracks.materialize_memory(state)
    mean = generator.tracks.predict_group(
        state, tracks=("__retrieval_query__",), control=control,
        effort=tape.effort, memory_snapshot=snapshot,
    )
    prior = generator.generator.facet_distribution(mean)
    log_density = prior.log_prob(selected).sum()
    vector = realizer(
        selected[:, 0], EmbeddingSpec(
            tape.space_fingerprint, "query", realizer.dimension,
        ),
    )
    expected = torch.from_numpy(tape._vector_array()).to(device)
    if not torch.allclose(vector[0].detach().float(), expected, atol=atol, rtol=rtol):
        raise ValueError("Selected retrieval vector no longer matches frozen readout")
    next_state = generator.tracks.advance_group(
        state, selected, tracks=("__retrieval_query__",),
        role_ids=torch.full(
            (1, 1), ROLE_IDS["assistant"], device=device, dtype=torch.long,
        ),
        memory_snapshot=snapshot,
    )
    return next_state, vector, log_density
