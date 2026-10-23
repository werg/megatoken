"""Byte/ngram ingestion through a shared hierarchy of ordered pair reductions.

The stride frontend covers all source bytes. Each reduction halves positions
and doubles their aggregation span; local attention uses a fixed-size window.
Only a record's final at-most-n_facets positions enter facet-query attention.
There is no quadratic attention over the full byte or stride sequence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import AttentionBranch
from .blocks import DenseFacetBlock
from .config import ModelConfig
from .expert_pools import current_adapter_weights, expert_scope
from .frontend import ByteFrontend, SourceFeatures
from .layers import RMSNorm, SwiGLU, position_features


class PairReduction(nn.Module):
    """One reusable ordered 2:1 merge followed by bounded local processing."""

    def __init__(self, config: ModelConfig, window_size: int = 8) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError("The ingestion attention window must be positive")
        dim = config.source_dim
        self.window_size = window_size
        self.pair_norm = RMSNorm(2 * dim, config.norm_eps)
        self.ordered_merge = nn.Linear(2 * dim, dim, bias=False)
        self.occupancy = nn.Linear(2, dim, bias=False)
        self.level_projection = nn.Linear(dim, dim, bias=False)
        self.attention = AttentionBranch(
            dim,
            dim,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
        )
        self.ffn_norm = RMSNorm(dim, config.norm_eps)
        self.ffn = SwiGLU(dim, 2 * dim)
        self.output_norm = RMSNorm(dim, config.norm_eps)
        self.residual_gates = nn.Parameter(torch.full((3,), -2.0))

    def forward(self, features: Tensor, mask: Tensor, *, level: int) -> tuple[Tensor, Tensor]:
        batch, count, dim = features.shape
        pair_padding = count % 2
        pairs = F.pad(features * mask[..., None], (0, 0, 0, pair_padding)).reshape(
            batch, -1, 2, dim
        )
        pair_mask = F.pad(mask, (0, pair_padding)).reshape(batch, -1, 2)
        output_mask = pair_mask.any(-1)
        # An odd final position is an occupied left child, not a fabricated
        # zero-byte right child. Concatenation preserves the two child roles.
        mean = pairs.float().sum(-2) / pair_mask.sum(-1, keepdim=True).clamp_min(1)
        ordered = self.ordered_merge(self.pair_norm(pairs.flatten(-2)))
        occupancy = self.occupancy(pair_mask.to(features.dtype))
        scale = position_features(1, dim, features.device, offset=level).to(features.dtype)
        scale = scale[None].expand(batch, -1, -1)
        value = mean.to(features.dtype) + self.residual_gates[0].sigmoid() * (
            ordered + occupancy + self.level_projection(scale)
        )
        value = value * output_mask[..., None]

        reduced_count = value.shape[1]
        padding = (-reduced_count) % self.window_size
        windows = F.pad(value, (0, 0, 0, padding)).reshape(-1, self.window_size, dim)
        window_mask = F.pad(output_mask, (0, padding)).reshape(-1, self.window_size)
        weights = current_adapter_weights(features[:, :1])
        window_weights = None if weights is None else weights[:, 0].repeat_interleave(
            windows.shape[0] // batch, dim=0
        )
        # Window flattening is explicit so a neighboring session's mixture can
        # never be mistaken for another window in this session.
        with expert_scope(window_weights):
            windows = windows + self.residual_gates[1].sigmoid() * self.attention(
                windows, windows, window_mask
            )
            windows = windows + self.residual_gates[2].sigmoid() * self.ffn(self.ffn_norm(windows))
        value = self.output_norm(windows).reshape(batch, -1, dim)[:, :reduced_count]
        return value * output_mask[..., None], output_mask


class ChunkEncoder(ByteFrontend):
    """Encode one byte-exact chunk as [batch, facets, facet_dim].

    Subclassing the stride frontend keeps a single native n-gram parameter owner
    at ``core.frontend.ngrams``. ``source_features`` exposes those causal stride
    features to the independent seed-control encoder.

    A batch row is read out at its first level with at most n_facets occupied
    positions. Later reductions required by longer rows cannot change it. This
    makes source padding and unrelated batch membership semantically inert.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.reduction = PairReduction(config)
        self.facet_queries = nn.Parameter(torch.empty(config.n_facets, config.facet_dim))
        nn.init.normal_(self.facet_queries, std=0.02)
        self.aligned_projection = nn.Linear(config.source_dim, config.facet_dim, bias=False)
        self.facet_attention = AttentionBranch(
            config.facet_dim,
            config.source_dim,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
        )
        self.facet_ffn_norm = RMSNorm(config.facet_dim, config.norm_eps)
        self.facet_ffn = SwiGLU(config.facet_dim, 2 * config.facet_dim)
        self.refinement = nn.ModuleList(
            DenseFacetBlock(config, config.encoder_hidden) for _ in range(config.encoder_layers)
        )
        # Canonical facets share the recurrent model's unconstrained coordinate
        # space. Internal pre-norms stabilize processing; the output has no
        # compulsory RMS/unit-length constraint.
        self.output_projection = nn.Linear(config.facet_dim, config.facet_dim, bias=False)
        self.facet_gates = nn.Parameter(torch.full((2,), -2.0))

    def source_features(self, byte_ids: Tensor, lengths: Tensor | None = None) -> SourceFeatures:
        return super().forward(byte_ids, lengths)

    def _readout(self, features: Tensor, mask: Tensor) -> Tensor:
        count = min(features.shape[1], self.config.n_facets)
        source = F.pad(features[:, :count], (0, 0, 0, self.config.n_facets - count))
        source_mask = F.pad(mask[:, :count], (0, self.config.n_facets - count))
        source = source * source_mask[..., None]
        queries = self.facet_queries.to(source.dtype)[None].expand(source.shape[0], -1, -1)
        # The aligned path retains ordered source identities while bounded query
        # attention fills unused slots and shares information across all facets.
        facets = queries + self.aligned_projection(source)
        facets = facets + self.facet_gates[0].sigmoid() * self.facet_attention(
            facets, source, source_mask
        )
        facets = facets + self.facet_gates[1].sigmoid() * self.facet_ffn(
            self.facet_ffn_norm(facets)
        )
        return facets

    def _finish(self, facets: Tensor) -> Tensor:
        """Refine once at facet resolution, then expose the shared input format."""
        for block in self.refinement:
            facets = block(facets)
        return self.output_projection(facets)

    def forward(self, byte_ids: Tensor, lengths: Tensor | None = None) -> Tensor:
        source = self.source_features(byte_ids, lengths)
        features, mask = source.features, source.mask
        # With no lengths argument every row has the same real byte length.
        # Its finishing level follows from static shapes: skip readouts that
        # would be discarded without adding device-to-host synchronization.
        # Ragged batches keep the per-row first-ready semantics below.
        if lengths is None:
            level = 0
            while features.shape[1] > self.config.n_facets:
                level += 1
                features, mask = self.reduction(features, mask, level=level)
            return self._finish(self._readout(features, mask))
        completed = torch.zeros(features.shape[0], device=features.device, dtype=torch.bool)
        facets = features.new_zeros(
            (features.shape[0], self.config.n_facets, self.config.facet_dim)
        )
        level = 0
        while True:
            ready = ~completed & (mask.sum(-1) <= self.config.n_facets)
            candidate = self._readout(features, mask)
            facets = torch.where(ready[:, None, None], candidate, facets)
            completed = completed | ready
            if features.shape[1] <= self.config.n_facets:
                return self._finish(facets)
            level += 1
            features, mask = self.reduction(features, mask, level=level)

