"""Deterministic parallel realization of raw canonical facets using shared U."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.model.attention import AttentionBranch
from megatoken.model.layers import RMSNorm, position_features


class ParallelRealizer(nn.Module):
    """Shared growth, block attention and one dense lexical refinement.

    All learned inputs descend from the raw canonical facets. The conditioning
    vector contains deterministic decoded structure and public copied values;
    recurrent factual state and control are deliberately absent from this API.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.width = width = config.facet_dim
        self.seed_count = min(4, config.max_capacity)
        self.seed = nn.Linear(width, width, bias=False)
        self.seed_codes = nn.Parameter(torch.randn(self.seed_count, width) / math.sqrt(width))
        self.split = nn.Linear(width, 2 * width, bias=False)
        self.growth_norm = RMSNorm(width, config.norm_eps)
        self.growth_attention = AttentionBranch(
            width,
            width,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
        )
        self.plan_attention = AttentionBranch(
            width,
            width,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
        )
        self.local_attention = AttentionBranch(
            width,
            width,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
        )
        self.global_attention = AttentionBranch(
            width,
            width,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
        )
        self.final_norm = RMSNorm(width, config.norm_eps)
        self.local_window = 64

    def forward(self, core, raw_facets: Tensor, capacity: int, conditioning: Tensor) -> Tensor:
        if not 1 <= capacity <= core.config.max_capacity:
            raise ValueError("realization workspace exceeds the configured output limit")
        context = core.prepare_realization(raw_facets)
        raw = raw_facets.to(self.seed.weight.dtype)
        rows = self.seed(raw.mean(1))[:, None] + self.seed_codes[None]
        rows = (rows + conditioning[:, None])[:, :capacity]
        while rows.shape[1] < capacity:
            previous = rows.shape[1]
            children = self.split(self.growth_norm(rows)).reshape(
                rows.shape[0], previous, 2, self.width
            )
            rows = (rows[:, :, None] + children).reshape(rows.shape[0], 2 * previous, self.width)
            rows = rows[:, :capacity]
            rows = (
                rows
                + position_features(rows.shape[1], self.width, rows.device).to(rows.dtype)[None]
            )
            rows = rows + self.growth_attention(rows, rows)
            rows = core.apply_u(self.growth_norm(rows), context, level=rows.shape[1])
        return self.refine(core, context, rows)

    def refine(self, core, context, rows: Tensor) -> Tensor:
        batch, length, width = rows.shape
        window = min(self.local_window, length)
        padded = math.ceil(length / window) * window
        local = F.pad(rows, (0, 0, 0, padded - length)).reshape(
            batch * (padded // window), window, width
        )
        mask = (torch.arange(padded, device=rows.device)[None] < length).expand(batch, -1)
        mask = mask.reshape(batch * (padded // window), window)
        local_out = self.local_attention(local, local, None if padded == length else mask).reshape(
            batch, padded, width
        )[:, :length]
        pooled = (
            (local * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        ).reshape(batch, padded // window, width)
        rows = rows + local_out + self.global_attention(rows, pooled)
        rows = rows + self.plan_attention(rows, context.facets.to(rows.dtype))
        return core.lexical_refine(self.final_norm(rows))
