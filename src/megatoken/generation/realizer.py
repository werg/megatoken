"""Progressive realization directly from canonical raw facets."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.model.attention import AttentionBranch
from megatoken.model.layers import RMSNorm, position_features


class ParallelRealizer(nn.Module):
    """Keep all facet identities without reading causal core or KV history."""

    def __init__(self, config) -> None:
        super().__init__()
        self.width = width = config.facet_dim
        self.seed_count = config.n_facets
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

    def forward(self, core, context, capacity: int, conditioning: Tensor) -> Tensor:
        # Decoder entry points already install a retrieval-conditioned R
        # routing epoch. Keep direct realizer use (debugging and small training
        # utilities) valid while ensuring every routed site in this call shares
        # one context and palette.
        with core.expert_pools.routing_epoch_if_absent(
            context.facets.mean(1), role="R", stage=0
        ):
            return self._forward_impl(core, context, capacity, conditioning)

    def _forward_impl(self, core, context, capacity: int, conditioning: Tensor) -> Tensor:
        if not 1 <= capacity <= core.config.max_capacity:
            raise ValueError("realization workspace exceeds the configured output limit")
        raw = context.facets.to(self.seed.weight.dtype)
        if raw.shape[1] != self.seed_count:
            raise ValueError("realization requires every configured raw facet slot")
        rows = self.seed(raw) + self.seed_codes[None] + conditioning[:, None]
        if capacity < self.seed_count:
            # Short chunks still use all slots, rather than dropping the suffix.
            rows = F.adaptive_avg_pool1d(rows.transpose(1, 2), capacity).transpose(1, 2)
        rows = core.apply_u(self.growth_norm(rows), context, level=rows.shape[1])
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
        # Each sequence can yield several independent local windows. Preserve
        # its event route and adapter mixture for every flattened window.
        window_contexts = torch.arange(batch, device=rows.device).repeat_interleave(
            padded // window
        )
        with core.expert_pools.routing_subset(window_contexts):
            local_out = self.local_attention(
                local, local, None if padded == length else mask
            ).reshape(batch, padded, width)[:, :length]
        pooled = (
            (local * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        ).reshape(batch, padded // window, width)
        rows = rows + local_out + self.global_attention(rows, pooled)
        identities = self.seed_codes[None].to(rows.dtype)
        rows = rows + self.plan_attention(rows, context.facets.to(rows.dtype) + identities)
        return core.lexical_refine(self.final_norm(rows))
