"""Capacity-first parallel lexical realization with shared U expert weights."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def positional_features(length: int, width: int, *, device, dtype) -> Tensor:
    half = (width + 1) // 2
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1)
    )
    phase = torch.arange(length, device=device, dtype=torch.float32)[:, None] * frequency[None]
    return torch.cat((phase.sin(), phase.cos()), -1)[:, :width].to(dtype)


class CrossAttention(nn.Module):
    def __init__(self, width: int, memory_width: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(memory_width, width, bias=False)
        self.v = nn.Linear(memory_width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm = nn.RMSNorm(width)

    def forward(self, query: Tensor, memory: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, length, width = query.shape
        if not memory.shape[1]:
            return torch.zeros_like(query)

        def split(value):
            return value.reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)

        q, k, v = split(self.q(self.norm(query))), split(self.k(memory)), split(self.v(memory))
        attention_mask = None if mask is None else mask[:, None, None, :]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        return self.out(output.transpose(1, 2).reshape(batch, length, width))


class ParallelRealizer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        width = config.facet_dim
        self.width = width
        heads = max(head for head in range(1, config.query_heads + 1) if width % head == 0)
        self.seed_count = min(4, config.max_capacity)
        self.seed = nn.Linear(width, width, bias=False)
        self.seed_codes = nn.Parameter(torch.randn(self.seed_count, width) / math.sqrt(width))
        self.split = nn.Linear(width, 2 * width, bias=False)
        self.growth_norm = nn.RMSNorm(width)
        self.growth_attention = CrossAttention(width, width, heads)
        self.plan_attention = CrossAttention(width, width, heads)
        self.source_attention = CrossAttention(width, config.source_dim, heads)
        self.local_attention = CrossAttention(width, width, heads)
        self.global_attention = CrossAttention(width, width, heads)
        self.repair_adapter = nn.Linear(width, width, bias=False)
        self.final_norm = nn.RMSNorm(width)
        self.local_window = 64

    def begin(self, core, prediction, capacity: int) -> Tensor:
        """Stage 8: grow to scale 64 before exact-length information exists."""
        rows = self.seed(prediction.facets.mean(1))[:, None] + self.seed_codes[None]
        if capacity < rows.shape[1]:
            rows = rows[:, :capacity]
        return self.grow(core, prediction, rows, min(capacity, 64))

    def grow(self, core, prediction, rows: Tensor, capacity: int) -> Tensor:
        while rows.shape[1] < capacity:
            previous = rows.shape[1]
            children = self.split(self.growth_norm(rows)).reshape(
                rows.shape[0], previous, 2, self.width
            )
            rows = (rows[:, :, None] + children).reshape(rows.shape[0], 2 * previous, self.width)
            rows = rows[:, :capacity]
            rows = (
                rows
                + positional_features(
                    rows.shape[1], self.width, device=rows.device, dtype=rows.dtype
                )[None]
            )
            rows = rows + self.growth_attention(rows, rows)
            rows = core.apply_u(self.growth_norm(rows), prediction, level=rows.shape[1])
        return rows

    def finish_growth(
        self, core, prediction, rows: Tensor, capacity: int, conditioning: Tensor
    ) -> Tensor:
        """Actual stage-10 ancestors can affect future growth, never prior work."""
        rows = rows + conditioning[:, None]
        return self.grow(core, prediction, rows, capacity)

    def refine(self, core, prediction, rows: Tensor) -> Tensor:
        # Dense attention stays block-local; a second, separately normalized
        # branch sees pooled global groups. Neither uses lexical target tokens.
        batch, length, width = rows.shape
        window = min(self.local_window, length)
        padded = math.ceil(length / window) * window
        padding = padded - length
        local = F.pad(rows, (0, 0, 0, padding)).reshape(batch * (padded // window), window, width)
        mask = torch.arange(padded, device=rows.device)[None] < length
        mask = mask.expand(batch, -1).reshape(batch * (padded // window), window)
        local_out = self.local_attention(local, local, mask).reshape(batch, padded, width)[
            :, :length
        ]
        weighted = local * mask[..., None]
        pooled = (weighted.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)).reshape(
            batch, padded // window, width
        )
        rows = rows + local_out + self.global_attention(rows, pooled)
        rows = rows + self.plan_attention(rows, prediction.facets)
        rows = rows + self.source_attention(rows, prediction.source, prediction.source_mask)
        # This is the same dense U always branch used throughout growth/core.
        return core.lexical_refine(self.final_norm(rows))

    def repair(self, core, prediction, rows: Tensor, draft_embeddings: Tensor) -> Tensor:
        if rows.shape != draft_embeddings.shape:
            raise ValueError("repair embeddings must cover the complete capacity workspace")
        repaired = rows + self.repair_adapter(draft_embeddings)
        repaired = repaired + self.growth_attention(repaired, repaired)
        return self.refine(core, prediction, repaired)
