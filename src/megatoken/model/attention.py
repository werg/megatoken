"""Separate normalization for historical A, historical B, live, and source reads."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .layers import RMSNorm
from .state import FactualState, KVMemory


class AttentionBranch(nn.Module):
    def __init__(
        self,
        query_dim: int,
        source_dim: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.query_heads, self.kv_heads, self.head_dim = query_heads, kv_heads, head_dim
        self.query_norm = RMSNorm(query_dim, eps)
        self.source_norm = RMSNorm(source_dim, eps)
        self.query = nn.Linear(query_dim, query_heads * head_dim, bias=False)
        self.key_value = nn.Linear(source_dim, 2 * kv_heads * head_dim, bias=False)
        self.output = nn.Linear(query_heads * head_dim, query_dim, bias=False)

    def encode(self, source: Tensor, mask: Tensor) -> KVMemory:
        b, n, _ = source.shape
        kv = self.key_value(self.source_norm(source)).reshape(b, n, 2, self.kv_heads, self.head_dim)
        key, value = kv.unbind(2)
        return KVMemory(key.transpose(1, 2), value.transpose(1, 2), mask)

    def read(self, query: Tensor, memory: KVMemory) -> Tensor:
        if not memory.keys.shape[2]:
            return torch.zeros_like(query)
        b, n, _ = query.shape
        q = (
            self.query(self.query_norm(query))
            .reshape(b, n, self.query_heads, self.head_dim)
            .transpose(1, 2)
        )
        k, v = memory.keys.to(q.dtype), memory.values.to(q.dtype)
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=memory.mask[:, None, None, :],
            dropout_p=0.0,
            enable_gqa=self.query_heads != self.kv_heads,
        )
        return self.output(result.transpose(1, 2).reshape(b, n, -1))

    def forward(self, query: Tensor, source: Tensor, mask: Tensor) -> Tensor:
        return self.read(query, self.encode(source, mask))


class PooledWriter(nn.Module):
    """Create a fixed number of factual slots once per committed record."""

    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(slots, dim))
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, source: Tensor, mask: Tensor | None = None) -> Tensor:
        with torch.autocast(device_type=source.device.type, enabled=False):
            logits = torch.einsum("sd,bnd->bsn", self.queries.float(), source.float()) / math.sqrt(
                source.shape[-1]
            )
            if mask is not None:
                logits = logits.masked_fill(~mask[:, None], -torch.inf)
                nonempty = mask.any(-1)[:, None, None]
                logits = torch.where(nonempty, logits, torch.zeros_like(logits))
            weights = logits.softmax(-1)
            if mask is not None:
                weights = weights * nonempty
        return torch.einsum("bsn,bnd->bsd", weights.to(source.dtype), source)


def append_memory(old: KVMemory, new: KVMemory, max_slots: int) -> KVMemory:
    return KVMemory(
        torch.cat((old.keys, new.keys), dim=2)[:, :, -max_slots:],
        torch.cat((old.values, new.values), dim=2)[:, :, -max_slots:],
        torch.cat((old.mask, new.mask), dim=1)[:, -max_slots:],
    )


class MemoryInterface(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.history_a = AttentionBranch(
            c.facet_dim,
            c.source_dim,
            c.query_heads,
            c.historical_kv_heads,
            c.attention_head_dim,
            c.norm_eps,
        )
        self.history_b = AttentionBranch(
            c.facet_dim,
            c.facet_dim,
            c.query_heads,
            c.historical_kv_heads,
            c.attention_head_dim,
            c.norm_eps,
        )
        self.live = AttentionBranch(
            c.facet_dim,
            c.facet_dim,
            c.query_heads,
            c.live_kv_heads,
            c.attention_head_dim,
            c.norm_eps,
        )
        self.literal = AttentionBranch(
            c.facet_dim,
            c.source_dim,
            c.query_heads,
            c.live_kv_heads,
            c.attention_head_dim,
            c.norm_eps,
        )
        self.pool_a = PooledWriter(c.source_dim, c.historical_slots)
        self.pool_b = PooledWriter(c.facet_dim, c.historical_slots)
        self.read_gates = nn.Parameter(torch.full((4,), -2.0))

    def forward(self, facets: Tensor, state: FactualState) -> Tensor:
        mask = torch.ones(facets.shape[:2], device=facets.device, dtype=torch.bool)
        # Normalization belongs to each branch independently, including live KV.
        values = (
            self.history_a.read(facets, state.history_a),
            self.history_b.read(facets, state.history_b),
            self.live(facets, facets, mask),
            self.literal.read(facets, state.literal),
        )
        return facets + sum(
            g.sigmoid() * value for g, value in zip(self.read_gates, values, strict=True)
        )

    def write(
        self, source: Tensor, source_mask: Tensor, facets: Tensor, state: FactualState
    ) -> tuple[KVMemory, KVMemory, KVMemory]:
        slots = self.config.historical_slots
        mask = torch.ones(source.shape[0], slots, device=source.device, dtype=torch.bool)
        a = self.history_a.encode(self.pool_a(source, source_mask), mask)
        b = self.history_b.encode(self.pool_b(facets), mask)
        literal = self.literal.encode(source, source_mask)
        maximum = self.config.max_history_records * slots
        return (
            append_memory(state.history_a, a, maximum),
            append_memory(state.history_b, b, maximum),
            append_memory(state.literal, literal, self.config.literal_window),
        )
