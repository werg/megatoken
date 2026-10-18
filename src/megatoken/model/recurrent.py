"""Independent record-time recurrent memory; prediction can only read it."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.kernels.recurrent import delta_rule_read, delta_rule_step

from .config import ModelConfig
from .layers import RMSNorm


class RecordGDN2(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.pool_queries = nn.Parameter(torch.empty(config.gdn_streams, config.facet_dim))
        nn.init.normal_(self.pool_queries, std=0.02)
        hd = config.gdn_heads * config.gdn_head_dim
        self.norm = RMSNorm(config.facet_dim, config.norm_eps)
        self.qkv = nn.Linear(config.facet_dim, 3 * hd, bias=False)
        self.gates = nn.Linear(config.facet_dim, 2 * config.gdn_heads, bias=True)
        self.output_gate = nn.Linear(config.facet_dim, hd, bias=False)
        self.output = nn.Linear(hd, config.facet_dim, bias=False)
        self.read_norm = RMSNorm(hd, config.norm_eps)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        with torch.no_grad():
            self.gates.bias[: config.gdn_heads].fill_(3.0)  # initially slow forgetting
            self.gates.bias[config.gdn_heads :].fill_(-1.0)

    def initial_state(self, batch: int, device: torch.device) -> Tensor:
        c = self.config
        return torch.zeros(
            batch,
            c.gdn_streams,
            c.gdn_heads,
            c.gdn_head_dim,
            c.gdn_head_dim,
            device=device,
            dtype=torch.float32,
        )

    def forward(self, facets: Tensor, state: Tensor, *, write: bool) -> tuple[Tensor, Tensor]:
        c = self.config
        normalized = self.norm(facets)
        with torch.autocast(device_type=facets.device.type, enabled=False):
            weights = torch.einsum(
                "sd,bfd->bsf", self.pool_queries.float(), normalized.float()
            ) / math.sqrt(c.facet_dim)
            weights = weights.softmax(-1)
        pooled = torch.einsum("bsf,bfd->bsd", weights.to(facets.dtype), normalized)
        q, k, v = (
            self.qkv(pooled).reshape(*pooled.shape[:2], 3, c.gdn_heads, c.gdn_head_dim).unbind(2)
        )
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        if write:
            decay_logit, beta_logit = self.gates(pooled).float().chunk(2, -1)
            decay = decay_logit.sigmoid().clamp(1e-4, 1 - 1e-4)
            beta = beta_logit.sigmoid()
            next_state, reads = delta_rule_step(state, q, k, v, decay, beta, c.recurrent_backend)
        else:
            next_state, reads = state, delta_rule_read(state, q)
        reads = reads.flatten(-2).to(facets.dtype)
        updates = self.output(self.read_norm(reads) * self.output_gate(pooled).sigmoid())
        # Normalize each facet's aggregate, with stream pooling independent of
        # temporal recurrence. No stream is accidentally scanned as time.
        distribute = weights.transpose(1, 2)
        distribute = distribute / distribute.sum(-1, keepdim=True).clamp_min(1e-6)
        delta = torch.einsum("bfs,bsd->bfd", distribute.to(updates.dtype), updates)
        return facets + self.residual_gate.sigmoid() * delta, next_state
