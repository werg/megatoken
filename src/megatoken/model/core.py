"""Shared raw facets feed direct realization and causal prediction separately.

E ingests to raw facets, P predicts raw facets, and R realizes them without C.
C processes committed facets for *future* predictions and writes causal KV.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from megatoken.contracts import ROLE_TO_ID
from megatoken.tensors import parameter_dtype

from .adapters import AdapterBank, AdapterPlan
from .attention import AttentionFeatures, AttentionRead, KVProjection, join_attention
from .blocks import DenseFacetBlock
from .config import ModelConfig
from .expert_pools import (
    ShapePools,
    checkpoint_context_fn,
    current_adapter_weights,
    expert_scope,
)
from .experts import ExpertBank
from .frontend import SeedControlEncoder
from .ingestion import ChunkEncoder
from .layers import RMSNorm, SwiGLU
from .media import MediaEncoder
from .routing import EpochRouter, RowRoute
from .state import CoreState, KVHistory, RealizationContext, assert_cache_version

ROLE_IDS = ROLE_TO_ID
RECORD_ROLES = tuple(ROLE_TO_ID)
EFFORTS = (1, 2, 3, 6)
P_HISTORY_READ_STAGES = (3, 6, 9, 12)
MODALITY_IDS = {"text": 0, "audio": 1, "image": 2, "video": 3, "compressed_text": 4}


def _byte_routing_summary(byte_ids: Tensor, lengths: Tensor | None) -> Tensor:
    """Cheap input-only routing context, including a finite empty-record value."""
    if byte_ids.ndim != 2:
        raise ValueError("A physical record batch must have shape [batch,bytes]")
    if byte_ids.dtype not in (torch.uint8, torch.int32, torch.int64):
        raise TypeError("byte_ids must be integer native bytes")
    if byte_ids.shape[0] < 1:
        raise ValueError("A physical record batch must contain at least one record")
    values = byte_ids.float()
    if lengths is None:
        return values.sum(-1, keepdim=True) / max(1, values.shape[1]) / 255.0
    if lengths.dtype not in (torch.int32, torch.int64, torch.uint8):
        raise TypeError("Record lengths must be integer byte counts")
    if lengths.shape != (values.shape[0],):
        raise ValueError("Every record length must be in [0, padded byte length]")
    lengths = lengths.to(device=values.device, dtype=torch.long)
    valid = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
    return (values * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1) / 255.0


class PairStage(nn.Module):
    """Local routes and residual gates over globally shared pair expert weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = AttentionRead(
            config.facet_dim,
            config.query_heads,
            config.kv_heads,
            config.attention_head_dim,
            config.norm_eps,
            track_relations=True,
            sink=True,
        )
        self.attention_gate = nn.Parameter(torch.tensor(-2.0))
        self.norm = RMSNorm(config.pair_dim, config.norm_eps)
        self.always = SwiGLU(config.pair_dim, config.p_always_hidden)
        self.residual_gates = nn.Parameter(torch.full((2,), -2.0))

    def update(
        self,
        facets: Tensor,
        left: Tensor,
        right: Tensor,
        route: RowRoute,
        bank: ExpertBank,
        update_index: int,
        reduction_indices: Tensor,
        reduction_mask: Tensor,
        reduction_degree: Tensor,
    ) -> Tensor:
        dim = facets.shape[-1]
        pairs = self.norm(torch.cat((facets[:, left], facets[:, right]), -1))
        delta = bank(
            pairs,
            route.expert_ids,
            route.confidence,
            adapter_weights=current_adapter_weights(pairs),
        ) + self.always(pairs)
        delta = delta * self.residual_gates[update_index].sigmoid()
        left_delta, right_delta = delta.split(dim, -1)
        contributions = torch.cat((left_delta, right_delta), dim=1)
        ordered = contributions[:, reduction_indices] * reduction_mask[None, :, :, None]
        # Unlike repeated-destination CUDA index_add, this fixed reduction order
        # does not depend on atomic scheduling during sampled/replayed policies.
        accumulated = ordered.sum(2)
        return facets + accumulated / reduction_degree[None, :, None]


class FacetRemix(nn.Module):
    """The single internal boundary without an identity/residual shortcut.

    A learned channel transform and learned slot transform form a fresh seed.
    Subsequent residual stages build the actual next-input prediction.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.facet_dim, config.norm_eps)
        self.channels = SwiGLU(config.facet_dim, config.remix_hidden)
        self.slots = nn.Linear(config.n_facets, config.n_facets, bias=False)
        nn.init.orthogonal_(self.slots.weight)

    def forward(self, facets: Tensor) -> Tensor:
        channels = self.channels(self.norm(facets))
        return self.slots(channels.transpose(1, 2)).transpose(1, 2)


class CentralProcessor(nn.Module):
    """Expert-routed causal C with temporary KV writers and one final writer.

    Routes depend on committed input only. Its derived features may be cached
    until model weights change. The raw input remains intact outside this module.
    """

    def __init__(self, config: ModelConfig, pools: ShapePools) -> None:
        super().__init__()
        self.config = c = config
        object.__setattr__(self, "_expert_pools", pools)
        # C1-C3 write intermediate KV read only by the next central block.
        # C4's separate projection alone owns the cross-record cached KV.
        self.temporary_memories = nn.ModuleList(
            KVProjection(c.facet_dim, c.kv_heads, c.attention_head_dim, c.norm_eps)
            for _ in range(c.central_layers - 1)
        )
        self.memory = KVProjection(c.facet_dim, c.kv_heads, c.attention_head_dim, c.norm_eps)
        self.layers = nn.ModuleList(
            DenseFacetBlock(c, c.central_hidden, shared_kv=True, temporary_read=index > 0)
            for index in range(c.central_layers)
        )
        self.layer_codes = nn.Parameter(torch.empty(c.central_layers, c.facet_dim))
        self.slot_embedding = nn.Parameter(torch.empty(c.n_facets, c.facet_dim))
        self.slot_gate = nn.Parameter(torch.tensor(-2.0))
        self.role_embedding = nn.Embedding(len(RECORD_ROLES), c.facet_dim)
        self.modality_embedding = nn.Embedding(len(MODALITY_IDS), c.facet_dim)
        nn.init.normal_(self.layer_codes, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        nn.init.zeros_(self.modality_embedding.weight)

    def slot_codes(self) -> Tensor:
        """Stable learned facet addresses, independent of temporal RoPE phases."""
        return self.slot_gate.sigmoid() * self.slot_embedding

    def forward(
        self,
        raw: Tensor,
        history: KVHistory,
        roles: Tensor,
        positions: Tensor,
        *,
        checkpointing: bool | None = None,
        track_ids: Tensor | None = None,
        modality_ids: Tensor | None = None,
        history_track_ids: Tensor | None = None,
        track_semantics: Tensor | None = None,
        attention_mask: Tensor | None = None,
        adapters: AdapterPlan | None = None,
    ) -> Tensor:
        # A meridian may contain several complete 32-facet megatokens.  The
        # original one-event shape remains exactly the same, so singleton
        # inference and checkpoints retain their previous semantics.
        if raw.ndim != 3 or raw.shape[1] % self.config.n_facets:
            raise ValueError("Central inputs must contain complete facet blocks")
        events = raw.shape[1] // self.config.n_facets
        addresses = self.slot_codes().repeat(events, 1)
        if roles.shape == (raw.shape[0],):
            role_features = self.role_embedding(roles)[:, None]
        elif roles.shape == (raw.shape[0], events):
            role_features = self.role_embedding(roles).repeat_interleave(
                self.config.n_facets, dim=1
            )
        else:
            raise ValueError("Roles must identify the batch or each selected event")
        facets = raw.to(self.role_embedding.weight.dtype)
        facets = facets + role_features + addresses.to(facets.dtype)[None]
        if track_ids is not None and track_ids.shape != (raw.shape[0], events):
            raise ValueError("Track IDs must identify every event in the meridian")
        live_tracks = (
            None if track_ids is None else track_ids.repeat_interleave(self.config.n_facets, dim=1)
        )
        if history_track_ids is not None and history_track_ids.shape != (
            raw.shape[0],
            history.slots,
        ):
            raise ValueError("Historical track IDs must identify every retained facet")
        if modality_ids is not None:
            if modality_ids.shape != (raw.shape[0], events):
                raise ValueError("Modality IDs must identify every event in the meridian")
            modality_features = self.modality_embedding(modality_ids).repeat_interleave(
                self.config.n_facets, dim=1
            )
            facets = facets + modality_features
        projected = AttentionFeatures(history.keys, history.values, history.mask, history_track_ids)
        if attention_mask is not None and attention_mask.shape != (
            raw.shape[0],
            raw.shape[1],
            projected.keys.shape[2] + raw.shape[1],
        ):
            raise ValueError("C attention mask must address all native query/key facets")
        temporary = None
        for index, layer in enumerate(self.layers):
            facets = facets + self.layer_codes[index]
            projector = (
                self.temporary_memories[index]
                if index < len(self.temporary_memories)
                else self.memory
            )
            enabled = self.config.checkpoint_stages if checkpointing is None else checkpointing
            weights = (
                None
                if adapters is None
                else adapters.weights[index].repeat_interleave(self.config.n_facets, dim=1)
            )
            routing_context = facets.reshape(
                facets.shape[0], events, self.config.n_facets, self.config.facet_dim
            ).mean(2)
            with (
                self._expert_pools.routing_epoch(
                    routing_context,
                    track_ids=track_ids,
                    track_semantics=track_semantics,
                    role="c",
                    stage=index,
                    adapter_weights=None if adapters is None else adapters.weights[index],
                ),
                expert_scope(weights),
            ):
                if enabled and self.training and torch.is_grad_enabled():
                    # Pin the causal adapter scope as well as the intermediate KV.
                    facets = checkpoint(
                        lambda value, block=layer, writer=projector, previous=temporary: block(
                            value,
                            projected,
                            projector=writer,
                            temporary=previous,
                            temporal_positions=positions,
                            slot_codes=addresses,
                            query_track_ids=live_tracks,
                            query_key_mask=attention_mask,
                        ),
                        facets,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        context_fn=checkpoint_context_fn,
                    )
                else:
                    facets = layer(
                        facets,
                        projected,
                        projector=projector,
                        temporary=temporary,
                        temporal_positions=positions,
                        slot_codes=addresses,
                        query_track_ids=live_tracks,
                        query_key_mask=attention_mask,
                    )
                # Intermediate projections use the same committed adapter choice.
                if index < len(self.temporary_memories):
                    temporary = projector(
                        facets,
                        temporal_positions=positions,
                        slot_codes=addresses,
                        track_ids=live_tracks,
                    )
        return facets

    def write(
        self,
        final: Tensor,
        history: KVHistory,
        positions: Tensor,
        *,
        adapters: AdapterPlan | None = None,
        track_ids: Tensor | None = None,
        track_semantics: Tensor | None = None,
    ) -> KVHistory:
        """Append all final-stage facets; evict only complete oldest megatokens.

        Every retained record has exactly n_facets K/V slots for its lifetime.
        No pooling, age-dependent compression, or re-projection of older records
        occurs. Temporal RoPE is applied once, at the original record position.
        """
        c = self.config
        if final.ndim != 3 or final.shape[1] % c.n_facets:
            raise ValueError("KV writes require complete facet blocks")
        if final.shape[1] > c.max_history_chunks * c.n_facets:
            raise ValueError("One meridian exceeds the historical event capacity")
        events = final.shape[1] // c.n_facets
        weights = (
            None if adapters is None else adapters.weights[-1].repeat_interleave(c.n_facets, dim=1)
        )
        with (
            self._expert_pools.routing_epoch(
                final.reshape(final.shape[0], events, c.n_facets, c.facet_dim).mean(2),
                track_ids=track_ids,
                track_semantics=track_semantics,
                role="c",
                stage=c.central_layers,
                adapter_weights=None if adapters is None else adapters.weights[-1],
            ),
            expert_scope(weights),
        ):
            added = self.memory(
                final,
                temporal_positions=positions,
                slot_codes=self.slot_codes().repeat(events, 1).to(final.dtype),
            )
        max_slots = c.max_history_chunks * c.n_facets
        keys = torch.cat((history.keys, added.keys), dim=2)[:, :, -max_slots:]
        values = torch.cat((history.values, added.values), dim=2)[:, :, -max_slots:]
        mask = None
        if history.mask is not None:
            mask = torch.cat(
                (
                    history.mask,
                    torch.ones(
                        final.shape[0], final.shape[1], dtype=torch.bool, device=final.device
                    ),
                ),
                dim=1,
            )[:, -max_slots:]
        return KVHistory(keys, values, mask)


class FacetedCore(nn.Module):
    """E and P share canonical raw facets; R reads them independently of C.

    C still processes selected facets once and writes history for the next P.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.model_version = 0
        self.expert_pools = ShapePools(c)
        self.expert_pools.ffn_bank(c.facet_dim)
        self.expert_pools.ffn_bank(c.pair_dim)
        self.frontend = ChunkEncoder(c)
        self.media_encoder = MediaEncoder(c)
        self.control_encoder = SeedControlEncoder(c)
        self.initial_facets = nn.Parameter(torch.empty(c.n_facets, c.facet_dim))
        self.stage_codes = nn.Parameter(torch.empty(c.prediction_layers, c.facet_dim))
        nn.init.normal_(self.initial_facets, std=0.02)
        nn.init.normal_(self.stage_codes, std=0.02)
        self.central = CentralProcessor(c, self.expert_pools)
        self.adapters = AdapterBank(c, feature_provider=self.expert_pools)
        self.effort_embedding = nn.Embedding(len(EFFORTS), c.facet_dim)
        nn.init.normal_(self.effort_embedding.weight, std=0.02)
        self.control_injection = nn.Linear(c.control_dim, c.facet_dim, bias=False)
        self.modality_head = nn.Linear(c.facet_dim + c.control_dim, 4)
        self.reference_projection = nn.Linear(
            c.kv_heads * c.attention_head_dim, c.control_dim, bias=False
        )
        self.remix = FacetRemix(c)
        self.stages = nn.ModuleList(PairStage(c) for _ in range(c.prediction_layers))
        self.p_router = EpochRouter(
            c.facet_dim, c.router_dim, c.p_experts, c.eligible_experts, version=c.routing_version
        )
        self.u_always = SwiGLU(c.facet_dim, c.u_always_hidden)
        self.u_router = EpochRouter(
            c.facet_dim, c.router_dim, c.u_experts, c.eligible_experts, version=c.routing_version
        )
        self.p_router.bind_feature_provider(self.expert_pools, role="P")
        self.u_router.bind_feature_provider(self.expert_pools, role="U")
        self.u_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.u_gate = nn.Parameter(torch.tensor(-2.0))
        self.lexical_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.lexical_gate = nn.Parameter(torch.tensor(-2.0))
        for name, tensor in self._pair_topology().items():
            self.register_buffer(name, tensor, persistent=False)
        self._routing_losses: list[tuple[Tensor, Tensor]] = []
        self.expert_pools.bind(self, "core", exclusions=("slots",))

    @property
    def p_bank(self) -> ExpertBank:
        return self.expert_pools.ffn_bank(self.config.pair_dim)

    @property
    def u_bank(self) -> ExpertBank:
        return self.expert_pools.ffn_bank(self.config.facet_dim)

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        """The single owner of all expert matrices and per-expert adapter factors."""
        yield from self.expert_pools.parameters()

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        """Only E owners; contextual C and P are independently owned."""
        yield from self.frontend.parameters()
        yield from self.media_encoder.parameters()
        yield from self.adapters.owners["e"].parameters()

    def central_parameters(self) -> Iterator[nn.Parameter]:
        """C, its KV projections and its adapters."""
        yield self.initial_facets
        yield from self.central.parameters()
        yield from self.adapters.owners["c"].parameters()

    def predictor_parameters(self) -> Iterator[nn.Parameter]:
        """All P owners; excludes E, C, R/U."""
        yield self.stage_codes
        for module in (
            self.control_injection,
            self.modality_head,
            self.reference_projection,
            self.effort_embedding,
            self.remix,
            self.stages,
            self.p_router,
            self.adapters.owners["p"],
        ):
            yield from module.parameters()

    def control_parameters(self) -> Iterator[nn.Parameter]:
        """Public-control encoder read by P and by every media realizer (R side)."""
        yield from self.control_encoder.parameters()

    def realizer_parameters(self) -> Iterator[nn.Parameter]:
        """Core-owned U decoder parameters and R's adapters."""
        yield from self.adapters.owners["r"].parameters()
        yield self.u_gate
        yield self.lexical_gate
        for module in (self.u_always, self.u_router, self.u_norm, self.lexical_norm):
            yield from module.parameters()

    def compile_kernels(self, mode: str = "default") -> None:
        """Compile tensor operators without hiding state transitions or cache writes."""
        for stage in self.stages:
            stage.always.compile(mode=mode, dynamic=True)
        for layer in self.central.layers:
            layer.ffn.compile(mode=mode, dynamic=True)
        for layer in self.frontend.refinement:
            layer.ffn.compile(mode=mode, dynamic=True)
        self.remix.compile(mode=mode, dynamic=True)
        self.frontend.reduction.compile(mode=mode, dynamic=True)
        self.frontend.facet_ffn.compile(mode=mode, dynamic=True)
        self.u_always.compile(mode=mode, dynamic=True)
        self.control_encoder.compile(mode=mode, dynamic=True)

    def _pair_topology(self) -> dict[str, Tensor]:
        """Precompute pairing and deterministic facet reductions for every effort."""
        facets = self.config.n_facets
        offsets = (1, facets // 2 + 1, facets // 4 + 1)
        left_stages, right_stages, reductions = [], [], []
        maximum_degree = 1
        for stage in range(self.config.prediction_layers):
            left_updates, right_updates, update_reductions = [], [], []
            for update in range(2):
                left = [
                    (2 * (row % (facets // 2)) + stage + update) % facets
                    for row in range(3 * facets // 2)
                ]
                right = [
                    (value + offsets[row // (facets // 2)]) % facets
                    for row, value in enumerate(left)
                ]
                left_updates.append(left)
                right_updates.append(right)
                effort_reductions = []
                for effort in EFFORTS:
                    count = self._row_counts(facets, effort)[update]
                    incident = [[] for _ in range(facets)]
                    for position, facet in enumerate(left[:count] + right[:count]):
                        incident[facet].append(position)
                    maximum_degree = max(maximum_degree, max(map(len, incident)))
                    effort_reductions.append(incident)
                update_reductions.append(effort_reductions)
            left_stages.append(left_updates)
            right_stages.append(right_updates)
            reductions.append(update_reductions)
        degree = [
            [
                [[max(1, len(indices)) for indices in effort] for effort in update]
                for update in stage
            ]
            for stage in reductions
        ]
        masks = [
            [
                [
                    [[slot < len(indices) for slot in range(maximum_degree)] for indices in effort]
                    for effort in update
                ]
                for update in stage
            ]
            for stage in reductions
        ]
        padded = [
            [
                [
                    [indices + [0] * (maximum_degree - len(indices)) for indices in effort]
                    for effort in update
                ]
                for update in stage
            ]
            for stage in reductions
        ]
        return {
            "pair_left": torch.tensor(left_stages, dtype=torch.long),
            "pair_right": torch.tensor(right_stages, dtype=torch.long),
            "pair_reduction_indices": torch.tensor(padded, dtype=torch.long),
            "pair_reduction_mask": torch.tensor(masks, dtype=torch.bool),
            "pair_reduction_degree": torch.tensor(degree, dtype=torch.long),
        }

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        *,
        model_version: int | None = None,
        episode_ids: Tensor | None = None,
    ) -> CoreState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        device = torch.device(device or self.initial_facets.device)
        home = self.initial_facets.device
        # "cuda" names the model's own "cuda:0": an unindexed device is the same device.
        if device != home and not (device.index is None and device.type == home.type):
            raise ValueError("Initial state and model must be on the same device")
        device = home
        version = self.model_version if model_version is None else model_version
        if episode_ids is None:
            episode_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        if episode_ids.shape != (batch_size,):
            raise ValueError("episode_ids must have one stable identity per batch row")
        ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        empty = self.initial_facets.new_empty(
            batch_size, self.config.kv_heads, 0, self.config.attention_head_dim
        )
        return CoreState(
            facets=self.initial_facets[None].expand(batch_size, -1, -1),
            final=None,
            history=KVHistory(empty, empty),
            records_seen=ids,
            last_record_ids=torch.full_like(ids, -1),
            role_ids=torch.full_like(ids, ROLE_IDS["document"]),
            episode_ids=episode_ids.to(device=device, dtype=torch.long),
            model_version=version,
            routing_version=self.config.routing_version,
            geometry=self.config.state_geometry,
        )

    def set_model_version(self, version: int) -> None:
        if version < 0:
            raise ValueError("Model version must be nonnegative")
        self.model_version = int(version)

    def _check_facets(self, facets: Tensor) -> None:
        if facets.ndim != 3 or facets.shape[1:] != (self.config.n_facets, self.config.facet_dim):
            raise ValueError("Facets must have shape [batch, n_facets, facet_dim]")
        if not facets.is_floating_point():
            raise ValueError("Facets must be floating point")
        if facets.device != self.initial_facets.device:
            raise ValueError("Facets and model must be on the same device")

    def _check_state(self, state: CoreState) -> None:
        assert_cache_version(state, self.model_version, self.config.routing_version)
        if state.geometry != self.config.state_geometry:
            raise ValueError("State belongs to an incompatible model geometry")
        self._check_facets(state.facets)
        if state.final is not None:
            self._check_facets(state.final)
            if state.final.shape != state.facets.shape:
                raise ValueError("Central features must preserve every raw facet slot")
        c, history = self.config, state.history
        expected = (state.batch_size, c.kv_heads, history.slots, c.attention_head_dim)
        if history.keys.shape != expected or history.values.shape != expected:
            raise ValueError("Causal KV history has incompatible geometry")
        maximum_slots = c.max_history_chunks * c.n_facets
        if history.slots > maximum_slots or history.slots % c.n_facets:
            raise ValueError("Causal KV history must contain complete, full-resolution megatokens")
        if state.final is None and history.slots:
            raise ValueError("Only the empty initial sentinel may lack central features")
        if history.mask is not None and (
            history.mask.shape != (state.batch_size, history.slots)
            or history.mask.dtype != torch.bool
        ):
            raise ValueError("History validity mask has incompatible geometry")
        for name in ("records_seen", "last_record_ids", "role_ids", "episode_ids"):
            value = getattr(state, name)
            if value.shape != (state.batch_size,) or value.dtype != torch.long:
                raise ValueError(f"State {name} must be one integer per batch row")

    def encode_chunk(
        self,
        byte_ids: Tensor,
        lengths: Tensor | None = None,
        *,
        track_ids: Tensor | None = None,
        track_semantics: Tensor | None = None,
    ) -> Tensor:
        """Encode bytes under one input- and track-conditioned initial route."""
        summary = _byte_routing_summary(byte_ids, lengths)
        weights = current_adapter_weights(summary)
        with self.expert_pools.routing_epoch_if_absent(
            summary,
            track_ids=track_ids,
            track_semantics=track_semantics,
            role="e",
            adapter_weights=weights,
        ):
            return self.frontend(byte_ids, lengths)

    def encode_media(
        self,
        features: Tensor,
        mask: Tensor | None = None,
        *,
        extractor: str,
        track_ids: Tensor | None = None,
        track_semantics: Tensor | None = None,
    ) -> Tensor:
        """Pinned teacher hidden states [B, T, 512] -> raw facets, through the shared E."""
        return self.media_encoder(
            self,
            features,
            mask,
            extractor=extractor,
            track_ids=track_ids,
            track_semantics=track_semantics,
        )

    def encode_control(
        self,
        byte_ids: Tensor,
        lengths: Tensor | None = None,
        *,
        encoder_grad: bool = True,
        track_ids: Tensor | None = None,
    ) -> Tensor:
        """Keep the control head differentiable while optionally isolating shared E features."""
        summary = _byte_routing_summary(byte_ids, lengths)
        weights = current_adapter_weights(summary)
        with self.expert_pools.routing_epoch_if_absent(
            summary, track_ids=track_ids, role="p", adapter_weights=weights
        ):
            with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
                source = self.frontend.source_features(byte_ids, lengths)
            return self.control_encoder(source)

    def _roles(self, role_ids: Tensor | None, state: CoreState, default: str) -> Tensor:
        if role_ids is None:
            return torch.full_like(state.role_ids, ROLE_IDS[default])
        if role_ids.shape != (state.batch_size,):
            raise ValueError("role_ids must identify one supported causal role per batch row")
        return role_ids.to(device=state.device, dtype=torch.long)

    def advance_raw(
        self,
        state: CoreState,
        raw_facets: Tensor,
        *,
        role_ids: Tensor | None = None,
        record_ids: Tensor | None = None,
        checkpointing: bool | None = None,
    ) -> CoreState:
        """Select exact raw facets, run C once, and append their contextual KV now.

        The same path accepts E observations, sampled predictions, and taped raw
        events, including invalid or empty actions. It never runs E, P, or R and
        never reprocesses the previous selected input. Controls belong only to P.
        """
        self._check_state(state)
        self._check_facets(raw_facets)
        if raw_facets.shape[0] != state.batch_size:
            raise ValueError("Raw input and state batches differ")
        roles = self._roles(role_ids, state, "assistant")
        if record_ids is None:
            record_ids = state.last_record_ids + 1
        else:
            record_ids = record_ids.to(device=state.device, dtype=torch.long)
            if record_ids.shape != (state.batch_size,) or bool(
                (record_ids <= state.last_record_ids).any()
            ):
                raise ValueError("Record IDs must increase; duplicate selection is forbidden")
        final = self.central(
            raw_facets, state.history, roles, state.records_seen, checkpointing=checkpointing
        )
        return replace(
            state,
            facets=raw_facets,
            final=final,
            history=self.central.write(final, state.history, state.records_seen),
            records_seen=state.records_seen + 1,
            last_record_ids=record_ids,
            role_ids=roles,
        )

    def observe(
        self,
        byte_ids: Tensor,
        state: CoreState,
        lengths: Tensor | None = None,
        record_ids: Tensor | None = None,
        role_ids: Tensor | None = None,
        *,
        encoder_grad: bool = True,
    ) -> CoreState:
        """Encode this observation and process it immediately through the shared C.

        No prior-input consumption, remix, prediction, or realization is hidden
        here. Detaching E leaves C/history gradients live in recurrent training.
        """
        self._check_state(state)
        if byte_ids.shape[0] != state.batch_size or byte_ids.device != state.device:
            raise ValueError("Observation and state batch/device differ")
        with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
            encoded = self.encode_chunk(byte_ids, lengths)
        return self.advance_raw(
            state,
            encoded,
            role_ids=self._roles(role_ids, state, "document"),
            record_ids=record_ids,
        )

    def predict_facets(
        self,
        state: CoreState,
        control: Tensor | None = None,
        effort: int = 6,
    ) -> Tensor:
        """Pure remix/P prediction from cached final C features and updated KV.

        Selected inputs incur no additional central pass. An empty startup
        sentinel is processed locally without a KV write or record count; this
        startup-only work is repeated if that same empty state is queried again.
        Neither the raw selected input nor cached history is mutated.
        """
        self._check_state(state)
        if effort not in EFFORTS:
            raise ValueError(f"Predictive effort must be one of {EFFORTS}")
        final = state.final
        if final is None:
            final = self.central(state.facets, state.history, state.role_ids, state.records_seen)
        if control is None:
            control = final.new_zeros(state.batch_size, self.config.control_dim)
        if (
            control.shape != (state.batch_size, self.config.control_dim)
            or control.device != state.device
        ):
            raise ValueError("Causal control has the wrong shape or device")
        with self.expert_pools.routing_epoch(final.mean(1), role="p", stage=0):
            facets = self.remix(final)
            injection = self.control_injection(control.to(facets.dtype))
        injection = injection + self.effort_embedding.weight[EFFORTS.index(effort)]
        facets = facets + injection[:, None]
        stages = tuple(range(1, self.config.prediction_layers + 1))
        with self.expert_pools.routing_epoch(facets.mean(1), role="p", stage=0):
            program = self.p_router.compile(
                facets.mean(1),
                stages,
                sum(self._row_counts(self.config.n_facets, effort)),
                epoch="prediction",
            )
        self._record_routing_loss(program.balance_loss, program.stabilization_loss)
        for stage in stages:
            facets = self._execute_pair_stage(
                facets, stage, program.at(stage), effort, state.history, state.records_seen
            )
        return facets

    def prepare_realization(
        self, selected: CoreState | Tensor | RealizationContext
    ) -> RealizationContext:
        """Expose the shared raw facets without running C or reading history."""
        if isinstance(selected, CoreState):
            self._check_state(selected)
            if selected.final is None:
                raise ValueError("The initial sentinel is not a selected realization input")
            selected = selected.facets
        if isinstance(selected, RealizationContext):
            self._check_facets(selected.facets)
            return selected
        if isinstance(selected, Tensor):
            self._check_facets(selected)
            return RealizationContext(selected)
        raise TypeError("Realization requires a state or a raw facet tensor")

    @staticmethod
    def _row_counts(facets: int, effort: int) -> tuple[int, int]:
        return {
            1: (facets // 4, facets // 4),
            2: (facets // 2, facets // 2),
            3: (facets, facets // 2),
            6: (3 * facets // 2, 3 * facets // 2),
        }[effort]

    def _pair_stage(
        self,
        facets: Tensor,
        stage: int,
        route: RowRoute,
        effort: int,
        history: KVHistory,
        positions: Tensor,
    ) -> Tensor:
        wrapper = self.stages[stage - 1]
        facets = facets + self.stage_codes[stage - 1]
        addresses = self.central.slot_codes()
        # For local-only attention every live facet shares time t, so RoPE
        # cancels out of Q·K. Avoid unnecessary rotations at eight P sites.
        reads_history = stage in P_HISTORY_READ_STAGES
        time = positions if reads_history else None
        live = self.central.memory(facets, temporal_positions=time, slot_codes=addresses)
        # Every P stage mixes live facets. Exactly four fixed stages also query
        # old committed records; prediction never writes a new KV representation.
        historical = (
            AttentionFeatures(history.keys, history.values, history.mask) if reads_history else None
        )
        projected = join_attention(live, historical)
        facets = facets + wrapper.attention_gate.sigmoid() * wrapper.attention.read(
            facets, projected, temporal_positions=time, slot_codes=addresses
        )
        return self.pair_updates(facets, stage, route, effort)

    def pair_updates(self, facets: Tensor, stage: int, route: RowRoute, effort: int) -> Tensor:
        """The stage's routed expert updates over [rows, facets, width] blocks."""
        wrapper = self.stages[stage - 1]
        begin = 0
        for update, count in enumerate(self._row_counts(self.config.n_facets, effort)):
            selection = RowRoute(
                route.expert_ids[:, begin : begin + count],
                route.confidence[:, begin : begin + count],
                route.eligible,
                route.palette_probabilities,
                route.balance_loss,
                route.stabilization_loss,
            )
            facets = wrapper.update(
                facets,
                self.pair_left[stage - 1, update, :count],
                self.pair_right[stage - 1, update, :count],
                selection,
                self.p_bank,
                update,
                self.pair_reduction_indices[stage - 1, update, EFFORTS.index(effort)],
                self.pair_reduction_mask[stage - 1, update, EFFORTS.index(effort)],
                self.pair_reduction_degree[stage - 1, update, EFFORTS.index(effort)],
            )
            begin += count
        return facets

    def _execute_pair_stage(
        self,
        facets: Tensor,
        stage: int,
        route: RowRoute,
        effort: int,
        history: KVHistory,
        positions: Tensor,
    ) -> Tensor:
        with self.expert_pools.routing_epoch(facets.mean(1), role="p", stage=stage):
            return self.run_pair_stage(
                facets,
                stage,
                route,
                effort,
                lambda value: self._pair_stage(value, stage, route, effort, history, positions),
            )

    def run_pair_stage(
        self, facets: Tensor, stage: int, route: RowRoute, effort: int, compute
    ) -> Tensor:
        """Record the executed route once, then run ``compute(facets)`` for the stage.

        Recording stays outside the computation, so activation recomputation is
        never counted as learning exposure.
        """
        self.p_router.record_executed(route)
        start = 0
        for update, count in enumerate(self._row_counts(self.config.n_facets, effort)):
            self.p_bank.record_routes(
                route.expert_ids[:, start : start + count],
                route.eligible if update == 0 else None,
                gradient_enabled=torch.is_grad_enabled(),
            )
            start += count
        if self.config.checkpoint_stages and self.training and torch.is_grad_enabled():
            return checkpoint(
                compute,
                facets,
                use_reentrant=False,
                preserve_rng_state=False,
                context_fn=checkpoint_context_fn,
            )
        return compute(facets)

    def _record_routing_loss(self, balance: Tensor, stabilization: Tensor) -> None:
        # Outside checkpointed operators: loss collection never affects routing.
        if torch.is_grad_enabled():
            self._routing_losses.append((balance, stabilization))

    def apply_u(
        self, rows: Tensor, context: RealizationContext, level: int, refinement: bool = False
    ) -> Tensor:
        """Shared output transform routed directly from raw facets."""
        summary = context.route_context
        weights = current_adapter_weights(summary)
        with self.expert_pools.routing_epoch_if_absent(
            summary, role="r", stage=level, adapter_weights=weights
        ):
            return self._apply_u(rows, context, level, refinement)

    def _apply_u(
        self, rows: Tensor, context: RealizationContext, level: int, refinement: bool = False
    ) -> Tensor:
        if (
            rows.ndim != 3
            or rows.shape[0] != context.batch_size
            or rows.shape[-1] != self.config.facet_dim
        ):
            raise ValueError("Output transform expects [batch, rows, facet_dim]")
        if level < 1 or rows.shape[1] < 1:
            raise ValueError("Growth level and number of rows must be positive")
        if refinement:
            return self.lexical_refine(rows)
        rows = rows.to(parameter_dtype(self.u_always))
        eligible = min(self.config.eligible_experts, 4 if rows.shape[1] < 64 else 8, rows.shape[1])
        route = self.u_router.route(
            context.route_context.to(parameter_dtype(self.u_router)),
            level,
            rows.shape[1],
            eligible_limit=eligible,
        )
        self.u_router.record_executed(route)
        self.u_bank.record_routes(
            route.expert_ids, route.eligible, gradient_enabled=torch.is_grad_enabled()
        )
        self._record_routing_loss(route.balance_loss, route.stabilization_loss)
        normalized = self.u_norm(rows)
        delta = self.u_bank(
            normalized,
            route.expert_ids,
            route.confidence,
            adapter_weights=current_adapter_weights(normalized),
        ) + self.u_always(normalized)
        return rows + self.u_gate.sigmoid() * delta

    def lexical_refine(self, rows: Tensor) -> Tensor:
        """The final routed refinement reuses the shared single-facet pool."""
        rows = rows.to(parameter_dtype(self.u_always))
        summary = rows.mean(1)
        weights = current_adapter_weights(summary)
        with self.expert_pools.routing_epoch_if_absent(summary, role="r", adapter_weights=weights):
            return rows + self.lexical_gate.sigmoid() * self.u_always(self.lexical_norm(rows))

    def dispatch_metrics(self, *, reset: bool = False) -> dict[str, dict[str, Tensor]]:
        return {"p": self.p_bank.metrics(reset=reset), "u": self.u_bank.metrics(reset=reset)}

    def routing_aux_loss(self, *, clear: bool = True) -> tuple[Tensor, Tensor]:
        """Unweighted sums of per-operation router balance/stabilization losses.

        Clear before a decision or transaction; consume when forming its loss.
        Calling this method never changes routing or core state. No-gradient
        inference does not retain loss tensors or grow this collector.
        """
        if self._routing_losses:
            balance, stabilization = zip(*self._routing_losses, strict=True)
            result = torch.stack(balance).sum(), torch.stack(stabilization).sum()
        else:
            zero = self.initial_facets.new_zeros((), dtype=torch.float32)
            result = zero, zero
        if clear:
            self._routing_losses.clear()
        return result

    @contextmanager
    def discard_routing_losses(self) -> Iterator[None]:
        """Isolate an auxiliary forward without changing an outer routing graph.

        Selection and dispatch are unchanged. Nested contexts and exceptions
        restore the original collector; no graph is detached or overwritten.
        """
        previous = self._routing_losses
        self._routing_losses = []
        # Auxiliary branches also stay out of the global router-balance replay.
        routers = (
            self.p_router,
            self.u_router,
            *self.adapters.owners.values(),
            *self.expert_pools.routing_selectors().values(),
        )
        recorders = [router.balance_recorder for router in routers]
        for router in routers:
            router.balance_recorder = None
        try:
            yield
        finally:
            self._routing_losses = previous
            for router, recorder in zip(routers, recorders, strict=True):
                router.balance_recorder = recorder

    def reset_state(
        self,
        state: CoreState,
        rows: Tensor | None = None,
        episode_ids: Tensor | None = None,
    ) -> CoreState:
        """Clear episode context; partial resets preserve other rows' computed features."""
        self._check_state(state)
        if rows is None:
            return self.initial_state(state.batch_size, state.device, episode_ids=episode_ids)
        if rows.shape != (state.batch_size,) or rows.dtype != torch.bool:
            raise ValueError("Selective resets require a boolean vector over episodes")
        rows = rows.to(state.device)
        keep = ~rows
        ids = state.episode_ids if episode_ids is None else episode_ids.to(state.episode_ids)
        if ids.shape != (state.batch_size,):
            raise ValueError("episode_ids must have one stable identity per batch row")
        history_mask = state.history.mask
        if history_mask is None:
            history_mask = torch.ones(
                state.batch_size,
                state.history.slots,
                device=state.device,
                dtype=torch.bool,
            )
        final = state.final
        if final is not None:
            # Compute one empty-context sentinel, not another pass over any
            # selected input. Mixed batches then have valid reusable caches.
            initial = self.initial_state(1)
            start_final = self.central(
                initial.facets, initial.history, initial.role_ids, initial.records_seen
            )
            final = torch.where(rows[:, None, None], start_final, final)
        return replace(
            state,
            facets=torch.where(rows[:, None, None], self.initial_facets[None], state.facets),
            final=final,
            history=KVHistory(
                state.history.keys * keep[:, None, None, None],
                state.history.values * keep[:, None, None, None],
                history_mask & keep[:, None],
            ),
            records_seen=state.records_seen * keep,
            last_record_ids=torch.where(rows, -1, state.last_record_ids),
            role_ids=torch.where(rows, ROLE_IDS["document"], state.role_ids),
            episode_ids=torch.where(rows, ids, state.episode_ids),
        )
