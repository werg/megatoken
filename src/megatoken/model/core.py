"""Raw facet recurrence: residual body, nonresidual remix, residual prediction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from megatoken.contracts import ROLE_TO_ID

from .attention import AttentionBranch, AttentionFeatures
from .config import ModelConfig
from .experts import ExpertBank
from .frontend import SeedControlEncoder
from .ingestion import ChunkEncoder
from .layers import RMSNorm, SwiGLU, position_features_at
from .routing import EpochRouter, RowRoute
from .state import ConsumedInput, CoreState, KVHistory, RealizationContext, assert_cache_version

ROLE_IDS = ROLE_TO_ID
RECORD_ROLES = tuple(ROLE_TO_ID)
EFFORTS = (1, 2, 3, 6)


class PairStage(nn.Module):
    """A stage-private dense branch around one globally shared expert bank."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
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
        delta = bank(pairs, route.expert_ids, route.confidence) + self.always(pairs)
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
    Ten subsequent residual stages build the actual next-input prediction.
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


class FacetedCore(nn.Module):
    """Final prediction facets are both decoder input and the next core input.

    A process call consumes the current input once: six residual body stages,
    a nonresidual slot/channel remix, then ten residual prediction stages.
    Ordinary causal KV history stores processed input bodies. Observation uses
    the same body path to capture a pending input before installing E(new bytes);
    no separate learned ingestion writer or output reconciliation exists.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.model_version = 0
        self.frontend = ChunkEncoder(c)
        self.control_encoder = SeedControlEncoder(c)
        self.initial_facets = nn.Parameter(torch.empty(c.n_facets, c.facet_dim))
        self.stage_codes = nn.Parameter(torch.empty(c.n_stages, c.facet_dim))
        nn.init.normal_(self.initial_facets, std=0.02)
        nn.init.normal_(self.stage_codes, std=0.02)
        self.live_attention = AttentionBranch(
            c.facet_dim,
            c.facet_dim,
            c.query_heads,
            c.kv_heads,
            c.attention_head_dim,
            c.norm_eps,
        )
        self.live_gate = nn.Parameter(torch.tensor(-2.0))
        self.position_gate = nn.Parameter(torch.tensor(-2.0))
        self.role_embedding = nn.Embedding(len(RECORD_ROLES), c.facet_dim)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        self.effort_embedding = nn.Embedding(len(EFFORTS), c.facet_dim)
        nn.init.normal_(self.effort_embedding.weight, std=0.02)
        self.control_injection = nn.Linear(c.control_dim, c.facet_dim, bias=False)
        self.remix = FacetRemix(c)
        # The primary/compact presets retain every facet. A compressed cache is
        # an explicit ablation and owns this projection only when requested.
        self.history_pool = (
            nn.Linear(c.n_facets, c.history_slots, bias=False)
            if c.history_slots < c.n_facets
            else None
        )
        self.p_bank = ExpertBank(c.p_experts, c.pair_dim, c.p_hidden, c.expert_backend, "p")
        self.stages = nn.ModuleList(PairStage(c) for _ in range(c.n_stages))
        self.p_router = EpochRouter(
            c.facet_dim,
            c.router_dim,
            c.p_experts,
            c.eligible_experts,
            version=c.routing_version,
        )
        self.u_bank = ExpertBank(c.u_experts, c.facet_dim, c.u_hidden, c.expert_backend, "u")
        self.u_always = SwiGLU(c.facet_dim, c.u_always_hidden)
        self.u_router = EpochRouter(
            c.facet_dim,
            c.router_dim,
            c.u_experts,
            c.eligible_experts,
            version=c.routing_version,
        )
        self.u_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.u_gate = nn.Parameter(torch.tensor(-2.0))
        self.lexical_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.lexical_gate = nn.Parameter(torch.tensor(-2.0))
        for name, tensor in self._pair_topology().items():
            self.register_buffer(name, tensor, persistent=False)
        self._routing_losses: list[tuple[Tensor, Tensor]] = []

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        """Exact E(bytes) owners; no body, cache, remix, or decoder parameters."""
        yield from self.frontend.parameters()

    def realizer_parameters(self) -> Iterator[nn.Parameter]:
        """Core-owned decoder weights; the generator owns its other modules."""
        yield self.u_gate
        yield self.lexical_gate
        for module in (self.u_bank, self.u_always, self.u_router, self.u_norm, self.lexical_norm):
            yield from module.parameters()

    def compile_kernels(self, mode: str = "default") -> None:
        """Compile reusable tensor operators; keep stream/tape orchestration explicit."""
        for stage in self.stages:
            stage.always.compile(mode=mode, dynamic=True)
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
        for stage in range(self.config.n_stages):
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
        if device != self.initial_facets.device:
            raise ValueError("Initial state and model must be on the same device")
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
            history=KVHistory(empty, empty),
            records_seen=ids,
            last_record_ids=torch.full_like(ids, -1),
            input_record_ids=ids,
            role_ids=torch.full_like(ids, ROLE_IDS["document"]),
            episode_ids=episode_ids.to(device=device, dtype=torch.long),
            pending_mask=torch.zeros(batch_size, device=device, dtype=torch.bool),
            has_pending_input=False,
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
        c, history = self.config, state.history
        expected = (state.batch_size, c.kv_heads, history.slots, c.attention_head_dim)
        if history.keys.shape != expected or history.values.shape != expected:
            raise ValueError("Causal KV history has incompatible geometry")
        if history.slots > c.history_slots * c.max_history_chunks:
            raise ValueError("Causal KV history exceeds its configured bound")
        if history.mask is not None and (
            history.mask.shape != (state.batch_size, history.slots)
            or history.mask.dtype != torch.bool
        ):
            raise ValueError("History validity mask has incompatible geometry")
        if state.pending_mask is not None and (
            state.pending_mask.shape != (state.batch_size,)
            or state.pending_mask.dtype != torch.bool
        ):
            raise ValueError("Pending-input validity mask must identify batch rows")
        for name in (
            "records_seen",
            "last_record_ids",
            "input_record_ids",
            "role_ids",
            "episode_ids",
        ):
            value = getattr(state, name)
            if value.shape != (state.batch_size,) or value.dtype != torch.long:
                raise ValueError(f"State {name} must be one integer per batch row")

    def encode_chunk(self, byte_ids: Tensor, lengths: Tensor | None = None) -> Tensor:
        """History-, role-, and control-independent hierarchical E(bytes)."""
        return self.frontend(byte_ids, lengths)

    def encode_control(
        self,
        byte_ids: Tensor,
        lengths: Tensor | None = None,
        *,
        encoder_grad: bool = True,
    ) -> Tensor:
        """Encode causal control, optionally isolating the shared byte frontend.

        Later recurrent stages train E only through its detached-target alignment
        objective. The control encoder remains differentiable when its shared
        frontend features are detached; no parameter requires_grad flags change.
        """
        with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
            source = self.frontend.source_features(byte_ids, lengths)
        return self.control_encoder(source)

    @staticmethod
    def _row_counts(facets: int, effort: int) -> tuple[int, int]:
        return {
            1: (facets // 4, facets // 4),
            2: (facets // 2, facets // 2),
            3: (facets, facets // 2),
            6: (3 * facets // 2, 3 * facets // 2),
        }[effort]

    def _attention(self, facets: Tensor, history: KVHistory) -> Tensor:
        live = self.live_attention.encode(facets)
        if not history.slots:
            return self.live_attention.read(facets, live)
        mask = None
        if history.mask is not None:
            mask = torch.cat(
                (
                    history.mask,
                    torch.ones(facets.shape[:2], device=facets.device, dtype=torch.bool),
                ),
                dim=1,
            )
        features = AttentionFeatures(
            torch.cat((history.keys.to(live.keys.dtype), live.keys), dim=2),
            torch.cat((history.values.to(live.values.dtype), live.values), dim=2),
            mask,
        )
        return self.live_attention.read(facets, features)

    def _pair_stage(
        self,
        facets: Tensor,
        stage: int,
        route: RowRoute,
        effort: int,
        history: KVHistory,
    ) -> Tensor:
        wrapper = self.stages[stage - 1]
        facets = facets + self.stage_codes[stage - 1]
        facets = facets + self.live_gate.sigmoid() * self._attention(facets, history)
        route_start = 0
        for update, count in enumerate(self._row_counts(self.config.n_facets, effort)):
            selection = RowRoute(
                route.expert_ids[:, route_start : route_start + count],
                route.confidence[:, route_start : route_start + count],
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
            route_start += count
        return facets

    def _execute_pair_stage(
        self,
        facets: Tensor,
        stage: int,
        route: RowRoute,
        effort: int,
        history: KVHistory,
    ) -> Tensor:
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
                lambda value: self._pair_stage(value, stage, route, effort, history),
                facets,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self._pair_stage(facets, stage, route, effort, history)

    def _record_routing_loss(self, balance: Tensor, stabilization: Tensor) -> None:
        # Outside checkpointed operators: loss collection never affects routing.
        if torch.is_grad_enabled():
            self._routing_losses.append((balance, stabilization))

    def _append_history(self, body: Tensor, state: CoreState) -> KVHistory:
        if not state.has_pending_input:
            return state.history
        pooled = (
            body
            if self.history_pool is None
            else self.history_pool(body.transpose(1, 2)).transpose(1, 2)
        )
        addition = self.live_attention.encode(pooled)
        old = state.history
        mask = None
        if old.mask is not None or state.pending_mask is not None:
            before = (
                old.mask
                if old.mask is not None
                else torch.ones(
                    state.batch_size,
                    old.slots,
                    device=state.device,
                    dtype=torch.bool,
                )
            )
            after = torch.ones(
                state.batch_size,
                self.config.history_slots,
                device=state.device,
                dtype=torch.bool,
            )
            if state.pending_mask is not None:
                after = after & state.pending_mask[:, None]
            mask = torch.cat((before, after), dim=1)
        limit = self.config.history_slots * self.config.max_history_chunks
        return KVHistory(
            torch.cat((old.keys.to(addition.keys.dtype), addition.keys), dim=2)[:, :, -limit:],
            torch.cat((old.values.to(addition.values.dtype), addition.values), dim=2)[
                :, :, -limit:
            ],
            None if mask is None else mask[:, -limit:],
        )

    def consume_input(
        self,
        state: CoreState,
        control: Tensor | None = None,
        effort: int = 6,
    ) -> ConsumedInput:
        """Shared six-stage body; capture this input's causal context exactly once.

        This call is pure: the supplied state remains unchanged. Process and
        observe consume the returned context instead of reprocessing that input.
        The learned initial sentinel may seed unconditional prediction, but is
        never entered into history or counted as a real record. Control and
        effort condition this computation, not a stored pending-input object.
        An observation-only consumption uses neutral control/full effort; the
        raw pending facets already contain their earlier generation conditioning.
        """
        self._check_state(state)
        if effort not in EFFORTS:
            raise ValueError(f"Predictive effort must be one of {EFFORTS}")
        facets = state.facets.to(self.initial_facets.dtype)
        if control is None:
            control = facets.new_zeros(state.batch_size, self.config.control_dim)
        if (
            control.shape != (state.batch_size, self.config.control_dim)
            or control.device != state.device
        ):
            raise ValueError("Causal control has the wrong shape or device")
        injection = self.control_injection(control.to(facets.dtype))
        injection = injection + self.effort_embedding.weight[EFFORTS.index(effort)]
        position = position_features_at(state.records_seen, self.config.facet_dim).to(facets.dtype)
        facets = facets + self.role_embedding(state.role_ids)[:, None]
        facets = facets + self.position_gate.sigmoid() * position[:, None] + injection[:, None]
        program = self.p_router.compile(
            facets.mean(1),
            tuple(range(1, self.config.body_stages + 1)),
            3 * self.config.n_facets,
            epoch="body",
            allow_exploration=False,
        )
        self._record_routing_loss(program.balance_loss, program.stabilization_loss)
        for stage in range(1, self.config.body_stages + 1):
            facets = self._execute_pair_stage(facets, stage, program.at(stage), 6, state.history)
        history = self._append_history(facets, state)
        valid = state.pending_mask
        if valid is None:
            records_seen = state.records_seen + 1
            last_record_ids = state.input_record_ids
        else:
            records_seen = state.records_seen + valid.to(torch.long)
            last_record_ids = torch.where(valid, state.input_record_ids, state.last_record_ids)
        return ConsumedInput(facets, history, records_seen, last_record_ids)

    def _roles(self, role_ids: Tensor | None, state: CoreState, default: str) -> Tensor:
        if role_ids is None:
            return torch.full_like(state.role_ids, ROLE_IDS[default])
        if role_ids.shape != (state.batch_size,):
            raise ValueError("role_ids must identify one supported causal role per batch row")
        return role_ids.to(device=state.device, dtype=torch.long)

    @staticmethod
    def _pending_state(
        state: CoreState,
        raw_facets: Tensor,
        consumed: ConsumedInput,
        role_ids: Tensor,
        record_ids: Tensor | None = None,
    ) -> CoreState:
        """One pure transition shared by observations, predictions, and tape replay."""
        return replace(
            state,
            facets=raw_facets,
            history=consumed.history,
            records_seen=consumed.records_seen,
            last_record_ids=consumed.last_record_ids,
            input_record_ids=consumed.last_record_ids + 1 if record_ids is None else record_ids,
            role_ids=role_ids,
            pending_mask=None,
            has_pending_input=True,
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
        """Capture a pending prior input's body, then install E(observed bytes).

        There is no prediction/remix/decoder work for this observation transition.
        E may be detached in recurrent training while body/cache gradients remain
        live. Consecutive observations and action-to-tool feedback retain context.
        """
        self._check_state(state)
        if byte_ids.shape[0] != state.batch_size or byte_ids.device != state.device:
            raise ValueError("Observation and state batch/device differ")
        if state.has_pending_input:
            consumed = self.consume_input(state)
        else:
            consumed = ConsumedInput(
                state.facets,
                state.history,
                state.records_seen,
                state.last_record_ids,
            )
        if record_ids is None:
            record_ids = consumed.last_record_ids + 1
        else:
            record_ids = record_ids.to(device=state.device, dtype=torch.long)
            if record_ids.shape != (state.batch_size,) or bool(
                (record_ids <= consumed.last_record_ids).any()
            ):
                raise ValueError("Record IDs must increase; duplicate observation is forbidden")
        with torch.set_grad_enabled(torch.is_grad_enabled() and encoder_grad):
            encoded = self.encode_chunk(byte_ids, lengths)
        return self._pending_state(
            state,
            encoded,
            consumed,
            self._roles(role_ids, state, "document"),
            record_ids,
        )

    def process(
        self,
        state: CoreState,
        control: Tensor | None = None,
        effort: int = 6,
        *,
        next_role_ids: Tensor | None = None,
    ) -> CoreState:
        """Build raw next-input facets after the remix and ten residual stages.

        The returned tensor feeds both deterministic realization and the next
        iteration directly. A generator may replace its mean with one sampled
        tensor using replace_facets; that operation contains no neural work.
        """
        consumed = self.consume_input(state, control, effort)
        facets = self.remix(consumed.facets)
        stages = tuple(range(self.config.body_stages + 1, self.config.n_stages + 1))
        program = self.p_router.compile(
            facets.mean(1),
            stages,
            sum(self._row_counts(self.config.n_facets, effort)),
            epoch="prediction",
            allow_exploration=False,
        )
        self._record_routing_loss(program.balance_loss, program.stabilization_loss)
        for stage in stages:
            facets = self._execute_pair_stage(
                facets,
                stage,
                program.at(stage),
                effort,
                consumed.history,
            )
        return self._pending_state(
            state,
            facets,
            consumed,
            self._roles(next_role_ids, state, "assistant"),
        )

    def advance_raw(
        self,
        state: CoreState,
        raw_facets: Tensor,
        control: Tensor | None = None,
        effort: int = 6,
        role_ids: Tensor | None = None,
    ) -> CoreState:
        """Replay a fixed raw action without computing a discarded prediction.

        Equivalent state to process(...), then replace_facets(..., raw_facets).
        Only the common six-stage input body is needed when the next raw tensor
        is already recorded. This is not a density-scoring or sampling operation.
        """
        self._check_facets(raw_facets)
        if raw_facets.shape[0] != state.batch_size:
            raise ValueError("Raw replay and state batches differ")
        consumed = self.consume_input(state, control, effort)
        return self._pending_state(
            state,
            raw_facets,
            consumed,
            self._roles(role_ids, state, "assistant"),
        )

    def replace_facets(
        self,
        state: CoreState,
        raw_facets: Tensor,
        role_ids: Tensor | None = None,
    ) -> CoreState:
        """Replace the pending prediction verbatim; no encoding, projection, or write."""
        self._check_state(state)
        self._check_facets(raw_facets)
        if raw_facets.shape[0] != state.batch_size:
            raise ValueError("Raw prediction and state batches differ")
        roles = state.role_ids if role_ids is None else self._roles(role_ids, state, "assistant")
        return replace(state, facets=raw_facets, role_ids=roles)

    def predict_facets(
        self,
        state: CoreState,
        control: Tensor | None = None,
        effort: int = 6,
    ) -> Tensor:
        """Inspect a mean without advancing the caller's immutable state.

        Training/generation must use process when its returned context will be
        continued; this convenience method intentionally discards that new state.
        """
        return self.process(state, control, effort).facets

    def prepare_realization(self, raw_facets: Tensor) -> RealizationContext:
        self._check_facets(raw_facets)
        return RealizationContext(raw_facets)

    def apply_u(
        self, rows: Tensor, context: RealizationContext, level: int, refinement: bool = False
    ) -> Tensor:
        """Shared output transform with routing determined only by the raw final prediction."""
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
        rows = rows.to(self.u_always.gate_up.weight.dtype)
        eligible = min(self.config.eligible_experts, 4 if rows.shape[1] < 64 else 8, rows.shape[1])
        route = self.u_router.route(
            context.route_context.to(self.u_router.context_projection.weight.dtype),
            level,
            rows.shape[1],
            eligible_limit=eligible,
            allow_exploration=False,
        )
        self.u_bank.record_routes(
            route.expert_ids, route.eligible, gradient_enabled=torch.is_grad_enabled()
        )
        self._record_routing_loss(route.balance_loss, route.stabilization_loss)
        normalized = self.u_norm(rows)
        delta = self.u_bank(normalized, route.expert_ids, route.confidence) + self.u_always(
            normalized
        )
        return rows + self.u_gate.sigmoid() * delta

    def lexical_refine(self, rows: Tensor) -> Tensor:
        """The final dense refinement aliases the same U always branch."""
        rows = rows.to(self.u_always.gate_up.weight.dtype)
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

    def reset_state(
        self,
        state: CoreState,
        rows: Tensor | None = None,
        episode_ids: Tensor | None = None,
    ) -> CoreState:
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
        pending = keep if state.pending_mask is None else keep & state.pending_mask
        return replace(
            state,
            facets=torch.where(rows[:, None, None], self.initial_facets[None], state.facets),
            history=KVHistory(
                state.history.keys * keep[:, None, None, None],
                state.history.values * keep[:, None, None, None],
                history_mask & keep[:, None],
            ),
            records_seen=state.records_seen * keep,
            last_record_ids=torch.where(rows, -1, state.last_record_ids),
            input_record_ids=torch.where(rows, 0, state.input_record_ids),
            role_ids=torch.where(rows, ROLE_IDS["document"], state.role_ids),
            episode_ids=torch.where(rows, ids, state.episode_ids),
            pending_mask=pending,
        )
