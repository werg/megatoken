"""Shared factual and predictive core with explicit causal stage boundaries."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from megatoken.contracts import ROLE_TO_ID

from .attention import AttentionBranch, MemoryInterface
from .config import ModelConfig
from .experts import ExpertBank
from .frontend import ByteFrontend, SeedControlEncoder
from .layers import RMSNorm, SwiGLU, masked_mean
from .recurrent import RecordGDN2
from .routing import EpochRouter, RowRoute
from .state import FactualState, KVMemory, PredictionState, assert_cache_version

ROLE_IDS = ROLE_TO_ID
RECORD_ROLES = tuple(ROLE_TO_ID)
PREDICTION_BOUNDARIES = (7, 8, 10, 15, 16)
EFFORTS = (1, 2, 3, 6)


class PairStage(nn.Module):
    """A stage-private dense branch around one globally shared expert bank."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.pair_dim, config.norm_eps)
        self.always = SwiGLU(config.pair_dim, config.p_always_hidden)
        self.residual_gates = nn.Parameter(torch.full((2,), -2.0))
        self.output_norm = RMSNorm(config.facet_dim, config.norm_eps)

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


class FacetedCore(nn.Module):
    """One implementation of ingestion, prediction, and shared output transforms.

    ``ingest`` commits actual bytes through six factual stages and writes all four
    recurrent sites exactly once. Prediction creates a separate immutable record;
    no speculative root, target-posterior data, or private carry writes factual
    history. A changed optimizer/model version invalidates all runtime caches.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.model_version = 0
        self.frontend = ByteFrontend(c)
        self.control_encoder = SeedControlEncoder(c)
        self.facet_codes = nn.Parameter(torch.empty(c.n_facets, c.facet_dim))
        self.stage_codes = nn.Parameter(torch.empty(c.n_stages, c.facet_dim))
        nn.init.normal_(self.facet_codes, std=0.02)
        nn.init.normal_(self.stage_codes, std=0.02)
        self.role_embedding = nn.Embedding(len(RECORD_ROLES), c.facet_dim)
        self.effort_embedding = nn.Embedding(len(EFFORTS), c.facet_dim)
        self.source_attention = AttentionBranch(
            c.facet_dim,
            c.source_dim,
            c.query_heads,
            c.live_kv_heads,
            c.attention_head_dim,
            c.norm_eps,
        )
        self.source_context = nn.Linear(c.source_dim, c.facet_dim, bias=False)
        self.stem_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.memory = MemoryInterface(c)
        self.gdn = nn.ModuleList(RecordGDN2(c) for _ in c.gdn_sites)
        self.p_bank = ExpertBank(c.p_experts, c.pair_dim, c.p_hidden, c.expert_backend, "p")
        self.stages = nn.ModuleList(PairStage(c) for _ in range(c.n_stages))
        self.u_bank = ExpertBank(c.u_experts, c.facet_dim, c.u_hidden, c.expert_backend, "u")
        self.u_always = SwiGLU(c.facet_dim, c.u_always_hidden)
        self.u_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.u_gate = nn.Parameter(torch.tensor(-2.0))
        self.lexical_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.lexical_gate = nn.Parameter(torch.tensor(-2.0))
        self.p_router = EpochRouter(
            c.facet_dim,
            c.router_dim,
            c.p_experts,
            c.eligible_experts,
            exploration=c.router_exploration,
            version=c.routing_version,
        )
        self.u_router = EpochRouter(
            c.facet_dim,
            c.router_dim,
            c.u_experts,
            c.eligible_experts,
            exploration=c.router_exploration,
            version=c.routing_version,
        )
        self.root_injection = nn.Linear(c.root_dim, c.facet_dim, bias=False)
        self.control_injection = nn.Linear(c.control_dim, c.facet_dim, bias=False)
        self.conditioning_injection = nn.Linear(c.facet_dim, c.facet_dim, bias=False)
        self.carry_read = nn.Linear(c.facet_dim, c.facet_dim, bias=False)
        self.carry_cell = nn.GRUCell(c.facet_dim + c.root_dim, c.facet_dim)
        self.carry_norm = RMSNorm(c.facet_dim, c.norm_eps)
        self.carry_slot_codes = nn.Parameter(torch.empty(c.carry_slots, c.facet_dim))
        nn.init.normal_(self.carry_slot_codes, std=0.02)
        for name, tensor in self._pair_topology().items():
            self.register_buffer(name, tensor, persistent=False)
        self._u_routing_losses: list[tuple[Tensor, Tensor]] = []

    def compile_kernels(self, mode: str = "default") -> None:
        """Compile tensor-only operators without wrapping stateful policy logic.

        ``Module.compile`` retains normal state-dict names and parameter identity.
        Group dispatch and immutable-state orchestration stay explicit. The
        hardware validation command must test this setting before a long run.
        """
        for stage in self.stages:
            stage.always.compile(mode=mode, dynamic=True)
        self.u_always.compile(mode=mode, dynamic=True)
        self.control_encoder.compile(mode=mode, dynamic=True)
        self.carry_cell.compile(mode=mode, dynamic=True)

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
        dtype: torch.dtype | None = None,
        model_version: int | None = None,
        episode_ids: Tensor | None = None,
    ) -> FactualState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        c = self.config
        device = torch.device(device or self.facet_codes.device)
        dtype = dtype or self.facet_codes.dtype
        version = self.model_version if model_version is None else model_version
        if episode_ids is None:
            episode_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        if episode_ids.shape != (batch_size,):
            raise ValueError("episode_ids must have one stable identity per batch row")

        def empty_memory(heads: int) -> KVMemory:
            return KVMemory(
                torch.empty(batch_size, heads, 0, c.attention_head_dim, device=device, dtype=dtype),
                torch.empty(batch_size, heads, 0, c.attention_head_dim, device=device, dtype=dtype),
                torch.empty(batch_size, 0, device=device, dtype=torch.bool),
            )

        return FactualState(
            facets=torch.zeros(batch_size, c.n_facets, c.facet_dim, device=device, dtype=dtype),
            recurrent=tuple(site.initial_state(batch_size, device) for site in self.gdn),
            history_a=empty_memory(c.historical_kv_heads),
            history_b=empty_memory(c.historical_kv_heads),
            literal=empty_memory(c.live_kv_heads),
            source=torch.empty(batch_size, 0, c.source_dim, device=device, dtype=dtype),
            source_mask=torch.empty(batch_size, 0, device=device, dtype=torch.bool),
            carry=torch.zeros(batch_size, c.carry_slots, c.facet_dim, device=device, dtype=dtype),
            records_seen=torch.zeros(batch_size, device=device, dtype=torch.long),
            last_record_ids=torch.full((batch_size,), -1, device=device, dtype=torch.long),
            episode_ids=episode_ids.to(device=device, dtype=torch.long),
            model_version=version,
            routing_version=c.routing_version,
            routing_aux_loss=torch.zeros((), device=device),
        )

    def set_model_version(self, version: int) -> None:
        if version < 0:
            raise ValueError("Model version must be nonnegative")
        self.model_version = int(version)

    def _check_state(self, state: FactualState) -> None:
        assert_cache_version(state, self.model_version, self.config.routing_version)
        if state.facets.shape[1:] != (self.config.n_facets, self.config.facet_dim):
            raise ValueError("State belongs to an incompatible model geometry")

    def encode_control(self, byte_ids: Tensor, lengths: Tensor | None = None) -> Tensor:
        return self.control_encoder(self.frontend(byte_ids, lengths))

    @staticmethod
    def _row_counts(facets: int, effort: int) -> tuple[int, int]:
        return {
            1: (facets // 4, facets // 4),
            2: (facets // 2, facets // 2),
            3: (facets, facets // 2),
            6: (3 * facets // 2, 3 * facets // 2),
        }[effort]

    def _pair_stage(
        self, facets: Tensor, state: FactualState, stage: int, route: RowRoute, effort: int
    ) -> Tensor:
        wrapper = self.stages[stage - 1]
        facets = self.memory(facets + self.stage_codes[stage - 1], state)
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
        return wrapper.output_norm(facets)

    def _execute_pair_stage(
        self, facets: Tensor, state: FactualState, stage: int, route: RowRoute, effort: int
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
                lambda x: self._pair_stage(x, state, stage, route, effort),
                facets,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return self._pair_stage(facets, state, stage, route, effort)

    def ingest(
        self,
        byte_ids: Tensor,
        state: FactualState,
        lengths: Tensor | None = None,
        record_ids: Tensor | None = None,
        role_ids: Tensor | None = None,
        deterministic: bool = True,
    ) -> FactualState:
        self._check_state(state)
        if byte_ids.shape[0] != state.batch_size:
            raise ValueError("Input and state batch sizes differ")
        source = self.frontend(byte_ids, lengths)
        if record_ids is None:
            record_ids = state.last_record_ids + 1
        record_ids = record_ids.to(device=byte_ids.device, dtype=torch.long)
        if record_ids.shape != state.last_record_ids.shape or bool(
            (record_ids <= state.last_record_ids).any()
        ):
            raise ValueError(
                "Physical record IDs must be strictly increasing; duplicate ingestion is forbidden"
            )
        if role_ids is None:
            role_ids = torch.full_like(record_ids, ROLE_IDS["document"])
        if role_ids.shape != record_ids.shape or bool(
            ((role_ids < 0) | (role_ids >= len(RECORD_ROLES))).any()
        ):
            raise ValueError(
                "role_ids must identify one supported causal record role per batch row"
            )
        role = self.role_embedding(role_ids.to(byte_ids.device, torch.long))
        facets = self.stem_norm(state.facets + self.facet_codes + role[:, None])
        facets = facets + self.source_attention(facets, source.features, source.mask)
        context = facets.mean(1) + self.source_context(masked_mean(source.features, source.mask))
        identities = state.episode_ids * 1_000_003 + record_ids
        program = self.p_router.compile(
            context,
            tuple(range(1, 7)),
            3 * self.config.n_facets,
            epoch="factual_1_6",
            record_ids=identities,
            allow_exploration=not deterministic,
        )
        recurrent = list(state.recurrent)
        for stage in range(1, 7):
            facets = self._execute_pair_stage(facets, state, stage, program.at(stage), 6)
            if stage in self.config.gdn_sites:
                index = self.config.gdn_sites.index(stage)
                facets, recurrent[index] = self.gdn[index](facets, recurrent[index], write=True)
        history_a, history_b, literal = self.memory.write(
            source.features, source.mask, facets, state
        )
        recent_source = torch.cat((state.source, source.features), dim=1)[
            :, -self.config.literal_window :
        ]
        recent_mask = torch.cat((state.source_mask, source.mask), dim=1)[
            :, -self.config.literal_window :
        ]
        return replace(
            state,
            facets=facets,
            recurrent=tuple(recurrent),
            history_a=history_a,
            history_b=history_b,
            literal=literal,
            source=recent_source,
            source_mask=recent_mask,
            records_seen=state.records_seen + 1,
            last_record_ids=record_ids,
            routing_aux_loss=program.aux_loss,
            last_program=program,
        )

    def begin_prediction(
        self,
        state: FactualState,
        root: Tensor,
        control: Tensor | None = None,
        effort: int = 6,
        deterministic: bool = True,
    ) -> PredictionState:
        self._check_state(state)
        if effort not in EFFORTS:
            raise ValueError(f"Predictive effort must be one of {EFFORTS}")
        c = self.config
        self._u_routing_losses = []
        if root.shape != (state.batch_size, c.root_dim):
            raise ValueError("Root innovation has the wrong shape")
        if control is None:
            control = state.facets.new_zeros((state.batch_size, c.control_dim))
        if control.shape != (state.batch_size, c.control_dim):
            raise ValueError("Causal control has the wrong shape")
        root, control = root.to(state.facets.dtype), control.to(state.facets.dtype)
        effort_code = self.effort_embedding.weight[EFFORTS.index(effort)]
        innovation = (
            self.root_injection(root)
            + self.control_injection(control)
            + self.carry_read(self.carry_norm(state.carry).mean(1))
            + effort_code
        )
        facets = state.facets + innovation[:, None]
        context = state.facets.mean(1) + innovation
        identities = state.episode_ids * 1_000_003 + state.last_record_ids
        program = self.p_router.compile(
            context,
            (7, 8, 9, 10),
            3 * c.n_facets,
            epoch="prediction_7_10",
            record_ids=identities,
            allow_exploration=not deterministic,
        )
        return PredictionState(
            state, facets, root, control, effort, 6, program, (), context, deterministic
        )

    def advance_prediction(
        self, prediction: PredictionState, stop_stage: int, conditioning: Tensor | None = None
    ) -> PredictionState:
        self._check_state(prediction.factual)
        if stop_stage not in PREDICTION_BOUNDARIES or not prediction.stage < stop_stage:
            raise ValueError(
                f"Advance monotonically to a causal boundary in {PREDICTION_BOUNDARIES}"
            )
        facets, context = prediction.facets, prediction.route_context
        if conditioning is not None:
            if conditioning.shape != (facets.shape[0], self.config.facet_dim):
                raise ValueError(
                    "Conditioning must encode actually chosen ancestors as [batch,facet_dim]"
                )
            injected = self.conditioning_injection(conditioning.to(facets.dtype))
            facets = facets + injected[:, None]
            if prediction.stage <= 10:
                if prediction.stage == 10 and prediction.program.epoch == "after_stage_10":
                    raise ValueError(
                        "Stage-10 ancestors were already compiled; pass them once to refresh_prediction"
                    )
                context = context + injected
        program, previous = prediction.program, prediction.previous_programs
        for stage in range(prediction.stage + 1, stop_stage + 1):
            if stage == 11 and program.epoch != "after_stage_10":
                previous = (*previous, program)
                identities = (
                    prediction.factual.episode_ids * 1_000_003 + prediction.factual.last_record_ids
                )
                program = self.p_router.compile(
                    context,
                    tuple(range(11, 17)),
                    3 * self.config.n_facets,
                    epoch="after_stage_10",
                    record_ids=identities,
                    allow_exploration=not prediction.deterministic,
                )
            facets = self._execute_pair_stage(
                facets, prediction.factual, stage, program.at(stage), prediction.effort
            )
            site = (stage - 7) % len(self.gdn)
            facets, _ = self.gdn[site](facets, prediction.factual.recurrent[site], write=False)
        return replace(
            prediction,
            facets=facets,
            stage=stop_stage,
            program=program,
            previous_programs=previous,
            route_context=context,
        )

    def refresh_prediction(
        self, prediction: PredictionState, conditioning: Tensor
    ) -> PredictionState:
        """Compile future routes after the actual stage-10 sampled ancestors.

        This explicit boundary also lets output growth use the refreshed program
        before stages 11–15. Calling it twice or at another stage is an error.
        """
        if prediction.stage != 10 or prediction.program.epoch == "after_stage_10":
            raise ValueError("Prediction routes can refresh once, after stage-10 choices")
        if conditioning.shape != prediction.route_context.shape:
            raise ValueError("Stage-10 conditioning must have shape [batch, facet_dim]")
        injected = self.conditioning_injection(conditioning.to(prediction.facets.dtype))
        context = prediction.route_context + injected
        identities = prediction.factual.episode_ids * 1_000_003 + prediction.factual.last_record_ids
        program = self.p_router.compile(
            context,
            tuple(range(11, 17)),
            3 * self.config.n_facets,
            epoch="after_stage_10",
            record_ids=identities,
            allow_exploration=not prediction.deterministic,
        )
        return replace(
            prediction,
            facets=prediction.facets + injected[:, None],
            route_context=context,
            program=program,
            previous_programs=(*prediction.previous_programs, prediction.program),
        )

    def apply_u(
        self, rows: Tensor, prediction: PredictionState, level: int, refinement: bool = False
    ) -> Tensor:
        """Apply shared routed+dense U256 weights at a capacity growth level."""
        self._check_state(prediction.factual)
        if (
            rows.ndim != 3
            or rows.shape[0] != prediction.facets.shape[0]
            or rows.shape[-1] != self.config.facet_dim
        ):
            raise ValueError("U256 expects [batch, rows, facet_dim]")
        if level < 1 or rows.shape[1] < 1:
            raise ValueError("Growth level and number of rows must be positive")
        if refinement:
            return self.lexical_refine(rows)
        eligible = min(self.config.eligible_experts, 4 if rows.shape[1] < 64 else 8, rows.shape[1])
        identities = prediction.factual.episode_ids * 1_000_003 + prediction.factual.last_record_ids
        route = self.u_router.route(
            prediction.route_context,
            level,
            rows.shape[1],
            eligible_limit=eligible,
            record_ids=identities,
            allow_exploration=not prediction.deterministic,
        )
        self.u_bank.record_routes(
            route.expert_ids, route.eligible, gradient_enabled=torch.is_grad_enabled()
        )
        self._u_routing_losses.append((route.balance_loss, route.stabilization_loss))
        normalized = self.u_norm(rows)
        delta = self.u_bank(normalized, route.expert_ids, route.confidence) + self.u_always(
            normalized
        )
        return rows + self.u_gate.sigmoid() * delta

    def lexical_refine(self, rows: Tensor) -> Tensor:
        """The final dense refinement aliases the SAME U always branch."""
        return rows + self.lexical_gate.sigmoid() * self.u_always(self.lexical_norm(rows))

    def dispatch_metrics(self, *, reset: bool = False) -> dict[str, dict[str, Tensor]]:
        return {"p": self.p_bank.metrics(reset=reset), "u": self.u_bank.metrics(reset=reset)}

    def u_routing_losses(self) -> tuple[Tensor, Tensor]:
        """Unweighted U-family auxiliary losses for the current prediction.

        Consume when forming that decision's objective, before starting another
        prediction. These diagnostics never feed routing selection or caches.
        """
        if not self._u_routing_losses:
            zero = self.facet_codes.sum() * 0
            return zero, zero
        balance, stabilization = zip(*self._u_routing_losses, strict=True)
        return torch.stack(balance).mean(), torch.stack(stabilization).mean()

    def refresh_output(
        self, prediction: PredictionState, summary: Tensor, *, after_scale: int = 64
    ) -> PredictionState:
        if after_scale != 64 or prediction.output_refreshed:
            raise ValueError("One output routing refresh is permitted immediately after scale 64")
        if summary.shape != prediction.route_context.shape:
            raise ValueError("Output refresh summary must have shape [batch, facet_dim]")
        return replace(
            prediction,
            route_context=prediction.route_context + self.conditioning_injection(summary),
            output_refreshed=True,
        )

    def reconcile_carry(
        self, state: FactualState, content: Tensor, innovation: Tensor
    ) -> FactualState:
        self._check_state(state)
        c = self.config
        if content.shape != (state.batch_size, c.facet_dim) or innovation.shape != (
            state.batch_size,
            c.root_dim,
        ):
            raise ValueError(
                "Carry reconciliation needs committed content and the actual sampled root"
            )
        inputs = torch.cat((content.to(state.carry.dtype), innovation.to(state.carry.dtype)), -1)
        inputs = inputs[:, None].expand(-1, c.carry_slots, -1).reshape(-1, c.facet_dim + c.root_dim)
        previous = state.carry + self.carry_slot_codes
        carry = self.carry_cell(inputs, previous.reshape(-1, c.facet_dim)).reshape_as(state.carry)
        return replace(state, carry=self.carry_norm(carry))

    def reset_state(
        self, state: FactualState, rows: Tensor | None = None, episode_ids: Tensor | None = None
    ) -> FactualState:
        self._check_state(state)
        if rows is None:
            return self.initial_state(
                state.batch_size, state.facets.device, state.facets.dtype, episode_ids=episode_ids
            )
        if rows.shape != (state.batch_size,) or rows.dtype != torch.bool:
            raise ValueError("Selective resets require a boolean vector over episodes")
        keep = ~rows
        ids = state.episode_ids if episode_ids is None else episode_ids.to(state.episode_ids)
        return replace(
            state,
            facets=state.facets * keep[:, None, None],
            recurrent=tuple(s * keep[:, None, None, None, None] for s in state.recurrent),
            history_a=state.history_a.reset(rows),
            history_b=state.history_b.reset(rows),
            literal=state.literal.reset(rows),
            source=state.source * keep[:, None, None],
            source_mask=state.source_mask & keep[:, None],
            carry=state.carry * keep[:, None, None],
            records_seen=state.records_seen * keep,
            last_record_ids=torch.where(rows, -1, state.last_record_ids),
            episode_ids=torch.where(rows, ids, state.episode_ids),
            last_program=None,
            routing_aux_loss=state.routing_aux_loss.new_zeros(()),
        )
