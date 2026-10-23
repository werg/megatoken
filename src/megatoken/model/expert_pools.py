"""Shared shape-bucketed expert pools and hierarchical routing.

``ShapePools`` owns every expert bank and selector codebook exactly once. Sites
hold non-owning references to their pool and site path. One small bootstrap
expert bank computes routing features per event, and each shared shape pool
uses a cached eligible palette with a deterministic site phase.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_CURRENT_ADAPTER_WEIGHTS: ContextVar[Tensor | None] = ContextVar(
    "megatoken_expert_adapter_weights", default=None
)
_EXPERT_REPLAY: ContextVar[bool] = ContextVar("megatoken_expert_replay", default=False)
_CURRENT_ROUTING_PLAN: ContextVar[object | None] = ContextVar(
    "megatoken_shared_routing_plan", default=None
)
_INHERIT_ADAPTER = object()


@contextmanager
def expert_scope(weights: Tensor | None) -> Iterator[None]:
    """Set retrieval-conditioned adapter weights for the current forward pass.

    Weights have shape ``[..., adapter_experts]``.  Each routed site must have
    leading dimensions matching that prefix; any remaining row dimensions are
    repeated per event.  A mismatch raises instead of mixing sessions silently.
    """

    if weights is not None:
        if weights.ndim < 2 or not weights.is_floating_point():
            raise ValueError("Expert scope weights must be floating [..., adapters]")
    token = _CURRENT_ADAPTER_WEIGHTS.set(weights)
    try:
        yield
    finally:
        _CURRENT_ADAPTER_WEIGHTS.reset(token)


def current_adapter_weights(rows: Tensor) -> Tensor | None:
    """Return scope weights expanded to a bank's row geometry."""

    return _adapter_rows(_CURRENT_ADAPTER_WEIGHTS.get(), tuple(rows.shape[:-1]), rows.device)


def flatten_expert_scope(weights: Tensor) -> Tensor:
    """Flatten explicit batch/event mixture dimensions before flattening rows.

    Use this when activations are reshaped from ``[B,E,...,D]`` to
    ``[B*E,...,D]``. Row ordering must use the same contiguous B/E order.
    """

    if weights.ndim < 2:
        raise ValueError("Expert scope weights must include a batch/event axis")
    return weights.reshape(-1, weights.shape[-1])


def checkpoint_context_fn():
    """Return the ``context_fn`` expected by non-reentrant checkpointing."""

    # ContextVar state is thread/task local.  Capture the value now so backward
    # replay sees the same adapter mixture even if the caller's scope has ended.
    weights = _CURRENT_ADAPTER_WEIGHTS.get()
    active_plan = _CURRENT_ROUTING_PLAN.get()
    palette_snapshot = None if active_plan is None else dict(active_plan.palette_cache)

    def replay_plan():
        if active_plan is None:
            return None
        clone = _RoutingPlan(
            active_plan.pools,
            active_plan.features,
            active_plan.router_features,
            active_plan.adapter_weights,
            active_plan.stage,
            context_shape=active_plan.context_shape,
        )
        clone.palette_cache.update(palette_snapshot or {})
        return clone

    @contextmanager
    def with_weights():
        token = _CURRENT_ADAPTER_WEIGHTS.set(weights)
        route_token = _CURRENT_ROUTING_PLAN.set(active_plan)
        try:
            yield
        finally:
            _CURRENT_ROUTING_PLAN.reset(route_token)
            _CURRENT_ADAPTER_WEIGHTS.reset(token)

    @contextmanager
    def with_replay():
        token = _CURRENT_ADAPTER_WEIGHTS.set(weights)
        route_token = _CURRENT_ROUTING_PLAN.set(replay_plan())
        replay_token = _EXPERT_REPLAY.set(True)
        try:
            yield
        finally:
            _EXPERT_REPLAY.reset(replay_token)
            _CURRENT_ROUTING_PLAN.reset(route_token)
            _CURRENT_ADAPTER_WEIGHTS.reset(token)

    return with_weights(), with_replay()


def _bucket(value: int, base: int) -> int:
    if value <= base:
        return base
    return base * (1 << math.ceil(math.log2(value / base)))


def _module_key(prefix: str, kind: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in prefix)
    return f"{safe or 'site'}__{kind}"


def _adapter_rows(
    weights: Tensor | None, leading: tuple[int, ...], device: torch.device
) -> Tensor | None:
    if weights is None:
        return None
    if weights.device != device:
        raise ValueError("Expert scope and routed rows must be on the same device")
    scope_leading = tuple(weights.shape[:-1])
    if len(scope_leading) > len(leading) or tuple(leading[: len(scope_leading)]) != scope_leading:
        raise ValueError(
            f"Expert scope prefix {scope_leading} does not align with routed rows {leading}"
        )
    expanded = weights
    for _ in range(len(leading) - len(scope_leading)):
        expanded = expanded.unsqueeze(-2)
    expanded = expanded.expand(*leading, weights.shape[-1])
    return expanded


def _fallback_context(rows: Tensor, adapter_weights: Tensor | None = None) -> Tensor:
    """Derive one deterministic standalone context per batch item."""
    if rows.ndim <= 2:
        return rows
    leading = tuple(rows.shape[:-1])
    if adapter_weights is not None:
        scope = tuple(adapter_weights.shape[:-1])
        if len(scope) > len(leading) or leading[: len(scope)] != scope:
            raise ValueError(
                f"Expert scope prefix {scope} does not align with routed rows {leading}"
            )
        reduce_from = len(scope)
    else:
        reduce_from = 1
    if reduce_from >= len(leading):
        return rows
    axes = tuple(range(reduce_from, rows.ndim - 1))
    return rows.mean(dim=axes)


class _ExecutedSiteRoute(NamedTuple):
    expert_ids: Tensor
    router_context: Tensor
    router_stage: int


class _Selector:
    """Balance-training facade over a pool-owned selector parameter set."""

    def __init__(self, pools: ShapePools, name: str, experts: int, kind: str):
        self.pools, self.name, self.experts, self.kind = pools, name, experts, kind
        self.balance_recorder = None

    def balance_probabilities(self, context: Tensor, stage: int = 0) -> Tensor:
        return self.pools._selector_probabilities(self, context, stage)

    def record_executed(self, route) -> None:
        if self.balance_recorder is not None and not _EXPERT_REPLAY.get():
            self.balance_recorder.capture(self, route)

    @property
    def weight(self) -> Tensor:
        if self.kind == "bootstrap":
            return self.pools.router_bootstrap_codes
        return self.pools.pool_codes[_module_key(self.name, "codes")]


class _RoutedSite(nn.Module):
    def _init_site(self, registry: ShapePools, pool_name: str, prefix: str) -> None:
        object.__setattr__(self, "_registry", registry)
        self._pool_name = pool_name
        self._site_prefix = prefix
        self._site_phase = int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:4], "little")

    @property
    def _bank(self) -> nn.Module:
        return self._registry.pools[self._pool_name]

    @property
    def _selector(self) -> _Selector:
        return self._registry.selector_for_pool(self._pool_name)

    # Compatibility for existing diagnostics; these are non-owning facades,
    # and no per-site selector weights are registered.
    @property
    def router(self):
        return self._selector

    @property
    def experts(self) -> int:
        return self._bank.experts

    @property
    def balance_recorder(self):
        return self._selector.balance_recorder

    @balance_recorder.setter
    def balance_recorder(self, value):
        self._selector.balance_recorder = value

    def balance_probabilities(self, context: Tensor, stage: int = 0) -> Tensor:
        return self._selector.balance_probabilities(context, stage)

    def record_executed(self, route) -> None:
        self._selector.record_executed(route)

    def _route(
        self, rows: Tensor, *, phase: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        plan = _CURRENT_ROUTING_PLAN.get()
        if plan is None or plan.pools is not self._registry:
            # Standalone module calls get a deterministic one-call context.
            context = _fallback_context(rows, _CURRENT_ADAPTER_WEIGHTS.get())
            with self._registry.routing_epoch_if_absent(context):
                return self._route(rows, phase=phase)
        ids, confidence, features = self._registry._route_rows(
            plan, self._pool_name, self._site_phase if phase is None else phase, rows
        )
        adapters = _adapter_rows(
            _CURRENT_ADAPTER_WEIGHTS.get(), tuple(rows.shape[:-1]), rows.device
        )
        return ids, confidence, adapters


class RoutedLinear(_RoutedSite):
    """A linear site routed through a shared, padded linear-expert pool."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        registry: ShapePools,
        pool_name: str,
        prefix: str,
        *,
        bias: Tensor | None,
        zero_initialized: bool = False,
    ):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias is not None else None
        if bias is not None:
            with torch.no_grad():
                self.bias.copy_(bias.detach())
        self.output_gain = nn.Parameter(torch.tensor(0.0 if zero_initialized else 1.0))
        self._init_site(registry, pool_name, prefix)

    @property
    def in_features(self) -> int:
        return self.input_dim

    @property
    def out_features(self) -> int:
        return self.output_dim

    @property
    def weight(self) -> Tensor:
        """Representative compatibility view; forward always uses the pool."""
        return self._bank.weight[0, : self.output_dim, : self.input_dim]

    def forward(self, x: Tensor) -> Tensor:
        padded = F.pad(x, (0, self._bank.dim - self.input_dim))
        expert_ids, confidence, adapters = self._route(x)
        output = self._bank(padded, expert_ids, confidence, adapter_weights=adapters)
        output = output[..., : self.output_dim] * self.output_gain.to(output.dtype)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output


class RoutedConv1d(_RoutedSite):
    """A groups=1 Conv1d whose kernel taps share one routed linear pool.

    Each tap is a D-to-output projection from the same canonical bank. The
    tap outputs are summed with fan-in normalization; an optional site bias is
    added once after the sum. Inputs and outputs retain PyTorch's [B,C,L]
    convention.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        registry: ShapePools,
        pool_name: str,
        prefix: str,
        *,
        stride: int = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: Tensor | None = None,
    ):
        super().__init__()
        if padding_mode != "zeros":
            raise NotImplementedError(
                f"RoutedConv1d only supports zero padding, got {padding_mode!r}"
            )
        if isinstance(padding, str) and padding not in {"same", "valid"}:
            raise ValueError(f"unsupported Conv1d padding: {padding!r}")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.kernel_size, self.stride, self.padding, self.dilation = (
            kernel_size,
            stride,
            padding,
            dilation,
        )
        self.padding_mode = padding_mode
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias is not None else None
        if bias is not None:
            with torch.no_grad():
                self.bias.copy_(bias.detach())
        self._tap_phases = tuple(
            int.from_bytes(hashlib.sha256(f"{prefix}.tap.{index}".encode()).digest()[:4], "little")
            for index in range(kernel_size)
        )
        self._init_site(registry, pool_name, prefix)

    @property
    def in_channels(self) -> int:
        return self.input_dim

    @property
    def out_channels(self) -> int:
        return self.output_dim

    @property
    def groups(self) -> int:
        return 1

    def _padding_pair(self) -> tuple[int, int]:
        if self.padding == "valid":
            return 0, 0
        if self.padding == "same":
            total = self.dilation * (self.kernel_size - 1)
            return total // 2, total - total // 2
        if isinstance(self.padding, tuple):
            if len(self.padding) != 1:
                raise ValueError("Conv1d padding tuple must contain one value")
            value = int(self.padding[0])
        else:
            value = int(self.padding)
        return value, value

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"RoutedConv1d expects [batch, channels, length], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(f"RoutedConv1d expected {self.input_dim} channels, got {x.shape[1]}")
        sequence = x.transpose(1, 2)
        left, right = self._padding_pair()
        if left or right:
            sequence = F.pad(sequence.transpose(1, 2), (left, right)).transpose(1, 2)
        effective = self.dilation * (self.kernel_size - 1) + 1
        windows = sequence.unfold(1, effective, self.stride)
        if windows.shape[1] == 0:
            raise ValueError("RoutedConv1d kernel is longer than the padded input")

        # Establish one shared epoch around all kernel taps when called alone.
        plan = _CURRENT_ROUTING_PLAN.get()
        if plan is None or plan.pools is not self._registry:
            context = _fallback_context(sequence, _CURRENT_ADAPTER_WEIGHTS.get())
            with self._registry.routing_epoch_if_absent(context):
                return self.forward(x)

        result = None
        for index, phase in enumerate(self._tap_phases):
            rows = windows[..., index * self.dilation]
            padded = F.pad(rows, (0, self._bank.dim - self.input_dim))
            ids, confidence, adapters = self._route(rows, phase=phase)
            tap_output = self._bank(padded, ids, confidence, adapter_weights=adapters)
            tap_output = tap_output[..., : self.output_dim]
            result = tap_output if result is None else result + tap_output
        result = result * (self.kernel_size**-0.5)
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result.transpose(1, 2)


class RoutedSwiGLU(_RoutedSite):
    """A SwiGLU-shaped site routed through a shared expert FFN pool."""

    def __init__(self, dim: int, registry: ShapePools, pool_name: str, prefix: str):
        super().__init__()
        self.dim = dim
        self._init_site(registry, pool_name, prefix)

    @property
    def gate_up(self) -> nn.Module:
        return _WeightView(self._bank, "gate_up")

    @property
    def down(self) -> nn.Module:
        return _WeightView(self._bank, "down")

    def forward(self, x: Tensor) -> Tensor:
        padded = F.pad(x, (0, self._bank.dim - self.dim))
        expert_ids, confidence, adapters = self._route(x)
        result = self._bank(padded, expert_ids, confidence, adapter_weights=adapters)
        return result[..., : self.dim]


class _WeightView:
    """Read-only first-expert compatibility view for old dtype/shape probes."""

    def __init__(self, bank: nn.Module, name: str):
        self.bank, self.name = bank, name

    @property
    def weight(self) -> Tensor:
        return getattr(self.bank, self.name)[0]


class RoutedGRUCell(_RoutedSite):
    """GRUCell with shared routed input/hidden gate projections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        registry: ShapePools,
        input_pool_name: str,
        hidden_pool_name: str,
        prefix: str,
        bias_ih: Tensor | None = None,
        bias_hh: Tensor | None = None,
    ):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.input_bias = (
            nn.Parameter(torch.zeros(3 * hidden_size)) if bias_ih is not None else None
        )
        self.hidden_bias = (
            nn.Parameter(torch.zeros(3 * hidden_size)) if bias_hh is not None else None
        )
        if bias_ih is not None:
            with torch.no_grad():
                self.input_bias.copy_(bias_ih.detach())
        if bias_hh is not None:
            with torch.no_grad():
                self.hidden_bias.copy_(bias_hh.detach())
        self._init_site(registry, input_pool_name, prefix + ".input")
        self.hidden_pool_name = hidden_pool_name
        self.hidden_site_phase = int.from_bytes(
            hashlib.sha256((prefix + ".hidden").encode()).digest()[:4], "little"
        )

    @property
    def balance_recorder(self):
        return self._selector.balance_recorder

    @balance_recorder.setter
    def balance_recorder(self, value):
        self._selector.balance_recorder = value
        self._registry.selector_for_pool(self.hidden_pool_name).balance_recorder = value

    def forward(self, x: Tensor, hx: Tensor | None = None) -> Tensor:
        plan = _CURRENT_ROUTING_PLAN.get()
        if plan is None or plan.pools is not self._registry:
            context = _fallback_context(x, _CURRENT_ADAPTER_WEIGHTS.get())
            with self._registry.routing_epoch_if_absent(context):
                return self.forward(x, hx)
        if hx is None:
            hx = x.new_zeros((*x.shape[:-1], self.hidden_size))
        in_pad = F.pad(x, (0, self._bank.dim - self.input_size))
        hidden_bank = self._registry.pools[self.hidden_pool_name]
        hidden_pad = F.pad(hx, (0, hidden_bank.dim - self.hidden_size))
        in_ids, in_conf, adapters = self._route(x)
        input_gates = self._bank(in_pad, in_ids, in_conf, adapter_weights=adapters)
        hidden_ids, hidden_conf, _ = self._registry._route_rows(
            _CURRENT_ROUTING_PLAN.get(), self.hidden_pool_name, self.hidden_site_phase, hx
        )
        hidden_gates = hidden_bank(hidden_pad, hidden_ids, hidden_conf, adapter_weights=adapters)
        input_gates = input_gates[..., : 3 * self.hidden_size]
        hidden_gates = hidden_gates[..., : 3 * self.hidden_size]
        if self.input_bias is not None:
            input_gates = input_gates + self.input_bias
        if self.hidden_bias is not None:
            hidden_gates = hidden_gates + self.hidden_bias
        i_r, i_z, i_n = input_gates.chunk(3, -1)
        h_r, h_z, h_n = hidden_gates.chunk(3, -1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        candidate = torch.tanh(i_n + reset * h_n)
        return candidate + update * (hx - candidate)


class _RoutingPlan:
    def __init__(self, pools, features, router_features, weights, stage, *, context_shape=None):
        self.pools = pools
        self.features = features
        self.router_features = router_features
        self.adapter_weights = weights
        self.stage = stage
        self.context_shape = tuple(context_shape or (router_features.shape[0],))
        self.palette_cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}


class ShapePools(nn.Module):
    """One owner for banks shared by every bound learned site."""

    def __init__(self, config):
        super().__init__()
        from .experts import ExpertBank, LinearExpertBank

        self.config = config
        self.facet_dim = config.facet_dim
        self.projection_experts = config.projection_experts
        self.adapter_experts = config.adapter_experts
        self.adapter_rank = config.adapter_rank
        self.router_dim = int(getattr(config, "router_dim", 64))
        self.router_experts = int(getattr(config, "router_experts", 8))
        self.eligible_experts = int(getattr(config, "eligible_experts", 4))
        self._linear_keys: dict[tuple[int, int], str] = {}
        self._swiglu_keys: dict[int, str] = {}
        self._selectors: dict[str, _Selector] = {}
        self._site_manifest: list[dict[str, object]] = []
        self._exclusions: list[dict[str, str]] = []
        self.pools = nn.ModuleDict()
        self._ExpertBank, self._LinearExpertBank = ExpertBank, LinearExpertBank
        self.register_buffer("_pool_anchor", torch.empty(0), persistent=False)
        self.router_bank = self._ExpertBank(
            self.router_experts,
            self.router_dim,
            hidden=2 * self.router_dim,
            backend=config.expert_backend,
            family="shared_router",
            adapter_experts=self.adapter_experts,
            adapter_rank=self.adapter_rank,
            adapter_topk=config.adapter_topk,
        )
        self.router_bootstrap_codes = nn.Parameter(
            torch.empty(self.router_experts, self.router_dim)
        )
        nn.init.normal_(self.router_bootstrap_codes, std=0.02)
        self.router_semantic_projection = nn.Linear(
            getattr(config, "control_dim", config.facet_dim), self.router_dim, bias=False
        )
        self.router_role_embedding = nn.Embedding(5, self.router_dim)
        self.router_stage_embedding = nn.Embedding(64, self.router_dim)
        nn.init.normal_(self.router_role_embedding.weight, std=0.02)
        nn.init.normal_(self.router_stage_embedding.weight, std=0.02)
        self.pool_codes = nn.ParameterDict()
        self.router_evaluations = 0
        self.palette_evaluations = 0
        self.expert_rows = 0

    def _linear_bank(self, input_dim: int, output_dim: int, prefix: str) -> nn.Module:
        in_bucket = _bucket(input_dim, self.facet_dim)
        out_bucket = _bucket(output_dim, self.facet_dim)
        key = (in_bucket, out_bucket)
        if key not in self._linear_keys:
            name = _module_key(f"linear_{in_bucket}_{out_bucket}", "pool")
            experts = self.projection_experts
            bank = self._LinearExpertBank(
                experts,
                in_bucket,
                out_bucket,
                backend=self.config.expert_backend,
                family=name,
                adapter_experts=self.adapter_experts,
                adapter_rank=self.adapter_rank,
                adapter_topk=self.config.adapter_topk,
            ).to(device=self._pool_anchor.device, dtype=self._pool_anchor.dtype)
            self.pools[name] = bank
            self._linear_keys[key] = name
            self._add_pool_selector(name, experts)
        return self.pools[self._linear_keys[key]]

    def _swiglu_bank(self, dim: int) -> nn.Module:
        bucket = _bucket(dim, self.facet_dim)
        if bucket not in self._swiglu_keys:
            name = _module_key(f"swiglu_{bucket}", "pool")
            experts = (
                self.config.p_experts if bucket >= 2 * self.facet_dim else self.config.u_experts
            )
            bank = self._ExpertBank(
                experts,
                bucket,
                hidden=2 * bucket,
                backend=self.config.expert_backend,
                family=name,
                adapter_experts=self.adapter_experts,
                adapter_rank=self.adapter_rank,
                adapter_topk=self.config.adapter_topk,
            ).to(device=self._pool_anchor.device, dtype=self._pool_anchor.dtype)
            self.pools[name] = bank
            self._swiglu_keys[bucket] = name
            self._add_pool_selector(name, experts)
        return self.pools[self._swiglu_keys[bucket]]

    def _add_pool_selector(self, name: str, experts: int) -> None:
        code_name = _module_key(name, "codes")
        codes = nn.Parameter(
            torch.empty(
                experts,
                self.router_dim,
                device=self._pool_anchor.device,
                dtype=self._pool_anchor.dtype,
            )
        )
        nn.init.normal_(codes, std=0.02)
        self.pool_codes[code_name] = codes
        self._selectors[name] = _Selector(self, name, experts, "pool")

    @property
    def bootstrap_selector(self) -> _Selector:
        if "bootstrap" not in self._selectors:
            self._selectors["bootstrap"] = _Selector(
                self, "bootstrap", self.router_experts, "bootstrap"
            )
        return self._selectors["bootstrap"]

    def selector_for_pool(self, name: str) -> _Selector:
        return self._selectors[name]

    def routing_selectors(self) -> dict[str, _Selector]:
        """Return one bootstrap selector and one selector per shared shape pool."""
        return {
            "shape_bootstrap": self.bootstrap_selector,
            **{
                f"shape_pool_{name}": selector
                for name, selector in self._selectors.items()
                if name != "bootstrap"
            },
        }

    def router_features(
        self,
        context: Tensor,
        *,
        track_ids: Tensor | None = None,
        track_semantics: Tensor | None = None,
        role: str = "other",
        stage: int = 0,
    ) -> Tensor:
        """Fold arbitrary context widths to the small router width and add codes."""
        if context.ndim < 1:
            raise ValueError("Routing context needs a feature axis")
        width = context.shape[-1]
        if width < self.router_dim:
            folded = F.pad(context, (0, self.router_dim - width))
        else:
            pieces = math.ceil(width / self.router_dim)
            folded = F.pad(context, (0, pieces * self.router_dim - width))
            folded = folded.reshape(*context.shape[:-1], pieces, self.router_dim).mean(-2)
        if track_ids is not None and track_ids.shape != context.shape[:-1]:
            raise ValueError("track_ids must match routing context leading dimensions")
        semantic = 0
        if track_semantics is not None:
            if track_semantics.shape != (
                *context.shape[:-1],
                self.router_semantic_projection.in_features,
            ):
                raise ValueError("Track semantics must match routing context leading dimensions")
            semantic = self.router_semantic_projection(
                track_semantics.to(self.router_semantic_projection.weight.dtype)
            )
        role = role.lower()
        roles = {"e": 0, "c": 1, "p": 2, "r": 3, "other": 4}
        try:
            role_id = roles[role]
        except KeyError as exc:
            raise ValueError(f"Unknown routing role {role!r}") from exc
        if not 0 <= int(stage) < self.router_stage_embedding.num_embeddings:
            raise ValueError("Routing stage must be in [0, 63]")
        folded = folded.to(self.router_bootstrap_codes.dtype)
        enriched = (
            folded
            + semantic
            + self.router_role_embedding.weight[role_id]
            + self.router_stage_embedding.weight[int(stage)]
        )
        return F.layer_norm(enriched, (self.router_dim,))

    def controller_features(
        self,
        context: Tensor,
        *,
        track_ids: Tensor | None = None,
        track_semantics: Tensor | None = None,
        role: str = "R",
        stage: int = 0,
        reuse_active: bool = False,
        adapter_weights: Tensor | None | object = _INHERIT_ADAPTER,
        precomputed_features: Tensor | None = None,
    ) -> Tensor:
        """Fold lookup context, then run the shared router expert bank once.

        Adapter controllers can compute this once for a lookup and fan the
        returned small feature to their four inexpensive role selectors. An
        explicit ``adapter_weights=None`` keeps that controller lookup
        unadapted and prevents recursion; ordinary routing epochs inherit the
        active committed mixture so the shared router bank adapts with other
        learned projections.
        """
        active = _CURRENT_ROUTING_PLAN.get()
        if reuse_active and active is not None and active.pools is self:
            expected = math.prod(context.shape[:-1])
            if active.router_features.shape[0] != expected:
                raise ValueError(
                    "Active routing feature batch does not match the requested context"
                )
            return active.router_features
        features = (
            precomputed_features
            if precomputed_features is not None
            else self.router_features(
                context,
                track_ids=track_ids,
                track_semantics=track_semantics,
                role=role,
                stage=stage,
            )
        )
        flat = features.reshape(-1, self.router_dim)
        if adapter_weights is _INHERIT_ADAPTER:
            adapter_weights = _CURRENT_ADAPTER_WEIGHTS.get()
        adapter_rows = _adapter_rows(adapter_weights, tuple(features.shape[:-1]), features.device)
        probabilities = self._selector_probabilities(self.bootstrap_selector, flat, 0)
        confidence, ids = probabilities.max(-1)
        output = self.router_bank(
            flat,
            ids,
            0.75 + 0.25 * confidence,
            adapter_weights=None
            if adapter_rows is None
            else adapter_rows.reshape(-1, self.adapter_experts),
        )
        if not _EXPERT_REPLAY.get():
            self.router_evaluations += 1
            self.router_bank.record_routes(
                ids,
                ids[:, None],
                gradient_enabled=torch.is_grad_enabled(),
            )
        selector = self.bootstrap_selector
        if selector.balance_recorder is not None and not _EXPERT_REPLAY.get():
            selector.record_executed(_ExecutedSiteRoute(ids[:, None].detach(), flat.detach(), 0))
        return output

    def _selector_probabilities(self, selector: _Selector, context: Tensor, stage: int) -> Tensor:
        if selector.kind == "bootstrap":
            if stage != 0:
                raise ValueError("Bootstrap selector replay only supports stage 0")
            feature = context.to(self.router_bootstrap_codes.dtype)
            return (
                feature.float() @ self.router_bootstrap_codes.float().T / math.sqrt(self.router_dim)
            ).softmax(-1)
        if stage != 0:
            raise ValueError("Pool selector replay only supports stage 0")
        codes = self.pool_codes[_module_key(selector.name, "codes")]
        return (context.float() @ codes.float().T / math.sqrt(self.router_dim)).softmax(-1)

    @contextmanager
    def routing_epoch(
        self,
        context: Tensor,
        *,
        track_ids: Tensor | None = None,
        track_semantics: Tensor | None = None,
        role: str = "other",
        stage: int = 0,
        adapter_weights: Tensor | None | object = _INHERIT_ADAPTER,
    ):
        features = self.router_features(
            context, track_ids=track_ids, track_semantics=track_semantics, role=role, stage=stage
        )
        flat = features.reshape(-1, self.router_dim)
        router_output = self.controller_features(
            context,
            track_ids=track_ids,
            role=role,
            stage=stage,
            adapter_weights=adapter_weights,
            precomputed_features=features,
        )
        if adapter_weights is _INHERIT_ADAPTER:
            adapter_weights = _CURRENT_ADAPTER_WEIGHTS.get()
        plan = _RoutingPlan(
            self,
            flat,
            router_output,
            adapter_weights,
            stage,
            context_shape=tuple(features.shape[:-1]),
        )
        token = _CURRENT_ROUTING_PLAN.set(plan)
        try:
            with expert_scope(adapter_weights):
                yield plan
        finally:
            _CURRENT_ROUTING_PLAN.reset(token)

    @contextmanager
    def routing_epoch_if_absent(self, context: Tensor, **kwargs):
        active = _CURRENT_ROUTING_PLAN.get()
        if active is not None and active.pools is self:
            yield active
            return
        with self.routing_epoch(context, **kwargs) as plan:
            yield plan

    @contextmanager
    def routing_subset(self, indices: Tensor | list[int] | tuple[int, ...]):
        """Temporarily run a subset of active context rows with cached routing.

        ``indices`` address flattened rows of the active epoch context. This
        supports sequential per-example decode/workspace calls without
        recomputing the shared router or losing each example's cached palette.
        Within the context, routed activations must have a first leading axis
        equal to the number of selected indices.
        """
        active = _CURRENT_ROUTING_PLAN.get()
        if active is None or active.pools is not self:
            raise RuntimeError(
                "routing_subset requires an active routing epoch from this ShapePools"
            )
        index = torch.as_tensor(indices, device=active.router_features.device, dtype=torch.long)
        if index.ndim != 1 or index.numel() == 0:
            raise ValueError("routing_subset indices must be a non-empty one-dimensional selection")
        total = active.router_features.shape[0]
        valid = torch.all((index >= 0) & (index < total))
        if index.device.type == "meta":
            pass
        elif index.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(valid, "routing_subset index exceeds active context rows")
        elif not bool(valid):
            raise IndexError("routing_subset index exceeds active context rows")

        context_shape = active.context_shape
        if math.prod(context_shape) != total:
            raise RuntimeError("Active routing context geometry is inconsistent")
        weights = active.adapter_weights
        if weights is not None:
            expanded = _adapter_rows(weights, context_shape, active.router_features.device)
            weights = expanded.reshape(total, self.adapter_experts).index_select(0, index)
        subset = _RoutingPlan(
            self,
            active.features.index_select(0, index),
            active.router_features.index_select(0, index),
            weights,
            active.stage,
            context_shape=(index.numel(),),
        )
        for name, (palette, probabilities, features) in active.palette_cache.items():
            subset.palette_cache[name] = (
                palette.index_select(0, index),
                probabilities.index_select(0, index),
                features.index_select(0, index),
            )
        route_token = _CURRENT_ROUTING_PLAN.set(subset)
        try:
            with expert_scope(weights):
                yield subset
        finally:
            _CURRENT_ROUTING_PLAN.reset(route_token)

    def _route_rows(self, plan, pool_name: str, phase: int, rows: Tensor):
        leading = tuple(rows.shape[:-1])
        flat = rows.reshape(-1, rows.shape[-1])
        contexts = plan.router_features.shape[0]
        if contexts < 1 or flat.shape[0] % contexts:
            raise ValueError(
                f"Routed rows ({flat.shape[0]}) must divide evenly across routing contexts ({contexts})"
            )
        per_context = flat.shape[0] // contexts
        if pool_name not in plan.palette_cache:
            selector = self.selector_for_pool(pool_name)
            probabilities = selector.balance_probabilities(plan.router_features, 0)
            count = min(self.eligible_experts, selector.experts)
            palette = probabilities.topk(count, dim=-1).indices
            plan.palette_cache[pool_name] = (palette, probabilities, plan.router_features)
            if not _EXPERT_REPLAY.get():
                self.palette_evaluations += 1
        palette, probabilities, features = plan.palette_cache[pool_name]
        local_rows = torch.arange(per_context, device=rows.device)
        slots = (local_rows + phase) % palette.shape[-1]
        ids = palette[:, slots].reshape(-1)
        selected_prob = probabilities.gather(-1, ids.reshape(contexts, per_context))
        confidence = 0.75 + 0.25 * selected_prob
        if (
            self.selector_for_pool(pool_name).balance_recorder is not None
            and not _EXPERT_REPLAY.get()
        ):
            route = _ExecutedSiteRoute(
                ids.detach().reshape(contexts, per_context), features.detach(), 0
            )
            self.selector_for_pool(pool_name).record_executed(route)
        if not _EXPERT_REPLAY.get():
            self.expert_rows += flat.shape[0]
            self.pools[pool_name].record_routes(
                ids,
                palette,
                gradient_enabled=torch.is_grad_enabled(),
            )
        return ids.reshape(leading), confidence.reshape(leading), features

    def ffn_bank(self, dim: int) -> nn.Module:
        """Return the shared SwiGLU expert bank for a native width."""

        return self._swiglu_bank(dim)

    def _replace(self, parent: nn.Module, name: str, child: nn.Module, prefix: str) -> None:
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            bank = self._linear_bank(child.in_features, child.out_features, path)
            pool_name = self._pool_name(bank)
            zero_initialized = bool(getattr(child, "_zero_initialized", False)) or path.endswith(
                "facet_distribution.log_scale"
            )
            if child.weight.device.type != "meta" and child.weight.numel() <= 1_000_000:
                zero_initialized = zero_initialized or not bool(
                    torch.count_nonzero(child.weight.detach())
                )
            replacement = RoutedLinear(
                child.in_features,
                child.out_features,
                self,
                pool_name,
                path,
                bias=child.bias,
                zero_initialized=zero_initialized,
            ).to(device=bank.weight.device, dtype=bank.weight.dtype)
            setattr(parent, name, replacement)
            kind = "linear"
            shape = [child.in_features, child.out_features]
        elif child.__class__.__name__ == "SwiGLU" and isinstance(
            getattr(child, "gate_up", None), nn.Linear
        ):
            dim = child.gate_up.in_features
            bank = self._swiglu_bank(dim)
            pool_name = self._pool_name(bank)
            replacement = RoutedSwiGLU(dim, self, pool_name, path)
            replacement = replacement.to(device=bank.gate_up.device, dtype=bank.gate_up.dtype)
            setattr(parent, name, replacement)
            kind = "swiglu"
            shape = [dim, child.gate_up.out_features // 2]
        else:
            return
        self._site_manifest.append(
            {"site": path, "kind": kind, "shape": shape, "pool": self._pool_name(bank)}
        )

    def _pool_name(self, bank: nn.Module) -> str:
        for name, module in self.pools.items():
            if module is bank:
                return name
        return "unknown"

    def bind(self, module: nn.Module, prefix: str, *, exclusions: tuple[str, ...] = ()) -> None:
        """Replace eligible descendants; named exclusions are recorded, not traversed."""
        excluded = set(exclusions)

        def visit(parent: nn.Module, base: str) -> None:
            for name, child in list(parent.named_children()):
                path = f"{base}.{name}" if base else name
                if path in excluded or name in excluded:
                    self._exclusions.append({"site": path, "reason": "explicit bind exclusion"})
                    continue
                if name == "import_map":
                    self._exclusions.append(
                        {
                            "site": path,
                            "reason": "native n-gram import map is an optional baked bridge",
                        }
                    )
                    continue
                if name == "slots":
                    self._exclusions.append(
                        {"site": path, "reason": "fixed slot geometry transform"}
                    )
                    continue
                if name == "ngram_mixture":
                    self._exclusions.append(
                        {"site": path, "reason": "learned input-order selector"}
                    )
                    continue
                if isinstance(child, (nn.Embedding, nn.LayerNorm)) or child.__class__.__name__ in {
                    "RMSNorm",
                }:
                    self._exclusions.append(
                        {"site": path, "reason": "lookup table or normalization"}
                    )
                    continue
                if parent.__class__.__name__ == "TiedVocabulary" and name == "projection":
                    self._exclusions.append(
                        {
                            "site": path,
                            "reason": "weight is consumed directly by tied vocabulary embedding",
                        }
                    )
                    continue
                if child.__class__.__module__.startswith("megatoken.omni.realizers"):
                    self._exclusions.append(
                        {"site": path, "reason": "pretrained external realizer"}
                    )
                    continue
                if child.__class__.__name__ in {
                    "EpochRouter",
                    "AdapterBank",
                    "OwnerAdapters",
                    "ShapePools",
                }:
                    self._exclusions.append(
                        {"site": path, "reason": "router/controller or expert registry"}
                    )
                    continue
                parameters = tuple(child.parameters())
                if parameters and all(not parameter.requires_grad for parameter in parameters):
                    self._exclusions.append({"site": path, "reason": "frozen external module"})
                    continue
                if isinstance(child, (RoutedLinear, RoutedConv1d, RoutedSwiGLU, RoutedGRUCell)):
                    continue
                if isinstance(child, nn.GRUCell):
                    input_bank = self._linear_bank(
                        child.input_size, 3 * child.hidden_size, path + ".input"
                    )
                    input_pool = self._pool_name(input_bank)
                    hidden_bank = self._linear_bank(
                        child.hidden_size, 3 * child.hidden_size, path + ".hidden"
                    )
                    hidden_pool = self._pool_name(hidden_bank)
                    replacement = RoutedGRUCell(
                        child.input_size,
                        child.hidden_size,
                        self,
                        input_pool,
                        hidden_pool,
                        path,
                        child.bias_ih,
                        child.bias_hh,
                    )
                    replacement = replacement.to(
                        device=input_bank.weight.device,
                        dtype=input_bank.weight.dtype,
                    )
                    setattr(parent, name, replacement)
                    self._site_manifest.append(
                        {
                            "site": path,
                            "kind": "gru_cell",
                            "shape": [child.input_size, child.hidden_size],
                            "pool": self._pool_name(input_bank),
                        }
                    )
                    continue
                if isinstance(child, nn.Conv1d):
                    if child.groups != 1:
                        raise NotImplementedError(
                            f"trainable grouped Conv1d at {path!r} is unsupported (groups={child.groups})"
                        )
                    if child.padding_mode != "zeros":
                        raise NotImplementedError(
                            f"trainable Conv1d at {path!r} uses unsupported padding_mode={child.padding_mode!r}"
                        )
                    bank = self._linear_bank(child.in_channels, child.out_channels, path)
                    pool_name = self._pool_name(bank)
                    replacement = RoutedConv1d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size[0],
                        self,
                        pool_name,
                        path,
                        stride=child.stride[0],
                        padding=child.padding if isinstance(child.padding, str) else child.padding,
                        dilation=child.dilation[0],
                        padding_mode=child.padding_mode,
                        bias=child.bias,
                    ).to(device=bank.weight.device, dtype=bank.weight.dtype)
                    setattr(parent, name, replacement)
                    self._site_manifest.append(
                        {
                            "site": path,
                            "kind": "conv1d",
                            "shape": [child.in_channels, child.out_channels],
                            "kernel_size": child.kernel_size[0],
                            "pool": pool_name,
                        }
                    )
                    continue
                if isinstance(child, nn.Linear):
                    self._replace(parent, name, child, base)
                    continue
                if child.__class__.__name__ == "SwiGLU":
                    self._replace(parent, name, child, base)
                    continue
                visit(child, path)

        visit(module, prefix)

    def _coverage(self, module: nn.Module) -> dict[str, object]:
        from .layers import SwiGLU

        excluded_roots = {row["site"] for row in self._exclusions}
        routed_roots = {
            name for name, child in module.named_modules() if isinstance(child, _RoutedSite)
        }
        frozen_roots = {
            name
            for name, child in module.named_modules()
            if name
            and (parameters := tuple(child.parameters()))
            and all(not parameter.requires_grad for parameter in parameters)
        }
        unconverted = []
        for name, child in module.named_modules():
            if isinstance(
                child, (RoutedLinear, RoutedConv1d, RoutedSwiGLU, RoutedGRUCell, _RoutedSite)
            ):
                continue
            if any(name == root or name.startswith(root + ".") for root in excluded_roots):
                continue
            if any(name == root or name.startswith(root + ".") for root in routed_roots):
                continue
            if any(name == root or name.startswith(root + ".") for root in frozen_roots):
                continue
            if isinstance(child, (nn.Linear, nn.Conv1d, nn.GRUCell, SwiGLU)):
                unconverted.append({"site": name, "kind": type(child).__name__})
        return {
            "routed": len(tuple(iter_sites(module))),
            "unconverted": tuple(unconverted),
            "frozen_external": tuple(sorted(frozen_roots)),
        }

    def manifest(self, module: nn.Module | None = None) -> dict[str, object]:
        def bank_geometry(bank: nn.Module) -> dict[str, object]:
            return {
                "type": type(bank).__name__,
                "experts": bank.experts,
                "input_dim": bank.dim,
                "output_dim": getattr(bank, "output_dim", bank.dim),
                "hidden_dim": getattr(bank, "hidden", None),
                "parameters": sum(parameter.numel() for parameter in bank.parameters()),
            }

        report = {
            "sites": tuple(dict(item) for item in self._site_manifest),
            "pools": tuple(
                {
                    "name": name,
                    **bank_geometry(bank),
                }
                for name, bank in self.pools.items()
            ),
            "routing": {
                "router_bank": bank_geometry(self.router_bank),
                "router_experts": self.router_experts,
                "router_dim": self.router_dim,
                "bootstrap_code_parameters": self.router_bootstrap_codes.numel(),
                "pool_selector_count": len(self._selectors) - int("bootstrap" in self._selectors),
                "selectors": tuple(self.routing_selectors().keys()),
                "router_evaluations": self.router_evaluations,
                "palette_evaluations": self.palette_evaluations,
                "expert_rows": self.expert_rows,
            },
            "exclusions": tuple(dict(item) for item in self._exclusions),
        }
        if module is not None:
            report["coverage"] = self._coverage(module)
        return report

    def metrics(self, *, reset: bool = False) -> dict[str, object]:
        """Routing work and per-bank dispatch counts across the whole registry."""
        result: dict[str, object] = {
            "router_evaluations": self.router_evaluations,
            "palette_evaluations": self.palette_evaluations,
            "expert_rows": self.expert_rows,
            "router_bank": self.router_bank.metrics(reset=reset),
            "shape_pools": {name: bank.metrics(reset=reset) for name, bank in self.pools.items()},
        }
        if reset:
            self.router_evaluations = 0
            self.palette_evaluations = 0
            self.expert_rows = 0
        return result


def iter_sites(module: nn.Module):
    """Yield each bound routed site once, with its fully qualified path."""

    for name, child in module.named_modules():
        if isinstance(child, _RoutedSite):
            yield name, child
