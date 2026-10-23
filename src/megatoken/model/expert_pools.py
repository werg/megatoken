"""Shared shape-bucketed expert pools and non-owning learned sites.

``ShapePools`` owns each expert bank exactly once.  Sites keep their router
parameters locally and hold an unregistered reference to the relevant bank;
this lets the same pool serve many module locations without duplicate state
dict entries or repeated optimizer parameters.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Iterator, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_CURRENT_ADAPTER_WEIGHTS: ContextVar[Tensor | None] = ContextVar(
    "megatoken_expert_adapter_weights", default=None
)
_EXPERT_REPLAY: ContextVar[bool] = ContextVar("megatoken_expert_replay", default=False)


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


def capture_expert_context():
    """Capture the current context for explicit checkpoint replay."""

    return copy_context()


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

    @contextmanager
    def with_weights():
        token = _CURRENT_ADAPTER_WEIGHTS.set(weights)
        try:
            yield
        finally:
            _CURRENT_ADAPTER_WEIGHTS.reset(token)

    @contextmanager
    def with_replay():
        token = _CURRENT_ADAPTER_WEIGHTS.set(weights)
        replay_token = _EXPERT_REPLAY.set(True)
        try:
            yield
        finally:
            _EXPERT_REPLAY.reset(replay_token)
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


class _RoutedSite(nn.Module):
    def _init_router(self, input_dim: int, experts: int, bank: nn.Module) -> None:
        self.router = nn.Linear(input_dim, experts, bias=True)
        nn.init.normal_(self.router.weight, std=0.02 / math.sqrt(input_dim))
        nn.init.zeros_(self.router.bias)
        object.__setattr__(self, "_bank", bank)
        self.input_dim = input_dim
        self.experts = experts
        self.balance_recorder = None

    def balance_probabilities(self, context: Tensor, stage: int) -> Tensor:
        if stage != 0:
            raise ValueError("A routed site has only its single input-state routing stage")
        return self.router(context.to(self.router.weight.dtype)).float().softmax(-1)

    def record_executed(self, route) -> None:
        if self.balance_recorder is not None and not _EXPERT_REPLAY.get():
            self.balance_recorder.capture(self, route)

    def _route(self, rows: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        leading = tuple(rows.shape[:-1])
        flat = rows.reshape(-1, rows.shape[-1])
        probabilities = self.router(flat.to(self.router.weight.dtype)).float().softmax(-1)
        maximum, expert_ids = probabilities.max(-1)
        confidence = 0.75 + 0.25 * maximum
        adapters = _adapter_rows(_CURRENT_ADAPTER_WEIGHTS.get(), leading, rows.device)
        if self.balance_recorder is not None and not _EXPERT_REPLAY.get():
            route = _ExecutedSiteRoute(expert_ids.reshape(-1, 1), flat.detach(), 0)
            self.record_executed(route)
        return expert_ids.reshape(leading), confidence.reshape(leading), adapters


class _ExecutedSiteRoute(NamedTuple):
    expert_ids: Tensor
    router_context: Tensor
    router_stage: int


class RoutedLinear(_RoutedSite):
    """A linear site routed through a shared, padded linear-expert pool."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bank: nn.Module,
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
        self._init_router(input_dim, bank.experts, bank)

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


class RoutedSwiGLU(_RoutedSite):
    """A SwiGLU-shaped site routed through a shared expert FFN pool."""

    def __init__(self, dim: int, bank: nn.Module):
        super().__init__()
        self.dim = dim
        self._init_router(dim, bank.experts, bank)

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
        input_bank: nn.Module,
        hidden_bank: nn.Module,
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
        self._init_router(input_size, input_bank.experts, input_bank)
        object.__setattr__(self, "_hidden_bank", hidden_bank)
        self.hidden_router = nn.Linear(hidden_size, hidden_bank.experts, bias=True)
        nn.init.normal_(self.hidden_router.weight, std=0.02 / math.sqrt(hidden_size))
        nn.init.zeros_(self.hidden_router.bias)

    def forward(self, x: Tensor, hx: Tensor | None = None) -> Tensor:
        if hx is None:
            hx = x.new_zeros((*x.shape[:-1], self.hidden_size))
        in_pad = F.pad(x, (0, self._bank.dim - self.input_size))
        hidden_pad = F.pad(hx, (0, self._hidden_bank.dim - self.hidden_size))
        in_ids, in_conf, adapters = self._route(x)
        hidden_probs = self.hidden_router(
            hx.to(self.hidden_router.weight.dtype)
        ).float().softmax(-1)
        hidden_conf, hidden_ids = hidden_probs.max(-1)
        input_gates = self._bank(in_pad, in_ids, in_conf, adapter_weights=adapters)
        hidden_gates = self._hidden_bank(
            hidden_pad, hidden_ids, hidden_conf, adapter_weights=adapters
        )
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
        self._linear_keys: dict[tuple[int, int], str] = {}
        self._swiglu_keys: dict[int, str] = {}
        self._site_manifest: list[dict[str, object]] = []
        self._exclusions: list[dict[str, str]] = []
        self.pools = nn.ModuleDict()
        self.register_buffer("_pool_anchor", torch.empty(0), persistent=False)
        self._ExpertBank, self._LinearExpertBank = ExpertBank, LinearExpertBank

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
        return self.pools[self._swiglu_keys[bucket]]

    def ffn_bank(self, dim: int) -> nn.Module:
        """Return the shared SwiGLU expert bank for a native width."""

        return self._swiglu_bank(dim)

    def _replace(self, parent: nn.Module, name: str, child: nn.Module, prefix: str) -> None:
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            bank = self._linear_bank(child.in_features, child.out_features, path)
            zero_initialized = not bool(torch.count_nonzero(child.weight.detach()))
            replacement = RoutedLinear(
                child.in_features,
                child.out_features,
                bank,
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
            replacement = RoutedSwiGLU(dim, bank)
            replacement = replacement.to(
                device=bank.gate_up.device, dtype=bank.gate_up.dtype
            )
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
                if name in {"slots", "level_projection"}:
                    self._exclusions.append(
                        {"site": path, "reason": "fixed slot or level geometry transform"}
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
                if isinstance(child, (RoutedLinear, RoutedSwiGLU, RoutedGRUCell)):
                    continue
                if isinstance(child, nn.GRUCell):
                    input_bank = self._linear_bank(
                        child.input_size, 3 * child.hidden_size, path + ".input"
                    )
                    hidden_bank = self._linear_bank(
                        child.hidden_size, 3 * child.hidden_size, path + ".hidden"
                    )
                    replacement = RoutedGRUCell(
                        child.input_size,
                        child.hidden_size,
                        input_bank,
                        hidden_bank,
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
        routed_roots = {name for name, child in module.named_modules() if isinstance(child, _RoutedSite)}
        unconverted = []
        for name, child in module.named_modules():
            if isinstance(child, (RoutedLinear, RoutedSwiGLU, RoutedGRUCell, _RoutedSite)):
                continue
            if any(name == root or name.startswith(root + ".") for root in excluded_roots):
                continue
            if any(name == root or name.startswith(root + ".") for root in routed_roots):
                continue
            if isinstance(child, (nn.Linear, nn.GRUCell, SwiGLU)):
                unconverted.append(
                    {"site": name, "kind": type(child).__name__}
                )
        return {
            "routed": len(tuple(iter_sites(module))),
            "unconverted": tuple(unconverted),
        }

    def manifest(self, module: nn.Module | None = None) -> dict[str, object]:
        report = {
            "sites": tuple(dict(item) for item in self._site_manifest),
            "pools": tuple(
                {
                    "name": name,
                    "type": type(bank).__name__,
                    "parameters": sum(p.numel() for p in bank.parameters()),
                }
                for name, bank in self.pools.items()
            ),
            "exclusions": tuple(dict(item) for item in self._exclusions),
        }
        if module is not None:
            report["coverage"] = self._coverage(module)
        return report


def iter_sites(module: nn.Module):
    """Yield each bound routed site once, with its fully qualified path."""

    for name, child in module.named_modules():
        if isinstance(child, _RoutedSite):
            yield name, child
