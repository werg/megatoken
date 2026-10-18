"""AdamW or hybrid Muon/AdamW with FP32 masters and accumulation buffers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterRule:
    lr_multiplier: float
    weight_decay: float


def parameter_rules(module: nn.Module, weight_decay: float) -> dict[str, ParameterRule]:
    """Use owning module semantics first; shared tensors get one optimizer slot."""
    lookup_ids = {
        id(p)
        for m in module.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    result = {}
    for name, parameter in module.named_parameters():
        lower = name.lower()
        lookup = id(parameter) in lookup_ids or any(
            x in lower for x in ("lookup", "ngram", "embedding", "vocabulary", "byte_table")
        )
        slower = lookup or "control" in lower
        multiplier = 0.5 if slower else 1.0
        no_decay = lookup or parameter.ndim < 2 or lower.endswith("bias") or "norm" in lower
        result[name] = ParameterRule(multiplier, 0.0 if no_decay else weight_decay)
    return result


class MasterAdamW:
    """A small adapter around PyTorch's maintained fused/foreach AdamW kernels.

    BF16 leaf gradients are moved into FP32 buffers after each backward, avoiding
    low-precision accumulation over many windows. Parameters already in FP32 are
    their own masters. Checkpoints retain both precise masters and served weights.
    """

    def __init__(
        self,
        module: nn.Module,
        lr: float,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        inner_optimizer: str = "adamw",
    ) -> None:
        if inner_optimizer not in {"adamw", "muon"}:
            raise ValueError("inner_optimizer must be adamw or muon")
        self.inner_optimizer = inner_optimizer
        self.module = module
        self.rules = parameter_rules(module, weight_decay)
        self.model_parameters: dict[str, nn.Parameter] = {}
        self.masters: dict[str, nn.Parameter] = {}
        self._hooks: dict[str, Any] = {}
        self.frozen_names: set[str] = set()
        self.gradient_domains: dict[str, tuple[str, ...]] = {}
        groups: dict[ParameterRule, list[nn.Parameter]] = defaultdict(list)
        muon_groups: dict[ParameterRule, list[nn.Parameter]] = defaultdict(list)
        self.muon_names: list[str] = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            self.model_parameters[name] = parameter
            master = (
                parameter
                if parameter.dtype == torch.float32
                else nn.Parameter(parameter.detach().float())
            )
            self.masters[name] = master
            # Muon targets dense hidden 2D matrices, not embeddings, norms,
            # control lookups, 1D biases, or packed 3D+ experts. Packed expert
            # matrices retain AdamW until explicitly reparameterized per expert.
            use_muon = (
                inner_optimizer == "muon"
                and parameter.ndim == 2
                and self.rules[name].weight_decay > 0
                and "control" not in name.lower()
            )
            if use_muon:
                muon_groups[self.rules[name]].append(master)
                self.muon_names.append(name)
            else:
                groups[self.rules[name]].append(master)
            if master is not parameter:

                def accumulate(p: nn.Parameter, destination: nn.Parameter = master) -> None:
                    if p.grad is not None:
                        gradient = p.grad.detach().float()
                        if destination.grad is None:
                            destination.grad = gradient
                        else:
                            destination.grad.add_(gradient)
                        p.grad = None

                self._hooks[name] = parameter.register_post_accumulate_grad_hook(accumulate)
        if not groups:
            raise ValueError("The model has no trainable parameters")
        cuda = all(p.device.type == "cuda" for p in self.masters.values())
        self.optimizer = (
            torch.optim.AdamW(
                [
                    {
                        "params": params,
                        "lr": lr * rule.lr_multiplier,
                        "lr_multiplier": rule.lr_multiplier,
                        "weight_decay": rule.weight_decay,
                    }
                    for rule, params in groups.items()
                ],
                lr=lr,
                betas=betas,
                eps=eps,
                fused=cuda,
                foreach=None if cuda else True,
            )
            if groups
            else None
        )
        self.muon_optimizer = (
            torch.optim.Muon(
                [
                    {
                        "params": params,
                        "lr": lr * rule.lr_multiplier,
                        "lr_multiplier": rule.lr_multiplier,
                        "weight_decay": rule.weight_decay,
                    }
                    for rule, params in muon_groups.items()
                ],
                lr=lr,
                momentum=0.95,
                nesterov=True,
                adjust_lr_fn="match_rms_adamw",
            )
            if muon_groups
            else None
        )

    @property
    def parameters(self) -> list[nn.Parameter]:
        return list(self.masters.values())

    def set_lr(self, lr: float) -> None:
        for optimizer in (self.optimizer, self.muon_optimizer):
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr * group["lr_multiplier"]

    def zero_grad(self) -> None:
        for optimizer in (self.optimizer, self.muon_optimizer):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        for parameter in self.model_parameters.values():
            parameter.grad = None

    def freeze(self, names: list[str]) -> None:
        """Freeze owners without losing their checkpointed masters or moments."""
        missing = set(names) - self.model_parameters.keys()
        if missing:
            raise ValueError(f"Cannot freeze unowned parameters: {sorted(missing)}")
        for name in names:
            self.model_parameters[name].requires_grad_(False)
            self.model_parameters[name].grad = None
            self.masters[name].grad = None
        self.frozen_names.update(names)

    def set_gradient_domains(self, domains: dict[str, list[str]]) -> None:
        """Clip disjoint owners independently, without coupling follower losses.

        Domain membership is training policy, derived from the phase config;
        Adam moments and parameter ownership stay unchanged across phases.
        """
        names = [name for members in domains.values() for name in members]
        if (
            not domains
            or any(not key or not members for key, members in domains.items())
            or len(names) != len(set(names))
            or set(names) != self.masters.keys()
        ):
            raise ValueError("Gradient domains must partition all optimizer parameter owners")
        self.gradient_domains = {key: tuple(members) for key, members in domains.items()}

    def isolate_gradient_parameters(self, name: str, parameters: Iterable[nn.Parameter]) -> None:
        """Separate an auxiliary learner from all remaining model owners."""
        if name == "model":
            raise ValueError("The auxiliary gradient domain needs its own name")
        selected = {id(parameter) for parameter in parameters}
        owned = {id(parameter) for parameter in self.model_parameters.values()}
        if not selected <= owned:
            raise ValueError("The auxiliary learner contains unowned optimizer parameters")
        self.set_gradient_domains(
            {
                name: [key for key, p in self.model_parameters.items() if id(p) in selected],
                "model": [key for key, p in self.model_parameters.items() if id(p) not in selected],
            }
        )

    @torch.no_grad()
    def step(self, max_norm: float = 1.0) -> dict[str, float]:
        if any(self.masters[name].grad is not None for name in self.frozen_names):
            raise RuntimeError("A frozen deterministic decoder received an optimizer gradient")
        domains = self.gradient_domains or {"model": tuple(self.masters)}
        metric_device = next(iter(self.masters.values())).device
        norms = {
            key: torch.nn.utils.clip_grad_norm_(
                [self.masters[name] for name in names], max_norm, error_if_nonfinite=True
            ).to(metric_device)
            for key, names in domains.items()
        }
        grad_norm = torch.stack(list(norms.values())).square().sum().sqrt()
        if self.optimizer is not None:
            self.optimizer.step()
        if self.muon_optimizer is not None:
            self.muon_optimizer.step()
        for name, parameter in self.model_parameters.items():
            master = self.masters[name]
            if master is not parameter:
                parameter.copy_(master)
        metrics = {"gradient_norm": float(grad_norm)}
        if len(norms) > 1:
            metrics.update({f"gradient_norm_{key}": float(value) for key, value in norms.items()})
        return metrics

    @torch.no_grad()
    def rebase(self, names: list[str]) -> None:
        """Use after baking static features: replace masters and reset their moments."""
        for name in names:
            parameter, master = self.model_parameters[name], self.masters[name]
            master.copy_(parameter.float())
            for optimizer in (self.optimizer, self.muon_optimizer):
                if optimizer is not None:
                    optimizer.state.pop(master, None)

    @torch.no_grad()
    def on_engram_bake(
        self,
        table_parameter: nn.Parameter,
        rows: torch.Tensor,
        removed_parameters: list[nn.Parameter],
    ) -> None:
        """Reset only baked-row moments, and remove every temporary map slot."""
        table_name = next(
            (
                name
                for name, parameter in self.model_parameters.items()
                if parameter is table_parameter
            ),
            None,
        )
        if table_name is None:
            raise ValueError("Baked table is not owned by this optimizer")
        master = self.masters[table_name]
        row_ids = rows.to(device=master.device, dtype=torch.long)
        if master is not table_parameter:
            master.index_copy_(0, row_ids, table_parameter[row_ids].float())
        owner = self.muon_optimizer if table_name in self.muon_names else self.optimizer
        state = owner.state.get(master, {}) if owner is not None else {}
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                state[key].index_fill_(0, row_ids, 0.0)
        removed_ids = {id(parameter) for parameter in removed_parameters}
        removed_masters = set()
        for name, parameter in list(self.model_parameters.items()):
            if id(parameter) in removed_ids:
                removed_master = self.masters.pop(name)
                removed_masters.add(id(removed_master))
                for optimizer in (self.optimizer, self.muon_optimizer):
                    if optimizer is not None:
                        optimizer.state.pop(removed_master, None)
                if name in self.muon_names:
                    self.muon_names.remove(name)
                self.model_parameters.pop(name)
                self.rules.pop(name)
                self.frozen_names.discard(name)
                hook = self._hooks.pop(name, None)
                if hook is not None:
                    hook.remove()
        for optimizer in (self.optimizer, self.muon_optimizer):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["params"] = [
                    parameter for parameter in group["params"] if id(parameter) not in removed_masters
                ]
            optimizer.param_groups = [
                group for group in optimizer.param_groups if group["params"]
            ]
        self.gradient_domains = {
            key: tuple(name for name in names if name in self.masters)
            for key, names in self.gradient_domains.items()
            if any(name in self.masters for name in names)
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 2 if self.inner_optimizer == "muon" else 1,
            "names": list(self.masters),
            "master_weights": {name: p.detach() for name, p in self.masters.items()},
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            **(
                {
                    "muon_names": list(self.muon_names),
                    "muon": self.muon_optimizer.state_dict()
                    if self.muon_optimizer is not None
                    else None,
                }
                if self.inner_optimizer == "muon"
                else {}
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = 2 if self.inner_optimizer == "muon" else 1
        if state.get("format_version") != expected or state["names"] != list(self.masters):
            raise ValueError("Optimizer parameter ownership/type differs from the checkpoint")
        if expected == 2 and state["muon_names"] != self.muon_names:
            raise ValueError("Muon matrix ownership changed across the checkpoint")
        with torch.no_grad():
            for name, parameter in self.masters.items():
                parameter.copy_(state["master_weights"][name])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            # Optimizer.load_state_dict must not cast FP32 moments to served BF16.
            for value in self.optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in value and value[key].dtype != torch.float32:
                        raise RuntimeError(f"AdamW {key} lost FP32 precision")
        if self.muon_optimizer is not None:
            self.muon_optimizer.load_state_dict(state["muon"])
            for value in self.muon_optimizer.state.values():
                for slot in value.values():
                    if isinstance(slot, torch.Tensor) and slot.is_floating_point():
                        if slot.dtype != torch.float32:
                            raise RuntimeError("Muon momentum lost FP32 precision")

    def close(self) -> None:
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()
