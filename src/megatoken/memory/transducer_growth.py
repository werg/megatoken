"""Error-driven write-out, structural evidence and bounded evolution of a v10 bank."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from .transducer import Lookup, TransducerBank
from .transducer_packing import pack_fp8, rounding_seed, unit


@dataclass
class EdgeEvidence:
    utility: float = 0.0
    visits: float = 0.0
    dispersion: float = 0.0
    sketch: Tensor = field(default_factory=lambda: torch.zeros(64))
    energy: float = 0.0
    last_step: int = 0

    @property
    def coherence(self) -> float:
        if self.visits <= 0 or self.energy <= 0:
            return 1.0
        return float(self.sketch.square().sum()) / (self.visits * self.energy)

    def half(self) -> EdgeEvidence:
        return EdgeEvidence(
            self.utility / 2,
            self.visits / 2,
            self.dispersion / 2,
            self.sketch / 2,
            self.energy / 2,
            self.last_step,
        )


class TransducerGrowth:
    """Uses existing forward/backward evidence, not a separate growth query or replay."""

    def __init__(self, bank: TransducerBank, *, max_writeouts: int = 512) -> None:
        self.bank = bank
        self.max_writeouts = max_writeouts
        self.writeouts = 0
        self.evidence: dict[int, EdgeEvidence] = {}
        self._random_maps: dict[int, Tensor] = {}

    def begin_round(self) -> None:
        self.writeouts = 0

    def _projection(self, width: int) -> Tensor:
        if width not in self._random_maps:
            generator = torch.Generator().manual_seed(rounding_seed(
                self.bank.geometry.seed, "gradient-sketch", width
            ))
            self._random_maps[width] = (
                torch.randn(64, width, generator=generator) / math.sqrt(64)
            )
        return self._random_maps[width]

    def observe(
        self,
        lookup: Lookup,
        *,
        gradient_value: Tensor,
        gradient_location: Tensor,
        step: int,
        hindsight: Tensor | None = None,
        gradients: dict[int, Tensor] | None = None,
        sink_only: bool = False,
    ) -> None:
        """Exact first-order removal utility; phase-dependent dispersion or sketch."""
        if step < 0 or gradient_value.numel() != self.bank.geometry.value_dim:
            raise ValueError("Invalid structural observation")
        if gradient_location.numel() != self.bank.geometry.location_dim:
            raise ValueError("Location sensitivity must accompany value sensitivity")
        if not bool(torch.isfinite(gradient_value).all()) or not bool(
            torch.isfinite(gradient_location).all()
        ):
            raise FloatingPointError("Nonfinite structural gradient")
        for edge_id, weight in lookup.edge_weights.items():
            if edge_id not in self.bank.rows:
                continue
            w = float(weight.detach())
            if w >= 1:
                raise ValueError("Sink must leave nonzero removal denominator")
            row = self.bank.rows[edge_id]
            v = row.value.decode(device=gradient_value.device)
            e = unit(row.destination.decode(device=gradient_location.device))
            # F is the unnormalized value/location mixture. The lookup stores a
            # normalized bank landing, so recover its norm only for the reference
            # when used; supplied gradients refer to the same displayed landing.
            delta = w / (1 - w) * (
                torch.dot(gradient_value.float(), lookup.bank_value.detach() - v)
                + torch.dot(gradient_location.float(), lookup.bank_landing.detach() - e)
            )
            evidence = self.evidence.setdefault(edge_id, EdgeEvidence(last_step=step))
            age = step - evidence.last_step
            if age < 0:
                raise ValueError("Structural observation clocks are not monotonic")
            decay = 2 ** (-age / 16384)
            evidence.utility = decay * evidence.utility + float(delta.detach())
            evidence.visits = decay * evidence.visits + w
            evidence.dispersion *= decay
            evidence.sketch *= decay
            evidence.energy *= decay
            if not self.bank.learned_values:
                if hindsight is not None and not sink_only:
                    difference = hindsight.detach().float() - v
                    evidence.dispersion += w * float(difference.square().sum())
            elif gradients and edge_id in gradients:
                vector = gradients[edge_id].detach().float().flatten().cpu()
                sketch = self._projection(vector.numel()) @ vector
                evidence.sketch += sketch * w
                evidence.energy += w * float(sketch.square().sum())
            evidence.last_step = step

    def write_out(
        self,
        lookup: Lookup,
        hindsight: Tensor,
        gradient_value: Tensor,
        *,
        step: int,
        sink_only: bool = False,
        training_includes_hindsight: bool = True,
    ) -> int | None:
        """After backward only: admit beneficial, bank-unexplained hindsight."""
        if sink_only or not training_includes_hindsight or self.writeouts >= self.max_writeouts:
            return None
        if hindsight.shape != (self.bank.geometry.value_dim,) or (
            gradient_value.shape != hindsight.shape
        ):
            raise ValueError("Write-out needs a full-width value and future-loss gradient")
        benefit = -float(torch.dot(gradient_value.detach(), hindsight.detach()))
        if benefit <= 0:
            return None
        residual = float((hindsight.detach() - lookup.bank_value.detach()).square().sum())
        if lookup.pairs and residual <= 0.25 * float(hindsight.detach().square().sum()):
            return None
        mod_query = unit(
            lookup.query.detach()
            * (1 + torch.tanh(self.bank.modulation.detach() @ lookup.start.detach()))
        )
        logits = self.bank.sink_bias.detach() + (
            self.bank.sink_direction.detach() @ lookup.query.detach()
        )
        best = (
            float((lookup.weights[:-1].max() / lookup.sink_weight).log() + logits)
            if lookup.pairs
            else 0.0
        )
        bias = best - float(
            self.bank.log_beta_location.exp() + self.bank.log_beta_query.exp()
        )
        h = hindsight.detach()
        try:
            edge_id = self.bank.add_edge(
                departure=lookup.start.detach(),
                key=mod_query,
                value=h,
                destination=self.bank._content_location(h).detach(),
                bias=bias,
                step=step,
                write_mass=0.0 if self.bank.learned_values else 1.0,
            )
        except RuntimeError:
            # No extra write-out when out of byte capacity or at hard ceiling.
            return None
        self.evidence[edge_id] = EdgeEvidence(last_step=step)
        self.writeouts += 1
        return edge_id

    def _split(self, edge_id: int, step: int) -> tuple[int, int] | None:
        bank = self.bank
        row = bank.rows[edge_id]
        if len(bank.rows) >= bank.geometry.max_edges:
            return None
        if bank.packed_bytes + row.nbytes > bank.geometry.byte_budget:
            return None
        key = unit(row.key.decode())
        departure = unit(row.departure.decode())
        generator = torch.Generator().manual_seed(rounding_seed(
            bank.geometry.seed, "split", edge_id, step
        ))
        k_noise = torch.randn(key.shape, generator=generator)
        d_noise = torch.randn(departure.shape, generator=generator)
        k_noise = unit(k_noise - torch.dot(key, k_noise) * key)
        d_noise = unit(d_noise - torch.dot(departure, d_noise) * departure)
        key_a, key_b = unit(key + 0.01 * k_noise), unit(key - 0.01 * k_noise)
        dep_a, dep_b = unit(departure + 0.01 * d_noise), unit(departure - 0.01 * d_noise)
        new_id = max(bank.rows) + 1
        half = row.write_mass / 2
        bias = row.bias - math.log(2)
        evidence = self.evidence.pop(edge_id, EdgeEvidence(last_step=step)).half()
        bank.rows[edge_id] = replace(
            row,
            key=pack_fp8(key_a),
            departure=pack_fp8(dep_a),
            bias=bias,
            write_mass=half,
        )
        bank.rows[new_id] = replace(
            row,
            id=new_id,
            key=pack_fp8(key_b),
            departure=pack_fp8(dep_b),
            bias=bias,
            write_mass=half,
        )
        self.evidence[edge_id] = evidence
        self.evidence[new_id] = replace(evidence, sketch=evidence.sketch.clone())
        bank._masters = {
            key: value for key, value in bank._masters.items() if key[0] != edge_id
        }
        bank._invalidate()
        return edge_id, new_id

    def evolve(self, step: int) -> dict[str, int]:
        """Coordinator-only structure changes after a complete optimization round."""
        if step < 0:
            raise ValueError("Growth step must be nonnegative")
        removable = sorted(
            (
                (self.evidence.get(i, EdgeEvidence()).utility / max(row.nbytes, 1), i)
                for i, row in self.bank.rows.items()
                if step - row.birth_step >= 4096
                and self.evidence.get(i, EdgeEvidence()).utility <= 0
            )
        )
        removed = 0
        for _, edge_id in removable:
            self.bank.retire(edge_id)
            self.evidence.pop(edge_id, None)
            removed += 1
        split_candidates: list[int] = []
        bank = self.bank
        if bank.learned_values:
            active = [
                v.energy / max(v.visits, 1e-12)
                for i, v in self.evidence.items()
                if i in bank.rows and v.visits >= 64
            ]
            top_decile = sorted(active)[int(0.9 * (len(active) - 1))] if active else math.inf
        for i, state in sorted(self.evidence.items()):
            if i not in bank.rows or state.visits < 64:
                continue
            row = bank.rows[i]
            if bank.learned_values:
                if state.coherence < 0.1 and (
                    state.energy / max(state.visits, 1e-12) >= top_decile
                ):
                    split_candidates.append(i)
            elif row.write_mass > 0 and state.dispersion / row.write_mass > (
                0.5 * float(row.value.decode().square().sum())
            ):
                split_candidates.append(i)
        splits = 0
        for i in split_candidates:
            if self._split(i, step) is not None:
                splits += 1
        if bank.packed_bytes > 0.95 * bank.geometry.byte_budget:
            eligible = sorted(
                bank.rows, key=lambda i: (
                    self.evidence.get(i, EdgeEvidence()).utility / bank.rows[i].nbytes,
                    i,
                )
            )
            for edge_id in eligible:
                if bank.packed_bytes <= 0.9 * bank.geometry.byte_budget:
                    break
                bank.retire(edge_id)
                self.evidence.pop(edge_id, None)
                removed += 1
        return {"removed": removed, "split": splits}

    def state_dict(self) -> dict:
        return {
            "writeouts": self.writeouts,
            "evidence": {
                str(edge): {
                    "utility": evidence.utility,
                    "visits": evidence.visits,
                    "dispersion": evidence.dispersion,
                    "sketch": evidence.sketch.tolist(),
                    "energy": evidence.energy,
                    "last_step": evidence.last_step,
                }
                for edge, evidence in self.evidence.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.writeouts = int(state["writeouts"])
        self.evidence = {
            int(edge): EdgeEvidence(
                utility=value["utility"],
                visits=value["visits"],
                dispersion=value["dispersion"],
                sketch=torch.tensor(value["sketch"]),
                energy=value["energy"],
                last_step=value["last_step"],
            )
            for edge, value in state["evidence"].items()
        }
