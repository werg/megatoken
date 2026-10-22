"""v10 DiLoCo for the transducer bank: learner rounds, one canonical coordinator.

A learner trains its local copy of the newest published bank for 32 local
steps, then submits one ``LearnerRound``: per-field deltas against the rows it
started from (the 32,768-wide value deltas packed to FP4 with stochastic
rounding, so a round stays small; the coordinator merges exactly the rows it
decodes), its hindsight writes as (mass, packed mean), the structural evidence
it added, and its provisional write-outs as packed rows.
The coordinator alone allocates canonical IDs, merges with age weight
2^(-rounds behind / 8), adopts provisional rows (combining near-duplicates from
different learners), evolves structure and publishes a change set. There is no
outer KB optimizer and no momentum.

The coordinator's clock is the furthest learner step carried by any merged
round (``TransducerCoordinator.clock``). Row ages and evidence decay are
measured in learner steps, so the clock must follow the learners however many
rounds a merge combines; counting merges would run it up to the number of
workers times too fast under partial quorums.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from torch import Tensor

from megatoken.memory.transducer import (
    ADDRESS_FIELDS,
    FIELD_COLUMNS,
    PROVISIONAL_BASE,
    TransducerBank,
)
from megatoken.memory.transducer_growth import EvidenceRows, TransducerGrowth
from megatoken.memory.transducer_optimizer import WRITE_HALF_LIFE, BlockScaleDescent
from megatoken.memory.transducer_packing import FP4Rows, pack_fp4, row_seeds, unit

LOCAL_STEPS = 32
COORDINATOR_WORKER = -1
CHANGE_HISTORY = 64


@dataclass
class LearnerRound:
    """Everything one learner contributes from one 32-step round (tensors only)."""

    worker: int
    sequence: int
    base_version: int
    examples: int
    step: int
    deltas: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    write_ids: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    write_mass: Tensor = field(default_factory=lambda: torch.empty(0))
    write_means: dict[str, Tensor] = field(default_factory=dict)  # packed FP4 columns
    value_delta: dict[str, Tensor] = field(default_factory=dict)  # ids + packed FP4 columns
    evidence: EvidenceRows | None = None
    provisional: dict[str, Tensor] = field(default_factory=dict)  # packed rows, local IDs

    def __post_init__(self) -> None:
        if min(self.worker, self.sequence, self.base_version, self.step) < 0:
            raise ValueError("Learner round identity must be nonnegative")
        if self.examples < 1:
            raise ValueError("A learner round needs its processed-example weight")

    def state_dict(self) -> dict:
        return {
            "worker": self.worker,
            "sequence": self.sequence,
            "base_version": self.base_version,
            "examples": self.examples,
            "step": self.step,
            "deltas": {name: {"ids": i, "rows": d} for name, (i, d) in self.deltas.items()},
            "write_ids": self.write_ids,
            "write_mass": self.write_mass,
            "write_means": self.write_means,
            "value_delta": self.value_delta,
            "evidence": None if self.evidence is None else vars(self.evidence),
            "provisional": self.provisional,
        }

    def decoded_deltas(self, value_dim: int) -> dict[str, tuple[Tensor, Tensor]]:
        """Every field's (edge IDs, rows), the packed value rows decoded as sent."""
        result = dict(self.deltas)
        if self.value_delta:
            packed = self.value_delta
            result["value"] = (
                packed["ids"],
                FP4Rows(packed["codes"], packed["groups"], packed["rows"]).decode(value_dim),
            )
        return result

    @classmethod
    def from_state(cls, state: dict) -> LearnerRound:
        return cls(
            worker=int(state["worker"]),
            sequence=int(state["sequence"]),
            base_version=int(state["base_version"]),
            examples=int(state["examples"]),
            step=int(state["step"]),
            deltas={name: (v["ids"], v["rows"]) for name, v in state["deltas"].items()},
            write_ids=state["write_ids"],
            write_mass=state["write_mass"],
            write_means=state["write_means"],
            value_delta=state["value_delta"],
            evidence=None if state["evidence"] is None else EvidenceRows(**state["evidence"]),
            provisional=state["provisional"],
        )


def learner_round(
    optimizer: BlockScaleDescent,
    growth: TransducerGrowth,
    *,
    worker: int,
    sequence: int,
    base_version: int,
    examples: int,
    step: int,
) -> LearnerRound:
    """Collect and reset one learner's round contribution (called at a round boundary)."""
    bank = optimizer.bank
    deltas, writes = optimizer.take_round()
    # Each provisional row is sent once, whole; it has no canonical base to differ from.
    provisional_ids = [edge for edge in growth.provisional_ids if edge in bank]
    growth.submitted_ids.extend(provisional_ids)
    growth.provisional_ids = []
    for name, (ids, rows) in list(deltas.items()):
        keep = ids < PROVISIONAL_BASE
        deltas[name] = (ids[keep], rows[keep])
    value_delta: dict[str, Tensor] = {}
    if "value" in deltas:
        ids, rows = deltas.pop("value")
        if ids.numel():
            packed = pack_fp4(rows, row_seeds(ids.to(rows.device), "value", step, worker))
            value_delta = {
                "ids": ids.cpu(),
                "codes": packed.codes.cpu(),
                "groups": packed.group_scales.cpu(),
                "rows": packed.row_scales.cpu(),
            }
    write_ids = [edge for edge in writes.ids() if edge in bank]
    write_means: dict[str, Tensor] = {}
    write_mass = torch.empty(0)
    if write_ids:
        rows = writes.find(write_ids)
        positive = writes.mass[rows] > 0
        rows = rows[positive]
        write_ids = torch.tensor(write_ids)[positive.cpu()].tolist()
    if write_ids:
        write_mass = writes.mass[rows].cpu()
        means = writes.sums[rows] / writes.mass[rows][:, None]
        packed = pack_fp4(
            means, row_seeds(torch.tensor(write_ids, device=bank.device), "value", step, worker)
        )
        write_means = {
            "value_codes": packed.codes.cpu(),
            "value_groups": packed.group_scales.cpu(),
            "value_rows": packed.row_scales.cpu(),
        }
    evidence = growth.take_round(step)
    submission = LearnerRound(
        worker=worker,
        sequence=sequence,
        base_version=base_version,
        examples=examples,
        step=step,
        deltas=deltas,
        write_ids=torch.tensor(write_ids, dtype=torch.int64),
        write_mass=write_mass,
        write_means=write_means,
        value_delta=value_delta,
        evidence=evidence,
        provisional=bank.export_rows(provisional_ids) if provisional_ids else {},
    )
    growth.begin_round()
    return submission


SWITCH_EDGES = 262_144
SWITCH_EVALUATIONS = 16
SWITCH_EVERY = 1024


@dataclass
class SwitchRule:
    """Switch to learned values once the bank is large and bank-only validation
    loss improved by less than 0.5% across the last 16 evaluations."""

    history: list[tuple[int, float]] = field(default_factory=list)

    def record(self, step: int, loss: float, bank: TransducerBank) -> bool:
        if not math.isfinite(loss):
            raise FloatingPointError("Invalid bank-only continuation loss")
        if self.history and step <= self.history[-1][0]:
            return False  # a resumed learner replaying an evaluation it already reported
        self.history.append((step, float(loss)))
        del self.history[:-SWITCH_EVALUATIONS]
        if (
            bank.learned_values
            or bank.edge_count < SWITCH_EDGES
            or len(self.history) < SWITCH_EVALUATIONS
        ):
            return False
        baseline = self.history[0][1]
        return (baseline - loss) / max(abs(baseline), 1e-12) < 0.005

    def state_dict(self) -> dict:
        return {"history": list(self.history)}

    def load_state_dict(self, state: dict) -> None:
        self.history = [(int(s), float(v)) for s, v in state["history"]]


@dataclass
class ChangeSet:
    """What a learner at ``base_version`` must apply to reach ``version``."""

    base_version: int
    version: int
    rows: dict[str, Tensor]
    retired: list[int]
    next_id: int
    learned_values: bool

    def state_dict(self) -> dict:
        return dict(vars(self))


def install_changes(
    bank: TransducerBank,
    optimizer: BlockScaleDescent,
    growth: TransducerGrowth,
    changes: ChangeSet,
) -> None:
    """A learner installs the newest canonical bank at a window boundary.

    Its submitted provisional rows are dropped (the coordinator adopted or
    rejected them; unsubmitted ones stay), its masters restart from the
    installed rows, and it keeps its own statistics for every edge that still
    exists.
    """
    if optimizer.bank is not bank or growth.bank is not bank:
        raise ValueError("Install requires one learner's bank, optimizer and growth")
    if len(optimizer.writes):
        raise RuntimeError("Install only between optimizer steps")
    gone = [edge for edge in growth.submitted_ids if edge in bank]
    retired = gone + changes.retired
    bank.import_rows(changes.rows, retired=retired, next_id=changes.next_id)
    optimizer.forget(retired)
    # Rows the coordinator republished are this round's new base.
    optimizer.forget_origins(changes.rows["ids"].tolist())
    growth.evidence.drop(retired)
    growth.round_evidence.drop(retired)
    growth.submitted_ids = []
    bank.masters.clear()
    if changes.learned_values and not bank.learned_values:
        optimizer.switch_to_learned()
        growth.switch_to_learned()


def _check_packed(ids: Tensor, packed: dict[str, Tensor], dim: int, message: str) -> None:
    """Shapes, dtypes and finite scales of one FP4 row block (``decode`` would fail or lie)."""
    count = ids.numel()
    codes, groups, rows = packed["codes"], packed["groups"], packed["rows"]
    if (
        ids.ndim != 1
        or codes.dtype != torch.uint8
        or codes.shape != (count, -(-dim // 2))
        or groups.dtype != torch.float8_e4m3fn
        or groups.shape != (count, -(-dim // 16))
        or rows.dtype != torch.float32
        or rows.shape != (count,)
        or not bool(torch.isfinite(groups.float()).all() & torch.isfinite(rows).all())
    ):
        raise ValueError(message)


class TransducerCoordinator:
    """The canonical owner. Learners submit in any order; one merge per round."""

    def __init__(
        self, bank: TransducerBank, growth: TransducerGrowth, *, evidence_from: int = 0
    ) -> None:
        if growth.bank is not bank or growth.optimizer is not None:
            raise ValueError("The coordinator's growth must share its bank and have no optimizer")
        if growth.provisional:
            raise ValueError("The coordinator allocates canonical IDs only")
        self.bank = bank
        self.growth = growth
        self.evidence_from = evidence_from
        self.version = 0
        self.clock = 0
        # Highest merged sequence per worker: rounds arrive in order per worker.
        self.ledger: dict[int, int] = {}
        self.pending: list[LearnerRound] = []
        self.switch = SwitchRule()
        self.switch_requested = False
        self.switch_version: int | None = None
        # version -> (changed edge IDs, retired edge IDs) of the merge producing it.
        self.history: OrderedDict[int, tuple[set[int], set[int]]] = OrderedDict()

    @property
    def step(self) -> int:
        """The learners' step as of the last merged round (see the module docstring)."""
        return self.clock

    def submit(self, submission: LearnerRound) -> bool:
        """Queue one round; duplicates are refused, late rounds are age-weighted."""
        if submission.sequence <= self.ledger.get(submission.worker, -1) or any(
            (old.worker, old.sequence) == (submission.worker, submission.sequence)
            for old in self.pending
        ):
            return False
        if submission.base_version > self.version:
            raise ValueError("A learner cannot train from an unpublished bank")
        self._validate(submission)
        self.pending.append(submission)
        return True

    def _validate(self, submission: LearnerRound) -> None:
        """Refuse a round that could fail halfway through a merge."""
        g = self.bank.geometry
        for name, (ids, rows) in submission.deltas.items():
            width = 1 if name == "bias" else g.dims(name)
            if name not in FIELD_COLUMNS or name == "value" or rows.shape != (ids.numel(), width):
                raise ValueError("Malformed transducer delta")
            if not bool(torch.isfinite(rows).all()) or bool((ids >= PROVISIONAL_BASE).any()):
                raise ValueError("Nonfinite or provisional transducer delta")
        if submission.value_delta:
            _check_packed(
                submission.value_delta["ids"],
                {
                    "codes": submission.value_delta["codes"],
                    "groups": submission.value_delta["groups"],
                    "rows": submission.value_delta["rows"],
                },
                g.value_dim,
                "Malformed packed value delta",
            )
            if bool((submission.value_delta["ids"] >= PROVISIONAL_BASE).any()):
                raise ValueError("Nonfinite or provisional transducer delta")
        if submission.write_ids.numel():
            mass = submission.write_mass
            _check_packed(
                submission.write_ids,
                {
                    "codes": submission.write_means["value_codes"],
                    "groups": submission.write_means["value_groups"],
                    "rows": submission.write_means["value_rows"],
                },
                g.value_dim,
                "Malformed submitted hindsight writes",
            )
            if mass.shape != submission.write_ids.shape or not bool(
                torch.isfinite(mass).all() & (mass >= 0).all()
            ):
                raise ValueError("Invalid submitted hindsight writes")
        if submission.provisional:
            rows = submission.provisional
            if bool((rows["ids"] < PROVISIONAL_BASE).any()):
                raise ValueError("Write-outs must carry provisional IDs")
            for name in FIELD_COLUMNS:
                decoded = self.bank.table.decode_columns(name, rows)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError("Nonfinite submitted write-out")
                if name in ADDRESS_FIELDS and bool(
                    (torch.linalg.vector_norm(decoded.float(), dim=-1) < 1e-6).any()
                ):
                    raise ValueError("Zero address in a submitted write-out")

    # ---------------------------------------------------------------- switch

    def record_validation(self, step: int, bank_only_loss: float) -> bool:
        """Bank-only continuation loss; the switch, when due, lands with the next merge."""
        if self.switch.record(step, bank_only_loss, self.bank) and not self.switch_requested:
            self.switch_requested = True
        return self.switch_requested

    # ----------------------------------------------------------------- merge

    def _weights(self) -> list[float]:
        return [
            float(s.examples) * 2.0 ** (-(self.version - s.base_version) / 8) for s in self.pending
        ]

    def _merged_deltas(self, weights: list[float], denominator: float) -> dict[str, tuple]:
        device = self.bank.device
        merged = {}
        decoded = [s.decoded_deltas(self.bank.geometry.value_dim) for s in self.pending]
        for name in FIELD_COLUMNS:
            ids, rows = [], []
            for deltas, weight in zip(decoded, weights, strict=True):
                if name not in deltas:
                    continue
                edge_ids, delta = deltas[name]
                if name == "value" and not self.bank.learned_values:
                    raise ValueError("Written-phase values merge as mass-weighted means")
                if not bool(torch.isfinite(delta).all()):
                    raise FloatingPointError("Nonfinite submitted transducer delta")
                alive = torch.tensor(
                    [int(e) in self.bank for e in edge_ids.tolist()], dtype=torch.bool
                )
                ids.append(edge_ids[alive])
                rows.append(delta[alive].float() * weight)
            if not ids or not sum(x.numel() for x in ids):
                continue
            unique, inverse = torch.unique(torch.cat(ids), return_inverse=True)
            total = torch.zeros(unique.numel(), rows[0].shape[1]).index_add_(
                0, inverse, torch.cat(rows)
            )
            current = self.bank.decode(name, unique).reshape(unique.numel(), -1)
            target = current + (total / denominator).to(device)
            merged[name] = (unique, unit(target) if name in ADDRESS_FIELDS else target)
        return merged

    def _merged_writes(self, weights: list[float]) -> tuple[Tensor, Tensor, Tensor] | None:
        if self.bank.learned_values:
            return None
        ids, masses, means = [], [], []
        for submission, weight in zip(self.pending, weights, strict=True):
            if not submission.write_ids.numel():
                continue
            decoded = FP4Rows(
                submission.write_means["value_codes"],
                submission.write_means["value_groups"],
                submission.write_means["value_rows"],
            ).decode(self.bank.geometry.value_dim)
            alive = torch.tensor(
                [int(e) in self.bank for e in submission.write_ids.tolist()], dtype=torch.bool
            )
            mass = submission.write_mass[alive].float() * (weight / submission.examples)
            if not bool(torch.isfinite(mass).all() & (mass >= 0).all()):
                raise ValueError("Invalid submitted write mass")
            ids.append(submission.write_ids[alive])
            masses.append(mass)
            means.append(decoded[alive] * mass[:, None])
        if not ids or not sum(x.numel() for x in ids):
            return None
        unique, inverse = torch.unique(torch.cat(ids), return_inverse=True)
        mass = torch.zeros(unique.numel()).index_add_(0, inverse, torch.cat(masses))
        sums = torch.zeros(unique.numel(), self.bank.geometry.value_dim).index_add_(
            0, inverse, torch.cat(means)
        )
        table = self.bank.table
        slots = table.slots(unique)
        age = (self.step - table.last_write[slots]).float().clamp_min(0)
        old_mass = table.write_mass[slots] * torch.exp2(-age / WRITE_HALF_LIFE)
        new_mass = old_mass + mass.to(self.bank.device)
        value = (table.decode("value", slots) * old_mass[:, None] + sums.to(self.bank.device)) / (
            new_mass.clamp_min(1e-30)[:, None]
        )
        return unique, value, new_mass

    def _clusters(self, weights: list[float]) -> list[list[tuple[int, int, float]]]:
        """Group provisional rows from distinct learners that sit at the same place
        and answer the same query (departure and key cosine >= 0.9)."""
        rows = []  # (submission index, row index, weight)
        departures, keys = [], []
        for index, (submission, weight) in enumerate(zip(self.pending, weights, strict=True)):
            if not submission.provisional:
                continue
            packed = submission.provisional
            count = int(packed["ids"].numel())
            rows.extend((index, row, weight) for row in range(count))
            departures.append(self.bank.table.decode_columns("departure", packed))
            keys.append(self.bank.table.decode_columns("key", packed))
        if not rows:
            return []
        d = unit(torch.cat(departures).float())
        k = unit(torch.cat(keys).float())
        similar = ((d @ d.T) >= 0.9) & ((k @ k.T) >= 0.9)
        owner = torch.tensor([index for index, _, _ in rows])
        assigned = torch.zeros(len(rows), dtype=torch.bool)
        clusters = []
        for lead in range(len(rows)):
            if assigned[lead]:
                continue
            members = [lead]
            assigned[lead] = True
            used = {int(owner[lead])}
            for other in torch.nonzero(similar[lead] & ~assigned).flatten().tolist():
                if int(owner[other]) not in used:
                    members.append(other)
                    used.add(int(owner[other]))
                    assigned[other] = True
            clusters.append([rows[m] for m in members])
        return clusters

    def _adopt(self, weights: list[float]) -> tuple[Tensor, dict[tuple[int, int], int]]:
        """Combine provisional rows into new canonical edges; map (worker, local ID) to them."""
        clusters = self._clusters(weights)[: self.bank.edge_room]
        if not clusters:
            return torch.empty(0, dtype=torch.int64), {}
        table = self.bank.table
        decoded: dict[int, dict[str, Tensor]] = {}
        for index, submission in enumerate(self.pending):
            if submission.provisional:
                decoded[index] = {
                    name: table.decode_columns(name, submission.provisional).reshape(
                        int(submission.provisional["ids"].numel()), -1
                    )
                    for name in FIELD_COLUMNS
                }
        fields: dict[str, list[Tensor]] = {name: [] for name in FIELD_COLUMNS}
        masses = []
        for cluster in clusters:
            weight = torch.tensor([w for _, _, w in cluster])
            share = weight / weight.sum()
            mass = torch.stack(
                [self.pending[i].provisional["write_mass"][r].float() for i, r, _ in cluster]
            )
            for name in FIELD_COLUMNS:
                stack = torch.stack([decoded[i][name][r] for i, r, _ in cluster])
                if name == "value" and float(mass.sum()) > 0:
                    combined = (mass[:, None] * stack).sum(0) / mass.sum()
                else:
                    combined = (share[:, None] * stack).sum(0)
                fields[name].append(combined)
            masses.append(mass.sum())
        added = self.bank.add_edges(
            departure=torch.stack(fields["departure"]),
            key=torch.stack(fields["key"]),
            value=torch.stack(fields["value"]),
            destination=torch.stack(fields["destination"]),
            bias=torch.stack(fields["bias"]).reshape(-1),
            write_mass=torch.stack(masses),
            step=self.step,
            worker=COORDINATOR_WORKER,
        )
        mapping = {}
        for edge, cluster in zip(added.tolist(), clusters, strict=True):
            for index, row, _ in cluster:
                local = int(self.pending[index].provisional["ids"][row])
                mapping[self.pending[index].worker, local] = edge
        return added, mapping

    def _fold_evidence(self, weights: list[float], mapping: dict[tuple[int, int], int]) -> None:
        """Fold structural evidence increments, age-weighted, into canonical edges."""
        step = self.step
        for submission, weight in zip(self.pending, weights, strict=True):
            evidence = submission.evidence
            if evidence is None or not evidence.ids.numel():
                continue
            targets = []
            for edge in evidence.ids.tolist():
                edge = mapping.get((submission.worker, edge), edge)
                targets.append(edge if edge in self.bank else None)
            keep = torch.tensor([t is not None for t in targets], dtype=torch.bool)
            if not bool(keep.any()):
                continue
            moved = EvidenceRows(
                torch.tensor([t for t in targets if t is not None], dtype=torch.int64),
                *(
                    getattr(evidence, name)[keep]
                    for name in (
                        "utility",
                        "visits",
                        "dispersion",
                        "sketch",
                        "sketch_count",
                        "energy",
                        "last_step",
                    )
                ),
            )
            self.growth.evidence.add(moved, step, scale=weight / submission.examples)

    def merge(self) -> dict[str, int]:
        """Merge every pending round atomically, evolve structure, publish a version."""
        if not self.pending:
            return {"submitted": 0, "fields": 0, "written": 0, "adopted": 0}
        try:
            weights = self._weights()
            denominator = sum(weights)
            if not denominator > 0:
                raise ValueError("No positive merge evidence")
            self.clock = max(self.clock, *(s.step for s in self.pending))
            # Stage every computed row first: a bad submission changes nothing.
            deltas = self._merged_deltas(weights, denominator)
            writes = self._merged_writes(weights)
            before = set(self.bank.edge_ids().tolist())
            return self._apply(weights, deltas, writes, before, self.clock)
        finally:
            # Applied, staged badly or not, these rounds are consumed: a round that cannot
            # merge must not wedge every later merge, and a retry must never reapply one.
            for submission in self.pending:
                self.ledger[submission.worker] = max(
                    submission.sequence, self.ledger.get(submission.worker, -1)
                )
            self.pending = []

    def _apply(self, weights, deltas, writes, before: set[int], step: int) -> dict[str, int]:
        # The version moves and the touched rows are recorded even if a step below
        # fails halfway, so learners re-sync whatever the merge already changed.
        changed = {int(edge) for ids, _ in deltas.values() for edge in ids.tolist()}
        if writes is not None:
            changed.update(writes[0].tolist())
        submitted = len(self.pending)
        added = torch.empty(0, dtype=torch.int64)
        evolution = {"removed": 0, "split": 0, "touched": ()}
        try:
            for name, (ids, rows) in deltas.items():
                if name in ADDRESS_FIELDS:
                    self.bank.publish_addresses(
                        name, ids, rows, step=step, worker=COORDINATOR_WORKER
                    )
                elif name == "value":
                    self.bank.publish_values(ids, rows, step=step, worker=COORDINATOR_WORKER)
                else:
                    self.bank.publish_bias(ids, rows.reshape(-1))
            if writes is not None:
                ids, value, mass = writes
                self.bank.publish_values(ids, value, step=step, worker=COORDINATOR_WORKER)
                self.bank.set_write_mass(ids, mass, step=step)
            added, mapping = self._adopt(weights)
            self._fold_evidence(weights, mapping)
            self.bank.masters.clear()
            evolution = self.growth.evolve(step, evidence_from=self.evidence_from)
            if self.switch_requested and not self.bank.learned_values:
                self.bank.switch_to_learned()
                self.growth.switch_to_learned()
                self.switch_version = self.version + 1
        finally:
            self.version += 1
            self.bank._changed()
            after = set(self.bank.edge_ids().tolist())
            retired = before - after
            # Splits rewrite their parents; every new edge is a change too.
            changed = (changed | (after - before) | set(evolution["touched"])) & after
            self.history[self.version] = (changed, retired)
            while len(self.history) > CHANGE_HISTORY:
                self.history.popitem(last=False)
        return {
            "submitted": submitted,
            "fields": sum(ids.numel() for ids, _ in deltas.values()),
            "written": 0 if writes is None else int(writes[0].numel()),
            "adopted": int(added.numel()),
            "removed": evolution["removed"],
            "split": evolution["split"],
        }

    def changes_since(self, base_version: int) -> ChangeSet | None:
        """Rows a learner at ``base_version`` must install, or None if it needs a snapshot."""
        if base_version > self.version or (
            base_version < self.version and base_version + 1 not in self.history
        ):
            return None
        changed: set[int] = set()
        retired: set[int] = set()
        for version in range(base_version + 1, self.version + 1):
            new, gone = self.history[version]
            changed = (changed - gone) | new
            retired = (retired - new) | gone
        alive = [edge for edge in changed if edge in self.bank]
        return ChangeSet(
            base_version,
            self.version,
            self.bank.export_rows(alive)
            if alive
            else {k: v[:0] for k, v in self.bank.table.state([]).items()},
            sorted(retired),
            self.bank.table.next_id,
            self.bank.learned_values,
        )

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict:
        if self.pending:
            raise RuntimeError("Checkpoint only after a fully merged coordinator round")
        return {
            "identity": self.bank.identity,
            "version": self.version,
            "clock": self.clock,
            "ledger": dict(self.ledger),
            "switch": self.switch.state_dict(),
            "switch_requested": self.switch_requested,
            "switch_version": self.switch_version,
            "evidence_from": self.evidence_from,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["identity"] != self.bank.identity:
            raise ValueError("Only a v10 coordinator state can be restored")
        self.version = int(state["version"])
        self.clock = int(state["clock"])
        self.ledger = {int(k): int(v) for k, v in state["ledger"].items()}
        self.switch.load_state_dict(state["switch"])
        self.switch_requested = bool(state["switch_requested"])
        self.switch_version = state["switch_version"]
        self.evidence_from = int(state["evidence_from"])
        self.pending = []
        # Change history is not checkpointed: learners re-sync from a snapshot.
        self.history.clear()
