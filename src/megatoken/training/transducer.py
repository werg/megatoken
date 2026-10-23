"""Memory training with prepared teacher values and sparse bank updates.

Each window plans causal lookups. Supported text windows attach immutable teacher
values; an isolated future-only pass distills the serving hindsight head. Item
encoders learn against the same prepared targets. Backward gradients drive sparse
updates and candidate entries, committed at the bank barrier. Windows without
teacher support perform causal retrieval without substitute targets or writes.

Serving uses BoundaryScheduler for the same boundary rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F

from megatoken.data.media_assets import FEATURE_MODALITIES
from megatoken.memory.transducer import Lookup, Mounted, SearchAgenda
from megatoken.memory.transducer_growth import TransducerGrowth, WriteOut
from megatoken.memory.transducer_optimizer import BlockScaleDescent
from megatoken.model.multitrack import MultiTrackState
from megatoken.model.transducer import HindsightValues, NativeTransducer, OrdinaryEvents
from megatoken.retrieval.session import Search, SourceSession

from .reading_base import ReadingBase
from .transducer_diloco import LOCAL_STEPS, SWITCH_EVERY, SwitchRule

ARTIFICIAL_INTERVAL = 8
TRAINING_INTERVALS = (4, 12)
CREDIT_GROUPS = 4
# Weights of the ingest losses (each a mean over the window's matched lookups) and the
# contrast's temperature: keys and queries are unit vectors, so logits are cosines / this.
INGEST_VALUE_WEIGHT = 0.1
INGEST_KEY_WEIGHT = 0.05
INGEST_KEY_TEMPERATURE = 0.05


@dataclass
class BoundaryScheduler:
    """When a session looks up: after metadata, at real boundaries, and after
    ``interval`` ordinary meridians without one. Internal events never count.

    Each lookup is finished before the next meridian is counted; a cancelled
    lookup waits for the next boundary.
    """

    # Training draws each interval from this generator; serving uses the fixed one.
    generator: torch.Generator | None = field(default=None, repr=False)
    max_lookups: int = 256
    lookups: int = 0
    ordinary_since: int = 0
    interval: int = ARTIFICIAL_INTERVAL

    def __post_init__(self) -> None:
        self._next_interval()

    def _next_interval(self) -> None:
        if self.generator is not None:
            low, high = TRAINING_INTERVALS
            self.interval = int(torch.randint(low, high + 1, (1,), generator=self.generator))
        else:
            self.interval = ARTIFICIAL_INTERVAL

    def first_metadata(self) -> bool:
        """Immediately after the metadata event, before any answer."""
        return self.lookups < self.max_lookups

    def ordinary_commit(self, *, real_boundary: bool = False) -> bool:
        """Count one committed ordinary meridian (observed or generated)."""
        self.ordinary_since += 1
        return (real_boundary or self.ordinary_since >= self.interval) and self.first_metadata()

    def finish(self, *, committed: bool) -> None:
        """Close a lookup; a cancelled one waits for the next boundary, not the next meridian."""
        self.ordinary_since = 0
        if committed:
            self.lookups += 1
            self._next_interval()


@dataclass(frozen=True)
class HindsightSchedule:
    """Explicit auxiliary teacher windows; causal student windows begin immediately."""

    evidence_steps: int = 16384
    hindsight_only_every: int = 8

    def overlay(self, step: int, sequence: int) -> float:
        """Only one explicitly selected teacher window in eight; no all-teacher warmup."""
        if step < 0 or sequence < 0:
            raise ValueError("Training step and sequence counters must be nonnegative")
        return 1.0 if sequence % self.hindsight_only_every == 0 else 0.0


@dataclass(frozen=True)
class Meridian:
    """What the boundary planner and hindsight pass need from one ordinary meridian."""

    tracks: tuple[str, ...]
    roles: tuple[int, ...]
    modalities: tuple[str, ...]
    real_boundary: bool
    # It finishes a user message: the lookup after it is the one that sees the question.
    asked: bool = False


def plan_boundaries(
    meridians: Sequence[Meridian], *, max_lookups: int, generator: torch.Generator | None
) -> list[int]:
    """Lookups per meridian: how many happen right before meridian i is committed.

    The first follows the metadata event (before meridian 0). A boundary after
    the last meridian would serve nothing in this window and is not planned.
    """
    scheduler = BoundaryScheduler(generator=generator, max_lookups=max_lookups)
    before = [0] * len(meridians)
    if not meridians or not scheduler.first_metadata():
        return before
    before[0] = 1
    scheduler.finish(committed=True)
    for index, meridian in enumerate(meridians[:-1]):
        if scheduler.ordinary_commit(real_boundary=meridian.real_boundary):
            before[index + 1] = 1
            scheduler.finish(committed=True)
    return before


def question_lookup(meridians: Sequence[Meridian], before: Sequence[int]) -> int | None:
    """The lookup that follows the window's first finished user message, if one is planned.

    ``before`` is ``plan_boundaries``'s answer. A message that is the window's last
    meridian is followed by no lookup of this window.
    """
    asked = next((i for i, meridian in enumerate(meridians) if meridian.asked), None)
    if asked is None or asked + 1 >= len(meridians) or not before[asked + 1]:
        return None
    return sum(before[: asked + 2]) - 1


@dataclass
class _Recorded:
    lookup: Lookup
    hindsight: Tensor | None
    sink_only: bool
    alpha: float
    writes: bool = True  # whether this lookup may write into the trained base


@dataclass(frozen=True)
class IngestedWindow:
    """The window's items read alone, and which lookup each one matches."""

    values: Tensor  # [items, value_dim] (FP4-exact)
    keys: Tensor  # [items, query_dim], unit
    matches: tuple[tuple[int, int], ...]  # (lookup, item) pairs: the lookup's group is/has the item


class TransducerTrainer:
    """The run's single owner of the bank's sparse optimizer, growth and schedule."""

    def __init__(
        self,
        native: NativeTransducer,
        *,
        worker: int = 0,
        provisional: bool = False,
        schedule: HindsightSchedule | None = None,
        max_writeouts: int = 512,
        cache_edges: int = 8192,
        reading: ReadingBase | None = None,
        teacher_targets=None,
        bank_trainable: bool = True,
        teacher_weight: float = 0.1,
        navigation_weight: float = 0.0,
        navigation_examples: int = 8,
        navigation_negatives: int = 32,
    ) -> None:
        self.native = native
        self.bank = native.bank
        self.optimizer = BlockScaleDescent(self.bank, worker=worker, cache_edges=cache_edges)
        self.growth = TransducerGrowth(
            self.bank,
            optimizer=self.optimizer,
            provisional=provisional,
            max_writeouts=max_writeouts,
        )
        self.teacher_targets = teacher_targets
        self.bank_trainable = bank_trainable
        self.teacher_weight = teacher_weight
        self.navigation_weight = navigation_weight
        self.navigation_examples = navigation_examples
        self.navigation_negatives = navigation_negatives
        self.schedule = schedule or HindsightSchedule()
        self.sequences = 0
        self.records: list[_Recorded] = []
        self.candidates: list[WriteOut] = []
        self._writes = 0
        self.switch = SwitchRule()
        # Ingested bases of the reading-from-a-base stream: built here, never checkpointed or synced.
        self.reading = reading if reading is not None else ReadingBase(native)

    def sequence(
        self,
        meridians: Sequence[Meridian],
        *,
        step: int,
        generator: torch.Generator | None = None,
        mounted: tuple[Mounted, ...] | None = None,
        sources: SourceSession | None = None,
        teacher_supervision: bool = False,
        memory_eligible: bool | None = None,
    ) -> TransducerSequence:
        """Plan one training window (or a bank-only evaluation window when not training).

        ``mounted`` are the bases the window searches (default: the trained base alone).
        ``memory_eligible`` explicitly controls lasting writes of this window's own targets;
        declaring reference bases alone never grants that permission. ``sources`` are the collections the
        window's session has mounted.
        """
        training = self.native.training
        if memory_eligible is None:
            memory_eligible = mounted is None
        lookups = plan_boundaries(
            meridians,
            max_lookups=self.bank.geometry.max_lookups,
            generator=None,
        )
        question = question_lookup(meridians, lookups)
        if not training or not self.bank_trainable:
            return TransducerSequence(
                self,
                lookups,
                step,
                alpha=0.0,
                mounted=mounted,
                sources=sources,
                question=question,
                memory_eligible=False,
            )
        sequence = self.sequences
        self.sequences += 1
        return TransducerSequence(
            self,
            lookups,
            step,
            alpha=self.schedule.overlay(step, sequence)
            if teacher_supervision and self.teacher_targets is not None
            else 0.0,
            generator=generator,
            mounted=mounted,
            sources=sources,
            question=question,
            memory_eligible=memory_eligible,
        )

    def after_backward(self, step: int) -> dict[str, int]:
        """Hindsight writes, structural evidence and write-out candidates of the window."""
        if not self.bank_trainable:
            if self.records or self.candidates:
                raise RuntimeError("Frozen memory cannot accumulate sparse writes")
            return {"transducer_observed": 0, "transducer_writes": 0}
        g = self.bank.geometry
        writeout_stage = self.teacher_targets is not None
        observed = writes = 0
        for record in self.records:
            lookup = record.lookup
            if record.alpha >= 1:
                # The bank's output never reached the loss: no structural evidence,
                # but h_t is still written into the edges it would have used.
                self._hindsight_only(record, step, writeout_stage)
                continue

            def gradient(tensor: Tensor, width: int) -> Tensor:
                grad = tensor.grad if tensor.requires_grad else None
                return torch.zeros(width, device=self.bank.device) if grad is None else grad

            self.growth.observe(
                lookup,
                gradient_value=gradient(lookup.bank_value, g.value_dim),
                gradient_location=gradient(lookup.bank_landing_raw, g.location_dim),
                step=step,
                hindsight=record.hindsight,
                sink_only=record.sink_only,
            )
            observed += 1
            usable = record.hindsight is not None and not record.sink_only and record.writes
            if usable and not self.bank.learned_values and lookup.edge_ids.numel():
                self.optimizer.add_write(lookup.edge_ids, lookup.edge_weights, record.hindsight)
                writes += 1
            if usable and writeout_stage and record.alpha > 0:
                entry = self.growth.candidate(
                    lookup, record.hindsight, gradient(lookup.value, g.value_dim)
                )
                if entry is not None:
                    self.candidates.append(entry)
        self.growth.observe_gradients(step)
        writes += self._writes
        self._writes = 0
        self.records.clear()
        return {"transducer_observed": observed, "transducer_writes": writes}

    def _hindsight_only(self, record: _Recorded, step: int, writeout_stage: bool) -> None:
        lookup = record.lookup
        if record.hindsight is None or record.sink_only or not record.writes:
            return
        if not self.bank.learned_values and lookup.edge_ids.numel():
            self.optimizer.add_write(lookup.edge_ids, lookup.edge_weights, record.hindsight)
            self._writes += 1
        if writeout_stage:
            grad = lookup.value.grad
            if grad is None:
                grad = torch.zeros_like(record.hindsight)
            entry = self.growth.candidate(lookup, record.hindsight, grad)
            if entry is not None:
                self.candidates.append(entry)

    def barrier(self, step: int, *, evolve: bool = True) -> dict[str, int]:
        """Step the touched rows, add write-outs; in a single-owner run, evolve per round."""
        if self.records:
            raise RuntimeError("Collect backward evidence before the sparse step")
        if not self.bank_trainable:
            return {
                "transducer_updated_fields": 0,
                "transducer_written_values": 0,
                "transducer_writeouts": 0,
                "transducer_edges": self.bank.edge_count,
                "transducer_frozen": 1,
            }
        result = self.optimizer.step(step)
        added = self.growth.write_out(self.candidates, step=step)
        self.candidates.clear()
        metrics = {
            "transducer_updated_fields": result["updated_fields"],
            "transducer_written_values": result["written_values"],
            "transducer_writeouts": int(added.numel()),
            "transducer_edges": self.bank.edge_count,
        }
        if evolve and step % LOCAL_STEPS == 0:
            evolution = self.growth.evolve(step, evidence_from=self.schedule.evidence_steps)
            # No coordinator consumes round increments here; drop them.
            self.growth.begin_round()
            self.growth.discard_round()
            self.optimizer.discard_round()
            metrics["transducer_removed"] = evolution["removed"]
            metrics["transducer_split"] = evolution["split"]
        return metrics

    def switch_due(self, step: int) -> bool:
        """Bank-only validation runs every 1,024 steps once the overlay has ended."""
        schedule = self.schedule
        return (
            self.bank_trainable
            and not self.bank.learned_values
            and step % SWITCH_EVERY == 0
            and step >= schedule.evidence_steps
        )

    def record_validation(self, step: int, bank_only_loss: float) -> bool:
        """Single-owner switch rule; returns True when values switch to learned."""
        if not self.switch.record(step, bank_only_loss, self.bank):
            return False
        self.optimizer.switch_to_learned()
        self.growth.switch_to_learned()
        return True

    def state_dict(self) -> dict:
        if self.records or self.candidates:
            raise RuntimeError("Only a complete forward/backward/step window is checkpointable")
        return {
            "schedule": asdict(self.schedule),
            "bank": self.bank.state_snapshot(),
            "optimizer": self.optimizer.state_dict(),
            "growth": self.growth.state_dict(),
            "sequences": self.sequences,
            "supervision": self.supervision_identity(),
            "switch": self.switch.state_dict(),
        }

    def supervision_identity(self):
        targets = self.teacher_targets
        return {
            "teacher": None if targets is None else targets.manifest["encoding_identity"],
            "targets": None if targets is None else targets.identity,
            "memory_policy": (
                "causal-only-v1" if targets is None else targets.manifest["memory_policy"]
            ),
            "bank_trainable": self.bank_trainable,
            "teacher_weight": self.teacher_weight,
            "navigation_weight": self.navigation_weight,
            "navigation_examples": self.navigation_examples,
            "navigation_negatives": self.navigation_negatives,
        }

    def validate_supervision(self, state, *, allow_target_transition=False):
        if state.get("schedule") != asdict(self.schedule):
            raise ValueError("Memory training requires the exact current schedule")
        if "supervision" not in state:
            raise ValueError("Memory training requires explicit supervision identity")
        prior = state["supervision"]
        current = self.supervision_identity()
        if not isinstance(prior, dict) or prior.keys() != current.keys():
            raise ValueError("Memory training requires complete current supervision fields")
        if prior["teacher"] != current["teacher"]:
            if state["bank"]["rows"]["ids"].numel():
                raise ValueError(
                    "Checkpoint teacher value identity changed; explicit handover required"
                )
        if not allow_target_transition and prior["targets"] != current["targets"]:
            raise ValueError("Checkpoint teacher target identity changed during resume")

    def load_state_dict(self, state: dict, *, allow_target_transition=False) -> None:
        self.validate_supervision(state, allow_target_transition=allow_target_transition)
        self.bank.load_snapshot(state["bank"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.growth.load_state_dict(state["growth"])
        self.sequences = int(state["sequences"])
        self.switch.load_state_dict(state["switch"])


class TransducerSequence:
    """One window's lookups, in order, through the native boundary transaction."""

    def __init__(
        self,
        trainer: TransducerTrainer,
        lookups_before: list[int],
        step: int,
        *,
        alpha: float,
        generator: torch.Generator | None = None,
        mounted: tuple[Mounted, ...] | None = None,
        sources: SourceSession | None = None,
        question: int | None = None,
        memory_eligible: bool = True,
    ) -> None:
        self.trainer = trainer
        self.native = trainer.native
        self.before = lookups_before
        self.step = step
        self.alpha = alpha
        self.generator = generator
        self.mounted = mounted
        self.memory_eligible = memory_eligible and self.native.training
        self.values: HindsightValues | None = None
        self.items: IngestedWindow | None = None
        self.queries: dict[int, Tensor] = {}  # the unit query of every lookup performed
        self.agenda: SearchAgenda | None = None
        self.lookups = 0
        self.stored: HindsightValues | None = None
        self.teacher_loss: Tensor | None = None
        self.sources = sources
        # The search of the lookup after the first finished user message (``question``,
        # counted among the window's lookups), which the data may annotate.
        self.question = question
        self.question_search: Search | None = None

    @property
    def planned(self) -> int:
        return sum(self.before)

    def groups(self) -> list[int]:
        """The retrieval group of every meridian (lookup t precedes group t)."""
        result, current = [], -1
        for count in self.before:
            current += count
            result.append(max(current, 0))
        return result

    def hindsight(
        self,
        meridians: Sequence[Meridian],
        raw: Callable[[int], Tensor],
    ) -> None:
        """Run the hindsight pass once for every planned lookup (training only)."""
        if self.stored is None or not self.planned:
            return
        groups = self.groups()
        blocks, roles, tracks, modalities, positions, owners = [], [], [], [], [], []
        for index, meridian in enumerate(meridians):
            block = raw(index)
            blocks.append(block[0])
            roles.extend(meridian.roles)
            tracks.extend(meridian.tracks)
            modalities.extend(meridian.modalities)
            positions.extend([index + 1] * len(meridian.tracks))
            owners.extend([groups[index]] * len(meridian.tracks))
        device = self.native.bank.device
        events = OrdinaryEvents(
            torch.cat(blocks)[None],
            torch.tensor([roles], device=device),
            tuple(modalities),
            tuple(tracks),
            torch.tensor([positions], device=device),
            torch.tensor([owners], device=device),
        )
        reads = [t + 1 if t + 1 <= groups[-1] else None for t in range(self.planned)]
        self.values = self.stored
        if torch.is_grad_enabled() and self.trainer.teacher_weight > 0:
            valid = ~self.stored.sink_only
            if bool(valid.any()):
                learned = self.native.hindsight_pass(events, reads)
                target = self.stored.values[valid].detach().float()
                error = (learned.values[valid].float() - target).square().sum(-1)
                self.teacher_loss = (error / target.square().sum(-1).clamp_min(1e-6)).mean()
        self._ingest(events, meridians, reads)

    def _ingest(
        self, events: OrdinaryEvents, meridians: Sequence[Meridian], reads: Sequence[int | None]
    ) -> None:
        """Read the window's items alone, and match each lookup whose group is or has one item.

        A group with no item, or more than one, matches nothing; neither does a
        lookup whose group lies past the end (its hindsight value is the sink's alone).
        """
        if not torch.is_grad_enabled():
            return
        modalities = [m for meridian in meridians for m in meridian.modalities]
        items = [
            k for k, modality in enumerate(modalities) if modality in (*FEATURE_MODALITIES, "text")
        ]
        owners = events.groups[0].tolist()
        matches = []
        for t, group in enumerate(reads):
            inside = [k for k in items if owners[k] == group]
            if group is not None and len(inside) == 1:
                matches.append((t, items.index(inside[0])))
        if not matches:
            return
        values, keys = self.native.ingest(
            self.native.core, events.raw[0, items], [modalities[k] for k in items]
        )
        self.items = IngestedWindow(values, keys, tuple(matches))

    def begin(
        self, multi, state: MultiTrackState, payload: bytes, *, encoder_grad: bool = False
    ) -> MultiTrackState:
        state, self.agenda = self.native.begin(
            multi, state, payload, encoder_grad=encoder_grad, mounted=self.mounted
        )
        return state

    def before_meridian(
        self, multi, state: MultiTrackState, index: int, *, effort: int = 6
    ) -> MultiTrackState:
        """Perform the lookups planned right before ordinary meridian ``index``."""
        if self.agenda is None:
            raise RuntimeError("The metadata prefix must be committed first")
        for _ in range(self.before[index]):
            state = self._lookup(multi, state, effort)
        return state

    def _lookup(self, multi, state: MultiTrackState, effort: int) -> MultiTrackState:
        t = self.lookups
        self.lookups += 1
        h = sink_only = None
        if self.values is not None:
            h = self.values.values[t]
            sink_only = bool(self.values.sink_only[t])
            if self.stored is not None and sink_only:
                # A dense zero placeholder is not a teacher value: retrieve normally.
                h = None
        alpha = self.alpha if h is not None else 0.0
        differentiable = torch.is_grad_enabled() and self.native.training
        if not differentiable and h is not None:
            h = h.detach()
        state, lookup = self.native.boundary(
            multi,
            state,
            self.agenda,
            training=self.native.training,
            hindsight=h,
            overlay=alpha,
            generator=self.generator,
            # With alpha = 1 the bank's output never reaches the loss.
            train_rows=differentiable and self.trainer.bank_trainable and alpha < 1,
            effort=effort,
            sources=self.sources,
        )
        if lookup is None:
            return state
        if t == self.question and self.sources is not None:
            self.question_search = self.sources.last
        self.agenda.truncate_credit(CREDIT_GROUPS)
        if differentiable and self.trainer.bank_trainable:
            for tensor in (lookup.value, lookup.bank_value, lookup.bank_landing_raw):
                if tensor.requires_grad:
                    tensor.retain_grad()
            self.trainer.records.append(
                _Recorded(lookup, h, bool(sink_only), alpha, writes=self.memory_eligible)
            )
            self.queries[t] = lookup.query.detach()
        return state

    def ingest_losses(self) -> dict[str, Tensor]:
        """Window means of the ingest heads' two losses; empty when no lookup matched an item.

        Value: the item read alone should equal the hindsight value of its lookup, relative
        to that value's size. Key: among the window's items, the lookup's own query should
        pick out its item. Hindsight values and queries are targets (no gradient).
        """
        window = self.items
        if window is None or self.values is None:
            return {}
        # A lookup that was cancelled or hit the cap has no query, so no pair.
        pairs = [(t, k) for t, k in window.matches if t in self.queries]
        if not pairs:
            return {}
        target = torch.stack([self.values.values[t].detach().float() for t, _ in pairs])
        item = torch.tensor([k for _, k in pairs], device=target.device)
        error = (window.values[item] - target).square().sum(-1)
        losses = {"ingest_value": (error / target.square().sum(-1).clamp_min(1e-6)).mean()}
        if window.keys.shape[0] > 1:
            queries = torch.stack([self.queries[t] for t, _ in pairs]).float()
            logits = queries @ window.keys.float().T / INGEST_KEY_TEMPERATURE
            losses["ingest_key"] = F.cross_entropy(logits, item)
        return losses
