"""v10 training orchestration: boundaries, hindsight curriculum, sparse barriers.

One ``TransducerTrainer`` owns the bank's sparse optimizer and growth for a
run. Each training window gets a ``TransducerSequence``: it plans the window's
latency boundaries, runs the hindsight pass once, performs every lookup and
records the differentiable ones. After the window's backward,
``after_backward`` turns their gradients into hindsight writes, structural
evidence and write-out candidates; ``barrier`` then steps the bank.

Serving uses ``BoundaryScheduler`` for the same boundary rule, live.

The same hindsight pass trains the ingest heads: wherever a lookup's hindsight
group is, or contains, exactly one item (an image, a clip, a document piece),
that item read alone should give the value the hindsight head gave
(distillation), and its key should be the one the lookup's own query picks out
among the window's items (contrast).
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

    full_steps: int = 4096
    gap1_steps: int = 4096
    evidence_steps: int = 16384
    hindsight_only_every: int = 8

    def stage(self, step: int) -> str:
        if step < self.full_steps:
            return "full"
        return "gap1" if step < self.full_steps + self.gap1_steps else "gap2"

    def overlay(self, step: int, sequence: int) -> float:
        """Only one explicitly selected teacher window in eight; no all-teacher warmup."""
        if step < 0 or sequence < 0:
            raise ValueError("Training step and sequence counters must be nonnegative")
        return 1.0 if sequence % self.hindsight_only_every == 0 else 0.0

    def computes_hindsight(self, step: int, sequence: int, *, learned: bool) -> bool:
        # Before the switch every lookup writes h_t into its edges; after it,
        # the pass runs only for the hindsight-only sequences.
        return not learned or self.overlay(step, sequence) > 0

    def reads(self, step: int, groups: int, lookups: int) -> list[int | None]:
        """The retrieval group each lookup's h_t reads (None: past the end)."""
        gap = 1 if self.stage(step) == "gap2" else 0
        return [t + gap if t + gap < groups else None for t in range(lookups)]


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
        bank_trainable: bool = True,
        worker: int = 0,
        provisional: bool = False,
        schedule: HindsightSchedule | None = None,
        max_writeouts: int = 512,
        cache_edges: int = 8192,
        reading: ReadingBase | None = None,
    ) -> None:
        self.native = native
        self.bank_trainable = bool(bank_trainable)
        self.bank = native.bank
        self.optimizer = BlockScaleDescent(self.bank, worker=worker, cache_edges=cache_edges)
        self.growth = TransducerGrowth(
            self.bank,
            optimizer=self.optimizer,
            provisional=provisional,
            max_writeouts=max_writeouts,
        )
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
    ) -> TransducerSequence:
        """Plan one training window (or a bank-only evaluation window when not training).

        ``mounted`` are the bases the window searches (default: the trained base alone).
        A window that mounts others learns nothing into the trained base: what those
        bases hold is not to be memorized by it. ``sources`` are the collections the
        window's session has mounted.
        """
        training = self.native.training
        lookups = plan_boundaries(
            meridians,
            max_lookups=self.bank.geometry.max_lookups,
            generator=generator if training else None,
        )
        question = question_lookup(meridians, lookups)
        if not training or not self.bank_trainable:
            return TransducerSequence(
                self,
                lookups,
                step,
                alpha=0.0,
                hindsight=False,
                mounted=mounted,
                sources=sources,
                question=question,
            )
        sequence = self.sequences
        self.sequences += 1
        learned = self.bank.learned_values
        return TransducerSequence(
            self,
            lookups,
            step,
            alpha=self.schedule.overlay(step, sequence) if teacher_supervision else 0.0,
            hindsight=teacher_supervision
            and self.schedule.computes_hindsight(step, sequence, learned=learned),
            generator=generator,
            mounted=mounted,
            sources=sources,
            question=question,
        )

    def after_backward(self, step: int) -> dict[str, int]:
        """Hindsight writes, structural evidence and write-out candidates of the window."""
        if not self.bank_trainable:
            if self.records or self.candidates:
                raise RuntimeError("Frozen memory cannot accumulate sparse writes")
            return {"transducer_observed": 0, "transducer_writes": 0}
        g = self.bank.geometry
        writeout_stage = self.schedule.stage(step) == "gap2"
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
            "switch": self.switch.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schedule") != asdict(self.schedule):
            raise ValueError("Memory training requires the exact current schedule")
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
        hindsight: bool,
        generator: torch.Generator | None = None,
        mounted: tuple[Mounted, ...] | None = None,
        sources: SourceSession | None = None,
        question: int | None = None,
    ) -> None:
        self.trainer = trainer
        self.native = trainer.native
        self.before = lookups_before
        self.step = step
        self.alpha = alpha
        self.wants_hindsight = hindsight
        self.generator = generator
        self.mounted = mounted
        self.values: HindsightValues | None = None
        self.items: IngestedWindow | None = None
        self.queries: list[Tensor] = []  # the unit query of every lookup performed
        self.agenda: SearchAgenda | None = None
        self.lookups = 0
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
        if not self.wants_hindsight or not self.planned:
            return
        schedule = self.trainer.schedule
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
        stage = schedule.stage(self.step)
        reads = schedule.reads(self.step, groups[-1] + 1, self.planned)
        # With no overlay h_t never reaches the loss: it only feeds writes and evidence.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.alpha > 0):
            self.values = self.native.hindsight_pass(
                events, reads, mode="full" if stage == "full" else "group-inverse"
            )
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
        items = [k for k, modality in enumerate(modalities) if modality in FEATURE_MODALITIES]
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
                _Recorded(lookup, h, bool(sink_only), alpha, writes=self.mounted is None)
            )
            self.queries.append(lookup.query.detach())
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
        pairs = [(t, k) for t, k in window.matches if t < len(self.queries)]
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
