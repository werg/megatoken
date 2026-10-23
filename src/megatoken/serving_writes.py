"""Opt-in, bounded conversation memory; model work is staged before publication."""

from __future__ import annotations

import copy
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from megatoken.model.ingest import IngestItem
from megatoken.model.multitrack import _GroupedTeacherReadout
from megatoken.model.transducer import OrdinaryEvents


@dataclass(frozen=True)
class WritePolicy:
    """Uncalibrated defaults: callers explicitly opt into either writing rule."""

    retrospective: bool = True
    surprise: bool = False
    lag: int = 1
    max_events: int = 128
    max_writes: int = 64
    autosave_every: int = 8
    novelty_cosine: float = 0.9
    surprise_alpha: float = 0.05
    surprise_multiplier: float = 1.2

    def __post_init__(self):
        if (
            self.lag < 0
            or min(self.max_events, self.max_writes, self.autosave_every) < 1
            or not 0 < self.surprise_alpha <= 1
            or not 0 < self.novelty_cosine <= 1
            or not math.isfinite(self.surprise_multiplier)
            or self.surprise_multiplier < 1
        ):
            raise ValueError("Invalid serving write policy")


@dataclass
class SurpriseMeter:
    """Lexical negative log likelihood per observed byte and an exponential baseline."""

    mean: float | None = None

    @torch.no_grad()
    def score(self, multi, state, event, *, effort=6) -> float | None:
        if event.modality != "text" or not event.payload:
            return None
        model = multi.generator
        prediction = multi.tracks.predict_group(
            state, tracks=(event.track,), modalities=("text",), effort=effort
        )
        plan = model.prepare_target(
            target_bytes=event.payload, terminal=event.terminal, role_ids=(event.role_id,)
        )
        track_ids, _ = multi.tracks._track_indices(state, (event.track,))
        controls = multi.tracks.description_controls(state, (event.track,))
        terms = _GroupedTeacherReadout(model.core, model.decoder, model.media_realizers)
        losses = terms.losses_for_selected(prediction, (plan,), controls, state.adapters, track_ids)
        return float(losses["lexical"][0]) / len(event.payload)

    def accept(self, score: float, policy: WritePolicy) -> bool:
        if not math.isfinite(score):
            return False
        high = self.mean is not None and score > self.mean * policy.surprise_multiplier
        self.mean = (
            score
            if self.mean is None
            else ((1 - policy.surprise_alpha) * self.mean + policy.surprise_alpha * score)
        )
        return high


@dataclass(frozen=True)
class RecordedGroup:
    raw: torch.Tensor
    frontier_roles: torch.Tensor
    frontier_modalities: tuple[str, ...]
    frontier_tracks: tuple[str, ...]
    position: int

    @property
    def frontier_count(self):
        return self.raw.shape[1]


@dataclass
class ConversationWrites:
    scope: str
    base: object
    metadata: bytes
    policy: WritePolicy
    meter: SurpriseMeter = field(default_factory=SurpriseMeter)
    events: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    writes: int = 0
    unsaved: int = 0
    turn: int = 0
    # A public journal lets callers inspect exactly which accepted steps wrote rows.
    journal: list = field(default_factory=list)

    def remember_lookup(self, lookup, group):
        if lookup is None or not self.policy.retrospective:
            return
        # Store only the facts needed later, never the live lookup or its autograd graph.
        self.pending.append(
            (
                group + 1,
                {
                    name: getattr(lookup, name).detach().clone()
                    for name in ("start", "start_key", "bank_value", "pair_logits")
                },
            )
        )
        self.pending = self.pending[-self.policy.max_events :]

    def _ordinary(self):
        return OrdinaryEvents(
            torch.cat([e.raw for e, _ in self.events], dim=1),
            torch.cat([e.frontier_roles for e, _ in self.events], dim=1),
            tuple(m for e, _ in self.events for m in e.frontier_modalities),
            tuple(t for e, _ in self.events for t in e.frontier_tracks),
            torch.cat(
                [torch.full_like(e.frontier_roles, e.position) for e, _ in self.events], dim=1
            ),
            torch.cat([torch.full_like(e.frontier_roles, g) for e, g in self.events], dim=1),
        )

    def _retrospective(self, native, *, flush=False):
        if not self.events:
            return []
        first, last = self.events[0][1], self.events[-1][1]
        ready = [
            (read, record)
            for read, record in self.pending
            if first <= read <= last and (flush or read + self.policy.lag <= last)
        ]
        self.pending = [(r, x) for r, x in self.pending if r >= first]
        chosen = {r for r, _ in ready}
        self.pending = [(r, x) for r, x in self.pending if r not in chosen]
        if not ready:
            return []
        hindsight = native.retrospective_values(self._ordinary(), [r for r, _ in ready])
        candidates = []
        beta = float(native.bank.log_beta_location.exp() + native.bank.log_beta_query.exp())
        for index, (_, record) in enumerate(ready):
            if bool(hindsight.sink_only[index]):
                continue
            h = hindsight.values[index].detach().float()
            if float((h - record["bank_value"]).square().sum()) <= 0.25 * float(h.square().sum()):
                continue
            logits = record["pair_logits"]
            best = float(logits.max()) if logits.numel() else 0.0
            candidates.append(
                (
                    record["start"],
                    record["start_key"],
                    h,
                    native.bank.content_location(h),
                    best - beta,
                    "retrospective",
                )
            )
        return candidates

    def _novel(self, key, mounted):
        for mount in mounted:
            table = mount.base.table
            slots = table.alive_slots()
            for chunk in slots.split(4096):
                keys = table.key_unit[chunk].float()
                if keys.numel() and bool(
                    (keys @ F.normalize(key.float(), dim=0) >= self.policy.novelty_cosine).any()
                ):
                    return False
        return True

    @torch.no_grad()
    def commit(self, session, native, multi, ordinary=None, lookup=None, scores=(), *, flush=False):
        """Called under the session lock, strictly after the lookup commits.

        The base transaction covers blends and evictions as well as fresh edges.
        No conversation payload is persisted; only packed memories and provenance.
        """
        fields = (
            self.events,
            self.pending,
            self.writes,
            self.unsaved,
            self.turn,
            copy.copy(self.meter),
            self.journal,
        )
        # Mutate copies so rollback also restores bounded recorder state.
        self.events, self.pending, self.journal = (
            list(self.events),
            list(self.pending),
            list(self.journal),
        )
        try:
            candidates = []
            if ordinary is not None:
                self.turn += 1
                group = session.agenda.lookups - 1 - int(lookup is not None)
                self.events.append(
                    (
                        RecordedGroup(
                            ordinary.raw.detach().clone(),
                            ordinary.frontier_roles.detach().clone(),
                            ordinary.frontier_modalities,
                            ordinary.frontier_tracks,
                            int(ordinary.meridians_seen[0]) - 1,
                        ),
                        group,
                    )
                )
                while sum(e.frontier_count for e, _ in self.events) > self.policy.max_events:
                    self.events.pop(0)
                for index, score in scores:
                    if score is None or not self.meter.accept(score, self.policy):
                        continue
                    rows = native.ingest_rows(
                        multi, [IngestItem(ordinary.raw[0, index], "text", self.metadata)]
                    )
                    candidates.append(
                        (
                            rows.departure[0],
                            rows.key[0],
                            rows.value[0],
                            rows.destination[0],
                            0.0,
                            "surprise",
                        )
                    )
            if self.policy.retrospective:
                candidates += self._retrospective(native, flush=flush)
            if lookup is not None:
                self.remember_lookup(lookup, session.agenda.lookups - 1)
            base = self.base
            with ExitStack() as locks:
                writable = {
                    id(m.base): m.base.lock for m in session.agenda.mounted if m.base.writable
                }
                for _, lock in sorted(writable.items()):
                    locks.enter_context(lock)
                locks.enter_context(base.transaction())
                if ordinary is not None:
                    base.clock += 1
                if lookup is not None:
                    for mount in session.agenda.mounted:
                        if mount.base is base:
                            ids, weights = lookup.base_edges.get(mount.name, ([], []))
                            for edge, weight in zip(ids, weights, strict=True):
                                base.use_counts[int(edge)] = base.use_counts.get(
                                    int(edge), 0
                                ) + float(weight)
                for departure, key, value, destination, bias, rule in candidates:
                    if self.writes >= self.policy.max_writes:
                        break
                    if rule == "surprise" and not self._novel(key, session.agenda.mounted):
                        continue
                    ids = base.edge_ids().tolist()
                    match = None
                    if ids and rule == "retrospective":
                        dep = F.normalize(base.decode("departure", ids), dim=-1)
                        keys = F.normalize(base.decode("key", ids), dim=-1)
                        similar = (dep @ F.normalize(departure, dim=0) >= 0.9) & (
                            keys @ F.normalize(key, dim=0) >= 0.9
                        )
                        hits = torch.nonzero(similar).flatten().tolist()
                        match = ids[hits[0]] if hits else None
                    if match is not None:
                        base.blend(
                            [match],
                            value[None],
                            1.0,
                            base.clock,
                            destination=native.bank.content_location,
                        )
                        added = [match]
                    else:
                        base.make_room(1)
                        added = base.add_edges(
                            departure=departure[None],
                            key=key[None],
                            value=value[None],
                            destination=destination[None],
                            bias=torch.tensor([bias]),
                            write_mass=torch.ones(1),
                            step=base.clock,
                        ).tolist()
                        base.masters[added[0]] = value.detach().float().clone()
                    for edge in added:
                        sessions = set(base.provenance.get(edge, {}).get("sessions", []))
                        sessions.add(session.session_id)
                        base.provenance[edge] = dict(
                            scope=self.scope,
                            session=session.session_id,
                            sessions=sorted(sessions),
                            turn=self.turn,
                            rule=rule,
                            time=time.time(),
                        )
                    self.journal.append(dict(turn=self.turn, rule=rule, ids=added))
                    self.writes += 1
                    self.unsaved += 1
                if self.unsaved >= self.policy.autosave_every or flush:
                    if base.path is not None:
                        base.save(base.path)
                        self.unsaved = 0
            self.journal = self.journal[-self.policy.max_writes :]
        except BaseException:
            (
                self.events,
                self.pending,
                self.writes,
                self.unsaved,
                self.turn,
                self.meter,
                self.journal,
            ) = fields
            raise
