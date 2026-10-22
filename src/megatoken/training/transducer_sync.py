"""Transport of v10 transducer rounds between DiLoCo learners and the coordinator.

The coordinator process hosts the canonical bank (``TransducerHost``); each
learner keeps a ``TransducerSync`` cursor. Learners upload one
``LearnerRound`` per 32 local steps and install published change sets between
optimizer steps; they never block on the network. Files are ``torch.save``
archives of tensors and plain values only, loaded with ``weights_only=True``.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from megatoken.io import atomic_binary_writer
from megatoken.memory.transducer import TransducerBank, TransducerGeometry
from megatoken.memory.transducer_growth import TransducerGrowth
from megatoken.memory.transducer_optimizer import BlockScaleDescent

from .checkpoint import snapshot_to_cpu
from .transducer_diloco import (
    LOCAL_STEPS,
    ChangeSet,
    LearnerRound,
    TransducerCoordinator,
    install_changes,
    learner_round,
)


class ChangesExpired(LookupError):
    """The learner's base is older than the kept change history: fetch a snapshot."""


class StaleRound(ValueError):
    """A round built on a bank newer than the canonical one: the learner must re-base."""


class DuplicateRound(ValueError):
    """A round the coordinator already holds; ``sequence`` is its latest for that learner."""

    def __init__(self, message: str, sequence: int) -> None:
        super().__init__(message)
        self.sequence = sequence


def _save(path: Path, value: dict) -> None:
    with atomic_binary_writer(path) as handle:
        torch.save(value, handle)


def _load(path: Path) -> dict:
    # Read through a handle: torch dispatches on the file suffix of a path.
    with Path(path).open("rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=True)


def load_transducer_state(checkpoint: str | Path) -> dict:
    """A checkpoint's ``transducer.pt``, memory-mapped so a caller reads only what it uses."""
    return torch.load(
        Path(checkpoint) / "transducer.pt", map_location="cpu", weights_only=True, mmap=True
    )


def resume_transducer(trainer: Any, rank_rounds: Any, checkpoint: str | Path) -> None:
    """Restore a run's transducer from a checkpoint, however many ranks wrote or read it."""
    saved = load_transducer_state(checkpoint)
    if rank_rounds is not None:
        rank_rounds.load_state_dict(saved)
    elif "canonical" in saved:
        # A multi-owner bank (ranks or coordinator); this learner's statistics restart.
        trainer.bank.load_snapshot(saved["bank"])
        trainer.sequences = int(saved.get("sequences", 0))
    else:
        trainer.load_state_dict(saved)


class CanonicalBank:
    """The run's canonical bank and its merging coordinator, kept on the CPU."""

    def __init__(self, geometry: TransducerGeometry, *, evidence_from: int) -> None:
        self.bank = TransducerBank(geometry)
        self.growth = TransducerGrowth(self.bank)
        self.coordinator = TransducerCoordinator(
            self.bank, self.growth, evidence_from=evidence_from
        )
        self.skipped_merges = 0

    def version(self) -> int:
        return self.coordinator.version

    def state_dict(self) -> dict:
        return {
            "bank": self.bank.state_snapshot(),
            "growth": self.growth.state_dict(),
            "coordinator": self.coordinator.state_dict(),
        }

    def checkpoint_state(self) -> dict:
        """The ``transducer.pt`` layout: the served bank plus the canonical structure state."""
        state = self.state_dict()
        return {"bank": state.pop("bank"), "canonical": state}

    def load_state_dict(self, state: dict) -> None:
        self.bank.load_snapshot(state["bank"])
        self.growth.load_state_dict(state["growth"])
        self.coordinator.load_state_dict(state["coordinator"])

    def seed(self, state: dict) -> None:
        """Start a new run from a checkpoint's bank, keeping its structural evidence if any.

        Coordinator versions, ledgers, switch history and clock belong to the old run: the
        new clock starts at 0, so every age stamp the old run left on a row (birth, last
        write, evidence recency) is reset to 0 with it. Otherwise rows could be neither
        removed nor decayed until the new clock passed the old run's last step.
        """
        self.bank.load_snapshot(state["bank"])
        if "canonical" in state:
            self.growth.load_state_dict(state["canonical"]["growth"])
        self.bank.table.birth.zero_()
        self.bank.table.last_write.zero_()
        self.growth.evidence.last_step.zero_()

    def merge_or_skip(self) -> bool:
        try:
            self.coordinator.merge()
        except (ValueError, FloatingPointError, RuntimeError) as error:
            # The failing rounds were consumed and are never retried. A failure that struck
            # before any row changed leaves the version where it was; one that struck
            # halfway has moved it, and learners re-sync from that version.
            self.skipped_merges += 1
            print(f"transducer merge failed and was skipped: {error}", flush=True)
            return False
        return True


def _pack(value: dict) -> bytes:
    # Plain pickling cannot carry FP8 storages between processes; torch archives can.
    buffer = io.BytesIO()
    torch.save(snapshot_to_cpu(value), buffer)
    return buffer.getvalue()


def _unpack(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


class TransducerHost(CanonicalBank):
    """The DiLoCo coordinator's canonical bank, its pending rounds and its publications."""

    def __init__(
        self,
        root: str | Path,
        settings: dict,
        *,
        workers: int,
        min_quorum: int,
        grace_seconds: float,
        persist_every: int = 16,
    ) -> None:
        super().__init__(
            TransducerGeometry(**settings["geometry"]),
            evidence_from=int(settings["evidence_from"]),
        )
        self.root = Path(root) / "transducer"
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.min_quorum = min_quorum
        self.grace_seconds = grace_seconds
        self.persist_every = persist_every
        self.quorum_since: float | None = None
        self.published_version = -1
        state = self.root / "state.pt"
        if state.exists():
            self.load_state_dict(_load(state))

    def persist(self) -> None:
        _save(self.root / "state.pt", self.state_dict())

    def submit(self, worker: int, sequence: int, base_version: int, incoming: Path) -> dict:
        try:
            submission = LearnerRound.from_state(_load(incoming))
        except Exception as exc:
            raise ValueError("Malformed transducer round") from exc
        if (submission.worker, submission.sequence, submission.base_version) != (
            worker,
            sequence,
            base_version,
        ):
            raise ValueError("Transducer round headers disagree with its contents")
        if not 0 <= worker < self.workers:
            raise ValueError("Unknown learner")
        if base_version > self.version():
            raise StaleRound("Stale transducer base: newer than the canonical bank")
        if not self.coordinator.submit(submission):
            raise DuplicateRound("Duplicate transducer round", self.sequence(worker))
        return {"transducer_version": self.version(), "pending": len(self.coordinator.pending)}

    def sequence(self, worker: int) -> int:
        """The newest round of ``worker`` the coordinator holds, merged or queued (0: none)."""
        held = [s.sequence for s in self.coordinator.pending if s.worker == worker]
        return max([self.coordinator.ledger.get(worker, 0), *held])

    def record_validation(self, step: int, loss: float) -> dict:
        return {"switch_requested": self.coordinator.record_validation(step, loss)}

    def try_merge(self) -> bool:
        pending = len(self.coordinator.pending)
        if pending < self.min_quorum:
            self.quorum_since = None
            return False
        if self.quorum_since is None:
            self.quorum_since = time.monotonic()
        if pending < self.workers and time.monotonic() - self.quorum_since < self.grace_seconds:
            return False
        self.quorum_since = None
        if not self.merge_or_skip():
            return False
        if self.version() % self.persist_every == 0:
            self.persist()
        return True

    def snapshot(self) -> tuple[int, Path]:
        """A full published bank, rewritten only when the version moved."""
        path = self.root / "published.pt"
        if self.published_version != self.version() or not path.exists():
            _save(
                path,
                {"version": self.version(), "bank": self.bank.state_snapshot()},
            )
            self.published_version = self.version()
        return self.version(), path

    def changes(self, after: int) -> tuple[int, Path] | None:
        """A change set file from ``after``, None when current; raises ChangesExpired."""
        if after == self.version():
            return None
        changes = self.coordinator.changes_since(after)
        if changes is None:
            raise ChangesExpired("Transducer change history expired: fetch a snapshot")
        exports = self.root / "exports"
        exports.mkdir(exist_ok=True)
        fd, name = tempfile.mkstemp(dir=exports, suffix=".pt")
        os.close(fd)
        _save(Path(name), changes.state_dict())
        return changes.version, Path(name)


@dataclass
class TransducerSync:
    """A learner's cursor: installed canonical version and its round uploads."""

    version: int = 0
    sequence: int = 0
    steps_since: int = 0
    sequences_at_round: int = 0
    pending: dict | None = None
    download: Future | None = None
    force_snapshot: bool = False

    def state_dict(self) -> dict:
        return {
            "version": self.version,
            "sequence": self.sequence,
            "steps_since": self.steps_since,
            "sequences_at_round": self.sequences_at_round,
        }

    def load_state_dict(self, state: dict) -> None:
        self.version = int(state["version"])
        self.sequence = int(state["sequence"])
        self.steps_since = int(state["steps_since"])
        self.sequences_at_round = int(state["sequences_at_round"])
        # An interrupted upload is dropped; the next install re-bases the learner.
        self.pending = None
        self.force_snapshot = True


def install_published(trainer: Any, sync: TransducerSync, kind: str, path: Path) -> None:
    """Apply a downloaded change set or snapshot between optimizer steps."""
    try:
        data = _load(path)
    finally:
        path.unlink(missing_ok=True)
    if kind == "changes":
        changes = ChangeSet(**data)
        if changes.base_version != sync.version:
            raise ValueError("Change set does not start from the installed version")
        install_changes(trainer.bank, trainer.optimizer, trainer.growth, changes)
        sync.version = changes.version
        return
    bank: TransducerBank = trainer.bank
    optimizer: BlockScaleDescent = trainer.optimizer
    growth: TransducerGrowth = trainer.growth
    before = set(bank.edge_ids().tolist())
    # Write-outs not yet submitted survive the reinstall.
    unsent = [edge for edge in growth.provisional_ids if edge in bank]
    kept = bank.export_rows(unsent) if unsent else None
    if data["bank"]["learned_values"] and not bank.learned_values:
        optimizer.switch_to_learned()
        growth.switch_to_learned()
    bank.load_snapshot(data["bank"])
    if kept is not None:
        bank.import_rows(kept, retired=[], next_id=bank.table.next_id)
    gone = sorted(before - set(bank.edge_ids().tolist()))
    optimizer.forget(gone)
    optimizer.reset_origins()
    growth.evidence.drop(gone)
    growth.round_evidence.drop(gone)
    growth.submitted_ids = []
    sync.version = int(data["version"])


def submit_round(trainer: Any, sync: TransducerSync, *, worker: int, step: int) -> LearnerRound:
    """Take this learner's round at a 32-step boundary."""
    examples = max(1, trainer.sequences - sync.sequences_at_round)
    sync.sequences_at_round = trainer.sequences
    sync.sequence += 1
    sync.steps_since = 0
    return learner_round(
        trainer.optimizer,
        trainer.growth,
        worker=worker,
        sequence=sync.sequence,
        base_version=sync.version,
        examples=examples,
        step=step,
    )


class RankRounds:
    """Synchronous data-parallel ranks sharing one bank through in-process rounds.

    Each rank trains its copy as a provisional learner. Every 32 steps (and
    before a checkpoint) all ranks' rounds are gathered on rank 0, merged there
    by the canonical coordinator exactly as decoupled learners' rounds are, and
    the resulting change set is broadcast back and installed by every rank.
    Checkpoints hold the canonical bank; a resumed rank restarts its learner
    copy from it.
    """

    def __init__(self, trainer: Any, context: Any) -> None:
        self.trainer = trainer
        self.context = context
        self.sync = TransducerSync()
        self.exchanged = -1  # the step of the last exchange
        self.canonical: CanonicalBank | None = None
        if context.is_primary:
            self.canonical = CanonicalBank(
                trainer.bank.geometry, evidence_from=trainer.schedule.warmup_steps
            )
            self.canonical.bank.load_snapshot(trainer.bank.state_snapshot())

    def step(self, step: int) -> dict[str, int]:
        """Exchange one round at every 32-step boundary (a collective on every rank)."""
        return {} if step % LOCAL_STEPS else self._exchange(step)

    def flush(self, step: int) -> None:
        """Merge the partial round before a checkpoint (a collective on every rank)."""
        if step % LOCAL_STEPS and step != self.exchanged:
            self._exchange(step)

    def _exchange(self, step: int) -> dict[str, int]:
        self.exchanged = step
        submission = submit_round(self.trainer, self.sync, worker=self.context.rank, step=step)
        rounds = self.context.all_gather_object(_pack(submission.state_dict()))
        changes = None
        if self.canonical is not None:
            for packed in rounds:
                self.canonical.coordinator.submit(LearnerRound.from_state(_unpack(packed)))
            self.canonical.merge_or_skip()
            changes = _pack(
                self.canonical.coordinator.changes_since(self.sync.version).state_dict()
            )
        changes = ChangeSet(**_unpack(self.context.broadcast_object(changes)))
        trainer = self.trainer
        install_changes(trainer.bank, trainer.optimizer, trainer.growth, changes)
        self.sync.version = changes.version
        return {"transducer_version": changes.version}

    def record_validation(self, step: int, loss: float) -> None:
        """Rank 0's bank-only loss feeds the canonical switch rule."""
        if self.canonical is not None:
            self.canonical.coordinator.record_validation(step, loss)

    def state_dict(self) -> dict:
        """The canonical state (rank 0 only) plus the shared round cursor."""
        if self.canonical is None:
            raise RuntimeError("Only rank 0 holds the canonical bank")
        return {
            **self.canonical.checkpoint_state(),
            "sequences": self.trainer.sequences,
            "rounds": self.sync.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Exact resume of a multi-rank run, or a new run seeded from any checkpoint."""
        self.trainer.bank.load_snapshot(state["bank"])
        self.trainer.sequences = int(state.get("sequences", 0))
        resumed = "rounds" in state
        if resumed:
            self.sync.load_state_dict(state["rounds"])
        if self.canonical is None:
            return
        if resumed:
            self.canonical.load_state_dict({"bank": state["bank"], **state["canonical"]})
        else:
            self.canonical.seed(state)
