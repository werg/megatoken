"""Transport of v10 transducer rounds between DiLoCo learners and the coordinator.

The coordinator process hosts the canonical bank (``TransducerHost``); each
learner keeps a ``TransducerSync`` cursor. Learners upload one
``LearnerRound`` per 32 local steps and install published change sets between
optimizer steps; they never block on the network. Files are ``torch.save``
archives of tensors and plain values only, loaded with ``weights_only=True``.
"""

from __future__ import annotations

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

from .transducer_diloco import (
    COORDINATOR_WORKER,
    ChangeSet,
    LearnerRound,
    TransducerCoordinator,
    install_changes,
    learner_round,
)


class ChangesExpired(LookupError):
    """The learner's base is older than the kept change history: fetch a snapshot."""


def _save(path: Path, value: dict) -> None:
    with atomic_binary_writer(path) as handle:
        torch.save(value, handle)


def _load(path: Path) -> dict:
    # Read through a handle: torch dispatches on the file suffix of a path.
    with Path(path).open("rb") as handle:
        return torch.load(handle, map_location="cpu", weights_only=True)


class TransducerHost:
    """The coordinator's canonical bank, its pending rounds and its publications."""

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
        self.root = Path(root) / "transducer"
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.min_quorum = min_quorum
        self.grace_seconds = grace_seconds
        self.persist_every = persist_every
        self.bank = TransducerBank(TransducerGeometry(**settings["geometry"]))
        self.optimizer = BlockScaleDescent(self.bank, worker=COORDINATOR_WORKER)
        self.growth = TransducerGrowth(self.bank, optimizer=self.optimizer)
        self.coordinator = TransducerCoordinator(
            self.bank, self.optimizer, self.growth, evidence_from=int(settings["evidence_from"])
        )
        self.quorum_since: float | None = None
        self.published_version = -1
        state = self.root / "state.pt"
        if state.exists():
            self._restore(_load(state))

    def version(self) -> int:
        return self.coordinator.version

    def _restore(self, state: dict) -> None:
        self.bank.load_snapshot(state["bank"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.growth.load_state_dict(state["growth"])
        self.coordinator.load_state_dict(state["coordinator"])

    def persist(self) -> None:
        _save(
            self.root / "state.pt",
            {
                "bank": self.bank.state_snapshot(),
                "optimizer": self.optimizer.state_dict(),
                "growth": self.growth.state_dict(),
                "coordinator": self.coordinator.state_dict(),
            },
        )

    def submit(self, worker: int, sequence: int, base_version: int, incoming: Path, conflict):
        submission = LearnerRound.from_state(_load(incoming))
        if (submission.worker, submission.sequence, submission.base_version) != (
            worker,
            sequence,
            base_version,
        ):
            raise ValueError("Transducer round headers disagree with its contents")
        if not 0 <= worker < self.workers:
            raise ValueError("Unknown learner")
        if base_version > self.version():
            raise conflict("Stale transducer base: newer than the canonical bank")
        if not self.coordinator.submit(submission):
            raise conflict("Stale or duplicate transducer round")
        return {"transducer_version": self.version(), "pending": len(self.coordinator.pending)}

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
        try:
            self.coordinator.merge()
        except (ValueError, FloatingPointError, RuntimeError) as error:
            # The failing rounds were consumed; learners re-base on the next version.
            print(f"transducer merge failed and was skipped: {error}", flush=True)
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
    bank.load_snapshot(data["bank"])
    if kept is not None:
        bank.import_rows(kept, retired=[], next_id=bank.table.next_id)
    gone = sorted(before - set(bank.edge_ids().tolist()))
    optimizer.forget(gone)
    optimizer.round_origin = {}
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
