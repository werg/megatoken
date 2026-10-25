"""Measurements of the real selected-serving process, with no optimizer."""

from __future__ import annotations

import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from megatoken.io import content_hash, sha256_file


def percentile(values, quantile):
    """Nearest-rank empirical percentile; no interpolation or population claim."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def state_counts(adapter):
    sessions = list(adapter.sessions.values())
    tracks = [
        track for session in sessions for track in session.runtime_session.tracks.tracks.values()
    ]
    return {
        "retainedSessions": len(sessions),
        "retainedInvocations": sum(len(session.work) for session in sessions),
        "activeInvocations": sum(
            not work.completed for session in sessions for work in session.work.values()
        ),
        "trackStatuses": dict(Counter(track.status for track in tracks)),
        "rendering": sum(len(session.runtime_session.rendering) for session in sessions),
        "selectedGroups": sum(len(session.runtime_session.selected_groups) for session in sessions),
        "reservedOutputBytes": sum(
            sum(session.runtime_session.reserved_output_bytes.values()) for session in sessions
        ),
    }


class ServingProfile:
    """An optional report alongside unchanged v4 RPC replies."""

    def __init__(self, served, device, *, origin="fixed", max_records=100000):
        if origin not in {"controlled", "fixed", "trained"}:
            raise ValueError("Profile origin must be controlled, fixed or trained")
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path, self.device, self.origin = (
            Path(served),
            torch.device(resolved_device if device == "auto" else device),
            origin,
        )
        self.max_records = max_records
        self.records = []
        self.dropped_records = 0
        self.started = time.perf_counter()
        self.cold_seconds = None
        self.identity = None
        self.initialization = None
        self.peak_active = 0
        self.before_close = None
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def loaded(self, bundle):
        self.synchronize()
        self.cold_seconds = time.perf_counter() - self.started
        self.identity = {
            "modelSha256": bundle.manifest["model_sha256"],
            "manifestSha256": sha256_file(self.path / "manifest.json"),
            "tokenizerIdentity": bundle.model.codec.fingerprint,
        }

    def synchronize(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def record(self, message, reply, seconds, adapter):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Profile duration must be finite and nonnegative")
        counts = state_counts(adapter)
        self.peak_active = max(self.peak_active, counts["activeInvocations"])
        op = message.get("op") if isinstance(message, dict) else None
        if op == "initialize" and reply["ok"]:
            self.initialization = reply["result"]
        result = reply.get("result") if reply["ok"] else (reply.get("error") or {}).get("evidence")
        selection = result.get("selection") if isinstance(result, dict) else None
        row = {
            "id": reply["id"],
            "operation": op,
            "ok": reply["ok"],
            "wallSeconds": seconds,
            "state": counts,
            "selectionAuthority": selection.get("origin") if isinstance(selection, dict) else None,
            # Original error evidence includes the generated IDs/bytes; never repair them.
            "failure": reply.get("error"),
        }
        if len(self.records) < self.max_records:
            self.records.append(row)
        else:
            self.dropped_records += 1

    def report(self, *, runtime_closed):
        groups = {}
        for operation in sorted({row["operation"] or "invalid-request" for row in self.records}):
            rows = [
                row for row in self.records if (row["operation"] or "invalid-request") == operation
            ]
            values = [row["wallSeconds"] for row in rows]
            groups[operation] = {
                "count": len(rows),
                "successfulRpcCalls": sum(row["ok"] for row in rows),
                "firstSeconds": values[0],
                "p50Seconds": percentile(values, 0.5),
                "p95Seconds": percentile(values, 0.95),
                "warmP50Seconds": percentile(values[1:], 0.5),
                "warmP95Seconds": percentile(values[1:], 0.95),
                "totalSeconds": sum(values),
            }
        memory = {
            "hostLifetimePeakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated(self.device)
            if self.cuda
            else None,
            "cudaPeakReservedBytes": torch.cuda.max_memory_reserved(self.device)
            if self.cuda
            else None,
            "scope": "This serving process; GPU peaks since before model load; host RSS is lifetime peak",
        }
        result = {
            "schema": "semantic-serving-profile/1",
            "declaredOrigin": self.origin,
            "originScope": "Caller declaration; model identity is verified, training provenance is not inferred",
            "device": str(self.device),
            "identity": self.identity,
            "initialization": self.initialization,
            "coldLoadSeconds": self.cold_seconds,
            "operations": groups,
            "records": self.records,
            "droppedRecords": self.dropped_records,
            "complete": self.dropped_records == 0,
            "memory": memory,
            "peakActiveInvocationsAtRpcBoundaries": self.peak_active,
            "stateBeforeRuntimeClose": self.before_close,
            "runtimeClosed": runtime_closed,
            "optimizerSteps": 0,
            "learnedSuccess": None,
            "trainingMemoryFit": "unmeasured",
            "scope": "Real served RPC wall time including CUDA synchronization; no queue/network time",
            "warmDefinition": "All calls after the first call of the same operation in this process",
        }
        return {**result, "digest": content_hash(result)}
