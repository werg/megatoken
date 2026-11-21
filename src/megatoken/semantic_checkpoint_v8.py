"""Safe identities for verified V8 model-session reconstruction.

Checkpoint JSON contains RNG bytes and public history, never a pickle or an
untrusted tensor cache. Restoration runs the loaded model again and compares
its original emitted tokens and complete committed continuation fingerprint.
"""

from __future__ import annotations

import base64
import hashlib
import random
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

from megatoken.data.semantic_json import stable_algorithm_hash


def capture_random_state():
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": [random.getstate()[0], list(random.getstate()[1]), random.getstate()[2]],
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode("ascii"),
        "cuda": [
            base64.b64encode(state.cpu().numpy().tobytes()).decode("ascii")
            for state in (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        ],
    }


def restore_random_state(state):
    import numpy as np
    import torch

    if type(state) is not dict or set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Invalid V8 checkpoint random state")
    python, numpy = state["python"], state["numpy"]
    if (
        type(python) is not list
        or len(python) != 3
        or type(python[1]) is not list
        or len(python[1]) != 625
        or any(type(value) is not int or not 0 <= value <= 2**32 for value in python[1])
        or type(numpy) is not list
        or len(numpy) != 5
        or numpy[0] != "MT19937"
        or type(numpy[1]) is not list
        or len(numpy[1]) != 624
        or any(type(value) is not int or not 0 <= value < 2**32 for value in numpy[1])
        or type(state["cuda"]) is not list
        or len(state["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    ):
        raise ValueError("V8 checkpoint RNG shape or device set differs")

    def tensor(encoded):
        if type(encoded) is not str or len(encoded) > 65536:
            raise ValueError("Invalid V8 checkpoint RNG bytes")
        raw = base64.b64decode(encoded, validate=True)
        if base64.b64encode(raw).decode("ascii") != encoded:
            raise ValueError("Noncanonical V8 checkpoint RNG bytes")
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()

    # Validate on independent generators before changing process-wide state.
    random.Random().setstate((python[0], tuple(python[1]), python[2]))
    numpy_value = (numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:])
    np.random.RandomState().set_state(numpy_value)
    cpu = tensor(state["torch"])
    torch.Generator(device="cpu").set_state(cpu)
    cuda = [tensor(value) for value in state["cuda"]]
    for index, value in enumerate(cuda):
        torch.Generator(device=f"cuda:{index}").set_state(value)
    random.setstate((python[0], tuple(python[1]), python[2]))
    np.random.set_state(numpy_value)
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def _state_value(value):
    """Fingerprint trusted in-memory state; never instantiate input-named types."""
    import torch

    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "tensor": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, torch.Generator):
        return {"generator": str(value.device), "state": _state_value(value.get_state())}
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (list, tuple)):
        return {"sequence": type(value).__name__, "items": [_state_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_state_value(item) for item in value]
        return {"set": sorted(items, key=stable_algorithm_hash)}
    if isinstance(value, dict):
        items = [[_state_value(key), _state_value(item)] for key, item in value.items()]
        return {"map": sorted(items, key=lambda row: stable_algorithm_hash(row[0]))}
    if is_dataclass(value):
        return {
            "type": type(value).__module__ + "." + type(value).__name__,
            "fields": {
                field.name: _state_value(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, SimpleNamespace):
        return _state_value(vars(value))
    raise ValueError("Unsupported V8 continuation state: " + type(value).__name__)


def runtime_identity(runtime):
    model = runtime.generator.generator
    values = {"modelConfig": model.config}
    for name in (
        "chunking",
        "max_event_output_bytes",
        "max_session_output_bytes",
        "context_archive_bytes",
        "context_archive_events",
        "max_retained_outputs",
        "output_specs",
    ):
        if hasattr(runtime, name):
            values[name] = getattr(runtime, name)
    return stable_algorithm_hash(_state_value(values))


def continuation_fingerprint(session):
    """Bind caches, retrieval state, budgets, source copies and routing together."""
    memory = session.memory
    if (
        memory.writes is not None
        or memory.sources is not None
        or session.rendering
        or session.pending
    ):
        raise ValueError("V8 checkpoint requires an idle writeless committed continuation")
    if getattr(memory, "closed", False):
        raise ValueError("A closed native continuation cannot checkpoint")
    agenda = memory.agenda
    if is_dataclass(agenda):
        # Mounted stores are immutable model resources, bound by model identity;
        # include their names/weights/versions without serializing entire banks.
        agenda = {field.name: getattr(agenda, field.name) for field in fields(agenda)}
        agenda["mounted"] = [
            {"name": mount.name, "weight": mount.weight, "version": mount.base.version}
            for mount in agenda["mounted"]
        ]
    omitted = {
        "memory",
        "lock",
        "tracks",
        "context_archive",
        "event_arrived_at_ns",
        "selection_at_ns",
    }
    state = {
        field.name: getattr(session, field.name)
        for field in fields(session)
        if field.name not in omitted
    }
    state.update(
        modelState=memory.state,
        agenda=agenda,
        scheduler=memory.scheduler,
        tracks=session.tracks.checkpoint(),
        contextArchive={
            "maxBytes": session.context_archive.max_bytes,
            "maxEvents": session.context_archive.max_events,
            "payloadBytes": session.context_archive.payload_bytes,
            "events": list(session.context_archive.events.items()),
        },
    )
    return stable_algorithm_hash(_state_value(state))
