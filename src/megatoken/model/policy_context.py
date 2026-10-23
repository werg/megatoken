"""Bounded, non-pickle causal inputs for exact outcome-policy reconstruction.

Only committed contextual transitions travel with a selected action. Public
retrieval results and their adapter mixture are frozen actor-state inputs;
current contextual weights reconstruct the attention history at optimization.
No renderer output, reward, future observation or cached attention is included.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import numpy as np
import torch

from megatoken.model.adapters import AdapterMixture

MAX_CONTEXT_BYTES = 64 << 20


def _pack(tensor):
    if tensor is None:
        return None
    array = tensor.detach().float().cpu().contiguous().numpy().astype("<f4")
    return {"shape": list(array.shape), "f32": base64.b64encode(array.tobytes()).decode()}


def _unpack(value, device):
    if value is None:
        return None
    shape = value["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or len(shape) > 4
        or any(type(n) is not int or n < 1 for n in shape)
        or np.prod(shape, dtype=np.float64) > 16_000_000
    ):
        raise ValueError("Invalid recorded policy-context geometry")
    raw = base64.b64decode(value["f32"], validate=True)
    array = np.frombuffer(raw, dtype="<f4")
    if array.size != int(np.prod(shape)) or not np.isfinite(array).all():
        raise ValueError("Invalid recorded policy-context coordinates")
    return torch.from_numpy(array.copy()).reshape(shape).to(device)


def capture(state, raw, tracks, event_ids, roles, modalities, semantics):
    mixture = state.adapters
    return json.dumps(
        {
            "raw": _pack(raw),
            "tracks": tracks,
            "events": event_ids,
            "roles": roles[0].tolist(),
            "modalities": modalities,
            "descriptions": state.track_descriptions,
            "context_policy": state.context_policy.to_json(),
            "semantics": _pack(semantics),
            "adapter_logits": _pack(None if mixture is None else mixture.logits),
            "adapter_strength": _pack(None if mixture is None else mixture.strength),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def reconstruct(multi, entries, *, device):
    if sum(map(len, entries)) > MAX_CONTEXT_BYTES:
        raise ValueError("Policy context exceeds the recorded-history memory budget")
    state = replace(multi.tracks.initial_state(1, device=device), policy_history=None)
    for entry in entries:
        value = json.loads(entry)
        logits = _unpack(value["adapter_logits"], device)
        strength = _unpack(value["adapter_strength"], device)
        if (logits is None) != (strength is None):
            raise ValueError("Recorded adapter context is incomplete")
        state = replace(
            state,
            adapters=None if logits is None else AdapterMixture(logits, strength),
            track_descriptions=tuple(tuple(d) for d in value["descriptions"]),
        )
        from megatoken.model.context import ContextPolicy

        state = multi.tracks.set_context_policy(
            state, ContextPolicy.from_json(value["context_policy"])
        )
        state = multi.tracks.advance_group(
            state,
            _unpack(value["raw"], device),
            tracks=tuple(value["tracks"]),
            event_ids=tuple(value["events"]),
            modalities=tuple(value["modalities"]),
            role_ids=torch.tensor([value["roles"]], device=device, dtype=torch.long),
            track_semantics=_unpack(value["semantics"], device),
        )
    return state
