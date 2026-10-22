"""Serving the v10 transducer: the same boundaries, metadata and effort as training."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.generation.structure import ActionSpec
from megatoken.loading import build_model, parameter_report
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.transducer import metadata_payload, real_boundary
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.training.meridians import _boundary_meridian, _metadata
from megatoken.training.transducer import plan_boundaries


def _runtime() -> MultiTrackRuntime:
    torch.manual_seed(3)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    return MultiTrackRuntime(MultiTrackGenerator(model))


def _event(role: str) -> ObservedEvent:
    return ObservedEvent("main", "text", b"x", role_id=ROLE_TO_ID[role])


def _step(roles: tuple[str, ...], *, part: tuple[int, int] = (0, 1)):
    records = [
        SimpleNamespace(record=SimpleNamespace(part_index=part[0], part_count=part[1]))
        for _ in roles
    ]
    return SimpleNamespace(
        tracks=("main",) * len(roles),
        roles=tuple(ROLE_TO_ID[role] for role in roles),
        modalities=("text",) * len(roles),
        records=tuple(records),
        generated=False,
    )


def test_serving_and_training_plan_the_same_lookups_from_roles():
    # "generate" is a sampled assistant meridian, the rest are observed events.
    # "name/i" is part i of a two-part message: only the last part ends it.
    script = [
        "user", "user/0", "user/1", "generate/0", "generate/1", "user", "tool", "document",
        "assistant/0", "assistant/1", "user", "environment", "user", "user", "user", "user", "user", "user",
        "user", "user", "user", "generate", "user",
    ]  # fmt: skip
    runtime = _runtime()
    session = runtime.open()
    served = [session.memory.agenda.lookups]  # the lookup before meridian 0
    for item in script[:-1]:
        name, _, part = item.partition("/")
        terminal = part != "0"
        before = session.memory.agenda.lookups
        if name == "generate":
            spec = ActionSpec(max_tokens=2, require_terminal=terminal)
            runtime.generate(session, tracks=("main",), specs=(spec,), effort=1)
        else:
            event = replace(_event(name), terminal=terminal)
            runtime.observe(session, (event,), effort=1)
        served.append(session.memory.agenda.lookups - before)
    meridians = []
    for item in script:
        name, _, part = item.partition("/")
        meridians.append(
            _boundary_meridian(
                _step(
                    ("assistant" if name == "generate" else name,),
                    part=(int(part or 0), 2 if part else 1),
                )
            )
        )
    planned = plan_boundaries(
        meridians, max_lookups=runtime.memory.bank.geometry.max_lookups, generator=None
    )
    assert served == planned and sum(planned) >= 6


def test_boundary_depends_on_roles_and_part_index_not_on_scoring():
    assert _boundary_meridian(_step(("assistant",))).real_boundary
    assert not _boundary_meridian(_step(("assistant",), part=(0, 2))).real_boundary
    assert not _boundary_meridian(_step(("assistant", "user"))).real_boundary
    # A finished user message is a boundary (the lookup after it has read the question)
    # and is the one that "asks"; a user message still arriving in parts is not.
    asked = _boundary_meridian(_step(("user",)))
    assert asked.real_boundary and asked.asked
    assert not _boundary_meridian(_step(("user",), part=(0, 2))).real_boundary
    assert not _boundary_meridian(_step(("assistant",))).asked
    assert all(
        _boundary_meridian(_step((role,), part=(0, 2))).real_boundary
        for role in ("tool", "document", "environment")
    )
    assert real_boundary([ROLE_TO_ID["user"]], [True])
    assert not real_boundary([ROLE_TO_ID["user"]], [False])
    assert not real_boundary([ROLE_TO_ID["system"]], [True])


def test_a_failed_lookup_leaves_the_session_untouched_and_retryable(monkeypatch):
    runtime = _runtime()
    session = runtime.open()
    native = runtime.memory.native
    boundary = native.boundary

    def broken(*args, **kwargs):
        raise RuntimeError("lookup failed")

    state, lookups = session.state, session.memory.agenda.lookups
    waited = session.memory.scheduler.ordinary_since
    draft = runtime.speculate(
        session, tracks=("main",), specs=(ActionSpec(max_tokens=2, require_terminal=True),)
    )
    monkeypatch.setattr(native, "boundary", broken)
    with pytest.raises(RuntimeError, match="lookup failed"):
        runtime.accept(session, draft)
    with pytest.raises(RuntimeError, match="lookup failed"):
        runtime.observe(session, (_event("tool"),))
    assert session.state is state and session.revision == 0 and session.events == []
    assert session.memory.agenda.lookups == lookups
    assert session.memory.scheduler.ordinary_since == waited
    monkeypatch.setattr(native, "boundary", boundary)
    runtime.accept(session, draft)
    assert session.revision == 1 and session.memory.agenda.lookups == lookups + 1


def test_lookups_use_the_requests_effort(monkeypatch):
    runtime = _runtime()
    seen = []
    boundary = runtime.memory.native.boundary

    def record(*args, effort, **kwargs):
        seen.append(effort)
        return boundary(*args, effort=effort, **kwargs)

    monkeypatch.setattr(runtime.memory.native, "boundary", record)
    session = runtime.open(effort=3)
    runtime.observe(session, (_event("tool"),), effort=2)
    runtime.generate(
        session,
        tracks=("main",),
        specs=(ActionSpec(max_tokens=2, require_terminal=True),),
        effort=1,
    )
    assert seen == [3, 2, 1]


def test_training_metadata_may_be_partial_or_empty_and_evaluation_is_complete():
    episode = SimpleNamespace(
        stream="chat",
        tracks=[SimpleNamespace(default_modality="text")],
        supervision=SimpleNamespace(annotations_json="{}"),
    )
    full = json.loads(_metadata(episode, None))
    assert (full["task"], full["modality"], full["version"]) == ("chat", ["text"], "v10")
    drawn = [_metadata(episode, seed) for seed in range(64)]
    assert metadata_payload({}) in drawn and len(set(drawn)) > 4
    for payload in drawn:
        fields = json.loads(payload)
        assert all(fields[k] in (None, full[k]) for k in full)


def test_the_hindsight_head_is_training_only():
    torch.manual_seed(0)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    served = FacetedGenerator(FacetedCore(model.config), model.codec, training_heads=False)
    assert served.transducer.hindsight_realizer is None
    weights = model.served_state_dict()
    assert not any("hindsight_realizer" in name for name in weights)
    served.load_state_dict(weights, strict=True)
    hindsight = sum(p.numel() for p in model.transducer.hindsight_realizer.parameters())
    totals = parameter_report(model)["totals"]
    assert hindsight and totals["training_only"] == hindsight
    assert totals["served"] == parameter_report(served)["totals"]["served"]
    assert parameter_report(served)["totals"]["training_only"] == 0


def test_a_window_scores_the_question_lookup_only_if_it_falls_before_its_own_meridians():
    from megatoken.training.meridians import _question_scored

    asked, answer, later = _step(("user",)), _step(("assistant",)), _step(("assistant",))
    assert _question_scored((), (asked, answer))
    # The lookup before the first scored meridian is trained, though the question is replayed.
    assert _question_scored((asked,), (answer,))
    # A window that starts after the lookup replays it without gradient; with no question, or
    # the question last, there is no lookup after it to score.
    assert not _question_scored((asked, answer), (later,))
    assert not _question_scored((), (answer, asked))
    assert not _question_scored((), (answer, later))
