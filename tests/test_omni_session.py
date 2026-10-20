"""Copy-on-write group speculation never publishes a subset or rewinds delivery."""
from types import SimpleNamespace

import pytest

from megatoken.omni.session import OmniSession


class FakeTracks:
    def _validate(self, state):
        if state.batch_size != 1:
            raise ValueError("bad session")
    def observe_group(self, state, events, *, encoder_grad):
        assert not encoder_grad
        return SimpleNamespace(batch_size=1, revision=state.revision + 1)


class FakeGenerator:
    def __init__(self):
        self.tracks = FakeTracks()
    def generate_group(self, state, *, tracks, specs, controls, effort, generator):
        tape = SimpleNamespace(
            meridian_id=state.revision,
            tracks=tuple(tracks),
            payloads=tuple(b"abcdef" for _ in tracks),
            controls=tuple(controls),
            effort=effort,
            to_json=lambda: "|".join(tuple(tracks) + tuple(c.hex() for c in controls)).encode().hex(),
        )
        return SimpleNamespace(
            state=SimpleNamespace(batch_size=1, revision=state.revision + 1),
            tape=tape,
        )
    def replay_group(self, state, tape, *, specs, effort):
        assert effort == tape.effort
        assert len(specs) == len(tape.tracks)
        return SimpleNamespace(
            state=SimpleNamespace(batch_size=1, revision=state.revision + 1),
        )


def _session():
    return OmniSession(
        FakeGenerator(), SimpleNamespace(batch_size=1, revision=0),
    )


def test_speculation_acceptance_is_atomic_and_replayable():
    session = _session()
    candidate = session.propose(
        tracks=("speech", "caption"), specs=(object(), object()),
        requests=(("Speak words", 1), ("Describe scene", 2)),
    )
    assert session.state.revision == 0
    assert session.history == []
    accepted = session.accept(candidate.candidate_id)
    assert session.state.revision == 1
    assert list(accepted.delivery_cursors) == ["speech", "caption"]
    assert len(session.history) == 1
    assert session.rebuild().revision == session.state.revision
    assert session.deliver(0, "speech", 4) == 4
    session.interrupt(0, "speech", "user spoke")
    assert accepted.interruptions == [("speech", "user spoke", 4)]
    with pytest.raises(ValueError, match="exceeds"):
        session.deliver(0, "speech", 3)
    assert accepted.delivery_cursors["speech"] == 4


def test_new_evidence_invalidates_unaccepted_work_but_not_played_output():
    session = _session()
    candidate = session.propose(
        tracks=("speech",), specs=(object(),),
        requests=(("Speak answer", 1),),
    )
    session.observe((object(),))
    assert session.state.revision == 1
    with pytest.raises(ValueError, match="stale"):
        session.accept(candidate.candidate_id)
    second = session.propose(
        tracks=("speech",), specs=(object(),),
        requests=(("Speak corrected answer", 2),),
    )
    session.reject(second.candidate_id)
    assert session.state.revision == 1
    assert session.published == {}
    assert session.rebuild().revision == 1
