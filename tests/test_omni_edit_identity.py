"""Edition ancestry and checkpoint residency regression tests."""

import hashlib

from megatoken.omni.trajectories import ArtifactVersion, EditTrajectory


def test_slotted_edit_replay_fingerprint_and_order():
    base = ArtifactVersion(
        "v0",
        "",
        hashlib.sha256(b"base").hexdigest(),
        "source-A",
    )
    first = ArtifactVersion(
        "v1",
        "v0",
        hashlib.sha256(b"edit-one").hexdigest(),
        "source-A",
        request="Change the voice, keep the words",
        preserve=("words",),
    )
    chain = EditTrajectory(base)
    chain.append(first)
    assert chain.replay_identity() == chain.replay_identity()
    assert chain.replay_identity() != EditTrajectory(base).replay_identity()
    assert chain.versions[-1].parent == "v0"
