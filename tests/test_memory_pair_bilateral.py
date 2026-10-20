"""Independent causal support is necessary for a learned pair to enter native KV."""
import numpy as np
import torch

from megatoken.memory.bilateral import FrozenMemoryFields, selected_joint_groups
from megatoken.memory.contracts import Collection, Footprint, SemanticSpace, SourceRange, Visibility
from megatoken.memory.sqlite import SQLiteKB


def _source(name):
    return SourceRange(name, "v1", 0, 16, f"hash-{name}", "team")


def _setup(bank, left_source, right_source):
    key = np.array([1.] + [0.] * 7, dtype=np.float32)
    a = bank.insert_memory(
        scope="team", footprint=Footprint((_source(left_source),)),
        key=key, value=np.ones(16), assignment="left", birth_step=0,
        group="pair-compatible",
    )
    b = bank.insert_memory(
        scope="team", footprint=Footprint((_source(right_source),)),
        key=key, value=np.ones(16) * 0.5, assignment="right", birth_step=0,
        group="pair-compatible",
    )
    bank.insert_relation(
        scope="team", left=a, right=b,
        prior=0.9, value=np.ones(16) * 2,
        dependence_logits=(1., -1.),
        footprint=Footprint((_source(left_source), _source(right_source)))
        if left_source != right_source else Footprint((_source(left_source),)),
    )
    return bank.scan(Collection.MEMORY, "team")


def _run(bank, sources, units):
    result, _ = selected_joint_groups(
        bank=bank, selected=FrozenMemoryFields(bank, torch.device("cpu")),
        query=torch.tensor([[1.] + [0.] * 7]),
        records=sources, visibility=Visibility("team", 10),
        anchor=10, max_groups=2,
        pair_support_units=units, pair_context_complete=True,
    )
    return [c.identity for group in result for c in group.constituents]


def test_single_repeated_support_unit_never_activates_relation(tmp_path):
    space = SemanticSpace("space", "v1", "teacher", "fixed", 8, "doc", "query", "context")
    with SQLiteKB(tmp_path / "single.sqlite", space, value_dim=16) as bank:
        sources = _setup(bank, "same", "same")
        identities = _run(bank, sources, ("same",))
        assert len(identities) == 2
        assert all(identity.startswith("memory:") for identity in identities)


def test_two_independent_checked_source_units_activate_pair_value(tmp_path):
    space = SemanticSpace("space", "v1", "teacher", "fixed", 8, "doc", "query", "context")
    with SQLiteKB(tmp_path / "distinct.sqlite", space, value_dim=16) as bank:
        sources = _setup(bank, "source-a", "source-b")
        identities = _run(bank, sources, ("source-a", "source-b"))
        assert len(identities) == 3
        assert any(identity.startswith("pair:") for identity in identities)
        absent = _run(bank, sources, ("source-a",))
        assert not any(identity.startswith("pair:") for identity in absent)
