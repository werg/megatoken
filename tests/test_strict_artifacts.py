"""No migrations, inferred saved fields, version modes, or obsolete module shims."""

import json
import sqlite3

import pytest

from megatoken.data.dedup import DedupStore
from megatoken.data.schema import Episode, Lineage, PhysicalRecord
from megatoken.generation import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import GroupActionTape
from megatoken.training.history import NativeHistory


def test_saved_geometry_is_complete_and_only_current_architecture_is_accepted():
    current = ModelConfig.tiny().to_dict()
    for key in ("context_local_events", "context_cross_events", "prior_noise_std"):
        incomplete = dict(current)
        incomplete.pop(key)
        with pytest.raises(ValueError, match="required fields"):
            ModelConfig.from_dict(incomplete)
    with pytest.raises(ValueError, match="architecture_version"):
        ModelConfig.from_dict({**current, "architecture_version": 13})


def test_episode_cannot_be_upgraded_by_a_reader():
    sample = Episode(
        "strict", (PhysicalRecord("q", "user", b"question"),), Lineage("fixture", "v1", "strict")
    )
    previous = {**sample.to_dict(), "schema_version": 5}
    with pytest.raises(ValueError):
        Episode.from_dict(previous)


def test_existing_dedup_database_requires_the_current_contract(tmp_path):
    path = tmp_path / "dedup.sqlite"
    with DedupStore(path):
        pass
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM meta WHERE key='format_version'")
    with pytest.raises(ValueError, match="Unsupported dedup database"):
        DedupStore(path)


def test_shared_experts_have_native_owners_instead_of_obsolete_dense_facades():
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    assert model.decoder.header_start.bank is model.decoder.header_start._bank
    assert model.decoder.header_start.selector.codes.numel() > 0
    for module, name in (
        (model.decoder.header_start, "weight"),
        (model.decoder.header_start, "router"),
        (model.core.u_always, "gate_up"),
        (model.core.u_router, "context_projection"),
    ):
        assert not hasattr(module, name)


def test_recorded_native_selections_require_explicit_member_bindings():
    from megatoken.generation import ActionSpec
    from megatoken.model.multitrack import MultiTrackGenerator

    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    multi = MultiTrackGenerator(model)
    selected = multi.generate_group(
        multi.tracks.initial_state(1),
        tracks=("assistant",),
        specs=(ActionSpec(max_tokens=0),),
        effort=1,
    )
    old = json.loads(selected.tape.to_json())
    old["version"] = "fmt-multitrack-facets-v16"
    with pytest.raises(ValueError, match="version"):
        GroupActionTape.from_json(json.dumps(old))
    from megatoken.data.schema import Supervision

    sample = Episode(
        "native",
        (PhysicalRecord("q", "user", b"question"),),
        Lineage("fixture", "v1", "native"),
        supervision=Supervision(
            annotations_json=json.dumps(
                {
                    "native_action_tapes": [
                        {"record_cursor": 1, "tape": json.loads(selected.tape.to_json())}
                    ]
                }
            )
        ),
    )
    with pytest.raises(ValueError, match="record_ids"):
        NativeHistory.from_episode(sample, model)


def test_no_internal_corpus_converter_is_installed():
    from importlib.util import find_spec

    assert find_spec("megatoken.data.legacy_linear") is None


def test_saved_memory_supervision_cannot_infer_missing_fields():
    from megatoken.training.transducer import TransducerTrainer

    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    trainer = TransducerTrainer(model.transducer)
    state = trainer.state_dict()
    for field in state["supervision"]:
        incomplete = {**state, "supervision": dict(state["supervision"])}
        incomplete["supervision"].pop(field)
        with pytest.raises(ValueError, match="complete current supervision"):
            trainer.load_state_dict(incomplete)
