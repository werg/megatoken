"""Production defaults must exercise the memory work, not just permit a toy opt-in."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from media_support import prepared_record
from test_teacher_graph import FrozenWriter, episode
from training_fixture import prepare_dataset, production_config, tiny_train_config

from megatoken.data import Episode, Lineage, PhysicalRecord, PreparedCorpus
from megatoken.data.hindsight import HindsightTargets
from megatoken.data.hindsight_corpus import prepare_corpus
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, TensorAssetStore
from megatoken.data.recipes import default_training_features
from megatoken.data.schema import sequential_meridians
from megatoken.generation.codec import ByteCodec
from megatoken.io import read_mapping, sha256_file
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.training.config import TrainConfig
from megatoken.training.engine import prepare_batches, window_config
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.readiness import check
from megatoken.training.stream import EpisodeWindow, _WindowPlan
from megatoken.training.transducer import TransducerTrainer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "name",
    [
        "coherence",
        "pretraining",
        "media-bridge",
        "repository-training",
        "verified-training",
        "long-context",
        "supervised-tuning",
        "student-corrections",
    ],
)
def test_every_production_template_requires_the_joint_memory_recipe(name):
    root = Path(__file__).resolve().parents[1]
    mapping = read_mapping(root / f"configs/train/{name}.example.yaml")
    config = TrainConfig.from_dict(mapping, expand_data=False)
    from megatoken.data.recipes import stage_recipe

    for key, value in stage_recipe(config.phase).training.items():
        assert getattr(config, key) == value
        assert key not in mapping, "Launch bindings must not duplicate curriculum decisions"
    assert config.budget is None, "Production examples must not retain rehearsal budgets"
    assert config.max_steps is None
    assert config.hindsight_targets == "/work/data/hindsight"
    for overrides in (
        {"hindsight_targets": None},
        {"navigation_weight": 0},
        {"hindsight_distill_weight": 0},
    ):
        with pytest.raises(ValueError, match="default teacher/structure recipe"):
            replace(config, **overrides)


def test_runtime_defaults_are_production_even_for_tiny_models():
    for model in ("tiny", "32x512"):
        with pytest.raises(ValueError, match="requires hindsight_targets"):
            TrainConfig(model=model)
        config = production_config("P1", model=model)
        assert config.purpose == "production" and config.use_stage_mixture
        for name, value in default_training_features().items():
            assert getattr(config, name) == value
        for overrides, message in (
            ({"eval_every": 0}, "held-out evaluation"),
            ({"use_stage_mixture": False}, "canonical stage recipe"),
            ({"allow_unverified_data": True}, "verification"),
        ):
            with pytest.raises(ValueError, match=message):
                replace(config, **overrides)


def test_old_recipe_selector_is_deleted_and_diagnostics_are_bounded():
    with pytest.raises(ValueError, match="Unknown training settings"):
        TrainConfig.from_dict({"memory_training": "learned"})
    with pytest.raises(ValueError, match="explicit max_steps or budget"):
        TrainConfig(purpose="diagnostic")
    config = TrainConfig(purpose="diagnostic", max_steps=1)
    assert config.eval_every == config.navigation_weight == 0


def test_default_cloud_and_preparation_jobs_start_the_same_initial_stage():
    import json

    from megatoken.data.recipes import stage_recipe

    root = Path(__file__).resolve().parents[1]
    launch = json.loads((root / "configs/cloud/train-default.example.json").read_text())
    preparation = json.loads(
        (root / "configs/cloud/prepare-teacher-structure.example.json").read_text()
    )
    config = TrainConfig.load(root / launch["config"], expand_data=False)
    assert stage_recipe(config.phase).previous_phase is None
    command = preparation["command"]
    assert command[command.index("--train-config") + 1] == launch["config"]
    assert "--resume" not in launch["coordinator"]["args"]
    assert config.decoupled.workers == launch["workers"]


@pytest.mark.parametrize("phase", ["B0", "MB", "P1", "M1", "M2", "LC", "S", "D"])
def test_production_cannot_select_another_training_method(phase):
    from megatoken.data.recipes import stage_recipe

    config = production_config(phase)
    recipe = stage_recipe(phase)
    assert config.schedule.peak_lr == recipe.learning_rate["peak_lr"]
    for overrides in (
        {"inner_optimizer": "adamw"},
        {"use_effort_mixture": False},
        {"representation_mode": "auto"},
        {"scored_decisions": config.scored_decisions + 1},
        {"navigation_weight": 0.2},
        {"hindsight_distill_weight": 0.2},
        {"loss_weights": {**config.loss_weights, "lexical": 0}},
        {"checkpoint_every": 0},
        {"decoupled": None},
    ):
        with pytest.raises(ValueError, match="Production"):
            replace(config, **overrides)
    if recipe.required_donor_profiles:
        with pytest.raises(ValueError, match="both selected donor imports"):
            replace(config, engram_imports=None)


def test_production_checks_pinned_donor_identity_and_order(tmp_path):
    import json

    from megatoken.data.recipes import stage_recipe, validate_training_donors
    from megatoken.donors import DONOR_PROFILES

    roots = []
    for name in stage_recipe("P1").required_donor_profiles:
        root = tmp_path / name
        root.mkdir()
        (root / "descriptors.npz").write_bytes(b"descriptor-checksum-fixture")
        profile = DONOR_PROFILES[name]
        metadata = {
            "donor_profile": name,
            "algorithm": "native-context-to-byte-hash-v1",
            "descriptor_sha256": sha256_file(root / "descriptors.npz"),
            "donor_metadata": {
                key: getattr(profile, key)
                for key in (
                    "repository",
                    "revision",
                    "tokenizer_sha256",
                    "config_sha256",
                    "index_sha256",
                )
            },
        }
        (root / "engram.json").write_text(json.dumps(metadata))
        roots.append(str(root))
    config = production_config("P1", engram_imports=roots)
    validate_training_donors(config)
    with pytest.raises(ValueError, match="curriculum order"):
        validate_training_donors(replace(config, engram_imports=roots[::-1]))
    (Path(roots[0]) / "descriptors.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_training_donors(config)


def test_production_requires_stage_lineage_and_media_alignment_targets():
    from megatoken.data.recipes import validate_stage_start, validate_teacher_encoding_source

    validate_stage_start(production_config("B0"), None)
    validate_stage_start(production_config("P1"), "MB")
    validate_stage_start(production_config("P1"), "P1")
    for previous in (None, "B0", "S"):
        with pytest.raises(ValueError, match="requires a checkpoint"):
            validate_stage_start(production_config("P1"), previous)
    with pytest.raises(ValueError, match="prepared text anchors"):
        replace(production_config("MB"), text_anchors=None)
    validate_teacher_encoding_source(production_config("B0"), None)
    with pytest.raises(ValueError, match="preceding teacher encoding"):
        validate_teacher_encoding_source(production_config("P1"), None)
    validate_teacher_encoding_source(production_config("P1"), "previous-hindsight")


def test_pretraining_batch_samples_the_declared_effort_distribution(tmp_path, monkeypatch):
    config = tiny_train_config(tmp_path)
    config = replace(config, phase="P1", use_effort_mixture=True)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())
    sampled = []

    def choose_lowest(probabilities, count):
        sampled.append(probabilities)
        return torch.tensor([0])

    monkeypatch.setattr(torch, "multinomial", choose_lowest)
    with PreparedCorpus(config.datasets) as corpus:
        episode, assets = corpus.fetch(0)
        window = _WindowPlan(episode, model.codec, window_config(config))[0]
        (batch,) = prepare_batches(model, [EpisodeWindow(episode, window, assets)], config)
    assert batch.effort == 1
    torch.testing.assert_close(sampled[0], torch.tensor([0.01, 0.04, 0.15, 0.8]))


def test_cli_rejects_incomplete_production_before_allocating_a_model(tmp_path, monkeypatch):
    from megatoken.cli import main

    path = tmp_path / "missing-teacher.json"
    path.write_text('{"model": "tiny"}')
    monkeypatch.setattr(
        "megatoken.training.engine.build_model",
        lambda *a, **kw: pytest.fail("Allocated an incomplete production run"),
    )
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(path)])
    assert error.value.code == 2
    assert main(["train-check", "--config", str(path)]) == 1


def production_fixture(tmp_path):
    source = tmp_path / "source"
    store = TensorAssetStore(source / "data")
    media = prepared_record(store, "audio", seed=31, target=False)
    ep = sequential_meridians(
        Episode(
            "audio",
            (
                PhysicalRecord("clip", "user", media.to_bytes(), media_type=MEDIA_RECORD_TYPE),
                PhysicalRecord("caption", "assistant", b"A clip.", policy_mask=True, turn_index=1),
            ),
            Lineage("audio", "v1", "audio"),
        )
    )
    data = prepare_dataset(
        source, [replace(episode(), stream="bootstrap"), replace(ep, stream="bootstrap")]
    )
    config = tiny_train_config(
        tmp_path / "fixture",
        datasets=[str(data)],
        max_window_records=8,
        max_record_bytes=16384,
        use_stage_mixture=True,
        hindsight_targets=str(tmp_path / "targets"),
        navigation_weight=0.05,
    )
    before = sha256_file(data / "manifest.json")
    manifest = prepare_corpus(config, tmp_path / "targets", FrozenWriter(), fit_windows=1)
    assert sha256_file(data / "manifest.json") == before
    return config, manifest


def test_production_preparation_preserves_media_roots_mixture_and_output(tmp_path):
    config, manifest = production_fixture(tmp_path)
    generated = TrainConfig.load(tmp_path / "targets/train.json")
    assert generated.datasets == config.datasets
    assert generated.output == config.output and generated.use_stage_mixture
    assert generated.packs == config.packs
    assert manifest["coverage"]["text_windows"] == manifest["coverage"]["unsupported_windows"] == 1
    assert manifest["coverage"]["target_values"] > 0
    assert manifest["coverage"]["navigation_examples"] > 0
    report = check(generated)
    assert report["ready"], report
    # A changed corpus cannot borrow targets from an unrelated production run.
    targets = HindsightTargets(config.hindsight_targets, config=config)
    with pytest.raises(ValueError, match="different training corpus"):
        targets.validate_corpus("different")
    targets.close()
    writer = FrozenWriter()
    writer.encode = lambda *a, **kw: pytest.fail(
        "Finished preparation must resume without encoding"
    )
    assert prepare_corpus(config, config.hindsight_targets, writer, fit_windows=1) == manifest


def test_mixed_recipe_uses_teacher_for_text_and_normal_retrieval_for_media(tmp_path, monkeypatch):
    config, _ = production_fixture(tmp_path)
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).train()
    targets = HindsightTargets(config.hindsight_targets, config=config)
    trainer = TransducerTrainer(model.transducer, teacher_targets=targets, navigation_weight=0.05)
    reports = {}
    with PreparedCorpus(config.datasets) as corpus:
        for i in range(len(corpus)):
            ep, assets = corpus.fetch(i)
            window = _WindowPlan(ep, model.codec, window_config(config))[0]
            (batch,) = prepare_batches(model, [EpisodeWindow(ep, window, assets)], config)
            ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
            reports[ep.episode_id] = forward_meridian_batch(
                model, batch, ledger, config, torch.device("cpu"), 0, transducer=trainer
            )
            ledger.finish()
            if ep.episode_id == "audio":
                assert all(
                    r.hindsight is None and r.alpha == 0 and not r.writes for r in trainer.records
                )
            trainer.after_backward(0)
            model.zero_grad(set_to_none=True)
    assert reports["joint"]["teacher_target_values"] > 0
    assert reports["joint"]["navigation_examples"] > 0
    assert reports["audio"]["teacher_unsupported_windows"] == 1
    assert "teacher_target_values" not in reports["audio"]
    targets.close()


def test_missing_production_index_is_rejected_before_model_allocation(tmp_path, monkeypatch):
    config, _ = production_fixture(tmp_path)
    (tmp_path / "targets/targets.sqlite").unlink()
    from megatoken.training.engine import train

    monkeypatch.setattr(
        "megatoken.training.engine.build_model",
        lambda *a, **kw: pytest.fail("Model allocation preceded artifact preflight"),
    )
    with pytest.raises(OSError):
        train(config)


def test_default_recipe_train_and_resume_exercise_both_objectives(tmp_path):
    import json

    from megatoken.training.engine import train

    config, _ = production_fixture(tmp_path)
    first = train(config)
    resumed = train(config, resume=first["checkpoint"])
    assert first["global_step"] == 1 and resumed["global_step"] == 2
    events = [
        json.loads(line)
        for line in (Path(config.output) / "metrics.jsonl").read_text().splitlines()
    ]
    updates = [event["generator"] for event in events if event["event"] == "train"]
    assert all(update["teacher_target_values"] > 0 for update in updates)
    assert all(update["navigation_examples"] > 0 for update in updates)
    assert all(update["teacher_unsupported_windows"] > 0 for update in updates)
    state = torch.load(Path(resumed["checkpoint"]) / "transducer.pt", weights_only=True)
    assert state["supervision"]["memory_policy"] == "teacher-text-only-v1"


def test_cached_coordinator_initializes_with_teacher_metadata_only(tmp_path):
    import json
    from argparse import Namespace
    from dataclasses import asdict

    from megatoken.cli import _diloco_initialize
    from megatoken.io import content_hash
    from megatoken.training.config import DecoupledConfig

    config, target_manifest = production_fixture(tmp_path)
    metadata_only = tmp_path / "coordinator-teacher"
    metadata_only.mkdir()
    (metadata_only / "manifest.json").write_bytes(
        (Path(config.hindsight_targets) / "manifest.json").read_bytes()
    )
    with PreparedCorpus(config.datasets) as corpus:
        roots = [
            dict(
                name=part.root.name,
                fingerprint=part.manifest["fingerprint"],
                extractors=sorted(part.feature_extractors()),
            )
            for part in corpus.parts
        ]
        metadata = dict(
            roots=roots,
            fingerprint=content_hash([r["fingerprint"] for r in roots]),
            codec_fingerprint=corpus.codec_fingerprint,
            chunking=corpus.chunking,
        )
    metadata_file = tmp_path / "corpus.json"
    metadata_file.write_text(json.dumps(metadata))
    coordinator_config = replace(
        config,
        datasets=["/work/data/prepared/*"],
        hindsight_targets=str(metadata_only),
        decoupled=DecoupledConfig(
            coordinator_url="https://coordinator", workers=1, run_id="default-memory"
        ),
    )
    config_path = tmp_path / "coordinator.json"
    config_path.write_text(json.dumps(asdict(coordinator_config)))
    root = tmp_path / "coordinator"
    _diloco_initialize(
        Namespace(
            config=str(config_path),
            root=str(root),
            resume=None,
            corpus_metadata=str(metadata_file),
            quorum=1,
            fragment_mib=1,
            outer_lr=0.7,
            outer_momentum=0.9,
            grace_seconds=0.0,
            max_staleness=2,
        )
    )
    saved = json.loads((root / "manifest.json").read_text())
    assert saved["dataset_fingerprint"] == target_manifest["corpus_identity"]
    assert saved["transducer"]["supervision"]["teacher"] == target_manifest["encoding_identity"]
    assert not (metadata_only / "targets.sqlite").exists()


def test_frozen_training_scope_keeps_search_active_but_prevents_memory_updates(monkeypatch):
    from megatoken.model.multitrack import MultiTrackGenerator
    from megatoken.model.transducer import metadata_payload
    from megatoken.training.transducer import Meridian

    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).train()
    multi = MultiTrackGenerator(model)
    bank = model.transducer.bank
    value = torch.randn(1, bank.geometry.value_dim)
    bank.add_edges(
        departure=torch.randn(1, bank.geometry.location_dim),
        key=torch.randn(1, bank.geometry.query_dim),
        value=value,
        destination=bank.content_location(value),
        bias=torch.zeros(1),
        write_mass=torch.ones(1),
        step=0,
    )
    before = bank.decode("value", bank.edge_ids()).clone()
    trainer = TransducerTrainer(model.transducer, bank_trainable=False)
    lookups = []
    boundary = model.transducer.boundary

    def capture(*args, **kwargs):
        result = boundary(*args, **kwargs)
        lookups.append(result[1])
        return result

    monkeypatch.setattr(model.transducer, "boundary", capture)
    sequence = trainer.sequence([Meridian(("a",), (1,), ("text",), False)], step=32768)
    assert sequence.values is None and sequence.alpha == 0
    state = sequence.begin(multi, multi.tracks.initial_state(1), metadata_payload({}))
    state = sequence.before_meridian(multi, state, 0, effort=1)
    assert len(lookups) == 1 and lookups[0].edge_ids.numel() == 1
    assert trainer.records == []
    state.final.square().sum().backward()
    trainer.after_backward(32768)
    report = trainer.barrier(32768)
    assert report["transducer_frozen"] == 1
    assert report["transducer_writeouts"] == report["transducer_updated_fields"] == 0
    torch.testing.assert_close(bank.decode("value", bank.edge_ids()), before)


def test_frozen_coordinator_does_not_prune_old_unvisited_memory():
    from megatoken.memory.transducer import TransducerBank
    from megatoken.memory.transducer_growth import TransducerGrowth
    from megatoken.training.transducer_diloco import LearnerRound, TransducerCoordinator

    bank = TransducerBank(ModelConfig.tiny().transducer_geometry)
    g = bank.geometry
    value = torch.randn(1, g.value_dim)
    bank.add_edges(
        departure=torch.randn(1, g.location_dim),
        key=torch.randn(1, g.query_dim),
        value=value,
        destination=bank.content_location(value),
        bias=torch.zeros(1),
        write_mass=torch.zeros(1),
        step=0,
    )
    coordinator = TransducerCoordinator(bank, TransducerGrowth(bank), frozen=True)
    # Without the freeze, zero-utility rows this old are eligible for pruning.
    row = LearnerRound(worker=0, sequence=1, base_version=0, examples=1, step=32768)
    coordinator.submit(row)
    coordinator.merge()
    assert bank.edge_count == 1
    assert not coordinator.record_validation(32768, 1.0)
