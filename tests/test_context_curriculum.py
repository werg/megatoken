"""Exercise original-task splitting, evidence checks and real training exposure."""

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megatoken.contracts import ROLE_TO_ID
from megatoken.data import DedupConfig, DedupStore, PreparedCorpus, PreparedDataset
from megatoken.data.context_curriculum import (
    STRATEGIES,
    TEMPLATES,
    WINDOW_PROFILES,
    curriculum_episode,
    curriculum_episodes,
    validate_curriculum_episode,
)
from megatoken.data.coverage import Coverage, behaviors, require_coverage
from megatoken.data.packing import WindowConfig
from megatoken.data.recipes import resolve_stream_producer, stage_recipe
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.context import DEFAULT_CROSS_EVENTS, DEFAULT_LOCAL_EVENTS
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.training.stream import MixtureWindowStream


def test_original_task_variants_stay_in_one_split_and_cover_every_task_and_choice():
    episodes = list(curriculum_episodes(8))
    families = {}
    choices = {}
    for episode in episodes:
        family = episode.lineage.family_id
        annotations = json.loads(episode.supervision.annotations_json)
        families.setdefault(family, set()).add(episode.lineage.frozen_split)
        choices.setdefault(family, set()).add(
            (annotations["context_strategy"], tuple(annotations["context_profile"]))
        )
    assert all(len(splits) == 1 for splits in families.values())
    assert all(
        len(variants) == len(STRATEGIES) * len(WINDOW_PROFILES) for variants in choices.values()
    )
    for split in ("train", "validation", "test"):
        assert {
            json.loads(e.supervision.annotations_json)["task_template"]
            for e in episodes
            if e.lineage.frozen_split == split
        } == set(TEMPLATES)
    assert resolve_stream_producer("context_management") is curriculum_episode
    assert (ModelConfig().context_local_events, ModelConfig().context_cross_events) == (
        DEFAULT_LOCAL_EVENTS,
        DEFAULT_CROSS_EVENTS,
    )


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_task_strategy_has_checked_evidence_and_cleanup(template, strategy):
    episode = curriculum_episode(
        2, template=template, strategy=strategy, local_events=16, cross_events=8
    )
    found = behaviors(episode)
    assert {
        "context_task_verified",
        "context_template_" + template,
        "context_strategy_" + strategy,
    } <= found
    cleanup = episode.actions[-1]
    assert cleanup.tool_name == "track_context"
    assert json.loads(cleanup.arguments_json)["local_events"] == 0
    if strategy == "unavailable":
        assert "context_unavailable" in found
        assert not episode.supervision.verified_success
    elif strategy == "fetch":
        assert "explicit_context_read" in found
        if template in ("evidence_join", "latest_revision"):
            assert "executed_parallel_calls" in found


def test_corrupted_answer_or_read_receipt_is_rejected_instead_of_counting_as_supply():
    episode = curriculum_episode(5)
    metadata = json.loads(episode.supervision.annotations_json)
    broken = replace(
        episode,
        records=tuple(
            replace(record, payload=b"unsupported answer")
            if record.record_id == metadata["context_answer"]
            else record
            for record in episode.records
        ),
    )
    with pytest.raises(ValueError, match="actual source evidence"):
        behaviors(broken)
    metadata["context_read_receipts"][0]["available"] = False
    broken = replace(
        episode, supervision=replace(episode.supervision, annotations_json=json.dumps(metadata))
    )
    with pytest.raises(ValueError, match="receipt"):
        behaviors(broken)
    metadata = json.loads(episode.supervision.annotations_json)
    metadata["context_profile"] = [64, 8]
    broken = replace(
        episode, supervision=replace(episode.supervision, annotations_json=json.dumps(metadata))
    )
    with pytest.raises(ValueError, match="public window"):
        behaviors(broken)
    broken = replace(
        episode,
        records=tuple(
            replace(r, payload=b"{}") if r.role == "system" and b'"evidence"' in r.payload else r
            for r in episode.records
        ),
    )
    with pytest.raises(ValueError, match="address absent"):
        behaviors(broken)


def test_one_original_task_cannot_fill_default_context_family_requirements():
    coverage = Coverage(sqlite3.connect(":memory:"))
    for local, cross in WINDOW_PROFILES:
        for strategy in STRATEGIES:
            episode = curriculum_episode(
                1, strategy=strategy, local_events=local, cross_events=cross
            )
            coverage.add(episode, "train", episode.stream)
    corpus = SimpleNamespace(
        parts=[SimpleNamespace(manifest={"behavior_coverage": coverage.report()})]
    )
    with pytest.raises(ValueError, match="lack prepared original families"):
        require_coverage(corpus, {"context_task_verified": 4}, {"context_management"})


@pytest.mark.parametrize("template", ("evidence_join", "user_constraint"))
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_actual_runtime_keeps_all_supervised_evidence_visible(template, strategy):
    torch.set_num_threads(1)
    episode = curriculum_episode(
        7, template=template, strategy=strategy, local_events=16, cross_events=8
    )
    model = build_model(
        ModelConfig.tiny(max_history_chunks=128, max_capacity=512), ByteCodec.byte_fallback()
    )
    runtime = MultiTrackRuntime(
        MultiTrackGenerator.of(model), context_archive_events=1 if strategy == "unavailable" else 64
    )
    records = {r.record_id: r for r in episode.records}
    try:
        session = runtime.open(effort=1)
        for group in episode.meridians:
            for event in group.events:
                needed = {
                    d.writer_id for d in episode.dependencies if d.reader_id == event.record_id
                }
                if needed:
                    assert needed <= set(
                        runtime.context_status(session, track=event.track_id)["visible_events"]
                    )
                record = records[event.record_id]
                if record.action_id:
                    arguments = json.loads(record.payload)["arguments"]
                    visible = runtime.context_status(session, track=event.track_id)[
                        "visible_events"
                    ]
                    text = b"\n".join(
                        records[address].payload for address in visible if address in records
                    )
                    handles = [arguments["address"], *arguments.get("pinned_events", ())]
                    if "source_event" in arguments:
                        handles.append(arguments["source_event"])
                        original = session.context_archive.get(arguments["source_event"])
                        assert (original is None) == (strategy == "unavailable")
                        if original is not None:
                            assert original.payload == records[arguments["source_event"]].payload
                    assert all(handle.encode() in text for handle in handles)
            runtime.observe(
                session,
                tuple(
                    ObservedEvent(
                        e.track_id,
                        e.modality,
                        records[e.record_id].payload,
                        ROLE_TO_ID[e.role],
                        event_id=e.record_id,
                    )
                    for e in group.events
                ),
                effort=1,
            )
            for event in group.events:
                record = records[event.record_id]
                if record.action_id:
                    arguments = json.loads(record.payload)["arguments"]
                    if arguments.get("pinned_events"):
                        assert set(arguments["pinned_events"]) <= set(
                            runtime.context_status(session, track=event.track_id)["visible_events"]
                        )
    finally:
        runtime.close()


def test_export_keeps_admitted_supply_and_normal_mixture_reads_scored_examples(tmp_path):
    codec = ByteCodec.byte_fallback()
    root = tmp_path / "prepared"
    with DedupStore(tmp_path / "spool.sqlite", DedupConfig(min_near_words=1_000_000)) as store:
        with store.transaction():
            for episode in curriculum_episodes(
                32, profiles=((DEFAULT_LOCAL_EVENTS, DEFAULT_CROSS_EVENTS),)
            ):
                store.add(episode, codec)
        store.freeze()
        report = export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=[],
            pipeline_fingerprint="context-curriculum-test",
        )
    requirements = {
        k: v
        for k, v in stage_recipe("P1").constraints["required_behaviors"].items()
        if k.startswith("context_") or k in ("public_context_window", "explicit_context_read")
    }
    for split in ("train", "validation"):
        with PreparedCorpus([root], split=split) as corpus:
            require_coverage(corpus, requirements, {"context_management"}, split=split)
    assert report["counts"]["split:train|stream:context_management"] > 0
    with PreparedDataset(root, split=None) as prepared:
        seen = {}
        for episode in prepared.iter_episodes():
            validate_curriculum_episode(episode)
            seen.setdefault(episode.lineage.family_id, set()).add(episode.lineage.frozen_split)
        assert all(len(splits) == 1 for splits in seen.values())
    stream = MixtureWindowStream(
        [root],
        codec,
        WindowConfig(max_records=16),
        {"context_management": 1.0},
        "D",
        seed=17,
        rank=0,
        world_size=1,
    )
    try:
        windows = [stream.next() for _ in range(8)]
        assert all(
            window is not None and any(r.record.policy_mask for r in window.window.records)
            for window in windows
        )
    finally:
        stream.close()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_context_choices_reach_prepared_losses_and_finite_predictor_gradients(tmp_path, strategy):
    from training_fixture import TEXT_ONLY_ASSETS, tiny_train_config

    from megatoken.data import iter_windows
    from megatoken.training.engine import prepare_batches
    from megatoken.training.losses import LossLedger
    from megatoken.training.meridians import forward_meridian_batch
    from megatoken.training.stream import EpisodeWindow
    from megatoken.training.transducer import TransducerTrainer

    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    episode = curriculum_episode(
        4, template="evidence_join", strategy=strategy, local_events=16, cross_events=8
    )
    model = build_model(ModelConfig.tiny(max_history_chunks=128, max_capacity=512), codec)
    config = tiny_train_config(
        tmp_path / strategy,
        phase="P1",
        representation_mode="bootstrap",
        generated_capacity=512,
        max_record_bytes=4096,
    )
    window = next(
        iter_windows(
            episode, codec, WindowConfig(max_records=128, max_decisions=16, max_window_tokens=16384)
        )
    )
    assert window.end == len(episode.records)
    (batch,) = prepare_batches(model, [EpisodeWindow(episode, window, TEXT_ONLY_ASSETS)], config)
    ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
    values = forward_meridian_batch(
        model,
        batch,
        ledger,
        config,
        torch.device("cpu"),
        0,
        transducer=TransducerTrainer(model.transducer),
    )
    ledger.finish()
    assert values["grouped_supervised_meridians"] >= 2
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.core.predictor_parameters()
    )
