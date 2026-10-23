"""Context demonstrations use the normal prepared-data and optimizer path."""

import torch
from training_fixture import tiny_train_config

from megatoken.data import DedupConfig, DedupStore, PreparedDataset, WindowConfig, iter_windows
from megatoken.data.context_tasks import context_episode
from megatoken.data.coverage import behaviors
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.training.engine import prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow
from megatoken.training.transducer import TransducerTrainer


def test_context_variants_keep_one_family_and_prove_actual_reads():
    variants = [context_episode(3, variant=v) for v in ("fetch", "pin", "local", "unavailable")]
    assert len({episode.lineage.family_id for episode in variants}) == 1
    assert all("public_context_window" in behaviors(episode) for episode in variants)
    assert "explicit_context_read" in behaviors(variants[0])
    assert "context_unavailable" in behaviors(variants[-1])
    assert not variants[-1].supervision.verified_success


def test_prepared_context_demo_restores_and_backpropagates_with_its_public_policy(tmp_path):
    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    episode = context_episode(7, split="train", local_events=4, cross_events=1)
    root = tmp_path / "prepared"
    with DedupStore(tmp_path / "spool.sqlite", DedupConfig(min_near_words=1_000_000)) as store:
        store.add(episode, codec)
        store.freeze()
        report = export_prepared(
            store, root, codec=codec, source_receipts=[], pipeline_fingerprint="context-test"
        )
    assert any(
        "explicit_context_read" in key for key in report["behavior_coverage"]["original_families"]
    )
    model = build_model(ModelConfig.tiny(max_history_chunks=32, max_capacity=256), codec)
    config = tiny_train_config(
        tmp_path / "run",
        phase="P1",
        representation_mode="bootstrap",
        max_record_bytes=4096,
        generated_capacity=256,
    )
    with PreparedDataset(root, split="train") as prepared:
        restored = prepared[0]
        window = next(
            iter_windows(restored, codec, WindowConfig(max_records=32, max_window_tokens=8192))
        )
        (batch,) = prepare_batches(
            model, [EpisodeWindow(restored, window, prepared.assets)], config
        )
        ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
        metrics = forward_meridian_batch(
            model,
            batch,
            ledger,
            config,
            torch.device("cpu"),
            0,
            transducer=TransducerTrainer(model.transducer),
        )
        ledger.finish()
        assert metrics["grouped_supervised_meridians"] == 2
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.core.predictor_parameters()
        )


def test_waiting_releases_execution_capacity_but_resumption_requires_admission():
    import pytest

    from megatoken.lifecycle import TrackRegistry

    tracks = TrackRegistry(max_active=1, max_waiting=1)
    parent = tracks.create("Coordinate")
    tracks.change(parent.address, "wait", prerequisites=())
    child = tracks.create("Read evidence", parent=parent.address)
    with pytest.raises(ValueError, match="Active work capacity"):
        tracks.change(parent.address, "continue")
    with pytest.raises(ValueError, match="Waiting work capacity"):
        tracks.change(child.address, "wait", prerequisites=())
    tracks.change(child.address, "complete")
    assert tracks.change(parent.address, "continue").status == "active"
