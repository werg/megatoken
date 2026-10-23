"""Training from several prepared roots, stage budgets in their own meters, held-out evaluation."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import socket
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from media_support import install_fake_realizers, prepared_record
from training_fixture import MODEL_OVERRIDES, make_episode, prepare_dataset, tiny_train_config

from megatoken.data import (
    DedupConfig,
    DedupStore,
    PreparedCorpus,
    PreparedDataset,
    WindowConfig,
)
from megatoken.data.media_assets import MediaRecord, TensorAssetStore
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model, load_model_config
from megatoken.model import ModelConfig
from megatoken.runtime import seed_everything
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import DecoupledConfig, TrainConfig
from megatoken.training.diloco_protocol import training_identity
from megatoken.training.diloco_server import initialize_coordinator, serve_coordinator
from megatoken.training.engine import train, window_config
from megatoken.training.heldout import HeldOutEvaluator, merge
from megatoken.training.schedule import PHASES
from megatoken.training.stream import MixtureWindowStream
from megatoken.training.transducer import HindsightSchedule, TransducerTrainer

CODEC = ByteCodec.byte_fallback()
WINDOWS = WindowConfig(max_window_tokens=4096, max_decisions=4, max_records=8)


def _identity_episode(record):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_media_identity.py"
    spec = importlib.util.spec_from_file_location("prepare_media_identity", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.identity_episode(record)


def _split(episode, split):
    return replace(episode, lineage=replace(episode.lineage, frozen_split=split))


def text(index, *, stream="text", split="train"):
    return _split(replace(make_episode(index), stream=stream), split)


def media(store, modality, seed, *, split="train"):
    return _split(_identity_episode(prepared_record(store, modality, seed=seed)), split)


def export(root, episodes, *, codec=CODEC):
    """One preparation script's output: an immutable root that holds its own episodes."""
    spool = root.parent / f"{root.name}-spool.sqlite"
    with DedupStore(spool, DedupConfig(train_fraction=1.0, validation_fraction=0.0)) as dedup:
        with dedup.transaction():
            for episode in episodes:
                dedup.add(episode, codec)
        dedup.freeze()
        export_prepared(
            dedup, root, codec=codec, source_receipts=[], pipeline_fingerprint="multi-root-test"
        )
    return root


def two_roots(tmp_path):
    """Text in one root; media identities, with their tensors, and more text in the other."""
    first = export(
        tmp_path / "first",
        [text(0, stream="alpha"), text(1, stream="alpha"), text(2, stream="alpha")],
    )
    store = TensorAssetStore(tmp_path / "second")
    second = export(
        tmp_path / "second",
        [
            text(3, stream="alpha"),
            text(4, stream="alpha"),
            media(store, "image", 1),
            media(store, "image", 2),
            media(store, "audio", 3),
        ],
    )
    return first, second


# --- the corpus -------------------------------------------------------------------------


def test_corpus_orders_roots_canonically_and_pairs_each_episode_with_its_own_assets(tmp_path):
    first, second = two_roots(tmp_path)
    forward = PreparedCorpus([first, second], stream="alpha")
    backward = PreparedCorpus([second, first], stream="alpha")
    assert len(forward) == 5 and forward.identity == backward.identity
    ids = [forward.fetch(i)[0].episode_id for i in range(5)]
    assert ids == [backward.fetch(i)[0].episode_id for i in range(5)]
    for index in range(5):
        episode, assets = forward.fetch(index)
        number = int(episode.episode_id.rsplit("-", 1)[1])
        assert assets.root == (first if number < 3 else second).resolve()

    images = PreparedCorpus([first, second], stream="image_identity")
    assert len(images) == 2 and len(images.parts) == 1  # the other root has none of it
    episode, assets = images.fetch(0)
    record = MediaRecord.from_bytes(episode.records[1].payload)
    assert assets.load(record.source).shape[0] > 0
    with pytest.raises((FileNotFoundError, ValueError)):
        TensorAssetStore(first).load(record.source)  # the tensors live in their own root

    everything = PreparedCorpus([first, second])
    assert len(everything) == 8
    assert {m: everything.feature_count("train", m) for m in ("image", "audio", "video")} == {
        "image": 2,
        "audio": 1,
        "video": 0,
    }
    item, item_assets = everything.feature_item("train", "audio", 0)
    assert item.modality == "audio" and item_assets.root == second.resolve()
    with pytest.raises(IndexError):
        everything.feature_item("train", "audio", 1)


def test_incompatible_or_repeated_roots_fail_clearly(tmp_path, monkeypatch):
    first, second = two_roots(tmp_path)
    with pytest.raises(ValueError, match="listed twice"):
        PreparedCorpus([first, first])
    other = ByteCodec(list(CODEC._pieces), fingerprint="f" * 64)
    foreign = export(tmp_path / "foreign", [text(9, stream="alpha")], codec=other)
    with pytest.raises(ValueError, match="different codecs"):
        PreparedCorpus([first, foreign])
    third_store = TensorAssetStore(tmp_path / "third")
    third = export(tmp_path / "third", [media(third_store, "video", 7)])
    monkeypatch.setattr(
        PreparedDataset, "feature_extractors", lambda self: {f"teacher-{self.root.name}"}
    )
    with pytest.raises(ValueError, match="different teachers"):
        PreparedCorpus([second, third])


# --- streams ----------------------------------------------------------------------------


def test_mixture_streams_resolve_assets_per_episode_and_shard_for_diloco_workers(tmp_path):
    first, second = two_roots(tmp_path)
    shares = {"alpha": 0.5, "image_identity": 0.5}

    def worker(rank, roots):
        return MixtureWindowStream(
            roots,
            CODEC,
            WINDOWS,
            shares,
            "G",
            seed=3,
            rank=rank,
            world_size=2,
            reduce_counts=lambda counts: counts,  # DiLoCo workers share no collective
            independent=True,
        )

    seen = {0: [], 1: []}
    for rank in (0, 1):
        stream = worker(rank, [first, second])
        try:
            for _ in range(8):
                item = stream.next()
                number = (
                    int(item.episode.episode_id.rsplit("-", 1)[1])
                    if item.episode.stream == "alpha"
                    else None
                )
                if number is not None:
                    assert item.assets.root == (first if number < 3 else second).resolve()
                    seen[rank].append(item.episode.episode_id)
                else:
                    assert item.assets.root == second.resolve()
                    record = MediaRecord.from_bytes(item.episode.records[1].payload)
                    item.assets.load(record.source)
        finally:
            stream.close()
    # One alpha epoch over both roots is split between the two workers without overlap.
    assert not set(seen[0]) & set(seen[1])
    assert len({*seen[0], *seen[1]}) >= 4

    # A cursor resumes under any listing order or location of the same roots.
    original = worker(0, [first, second])
    swapped = worker(0, [second, first])
    try:
        for _ in range(3):
            original.next()
        swapped.load_state_dict(original.state_dict())
        assert swapped.next().episode.episode_id == original.next().episode.episode_id
    finally:
        original.close()
        swapped.close()


# --- stage budgets ----------------------------------------------------------------------


def test_a_run_defaults_to_its_stages_own_meter_and_limit():
    for phase in ("B0", "MB", "P1", "M1", "S"):
        schedule = TrainConfig(
            phase=phase,
            trainable="media_bridge" if phase == "MB" else "all",
            packs="p" if phase == "MB" else None,
        ).schedule
        assert (schedule.meter, schedule.budget) == (PHASES[phase].meter, PHASES[phase].budget)
    assert [TrainConfig(phase=p).schedule.meter for p in ("B0", "P1", "S")] == ["I", "D", "P"]
    shorter = TrainConfig(phase="P1", budget=1000).schedule
    assert (shorter.meter, shorter.budget) == ("D", 1000)
    assert shorter.learning_rate(500) > shorter.learning_rate(1000)  # the decay fits the override
    with pytest.raises(ValueError, match="stage's own limit"):
        TrainConfig(phase="P1", budget=PHASES["P1"].budget + 1)
    with pytest.raises(ValueError, match="budget"):
        TrainConfig(phase="S", budget=0)
    with pytest.raises(ValueError, match="nonempty list"):
        TrainConfig(datasets="artifacts/demo")


@pytest.mark.parametrize("phase,meter", [("B0", "I"), ("P1", "D"), ("S", "P")])
def test_training_stops_at_the_budget_in_the_stages_meter(tmp_path, phase, meter):
    torch.set_num_threads(1)
    assert PHASES[phase].meter == meter
    config = tiny_train_config(tmp_path, phase=phase, max_steps=4)
    probe = train(config)
    events = [
        json.loads(line)
        for line in (Path(config.output) / "metrics.jsonl").read_text().splitlines()
    ]
    consumed = [e["phase_consumed"] for e in events if e["event"] == "train"]
    assert len(consumed) == 4 and consumed == sorted(set(consumed))
    assert probe["counters"][meter] == consumed[-1] and not probe["complete"]

    # The same trajectory with a budget that the second step crosses: it ends there.
    limited = replace(
        config, output=config.output + "-limited", max_steps=None, budget=consumed[0] + 1
    )
    result = train(limited)
    assert result["global_step"] == 2 and result["complete"]
    assert result["phase_consumed"] == consumed[1] == result["counters"][meter]
    # A budget that the first step already meets stops after one update.
    exact = train(replace(limited, output=config.output + "-exact", budget=consumed[0]))
    assert exact["global_step"] == 1 and exact["complete"]


# --- multi-root media training and held-out evaluation ----------------------------------


def media_bridge_roots(tmp_path):
    """MB's streams over two roots, each with a train and a validation episode."""
    first = export(
        tmp_path / "text-root",
        [text(0), text(1), text(2, split="validation"), text(3, split="validation")],
    )
    store = TensorAssetStore(tmp_path / "media-root")
    episodes = []
    for seed, modality in enumerate(("audio", "image", "video")):
        episodes.append(media(store, modality, seed))
        episodes.append(media(store, modality, seed + 10, split="validation"))
    second = export(tmp_path / "media-root", episodes)
    return first, second


def media_bridge_config(tmp_path, roots, **overrides):
    packs = tmp_path / "packs.json"
    packs.write_text("{}")
    return tiny_train_config(
        tmp_path / "base",
        phase="MB",
        trainable="media_bridge",
        packs=str(packs),
        datasets=[str(root) for root in roots],
        use_stage_mixture=True,
        max_record_bytes=4096,
        **overrides,
    )


@pytest.fixture
def fake_realizers(monkeypatch):
    monkeypatch.setattr(
        "megatoken.omni.loader.install_realizers",
        lambda model, path, *, device: install_fake_realizers(model),
    )


def test_media_bridge_trains_from_two_roots_and_evaluates_every_stream(tmp_path, fake_realizers):
    torch.set_num_threads(1)
    roots = media_bridge_roots(tmp_path)
    config = media_bridge_config(tmp_path, roots, max_steps=2, eval_every=1, eval_windows=2)
    result = train(config)
    assert result["global_step"] == 2 and not result["complete"]
    events = [
        json.loads(line)
        for line in (Path(config.output) / "metrics.jsonl").read_text().splitlines()
    ]
    evaluations = [e for e in events if e["event"] == "evaluation"]
    assert [e["global_step"] for e in evaluations] == [1, 2]
    for evaluation in evaluations:
        assert set(evaluation["streams"]) == {
            "text",
            "audio_identity",
            "image_identity",
            "video_identity",
        }
        # Text: two validation episodes, both scored; media: one each.
        assert {n: s["windows"] for n, s in evaluation["streams"].items()} == {
            "text": 2,
            "audio_identity": 1,
            "image_identity": 1,
            "video_identity": 1,
        }
        streams = evaluation["streams"]
        assert streams["text"]["loss"]["lexical"] > 0
        assert "audio_prediction" in streams["audio_identity"]["loss"]
        assert "visual_prediction" in streams["image_identity"]["loss"]
        assert "visual_reconstruction" in streams["video_identity"]["loss"]
        assert all(s["weighted_loss"] > 0 for s in streams.values())
        assert evaluation["pooled_weighted_loss"] > 0 and evaluation["evaluation_seconds"] >= 0
    # Evaluation is not a training step: the optimizer moved exactly twice.
    payload, _ = load_checkpoint(result["checkpoint"])
    assert payload["global_step"] == 2
    assert {int(s["step"]) for s in payload["optimizer"]["moments"].values()} == {2}


def test_stage_mixture_with_autoencode_resumes_exactly(tmp_path, fake_realizers):
    """The stream cursor, the encode-and-reconstruct draws and the random state all resume."""
    torch.set_num_threads(1)
    roots = media_bridge_roots(tmp_path)
    config = media_bridge_config(tmp_path, roots, max_steps=1)
    full = train(replace(config, output=config.output + "-full", max_steps=2))
    first = train(config)
    resumed = train(config, resume=first["checkpoint"])
    uninterrupted, _ = load_checkpoint(full["checkpoint"])
    recovered, _ = load_checkpoint(resumed["checkpoint"])
    sampler = uninterrupted["rank_state"][0]["autoencode"]
    assert sampler["turn"] > 64  # two steps of 64 draws each moved the sampler
    assert recovered["rank_state"][0]["autoencode"] == sampler
    assert recovered["rank_state"][0]["cursor"] == uninterrupted["rank_state"][0]["cursor"]
    for name, value in uninterrupted["model"].items():
        torch.testing.assert_close(recovered["model"][name], value, rtol=0, atol=0)


def test_heldout_evaluation_is_deterministic_bounded_and_splits_across_ranks(
    tmp_path, fake_realizers
):
    torch.set_num_threads(1)
    roots = media_bridge_roots(tmp_path)
    config = media_bridge_config(tmp_path, roots, eval_windows=1)
    seed_everything(config.seed)
    model = install_fake_realizers(build_model(ModelConfig.tiny(**MODEL_OVERRIDES), CODEC))
    model.train()
    transducer = TransducerTrainer(model.transducer)
    evaluator = HeldOutEvaluator(config, CODEC, window_config(config), require=True)
    try:
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state()
        first = evaluator.sums(model, transducer, 0, rank=0, world=1)
        second = evaluator.sums(model, transducer, 0, rank=0, world=1)
        assert first == second  # the same windows, the same draws
        assert model.training and torch.equal(torch.get_rng_state(), rng)
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        # eval_windows bounds each stream, although text holds two validation episodes.
        assert {name: entry["windows"] for name, entry in first.items()}["text"] == 1

        evaluator.config = replace(config, eval_windows=2)
        whole = evaluator.sums(model, transducer, 0, rank=0, world=1)
        halves = [evaluator.sums(model, transducer, 0, rank=r, world=2) for r in (0, 1)]
        assert whole["text"]["windows"] == 2
        assert [h["text"]["windows"] for h in halves] == [1, 1]
        streams, pooled = merge(halves)
        reference, reference_pooled = merge([whole])
        for name in reference:
            assert streams[name]["count"] == reference[name]["count"]
            for term, value in reference[name]["sum"].items():
                assert streams[name]["sum"][term] == pytest.approx(value, rel=1e-4)
        assert pooled["windows"] == reference_pooled["windows"] == 5
        report = evaluator.report(halves)
        assert report["streams"]["text"]["windows"] == 2
        assert report["pooled_weighted_loss"] == pytest.approx(
            evaluator.bank_only_loss(model, transducer, 0), rel=1e-4
        )
    finally:
        evaluator.close()


def test_evaluation_without_validation_episodes_fails_at_launch(tmp_path):
    config = tiny_train_config(tmp_path, eval_every=1)
    with pytest.raises(ValueError, match="validation episodes"):
        train(config)


def with_validation_root(tmp_path, config, count=3, first=5):
    held_out = prepare_dataset(
        tmp_path / "held-out", [make_episode(first + i) for i in range(count)], heldout=True
    )
    return replace(config, datasets=[*config.datasets, str(held_out)])


def test_an_unscoreable_validation_episode_stops_the_run_with_a_clear_error(tmp_path):
    torch.set_num_threads(1)
    # Its answer is longer than generated_capacity, as it would be in training.
    config = with_validation_root(
        tmp_path, tiny_train_config(tmp_path, eval_every=1), count=1, first=100
    )
    with pytest.raises(RuntimeError, match="Held-out evaluation failed.*generated_capacity"):
        train(config)


def test_whole_corpus_is_one_evaluated_stream_without_a_stage_mixture(tmp_path):
    torch.set_num_threads(1)
    config = with_validation_root(
        tmp_path, tiny_train_config(tmp_path, eval_every=1, eval_windows=2)
    )
    train(config)
    events = [
        json.loads(line)
        for line in (Path(config.output) / "metrics.jsonl").read_text().splitlines()
    ]
    (evaluation,) = [e for e in events if e["event"] == "evaluation"]
    # No stage mixture: the whole corpus is one stream, bounded by eval_windows.
    assert set(evaluation["streams"]) == {"all"} and evaluation["streams"]["all"]["windows"] == 2


def test_diloco_worker_evaluates_and_trains_from_several_roots(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = "test-diloco-token-" + "b" * 32
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    settings = DecoupledConfig(
        coordinator_url=f"http://127.0.0.1:{port}",
        run_id="multi-root-v1",
        workers=1,
        worker_id=0,
        local_steps=1,
        io_workers=2,
        timeout_seconds=30,
        max_offline_steps=20,
    )
    config = with_validation_root(
        tmp_path,
        tiny_train_config(
            tmp_path,
            output=str(tmp_path / "runs"),
            inner_optimizer="muon",
            decoupled=settings,
            eval_every=1,
            eval_windows=2,
        ),
    )
    with PreparedCorpus(config.datasets) as prepared:
        fingerprint = prepared.identity
    seed_everything(config.seed, config.deterministic)
    model_config = load_model_config(
        config.model,
        **{
            **config.model_overrides,
            "expert_backend": config.expert_backend,
            "checkpoint_stages": config.gradient_checkpointing,
        },
    )
    model = build_model(model_config, CODEC, device="cpu", precision="fp32")
    root = tmp_path / "coordinator"
    initialize_coordinator(
        root,
        run_id=settings.run_id,
        model_state=model.state_dict(),
        masters={
            name: p.detach().float() for name, p in model.named_parameters() if p.requires_grad
        },
        model_config=model_config.to_dict(),
        codec_fingerprint=CODEC.fingerprint,
        train_identity=training_identity(config),
        dataset_fingerprint=fingerprint,
        workers=1,
        min_quorum=1,
        fragment_bytes=1 << 20,
        grace_seconds=0,
        token_budget=config.schedule.budget,
        transducer={
            "geometry": dict(vars(model_config.transducer_geometry)),
            "evidence_from": HindsightSchedule().warmup_steps,
        },
    )
    server = multiprocessing.get_context("spawn").Process(
        target=serve_coordinator, args=(str(root), "127.0.0.1", port, token)
    )
    server.start()
    try:
        url = f"http://127.0.0.1:{port}/v1/status"
        for _ in range(100):
            try:
                request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
                with urllib.request.urlopen(request, timeout=1) as response:
                    assert response.status == 200
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise AssertionError("Coordinator failed to start")
        result = train(config)
        assert result["global_step"] == 1
    finally:
        server.terminate()
        server.join(timeout=10)
        if server.is_alive():
            server.kill()
            server.join()
    events = [
        json.loads(line)
        for line in (Path(config.output) / "worker-00" / "metrics.jsonl").read_text().splitlines()
    ]
    (evaluation,) = [e for e in events if e["event"] == "evaluation"]
    assert evaluation["streams"]["all"]["windows"] == 2


def test_a_dataset_pattern_stands_for_every_root_it_matches(tmp_path):
    from megatoken.training.config import TrainConfig

    for name in ("00002", "00000", "00001"):
        (tmp_path / "prepared" / name).mkdir(parents=True)
    (tmp_path / "prepared" / "stray.json").write_text("{}")  # a file is not a root
    config = TrainConfig.from_dict({"datasets": [str(tmp_path / "prepared" / "0*"), "plain/root"]})
    roots = [str(tmp_path / "prepared" / name) for name in ("00000", "00001", "00002")]
    assert config.datasets == [*roots, "plain/root"]
    with pytest.raises(ValueError, match="matches no prepared root"):
        TrainConfig.from_dict({"datasets": [str(tmp_path / "missing" / "*")]})
