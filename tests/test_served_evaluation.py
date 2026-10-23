from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatoken.data.schema import Meridian, PhysicalRecord, TrackEvent
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.retrieval.collection import SourceCollection
from megatoken.teacher import PINNED_TEACHER
from megatoken.training.config import TrainConfig
from megatoken.training.evaluation import evaluate_served_records


def _episode():
    prompt = PhysicalRecord(
        record_id="prompt",
        role="user",
        payload=b"What should I do next?",
        media_type="text/plain; charset=utf-8",
    )
    target = PhysicalRecord(
        record_id="target",
        role="assistant",
        payload=b"Proceed carefully.",
        policy_mask=True,
        media_type="text/plain; charset=utf-8",
    )
    return SimpleNamespace(
        stream="chat",
        tracks=[SimpleNamespace(default_modality="text")],
        records=(prompt, target),
        meridians=(
            Meridian(0, (TrackEvent("prompt", "user", "text", 0, "user"),)),
            Meridian(1, (TrackEvent("target", "assistant", "text", 1, "assistant"),)),
        ),
        control_seed=b"",
        control_reference=b"",
        reference_origin=None,
        tools_json="[]",
    )


class _Dataset:
    def __init__(self, episode):
        self.episode = episode

    def iter_episodes(self):
        yield self.episode


def _collection(path):
    collection = SourceCollection.create(path, name="evaluation", teacher=PINNED_TEACHER)
    for index in range(4):
        key = np.zeros(768, dtype=np.float32)
        key[index] = 1
        collection.add_text(f"evidence item {index}".encode(), key, origin=f"fixture-{index}")
    collection.reindex(cells=1, k=1)
    collection.set_threshold(-1.0)
    collection.close()


def _model():
    torch.manual_seed(41)
    return build_model(ModelConfig.tiny(), ByteCodec.byte_fallback())


def test_served_evaluation_uses_session_lookups_for_prompt_and_generation(tmp_path):
    _collection(tmp_path / "sources")
    config = TrainConfig(
        purpose="diagnostic",
        phase="D",
        precision="fp32",
        device="cpu",
        generated_capacity=4,
        use_effort_mixture=False,
        microbatch_episodes=1,
        budget=64,
    )
    model = _model()
    result = evaluate_served_records(
        model,
        _Dataset(_episode()),
        config,
        torch.device("cpu"),
        episodes=1,
        collections={"docs": str(tmp_path / "sources")},
        max_sources=1,
        threshold=-1,
    )
    assert result["evaluation_mode"] == "served_session_generation"
    assert result["episodes"] == 1 and result["generated_chunks"] == 1
    assert result["session_lookups"] >= 3  # metadata, observed prompt, generated answer
    assert result["collection_items_retrieved"] >= 3
    assert model.training


def test_served_session_benchmark_counts_actual_prompt_and_answer_lookups(tmp_path):
    from megatoken.benchmark import benchmark_served_session

    _collection(tmp_path / "sources")
    result = benchmark_served_session(
        config="tiny",
        device="cpu",
        steps=2,
        warmup=0,
        batch=2,
        record_bytes=8,
        capacity=2,
        effort=1,
        collections={"docs": str(tmp_path / "sources")},
        max_sources=1,
        threshold=-1,
    )
    assert result["workload"] == "served_session_inference"
    assert result["counts"]["session_opens"] == 4
    assert result["counts"]["prompt_events"] == 4
    assert result["counts"]["generated_groups"] == 4
    assert result["counts"]["runtime_lookups"] >= 8
    assert result["counts"]["collection_items_retrieved"] >= 8
    assert len(result["latency_seconds"]["per_session"]) == 4
    assert len(result["latency_seconds"]["per_batch"]) == 2


def test_session_benchmark_mounts_a_base_for_the_actual_shipped_weights(tmp_path):
    from test_served import checkpoint_model

    from megatoken.benchmark import benchmark_served_session
    from megatoken.kb import new_base
    from megatoken.served import ServedBundle, export_served

    _, checkpoint = checkpoint_model(tmp_path / "checkpoint")
    bundle = tmp_path / "bundle"
    export_served(checkpoint, bundle)
    shipped = ServedBundle(bundle, device="cpu")
    base = new_base(shipped.model, "notes")
    path = tmp_path / "notes.kb"
    base.save(path)
    result = benchmark_served_session(
        bundle=bundle,
        bases={"notes": str(path)},
        steps=1,
        warmup=0,
        capacity=2,
        record_bytes=8,
        effort=1,
    )
    assert result["configuration"]["knowledge_bases"] == ["notes"]
    assert result["configuration"]["model"]["max_capacity"] == 8
    assert result["counts"]["runtime_lookups"] >= 2


def test_evaluate_dataset_override_expands_new_glob_before_old_checkpoint_paths(
    tmp_path, monkeypatch
):
    from megatoken import cli
    from megatoken.training.config import TrainConfig

    local_root = tmp_path / "local-prepared-a"
    local_root.mkdir()
    saved = TrainConfig(
        purpose="diagnostic",
        phase="D",
        precision="fp32",
        device="cpu",
        datasets=[str(tmp_path / "old-cloud-root" / "*")],
        generated_capacity=4,
        use_effort_mixture=False,
        microbatch_episodes=1,
        budget=64,
    )
    trained = SimpleNamespace(
        model=object(),
        codec=object(),
        payload={"train_config": asdict(saved)},
    )
    monkeypatch.setattr("megatoken.loading.load_trained", lambda *args, **kwargs: trained)

    opened = {}

    class Corpus:
        def __init__(self, roots, *, split):
            opened["roots"], opened["split"] = roots, split

        def close(self):
            pass

    monkeypatch.setattr("megatoken.data.PreparedCorpus", Corpus)
    monkeypatch.setattr(
        "megatoken.training.evaluation.evaluate_records",
        lambda model, dataset, codec, config, device, *, episodes: {"roots": config.datasets},
    )
    result = cli._evaluate(
        SimpleNamespace(
            checkpoint="checkpoint",
            codec=None,
            device="cpu",
            seed=17,
            effort=6,
            datasets=[str(tmp_path / "local-prepared-*")],
            split="validation",
            core_only=True,
            episodes=1,
            collection=[],
            base=[],
        )
    )
    assert result["roots"] == [str(local_root)]
    assert opened == {"roots": [str(local_root)], "split": "validation"}
    with pytest.raises(ValueError, match="served evaluation"):
        cli._evaluate(
            SimpleNamespace(
                checkpoint="checkpoint",
                codec=None,
                device="cpu",
                seed=17,
                effort=6,
                datasets=[str(tmp_path / "local-prepared-*")],
                split="validation",
                core_only=True,
                episodes=1,
                collection=["docs=unused"],
                base=[],
            )
        )
