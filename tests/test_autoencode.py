"""Encode-and-reconstruct: feature items train E and the realizers without the core."""

import torch
from media_support import install_fake_realizers, prepared_record
from training_fixture import MODEL_OVERRIDES

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, TensorAssetStore
from megatoken.data.schema import sequential_meridians
from megatoken.data.storage import PreparedDataset, export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.training.autoencode import AutoencodePlan, FeatureSampler, autoencode_losses

CODEC = ByteCodec.byte_fallback()


def _dataset(root):
    store = TensorAssetStore(root)
    records = [prepared_record(store, m, seed=i) for i, m in enumerate(("audio", "image", "video"))]
    episodes = []
    for index, record in enumerate(records):
        for copy in range(2):  # the same item used twice is indexed once per split
            episodes.append(
                sequential_meridians(
                    Episode(
                        f"e{index}-{copy}",
                        (
                            PhysicalRecord(
                                f"r{index}-{copy}",
                                "user",
                                record.to_bytes(),
                                media_type=MEDIA_RECORD_TYPE,
                            ),
                            PhysicalRecord(
                                f"a{index}-{copy}",
                                "assistant",
                                b"ok",
                                turn_index=1,
                                policy_mask=True,
                            ),
                        ),
                        Lineage("test", "v1", f"family-{index}"),
                        stream="media",
                    )
                )
            )
    with DedupStore(
        root / "prep.sqlite", DedupConfig(train_fraction=1.0, validation_fraction=0.0)
    ) as dedup:
        with dedup.transaction():
            for episode in episodes:
                dedup.add(episode, CODEC)
        dedup.freeze()
        export_prepared(dedup, root, codec=CODEC, source_receipts=[], pipeline_fingerprint="t")
    return store, records


def test_feature_items_are_indexed_once_per_split(tmp_path):
    _dataset(tmp_path)
    dataset = PreparedDataset(tmp_path)
    assert {m: dataset.feature_count("train", m) for m in ("audio", "image", "video", "text")} == {
        "audio": 1,
        "image": 1,
        "video": 1,
        "text": 0,
    }
    item = dataset.feature_item("train", "image", 0)
    assert item.target is not None and item.embedding.kind == "embedding"
    sampler = FeatureSampler(dataset, seed=1, worker=0)
    drawn = [r.modality for r in sampler.draw(6)]
    assert sorted(drawn) == ["audio", "audio", "image", "image", "video", "video"]


def test_encode_and_reconstruct_trains_the_encoder_side_only(tmp_path):
    torch.manual_seed(3)
    store, records = _dataset(tmp_path)
    model = install_fake_realizers(build_model(ModelConfig.tiny(**MODEL_OVERRIDES), CODEC))
    center = (
        torch.nn.functional.normalize(torch.ones(model.embedding_realizer.dimension), dim=0) * 0.5
    )
    terms = autoencode_losses(model, records, store, space_fingerprint="space", center=center)
    assert terms["reconstruction"][1] == 3 and terms["embedding"][1] == 3
    (terms["reconstruction"][0] + terms["embedding"][0]).backward()
    assert model.core.media_encoder.projection.weight.grad.abs().sum() > 0
    assert model.core.frontend.reduction.ordered_merge.weight.grad is not None
    assert model.embedding_realizer.document.weight.grad.abs().sum() > 0
    for name in ("pocket", "sana", "sana_video"):
        assert model.media_realizers.packs[name].condition.project.weight.grad.abs().sum() > 0
    # Nothing passes through the core.
    assert all(p.grad is None for p in model.core.central.parameters())
    assert AutoencodePlan.from_stage({"autoencode": {"items": 8}}).items == 8
    assert AutoencodePlan.from_stage({}).items == 0


def test_media_bridge_stage_trains_encode_and_reconstruct_end_to_end(tmp_path, monkeypatch):
    import importlib.util
    from dataclasses import replace
    from pathlib import Path

    from training_fixture import make_episode, tiny_train_config

    from megatoken.training.engine import train

    script = Path(__file__).resolve().parents[1] / "scripts/prepare_media_identity.py"
    spec = importlib.util.spec_from_file_location("prepare_media_identity", script)
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    data = tmp_path / "data"
    store = TensorAssetStore(data)
    texts = [replace(make_episode(i), stream="text") for i in range(2)]
    media = [
        identity.identity_episode(prepared_record(store, m, seed=i))
        for i, m in enumerate(("audio", "image", "video"))
    ]
    with DedupStore(
        tmp_path / "prep.sqlite", DedupConfig(train_fraction=1.0, validation_fraction=0.0)
    ) as dedup:
        with dedup.transaction():
            for episode in (*texts, *media):
                dedup.add(episode, CODEC)
        dedup.freeze()
        export_prepared(dedup, data, codec=CODEC, source_receipts=[], pipeline_fingerprint="t")
    packs = tmp_path / "packs.json"
    packs.write_text("{}")
    monkeypatch.setattr(
        "megatoken.omni.loader.install_realizers",
        lambda model, path, *, device: install_fake_realizers(model),
    )
    calls = []
    original = autoencode_losses

    def watch(model, items, *args, **kwargs):
        calls.append(sorted(item.modality for item in items))
        return original(model, items, *args, **kwargs)

    monkeypatch.setattr("megatoken.training.engine.autoencode_losses", watch)
    config = tiny_train_config(
        tmp_path / "base",
        phase="MB",
        trainable="media_bridge",
        packs=str(packs),
        dataset=str(data),
        media_root=str(data),
        use_stage_mixture=True,
        max_record_bytes=4096,
        max_steps=1,
    )
    result = train(config)
    assert result["global_step"] == 1
    # The MB stage draws 64 feature items per step, in equal turns per modality.
    assert len(calls) == 1 and len(calls[0]) == 64
    assert {calls[0].count(m) for m in ("audio", "image", "video")} == {21, 22}
