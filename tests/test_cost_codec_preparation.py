"""Input-codec pretraining reconstructs public costs without importing choice labels."""

import importlib.util
from pathlib import Path

import torch

from megatoken.data import (
    DedupConfig,
    DedupStore,
    Episode,
    Lineage,
    PhysicalRecord,
    WindowConfig,
    iter_windows,
)
from megatoken.data.schema import sequential_meridians
from megatoken.data.storage import PreparedDataset, export_prepared
from megatoken.generation import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.training.config import TrainConfig
from megatoken.training.engine import prepare_batches
from megatoken.training.stream import EpisodeWindow


def test_public_copy_views_preserve_splits_and_train_actual_lexical_positions(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/accept_delegation_choice.py"
    spec = importlib.util.spec_from_file_location("cost_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    codec = ByteCodec.byte_fallback()
    source = tmp_path / "source"
    with DedupStore(tmp_path / "source.sqlite", DedupConfig(min_near_words=1_000_000)) as store:
        for index, split in enumerate(("train", "validation")):
            costs = f'{{"public_read_cost":{index + 1}}}'.encode()
            episode = sequential_meridians(
                Episode(
                    f"task-{index}",
                    (
                        PhysicalRecord("request", "user", b"Choose a procedure."),
                        PhysicalRecord("costs", "environment", costs),
                        PhysicalRecord(
                            "choice", "assistant", b"PRIVATE_CHOICE_LABEL", policy_mask=True
                        ),
                    ),
                    Lineage("fixture", "v1", f"family-{index}", frozen_split=split),
                )
            )
            store.add(episode, codec)
        store.freeze()
        export_prepared(
            store,
            source,
            codec=codec,
            pipeline_fingerprint="fixture",
            source_receipts=[{"origin": "authored"}],
        )
    output = tmp_path / "copies"
    module.prepare_codec_views(source, output)
    torch.set_num_threads(1)
    model = build_model(ModelConfig.tiny(max_capacity=512), codec)
    config = TrainConfig(
        purpose="diagnostic",
        max_steps=1,
        phase="B0",
        model="tiny",
        codec="byte",
        allow_byte_codec=True,
        device="cpu",
        precision="fp32",
        datasets=[str(output)],
        generated_capacity=512,
    )
    for index, split in enumerate(("train", "validation")):
        with PreparedDataset(output, split=split) as data:
            (sample,) = list(data.iter_episodes())
            assert sample.lineage.family_id == f"family-{index}"
            assert sample.lineage.frozen_split == split
            assert sample.records[0].payload == sample.records[1].payload
            assert sample.records[1].policy_mask
            assert b"PRIVATE_CHOICE_LABEL" not in sample.records[1].payload
            (window,) = list(
                iter_windows(
                    sample,
                    codec,
                    WindowConfig(max_records=8, max_decisions=4, max_window_tokens=4096),
                )
            )
            (batch,) = prepare_batches(model, [EpisodeWindow(sample, window, data.assets)], config)
            assert batch.counts["lexical"] == len(sample.records[1].payload)
            assert batch.counts["reconstruction"] == len(sample.records[1].payload)
