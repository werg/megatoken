"""Compressed reading: long observed text enters as teacher features, never generated."""

import pytest
import torch
from retrieval_fixture import MODEL_OVERRIDES, retrieval_aux, retrieval_train_config

from megatoken.data import WindowConfig, iter_windows
from megatoken.data.adapters import AdapterContext, adapt_row
from megatoken.data.compression import CompressionPolicy, DocumentCompressor
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.omni.embeddinggemma2 import TeacherOutput
from megatoken.teacher import HIDDEN_WIDTH, PINNED_TEACHER
from megatoken.training.engine import prepare_batches
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow

CODEC = ByteCodec.byte_fallback()


class CountingTeacher:
    """One feature row per 64 bytes; refuses pieces over ``limit`` bytes like the 8192 cap."""

    def __init__(self, limit=6000):
        self.limit, self.calls = limit, []

    def encode(self, modality, source, *, prompt=""):
        self.calls.append((modality, prompt, len(source)))
        if len(source) > self.limit:
            raise ValueError("Teacher context exceeds 8192 tokens; split the input first")
        generator = torch.Generator().manual_seed(len(source))
        rows = max(1, len(source) // 64)
        return TeacherOutput(
            torch.randn(rows, HIDDEN_WIDTH, generator=generator),
            torch.nn.functional.normalize(torch.ones(768), dim=0),
            modality,
            PINNED_TEACHER,
        )


def _row(log: str):
    return {
        "messages": [
            {"role": "user", "content": "What failed in the build?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "log", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": log},
            {"role": "assistant", "content": "The linker step failed."},
        ]
    }


def _episode(tmp_path, log, policy=None):
    teacher = CountingTeacher()
    compressor = DocumentCompressor(teacher, TensorAssetStore(tmp_path), policy)
    context = AdapterContext("s", "r", "u", "chat", compressor=compressor)
    return adapt_row(_row(log), "chat", context), teacher


def test_long_tool_results_are_read_compressed_and_split_under_the_cap(tmp_path):
    log = "".join(f"step {i}: compiling module {i % 7}\n" for i in range(800))
    episode, teacher = _episode(tmp_path, log, CompressionPolicy(min_bytes=4096, piece_bytes=8192))
    compressed = [r for r in episode.records if r.media_type == MEDIA_RECORD_TYPE]
    assert compressed and all(r.role == "tool" and not r.policy_mask for r in compressed)
    # Pieces cover the result exactly; one over the teacher's cap was halved.
    spans = [(r.source_span.start, r.source_span.end) for r in compressed]
    assert spans[0][0] == 0 and spans[-1][1] == len(log.encode())
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert any(size > teacher.limit for _, _, size in teacher.calls)
    assert all(prompt == "title: none | text: " for _, prompt, _ in teacher.calls)
    descriptors = [MediaRecord.from_bytes(r.payload) for r in compressed]
    assert all(d.modality == "text" and d.target is None for d in descriptors)
    events = {e.record_id: e for m in episode.meridians for e in m.events}
    assert {events[r.record_id].modality for r in compressed} == {"compressed_text"}
    # Short fields stay bytes.
    short, _ = _episode(tmp_path, "ok\n")
    assert not any(r.media_type == MEDIA_RECORD_TYPE for r in short.records)


@pytest.mark.parametrize("mode", ("bootstrap", "recurrent"))
def test_compressed_documents_train_through_the_media_encoder(tmp_path, mode):
    torch.manual_seed(5)
    log = "error: undefined reference to `main`\n" * 300
    episode, _ = _episode(tmp_path, log, CompressionPolicy(min_bytes=4096))
    model = build_model(ModelConfig.tiny(**MODEL_OVERRIDES), CODEC)
    config = retrieval_train_config(
        tmp_path / "retrieval",
        phase="B0" if mode == "bootstrap" else "P1",
        representation_mode=mode,
        media_root=str(tmp_path),
        generated_capacity=256,
        max_history_tokens=512,
        max_record_bytes=4096,
    )
    windows = list(iter_windows(episode, CODEC, WindowConfig(max_records=16, max_decisions=8)))
    batches = prepare_batches(model, [EpisodeWindow(episode, w) for w in windows], config)
    model.core.set_model_version(0)
    for batch in batches:
        ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
        forward_meridian_batch(
            model,
            batch,
            ledger,
            config,
            torch.device("cpu"),
            0,
            retrieval_aux(config, supervised=False),
        )
        ledger.finish()
    assert sum(b.counts["lexical"] for b in batches) > 0
    if mode == "bootstrap":  # recurrent training updates E on its own branch only
        assert model.core.media_encoder.projection.weight.grad is not None
    assert model.core.central.modality_embedding.weight.grad[4].abs().sum() > 0
