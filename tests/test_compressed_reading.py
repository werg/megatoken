"""Compressed reading: long observed text enters as teacher features, never generated."""

import pytest
import torch
from training_fixture import MODEL_OVERRIDES, tiny_train_config

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
from megatoken.training.transducer import TransducerTrainer

CODEC = ByteCodec.byte_fallback()


class CountingTeacher:
    """One feature row per 64 bytes; pieces of at most ``limit`` characters stand in
    for the real teacher's 8192-token pieces."""

    def __init__(self, limit=6000):
        self.limit, self.calls = limit, []

    def text_pieces(self, text, *, prompt=""):
        return [text[i : i + self.limit] for i in range(0, len(text), self.limit)]

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
    episode, teacher = _episode(tmp_path, log, CompressionPolicy(min_bytes=4096))
    compressed = [r for r in episode.records if r.media_type == MEDIA_RECORD_TYPE]
    assert compressed and all(r.role == "tool" and not r.policy_mask for r in compressed)
    # Pieces cover the result exactly, each within the teacher's input limit.
    spans = [(r.source_span.start, r.source_span.end) for r in compressed]
    assert spans[0][0] == 0 and spans[-1][1] == len(log.encode())
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert len(compressed) > 1 and all(size <= teacher.limit for _, _, size in teacher.calls)
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
    config = tiny_train_config(
        tmp_path / "retrieval",
        phase="B0" if mode == "bootstrap" else "P1",
        representation_mode=mode,
        generated_capacity=256,
        max_history_tokens=512,
        max_record_bytes=4096,
    )
    windows = list(iter_windows(episode, CODEC, WindowConfig(max_records=16, max_decisions=8)))
    batches = prepare_batches(
        model, [EpisodeWindow(episode, w, TensorAssetStore(tmp_path)) for w in windows], config
    )
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
            transducer=TransducerTrainer(model.transducer),
        )
        ledger.finish()
    assert sum(b.counts["lexical"] for b in batches) > 0
    if mode == "bootstrap":  # recurrent training updates E on its own branch only
        projection = model.core.media_encoder.projection
        assert projection.selector.codes.grad is not None
        assert projection._bank.weight.grad is not None
    assert model.core.central.modality_embedding.weight.grad[4].abs().sum() > 0


def test_long_document_reading_episodes_answer_from_a_compressed_document(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/prepare_long_document_reading.py"
    spec = importlib.util.spec_from_file_location("prepare_long_document_reading", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compressor = DocumentCompressor(
        CountingTeacher(),
        TensorAssetStore(tmp_path),
        CompressionPolicy(min_bytes=1, roles=frozenset({"document"})),
    )
    task = {
        "id": "q1",
        "question": "Which river?",
        "documents": ["Paris\nThe Seine flows through Paris.", "Rome\nThe Tiber."],
        "answer": "The Seine",
    }
    episode = module.reading_episode(task, compressor)
    assert episode.stream == "long_document_reading"
    roles = [(r.role, r.media_type == MEDIA_RECORD_TYPE, r.policy_mask) for r in episode.records]
    assert roles == [("user", False, False), ("document", True, False), ("assistant", False, True)]
    modalities = [e.modality for m in episode.meridians for e in m.events]
    assert modalities == ["text", "compressed_text", "text"]
