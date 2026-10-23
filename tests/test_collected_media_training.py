"""Actual collected media receipts survive preparation and joint training.

The decoder and teacher features are test doubles. Storage, collection, native
history validation, export/restore, batching, and backward use production code.
"""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import torch
from native_support import tiny_bundle
from training_fixture import tiny_train_config

from megatoken.data import DedupConfig, DedupStore, PreparedDataset, WindowConfig, iter_windows
from megatoken.data.collection import TeacherRequest
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, TensorAssetStore
from megatoken.data.schema import (
    Episode,
    Lineage,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    canonical_json,
)
from megatoken.data.storage import export_prepared
from megatoken.data.student import ModelTeacher
from megatoken.generation.codec import ByteCodec
from megatoken.model.multitrack import GroupActionTape
from megatoken.training.engine import prepare_batches
from megatoken.training.history import NativeHistory
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow
from megatoken.training.transducer import TransducerTrainer
from tests.media_support import install_fake_realizers, prepared_record


def test_collected_speech_and_caption_export_restore_and_train_together(tmp_path):
    torch.set_num_threads(1)
    codec = ByteCodec.byte_fallback()
    bundle = tiny_bundle(codec)
    model = install_fake_realizers(bundle.model)
    root = tmp_path / "prepared"
    assets = TensorAssetStore(root)

    def prepare_rendered(tensor, spec):
        record = prepared_record(assets, "audio")
        values = tensor.detach().cpu().numpy()[0]
        rendered = assets.put(
            values,
            modality="audio",
            kind="rendered",
            extractor=model.media_realizers.resident("audio").identity,
            source_family=hashlib.sha256(values.tobytes()).hexdigest(),
            sample_rate=spec.sample_rate,
        )
        return replace(
            record,
            source=replace(record.source, source_family=rendered.source_family),
            embedding=replace(record.embedding, source_family=rendered.source_family),
            target=replace(record.target, source_family=rendered.source_family),
            rendered=rendered,
        )

    student = ModelTeacher(
        bundle, effort=1, media_preparer=SimpleNamespace(store=assets, rendered=prepare_rendered)
    )
    prompt = PhysicalRecord("source", "user", b"Say hello with a caption.")
    try:
        response = student.respond(
            TeacherRequest(
                request_id="audio-caption",
                messages_json="[]",
                tools_json="[]",
                max_output_tokens=4,
                codec_fingerprint=codec.fingerprint,
                session_id="collection",
                records=(prompt,),
                outputs=(
                    {"track": "assistant", "modality": "audio", "sample_count": 48},
                    {"track": "assistant", "min_tokens": 1, "max_tokens": 4},
                ),
            )
        )
        tape = GroupActionTape.from_json(response.native_tape_json)
        assert tape.valid == (True, True)
        records = (prompt,) + tuple(
            PhysicalRecord(
                f"output-{index}",
                "assistant",
                event.payload,
                turn_index=1,
                policy_mask=True,
                emitted_ids=event.emitted_ids,
                codec_fingerprint=codec.fingerprint if event.emitted_ids is not None else None,
                media_type=MEDIA_RECORD_TYPE if event.modality != "text" else "text/plain",
            )
            for index, event in enumerate(response.events)
        )
        episode = Episode(
            "collected-media",
            records,
            Lineage("test", "v1", "original-conversation", frozen_split="train"),
            tracks=(TrackSpec("user", "user"), TrackSpec("assistant", "assistant")),
            meridians=(
                Meridian(0, (TrackEvent("source", "user", "text", 0, "user"),)),
                Meridian(
                    1,
                    tuple(
                        TrackEvent(f"output-{i}", "assistant", kind, 1, "assistant")
                        for i, kind in enumerate(tape.modalities)
                    ),
                ),
            ),
            supervision=Supervision(
                annotations_json=canonical_json(
                    {
                        "native_action_tapes": [
                            {
                                "record_cursor": 1,
                                "record_ids": ["output-0", "output-1"],
                                "tape": json.loads(tape.to_json()),
                            }
                        ]
                    }
                )
            ),
        )
        assert NativeHistory.from_episode(episode, model).group_for_records((1, 2)) == tape
        with DedupStore(
            tmp_path / "spool.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
        ) as store:
            store.add(episode, codec)
            store.freeze()
            export_prepared(
                store, root, codec=codec, source_receipts=[], pipeline_fingerprint="test"
            )
        with PreparedDataset(root, split="train") as prepared:
            restored, checked_assets = prepared[0], prepared.assets
            window = next(iter_windows(restored, codec, WindowConfig(max_records=8)))
            config = tiny_train_config(
                tmp_path / "run", phase="P1", representation_mode="recurrent", max_record_bytes=8192
            )
            (batch,) = prepare_batches(
                model, [EpisodeWindow(restored, window, checked_assets)], config
            )
            assert batch.counts["modality"] == 2
            assert batch.counts["audio_prediction"] == 1
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
            assert any(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.core.predictor_parameters()
            )
    finally:
        student.runtime.close()
