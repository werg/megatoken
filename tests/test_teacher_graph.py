"""Cheap correctness gates for the joint teacher/structural training path."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch
from training_fixture import prepare_dataset, tiny_train_config

from megatoken.data import Episode, Lineage, PhysicalRecord
from megatoken.data.hindsight import HindsightTargets, layout, pool_bytes
from megatoken.data.hindsight_corpus import prepare_corpus
from megatoken.data.knowledge import knowledge_episode
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.relations import local_relations, with_local_relations
from megatoken.data.schema import Supervision, sequential_meridians
from megatoken.generation.codec import ByteCodec
from megatoken.kb import add_items, new_base
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.training.engine import prepare_batches, window_config
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.stream import EpisodeWindow, _WindowPlan
from megatoken.training.transducer import TransducerTrainer


@pytest.fixture(autouse=True)
def small_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def episode():
    records = (
        PhysicalRecord(
            "a", "document", b"The north key is blue.", logical_id="doc", part_index=0, part_count=2
        ),
        PhysicalRecord(
            "b",
            "document",
            b"The blue key opens room seven.",
            logical_id="doc",
            part_index=1,
            part_count=2,
        ),
        PhysicalRecord("q", "user", b"Which room?", turn_index=1),
        PhysicalRecord("answer", "assistant", b"Seven.", turn_index=2, policy_mask=True),
    )
    return sequential_meridians(Episode("joint", records, Lineage("joint", "v1", "one")))


class FrozenWriter:
    identity = "test-frozen-writer"

    def encode(self, text, *, rows):
        offsets = [(i, i + 1) for i in range(len(text))]
        tokens = torch.tensor([ord(c) for c in text], dtype=torch.float32)
        hidden = torch.sin(tokens[:, None] * torch.arange(1, 33)[None] / 100)
        return pool_bytes(text, offsets, hidden, rows)


def prepared(tmp_path, *, references=False):
    ep = episode()
    if references:
        annotations = {
            "knowledge_items": [
                dict(text="A spare reference fact.", source="ref", document="", part=0)
            ]
        }
        ep = replace(ep, supervision=Supervision(annotations_json=json.dumps(annotations)))
    data = prepare_dataset(tmp_path / "source", [ep])
    config = tiny_train_config(
        tmp_path, datasets=[str(data)], max_window_records=8, max_steps=1, navigation_weight=0.05
    )
    output = tmp_path / "teacher"
    manifest = prepare_corpus(config, output, FrozenWriter(), fit_windows=1)
    config = replace(config, hindsight_targets=str(output))
    targets = HindsightTargets(
        output, config=config, codec=ByteCodec.byte_fallback(), value_dim=128
    )
    return config, targets, manifest


def test_packed_targets_fail_closed_on_corruption(tmp_path):
    store = TensorAssetStore(tmp_path)
    value = torch.linspace(-2, 2, 128)
    asset = store.put_hindsight(value, extractor="hindsight:test", source_family="a" * 64)
    decoded = torch.from_numpy(store.load(asset))
    assert asset.relative_path.suffix == ".fp4"
    assert (tmp_path / asset.relative_path).stat().st_size < 128 * 4
    assert torch.isfinite(decoded).all()
    assert float((decoded - value).square().mean()) < 0.03
    path = tmp_path / asset.relative_path
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        store.load(asset)


def test_raw_declared_graph_mounts_only_explicit_references(tmp_path):
    from megatoken.data.knowledge import declared_knowledge, episode_metadata
    from megatoken.training.reading_base import ReadingBase

    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).eval()
    multi = MultiTrackGenerator(model)
    ep = knowledge_episode(
        "refs",
        "Where?",
        "Never a reference.",
        [(b"A points to B.", "", 0, "x"), (b"B is in room 7.", "", 0, "y")],
        Lineage("test", "1", "refs"),
    )
    ann = json.loads(ep.supervision.annotations_json)
    ann["relations"] = [dict(kind="navigation", left="x", right="y")]
    ep = replace(ep, supervision=Supervision(annotations_json=json.dumps(ann)))
    reading = ReadingBase(model.transducer)
    mounted = reading.mount(multi, ep, TensorAssetStore(tmp_path), 0)
    base = reading.segments[ep.episode_id]
    assert base.edge_count == 3 and not base.writable
    assert set(base.sources.values()) == {"refs:x", "refs:y", "refs:x->refs:y"}
    assert all(m.base is not base or m.name.startswith("reading:") for m in mounted)
    items = declared_knowledge(ep, TensorAssetStore(tmp_path), episode_metadata(ep))
    other = new_base(model, "other", identity={"test": "test"})
    with pytest.raises(ValueError, match="dangling"):
        add_items(other, model, multi, items, relations=[("refs:x", "missing")])
    assert other.edge_count == 0


def test_exact_links_and_dangling_validation():
    ep = with_local_relations(episode())
    rows = local_relations(ep)
    assert [(r["left"], r["right"]) for r in rows] == [("a", "b")]
    assert with_local_relations(ep) == ep
    ann = {"relations": [dict(kind="navigation", left="a", right="absent")]}
    with pytest.raises(ValueError, match="dangling"):
        local_relations(replace(ep, supervision=Supervision(annotations_json=json.dumps(ann))))


def test_offset_pooling_uses_utf8_and_empty_rows():
    text = "[D] é中"
    pooled, mask, extent = pool_bytes(
        text, [(0, 4), (4, 5), (5, 6), (0, 0)], torch.arange(16).reshape(4, 4).float(), rows=8
    )
    assert extent == len(text.encode())
    assert mask.sum() == 2
    assert torch.equal(pooled[~mask], torch.zeros_like(pooled[~mask]))


def test_preparation_resume_and_target_alignment(tmp_path):
    config, targets, manifest = prepared(tmp_path)
    original = (targets.root / "manifest.json").read_bytes()
    # Same source settings, no writer work on resume.
    source_config = replace(
        config, datasets=[str(tmp_path / "source" / "data")], hindsight_targets=None
    )
    # Completed preparations resume without changing targets or the training config.
    resumed = prepare_corpus(source_config, targets.root, FrozenWriter(), fit_windows=1)
    assert resumed == json.loads(json.dumps(manifest))
    # The fixture's actual root is retained in preparation identity.
    intent = json.loads((targets.root / "preparation.json").read_text())
    assert intent["identity"]
    key, row = targets._index.execute("SELECT key, row_json FROM windows LIMIT 1").fetchone()
    entry = json.loads(row)
    assert entry["targets"][-1] is None
    from megatoken.data import PreparedCorpus

    with PreparedCorpus(config.datasets) as corpus:
        ep, _ = corpus.fetch(0)
        window = _WindowPlan(ep, ByteCodec.byte_fallback(), window_config(config))[0]
        _, before = layout(ep, window.end, 16)
        values, sink = targets.load(ep, window.start, window.end, before, device="cpu")
        assert values.shape == (sum(before), 128) and sink[-1]
        with pytest.raises(ValueError, match="boundary plan"):
            targets.load(ep, window.start, window.end, [0] * len(before), device="cpu")
        with pytest.raises(ValueError, match="no stored teacher coverage"):
            targets.load(
                replace(ep, episode_id="another"), window.start, window.end, before, device="cpu"
            )
    assert (targets.root / "manifest.json").read_bytes() == original
    assert source_config.hindsight_targets is None and key


def test_combined_backward_checkpoint_resume_and_bank_only_eval(tmp_path):
    config, targets, _ = prepared(tmp_path, references=True)
    from megatoken.data import PreparedCorpus

    codec = ByteCodec.byte_fallback()
    torch.manual_seed(17)
    model = build_model(ModelConfig.tiny(), codec).train()
    trainer = TransducerTrainer(model.transducer, teacher_targets=targets, navigation_weight=0.05)
    with PreparedCorpus(config.datasets) as corpus:
        ep, assets = corpus.fetch(0)
        window = _WindowPlan(ep, codec, window_config(config))[0]
        item = EpisodeWindow(ep, window, assets)
        (batch,) = prepare_batches(model, [item], config)
        ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
        metrics = forward_meridian_batch(
            model, batch, ledger, config, torch.device("cpu"), 0, transducer=trainer
        )
        ledger.finish()
        assert metrics["teacher_target_values"] > 0
        assert trainer.reading.segments and trainer.bank.edge_count == 0
        reference = trainer.reading.segments[ep.episode_id]
        reference_before = reference.decode("value", reference.edge_ids()).clone()
        assert trainer.records[-1].hindsight is None
        assert trainer.records[-1].sink_only and trainer.records[-1].alpha == 0
        assert all(r.writes for r in trainer.records)
        assert metrics["navigation_examples"] == 1
        assert torch.isfinite(metrics["navigation_loss"])
        assert torch.isfinite(metrics["hindsight_distill_loss"])
        for head in (
            model.transducer.query_realizer,
            model.transducer.hindsight_realizer,
            model.transducer.ingest,
        ):
            assert any(
                p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()
            )
        trainer.after_backward(0)
        trainer.barrier(1, evolve=False)
        assert trainer.bank.edge_count > 0
        torch.testing.assert_close(
            reference.decode("value", reference.edge_ids()), reference_before
        )
        snapshot = copy.deepcopy(trainer.state_dict())
        destination = tmp_path / "transducer.pt"
        torch.save(snapshot, destination)
        restored = TransducerTrainer(
            model.transducer, teacher_targets=targets, navigation_weight=0.05
        )
        restored.load_state_dict(torch.load(destination, weights_only=True))
        assert restored.sequences == trainer.sequences
        assert restored.bank.edge_count == trainer.bank.edge_count > 0
        # No teacher coverage for this new identity: evaluation must nevertheless work,
        # because it uses neither stored targets nor the structural auxiliary task.
        heldout = replace(ep, episode_id="fresh-held-out")
        heldout_window = replace(window, episode=heldout)
        (evaluation,) = prepare_batches(
            model, [EpisodeWindow(heldout, heldout_window, assets)], config
        )
        model.eval()
        with torch.no_grad():
            report = forward_meridian_batch(
                model,
                evaluation,
                LossLedger(evaluation.counts, evaluation.counts, config.loss_weights),
                config,
                torch.device("cpu"),
                0,
                transducer=restored,
                differentiate=False,
            )
        assert "teacher_target_values" not in report and "navigation_loss" not in report
        with pytest.raises(ValueError, match="identity changed"):
            TransducerTrainer(model.transducer, navigation_weight=0.05).load_state_dict(snapshot)


def test_navigation_inserts_a_positive_missing_from_the_shortlist(monkeypatch):
    from megatoken.training.structure import StructuralTeaching

    model = build_model(
        ModelConfig.tiny(transducer_neighbors=3, transducer_selected=1, transducer_exploration=0),
        ByteCodec.byte_fallback(),
    )
    bank = model.transducer.bank
    with torch.no_grad():
        bank.modulation.zero_()
    key = torch.eye(bank.geometry.query_dim)
    location = torch.zeros(bank.geometry.location_dim)
    location[0] = 1
    query = key[0].clone().requires_grad_()
    teaching = StructuralTeaching(
        model.transducer,
        {"a": [(0, 1.0)]},
        torch.stack((key[1], key[0], (key[0] + key[2]) / 2**0.5)),
        location[None].expand(3, -1),
        location[None].expand(3, -1),
        negatives=1,
    )
    monkeypatch.setattr(model.transducer, "causal_query", lambda *a, **kw: (None, query))
    teaching.after(None, None, ["a"], effort=1)
    loss, metrics = teaching.terms()
    assert metrics["navigation_shortlist_recall"] == 0
    loss.backward()
    assert query.grad is not None and query.grad.norm() > 0


def test_absent_targets_retrieve_normally_and_reuse_the_frozen_encoding(tmp_path, monkeypatch):
    from megatoken.data import PreparedCorpus
    from megatoken.model.transducer import metadata_payload

    config, previous, _ = prepared(tmp_path)
    ep = sequential_meridians(
        Episode(
            "single",
            (PhysicalRecord("only", "assistant", b"A.", policy_mask=True),),
            Lineage("joint", "v1", "single"),
        )
    )
    data = prepare_dataset(tmp_path / "single", [ep])
    config = replace(config, datasets=[str(data)], hindsight_targets=None)
    output = tmp_path / "single-targets"
    manifest = prepare_corpus(config, output, FrozenWriter(), encoding_from=previous.root)
    assert manifest["encoding_identity"] == previous.manifest["encoding_identity"]
    assert manifest["coverage"]["target_values"] == 0
    targets = HindsightTargets(output, config=config, value_dim=128)
    config = replace(config, hindsight_targets=str(output))
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).train()
    multi = MultiTrackGenerator(model)
    bank = model.transducer.bank
    value = torch.ones(1, bank.geometry.value_dim)
    bank.add_edges(
        departure=model.transducer.metadata_start(multi, metadata_payload({}))[None],
        key=torch.ones(1, bank.geometry.query_dim),
        value=value,
        destination=bank.content_location(value),
        bias=torch.zeros(1),
        write_mass=torch.ones(1),
        step=0,
    )
    trainer = TransducerTrainer(model.transducer, teacher_targets=targets)
    monkeypatch.setattr(
        model.transducer,
        "hindsight_pass",
        lambda *a, **kw: pytest.fail("Absent targets must not train the realizer"),
    )
    with PreparedCorpus(config.datasets) as corpus:
        ep, assets = corpus.fetch(0)
        window = _WindowPlan(ep, model.codec, window_config(config))[0]
        (batch,) = prepare_batches(model, [EpisodeWindow(ep, window, assets)], config)
        ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
        metrics = forward_meridian_batch(
            model, batch, ledger, config, torch.device("cpu"), 0, transducer=trainer
        )
        ledger.finish()
    assert metrics["teacher_target_values"] == 0
    assert "hindsight_distill_loss" not in metrics and "ingest_value_loss" not in metrics
    (record,) = trainer.records
    assert record.sink_only and record.hindsight is None and record.alpha == 0
    assert record.lookup.edge_ids.numel() == 1
    torch.testing.assert_close(record.lookup.value, record.lookup.bank_value)
    assert trainer.after_backward(0)["transducer_writes"] == 0
    report = trainer.barrier(1, evolve=False)
    assert report["transducer_writeouts"] == report["transducer_written_values"] == 0


def test_reading_questions_do_not_write_even_with_teacher_targets(tmp_path):
    config, previous, _ = prepared(tmp_path)
    ep = replace(episode(), stream="base_reading")
    annotations = {
        "knowledge_items": [dict(text="A reference fact.", source="ref", document="", part=0)]
    }
    ep = replace(ep, supervision=Supervision(annotations_json=json.dumps(annotations)))
    data = prepare_dataset(tmp_path / "reading", [ep])
    config = replace(config, datasets=[str(data)], hindsight_targets=None)
    output = tmp_path / "reading-targets"
    prepare_corpus(config, output, FrozenWriter(), encoding_from=previous.root)
    targets = HindsightTargets(output, config=config, value_dim=128)
    config = replace(config, hindsight_targets=str(output))
    model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).train()
    trainer = TransducerTrainer(model.transducer, teacher_targets=targets)
    from megatoken.data import PreparedCorpus

    with PreparedCorpus(config.datasets) as corpus:
        ep, assets = corpus.fetch(0)
        window = _WindowPlan(ep, model.codec, window_config(config))[0]
        (batch,) = prepare_batches(model, [EpisodeWindow(ep, window, assets)], config)
        ledger = LossLedger(batch.counts, batch.counts, config.loss_weights)
        forward_meridian_batch(
            model, batch, ledger, config, torch.device("cpu"), 0, transducer=trainer
        )
        ledger.finish()
    assert trainer.reading.segments
    assert trainer.records and all(not r.writes for r in trainer.records)
    trainer.after_backward(0)
    report = trainer.barrier(1, evolve=False)
    assert report["transducer_writeouts"] == report["transducer_written_values"] == 0


def test_split_tool_calls_link_the_completed_call_to_the_first_result():
    import hashlib

    from megatoken.data.schema import LogicalAction

    records = (
        PhysicalRecord(
            "c0", "assistant", b"{", logical_id="call", part_index=0, part_count=2, action_id="do"
        ),
        PhysicalRecord(
            "c1", "assistant", b"}", logical_id="call", part_index=1, part_count=2, action_id="do"
        ),
        PhysicalRecord(
            "r0", "tool", b"result", logical_id="result", part_index=0, part_count=2, result_of="do"
        ),
        PhysicalRecord(
            "r1", "tool", b" end", logical_id="result", part_index=1, part_count=2, result_of="do"
        ),
    )
    action = LogicalAction(
        "do", "do", "tool", "{}", ("c0", "c1"), hashlib.sha256(b"{}").hexdigest()
    )
    ep = sequential_meridians(
        Episode("tool", records, Lineage("tool", "1", "tool"), actions=(action,))
    )
    rows = local_relations(ep)
    assert [(r["left"], r["right"]) for r in rows if r["kind"] == "tool_result"] == [("c1", "r0")]
    assert [(r["left"], r["right"]) for r in rows if r["kind"] == "next_part"] == [
        ("c0", "c1"),
        ("r0", "r1"),
    ]


def test_plain_document_corpus_items_exclude_questions_and_generated_answers(tmp_path):
    from megatoken.data.knowledge import episode_items, episode_metadata

    ep = episode()
    items = episode_items(ep, TensorAssetStore(tmp_path), episode_metadata(ep))
    assert [i.source for i in items] == ["joint:a", "joint:b"]
    assert all(i.modality == "text" for i in items)


def test_preparation_preserves_validation_without_teacher_targets(tmp_path):
    from megatoken.data import PreparedCorpus

    config, _, _ = prepared(tmp_path)
    fresh = episode()
    records = tuple(replace(r, payload=r.payload.replace(b"blue", b"red")) for r in fresh.records)
    fresh = replace(
        fresh,
        episode_id="heldout",
        records=records,
        lineage=Lineage("joint", "v1", "heldout-family"),
    )
    heldout = prepare_dataset(tmp_path / "heldout", [fresh], heldout=True)
    config = replace(
        config,
        datasets=[str(tmp_path / "source" / "data"), str(heldout)],
        hindsight_targets=None,
        eval_every=1,
    )
    output = tmp_path / "with-validation"
    manifest = prepare_corpus(config, output, FrozenWriter())
    training = type(config).load(output / "train.json")
    with PreparedCorpus(training.datasets, split="train") as corpus:
        assert len(corpus) == 1 and corpus.fetch(0)[0].episode_id == "joint"
    with PreparedCorpus(training.datasets, split="validation") as corpus:
        assert len(corpus) == 1 and corpus.fetch(0)[0].episode_id == "heldout"
    assert manifest["corpus_identity"]
    from megatoken.data.hindsight import window_key

    targets = HindsightTargets(output)
    assert all(
        targets._index.execute(
            "SELECT 1 FROM windows WHERE key=?", (window_key(fresh, w.start, w.end),)
        ).fetchone()
        is None
        for w in _WindowPlan(fresh, ByteCodec.byte_fallback(), window_config(config))
    )
