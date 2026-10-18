import argparse
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_data_collection import RepairTeacher, registry, seed

from megatoken.data import PreparedDataset, WindowConfig, iter_windows, make_demo_episodes
from megatoken.data.cli import add_data_commands
from megatoken.data.collection import CollectionConfig, CollectionLedger, Collector
from megatoken.data.schema import canonical_json
from megatoken.data.student import ModelTeacher
from megatoken.environments import FixtureEnvironment
from megatoken.generation.codec import ByteCodec


def invoke(arguments):
    parser = argparse.ArgumentParser()
    add_data_commands(parser.add_subparsers(required=True))
    args = parser.parse_args(list(map(str, arguments)))
    return args.func(args)


def test_cli_demo_prepare_and_idempotent_resume(tmp_path):
    source = tmp_path / "source.jsonl"
    result = invoke(["demo", "--output", source, "--count", 4, "--max-record-bytes", 64])
    assert result["fixture_only"]
    command = [
        "prepare",
        "--demo",
        source,
        "--output",
        tmp_path / "ready",
        "--codec",
        "byte",
        "--max-record-tokens",
        128,
    ]
    prepared = invoke(command)
    assert prepared["census"]["accepted_episodes"] == 4
    assert prepared["census"]["D"] == prepared["census"]["P"] == 0
    assert prepared["fingerprint"] == invoke(command)["fingerprint"]
    with PreparedDataset(tmp_path / "ready", split=None) as dataset:
        assert len(dataset) == 4
        assert len(list(dataset.iter_episodes(rank=1, world_size=2))) == 2


def test_cli_source_pin_does_not_approve_terms(tmp_path):
    source = tmp_path / "external.jsonl"
    source.write_text('{"text":"A locally supplied document."}\n')
    result = invoke(
        [
            "sources",
            "pin-local",
            source,
            "--output",
            tmp_path / "sources.json",
            "--source-id",
            "ours",
            "--adapter",
            "document",
            "--stream",
            "natural_cc",
        ]
    )
    assert result["source"]["revision"].startswith("sha256:")
    assert not result["source"]["terms"]["approved"]
    with pytest.raises(ValueError, match="unapproved"):
        invoke(
            ["prepare", "--sources", tmp_path / "sources.json", "--output", tmp_path / "prepared"]
        )


def task_source(path):
    path.write_text(
        canonical_json(
            {
                "seed_id": "local-task",
                "family_id": "local-family",
                "category": "terminal",
                "prompt": "Change answer.txt to blue, then finish.",
                "environment": {
                    "initial_files": {"answer.txt": "red"},
                    "expected_files": {"answer.txt": "blue"},
                },
            }
        )
        + "\n"
    )


def test_collection_cli_dry_run_binds_sources_before_calls(tmp_path, monkeypatch):
    task_source(tmp_path / "tasks.jsonl")
    (tmp_path / "config.json").write_text(
        json.dumps({"distinct_tasks": 1, "seed_mix": {"terminal": 1.0}})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A dry run must not construct a network teacher")

    monkeypatch.setattr("megatoken.data.cli.OpenAICompatibleTeacher", forbidden)
    command = [
        "collect",
        "--tasks",
        tmp_path / "tasks.jsonl",
        "--output",
        tmp_path / "collection",
        "--teacher-endpoint",
        "http://127.0.0.1:9/v1/chat/completions",
        "--teacher-model",
        "fixture",
        "--config",
        tmp_path / "config.json",
        "--dry-run",
    ]
    result = invoke(command)
    assert result["families_frozen"] and result["usage"]["T"] == 0
    assert invoke(command)["usage"]["attempts"] == 0
    (tmp_path / "tasks.jsonl").write_text((tmp_path / "tasks.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        invoke(command)


def test_collection_cli_uses_shared_executed_collector(tmp_path, monkeypatch):
    task_source(tmp_path / "tasks.jsonl")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "distinct_tasks": 1,
                "seed_mix": {"terminal": 1.0},
                "max_record_tokens": 16,
                "max_record_bytes": 32,
            }
        )
    )
    monkeypatch.setattr(
        "megatoken.data.cli.OpenAICompatibleTeacher", lambda *a, **kw: RepairTeacher()
    )
    result = invoke(
        [
            "collect",
            "--tasks",
            tmp_path / "tasks.jsonl",
            "--output",
            tmp_path / "collection",
            "--teacher-endpoint",
            "http://127.0.0.1:9/v1/chat/completions",
            "--teacher-model",
            "fixture",
            "--config",
            tmp_path / "config.json",
            "--max-new-attempts",
            1,
        ]
    )
    assert result["usage"]["accepted_episodes"] == 1 and result["usage"]["T"] > 0
    prepared = invoke(
        [
            "prepare",
            "--sources",
            result["sources"],
            "--output",
            tmp_path / "prepared",
            "--max-record-tokens",
            16,
        ]
    )
    assert prepared["census"]["accepted_episodes"] == 1
    assert prepared["census"]["executed_transcript_U"] > 0


def test_frozen_student_preserves_native_action_and_does_not_reingest_it(tmp_path):
    task, codec = seed(), ByteCodec.byte_fallback()
    families = registry(tmp_path / "families", [task])
    ledger = CollectionLedger(
        tmp_path / "ledger", CollectionConfig(distinct_tasks=1, seed_mix=(("terminal", 1.0),))
    )

    class Engine:
        manifest = {"model_sha256": "a" * 64}
        config = SimpleNamespace(max_capacity=128)

        def __init__(self):
            self.codec, self.observed, self.draws = codec, [], 0

        def new_session(self):
            return SimpleNamespace()

        def observe(self, session, payload, *, role):
            self.observed.append((role, payload))

        def generate(self, session, **kwargs):
            self.draws += 1
            if self.draws == 1:
                payload = (
                    b'{"name": "write", "arguments": {"path": "answer.txt", "content": "green"}}'
                )
            else:
                assert [role for role, _ in self.observed] == ["user", "environment", "tool"]
                payload = b'{"name":"read","arguments":{"path":"answer.txt"}}'
            tape = SimpleNamespace(committed=True, to_json=lambda: json.dumps({"draw": self.draws}))
            return SimpleNamespace(
                payload=payload, emitted_ids=tuple(codec.encode(payload)), tape=tape
            )

    engine = Engine()
    student = ModelTeacher(_engine=engine)
    episode = Collector(RepairTeacher(), FixtureEnvironment, codec, ledger, families).collect(
        task, student=student, student_snapshot=student.snapshot, student_prefix_steps=2
    )
    assert episode.supervision.verified_success
    sampled = [record for record in episode.records if record.provenance == "student"]
    assert len(sampled) == 2 and all(record.emitted_ids for record in sampled)
    assert sampled[0].payload.startswith(b'{"name": "write"')
    assert len(json.loads(episode.supervision.annotations_json)["native_action_tapes"]) == 2
    assert len(list(ledger.connection.execute("SELECT * FROM requests"))) == 2
    ledger.close()
    families.close()


def test_record_limit_is_soft_at_complete_logical_boundaries():
    episode = next(make_demo_episodes(1, max_record_bytes=16))
    episode = replace(episode, dependencies=())
    windows = list(
        iter_windows(
            episode, ByteCodec.byte_fallback(), WindowConfig(max_window_tokens=4096, max_records=1)
        )
    )
    assert all(window.records[0].record.part_index == 0 for window in windows)
    assert all(
        window.records[-1].record.part_index + 1 == window.records[-1].record.part_count
        for window in windows
    )
    assert sum(len(window.records) for window in windows) == len(episode.records)
