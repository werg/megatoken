import argparse
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_data_collection import RepairTeacher, registry, seed

from megatoken.controls import build_control_payload
from megatoken.data import Episode, PreparedDataset, WindowConfig, iter_windows, make_demo_episodes
from megatoken.data.cli import add_data_commands
from megatoken.data.collection import CollectionConfig, CollectionLedger, Collector
from megatoken.data.schema import canonical_json
from megatoken.data.student import ModelTeacher
from megatoken.environments import FixtureEnvironment
from megatoken.generation.codec import ByteCodec
from megatoken.generation.structure import TAPE_VERSION, ActionTape


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


@pytest.mark.parametrize("invalid_prefix", [False, True])
def test_frozen_student_preserves_native_action_and_does_not_reingest_it(tmp_path, invalid_prefix):
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
            return SimpleNamespace(facet_advances=0)

        def observe_physical(self, session, payload, *, role, token_ids=None):
            assert codec.decode(token_ids) == payload
            self.observed.append((role, payload))

        def generate(self, session, **kwargs):
            assert session.facet_advances == self.draws
            session.facet_advances += 1
            self.draws += 1
            if self.draws == 1:
                self.initial_observations = tuple(self.observed)
            valid_draw = self.draws - int(invalid_prefix)
            if valid_draw == 0:
                payload = b'{"name":"write","arguments":'
            elif valid_draw == 1:
                assert tuple(self.observed) == self.initial_observations
                payload = (
                    b'{"name": "write", "arguments": {"path": "answer.txt", "content": "green"}}'
                )
            else:
                assert tuple(self.observed[:-1]) == self.initial_observations
                assert self.observed[-1][0] == "tool"
                assert all(role != "assistant" for role, _ in self.observed)
                payload = b'{"name":"read","arguments":{"path":"answer.txt"}}'
            tape = ActionTape(
                version=TAPE_VERSION,
                codec_fingerprint=codec.fingerprint,
                spec_fingerprint="fixture",
                facet_shape=(1, 1),
                facets=(float(self.draws),),
                effort=6,
                role_id=2,
                payload=payload,
                emitted_ids=tuple(codec.encode(payload)),
                valid=valid_draw > 0,
                terminal=False,
                behavior_log_prob=0.0,
            )
            return SimpleNamespace(
                payload=payload, emitted_ids=tuple(codec.encode(payload)), tape=tape
            )

    engine = Engine()
    student = ModelTeacher(_engine=engine)
    episode = Collector(RepairTeacher(), FixtureEnvironment, codec, ledger, families).collect(
        task,
        student=student,
        student_snapshot=student.snapshot,
        student_prefix_steps=2 + int(invalid_prefix),
    )
    assert episode.supervision.verified_success
    sampled = [record for record in episode.records if record.provenance == "student"]
    assert len(sampled) == 2 and all(record.emitted_ids for record in sampled)
    assert sampled[0].payload.startswith(b'{"name": "write"')
    tapes = json.loads(episode.supervision.annotations_json)["native_action_tapes"]
    assert len(tapes) == engine.draws == 2 + int(invalid_prefix)
    if invalid_prefix:
        assert tapes[0]["tape"]["valid"] is False
        assert tapes[0]["record_cursor"] == tapes[1]["record_cursor"]
        assert all(record.payload != b'{"name":"write","arguments":' for record in episode.records)
    assert len(list(ledger.connection.execute("SELECT * FROM executions"))) == 4
    assert len(list(ledger.connection.execute("SELECT * FROM requests"))) == 2
    ledger.close()
    families.close()


def test_student_public_controls_and_empty_draw_tapes_survive_collection(tmp_path):
    codec = ByteCodec.byte_fallback()
    private_target = "PRIVATE_VERIFIER_TARGET_DO_NOT_EXPOSE"
    initial_task = seed()
    environment = json.loads(initial_task.environment_json)
    environment["expected_files"]["answer.txt"] = private_target
    task = replace(
        initial_task,
        policy=b"PUBLIC_POLICY: read the current file before choosing an edit.\r\n",
        environment_json=canonical_json(environment),
    )
    expected_control = build_control_payload(seed=task.policy, tools_json=task.tools_json)

    class Engine:
        manifest = {"model_sha256": "b" * 64}
        config = SimpleNamespace(max_capacity=128)

        def __init__(self):
            self.codec, self.observed, self.tapes = codec, [], []

        def new_session(self):
            return SimpleNamespace(facet_advances=0)

        def observe_physical(self, session, payload, *, role, token_ids=None):
            assert codec.decode(token_ids) == payload
            self.observed.append((role, payload))

        def generate(self, session, **kwargs):
            assert kwargs["control_seed"] == task.policy
            assert kwargs["tools_json"] == task.tools_json
            assert private_target not in repr(kwargs)
            control = build_control_payload(
                seed=kwargs["control_seed"], tools_json=kwargs["tools_json"]
            )
            assert control == expected_control
            # A valid empty draw owns no physical record; the next request must
            # use the advanced session without re-observing its existing prefix.
            payload = (
                b"" if not self.tapes else b'{"name":"read","arguments":{"path":"answer.txt"}}'
            )
            if self.tapes:
                assert self.observed == self.initial_observations
            else:
                self.initial_observations = list(self.observed)
            session.facet_advances += 1
            tape = ActionTape(
                version=TAPE_VERSION,
                codec_fingerprint=codec.fingerprint,
                spec_fingerprint=kwargs["spec"].fingerprint,
                facet_shape=(1, 2),
                facets=(float(session.facet_advances), -0.5),
                effort=kwargs["effort"],
                role_id=2,
                payload=payload,
                emitted_ids=tuple(codec.encode(payload)),
                valid=True,
                terminal=False,
                behavior_log_prob=-1.25,
                control_payload=control,
            )
            self.tapes.append(tape)
            return SimpleNamespace(payload=payload, emitted_ids=tape.emitted_ids, tape=tape)

    class RecordingStudent(ModelTeacher):
        def __init__(self, engine):
            super().__init__(_engine=engine)
            self.requests = []

        def respond(self, request):
            self.requests.append(request)
            return super().respond(request)

    engine, teacher = Engine(), RepairTeacher()
    student = RecordingStudent(engine)
    with (
        registry(tmp_path / "families", [task]) as families,
        CollectionLedger(
            tmp_path / "ledger",
            CollectionConfig(distinct_tasks=1, seed_mix=(("terminal", 1.0),)),
        ) as ledger,
    ):
        episode = Collector(teacher, FixtureEnvironment, codec, ledger, families).collect(
            task,
            student=student,
            student_snapshot=student.snapshot,
            student_prefix_steps=2,
        )
        assert episode.control_seed == task.policy and episode.tools_json == task.tools_json
        assert episode.supervision.outcome == "failure"  # The private target stays unmet.
        for request in student.requests + teacher.requests:
            assert request.control_seed == task.policy and request.tools_json == task.tools_json
            assert private_target not in request.messages_json
            assert private_target.encode() not in b"".join(
                record.payload for record in request.records
            )
        assert private_target.encode() not in b"".join(payload for _, payload in engine.observed)
        entries = json.loads(episode.supervision.annotations_json)["native_action_tapes"]
        assert len(entries) == 2 and entries[0]["record_cursor"] == entries[1]["record_cursor"]
        assert [entry["tape"] for entry in entries] == [
            json.loads(tape.to_json()) for tape in engine.tapes
        ]
        assert all(entry["provenance"] == "student" for entry in entries)
        assert len([record for record in episode.records if record.provenance == "student"]) == 1
        assert len(list(ledger.connection.execute("SELECT * FROM executions"))) == 3
        ledger.export(tmp_path / "export")
        exported = Episode.from_dict(json.loads((tmp_path / "export" / "failed.jsonl").read_text()))
        assert exported.to_dict() == episode.to_dict()
        replay_entries = json.loads(exported.supervision.annotations_json)["native_action_tapes"]
        assert all(
            ActionTape.from_json(canonical_json(entry["tape"])).control_payload == expected_control
            for entry in replay_entries
        )


def test_record_limit_splits_semantic_parts_but_keeps_actions_complete():
    episode = next(make_demo_episodes(1, max_record_bytes=16))
    episode = replace(episode, dependencies=())
    windows = list(
        iter_windows(
            episode, ByteCodec.byte_fallback(), WindowConfig(max_window_tokens=4096, max_records=1)
        )
    )
    for window in windows:
        actions = [record.record for record in window.records if record.record.action_id]
        if actions:
            assert actions[0].part_index == 0
            assert actions[-1].part_index + 1 == actions[-1].part_count
        else:
            assert len(window.records) == 1
    assert sum(len(window.records) for window in windows) == len(episode.records)
