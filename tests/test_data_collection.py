import pytest

from megatoken.data import DedupConfig, DedupStore, encode_record
from megatoken.data.collection import (
    CollectionConfig,
    CollectionLedger,
    CollectionLimit,
    Collector,
    TaskSeed,
    TeacherCall,
    TeacherResponse,
)
from megatoken.data.schema import canonical_json
from megatoken.environments import FIXTURE_TOOLS, FixtureEnvironment
from megatoken.generation.codec import ByteCodec


def seed(name="one"):
    return TaskSeed(
        name,
        f"task-family:{name}",
        "terminal",
        b"Change answer.txt to blue, then finish.",
        canonical_json(
            {
                "prompt": "Change answer.txt to blue, then finish.",
                "initial_files": {"answer.txt": "red"},
                "expected_files": {"answer.txt": "blue"},
            }
        ),
        canonical_json(FIXTURE_TOOLS),
        "fixture-v1",
    )


def config(**overrides):
    return CollectionConfig(
        **{
            "distinct_tasks": 4,
            "attempts_per_task": 2,
            "teacher_output_limit": 1000,
            "max_output_per_request": 256,
            "max_requests_per_attempt": 5,
            "seed_mix": (("terminal", 1.0),),
            **overrides,
        }
    )


class RepairTeacher:
    model = "fixture-teacher-v1"

    def __init__(self, color="blue"):
        self.color, self.requests = color, []

    def respond(self, request):
        self.requests.append(request)
        assert "expected_files" not in request.messages_json
        if len(self.requests) == 1:
            return TeacherResponse(
                content=b"Updating the file.",
                calls=(
                    TeacherCall(
                        "change",
                        "write",
                        canonical_json({"path": "answer.txt", "content": self.color}),
                    ),
                ),
                provider_input_tokens=100,
                provider_output_tokens=30,
                provider_cache_read_tokens=20,
            )
        return TeacherResponse(
            calls=(TeacherCall("submit", "finish", "{}"),),
            provider_input_tokens=150,
            provider_output_tokens=10,
        )


def registry(path, seeds):
    store = DedupStore(path, DedupConfig(train_fraction=1, validation_fraction=0))
    with store.transaction():
        for item in seeds:
            store.register_family((item.family_id, *item.ancestor_ids))
    store.freeze()
    return store


def test_collection_executes_and_verifies_with_private_labels_outside_transcript(tmp_path):
    task = seed()
    codec = ByteCodec.byte_fallback()
    families = registry(tmp_path / "families.sqlite", [task])
    ledger = CollectionLedger(tmp_path / "usage.sqlite", config())
    teacher = RepairTeacher()
    episode = Collector(teacher, FixtureEnvironment, codec, ledger, families).collect(task)
    assert episode.supervision.verified_success
    assert episode.lineage.frozen_split == "train"
    assert all(
        record.result_of and record.execution_id
        for record in episode.records
        if record.role == "tool"
    )
    assert all(
        not record.policy_mask
        for record in episode.records
        if record.role in {"tool", "environment"}
    )
    assert b"expected_files" not in b"".join(record.payload for record in episode.records)
    meters = ledger.meters()
    assert meters["T"] > 0 and meters["teacher_input"] > meters["T"]
    assert meters["provider_input_tokens"] == 250 and meters["provider_cache_read_tokens"] == 20
    assert meters["accepted_episodes"] == 1 and meters["unresolved_output_reservations"] == 0
    assert len(teacher.requests[1].messages_json) > len(teacher.requests[0].messages_json)
    path = ledger.export(tmp_path / "export")
    assert path.exists()
    ledger.close()
    families.close()


def test_failed_collection_still_charges_teacher_output_and_masks_actions(tmp_path):
    task, codec = seed(), ByteCodec.byte_fallback()
    families = registry(tmp_path / "families", [task])
    ledger = CollectionLedger(tmp_path / "usage", config())
    episode = Collector(
        RepairTeacher("wrong"), FixtureEnvironment, codec, ledger, families
    ).collect(task)
    assert episode.supervision.outcome == "failure"
    assert sum(encode_record(record, codec).P for record in episode.records) == 0
    assert ledger.meters()["T"] > 0 and ledger.meters()["accepted_episodes"] == 0
    assert len(episode.actions) == 2
    ledger.close()
    families.close()


def test_student_changed_action_gets_fresh_execution_before_teacher_correction(tmp_path):
    task, codec = seed(), ByteCodec.byte_fallback()
    families = registry(tmp_path / "families", [task])
    ledger = CollectionLedger(tmp_path / "usage", config())
    environment = FixtureEnvironment()

    class Student:
        model = "frozen-student-step-10"

        def respond(self, request):
            return TeacherResponse(
                calls=(
                    TeacherCall(
                        "student-change",
                        "write",
                        canonical_json({"path": "answer.txt", "content": "green"}),
                    ),
                )
            )

    class Teacher(RepairTeacher):
        def respond(self, request):
            if not self.requests:
                assert environment.files["answer.txt"] == "green"
                assert (
                    "student-change" in request.messages_json and "green" in request.messages_json
                )
            return super().respond(request)

    episode = Collector(Teacher(), lambda: environment, codec, ledger, families).collect(
        task, student=Student(), student_snapshot="sha256:frozen", student_prefix_steps=1
    )
    assert episode.supervision.verified_success
    student_records = [record for record in episode.records if record.provenance == "student"]
    assert student_records and not any(record.policy_mask for record in student_records)
    executions = list(ledger.connection.execute("SELECT * FROM executions ORDER BY rowid"))
    assert len(executions) == 3
    assert executions[0]["output_state"] == executions[1]["input_state"]
    assert len({row["execution_id"] for row in executions}) == 3
    assert (
        len(list(ledger.connection.execute("SELECT * FROM requests"))) == 2
    )  # Student draws are not T.
    assert "outcome_credit" not in episode.supervision.annotations_json
    ledger.close()
    families.close()


@pytest.mark.parametrize("success", (False, True))
def test_student_only_uses_its_own_verified_outcome_without_teacher_help(tmp_path, success):
    import json
    from dataclasses import replace

    from native_support import group_tape

    codec = ByteCodec.byte_fallback()
    task = seed()
    environment = json.loads(task.environment_json)
    if success:
        environment["expected_files"] = environment["initial_files"]
    task = replace(task, environment_json=canonical_json(environment))
    call = TeacherCall("done", "finish", "{}")
    tape = replace(group_tape(codec, call.action_bytes), policy_context=())

    class Student:
        model = "frozen-student"

        def respond(self, request):
            assert "expected_files" not in request.messages_json
            return TeacherResponse(calls=(call,), native_tape_json=tape.to_json())

    teacher = RepairTeacher()
    with (
        registry(tmp_path / "families", [task]) as families,
        CollectionLedger(tmp_path / "usage", config()) as ledger,
    ):
        episode = Collector(teacher, FixtureEnvironment, codec, ledger, families).collect(
            task, student=Student(), student_snapshot="sha256:frozen", student_only=True
        )
        assert not teacher.requests
        assert episode.supervision.verified_success == success
        assert episode.stream == ("successful_replay" if success else "student_rollout_failures")
        (entry,) = json.loads(episode.supervision.annotations_json)["native_action_tapes"]
        assert entry["outcome_credit"]["reward"] == float(success)
        assert entry["outcome_credit"]["selected_events"] == 1
        assert not list(ledger.connection.execute("SELECT * FROM requests"))


def test_interrupted_requests_hold_reservations_and_attempt_caps(tmp_path):
    task = seed()
    ledger = CollectionLedger(tmp_path / "ledger", config(teacher_output_limit=100))
    attempt = ledger.begin_attempt(task, split="train")
    ledger.reserve(attempt, "request", output_limit=70, raw_input=200)
    ledger.recover_interrupted(attempt)
    assert ledger.remaining_output() == 30
    again = ledger.begin_attempt(task, split="train")
    with pytest.raises(CollectionLimit):
        ledger.reserve(again, "request2", output_limit=31, raw_input=10)
    ledger.recover_interrupted(again)
    with pytest.raises(CollectionLimit, match="attempt budget"):
        ledger.begin_attempt(task, split="train")
    assert ledger.meters()["T"] == 0
    assert ledger.meters()["unresolved_output_reservations"] == 70
    ledger.close()


def test_over_budget_provider_response_is_charged_without_clipping(tmp_path):
    ledger = CollectionLedger(tmp_path / "ledger", config(teacher_output_limit=100))
    attempt = ledger.begin_attempt(seed(), split="train")
    ledger.reserve(attempt, "req", output_limit=20, raw_input=10)
    with pytest.raises(CollectionLimit, match="exceeded"):
        ledger.settle("req", TeacherResponse(content=b"x" * 25), raw_output=25)
    assert ledger.meters()["T"] == 25
    assert ledger.remaining_output() == 75
    ledger.close()


def test_hidden_output_cannot_be_misreported_as_complete_raw_usage(tmp_path):
    ledger = CollectionLedger(tmp_path / "ledger", config(teacher_output_limit=100))
    attempt = ledger.begin_attempt(seed(), split="train")
    ledger.reserve(attempt, "req", output_limit=70, raw_input=10)
    with pytest.raises(RuntimeError, match="hidden output"):
        ledger.settle(
            "req", TeacherResponse(content=b"known", raw_output_complete=False), raw_output=5
        )
    assert ledger.meters()["T"] == 5
    assert not ledger.meters()["T_complete"]
    assert ledger.meters()["unresolved_output_reservations"] == 65
    assert ledger.remaining_output() == 30
    ledger.settle("req", TeacherResponse(), raw_output=25)
    assert ledger.meters()["T_complete"] and ledger.meters()["T"] == 25
    assert ledger.remaining_output() == 75
    ledger.close()


def test_seed_parser_does_not_infer_public_user_message_from_private_scenario():
    row = {
        "id": "s1",
        "policy": "public tool policy",
        "tools": [],
        "customer_scenario": "SECRET_INTENT",
    }
    task = TaskSeed.from_tool_use_seed(
        row,
        revision="pinned",
        environment_config={"endpoint": "simulator"},
        public_user_message="Hello, I need help.",
    )
    assert b"SECRET_INTENT" not in task.prompt + task.policy
    assert "SECRET_INTENT" in task.environment_json and task.mode == "simulated"


def test_families_must_be_frozen_before_generation(tmp_path):
    codec = ByteCodec.byte_fallback()
    families = DedupStore(tmp_path / "families")
    ledger = CollectionLedger(tmp_path / "ledger", config())
    with pytest.raises(ValueError, match="split before"):
        Collector(RepairTeacher(), FixtureEnvironment, codec, ledger, families)
    families.close()
    ledger.close()


def test_run_pilot_skips_a_quarantined_seed_instead_of_aborting(tmp_path):
    held_out, usable = seed("held-out"), seed("usable")
    store = DedupStore(tmp_path / "families", DedupConfig(train_fraction=1, validation_fraction=0))
    store.quarantine_benchmarks(family_keys=(held_out.family_id,))
    with store.transaction():
        for item in (held_out, usable):
            store.register_family((item.family_id,))
    store.freeze()
    ledger = CollectionLedger(tmp_path / "usage", config())
    teacher = RepairTeacher()
    collector = Collector(teacher, FixtureEnvironment, ByteCodec.byte_fallback(), ledger, store)
    with pytest.raises(ValueError, match="quarantined"):
        collector.collect(held_out)
    episodes = list(collector.run_pilot([held_out, usable]))
    assert [item.supervision.verified_success for item in episodes] == [True]
    meters = ledger.meters()
    # The held-out task never became a task, an attempt, or a teacher request.
    assert meters["distinct_tasks"] == 1 and meters["attempts"] == 1
    ledger.close()
    store.close()
