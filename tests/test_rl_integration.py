"""Real executed rollouts, complete replay gradients, and exact RL restarts."""

import json
from dataclasses import replace

import pytest
import torch

from megatoken.environments import FixtureEnvironment
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.training.checkpoint import load_checkpoint, save_checkpoint
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.rl import RLConfig, collect_rollout, score_rollout, train_rl

FIXED_TOOLS = [
    {
        "name": "write",
        "parameters": {"type": "object", "const": {"path": "answer.txt", "content": "blue"}},
    },
    {
        "name": "finish",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def fixture_tasks(count=2, *, tools=None):
    return [
        {
            "task_id": f"edit-{index}",
            "tools": FIXED_TOOLS if tools is None else tools,
            "initial_files": {"answer.txt": "red"},
            "expected_files": {"answer.txt": "blue"},
        }
        for index in range(count)
    ]


def assert_equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal_tree(a, b)
    else:
        assert left == right


@pytest.fixture
def initial_checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(901)
    codec = ByteCodec.byte_fallback()
    config = ModelConfig.tiny(max_capacity=64)
    model = build_model(config, codec)
    optimizer = MasterAdamW(model, lr=1e-4)
    # Populate Adam's moments before entering RL. A skipped equal-reward group
    # must preserve those moments and must not apply residual momentum or decay.
    plan = model.prepare_target(target_bytes=[b"initial"], capacity=8)
    step = model.supervised_step(model.core.initial_state(1), plan=plan, prior_probability=1)
    sum(value / max(count, 1) for value, count in step.losses.values()).backward()
    optimizer.step()
    optimizer.zero_grad()
    checkpoint = save_checkpoint(
        tmp_path / "initial",
        "supervised",
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": config.to_dict(),
            "codec_fingerprint": codec.fingerprint,
            "global_step": 1,
            "train_config": {"precision": "fp32", "phase": "D"},
        },
        {"global_step": 1, "phase": "D"},
        pinned=True,
    )
    optimizer.close()
    return checkpoint


def rl_config(checkpoint, output, **overrides):
    values = dict(
        checkpoint=str(checkpoint),
        output=str(output),
        device="cpu",
        task_seeds=2,
        rollouts_per_seed=4,
        max_steps_per_episode=2,
        max_scored_positions=1000,
        max_total_episodes=8,
        capacity=64,
        max_repairs=0,
        seed=17,
    )
    values.update(overrides)
    return RLConfig(**values)


def test_executed_structural_rl_updates_with_zero_free_lexical_positions(
    initial_checkpoint, tmp_path
):
    result = train_rl(rl_config(initial_checkpoint, tmp_path / "rl"), fixture_tasks())
    before, _ = load_checkpoint(initial_checkpoint)
    after, _ = load_checkpoint(result["checkpoint"])
    assert result["P"] == 0  # Public singleton tool arguments contain no lexical choice.
    assert result["episodes"] == 8 and 0 < result["successes"] < result["episodes"]
    assert result["optimizer_updates"] > 0 and result["global_step"] > before["global_step"]
    assert result["I"] == result["collection_I"] + result["replay_I"] == 2 * result["D"]
    assert not torch.equal(
        before["model"]["schema_codes.weight"], after["model"]["schema_codes.weight"]
    )
    assert_equal_tree(
        {k: v for k, v in before["model"].items() if k.startswith("posterior.")},
        {k: v for k, v in after["model"].items() if k.startswith("posterior.")},
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "rl" / "rollouts-rank0.jsonl").read_text().splitlines()
    ]
    assert len(records) == 8
    assert all(record["likelihood"] == "complete_augmented_action_tape" for record in records)
    assert all(step["execution_id"] for record in records for step in record["steps"])
    assert any(record["verified_success"] for record in records)


def test_rl_full_run_matches_resumed_weights_optimizer_rng_and_cursor(initial_checkpoint, tmp_path):
    tasks = fixture_tasks()
    config = rl_config(initial_checkpoint, tmp_path / "full")
    full = train_rl(config, tasks)
    first = train_rl(replace(config, output=str(tmp_path / "split"), max_task_groups=1), tasks)
    resumed = train_rl(
        replace(
            config,
            checkpoint=first["checkpoint"],
            output=str(tmp_path / "split"),
            max_task_groups=1,
        ),
        tasks,
    )
    uninterrupted, _ = load_checkpoint(full["checkpoint"])
    recovered, _ = load_checkpoint(resumed["checkpoint"])
    for key in (
        "model",
        "optimizer",
        "rank_state",
        "rl_counters",
        "global_step",
        "task_fingerprint",
        "next_task_group",
    ):
        assert_equal_tree(uninterrupted[key], recovered[key])
    assert recovered["next_task_group"] == 2
    assert recovered["rl_counters"]["episodes"] == 8
    # A completed resume performs no more sampling and does not overwrite files.
    again = train_rl(
        replace(config, checkpoint=resumed["checkpoint"], output=str(tmp_path / "split")), tasks
    )
    assert again["checkpoint"] == resumed["checkpoint"]


def test_equal_reward_groups_skip_decay_momentum_and_actual_update_budget(
    initial_checkpoint, tmp_path
):
    config = rl_config(initial_checkpoint, tmp_path / "equal", max_updates=1)
    result = train_rl(config, fixture_tasks(tools=[FIXED_TOOLS[1]]))
    before, _ = load_checkpoint(initial_checkpoint)
    after, _ = load_checkpoint(result["checkpoint"])
    assert result["episodes"] == 8 and result["task_seeds"] == 2
    assert result["optimizer_updates"] == 0 and result["global_step"] == before["global_step"]
    assert_equal_tree(before["model"], after["model"])
    assert_equal_tree(before["optimizer"], after["optimizer"])
    records = [
        json.loads(line) for line in (tmp_path / "equal" / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(record["skipped_equal_reward_update"] for record in records)


def test_raw_rollouts_replay_all_variable_length_tokens_with_history_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(231)
    codec = ByteCodec.byte_fallback()
    model = build_model(ModelConfig.tiny(max_capacity=8), codec).eval()
    config = RLConfig(
        checkpoint="unused",
        task_seeds=1,
        rollouts_per_seed=4,
        max_steps_per_episode=2,
        capacity=8,
        max_repairs=0,
        use_tool_schema=False,
        device="cpu",
    )
    rollouts = [
        collect_rollout(model, codec, fixture_tasks(1)[0], FixtureEnvironment, config)
        for _ in range(4)
    ]
    assert len({rollout.scored_positions for rollout in rollouts}) > 1
    assert sum(rollout.scored_positions for rollout in rollouts) > 0
    for rollout in rollouts:
        score = score_rollout(model, rollout)
        torch.testing.assert_close(
            score.detach(), score.new_tensor(rollout.behavior_log_prob), rtol=1e-5, atol=1e-4
        )
        score.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for name, parameter in model.named_parameters()
        if name.startswith("core.frontend.")
    )


def test_rl_replay_gate_rejects_mismatch_before_optimizer_update(
    initial_checkpoint, tmp_path, monkeypatch
):
    import megatoken.training.rl as rl

    original = rl.score_rollout
    monkeypatch.setattr(rl, "score_rollout", lambda model, rollout: original(model, rollout) + 1)
    calls = []
    monkeypatch.setattr(rl.MasterAdamW, "step", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(RuntimeError, match="replay failed before update"):
        train_rl(
            rl_config(initial_checkpoint, tmp_path / "rejected", max_task_groups=1), fixture_tasks()
        )
    assert not calls


def test_rl_resume_rejects_changed_task_content_and_objective(initial_checkpoint, tmp_path):
    config = rl_config(initial_checkpoint, tmp_path / "resume", max_task_groups=1)
    tasks = fixture_tasks()
    first = train_rl(config, tasks)
    resumed = replace(config, checkpoint=first["checkpoint"])
    with pytest.raises(ValueError, match="task content or order"):
        train_rl(resumed, list(reversed(tasks)))
    with pytest.raises(ValueError, match="max_repairs"):
        train_rl(replace(resumed, max_repairs=1), tasks)
