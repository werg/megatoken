"""Real executed rollouts, complete replay gradients, and exact RL restarts."""

import json
from dataclasses import replace

import pytest
import torch

from megatoken.controls import build_control_payload, encode_controls
from megatoken.data.chunking import ChunkingConfig
from megatoken.environments import FixtureEnvironment
from megatoken.generation.codec import ByteCodec
from megatoken.generation.structure import ActionSpec
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.tool_schema import compile_tool_spec
from megatoken.training.checkpoint import load_checkpoint, save_checkpoint
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.rl import RLConfig, _ingest, collect_rollout, score_rollout, train_rl

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


@torch.no_grad()
def initialize_balanced_tool_decoder(model, chunking):
    """Make the first tool depend on Gaussian noise around its causal prior mean.

    The real decoder emits either a complete fixed write or a complete finish.
    Its schema decision is a threshold on one mean facet coordinate, centered
    using the same public observation and tools as collection. No reward,
    sampled facet, or production policy is replaced by the fixture.
    """
    model.eval()
    observation = FixtureEnvironment().reset(fixture_tasks(1)[0]).observation
    state = _ingest(model, model.core.initial_state(1), observation, "user", chunking)
    tools_json = json.dumps(FIXED_TOOLS)
    payload = build_control_payload(tools_json=tools_json)
    control = encode_controls(model.core, [payload], encoder_grad=False)
    mean = model.core.predict_facets(state, control=control, effort=6)
    center = mean.mean(1)[0, 0]

    decoder = model.decoder
    width = model.config.facet_dim
    decoder.header_start.weight.zero_()
    decoder.header_start.weight[0, 0] = 1
    decoder.public_projection.weight.zero_()
    decoder.schema_codes.weight.zero_()
    decoder.schema_codes.weight[0, 0] = 1  # Write on positive noise.
    decoder.schema_codes.weight[1, 0] = -1  # Finish on negative noise.
    for parameter in decoder.header_cell.parameters():
        parameter.zero_()
    # GRU gates are (reset, update, candidate): r=1/2, z~=0, so the
    # forced-mode update maps h[0] to tanh(h[0] - center).
    decoder.header_cell.bias_ih[width : 2 * width] = -30
    decoder.header_cell.weight_hh[2 * width, 0] = 2
    decoder.header_cell.bias_hh[2 * width] = -2 * center

    # Validate the designed decision boundary through the complete serializer.
    spec = compile_tool_spec(tools_json, model.config.max_capacity)
    for direction, name in ((1, "write"), (-1, "finish")):
        raw = mean.clone()
        raw[:, :, 0] += direction
        decoded = model.decode_facets(raw, spec=spec)
        assert decoded.valid
        assert json.loads(decoded.payload)["name"] == name
    model.core.routing_aux_loss(clear=True)


@pytest.fixture
def initial_checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(901)
    codec = ByteCodec.byte_fallback()
    config = ModelConfig.tiny(max_capacity=128)
    model = build_model(config, codec)
    optimizer = MasterAdamW(model, lr=1e-4)
    # Populate Adam's moments before entering RL. A skipped equal-reward group
    # must preserve those moments and must not apply residual momentum or decay.
    plan = model.prepare_target(target_bytes=[b"initial"], capacity=8)
    step = model.supervised_step(model.core.initial_state(1), plan=plan)
    sum(value / max(count, 1) for value, count in step.losses.values()).backward()
    optimizer.step()
    optimizer.zero_grad()
    chunking = ChunkingConfig().with_hard_limit(config.max_capacity)
    # FP32 parameters are their own masters; retain the populated Adam moments
    # while setting a balanced, deterministic action map for the executed test.
    initialize_balanced_tool_decoder(model, chunking)
    checkpoint = save_checkpoint(
        tmp_path / "initial",
        "supervised",
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": config.to_dict(),
            "codec_fingerprint": codec.fingerprint,
            "chunking": chunking.identity(codec),
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
        max_latent_coordinates=100_000,
        max_total_episodes=8,
        capacity=128,
        seed=17,
    )
    values.update(overrides)
    return RLConfig(**values)


def test_executed_facet_rl_updates_predictor_and_preserves_fixed_decoder(
    initial_checkpoint, tmp_path
):
    result = train_rl(rl_config(initial_checkpoint, tmp_path / "rl"), fixture_tasks())
    before, _ = load_checkpoint(initial_checkpoint)
    after, _ = load_checkpoint(result["checkpoint"])
    assert result["P"] == 0  # All decoded structure and lexical choices are deterministic.
    assert result["Z"] > 0 and result["emitted_positions"] > 0
    assert result["episodes"] == 8 and 0 < result["successes"] < result["episodes"]
    assert result["optimizer_updates"] > 0 and result["global_step"] > before["global_step"]
    assert result["I"] == result["collection_I"] + result["replay_I"] == 2 * result["D"]
    frozen = after["frozen_conditional_parameters"]
    assert frozen and any("frontend" in name for name in frozen)
    assert any(name.startswith("core.central.") for name in frozen)
    assert not any(name.startswith("core.p_bank.") for name in frozen)
    assert not any(name.startswith("core.control_encoder.") for name in frozen)
    assert any(
        not torch.equal(value, after["model"][name])
        for name, value in before["model"].items()
        if name not in frozen and isinstance(value, torch.Tensor)
    )
    assert_equal_tree(
        {k: before["model"][k] for k in frozen},
        {k: after["model"][k] for k in frozen},
    )
    assert_equal_tree(
        {k: before["optimizer"]["master_weights"][k] for k in frozen},
        {k: after["optimizer"]["master_weights"][k] for k in frozen},
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "rl" / "rollouts-rank0.jsonl").read_text().splitlines()
    ]
    assert len(records) == 8
    assert all(record["likelihood"] == "canonical_facet_gaussian_density" for record in records)
    assert all(
        step["execution_id"]
        for record in records
        for step in record["steps"]
        if step["tape"]["valid"]
    )
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


def test_raw_rollouts_replay_full_facets_with_history_gradients_and_fixed_encoder():
    torch.set_num_threads(1)
    torch.manual_seed(231)
    codec = ByteCodec.byte_fallback()
    model = build_model(ModelConfig.tiny(max_capacity=8), codec).eval()
    model.freeze_decoder_for_rl()
    config = RLConfig(
        checkpoint="unused",
        task_seeds=1,
        rollouts_per_seed=4,
        max_steps_per_episode=2,
        capacity=8,
        use_tool_schema=False,
        device="cpu",
    )
    rollouts = [
        collect_rollout(model, codec, fixture_tasks(1)[0], FixtureEnvironment, config)
        for _ in range(4)
    ]
    assert all(rollout.scored_positions == 0 for rollout in rollouts)
    assert all(rollout.latent_dimensions == len(rollout.steps) * 4 * 32 for rollout in rollouts)
    assert len({rollout.steps[0].tape.facets for rollout in rollouts}) == len(rollouts)
    for rollout in rollouts:
        score = score_rollout(model, rollout)
        torch.testing.assert_close(
            score.detach(), score.new_tensor(rollout.behavior_log_prob), rtol=1e-5, atol=1e-4
        )
        score.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for name, parameter in model.named_parameters()
        if name.startswith("core.stages.") or name.startswith("core.remix.")
    )
    assert all(parameter.grad is None for parameter in model.core.encoder_parameters())
    assert all(parameter.grad is None for parameter in model.core.central_parameters())


def test_latent_update_preserves_fixed_action_context_and_readout_exactly():
    torch.set_num_threads(1)
    torch.manual_seed(611)
    codec = ByteCodec.byte_fallback()
    model = build_model(ModelConfig.tiny(max_capacity=8), codec).eval()
    optimizer = MasterAdamW(model, lr=0.01)
    names = list(model.freeze_decoder_for_rl())
    optimizer.freeze(names)
    chunking = ChunkingConfig().with_hard_limit(model.config.max_capacity)
    observation = b"public causal observation"
    original = _ingest(model, model.core.initial_state(1), observation, "user", chunking)
    spec = ActionSpec(text_kind="ascii", min_tokens=2, max_tokens=4)
    sampled = model.sample(
        original, spec=spec, control_payload=b"choose", generator=torch.Generator().manual_seed(551)
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    scored = model.score_tape(original, sampled.tape, spec=spec)
    (-scored.log_prob).backward()
    optimizer.step()
    try:
        for name in names:
            torch.testing.assert_close(
                before[name], dict(model.named_parameters())[name], atol=0, rtol=0
            )
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.named_parameters()
            if name not in names
        )
        model.core.set_model_version(1)
        rebuilt = _ingest(model, model.core.initial_state(1), observation, "user", chunking)
        replayed = model.score_tape(rebuilt, sampled.tape, spec=spec)
        assert replayed.payload == sampled.payload and replayed.emitted_ids == sampled.emitted_ids
        for field in ("facets", "early", "final"):
            # Compare like-for-like replay execution modes. Inference's
            # no-grad attention path may round derived features differently.
            torch.testing.assert_close(
                getattr(replayed.state, field), getattr(scored.state, field), atol=0, rtol=0
            )
        torch.testing.assert_close(
            replayed.state.history.keys, scored.state.history.keys, atol=0, rtol=0
        )
        torch.testing.assert_close(
            replayed.state.history.values, scored.state.history.values, atol=0, rtol=0
        )
        assert not torch.equal(replayed.log_prob.detach(), scored.log_prob.detach())
    finally:
        optimizer.close()


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
    with pytest.raises(ValueError, match="capacity"):
        train_rl(replace(resumed, capacity=64), tasks)
