import json
import subprocess
import sys

import pytest

from megatoken.cli import main, parser


def test_data_commands_and_help_do_not_require_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch']=None; from megatoken.cli import main; main(['recipes','--phase','S'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["meter"] == "P"


def test_inspect_counts_real_model_owners(capsys):
    assert main(["inspect", "--model", "tiny"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["totals"]["served"] > 0 and result["totals"]["training_only"] == 0
    assert result["totals"]["all"] == result["totals"]["served"] + result["totals"]["training_only"]


def test_unknown_configuration_has_a_useful_cli_error(tmp_path, capsys):
    config = tmp_path / "invalid.yaml"
    config.write_text("max_stepz: 1\n")
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(config)])
    assert error.value.code == 2 and "max_stepz" in capsys.readouterr().err


def test_launch_interfaces_parse_explicit_modes():
    args = parser().parse_args(
        [
            "train",
            "--config",
            "configs/train/b0.yaml",
            "--no-require-blackwell",
            "--compile",
            "--expert-backend",
            "triton",
            "--max-steps",
            "1",
        ]
    )
    assert not args.require_blackwell and args.compile and args.max_steps == 1
    args = parser().parse_args(
        ["rl", "--tasks", "tasks.jsonl", "--checkpoint", "checkpoint", "--max-task-groups", "1"]
    )
    assert args.max_task_groups == 1
