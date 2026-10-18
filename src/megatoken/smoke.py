"""One-command offline exercise of every stage and the served artifact."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from megatoken.io import write_json


def run_smoke(
    output: str | Path, *, device: str = "cpu", precision: str = "fp32"
) -> dict[str, Any]:
    import torch

    from megatoken.data import prepare_sources, write_demo_source
    from megatoken.generation.codec import ByteCodec
    from megatoken.generation.structure import ActionSpec
    from megatoken.inference import InferenceEngine, export_served
    from megatoken.training.config import TrainConfig
    from megatoken.training.engine import train
    from megatoken.training.rl import RLConfig, train_rl

    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("A smoke run is immutable; choose a fresh --output directory")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    source = write_demo_source(root / "source.jsonl", count=32, max_record_bytes=128)
    prepared = prepare_sources([source], root / "data", ByteCodec.byte_fallback())
    prepared.close()
    common = TrainConfig(
        model="tiny",
        codec="byte",
        dataset=str(root / "data"),
        device=device,
        precision=precision,
        microbatch_episodes=2,
        max_steps=1,
        budget=1,
        accumulate_scored=80,
        accumulate_ingested=1000,
        max_accumulation_windows=2,
        generated_capacity=128,
        max_record_bytes=512,
        max_history_tokens=1024,
        checkpoint_every=0,
        eval_every=0,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    phases, checkpoint = {}, None
    for phase in ("B0", "P1", "M1", "M2", "LC", "S", "D"):
        phases[phase] = train(
            replace(common, phase=phase, output=str(root / phase)),
            resume=checkpoint,
            phase_transition=checkpoint is not None,
        )
        checkpoint = phases[phase]["checkpoint"]
    tasks = [
        {
            "task_id": f"smoke-{index}",
            "prompt": "Write blue to answer.txt, then finish.",
            "initial_files": {"answer.txt": "red"},
            "expected_files": {"answer.txt": "blue"},
            "tools": [
                {
                    "name": "write",
                    "parameters": {"const": {"path": "answer.txt", "content": "blue"}},
                },
                {
                    "name": "finish",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            ],
        }
        for index in range(2)
    ]
    write_json(root / "rl-tasks.json", tasks)
    phases["R"] = train_rl(
        RLConfig(
            checkpoint=checkpoint,
            output=str(root / "R"),
            device=device,
            task_seeds=2,
            rollouts_per_seed=4,
            max_steps_per_episode=1,
            capacity=128,
        ),
        tasks,
    )
    exported = export_served(phases["R"]["checkpoint"], root / "served", codec_path="byte")
    engine = InferenceEngine(root / "served", device=device)
    session = engine.new_session()
    engine.observe(session, b"What changed in answer.txt?")
    sample = engine.generate(session, spec=ActionSpec(min_tokens=4, max_tokens=4), seed=17)
    result = {
        "stages": phases,
        "served": str(root / "served"),
        "export": exported,
        "sample_valid": sample.tape.valid,
        "sample_bytes_hex": sample.payload.hex(),
        "note": "Tiny offline integration gate; no model quality or Blackwell throughput claim.",
    }
    write_json(root / "summary.json", result)
    return result
