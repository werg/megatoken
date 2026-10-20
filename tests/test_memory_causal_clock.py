"""One causal clock for retrieval availability, independent of inserted meridians.

Query and RAG meridians inserted during training/serving advance the native
positional counter but never the causal clock that sidecars and source
availability are expressed in. Several queries per episode and windows that
start mid-episode must therefore keep validating.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from retrieval_fixture import live_kb_train_config

from megatoken.training.engine import train


def _metrics(config):
    lines = (Path(config.output) / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _generator(rows, name):
    return [row["generator"][name] for row in rows if name in row.get("generator", {})]


@pytest.mark.parametrize("window_records", [16, 2])
def test_second_query_after_rag_admissions_and_mid_episode_windows(tmp_path, window_records):
    config = live_kb_train_config(
        tmp_path, phase="M2", episodes=2, turns=2, max_steps=3,
        max_window_records=window_records, retrieval_growth_every=10,
    )
    result = train(config)
    assert result["global_step"] == 3
    rows = _metrics(config)
    # Each query inserts its own native meridian (and RAG may insert more), so
    # by the second turn the native counter has run ahead of the causal clock.
    # Every annotated query in the trained windows must still validate and train.
    counts = _generator(rows, "retrieval_embedding_count")
    assert counts and all(count > 0 for count in counts)
    if window_records == 16:
        assert max(counts) == 4  # 2 episodes x 2 turns in one window each
