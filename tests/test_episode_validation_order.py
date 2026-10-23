"""A malformed episode is rejected with a ValueError, never an internal KeyError."""

from dataclasses import replace

import pytest
from training_fixture import make_episode


def test_meridians_that_skip_a_record_raise_value_error():
    episode = make_episode(0)
    last = episode.meridians[-1]
    assert len(last.events) >= 1
    short = replace(last, events=last.events[:-1]) if len(last.events) > 1 else None
    meridians = (
        (*episode.meridians[:-1], short) if short is not None else episode.meridians[:-1]
    )
    with pytest.raises(ValueError, match="requires a meridian event"):
        replace(episode, meridians=meridians)
