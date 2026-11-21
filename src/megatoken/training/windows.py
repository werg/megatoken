"""Model-independent optimizer window boundaries shared by training and preflight."""

from megatoken.data import WindowConfig


def window_config(config):
    return WindowConfig(
        max_window_tokens=config.max_history_tokens,
        max_decisions=config.scored_decisions,
        max_records=config.max_window_records,
        respect_dependencies=config.phase == "LC",
        max_dependency_tokens=config.max_history_tokens,
    )
