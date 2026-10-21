from pathlib import Path

import pytest

from megatoken.data.concordance import TASK_FAMILIES
from megatoken.training.config import TrainConfig
from megatoken.training.task_recipe import RECIPE_VERSION, load_task_recipe

RECIPES = sorted(Path("configs/tasks").glob("*.yaml"))
STREAMS = {"text", "audio_identity", "image_identity", "video_identity", *TASK_FAMILIES}


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_shipped_recipes_name_only_prepared_streams(path):
    recipe = load_task_recipe(path)
    assert set(recipe.mixture) <= STREAMS
    assert "text" in recipe.mixture  # every media phase replays native text
    assert len(recipe.fingerprint) == 64


def test_phase_configs_point_at_shipped_recipes():
    for phase in ("media-bridge", "media-joint", "media-tasks"):
        fields = TrainConfig.__dataclass_fields__
        text = Path(f"configs/train/{phase}.example.yaml").read_text()
        assert f"task_recipe: configs/tasks/{phase}.yaml" in text
        assert "task_recipe" in fields and "trainable" in fields
    bridge = Path("configs/train/media-bridge.example.yaml").read_text()
    assert "trainable: media_bridge" in bridge


@pytest.mark.parametrize(
    "body, message",
    [
        ("recipe: other\nmixture: {text: 1.0}\nauxiliary_weight: 1.0\n", "Unsupported"),
        (f"recipe: {RECIPE_VERSION}\nmixture: {{text: 0.5}}\nauxiliary_weight: 1.0\n", "sum"),
        (f"recipe: {RECIPE_VERSION}\nmixture: {{text: 1.0}}\nauxiliary_weight: 0\n", "Auxiliary"),
        (f"recipe: {RECIPE_VERSION}\nmixture: {{text: 1.0}}\n", "exactly"),
        (
            f"recipe: {RECIPE_VERSION}\nmixture: {{text: 1.5, image_identity: -0.5}}\n"
            "auxiliary_weight: 1.0\n",
            "positive",
        ),
    ],
)
def test_invalid_recipes_are_rejected(tmp_path, body, message):
    path = tmp_path / "recipe.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_task_recipe(path)
