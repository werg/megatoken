import json
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from prepare_semantic_v8 import _write_source_recheck_artifact


def test_source_recheck_json_preserves_lone_utf16_surrogate(tmp_path):
    document = {"source": "return value", "value": chr(0xD800), "astral": "𝄞"}
    artifact = tmp_path / "artifact.json"

    _write_source_recheck_artifact(artifact, document)

    encoded = artifact.read_bytes()
    assert b'"value":"\\ud800"' in encoded
    assert json.loads(encoded.decode("utf-8")) == document
