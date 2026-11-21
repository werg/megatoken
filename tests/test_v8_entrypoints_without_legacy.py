"""Fresh-process import boundary for V8 preparation, training, export and JSONL serving."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_v8_lifecycle_entrypoints_import_without_legacy_wire_modules():
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    script = textwrap.dedent(
        r"""
        import importlib.abc
        import json
        import runpy
        import sys
        from pathlib import Path

        blocked = {
            "megatoken.data.algorithm_v7",
            "megatoken.data.detailed_v7",
            "megatoken.data.detailed_harness_v7",
            "megatoken.data.detailed_context_v7",
            "megatoken.data.semantic_harness_v6",
            "megatoken.data.semantic_contract",
            "megatoken.data.semantic_contract_v7",
        }
        class NoLegacyWire(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise AssertionError("Legacy wire dependency: " + fullname)
        sys.meta_path.insert(0, NoLegacyWire())

        repository, scripts = map(Path, sys.argv[1:3])
        sys.path.insert(0, str(scripts))
        for entrypoint in (
            "prepare_semantic_v8.py",
            "fit_semantic_v8.py",
            "serve_semantic_v8.py",
        ):
            runpy.run_path(str(scripts / entrypoint), run_name="v8_import_check")
        import megatoken.loading
        import megatoken.semantic_lineage_v8
        import megatoken.served
        import megatoken.training.diloco_export
        import megatoken.training.engine
        import megatoken.training.readiness
        import megatoken.training.semantic_v8
        assert not blocked.intersection(sys.modules)
        print(json.dumps({"entrypoints": 3, "legacyModulesLoaded": 0}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), str(scripts)],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"entrypoints": 3, "legacyModulesLoaded": 0}
