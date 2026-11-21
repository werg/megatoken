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
        import importlib.util
        import json
        import os
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
            "megatoken.training.semantic_budget",
        }
        class NoLegacyWire(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise AssertionError("Legacy wire dependency: " + fullname)
        sys.meta_path.insert(0, NoLegacyWire())

        repository, scripts = map(Path, sys.argv[1:3])
        package_spec = importlib.util.find_spec("megatoken")
        assert package_spec is not None and package_spec.origin is not None
        package_root = Path(package_spec.origin).resolve().parent
        if os.environ.get("MEGATOKEN_REQUIRE_INSTALLED_WHEEL") == "1":
            expected_root = os.environ.get("MEGATOKEN_EXPECT_PACKAGE_ROOT")
            if expected_root:
                assert package_root == Path(expected_root).resolve(), package_root
            else:
                import sysconfig

                installed_roots = {
                    Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib")
                }
                assert any(
                    package_root == installed_root or installed_root in package_root.parents
                    for installed_root in installed_roots
                ), package_root
            assert repository.resolve() not in package_root.parents, package_root
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
