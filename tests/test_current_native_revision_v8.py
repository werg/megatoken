"""Current native runs retain exact code identity without a frozen release pair."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import megatoken
from scripts.admit_semantic_v8 import checked_revision


class CurrentNativeRevisionTests(unittest.TestCase):
    def test_current_clean_code_is_observed_and_drift_or_shadow_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            module = root / "src/megatoken/__init__.py"
            module.parent.mkdir(parents=True)
            module.write_text("# first implementation\n")
            environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(root), *args], text=True, env=environment
                ).strip()

            git("init", "--quiet")
            git("add", ".")
            git("-c", "user.name=Fixture", "-c", "user.email=test@example.invalid", "commit", "-qm", "first")
            with patch.object(megatoken, "__file__", str(module)):
                first = checked_revision(root)
                self.assertEqual(first, git("rev-parse", "HEAD"))
                module.write_text("# next implementation\n")
                with self.assertRaisesRegex(ValueError, "exact clean consumer"):
                    checked_revision(root)
                git("add", ".")
                git("-c", "user.name=Fixture", "-c", "user.email=test@example.invalid", "commit", "-qm", "next")
                second = checked_revision(root)
                self.assertNotEqual(second, first)
                with self.assertRaisesRegex(ValueError, "exact clean consumer"):
                    checked_revision(root, first)
                shadow = module.parent / "shadow.py"
                shadow.write_text("# untracked importable code\n")
                with self.assertRaisesRegex(ValueError, "untracked importable"):
                    checked_revision(root, second)
                shadow.unlink()
                cache = module.parent / "__pycache__/fixture.cpython-312.pyc"
                cache.parent.mkdir()
                cache.write_bytes(b"ordinary source cache")
                self.assertEqual(checked_revision(root, second), second)


if __name__ == "__main__":
    unittest.main()
