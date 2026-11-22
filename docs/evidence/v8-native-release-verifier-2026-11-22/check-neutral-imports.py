"""Fresh-process import guard, not complete source/data/runtime qualification."""

import importlib.abc
import json
import re
import sys

PATTERN = r"(?:semantic[^.]*_v[67]|native_v[67]|algorithm_v7)(?:$|\.)"


class DenyLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if re.search(PATTERN, fullname):
            raise RuntimeError("Legacy semantic dependency reached: " + fullname)


sys.meta_path.insert(0, DenyLegacy())
for module in (
    "scripts.verify_semantic_v8_release",
    "prepare_semantic_v8",
    "megatoken.loading",
    "megatoken.served",
    "megatoken.training.engine",
):
    __import__(module)
reached = [name for name in sys.modules if re.search(PATTERN, name)]
if reached:
    raise RuntimeError("Legacy modules were already installed: " + repr(reached))
print(
    json.dumps(
        {
            "status": "passed",
            "deniedModulePattern": PATTERN,
            "legacyModulesReached": reached,
            "optimizerUpdates": 0,
            "decoderForwardPasses": 0,
            "freshSourceReplay": 0,
            "scope": "Fresh imports of verifier, preparer, checkpoint/export loaders and ordinary trainer only; no full runtime or physical-source-absence qualification",
        }
    )
)
