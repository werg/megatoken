"""Exercise native V8 replay and recovery with legacy wire modules unavailable."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_full_native_fixture_replay_and_recovery_without_legacy_wire_imports():
    script = textwrap.dedent(
        r"""
        import base64
        import copy
        import gzip
        import importlib.abc
        import json
        from pathlib import Path
        import sys

        blocked = {
            "megatoken.data.algorithm_v7",
            "megatoken.data.detailed_v7",
            "megatoken.data.detailed_harness_v7",
            "megatoken.data.detailed_context_v7",
            "megatoken.data.semantic_harness_v6",
            "megatoken.data.semantic_contract",
            "megatoken.data.semantic_contract_v7",
            "megatoken.data.witness_v6",
        }
        class NoLegacyWire(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise AssertionError("Legacy wire dependency: " + fullname)
        sys.meta_path.insert(0, NoLegacyWire())
        from megatoken.data.native_v8 import (
            NativeV8Error, _native_checkpoint, _restore_native_checkpoint,
            prepare_native_v8_training, validate_v8_artifact,
        )
        from megatoken.data.prefix_v8 import validate_v8_prediction_prefix
        from megatoken.data.semantic_json import stable_algorithm_hash
        from megatoken.generation.codec import ByteCodec

        directory = Path(sys.argv[1])
        checked = targets = source_refused = 0
        for filename in (
            "semantic-regions-v8.json.gz.b64",
            "semantic-protocol-region-v8.json.gz.b64",
            "semantic-callback-regions-v8.json.gz.b64",
            "semantic-callback-return-v8.json.gz.b64",
            "semantic-host-protocol-v8.json.gz.b64",
            "semantic-host-recovery-v8.json.gz.b64",
            "semantic-primitive-conversion-v8.json.gz.b64",
        ):
            fixture = json.loads(gzip.decompress(base64.b64decode((directory / filename).read_bytes())))
            examples = [fixture] if "public" in fixture else fixture.get("examples", fixture.get("cases", []))
            assert examples, filename
            for row in examples:
                if row.get("status") == "source-unqualified":
                    assert filename == "semantic-primitive-conversion-v8.json.gz.b64"
                    assert "artifact" not in row and row["boundary"] in {
                        "intrinsic-error-source-metadata", "node24-reference-disagreement"
                    }
                    source_refused += 1
                    continue
                example = row.get("artifact", row)
                result = validate_v8_artifact(example)
                assert result["admitted"] is False
                assert all(result["checks"][field] == "passed"
                           for field in ("replay", "contexts", "continuation"))
                context = example["contexts"][-1]
                public = {**example["public"],
                          "timeline": example["public"]["timeline"][:context["visibleThrough"]]}
                contexts = example["contexts"][:-1]
                prefix = validate_v8_prediction_prefix(public, contexts, context)
                packet = _native_checkpoint(public["timeline"], prefix["context"])
                restored = _restore_native_checkpoint(
                    packet, expected_prefix=public["timeline"],
                    expected_state=context["state"], expected_cursor=context["regionCursor"],
                    expected_obligations=context["obligations"],
                )
                assert restored == context["state"]
                checked += 1
                targets += len(example["contexts"])
                # Engineering codec only: exercise framing without claiming
                # pinned-tokenizer or private source admission.
                if checked == 1:
                    _, training = prepare_native_v8_training(example, codec=ByteCodec.byte_fallback())
                    assert training["maskValidation"]
                forged = copy.deepcopy(example)
                forged["contexts"][-1]["state"]["completion"] = {"kind":"return", "value":["num", 987654]}
                # Rehashing a claimed state cannot replace independent replay.
                forged["contexts"][-1]["prefixDigest"] = stable_algorithm_hash(public["timeline"])
                try:
                    validate_v8_artifact(forged)
                except (NativeV8Error, ValueError):
                    pass
                else:
                    raise AssertionError("Forged continuation accepted: " + str(row.get("id")))
        assert not blocked.intersection(sys.modules)
        print(json.dumps({"examples":checked, "prediction_contexts":targets,
                          "source_refused":source_refused}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).parent / "fixtures")],
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["examples"] >= 30
    assert result["prediction_contexts"] >= result["examples"]
    assert result["source_refused"] == 5
