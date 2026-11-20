"""Admission must distinguish replay from unverified source correspondence."""

import base64
import contextlib
import copy
import gzip
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec
from scripts import admit_semantic_v8


class NativeV8AdmissionTests(unittest.TestCase):
    def test_producer_checkout_is_verified_before_its_tool_can_claim_cleanliness(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / "producer"
            (root / "tools").mkdir(parents=True)
            marker = base / "executed"
            script = root / "tools/recheck-semantic-native-source.mjs"
            script.write_text(
                "import fs from 'node:fs';fs.writeFileSync("
                + json.dumps(str(marker))
                + ", 'ran');console.log('{}');"
            )
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "test producer authority",
                ],
                cwd=root,
                check=True,
            )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            artifact = base / "artifact.json"
            artifact.write_text("{}")
            with self.assertRaisesRegex(ValueError, "exact clean producer"):
                admit_semantic_v8.recheck_source_binding({}, artifact, root, "0" * 40)
            self.assertFalse(marker.exists())
            original = script.read_text()
            script.write_text(original + "\n// modified tool may make false claims\n")
            with self.assertRaisesRegex(ValueError, "exact clean producer"):
                admit_semantic_v8.recheck_source_binding({}, artifact, root, revision)
            self.assertFalse(marker.exists())
            script.write_text(original)
            (root / "untracked-tool.mjs").write_text("console.log('untracked');")
            with self.assertRaisesRegex(ValueError, "exact clean producer"):
                admit_semantic_v8.recheck_source_binding({}, artifact, root, revision)
            self.assertFalse(marker.exists())

    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent / "fixtures" / "semantic-regions-v8.json.gz.b64"
        cls.example = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))[
            "examples"
        ][0]

    def source_document(self, *, protocol=False):
        document = copy.deepcopy(self.example)
        source = {"sidecar": {"compiler": {"extensions": []}}} if protocol else {}
        document["sidecar"] = {
            "schema": "semantic-protocol-region-evidence/1"
            if protocol
            else "semantic-region-evidence/1",
            "representation": "semantic-trace/8",
            "profile": "compositional-regions/1",
            "admission": "locally_verified",
            "referenceStatus": "agreed",
            "optimizerSteps": 0,
            "learningClaim": None,
            "sourceArtifact": source,
            "sourceArtifactDigest": stable_algorithm_hash(source),
            "publicDigest": stable_algorithm_hash(document["public"]),
            "contextDigest": stable_algorithm_hash(document["contexts"]),
        }
        if protocol:
            document["sidecar"].update(
                protocolProfile="symbols-ordinary-properties/1", protocolExtensions=[]
            )
        return document

    def run_cli(self, document, *, producer_root=None, producer_revision=None):
        # This codec exercises real physical framing and masks; patching the
        # loader isolates the source-admission gate from a large tokenizer file.
        codec = ByteCodec.byte_fallback()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.json"
            output = root / "receipt.json"
            tokenizer = root / "tokenizer.json"
            source.write_text(json.dumps(document))
            tokenizer.write_text("{}")
            arguments = [
                "admit_semantic_v8.py",
                "--input",
                str(source),
                "--output",
                str(output),
                "--consumer-revision",
                "test-revision",
                "--tokenizer-json",
                str(tokenizer),
            ]
            if producer_root is not None:
                arguments.extend(
                    [
                        "--producer-root",
                        str(producer_root),
                        "--producer-revision",
                        producer_revision,
                    ]
                )
            with (
                patch("sys.argv", arguments),
                patch.object(admit_semantic_v8, "checked_revision", return_value="test-revision"),
                patch.object(ByteCodec, "from_tokenizer_json", return_value=codec),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(admit_semantic_v8.main(), 0)
            return json.loads(output.read_text())

    def test_self_asserted_source_evidence_cannot_admit_even_when_other_checks_pass(self):
        for protocol in (False, True):
            with self.subTest(protocol=protocol):
                receipt = self.run_cli(self.source_document(protocol=protocol))
                for name in ("replay", "contexts", "continuation", "masks", "tokenizer"):
                    self.assertEqual(receipt["checks"][name], "passed")
                self.assertEqual(receipt["checks"]["sourceBinding"], "untested")
                self.assertFalse(receipt["admitted"])

    def test_missing_evidence_is_not_admission(self):
        receipt = self.run_cli(self.example)
        self.assertEqual(receipt["checks"]["sourceBinding"], "untested")
        self.assertFalse(receipt["admitted"])

    def test_false_evidence_digests_are_still_rejected(self):
        for field in ("sourceArtifactDigest", "publicDigest", "contextDigest"):
            with self.subTest(field=field):
                document = self.source_document()
                document["sidecar"][field] = "0" * 64
                with self.assertRaisesRegex(ValueError, "no longer binds"):
                    self.run_cli(document)

    def test_protocol_extensions_must_match_the_recorded_inventory(self):
        document = self.source_document(protocol=True)
        document["sidecar"].update(
            protocolProfile="protocol-source-extensions/1",
            protocolExtensions=["owned-binary-memory/1"],
        )
        with self.assertRaisesRegex(ValueError, "mismatched extensions"):
            self.run_cli(document)

    def test_integrated_protocol_source_profiles_are_recognized(self):
        profiles = [
            "owned-binary-memory/1",
            "builtin-iterator/1",
            "class-private-source/1",
            "synchronous-generator-source/1",
            "promise-join/1",
            "number-conversion-source/1",
            "ordinary-data-proxy-source/1",
            "promise-executor/1",
            "ordinary-array-map-source/1",
        ]
        document = self.source_document(protocol=True)
        document["sidecar"].update(
            protocolProfile="protocol-source-extensions/1", protocolExtensions=profiles
        )
        document["sidecar"]["sourceArtifact"]["sidecar"]["compiler"]["extensions"] = profiles
        document["sidecar"]["sourceArtifactDigest"] = stable_algorithm_hash(
            document["sidecar"]["sourceArtifact"]
        )
        receipt = self.run_cli(document)
        self.assertEqual(receipt["checks"]["sourceBinding"], "untested")
        self.assertFalse(receipt["admitted"])

    def test_live_producer_rejects_internally_consistent_but_false_source_evidence(self):
        producer_root = os.environ.get("MEGATOKEN_NATIVE_SOURCE_PRODUCER")
        if not producer_root:
            self.skipTest("set MEGATOKEN_NATIVE_SOURCE_PRODUCER to run the live V8 authority test")
        root = Path(producer_root)
        revision = (
            __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
            .strip()
        )
        for mutation in ("source", "public", "contexts"):
            with self.subTest(mutation=mutation):
                document = self.source_document()
                if mutation == "source":
                    document["sidecar"]["sourceArtifact"]["invented"] = "internally consistent"
                    document["sidecar"]["sourceArtifactDigest"] = stable_algorithm_hash(
                        document["sidecar"]["sourceArtifact"]
                    )
                elif mutation == "public":
                    document["public"]["timeline"][0]["fields"]["invented"] = True
                    document["sidecar"]["publicDigest"] = stable_algorithm_hash(document["public"])
                else:
                    document["contexts"][0]["visibleThrough"] += 1
                    document["sidecar"]["contextDigest"] = stable_algorithm_hash(
                        document["contexts"]
                    )
                with tempfile.TemporaryDirectory() as folder:
                    artifact = Path(folder) / "artifact.json"
                    artifact.write_text(json.dumps(document))
                    with self.assertRaisesRegex(ValueError, "Pinned producer rejected"):
                        admit_semantic_v8.recheck_source_binding(document, artifact, root, revision)


if __name__ == "__main__":
    unittest.main()
