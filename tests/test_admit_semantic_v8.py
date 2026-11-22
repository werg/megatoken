"""Admission must distinguish replay from unverified source correspondence."""

import base64
import contextlib
import copy
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from megatoken.data.algorithm_v7 import stable_algorithm_hash, stable_algorithm_json
from megatoken.generation.codec import ByteCodec
from scripts import admit_semantic_v8
from tests.test_source_identity_v8 import direct_source_sidecar


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
            git_env = admit_semantic_v8._git_environment()
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True, env=git_env)
            subprocess.run(["git", "add", "."], cwd=root, check=True, env=git_env)
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
                env=git_env,
            )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, env=git_env
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

    def test_producer_git_checks_ignore_callers_consumer_repository_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / "producer"
            (root / "tools").mkdir(parents=True)
            (root / "tools/recheck-semantic-native-source.mjs").write_text("console.log('{}');")
            git_env = admit_semantic_v8._git_environment()
            subprocess.run(["git", "init", "--quiet"], cwd=root, check=True, env=git_env)
            subprocess.run(["git", "add", "."], cwd=root, check=True, env=git_env)
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
                env=git_env,
            )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, env=git_env
            ).strip()
            wrong_git_dir = base / "consumer.git"
            wrong_git_dir.mkdir()
            artifact = base / "artifact.json"
            artifact.write_text("{}")
            with patch.dict(os.environ, {"GIT_DIR": str(wrong_git_dir)}, clear=False):
                with self.assertRaisesRegex(ValueError, "exact clean producer revision"):
                    admit_semantic_v8.recheck_source_binding({}, artifact, root, "0" * 40)
                # A correctly pinned producer must reach the tool; the stub's
                # incomplete receipt is rejected only after that source check.
                with self.assertRaisesRegex(ValueError, "does not bind this artifact"):
                    admit_semantic_v8.recheck_source_binding({}, artifact, root, revision)

    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent / "fixtures" / "semantic-regions-v8.json.gz.b64"
        cls.example = json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))[
            "examples"
        ][0]
        conversion_fixture = (
            Path(__file__).parent / "fixtures" / "semantic-primitive-conversion-v8.json.gz.b64"
        )
        conversion_cases = json.loads(
            gzip.decompress(base64.b64decode(conversion_fixture.read_bytes()))
        )["cases"]
        cls.conversion_examples = {
            case["id"]: case["artifact"]
            for case in conversion_cases
            if case["id"] in ("getter-replacement", "nested-goals", "promise-key", "mapped-key")
        }
        assert len(cls.conversion_examples) == 4

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

    def test_direct_host_identity_without_live_source_replay_is_not_admission(self):
        document = self.source_document()
        side = document["sidecar"]
        del side["sourceArtifact"], side["sourceArtifactDigest"]
        side.update(
            direct_source_sidecar(), representation="mixed", protocolExtensions=["promise-join/1"]
        )
        receipt = self.run_cli(document)
        self.assertIsNone(receipt["sourceArtifactDigest"])
        self.assertEqual(receipt["sourceIdentityDigest"], side["identityDigest"])
        self.assertEqual(receipt["checks"]["sourceBinding"], "untested")
        self.assertFalse(receipt["admitted"])
        for mutation in ("identity", "inventory"):
            changed = copy.deepcopy(document)
            if mutation == "identity":
                changed["sidecar"]["identityDigest"] = "0" * 64
            else:
                changed["sidecar"]["protocolExtensions"] = ["invented"]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "direct"):
                self.run_cli(changed)

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
            "promise-adoption/1",
            "number-conversion-source/1",
            "primitive-conversion-source/1",
            "ordinary-completions-source/1",
            "custom-iterator-source/1",
            "ordinary-data-proxy-source/1",
            "promise-executor/1",
            "ordinary-array-map-source/1",
            "ordinary-for-in/1",
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

    def test_real_conversion_source_artifacts_reach_cli_checks_without_claiming_admission(self):
        # Exercise the production CLI boundary with the actual generated source
        # artifacts, including mixed profiles. Direct native replay alone does
        # not check this inventory registration. The byte fallback and absent
        # live producer deliberately leave complete admission unproven here.
        for identity, document in self.conversion_examples.items():
            with self.subTest(identity=identity):
                self.assertIn(
                    "primitive-conversion-source/1", document["sidecar"]["protocolExtensions"]
                )
                if identity == "promise-key":
                    self.assertIn(
                        "promise-adoption/1", document["sidecar"]["protocolExtensions"]
                    )
                receipt = self.run_cli(document)
                self.assertEqual(
                    receipt["protocolExtensions"], document["sidecar"]["protocolExtensions"]
                )
                for name in ("replay", "contexts", "continuation", "masks", "tokenizer"):
                    self.assertEqual(receipt["checks"][name], "passed")
                self.assertGreater(receipt["counts"]["protocolRecords"], 0)
                self.assertGreater(receipt["counts"]["generatedTokens"], 0)
                self.assertEqual(receipt["counts"]["contexts"], len(document["contexts"]))
                self.assertEqual(receipt["checks"]["sourceBinding"], "untested")
                self.assertFalse(receipt["admitted"])

    def test_real_conversion_source_inventory_rejects_unknown_duplicate_and_mismatched_profiles(
        self,
    ):
        for mutation in ("unknown", "duplicate", "mismatch"):
            with self.subTest(mutation=mutation):
                document = copy.deepcopy(self.conversion_examples["promise-key"])
                sidecar = document["sidecar"]
                extensions = sidecar["protocolExtensions"]
                source = sidecar["sourceArtifact"]
                if mutation == "unknown":
                    extensions.append("unreviewed-conversion-source/1")
                elif mutation == "duplicate":
                    extensions.append("primitive-conversion-source/1")
                if mutation != "mismatch":
                    source["sidecar"]["compiler"]["extensions"] = list(extensions)
                else:
                    source["sidecar"]["compiler"]["extensions"] = extensions[:-1]
                sidecar["sourceArtifactDigest"] = stable_algorithm_hash(source)
                with self.assertRaisesRegex(ValueError, "unsupported or mismatched extensions"):
                    self.run_cli(document)

    def test_live_conversion_admission_uses_exact_producer_and_real_tokenizer_in_cli_subprocess(
        self,
    ):
        producer_root = os.environ.get("MEGATOKEN_NATIVE_SOURCE_PRODUCER")
        tokenizer = os.environ.get("MEGATOKEN_TOKENIZER_JSON")
        if not producer_root or not tokenizer:
            self.skipTest("set the live producer and pinned tokenizer to check full CLI admission")
        consumer = Path(__file__).resolve().parents[1]
        producer = Path(producer_root).resolve()
        git_env = admit_semantic_v8._git_environment()

        def revision(root):
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, env=git_env
            ).strip()

        consumer_revision, producer_revision = revision(consumer), revision(producer)
        with tempfile.TemporaryDirectory() as folder:
            for identity, document in self.conversion_examples.items():
                with self.subTest(identity=identity):
                    source = Path(folder) / f"{identity}.json"
                    output = Path(folder) / f"{identity}.receipt.json"
                    source.write_text(json.dumps(document))
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(consumer / "scripts/admit_semantic_v8.py"),
                            "--input",
                            str(source),
                            "--output",
                            str(output),
                            "--consumer-revision",
                            consumer_revision,
                            "--producer-root",
                            str(producer),
                            "--producer-revision",
                            producer_revision,
                            "--tokenizer-json",
                            str(Path(tokenizer).resolve()),
                        ],
                        cwd=consumer,
                        env=git_env,
                        text=True,
                        capture_output=True,
                        timeout=180,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    receipt = json.loads(output.read_text())
                    self.assertTrue(receipt["admitted"])
                    self.assertTrue(all(value == "passed" for value in receipt["checks"].values()))
                    self.assertEqual(receipt["consumerRevision"], consumer_revision)
                    self.assertEqual(
                        receipt["sourceVerification"]["producerRevision"], producer_revision
                    )
                    self.assertTrue(receipt["sourceVerification"]["sourceValidated"])
                    self.assertGreater(receipt["counts"]["generatedTokens"], 0)
                    self.assertEqual(receipt["counts"]["contexts"], len(document["contexts"]))

    def test_composed_protocol_profile_uses_exact_inventory_and_still_requires_live_source(self):
        extensions = [
            "number-conversion-source/1",
            "ordinary-data-proxy-source/1",
            "owned-binary-memory/1",
            "promise-executor/1",
            "promise-join/1",
        ]
        document = self.source_document(protocol=True)
        sidecar = document["sidecar"]
        sidecar.update(protocolProfile="composed-protocol-source/1", protocolExtensions=extensions)
        sidecar["sourceArtifact"]["sidecar"]["compiler"]["extensions"] = extensions
        sidecar["sourceArtifactDigest"] = stable_algorithm_hash(sidecar["sourceArtifact"])

        # The profile makes the compiler inventory structurally acceptable,
        # but locally asserted evidence cannot pass the producer source check.
        receipt = self.run_cli(document)
        self.assertEqual(receipt["protocolProfile"], "composed-protocol-source/1")
        self.assertEqual(receipt["protocolExtensions"], extensions)
        self.assertEqual(receipt["checks"]["sourceBinding"], "untested")
        self.assertFalse(receipt["admitted"])

        invalid_inventories = (
            [],
            ["number-conversion-source/1", "invented-extension/1"],
            ["promise-join/1", "promise-join/1"],
        )
        for invalid in invalid_inventories:
            with self.subTest(extensions=invalid):
                changed = copy.deepcopy(document)
                changed["sidecar"]["protocolExtensions"] = invalid
                changed["sidecar"]["sourceArtifact"]["sidecar"]["compiler"]["extensions"] = invalid
                changed["sidecar"]["sourceArtifactDigest"] = stable_algorithm_hash(
                    changed["sidecar"]["sourceArtifact"]
                )
                with self.assertRaisesRegex(ValueError, "unsupported or mismatched extensions"):
                    self.run_cli(changed)

        changed = copy.deepcopy(document)
        changed["sidecar"]["protocolProfile"] = "unreviewed-composed-profile/1"
        with self.assertRaisesRegex(ValueError, "unsupported protocol profile"):
            self.run_cli(changed)

        changed = copy.deepcopy(document)
        changed["sidecar"]["sourceArtifact"]["sidecar"]["compiler"]["extensions"] = extensions[:-1]
        changed["sidecar"]["sourceArtifactDigest"] = stable_algorithm_hash(
            changed["sidecar"]["sourceArtifact"]
        )
        with self.assertRaisesRegex(ValueError, "unsupported or mismatched extensions"):
            self.run_cli(changed)

    def test_specialized_promise_profiles_require_their_exact_inventory_and_live_source(self):
        for profile, extensions in (
            ("promise-joins-source/1", ["promise-join/1"]),
            ("promise-executors-source/1", ["promise-executor/1"]),
            ("promise-executors-source/1", ["promise-executor/1", "promise-join/1"]),
        ):
            with self.subTest(profile=profile, extensions=extensions):
                document = self.source_document(protocol=True)
                sidecar = document["sidecar"]
                sidecar.update(protocolProfile=profile, protocolExtensions=extensions)
                source = sidecar["sourceArtifact"]
                source["sidecar"]["compiler"]["extensions"] = extensions
                sidecar["sourceArtifactDigest"] = stable_algorithm_hash(source)
                receipt = self.run_cli(document)
                self.assertEqual(receipt["protocolExtensions"], extensions)
                self.assertEqual(receipt["checks"]["sourceBinding"], "untested")
                self.assertFalse(receipt["admitted"])

                source["sidecar"]["compiler"]["extensions"] = []
                sidecar["sourceArtifactDigest"] = stable_algorithm_hash(source)
                with self.assertRaisesRegex(ValueError, "mismatched extensions"):
                    self.run_cli(document)

        for profile, extensions in (
            ("promise-joins-source/1", ["promise-executor/1"]),
            ("promise-joins-source/1", ["promise-join/1", "owned-binary-memory/1"]),
            ("promise-executors-source/1", ["promise-join/1"]),
            ("promise-executors-source/1", ["promise-join/1", "promise-executor/1"]),
        ):
            with self.subTest(profile=profile, invalid_extensions=extensions):
                document = self.source_document(protocol=True)
                sidecar = document["sidecar"]
                sidecar.update(protocolProfile=profile, protocolExtensions=extensions)
                source = sidecar["sourceArtifact"]
                source["sidecar"]["compiler"]["extensions"] = extensions
                sidecar["sourceArtifactDigest"] = stable_algorithm_hash(source)
                with self.assertRaisesRegex(ValueError, "invalid inventory"):
                    self.run_cli(document)

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

    def test_live_source_checks_the_body_return_after_normal_yield_delegation(self):
        producer_root = os.environ.get("MEGATOKEN_NATIVE_SOURCE_PRODUCER")
        if not producer_root:
            self.skipTest("set MEGATOKEN_NATIVE_SOURCE_PRODUCER to run the live V8 authority test")
        root = Path(producer_root)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            env=admit_semantic_v8._git_environment(),
        ).strip()
        cases = json.loads((root / "test/fixtures/semantic-protocol-region-v8.json").read_text())[
            "cases"
        ]
        document = copy.deepcopy(
            next(
                case["artifact"]
                for case in cases
                if case["id"] == "source-generator-delegated-cleanup"
            )
        )
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "artifact.json"
            artifact.write_text(json.dumps(document))
            receipt = admit_semantic_v8.recheck_source_binding(document, artifact, root, revision)
            self.assertTrue(receipt["sourceValidated"])
            # Normal yield* exposes its checked result to the remaining source;
            # it does not force an arbitrary later return to equal that result.
            # The live source authority must reject a forged enclosing return,
            # even after its enclosing public digest is recomputed consistently.
            ended = False
            for row in document["public"]["timeline"]:
                line = row["fields"].get("line", "")
                if not line.startswith("protocol "):
                    continue
                record = json.loads(line[len("protocol ") :])
                if record["kind"] == "generator.delegate.end":
                    ended = True
                elif ended and record["kind"] == "generator.finish":
                    record["completion"]["value"] = ["num", 99]
                    row["fields"]["line"] = "protocol " + stable_algorithm_json(record)
                    break
            else:
                self.fail("source golden has no enclosing completion after yield delegation")
            document["sidecar"]["publicDigest"] = stable_algorithm_hash(document["public"])
            artifact.write_text(json.dumps(document))
            completed = subprocess.run(
                [
                    "node",
                    str(root / "tools" / "recheck-semantic-native-source.mjs"),
                    "--producer-root",
                    str(root),
                    "--producer-revision",
                    revision,
                    "--input",
                    str(artifact),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=admit_semantic_v8._git_environment(),
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "public differs from the source-certified protocol projection", completed.stderr
            )


if __name__ == "__main__":
    unittest.main()
