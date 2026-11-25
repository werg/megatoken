from pathlib import Path
import ast
import base64
import collections
import gzip
import hashlib
import json
import shutil
import subprocess

root = Path(__file__).parent
producer, consumer = root / "producer-dev", root / "consumer"
for repo in (producer, consumer):
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True), repo
revision = subprocess.check_output(["git", "-C", str(producer), "rev-parse", "HEAD"], text=True).strip()
src, dst = producer / "test/fixtures", consumer / "tests/fixtures"
def compressed(path):
    return json.loads(gzip.decompress(base64.b64decode(path.read_bytes())))
canonical = compressed(src / "semantic-protocol-region-v8.json.gz.b64")
previous = compressed(dst / "semantic-protocol-region-v8.json.gz.b64")
baseline_revision = "78a284daf30b3dba8152659cfb43a8a8429ee5d7"
baseline_bytes = subprocess.check_output([
    "git", "-C", str(consumer), "show",
    baseline_revision + ":tests/fixtures/semantic-protocol-region-v8.json.gz.b64",
])
baseline_cases = json.loads(gzip.decompress(base64.b64decode(baseline_bytes)))["cases"]
baseline = {row["id"]: row for row in baseline_cases}
assert len(baseline) == len(baseline_cases) == 93
contract = json.loads((producer / "contracts/semantic-execution-v8.json").read_bytes())
before = {row["id"]: row for row in previous["cases"]}
after = {row["id"]: row for row in canonical["cases"]}
assert len(after) == len(canonical["cases"]) and set(before) <= set(after)
for key in before:
    for field in ("source", "entry", "args", "family", "ancestorIds"):
        previous_fixture, current_fixture = before[key]["fixture"], after[key]["fixture"]
        assert (field in previous_fixture) == (field in current_fixture), (key, field, "presence")
        assert json.dumps(previous_fixture.get(field), ensure_ascii=False, separators=(",",":")) == json.dumps(current_fixture.get(field), ensure_ascii=False, separators=(",",":")), (key, field)
for key, old in baseline.items():
    assert key in after, key
    for field in ("source", "entry", "args", "family", "ancestorIds"):
        prior, current = old["fixture"], after[key]["fixture"]
        assert (field in prior) == (field in current), (key, field, "baseline presence")
        assert json.dumps(prior.get(field), ensure_ascii=False, separators=(",",":")) == json.dumps(current.get(field), ensure_ascii=False, separators=(",",":")), (key, field, "baseline")
assert {"protocol-region:" + key for key in after} == {
    row["id"] for row in contract["interoperability"] if row["id"].startswith("protocol-region:")
}
shutil.copyfile(producer / "contracts/semantic-execution-v8.json",
                consumer / "src/megatoken/data/contracts/semantic-execution-v8.json")
names = (
    "semantic-protocol-v8.json", "semantic-iterator-v8.json", "semantic-generator-v8.json",
    "semantic-protocol-region-v8.json.gz.b64", "semantic-protocol-source-v8.json",
    "semantic-host-protocol-v8.json.gz.b64", "semantic-iterator-region-v8.json",
    "semantic-primitive-conversion-v8.json.gz.b64",
)
for name in names:
    shutil.copyfile(src / name, dst / name)
(dst / "semantic-regions-v8.json.gz.b64").write_bytes(
    base64.b64encode(gzip.compress((src / "semantic-regions-v8.json").read_bytes(), mtime=0)) + b"\n"
)
path = consumer / "tests/test_protocol_v8.py"
text = path.read_text()
point = text.index("        expected = {", text.index(
    "def test_source_qualified_protocol_region_public_golden_replays_and_continues"))
start, end = text.index("{", point), text.index("\n        }", point) + len("\n        }")
expected = ast.literal_eval(text[start:end])
assert set(expected) == set(before)
observed = {}
for key, case in after.items():
    counts = collections.Counter(row["kind"] for row in case["artifact"]["public"]["timeline"]
                                 if row["mode"] == "generated")
    assert not set(counts) - {"protocol", "region", "semantic"}
    observed[key] = tuple(counts[name] for name in ("protocol", "region", "semantic"))
replacement = "{\n" + "".join(f"            {json.dumps(key)}: {value},\n"
                               for key, value in observed.items()) + "        }"
path.write_text(text[:start] + replacement + text[end:])
assert subprocess.check_output(["git", "-C", str(producer), "rev-parse", "HEAD"], text=True).strip() == revision
assert not subprocess.check_output(["git", "-C", str(producer), "status", "--porcelain", "--untracked-files=all"], text=True)
print(json.dumps({
    "producerRevision": revision, "producerTreeClean": True,
    "sourceCases": len(after), "preservedSources": len(baseline),
    "preservedCurrentSources": len(before),
    "preservationBaselineRevision": baseline_revision,
    "preservationBaselineArchiveSha256": hashlib.sha256(baseline_bytes).hexdigest(),
    "newSources": sorted(set(after) - set(before)),
    "changedCounts": {key: {"old": expected[key], "new": observed[key]} for key in before
                      if expected[key] != observed[key]},
    "contractDigest": contract["digest"], "fixtureDigest": contract["fixtureDigest"],
    "interoperabilityDigest": contract["interoperabilityDigest"],
}, indent=2))
