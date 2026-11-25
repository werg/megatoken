from pathlib import Path
import base64
import collections
import gzip
import json
import subprocess
import textwrap

root = Path(__file__).parent
producer, consumer = root / 'producer-dev', root / 'consumer'
source = root / 'cohort-finish.json'
target = consumer / 'tests/fixtures/semantic-issue51-objects-v8.json.gz.b64'
report = json.loads(source.read_bytes())
previous = json.loads(gzip.decompress(base64.b64decode(target.read_bytes())))
revision = subprocess.check_output(['git','-C',str(producer),'rev-parse','HEAD'], text=True).strip()
contract = json.loads((producer / 'contracts/semantic-execution-v8.json').read_bytes())
assert report['producerRevision'] == revision and report['producerTreeClean'] is True
assert not subprocess.check_output(['git','-C',str(producer),'status','--porcelain','--untracked-files=all'], text=True)
before = {case['id']: case for case in previous['cases']}
after = {case['id']: case for case in report['cases']}
assert len(after) == len(report['cases']) and set(before) <= set(after)
for key, old in before.items():
 for field in ('source','entry','args','family','ancestorIds'):
  previous_fixture, current_fixture = old['fixture'], after[key]['fixture']
  assert (field in previous_fixture) == (field in current_fixture), (key,field,'presence')
  assert json.dumps(previous_fixture.get(field), ensure_ascii=False, separators=(',',':')) == json.dumps(current_fixture.get(field), ensure_ascii=False, separators=(',',':')), (key,field)
assert all(case['fixture']['ancestorIds'] == ['issue51-original-objects'] for case in after.values())
status = {case['id']: case for case in report['dispositions']}
assert set(status) == set(after)
for key, case in after.items():
 assert status[key]['status'] == ('passed' if case['artifact'] is not None else 'refused'), key
 assert status[key]['expected'] == ('supported' if case['artifact'] is not None else 'refused'), key
encoded = base64.b64encode(gzip.compress(source.read_bytes(), mtime=0)).decode('ascii')
target.write_text('\n'.join(textwrap.wrap(encoded, 76)) + '\n')
artifacts = [case['artifact'] for case in report['cases'] if case['artifact'] is not None]
(root / 'artifacts-finish.jsonl').write_text(''.join(json.dumps(case, separators=(',',':')) + '\n' for case in artifacts))
assert subprocess.check_output(['git','-C',str(producer),'rev-parse','HEAD'], text=True).strip() == revision
assert not subprocess.check_output(['git','-C',str(producer),'status','--porcelain','--untracked-files=all'], text=True)
print(json.dumps({'producerRevision':revision, 'producerTreeClean':True,
 'declared':len(after), 'preservedOriginals':len(before),
 'supported':len(artifacts), 'refused':len(after)-len(artifacts),
 'newMembers':sorted(set(after)-set(before)), 'operationCoverage':report['operationCoverage'],
 'contractDigest':contract['digest'], 'fixtureDigest':contract['fixtureDigest'],
 'interoperabilityDigest':contract['interoperabilityDigest']},indent=2))

# The fixed expected population changes only after all frozen inputs above
# have been checked byte-for-byte and every disposition matches the source gate.
import re
test_path = consumer / 'tests/test_issue51_object_source_v8.py'
test = test_path.read_text()
test = re.sub(r'assert len\(CASES\) == [0-9]+', f'assert len(CASES) == {len(after)}', test)
test = re.sub(r'assert len\(ACCEPTED\) == [0-9]+', f'assert len(ACCEPTED) == {len(artifacts)}', test)
test = re.sub(r'assert len\(\{row\["id"\] for row in CASES\}\) == [0-9]+',
              f'assert len({{row["id"] for row in CASES}}) == {len(after)}', test)
start = test.index('    assert {row["id"] for row in CASES if row["artifact"] is None} == ')
end = test.index('    for row in CASES:', start)
refusals = sorted(row['id'] for row in report['cases'] if row['artifact'] is None)
expected = '{\n' + ''.join(f'        {json.dumps(identity)},\n' for identity in refusals) + '    }\n' if refusals else 'set()\n'
test = test[:start] + '    assert {row["id"] for row in CASES if row["artifact"] is None} == ' + expected + test[end:]
test_path.write_text(test)
