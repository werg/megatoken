from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

root = Path(__file__).parent
repo = root / 'producer-dev'
def git(*args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()
def last_json(path):
    rows = [line for line in Path(path).read_text().splitlines() if line.lstrip().startswith('{')]
    assert rows, path
    return json.loads(rows[-1])
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert not git('status', '--porcelain', '--untracked-files=all')
revision = git('rev-parse', 'HEAD')
generation = json.loads((root / 'finish-canonical-status.json').read_bytes())
generation_commands = [
    ['node', 'tools/generate-semantic-region-contract.mjs', '--definition-only'],
    ['node', 'tools/generate-protocol-machine-fixtures.mjs'],
    ['node', 'tools/generate-protocol-iterator-fixtures.mjs'],
    ['node', 'tools/generate-protocol-generator-fixtures.mjs'],
    ['node', 'tools/generate-protocol-iterator-source-fixtures.mjs'],
    ['node', 'tools/generate-protocol-host-fixtures.mjs'],
    ['node', 'tools/generate-primitive-conversion-fixtures.mjs'],
    ['node', 'tools/generate-protocol-source-fixtures.mjs'],
    ['node', 'tools/generate-semantic-region-contract.mjs'],
    ['npm', 'run', 'contracts:check'],
]
assert len(generation['steps']) == 10
assert [step['command'] for step in generation['steps']] == generation_commands
assert all(step['exitCode'] == 0 for step in generation['steps'])
assert all(Path(step['log']).is_file() for step in generation['steps'])

commands = [
    ['node', 'tools/generate-protocol-source-fixtures.mjs', '--check'],
    ['npm', 'run', 'contracts:check'],
]
env = dict(os.environ)
env['PATH'] = str(root / 'node24/node_modules/.bin') + os.pathsep + env['PATH']
status = {
    'producerRevision': revision,
    'generationSourceRevision': generation['implementationRevision'],
    'generatedArtifactRevision': generation['producerRevision'],
    'generatedDataUnchanged': True,
    'changesSinceGeneration': git('diff', '--name-only', generation['producerRevision'], revision).splitlines(),
    'steps': [],
}
if revision == generation['producerRevision']:
    changed = git('diff', '--name-only', generation['implementationRevision'], generation['producerRevision']).splitlines()
    generated_contracts = {
        'contracts/semantic-execution-v8.json',
        'contracts/semantic-native-policy-v8.json',
    }
    assert all(path in generated_contracts or path.startswith('test/fixtures/semantic-') for path in changed), changed
    source_log = Path(generation['steps'][7]['log'])
    gate_log = Path(generation['steps'][9]['log'])
    source_result, gate_result = last_json(source_log), last_json(gate_log)
    assert source_result.get('valid') is True and source_result.get('mode') == 'generated', source_result
    assert gate_result.get('valid') is True and gate_result.get('mode') == 'checked', gate_result
    assert not git('status', '--porcelain', '--untracked-files=all')
    status.update(
        mode='verified-generation-artifact',
        generatedPaths=changed,
        evidenceLogs=[
            {'stepIndex': 7, 'purpose': 'protocol-source-fixture-generation',
             'command': generation['steps'][7]['command'], 'log': str(source_log),
             'sha256': sha256(source_log), 'valid': source_result['valid'], 'mode': source_result['mode']},
            {'stepIndex': 9, 'purpose': 'standard-contract-gate',
             'command': generation['steps'][9]['command'], 'log': str(gate_log),
             'sha256': sha256(gate_log), 'valid': gate_result['valid'], 'mode': gate_result['mode']},
        ],
    )
    (root / 'finish-canonical-recheck-status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({'mode': status['mode'], 'producerRevision': revision,
                      'generatedPaths': changed, 'evidenceLogs': status['evidenceLogs'],
                      'rerunSteps': []}, indent=2), flush=True)
    print('FINAL GENERATED ARTIFACT PIN AND STANDARD CONTRACT GATE VERIFIED', flush=True)
else:
    assert git('merge-base', '--is-ancestor', generation['producerRevision'], revision) == ''
    data_changes = git('diff', '--name-only', generation['producerRevision'], revision, '--', 'contracts', 'test/fixtures')
    assert not data_changes, data_changes
    status['mode'] = 'fresh-recheck'
    for index, command in enumerate(commands):
        log = root / ('finish-canonical-recheck-%02d.log' % index)
        status['current'] = {'command': command, 'log': log.name}
        (root / 'finish-canonical-recheck-status.json').write_text(json.dumps(status, indent=2) + '\n')
        started = time.monotonic()
        with log.open('w') as output:
            result = subprocess.run(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
        unchanged = revision == git('rev-parse', 'HEAD') and not git('status', '--porcelain', '--untracked-files=all')
        status['steps'].append({
            'command': command, 'log': log.name, 'exitCode': result.returncode,
            'producerTreeUnchanged': unchanged, 'seconds': round(time.monotonic() - started, 3),
        })
        status['current'] = None
        (root / 'finish-canonical-recheck-status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status['steps'][-1]), flush=True)
        if result.returncode or not unchanged:
            raise SystemExit(result.returncode or 1)
    print('FRESH CANONICAL SOURCE AND STANDARD CONTRACT CHECKS PASSED', flush=True)
