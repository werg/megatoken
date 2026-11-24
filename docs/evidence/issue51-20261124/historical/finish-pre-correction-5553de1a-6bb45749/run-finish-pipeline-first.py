"""Continue the authorized final checks after canonical artifacts are committed.

Existing focused drivers own the evidence. This coordinator never records a
passing check without a successful driver, and stops before publishing receipts.
"""
from pathlib import Path
import json
import os
import subprocess
import time
import traceback

root = Path(__file__).parent
producer, consumer = root / 'producer-dev', root / 'consumer'
python = str(root / '.venv/bin/python')
env = dict(os.environ)
env['PATH'] = str(root / 'node24/node_modules/.bin') + os.pathsep + env['PATH']
status_path = root / 'finish-pipeline-status.json'
assert not status_path.exists(), 'Inspect the existing pipeline before resuming'
status = {'implementationRevision': '76fd24a91ad7e535aac33af542b481e1337838df',
          'phase': 'awaiting-canonical-artifact', 'steps': [], 'jobs': {}}
jobs = {}


def save():
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(status_path)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def phase(name):
    status['phase'] = name
    save()
    print(name, flush=True)


def generation():
    try:
        return json.loads((root / 'finish-canonical-status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def wait_generation(complete):
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        value = generation()
        if value:
            assert all(step['exitCode'] == 0 for step in value['steps']), value
            if 'implementationRevision' in value:
                assert value['implementationRevision'] == status['implementationRevision']
                if not complete or len(value['steps']) == 10:
                    assert len(value['steps']) >= 9
                    assert value['producerRevision'] == git(producer, 'rev-parse', 'HEAD')
                    assert not git(producer, 'status', '--porcelain', '--untracked-files=all')
                    return value
        time.sleep(5)
    raise TimeoutError('Canonical generation has not reached the required boundary')


def run(name, command, output=None):
    started = time.monotonic()
    path = root / (output or ('finish-pipeline-' + name + '.log'))
    with path.open('w') as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    status['steps'].append({'name': name, 'command': command, 'log': path.name,
                            'exitCode': result.returncode,
                            'seconds': round(time.monotonic() - started, 3)})
    save()
    assert result.returncode == 0, (name, result.returncode, str(path))


def launch(name, driver):
    path = root / ('finish-pipeline-' + name + '.log')
    with path.open('w') as stream:
        process = subprocess.Popen([python, '-u', str(root / driver)], cwd=root,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    jobs[name] = process
    status['jobs'][name] = {'pid': process.pid, 'driver': driver, 'log': path.name}
    save()


def wait_job(name):
    result = jobs[name].wait()
    status['jobs'][name]['exitCode'] = result
    save()
    assert result == 0, (name, result, status['jobs'][name]['log'])


def commit_data(message):
    assert not git(consumer, 'ls-files', '--others', '--exclude-standard')
    changed = git(consumer, 'diff', '--name-only').splitlines()
    exact = {'src/megatoken/data/contracts/semantic-execution-v8.json',
             'tests/test_protocol_v8.py', 'tests/test_issue51_object_source_v8.py'}
    assert all(path in exact or (path.startswith('tests/fixtures/semantic-')
               and path.endswith('.json.gz.b64')) for path in changed), changed
    assert not git(consumer, 'diff', '--name-only', '--diff-filter=D')
    if changed:
        subprocess.run(['git', '-C', str(consumer), 'add', '--', *changed], check=True)
        subprocess.run(['git', '-C', str(consumer), 'commit', '-m', message], check=True)
    assert not git(consumer, 'status', '--porcelain', '--untracked-files=all')
    status['consumerRevision'] = git(consumer, 'rev-parse', 'HEAD')
    save()


def push(repo, name):
    run(name, ['git', '-C', str(repo), '-c', 'credential.helper=', '-c',
              'credential.helper=!gh auth git-credential', 'push', 'origin',
              'HEAD:refs/heads/codex/issue-51-object-semantics'])


try:
    save()
    assert not git(consumer, 'status', '--porcelain', '--untracked-files=all')
    generated = wait_generation(False)
    status['producerRevision'] = generated['producerRevision']
    phase('producer-tests-and-export')
    launch('producer', 'run-finish-producer-checks.py')
    launch('export', 'run-finish-cohort-export.py')

    phase('synchronizing-canonical-consumer-data')
    run('canonical-sync', [python, str(root / 'sync-object-canonical.py')],
        'finish-canonical-sync.json')
    run('support', [python, str(root / 'run-finish-support-fixtures.py')])
    run('format-canonical-test', [str(root / '.venv/bin/ruff'), 'format',
        str(consumer / 'tests/test_protocol_v8.py')])
    commit_data('Synchronize corrected construction source fixtures')

    phase('awaiting-complete-issue-export')
    wait_job('export')
    run('cohort-sync', [python, str(root / 'sync-finish-cohort.py')],
        'finish-cohort-sync.json')
    run('format-cohort-test', [str(root / '.venv/bin/ruff'), 'format',
        str(consumer / 'tests/test_issue51_object_source_v8.py')])
    commit_data('Synchronize complete retained object source cohort')

    phase('consumer-verification-and-preparation')
    launch('native', 'run-finish-native-checks.py')
    launch('cohort', 'run-finish-cohort-checks.py')
    launch('preparation', 'run-finish-preparation.py')
    run('python-style', [python, str(root / 'check-finish-python-style.py')])

    phase('confirming-canonical-gate')
    wait_generation(True)
    run('canonical-verification', [python, str(root / 'recheck-finish-canonical.py')])
    push(producer, 'push-producer-artifacts')
    push(consumer, 'push-consumer-artifacts')

    phase('awaiting-all-final-checks')
    for name in ('producer', 'native', 'cohort', 'preparation'):
        wait_job(name)
    assert git(producer, 'rev-parse', 'HEAD') == status['producerRevision']
    assert git(consumer, 'rev-parse', 'HEAD') == status['consumerRevision']
    for repo in (producer, consumer):
        assert not git(repo, 'status', '--porcelain', '--untracked-files=all')
    phase('ready-for-evidence-review')
    print('All final drivers passed; review receipts before documentation commits.', flush=True)
except BaseException as error:
    status['phase'] = 'stopped-for-review'
    status['error'] = str(error)
    status['jobsStillRunning'] = [name for name, process in jobs.items() if process.poll() is None]
    save()
    traceback.print_exc()
    raise
