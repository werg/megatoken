from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root = Path(__file__).parent
consumer = root / 'consumer'
producer = root / 'producer-dev'
generator = consumer / 'scripts/generate_protocol_support_v8.mjs'
support_dir = consumer / 'tests/fixtures'
allowed = [
    'semantic-generator-close-v8.json.gz.b64',
    'semantic-enumeration-v8.json.gz.b64',
    'semantic-host-recovery-v8.json.gz.b64',
]
command = ['node', 'scripts/generate_protocol_support_v8.mjs', '../producer-dev']
env = dict(os.environ)
env['PATH'] = str(root / 'node24/node_modules/.bin') + os.pathsep + env['PATH']

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git(path, *args, text=True):
    return subprocess.check_output(['git', '-C', str(path), *args], text=text)

def tracked_paths():
    return {
        os.fsdecode(name)
        for name in git(consumer, 'ls-files', '-z', text=False).split(b'\0') if name
    }

def snapshot(paths):
    result = {}
    for name in paths:
        path = consumer / name
        if path.is_file():
            result[name] = digest(path.read_bytes())
        else:
            result[name] = None
    return result

def untracked_paths():
    return sorted(os.fsdecode(name) for name in git(
        consumer, 'ls-files', '--others', '--exclude-standard', '-z', text=False
    ).split(b'\0') if name)

producer_revision = git(producer, 'rev-parse', 'HEAD').strip()
producer_clean_start = not git(producer, 'status', '--porcelain', '--untracked-files=all').strip()
consumer_revision = git(consumer, 'rev-parse', 'HEAD').strip()
tracked_before = tracked_paths()
untracked_before = untracked_paths()
before = snapshot(tracked_before | set(untracked_before))
runner_hash = digest(Path(__file__).read_bytes())
generator_hash = digest(generator.read_bytes())
started = time.monotonic()
generator_exit = 1
if producer_clean_start:
    with (root / 'finish-support-fixtures.log').open('w') as stream:
        result = subprocess.run(command, cwd=consumer, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    generator_exit = result.returncode
else:
    (root / 'finish-support-fixtures.log').write_text(
        'Producer must be clean before support fixture generation.\n'
    )
tracked_after = tracked_paths()
untracked_after = untracked_paths()
after = snapshot(tracked_after | set(untracked_after))
consumer_revision_after = git(consumer, 'rev-parse', 'HEAD').strip()
producer_revision_after = git(producer, 'rev-parse', 'HEAD').strip()
producer_clean_end = not git(producer, 'status', '--porcelain', '--untracked-files=all').strip()
changed = sorted(name for name in before.keys() | after.keys()
                 if before.get(name) != after.get(name))
changed_tracked = sorted(name for name in changed if name in tracked_before | tracked_after)
changed_untracked = sorted(name for name in changed if name not in tracked_before | tracked_after)
allowed_paths = {f'tests/fixtures/{name}' for name in allowed}
unexpected = sorted(set(changed) - allowed_paths)
unexpected_untracked_path_changes = sorted(
    (set(untracked_before) ^ set(untracked_after)) - allowed_paths
)
file_hashes = {
    name: digest((support_dir / name).read_bytes())
    for name in allowed if (support_dir / name).is_file()
}
log_path = root / 'finish-support-fixtures.log'
status = {
    'schema': 'issue51-support-fixture-run/1',
    'producerRevision': producer_revision,
    'producerRevisionAfter': producer_revision_after,
    'producerTreeClean': producer_clean_start and producer_clean_end,
    'consumerRevision': consumer_revision,
    'consumerRevisionAfter': consumer_revision_after,
    'runnerScriptSha256': runner_hash,
    'generatorScriptSha256': generator_hash,
    'command': command,
    'log': log_path.name,
    'logSha256': digest(log_path.read_bytes()),
    'generatorExitCode': generator_exit,
    'exitCode': 0,
    'seconds': round(time.monotonic() - started, 3),
    'allowedGeneratedFiles': allowed,
    'changedFiles': changed,
    'changedTrackedFiles': changed_tracked,
    'changedUntrackedFiles': changed_untracked,
    'unexpectedChanges': unexpected,
    'unexpectedUntrackedPathChanges': unexpected_untracked_path_changes,
    'untrackedBefore': untracked_before,
    'untrackedAfter': untracked_after,
    'generatedSupportFileSha256': file_hashes,
}
valid = (
    producer_clean_start and producer_clean_end
    and producer_revision_after == producer_revision
    and consumer_revision_after == consumer_revision
    and generator_exit == 0
    and not unexpected
    and not unexpected_untracked_path_changes
    and set(file_hashes) == set(allowed)
)
if not valid:
    status['exitCode'] = 1
status['verificationPassed'] = valid
(root / 'finish-support-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status, indent=2))
raise SystemExit(status['exitCode'])
