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
assert not git('status', '--porcelain', '--untracked-files=all')
revision = git('rev-parse', 'HEAD')
env = dict(os.environ)
env['PATH'] = str(root / 'node24/node_modules/.bin') + os.pathsep + env['PATH']
command = ['node', 'tools/export-issue51-evidence.mjs', '--out', '../cohort-finish.json']
started = time.monotonic()
with (root / 'finish-cohort-export.log').open('w') as output:
    result = subprocess.run(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT)
unchanged = revision == git('rev-parse', 'HEAD') and not git('status', '--porcelain', '--untracked-files=all')
status = {
    'producerRevision': revision, 'producerTreeUnchanged': unchanged,
    'command': command, 'exitCode': result.returncode,
    'seconds': round(time.monotonic() - started, 3),
    'logSha256': hashlib.sha256((root / 'finish-cohort-export.log').read_bytes()).hexdigest(),
}
if result.returncode == 0:
    status['reportSha256'] = hashlib.sha256((root / 'cohort-finish.json').read_bytes()).hexdigest()
(root / 'finish-cohort-export-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status), flush=True)
raise SystemExit(result.returncode or (0 if unchanged else 1))
