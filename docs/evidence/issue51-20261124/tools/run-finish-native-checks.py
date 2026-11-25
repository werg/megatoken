from pathlib import Path
import json
import os
import subprocess
import time

root = Path(__file__).parent
repo = root / 'consumer'
env = dict(os.environ)
env['PATH'] = str(root / 'node24/node_modules/.bin') + os.pathsep + env['PATH']
env['PYTHONPATH'] = 'src'
files = [
 'tests/test_proxy_v8.py', 'tests/test_proxy_reflection_v8.py', 'tests/test_intrinsic_error_v8.py',
 'tests/test_collection_v8.py', 'tests/test_iterator_collections_v8.py', 'tests/test_iterator_spread_v8.py',
 'tests/test_primitive_conversion_v8.py', 'tests/test_primitive_conversion_source_v8.py',
 'tests/test_construction_v8.py', 'tests/test_protocol_v8.py', 'tests/test_protocol_properties.py',
 'tests/test_promise_scheduler_v8.py', 'tests/test_promise_executor_v8.py',
 'tests/test_promise_executor_boundaries_v8.py', 'tests/test_promise_join_v8.py',
 'tests/test_async_call_v8.py', 'tests/test_semantic_observation_unicode.py', 'tests/test_prepare_semantic_v8.py',
 'tests/test_native_v8.py', 'tests/test_native_generator_close_v8.py',
 'tests/test_prediction_prefix_v8.py', 'tests/test_enumeration_v8.py',
 'tests/test_lifecycle_recovery_v8.py', 'tests/test_native_v8_without_legacy.py',
 'tests/test_protocol_host_composition_v8.py', 'tests/test_protocol_host_v8.py',
]
for checkout in (repo,root/'producer-dev'):
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=checkout,text=True), checkout
for name in files:
 assert (repo / name).is_file(), name
command = [str(root / '.venv/bin/python'), '-m', 'pytest', '-q', *files]
status = {'consumerRevision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
 'producerRevision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root/'producer-dev', text=True).strip(),
 'command': command}
start = time.monotonic()
with (root / 'finish-native-final.log').open('w') as stream:
 result = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
status.update(exitCode=result.returncode,seconds=round(time.monotonic()-start, 3))
(root / 'finish-native-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status))
raise SystemExit(result.returncode)
