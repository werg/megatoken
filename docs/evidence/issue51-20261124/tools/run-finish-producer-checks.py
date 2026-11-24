from pathlib import Path
import json
import os
import subprocess
import time

root=Path(__file__).parent
repo=root/'producer-dev'
env=dict(os.environ)
env['PATH']=str(root/'node24/node_modules/.bin')+os.pathsep+env['PATH']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
commands=[
    ['node','--experimental-vm-modules','--test','--test-concurrency=4',
     *[str(p.relative_to(repo)) for p in sorted((repo/'test').glob('*.test.js'))]],
    ['node','fixtures/reference/reference-fixture.mjs']
]
status={'producerRevision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'testFiles':len(commands[0])-4,'testConcurrency':4,'steps':[]}
for index,command in enumerate(commands):
    name='finish-producer-all.log' if index==0 else 'finish-reference.log'
    status['current']={'command':command,'log':name}
    (root/'finish-producer-status.json').write_text(json.dumps(status,indent=2)+'\n')
    started=time.monotonic()
    with (root/name).open('w') as stream:
        result=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
    status['steps'].append({'command':command,'log':name,'exitCode':result.returncode,
                            'seconds':round(time.monotonic()-started,3)})
    status['current']=None
    (root/'finish-producer-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status['steps'][-1]),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
print('ALL PRODUCER TESTS AND REFERENCE FIXTURE PASSED',flush=True)
