from pathlib import Path
import json
import os
import subprocess
import time

root=Path(__file__).parent
repo=root/'consumer'
producer=root/'producer-dev'
env=dict(os.environ)
env['PATH']=str(root/'node24/node_modules/.bin')+os.pathsep+env['PATH']
env['PYTHONPATH']='src'
def revision(path):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=path,text=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
producer_revision,consumer_revision=revision(producer),revision(repo)

command=[str(root/'.venv/bin/python'),'-m','pytest','-q','tests/test_issue51_object_source_v8.py']
status={'producerRevision':producer_revision,'consumerRevision':consumer_revision,'command':command}
started=time.monotonic()
with (root/'finish-cohort-tests.log').open('w') as stream:
    result=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
status.update(exitCode=result.returncode,seconds=round(time.monotonic()-started,3))
(root/'finish-cohort-status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
raise SystemExit(result.returncode)
