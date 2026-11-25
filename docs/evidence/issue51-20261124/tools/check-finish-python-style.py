from pathlib import Path
import os,subprocess,json
root=Path(__file__).parent
repo=root/'consumer'
files=[name for name in subprocess.check_output(['git','diff','--name-only','b432f976e85382fed59aa7f15a50dde215e8c6fd','HEAD'],cwd=repo,text=True).splitlines() if name.endswith('.py') and (repo/name).is_file()]
assert files
commands=[[str(root/'.venv/bin/ruff'),'check',*files],[str(root/'.venv/bin/ruff'),'format','--check',*files]]
status={'consumerRevision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'files':files,'steps':[]}
with (root/'finish-python-style.log').open('w') as log:
 for command in commands:
  result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
  status['steps'].append({'command':command,'exitCode':result.returncode})
(root/'finish-python-style-status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'files':len(files),'results':[step['exitCode'] for step in status['steps']]}))
raise SystemExit(any(step['exitCode'] for step in status['steps']))
