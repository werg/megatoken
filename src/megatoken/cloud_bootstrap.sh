#!/bin/bash
# In-sandbox bootstrap for megatoken CoreWeave jobs (mounted by `megatoken.cloud launch`; the sandbox's
# main process is `bash bootstrap.sh <job command...>`, so its stdout/stderr are the sandbox logs).
#
#   1. wait for the controller's pushed inputs ($CW/inputs-ready): the code snapshot, inputs.json and any
#      pushed files, all named by sha256 in $CW/inbox;
#   2. verify and unpack the code snapshot at $REPO, `pip install -e` it plus the job's extra requirements;
#   3. materialize every input (pushed or public URL) at its declared path, verifying sha256;
#   4. run the job in $REPO, forwarding SIGTERM so trainers write their final state;
#   5. write a sha256 manifest of every output directory, signal $CW/outputs-ready and stay up until the
#      controller has pulled them ($CW/outputs-collected) or MEGATOKEN_CW_COLLECT_WAIT seconds pass.
#
# Always exits 0: a nonzero main-process exit makes the platform restart the container and rerun the job.
# The job's real status is in the log and $CW/exit-status.
set -uo pipefail
WORK=${MEGATOKEN_CW_WORK:-/work}
CW="$WORK/.cloud"
REPO="$WORK/megatoken"
log() { printf '{"bootstrap":"%s","job":"%s","at":"%s"%s}\n' "$1" "${MEGATOKEN_CW_JOB:-}" "$(date -u +%FT%TZ)" "${2:+,$2}"; }
finish() { log done "\"status\":$1"; exit 0; }
die() { log failed "\"step\":\"$1\""; echo 1 > "$CW/exit-status"; touch "$CW/outputs-ready"; finish 1; }

mkdir -p "$CW/inbox" "$REPO"
log start "\"sandbox\":\"${CW_SANDBOX_ID:-}\""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
log waiting-for-inputs
until [ -f "$CW/inputs-ready" ]; do sleep 2; done

python3 - <<'PY' || die inputs
import hashlib, json, os, shutil, tarfile, urllib.request
from pathlib import Path

work = Path(os.environ.get("MEGATOKEN_CW_WORK", "/work"))
cw, repo = work / ".cloud", work / "megatoken"
spec = json.loads((cw / "inputs.json").read_text())

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()

code = cw / "inbox" / spec["code"]["sha256"]
if digest(code) != spec["code"]["sha256"]:
    raise SystemExit("code snapshot checksum mismatch")
with tarfile.open(code) as tar:
    tar.extractall(repo, filter="data")
for item in spec["inputs"]:
    target = work / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    pushed = cw / "inbox" / item["sha256"]
    staging = target.with_name(f".{target.name}.partial")
    if pushed.exists():
        shutil.move(pushed, staging)
    else:
        with urllib.request.urlopen(item["url"], timeout=600) as response, open(staging, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 22)
    if digest(staging) != item["sha256"]:
        raise SystemExit(f"{item['path']}: checksum mismatch")
    staging.replace(target)
    print(json.dumps({"bootstrap": "input", "path": item["path"]}), flush=True)
PY
log inputs-verified

python3 -m pip install -q --no-cache-dir -e "$REPO" ${MEGATOKEN_CW_PIP:-} || die pip
log installed

cd "$REPO" || die workdir
log run "\"command\":\"$*\""
"$@" &
child=$!
trap 'log term; kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done  # a trapped TERM interrupts wait
log exited "\"status\":$status"
echo "$status" > "$CW/exit-status"

python3 - <<'PY' || log manifest-failed
import hashlib, json, os
from pathlib import Path

work = Path(os.environ.get("MEGATOKEN_CW_WORK", "/work"))
files = []
for name in os.environ.get("MEGATOKEN_CW_OUTPUTS", "").split():
    root = work / name
    for path in sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        files.append({"path": str(path.relative_to(work)), "bytes": path.stat().st_size, "sha256": h.hexdigest()})
(work / ".cloud" / "outputs.json").write_text(json.dumps({"files": files}))
PY
touch "$CW/outputs-ready"
log outputs-ready
waited=0
until [ -f "$CW/outputs-collected" ] || [ $waited -ge "${MEGATOKEN_CW_COLLECT_WAIT:-21600}" ]; do
  sleep 5; waited=$((waited + 5))
done
log outputs-collected "\"waited\":$waited"
finish "$status"
