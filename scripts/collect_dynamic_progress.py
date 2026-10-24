#!/usr/bin/env python3
"""Copy committed experiment outputs while training runs; never stop the machine.

Collected bytes are addressed by their SHA256 and an immutable receipt retains
original paths. This transports exact artifacts; their normal readers still
require the current contracts when they are restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from megatoken import cloud
from megatoken.io import sha256_file, write_json

SNAPSHOT = r"""
import hashlib,json,os,shutil,sys,uuid
from pathlib import Path
work=Path(sys.argv[1]).resolve();root=Path(sys.argv[2]).resolve()
if not root.is_relative_to(work/'out') or root==work/'out':
 raise ValueError('Select one experiment directory under the cloud output root')
snapshot=work/'.cloud'/('progress-'+uuid.uuid4().hex);snapshot.mkdir(parents=True)
selected=set();errors=[]
# Reports and bounded log tails survive failures before any model is exported.
for path in root.rglob('*'):
 if not path.is_file() or path.is_symlink():continue
 rel=path.relative_to(root)
 if any(part in {'packs','originals','conversation-originals','paired-originals','visual-prefixes'} for part in rel.parts):continue
 if path.suffix=='.json' and 'checkpoints' not in rel.parts:selected.add(path)
 if any(part in {'decoded','translation-comparison'} for part in rel.parts):selected.add(path)
# Only a committed prepared manifest admits its immutable index, shards and assets.
for path in root.rglob('manifest.json'):
 try:value=json.loads(path.read_text())
 except (OSError,ValueError):continue
 if value.get('schema_version')==6 and 'shards' in value:
  prepared=path.parent
  selected.add(path)
  selected.add(prepared/'index.sqlite')
  selected.update(prepared.glob('episodes-*.jsonl'))
  selected.update(p for p in (prepared/'media').rglob('*') if p.is_file())
 if value.get('format_version')==16 and value.get('training_heads') is False:
  selected.update(p for p in path.parent.iterdir() if p.is_file())
 if value.get('format')=='hindsight-targets-v1' and value.get('scope')=='text_windows':
  selected.update(p for p in path.parent.rglob('*') if p.is_file() and '.building.' not in p.name)
# Pin complete checkpoint members before the trainer can prune their source directory.
items=[]
for pointer in root.rglob('checkpoints/latest.json'):
 pinned=None
 try:
  pointer_bytes=pointer.read_bytes();value=json.loads(pointer_bytes);name=value['checkpoint']
  if not isinstance(name,str) or name in {'','.', '..'} or Path(name).name!=name:
   raise ValueError('Unsafe checkpoint name')
  checkpoint=pointer.parent/name;metadata=checkpoint/'metadata.json'
  metadata_bytes=metadata.read_bytes()
  if hashlib.sha256(metadata_bytes).hexdigest()!=value['metadata_sha256']:
   raise ValueError('Uncommitted checkpoint metadata')
  contract=json.loads(metadata_bytes)
  if contract['format_version']!=16 or contract['checkpoint_name']!=name:
   raise ValueError('Only named current checkpoints can be collected')
  transducer=contract['transducer_checkpoint']
  if transducer['file']!='transducer.pt':raise ValueError('Unsafe memory checkpoint path')
  expected={'metadata.json':value['metadata_sha256'],'state.pt':contract['state_sha256'],
            'transducer.pt':transducer['sha256']}
  pinned=snapshot/checkpoint.relative_to(root);pinned.mkdir(parents=True)
  group=[]
  for filename,digest in expected.items():
   source=checkpoint/filename
   if source.is_symlink():raise ValueError('Checkpoint members must be regular files')
   destination=pinned/filename;os.link(source,destination)
   if hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()!=digest:
    raise ValueError('Checkpoint member digest differs from its committed metadata')
   group.append({'path':str(destination.relative_to(work)),
                 'original':str(source.relative_to(root)),'bytes':destination.stat().st_size,
                 'sha256':digest})
  destination=snapshot/pointer.relative_to(root);destination.write_bytes(pointer_bytes)
  group.append({'path':str(destination.relative_to(work)),'original':str(pointer.relative_to(root)),
                'bytes':len(pointer_bytes),'sha256':hashlib.sha256(pointer_bytes).hexdigest()})
  items.extend(group)
 except (OSError,ValueError,KeyError,TypeError) as e:
  if pinned is not None:shutil.rmtree(pinned,ignore_errors=True)
  errors.append({'path':str(pointer.relative_to(root)),'error':type(e).__name__})
for source in sorted(selected):
 try:
  relative=source.relative_to(root)
  if not source.is_file() or source.is_symlink():raise ValueError('Missing immutable output')
  destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
  if source.suffix=='.json':
   data=source.read_bytes();json.loads(data);destination.write_bytes(data)
  else:os.link(source,destination)
  digest=hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()
  items.append({'path':str(destination.relative_to(work)),'original':str(relative),
                'bytes':destination.stat().st_size,'sha256':digest})
 except (OSError,ValueError) as e:errors.append({'path':str(source.relative_to(root)),'error':type(e).__name__})
for source in root.glob('*.log'):
 with source.open('rb') as stream:
  stream.seek(max(0,source.stat().st_size-(1<<20)));data=stream.read(1<<20)
 destination=snapshot/(source.name+'.tail');destination.write_bytes(data)
 items.append({'path':str(destination.relative_to(work)),'original':source.name+'.tail',
               'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
manifest={'files':items,'capture_errors':errors}
(snapshot/'receipt.json').write_text(json.dumps(manifest))
print(json.dumps({'manifest':str((snapshot/'receipt.json').relative_to(work)),
                  'snapshot':str(snapshot.relative_to(work)),'files':len(items),
                  'bytes':sum(i['bytes'] for i in items),'capture_errors':errors}))
"""


def collect(sandbox, remote_root, destination, *, reserve_bytes):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = sandbox.exec(["python3", "-c", SNAPSHOT, "/work", remote_root]).result()
    if result.returncode:
        raise cloud.CloudError("Could not capture committed experiment outputs")
    capture = json.loads(result.stdout)
    receipt = json.loads(sandbox.read_file("/work/" + capture["manifest"]).result())
    downloaded = 0
    for item in receipt["files"]:
        digest = item["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid captured content identity")
        target = destination / "objects" / digest
        if target.is_file() and sha256_file(target) == digest:
            continue
        if shutil.disk_usage(destination).free - item["bytes"] < reserve_bytes:
            raise OSError("Progress collection would cross the reserved local disk floor")
        target.parent.mkdir(exist_ok=True)
        cloud._pull(sandbox, item, target)
        downloaded += item["bytes"]
    record = {
        "sandbox": sandbox.sandbox_id,
        "remote_root": remote_root,
        "captured_at": time.time(),
        "downloaded_bytes": downloaded,
        **receipt,
    }
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    write_json(destination / "receipts" / (identity + ".json"), record)
    if not receipt["capture_errors"]:
        write_json(destination / "latest.json", {"receipt": "receipts/" + identity + ".json"})
    sandbox.exec(
        [
            "python3",
            "-c",
            "import shutil,sys; shutil.rmtree(sys.argv[1])",
            "/work/" + capture["snapshot"],
        ]
    ).result()
    return {
        "files": len(receipt["files"]),
        "downloaded_bytes": downloaded,
        "capture_errors": receipt["capture_errors"],
        "complete_capture": not receipt["capture_errors"],
        "receipt": identity,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=int, default=20)
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.reserve_gib < 1 or args.interval_seconds < 10:
        parser.error("Preserve a positive disk reserve and at least ten seconds between captures")
    cloud.load_credentials()
    sandbox = cloud._attach(args.sandbox)
    while True:
        print(
            json.dumps(
                collect(
                    sandbox, args.remote_root, args.output, reserve_bytes=args.reserve_gib << 30
                )
            ),
            flush=True,
        )
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
