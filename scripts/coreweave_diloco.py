#!/usr/bin/env python3
"""Run eight independent Megatoken learners on Forge-managed CoreWeave GPUs.

The coordinator runs elsewhere on persistent storage. This launcher keeps each
sandbox alive for the training command and injects the shared auth token via
Forge Secret Manager. No GPU-to-GPU connectivity or torchrun is required.

Requires: pip install 'cwsandbox>=1.14.2'
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from dataclasses import dataclass

from cwsandbox import AuthHeaders, Sandbox, ScratchVolumeOptions, Secret


@dataclass(frozen=True)
class WorkerResult:
    worker: int
    exit_code: int
    stdout: str
    stderr: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Registry image containing code, model and data")
    parser.add_argument("--config", default="configs/train/p1-diloco-coreweave.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32Gi")
    parser.add_argument("--scratch", default="80Gi")
    parser.add_argument("--data-s3", help="S3 prefix containing codec/ and prepared/ inputs")
    parser.add_argument("--s3-endpoint", default="https://cwobject.com")
    parser.add_argument("--s3-region", help="CoreWeave object-storage region, if required")
    parser.add_argument("--lifetime-seconds", type=int, default=86400)
    parser.add_argument(
        "--secret", action="append", default=["MEGATOKEN_DILOCO_TOKEN"],
        help="Forge Secret Manager name to inject (repeat for AWS credentials)",
    )
    parser.add_argument(
        "--resume-template",
        help="Optional sandbox checkpoint path with {worker} placeholder; "
        "leave unset for automatic S3 recovery",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 256 or not 120 <= args.lifetime_seconds <= 2592000:
        parser.error("workers must be 1..256 and lifetime must be 120..2592000 seconds")
    forge_key = os.environ.get("FORGE_API_KEY")
    if not forge_key:
        parser.error("Set FORGE_API_KEY for Forge-managed serverless GPU sandboxes")
    auth = AuthHeaders(
        headers={
            "x-wandb-api-key": forge_key,
            **({"x-entity-id": os.environ["FORGE_TEAM"]} if os.environ.get("FORGE_TEAM") else {}),
        },
        strategy="wandb_api_key",
    )
    secrets = [Secret(store="wandb", name=name) for name in dict.fromkeys(args.secret)]

    def execute(worker: int) -> WorkerResult:
        with Sandbox.run(
            auth=auth,
            placement_mode="serverless",
            container_image=args.image,
            resources={"cpu": args.cpu, "memory": args.memory, "gpu": 1},
            volumes=[
                ScratchVolumeOptions(
                    name="learner-work", mount_path="/work", size=args.scratch
                )
            ],
            secrets=secrets,
            environment_variables={
                "MEGATOKEN_WORKER_ID": str(worker),
                "AWS_ENDPOINT_URL_S3": args.s3_endpoint,
                **({"AWS_REGION": args.s3_region} if args.s3_region else {}),
            },
            network={"egress_mode": "internet"},
            max_lifetime_seconds=args.lifetime_seconds,
        ) as sandbox:
            check = sandbox.exec(["nvidia-smi", "-L"], timeout_seconds=120).result()
            print(f"worker {worker}: GPU {check.stdout.strip()}", flush=True)
            if args.data_s3:
                staged = sandbox.exec(
                    ["python", "scripts/stage_coreweave_data.py",
                     "--s3", args.data_s3, "--destination", "/work/artifacts"],
                    timeout_seconds=max(120, args.lifetime_seconds // 3),
                ).result()
                print(f"worker {worker}: {staged.stdout.strip()}", flush=True)
            command = ["megatoken", "train", "--config", args.config]
            if args.resume_template:
                command += ["--resume", args.resume_template.format(worker=worker)]
            result = sandbox.exec(
                command, timeout_seconds=args.lifetime_seconds - 60
            ).result()
            return WorkerResult(
                worker=worker,
                exit_code=getattr(result, "exit_code", getattr(result, "returncode", 0)),
                stdout=getattr(result, "stdout", ""),
                stderr=getattr(result, "stderr", ""),
            )

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, i) for i in range(args.workers)]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(
                    f"worker {result.worker}: exit={result.exit_code}\n"
                    f"{result.stdout[-8000:]}",
                    flush=True,
                )
                if result.stderr:
                    print(
                        f"worker {result.worker} stderr:\n{result.stderr[-8000:]}",
                        file=sys.stderr,
                    )
                if result.exit_code != 0:
                    failures.append(result.worker)
            except Exception as exc:
                failures.append(str(exc))
                print(f"Sandbox worker error: {exc}", file=sys.stderr)
    if failures:
        print(f"Some workers failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
