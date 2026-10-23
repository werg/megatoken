#!/usr/bin/env python3
"""Prepare all required domains, train current weights, and retain generated failures.

This executes a bounded integrated experiment. Its completion report distinguishes
finished computation from demonstrated task quality and required human review.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from megatoken.io import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--writer-sha256", required=True)
    parser.add_argument("--writer-wait-seconds", type=int, default=3600)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--foundation-steps", type=int, default=256)
    parser.add_argument("--bridge-steps", type=int, default=128)
    parser.add_argument("--joint-steps", type=int, default=512)
    args = parser.parse_args()
    if min(args.foundation_steps, args.bridge_steps, args.joint_steps) < 1:
        parser.error("All training segments require a positive update budget")
    if len(args.writer_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.writer_sha256
    ):
        parser.error("The writer requires its exact SHA256 identity")
    if args.writer_wait_seconds < 0:
        parser.error("The writer wait bound cannot be negative")
    for path in (args.annotations, args.terms):
        if not path.is_file():
            raise FileNotFoundError(path)
    from megatoken.data.sources import TermsReview

    TermsReview(**json.loads(args.terms.read_text())).require_approval()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    scripts = repo / "scripts"
    env = {**os.environ, "PYTHONPATH": str(repo / "src"), "OMP_NUM_THREADS": "1"}
    receipts = []
    write_json(
        root / "experiment.json",
        {
            "script_sha256": sha256_file(__file__),
            "terms_sha256": sha256_file(args.terms),
            "annotations_sha256": sha256_file(args.annotations),
            "writer_sha256": args.writer_sha256,
            "budgets": {
                "foundation_updates": args.foundation_steps,
                "bridge_updates": args.bridge_steps,
                "joint_updates": args.joint_steps,
            },
            "purpose": "bounded integrated experiment, not production acceptance",
            "required_review": "independent decoded checks and human-reviewed original families",
        },
    )

    def run(name, script, *arguments, required=True):
        command = [sys.executable, str(scripts / script), *map(str, arguments)]
        write_json(root / "status.json", {"state": "running", "step": name})
        started = time.time()
        with (root / f"{name}.log").open("wb") as log:
            result = subprocess.run(
                command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        receipt = {
            "step": name,
            "command": command,
            "started_at": started,
            "finished_at": time.time(),
            "exit_code": result.returncode,
            "log_sha256": sha256_file(root / f"{name}.log"),
        }
        receipts.append(receipt)
        write_json(root / "steps.json", receipts)
        if result.returncode and required:
            write_json(root / "status.json", {"state": "failed", "step": name, **receipt})
            raise SystemExit(result.returncode)
        return result.returncode == 0

    packs = root / "packs/lock.json"
    run("donors", "acquire_packs.py", "--root", root / "packs")
    speech = root / "speech"
    run(
        "speech-preparation",
        "prepare_open_media.py",
        "--root",
        speech,
        "--packs",
        packs,
        "--work",
        root / "speech-in-flight",
        "--codec",
        "byte",
        "--count",
        "fleurs=128",
        "--count",
        "covost2=256",
        "--option",
        'fleurs.languages=["fr_fr","de_de","es_419"]',
    )
    executed = root / "executed"
    run("executed-preparation", "compare_delegation.py", "--root", executed, "--families", 128)
    from megatoken.data.prepare import prepare_sources
    from megatoken.data.sources import load_source_manifest
    from megatoken.generation.codec import ByteCodec

    selected = [
        replace(source, max_rows=2048 if source.stream == "natural_documents" else 256)
        for source in load_source_manifest(repo / "configs/data/sources.open.json")
        if source.source_id in {"fineweb-edu-10bt", "stackv2-edu-0000"}
    ]
    if len(selected) != 2:
        raise ValueError("Both approved broad-text and code sources are required")
    text = root / "text-code"
    write_json(root / "status.json", {"state": "running", "step": "text-code-preparation"})
    with prepare_sources(selected, text, ByteCodec.byte_fallback(), reject_invalid=True) as corpus:
        write_json(root / "text-code.json", corpus.manifest)
    conversation = root / "conversation"
    meetings = [
        "--meeting",
        "ES2002c:train",
        "--meeting",
        "ES2004c:train",
        "--meeting",
        "ES2003c:validation",
        "--meeting",
        "ES2009c:validation",
    ]
    run(
        "conversation-preparation",
        "prepare_ami_overlap.py",
        "--root",
        conversation,
        "--source-root",
        root / "conversation-originals",
        "--annotations",
        args.annotations,
        "--terms",
        args.terms,
        "--packs",
        packs,
        "--pairs",
        16,
        *meetings,
    )
    audiovisual = root / "audiovisual"
    run(
        "audiovisual-preparation",
        "prepare_open_media.py",
        "--root",
        audiovisual,
        "--packs",
        packs,
        "--work",
        root / "av-in-flight",
        "--codec",
        "byte",
        "--count",
        "rekadaily_processed=64",
    )
    from megatoken.data.fetch import download

    original_video = download(
        "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/video/ES2002a.Closeup1.avi",
        root / "original-video.avi",
    )
    visual = root / "visual"
    run(
        "visual-preparation",
        "prepare_visual_prefixes.py",
        "--root",
        visual,
        "--source-root",
        root / "visual-prefixes",
        "--train-video",
        original_video,
        "--terms",
        args.terms,
        "--packs",
        packs,
    )
    paired = root / "paired-translation"
    run(
        "paired-preparation",
        "prepare_paired_translation.py",
        "--root",
        paired,
        "--source-root",
        root / "paired-originals",
        "--packs",
        packs,
        "--train-families",
        32,
        "--validation-families",
        8,
    )
    context = root / "context"
    run(
        "context-preparation",
        "prepare_context_tasks.py",
        "--root",
        context,
        "--codec",
        "byte",
        "--families",
        16,
    )
    datasets = [
        speech,
        executed / "prepared",
        text,
        conversation,
        audiovisual,
        visual,
        paired,
        context,
    ]
    from megatoken.data.storage import PreparedCorpus

    with PreparedCorpus(datasets) as train, PreparedCorpus(datasets, split="validation") as heldout:
        if not len(train) or not len(heldout):
            raise ValueError("Integrated preparation needs both training and held-out examples")
        write_json(
            root / "prepared-identity.json",
            {
                "identity": train.identity,
                "roots": [str(path) for path in datasets],
                "train_episodes": len(train),
                "heldout_episodes": len(heldout),
            },
        )
    deadline = time.monotonic() + args.writer_wait_seconds
    writer_weights = args.writer / "training.safetensors"
    while not (args.writer / "receipt.json").is_file():
        write_json(root / "status.json", {"state": "awaiting_verified_writer"})
        if time.monotonic() >= deadline:
            raise TimeoutError("The required pinned writer has not been supplied")
        time.sleep(10)
    if sha256_file(writer_weights) != args.writer_sha256:
        raise ValueError("The writer differs from the declared immutable source")
    common = [
        "--packs",
        packs,
        "--model",
        "32x256",
        "--precision",
        "bf16",
        "--capacity",
        1024,
        "--microbatch-episodes",
        4,
        "--gradient-checkpointing",
        "--expert-backend",
        "triton",
        "--audio-samples",
        192000,
        "--visual-output",
        "--teacher-writer",
        args.writer,
    ]
    for dataset in datasets:
        common.extend(["--data", dataset])
    previous = None
    for name, phase, budget in (
        ("foundation", "B0", args.foundation_steps),
        ("bridge", "MB", args.bridge_steps),
        ("joint", "P1", args.joint_steps),
    ):
        arguments = [*common, "--root", root / name, "--phase", phase, "--steps", budget]
        if previous is not None:
            arguments.extend(
                [
                    "--resume",
                    root / previous / "training/checkpoints/latest.json",
                    "--teacher-encoding-from",
                    root / previous / "teacher-targets",
                ]
            )
        run(name, "accept_streaming_translation.py", *arguments)
        previous = name
    bundle = root / "joint/served"
    run(
        "autonomous-execution",
        "evaluate_dynamic_policy.py",
        "--bundle",
        bundle,
        "--packs",
        packs,
        "--root",
        root / "autonomous-execution",
        "--families",
        8,
        required=False,
    )
    run(
        "student-outcomes",
        "accept_student_outcomes.py",
        "--bundle",
        bundle,
        "--packs",
        packs,
        "--root",
        root / "student-outcomes",
        "--tasks",
        32,
        "--steps",
        32,
        required=False,
    )
    run(
        "translation-comparison",
        "compare_translation_paths.py",
        "--bundle",
        bundle,
        "--data",
        paired,
        "--packs",
        packs,
        "--root",
        root / "translation-comparison",
        required=False,
    )
    run(
        "decoded-quality",
        "evaluate_media_archive.py",
        "--archive",
        root / "translation-comparison",
        "--calibration-source",
        root / "paired-originals/acquisition.json",
        "--root",
        root / "decoded-quality",
        required=False,
    )
    run(
        "duplex-execution",
        "accept_duplex_runtime.py",
        "--bundle",
        bundle,
        "--packs",
        packs,
        "--recording-data",
        conversation,
        "--split",
        "validation",
        "--root",
        root / "duplex-execution",
        required=False,
    )
    write_json(
        root / "report.json",
        {
            "steps": receipts,
            "software_sequence_finished": True,
            "all_commands_succeeded": all(r["exit_code"] == 0 for r in receipts),
            "whole_plan_complete": False,
            "production_quality_established": False,
            "human_review": "pending; generated outputs and independent checks are retained",
            "acceptance_rule": "Finished optimization cannot substitute for verified autonomous behavior, decoded quality and all-cost matched timing.",
        },
    )
    write_json(root / "status.json", {"state": "awaiting_quality_review"})


if __name__ == "__main__":
    main()
