"""Measure observed-text surprise and retrospective lag on a supplied conversation.

Input is JSONL: {"text": "...", "role": "user", "novel": true}. Mark planted
facts novel and leave routine messages false. Use --model for a current trained
checkpoint or served bundle; --smoke uses random tiny weights and tests mechanics
only. Reported quality numbers from --smoke have no scientific interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from megatoken.contracts import ROLE_TO_ID
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model, load_trained
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.served import ServedBundle
from megatoken.serving_writes import SurpriseMeter, WritePolicy
from megatoken.transducer_service import TransducerInferenceEngine


@torch.inference_mode()
def probe(model, conversation, question="What is my code?", answer="731"):
    engine = TransducerInferenceEngine(MultiTrackGenerator(model))
    # Tiny presets otherwise exhaust their lookup budget before the delayed question.
    engine.bank.geometry = replace(
        engine.bank.geometry,
        max_lookups=max(engine.bank.geometry.max_lookups, 2 * len(conversation) + 8),
    )
    with tempfile.TemporaryDirectory(prefix="serving-writes-") as root:
        base = engine.create_overlay("probe", "probe", Path(root) / "memory.kb")
        session = engine.new_session(
            {}, overlay="probe", scope="probe", write_policy=WritePolicy(lag=4, max_events=256)
        )
        meter = SurpriseMeter()
        scores = []
        for row in conversation:
            event = ObservedEvent(
                "main", "text", row["text"].encode(), role_id=ROLE_TO_ID[row.get("role", "user")]
            )
            score = meter.score(engine.multi, session.state, event, effort=1)
            scores.append(dict(loss_nats_per_byte=score, novel=bool(row.get("novel", False))))
            engine.observe(session, (event,), effort=1)
        events = session.writes._ordinary()
        unique = sorted(set(events.groups.flatten().tolist()))
        reads = unique[1:]  # exclude initial group's same-group evidence
        lag_results = {}
        if reads:
            full = engine.native.retrospective_values(events, reads).values
            for lag in (0, 1, 2, 4):
                cosines = []
                for i, read in enumerate(reads):
                    cutoff = min(unique[-1], read + lag)
                    keep = events.groups[0] <= cutoff
                    partial = type(events)(
                        events.raw[:, keep],
                        events.roles[:, keep],
                        tuple(
                            m for m, k in zip(events.modalities, keep.tolist(), strict=True) if k
                        ),
                        tuple(t for t, k in zip(events.tracks, keep.tolist(), strict=True) if k),
                        events.meridians[:, keep],
                        events.groups[:, keep],
                    )
                    value = engine.native.retrospective_values(partial, [read]).values[0]
                    cosines.append(float(F.cosine_similarity(value, full[i], dim=0)))
                lag_results[str(lag)] = dict(
                    mean_cosine=sum(cosines) / len(cosines), comparisons=len(cosines)
                )
        finite = [(i, r) for i, r in enumerate(scores) if r["loss_nats_per_byte"] is not None]
        top = {
            i
            for i, _ in sorted(
                finite, key=lambda pair: pair[1]["loss_nats_per_byte"], reverse=True
            )[: math.ceil(len(finite) * 0.1)]
        }
        planted = [i for i, row in enumerate(scores) if row["novel"]]
        engine.close_session(session)
        written = base.edge_count
        engine.forget("probe", scope="probe")
        reopened = engine.open_base("reopened", base.path)
        recall = {}
        for rule in ("retrospective", "surprise"):
            store = engine.create_overlay(rule, rule, Path(root) / f"{rule}.kb")
            on = engine.new_session(
                {},
                overlay=rule,
                scope=rule,
                write_policy=WritePolicy(
                    retrospective=rule == "retrospective", surprise=rule == "surprise"
                ),
            )
            off = engine.new_session({})
            for row in conversation:
                event = ObservedEvent(
                    "main",
                    "text",
                    row["text"].encode(),
                    role_id=ROLE_TO_ID[row.get("role", "user")],
                )
                engine.observe(on, (event,), effort=1)
                engine.observe(off, (event,), effort=1)
            lookup = engine.observe(
                on, (ObservedEvent("main", "text", question.encode()),), effort=1
            )
            engine.observe(off, (ObservedEvent("main", "text", question.encode()),), effort=1)
            expected = ObservedEvent(
                "main", "text", answer.encode(), role_id=ROLE_TO_ID["assistant"]
            )
            loss_on = meter.score(engine.multi, on.state, expected, effort=1)
            loss_off = meter.score(engine.multi, off.state, expected, effort=1)
            recall[rule] = dict(
                written_edges=store.edge_count,
                selected_overlay_edges=(
                    sorted({e for _, n, e in lookup.pairs if n == rule})
                    if lookup is not None
                    else []
                ),
                answer_loss_with_writes=loss_on,
                answer_loss_without_writes=loss_off,
                loss_gain=loss_off - loss_on,
                input_groups=len(conversation),
                retained_event_limit=model.config.max_history_chunks,
            )
            engine.close_session(on)
            engine.close_session(off)
        return dict(
            recall=recall,
            surprise=scores,
            planted_facts_in_top_tenth=(
                sum(i in top for i in planted) / len(planted) if planted else None
            ),
            planted_fact_count=len(planted),
            lag=lag_results,
            memories_written=written,
            memories_after_forget=reopened.edge_count,
            definitions={
                "loss_nats_per_byte": "Lexical negative log likelihood divided by UTF-8 byte count; higher is less expected.",
                "planted_facts_in_top_tenth": "Fraction of marked novel messages in the highest-surprise 10% of messages; provisional target >= 0.8.",
                "lag": "Mean cosine to the completed-window retrospective value at each delay in retrieval groups; provisional target >= 0.9.",
                "recall": "For each rule, answer lexical loss in nats per byte with and without writes after the same inputs. Positive loss_gain is better; selected_overlay_edges should be nonempty. Inputs must push the planted fact beyond the retained event limit.",
                "memories_after_forget": "Live edges after deleting and reopening the private store; must be zero.",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--packs")
    parser.add_argument("--conversation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--question", default="What is my code?")
    parser.add_argument("--answer", default="731")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(4)
    if args.smoke:
        model = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback(), device=args.device)
    elif args.model:
        if (Path(args.model) / "manifest.json").exists():
            model = ServedBundle(args.model, packs=args.packs, device=args.device).model
        else:
            model = load_trained(args.model, packs=args.packs, device=args.device).model
    else:
        parser.error("Supply --model, or --smoke for a mechanics check")
    conversation = (
        [json.loads(line) for line in args.conversation.read_text().splitlines() if line]
        if args.conversation
        else [
            {"text": "Hello."},
            {"text": "Thanks."},
            {"text": "OK."},
            {"text": "My code is 731.", "novel": True},
            *[{"text": "Thanks."} for _ in range(16)],
        ]
    )
    report = dict(smoke=args.smoke, **probe(model.eval(), conversation, args.question, args.answer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("surprise", "definitions")}))


if __name__ == "__main__":
    main()
