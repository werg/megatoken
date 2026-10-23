"""Durable decoded-output evaluation, separate from training targets and losses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from megatoken.io import sha256_file, write_json
from megatoken.omni.bundles import RELATIONS, Relation
from megatoken.omni.concordance import score_sampled_relations
from megatoken.omni.evaluation import MetricAccumulator, character_error_rate, word_error_rate


def canonical_decoded_array(modality, array):
    """Remove the declared singleton batch/frame axes, without guessing layout."""
    values = np.asarray(array)
    if modality == "image" and values.ndim == 5 and values.shape[:2] == (1, 1):
        values = values[0, 0]  # Sana uses one displayed frame even for an image.
    elif modality == "audio" and values.ndim == 3 and values.shape[:2] == (1, 1):
        values = values[0, 0]
    expected = {"audio": 1, "image": 3, "video": 4}[modality]
    if values.ndim == expected + 1 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != expected:
        raise ValueError("Decoded event has an unsupported batch or channel layout")
    if modality == "image" and values.shape[0] != 3 or modality == "video" and values.shape[1] != 3:
        raise ValueError("Visual verification requires channel-first RGB pixels")
    return values


class DecodedQualityArchive:
    """Archive exactly sampled arrays/text before buffers are released.

    References are accessed only in this post-generation callback. Word/character
    agreement is a reproducible reference diagnostic, not a semantic adequacy
    judgment. Independent learned verifiers require recorded version/provenance.
    """

    def __init__(self, root, *, evaluator_metadata, speech_recognizer=None, pair_verifiers=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        if not isinstance(evaluator_metadata, dict) or not evaluator_metadata:
            raise ValueError("Decoded evaluation requires evaluator provenance")
        if speech_recognizer is not None and not evaluator_metadata.get("speech_recognizer"):
            raise ValueError("Record the independent recognizer identity and sample-rate policy")
        if pair_verifiers and not evaluator_metadata.get("pair_verifiers"):
            raise ValueError("Record each independent pair verifier's identity")
        self.metadata = dict(evaluator_metadata)
        self.speech_recognizer = speech_recognizer
        self.pair_verifiers = pair_verifiers
        self.seen = set()
        write_json(self.root / "evaluator.json", self.metadata)

    def record_generation(self, episode, generated):
        """Keep exact bytes/probabilities even when text is not valid UTF-8."""
        directory = self.root / hashlib.sha256(episode.episode_id.encode()).hexdigest()
        directory.mkdir(exist_ok=True)
        path = directory / f"selected-{generated.tape.meridian_id}.json"
        if path.exists():
            raise ValueError("A sampled decision cannot overwrite an earlier archived decision")
        write_json(path, json.loads(generated.tape.to_json()))

    def __call__(self, episode, outputs):
        identity = getattr(episode, "episode_id", None)
        if identity is None:
            raise ValueError("Quality archives require a stable episode identity")
        if identity in self.seen:
            raise ValueError("An evaluated episode cannot silently overwrite earlier samples")
        self.seen.add(identity)
        directory = self.root / hashlib.sha256(identity.encode()).hexdigest()
        directory.mkdir(exist_ok=True)
        assets = {}
        references = {}
        for record in episode.records:
            if (
                record.policy_mask
                and not record.action_id
                and record.media_type.startswith("text/")
            ):
                try:
                    references[record.record_id] = record.payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    pass
        metrics = MetricAccumulator()
        controls = []
        records = {r.record_id: r for r in episode.records}
        for event_id, sampled in outputs.items():
            if event_id not in records or not records[event_id].policy_mask:
                raise ValueError("A decoded sample must name an actually generated target")
            basename = hashlib.sha256(event_id.encode()).hexdigest()
            if sampled.modality == "text":
                path = directory / (basename + ".txt")
                path.write_bytes(sampled.decoded.encode("utf-8"))
                record = records[event_id]
                if not record.action_id:
                    try:
                        reference = record.payload.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        pass
                    else:
                        references[event_id] = reference
                        metrics.record(
                            "reference_word_error", word_error_rate(reference, sampled.decoded)
                        )
                        metrics.record(
                            "reference_character_error",
                            character_error_rate(reference, sampled.decoded),
                        )
                        wrong = "__deliberately_unrelated_output__"
                        if reference == wrong:
                            wrong += " different"
                        positive = word_error_rate(reference, reference)
                        negative = word_error_rate(reference, wrong)
                        controls.append(
                            {
                                "event_id": event_id,
                                "matching_reference_word_error": positive,
                                "wrong_text_word_error": negative,
                                "separates": negative > positive,
                            }
                        )
            else:
                path = directory / (basename + ".npy")
                np.save(path, np.asarray(sampled.decoded), allow_pickle=False)
            assets[event_id] = {
                "path": str(path.relative_to(self.root)),
                "sha256": sha256_file(path),
                "modality": sampled.modality,
            }
        declared = {
            (r.kind, r.left, r.right): Relation(r.kind, r.left, r.right, r.weight)
            for r in getattr(episode, "omni_relations", ())
        }
        for row in json.loads(episode.supervision.annotations_json).get("relations", ()):
            if row.get("kind") in RELATIONS:
                key = (row["kind"], row["left"], row["right"])
                declared.setdefault(key, Relation(*key, weight=row.get("weight", 1.0)))
        relations = tuple(declared.values())
        concordance = score_sampled_relations(
            relations,
            outputs,
            speech_recognizer=self.speech_recognizer,
            pair_verifiers=self.pair_verifiers,
        )
        report = {
            "format": "decoded-quality/v1",
            "episode_id": identity,
            "family_id": episode.lineage.family_id,
            "split": episode.lineage.frozen_split,
            "sampled_assets": assets,
            "references_post_generation_only": references,
            "generated_targets_without_decoded_sample": [
                r.record_id for r in episode.records if r.policy_mask and r.record_id not in outputs
            ],
            "reference_agreement": metrics.report(),
            "negative_controls": controls,
            "sampled_peer_concordance": asdict(concordance),
            "declared_relations": [asdict(r) for r in relations],
            "semantic_adequacy_established": False,
            "human_review": {"status": "pending", "reviewed_assets": []},
        }
        write_json(directory / "report.json", report)
        return report
