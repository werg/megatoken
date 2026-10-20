"""Prepare retrieval supervision from pinned teacher vector exports.

This is a strict importer, not an online teacher or an ungrounded embedding
ingester. Every query/document label must correspond to a verified episode
event in the already checksummed prepared corpus.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from megatoken.data.document_embeddings import (
    DocumentEmbeddingExample,
    write_document_sidecar,
)
from megatoken.data.retrieval import RetrievalExample, write_sidecar
from megatoken.data.storage import PreparedDataset
from megatoken.memory.recipe import RetrievalRecipe


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as input:
        for line_number, raw in enumerate(input, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield item


def _canonical_query(row: dict) -> dict:
    result = dict(row)
    for name in (
        "query_teacher_inputs", "document_key_context",
        "optimization_supervision", "payload_support",
    ):
        if isinstance(result.get(name), list):
            result[name] = json.dumps(result[name], sort_keys=True)
    return result


def _canonical_document(row: dict) -> dict:
    result = dict(row)
    if isinstance(result.get("teacher_inputs"), list):
        result["teacher_inputs"] = json.dumps(result["teacher_inputs"], sort_keys=True)
    return result


def prepare_supervision(
    *, recipe: RetrievalRecipe, prepared_dataset: str | Path,
    query_jsonl: str | Path, document_jsonl: str | Path,
    destination: str | Path, source_snapshot: str,
) -> dict[str, object]:
    if (recipe.space.teacher_revision.startswith("PIN_")
            or recipe.growth.partition_id.startswith("PIN_")):
        raise ValueError("Pin teacher weight/preprocessing and source partition first")
    if not source_snapshot or source_snapshot.startswith("PIN_"):
        raise ValueError("The canonical source catalog snapshot is required")
    queries = [
        RetrievalExample.from_dict(_canonical_query(x)) for x in _rows(Path(query_jsonl))
    ]
    documents = [
        DocumentEmbeddingExample.from_dict(_canonical_document(x))
        for x in _rows(Path(document_jsonl))
    ]
    if not queries or not documents:
        raise ValueError("Both causal queries and original-source document targets are required")
    q_lookup = {}
    d_lookup = {}
    for entry in queries:
        identity = (entry.episode_id, entry.event_id)
        if identity in q_lookup:
            raise ValueError(f"Duplicated query supervision event: {identity}")
        if entry.source_snapshot != source_snapshot:
            raise ValueError("Query source snapshot differs from original catalog")
        if entry.query.space_fingerprint != recipe.space.fingerprint:
            raise ValueError("Query belongs to another learned semantic space")
        if entry.teacher_target is None or (
            len(entry.teacher_target) != recipe.space.dimension
        ):
            raise ValueError("Wrong teacher query dimension for learned space")
        entry.request_content()
        q_lookup[identity] = entry
    for item in documents:
        identity = (item.episode_id, item.event_id)
        if identity in d_lookup:
            raise ValueError(f"Duplicate document supervision event: {identity}")
        if (len(item.teacher_target) != recipe.space.dimension or
                item.preprocessing_revision != recipe.space.document_recipe):
            raise ValueError("Document teacher role/preprocessing/dimension mismatch")
        d_lookup[identity] = item
    found_q, found_d = set(), set()
    with PreparedDataset(prepared_dataset, split="train", verify_checksums=True) as data:
        for episode in data.iter_episodes():
            for ordinal, record in enumerate(episode.records):
                identity = (episode.episode_id, record.record_id)
                annotation = q_lookup.get(identity)
                if annotation is not None:
                    if record.role != "assistant" or not record.policy_mask:
                        raise ValueError("Retrieval query annotation must attach to a generated action")
                    request = annotation.request_content()
                    if not any(
                        before.role in {"user", "system"} and before.payload == request
                        for before in episode.records[:ordinal]
                    ):
                        raise ValueError("Main request was not in the completed causal prefix")
                    found_q.add(identity)
                doc = d_lookup.get(identity)
                if doc is not None:
                    if record.role != "document" or (
                        sha256(record.payload).hexdigest() != doc.payload_sha256
                    ):
                        raise ValueError("Document label is not the original prepared source event")
                    # Verify source coordinates against this exact physical chunk
                    # before model training checks the actual enriched native clock.
                    doc.validate_event(
                        payload=record.payload,
                        clock=max(x.available_at for x in doc.teacher_inputs.ranges),
                    )
                    found_d.add(identity)
    if found_q != set(q_lookup) or found_d != set(d_lookup):
        raise ValueError(
            f"Orphaned teacher labels: query={len(set(q_lookup)-found_q)}, "
            f"document={len(set(d_lookup)-found_d)}"
        )
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    q_path = root / "retrieval-queries.jsonl"
    d_path = root / "retrieval-documents.jsonl"
    if q_path.exists() or d_path.exists():
        raise FileExistsError("Refusing to replace immutable prepared sidecars")
    qm = write_sidecar(q_path, queries)
    dm = write_document_sidecar(d_path, documents)
    return {
        "query_file": str(q_path), "document_file": str(d_path),
        "queries": len(queries), "documents": len(documents),
        "source_snapshot": source_snapshot,
        "semantic_space": recipe.space.fingerprint,
        "query_sha256": qm["sha256"], "document_sha256": dm["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and publish original-source, role-asymmetric teacher sidecars"
    )
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--prepared-dataset", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = prepare_supervision(
        recipe=RetrievalRecipe.load(args.recipe),
        prepared_dataset=args.prepared_dataset,
        query_jsonl=args.queries, document_jsonl=args.documents,
        destination=args.output, source_snapshot=args.source_snapshot,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
