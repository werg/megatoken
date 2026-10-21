"""Build a real retrieval-memory corpus from HotpotQA (distractor) and MuSiQue-Ans.

Outputs, under ``--output``:

- ``assets/``: immutable paragraph shards (``title\\ntext``); KB sources are byte ranges.
- ``teacher/``: EmbeddingGemma-2 document and query vectors (isolated chunk-wise).
- ``recipes/``: coherence_embedding, progressive_warmup and free_kb_pretraining recipes.
- ``catalog.jsonl`` + ``kb/catalog.sqlite``: the source catalog keyed by teacher vectors.
- ``datasets/{rc,open}-{train,heldout}``: prepared episodes (real Ling codec) and sidecars.
- ``kb/centroids.npy`` and the IVF publication in ``kb/catalog.sqlite``, its probe
  count calibrated on the held-out queries (``kb/catalog.sqlite.index.json``).

Two views of every question:

- ``rc`` (reading comprehension, P1): gold + distractor paragraphs are observed in
  meridian 0 with the question; the answer follows in meridian 1.
- ``open`` (M1/M2): only distractors are observed; gold paragraphs live solely in
  the KB, so the evidence must be retrieved.

Held-out questions come from the official dev splits and never enter training data.
Their paragraphs are part of the KB corpus (a corpus, not labels).
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

QUERY_PROMPT = "task: question answering | query: "
DOCUMENT_PROMPT = "title: {title} | text: "
SCOPE = "wiki-qa"
MAX_DOCS = 5
MAX_ANSWER_BYTES = 360
MAX_PARAGRAPH_BYTES = 1536


def _paragraph(title: str, text: str) -> dict:
    text = " ".join(text.split())
    payload = f"{title}\n{text}".encode()
    if len(payload) > MAX_PARAGRAPH_BYTES:
        cut = payload[:MAX_PARAGRAPH_BYTES].decode("utf-8", "ignore")
        payload = cut[: cut.rfind(" ")].encode() if " " in cut else cut.encode()
    return {
        "title": title,
        "payload": payload,
        "content_id": "p-" + sha256(payload).hexdigest()[:24],
    }


def _clip(text: str, limit: int) -> str:
    data = " ".join(text.split()).encode()
    if len(data) <= limit:
        return data.decode()
    cut = data[:limit].decode("utf-8", "ignore")
    return cut[: cut.rfind(" ")] if " " in cut else cut


def load_hotpot(root: Path, split: str) -> list[dict]:
    """HotpotQA distractor parquet shards (``hotpotqa/hotpot_qa`` ``distractor/``)."""
    import pyarrow.parquet as pq

    files = sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No HotpotQA {split} shards under {root}")
    questions = []
    for file in files:
        for row in pq.read_table(file).to_pylist():
            titles, sentences = row["context"]["title"], row["context"]["sentences"]
            gold_titles = set(row["supporting_facts"]["title"])
            paragraphs, gold, by_title = [], [], {}
            for title, sents in zip(titles, sentences, strict=True):
                item = _paragraph(title, "".join(sents))
                by_title[title] = sents
                paragraphs.append(item)
                if title in gold_titles:
                    gold.append(item["content_id"])
            evidence = " ".join(
                by_title[t][i].strip()
                for t, i in zip(
                    row["supporting_facts"]["title"],
                    row["supporting_facts"]["sent_id"],
                    strict=True,
                )
                if t in by_title and i < len(by_title[t])
            )
            questions.append(
                {
                    "qid": f"hotpot-{row['id']}",
                    "question": " ".join(row["question"].split()),
                    "answer": _clip(f"{row['answer']}\nEvidence: {evidence}", MAX_ANSWER_BYTES),
                    "paragraphs": paragraphs,
                    "gold": gold,
                }
            )
    return questions


def load_musique(root: Path, split: str) -> list[dict]:
    """MuSiQue-Ans JSONL (``dgslibisey/MuSiQue``)."""
    path = root / f"musique_ans_v1.0_{split}.jsonl"
    questions = []
    for line in open(path):
        row = json.loads(line)
        if not row.get("answerable", True):
            continue
        paragraphs = [_paragraph(p["title"], p["paragraph_text"]) for p in row["paragraphs"]]
        gold = [
            item["content_id"]
            for item, p in zip(paragraphs, row["paragraphs"], strict=True)
            if p["is_supporting"]
        ]
        steps = "; ".join(step["answer"] for step in row["question_decomposition"])
        questions.append(
            {
                "qid": f"musique-{row['id']}",
                "question": " ".join(row["question"].split()),
                "answer": _clip(f"{row['answer']}\nSteps: {steps}", MAX_ANSWER_BYTES),
                "paragraphs": paragraphs,
                "gold": gold,
            }
        )
    return questions


def select(questions: list[dict], count: int, rng: random.Random) -> list[dict]:
    usable = [
        q
        for q in questions
        if q["question"]
        and q["gold"]
        and len(set(q["gold"])) == len(q["gold"])
        and len(q["gold"]) < MAX_DOCS
    ]
    rng.shuffle(usable)
    return sorted(usable[:count], key=lambda q: q["qid"])


class Teacher:
    """EmbeddingGemma-2: mean pooling over the prompted input, then L2 normalization."""

    def __init__(self, path: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16).to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts: list[str], *, batch_tokens: int = 32768) -> np.ndarray:
        torch = self.torch
        lengths = [len(t) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        out = np.zeros((len(texts), 768), dtype=np.float32)
        start = 0
        while start < len(order):
            longest = max(1, lengths[order[min(start + 255, len(order) - 1)]] // 3)
            size = max(1, min(256, batch_tokens // longest))
            ids = order[start : start + size]
            batch = self.tokenizer(
                [texts[i] for i in ids],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out[ids] = pooled.cpu().numpy()
            start += size
        return out


def encode_resumable(
    teacher: Callable[[], Teacher],
    texts: list[str],
    directory: Path,
    *,
    chunk: int = 16384,
) -> np.ndarray:
    """Encode in fixed chunks, each published atomically; reruns skip finished chunks."""
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, start in enumerate(range(0, len(texts), chunk)):
        path = directory / f"{index:05d}.npy"
        if not path.exists():
            staging = directory / f".{index:05d}.npy"
            np.save(staging, teacher().encode(texts[start : start + chunk]))
            staging.replace(path)
            print(f"{directory.name}: {min(start + chunk, len(texts))}/{len(texts)}", flush=True)
        parts.append(np.load(path))
    return np.concatenate(parts) if parts else np.zeros((0, 768), dtype=np.float32)


def build(args: argparse.Namespace) -> dict:
    from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
    from megatoken.data.document_embeddings import (
        DocumentEmbeddingExample,
        write_document_sidecar,
    )
    from megatoken.data.retrieval import RetrievalExample, write_sidecar
    from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
    from megatoken.data.storage import export_prepared
    from megatoken.io import sha256_file
    from megatoken.loading import load_codec, load_model_config
    from megatoken.memory.catalog_cli import build_source_catalog
    from megatoken.memory.contracts import Footprint, QueryBoundary, SemanticSpace, SourceRange
    from megatoken.memory.recipe import RetrievalRecipe

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    splits = {
        "train": select(load_hotpot(args.hotpot, "train"), args.hotpot_train, rng)
        + select(load_musique(args.musique, "train"), args.musique_train, rng),
        "heldout": select(load_hotpot(args.hotpot, "validation"), args.hotpot_heldout, rng)
        + select(load_musique(args.musique, "dev"), args.musique_heldout, rng),
    }
    print({k: len(v) for k, v in splits.items()}, flush=True)

    # ---- immutable paragraph assets (the KB corpus) ----
    paragraphs: dict[str, dict] = {}
    windows: dict[str, list[str]] = {}
    for split in splits.values():
        for q in split:
            windows[q["qid"]] = sorted({p["content_id"] for p in q["paragraphs"]})
            for p in q["paragraphs"]:
                paragraphs.setdefault(p["content_id"], p)
    content_ids = sorted(paragraphs)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    spans: dict[str, tuple[str, int, int]] = {}
    shard, handle, offset = -1, None, 0
    for cid in content_ids:
        if handle is None or offset >= 64 << 20:
            if handle is not None:
                handle.close()
            shard += 1
            name = f"shard-{shard:05d}.txt"
            handle, offset = open(assets / name, "wb"), 0
        payload = paragraphs[cid]["payload"]
        handle.write(payload + b"\n")
        spans[cid] = (name, offset, offset + len(payload))
        offset += len(payload) + 1
    handle.close()
    digests = {
        name: sha256_file(assets / name) for name in sorted({a for a, _, _ in spans.values()})
    }
    print("paragraphs", len(content_ids), "shards", len(digests), flush=True)

    # ---- teacher vectors ----
    teacher_dir = out / "teacher"
    teacher_dir.mkdir(exist_ok=True)
    queries = [q for split in splits.values() for q in split]
    selection = {"documents": content_ids, "queries": [q["qid"] for q in queries]}
    manifest = teacher_dir / "ids.json"
    if manifest.exists() and json.loads(manifest.read_text()) != selection:
        raise ValueError("Cached teacher vectors belong to a different selection")
    manifest.write_text(json.dumps(selection))
    loaded: list[Teacher] = []

    def teacher() -> Teacher:
        if not loaded:
            loaded.append(Teacher(args.teacher))
        return loaded[0]

    doc_texts = []
    for cid in content_ids:
        title, _, body = paragraphs[cid]["payload"].decode().partition("\n")
        doc_texts.append(DOCUMENT_PROMPT.format(title=title) + body)
    doc_vectors = dict(
        zip(
            content_ids,
            encode_resumable(teacher, doc_texts, teacher_dir / "documents"),
            strict=True,
        )
    )
    query_vectors = dict(
        zip(
            selection["queries"],
            encode_resumable(
                teacher, [QUERY_PROMPT + q["question"] for q in queries], teacher_dir / "queries"
            ),
            strict=True,
        )
    )
    loaded.clear()

    # ---- recipes ----
    pilot = json.loads(Path(args.pilot_recipe).read_text())
    # The pilot recipe pins the teacher space; the prompts above implement its recipes.
    space = pilot["space"]
    space_fp = SemanticSpace(**space).fingerprint
    partition = sha256(
        json.dumps({k: [q["qid"] for q in v] for k, v in splits.items()}).encode()
    ).hexdigest()
    model_config = load_model_config(args.model, **json.loads(args.model_overrides))
    recipes = {}
    for phase in ("coherence_embedding", "progressive_warmup", "free_kb_pretraining"):
        recipe = {
            **pilot,
            "phase": phase,
            "space": space,
            "model": model_config.to_dict(),
            "growth": {
                **pilot["growth"],
                "value_dim": model_config.semantic_value_dim,
                "partition_id": partition,
            },
            "max_kb_positions": model_config.max_kb_positions,
        }
        path = out / "recipes" / f"{phase}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(recipe, indent=2, sort_keys=True))
        recipes[phase] = str(path)

    # ---- source catalog + bank ----
    def kb_range(cid: str) -> SourceRange:
        asset, start, end = spans[cid]
        return SourceRange(
            cid, "v1", start, end, sha256(paragraphs[cid]["payload"]).hexdigest(), SCOPE
        )

    catalog = out / "catalog.jsonl"
    if not catalog.exists():
        membership: dict[str, list[str]] = {}
        for qid, members in windows.items():
            for cid in members:
                membership.setdefault(cid, []).append(qid)
        with open(catalog, "w") as file:
            for cid in content_ids:
                span = kb_range(cid)
                file.write(
                    json.dumps(
                        {
                            "semantic_space_fingerprint": space_fp,
                            "scope": SCOPE,
                            "content_id": cid,
                            "asset_ref": spans[cid][0],
                            "asset_sha256": digests[spans[cid][0]],
                            "footprint": [asdict(span)],
                            "evidence_span": asdict(span),
                            "key": [float(x) for x in doc_vectors[cid]],
                            "source_windows": sorted(membership[cid]),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    snapshot = sha256_file(catalog)
    bank = out / "kb" / "catalog.sqlite"
    if not bank.exists():
        report = build_source_catalog(
            RetrievalRecipe.load(recipes["coherence_embedding"]),
            catalog=catalog,
            assets_root=assets,
            output=bank,
        )
        print("catalog", report["sources"], "windows", report["independent_source_windows"])

    # ---- episodes, prepared datasets and sidecars ----
    codec = load_codec(args.codec, vocab_size=model_config.vocab_size)
    summary = {}
    for split, items in splits.items():
        for view in ("rc", "open"):
            root = out / "datasets" / f"{view}-{split}"
            if (root / "data" / "manifest.json").exists():
                continue
            # The manifest is the commit marker: anything without one is partial.
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True)
            episodes, query_rows, doc_rows = [], [], []
            for q in items:
                local = random.Random(f"{args.seed}:{q['qid']}")
                gold = [paragraphs[c] for c in q["gold"]]
                distractors = [p for p in q["paragraphs"] if p["content_id"] not in q["gold"]]
                local.shuffle(distractors)
                if view == "rc":
                    observed = gold + distractors[: MAX_DOCS - len(gold)]
                else:
                    observed = distractors[: MAX_DOCS - 1]
                local.shuffle(observed)
                eid = f"{q['qid']}-{view}"
                records, names = [], []
                for index, p in enumerate(observed):
                    names.append(f"doc{index}")
                    records.append(PhysicalRecord(names[-1], "document", p["payload"]))
                request = q["question"].encode()
                answer = q["answer"].encode()
                names += ["request", "answer"]
                records += [
                    PhysicalRecord("request", "user", request),
                    PhysicalRecord("answer", "assistant", answer, turn_index=1, policy_mask=True),
                ]
                meridians = (
                    Meridian(
                        0,
                        tuple(
                            TrackEvent(n, n, "text", 0, r.role)
                            for n, r in zip(names[:-1], records[:-1], strict=True)
                        ),
                    ),
                    Meridian(1, (TrackEvent("answer", "answer", "text", 1, "assistant"),)),
                )
                episodes.append(
                    Episode(
                        eid,
                        tuple(records),
                        Lineage(q["qid"].split("-")[0], "v1", q["qid"]),
                        tracks=tuple(
                            TrackSpec(n, r.role) for n, r in zip(names, records, strict=True)
                        ),
                        meridians=meridians,
                    )
                )

                def episode_range(name: str, payload: bytes, eid: str = eid) -> SourceRange:
                    return SourceRange(
                        f"{eid}/{name}", "v1", 0, len(payload), sha256(payload).hexdigest(), SCOPE
                    )

                support = Footprint(tuple(kb_range(c) for c in q["gold"]))
                query_rows.append(
                    RetrievalExample(
                        episode_id=eid,
                        event_id="answer",
                        query=QueryBoundary(
                            "answer",
                            1,
                            "rag",
                            "request",
                            sha256(request).hexdigest(),
                            space_fp,
                            snapshot,
                            1,
                        ),
                        query_teacher_inputs=Footprint((episode_range("request", request),)),
                        document_key_context=support,
                        optimization_supervision=support,
                        payload_support=support,
                        positive_content_ids=tuple(q["gold"]),
                        support_weights=tuple(1.0 for _ in q["gold"]),
                        annotations_complete=True,
                        source_snapshot=snapshot,
                        split_family=q["qid"],
                        teacher_target=tuple(float(x) for x in query_vectors[q["qid"]]),
                        main_request_b64=base64.b64encode(request).decode(),
                    )
                )
                for name, p in zip(names, observed, strict=False):
                    span = episode_range(name, p["payload"])
                    doc_rows.append(
                        DocumentEmbeddingExample(
                            eid,
                            name,
                            sha256(p["payload"]).hexdigest(),
                            Footprint((span,)),
                            span,
                            tuple(float(x) for x in doc_vectors[p["content_id"]]),
                            space["document_recipe"],
                        )
                    )
            training = split == "train"
            store = DedupStore(
                root / "spool.sqlite",
                DedupConfig(
                    train_fraction=1.0 if training else 0.0,
                    validation_fraction=0.0 if training else 1.0,
                    near_threshold=1.0,
                ),
            )
            try:
                with store.transaction():
                    for episode in episodes:
                        store.add(episode, codec)
                store.freeze()
                manifest = export_prepared(
                    store,
                    root / "data",
                    codec=codec,
                    source_receipts=[],
                    pipeline_fingerprint=f"retrieval-qa-{view}-{partition[:16]}",
                )
            finally:
                store.close()
            (root / "spool.sqlite").unlink(missing_ok=True)
            write_sidecar(root / "queries.jsonl", query_rows)
            write_document_sidecar(root / "documents.jsonl", doc_rows)
            summary[f"{view}-{split}"] = {
                "episodes": len(episodes),
                "queries": len(query_rows),
                "documents": len(doc_rows),
                "counts": manifest["counts"],
            }
            print(f"{view}-{split}", summary[f"{view}-{split}"], flush=True)
    # ---- IVF centroids and the calibrated index publication ----
    index = publish_ivf(
        out, RetrievalRecipe.load(recipes["coherence_embedding"]), bank, args.ivf_cells
    )
    result = {
        "recipes": recipes,
        "bank": str(bank),
        "ivf": {
            "centroids": str(out / "kb" / "centroids.npy"),
            "centroids_sha256": index["centroid_file_sha256"],
            "nprobe": index["nprobe"],
        },
        "assets": str(assets),
        "catalog_snapshot": snapshot,
        "partition_id": partition,
        "paragraphs": len(content_ids),
        "splits": {k: len(v) for k, v in splits.items()},
        "datasets": summary,
    }
    (out / "corpus.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def publish_ivf(out: Path, recipe, bank: Path, cells: int) -> dict:
    """Fit pinned centroids and publish the IVF index, its probe count calibrated on
    the held-out teacher queries. Idempotent: an existing receipt is kept."""
    from megatoken.data.retrieval import read_sidecar
    from megatoken.io import sha256_file
    from megatoken.memory.centroids_cli import fit_centroids
    from megatoken.memory.index_cli import publish_index

    centroids = out / "kb" / "centroids.npy"
    if not centroids.exists():
        fit_centroids(recipe, bank_path=bank, output=centroids, cells=cells)
    receipt_path = bank.with_suffix(bank.suffix + ".index.json")
    if not receipt_path.exists():
        queries = np.stack(
            [
                np.asarray(row.teacher_target, dtype=np.float32)
                for view in ("rc", "open")
                for row in read_sidecar(out / "datasets" / f"{view}-heldout" / "queries.jsonl")
            ]
        )
        receipt = publish_index(
            recipe,
            bank_path=bank,
            centroid_path=centroids,
            centroid_sha256=sha256_file(centroids),
            queries=queries,
        )
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return json.loads(receipt_path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--hotpot", required=True, type=Path, help="HotpotQA distractor/ dir")
    parser.add_argument("--musique", required=True, type=Path, help="MuSiQue-Ans JSONL dir")
    parser.add_argument("--teacher", required=True, help="Local embeddinggemma-2 snapshot")
    parser.add_argument("--codec", required=True, help="Pinned Ling codec directory")
    parser.add_argument("--pilot-recipe", default="configs/retrieval-memory-pilot.json")
    parser.add_argument("--model", default="32x512")
    parser.add_argument("--model-overrides", default="{}")
    parser.add_argument("--hotpot-train", type=int, default=20000)
    parser.add_argument("--musique-train", type=int, default=5000)
    parser.add_argument("--hotpot-heldout", type=int, default=1000)
    parser.add_argument("--musique-heldout", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--ivf-cells", type=int, default=256)
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
