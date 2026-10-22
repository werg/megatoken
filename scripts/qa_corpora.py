"""HotpotQA (distractor) and MuSiQue-Ans loaders: questions with their paragraphs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

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
