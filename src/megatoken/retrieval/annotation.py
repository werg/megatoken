"""The data side of reading from a collection: what an episode says about its lookup.

A reading-from-a-collection episode has no supporting documents in its context.
They live in source collections that sit next to the episode's prepared root, and
the episode's supervision annotations (``annotations_json["retrieval"]``) say:

- ``collections``: the collection directories to mount, relative to the root;
- ``query_text`` and ``query``: the question, and the teacher's own embedding of
  it (a stored tensor asset in the prepared root), the target of the query head;
- ``positives``: the identities of the collection items that support the answer.

The annotated lookup is the one right after the user's question: a finished user
message is a lookup boundary, so that lookup has read the question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from megatoken.data.media_assets import TensorAsset
from megatoken.io import strict_json_loads

KEY = "retrieval"


@dataclass(frozen=True)
class RetrievalAnnotation:
    collections: tuple[str, ...]
    query_text: str
    query: TensorAsset
    positives: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.collections or not self.query_text:
            raise ValueError("A retrieval annotation names its collections and its question")
        if self.query.kind != "embedding" or self.query.modality != "text":
            raise ValueError("The query target is the teacher's embedding of the question text")
        if any(path.startswith("/") or ".." in path.split("/") for path in self.collections):
            raise ValueError("Collections are directories inside the prepared root")

    def to_json(self) -> dict[str, Any]:
        return {
            "collections": list(self.collections),
            "query_text": self.query_text,
            "query": asdict(self.query),
            "positives": list(self.positives),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> RetrievalAnnotation:
        return cls(
            tuple(value["collections"]),
            value["query_text"],
            TensorAsset.from_dict(value["query"]),
            tuple(value["positives"]),
        )

    @classmethod
    def of(cls, episode: Any) -> RetrievalAnnotation | None:
        """The episode's annotation, if it has one."""
        raw = episode.supervision.annotations_json
        # Other streams' annotations can be large; only parse what might hold ours.
        if f'"{KEY}"' not in raw:
            return None
        value = strict_json_loads(raw).get(KEY)
        return None if value is None else cls.from_json(value)
