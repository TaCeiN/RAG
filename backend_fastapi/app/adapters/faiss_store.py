from __future__ import annotations

import json
from pathlib import Path

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class FaissStore:
    def __init__(self, index_path: Path, meta_path: Path) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.items: list[dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if self.meta_path.exists():
            self.items = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.items = []
        if faiss is None:
            self.index = None
            return
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None
        if self.index is None and self.items:
            # Recover FAISS index from persisted metadata.
            self.rebuild_index()

    def _save(self) -> None:
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")

    def clear_chat(self, chat_id: str) -> None:
        self.items = [row for row in self.items if row["chat_id"] != chat_id]
        self.rebuild_index()

    def rebuild_index(self) -> None:
        if faiss is None:
            self.index = None
            self._save()
            return
        vectors = [row["vector"] for row in self.items]
        if not vectors:
            self.index = None
            self._save()
            return
        dim = len(vectors[0])
        index = faiss.IndexFlatIP(dim)
        import numpy as np

        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        index.add(matrix)
        self.index = index
        self._save()

    def add(
        self,
        chat_id: str,
        chunk_id: str,
        text: str,
        vector: list[float],
        metadata: dict[str, object] | None = None,
    ) -> None:
        row = {"chat_id": chat_id, "chunk_id": chunk_id, "text": text, "vector": vector}
        if metadata:
            row.update(metadata)
        self.items.append(row)
        self.rebuild_index()

    def total_items(self) -> int:
        return len(self.items)

    def replace_all(self, items: list[dict[str, object]]) -> None:
        self.items = [dict(row) for row in items]
        self.rebuild_index()

    def list_for_chat(self, chat_id: str) -> list[dict[str, object]]:
        return [dict(row) for row in self.items if row["chat_id"] == chat_id]

    def search(self, chat_id: str, query_vector: list[float], top_k: int) -> list[dict[str, object]]:
        rows = [row for row in self.items if row["chat_id"] == chat_id]
        if not rows:
            return []

        if faiss is None:
            return self._fallback_search(rows, query_vector, top_k)

        import numpy as np

        dim = len(query_vector)
        local_index = faiss.IndexFlatIP(dim)
        matrix = np.array([row["vector"] for row in rows], dtype="float32")
        faiss.normalize_L2(matrix)
        local_index.add(matrix)

        query = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, ids = local_index.search(query, min(top_k, len(rows)))

        hits: list[dict[str, object]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            row = rows[int(idx)]
            hits.append(
                {
                    "chat_id": row["chat_id"],
                    "chunk_id": row["chunk_id"],
                    "file_id": row.get("file_id"),
                    "file_name": row.get("file_name"),
                    "chunk_index": row.get("chunk_index"),
                    "text": row["text"],
                    "score": float(score),
                }
            )
        hits.sort(key=lambda x: (-float(x.get("score", 0.0)), str(x.get("chunk_id", ""))))
        return hits[:top_k]

    @staticmethod
    def _fallback_search(rows: list[dict[str, object]], query_vector: list[float], top_k: int) -> list[dict[str, object]]:
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            if len(a) != len(b):
                return 0.0
            num = sum(x * y for x, y in zip(a, b))
            den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            return 0.0 if den == 0 else num / den

        hits = []
        for row in rows:
            score = cosine(query_vector, list(row["vector"]))
            hits.append(
                {
                    "chat_id": row["chat_id"],
                    "chunk_id": row["chunk_id"],
                    "file_id": row.get("file_id"),
                    "file_name": row.get("file_name"),
                    "chunk_index": row.get("chunk_index"),
                    "text": row["text"],
                    "score": float(score),
                }
            )
        hits.sort(key=lambda x: (-float(x.get("score", 0.0)), str(x.get("chunk_id", ""))))
        return hits[:top_k]
