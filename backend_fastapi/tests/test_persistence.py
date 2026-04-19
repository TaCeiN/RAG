import json
import tempfile
import unittest
from pathlib import Path

from app.adapters.faiss_store import FaissStore
from app.db.sqlite_store import SqliteStore
from app.domain.service import RagService


class _FakeEmbedder:
    def embed(self, text: str, keep_alive=-1):
        if "alpha" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_many(self, texts, keep_alive=-1):
        return [self.embed(t) for t in texts]


class _FakeGenerator:
    def stream_chat(self, messages, think=False, keep_alive=-1):
        yield {"message": {"content": "ok"}, "done": False}
        yield {"done": True}


class PersistenceTests(unittest.TestCase):
    def test_rebuilds_vector_store_from_sqlite_on_startup_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "rag.sqlite3"
            index_path = base / "faiss.index"
            meta_path = base / "faiss_meta.json"

            db = SqliteStore(db_path)
            chat_id = db.create_chat("Test")
            file_id = db.store_file(chat_id, "a.txt", "alpha", status="ready")
            db.store_chunk(chat_id, file_id, 0, "alpha fact", json.dumps([1.0, 0.0]))
            db.close()

            # Simulate restart with missing vector artifacts.
            store = FaissStore(index_path=index_path, meta_path=meta_path)
            self.assertEqual(store.total_items(), 0)

            db2 = SqliteStore(db_path)
            service = RagService(
                db=db2,
                vector_store=store,
                embedder=_FakeEmbedder(),
                generator=_FakeGenerator(),
                reranker=None,
            )

            self.assertGreater(service.vector_store.total_items(), 0)
            retrieval = service._retrieve(chat_id, "alpha?", "alpha?", "hybrid_plus", 1)
            self.assertTrue(retrieval["final"])
            self.assertIn("alpha", str(retrieval["final"][0]["text"]).lower())

            db2.close()


if __name__ == "__main__":
    unittest.main()
