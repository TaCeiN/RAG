import json
import unittest

from app.domain.service import RagService, _rule_intent_scores


class _FakeDb:
    def __init__(self):
        self._chats = [{"id": "c1", "title": "Chat"}]
        self._chunks = [
            {
                "id": "k1",
                "chat_id": "c1",
                "file_id": "f1",
                "chunk_index": 0,
                "text": "Алгоритмы FCFS, Round Robin и SJF рассматриваются в работе.",
                "vector_json": json.dumps([1.0, 0.0]),
            },
            {
                "id": "k2",
                "chat_id": "c1",
                "file_id": "f1",
                "chunk_index": 1,
                "text": "Алгоритмы FIFO, LRU и OPT используются для замещения страниц.",
                "vector_json": json.dumps([0.9, 0.1]),
            },
        ]

    def list_chats(self):
        return self._chats

    def list_chunks_for_chat(self, chat_id):
        return [dict(x) for x in self._chunks if x["chat_id"] == chat_id]

    def create_chat(self, title):
        return "c1"

    def list_messages(self, chat_id):
        return []

    def list_files(self, chat_id):
        return []

    def store_file(self, *args, **kwargs):
        return "f1"

    def update_file_status(self, *args, **kwargs):
        pass

    def update_file_text(self, *args, **kwargs):
        pass

    def store_chunk(self, *args, **kwargs):
        return "k3"

    def store_message(self, *args, **kwargs):
        return "m1"


class _FakeVectorStore:
    def __init__(self):
        self.items = []

    def total_items(self):
        return len(self.items)

    def replace_all(self, items):
        self.items = [dict(x) for x in items]

    def add(self, chat_id, chunk_id, text, vector):
        self.items.append({"chat_id": chat_id, "chunk_id": chunk_id, "text": text, "vector": vector})

    def list_for_chat(self, chat_id):
        return [dict(x) for x in self.items if x["chat_id"] == chat_id]

    def search(self, chat_id, query_vector, top_k):
        rows = [x for x in self.items if x["chat_id"] == chat_id]
        return [
            {"chat_id": row["chat_id"], "chunk_id": row["chunk_id"], "text": row["text"], "score": 0.9 - (idx * 0.1)}
            for idx, row in enumerate(rows[:top_k])
        ]


class _FakeEmbedder:
    def embed(self, text, keep_alive=-1):
        low = text.lower()
        if "срав" in low or "разниц" in low:
            return [0.0, 1.0]
        if "сумм" in low or "overview" in low:
            return [0.7, 0.7]
        return [1.0, 0.0]

    def embed_many(self, texts, keep_alive=-1):
        return [self.embed(text) for text in texts]


class _FakeGenerator:
    def stream_chat(self, messages, think=False, keep_alive=-1):
        yield {"message": {"content": "ok"}, "done": False}
        yield {"done": True}


class RouterTests(unittest.TestCase):
    def test_rule_scores_detect_compare_intent(self):
        scores = _rule_intent_scores("Сравни два алгоритма и покажи различия")
        self.assertGreater(scores["compare"], scores["qa"])

    def test_auto_router_sets_route_meta(self):
        service = RagService(
            db=_FakeDb(),
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            generator=_FakeGenerator(),
            reranker=None,
        )

        result = service._retrieve(
            chat_id="c1",
            user_question="Сравни FCFS и SJF",
            normalized_question="Сравни FCFS и SJF",
            retrieval_mode="auto",
            final_k=3,
        )
        route_meta = result["trace"].get("route_meta", {})
        self.assertEqual(route_meta.get("reason"), "auto_router")
        self.assertIn(route_meta.get("intent"), {"compare", "qa"})
        self.assertEqual(route_meta.get("selected_retrieval_mode"), "hybrid_plus")

    def test_auto_summary_query_routes_to_report_with_expanded_context_budget(self):
        service = RagService(
            db=_FakeDb(),
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            generator=_FakeGenerator(),
            reranker=None,
        )

        result = service._retrieve(
            chat_id="c1",
            user_question="Суммаризируй файл целиком",
            normalized_question="Суммаризируй файл целиком",
            retrieval_mode="auto",
            final_k=3,
        )
        route_meta = result["trace"].get("route_meta", {})
        self.assertEqual(route_meta.get("intent"), "report")
        self.assertEqual(route_meta.get("selected_retrieval_mode"), "hybrid_plus")
        self.assertGreaterEqual(int(route_meta.get("selected_recall_k", 0)), 100)


if __name__ == "__main__":
    unittest.main()
