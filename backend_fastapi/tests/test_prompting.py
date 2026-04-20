import json
import re
import unittest

from app.domain.service import RagService, _build_context_blocks, _fit_context_to_token_budget, _with_source_labels


class PromptingTests(unittest.TestCase):
    def test_source_labels_are_human_readable(self) -> None:
        hits = [
            {"chunk_id": "abc-uuid-1", "file_name": "a.docx", "text": "Первый факт"},
            {"chunk_id": "abc-uuid-2", "file_name": "b.docx", "text": "Второй факт"},
        ]

        labeled = _with_source_labels(hits)
        context = _build_context_blocks(labeled)

        self.assertEqual(labeled[0]["source_label"], "a.docx")
        self.assertEqual(labeled[1]["source_label"], "b.docx")
        self.assertIn("[Файл: a.docx]", context)
        self.assertIn("[Файл: b.docx]", context)
        self.assertNotIn("abc-uuid-1", context)

    def test_context_includes_readable_file_metadata_when_available(self) -> None:
        hits = [
            {"chunk_id": "abc-uuid-1", "file_name": "report.docx", "chunk_index": 2, "text": "Первый факт"},
        ]

        context = _build_context_blocks(_with_source_labels(hits))

        self.assertIn("[Файл: report.docx]", context)
        self.assertIn("Раздел: фрагмент 3", context)
        self.assertNotIn("abc-uuid-1", context)

    def test_chat_messages_enforce_russian_output_and_source_labels(self) -> None:
        context = "[Файл: report.docx]\nТекст"
        messages = RagService._build_chat_messages("Что в документе?", context)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Final answer language must be Russian", messages[0]["content"])
        self.assertIn("reference file names", messages[0]["content"])
        self.assertIn("Do not force any fixed response template", messages[0]["content"])
        self.assertNotIn("Краткий ответ", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("CONTEXT_BLOCKS", messages[1]["content"])

    def test_report_intent_adds_whole_document_coverage_hint(self) -> None:
        context = "[Файл: report.docx]\nТекст"
        messages = RagService._build_chat_messages("Подготовь подробный отчет", context, intent="report")

        self.assertIn("whole-document coverage", messages[0]["content"])

    def test_followup_rag_prompt_uses_rewritten_question_and_history(self) -> None:
        db = _FakeDbWithHistory(
            messages=[
                {"role": "user", "content": "Какие алгоритмы планирования процессов использовались?"},
                {"role": "assistant", "content": "Использовались FCFS, RR, SJF и приоритетное планирование."},
            ]
        )
        generator = _CapturingGenerator()
        service = RagService(
            db=db,
            vector_store=_FakeVectorStore(),
            embedder=_FakeEmbedder(),
            generator=generator,
            reranker=None,
        )

        events = list(
            service.stream_answer(
                chat_id="c1",
                question="Подробнее",
                think=False,
                debug_retrieval=False,
                retrieval_mode="hybrid_plus",
                top_k=1,
            )
        )

        self.assertEqual(events[-1]["type"], "done")
        prompt = generator.messages[-1]["content"]
        self.assertIn("CONVERSATION_HISTORY", prompt)
        self.assertIn("Какие алгоритмы планирования процессов использовались?", prompt)
        self.assertIn("Использовались FCFS, RR, SJF", prompt)
        self.assertIn("QUESTION:\nКакие алгоритмы планирования процессов использовались?. Подробнее", prompt)

    def test_context_budget_keeps_prompt_under_configured_token_limit(self) -> None:
        hits = _with_source_labels(
            [
                {"chunk_id": "k1", "text": " ".join(f"alpha{i}" for i in range(80))},
                {"chunk_id": "k2", "text": " ".join(f"beta{i}" for i in range(80))},
            ]
        )

        context, used_hits = _fit_context_to_token_budget(hits, intent="qa", max_tokens=35)

        self.assertLessEqual(_rough_token_count(context), 35)
        self.assertEqual(len(used_hits), 1)
        self.assertIn("[Файл: Файл 1]", context)


class _FakeDbWithHistory:
    def __init__(self, messages):
        self.messages = [
            {"id": f"m{idx}", "chat_id": "c1", "role": row["role"], "content": row["content"], "created_at": idx}
            for idx, row in enumerate(messages)
        ]
        self._chunks = [
            {
                "id": "k1",
                "chat_id": "c1",
                "file_id": "f1",
                "chunk_index": 0,
                "text": (
                    "В документе описаны алгоритмы планирования процессов: FCFS, Round Robin, "
                    "SJF и приоритетное планирование. Эти алгоритмы сравниваются по времени ожидания."
                ),
                "vector_json": json.dumps([1.0, 0.0]),
            }
        ]

    def list_chats(self):
        return [{"id": "c1", "title": "Chat"}]

    def list_chunks_for_chat(self, chat_id):
        return [dict(x) for x in self._chunks if x["chat_id"] == chat_id]

    def list_messages(self, chat_id):
        return [dict(row) for row in self.messages if row["chat_id"] == chat_id]

    def list_files(self, chat_id):
        return []

    def store_message(self, chat_id, role, content):
        message_id = f"m{len(self.messages)}"
        self.messages.append(
            {"id": message_id, "chat_id": chat_id, "role": role, "content": content, "created_at": len(self.messages)}
        )
        return message_id


class _FakeVectorStore:
    def __init__(self):
        self.items = [
            {
                "chat_id": "c1",
                "chunk_id": "k1",
                "text": (
                    "В документе описаны алгоритмы планирования процессов: FCFS, Round Robin, "
                    "SJF и приоритетное планирование. Эти алгоритмы сравниваются по времени ожидания."
                ),
                "vector": [1.0, 0.0],
            }
        ]

    def total_items(self):
        return len(self.items)

    def list_for_chat(self, chat_id):
        return [dict(x) for x in self.items if x["chat_id"] == chat_id]

    def search(self, chat_id, query_vector, top_k):
        return [
            {
                "chat_id": row["chat_id"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "score": 0.91,
            }
            for row in self.list_for_chat(chat_id)[:top_k]
        ]


class _FakeEmbedder:
    def embed(self, text, keep_alive=-1):
        return [1.0, 0.0]

    def embed_many(self, texts, keep_alive=-1):
        return [self.embed(text) for text in texts]


class _CapturingGenerator:
    def __init__(self):
        self.messages = []

    def stream_chat(self, messages, think=False, keep_alive=-1):
        self.messages = messages
        yield {"message": {"content": "ok"}, "done": False}
        yield {"done": True}


def _rough_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


if __name__ == "__main__":
    unittest.main()
