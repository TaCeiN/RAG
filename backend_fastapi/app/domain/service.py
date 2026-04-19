from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError

from app.adapters.faiss_store import FaissStore
from app.adapters.ollama import OllamaEmbedder, OllamaGenerator, OllamaReranker
from app.core.config import settings
from app.core.logging import log
from app.db.sqlite_store import SqliteStore
from app.domain.ingest import chunk_text, extract_text

RETRIEVAL_MODES = {"hybrid_plus"}
ROUTER_INTENTS = {"qa", "compare", "report", "extract"}
ROUTER_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "qa": (
        "ответь на конкретный вопрос по документу",
        "найди точный факт, дату, число или определение",
        "answer a specific fact question from the document",
    ),
    "compare": (
        "сравни объекты, подходы или алгоритмы и выдели различия",
        "в чем разница между A и B",
        "compare alternatives and explain differences",
    ),
    "report": (
        "подготовь подробный отчет со структурой и выводами",
        "сделай развернутый аналитический разбор",
        "create a detailed structured report from the document",
    ),
    "extract": (
        "извлеки список сущностей, шагов, требований или пунктов",
        "перечисли элементы по документу",
        "extract structured items from the document",
    ),
}


class RagService:
    def __init__(
        self,
        db: SqliteStore,
        vector_store: FaissStore,
        embedder: OllamaEmbedder,
        generator: OllamaGenerator,
        reranker: OllamaReranker | None,
    ) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder
        self.generator = generator
        self.reranker = reranker
        self._router_vectors: dict[str, list[float]] = {}
        self._router_vectors_initialized = False
        self._bootstrap_vector_store_from_db()

    def _init_router_vectors(self) -> dict[str, list[float]]:
        samples: list[str] = []
        owners: list[str] = []
        for intent, phrases in ROUTER_PROTOTYPES.items():
            for phrase in phrases:
                samples.append(phrase)
                owners.append(intent)
        if not samples:
            return {}
        vectors = self.embedder.embed_many(samples, keep_alive=settings.embed_keep_alive_query)
        grouped: dict[str, list[list[float]]] = {intent: [] for intent in ROUTER_INTENTS}
        for intent, vector in zip(owners, vectors):
            grouped[intent].append(vector)
        averaged: dict[str, list[float]] = {}
        for intent, rows in grouped.items():
            if not rows:
                continue
            dim = len(rows[0])
            acc = [0.0] * dim
            for row in rows:
                if len(row) != dim:
                    continue
                for idx, value in enumerate(row):
                    acc[idx] += float(value)
            averaged[intent] = [value / max(1, len(rows)) for value in acc]
        return averaged

    def _ensure_router_vectors(self) -> None:
        if self._router_vectors_initialized:
            return
        try:
            self._router_vectors = self._init_router_vectors()
        except Exception as exc:  # noqa: BLE001
            log(f"Router vectors lazy init failed: {exc}")
            self._router_vectors = {}
        finally:
            self._router_vectors_initialized = True

    def _bootstrap_vector_store_from_db(self) -> None:
        if self.vector_store.total_items() > 0:
            return
        rows: list[dict[str, object]] = []
        for chat in self.db.list_chats():
            chat_id = str(chat["id"])
            for chunk in self.db.list_chunks_for_chat(chat_id):
                vector_raw = chunk.get("vector_json")
                try:
                    vector = json.loads(str(vector_raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(vector, list):
                    continue
                rows.append(
                    {
                        "chat_id": chunk["chat_id"],
                        "chunk_id": chunk["id"],
                        "file_id": chunk.get("file_id"),
                        "file_name": chunk.get("file_name"),
                        "chunk_index": chunk.get("chunk_index"),
                        "text": chunk["text"],
                        "vector": vector,
                    }
                )
        if rows:
            log(f"Restoring vector store from SQLite chunks: {len(rows)}")
            self.vector_store.replace_all(rows)

    def create_chat(self, title: str) -> str:
        return self.db.create_chat(title)

    def list_chats(self) -> list[dict[str, object]]:
        return self.db.list_chats()

    def list_messages(self, chat_id: str) -> list[dict[str, object]]:
        return self.db.list_messages(chat_id)

    def list_files(self, chat_id: str) -> list[dict[str, object]]:
        return self.db.list_files(chat_id)

    def queue_ingest_path(self, chat_id: str, file_path: Path) -> str:
        file_id = self.db.store_file(chat_id, file_path.name, "", status="uploaded")
        worker = threading.Thread(target=self._index_file_worker, args=(chat_id, file_id, file_path), daemon=True)
        worker.start()
        return file_id

    def _index_file_worker(self, chat_id: str, file_id: str, file_path: Path) -> None:
        try:
            self.db.update_file_status(file_id, "indexing")
            text = extract_text(file_path)
            self.db.update_file_text(file_id, text)
            self._index_text(chat_id, file_id, text)
            self.db.update_file_status(file_id, "ready")
        except Exception as exc:  # noqa: BLE001
            self.db.update_file_status(file_id, "error")
            log(f"Indexing failed for {file_path.name}: {exc}")

    def _index_text(self, chat_id: str, file_id: str, text: str) -> None:
        chunks = chunk_text(text)
        vectors = self.embedder.embed_many(chunks, keep_alive=settings.embed_keep_alive_ingest)
        file_name = ""
        for row in self.db.list_files(chat_id):
            if str(row.get("id")) == file_id:
                file_name = str(row.get("name", ""))
                break
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            chunk_id = self.db.store_chunk(chat_id, file_id, index, chunk, json.dumps(vector))
            self.vector_store.add(
                chat_id,
                chunk_id,
                chunk,
                vector,
                metadata={"file_id": file_id, "file_name": file_name, "chunk_index": index},
            )

    def stream_answer(
        self,
        chat_id: str,
        question: str,
        think: bool,
        debug_retrieval: bool,
        retrieval_mode: str,
        top_k: int,
        force_rag_on_upload: bool = False,
    ):
        normalized_question = _normalize_query_text(question)
        retrieval_question, followup_meta = self._resolve_retrieval_question(chat_id, question, normalized_question)
        mode = _resolve_retrieval_mode(retrieval_mode)
        yield {
            "type": "status",
            "message": "Модель сейчас анализирует ваш запрос (роутер определяет тип задачи и стратегию поиска).",
        }
        index_wait_meta = self._wait_for_indexed_chunks(chat_id)
        if index_wait_meta.get("waited"):
            yield {
                "type": "status",
                "message": "Модель сейчас подготавливает документ (ожидание завершения индексации файла).",
            }

        if (
            int(index_wait_meta.get("files_total", 0)) > 0
            and int(index_wait_meta.get("chunk_count", 0)) == 0
            and int(index_wait_meta.get("pending_files", 0)) > 0
        ):
            self.db.store_message(chat_id, "user", question)
            answer = (
                "Файл загружен, но его индексация ещё не завершилась. "
                "Попробуйте повторить запрос через несколько секунд."
            )
            trace = {
                "retrieval_mode": mode,
                "original_query": question,
                "normalized_query": normalized_question,
                "total_chunks_in_chat": 0,
                "route_meta": {
                    "response_mode": "indexing_wait",
                    "response_reason": "indexing_in_progress",
                    "index_wait_meta": index_wait_meta,
                },
            }
            self.db.store_message(chat_id, "assistant", answer)
            yield {"type": "answer", "delta": answer}
            yield {
                "type": "done",
                "answer": answer,
                "thinking": "",
                "context": "(indexing_pending)",
                "sources": [],
                "trace": trace,
            }
            return

        retrieval = self._retrieve(chat_id, question, retrieval_question, mode, top_k)
        hits = retrieval["final"]
        labeled_hits = _with_source_labels(hits)
        trace = retrieval["trace"]
        trace["normalized_query_original"] = normalized_question
        trace["followup_meta"] = followup_meta
        if retrieval_question != normalized_question:
            trace["rewrites"] = [retrieval_question]
        route_meta = trace.setdefault("route_meta", {})
        route_meta["index_wait_meta"] = index_wait_meta
        selected_intent = str(route_meta.get("intent", "qa"))
        context_text, labeled_hits = _fit_context_to_token_budget(
            labeled_hits,
            intent=selected_intent,
            max_tokens=settings.rag_context_token_budget,
        )
        route_meta["context_budget_meta"] = {
            "max_tokens": settings.rag_context_token_budget,
            "selected_sources": len(labeled_hits),
            "estimated_tokens": _token_count(context_text),
        }
        response_mode, response_reason, off_topic_meta = _decide_response_mode(
            trace=trace,
            final_hits=hits,
            context_text=context_text,
        )
        if force_rag_on_upload:
            if int(trace.get("total_chunks_in_chat", 0) or 0) > 0 and len(hits) > 0:
                response_mode = "rag"
                response_reason = "forced_rag_uploaded_with_message"
            elif int(index_wait_meta.get("files_total", 0)) > 0 and int(index_wait_meta.get("pending_files", 0)) > 0:
                response_mode = "indexing_wait"
                response_reason = "forced_rag_waiting_index"
            else:
                response_mode = "rag"
                response_reason = "forced_rag_uploaded_with_message"
        route_meta["response_mode"] = response_mode
        route_meta["response_reason"] = response_reason
        effective_min_hits = _effective_min_hits(trace)
        route_meta["direct_chat_thresholds"] = {
            "min_hits": settings.direct_chat_min_hits,
            "effective_min_hits": effective_min_hits,
            "min_best_score": settings.direct_chat_min_best_score,
            "min_context_chars": settings.direct_chat_min_context_chars,
            "off_topic_semantic_min_score": settings.off_topic_semantic_min_score,
            "off_topic_min_overlap_ratio": settings.off_topic_min_overlap_ratio,
            "off_topic_small_corpus_semantic_min_score": settings.off_topic_small_corpus_semantic_min_score,
            "off_topic_small_corpus_min_overlap_ratio": settings.off_topic_small_corpus_min_overlap_ratio,
        }
        route_meta["off_topic_meta"] = off_topic_meta

        self.db.store_message(chat_id, "user", question)

        if debug_retrieval:
            debug_answer = json.dumps(trace, ensure_ascii=False, indent=2)
            self.db.store_message(chat_id, "assistant", debug_answer)
            yield {"type": "answer", "delta": debug_answer}
            yield {
                "type": "done",
                "answer": debug_answer,
                "thinking": "",
                "context": context_text,
                "sources": labeled_hits,
                "trace": trace,
            }
            return

        if response_mode == "rag":
            yield {"type": "search_started"}
            selected_mode = str(route_meta.get("selected_retrieval_mode", "hybrid"))
            yield {
                "type": "status",
                "message": (
                    "Модель сейчас изучает ваш документ "
                    f"(поиск релевантных фрагментов: intent={selected_intent}, retrieval={selected_mode})."
                ),
            }
            yield {"type": "search_ready", "context": context_text, "sources": labeled_hits, "trace": trace}
            conversation_history, history_meta = self._build_rag_history(chat_id, current_question=question)
            route_meta["rag_history_meta"] = history_meta
            prompt_question = retrieval_question if followup_meta.get("is_followup") else question
            messages = self._build_chat_messages(
                prompt_question,
                context_text,
                intent=selected_intent,
                conversation_history=conversation_history,
            )
            yield {
                "type": "status",
                "message": "Модель сейчас формирует ответ (генератор собирает итог по найденным источникам).",
            }
            result_context = context_text
            result_sources = labeled_hits
        elif response_mode == "indexing_wait":
            answer = (
                "Файл получен, но контекст ещё формируется. "
                "Попробуйте повторить вопрос через несколько секунд."
            )
            self.db.store_message(chat_id, "assistant", answer)
            yield {"type": "answer", "delta": answer}
            yield {
                "type": "done",
                "answer": answer,
                "thinking": "",
                "context": "(indexing_pending)",
                "sources": [],
                "trace": trace,
            }
            return
        else:
            yield {
                "type": "status",
                "message": "Модель сейчас отвечает в дружеском разговорном режиме (без поиска по документам).",
            }
            messages = self._build_direct_chat_messages(chat_id, question)
            result_context = "(direct_chat)"
            result_sources = []

        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_limit = settings.thinking_max_chars
        thinking_used_chars = 0
        try:
            for chunk in self.generator.stream_chat(
                messages,
                think=think,
                keep_alive=settings.generator_keep_alive,
            ):
                message = chunk.get("message") or {}
                thinking = message.get("thinking")
                if thinking:
                    delta = str(thinking)
                    if thinking_limit is None:
                        thinking_parts.append(delta)
                        yield {"type": "thinking", "delta": delta}
                    elif thinking_limit > 0:
                        if thinking_used_chars < thinking_limit:
                            remaining = thinking_limit - thinking_used_chars
                            clipped = delta[:remaining]
                            if clipped:
                                thinking_parts.append(clipped)
                                thinking_used_chars += len(clipped)
                                yield {"type": "thinking", "delta": clipped}
                content = message.get("content")
                if content:
                    delta = str(content)
                    answer_parts.append(delta)
                    yield {"type": "answer", "delta": delta}
                if chunk.get("done"):
                    break

            if think and not answer_parts:
                yield {
                    "type": "status",
                    "message": "Модель завершает длинное рассуждение, продолжаю генерацию ответа без режима размышления.",
                }
                for chunk in self.generator.stream_chat(
                    messages,
                    think=False,
                    keep_alive=settings.generator_keep_alive,
                ):
                    message = chunk.get("message") or {}
                    content = message.get("content")
                    if content:
                        delta = str(content)
                        answer_parts.append(delta)
                        yield {"type": "answer", "delta": delta}
                    if chunk.get("done"):
                        break
        except HTTPError as exc:
            log(f"Ollama HTTPError while streaming chat: status={exc.code}, reason={exc.reason}")
            answer = (
                "Не получилось получить ответ от модели: Ollama вернул ошибку сервера. "
                "Обычно это из-за нехватки RAM/VRAM для выбранной модели. "
                "Попробуйте более легкую модель или уменьшите нагрузку."
            )
            self.db.store_message(chat_id, "assistant", answer)
            yield {"type": "error", "message": answer, "trace": trace}
            yield {
                "type": "done",
                "answer": answer,
                "thinking": "".join(thinking_parts),
                "context": result_context,
                "sources": result_sources,
                "trace": trace,
            }
            return
        except URLError as exc:
            log(f"Ollama URLError while streaming chat: {exc}")
            answer = (
                "Не получилось подключиться к Ollama. "
                "Проверьте, что `ollama serve` запущен и доступен."
            )
            self.db.store_message(chat_id, "assistant", answer)
            yield {"type": "error", "message": answer, "trace": trace}
            yield {
                "type": "done",
                "answer": answer,
                "thinking": "".join(thinking_parts),
                "context": result_context,
                "sources": result_sources,
                "trace": trace,
            }
            return
        except Exception as exc:  # noqa: BLE001
            log(f"Unexpected generator error while streaming chat: {exc}")
            answer = "Во время генерации произошла непредвиденная ошибка. Попробуйте повторить запрос."
            self.db.store_message(chat_id, "assistant", answer)
            yield {"type": "error", "message": answer, "trace": trace}
            yield {
                "type": "done",
                "answer": answer,
                "thinking": "".join(thinking_parts),
                "context": result_context,
                "sources": result_sources,
                "trace": trace,
            }
            return

        answer = "".join(answer_parts)
        self.db.store_message(chat_id, "assistant", answer)
        yield {
            "type": "done",
            "answer": answer,
            "thinking": "".join(thinking_parts),
            "context": result_context,
            "sources": result_sources,
            "trace": trace,
        }

    @staticmethod
    def _build_chat_messages(
        question: str,
        context_text: str,
        intent: str = "qa",
        conversation_history: str = "",
    ) -> list[dict[str, str]]:
        intent_hint = ""
        if intent == "report":
            intent_hint = (
                "\n9) For report requests, prioritize whole-document coverage over a single local detail.\n"
                "10) Mention all major themes/tasks found in context, then briefly describe important specifics.\n"
                "11) Ignore boilerplate, bibliography sections, and raw source code listings unless explicitly requested.\n"
                "12) Prefer concrete facts from context (tasks, algorithm names, input sequences, numeric results).\n"
                "13) If numbered tasks are present, summarize each task briefly and include key outcomes.\n"
                "14) Never infer missing parts from general knowledge; state explicitly when data is absent.\n"
            )
        elif intent == "compare":
            intent_hint = (
                "\n9) For comparison requests, explicitly contrast alternatives and highlight key differences.\n"
            )
        history_hint = (
            "\nUse CONVERSATION_HISTORY only for dialogue continuity and pronoun/reference resolution.\n"
            "Never treat CONVERSATION_HISTORY as a factual source; factual claims must come from CONTEXT_BLOCKS.\n"
        )
        history_block = f"CONVERSATION_HISTORY:\n{conversation_history}\n\n" if conversation_history.strip() else ""
        return [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-grounded assistant. Follow these rules strictly:\n"
                    "1) Use only facts from CONTEXT_BLOCKS.\n"
                    "2) Final answer language must be Russian.\n"
                    "3) Keep wording natural, clear, friendly, and practical.\n"
                    "4) If context is insufficient, explicitly say what is missing.\n"
                    "5) Do not expose internal IDs or chunk UUIDs.\n"
                    "6) When citing evidence, reference only SOURCE labels from the context (e.g., SOURCE 1).\n"
                    "7) Do not force any fixed response template unless the user explicitly asks for one.\n"
                    "8) Choose structure and depth based on user intent and question complexity.\n"
                    "9) Never infer, guess, or complete missing facts from general knowledge.\n"
                    "10) If a fact is absent in CONTEXT_BLOCKS, say this directly and stop speculation.\n"
                    "11) Never use phrases like 'скорее всего', 'вероятно', or similar uncertainty guesses."
                    f"{history_hint}"
                    f"{intent_hint}"
                ),
            },
            {"role": "user", "content": f"{history_block}CONTEXT_BLOCKS:\n{context_text}\n\nQUESTION:\n{question}"},
        ]

    def _build_direct_chat_messages(self, chat_id: str, question: str) -> list[dict[str, str]]:
        history = self.db.list_messages(chat_id)
        tail = history[-settings.direct_chat_history_messages :] if settings.direct_chat_history_messages > 0 else []
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Ты - полезный AI-ассистент. Твоя задача - помогать пользователям с их вопросами.\n\n"
                    "Основные принципы:\n"
                    "- Будь вежливым и уважительным\n"
                    "- Давай точные и полезные ответы\n"
                    "- Если не знаешь ответа, признайся в этом\n"
                    "- Не выдумывай информацию\n"
                    "- Соблюдай этические нормы\n\n"
                    "Возможности:\n"
                    "- Анализ текста\n"
                    "- Написание кода\n"
                    "- Ответы на вопросы\n"
                    "- Помощь с задачами\n\n"
                    "Ограничения:\n"
                    "- Не помогай с незаконной деятельностью\n"
                    "- Не создавай вредоносный контент\n"
                    "- Уважай авторские права"
                ),
            }
        ]
        for row in tail:
            role = str(row.get("role", "")).strip().lower()
            content = str(row.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def _wait_for_indexed_chunks(self, chat_id: str) -> dict[str, object]:
        timeout_seconds = max(0.0, float(settings.rag_index_wait_timeout_seconds))
        poll_seconds = max(0.05, float(settings.rag_index_wait_poll_seconds))
        waited = False
        start = time.monotonic()

        state = self._chat_index_state(chat_id)
        if state["files_total"] <= 0:
            state["waited"] = False
            state["wait_seconds"] = 0.0
            return state
        if state["chunk_count"] > 0:
            state["waited"] = False
            state["wait_seconds"] = 0.0
            return state
        if state["pending_files"] <= 0 or timeout_seconds <= 0:
            state["waited"] = False
            state["wait_seconds"] = 0.0
            return state

        deadline = start + timeout_seconds
        while time.monotonic() < deadline:
            waited = True
            time.sleep(poll_seconds)
            state = self._chat_index_state(chat_id)
            if state["chunk_count"] > 0:
                break
            if state["pending_files"] <= 0:
                break

        state["waited"] = waited
        state["wait_seconds"] = round(max(0.0, time.monotonic() - start), 3)
        return state

    def _chat_index_state(self, chat_id: str) -> dict[str, object]:
        files = self.db.list_files(chat_id)
        pending_statuses = {"uploaded", "indexing"}
        pending_files = sum(1 for row in files if str(row.get("status", "")).strip().lower() in pending_statuses)
        ready_files = sum(1 for row in files if str(row.get("status", "")).strip().lower() == "ready")
        error_files = sum(1 for row in files if str(row.get("status", "")).strip().lower() == "error")
        chunk_count = len(self.vector_store.list_for_chat(chat_id))
        return {
            "files_total": len(files),
            "ready_files": ready_files,
            "pending_files": pending_files,
            "error_files": error_files,
            "chunk_count": chunk_count,
        }

    def _resolve_retrieval_question(
        self,
        chat_id: str,
        question: str,
        normalized_question: str,
    ) -> tuple[str, dict[str, object]]:
        meta: dict[str, object] = {
            "is_followup": False,
            "reason": "default",
            "anchor_query": "",
            "effective_query": normalized_question,
        }
        if not _is_followup_request(normalized_question):
            return normalized_question, meta

        history = self.db.list_messages(chat_id)
        anchor_query = ""
        for row in reversed(history):
            role = str(row.get("role", "")).strip().lower()
            if role != "user":
                continue
            candidate_raw = str(row.get("content", "")).strip()
            candidate = _normalize_query_text(candidate_raw)
            if not candidate:
                continue
            if _is_followup_request(candidate):
                continue
            anchor_query = candidate_raw
            break

        if not anchor_query:
            meta["is_followup"] = True
            meta["reason"] = "followup_without_anchor"
            return normalized_question, meta

        effective = _normalize_query_text(f"{anchor_query}. {question}")
        if not effective:
            meta["is_followup"] = True
            meta["reason"] = "followup_anchor_empty_after_normalization"
            meta["anchor_query"] = anchor_query
            return normalized_question, meta

        meta["is_followup"] = True
        meta["reason"] = "followup_rewritten_from_previous_user_query"
        meta["anchor_query"] = anchor_query
        meta["effective_query"] = effective
        return effective, meta

    def _build_rag_history(self, chat_id: str, current_question: str) -> tuple[str, dict[str, int]]:
        max_messages = max(0, int(settings.rag_history_recent_messages))
        token_budget = max(0, int(settings.rag_history_token_budget))
        if max_messages <= 0 or token_budget <= 0:
            return "", {
                "requested_messages": max_messages,
                "included_messages": 0,
                "token_budget": token_budget,
                "used_tokens": 0,
            }

        rows = self.db.list_messages(chat_id)
        included: list[tuple[str, str]] = []
        used_tokens = 0
        skipped_current = False

        for row in reversed(rows):
            role = str(row.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            if role == "user" and not skipped_current and content == current_question.strip():
                skipped_current = True
                continue
            estimated = _token_count(content)
            if estimated <= 0:
                continue
            if included and used_tokens + estimated > token_budget:
                break
            if not included and estimated > token_budget:
                included.append((role, content))
                used_tokens = estimated
                break
            included.append((role, content))
            used_tokens += estimated
            if len(included) >= max_messages:
                break

        included.reverse()
        labels = {"user": "USER", "assistant": "ASSISTANT"}
        lines = [f"{labels.get(role, role.upper())}_{index + 1}: {text}" for index, (role, text) in enumerate(included)]
        history_text = "\n".join(lines)
        return history_text, {
            "requested_messages": max_messages,
            "included_messages": len(included),
            "token_budget": token_budget,
            "used_tokens": used_tokens,
        }

    def _retrieve(
        self,
        chat_id: str,
        user_question: str,
        normalized_question: str,
        retrieval_mode: str,
        final_k: int,
    ) -> dict[str, object]:
        recall_k = max(20, final_k * 4)
        all_chunks = self.vector_store.list_for_chat(chat_id)
        query_vector = self.embedder.embed(normalized_question, keep_alive=settings.embed_keep_alive_query)
        route_meta = self._resolve_route(
            user_question=user_question,
            normalized_question=normalized_question,
            retrieval_mode=retrieval_mode,
            query_vector=query_vector,
            total_chunks=len(all_chunks),
            final_k=final_k,
            base_recall_k=recall_k,
        )
        intent = str(route_meta.get("intent", "qa"))
        selected_mode = str(route_meta["selected_retrieval_mode"])
        selected_k = int(route_meta["selected_top_k"])
        selected_recall_k = int(route_meta["selected_recall_k"])
        vector_weight, bm25_weight = _resolve_mode_weights(selected_mode, normalized_question, len(all_chunks))
        search_query = normalized_question
        if intent == "report":
            acronyms = _extract_document_acronyms(all_chunks, limit=8)
            extra = " ".join(acronyms)
            search_query = f"{normalized_question} задания алгоритмы результаты вывод {extra}".strip()

        if search_query != normalized_question:
            query_vector = self.embedder.embed(search_query, keep_alive=settings.embed_keep_alive_query)
        vector_hits = self.vector_store.search(chat_id, query_vector, top_k=selected_recall_k)
        bm25_hits = _bm25_search(search_query, all_chunks, top_n=selected_recall_k) if selected_mode != "embeddings" else []
        merged = self._merge_hits(vector_hits, bm25_hits, selected_recall_k, vector_weight, bm25_weight)

        rerank_allowed = selected_mode == "hybrid_plus" and settings.reranker_enabled
        min_rerank = settings.rerank_min_candidates
        if rerank_allowed:
            reranked = self._rerank(normalized_question, merged, min_rerank)
            candidate_pool = reranked
        else:
            reranked = []
            candidate_pool = merged

        if intent == "report":
            quality_filtered = [
                row
                for row in candidate_pool
                if (
                    _chunk_quality_score(str(row.get("text", ""))) >= 0.22
                    or re.search(r"(?:\b\d+\b[\s,;:]+){6,}\b\d+\b", str(row.get("text", "")))
                )
                and not _is_summary_noise(str(row.get("text", "")))
            ]
            if len(quality_filtered) >= max(4, selected_k // 2):
                candidate_pool = quality_filtered
            final_hits = _select_diverse_hits(candidate_pool, limit=selected_k)
            if not _contains_long_numeric_sequence(final_hits):
                sequence_row = _find_sequence_chunk(all_chunks)
                if sequence_row is not None:
                    sequence_chunk_id = str(sequence_row.get("chunk_id", ""))
                    if not any(str(item.get("chunk_id", "")) == sequence_chunk_id for item in final_hits):
                        candidate = {
                            "chat_id": sequence_row.get("chat_id"),
                            "chunk_id": sequence_row.get("chunk_id"),
                            "file_id": sequence_row.get("file_id"),
                            "file_name": sequence_row.get("file_name"),
                            "chunk_index": sequence_row.get("chunk_index"),
                            "text": sequence_row.get("text"),
                            "vector_score": 0.0,
                            "bm25_score": 0.0,
                            "hybrid_score": 0.0,
                            "rerank_score": 0.0,
                        }
                        if len(final_hits) >= selected_k and selected_k > 0:
                            final_hits[-1] = candidate
                        else:
                            final_hits.append(candidate)
        else:
            filtered_pool: list[dict[str, object]] = []
            for row in candidate_pool:
                text = str(row.get("text", ""))
                overlap_count, _overlap_ratio = _token_overlap_stats(search_query, text)
                if _is_summary_noise(text) and overlap_count == 0:
                    continue
                filtered_pool.append(row)
            if filtered_pool:
                candidate_pool = filtered_pool
            final_hits = candidate_pool[:selected_k]

        rerank_enabled = rerank_allowed and len(merged) >= min_rerank and self.reranker is not None
        if not rerank_allowed:
            rerank_reason = "mode_disabled"
        elif len(merged) < min_rerank:
            rerank_reason = "not_enough_candidates"
        elif self.reranker is None:
            rerank_reason = "reranker_disabled" if not settings.reranker_enabled else "reranker_missing"
        else:
            rerank_reason = "ok"
        trace = {
            "retrieval_mode": selected_mode,
            "original_query": user_question,
            "normalized_query": normalized_question,
            "search_query": search_query,
            "rewrites": [],
            "queries": [normalized_question] if search_query == normalized_question else [normalized_question, search_query],
            "per_query": [
                {
                    "query": normalized_question,
                    "vector_candidates": _trace_rows(vector_hits, limit=10),
                    "bm25_candidates": _trace_rows(bm25_hits, limit=10),
                    "merged": _trace_rows(merged, limit=10),
                }
            ],
            "total_chunks_in_chat": len(all_chunks),
            "mode_meta": {
                "vector_weight": vector_weight,
                "bm25_weight": bm25_weight,
                "min_rerank_candidates": min_rerank,
                "top_k": selected_k,
                "recall_k": selected_recall_k,
            },
            "route_meta": route_meta,
            "vector_candidates": _trace_rows(vector_hits, limit=10),
            "bm25_candidates": _trace_rows(bm25_hits, limit=10),
            "merged": _trace_rows(merged),
            "reranked": _trace_rows(reranked),
            "final": _trace_rows(final_hits),
            "rerank_meta": {
                "enabled": rerank_enabled,
                "candidate_count": len(merged),
                "min_candidates": min_rerank,
                "reason": rerank_reason,
            },
        }
        return {"final": final_hits, "trace": trace}

    @staticmethod
    def _merge_hits(
        vector_hits: list[dict[str, object]],
        bm25_hits: list[dict[str, object]],
        recall_k: int,
        vector_weight: float,
        bm25_weight: float,
    ) -> list[dict[str, object]]:
        rows: dict[str, dict[str, object]] = {}
        for hit in vector_hits:
            chunk_id = str(hit["chunk_id"])
            row = rows.setdefault(
                chunk_id,
                {
                    "chat_id": hit["chat_id"],
                    "chunk_id": hit["chunk_id"],
                    "file_id": hit.get("file_id"),
                    "file_name": hit.get("file_name"),
                    "chunk_index": hit.get("chunk_index"),
                    "text": hit["text"],
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                },
            )
            row["vector_score"] = float(hit["score"])

        for hit in bm25_hits:
            chunk_id = str(hit["chunk_id"])
            row = rows.setdefault(
                chunk_id,
                {
                    "chat_id": hit["chat_id"],
                    "chunk_id": hit["chunk_id"],
                    "file_id": hit.get("file_id"),
                    "file_name": hit.get("file_name"),
                    "chunk_index": hit.get("chunk_index"),
                    "text": hit["text"],
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                },
            )
            row["file_id"] = row.get("file_id") or hit.get("file_id")
            row["file_name"] = row.get("file_name") or hit.get("file_name")
            row["chunk_index"] = row.get("chunk_index") if row.get("chunk_index") is not None else hit.get("chunk_index")
            row["bm25_score"] = _normalize_bm25(float(hit["score"]))

        merged = list(rows.values())
        for row in merged:
            row["hybrid_score"] = (vector_weight * float(row.get("vector_score", 0.0))) + (
                bm25_weight * float(row.get("bm25_score", 0.0))
            )
        merged.sort(
            key=lambda x: (
                -float(x.get("hybrid_score", 0.0)),
                -float(x.get("vector_score", 0.0)),
                -float(x.get("bm25_score", 0.0)),
                str(x.get("chunk_id", "")),
            )
        )
        return merged[:recall_k]

    def _rerank(self, question: str, merged: list[dict[str, object]], min_candidates: int) -> list[dict[str, object]]:
        if not merged:
            return []
        if len(merged) < min_candidates or self.reranker is None:
            result = []
            for row in merged:
                cloned = dict(row)
                cloned["rerank_score"] = float(cloned.get("hybrid_score", 0.0))
                result.append(cloned)
            return result

        docs = [str(item["text"]) for item in merged]
        scores = self.reranker.score_many(question, docs)
        if len(scores) < len(merged):
            scores.extend([0.0] * (len(merged) - len(scores)))

        reranked = []
        for row, score in zip(merged, scores):
            cloned = dict(row)
            cloned["rerank_score"] = float(score)
            reranked.append(cloned)
        reranked.sort(
            key=lambda x: (
                -float(x.get("rerank_score", 0.0)),
                -float(x.get("hybrid_score", 0.0)),
                str(x.get("chunk_id", "")),
            )
        )
        return reranked

    def _resolve_route(
        self,
        user_question: str,
        normalized_question: str,
        retrieval_mode: str,
        query_vector: list[float],
        total_chunks: int,
        final_k: int,
        base_recall_k: int,
    ) -> dict[str, object]:
        retrieval_mode = settings.retrieval_mode_default

        rule_scores = _rule_intent_scores(normalized_question)
        semantic_scores = self._semantic_intent_scores(query_vector)
        fused = _fuse_intent_scores(rule_scores, semantic_scores)
        intent, confidence = _select_intent(fused)
        if confidence < float(settings.router_min_confidence):
            intent = "qa"

        selected_k = final_k
        selected_recall_k = base_recall_k
        selected_mode = settings.retrieval_mode_default
        if intent == "report":
            selected_k = max(final_k, 16)
            selected_recall_k = max(base_recall_k, selected_k * 7)
            selected_mode = settings.retrieval_mode_default
        elif intent == "compare":
            selected_k = max(final_k, 8)
            selected_recall_k = max(base_recall_k, selected_k * 5)
            selected_mode = settings.retrieval_mode_default
        elif intent == "extract":
            selected_k = max(final_k, 8)
            selected_recall_k = max(base_recall_k, selected_k * 4)
            selected_mode = settings.retrieval_mode_default
        else:
            selected_k = final_k
            selected_recall_k = base_recall_k
            selected_mode = settings.retrieval_mode_default

        if total_chunks < selected_k:
            selected_k = max(1, total_chunks)
        if total_chunks == 0:
            selected_k = final_k

        return {
            "intent": intent,
            "confidence": confidence,
            "reason": "auto_router",
            "score_breakdown": {
                "rules": rule_scores,
                "semantic": semantic_scores,
                "fused": fused,
            },
            "selected_retrieval_mode": selected_mode,
            "selected_top_k": selected_k,
            "selected_recall_k": selected_recall_k,
        }

    def _semantic_intent_scores(self, query_vector: list[float]) -> dict[str, float]:
        self._ensure_router_vectors()
        if not self._router_vectors:
            return {intent: 0.0 for intent in ROUTER_INTENTS}
        scores: dict[str, float] = {}
        for intent in ROUTER_INTENTS:
            prototype = self._router_vectors.get(intent)
            if not prototype:
                scores[intent] = 0.0
                continue
            scores[intent] = _cosine_similarity(query_vector, prototype)
        return scores


def build_service() -> RagService:
    db = SqliteStore(settings.sqlite_path)
    vector_store = FaissStore(settings.faiss_index_path, settings.faiss_meta_path)
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    generator = OllamaGenerator(
        settings.ollama_base_url,
        settings.generator_model,
        temperature=settings.generator_temperature,
        top_p=settings.generator_top_p,
        repeat_penalty=settings.generator_repeat_penalty,
        num_ctx=settings.generator_num_ctx,
        num_batch=settings.generator_num_batch,
    )
    reranker = (
        OllamaReranker(
            settings.ollama_base_url,
            settings.reranker_model,
            keep_alive=settings.reranker_keep_alive,
        )
        if settings.reranker_enabled
        else None
    )
    return RagService(db=db, vector_store=vector_store, embedder=embedder, generator=generator, reranker=reranker)


def _with_source_labels(hits: list[dict[str, object]]) -> list[dict[str, object]]:
    labeled: list[dict[str, object]] = []
    for index, hit in enumerate(hits, start=1):
        row = dict(hit)
        row["source_label"] = f"SOURCE {index}"
        labeled.append(row)
    return labeled


def _build_context_blocks(hits: list[dict[str, object]], intent: str = "qa") -> str:
    if not hits:
        return "(no retrieved context)"
    blocks: list[str] = []
    for hit in hits:
        label = str(hit.get("source_label", "SOURCE"))
        text = str(hit.get("text", "")).strip()
        source_ref = _source_reference(hit)
        meta_line = f"Источник: {source_ref}\n" if source_ref else ""
        blocks.append(f"[{label}]\n{meta_line}{text}")
    return "\n\n".join(blocks)


def _source_reference(hit: dict[str, object]) -> str:
    file_name = str(hit.get("file_name") or "").strip()
    chunk_index_raw = hit.get("chunk_index")
    parts: list[str] = []
    if file_name:
        parts.append(file_name)
    try:
        chunk_number = int(chunk_index_raw) + 1
    except (TypeError, ValueError):
        chunk_number = 0
    if chunk_number > 0:
        parts.append(f"фрагмент {chunk_number}")
    return ", ".join(parts)


def _fit_context_to_token_budget(
    hits: list[dict[str, object]],
    intent: str,
    max_tokens: int,
) -> tuple[str, list[dict[str, object]]]:
    if not hits:
        return "(no retrieved context)", []
    if max_tokens <= 0:
        return "(no retrieved context)", []

    selected: list[dict[str, object]] = []
    best_context = "(no retrieved context)"
    for hit in hits:
        candidate = [*selected, dict(hit)]
        context = _build_context_blocks(candidate, intent=intent)
        if _token_count(context) <= max_tokens:
            selected = candidate
            best_context = context
            continue

        if not selected:
            trimmed = dict(hit)
            label = str(trimmed.get("source_label", "SOURCE"))
            header_budget = _token_count(f"[{label}]\n")
            text_budget = max(1, max_tokens - header_budget)
            trimmed["text"] = _truncate_to_token_budget(str(trimmed.get("text", "")), text_budget)
            selected = [trimmed]
            best_context = _build_context_blocks(selected, intent=intent)
        break

    if selected and _token_count(best_context) <= max_tokens:
        return best_context, selected
    if selected:
        return _shrink_context_preserving_first_label(selected[0], intent=intent, max_tokens=max_tokens), [selected[0]]
    return "(no retrieved context)", []


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max(1, max_tokens)]).strip()


def _shrink_context_preserving_first_label(hit: dict[str, object], intent: str, max_tokens: int) -> str:
    label = str(hit.get("source_label", "SOURCE"))
    low = 1
    high = max(1, _token_count(str(hit.get("text", ""))))
    best = f"[{label}]\n"
    while low <= high:
        mid = (low + high) // 2
        trimmed = dict(hit)
        trimmed["text"] = _truncate_to_token_budget(str(hit.get("text", "")), mid)
        context = _build_context_blocks([trimmed], intent=intent)
        if _token_count(context) <= max_tokens:
            best = context
            low = mid + 1
        else:
            high = mid - 1
    return best.strip() or "(no retrieved context)"


def _build_summary_context_blocks(hits: list[dict[str, object]]) -> str:
    candidates: list[tuple[float, str, str]] = []
    for hit in hits:
        label = str(hit.get("source_label", "SOURCE"))
        text = str(hit.get("text", ""))
        if _is_summary_noise(text) and _sequence_context_score(text) < 1.2:
            continue
        for sequence in re.findall(r"(?:\b\d+\b[\s,;:]+){10,}\b\d+\b", text):
            compact_sequence = re.sub(r"\s+", " ", sequence).strip()
            if compact_sequence and _sequence_context_score(text) >= 1.2:
                candidates.append((1.45, label, f"Последовательность данных: {compact_sequence}"))
        sentences = _split_sentences(text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) < 35:
                continue
            score = _fact_sentence_score(clean)
            if score <= 0:
                continue
            candidates.append((score, label, clean))

    if not candidates:
        blocks: list[str] = []
        for hit in hits:
            label = str(hit.get("source_label", "SOURCE"))
            text = str(hit.get("text", "")).strip()
            blocks.append(f"[{label}]\n{text}")
        return "\n\n".join(blocks)

    candidates.sort(key=lambda row: row[0], reverse=True)
    selected: list[tuple[str, str]] = []
    seen_norm: set[str] = set()
    per_source_limit = 4
    source_counts: dict[str, int] = {}
    for _score, label, sentence in candidates:
        norm = _normalize_sentence(sentence)
        if norm in seen_norm:
            continue
        if source_counts.get(label, 0) >= per_source_limit:
            continue
        selected.append((label, sentence))
        seen_norm.add(norm)
        source_counts[label] = source_counts.get(label, 0) + 1
        if len(selected) >= 24:
            break

    grouped: dict[str, list[str]] = {}
    for label, sentence in selected:
        grouped.setdefault(label, []).append(sentence)

    blocks: list[str] = []
    for hit in hits:
        label = str(hit.get("source_label", "SOURCE"))
        facts = grouped.get(label)
        if not facts:
            continue
        facts_text = "\n".join(f"- {fact}" for fact in facts)
        blocks.append(f"[{label}]\n{facts_text}")
    return "\n\n".join(blocks) if blocks else "(no retrieved context)"


def _select_diverse_hits(candidates: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        return [dict(row) for row in candidates]

    selected: list[dict[str, object]] = []
    used_ids: set[str] = set()

    ranked = sorted(
        candidates,
        key=lambda row: (
            0.82 * float(row.get("rerank_score", row.get("hybrid_score", row.get("vector_score", 0.0))))
            + 0.18 * _chunk_quality_score(str(row.get("text", "")))
        ),
        reverse=True,
    )
    first_idx = 0
    for idx, row in enumerate(ranked):
        if _chunk_quality_score(str(row.get("text", ""))) >= 0.16:
            first_idx = idx
            break

    # Keep a strong and readable candidate first.
    first = dict(ranked[first_idx])
    selected.append(first)
    used_ids.add(str(first.get("chunk_id", "")))

    while len(selected) < limit:
        best_idx = -1
        best_score = -1e9
        for idx, row in enumerate(ranked):
            chunk_id = str(row.get("chunk_id", ""))
            if chunk_id in used_ids:
                continue
            relevance = float(row.get("rerank_score", row.get("hybrid_score", row.get("vector_score", 0.0))))
            novelty = _novelty_score(str(row.get("text", "")), [str(x.get("text", "")) for x in selected])
            quality = _chunk_quality_score(str(row.get("text", "")))
            if quality < 0.16 and len(selected) < max(2, limit // 3):
                continue
            mmr_score = (0.62 * relevance) + (0.23 * novelty) + (0.15 * quality)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx < 0:
            break
        picked = dict(ranked[best_idx])
        selected.append(picked)
        used_ids.add(str(picked.get("chunk_id", "")))

    if len(selected) < limit:
        for row in ranked:
            chunk_id = str(row.get("chunk_id", ""))
            if chunk_id in used_ids:
                continue
            selected.append(dict(row))
            used_ids.add(chunk_id)
            if len(selected) >= limit:
                break
    return selected


def _novelty_score(text: str, selected_texts: list[str]) -> float:
    if not selected_texts:
        return 1.0
    base = _token_set(text)
    if not base:
        return 0.0
    max_overlap = 0.0
    for other in selected_texts:
        overlap = _jaccard(base, _token_set(other))
        if overlap > max_overlap:
            max_overlap = overlap
    return 1.0 - max_overlap


def _chunk_quality_score(text: str) -> float:
    if not text:
        return 0.0
    total = max(1, len(text))
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))
    letter_ratio = letters / total

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    noisy_lines = 0
    for line in lines:
        symbols = len(re.findall(r"[^A-Za-zА-Яа-яЁё0-9\s]", line))
        if symbols / max(1, len(line)) > 0.35:
            noisy_lines += 1
    noisy_ratio = (noisy_lines / max(1, len(lines))) if lines else 0.0

    code_markers = 0
    lower = text.lower()
    for marker in (
        "#include",
        "cout <<",
        "vector<",
        "for (",
        "while (",
        "return ",
        "::",
        "{",
        "}",
        ";",
    ):
        code_markers += lower.count(marker)

    bib_markers = 0
    for marker in ("isbn", "изд.", "учебное пособие", "с.", "том "):
        bib_markers += lower.count(marker)

    score = letter_ratio
    score -= 0.35 * noisy_ratio
    score -= min(0.45, code_markers * 0.02)
    score -= min(0.25, bib_markers * 0.03)
    return max(0.0, min(1.0, score))


def _is_summary_noise(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    if re.search(r"\b(isbn|учебн(ое|ый)\s+пособ|изд\.|санкт-петербург|вильямс)\b", lower):
        return True
    if re.search(r"\b(росжелдор|федеральн\w+\s+государствен\w+\s+бюджетн\w+|кафедр\w+)\b", lower):
        return True
    if re.search(r"_{8,}|[-=]{8,}", text):
        return True
    symbol_density = len(re.findall(r"[^A-Za-zА-Яа-яЁё0-9\s]", text)) / max(1, len(text))
    if symbol_density > 0.28:
        return True
    if _single_char_token_ratio(text) > 0.42:
        return True
    if re.search(r"\b(#include|int\s+main|cout\s*<<|vector<|for\s*\(|while\s*\()\b", lower):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return re.split(r"(?<=[.!?])\s+|(?<=;)\s+|(?<=:)\s+", normalized)


def _fact_sentence_score(sentence: str) -> float:
    s = sentence.strip()
    if not s:
        return 0.0
    quality = _chunk_quality_score(s)
    if quality < 0.12:
        return 0.0

    lower = s.lower()
    if re.search(r"\b(#include|cout\s*<<|int\s+main|vector<|for\s*\(|while\s*\(|return\s+)\b", lower):
        return 0.0
    symbol_density = len(re.findall(r"[^A-Za-zА-Яа-яЁё0-9\s]", s)) / max(1, len(s))
    if symbol_density > 0.22:
        return 0.0
    if _single_char_token_ratio(s) > 0.36:
        return 0.0
    score = 0.1 + (0.35 * quality)
    marker_hits = 0

    if re.search(r"\b(задани[ея]|task|section|раздел)\b", lower):
        score += 0.5
        marker_hits += 1
    if re.search(r"\b(fcfs|round robin|sjf|priority|fifo|lru|opt|page fault|pf)\b", lower):
        score += 0.7
        marker_hits += 1
    if re.search(r"\b(аномал|belady|белад)\w*", lower):
        score += 0.4
        marker_hits += 1
    if re.search(r"\b(вывод|результат|итог|составил|получено)\b", lower):
        score += 0.3
        marker_hits += 1
    if re.search(r"(?:\b\d+\b[\s,;:]+){4,}\b\d+\b", lower):
        score += 0.55
        marker_hits += 1
    if re.search(r"\b(таблиц|рисунок|листинг)\b", lower):
        score += 0.15
        marker_hits += 1

    if marker_hits == 0:
        return 0.0

    if re.search(r"\b(cout|vector<|for\s*\(|while\s*\(|return\s+)\b", lower):
        score -= 0.6
    if re.search(r"\b(isbn|изд\.|учебн(ое|ый) пособ)\b", lower):
        score -= 0.35

    return max(0.0, score)


def _normalize_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip().lower()


def _single_char_token_ratio(text: str) -> float:
    tokens = re.findall(r"\b[^\W_]+\b", text, flags=re.UNICODE)
    if not tokens:
        return 0.0
    single = 0
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            single += 1
    return single / max(1, len(tokens))


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return 0.0 if union == 0 else (inter / union)


def _extract_document_acronyms(chunks: list[dict[str, object]], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        tokens = re.findall(r"\b[A-ZА-ЯЁ]{2,8}\b", text)
        for token in tokens:
            if token.isdigit():
                continue
            counter[token] += 1
    if not counter:
        return []
    return [token for token, _freq in counter.most_common(limit)]


def _contains_long_numeric_sequence(rows: list[dict[str, object]]) -> bool:
    for row in rows:
        text = str(row.get("text", ""))
        if re.search(r"(?:\b\d+\b[\s,;:]+){10,}\b\d+\b", text):
            return True
    return False


def _find_sequence_chunk(chunks: list[dict[str, object]]) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_score = -1.0
    best_len = -1
    for row in chunks:
        text = str(row.get("text", ""))
        context_score = _sequence_context_score(text)
        if context_score <= 0:
            continue
        for sequence in re.findall(r"(?:\b\d+\b[\s,;:]+){10,}\b\d+\b", text):
            length = len(sequence)
            combined = (2.0 * context_score) + (length / 100.0)
            if combined > best_score or (combined == best_score and length > best_len):
                best_score = combined
                best_len = length
                best = row
    return None if best is None else best


def _sequence_context_score(text: str) -> float:
    lower = text.lower()
    score = 0.0
    if re.search(r"\b(задани[ея]|task)\b", lower):
        score += 1.2
    if re.search(r"\b(строк[аи]\s+обращени|reference string|последовательност[ьи]\s+обращени)\b", lower):
        score += 1.5
    if re.search(r"\b(fifo|lru|opt|page fault|pf|страничн\w+\s+нарушени)\b", lower):
        score += 1.6
    if re.search(r"\b(кадр\w+\s+памят|frames?)\b", lower):
        score += 0.7
    if re.search(r"\b(fcfs|round robin|sjf|priority)\b", lower):
        score += 0.2
    if re.search(r"\b(cout|#include|int\s+main|vector<|for\s*\(|while\s*\()\b", lower):
        score -= 1.0
    if _single_char_token_ratio(text) > 0.34:
        score -= 0.8
    return score


def _rule_intent_scores(question: str) -> dict[str, float]:
    text = question.lower()
    scores = {intent: 0.0 for intent in ROUTER_INTENTS}
    if re.search(r"\b(сумм|кратк|overview|summary|резюм|подыт)\w*", text):
        scores["report"] += 1.2
    if re.search(r"\b(сравн|разниц|отлич|vs|versus|compare)\w*", text):
        scores["compare"] += 1.4
    if re.search(r"\b(отчет|доклад|развернут|разбор|аналит|report)\w*", text):
        scores["report"] += 1.1
    if re.search(r"\b(извлек|перечисл|список|таблиц|extract|list)\w*", text):
        scores["extract"] += 1.0
    if re.search(r"\b(что|когда|сколько|какой|какие|какова|каков|кто|where|when|what|how many)\b", text):
        scores["qa"] += 0.8
    return scores


def _fuse_intent_scores(rule_scores: dict[str, float], semantic_scores: dict[str, float]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for intent in ROUTER_INTENTS:
        rule = float(rule_scores.get(intent, 0.0))
        semantic = max(0.0, float(semantic_scores.get(intent, 0.0)))
        fused[intent] = (0.45 * min(1.5, rule)) + (0.55 * semantic)
    return fused


def _select_intent(fused_scores: dict[str, float]) -> tuple[str, float]:
    if not fused_scores:
        return "qa", 0.0
    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    top_intent, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = max(0.0, min(1.0, top_score - (0.35 * second_score)))
    return top_intent, confidence


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _resolve_retrieval_mode(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in RETRIEVAL_MODES else settings.retrieval_mode_default


def _resolve_mode_weights(retrieval_mode: str, question: str, total_chunks: int) -> tuple[float, float]:
    if retrieval_mode == "embeddings":
        return 1.0, 0.0
    if retrieval_mode == "hybrid":
        return settings.hybrid_vector_weight, settings.hybrid_bm25_weight

    bm25_weight = settings.hybrid_bm25_weight
    if total_chunks < 8:
        bm25_weight *= 0.45
    elif total_chunks < 20:
        bm25_weight *= 0.7

    if len(set(_tokenize(question))) <= 3:
        bm25_weight *= 0.7

    bm25_weight = max(0.08, min(settings.hybrid_bm25_weight, bm25_weight))
    return 1.0 - bm25_weight, bm25_weight


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if token]


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "и",
        "в",
        "во",
        "на",
        "с",
        "со",
        "к",
        "ко",
        "по",
        "о",
        "об",
        "от",
        "до",
        "за",
        "из",
        "у",
        "а",
        "но",
        "или",
        "что",
        "это",
        "как",
        "ли",
        "же",
        "не",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "be",
        "what",
        "how",
        "when",
        "where",
        "who",
        "будет",
        "может",
        "нужно",
        "надо",
        "сделать",
        "сделал",
        "был",
        "была",
        "были",
    }
    return {
        token
        for token in _tokenize(text)
        if len(token) >= 3 and not token.isdigit() and token not in stopwords
    }


def _token_overlap_stats(query_text: str, chunk_text: str) -> tuple[int, float]:
    query_tokens = _content_tokens(query_text)
    if not query_tokens:
        return 0, 0.0
    chunk_tokens = _content_tokens(chunk_text)
    if not chunk_tokens:
        return 0, 0.0
    overlap = len(query_tokens.intersection(chunk_tokens))
    ratio = overlap / max(1, len(query_tokens))
    return overlap, ratio


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _bm25_search(question: str, chunks: list[dict[str, object]], top_n: int) -> list[dict[str, object]]:
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []

    docs = [(_tokenize(str(chunk["text"])), chunk) for chunk in chunks]
    docs = [(tokens, chunk) for tokens, chunk in docs if tokens]
    if not docs:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens, _chunk in docs:
        doc_freq.update(set(tokens))

    total_docs = len(docs)
    avg_len = sum(len(tokens) for tokens, _chunk in docs) / max(1, total_docs)
    k1 = 1.5
    b = 0.75

    hits: list[dict[str, object]] = []
    query_counter = Counter(query_tokens)
    for tokens, chunk in docs:
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for token, qf in query_counter.items():
            df = doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * (dl / max(avg_len, 1e-9)))
            score += idf * (freq * (k1 + 1)) / denom * qf
        if score > 0:
            hits.append(
                {
                    "chat_id": chunk["chat_id"],
                    "chunk_id": chunk["chunk_id"],
                    "file_id": chunk.get("file_id"),
                    "file_name": chunk.get("file_name"),
                    "chunk_index": chunk.get("chunk_index"),
                    "text": chunk["text"],
                    "score": score,
                }
            )
    hits.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("chunk_id", ""))))
    return hits[:top_n]


def _normalize_bm25(score: float) -> float:
    if score <= 0:
        return 0.0
    return score / (score + 1.5)


def _normalize_query_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""

    candidate = compact
    recovered_candidates = []
    for encoding in ("cp1251", "latin1"):
        try:
            recovered = compact.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        recovered = re.sub(r"\s+", " ", recovered).strip()
        if recovered:
            recovered_candidates.append(recovered)

    best = candidate
    best_score = _text_quality_score(candidate)
    for recovered in recovered_candidates:
        score = _text_quality_score(recovered)
        if score > best_score:
            best = recovered
            best_score = score
    return best


def _is_followup_request(text: str) -> bool:
    normalized = _normalize_query_text(text).lower().strip()
    if not normalized:
        return False
    patterns = [
        r"^подробнее\b",
        r"^поподробнее\b",
        r"^подробней\b",
        r"^детальнее\b",
        r"^раскрой\b",
        r"^распиши\b",
        r"^расскажи подробнее\b",
        r"^можно подробнее\b",
        r"^more details\b",
        r"^elaborate\b",
        r"^expand\b",
    ]
    if any(re.search(pattern, normalized) for pattern in patterns):
        return True
    short_tokens = _tokenize(normalized)
    if len(short_tokens) <= 3 and any(token in {"подробнее", "детальнее", "подробней"} for token in short_tokens):
        return True
    return False


def _text_quality_score(text: str) -> int:
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))
    mojibake_pairs = len(re.findall(r"[РС][A-Za-zА-Яа-яЁё]", text))
    latin_mojibake = text.count("Ãƒ") + text.count("Ã") + text.count("Ã‘")
    replacement_marks = text.count("ï¿½")
    return (cyrillic * 4) + letters - (mojibake_pairs * 6) - (latin_mojibake * 4) - (replacement_marks * 10)


def _trace_rows(rows: list[dict[str, object]], limit: int = 30) -> list[dict[str, object]]:
    trace = []
    for row in rows[:limit]:
        text = str(row.get("text", ""))
        preview = text if settings.debug_text_preview_limit is None else text[: settings.debug_text_preview_limit]
        trace.append(
            {
                "chunk_id": row.get("chunk_id"),
                "file_name": row.get("file_name"),
                "chunk_index": row.get("chunk_index"),
                "score": row.get("score"),
                "vector_score": row.get("vector_score"),
                "bm25_score": row.get("bm25_score"),
                "hybrid_score": row.get("hybrid_score"),
                "rerank_score": row.get("rerank_score"),
                "text_preview": preview,
            }
        )
    return trace


def _best_retrieval_score(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    best = rows[0]
    values: list[float] = []
    for key in ("rerank_score", "hybrid_score", "vector_score", "score", "bm25_score"):
        value = best.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return max(values)


def _decide_response_mode(
    trace: dict[str, object],
    final_hits: list[dict[str, object]],
    context_text: str,
) -> tuple[str, str, dict[str, float | int | bool]]:
    total_chunks = int(trace.get("total_chunks_in_chat", 0) or 0)
    if total_chunks <= 0:
        return "direct_chat", "no_chunks_in_chat", {"triggered": False}

    effective_min_hits = _effective_min_hits(trace)
    if len(final_hits) < effective_min_hits:
        return "direct_chat", "low_hit_count", {"triggered": False}

    best_score = _best_retrieval_score(final_hits)
    query_text = str(trace.get("normalized_query") or trace.get("original_query") or "").strip()
    top_text = str((final_hits[0] or {}).get("text", "") if final_hits else "")
    overlap_count, overlap_ratio = _token_overlap_stats(query_text, top_text)
    small_corpus = total_chunks <= 2
    effective_best_score_threshold = float(settings.direct_chat_min_best_score)
    if overlap_count >= 2 and overlap_ratio >= 0.18:
        effective_best_score_threshold = min(effective_best_score_threshold, effective_best_score_threshold * 0.65)
    if best_score < effective_best_score_threshold:
        return "direct_chat", "low_best_score", {
            "triggered": False,
            "effective_best_score_threshold": float(effective_best_score_threshold),
            "best_score": float(best_score),
            "overlap_count": int(overlap_count),
            "overlap_ratio": float(overlap_ratio),
            "total_chunks": int(total_chunks),
        }

    if small_corpus:
        off_topic_triggered = (
            best_score < settings.off_topic_small_corpus_semantic_min_score
            and (overlap_ratio < settings.off_topic_small_corpus_min_overlap_ratio or overlap_count <= 1)
        )
    else:
        off_topic_triggered = (
            best_score < settings.off_topic_semantic_min_score
            and overlap_ratio < settings.off_topic_min_overlap_ratio
        )
    off_topic_meta: dict[str, float | int | bool] = {
        "triggered": off_topic_triggered,
        "small_corpus": small_corpus,
        "best_score": float(best_score),
        "effective_best_score_threshold": float(effective_best_score_threshold),
        "overlap_count": int(overlap_count),
        "overlap_ratio": float(overlap_ratio),
        "total_chunks": int(total_chunks),
    }
    if off_topic_triggered:
        return "direct_chat", "off_topic_low_alignment", off_topic_meta

    normalized_context = context_text.strip()
    if (not normalized_context) or (normalized_context == "(no retrieved context)"):
        return "direct_chat", "empty_context", off_topic_meta
    if len(normalized_context) < settings.direct_chat_min_context_chars:
        return "direct_chat", "short_context", off_topic_meta

    return "rag", "retrieval_confident", off_topic_meta


def _effective_min_hits(trace: dict[str, object]) -> int:
    total_chunks = max(0, int(trace.get("total_chunks_in_chat", 0) or 0))
    route_meta = trace.get("route_meta") or {}
    selected_top_k_raw = route_meta.get("selected_top_k", 0) if isinstance(route_meta, dict) else 0
    try:
        selected_top_k = max(0, int(selected_top_k_raw or 0))
    except (TypeError, ValueError):
        selected_top_k = 0

    effective = max(1, int(settings.direct_chat_min_hits))
    if selected_top_k > 0:
        effective = min(effective, selected_top_k)
    if total_chunks > 0:
        effective = min(effective, total_chunks)
    return max(1, effective)
