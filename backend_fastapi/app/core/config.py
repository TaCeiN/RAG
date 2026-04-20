from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Хост FastAPI-сервера для локального запуска.
    # Обычно 127.0.0.1 (доступ только с этой машины).
    host: str = "127.0.0.1"
    # Порт FastAPI-сервера.
    # Фронтенд обращается к этому порту по API.
    port: int = 8010

    # Абсолютный путь к корню backend_fastapi.
    # Используется для построения путей к файлам данных/индекса.
    project_root: Path = Path(__file__).resolve().parents[2]

    # Путь к SQLite-базе с "реляционными" сущностями:
    # чаты, сообщения, файлы, чанки, метаданные.
    sqlite_path: Path = project_root / "data" / "rag_fastapi.sqlite3"

    # Путь к FAISS-индексу векторов (поиск похожих чанков по эмбеддингам).
    faiss_index_path: Path = project_root / "data" / "faiss.index"
    # Путь к карте соответствия "позиция вектора в FAISS -> chunk_id".
    # Нужен, чтобы после поиска в FAISS восстановить исходный текст чанка.
    faiss_meta_path: Path = project_root / "data" / "faiss_meta.json"

    # Базовый URL локального Ollama-сервера.
    # Должен совпадать с адресом, где работает `ollama serve`.
    ollama_base_url: str = "http://localhost:11434"
    # Имя модели эмбеддингов в Ollama.
    # Используется для индексации документов и векторизации запросов.
    embedding_model: str = "embeddinggemma:300m"
    # Имя генеративной модели (основной чат-модели).
    # Она формирует финальный ответ пользователю.
    generator_model: str = "qwen3.5:4b"
    # Время удержания embedding-модели в памяти после embed-запроса (сек).
    # 0 = выгружать сразу после запроса (минимум VRAM, но больше cold-start).
    # -1 = держать бесконечно (быстрее повторные вызовы, но больше риск OOM).
    embed_keep_alive_query: int = 0
    # Время удержания embedding-модели при индексации документов (сек).
    # Имеет тот же смысл, что и embed_keep_alive_query, но для ingest-пайплайна.
    embed_keep_alive_ingest: int = 0
    # Temperature генератора.
    # Ниже = более "строгие/стабильные" ответы, выше = больше вариативности.
    generator_temperature: float = 0.0
    # Top-p генератора (nucleus sampling).
    # Ограничивает распределение токенов верхним вероятностным "хвостом".
    generator_top_p: float = 1.0
    # Штраф за повторы токенов.
    # Чуть повышенное значение помогает уменьшить зацикливание формулировок.
    generator_repeat_penalty: float = 1.0
    # Размер контекста генератора (num_ctx, в токенах).
    # Больше = лучше держит длинный диалог/контекст, но выше VRAM и риск OOM.
    generator_num_ctx: int = 4096
    # Размер батча генерации (num_batch).
    # Больше = иногда быстрее, но тоже увеличивает потребление VRAM/RAM.
    generator_num_batch: int = 64
    # Время удержания генеративной модели в памяти между запросами (сек).
    # 0 = выгружать сразу; -1 = держать постоянно; >0 = компромисс по latency/VRAM.
    generator_keep_alive: int = 120
    # Включать ли отдельную модель реранкера.
    # False = быстрее/легче по памяти, True = потенциально точнее ранжирование.
    reranker_enabled: bool = False
    # Имя модели реранкера в Ollama.
    reranker_model: str = "ExpedientFalcon/qwen3-reranker:0.6b-q4_k_m"
    # Время удержания реранкер-модели в памяти после запроса (сек).
    # Для экономии VRAM обычно 0.
    reranker_keep_alive: int = 0

    # Единственный поддерживаемый режим retrieval: высокий recall + BM25 + rerank.
    retrieval_mode_default: str = "hybrid_plus"
    # Сколько чанков передаём в сборщик контекста (top-k финальной выдачи).
    top_k_default: int = 4
    # Нижняя/верхняя граница top_k из API. Больше лимита не даем, чтобы не раздувать prompt.
    top_k_min: int = 1
    top_k_max: int = 16
    # Бюджет токенов для документного контекста внутри prompt.
    # Держим запас под system prompt, историю, вопрос и ответ.
    rag_context_token_budget: int = 2200
    # Минимум кандидатов, после которого можно применять реранк.
    # Если кандидатов меньше — реранк пропускается.
    rerank_min_candidates: int = 10
    # Веса смешивания vector + BM25 в гибридном поиске.
    # Итоговый score = vector_weight * vector_score + bm25_weight * bm25_score.
    hybrid_vector_weight: float = 0.58
    hybrid_bm25_weight: float = 0.42

    # Минимум найденных чанков, чтобы остаться в RAG-режиме.
    # Ниже порога — переключаемся в direct_chat.
    direct_chat_min_hits: int = 2
    # Минимальный score лучшего чанка для продолжения в RAG.
    direct_chat_min_best_score: float = 0.23
    # Минимальная уверенность роутера intent-классификации.
    # Ниже порога intent принудительно нормализуется в QA (без агрессивных спец-режимов).
    router_min_confidence: float = 0.22
    # Минимальная длина собранного контекста (символы), чтобы считать его пригодным.
    direct_chat_min_context_chars: int = 80
    # Порог "семантической уверенности" для off-topic защиты.
    # Если низко и запрос не пересекается по смыслу с документом — уход в direct_chat.
    off_topic_semantic_min_score: float = 0.34
    # Минимальная доля пересечения содержательных токенов запроса с топ-чанком.
    off_topic_min_overlap_ratio: float = 0.08
    # Порог off-topic для маленького корпуса (1-2 чанка), где статистика менее стабильна.
    off_topic_small_corpus_semantic_min_score: float = 0.40
    # Минимальное пересечение токенов в режиме маленького корпуса.
    off_topic_small_corpus_min_overlap_ratio: float = 0.05
    # Сколько последних сообщений подтягивать в direct_chat (без RAG-контекста).
    direct_chat_history_messages: int = 12
    # Сколько последних сообщений добавлять в RAG для follow-up логики.
    rag_history_recent_messages: int = 6
    # Старая настройка оставлена для совместимости с тестами/локальными скриптами.
    rag_history_recent_user_queries: int = 0
    # Бюджет токенов на блок истории в RAG.
    # Ограничивает разрастание промпта и потребление контекста модели.
    rag_history_token_budget: int = 700
    # Максимальная длина видимого "thinking" в стриме (символы).
    # None = без ограничения (не обрезать рассуждение).
    thinking_max_chars: int | None = None
    # Сколько максимум ждать завершения индексации файла перед ответом (сек).
    rag_index_wait_timeout_seconds: float = 12.0
    # Интервал опроса статуса индексации (сек).
    rag_index_wait_poll_seconds: float = 0.35

    # Целевой размер чанка в "лексических токенах"
    # (приближенно слова/пунктуация, не BPE токены модели).
    chunk_size: int = 320
    # Перекрытие соседних чанков в лексических токенах.
    # Помогает не терять смысл на границах разреза.
    chunk_overlap: int = 40
    # Минимальный размер чанка.
    # Более маленькие фрагменты по возможности склеиваются.
    chunk_min_size: int = 120

    # Ограничение длины text_preview в debug trace.
    # None = показывать полный текст чанка.
    debug_text_preview_limit: int | None = None

    # Разрешенные расширения файлов для upload endpoint.
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")


settings = Settings()
