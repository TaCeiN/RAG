# Local RAG Chat

Локальный RAG-чат для работы с документами. Проект состоит из FastAPI-бэкенда,
React/Vite-фронтенда, SQLite-хранилища, FAISS-поиска и локальных моделей Ollama.

## Возможности

- Загрузка документов `.txt`, `.md`, `.pdf`, `.docx`.
- Индексация документов в SQLite + FAISS.
- Гибридный поиск `hybrid_plus`: embeddings + BM25 + rerank.
- Потоковые ответы модели через NDJSON stream.
- Русскоязычные ответы по умолчанию.
- Поддержка follow-up вопросов вроде `Подробнее`.
- Отображение источников и файлов в интерфейсе чата.
- Debug-режим retrieval trace.

## Структура

```text
.
├── backend_fastapi/        # FastAPI backend, RAG pipeline, SQLite, FAISS
│   ├── app/
│   │   ├── adapters/       # Ollama and FAISS adapters
│   │   ├── api/            # HTTP routes
│   │   ├── core/           # config and logging
│   │   ├── db/             # SQLite store
│   │   └── domain/         # ingest, retrieval, prompt assembly, generation
│   ├── tests/              # backend tests
│   └── requirements.txt
├── frontend_react/         # React + Vite frontend
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── .gitignore
```

## Требования

- Python 3.12+
- Node.js 20+
- Ollama

Модели Ollama, ожидаемые конфигом:

```powershell
ollama pull embeddinggemma:300m
ollama pull qwen3.5:4b
ollama pull ExpedientFalcon/qwen3-reranker:0.6b-q4_k_m
```

Перед запуском убедитесь, что Ollama работает:

```powershell
ollama serve
```

## Запуск backend

```powershell
cd backend_fastapi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

Backend по умолчанию доступен на:

```text
http://127.0.0.1:8010
```

Healthcheck:

```text
http://127.0.0.1:8010/health
```

## Запуск frontend

В отдельном терминале:

```powershell
cd frontend_react
npm install
npm run dev
```

Frontend по умолчанию доступен на:

```text
http://127.0.0.1:5173
```

## Тесты

Backend:

```powershell
cd backend_fastapi
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Frontend:

```powershell
cd frontend_react
npm test
npm run build
```

## Конфигурация

Основные настройки находятся в:

```text
backend_fastapi/app/core/config.py
```

Важные параметры:

- `ollama_base_url` - адрес Ollama.
- `embedding_model` - модель эмбеддингов.
- `generator_model` - генеративная модель.
- `reranker_model` - модель rerank.
- `retrieval_mode_default` - единственный режим retrieval, сейчас `hybrid_plus`.
- `rag_context_token_budget` - лимит документного контекста для prompt.
- `top_k_default`, `top_k_min`, `top_k_max` - лимиты количества источников.
- `chunk_size`, `chunk_overlap`, `chunk_min_size` - настройки нарезки документов.

## Runtime-данные

Локальные данные не должны попадать в git:

- `backend_fastapi/data/`
- SQLite-файлы
- FAISS-индексы
- загруженные документы
- `.venv/`
- `node_modules/`
- `frontend_react/dist/`

Это уже закрыто в `.gitignore`.

## API

Основные endpoints:

- `GET /health`
- `GET /api/chats`
- `POST /api/chats`
- `GET /api/chats/{chat_id}/messages`
- `GET /api/chats/{chat_id}/files`
- `POST /api/chats/{chat_id}/files`
- `POST /api/chats/{chat_id}/messages`

Ответы чата стримятся как `application/x-ndjson`.

## Заметки по качеству ответов

Pipeline держит память диалога и документный контекст отдельно:

1. Нормализует запрос.
2. Переписывает короткие follow-up запросы для retrieval.
3. Ищет кандидаты через `hybrid_plus`.
4. Ограничивает контекст token budget.
5. Собирает prompt с `CONVERSATION_HISTORY` и `CONTEXT_BLOCKS`.
6. Стримит ответ из Ollama.

