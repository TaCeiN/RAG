from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from uuid import uuid4


class SqliteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                name TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """
        )
        self.conn.commit()

    def create_chat(self, title: str) -> str:
        chat_id = uuid4().hex
        self.conn.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (chat_id, title, time.time_ns()),
        )
        self.conn.commit()
        return chat_id

    def list_chats(self) -> list[dict[str, object]]:
        rows = self.conn.execute("SELECT id, title, created_at FROM chats ORDER BY seq DESC").fetchall()
        return [dict(row) for row in rows]

    def store_file(self, chat_id: str, name: str, text: str, status: str = "ready") -> str:
        file_id = uuid4().hex
        self.conn.execute(
            "INSERT INTO files (id, chat_id, name, text, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (file_id, chat_id, name, text, status, time.time_ns()),
        )
        self.conn.commit()
        return file_id

    def update_file_status(self, file_id: str, status: str) -> None:
        self.conn.execute("UPDATE files SET status = ? WHERE id = ?", (status, file_id))
        self.conn.commit()

    def update_file_text(self, file_id: str, text: str) -> None:
        self.conn.execute("UPDATE files SET text = ? WHERE id = ?", (text, file_id))
        self.conn.commit()

    def list_files(self, chat_id: str) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT id, chat_id, name, text, status, created_at FROM files WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def store_chunk(self, chat_id: str, file_id: str, chunk_index: int, text: str, vector_json: str) -> str:
        chunk_id = uuid4().hex
        self.conn.execute(
            """
            INSERT INTO chunks (id, chat_id, file_id, chunk_index, text, vector_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, chat_id, file_id, chunk_index, text, vector_json, time.time_ns()),
        )
        self.conn.commit()
        return chunk_id

    def list_chunks_for_chat(self, chat_id: str) -> list[dict[str, object]]:
        rows = self.conn.execute(
            """
            SELECT chunks.id,
                   chunks.chat_id,
                   chunks.file_id,
                   files.name AS file_name,
                   chunks.chunk_index,
                   chunks.text,
                   chunks.vector_json,
                   chunks.created_at
            FROM chunks
            LEFT JOIN files ON files.id = chunks.file_id
            WHERE chunks.chat_id = ?
            ORDER BY chunks.created_at ASC
            """,
            (chat_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def store_message(self, chat_id: str, role: str, content: str) -> str:
        message_id = uuid4().hex
        self.conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, chat_id, role, content, time.time_ns()),
        )
        self.conn.commit()
        return message_id

    def list_messages(self, chat_id: str) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT id, chat_id, role, content, created_at FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()
