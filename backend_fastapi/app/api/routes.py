from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.domain.service import RagService


class CreateChatRequest(BaseModel):
    title: str = "New chat"


class UpdateChatRequest(BaseModel):
    title: str


class SendMessageRequest(BaseModel):
    content: str
    think: bool = False
    debug_retrieval: bool = False
    retrieval_mode: str = settings.retrieval_mode_default
    top_k: int = Field(default=settings.top_k_default, ge=settings.top_k_min, le=settings.top_k_max)
    force_rag_on_upload: bool = False


def build_router(service: RagService) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["chat"])

    @router.get("/chats")
    def list_chats():
        return service.list_chats()

    @router.post("/chats", status_code=201)
    def create_chat(payload: CreateChatRequest):
        chat_id = service.create_chat(payload.title or "New chat")
        return {"id": chat_id}

    @router.patch("/chats/{chat_id}")
    def update_chat(chat_id: str, payload: UpdateChatRequest):
        updated = service.update_chat_title(chat_id, payload.title)
        if not updated:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"id": chat_id, "title": payload.title}

    @router.get("/chats/{chat_id}/messages")
    def list_messages(chat_id: str):
        return service.list_messages(chat_id)

    @router.get("/chats/{chat_id}/files")
    def list_files(chat_id: str):
        return service.list_files(chat_id)

    @router.post("/chats/{chat_id}/files", status_code=201)
    async def upload_files(chat_id: str, file: list[UploadFile] = File(...)):
        uploaded = []
        chat_dir = settings.project_root / "data" / "uploads" / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)

        for item in file:
            suffix = Path(item.filename or "").suffix.lower()
            if suffix not in settings.allowed_extensions:
                raise HTTPException(status_code=400, detail=f"Unsupported extension: {suffix}")
            target = chat_dir / Path(item.filename or "upload").name
            target.write_bytes(await item.read())
            file_id = service.queue_ingest_path(chat_id, target)
            uploaded.append({"file_id": file_id, "filename": target.name, "status": "uploaded"})

        return {"files": uploaded}

    @router.post("/chats/{chat_id}/files/{file_id}/cancel")
    def cancel_file(chat_id: str, file_id: str):
        cancelled = service.cancel_file_processing(chat_id, file_id)
        return {"file_id": file_id, "cancelled": cancelled}

    @router.post("/chats/{chat_id}/messages", status_code=201)
    def stream_message(chat_id: str, payload: SendMessageRequest):
        def event_stream():
            for event in service.stream_answer(
                chat_id=chat_id,
                question=payload.content,
                think=payload.think,
                debug_retrieval=payload.debug_retrieval,
                retrieval_mode=settings.retrieval_mode_default,
                top_k=payload.top_k,
                force_rag_on_upload=payload.force_rag_on_upload,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson; charset=utf-8")

    return router
