from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import build_router
from app.domain.service import build_service

service = build_service()
app = FastAPI(title="RAG FastAPI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(build_router(service))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
