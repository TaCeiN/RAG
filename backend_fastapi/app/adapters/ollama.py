from __future__ import annotations

import json
import re
from urllib.request import Request, urlopen


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, opener=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.opener = opener or urlopen

    def embed(self, text: str, keep_alive: int = -1) -> list[float]:
        return self.embed_many([text], keep_alive=keep_alive)[0]

    def embed_many(self, texts: list[str], keep_alive: int = -1) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "keep_alive": keep_alive}
        request = Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener(request) as response:
            body = json.loads(response.read().decode("utf-8"))
        embeddings = body.get("embeddings")
        if embeddings is None and "embedding" in body:
            embeddings = [body["embedding"]]
        return [list(vector) for vector in (embeddings or [])]


class OllamaGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        opener=None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        num_batch: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.opener = opener or urlopen
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.num_ctx = num_ctx
        self.num_batch = num_batch

    def stream_chat(self, messages: list[dict[str, str]], think: bool = False, keep_alive: int = -1):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think,
            "keep_alive": keep_alive,
        }
        options: dict[str, float] = {}
        if self.temperature is not None:
            options["temperature"] = float(self.temperature)
        if self.top_p is not None:
            options["top_p"] = float(self.top_p)
        if self.repeat_penalty is not None:
            options["repeat_penalty"] = float(self.repeat_penalty)
        if self.num_ctx is not None:
            options["num_ctx"] = int(self.num_ctx)
        if self.num_batch is not None:
            options["num_batch"] = int(self.num_batch)
        if options:
            payload["options"] = options
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener(request) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line.decode("utf-8"))


class OllamaReranker:
    def __init__(self, base_url: str, model: str, opener=None, keep_alive: int = -1) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.opener = opener or urlopen
        self.keep_alive = keep_alive

    def score_many(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        prompt = self._build_prompt(query, documents)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener(request) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = str((body.get("message") or {}).get("content", ""))
        return self._parse_scores(content, len(documents))

    @staticmethod
    def _build_prompt(query: str, documents: list[str]) -> str:
        parts = [
            "You are a reranker. Return strict JSON only.",
            '{"scores":[{"index":0,"score":0.0}]}',
            "Score must be in [0,1].",
            f"Query: {query}",
            "Documents:",
        ]
        for index, doc in enumerate(documents):
            parts.append(f"[{index}] {doc}")
        return "\n".join(parts)

    @staticmethod
    def _parse_scores(content: str, count: int) -> list[float]:
        scores = [0.0] * count
        parsed: dict[str, object] | None = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None
        if not parsed:
            return scores
        raw = parsed.get("scores")
        if not isinstance(raw, list):
            return scores
        for row in raw:
            if not isinstance(row, dict):
                continue
            index = row.get("index")
            value = row.get("score")
            if isinstance(index, int) and 0 <= index < count:
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    score = 0.0
                scores[index] = max(0.0, min(1.0, score))
        return scores
