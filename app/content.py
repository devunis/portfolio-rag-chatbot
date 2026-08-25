import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    title: str
    url: str
    type: str
    text: str


class ContentLoader:
    def __init__(self, source_url: str, timeout_seconds: float = 15.0) -> None:
        self.source_url = source_url
        self.timeout_seconds = timeout_seconds

    async def load(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "devunis-portfolio-rag/1.0"},
        ) as client:
            response = await client.get(self.source_url)
            response.raise_for_status()
            payload = response.json()

        documents = payload.get("documents", [])
        if not isinstance(documents, list):
            raise ValueError("Source index must contain a documents array")
        return [document for document in documents if isinstance(document, dict)]


def split_text(value: str, max_chars: int = 900) -> list[str]:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    if not clean:
        return []

    sentences = re.findall(r"[^.!?。！？]+[.!?。！？]?", clean) or [clean]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_chunks(documents: list[dict[str, Any]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document_index, document in enumerate(documents):
        document_id = str(document.get("id") or f"document-{document_index}")
        title = str(document.get("title") or "제목 없음")[:200]
        url = str(document.get("url") or "/")
        source_type = str(document.get("type") or "사이트")[:80]
        content = str(document.get("answer") or document.get("content") or "")
        for chunk_index, text in enumerate(split_text(content)):
            chunks.append(
                Chunk(
                    id=f"{document_id}-{chunk_index}",
                    title=title,
                    url=url,
                    type=source_type,
                    text=text,
                )
            )
    return chunks
