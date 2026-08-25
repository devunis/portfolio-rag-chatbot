from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=2000)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 2:
            raise ValueError("질문을 두 글자 이상 입력해 주세요.")
        return normalized


class Source(BaseModel):
    title: str
    url: str
    type: str
    excerpt: str
    score: float = Field(ge=-1, le=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: Literal["rag", "extractive"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    indexed: bool
    chunks: int
    semantic_search: bool
    source: str


class ReindexResponse(BaseModel):
    indexed: bool
    documents: int
    chunks: int
    semantic_search: bool
