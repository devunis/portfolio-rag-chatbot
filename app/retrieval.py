import math
import re
from dataclasses import dataclass

from app.content import Chunk

PARTICLES = (
    "으로",
    "에서",
    "에게",
    "부터",
    "까지",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "의",
    "에",
    "로",
    "도",
    "만",
)


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[0-9a-zA-Z가-힣+#.]+", value.lower())
    normalized: list[str] = []
    for token in tokens:
        for particle in PARTICLES:
            if len(token) > len(particle) + 1 and token.endswith(particle):
                token = token[: -len(particle)]
                break
        if len(token) > 1:
            normalized.append(token)
    return normalized


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def lexical_score(query: str, chunk: Chunk) -> float:
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    haystack = f"{chunk.title} {chunk.type} {chunk.text}".lower()
    matches = sum(1 for term in terms if term in haystack)
    phrase_bonus = 0.2 if query.lower() in haystack else 0.0
    return min(1.0, matches / len(terms) + phrase_bonus)


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryRetriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def semantic(self) -> bool:
        return bool(self._vectors) and len(self._vectors) == len(self._chunks)

    def replace(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        if vectors and len(vectors) != len(chunks):
            raise ValueError("Each chunk must have exactly one embedding")
        self._chunks = list(chunks)
        self._vectors = list(vectors or [])

    def search(
        self, query: str, query_vector: list[float] | None = None, limit: int = 4
    ) -> list[SearchResult]:
        ranked: list[SearchResult] = []
        use_semantic = bool(query_vector) and self.semantic

        for index, chunk in enumerate(self._chunks):
            lexical = lexical_score(query, chunk)
            if use_semantic:
                semantic = cosine_similarity(query_vector or [], self._vectors[index])
                score = semantic * 0.88 + lexical * 0.12
            else:
                score = lexical
            if score > (0.12 if use_semantic else 0.0):
                ranked.append(SearchResult(chunk=chunk, score=score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]
