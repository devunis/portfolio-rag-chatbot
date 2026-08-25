from app.content import Chunk
from app.retrieval import InMemoryRetriever, cosine_similarity, tokenize


def test_korean_particle_normalization() -> None:
    assert "경력" in tokenize("경력은 어떻게 되나요?")


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_lexical_retrieval_ranks_matching_chunk() -> None:
    retriever = InMemoryRetriever()
    retriever.replace(
        [
            Chunk("career", "백엔드 경력", "/about/", "경력", "병원 연동 API를 개발했습니다."),
            Chunk("project", "SenPick", "/projects/", "프로젝트", "RAG 선물 추천 서비스입니다."),
        ]
    )
    result = retriever.search("RAG 프로젝트", limit=1)
    assert result[0].chunk.id == "project"
