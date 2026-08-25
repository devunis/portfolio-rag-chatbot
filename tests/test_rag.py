import asyncio
from types import SimpleNamespace

from app.config import Settings
from app.rag import RagService


class FakeLoader:
    async def load(self):
        return [
            {
                "id": "career",
                "title": "백엔드 실무 경력",
                "url": "/about/",
                "type": "경력",
                "answer": "푸른소나무에서 백엔드 개발자로 근무했습니다.",
            }
        ]


class FakeEmbeddings:
    async def create(self, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in kwargs["input"]]
        )


class FakeResponses:
    async def create(self, **_kwargs):
        return SimpleNamespace(output_text="허정윤은 푸른소나무에서 백엔드 개발자로 근무했습니다.")


class FakeOpenAI:
    def __init__(self):
        self.embeddings = FakeEmbeddings()
        self.responses = FakeResponses()


def test_rag_answer_contains_sources() -> None:
    settings = Settings(openai_api_key="test", index_on_startup=False)
    service = RagService(settings, loader=FakeLoader(), client=FakeOpenAI())
    response = asyncio.run(service.ask("어느 회사에서 일했나요?"))
    assert response.mode == "rag"
    assert "푸른소나무" in response.answer
    assert response.sources[0].url == "/about/"
