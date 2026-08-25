from app.content import build_chunks, split_text


def test_split_text_keeps_short_content_together() -> None:
    assert split_text("첫 문장입니다. 두 번째 문장입니다.") == [
        "첫 문장입니다. 두 번째 문장입니다."
    ]


def test_build_chunks_prefers_curated_answer() -> None:
    chunks = build_chunks(
        [
            {
                "id": "career",
                "title": "경력",
                "url": "/about/",
                "type": "경력",
                "content": "긴 원문",
                "answer": "2년 7개월 동안 백엔드 개발자로 근무했습니다.",
            }
        ]
    )
    assert len(chunks) == 1
    assert "2년 7개월" in chunks[0].text
