# Devunis Portfolio RAG

`devunis.github.io`의 공개 프로필·프로젝트·기술 글을 검색하고, 확인된 근거만으로 답하는 FastAPI 백엔드입니다.

## 구조

1. `https://devunis.github.io/rag-index.json`에서 공개 자료를 가져옵니다.
2. 문서를 문장 단위로 청크하고 `text-embedding-3-small`로 임베딩합니다.
3. 질문 임베딩과 cosine similarity로 관련 근거를 검색합니다.
4. 검색된 근거만 Responses API에 전달해 한국어 답변을 생성합니다.
5. API 응답에 사용한 출처와 URL을 함께 반환합니다.

`OPENAI_API_KEY`가 없거나 OpenAI 요청이 실패하면 lexical 검색과 원문 답변으로 안전하게 전환됩니다.

공식 참고 문서: [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings), [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)

## 로컬 실행

Python 3.11 이상이 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

`.env`에 `OPENAI_API_KEY`를 설정하면 semantic RAG 모드가 활성화됩니다. API 키는 절대 Git에 커밋하지 않습니다.

## API

### 상태 확인

```bash
curl http://localhost:8000/health
```

### 질문

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"어느 회사에서 어떤 일을 했나요?"}'
```

응답 예시:

```json
{
  "answer": "...",
  "sources": [
    {
      "title": "백엔드 실무 경력",
      "url": "/about/",
      "type": "경력",
      "excerpt": "...",
      "score": 0.91
    }
  ],
  "mode": "rag"
}
```

### 인덱스 갱신

`ADMIN_TOKEN`이 설정된 경우에만 사용할 수 있습니다.

```bash
curl -X POST http://localhost:8000/admin/reindex \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

## 운영 설정

- `OPENAI_API_KEY`: 서버에만 저장하는 OpenAI API 키
- `OPENAI_RESPONSE_MODEL`: 기본값 `gpt-4.1-mini`
- `OPENAI_EMBEDDING_MODEL`: 기본값 `text-embedding-3-small`
- `SOURCE_INDEX_URL`: 포트폴리오 지식 인덱스 URL
- `ALLOWED_ORIGINS`: 쉼표로 구분한 CORS 허용 출처
- `ADMIN_TOKEN`: 재인덱싱 엔드포인트 보호 토큰
- `REQUEST_LIMIT_PER_MINUTE`: IP당 분당 질문 수

Docker를 지원하는 Render, Railway, Fly.io 등의 서비스에 배포할 수 있습니다. 배포 환경에는 `.env` 파일 대신 서비스의 secret/environment 설정을 사용하세요.

애플리케이션의 메모리 기반 제한은 기본적인 비용 보호 장치입니다. 공개 운영 시에는 배포 서비스나 Cloudflare의 edge rate limiting을 함께 설정하세요.

## 프런트엔드 연결

배포 URL이 `https://portfolio-rag.example.com`이라면 GitHub Pages의 챗봇이 다음 요청을 보내도록 설정합니다.

```javascript
fetch("https://portfolio-rag.example.com/v1/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: question })
});
```

백엔드 배포가 완료되기 전까지는 현재 GitHub Pages의 클라이언트 검색 모드를 유지하는 것이 안전합니다.
