import asyncio
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.rag import RagService
from app.rate_limit import FixedWindowRateLimiter
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ReindexResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()
rag = RagService(settings)
limiter = FixedWindowRateLimiter(settings.request_limit_per_minute)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task: asyncio.Task[ReindexResponse | None] | None = None
    if settings.index_on_startup:

        async def warm_index() -> ReindexResponse | None:
            try:
                return await rag.reindex()
            except Exception:
                logger.exception("Initial indexing failed; the first chat request will retry")
                return None

        task = asyncio.create_task(warm_index())
    yield
    if task and not task.done():
        task.cancel()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token"],
    max_age=86400,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        indexed=rag.indexed,
        chunks=rag.retriever.size,
        semantic_search=rag.retriever.semantic,
        source=settings.source_index_url,
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    client_key = request.client.host if request.client else "unknown"
    if not await limiter.allow(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="잠시 후 다시 질문해 주세요.",
        )
    if len(payload.message) > settings.max_question_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"질문은 {settings.max_question_length}자 이하로 입력해 주세요.",
        )
    try:
        return await rag.ask(payload.message)
    except Exception as error:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="지식 인덱스를 준비하지 못했습니다. 잠시 후 다시 시도해 주세요.",
        ) from error


@app.post("/admin/reindex", response_model=ReindexResponse)
async def reindex(x_admin_token: str | None = Header(default=None)) -> ReindexResponse:
    if not settings.admin_token:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")
    return await rag.reindex()
