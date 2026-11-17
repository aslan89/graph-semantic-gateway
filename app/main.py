from contextlib import asynccontextmanager
import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from app.clients.llm_proxy import LLMProxyClient
from app.config import settings
from app.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.services.gateway_router import GatewayRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Yüksek trafikli istekler için kalıcı (persistent) HTTP bağlantı havuzu
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    timeout = httpx.Timeout(settings.llm_timeout_seconds)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as http_client:
        llm_client = LLMProxyClient(http_client=http_client)
        app.state.gateway_router = GatewayRouter(llm_client=llm_client)
        yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)


def get_gateway_router() -> GatewayRouter:
    return app.state.gateway_router


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gateway_mode": settings.gateway_mode,
        "default_model": settings.default_llm_model,
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    router: GatewayRouter = Depends(get_gateway_router),
):
    if not request.extract_latest_user_prompt():
        raise HTTPException(status_code=400, detail="En az bir 'user' mesajı gereklidir.")

    try:
        result = await router.handle_chat_completion(request, background_tasks)
        # Kıyaslama ve yük testleri için HTTP başlıklarına metrik ekleme
        response.headers["X-Cache-Status"] = result.telemetry.cache_status
        response.headers["X-Total-Latency-Ms"] = str(result.telemetry.total_latency_ms)
        return result
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream LLM hatası: {exc.response.text}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Dış LLM servisine erişilemedi: {str(exc)}",
        ) from exc