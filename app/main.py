import asyncio
from contextlib import asynccontextmanager
import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from app.clients.llm_proxy import LLMProxyClient
from app.clients.neo4j_graph import Neo4jGraphClient
from app.clients.redis_cache import RedisCacheClient
from app.config import settings
from app.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.services.feature_extractor import FeatureExtractor
from app.services.gateway_router import GatewayRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Katman 2 modellerini belleğe yükle
    feature_extractor = FeatureExtractor()
    await asyncio.to_thread(feature_extractor.load_models)

    # 2. Katman 3 (Neo4j) ve Katman 4 (Redis) istemcilerini başlat
    neo4j_client = Neo4jGraphClient()
    await neo4j_client.initialize_schema()
    redis_client = RedisCacheClient()

    # 3. Redis TTL süresi dolduğunda Neo4j düğümünü temizleyen arka plan görevini başlat
    ttl_sync_task = asyncio.create_task(
        redis_client.listen_expired_events(neo4j_client.delete_message_node)
    )

    # 4. Kalıcı HTTP havuzu ve GatewayRouter kurulumu
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    timeout = httpx.Timeout(settings.llm_timeout_seconds)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as http_client:
        llm_client = LLMProxyClient(http_client=http_client)
        app.state.gateway_router = GatewayRouter(
            llm_client=llm_client,
            feature_extractor=feature_extractor,
            neo4j_client=neo4j_client,
            redis_client=redis_client,
        )
        try:
            yield
        finally:
            ttl_sync_task.cancel()
            await redis_client.close()
            await neo4j_client.close()


app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    lifespan=lifespan,
)


def get_gateway_router() -> GatewayRouter:
    return app.state.gateway_router


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gateway_mode": settings.gateway_mode,
        "neo4j_uri": settings.neo4j_uri,
        "redis_url": settings.redis_url,
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
        response.headers["X-Cache-Status"] = result.telemetry.cache_status
        response.headers["X-Total-Latency-Ms"] = str(result.telemetry.total_latency_ms)
        response.headers["X-Cache-Lookup-Ms"] = str(result.telemetry.cache_lookup_ms)
        if result.telemetry.similarity_score is not None:
            response.headers["X-Similarity-Score"] = str(result.telemetry.similarity_score)
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