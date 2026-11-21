import logging
import time
import uuid
from typing import Optional
from fastapi import BackgroundTasks
from app.clients.llm_proxy import LLMProxyClient
from app.clients.neo4j_graph import Neo4jGraphClient
from app.clients.redis_cache import RedisCacheClient
from app.config import settings
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ExtractedFeatures,
    GatewayTelemetry,
)
from app.services.feature_extractor import FeatureExtractor

logger = logging.getLogger("gateway.router")


class GatewayRouter:
    """
    İstek karşılama, paralel özellik çıkarımı, Neo4j + Redis önbellek kontrolü
    ve arka planda asenkron kayıt süreçlerini koordine eden çekirdek yönlendirici.
    """

    def __init__(
        self,
        llm_client: LLMProxyClient,
        feature_extractor: FeatureExtractor,
        neo4j_client: Neo4jGraphClient,
        redis_client: RedisCacheClient,
    ) -> None:
        self.llm_client = llm_client
        self.feature_extractor = feature_extractor
        self.neo4j_client = neo4j_client
        self.redis_client = redis_client

    async def _lookup_semantic_cache(
        self,
        prompt: str,
        embedding: list[float],
        features: ExtractedFeatures,
        active_mode: str,
    ) -> Optional[dict]:
        """
        1) Neo4j hibrit Cypher sorgusu ile yapısal/vektörel eşleşmeyi doğrular.
        2) Eşleşme bulunursa Redis üzerinden milisaniyeler içinde yanıtı çeker.
        3) Self-healing: Neo4j kaydı var ancak Redis anahtarı yoksa yetim düğümü temizler.
        """
        match_info = await self.neo4j_client.find_semantic_match(
            prompt=prompt,
            embedding=embedding,
            intent=features.intent,
            entities=features.entities,
            mode=active_mode,
        )
        if not match_info:
            return None

        message_id = match_info["message_id"]
        cached_answer = await self.redis_client.get_answer(message_id)

        if cached_answer is None:
            logger.warning(
                "Tutarsız önbellek kaydı tespit edildi (Redis'te yok, Neo4j'den siliniyor): %s",
                message_id,
            )
            await self.neo4j_client.delete_message_node(message_id)
            return None

        return {
            "answer": cached_answer,
            "message_id": message_id,
            "score": match_info["score"],
            "threshold": match_info["threshold"],
        }

    async def _persist_cache_async(
        self,
        prompt: str,
        answer: str,
        embedding: list[float],
        features: ExtractedFeatures,
    ) -> None:
        """
        Cache Miss sonrasında kullanıcıya yanıt dönülürken arka planda
        önce Redis'e yanıtı, ardından Neo4j'ye graf düğümlerini kaydeder.
        """
        message_id = uuid.uuid4().hex
        try:
            await self.redis_client.set_answer(message_id=message_id, answer=answer)
            await self.neo4j_client.upsert_message_node(
                message_id=message_id,
                prompt=prompt,
                embedding=embedding,
                intent=features.intent,
                entities=features.entities,
            )
        except Exception as exc:
            logger.error("Arka plan önbellek kaydı başarısız: %s", str(exc))

    async def handle_chat_completion(
        self,
        request: ChatCompletionRequest,
        background_tasks: BackgroundTasks,
        mode_override: Optional[str] = None,
        skip_cache_write: bool = False,
    ) -> ChatCompletionResponse:
        total_start = time.perf_counter()
        prompt = request.extract_latest_user_prompt()
        active_mode = mode_override or settings.gateway_mode

        # 1. Passthrough (Önbelleksiz Baseline) Modu
        if active_mode == "passthrough":
            llm_result = await self.llm_client.generate_completion(request)
            total_ms = (time.perf_counter() - total_start) * 1000
            return ChatCompletionResponse(
                id=llm_result["id"],
                model=llm_result["model"],
                content=llm_result["content"],
                telemetry=GatewayTelemetry(
                    cache_status="BYPASS",
                    gateway_mode=active_mode,
                    total_latency_ms=round(total_ms, 2),
                    llm_upstream_ms=llm_result["upstream_latency_ms"],
                    prompt_tokens=llm_result["prompt_tokens"],
                    completion_tokens=llm_result["completion_tokens"],
                    estimated_cost_saved=False,
                ),
            )

        # 2. Paralel Özellik Çıkarımı (Embedding + Intent/Entity)
        embedding, features, extract_ms = await self.feature_extractor.extract(prompt)

        # 3. Hibrit Neo4j + Redis Önbellek Sorgusu
        lookup_start = time.perf_counter()
        cache_hit_data = await self._lookup_semantic_cache(
            prompt=prompt,
            embedding=embedding,
            features=features,
            active_mode=active_mode,
        )
        lookup_ms = (time.perf_counter() - lookup_start) * 1000

        # 4. Cache Hit Durumu
        if cache_hit_data is not None:
            total_ms = (time.perf_counter() - total_start) * 1000
            return ChatCompletionResponse(
                id=f"cache-{cache_hit_data['message_id'][:12]}",
                model=request.model or settings.default_llm_model,
                content=cache_hit_data["answer"],
                telemetry=GatewayTelemetry(
                    cache_status="HIT",
                    gateway_mode=active_mode,
                    total_latency_ms=round(total_ms, 2),
                    feature_extraction_ms=extract_ms,
                    cache_lookup_ms=round(lookup_ms, 2),
                    llm_upstream_ms=0.0,
                    matched_message_id=cache_hit_data["message_id"],
                    similarity_score=cache_hit_data["score"],
                    applied_threshold=cache_hit_data["threshold"],
                    estimated_cost_saved=True,
                    extracted_features=features,
                ),
            )

        # 5. Cache Miss Durumu
        llm_result = await self.llm_client.generate_completion(request)
        if not skip_cache_write:
            background_tasks.add_task(
                self._persist_cache_async,
                prompt=prompt,
                answer=llm_result["content"],
                embedding=embedding,
                features=features,
            )

        total_ms = (time.perf_counter() - total_start) * 1000
        return ChatCompletionResponse(
            id=llm_result["id"],
            model=llm_result["model"],
            content=llm_result["content"],
            telemetry=GatewayTelemetry(
                cache_status="MISS",
                gateway_mode=active_mode,
                total_latency_ms=round(total_ms, 2),
                feature_extraction_ms=extract_ms,
                cache_lookup_ms=round(lookup_ms, 2),
                llm_upstream_ms=llm_result["upstream_latency_ms"],
                prompt_tokens=llm_result["prompt_tokens"],
                completion_tokens=llm_result["completion_tokens"],
                estimated_cost_saved=False,
                extracted_features=features,
            ),
        )