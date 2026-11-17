import asyncio
import time
import uuid
import httpx
from app.config import settings
from app.schemas import ChatCompletionRequest


class LLMProxyClient:
    """Cache Miss durumunda istekleri dış LLM servisine ileten asenkron istemci."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self.http_client = http_client

    async def generate_completion(self, request: ChatCompletionRequest) -> dict:
        target_model = request.model or settings.default_llm_model
        start_time = time.perf_counter()

        # API anahtarı tanımlı değilse geliştirme/test için gerçekçi gecikmeli yanıt döner
        if not settings.llm_api_key:
            await asyncio.sleep(0.35)  # Dış ağ ve LLM çıkarım gecikmesi simülasyonu
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            prompt_text = request.extract_latest_user_prompt()
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "model": target_model,
                "content": f"[Simulated LLM Response] '{prompt_text}' sorgusu için üretilen yanıt.",
                "prompt_tokens": max(1, len(prompt_text.split()) * 2),
                "completion_tokens": 24,
                "upstream_latency_ms": round(elapsed_ms, 2),
            }

        headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": target_model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        response = await self.http_client.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        usage = data.get("usage", {})
        return {
            "id": data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            "model": data.get("model", target_model),
            "content": data["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "upstream_latency_ms": round(elapsed_ms, 2),
        }