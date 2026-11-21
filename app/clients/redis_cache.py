import asyncio
import logging
from typing import Awaitable, Callable, Optional
import redis.asyncio as redis
from app.config import settings

logger = logging.getLogger("gateway.redis")

KEY_PREFIX = "semcache:resp:"


class RedisCacheClient:
    """
    Katman 4 (Yanıt Motoru):
    Eşleşen Mesaj ID'sine ait yanıt metnini milisaniyeler içinde sunan ve
    TTL süre dolumlarını izleyerek Neo4j ile veri tutarlılığı sağlayan istemci.
    """

    def __init__(self) -> None:
        self.client: redis.Redis = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=100,
        )

    async def close(self) -> None:
        await self.client.aclose()

    def _format_key(self, message_id: str) -> str:
        return f"{KEY_PREFIX}{message_id}"

    async def get_answer(self, message_id: str) -> Optional[str]:
        return await self.client.get(self._format_key(message_id))

    async def set_answer(
        self,
        message_id: str,
        answer: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        ttl = ttl_seconds or settings.cache_ttl_seconds
        await self.client.set(self._format_key(message_id), answer, ex=ttl)

    async def listen_expired_events(
        self,
        on_expire: Callable[[str], Awaitable[None]],
    ) -> None:
        """
        Redis keyspace expiration olaylarını dinler.
        Bir yanıtın süresi dolduğunda ilgili message_id'yi Neo4j'den silmek üzere
        geri çağırım (callback) fonksiyonunu tetikler.
        """
        pubsub = self.client.pubsub()
        await pubsub.psubscribe("__keyevent@0__:expired")
        logger.info("Redis TTL tutarlılık dinleyicisi başlatıldı.")

        try:
            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    expired_key: str = message["data"]
                    if expired_key.startswith(KEY_PREFIX):
                        message_id = expired_key.removeprefix(KEY_PREFIX)
                        await on_expire(message_id)
        except asyncio.CancelledError:
            await pubsub.punsubscribe("__keyevent@0__:expired")
            await pubsub.aclose()
            raise