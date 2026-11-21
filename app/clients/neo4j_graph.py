import logging
from typing import Optional
from neo4j import AsyncDriver, AsyncGraphDatabase
from app.config import settings
from app.schemas import EntityItem

logger = logging.getLogger("gateway.neo4j")


class Neo4jGraphClient:
    """
    Katman 3 (Kural/Karar Motoru):
    Mesaj, Niyet (Intent) ve Varlık (Entity) düğümleri üzerinde vektör benzerliği
    ve graf komşuluk doğrulamasını hibrit Cypher sorgularıyla yürüten istemci.
    """

    def __init__(self) -> None:
        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )

    async def close(self) -> None:
        await self.driver.close()

    async def initialize_schema(self) -> None:
        """Düğüm kısıtlarını (constraints) ve 384 boyutlu kosinüs vektör indeksini oluşturur."""
        queries = [
            "CREATE CONSTRAINT message_id_unique IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT intent_name_unique IF NOT EXISTS FOR (i:Intent) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
            """
            CREATE VECTOR INDEX message_embedding_idx IF NOT EXISTS
            FOR (m:Message) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """,
        ]
        async with self.driver.session() as session:
            for query in queries:
                await session.run(query)
        logger.info("Neo4j graf kısıtları ve vektör indeksi doğrulandı.")

    def compute_dynamic_threshold(self, prompt: str, entity_count: int) -> float:
        """
        Sorgu uzunluğu ve çıkarılan varlık sayısına göre dinamik benzerlik eşiği hesaplar.
        - Çok kısa sorgularda yanlış pozitif (False Positive) riskini azaltmak için eşik yükseltilir.
        - Zengin varlık bağlamına sahip uzun ifadelerde eşik kontrollü biçimde esnetilir.
        """
        threshold = settings.base_similarity_threshold
        word_count = len(prompt.split())

        if word_count <= 4:
            threshold += 0.05
        elif word_count >= 12:
            threshold -= 0.02

        if entity_count >= 2:
            threshold -= 0.02
        elif entity_count == 0:
            threshold += 0.03

        return round(min(max(threshold, 0.76), 0.94), 4)

    async def find_semantic_match(
        self,
        prompt: str,
        embedding: list[float],
        intent: str,
        entities: list[EntityItem],
        mode: str = "hybrid",
    ) -> Optional[dict]:
        """
        Hibrit veya salt vektör modunda Neo4j üzerinde aday mesaj düğümünü arar.
        Eşleşme varsa {'message_id': str, 'score': float, 'threshold': float} döner.
        """
        threshold = self.compute_dynamic_threshold(prompt, len(entities))
        entity_keys = [f"{e.text}:{e.label}" for e in entities]

        if mode == "vector_only":
            # Baseline karşılaştırması için sadece vektör benzerliğine bakan sorgu
            cypher = """
            CALL db.index.vector.queryNodes('message_embedding_idx', 1, $embedding)
            YIELD node AS m, score
            WHERE score >= $threshold
            RETURN m.id AS message_id, score
            LIMIT 1
            """
            params = {"embedding": embedding, "threshold": threshold}
        else:
            # Önerilen Hibrit Mimari:
            # 1) Vektör indeksi ile en yakın 5 aday düğümü ön filtrele
            # 2) Niyet (Intent) düğümü eşleşmesini doğrula
            # 3) Varlık (Entity) kümesi birebir örtüşmesini kontrol et
            cypher = """
            CALL db.index.vector.queryNodes('message_embedding_idx', 5, $embedding)
            YIELD node AS m, score
            WHERE score >= $threshold
            MATCH (m)-[:HAS_INTENT]->(:Intent {name: $intent})
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            WITH m, score, collect(DISTINCT e.key) AS candidate_entities
            WHERE size(candidate_entities) = size($entity_keys)
              AND ALL(ek IN $entity_keys WHERE ek IN candidate_entities)
            RETURN m.id AS message_id, score
            ORDER BY score DESC
            LIMIT 1
            """
            params = {
                "embedding": embedding,
                "threshold": threshold,
                "intent": intent,
                "entity_keys": entity_keys,
            }

        async with self.driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()
            if record:
                return {
                    "message_id": record["message_id"],
                    "score": round(float(record["score"]), 4),
                    "threshold": threshold,
                }
        return None

    async def upsert_message_node(
        self,
        message_id: str,
        prompt: str,
        embedding: list[float],
        intent: str,
        entities: list[EntityItem],
    ) -> None:
        """Cache Miss sonrasında Mesaj, Niyet ve Varlık düğümleri ile ilişkilerini yazar."""
        entity_payload = [
            {"key": f"{e.text}:{e.label}", "text": e.text, "label": e.label}
            for e in entities
        ]
        cypher = """
        MERGE (m:Message {id: $message_id})
        SET m.prompt = $prompt,
            m.embedding = $embedding,
            m.created_at = datetime()
        MERGE (i:Intent {name: $intent})
        MERGE (m)-[:HAS_INTENT]->(i)
        FOREACH (ent IN $entities |
            MERGE (e:Entity {key: ent.key})
            ON CREATE SET e.text = ent.text, e.label = ent.label
            MERGE (m)-[:MENTIONS]->(e)
        )
        """
        async with self.driver.session() as session:
            await session.run(
                cypher,
                message_id=message_id,
                prompt=prompt,
                embedding=embedding,
                intent=intent,
                entities=entity_payload,
            )

    async def delete_message_node(self, message_id: str) -> None:
        """Redis TTL süresi dolduğunda Neo4j'deki karşılık gelen Mesaj düğümünü siler."""
        cypher = """
        MATCH (m:Message {id: $message_id})
        DETACH DELETE m
        """
        async with self.driver.session() as session:
            await session.run(cypher, message_id=message_id)
        logger.info("Süresi dolan önbellek düğümü Neo4j'den temizlendi: %s", message_id)