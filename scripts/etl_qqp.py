import argparse
import asyncio
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from app.clients.neo4j_graph import Neo4jGraphClient
from app.clients.redis_cache import RedisCacheClient
from app.services.feature_extractor import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("etl.qqp")


def generate_sample_qqp_csv(target_path: Path) -> None:
    """
    Kaggle train.csv henüz indirilmemişse, hibrit mimarinin üstünlüğünü
    test etmek için eşdeğer (is_duplicate=1) ve tuzak (is_duplicate=0)
    çiftlerden oluşan örnek bir QQP veri seti üretir.
    """
    sample_rows = [
        # --- POZİTİF ÇİFTLER (is_duplicate = 1 -> Cache Hit Beklenir) ---
        (1, 101, 102, "How many people live in Ankara?", "What is the population count of Ankara?", 1),
        (2, 103, 104, "How can I learn Python programming fast?", "What is the fastest way to learn Python?", 1),
        (3, 105, 106, "Why is the sky blue during the day?", "What is the reason the sky appears blue?", 1),
        (4, 107, 108, "Where is the headquarters of Google located?", "In which city is Google based?", 1),
        (5, 109, 110, "When was the Republic of Turkey founded?", "In what year was Turkey established as a republic?", 1),
        (6, 111, 112, "What is the difference between TCP and UDP?", "How do TCP and UDP protocols compare?", 1),
        (7, 113, 114, "How much does an iPhone 15 cost in Germany?", "What is the price of iPhone 15 in Germany?", 1),
        (8, 115, 116, "Who invented the World Wide Web?", "Who is the creator of the World Wide Web?", 1),
        (9, 117, 118, "What is semantic caching in LLM gateways?", "Define semantic caching for large language models.", 1),
        (10, 119, 120, "Can Docker containers run on Windows 11?", "Is it possible to run Docker on Windows 11?", 1),
        # --- NEGATİF TUZAK ÇİFTLER (is_duplicate = 0 -> Cache Miss Beklenir) ---
        # Tuzak Türü A: Varlık (Entity) değişimi (yüksek kelime örtüşmesi)
        (11, 201, 202, "How many people live in Istanbul?", "How many people live in Izmir?", 0),
        (12, 203, 204, "Where is the headquarters of Apple located?", "Where is the headquarters of Microsoft located?", 0),
        (13, 205, 206, "How much does a Tesla Model 3 cost in Norway?", "How much does a Tesla Model 3 cost in Japan?", 0),
        (14, 207, 208, "When was the Eiffel Tower built in Paris?", "When was the Colosseum built in Rome?", 0),
        # Tuzak Türü B: Olumsuzlama (Negation) kutupsallığı tersine çevrilmiş sorgular
        (15, 209, 210, "Why should companies use Redis for caching?", "Why should companies not use Redis for caching?", 0),
        (16, 211, 212, "Can Python run without an interpreter?", "Can Python run with an interpreter?", 0),
        # Tuzak Türü C: Aynı varlık ancak tamamen farklı Niyet (Intent)
        (17, 213, 214, "Who founded Amazon?", "When was Amazon founded?", 0),
        (18, 215, 216, "Where is Neo4j developed?", "Why is Neo4j used for knowledge graphs?", 0),
        (19, 217, 218, "How to install PostgreSQL on Ubuntu?", "How to uninstall PostgreSQL on Ubuntu?", 0),
        (20, 219, 220, "What is the population of Berlin?", "Where is Berlin located?", 0),
    ]
    df = pd.DataFrame(
        sample_rows,
        columns=["id", "qid1", "qid2", "question1", "question2", "is_duplicate"],
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logger.info("Örnek QQP veri seti oluşturuldu: %s (%d çift)", target_path, len(df))


def load_balanced_qqp(csv_path: Path, limit: int, seed: int = 42) -> pd.DataFrame:
    """
    QQP veri setini okur, boş değerleri temizler, benzersiz question1 kayıtlarına
    sahip dengeli (%50 pozitif, %50 negatif) bir test alt kümesi seçer.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"id", "qid1", "qid2", "question1", "question2", "is_duplicate"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV dosyası şu sütunları içermelidir: {required_cols}")

    df = df.dropna(subset=["question1", "question2", "is_duplicate"]).copy()
    df["question1"] = df["question1"].astype(str).str.strip()
    df["question2"] = df["question2"].astype(str).str.strip()
    df["is_duplicate"] = df["is_duplicate"].astype(int)

    # Aynı question1'in hem pozitif hem negatif eşleşmede çakışmasını önlemek için tekilleştir
    df = df.drop_duplicates(subset=["question1"]).drop_duplicates(subset=["question2"])

    half = limit // 2
    pos_df = df[df["is_duplicate"] == 1]
    neg_df = df[df["is_duplicate"] == 0]

    pos_sample = pos_df.sample(n=min(half, len(pos_df)), random_state=seed)
    neg_sample = neg_df.sample(n=min(half, len(neg_df)), random_state=seed)

    balanced = pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


async def run_etl(
    csv_path: Path,
    eval_output_path: Path,
    limit: int,
    reset_db: bool,
    create_sample: bool,
) -> None:
    if not csv_path.exists():
        if create_sample:
            generate_sample_qqp_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"'{csv_path}' bulunamadı. Kaggle QQP 'train.csv' dosyasını bu yola koyun "
                "veya '--create-sample' parametresiyle çalıştırın."
            )

    df = load_balanced_qqp(csv_path, limit=limit)
    logger.info(
        "Dengeli veri kümesi hazırlandı | Toplam=%d | Pozitif (is_duplicate=1)=%d | Negatif (is_duplicate=0)=%d",
        len(df),
        int((df["is_duplicate"] == 1).sum()),
        int((df["is_duplicate"] == 0).sum()),
    )

    feature_extractor = FeatureExtractor()
    await asyncio.to_thread(feature_extractor.load_models)

    neo4j_client = Neo4jGraphClient()
    redis_client = RedisCacheClient()

    try:
        if reset_db:
            logger.warning("Mevcut Neo4j ve Redis önbellek kayıtları temizleniyor...")
            async with neo4j_client.driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            await redis_client.client.flushdb()

        await neo4j_client.initialize_schema()

        # question1 kayıtlarını Neo4j + Redis başlangıç durumu olarak yükle
        for _, row in tqdm(df.iterrows(), total=len(df), desc="QQP question1 Yükleniyor"):
            q1_text = row["question1"]
            message_id = f"qqp-{row['qid1']}"

            embedding, features, _ = await feature_extractor.extract(q1_text)

            # Tez yöntemi gereği question1 metni Redis'e yanıt (value) olarak yazılır
            await redis_client.set_answer(message_id=message_id, answer=q1_text)
            await neo4j_client.upsert_message_node(
                message_id=message_id,
                prompt=q1_text,
                embedding=embedding,
                intent=features.intent,
                entities=features.entities,
            )

        # Simülatör için question2 değerlendirme dosyasını kaydet
        eval_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(eval_output_path, index=False)
        logger.info("ETL tamamlandı! Değerlendirme dosyası kaydedildi: %s", eval_output_path)
    finally:
        await redis_client.close()
        await neo4j_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="QQP Veri Seti Neo4j & Redis ETL Betiği")
    parser.add_argument("--csv-path", type=Path, default=Path("data/train.csv"), help="QQP train.csv yolu")
    parser.add_argument("--eval-out", type=Path, default=Path("data/eval_pairs.csv"), help="Test çiftleri çıktı yolu")
    parser.add_argument("--limit", type=int, default=200, help="Yüklenecek dengeli çift sayısı (örn. 200, 1000)")
    parser.add_argument("--reset-db", action="store_true", help="Yükleme öncesi Neo4j ve Redis veritabanlarını sıfırla")
    parser.add_argument("--create-sample", action="store_true", help="train.csv yoksa örnek veri seti oluştur")
    args = parser.parse_args()

    asyncio.run(
        run_etl(
            csv_path=args.csv_path,
            eval_output_path=args.eval_out,
            limit=args.limit,
            reset_db=args.reset_db,
            create_sample=args.create_sample,
        )
    )


if __name__ == "__main__":
    main()