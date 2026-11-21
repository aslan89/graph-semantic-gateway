import asyncio
import logging
import re
import time
from typing import Optional
import spacy
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.schemas import EntityItem, ExtractedFeatures

logger = logging.getLogger("gateway.feature_extractor")


class FeatureExtractor:
    """
    Gelen istemlerden eş zamanlı (paralel) olarak:
    1) Sayısal vektör gömüsü (all-MiniLM-L6-v2)
    2) Niyet (Intent) ve Varlık (Entity) düğüm parametreleri
    çıkaran 2. Katman servisi.
    """

    def __init__(self) -> None:
        self._embedder: Optional[SentenceTransformer] = None
        self._nlp = None

    def load_models(self) -> None:
        """Uygulama başlangıcında modelleri belleğe yükler (soğuk başlatma gecikmesini önler)."""
        logger.info("Embedding modeli yükleniyor: %s", settings.embedding_model_name)
        self._embedder = SentenceTransformer(settings.embedding_model_name)

        try:
            logger.info("spaCy NLP modeli yükleniyor: %s", settings.spacy_model_name)
            self._nlp = spacy.load(settings.spacy_model_name)
        except OSError:
            logger.warning(
                "'%s' paketi bulunamadı. Yedek kural tabanlı ayrıştırıcı (spacy.blank) devreye alınıyor. "
                "Tam NER için 'python -m spacy download %s' çalıştırabilirsiniz.",
                settings.spacy_model_name,
                settings.spacy_model_name,
            )
            self._nlp = spacy.blank("en")

    def _compute_embedding(self, text: str) -> list[float]:
        if self._embedder is None:
            raise RuntimeError("Embedding modeli henüz yüklenmedi.")
        vector = self._embedder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def _detect_intent(self, text: str, doc) -> str:
        """
        Sorgu yapısı ve olumsuzlama (negation) kutupsallığını birleştirerek
        deterministik bir Niyet (Intent) etiketi üretir.
        """
        lowered = text.lower().strip()

        # Olumsuzlama kontrolü (False Positive engelleme)
        negation_tokens = {"not", "never", "no", "without", "n't", "değil", "yok", "hiç", "olmadan"}
        tokens_set = {t.text.lower() for t in doc}
        has_negation = bool(tokens_set & negation_tokens) or any(
            getattr(t, "dep_", "") == "neg" for t in doc
        )
        polarity = "NEG" if has_negation else "POS"

        # 1. Spesifik soru amaçları (Genel 'what is' kalıbından önce kontrol edilir)
        if re.search(
                r"\b(how many|how much|population|count|number of|amount|total|percentage|rate|price|cost|kaç|ne kadar|sayısı|nüfusu|fiyatı)\b",
                lowered,
        ):
            base_intent = "QUERY_QUANTITY"
        elif re.search(r"\b(how to|how do|how can|steps to|way to|nasıl)\b", lowered):
            base_intent = "QUERY_METHOD"
        elif re.search(r"\b(why|reason|cause of|neden|niçin|niye)\b", lowered):
            base_intent = "QUERY_REASON"
        elif re.search(r"\b(vs|versus|difference between|compare|farkı|karşılaştır)\b", lowered):
            base_intent = "QUERY_COMPARISON"
        elif re.search(r"\b(where|location of|located|nerede|nereye|hangi şehir|hangi ülke)\b", lowered):
            base_intent = "QUERY_LOCATION"
        elif re.search(r"\b(when|what year|what date|ne zaman|hangi yıl|kaç yılında)\b", lowered):
            base_intent = "QUERY_TIME"
        elif re.search(r"\b(who|whom|whose|kim|kimdir)\b", lowered):
            base_intent = "QUERY_PERSON"
        elif re.search(r"\b(what is|what are|define|meaning of|nedir|ne demek)\b", lowered):
            base_intent = "QUERY_DEFINITION"
        elif re.match(r"^(is|are|can|could|do|does|did|will|should)\b", lowered) or re.search(
                r"\b(mı|mi|mu|mü)\??$", lowered
        ):
            base_intent = "QUERY_BOOLEAN"
        else:
            base_intent = "GENERAL_STATEMENT"

        return f"{base_intent}:{polarity}"

    def _extract_nlp_features(self, text: str) -> tuple[str, list[EntityItem]]:
        if self._nlp is None:
            raise RuntimeError("NLP modeli henüz yüklenmedi.")

        doc = self._nlp(text)
        intent = self._detect_intent(text, doc)

        seen: set[tuple[str, str]] = set()
        entities: list[EntityItem] = []

        # 1. spaCy NER varlıkları
        for ent in doc.ents:
            norm_text = ent.text.strip().lower()
            if norm_text and (norm_text, ent.label_) not in seen:
                seen.add((norm_text, ent.label_))
                entities.append(EntityItem(text=norm_text, label=ent.label_))

        # 2. NER boş dönerse özel isimleri (PROPN) veya anahtar isimleri bağlam düğümü olarak ekle
        if not entities:
            for token in doc:
                if getattr(token, "pos_", "") in {"PROPN", "NOUN"} and len(token.text) > 2:
                    norm_text = token.lemma_.lower() if token.lemma_ else token.text.lower()
                    if (norm_text, "CONCEPT") not in seen:
                        seen.add((norm_text, "CONCEPT"))
                        entities.append(EntityItem(text=norm_text, label="CONCEPT"))

            # Yedek model (blank) kullanılıyorsa büyük harfle başlayan kelimeleri yakala
            if not entities:
                for match in re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ0-9]+\b", text[1:]):
                    norm_text = match.lower()
                    if (norm_text, "PROPER_NOUN") not in seen:
                        seen.add((norm_text, "PROPER_NOUN"))
                        entities.append(EntityItem(text=norm_text, label="PROPER_NOUN"))

        return intent, entities

    async def extract(self, text: str) -> tuple[list[float], ExtractedFeatures, float]:
        """
        Vektör gömüsü ve Niyet/Varlık çıkarımını ayrı thread'lerde paralel çalıştırır.
        Dönüş: (tam_vektör, özet_özellikler, geçen_süre_ms)
        """
        start_time = time.perf_counter()

        embedding_task = asyncio.to_thread(self._compute_embedding, text)
        nlp_task = asyncio.to_thread(self._extract_nlp_features, text)

        embedding, (intent, entities) = await asyncio.gather(embedding_task, nlp_task)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        features = ExtractedFeatures(
            intent=intent,
            entities=entities,
            embedding_dim=len(embedding),
            embedding_sample=[round(v, 4) for v in embedding[:5]],
        )
        return embedding, features, elapsed_ms