from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)

    def extract_latest_user_prompt(self) -> str:
        """Önbellek sorgusu için son kullanıcı mesajını ayrıştırır."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content.strip()
        return ""


class EntityItem(BaseModel):
    text: str
    label: str


class ExtractedFeatures(BaseModel):
    intent: str
    entities: list[EntityItem]
    embedding_dim: int
    embedding_sample: list[float]


class GatewayTelemetry(BaseModel):
    cache_status: Literal["HIT", "MISS", "BYPASS"]
    gateway_mode: str
    total_latency_ms: float
    feature_extraction_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    llm_upstream_ms: float = 0.0
    matched_message_id: Optional[str] = None
    similarity_score: Optional[float] = None
    applied_threshold: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_saved: bool = False
    extracted_features: Optional[ExtractedFeatures] = None


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    content: str
    telemetry: GatewayTelemetry